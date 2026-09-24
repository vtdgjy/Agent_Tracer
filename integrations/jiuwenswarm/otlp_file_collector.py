"""Minimal local OTLP/HTTP trace receiver that writes one JSON file per trace.

This receiver is intentionally small: it is for deterministic replay and local
acceptance data collection, not as a production telemetry backend.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
from concurrent import futures
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from google.protobuf.json_format import MessageToDict, ParseDict
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer,
    add_TraceServiceServicer_to_server,
)
import grpc


def _hex_id(value: str) -> str:
    if not value:
        return ""
    try:
        return base64.b64decode(value).hex()
    except Exception:
        return value


def request_to_otlp_json(request: ExportTraceServiceRequest) -> dict[str, Any]:
    payload = MessageToDict(request, preserving_proto_field_name=False)
    for resource_spans in payload.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                span["traceId"] = _hex_id(span.get("traceId", ""))
                span["spanId"] = _hex_id(span.get("spanId", ""))
                if span.get("parentSpanId"):
                    span["parentSpanId"] = _hex_id(span["parentSpanId"])
                for link in span.get("links", []):
                    link["traceId"] = _hex_id(link.get("traceId", ""))
                    link["spanId"] = _hex_id(link.get("spanId", ""))
    return payload


def split_by_trace(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    traces: dict[str, dict[str, Any]] = {}
    for resource_spans in payload.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                trace_id = str(span.get("traceId") or "")
                if not trace_id:
                    continue
                trace_payload = traces.setdefault(trace_id, {"resourceSpans": []})
                copied_resource = {
                    key: deepcopy(value)
                    for key, value in resource_spans.items()
                    if key != "scopeSpans"
                }
                copied_scope = {
                    key: deepcopy(value)
                    for key, value in scope_spans.items()
                    if key != "spans"
                }
                copied_scope["spans"] = [deepcopy(span)]
                copied_resource["scopeSpans"] = [copied_scope]
                trace_payload["resourceSpans"].append(copied_resource)
    return traces


class TraceStore:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.trace_dir = output_dir / "traces"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, payload: dict[str, Any]) -> list[str]:
        trace_ids = []
        with self.lock:
            for trace_id, fragment in split_by_trace(payload).items():
                path = self.trace_dir / f"{trace_id}.json"
                if path.exists():
                    current = json.loads(path.read_text(encoding="utf-8"))
                else:
                    current = {"resourceSpans": []}
                current["resourceSpans"].extend(fragment["resourceSpans"])
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(current, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary.replace(path)
                trace_ids.append(trace_id)
            self.write_manifest()
        return trace_ids

    def write_manifest(self) -> None:
        rows = []
        for path in sorted(self.trace_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            span_count = sum(
                len(scope.get("spans", []))
                for resource in payload.get("resourceSpans", [])
                for scope in resource.get("scopeSpans", [])
            )
            rows.append({"trace_id": path.stem, "span_count": span_count, "file": str(path)})
        (self.output_dir / "manifest.json").write_text(
            json.dumps({"trace_count": len(rows), "traces": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_handler(store: TraceStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _reply(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._reply(200, b'{"status":"ok"}', "application/json")
                return
            self._reply(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") not in {"/v1/traces", "/traces"}:
                self._reply(404, b"not found", "text/plain")
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            request = ExportTraceServiceRequest()
            try:
                content_type = self.headers.get("Content-Type", "").lower()
                if "json" in content_type:
                    ParseDict(json.loads(body.decode("utf-8")), request)
                else:
                    request.ParseFromString(body)
                store.append(request_to_otlp_json(request))
                response = ExportTraceServiceResponse().SerializeToString()
                self._reply(200, response, "application/x-protobuf")
            except Exception as exc:
                self._reply(
                    400,
                    json.dumps({"error": str(exc)}).encode("utf-8"),
                    "application/json",
                )

    return Handler


class GrpcTraceService(TraceServiceServicer):
    def __init__(self, store: TraceStore) -> None:
        self.store = store

    def Export(self, request, context):  # noqa: N802
        self.store.append(request_to_otlp_json(request))
        return ExportTraceServiceResponse()


def main() -> int:
    parser = argparse.ArgumentParser(description="Write OTLP/HTTP traces to local JSON files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=4318)
    parser.add_argument("--grpc-port", type=int, default=4317)
    args = parser.parse_args()

    store = TraceStore(Path(args.output_dir).resolve())
    server = ThreadingHTTPServer((args.host, args.http_port), build_handler(store))
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    add_TraceServiceServicer_to_server(GrpcTraceService(store), grpc_server)
    grpc_server.add_insecure_port(f"{args.host}:{args.grpc_port}")
    grpc_server.start()
    print(json.dumps({
        "status": "ready",
        "host": args.host,
        "http_port": args.http_port,
        "grpc_port": args.grpc_port,
    }), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        grpc_server.stop(grace=2)
        store.write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
