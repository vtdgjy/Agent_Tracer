"""Four-layer sidecar probe for a running JiuwenSwarm instance.

This process is intentionally independent of OpenTelemetry tracing.  It emits
MonitoringEvent batches to Agent Tracer while ``jiuwenswarm-instrumentor``
continues to emit traces/metrics through OTLP.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multilayer_monitor import collect_hardware_snapshot  # noqa: E402
from integrations.jiuwenswarm.production_collectors import ProductionCollectorSuite  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def metric_values(layer: dict[str, Any]) -> dict[str, Any]:
    return {item["key"]: item.get("value") for item in layer.get("metrics", [])}


def tcp_probe(host: str, port: int, timeout: float = 0.5) -> tuple[bool, float]:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, round((time.perf_counter() - started) * 1000, 2)
    except OSError:
        return False, round((time.perf_counter() - started) * 1000, 2)


def process_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"process_count": None, "rss_bytes": None, "cpu_percent": None}
    try:
        import psutil  # type: ignore

        matches = []
        for process in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            try:
                command = " ".join(process.info.get("cmdline") or []).lower()
                is_agent_process = (
                    "jiuwenswarm-start" in command
                    or "-m jiuwenswarm." in command
                    or command.rstrip().endswith("jiuwenswarm.exe")
                )
                if is_agent_process and "runtime_probe.py" not in command:
                    matches.append(process)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
        result["process_count"] = len(matches)
        result["rss_bytes"] = sum(p.memory_info().rss for p in matches if p.is_running())
        result["cpu_percent"] = round(sum(p.cpu_percent(interval=None) for p in matches), 2)
    except ImportError:
        pass
    return result


def git_branch(project: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project), "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return completed.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def probe_version() -> str:
    try:
        return importlib.metadata.version("jiuwenswarm-instrumentor")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_events(
    system_id: str,
    project: Path,
    host: str,
    ports: list[int],
    production_collectors: ProductionCollectorSuite | None = None,
) -> list[dict[str, Any]]:
    timestamp = utc_now()
    correlation = {
        key: value
        for key, value in {
            "trace_id": os.getenv("MONITOR_TRACE_ID"),
            "session_id": os.getenv("MONITOR_SESSION_ID"),
        }.items()
        if value
    }

    hardware = collect_hardware_snapshot()
    connections = {str(port): dict(zip(("reachable", "latency_ms"), tcp_probe(host, port))) for port in ports}
    processes = process_snapshot()
    services_up = sum(1 for item in connections.values() if item["reachable"])
    common = {
        "timestamp": timestamp,
        "system_id": system_id,
        "labels": {"agent_framework": "jiuwenswarm", "host": socket.gethostname()},
        "correlation": correlation,
    }
    events = [
        {
            **common,
            "layer": "hardware",
            "event_type": "host.resource.snapshot",
            "measurement": metric_values(hardware),
        },
        {
            **common,
            "layer": "virtualization",
            "event_type": "runtime.process.snapshot",
            "measurement": {
                **processes,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
        },
        {
            **common,
            "layer": "communication",
            "event_type": "gateway.endpoint.snapshot",
            "measurement": {"host": host, "ports": connections, "reachable_count": services_up},
        },
        {
            **common,
            "layer": "application",
            "event_type": "agent.runtime.snapshot",
            "measurement": {
                "branch": git_branch(project),
                "instrumentor_version": probe_version(),
                "services_ready": services_up == len(ports),
                "services_ready_count": services_up,
            },
        },
    ]
    if production_collectors is not None:
        events.extend(production_collectors.collect(common))
    return events


def send_batch(endpoint: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    payload = json.dumps({"events": events}, ensure_ascii=False).encode("utf-8")
    request = Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=5) as response:  # noqa: S310 - endpoint is operator supplied
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-id", default="jiuwenswarm-develop")
    parser.add_argument("--agent-project", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:5000/api/v1/monitor/events:batch",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ports", default="18092,19000,19001")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--kubernetes", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--cgroup", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--dcgm-url", default=os.getenv("DCGM_EXPORTER_URL", ""))
    parser.add_argument("--npu-url", default=os.getenv("NPU_EXPORTER_URL", ""))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ports = [int(value.strip()) for value in args.ports.split(",") if value.strip()]
    production_collectors = ProductionCollectorSuite(
        kubernetes=args.kubernetes,
        cgroup=args.cgroup,
        dcgm_url=args.dcgm_url,
        npu_url=args.npu_url,
    )
    while True:
        events = build_events(
            args.system_id,
            args.agent_project.resolve(),
            args.host,
            ports,
            production_collectors,
        )
        if args.dry_run:
            print(json.dumps({"events": events}, ensure_ascii=False, indent=2))
        else:
            result = send_batch(args.endpoint, events)
            print(json.dumps(result, ensure_ascii=False))
        if args.once:
            return 0
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
