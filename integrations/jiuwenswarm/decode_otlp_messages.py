"""Decode user/assistant text from JiuwenSwarm OTLP JSON trace attributes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_trace_validator import classify_tool_result


USER_ATTRIBUTE_PATH = (
    "resourceSpans[].scopeSpans[].spans[name=jiuwenclaw.agent.invoke]."
    "attributes[key=gen_ai.input.messages].value.stringValue"
)
OUTPUT_ATTRIBUTE_PATH = (
    "resourceSpans[].scopeSpans[].spans[name=gen_ai.chat]."
    "attributes[key=gen_ai.output.messages].value.stringValue"
)


def iter_spans(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for resource in payload.get("resourceSpans", []):
        for scope in resource.get("scopeSpans", []):
            yield from scope.get("spans", [])


def attribute_map(span: dict[str, Any]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for attribute in span.get("attributes", []):
        key = attribute.get("key")
        value = attribute.get("value") or {}
        if not key:
            continue
        for field in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if field in value:
                decoded[str(key)] = value[field]
                break
    return decoded


def decode_messages(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    try:
        parsed = json.loads(str(value)) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return []
    return [message for message in parsed if isinstance(message, dict)] if isinstance(parsed, list) else []


def message_text(message: dict[str, Any]) -> str:
    parts = message.get("parts")
    if isinstance(parts, list):
        return "".join(
            str(part.get("content") or part.get("text") or "")
            for part in parts
            if isinstance(part, dict)
        ).strip()
    return str(message.get("content") or "").strip()


def unwrap_user_envelope(text: str) -> str:
    marker = "你收到一条消息："
    if marker not in text:
        return text
    encoded = text.split(marker, 1)[1].strip()
    try:
        envelope = json.loads(encoded)
    except json.JSONDecodeError:
        return text
    if isinstance(envelope, dict) and envelope.get("content"):
        return str(envelope["content"]).strip()
    return text


def decode_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    spans = list(iter_spans(payload))
    trace_id = next((str(span.get("traceId")) for span in spans if span.get("traceId")), path.stem)
    user_input = ""
    assistant_outputs: list[str] = []
    role_counts: Counter[str] = Counter()
    model = ""
    input_tokens: Any = None
    output_tokens: Any = None
    capture_spans = []
    tool_calls: list[dict[str, Any]] = []

    for span in spans:
        name = str(span.get("name") or "")
        attributes = attribute_map(span)
        input_messages = decode_messages(attributes.get("gen_ai.input.messages"))
        output_messages = decode_messages(attributes.get("gen_ai.output.messages"))
        for message in input_messages + output_messages:
            role_counts[str(message.get("role") or "unknown")] += 1

        if name == "jiuwenclaw.agent.invoke" and not user_input:
            for message in input_messages:
                if message.get("role") == "user" and message_text(message):
                    user_input = unwrap_user_envelope(message_text(message))
                    break
        if name == "gen_ai.chat":
            model = str(
                attributes.get("gen_ai.response.model")
                or attributes.get("gen_ai.request.model")
                or model
            )
            input_tokens = attributes.get("gen_ai.usage.input_tokens", input_tokens)
            output_tokens = attributes.get("gen_ai.usage.output_tokens", output_tokens)
            for message in output_messages:
                if message.get("role") == "assistant" and message_text(message):
                    assistant_outputs.append(message_text(message))
        if name == "gen_ai.tool":
            raw_status = str((span.get("status") or {}).get("code") or "UNSET")
            tool_result = attributes.get("gen_ai.tool.result")
            semantic_status = classify_tool_result(tool_result, raw_status)
            tool_calls.append({
                "name": str(attributes.get("gen_ai.tool.name") or "unknown"),
                "arguments": attributes.get("gen_ai.tool.arguments"),
                "result": tool_result,
                "status": "ERROR" if semantic_status["is_error"] else raw_status.replace("STATUS_CODE_", ""),
                "raw_span_status": raw_status.replace("STATUS_CODE_", ""),
                "semantic_error": bool(semantic_status["is_error"]),
            })
        if input_messages or output_messages:
            capture_spans.append({
                "span_name": name,
                "has_input_messages": bool(input_messages),
                "has_output_messages": bool(output_messages),
            })

    return {
        "trace_id": trace_id,
        "source_file": str(path.resolve()),
        "message_capture_present": bool(capture_spans),
        "user_input": user_input or None,
        "assistant_output": assistant_outputs[-1] if assistant_outputs else None,
        "assistant_outputs": assistant_outputs,
        "tool_calls": tool_calls,
        "model": model or None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "message_role_counts": dict(role_counts),
        "capture_spans": capture_spans,
        "raw_attribute_locations": {
            "user_input": USER_ATTRIBUTE_PATH,
            "assistant_output": OUTPUT_ATTRIBUTE_PATH,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="OTLP JSON file or directory")
    parser.add_argument("--output", type=Path, help="Write decoded JSON report")
    args = parser.parse_args()

    paths = sorted(args.input.glob("*.json")) if args.input.is_dir() else [args.input]
    traces = [decode_trace(path) for path in paths]
    report = {
        "trace_count": len(traces),
        "message_capture_count": sum(row["message_capture_present"] for row in traces),
        "user_input_count": sum(row["user_input"] is not None for row in traces),
        "assistant_output_count": sum(row["assistant_output"] is not None for row in traces),
        "traces": traces,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
