import json
import hashlib
import re
from datetime import datetime, timezone


def _safe_json_parse_text(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _decode_otel_value(value_obj):
    if not isinstance(value_obj, dict):
        return value_obj

    scalar_keys = [
        "stringValue",
        "string_value",
        "boolValue",
        "bool_value",
        "intValue",
        "int_value",
        "doubleValue",
        "double_value",
        "bytesValue",
        "bytes_value",
    ]
    for key in scalar_keys:
        if key in value_obj:
            return value_obj.get(key)

    array_value = value_obj.get("arrayValue")
    if array_value is None:
        array_value = value_obj.get("array_value")
    if array_value is not None:
        values = array_value.get("values", []) if isinstance(array_value, dict) else []
        return [_decode_otel_value(v) for v in values]

    kvlist_value = value_obj.get("kvlistValue")
    if kvlist_value is None:
        kvlist_value = value_obj.get("kvlist_value")
    if kvlist_value is not None:
        values = kvlist_value.get("values", []) if isinstance(kvlist_value, dict) else []
        out = {}
        for item in values:
            if not isinstance(item, dict):
                continue
            k = item.get("key")
            out[k] = _decode_otel_value(item.get("value", {}))
        return out

    return value_obj


def _decode_otel_attributes(attrs):
    if isinstance(attrs, dict):
        return attrs
    if not isinstance(attrs, list):
        return {}

    out = {}
    for item in attrs:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        if not key:
            continue
        out[key] = _decode_otel_value(item.get("value", {}))
    return out


def _ns_to_iso(ts_ns):
    try:
        if ts_ns in [None, "", 0, "0"]:
            return ""
        ns_int = int(str(ts_ns))
        seconds = ns_int / 1_000_000_000
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return str(ts_ns) if ts_ns is not None else ""


def _duration_ns_to_iso8601(duration_ns):
    try:
        ns_int = int(duration_ns)
        if ns_int < 0:
            ns_int = 0
        sec = ns_int / 1_000_000_000
        return f"PT{sec:.6f}S"
    except Exception:
        return ""


def _ms_offset_to_iso(at_ms):
    try:
        ms = float(at_ms)
    except Exception:
        ms = 0.0
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _stable_span_id(trace_id, idx, label, event_type):
    raw = f"{trace_id}|{idx}|{label}|{event_type}"
    return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _stringify_json(value):
    return json.dumps(value, ensure_ascii=False) if value is not None else ""


def _first_present(obj, *keys, default=None):
    if not isinstance(obj, dict):
        return default
    for key in keys:
        if key in obj:
            value = obj.get(key)
            if value is not None:
                return value
    return default


def _extract_message_text_from_gen_ai_messages(value):
    parsed = _safe_json_parse_text(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        return ""
    parts = []
    for message in parsed:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content_parts = message.get("parts")
        content = ""
        if isinstance(content_parts, list):
            text_parts = []
            for part in content_parts:
                if isinstance(part, dict) and isinstance(part.get("content"), str):
                    text_parts.append(part.get("content"))
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part.get("text"))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(t for t in text_parts if t)
        elif isinstance(message.get("content"), str):
            content = message.get("content")
        if content:
            prefix = f"{role}: " if role else ""
            parts.append(prefix + content)
    return "\n".join(parts)


def _looks_like_tool_error(text):
    if not isinstance(text, str) or not text.strip():
        return False
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in [
            "[error]",
            "success=false",
            "traceback",
            "exception",
            "failed",
            "timeout",
            "cannot find",
            "not recognized",
        ]
    )


def _enrich_gen_ai_span(span_name, span_attributes):
    if not isinstance(span_attributes, dict):
        return span_name, {}

    attrs = dict(span_attributes)
    original_span_name = str(span_name or "")
    if original_span_name:
        attrs.setdefault("otel.original_span_name", original_span_name)

    if original_span_name in {"jiuwenclaw.agent.invoke", "jiuwenclaw.subagent.invoke"}:
        span_name = "CodeAgent.run"
        input_text = _extract_message_text_from_gen_ai_messages(attrs.get("gen_ai.input.messages"))
        if input_text:
            attrs.setdefault("input.value", _stringify_json({"task": input_text}))
        agent_name = attrs.get("jiuwenclaw.agent.name") or attrs.get("gen_ai.agent.name")
        if agent_name:
            attrs.setdefault("chat.name", str(agent_name))

    elif original_span_name == "gen_ai.chat":
        span_name = "LiteLLMModel.__call__"
        input_text = _extract_message_text_from_gen_ai_messages(attrs.get("gen_ai.input.messages"))
        output_text = _extract_message_text_from_gen_ai_messages(attrs.get("gen_ai.output.messages"))
        if input_text:
            attrs.setdefault("input.value", _stringify_json({"messages": input_text}))
            attrs.setdefault("llm.input_messages.0.message.content", input_text)
        if output_text:
            attrs.setdefault("output.value", output_text)
            attrs.setdefault("llm.output_messages.0.message.content", output_text)

    elif original_span_name == "gen_ai.tool" or "gen_ai.tool.name" in attrs:
        tool_name = str(attrs.get("gen_ai.tool.name") or "tool")
        span_name = f"{tool_name}Tool"
        tool_args = attrs.get("gen_ai.tool.arguments")
        tool_result = attrs.get("gen_ai.tool.result")
        attrs.setdefault("openinference.span.kind", "TOOL")
        attrs.setdefault("tool.name", tool_name)
        attrs.setdefault("tool.id", str(attrs.get("gen_ai.tool.call.id") or ""))
        attrs.setdefault("input.value", tool_args if isinstance(tool_args, str) else _stringify_json(tool_args))
        attrs.setdefault("output.value", tool_result if isinstance(tool_result, str) else _stringify_json(tool_result))

    return span_name, attrs


def _extract_huawei_task(payload):
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    for key in ["input", "query", "task", "question", "prompt", "phase"]:
        value = data.get(key)
        if value not in [None, ""]:
            return str(value)
    if data:
        return _stringify_json(data)
    for key in ["content", "message", "name", "event_type"]:
        value = payload.get(key)
        if value not in [None, ""]:
            return str(value)
    return _stringify_json(payload)


def _extract_huawei_error(payload):
    for key in ["error", "message", "exception", "exception_type"]:
        value = payload.get(key)
        if value not in [None, ""]:
            return str(value)
    return _stringify_json(payload)


def _normalize_huawei_event_trace(obj):
    if not isinstance(obj, dict):
        return None

    events = obj.get("events")
    if not isinstance(events, list) or not events:
        return None

    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    trace_id = str(meta.get("trace_id") or "")
    for ev in events:
        if not isinstance(ev, dict):
            continue
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else ev
        if not isinstance(payload, dict):
            continue
        trace_id = trace_id or str(payload.get("traceId") or payload.get("trace_id") or "")
    if not trace_id:
        trace_id = hashlib.md5(_stringify_json(obj).encode("utf-8", errors="ignore")).hexdigest()

    flat_spans = []
    prev_id = None
    for idx, ev in enumerate(events):
        if not isinstance(ev, dict):
            continue

        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else ev
        if not isinstance(payload, dict):
            continue

        event_type = str(payload.get("event_type") or "")
        if not event_type:
            continue

        label = str(ev.get("label") or event_type or f"event_{idx + 1}")
        span_id = _stable_span_id(trace_id, idx, label, event_type)
        timestamp = _ms_offset_to_iso(ev.get("at_ms", idx))

        attrs = {
            "huawei.event_type": event_type,
            "huawei.label": label,
            "huawei.payload": _stringify_json(payload),
        }
        span_name = f"HuaweiEvent.{event_type}"
        span_kind = "Internal"
        status_code = "Unset"
        status_message = ""

        if event_type == "chat.tracer_agent":
            span_name = "CodeAgent.run"
            attrs.update(
                {
                    "chat.role": str(payload.get("role") or ""),
                    "chat.name": str(payload.get("name") or ""),
                    "input.value": _stringify_json({"task": _extract_huawei_task(payload)}),
                }
            )
            if isinstance(payload.get("data"), dict):
                attrs["output.value"] = _stringify_json(payload.get("data"))

        elif event_type == "tool.use":
            tool_name = str(payload.get("name") or "tool")
            span_name = f"{tool_name}Tool"
            span_kind = "TOOL"
            attrs.update(
                {
                    "openinference.span.kind": "TOOL",
                    "tool.name": tool_name,
                    "tool.id": str(payload.get("id") or ""),
                    "input.value": _stringify_json(payload.get("args", {})),
                }
            )

        elif event_type in {"chat.error", "team.error"}:
            span_name = "Huawei Error"
            status_code = "Error"
            status_message = _extract_huawei_error(payload)
            attrs["output.value"] = status_message

        elif event_type == "chat.final":
            span_name = "LiteLLMModel.__call__"
            content = str(payload.get("content") or payload.get("message") or "")
            if content and "FINAL ANSWER:" not in content.upper():
                content = f"FINAL ANSWER: {content}"
            attrs["llm.output_messages.0.message.content"] = content
            attrs["output.value"] = content

        else:
            attrs["output.value"] = _stringify_json(payload)

        flat_spans.append(
            {
                "timestamp": timestamp,
                "trace_id": trace_id,
                "span_id": span_id,
                "parent_span_id": prev_id,
                "trace_state": "",
                "span_name": span_name,
                "span_kind": span_kind,
                "service_name": "huawei-agenttrace",
                "resource_attributes": {
                    "service.name": "huawei-agenttrace",
                    "source.module": str(meta.get("source_module") or ""),
                },
                "scope_name": "huawei.event.adapter",
                "scope_version": "1.0.0",
                "span_attributes": attrs,
                "duration": "",
                "status_code": status_code,
                "status_message": status_message,
                "events": [],
                "links": [],
                "logs": [],
                "child_spans": [],
            }
        )
        prev_id = span_id

    if not flat_spans:
        return None

    id_to_span = {span["span_id"]: span for span in flat_spans}
    roots = []
    for span in flat_spans:
        pid = span.get("parent_span_id")
        if pid and pid in id_to_span and pid != span.get("span_id"):
            id_to_span[pid].setdefault("child_spans", []).append(span)
        else:
            roots.append(span)

    return {"trace_id": trace_id, "spans": roots}


def _normalize_gaia_like_trace(obj):
    if not isinstance(obj, dict):
        return None
    spans = obj.get("spans")
    if not isinstance(spans, list):
        return None
    if not spans:
        return {"trace_id": obj.get("trace_id"), "spans": []}

    first = spans[0]
    if isinstance(first, dict) and ("span_id" in first or "child_spans" in first):
        return obj
    return None


def _normalize_otlp_trace(obj):
    if not isinstance(obj, dict):
        return None

    resource_spans = _first_present(obj, "resourceSpans", "resource_spans")
    if not isinstance(resource_spans, list):
        return None

    flat_spans = []
    trace_id = None

    for rs in resource_spans:
        if not isinstance(rs, dict):
            continue

        resource = rs.get("resource", {}) if isinstance(rs.get("resource"), dict) else {}
        resource_attrs = _decode_otel_attributes(resource.get("attributes", []))
        service_name = str(resource_attrs.get("service.name", ""))
        scope_spans = _first_present(rs, "scopeSpans", "scope_spans", default=[])

        for ss in scope_spans:
            if not isinstance(ss, dict):
                continue
            scope_info = ss.get("scope", {}) if isinstance(ss.get("scope"), dict) else {}
            scope_name = scope_info.get("name", "")
            scope_version = scope_info.get("version", "")

            for sp in ss.get("spans", []):
                if not isinstance(sp, dict):
                    continue

                span_id = _first_present(sp, "spanId", "span_id")
                parent_span_id = _first_present(sp, "parentSpanId", "parent_span_id") or None
                otel_trace_id = _first_present(sp, "traceId", "trace_id")
                if trace_id is None and otel_trace_id:
                    trace_id = otel_trace_id

                start_ns = _first_present(sp, "startTimeUnixNano", "start_time_unix_nano")
                end_ns = _first_present(sp, "endTimeUnixNano", "end_time_unix_nano")
                duration_iso = ""
                try:
                    duration_iso = _duration_ns_to_iso8601(int(str(end_ns)) - int(str(start_ns)))
                except Exception:
                    duration_iso = ""

                status = sp.get("status", {}) if isinstance(sp.get("status"), dict) else {}
                status_code = status.get("code", "Unset")
                if isinstance(status_code, int):
                    status_code = {0: "Unset", 1: "Ok", 2: "Error"}.get(status_code, str(status_code))

                span_kind = sp.get("kind", "Internal")
                if isinstance(span_kind, int):
                    span_kind = {
                        0: "Internal",
                        1: "Internal",
                        2: "Server",
                        3: "Client",
                        4: "Producer",
                        5: "Consumer",
                    }.get(span_kind, str(span_kind))

                span_attributes = _decode_otel_attributes(sp.get("attributes", []))
                if scope_name:
                    span_attributes.setdefault("otel.scope.name", scope_name)
                if scope_version:
                    span_attributes.setdefault("otel.scope.version", scope_version)
                span_name, span_attributes = _enrich_gen_ai_span(sp.get("name", ""), span_attributes)

                events = []
                for ev in sp.get("events", []) or []:
                    if not isinstance(ev, dict):
                        continue
                    events.append(
                        {
                            "name": ev.get("name", ""),
                            "timestamp": _ns_to_iso(_first_present(ev, "timeUnixNano", "time_unix_nano")),
                            "attributes": _decode_otel_attributes(ev.get("attributes", [])),
                        }
                    )
                logs = []
                tool_result = span_attributes.get("output.value") if span_attributes.get("tool.name") else None
                if tool_result not in [None, ""]:
                    logs.append(
                        {
                            "timestamp": _ns_to_iso(start_ns),
                            "body": {
                                "function.output": tool_result,
                            },
                        }
                    )

                normalized_status_code = status_code
                normalized_status_message = status.get("message", "")
                # The instrumentor records the authoritative OTel span status.
                # Do not downgrade a successful tool merely because its result
                # is an incident/log document containing words such as
                # "timeout" or "failed".  Textual fallback remains useful for
                # legacy traces whose tool status is unset.
                status_is_unset = str(status_code or "").strip().lower() in {
                    "", "unset", "status_code_unset", "status_code_unspecified",
                }
                if (
                    span_attributes.get("tool.name")
                    and status_is_unset
                    and _looks_like_tool_error(str(tool_result or ""))
                ):
                    normalized_status_code = "Error"
                    normalized_status_message = str(tool_result or "")[:1200]

                flat_spans.append(
                    {
                        "timestamp": _ns_to_iso(start_ns),
                        "trace_id": otel_trace_id,
                        "span_id": span_id,
                        "parent_span_id": parent_span_id,
                        "trace_state": "",
                        "span_name": span_name,
                        "span_kind": span_kind,
                        "service_name": service_name,
                        "resource_attributes": resource_attrs,
                        "scope_name": scope_name,
                        "scope_version": scope_version,
                        "span_attributes": span_attributes,
                        "duration": duration_iso,
                        "status_code": normalized_status_code,
                        "status_message": normalized_status_message,
                        "events": events,
                        "links": [],
                        "logs": logs,
                        "child_spans": [],
                    }
                )

    if not flat_spans:
        return None

    id_to_span = {}
    roots = []
    for span in flat_spans:
        sid = span.get("span_id")
        if sid:
            id_to_span[sid] = span

    for span in flat_spans:
        pid = span.get("parent_span_id")
        if pid and pid in id_to_span and pid != span.get("span_id"):
            id_to_span[pid].setdefault("child_spans", []).append(span)
        else:
            roots.append(span)

    def _sort_children(items):
        items.sort(key=lambda x: x.get("timestamp") or "")
        for it in items:
            children = it.get("child_spans", [])
            if isinstance(children, list) and children:
                _sort_children(children)

    _sort_children(roots)
    return {"trace_id": trace_id, "spans": roots}


def _dialogue_history(obj):
    if not isinstance(obj, dict):
        return None
    for key in ("history", "conversation", "trajectory", "messages"):
        value = obj.get(key)
        if isinstance(value, list) and value:
            return value
    return None


def _dialogue_task(obj):
    task = obj.get("task")
    if isinstance(task, dict):
        for key in ("query", "question", "prompt", "instruction", "description"):
            if task.get(key) not in (None, ""):
                return str(task[key])
    elif task not in (None, ""):
        return str(task)
    for key in ("question", "query", "prompt", "instruction"):
        if obj.get(key) not in (None, ""):
            return str(obj[key])
    return ""


def _dialogue_expected_answer(obj):
    task = obj.get("task")
    if isinstance(task, dict):
        for key in ("gold_answer", "ground_truth", "answer", "reference_answer"):
            if task.get(key) not in (None, ""):
                return str(task[key])
    ground_truth = obj.get("ground_truth")
    if ground_truth not in (None, "") and not isinstance(ground_truth, dict):
        return str(ground_truth)
    for key in ("reference_answer", "gold_answer", "answer"):
        if obj.get(key) not in (None, ""):
            return str(obj[key])
    return ""


def _dialogue_content(message):
    if not isinstance(message, dict):
        return str(message)
    value = _first_present(message, "content", "text", "message", "output", "response", default="")
    if isinstance(value, str):
        return value
    if value in (None, ""):
        return ""
    return _stringify_json(value)


def _canonical_dialogue_agent(message, index):
    raw_role = str(message.get("role") or "") if isinstance(message, dict) else ""
    candidate = _first_present(
        message,
        "name",
        "agent",
        "agent_name",
        "speaker",
        "sender",
        default="",
    ) if isinstance(message, dict) else ""
    candidate = str(candidate or "").strip()
    if not candidate and raw_role.lower() not in {"assistant", "user", "human", "system", "tool"}:
        candidate = raw_role.strip()
    if not candidate:
        candidate = {
            "assistant": "Assistant",
            "human": "Human",
            "user": "User",
            "system": "System",
            "tool": "Tool",
        }.get(raw_role.lower(), f"Participant_{index}")
    canonical = re.sub(r"\s*\([^)]*(?:->|thought)[^)]*\)\s*$", "", candidate, flags=re.IGNORECASE).strip()
    return canonical or candidate, candidate, raw_role


def _is_dialogue_control_message(content):
    normalized = re.sub(r"\s+", " ", str(content or "")).strip().lower()
    return bool(
        normalized in {"terminate", "done", "stop", "next speaker"}
        or normalized.startswith("next speaker ")
    )


def _normalize_dialogue_history_trace(obj):
    if not isinstance(obj, dict):
        return None
    history = _dialogue_history(obj)
    if not history:
        return None

    qid = str(
        _first_present(
            obj,
            "question_ID",
            "trace_id",
            "id",
            "task_id",
            "example_id",
            default="",
        )
        or ""
    )
    question = _dialogue_task(obj)
    expected_answer = _dialogue_expected_answer(obj)
    if not qid:
        qid = hashlib.md5(question.encode("utf-8", errors="ignore")).hexdigest()

    # Dialogue datasets usually have ordered turns but no wall-clock time.
    # Use a deterministic synthetic epoch so repeated conversions are byte-stable.
    base_ts = datetime.fromtimestamp(0, tz=timezone.utc)
    spans = []
    root_id = "root000000000001"

    root_attrs = {
        "input.value": json.dumps({"task": question}, ensure_ascii=False),
        "task.level": str(obj.get("level", "")),
        "task.ground_truth": expected_answer,
        "task.is_correct": bool(obj.get("is_correct", False)),
        "dataset.name": str(obj.get("dataset") or obj.get("dataset_name") or "dialogue"),
        "dataset.subset": str(obj.get("subset") or obj.get("split") or ""),
    }

    root_span = {
        "timestamp": base_ts.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "trace_id": qid,
        "span_id": root_id,
        "parent_span_id": None,
        "trace_state": "",
        "span_name": "CodeAgent.run",
        "span_kind": "Internal",
        "service_name": "nl-dialog-trace",
        "resource_attributes": {"service.name": "nl-dialog-trace"},
        "scope_name": "dialogue.adapter",
        "scope_version": "1.0.0",
        "span_attributes": root_attrs,
        "duration": "",
        "status_code": "Unset",
        "status_message": "",
        "events": [],
        "links": [],
        "logs": [],
        "child_spans": [],
    }
    spans.append(root_span)

    for idx, h in enumerate(history):
        if not isinstance(h, dict):
            continue
        agent, raw_agent, role = _canonical_dialogue_agent(h, idx)
        content = _dialogue_content(h)

        span_id = f"dlg{idx + 1:013d}"[-16:]
        ts = base_ts.timestamp() + idx * 0.1
        ts_iso = (
            datetime.fromtimestamp(ts, tz=timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )

        attrs = {
            "chat.role": role,
            "chat.name": raw_agent,
            "agent.id": agent,
            "agent.name": agent,
            "conversation.step_index": idx,
            "conversation.step_number": idx + 1,
            "conversation.raw_agent": raw_agent,
            "output.value": content,
        }

        span_name = "Step Dialogue"
        role_lower = role.lower()
        agent_lower = agent.lower()
        is_terminal = agent_lower in {"computer_terminal", "computerterminal", "terminal", "tool"}
        is_human = role_lower in {"human"} or agent_lower == "human"
        if not is_terminal and not is_human and not _is_dialogue_control_message(content):
            span_name = "LiteLLMModel.__call__"
            attrs["llm.output_messages.0.message.content"] = content
        elif is_terminal or content.lower().startswith("exitcode"):
            span_name = f"Step {idx + 1}"

        span = {
            "timestamp": ts_iso,
            "trace_id": qid,
            "span_id": span_id,
            "parent_span_id": root_id,
            "trace_state": "",
            "span_name": span_name,
            "span_kind": "Internal",
            "service_name": "nl-dialog-trace",
            "resource_attributes": {"service.name": "nl-dialog-trace"},
            "scope_name": "dialogue.adapter",
            "scope_version": "1.0.0",
            "span_attributes": attrs,
            "duration": "",
            "status_code": "Unset",
            "status_message": "",
            "events": [],
            "links": [],
            "logs": [],
            "child_spans": [],
        }
        spans.append(span)

    id_to_span = {s["span_id"]: s for s in spans}
    roots = []
    for span in spans:
        pid = span.get("parent_span_id")
        if pid and pid in id_to_span and pid != span.get("span_id"):
            id_to_span[pid].setdefault("child_spans", []).append(span)
        else:
            roots.append(span)

    source_metadata = {
        "dataset": str(obj.get("dataset") or obj.get("dataset_name") or "dialogue"),
        "subset": str(obj.get("subset") or obj.get("split") or ""),
        "question": question,
        "ground_truth_answer": expected_answer,
        "history_length": len(history),
    }
    return {"trace_id": qid, "spans": roots, "source_metadata": source_metadata}


def find_first_trace_obj(obj):
    if isinstance(obj, dict):
        if "resourceSpans" in obj and isinstance(obj["resourceSpans"], list):
            return obj
        if _dialogue_history(obj):
            return obj
        if "spans" in obj and isinstance(obj["spans"], list):
            return obj
        for value in obj.values():
            result = find_first_trace_obj(value)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for value in obj:
            result = find_first_trace_obj(value)
            if result is not None:
                return result
    elif isinstance(obj, str):
        parsed = _safe_json_parse_text(obj)
        if parsed is not None:
            return find_first_trace_obj(parsed)
    return None


def normalize_trace_for_parser(obj):
    normalized = _normalize_gaia_like_trace(obj)
    if normalized is not None:
        return normalized

    normalized = _normalize_otlp_trace(obj)
    if normalized is not None:
        return normalized

    normalized = _normalize_dialogue_history_trace(obj)
    if normalized is not None:
        return normalized

    normalized = _normalize_huawei_event_trace(obj)
    if normalized is not None:
        return normalized

    candidate = find_first_trace_obj(obj)
    if candidate is not None and candidate is not obj:
        return normalize_trace_for_parser(candidate)
    return candidate
