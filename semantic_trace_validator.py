"""Generic, deterministic validation for every tool call in an Agent trace.

The core validator contains no task- or website-specific knowledge.  It checks
transport/business status, argument shape, required parameters, result
presence, temporal constraints, provenance of opaque arguments, and whether a
result is consumed or explicitly ignored downstream.  Domain knowledge can be
added through optional rule callables without changing the core pipeline.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse


ValidationRule = Callable[["ToolCallContext"], List[dict]]

PROPAGATION_STATES = {
    "NOT_EXPOSED",
    "EXPOSED",
    "PROPAGATED",
    "TRANSFORMED",
    "REJECTED",
    "UNKNOWN",
}

_FULL_DATE_RE = re.compile(
    r"(?P<year>20\d{2})[年\-/\.](?P<month>\d{1,2})[月\-/\.](?P<day>\d{1,2})日?"
)
_MONTH_DAY_RE = re.compile(r"(?<!\d)(?P<month>\d{1,2})月(?P<day>\d{1,2})日")
_HTTP_STATUS_RE = re.compile(r"^Status:\s*(?P<status>\d{3})\b", re.M | re.I)
_STRONG_TOOL_ERROR_RE = re.compile(
    r"(?:^|\n)\s*(?:file system operation execution error\b|"
    r"error when executing tool\b|tool execution (?:error|failed)\b|"
    r"execution failed(?:\s+at|\s*:)|\[error\])",
    re.I,
)
_OPAQUE_URL_TOKEN_RE = re.compile(r"(?<=[/=])([A-Za-z0-9_-]{7,})(?=[/?#.&]|$)")
_MEASURE_RE = re.compile(
    r"(?<!\w)[<>]?\d+(?:\.\d+)?\s*(?:℃|°C|%|级|km|m|kg|g|ms|s|秒|分钟|小时|元|美元|MB|GB)",
    re.I,
)
_TASK_TERM_RE = re.compile(
    r"(?:查询|搜索|查看|获取|分析|读取|比较|检查|打开|下载|调用|验证)"
    r"(?P<term>[\u4e00-\u9fffA-Za-z0-9_-]{2,16}?)"
    r"(?:今天|今日|当前|的|数据|信息|情况|文件|网页|页面|报告|内容|结果|$)"
)
_IGNORE_TERMS = (
    r"(?:不可靠|忽略|未采用|不采用|不使用|无关|过期|discard|unreliable|"
    r"outdated|doesn['’]t\s+match|ignored|not\s+used)"
)


@dataclass
class ToolCallContext:
    span: dict
    span_index: int
    spans: Sequence[dict]
    task_context: str
    tool_name: str
    arguments: Any
    parsed_arguments: Optional[dict]
    result: Any
    result_text: str
    origin_span_id: Optional[str]
    prior_tool_evidence: str
    consumer_links: List[dict]


def _parse_mapping(value: Any) -> Optional[dict]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _meaningful_error(value: Any) -> bool:
    return value not in (None, "", False, [], {})


def classify_tool_result(result: Any, span_status: str = "") -> dict:
    """Classify execution status without treating ``error=None`` as failure."""

    transport_success = str(span_status or "").strip().lower() not in {
        "error",
        "status_code_error",
    }
    parsed = _parse_mapping(result)
    business_success: Optional[bool] = None
    error_value: Any = None
    if parsed is not None:
        if isinstance(parsed.get("success"), bool):
            business_success = parsed["success"]
        error_value = parsed.get("error")

    is_error = not transport_success or business_success is False or _meaningful_error(error_value)
    text = str(result or "")
    # A successful OTel tool span may legitimately return an incident/log
    # document containing words such as "timeout" or "failed".  Textual
    # fallback is therefore restricted to legacy records whose span status is
    # absent; an explicit OK status is authoritative.
    status_is_unset = str(span_status or "").strip().lower() in {
        "", "unset", "status_code_unset", "status_code_unspecified",
    }
    if parsed is None:
        if _STRONG_TOOL_ERROR_RE.search(text):
            is_error = True
        elif status_is_unset and re.search(
            r"(?:\btraceback\b|\bexception\b|\bsuccess\s*[:=]\s*false\b|"
            r"\bfailed\b|\btimeout\b)",
            text,
            flags=re.I,
        ):
            is_error = True
    return {
        "transport_success": transport_success,
        "business_success": business_success,
        "semantic_valid": None,
        "is_error": is_error,
        "error": error_value if _meaningful_error(error_value) else None,
    }


def _flatten_spans(spans: Iterable[dict]) -> List[dict]:
    flattened: List[dict] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        flattened.append(span)
        children = span.get("child_spans", [])
        if isinstance(children, list):
            flattened.extend(_flatten_spans(children))
    flattened.sort(key=lambda item: str(item.get("timestamp") or ""))
    return flattened


def _tool_span(span: dict) -> bool:
    attrs = span.get("span_attributes", {}) if isinstance(span, dict) else {}
    return isinstance(attrs, dict) and bool(attrs.get("tool.name"))


def _ancestor_task_context(span: dict, span_by_id: Dict[str, dict]) -> str:
    current = span_by_id.get(str(span.get("parent_span_id") or ""))
    visited: Set[str] = set()
    while isinstance(current, dict):
        attrs = current.get("span_attributes", {})
        text = attrs.get("input.value", "") if isinstance(attrs, dict) else ""
        if isinstance(text, str) and text.strip():
            return text
        parent_id = str(current.get("parent_span_id") or "")
        if not parent_id or parent_id in visited:
            break
        visited.add(parent_id)
        current = span_by_id.get(parent_id)
    return ""


def _extract_url(arguments: Any) -> str:
    parsed = _parse_mapping(arguments)
    if parsed and isinstance(parsed.get("url"), str):
        return parsed["url"]
    match = re.search(r"https?://[^\s'\"}]+", str(arguments or ""))
    return match.group(0) if match else ""


def _find_tool_origin_span(
    spans: Sequence[dict], span_index: int, tool_span: dict, arguments: Any
) -> Optional[str]:
    attrs = tool_span.get("span_attributes", {})
    tool_id = str(attrs.get("tool.id") or "") if isinstance(attrs, dict) else ""
    argument_text = str(arguments or "")
    for earlier in reversed(spans[:span_index]):
        earlier_attrs = earlier.get("span_attributes", {}) if isinstance(earlier, dict) else {}
        if not isinstance(earlier_attrs, dict):
            continue
        raw_output = earlier_attrs.get("gen_ai.output.messages")
        if raw_output in (None, ""):
            continue
        try:
            haystack = json.dumps(raw_output, ensure_ascii=False)
        except TypeError:
            haystack = str(raw_output)
        if (tool_id and tool_id in haystack) or (argument_text and argument_text in haystack):
            return earlier.get("span_id")
    return None


def _extract_requested_date(text: str) -> Optional[str]:
    match = _FULL_DATE_RE.search(str(text or ""))
    if not match:
        return None
    try:
        return date(
            int(match.group("year")), int(match.group("month")), int(match.group("day"))
        ).isoformat()
    except ValueError:
        return None


def _extract_dates(text: str) -> Set[str]:
    result: Set[str] = set()
    for match in _FULL_DATE_RE.finditer(str(text or "")):
        try:
            result.add(
                date(
                    int(match.group("year")),
                    int(match.group("month")),
                    int(match.group("day")),
                ).isoformat()
            )
        except ValueError:
            continue
    return result


def _contains_requested_date(text: str, requested: str) -> bool:
    try:
        parsed = date.fromisoformat(requested)
    except (TypeError, ValueError):
        return False
    variants = {
        requested,
        f"{parsed.year}年{parsed.month}月{parsed.day}日",
        f"{parsed.month}月{parsed.day}日",
    }
    return any(item in str(text or "") for item in variants)


def _task_terms(text: str) -> Set[str]:
    terms = {match.group("term") for match in _TASK_TERM_RE.finditer(str(text or ""))}
    for quoted in re.findall(r"[\"“]([^\"”]{2,40})[\"”]", str(text or "")):
        for token in re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_-]{2,}", quoted):
            terms.add(token)
    return {term for term in terms if term not in {"相关", "详细", "一下", "一下子"}}


def _opaque_argument_tokens(arguments: Any) -> Set[str]:
    url = _extract_url(arguments)
    candidates = set(_OPAQUE_URL_TOKEN_RE.findall(url)) if url else set()
    tokens = {
        token
        for token in candidates
        if (len(token) >= 7 and any(char.isdigit() for char in token)) or len(token) >= 16
    }
    parsed = _parse_mapping(arguments) or {}
    for key, value in parsed.items():
        if re.search(r"(?:^|_)(?:id|code|key|path|resource)(?:$|_)", str(key), flags=re.I):
            text = str(value)
            if len(text) >= 7 and re.fullmatch(r"[A-Za-z0-9_-]+", text):
                tokens.add(text)
    return tokens


def _fact_signatures(text: str) -> Set[str]:
    signatures = {re.sub(r"\s+", "", item) for item in _MEASURE_RE.findall(str(text or ""))}
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"[*#|]", "", raw_line).strip()
        if not 2 <= len(line) <= 24:
            continue
        if re.match(r"^(?:URL|Status|Title|Provider|Content|Query|Source):", line, flags=re.I):
            continue
        if re.search(r"[\u4e00-\u9fffA-Za-z]", line):
            signatures.add(re.sub(r"\s+", "", line))
    return signatures


def _parse_messages(value: Any) -> List[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _message_text(message: dict) -> str:
    parts: List[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    for part in message.get("parts", []) or []:
        if isinstance(part, dict) and isinstance(part.get("content"), str):
            parts.append(part["content"])
    return "\n".join(parts)


def _model_span(span: dict) -> bool:
    attrs = span.get("span_attributes", {}) if isinstance(span, dict) else {}
    return isinstance(attrs, dict) and (
        attrs.get("otel.original_span_name") == "gen_ai.chat"
        or span.get("span_name") == "LiteLLMModel.__call__"
    )


def _model_output_text(span: dict) -> str:
    attrs = span.get("span_attributes", {}) if isinstance(span, dict) else {}
    if not isinstance(attrs, dict):
        return ""
    direct = attrs.get("llm.output_messages.0.message.content")
    if isinstance(direct, str) and direct.strip():
        return direct
    messages = _parse_messages(attrs.get("gen_ai.output.messages"))
    message_text = "\n".join(_message_text(item) for item in messages if item.get("role") == "assistant")
    if message_text.strip():
        return message_text
    return str(attrs.get("output.value") or "")


def _input_tool_results(span: dict) -> Dict[str, str]:
    attrs = span.get("span_attributes", {}) if isinstance(span, dict) else {}
    messages = _parse_messages(attrs.get("gen_ai.input.messages")) if isinstance(attrs, dict) else []
    results: Dict[str, str] = {}
    for message in messages:
        tool_call_id = str(message.get("tool_call_id") or "")
        if message.get("role") == "tool" and tool_call_id:
            results[tool_call_id] = _message_text(message)
    return results


def _output_has_tool_calls(span: dict) -> bool:
    attrs = span.get("span_attributes", {}) if isinstance(span, dict) else {}
    messages = _parse_messages(attrs.get("gen_ai.output.messages")) if isinstance(attrs, dict) else []
    return any(message.get("tool_calls") for message in messages)


def _same_branch_consumers(spans: Sequence[dict], span_index: int, tool_span: dict) -> List[dict]:
    parent_id = str(tool_span.get("parent_span_id") or "")
    return [
        later
        for later in spans[span_index + 1 :]
        if _model_span(later) and str(later.get("parent_span_id") or "") == parent_id
    ]


def _claim_links(
    spans: Sequence[dict],
    span_index: int,
    tool_span: dict,
    result_text: str,
    exact_consumers: Sequence[dict],
) -> List[dict]:
    signatures = _fact_signatures(result_text)
    url = _extract_url((tool_span.get("span_attributes") or {}).get("input.value", ""))
    source_host = urlparse(url).netloc.lower().removeprefix("www.") if url else ""
    attrs = tool_span.get("span_attributes", {}) or {}
    tool_call_id = str(attrs.get("tool.id") or "")
    candidates: List[Tuple[dict, str]] = [(span, "tool_call_id") for span in exact_consumers]
    if not candidates:
        candidates = [(span, "same_branch_order") for span in _same_branch_consumers(spans, span_index, tool_span)]
    links: List[dict] = []
    for later, delivery_method in candidates:
        output = _model_output_text(later)
        if not output.strip():
            continue
        normalized_output = re.sub(r"\s+", "", output)
        overlap = sorted(signature for signature in signatures if signature in normalized_output)
        ignored = False
        if source_host:
            ignored = bool(
                re.search(
                    rf"(?:{re.escape(source_host)}.{{0,160}}{_IGNORE_TERMS}|"
                    rf"{_IGNORE_TERMS}.{{0,160}}{re.escape(source_host)})",
                    output,
                    flags=re.I | re.S,
                )
            )
        if not ignored and not source_host:
            for signature in overlap[:12]:
                ignored = bool(
                    re.search(
                        rf"(?:{re.escape(signature)}.{{0,100}}{_IGNORE_TERMS}|"
                        rf"{_IGNORE_TERMS}.{{0,100}}{re.escape(signature)})",
                        normalized_output,
                        flags=re.I | re.S,
                    )
                )
                if ignored:
                    break
        if ignored:
            state = "REJECTED"
        elif len(overlap) >= 3:
            state = "PROPAGATED"
        elif delivery_method == "tool_call_id" and len(overlap) >= 1:
            state = "TRANSFORMED"
        elif delivery_method == "tool_call_id":
            state = "EXPOSED"
        else:
            # Ordering alone is not enough to prove delivery or non-use.
            continue
        confidence = min(1.0, 0.45 + 0.1 * len(overlap))
        if delivery_method == "tool_call_id":
            confidence = max(confidence, 0.9 if state in {"EXPOSED", "REJECTED"} else 0.95)
        elif state == "PROPAGATED":
            confidence = min(confidence, 0.82)
        links.append(
            {
                "source_span_id": tool_span.get("span_id"),
                "consumer_span_id": later.get("span_id"),
                "tool_call_id": tool_call_id or None,
                "delivery_method": delivery_method,
                "state": state,
                "matched_signatures": overlap[:12],
                "confidence": round(confidence, 2),
                "relation": (
                    "ignored"
                    if state == "REJECTED"
                    else "consumed"
                    if state in {"PROPAGATED", "TRANSFORMED"}
                    else "exposed"
                ),
            }
        )
    return links


def _final_answer_span_ids(spans: Sequence[dict], span_by_id: Dict[str, dict]) -> List[str]:
    candidates: List[dict] = []
    for span in spans:
        if not _model_span(span) or not _model_output_text(span).strip() or _output_has_tool_calls(span):
            continue
        ancestor_ids = _ancestor_span_ids(span, span_by_id)
        if any(_tool_span(span_by_id.get(parent_id, {})) for parent_id in ancestor_ids):
            continue
        candidates.append(span)
    if not candidates:
        candidates = [span for span in spans if _model_span(span) and _model_output_text(span).strip() and not _output_has_tool_calls(span)]
    if not candidates:
        return []
    latest = max(candidates, key=lambda item: str(item.get("timestamp") or ""))
    return [str(latest.get("span_id"))]


def _subagent_return_links(spans: Sequence[dict], span_by_id: Dict[str, dict]) -> List[dict]:
    links: List[dict] = []
    for wrapper in spans:
        if not _tool_span(wrapper):
            continue
        result_text = str((wrapper.get("span_attributes", {}) or {}).get("output.value") or "")
        if not result_text:
            continue
        wrapper_id = str(wrapper.get("span_id") or "")
        descendants = [
            span
            for span in spans
            if _model_span(span) and wrapper_id in _ancestor_span_ids(span, span_by_id)
        ]
        for model in sorted(descendants, key=lambda item: str(item.get("timestamp") or ""), reverse=True):
            output = _model_output_text(model)
            signatures = _fact_signatures(output)
            overlap = sorted(item for item in signatures if item in re.sub(r"\s+", "", result_text))
            if len(overlap) < 3 and output.strip() not in result_text:
                continue
            links.append(
                {
                    "source_span_id": model.get("span_id"),
                    "consumer_span_id": wrapper.get("span_id"),
                    "tool_call_id": (wrapper.get("span_attributes", {}) or {}).get("tool.id"),
                    "delivery_method": "subagent_return",
                    "state": "PROPAGATED",
                    "matched_signatures": overlap[:12],
                    "confidence": 0.96,
                    "relation": "consumed",
                }
            )
            break
    return links


def _propagation_report(
    source_span_id: str,
    links: Sequence[dict],
    final_answer_ids: Sequence[str],
) -> dict:
    graph: Dict[str, List[dict]] = defaultdict(list)
    for link in links:
        if link.get("state") in {"PROPAGATED", "TRANSFORMED"}:
            graph[str(link.get("source_span_id") or "")].append(link)
    queue = deque([(source_span_id, [source_span_id], 1.0)])
    visited = {source_span_id}
    finals = set(final_answer_ids)
    while queue:
        node_id, path, confidence = queue.popleft()
        if node_id in finals:
            return {
                "source_span_id": source_span_id,
                "state": "PROPAGATED",
                "reaches_final_answer": True,
                "final_answer_span_id": node_id,
                "path": path,
                "confidence": round(confidence, 4),
            }
        for link in graph.get(node_id, []):
            target = str(link.get("consumer_span_id") or "")
            if not target or target in visited:
                continue
            visited.add(target)
            queue.append((target, path + [target], min(confidence, float(link.get("confidence") or 0.0))))
    source_links = [link for link in links if str(link.get("source_span_id") or "") == source_span_id]
    states = {str(link.get("state") or "UNKNOWN") for link in source_links}
    if "REJECTED" in states:
        state = "REJECTED"
    elif "EXPOSED" in states:
        state = "EXPOSED"
    elif states & {"PROPAGATED", "TRANSFORMED"}:
        state = "PROPAGATED"
    elif source_links:
        state = "UNKNOWN"
    else:
        state = "NOT_EXPOSED"
    return {
        "source_span_id": source_span_id,
        "state": state,
        "reaches_final_answer": False,
        "final_answer_span_id": None,
        "path": [],
        "confidence": max((float(link.get("confidence") or 0.0) for link in source_links), default=1.0 if state == "NOT_EXPOSED" else 0.0),
    }


def _ancestor_span_ids(span: dict, span_by_id: Dict[str, dict]) -> Set[str]:
    result: Set[str] = set()
    parent_id = str(span.get("parent_span_id") or "")
    while parent_id and parent_id not in result:
        result.add(parent_id)
        parent = span_by_id.get(parent_id)
        if not parent:
            break
        parent_id = str(parent.get("parent_span_id") or "")
    return result


def _prior_tool_evidence(
    spans: Sequence[dict], span_index: int, current_span: dict, span_by_id: Dict[str, dict]
) -> str:
    parts: List[str] = []
    ancestors = _ancestor_span_ids(current_span, span_by_id)
    for earlier in spans[:span_index]:
        if not _tool_span(earlier):
            continue
        # A wrapper/parent tool often records its final result at span start in
        # normalized traces.  Counting it here would leak future child output
        # into argument provenance.
        if str(earlier.get("span_id") or "") in ancestors:
            continue
        attrs = earlier.get("span_attributes", {})
        output = attrs.get("output.value", "") if isinstance(attrs, dict) else ""
        if output:
            parts.append(str(output))
    return "\n".join(parts)[-30000:]


def _required_argument_violations(context: ToolCallContext) -> List[dict]:
    attrs = context.span.get("span_attributes", {})
    schema = _parse_mapping(attrs.get("tool.parameters")) if isinstance(attrs, dict) else None
    if not schema or not isinstance(schema.get("required"), list):
        return []
    args = context.parsed_arguments or {}
    missing = [name for name in schema["required"] if name not in args]
    if not missing:
        return []
    return [
        {
            "violation_type": "missing_required_argument",
            "dimension": "argument_schema",
            "expected": schema["required"],
            "observed": sorted(args),
            "evidence": str(context.arguments)[:500],
            "severity": "HIGH",
            "confidence": 1.0,
        }
    ]


def _explicit_field_consistency_violations(context: ToolCallContext) -> List[dict]:
    result = _parse_mapping(context.result)
    args = context.parsed_arguments or {}
    if not result:
        return []
    comparable = {"entity", "location", "city", "resource_id", "document_id", "user_id"}
    violations: List[dict] = []
    for key in comparable & args.keys() & result.keys():
        if str(args[key]).strip().casefold() == str(result[key]).strip().casefold():
            continue
        violations.append(
            {
                "violation_type": "argument_result_mismatch",
                "dimension": key,
                "expected": args[key],
                "observed": result[key],
                "evidence": f"argument.{key}={args[key]!r}, result.{key}={result[key]!r}",
                "severity": "HIGH",
                "confidence": 0.99,
            }
        )
    return violations


class ToolCallValidator:
    """Validate all tool calls using generic built-ins plus optional rules."""

    def __init__(self, rules: Optional[Sequence[ValidationRule]] = None):
        self.rules = list(rules or [])

    def validate_trace(self, trace_data: dict) -> dict:
        roots = trace_data.get("spans", []) if isinstance(trace_data, dict) else []
        spans = _flatten_spans(roots)
        span_by_id = {str(span.get("span_id")): span for span in spans if span.get("span_id")}
        checks: List[dict] = []
        violations: List[dict] = []
        claim_links: List[dict] = []
        exact_consumers: Dict[str, List[dict]] = defaultdict(list)
        for candidate in spans:
            if not _model_span(candidate):
                continue
            for tool_call_id in _input_tool_results(candidate):
                exact_consumers[tool_call_id].append(candidate)
        final_answer_ids = _final_answer_span_ids(spans, span_by_id)

        for index, span in enumerate(spans):
            if not _tool_span(span):
                continue
            attrs = span.get("span_attributes", {})
            arguments = attrs.get("input.value", "")
            result = attrs.get("output.value", "")
            result_text = str(result or "")
            task_context = _ancestor_task_context(span, span_by_id)
            tool_call_id = str(attrs.get("tool.id") or "")
            links = _claim_links(
                spans,
                index,
                span,
                result_text,
                exact_consumers.get(tool_call_id, []),
            )
            claim_links.extend(links)
            consumed_by = list(
                dict.fromkeys(
                    link["consumer_span_id"]
                    for link in links
                    if link.get("state") in {"PROPAGATED", "TRANSFORMED"}
                )
            )
            context = ToolCallContext(
                span=span,
                span_index=index,
                spans=spans,
                task_context=task_context,
                tool_name=str(attrs.get("tool.name") or ""),
                arguments=arguments,
                parsed_arguments=_parse_mapping(arguments),
                result=result,
                result_text=result_text,
                origin_span_id=_find_tool_origin_span(spans, index, span, arguments),
                prior_tool_evidence=_prior_tool_evidence(spans, index, span, span_by_id),
                consumer_links=links,
            )
            status = classify_tool_result(result, span.get("status_code", ""))
            call_violations: List[dict] = []
            call_violations.extend(_required_argument_violations(context))
            call_violations.extend(_explicit_field_consistency_violations(context))

            if status["is_error"]:
                call_violations.append(
                    {
                        "violation_type": "tool_execution_failure",
                        "dimension": "execution_status",
                        "expected": "successful tool result",
                        "observed": status,
                        "evidence": result_text[:500],
                        "severity": "HIGH",
                        "confidence": 1.0,
                    }
                )
            elif result in (None, "", [], {}):
                call_violations.append(
                    {
                        "violation_type": "empty_tool_result",
                        "dimension": "result_presence",
                        "expected": "non-empty result",
                        "observed": "empty",
                        "evidence": str(result),
                        "severity": "MEDIUM",
                        "confidence": 1.0,
                    }
                )

            http_match = _HTTP_STATUS_RE.search(result_text)
            if http_match and int(http_match.group("status")) >= 400:
                call_violations.append(
                    {
                        "violation_type": "http_failure",
                        "dimension": "http_status",
                        "expected": "2xx or 3xx",
                        "observed": int(http_match.group("status")),
                        "evidence": result_text[:300],
                        "severity": "HIGH",
                        "confidence": 1.0,
                    }
                )

            requested_date = _extract_requested_date(task_context)
            observed_dates = _extract_dates(result_text)
            if requested_date and observed_dates and not _contains_requested_date(result_text, requested_date):
                call_violations.append(
                    {
                        "violation_type": "temporal_mismatch",
                        "dimension": "date",
                        "expected": requested_date,
                        "observed": sorted(observed_dates)[:20],
                        "evidence": result_text[:500],
                        "severity": "HIGH" if consumed_by else "MEDIUM",
                        "confidence": 0.96,
                    }
                )

            opaque_tokens = _opaque_argument_tokens(arguments)
            task_terms = _task_terms(task_context)
            ungrounded = sorted(
                token
                for token in opaque_tokens
                if token not in task_context and token not in context.prior_tool_evidence
            )
            target_echoed = not task_terms or any(term in result_text for term in task_terms)
            if ungrounded:
                call_violations.append(
                    {
                        "violation_type": "ungrounded_tool_argument",
                        "dimension": "argument_provenance",
                        "expected": "opaque identifiers must come from user input or prior tool evidence",
                        "observed": ungrounded,
                        "evidence": str(arguments)[:500],
                        "target_terms": sorted(task_terms),
                        "target_echoed_in_result": target_echoed,
                        "severity": "HIGH" if consumed_by and not target_echoed else "MEDIUM",
                        "confidence": 0.94 if not target_echoed else 0.78,
                    }
                )

            for rule in self.rules:
                call_violations.extend(rule(context) or [])

            for violation in call_violations:
                violation.setdefault("source_span_id", span.get("span_id"))
                violation.setdefault("origin_span_id", context.origin_span_id)
                violation.setdefault("consumer_span_ids", consumed_by)
                violation.setdefault("consumed_by_any", bool(consumed_by))
                violation.setdefault("reaches_final_answer", False)
                violation.setdefault("used_by_final", False)
                violations.append(violation)

            semantic_verdict = "invalid" if call_violations else "valid"
            if any(item["violation_type"] == "ungrounded_tool_argument" for item in call_violations):
                semantic_verdict = "uncertain"
            checks.append(
                {
                    "span_id": span.get("span_id"),
                    "origin_span_id": context.origin_span_id,
                    "tool": context.tool_name,
                    "task_context": task_context[:1200],
                    "arguments": arguments,
                    "result_preview": result_text[:1600],
                    "status": status,
                    "consumer_span_ids": consumed_by,
                    "consumed_by_any": bool(consumed_by),
                    "verdict": semantic_verdict,
                    "violation_types": [item["violation_type"] for item in call_violations],
                    "semantic_review_required": semantic_verdict == "uncertain",
                }
            )

        claim_links.extend(_subagent_return_links(spans, span_by_id))
        propagation_reports: List[dict] = []
        report_by_source: Dict[str, dict] = {}
        for source_span_id in dict.fromkeys(
            str(item.get("source_span_id") or "") for item in violations
        ):
            if not source_span_id:
                continue
            report = _propagation_report(source_span_id, claim_links, final_answer_ids)
            propagation_reports.append(report)
            report_by_source[source_span_id] = report

        for violation in violations:
            report = report_by_source.get(str(violation.get("source_span_id") or ""), {})
            reaches_final = bool(report.get("reaches_final_answer"))
            violation["reaches_final_answer"] = reaches_final
            violation["used_by_final"] = reaches_final
            violation["propagation_state"] = report.get("state", "UNKNOWN")
            violation["propagation_path"] = report.get("path", [])
            violation["final_reach_probability"] = (
                float(report.get("confidence") or 0.0) if reaches_final else 0.0
            )

        for check in checks:
            report = report_by_source.get(str(check.get("span_id") or ""), {})
            check["propagation_state"] = report.get("state", "NOT_EXPOSED")
            check["reaches_final_answer"] = bool(report.get("reaches_final_answer"))

        result = {
            "tool_call_checks": checks,
            "semantic_observations": checks,
            "semantic_violations": violations,
            "claim_evidence_links": claim_links,
            "delivery_links": claim_links,
            "propagation_reports": propagation_reports,
            "final_answer_span_ids": final_answer_ids,
        }
        from fault_taxonomy import detect_fault_signatures

        result["fault_taxonomy_findings"] = detect_fault_signatures(trace_data, result)
        return result


def analyze_semantic_trace(
    trace_data: dict,
    rules: Optional[Sequence[ValidationRule]] = None,
    **_: Any,
) -> dict:
    """Compatibility wrapper used by the graph and Judge pipelines."""

    result = ToolCallValidator(rules=rules).validate_trace(trace_data)
    result.setdefault("task_contracts", [])
    return result
