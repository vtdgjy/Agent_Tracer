"""Trace-grounded Agent failure signatures.

Deterministic signatures are deliberately conservative.  They identify
observable candidates and preserve evidence; semantic judgments such as
constraint omission or noisy evidence remain review tasks.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any


_FORMAT_ERROR = re.compile(
    r"format(?:ting)?\s*error|missing required tag|invalid (?:json|format)|"
    r"schema (?:error|mismatch)|protocol (?:error|violation)", re.I
)
_UNKNOWN_TOOL = re.compile(
    r"unknown tool|tool (?:is )?not (?:available|registered|found)|"
    r"no such tool|unsupported tool|tool does not exist", re.I
)
_FINAL = re.compile(r"final[_ ]?answer|finalanswer", re.I)
_ERROR_NAME = re.compile(r"(?:error|exception|failure|timeout|crash)", re.I)


def _flatten(spans: Any) -> list[dict]:
    result: list[dict] = []
    if not isinstance(spans, list):
        return result
    stack = list(reversed(spans))
    while stack:
        span = stack.pop()
        if not isinstance(span, dict):
            continue
        result.append(span)
        children = span.get("child_spans") or span.get("children") or []
        if isinstance(children, list):
            stack.extend(reversed(children))
    return result


def _text(span: dict) -> str:
    attrs = span.get("span_attributes") or span.get("attributes") or {}
    values = [span.get("span_name", ""), span.get("name", ""), span.get("status_code", "")]
    if isinstance(attrs, dict):
        values.extend(str(attrs.get(k, "")) for k in ("input.value", "output.value", "tool.name"))
    return " ".join(map(str, values))


def _finding(fid: str, name: str, category: str, evidence: list[str], confidence: float,
             explanation: str, review: bool = False) -> dict:
    return {
        "fault_id": fid,
        "name": name,
        "category": category,
        "matched": True,
        "confidence": confidence,
        "evidence_span_ids": list(dict.fromkeys(x for x in evidence if x)),
        "evidence": explanation,
        "evidence_kind": "trace_rule_candidate",
        "requires_review": review,
    }


def detect_fault_signatures(trace_data: dict, semantic: dict | None = None) -> list[dict]:
    """Return conservative F1-F12 candidates supported by available trace data."""
    semantic = semantic if isinstance(semantic, dict) else {}
    spans = _flatten(trace_data.get("spans", []) if isinstance(trace_data, dict) else [])
    violations = semantic.get("semantic_violations") or []
    checks = semantic.get("tool_call_checks") or []
    findings: list[dict] = []

    def add(fid: str, name: str, category: str, ids: list[str], confidence: float,
            why: str, review: bool = False) -> None:
        findings.append(_finding(fid, name, category, ids, confidence, why, review))

    violations_by_type: dict[str, list[dict]] = defaultdict(list)
    for item in violations:
        if isinstance(item, dict):
            violations_by_type[str(item.get("violation_type") or "")].append(item)

    execution = violations_by_type.get("tool_execution_failure", []) + violations_by_type.get("http_failure", [])
    if execution:
        add("F4", "动作执行崩溃", "tool_interaction", [str(x.get("source_span_id") or "") for x in execution],
            0.98, "工具调用状态或 HTTP 响应明确失败。")

    contract = violations_by_type.get("missing_required_argument", []) + violations_by_type.get("argument_result_mismatch", [])
    if contract:
        add("F9", "工具接口模式失配", "tool_interaction", [str(x.get("source_span_id") or "") for x in contract],
            0.97, "工具必需参数缺失，或参数与返回对象字段不一致。")

    ungrounded = violations_by_type.get("ungrounded_tool_argument", [])
    if ungrounded:
        add("F9-U", "工具参数来源不可追溯", "tool_interaction", [str(x.get("source_span_id") or "") for x in ungrounded],
            0.82, "参数中的不透明标识无法从任务或先前工具证据中追溯。", True)

    format_spans = [s for s in spans if _FORMAT_ERROR.search(_text(s))]
    if format_spans:
        add("F8", "格式协议失配", "tool_interaction", [str(s.get("span_id") or "") for s in format_spans],
            0.94, "Span 文本包含格式、协议或必需标签错误信号。")

    unknown_tool_spans = [s for s in spans if _UNKNOWN_TOOL.search(_text(s))]
    if unknown_tool_spans:
        add("F5", "工具幻觉或能力越界", "tool_interaction", [str(s.get("span_id") or "") for s in unknown_tool_spans],
            0.86, "环境明确报告工具未知、未注册或不可用。", True)

    # Group tool attempts by exact tool name and canonicalized input.  Similar
    # arguments are not treated as identical without a stable representation.
    attempts: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for span in spans:
        attrs = span.get("span_attributes") or span.get("attributes") or {}
        if not isinstance(attrs, dict) or not attrs.get("tool.name"):
            continue
        raw = attrs.get("input.value", "")
        try:
            args_key = json.dumps(json.loads(raw) if isinstance(raw, str) else raw,
                                  sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError, json.JSONDecodeError):
            args_key = str(raw).strip()
        attempts[(str(attrs.get("tool.name")), args_key)].append(span)
    repeated = [group for group in attempts.values() if len(group) >= 3]
    if repeated:
        group = max(repeated, key=len)
        add("F3", "动作循环", "reasoning_planning", [str(s.get("span_id") or "") for s in group],
            0.78, f"同一工具和规范化输入至少重复 {len(group)} 次。", True)
        failed_ids = {str(v.get("source_span_id") or "") for v in execution}
        failed_group = [s for s in group if str(s.get("span_id") or "") in failed_ids]
        if len(failed_group) >= 2:
            add("F10", "错误放大型重试", "agent_recovery", [str(s.get("span_id") or "") for s in failed_group],
                0.84, "相同工具和输入重复执行，且至少两次尝试伴随执行错误。", True)

    errors = [s for s in spans if _ERROR_NAME.search(str(s.get("span_name") or s.get("name") or ""))
              or str(s.get("status_code") or "").lower() in {"error", "status_code_error"}]
    finals = [s for s in spans if _FINAL.search(str(s.get("span_name") or s.get("name") or ""))]
    if errors and finals:
        positions = {str(s.get("span_id") or ""): i for i, s in enumerate(spans)}
        prior_errors = [s for s in errors if positions.get(str(s.get("span_id") or ""), 10**9) <
                        positions.get(str(finals[-1].get("span_id") or ""), -1)]
        if prior_errors:
            add("F11", "错误后继续作答", "agent_outcome",
                [str(s.get("span_id") or "") for s in prior_errors[-3:]] + [str(finals[-1].get("span_id") or "")],
                0.72, "终止错误之后仍出现 final-answer Span；需结合错误是否已恢复判断。", True)

    # Shared-parent bursts of multiple Error spans are observable cascade
    # surfaces, but not proof of independent root causes.
    error_parents: dict[str, list[dict]] = defaultdict(list)
    for span in errors:
        parent = str(span.get("parent_span_id") or "")
        if parent:
            error_parents[parent].append(span)
    cascades = [group for group in error_parents.values() if len(group) >= 2]
    if cascades:
        group = max(cascades, key=len)
        add("F12", "级联外显错误", "tool_interaction", [str(s.get("span_id") or "") for s in group],
            0.76, "多个错误 Span 共享父 Span，可能是单次故障的多层外显。", True)

    # Evidence-flow and semantic checks are direct observations, not proofs
    # that the agent ignored a fact or used noisy evidence.
    unconsumed = [c for c in checks if isinstance(c, dict) and c.get("result_preview")
                  and not c.get("consumed_by_any")]
    if unconsumed:
        add("F6-candidate", "关键事实可能未被后续消费", "information_perception",
            [str(c.get("span_id") or "") for c in unconsumed], 0.55,
            "工具结果未观察到后续消费；是否为关键事实需要语义判断。", True)

    date_mismatch = violations_by_type.get("temporal_mismatch", [])
    if date_mismatch:
        add("F1/F6", "约束遗漏或时间事实失配", "intent_or_information",
            [str(x.get("source_span_id") or "") for x in date_mismatch], 0.88,
            "任务要求日期与工具返回日期不一致；不能仅凭该信号确定是遗漏约束还是感知错误。", True)

    return findings
