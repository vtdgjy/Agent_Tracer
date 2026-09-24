"""Build a readable, structured report from a trace judgement.

The judge is encouraged to produce ``diagnostic_report`` directly.  This
module also supplies deterministic fallbacks so older judgement files remain
useful in the web UI and downstream consumers.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


IMPACT_ORDER = {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
IMPACT_LABELS = {"HIGH": "高影响", "MEDIUM": "中影响", "LOW": "低影响", "UNKNOWN": "影响待确认"}
RELATIONSHIP_LABELS = {
    "triggers": "触发",
    "amplifies": "放大",
    "blocks_recovery": "阻断恢复",
    "parallel": "并行作用",
}
MERGED_JUDGEMENT_KEYS = {
    "raw": "raw_trace_judgement",
    "graph": "graph_judgement",
    "pure_graph": "pure_graph_judgement",
}
SEMANTIC_EDGE_TYPES = {
    "SemanticOrigin",
    "SemanticValidation",
    "SemanticPropagation",
    "SemanticRejection",
    "EvidenceExposure",
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _impact(value: Any) -> str:
    normalized = _text(value).upper()
    return normalized if normalized in IMPACT_ORDER else "UNKNOWN"


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _highest_impact(items: list[dict[str, Any]]) -> str:
    impacts = [_impact(item.get("impact")) for item in items]
    return max(impacts, key=lambda value: IMPACT_ORDER[value], default="UNKNOWN")


def _matching_error(root_cause: dict[str, Any], errors: list[dict[str, Any]]) -> dict[str, Any]:
    location = _text(root_cause.get("location"))
    module = _text(root_cause.get("module")).lower()
    for error in errors:
        if location and location != "unknown" and _text(error.get("location")) == location:
            return error
    for error in errors:
        if module and _text(error.get("module")).lower() == module:
            return error
    return {}


def _fallback_root_cause_details(
    root_causes: list[dict[str, Any]], errors: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    details = []
    for index, root_cause in enumerate(root_causes, start=1):
        error = _matching_error(root_cause, errors)
        module = _text(root_cause.get("module")) or "未识别模块"
        reason = _text(root_cause.get("reason")) or "诊断结果未提供具体机制说明。"
        impact_level = _impact(root_cause.get("impact") or error.get("impact"))
        details.append(
            {
                "root_cause_index": index,
                "title": f"{module} 中的根因",
                "module": module,
                "location": _text(root_cause.get("location")) or "unknown",
                "mechanism": reason,
                "impact": _text(error.get("description")) or reason,
                "impact_level": impact_level,
                "confidence": _confidence(root_cause.get("confidence")),
                "evidence": _text(error.get("evidence")),
                "role": "首要根因" if index == 1 else "次要根因或放大因素",
            }
        )
    return details


def _normalize_root_cause_details(
    supplied: Any,
    root_causes: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fallback = _fallback_root_cause_details(root_causes, errors)
    if not isinstance(supplied, list):
        return fallback

    by_index = {
        item["root_cause_index"]: item
        for item in fallback
        if isinstance(item.get("root_cause_index"), int)
    }
    normalized = []
    for position, item in enumerate(supplied, start=1):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("root_cause_index") or position)
        except (TypeError, ValueError):
            index = position
        base = dict(by_index.get(index, {}))
        base.update(
            {
                "root_cause_index": index,
                "title": _text(item.get("title")) or base.get("title", f"根因 {index}"),
                "module": _text(item.get("module")) or base.get("module", "未识别模块"),
                "location": _text(item.get("location")) or base.get("location", "unknown"),
                "mechanism": _text(item.get("mechanism")) or base.get("mechanism", ""),
                "impact": _text(item.get("impact")) or base.get("impact", ""),
                "impact_level": _impact(item.get("impact_level") or base.get("impact_level")),
                "confidence": _confidence(item.get("confidence", base.get("confidence"))),
                "evidence": _text(item.get("evidence")) or base.get("evidence", ""),
                "role": _text(item.get("role")) or base.get("role", "相关根因"),
            }
        )
        normalized.append(base)

    seen = {item["root_cause_index"] for item in normalized}
    normalized.extend(item for item in fallback if item["root_cause_index"] not in seen)
    return sorted(normalized, key=lambda item: item["root_cause_index"])


def _fallback_failure_chain(causal_chains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_chains = [chain for chain in causal_chains if isinstance(chain, dict)]
    if not valid_chains:
        return []
    chain = max(valid_chains, key=lambda item: _confidence(item.get("confidence")))
    raw_steps = [
        ("故障产生", chain.get("fault_origin"), "异常信息、行为或状态首次出现"),
        ("故障传播", chain.get("fault_usage"), "下游步骤使用了异常结果，或未能正确处理异常"),
        ("最终失败", chain.get("final_failure"), "异常影响到最终输出或任务完成状态"),
    ]
    steps = []
    for stage, location, effect in raw_steps:
        location_text = _text(location)
        if not location_text or location_text == "unknown":
            continue
        if steps and steps[-1]["location"] == location_text:
            continue
        steps.append(
            {
                "order": len(steps) + 1,
                "stage": stage,
                "location": location_text,
                "event": _text(chain.get("evidence")) if stage == "故障产生" else effect,
                "effect": effect,
            }
        )
    return steps


def _normalize_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value] if isinstance(value, list) else []


def _fallback_remediation(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priorities = []
    for detail in details[:3]:
        priorities.append(
            {
                "priority": len(priorities) + 1,
                "target": detail.get("module") or detail.get("title") or "相关模块",
                "action": f"优先修复并验证：{detail.get('mechanism') or '该根因对应的异常行为'}",
                "expected_effect": f"阻断{detail.get('impact') or '该异常向下游传播'}",
            }
        )
    return priorities


def build_diagnostic_report(judgement: dict[str, Any]) -> dict[str, Any]:
    """Return a complete report, preserving model-authored details when present."""

    root_causes = _normalize_list_of_dicts(judgement.get("root_causes"))
    errors = _normalize_list_of_dicts(judgement.get("errors"))
    causal_chains = _normalize_list_of_dicts(judgement.get("causal_chains"))
    supplied = judgement.get("diagnostic_report")
    supplied = supplied if isinstance(supplied, dict) else {}

    details = _normalize_root_cause_details(
        supplied.get("root_cause_details"), root_causes, errors
    )
    relationships = _normalize_list_of_dicts(supplied.get("relationships"))
    failure_chain = _normalize_list_of_dicts(supplied.get("failure_chain"))
    if not failure_chain:
        failure_chain = _fallback_failure_chain(causal_chains)
    remediation = _normalize_list_of_dicts(supplied.get("remediation_priorities"))
    if not remediation:
        remediation = _fallback_remediation(details)

    diagnosis = _text(judgement.get("final_diagnosis"))
    if not diagnosis:
        diagnosis = "未发现足够证据形成明确的整体失败结论。"
    relationship_summary = _text(supplied.get("relationship_summary"))
    if not relationship_summary:
        relationship_summary = (
            "当前证据未证明多个根因之间存在直接的先后因果关系；它们可能分别作用于同一失败链。"
            if len(details) > 1
            else "当前只识别到一个主要根因，无需建立根因间关系。"
        )

    return {
        "overall_failure": _text(supplied.get("overall_failure")) or diagnosis,
        "failure_stage": _text(supplied.get("failure_stage")) or (
            details[0].get("module", "未识别阶段") if details else "未识别阶段"
        ),
        "severity": _impact(supplied.get("severity") or _highest_impact(root_causes + errors)),
        "root_cause_details": details,
        "relationship_summary": relationship_summary,
        "relationships": relationships,
        "failure_chain": failure_chain,
        "remediation_priorities": remediation,
        "conclusion": _text(supplied.get("conclusion")) or diagnosis,
    }


def _md_text(value: Any, fallback: str = "—") -> str:
    text = str(value if value is not None else "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return text or fallback


def _md_inline(value: Any, fallback: str = "—") -> str:
    return _md_text(value, fallback).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _md_confidence(value: Any) -> str:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "—"
    percentage = confidence * 100 if confidence <= 1 else confidence
    return f"{round(percentage)}%"


def _append_section(lines: list[str], title: str) -> None:
    lines.extend(["", f"## {title}", ""])


def _graph_summary(graph: dict[str, Any] | None) -> tuple[int, int, Counter[str], list[dict[str, Any]]]:
    graph = graph if isinstance(graph, dict) else {}
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
    edge_types = Counter(
        _text(edge.get("edge_type") or edge.get("label") or "Unknown")
        for edge in edges
        if isinstance(edge, dict)
    )
    semantic_edges = [
        edge
        for edge in edges
        if isinstance(edge, dict)
        and _text(edge.get("edge_type") or edge.get("label")) in SEMANTIC_EDGE_TYPES
    ]
    return len(nodes), len(edges), edge_types, semantic_edges


def render_diagnostic_markdown(
    judgement: dict[str, Any],
    *,
    trace_id: str = "trace",
    source_name: str = "graph",
    graph: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> str:
    """Render one normalized judgement as a standalone Markdown report."""

    judgement = judgement if isinstance(judgement, dict) else {}
    report = build_diagnostic_report(judgement)
    root_causes = report.get("root_cause_details") or []
    relationships = report.get("relationships") or []
    failure_chain = report.get("failure_chain") or []
    priorities = report.get("remediation_priorities") or []
    errors = judgement.get("errors") if isinstance(judgement.get("errors"), list) else []
    taxonomy = judgement.get("taxonomy_hits") if isinstance(judgement.get("taxonomy_hits"), list) else []
    node_count, edge_count, edge_types, semantic_edges = _graph_summary(graph)
    exported_at = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Agent Tracer 诊断报告",
        "",
        f"- Trace ID：{_md_inline(trace_id)}",
        f"- 诊断来源：{_md_inline(source_name)}",
        f"- 导出时间：{_md_inline(exported_at)}",
    ]
    if graph is not None:
        lines.append(f"- 图规模：{node_count} 个节点，{edge_count} 条边")

    _append_section(lines, "诊断概览")
    severity = _impact(report.get("severity"))
    lines.extend(
        [
            f"- 影响等级：{IMPACT_LABELS[severity]}",
            f"- 关键失败阶段：{_md_inline(report.get('failure_stage'), '未识别')}",
            "",
            _md_text(report.get("overall_failure"), "未形成整体诊断。"),
        ]
    )

    if root_causes:
        _append_section(lines, "根因如何导致失败")
        for position, item in enumerate(root_causes, start=1):
            if not isinstance(item, dict):
                continue
            index = item.get("root_cause_index") or position
            lines.extend(
                [
                    f"### {_md_inline(index)}. {_md_inline(item.get('title'), f'根因 {index}')}",
                    "",
                    f"- 角色：{_md_inline(item.get('role'), '相关根因')}",
                    f"- 模块：{_md_inline(item.get('module'), '未识别模块')}",
                    f"- 位置：{_md_inline(item.get('location'), 'unknown')}",
                    f"- 影响：{_md_inline(item.get('impact'), '未说明')}",
                    f"- 置信度：{_md_confidence(item.get('confidence'))}",
                    "",
                    "#### 形成机制",
                    "",
                    _md_text(item.get("mechanism"), "未提供。"),
                    "",
                    "#### 对失败的影响",
                    "",
                    _md_text(item.get("impact"), "未提供。"),
                ]
            )
            if item.get("evidence"):
                lines.extend(["", "#### 关键证据", "", _md_text(item.get("evidence"))])
            lines.append("")

    _append_section(lines, "根因之间的联系")
    lines.extend([_md_text(report.get("relationship_summary")), ""])
    if relationships:
        lines.extend(["| 起点 | 终点 | 关系 | 说明 |", "| --- | --- | --- | --- |"])
        for item in relationships:
            if not isinstance(item, dict):
                continue
            relation = RELATIONSHIP_LABELS.get(_text(item.get("relation_type")), _text(item.get("relation_type")) or "关联")
            lines.append(
                f"| 根因 {_md_inline(item.get('from_root_cause_index'))} | "
                f"根因 {_md_inline(item.get('to_root_cause_index'))} | {_md_inline(relation)} | "
                f"{_md_inline(item.get('explanation'))} |"
            )

    if failure_chain:
        _append_section(lines, "完整失败链")
        for position, item in enumerate(failure_chain, start=1):
            if not isinstance(item, dict):
                continue
            lines.extend(
                [
                    f"{position}. **{_md_inline(item.get('stage'), f'阶段 {position}')}**（{_md_inline(item.get('location'), 'unknown')}）",
                    f"   - 事件：{_md_inline(item.get('event'))}",
                    f"   - 影响：{_md_inline(item.get('effect'))}",
                ]
            )

    if errors:
        _append_section(lines, "检测到的错误")
        lines.extend(["| 类别 | 影响 | 位置 | 描述 | 证据 |", "| --- | --- | --- | --- | --- |"])
        for item in errors:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {_md_inline(item.get('category'))} | {_md_inline(item.get('impact'))} | "
                f"{_md_inline(item.get('location') or item.get('node_location'), 'unknown')} | "
                f"{_md_inline(item.get('description'))} | {_md_inline(item.get('evidence'))} |"
            )

    if taxonomy:
        _append_section(lines, "故障分类（CCG Taxonomy）")
        lines.extend(["| 分类 ID | 故障名称 | 是否命中 | 置信度 | 证据 |", "| --- | --- | --- | --- | --- |"])
        for item in taxonomy:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {_md_inline(item.get('fault_id'))} | {_md_inline(item.get('fault_name'))} | "
                f"{'是' if item.get('matched') else '否'} | {_md_confidence(item.get('confidence'))} | "
                f"{_md_inline(item.get('evidence'))} |"
            )

    if edge_types or semantic_edges:
        _append_section(lines, "图证据摘要")
        if edge_types:
            lines.extend(["### 边类型统计", "", "| 边类型 | 数量 |", "| --- | ---: |"])
            for edge_type, count in edge_types.most_common():
                lines.append(f"| {_md_inline(edge_type)} | {count} |")
        if semantic_edges:
            nodes = graph.get("nodes") if isinstance(graph, dict) and isinstance(graph.get("nodes"), list) else []
            node_labels = {
                node.get("id"): node.get("label") or node.get("content") or node.get("id")
                for node in nodes
                if isinstance(node, dict) and node.get("id") is not None
            }
            lines.extend(
                [
                    "",
                    "### 语义传播与校验边",
                    "",
                    "| 类型 | 来源 | 目标 | 状态 | 方法 | 得分 |",
                    "| --- | --- | --- | --- | --- | ---: |",
                ]
            )
            for edge in semantic_edges:
                edge_type = edge.get("edge_type") or edge.get("label")
                source = edge.get("source") if edge.get("source") is not None else edge.get("from")
                target = edge.get("target") if edge.get("target") is not None else edge.get("to")
                lines.append(
                    f"| {_md_inline(edge_type)} | {_md_inline(node_labels.get(source, source))} | "
                    f"{_md_inline(node_labels.get(target, target))} | {_md_inline(edge.get('propagation_state'))} | "
                    f"{_md_inline(edge.get('method'))} | {_md_inline(edge.get('score'))} |"
                )

    if priorities:
        _append_section(lines, "建议修复顺序")
        for position, item in enumerate(priorities, start=1):
            if not isinstance(item, dict):
                continue
            lines.extend(
                [
                    f"### {_md_inline(item.get('priority') or position)}. {_md_inline(item.get('target'), '相关模块')}",
                    "",
                    _md_text(item.get("action"), "未提供具体动作。"),
                    "",
                    f"预期效果：{_md_text(item.get('expected_effect'), '未说明。')}",
                    "",
                ]
            )

    _append_section(lines, "结论")
    lines.extend([_md_text(report.get("conclusion"), "当前证据不足以形成结论。"), ""])
    return "\n".join(lines)


def write_diagnostic_markdown(
    output_path: str | Path,
    judgement: dict[str, Any],
    **render_options: Any,
) -> Path:
    """Write a Markdown report with a UTF-8 BOM for Windows compatibility."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_diagnostic_markdown(judgement, **render_options), encoding="utf-8-sig")
    return path


def export_merged_diagnostic_markdown(
    merged_path: str | Path,
    output_path: str | Path,
    *,
    judgement_source: str = "graph",
) -> Path:
    """Export one ``merged/<trace_id>.json`` judgement file to Markdown."""

    if judgement_source not in MERGED_JUDGEMENT_KEYS:
        raise ValueError(f"Unknown judgement source: {judgement_source}")
    merged_file = Path(merged_path)
    with merged_file.open("r", encoding="utf-8-sig") as handle:
        merged = json.load(handle)
    judgement = merged.get(MERGED_JUDGEMENT_KEYS[judgement_source])
    if not isinstance(judgement, dict):
        raise ValueError(f"Missing {MERGED_JUDGEMENT_KEYS[judgement_source]} in {merged_file}")

    graph = None
    graph_file_value = (merged.get("meta") or {}).get("graph_file") if isinstance(merged.get("meta"), dict) else None
    if graph_file_value:
        graph_file = Path(graph_file_value)
        if graph_file.exists():
            with graph_file.open("r", encoding="utf-8-sig") as handle:
                graph = json.load(handle)

    return write_diagnostic_markdown(
        output_path,
        judgement,
        trace_id=_text(merged.get("trace_id")) or merged_file.stem,
        source_name=judgement_source,
        graph=graph,
    )
