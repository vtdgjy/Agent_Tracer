import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from diagnostic_report import build_diagnostic_report, export_merged_diagnostic_markdown
from semantic_trace_validator import analyze_semantic_trace

try:
    from trace_input_adapter import normalize_trace_for_parser
except Exception:
    normalize_trace_for_parser = None


load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_THINKING_MODE = "disabled"
DEEPSEEK_API_KEY = (
    os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("SILICONFLOW_API_KEY")
    or ""
)
GRAPH_REPRESENTATION_HYBRID = "hybrid"
GRAPH_REPRESENTATION_PURE = "pure"
DEFAULT_JUDGE_MAX_TOKENS = 0
DEFAULT_JUDGE_REVIEW_MAX_TOKENS = 0
DEFAULT_JSON_REPAIR_MAX_TOKENS = 0
DEFAULT_REVIEW_FIRST_PASS_MAX_CHARS = 0

CCG_TAXONOMY = [
    {
        "id": "F1",
        "name": "Constraint Omission",
        "group": "Intent Comprehension Faults",
        "description": "忽略用户关键约束（时间/格式/精度等）",
        "ccg_feature": "I 与后续 P/C 约束依赖弱或缺失；干预 I 约束后 C 不变",
    },
    {
        "id": "F2",
        "name": "Knowledge Hallucination",
        "group": "Reasoning & Planning Faults",
        "description": "无事实支撑直接作答",
        "ccg_feature": "P_final 或 C 缺少来自 F 的认知依赖边",
    },
    {
        "id": "F3",
        "name": "Action Looping",
        "group": "Reasoning & Planning Faults",
        "description": "重复调用同工具或陷入循环",
        "ccg_feature": "高相似 P->A->F 子图重复，Retry/重复工具特征显著",
    },
    {
        "id": "F4",
        "name": "Action Execution Crash",
        "group": "Tool & Interaction Faults",
        "description": "动作执行报错、超时、异常",
        "ccg_feature": "P->A->F 时序存在且 F/Error 含 exception/error",
    },
    {
        "id": "F5",
        "name": "Tool Hallucination/Insufficient Tools",
        "group": "Tool & Interaction Faults",
        "description": "调用非法工具或能力越界",
        "ccg_feature": "P 指向非法 A_invalid，随后 F_reject",
    },
    {
        "id": "F6",
        "name": "Crucial Fact Ignored",
        "group": "Information Perception Faults",
        "description": "关键事实存在但后续未被利用",
        "ccg_feature": "存在 F_key，但 F_key -> P_next/C 的认知依赖缺失",
    },
    {
        "id": "F7",
        "name": "Noise Distraction",
        "group": "Information Perception Faults",
        "description": "被无关噪声事实误导",
        "ccg_feature": "C_wrong 强依赖 F_noise，切断后结论应变化",
    },
]

JUDGE_DETECTION_CHECKS = [
    {
        "id": "format_constraints",
        "label": "Formatting / explicit constraints",
        "mapped_category": "Formatting Errors",
        "focus": "Missing required tags, malformed tool arguments, invalid output shape, stop-tag violations, schema or formatting mismatches.",
    },
    {
        "id": "tool_invocation",
        "label": "Tool invocation / environment / service failures",
        "mapped_category": "Tool-related",
        "focus": "Wrong tool choice, invalid tool capability assumptions, forbidden API usage, auth/network/403/404/timeout, path or environment mistakes.",
    },
    {
        "id": "retry_context_learning",
        "label": "Retry / context handling / learning from failure",
        "mapped_category": "Resource Abuse",
        "focus": "Repeated failed retries, unchanged wrong parameters, inability to adapt after an error, repeated invalid navigation or search.",
    },
    {
        "id": "plan_execution_alignment",
        "label": "Plan execution alignment",
        "mapped_category": "Goal Deviation",
        "focus": "Skipped planned steps, premature final_answer, plan says use tool but execution does not, orchestration failures.",
    },
    {
        "id": "retrieval_quality",
        "label": "Information retrieval quality",
        "mapped_category": "Poor Information Retrieval",
        "focus": "Wrong source, weak query choice, wrong page, failure to inspect required page or file, missing authoritative evidence.",
    },
    {
        "id": "semantic_contract",
        "label": "Per-tool call contract consistency",
        "mapped_category": "Hallucination",
        "focus": "For every tool call, inspect argument provenance/schema, execution envelope, result presence, explicit constraints, downstream consumption, and whether opaque identifiers are supported by prior evidence. STATUS_CODE_OK is not semantic correctness.",
    },
    {
        "id": "reasoning_support",
        "label": "Reasoning and evidence support",
        "mapped_category": "Hallucination",
        "focus": "Unsupported claims, tool output misread, conclusion lacks fact support, key evidence ignored.",
    },
    {
        "id": "instruction_following",
        "label": "Instruction adherence",
        "mapped_category": "Goal Deviation",
        "focus": "Ignored explicit user or system requirements, failed mandatory verification, failed required output or process constraints.",
    },
    {
        "id": "other_residual",
        "label": "Other residual risks",
        "mapped_category": "Other",
        "focus": "Any important causal issue not covered above.",
    },
]


def _build_detection_checklist_text():
    lines = []
    for item in JUDGE_DETECTION_CHECKS:
        lines.append(
            f"- {item['id']} | {item['label']} | default_category={item['mapped_category']} | inspect: {item['focus']}"
        )
    return "\n".join(lines)


def _empty_coverage_checklist():
    return [
        {
            "check_id": item["id"],
            "label": item["label"],
            "status": "uncertain",
            "mapped_category": item["mapped_category"],
            "location": "unknown",
            "evidence": "",
            "note": "",
        }
        for item in JUDGE_DETECTION_CHECKS
    ]

def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def safe_dump(obj, max_chars=None):
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if max_chars and len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def parse_trace_id_csv(text):
    if not text:
        return set()
    return {
        item.strip()
        for item in str(text).split(",")
        if item and item.strip()
    }


def iter_spans(root):
    if not isinstance(root, dict):
        return
    stack = list(root.get("spans", []))
    while stack:
        span = stack.pop()
        yield span
        for child in reversed(span.get("child_spans", []) or []):
            if isinstance(child, dict):
                stack.append(child)


def find_true_answer(trace_obj):
    for span in iter_spans(trace_obj):
        for log in span.get("logs", []) or []:
            body = log.get("body", {}) if isinstance(log, dict) else {}
            output = body.get("function.output")
            if isinstance(output, list):
                for item in output:
                    if isinstance(item, dict) and "true_answer" in item:
                        return str(item.get("true_answer"))
            if isinstance(output, dict) and "true_answer" in output:
                return str(output.get("true_answer"))
    return None


def extract_final_answer_candidates(trace_obj, max_items=8):
    cands = []
    pat = re.compile(r"final\s*answer\s*[:：]\s*(.+)", re.IGNORECASE)
    for span in iter_spans(trace_obj):
        attrs = span.get("span_attributes", {})
        text_fields = []
        if isinstance(attrs, dict):
            for k in ["llm.output_messages.0.message.content", "output.value", "input.value"]:
                v = attrs.get(k)
                if isinstance(v, str):
                    text_fields.append(v)
        for log in span.get("logs", []) or []:
            body = log.get("body", {}) if isinstance(log, dict) else {}
            out = body.get("function.output")
            if isinstance(out, str):
                text_fields.append(out)
        for text in text_fields:
            m = pat.search(text)
            if m:
                ans = m.group(1).strip().splitlines()[0][:120]
                cands.append({"span_id": span.get("span_id"), "answer": ans})
            elif "final_answer(" in text:
                cands.append({"span_id": span.get("span_id"), "answer": "final_answer() called"})
            if len(cands) >= max_items:
                return cands
    return cands


def extract_required_tags(input_prompt, input_value):
    tags = set()

    prompt_patterns = [
        r"write the ['\"](?:\\n)?(<[^>\s]+>)['\"] tag and stop there",
        r"write the (?:['\"])?(<[^>\s]+>)(?:['\"])? tag and stop there",
    ]

    if isinstance(input_prompt, str) and input_prompt:
        for pattern in prompt_patterns:
            for match in re.findall(pattern, input_prompt, flags=re.IGNORECASE):
                tags.add(match)

    parsed_input = None
    if isinstance(input_value, str):
        try:
            parsed_input = json.loads(input_value)
        except Exception:
            parsed_input = None

    if isinstance(parsed_input, dict):
        for seq in parsed_input.get("stop_sequences", []) or []:
            if isinstance(seq, str):
                seq = seq.strip()
                if re.fullmatch(r"<[^>\s]+>", seq):
                    tags.add(seq)

    return sorted(tags)


def extract_missing_format_violation(span):
    attrs = span.get("span_attributes", {}) if isinstance(span.get("span_attributes", {}), dict) else {}
    input_prompt = attrs.get("llm.input_messages.0.message.content", "")
    input_value = attrs.get("input.value", "")
    output_content = attrs.get("llm.output_messages.0.message.content", "")

    required_tags = extract_required_tags(input_prompt, input_value)
    if not required_tags or not isinstance(output_content, str):
        return None

    missing_tags = [tag for tag in required_tags if tag not in output_content]
    if not missing_tags:
        return None

    return {
        "type": "missing_required_tag",
        "span_id": span.get("span_id"),
        "span_name": span.get("span_name", ""),
        "required_tags": missing_tags,
        "evidence": f"required {', '.join(missing_tags)} in prompt but missing in output",
        "module": "planning" if "plan" in (input_prompt or "").lower() else "unknown",
    }


def build_raw_trace_digest(trace_obj, max_spans=80):
    span_count = 0
    span_name_counter = Counter()
    tool_counter = Counter()
    suspicious = []
    format_violations = []
    selected_spans = []

    for span in iter_spans(trace_obj):
        span_count += 1
        span_id = span.get("span_id")
        span_name = span.get("span_name", "")
        status_code = span.get("status_code", "")
        status_message = span.get("status_message", "")
        attrs = span.get("span_attributes", {}) if isinstance(span.get("span_attributes", {}), dict) else {}

        span_name_counter[span_name] += 1

        if span_name.startswith("Step "):
            out = attrs.get("output.value", "")
            if isinstance(out, str) and ("error" in out.lower() or "failed" in out.lower() or "exception" in out.lower()):
                suspicious.append({
                    "type": "step_output_error_keyword",
                    "span_id": span_id,
                    "span_name": span_name,
                    "evidence": out[:260],
                })

        if status_code and str(status_code).lower() == "error":
            suspicious.append({
                "type": "span_status_error",
                "span_id": span_id,
                "span_name": span_name,
                "evidence": (status_message or "status_code=Error")[:260],
            })

        if span_name in {"ToolCallingAgent.execute_tool_call", "CodeAgent.step", "Tool.__call__"}:
            tool_name = attrs.get("tool.name") or attrs.get("function.name") or span_name
            tool_counter[str(tool_name)] += 1

        if isinstance(attrs, dict):
            in_val = attrs.get("input.value")
            if isinstance(in_val, str) and '"kwargs": {"": ""}' in in_val:
                suspicious.append({
                    "type": "invalid_tool_kwargs",
                    "span_id": span_id,
                    "span_name": span_name,
                    "evidence": in_val[:260],
                })

        format_violation = extract_missing_format_violation(span)
        if format_violation:
            format_violations.append(format_violation)
            suspicious.append({
                "type": format_violation["type"],
                "span_id": format_violation["span_id"],
                "span_name": format_violation["span_name"],
                "evidence": format_violation["evidence"][:260],
            })

        if len(selected_spans) < max_spans:
            selected_spans.append({
                "span_id": span_id,
                "parent_span_id": span.get("parent_span_id"),
                "span_name": span_name,
                "status_code": status_code,
                "timestamp": span.get("timestamp"),
                "tool_name": attrs.get("tool.name") if isinstance(attrs, dict) else None,
            })

    repeated_tools = [
        {"tool": t, "count": c}
        for t, c in tool_counter.items()
        if c >= 4
    ]

    semantic_analysis = analyze_semantic_trace(trace_obj)
    for violation in semantic_analysis.get("semantic_violations", []):
        suspicious.append(
            {
                "type": violation.get("violation_type"),
                "span_id": violation.get("source_span_id"),
                "span_name": "semantic_contract",
                "evidence": str(violation.get("evidence") or "")[:260],
                "expected": violation.get("expected"),
                "observed": violation.get("observed"),
                "used_by_final": violation.get("used_by_final"),
                "reaches_final_answer": violation.get("reaches_final_answer"),
                "propagation_state": violation.get("propagation_state"),
                "propagation_path": violation.get("propagation_path", []),
            }
        )

    digest = {
        "trace_id": trace_obj.get("trace_id"),
        "stats": {
            "span_count": span_count,
            "top_span_names": span_name_counter.most_common(20),
            "top_tools": tool_counter.most_common(20),
            "repeated_tools_ge4": repeated_tools,
            "suspicious_event_count": len(suspicious),
            "format_violation_count": len(format_violations),
        },
        "true_answer": find_true_answer(trace_obj),
        "final_answer_candidates": extract_final_answer_candidates(trace_obj),
        "format_violations": format_violations[:20],
        "suspicious_events": suspicious[:60],
        "span_timeline_head": selected_spans,
        "task_contracts": semantic_analysis.get("task_contracts", []),
        "semantic_observations": semantic_analysis.get("semantic_observations", []),
        "semantic_violations": semantic_analysis.get("semantic_violations", []),
        "claim_evidence_links": semantic_analysis.get("claim_evidence_links", []),
        "delivery_links": semantic_analysis.get("delivery_links", []),
        "propagation_reports": semantic_analysis.get("propagation_reports", []),
        "final_answer_span_ids": semantic_analysis.get("final_answer_span_ids", []),
        "tool_call_checks": semantic_analysis.get("tool_call_checks", []),
    }
    return digest


def resolve_raw_input_mode(trace_id, default_mode, full_trace_ids):
    if trace_id in full_trace_ids:
        return "full"
    return default_mode


def build_raw_trace_payload(trace_obj, raw_input_mode, max_spans):
    if raw_input_mode == "full":
        return trace_obj
    if normalize_trace_for_parser is not None:
        normalized = normalize_trace_for_parser(trace_obj)
        if isinstance(normalized, dict) and normalized.get("spans"):
            trace_obj = normalized
    return build_raw_trace_digest(trace_obj, max_spans=max_spans)


def _extract_graph_format_violations(nodes):
    format_violations = []
    for node in nodes:
        if not isinstance(node, dict) or node.get("type") != "Error":
            continue
        content = str(node.get("content", ""))
        if not content.startswith("FormattingError: missing required tag"):
            continue
        format_violations.append(
            {
                "node_id": node.get("id"),
                "span_id": node.get("span_id"),
                "required_tags": re.findall(r"<[^>\s]+>", content),
                "evidence": (node.get("display") or content)[:220],
                "module": "planning",
            }
        )
    return format_violations


def _extract_tool_name_from_action_text(content):
    if not isinstance(content, str):
        return None
    match = re.search(r"Tool:\s*([^|]+)", content)
    if match:
        return match.group(1).strip()[:80]
    return None


def _build_content_flags(node):
    content = str(node.get("content", ""))
    text = content.lower()
    flags = []
    if re.search(r"<[^>\s]+>", content):
        flags.append("has_tag")
    if re.search(r"\b\d+(?:\.\d+)?\b", content):
        flags.append("has_number")
    if any(keyword in text for keyword in ["error", "exception", "failed", "timeout", "traceback"]):
        flags.append("has_error_terms")
    if "final answer" in text or "final_answer(" in text:
        flags.append("has_final_answer")
    if node.get("type") == "Action":
        tool_name = _extract_tool_name_from_action_text(content)
        if tool_name:
            flags.append(f"tool:{tool_name}")
    if content.lstrip().startswith("{") or content.lstrip().startswith("["):
        flags.append("json_like")
    return flags


def _project_graph_node(node, in_degree, out_degree, representation_mode):
    projected = {
        "id": node.get("id"),
        "type": node.get("type"),
        "saliency": node.get("saliency"),
        "span_id": node.get("span_id"),
        "in_degree": in_degree,
        "out_degree": out_degree,
    }
    if representation_mode == GRAPH_REPRESENTATION_PURE:
        projected.update(
            {
                "content_chars": len(str(node.get("content", ""))),
                "content_flags": _build_content_flags(node),
            }
        )
    else:
        projected["display"] = (node.get("display") or node.get("content", ""))[:220]
    return projected


def _project_cognitive_edge(edge, node_by_id, representation_mode):
    projected = {
        "source": edge.get("source"),
        "target": edge.get("target"),
        "score": edge.get("score"),
        "method": edge.get("method"),
    }
    if representation_mode == GRAPH_REPRESENTATION_PURE:
        projected.update(
            {
                "source_type": node_by_id.get(edge.get("source"), {}).get("type"),
                "target_type": node_by_id.get(edge.get("target"), {}).get("type"),
            }
        )
    return projected


def _compact_candidate_chain(chain, representation_mode):
    if not isinstance(chain, dict):
        return None

    compact = {
        "rank": chain.get("rank"),
        "root_cause_node": chain.get("root_cause_node"),
        "fault_origin": chain.get("fault_origin"),
        "fault_usage": chain.get("fault_usage"),
        "final_failure": chain.get("final_failure"),
        "causal_path": chain.get("causal_path", []),
        "path_valid": chain.get("path_valid"),
        "root_cause_type": chain.get("root_cause_type"),
        "root_cause_score": chain.get("root_cause_score"),
        "score_components": chain.get("score_components", {}),
        "dependency_metrics": chain.get("dependency_metrics", {}),
        "diversity": chain.get("diversity", {}),
        "selection_rank": chain.get("selection_rank"),
        "edge_chain": chain.get("edge_chain", [])[:12],
        "fault_origin_type": chain.get("fault_origin_type"),
        "fault_usage_type": chain.get("fault_usage_type"),
        "final_failure_type": chain.get("final_failure_type"),
        "fault_origin_span_id": chain.get("fault_origin_span_id"),
        "fault_usage_span_id": chain.get("fault_usage_span_id"),
        "final_failure_span_id": chain.get("final_failure_span_id"),
        "counterfactual_question": chain.get("counterfactual_question"),
        "location_policy": "type_conditioned_human_anchor",
    }

    if representation_mode == GRAPH_REPRESENTATION_HYBRID:
        compact.update(
            {
                "fault_origin_evidence": chain.get("fault_origin_evidence", ""),
                "fault_usage_evidence": chain.get("fault_usage_evidence", ""),
                "final_failure_evidence": chain.get("final_failure_evidence", ""),
            }
        )
    else:
        compact.update(
            {
                "fault_origin_evidence_chars": len(str(chain.get("fault_origin_evidence", ""))),
                "fault_usage_evidence_chars": len(str(chain.get("fault_usage_evidence", ""))),
                "final_failure_evidence_chars": len(str(chain.get("final_failure_evidence", ""))),
            }
        )
    return compact


def _extract_candidate_chains_for_digest(graph_obj, representation_mode, max_items=8):
    meta = graph_obj.get("meta", {}) if isinstance(graph_obj, dict) else {}
    chains = meta.get("candidate_chains") or meta.get("root_cause_ranking") or []
    if not isinstance(chains, list):
        return []
    compact = []
    for chain in chains[:max_items]:
        item = _compact_candidate_chain(chain, representation_mode)
        if item:
            compact.append(item)
    return compact


def _extract_rca_candidate_table_for_digest(graph_obj, representation_mode, max_items=10):
    meta = graph_obj.get("meta", {}) if isinstance(graph_obj, dict) else {}
    candidates = meta.get("rca_candidate_table") or []
    if not isinstance(candidates, list):
        return []
    compact = []
    for item in candidates[:max_items]:
        if not isinstance(item, dict):
            continue
        row = {
            "rank": item.get("rank"),
            "node_id": item.get("node_id"),
            "span_id": item.get("span_id"),
            "role": item.get("role"),
            "artifacts": item.get("artifacts", []),
            "consumer_node": item.get("consumer_node"),
            "symptom_or_error_node": item.get("symptom_or_error_node"),
            "suggested_root_cause_type": item.get("suggested_root_cause_type"),
            "candidate_score": item.get("candidate_score"),
            "why_candidate": item.get("why_candidate"),
            "counterfactual_hint": item.get("counterfactual_hint"),
        }
        if representation_mode == GRAPH_REPRESENTATION_HYBRID:
            row.update(
                {
                    "evidence": item.get("evidence", ""),
                    "consumer_evidence": item.get("consumer_evidence", ""),
                    "symptom_evidence": item.get("symptom_evidence", ""),
                }
            )
        else:
            row.update(
                {
                    "evidence_chars": len(str(item.get("evidence", ""))),
                    "consumer_evidence_chars": len(str(item.get("consumer_evidence", ""))),
                    "symptom_evidence_chars": len(str(item.get("symptom_evidence", ""))),
                }
            )
        compact.append(row)
    return compact


def _compact_failure_slice_meta(graph_obj):
    meta = graph_obj.get("meta", {}) if isinstance(graph_obj, dict) else {}
    failure_slice = meta.get("failure_slice", {})
    if not isinstance(failure_slice, dict):
        failure_slice = {}
    return {
        "strategy": failure_slice.get("strategy"),
        "target_count": failure_slice.get("target_count"),
        "targets": failure_slice.get("targets", []),
        "sliced_node_count": failure_slice.get("sliced_node_count"),
        "inspected_backward_edges": failure_slice.get("inspected_backward_edges"),
        "dropped_low_dependency_edges": failure_slice.get("dropped_low_dependency_edges"),
        "kept_by_reason": failure_slice.get("kept_by_reason", {}),
        "min_dependency_confidence": failure_slice.get("min_dependency_confidence"),
    }


def build_graph_digest(graph_obj, representation_mode=GRAPH_REPRESENTATION_HYBRID):
    nodes = graph_obj.get("nodes", []) if isinstance(graph_obj, dict) else []
    edges = graph_obj.get("edges", []) if isinstance(graph_obj, dict) else []
    summary = graph_obj.get("summary", {}) if isinstance(graph_obj, dict) else {}
    meta = graph_obj.get("meta", {}) if isinstance(graph_obj, dict) else {}
    semantic_analysis = graph_obj.get("semantic_analysis", {}) if isinstance(graph_obj, dict) else {}
    if not semantic_analysis:
        semantic_violations = [
            node.get("semantic_violation")
            for node in nodes
            if isinstance(node, dict) and isinstance(node.get("semantic_violation"), dict)
        ]
        semantic_observations = [
            node.get("semantic_observation")
            for node in nodes
            if isinstance(node, dict) and isinstance(node.get("semantic_observation"), dict)
        ]
        semantic_analysis = {
            "task_contracts": [],
            "tool_call_checks": semantic_observations,
            "semantic_observations": list(
                {json.dumps(item, ensure_ascii=False, sort_keys=True): item for item in semantic_observations}.values()
            ),
            "semantic_violations": list(
                {json.dumps(item, ensure_ascii=False, sort_keys=True): item for item in semantic_violations}.values()
            ),
            "claim_evidence_links": [],
            "delivery_links": [],
            "propagation_reports": [],
            "final_answer_span_ids": [],
        }

    top_nodes = sorted(
        [n for n in nodes if isinstance(n, dict)],
        key=lambda n: n.get("saliency", 0),
        reverse=True,
    )[:30]

    type_counter = Counter(n.get("type", "Unknown") for n in nodes if isinstance(n, dict))
    edge_type_counter = Counter(e.get("edge_type", "Unknown") for e in edges if isinstance(e, dict))

    node_by_id = {n.get("id"): n for n in nodes if isinstance(n, dict)}
    in_degree_counter = Counter()
    out_degree_counter = Counter()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        if edge.get("source"):
            out_degree_counter[edge.get("source")] += 1
        if edge.get("target"):
            in_degree_counter[edge.get("target")] += 1

    cognitive_edges = [
        _project_cognitive_edge(edge, node_by_id, representation_mode)
        for edge in edges
        if isinstance(edge, dict) and edge.get("edge_type") == "Cognitive"
    ]
    cognitive_edges = sorted(cognitive_edges, key=lambda x: (x.get("score") or 0), reverse=True)[:40]

    top_nodes_compact = [
        _project_graph_node(
            node,
            in_degree_counter.get(node.get("id"), 0),
            out_degree_counter.get(node.get("id"), 0),
            representation_mode,
        )
        for node in top_nodes
    ]

    format_violations = _extract_graph_format_violations(nodes)

    def _is_fact(node_id):
        n = node_by_id.get(node_id, {})
        return n.get("type") == "Fact"

    def _is_conclusion(node_id):
        n = node_by_id.get(node_id, {})
        return n.get("type") == "Conclusion"

    conclusion_ids = [nid for nid, n in node_by_id.items() if n.get("type") == "Conclusion"]
    cognitive_edges_all = [e for e in edges if isinstance(e, dict) and e.get("edge_type") == "Cognitive"]
    fact_to_conclusion_edges = [e for e in cognitive_edges_all if _is_fact(e.get("source")) and _is_conclusion(e.get("target"))]

    conclusion_incoming_from_fact = {}
    for cid in conclusion_ids:
        conclusion_incoming_from_fact[cid] = sum(
            1 for e in cognitive_edges_all if e.get("target") == cid and _is_fact(e.get("source"))
        )

    action_error_observation = sum(
        1
        for e in edges
        if isinstance(e, dict)
        and e.get("edge_type") == "Observation"
        and node_by_id.get(e.get("source"), {}).get("type") == "Action"
        and node_by_id.get(e.get("target"), {}).get("type") in {"Error", "Fact"}
        and any(k in str(node_by_id.get(e.get("target"), {}).get("content", "")).lower() for k in ["error", "exception", "failed", "timeout"])
    )

    isolated_fact_candidates = []
    for nid, n in node_by_id.items():
        if n.get("type") != "Fact":
            continue
        out_cognitive = [e for e in cognitive_edges_all if e.get("source") == nid]
        if len(out_cognitive) == 0 and (n.get("saliency") or 0) >= 4.0:
            isolated_fact_candidates.append({
                "id": nid,
                "saliency": n.get("saliency"),
                "display": (n.get("display") or "")[:160],
            })

    likely_noise_fact_to_conclusion = 0
    noise_keys = ["google", "search results", "web results", "reddit", "github", "viewport position"]
    for e in fact_to_conclusion_edges:
        src = node_by_id.get(e.get("source"), {})
        txt = str(src.get("content", "")).lower()
        if any(k in txt for k in noise_keys):
            likely_noise_fact_to_conclusion += 1

    ccg_signals = {
        "f2_hallucination_signal": {
            "conclusion_count": len(conclusion_ids),
            "fact_to_conclusion_cognitive_edges": len(fact_to_conclusion_edges),
            "conclusion_incoming_from_fact": conclusion_incoming_from_fact,
            "possible_no_fact_support": all(v == 0 for v in conclusion_incoming_from_fact.values()) if conclusion_incoming_from_fact else False,
        },
        "f3_looping_signal": {
            "retry_edges": edge_type_counter.get("Retry", 0),
            "observation_edges": edge_type_counter.get("Observation", 0),
            "repeated_tools_hint": edge_type_counter.get("Retry", 0) >= 2,
        },
        "f4_execution_crash_signal": {
            "action_to_error_observation_edges": action_error_observation,
            "likely_action_crash": action_error_observation > 0,
        },
        "f6_crucial_fact_ignored_signal": {
            "isolated_salient_fact_count": len(isolated_fact_candidates),
            "isolated_salient_fact_examples": isolated_fact_candidates[:8],
            "likely_fact_ignored": len(isolated_fact_candidates) > 0,
        },
        "f7_noise_distraction_signal": {
            "noise_fact_to_conclusion_edges": likely_noise_fact_to_conclusion,
            "likely_noise_distraction": likely_noise_fact_to_conclusion > 0,
        },
    }

    return {
        "representation_mode": representation_mode,
        "summary": summary,
        "stats": {
            "node_type_counter": dict(type_counter),
            "edge_type_counter": dict(edge_type_counter),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "format_violation_count": len(format_violations),
            "text_preservation": "minimal" if representation_mode == GRAPH_REPRESENTATION_PURE else "hybrid",
        },
        "top_salient_nodes": top_nodes_compact,
        "top_cognitive_edges": cognitive_edges,
        "failure_slice": _compact_failure_slice_meta(graph_obj),
        "rca_candidate_table": _extract_rca_candidate_table_for_digest(graph_obj, representation_mode),
        "candidate_causal_chains": _extract_candidate_chains_for_digest(graph_obj, representation_mode),
        "root_cause_score_weights": meta.get("root_cause_score_weights", {}),
        "chain_schema_version": meta.get("chain_schema_version"),
        "format_violations": format_violations[:20],
        "ccg_topology_signals": ccg_signals,
        # Structured semantic facts are preserved even in pure graph mode;
        # opaque identifiers cannot be recovered from topology alone.
        "task_contracts": semantic_analysis.get("task_contracts", []),
        "semantic_observations": semantic_analysis.get("semantic_observations", []),
        "semantic_violations": semantic_analysis.get("semantic_violations", []),
        "claim_evidence_links": semantic_analysis.get("claim_evidence_links", []),
        "delivery_links": semantic_analysis.get("delivery_links", []),
        "propagation_reports": semantic_analysis.get("propagation_reports", []),
        "final_answer_span_ids": semantic_analysis.get("final_answer_span_ids", []),
        "tool_call_checks": semantic_analysis.get("tool_call_checks", []),
    }


def make_judge_prompt_swe(source_type, trace_id, digest_json):
    """SWE-specific variant v4: A+B+C optimizations.
    A: stronger proximity rule (prefer execution/tool-call span over upstream planning)
    B: SWE category mapping hints
    C: fine-grained impact calibration
    """
    taxonomy_text = "\n".join(
        f"- {t['id']} {t['name']}: {t['ccg_feature']}" for t in CCG_TAXONOMY
    )
    checklist_text = _build_detection_checklist_text()
    return f"""
You are a strict Agent trace root-cause analyst.
Task domain: software engineering / code repair (SWE-Bench).
Analyze {source_type} and find the earliest causal failures that make the answer wrong or unreliable.
You must follow a fixed diagnosis workflow instead of jumping directly to a conclusion.
You must prioritize the CCG taxonomy when grouping root causes.

CCG taxonomy:
{taxonomy_text}

SWE-specific failure patterns to actively look for:
1. File/output size constraint violations: model prints full file content despite "STRICTLY DO NOT print files > X chars" or "print up to 500 characters at a time" instructions → category: Instruction Non-compliance.
2. Runtime code errors: unterminated string literals, Unicode/encoding errors, type errors from wrong variable types → category: Formatting Errors.
3. Missing final_answer tool call: model produces output but does not call final_answer() as required → category: Instruction Non-compliance.
4. Printing full tree/file when only partial output was requested → category: Instruction Non-compliance.
5. Hallucinated file paths: model assumes a path from partial tree structure without evidence → category: Incorrect Problem Identification.
6. Repeated same wrong tool call with identical bad arguments → category: Resource Abuse + Formatting Errors.
7. 403/auth error from visit_page → category: Authentication Errors (LOW unless it blocks the entire task).

SWE category mapping guide (direction B):
- "file/output size constraint violation" → Instruction Non-compliance
- "repeated same wrong tool call" → Resource Abuse
- "wrong tool arguments (e.g. page_down with args)" → Formatting Errors
- "403/auth error" → Authentication Errors
- "hallucinated path" → Incorrect Problem Identification

Mandatory diagnosis workflow:
1. Read the digest and identify the earliest failure chain.
2. Run every checklist item below one by one; do not skip any item.
3. For each checklist item, decide whether the issue is present / absent / uncertain.
4. Only after checklist coverage is complete, synthesize errors, root causes, taxonomy hits, and final diagnosis.
5. STRONGLY PREFER execution and tool-call spans for location anchoring. Only use planning spans if the defect is in the plan itself (wrong logic, wrong decision), not merely because the plan led to a bad execution.
6. Apply type-conditioned location disambiguation (SWE-specific):
   - PRIMARY RULE: anchor at the CODE EXECUTION or TOOL CALL span where the error is directly triggered or observed. This is the most important rule.
   - PROXIMITY RULE: when multiple spans are on the same causal chain, prefer the span closest to the observable failure output. Do NOT walk back to an upstream planning span unless the planning span itself contains the defect.
   - File size / output length violations: anchor at the code execution span that produced the oversized output.
   - Runtime errors (type error, encoding error, syntax error): anchor at the code execution span where the error occurred.
   - Wrong tool arguments: anchor at the tool call span with the bad arguments.
   - Missing final_answer: anchor at the last code execution span before the trace ends.
   - Hallucinated file paths: anchor at the code execution span that used the hallucinated path.
   - Retrieval failures: anchor at the bad search/tool call span.
   - Retry/resource-abuse: anchor at the first repeated bad action span.
7. Impact calibration for SWE (direction C):
   - HIGH: error directly causes wrong final answer or complete task failure (e.g. wrong patch applied, task not completed at all).
   - MEDIUM: error causes significant detour, repeated failures, missed key information, or substantial wasted computation (e.g. repeated wrong tool calls, large output constraint violations that consume context).
   - LOW: minor constraint violation that does not affect the final result (e.g. printed 600 chars instead of 500 once, minor formatting issue with no downstream effect).

Tool-call validation rules:
- Inspect every item in tool_call_checks; do not infer correctness from STATUS_CODE_OK alone.
- Treat deterministic schema/status/constraint violations as evidence and ungrounded opaque arguments as requiring semantic review.
- Use delivery_links and propagation_reports before semantic guesswork: exact tool_call_id delivery proves exposure, while PROPAGATED/TRANSFORMED proves downstream adoption.
- Treat REJECTED and EXPOSED as non-propagated unless another path reaches a final_answer_span_id; treat UNKNOWN as requiring semantic review.
- Prefer a bad or unverified call with reaches_final_answer=true over an unused or merely exposed noisy result.

Checklist to run on every trace:
{checklist_text}

Input trace_id={trace_id} (JSON):
{digest_json}

Output exactly one JSON object. No markdown. No prose outside JSON.
Write all diagnostic_report explanatory text in Chinese. Keep identifiers, span ids, and code unchanged.

JSON schema:
{{
  "trace_id": "{trace_id}",
  "source": "{source_type}",
  "coverage_checks": [
    {{
      "check_id": "one of the checklist ids above",
      "label": "copy the checklist label",
      "status": "present|absent|uncertain",
      "mapped_category": "Formatting Errors|Tool-related|Goal Deviation|Poor Information Retrieval|Hallucination|Reasoning Error|Resource Abuse|Other",
      "location": "span_id (16 hex) or unknown",
      "evidence": "short evidence from input",
      "note": "why this check is or is not triggered"
    }}
  ],
  "errors": [
    {{
      "category": "Formatting Errors|Tool-related|Goal Deviation|Poor Information Retrieval|Hallucination|Reasoning Error|Resource Abuse|Other",
      "module": "module/stage name such as planning/tool_calling/search/final_answer/graph_reasoning",
      "location": "span_id (16 hex) or graph node id such as E_31/C_2/F_53",
      "evidence": "short evidence from input",
      "description": "mechanism of failure",
      "impact": "LOW|MEDIUM|HIGH"
    }}
  ],
  "scores": [
    {{
      "reliability_score": 1-5,
      "reliability_reasoning": "简要理由",
      "security_score": 1-5,
      "security_reasoning": "简要理由",
      "instruction_adherence_score": 1-5,
      "instruction_adherence_reasoning": "简要理由",
      "plan_opt_score": 1-5,
      "plan_opt_reasoning": "简要理由",
      "overall": 1-5
    }}
  ],
  "root_causes": [
    {{
      "module": "模块名",
      "location": "span_id (16 hex) — must be execution/tool-call span, not planning span",
      "causal_origin_location": "earliest causal origin span/node if visible, otherwise same as location",
      "human_preferred_location": "type-conditioned human anchor, usually same as location",
      "location_policy": "type_conditioned_human_anchor",
      "causal_origin_node": "graph node id for the upstream producer when visible, such as A_5",
      "node_location": "same graph node id used as the location anchor when visible",
      "reason": "根因",
      "impact": "LOW|MEDIUM|HIGH",
      "confidence": 0-1
    }}
  ],
  "causal_chains": [
    {{
      "root_cause_node": "graph node id such as F_12/E_4/A_7 or unknown",
      "fault_origin": "node/span where the bad evidence/action/failure first appears",
      "fault_usage": "node/span where the bad evidence/action/failure is used or ignored",
      "final_failure": "final answer/failure node/span",
      "causal_path": ["node_a", "node_b", "node_c"],
      "path_valid": true,
      "root_cause_type": "wrong_evidence_used|tool_or_execution_failure|format_or_tool_argument_error|unsupported_or_invalid_reasoning|wrong_action_or_tool_choice|retry_or_recovery_failure|other",
      "evidence": "short evidence from input",
      "counterfactual_effect": "yes/no/uncertain with short reason",
      "confidence": 0-1
    }}
  ],
  "taxonomy_hits": [
    {{
      "fault_id": "F1|F2|F3|F4|F5|F6|F7",
      "fault_name": "taxonomy name",
      "matched": true,
      "evidence": "evidence summary",
      "ccg_feature_match": "matched topology or causal feature",
      "confidence": 0-1
    }}
  ],
  "final_diagnosis": "one-sentence diagnosis",
  "diagnostic_report": {{
    "overall_failure": "2-4 sentences explaining what failed and why",
    "failure_stage": "the stage where the decisive failure occurred",
    "severity": "LOW|MEDIUM|HIGH",
    "root_cause_details": [
      {{
        "root_cause_index": 1,
        "title": "short root-cause title",
        "module": "module/stage name",
        "location": "same concrete location as the matching root cause",
        "mechanism": "how this root cause was produced",
        "impact": "what downstream behavior and final result it affected",
        "impact_level": "LOW|MEDIUM|HIGH",
        "confidence": 0-1,
        "evidence": "short visible evidence",
        "role": "primary cause|trigger|amplifier|recovery failure"
      }}
    ],
    "relationship_summary": "how the root causes jointly led to failure, without inventing links",
    "relationships": [
      {{
        "from_root_cause_index": 1,
        "to_root_cause_index": 2,
        "relation_type": "triggers|amplifies|blocks_recovery|parallel",
        "explanation": "evidence-backed relationship"
      }}
    ],
    "failure_chain": [
      {{
        "order": 1,
        "stage": "failure stage",
        "location": "span or node id",
        "event": "what happened",
        "effect": "what it caused next"
      }}
    ],
    "remediation_priorities": [
      {{
        "priority": 1,
        "target": "module or behavior to fix",
        "action": "specific corrective action",
        "expected_effect": "which part of the failure chain this blocks"
      }}
    ],
    "conclusion": "plain-language conclusion for the trace owner"
  }}
}}

Rules:
1. Complete all coverage_checks first. There must be one item for every checklist id above.
2. Do not skip a category just because another already matched.
3. root_causes[].location MUST be an execution or tool-call span. Only use a planning span if the plan itself is defective.
4. Evidence must come from visible input fields. Do not invent evidence.
5. Every errors and root_causes item must have a concrete location (span_id).
6. Target >=3 errors and >=3 taxonomy_hits when evidence exists.
7. Output must be valid JSON with correct types.
""".strip()
    taxonomy_text = "\n".join(
        f"- {t['id']} {t['name']}: {t['ccg_feature']}" for t in CCG_TAXONOMY
    )
    checklist_text = _build_detection_checklist_text()
    return f"""
You are a strict Agent trace root-cause analyst.
Task domain: software engineering / code repair (SWE-Bench).
Analyze {source_type} and find the earliest causal failures that make the answer wrong or unreliable.
You must follow a fixed diagnosis workflow instead of jumping directly to a conclusion.
You must prioritize the CCG taxonomy when grouping root causes.

CCG taxonomy:
{taxonomy_text}

SWE-specific failure patterns to actively look for:
1. File/output size constraint violations: model prints full file content despite "STRICTLY DO NOT print files > X chars" or "print up to 500 characters at a time" instructions. These are Instruction Non-compliance, typically MEDIUM impact.
2. Runtime code errors: unterminated string literals, Unicode/encoding errors (UTF-16 decoded as UTF-8), type errors from wrong variable types passed to functions. These are Formatting Errors.
3. Missing final_answer tool call: model produces output but does not call final_answer() as required.
4. Printing full tree/file when only partial output was requested.
5. Hallucinated file paths: model assumes a path from partial tree structure without evidence.

Mandatory diagnosis workflow:
1. Read the digest and identify the earliest failure chain.
2. Run every checklist item below one by one; do not skip any item.
3. For each checklist item, decide whether the issue is present / absent / uncertain.
4. Only after checklist coverage is complete, synthesize errors, root causes, taxonomy hits, and final diagnosis.
5. Preserve the earliest causal origin, but choose the reported root-cause location using human-style type-conditioned anchoring.
6. Apply type-conditioned location disambiguation (SWE-specific):
   - SWE primary rule: anchor at the specific CODE EXECUTION or TOOL CALL span where the error FIRST MANIFESTS, not at the planning span.
   - DO NOT over-trace: prefer the span where the problem is directly triggered or observed over a distant upstream planning span. If the error is visible at a tool call or code execution span, anchor there — do not walk further back to an earlier planning span unless the planning span itself contains the defect.
   - File size / output length violations: anchor at the code execution span that produced the oversized output.
   - Runtime errors (type error, encoding error, syntax error): anchor at the code execution span where the error occurred.
   - Missing final_answer: anchor at the last code execution span before the trace ends.
   - Hallucinated file paths: anchor at the code execution span that used the hallucinated path.
   - Retrieval failures: anchor at the bad search/tool call span.
   - Retry/resource-abuse: anchor at the first repeated bad action span.
7. Impact calibration for SWE: most errors are MEDIUM. Use HIGH only when the error directly causes the final answer to be wrong or the task to fail completely. Use LOW for minor constraint violations that do not affect the final result.

Tool-call validation rules:
- Inspect every item in tool_call_checks; do not infer correctness from STATUS_CODE_OK alone.
- Treat deterministic schema/status/constraint violations as evidence and ungrounded opaque arguments as requiring semantic review.
- Use delivery_links and propagation_reports before semantic guesswork: exact tool_call_id delivery proves exposure, while PROPAGATED/TRANSFORMED proves downstream adoption.
- Treat REJECTED and EXPOSED as non-propagated unless another path reaches a final_answer_span_id; treat UNKNOWN as requiring semantic review.
- Prefer a bad or unverified call with reaches_final_answer=true over an unused or merely exposed noisy result.

Checklist to run on every trace:
{checklist_text}

Input trace_id={trace_id} (JSON):
{digest_json}

Output exactly one JSON object. No markdown. No prose outside JSON.
Write all diagnostic_report explanatory text in Chinese. Keep identifiers, span ids, and code unchanged.

JSON schema:
{{
  "trace_id": "{trace_id}",
  "source": "{source_type}",
  "coverage_checks": [
    {{
      "check_id": "one of the checklist ids above",
      "label": "copy the checklist label",
      "status": "present|absent|uncertain",
      "mapped_category": "Formatting Errors|Tool-related|Goal Deviation|Poor Information Retrieval|Hallucination|Reasoning Error|Resource Abuse|Other",
      "location": "span_id (16 hex) or unknown",
      "evidence": "short evidence from input",
      "note": "why this check is or is not triggered"
    }}
  ],
  "errors": [
    {{
      "category": "Formatting Errors|Tool-related|Goal Deviation|Poor Information Retrieval|Hallucination|Reasoning Error|Resource Abuse|Other",
      "module": "module/stage name such as planning/tool_calling/search/final_answer/graph_reasoning",
      "location": "span_id (16 hex) or graph node id such as E_31/C_2/F_53",
      "evidence": "short evidence from input",
      "description": "mechanism of failure",
      "impact": "LOW|MEDIUM|HIGH"
    }}
  ],
  "scores": [
    {{
      "reliability_score": 1-5,
      "reliability_reasoning": "简要理由",
      "security_score": 1-5,
      "security_reasoning": "简要理由",
      "instruction_adherence_score": 1-5,
      "instruction_adherence_reasoning": "简要理由",
      "plan_opt_score": 1-5,
      "plan_opt_reasoning": "简要理由",
      "overall": 1-5
    }}
  ],
  "root_causes": [
    {{
      "module": "模块名",
      "location": "span_id (16 hex) — must be the execution/tool-call span, not the planning span",
      "causal_origin_location": "earliest causal origin span/node if visible, otherwise same as location",
      "human_preferred_location": "type-conditioned human anchor, usually same as location",
      "location_policy": "type_conditioned_human_anchor",
      "reason": "根因",
      "impact": "LOW|MEDIUM|HIGH",
      "confidence": 0-1
    }}
  ],
  "causal_chains": [
    {{
      "root_cause_node": "graph node id such as F_12/E_4/A_7 or unknown",
      "fault_origin": "node/span where the bad evidence/action/failure first appears",
      "fault_usage": "node/span where the bad evidence/action/failure is used or ignored",
      "final_failure": "final answer/failure node/span",
      "causal_path": ["node_a", "node_b", "node_c"],
      "path_valid": true,
      "root_cause_type": "wrong_evidence_used|tool_or_execution_failure|format_or_tool_argument_error|unsupported_or_invalid_reasoning|wrong_action_or_tool_choice|retry_or_recovery_failure|other",
      "evidence": "short evidence from input",
      "counterfactual_effect": "Would removing this origin likely prevent the final failure? yes/no/uncertain with short reason",
      "confidence": 0-1
    }}
  ],
  "taxonomy_hits": [
    {{
      "fault_id": "F1|F2|F3|F4|F5|F6|F7",
      "fault_name": "taxonomy name",
      "matched": true,
      "evidence": "evidence summary",
      "ccg_feature_match": "matched topology or causal feature",
      "confidence": 0-1
    }}
  ],
  "final_diagnosis": "one-sentence diagnosis",
  "diagnostic_report": {{
    "overall_failure": "2-4 sentences explaining what failed and why",
    "failure_stage": "the stage where the decisive failure occurred",
    "severity": "LOW|MEDIUM|HIGH",
    "root_cause_details": [{{"root_cause_index": 1, "title": "short title", "module": "module", "location": "span/node id", "mechanism": "how it happened", "impact": "downstream and final impact", "impact_level": "LOW|MEDIUM|HIGH", "confidence": 0-1, "evidence": "visible evidence", "role": "primary cause|trigger|amplifier|recovery failure"}}],
    "relationship_summary": "how the root causes jointly led to failure",
    "relationships": [{{"from_root_cause_index": 1, "to_root_cause_index": 2, "relation_type": "triggers|amplifies|blocks_recovery|parallel", "explanation": "evidence-backed relationship"}}],
    "failure_chain": [{{"order": 1, "stage": "stage", "location": "span/node id", "event": "what happened", "effect": "what it caused next"}}],
    "remediation_priorities": [{{"priority": 1, "target": "module/behavior", "action": "specific correction", "expected_effect": "blocked failure effect"}}],
    "conclusion": "plain-language conclusion"
  }}
}}

Rules:
1. Complete all coverage_checks first. There must be one coverage_checks item for every checklist id above.
2. Do not skip a category just because another category already matched.
3. Keep the earliest causal location in `causal_origin_location`, but set `root_causes[].location` to the execution-span anchor.
4. Evidence must come from visible input fields. Do not invent evidence.
5. Every errors item and every root_causes item must have a concrete location. Prefer span_id as location.
5a. For coverage_checks with status=present, use the concrete span_id/node_id of the evidence whenever possible.
6. If explicit format_violations exist, you must inspect and report them even if impact is low.
7. If the digest contains candidate_causal_chains, treat them as Top-K graph-ranked candidates.
8. causal_chains must use the three-part form fault_origin -> fault_usage -> final_failure.
9. root_causes[].location must follow the SWE execution-span anchoring rule above.
10. Distinguish these common cases: invalid tool args, wrong tool choice, repeated failed retry, service/auth/environment failure, skipped planned steps, premature final_answer, unsupported conclusion.
11. Target >=3 errors and >=3 taxonomy_hits when evidence exists.
12. Output must be valid JSON with correct types.
""".strip()


def make_judge_prompt(source_type, trace_id, digest_json):
    taxonomy_text = "\n".join(
        f"- {t['id']} {t['name']}: {t['ccg_feature']}" for t in CCG_TAXONOMY
    )
    checklist_text = _build_detection_checklist_text()
    return f"""
You are a strict Agent trace root-cause analyst.
Analyze {source_type} and find the earliest causal failures that make the answer wrong or unreliable.
You must follow a fixed diagnosis workflow instead of jumping directly to a conclusion.
You must prioritize the CCG taxonomy when grouping root causes.

CCG taxonomy:
{taxonomy_text}

Mandatory diagnosis workflow:
1. Read the digest and identify the earliest failure chain.
2. Run every checklist item below one by one; do not skip any item.
3. For each checklist item, decide whether the issue is present / absent / uncertain.
4. Only after checklist coverage is complete, synthesize errors, root causes, taxonomy hits, and final diagnosis.
5. First locate the observable Error/symptom. Then inspect the same causal chain upstream.
   If an earlier Action/Fact created a bad configuration, path, value, artifact, environment setting,
   command string, model name, schema, or other state later consumed by the failing step, treat that
   earlier producer as the root cause and treat the Error node as the symptom.
6. Preserve the earliest causal origin, but choose the reported root-cause location using human-style
   type-conditioned anchoring. For upstream state/configuration faults, the human-preferred root-cause
   location is the upstream bad setup Action/Fact, not the downstream Error.
7. Apply type-conditioned location disambiguation:
   - upstream configuration/state contamination anchors at the Action/Fact that produced the bad state.
   - format constraint violations and tool execution crashes anchor at the first visible Error/execution message only when no upstream bad producer is visible on the same chain.
   - invalid tool arguments and wrong tool choices anchor at the bad tool call when it is explicit; otherwise at the first visible Error.
   - plan/goal deviation, unsupported reasoning, hallucination, and incorrect problem identification anchor at the first bad planning/reasoning/claim span.
   - retrieval/evidence-quality failures anchor at the bad search/tool call or the first reasoning claim that uses the bad evidence.
   - retry/resource-abuse failures anchor at the first repeated bad action or first visible repeated failure.
   - final-answer/completion failures anchor at the final answer call/text.
8. When candidate_causal_chains are provided, read each candidate as:
   fault_origin -> fault_usage -> final_failure. Prefer a valid chain whose fault_origin is an upstream
   Action/Fact that counterfactually explains the downstream Error. Do not collapse fault_origin into
   final_failure unless the Error action itself created the bad state.
   Treat dependency_metrics as path-level evidence: prefer high final_answer_dependency and
   causal_path_strength, and reject a chain dominated by temporal edges even if its nodes are adjacent.
9. When rca_candidate_table is provided, treat it as the primary root-cause candidate set.
   Each row is a producer-consumer hypothesis: node_id produced an artifact/value, consumer_node used it,
   and symptom_or_error_node is the downstream symptom. Prefer a row with role=upstream_state_producer
   when its evidence explains the final wrong state. Report its node_id/span_id as the root cause location.
   Do not choose repository-inspection, read_file, retry, Error, or final-answer nodes over a valid producer row.
   Order root_causes by RCA priority: root_causes[0] MUST be the best valid producer candidate when one exists.

Tool-call validation rules:
- Inspect every item in tool_call_checks; do not infer correctness from STATUS_CODE_OK alone.
- Treat deterministic schema/status/constraint violations as evidence and ungrounded opaque arguments as requiring semantic review.
- Use delivery_links and propagation_reports before semantic guesswork: exact tool_call_id delivery proves exposure, while PROPAGATED/TRANSFORMED proves downstream adoption.
- Treat REJECTED and EXPOSED as non-propagated unless another path reaches a final_answer_span_id; treat UNKNOWN as requiring semantic review.
- Prefer a bad or unverified call with reaches_final_answer=true over an unused or merely exposed noisy result.

Checklist to run on every trace:
{checklist_text}

Input trace_id={trace_id} (JSON):
{digest_json}

Output exactly one JSON object. No markdown. No prose outside JSON.
Write all diagnostic_report explanatory text in Chinese. Keep identifiers, span ids, and code unchanged.

JSON schema:
{{
  "trace_id": "{trace_id}",
  "source": "{source_type}",
  "coverage_checks": [
    {{
      "check_id": "one of the checklist ids above",
      "label": "copy the checklist label",
      "status": "present|absent|uncertain",
      "mapped_category": "Formatting Errors|Tool-related|Goal Deviation|Poor Information Retrieval|Hallucination|Reasoning Error|Resource Abuse|Other",
      "location": "span_id (16 hex) or unknown",
      "evidence": "short evidence from input",
      "note": "why this check is or is not triggered"
    }}
  ],
  "errors": [
    {{
      "category": "Formatting Errors|Tool-related|Goal Deviation|Poor Information Retrieval|Hallucination|Reasoning Error|Resource Abuse|Other",
      "module": "module/stage name such as planning/tool_calling/search/final_answer/graph_reasoning",
      "location": "span_id (16 hex) or graph node id such as E_31/C_2/F_53",
      "evidence": "short evidence from input",
      "description": "mechanism of failure",
      "impact": "LOW|MEDIUM|HIGH"
    }}
  ],
  "scores": [
    {{
      "reliability_score": 1-5,
      "reliability_reasoning": "简要理由",
      "security_score": 1-5,
      "security_reasoning": "简要理由",
      "instruction_adherence_score": 1-5,
      "instruction_adherence_reasoning": "简要理由",
      "plan_opt_score": 1-5,
      "plan_opt_reasoning": "简要理由",
      "overall": 1-5
    }}
  ],
  "root_causes": [
    {{
      "module": "模块名",
      "location": "span_id (16 hex) or graph node id such as E_31/C_2/F_53",
      "causal_origin_location": "earliest causal origin span/node if visible, otherwise same as location",
      "human_preferred_location": "type-conditioned human anchor, usually same as location",
      "location_policy": "type_conditioned_human_anchor",
      "causal_origin_node": "graph node id for the upstream producer when visible, such as A_5",
      "node_location": "same graph node id used as the location anchor when visible",
      "reason": "根因",
      "impact": "LOW|MEDIUM|HIGH",
      "confidence": 0-1
    }}
  ],
  "causal_chains": [
    {{
      "root_cause_node": "graph node id such as F_12/E_4/A_7 or unknown",
      "fault_origin": "node/span where the bad evidence/action/failure first appears",
      "fault_usage": "node/span where the bad evidence/action/failure is used or ignored",
      "final_failure": "final answer/failure node/span",
      "causal_path": ["node_a", "node_b", "node_c"],
      "path_valid": true,
      "root_cause_type": "upstream_configuration_or_state_fault|wrong_evidence_used|tool_or_execution_failure|format_or_tool_argument_error|unsupported_or_invalid_reasoning|wrong_action_or_tool_choice|retry_or_recovery_failure|other",
      "evidence": "short evidence from input",
      "counterfactual_effect": "Would removing this origin likely prevent the final failure? yes/no/uncertain with short reason",
      "confidence": 0-1
    }}
  ],
  "taxonomy_hits": [
    {{
      "fault_id": "F1|F2|F3|F4|F5|F6|F7",
      "fault_name": "taxonomy name",
      "matched": true,
      "evidence": "evidence summary",
      "ccg_feature_match": "matched topology or causal feature",
      "confidence": 0-1
    }}
  ],
  "final_diagnosis": "one-sentence diagnosis",
  "diagnostic_report": {{
    "overall_failure": "2-4 sentences explaining what failed and why",
    "failure_stage": "the stage where the decisive failure occurred",
    "severity": "LOW|MEDIUM|HIGH",
    "root_cause_details": [{{"root_cause_index": 1, "title": "short title", "module": "module", "location": "span/node id", "mechanism": "how it happened", "impact": "downstream and final impact", "impact_level": "LOW|MEDIUM|HIGH", "confidence": 0-1, "evidence": "visible evidence", "role": "primary cause|trigger|amplifier|recovery failure"}}],
    "relationship_summary": "how the root causes jointly led to failure",
    "relationships": [{{"from_root_cause_index": 1, "to_root_cause_index": 2, "relation_type": "triggers|amplifies|blocks_recovery|parallel", "explanation": "evidence-backed relationship"}}],
    "failure_chain": [{{"order": 1, "stage": "stage", "location": "span/node id", "event": "what happened", "effect": "what it caused next"}}],
    "remediation_priorities": [{{"priority": 1, "target": "module/behavior", "action": "specific correction", "expected_effect": "blocked failure effect"}}],
    "conclusion": "plain-language conclusion"
  }}
}}

Rules:
1. Complete all coverage_checks first. There must be one coverage_checks item for every checklist id above.
2. Do not skip a category just because another category already matched.
3. Keep the earliest causal location in `causal_origin_location`, and for upstream configuration/state
   faults set both `causal_origin_location` and `root_causes[].location` to the upstream producer
   Action/Fact. The downstream Error belongs in errors[] and final_failure, not as the root cause.
4. Evidence must come from visible input fields. Do not invent evidence.
5. Every errors item and every root_causes item must have a concrete location. Prefer span_id as location. If no span_id is visible for that evidence, use the best visible graph node id such as E_31/C_2/F_53. Do not use unknown for errors/root_causes unless the input contains no concrete span/node identifier for the evidence.
5a. For coverage_checks with status=present, use the concrete span_id/node_id of the evidence whenever possible. status=absent may use unknown.
6. If explicit format_violations exist, you must inspect and report them even if impact is low.
7. If the digest contains candidate_causal_chains, treat them as Top-K graph-ranked candidates. Verify or reject the chains instead of searching the whole graph blindly.
   Use dependency_metrics and edge_chain to verify that each reported root cause can propagate to final_failure;
   temporal adjacency alone is not sufficient causal evidence.
8. If the digest contains rca_candidate_table, evaluate it before writing root_causes.
   For valid producer-consumer candidates, root_causes[].causal_origin_node and root_causes[].node_location
   must be the candidate `node_id`; root_causes[].location and causal_origin_location should use the
   candidate span_id when available, otherwise the candidate node_id.
   The first root_causes item must be the highest-priority valid producer candidate. Put secondary
   retrieval/retry/final-answer issues after the producer candidate, not before it.
9. causal_chains must use the three-part form fault_origin -> fault_usage -> final_failure. Prefer the highest-scoring valid candidate chain, but correct it if the visible evidence shows a better adjacent node on the same causal chain.
10. Before finalizing root_causes, ask: "If the downstream Error node were removed but the upstream bad Action/Fact stayed, would the next consumer still fail?" If yes, the upstream Action/Fact is the root cause and the Error is only a symptom.
11. root_causes[].location should follow the type-conditioned human-anchor rules above. Always preserve the chain's earliest origin in `causal_origin_location`.
12. Distinguish these common cases instead of collapsing them: invalid tool args, wrong tool choice, repeated failed retry, service/auth/environment failure, skipped planned steps, premature final_answer, unsupported conclusion.
13. Target >=3 errors and >=3 taxonomy_hits when evidence exists; if evidence is weak, use coverage_checks to mark uncertain and explain why.
14. Output must be valid JSON with correct types.
""".strip()


def make_judge_review_prompt(source_type, trace_id, digest_json, first_pass_json):
    checklist_text = _build_detection_checklist_text()
    return f"""
You are the second-pass reviewer for an Agent trace diagnosis.
Your job is to review the first-pass diagnosis and find omissions, especially inconsistent misses across similar traces.

Review workflow:
1. Re-run every checklist item below against the source digest.
2. Compare the checklist coverage against the first-pass JSON.
3. Add any missed root causes, fix weak locations, and remove unsupported claims.
4. Return a revised full JSON object with the exact same schema as the first pass.

Checklist:
{checklist_text}

Source digest:
{digest_json}

First-pass diagnosis JSON:
{first_pass_json}

Review requirements:
- Ensure every checklist item appears exactly once in coverage_checks.
- If coverage_checks says an issue is present but errors/root_causes missed it, add it.
- Ensure every errors item and every root_causes item has a concrete location. Prefer span_id; otherwise use the best visible graph node id. Do not leave errors/root_causes location as unknown when any supporting span/node id is visible.
- For present coverage_checks, fill a concrete location when the source digest contains a supporting span/node id. absent checks may keep unknown.
- If a tool/service/context/planning issue exists, do not let it be hidden only as Hallucination/Other.
- Keep causal origins separate from human-preferred exact anchors.
- If source digest contains rca_candidate_table, review root_causes against that table first.
  A valid upstream_state_producer candidate should stay the root unless the visible evidence contradicts it.
  Do not replace a valid producer candidate with an Error, retry, repository-inspection, or final-answer node.
  For valid producer candidates, keep causal_origin_node/node_location as the candidate node_id and use span_id when available.
  root_causes[0] must be the best valid producer candidate when one exists; move secondary retrieval/retry/final-answer issues after it.
- Preserve and validate causal_chains. Each valid chain should explain fault_origin -> fault_usage -> final_failure and include a counterfactual_effect judgment.
- Apply type-conditioned human-anchor tie-breaking: format/tool execution failures prefer first visible Error; bad tool-choice/argument failures prefer explicit bad Action else Error; planning/reasoning/hallucination failures prefer first bad Planning/Conclusion claim; final-answer failures prefer final answer span. Preserve earliest origin in `causal_origin_location`.
- Keep output as one valid JSON object only.
""".strip()


SPAN_ID_RE = re.compile(r"\b[0-9a-f]{16}\b", re.IGNORECASE)
NODE_ID_RE = re.compile(r"\b[A-Z]_[0-9]+\b")


def extract_best_visible_location(text: str, node_to_span=None) -> str:
    text = str(text or "")
    span_match = SPAN_ID_RE.search(text)
    if span_match:
        return span_match.group(0).lower()

    node_match = NODE_ID_RE.search(text)
    if node_match:
        node_id = node_match.group(0)
        if isinstance(node_to_span, dict) and node_id in node_to_span:
            return node_to_span[node_id]
        return node_id

    return "unknown"


def build_node_to_span_map(graph_obj):
    node_to_span = {}
    if not isinstance(graph_obj, dict):
        return node_to_span
    nodes = graph_obj.get("nodes", [])
    if not isinstance(nodes, list):
        return node_to_span

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        span_id = str(node.get("span_id") or "").strip().lower()
        if node_id and span_id and SPAN_ID_RE.fullmatch(span_id):
            node_to_span[node_id] = span_id
    return node_to_span


def remap_error_locations_to_span_id(result, node_to_span):
    if not isinstance(result, dict):
        return result
    if not isinstance(node_to_span, dict) or not node_to_span:
        return result

    errors = result.get("errors")
    if not isinstance(errors, list):
        return result

    for item in errors:
        if not isinstance(item, dict):
            continue

        raw_location = str(item.get("location") or "").strip()
        if SPAN_ID_RE.fullmatch(raw_location):
            item["location"] = raw_location.lower()
            continue

        if raw_location in node_to_span:
            item["node_location"] = raw_location
            item["location"] = node_to_span[raw_location]
            continue

        evidence = str(item.get("evidence") or "")
        description = str(item.get("description") or "")
        note = str(item.get("note") or "")
        reason = str(item.get("reason") or "")
        fallback = extract_best_visible_location(f"{evidence} {description} {note} {reason}", node_to_span)
        if fallback != "unknown":
            item["location"] = fallback
        elif raw_location.lower() == "unknown":
            item["location"] = "unknown"
        else:
            # Keep backward compatible value if no deterministic span mapping exists.
            item["location"] = raw_location

    return result


def remap_location_list_to_span_id(items, node_to_span):
    if not isinstance(items, list):
        return items
    if not isinstance(node_to_span, dict) or not node_to_span:
        return items

    for item in items:
        if not isinstance(item, dict):
            continue

        raw_location = str(item.get("location") or "").strip()
        if SPAN_ID_RE.fullmatch(raw_location):
            item["location"] = raw_location.lower()
            continue

        if raw_location in node_to_span:
            item["node_location"] = raw_location
            item["location"] = node_to_span[raw_location]
            continue

        evidence = str(item.get("evidence") or "")
        note = str(item.get("note") or "")
        description = str(item.get("description") or "")
        reason = str(item.get("reason") or "")
        fallback = extract_best_visible_location(f"{evidence} {note} {description} {reason}", node_to_span)
        if fallback != "unknown":
            item["location"] = fallback
        elif raw_location.lower() == "unknown":
            item["location"] = "unknown"
        else:
            item["location"] = raw_location

    return items


def _normalize_chain_path(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in re.split(r"->|,|\s+", value) if part.strip()]
    return []


def _node_or_unknown(value) -> str:
    text = str(value or "").strip()
    return text if text else "unknown"


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "valid"}:
            return True
        if text in {"false", "no", "n", "0", "invalid"}:
            return False
    return default


def normalize_causal_chains(result, node_to_span=None):
    chains = result.get("causal_chains")
    if not isinstance(chains, list):
        result["causal_chains"] = []
        return result

    normalized = []
    for item in chains:
        if not isinstance(item, dict):
            continue

        path = _normalize_chain_path(item.get("causal_path"))
        root_node = _node_or_unknown(item.get("root_cause_node") or item.get("fault_origin"))
        original_root_node = root_node
        fault_origin = _node_or_unknown(item.get("fault_origin") or root_node)
        fault_usage = _node_or_unknown(item.get("fault_usage"))
        final_failure = _node_or_unknown(item.get("final_failure"))

        if not path:
            path = [node for node in [fault_origin, fault_usage, final_failure] if node != "unknown"]

        origin_disambiguation = None
        if fault_origin != "unknown":
            downstream_nodes = {fault_usage, final_failure}
            if root_node in {"unknown", *downstream_nodes}:
                root_node = fault_origin
                origin_disambiguation = {
                    "policy": "causal_origin_normalization",
                    "original_root_cause_node": original_root_node,
                    "reason": "root_cause_node pointed to a usage/final node on the same chain",
                }
            elif root_node in path and fault_origin in path and path.index(root_node) > path.index(fault_origin):
                root_node = fault_origin
                origin_disambiguation = {
                    "policy": "causal_origin_normalization",
                    "original_root_cause_node": original_root_node,
                    "reason": "root_cause_node was downstream of fault_origin on the causal_path",
                }

        normalized_item = {
            "root_cause_node": root_node,
            "fault_origin": fault_origin,
            "fault_usage": fault_usage,
            "final_failure": final_failure,
            "causal_path": path,
            "path_valid": _coerce_bool(item.get("path_valid"), default=len(path) >= 2),
            "root_cause_type": str(item.get("root_cause_type") or "other").strip() or "other",
            "evidence": str(item.get("evidence") or "").strip(),
            "counterfactual_effect": str(item.get("counterfactual_effect") or "").strip(),
            "confidence": item.get("confidence", 0),
        }
        if origin_disambiguation:
            normalized_item["location_disambiguation"] = origin_disambiguation

        if isinstance(node_to_span, dict):
            for field in ["root_cause_node", "fault_origin", "fault_usage", "final_failure"]:
                node_id = normalized_item.get(field)
                if node_id in node_to_span:
                    normalized_item[f"{field}_span_id"] = node_to_span[node_id]

        normalized.append(normalized_item)

    result["causal_chains"] = normalized
    return result


def _chain_field_aliases(chain: dict, field: str, node_to_span=None) -> set:
    aliases = set()
    value = str(chain.get(field) or "").strip()
    if value and value != "unknown":
        aliases.add(value)
    span_value = str(chain.get(f"{field}_span_id") or "").strip()
    if span_value and span_value != "unknown":
        aliases.add(span_value.lower())
    if isinstance(node_to_span, dict) and value in node_to_span:
        aliases.add(str(node_to_span[value]).lower())
    return aliases


def _norm_location_id(value: str) -> str:
    text = str(value or "").strip()
    return text.lower() if SPAN_ID_RE.fullmatch(text) else text


def _origin_location_for_chain(chain: dict, node_to_span=None) -> str:
    origin = str(chain.get("fault_origin") or chain.get("root_cause_node") or "").strip()
    origin_span = str(chain.get("fault_origin_span_id") or "").strip()
    if origin_span:
        return origin_span.lower()
    if isinstance(node_to_span, dict) and origin in node_to_span:
        return node_to_span[origin]
    return origin or "unknown"


NODE_TYPE_BY_PREFIX = {
    "I": "Intent",
    "P": "Planning",
    "A": "Action",
    "F": "Fact",
    "E": "Error",
    "C": "Conclusion",
}


def _node_type_from_id(node_id: str) -> str:
    text = str(node_id or "").strip()
    match = re.match(r"^([A-Z])_\d+$", text)
    if not match:
        return "unknown"
    return NODE_TYPE_BY_PREFIX.get(match.group(1), "unknown")


def _location_for_node_id(node_id: str, node_to_span=None) -> str:
    node_id = str(node_id or "").strip()
    if not node_id or node_id == "unknown":
        return "unknown"
    if SPAN_ID_RE.fullmatch(node_id):
        return node_id.lower()
    if isinstance(node_to_span, dict) and node_id in node_to_span:
        return str(node_to_span[node_id]).lower()
    return node_id


def _chain_ordered_nodes(chain: dict, node_to_span=None) -> list:
    ordered = []
    seen = set()
    role_by_node = {}
    for role in ["fault_origin", "root_cause_node", "fault_usage", "final_failure"]:
        node_id = str(chain.get(role) or "").strip()
        if node_id and node_id != "unknown":
            role_by_node.setdefault(node_id, role)

    for node_id in _normalize_chain_path(chain.get("causal_path")):
        if not node_id or node_id == "unknown" or node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(
            {
                "node": node_id,
                "location": _location_for_node_id(node_id, node_to_span),
                "type": _node_type_from_id(node_id),
                "role": role_by_node.get(node_id, "path"),
            }
        )

    for role in ["fault_origin", "root_cause_node", "fault_usage", "final_failure"]:
        node_id = str(chain.get(role) or "").strip()
        if not node_id or node_id == "unknown" or node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(
            {
                "node": node_id,
                "location": _location_for_node_id(node_id, node_to_span),
                "type": _node_type_from_id(node_id),
                "role": role,
            }
        )
    return ordered


def _first_node_of_types(nodes: list, node_types: set):
    for node in nodes:
        if node.get("type") in node_types and node.get("location") not in {"", "unknown"}:
            return node
    return None


def _role_node(nodes: list, role: str):
    for node in nodes:
        if node.get("role") == role and node.get("location") not in {"", "unknown"}:
            return node
    return None


def _chain_text_for_policy(chain: dict, root_cause_item: dict) -> str:
    fields = [
        chain.get("root_cause_type"),
        chain.get("evidence"),
        chain.get("counterfactual_effect"),
        root_cause_item.get("module"),
        root_cause_item.get("reason"),
        root_cause_item.get("evidence"),
        root_cause_item.get("description"),
        root_cause_item.get("category"),
    ]
    return " ".join(str(value or "") for value in fields).lower()


def _select_type_conditioned_anchor(chain: dict, root_cause_item: dict, node_to_span=None) -> dict:
    nodes = _chain_ordered_nodes(chain, node_to_span)
    origin = _role_node(nodes, "fault_origin") or _role_node(nodes, "root_cause_node")
    usage = _role_node(nodes, "fault_usage")
    final = _role_node(nodes, "final_failure")
    first_error = _first_node_of_types(nodes, {"Error"})
    first_action = _first_node_of_types(nodes, {"Action"})
    first_reasoning = _first_node_of_types(nodes, {"Planning", "Conclusion"})
    first_planning = _first_node_of_types(nodes, {"Planning"})
    first_fact = _first_node_of_types(nodes, {"Fact"})

    root_type = str(chain.get("root_cause_type") or "other").strip()
    root_type_l = root_type.lower()
    text = _chain_text_for_policy(chain, root_cause_item)

    selected = origin or usage or final or (nodes[0] if nodes else None)
    anchor_role = "earliest_origin"
    reason = "fallback to earliest causal origin"

    format_terms = any(
        term in text
        for term in [
            "format",
            "formatting",
            "missing required tag",
            "schema",
            "syntax",
            "parse",
            "invalid json",
            "typeerror",
            "code execution failed",
        ]
    )
    final_terms = any(
        term in text
        for term in [
            "completion failure",
            "did not provide the final answer",
            "failed to provide the final answer",
            "missing final answer",
            "no final answer",
            "stops on the tool call",
        ]
    )
    retry_terms = any(term in text for term in ["retry", "again", "repeated", "resource abuse", "loop"])
    evidence_terms = any(term in text for term in ["retrieval", "search", "evidence", "observation", "result", "misinterpret"])
    unsupported_terms = any(term in text for term in ["unsupported", "hallucinat", "language-only", "claim", "reasoning"])
    upstream_state_terms = any(
        term in text
        for term in [
            "upstream",
            "setup",
            "configured",
            "configuration",
            "config",
            "stored",
            "wrote",
            "written",
            "bad path",
            "bad state",
            "bad value",
            "bad artifact",
            "invalid value",
            "nonexistent",
            "unsupported model",
            "negative timeout",
            "missing key",
            "schema key",
        ]
    )

    if upstream_state_terms and (origin or first_action):
        selected = origin or first_action or selected
        anchor_role = "upstream_root_action"
        reason = "upstream configuration/state contamination should anchor at the first bad setup Action/Fact, not the downstream Error symptom"
    elif root_type_l in {"final_answer_symptom", "final_answer_or_completion_failure"} or final_terms:
        selected = final or _first_node_of_types(nodes, {"Conclusion", "Action"}) or selected
        anchor_role = "final_symptom"
        reason = "final-answer/completion failures match human final-symptom anchors"
    elif root_type_l in {"tool_or_execution_failure"}:
        selected = first_error or first_fact or origin or selected
        anchor_role = "first_visible_error"
        reason = "tool execution failures match human first-visible-error anchors"
    elif root_type_l in {"format_or_tool_argument_error"}:
        if format_terms and first_error:
            selected = first_error
            anchor_role = "first_visible_error"
            reason = "format/constraint failures are usually annotated at the Error span"
        elif first_action:
            selected = first_action
            anchor_role = "bad_action_call"
            reason = "explicit tool argument/choice failures are annotated at the bad Action span"
        else:
            selected = first_error or origin or selected
            anchor_role = "first_visible_error" if first_error else "earliest_origin"
            reason = "tool argument failure fallback"
    elif root_type_l in {"wrong_action_or_tool_choice"}:
        selected = first_action or first_planning or origin or selected
        anchor_role = "bad_action_call" if first_action else "earliest_origin"
        reason = "wrong tool/action choices match human bad-action anchors"
    elif root_type_l in {"retry_or_recovery_failure"} or retry_terms:
        selected = first_action or first_error or usage or origin or selected
        anchor_role = "bad_action_call" if first_action else "first_visible_error"
        reason = "retry/resource-abuse failures match human repeated-action or first-error anchors"
    elif root_type_l in {
        "wrong_evidence_used",
        "wrong_or_low_quality_evidence_used",
        "crucial_fact_ignored",
        "tool_output_misinterpretation",
    } or evidence_terms:
        selected = first_reasoning or first_action or first_fact or usage or origin or selected
        anchor_role = "unsupported_usage" if selected == usage else "earliest_origin"
        reason = "retrieval/evidence failures match human bad-query or first bad-claim anchors"
    elif root_type_l in {
        "unsupported_or_invalid_reasoning",
        "agent_trace_failure",
        "instruction_violation",
        "context_memory_failure",
    } or unsupported_terms:
        selected = first_reasoning or origin or usage or selected
        anchor_role = "earliest_origin"
        reason = "reasoning/hallucination failures match human first bad planning/claim anchors"

    if not selected:
        selected = {"node": "unknown", "location": "unknown", "type": "unknown", "role": "unknown"}

    return {
        "selected": selected,
        "origin": origin or selected,
        "anchor_role": anchor_role,
        "rule_reason": reason,
        "root_cause_type": root_type,
    }


def _apply_origin_preferred_root_cause_locations(result, node_to_span=None):
    chains = result.get("causal_chains")
    root_causes = result.get("root_causes")
    if not isinstance(chains, list) or not chains or not isinstance(root_causes, list):
        result.setdefault(
            "location_disambiguation",
            {"policy": "type_conditioned_human_anchor", "rewritten_root_causes": 0},
        )
        return result

    chain_records = []
    for chain in chains:
        if not isinstance(chain, dict):
            continue
        origin_location = _origin_location_for_chain(chain, node_to_span)
        if not origin_location or origin_location == "unknown":
            continue
        all_aliases = set()
        for field in ["root_cause_node", "fault_origin", "fault_usage", "final_failure"]:
            all_aliases.update(_chain_field_aliases(chain, field, node_to_span))
        for node in _normalize_chain_path(chain.get("causal_path")):
            if node and node != "unknown":
                all_aliases.add(node)
                if isinstance(node_to_span, dict) and node in node_to_span:
                    all_aliases.add(str(node_to_span[node]).lower())
        chain_records.append(
            {
                "chain": chain,
                "origin_location": origin_location,
                "origin_node": chain.get("fault_origin") or chain.get("root_cause_node"),
                "aliases": {_norm_location_id(v) for v in all_aliases if v},
            }
        )

    rewritten = 0
    unchanged = 0
    for idx, item in enumerate(root_causes):
        if not isinstance(item, dict):
            continue
        raw_location = str(item.get("location") or "unknown").strip() or "unknown"
        normalized_location = _norm_location_id(raw_location)
        record = None
        if normalized_location != "unknown":
            record = next((row for row in chain_records if normalized_location in row["aliases"]), None)
        if record is None and len(root_causes) == 1 and chain_records:
            record = chain_records[0]
        if record is None:
            continue

        anchor = _select_type_conditioned_anchor(record["chain"], item, node_to_span)
        origin_location = anchor["origin"].get("location") or record["origin_location"]
        preferred_location = anchor["selected"].get("location") or origin_location
        if not preferred_location or preferred_location == "unknown":
            continue

        item["causal_origin_location"] = origin_location
        item["human_preferred_location"] = preferred_location
        item["location_policy"] = "type_conditioned_human_anchor"
        if record.get("origin_node"):
            item["causal_origin_node"] = record["origin_node"]
        if anchor["selected"].get("node"):
            item["node_location"] = anchor["selected"]["node"]
        if anchor["root_cause_type"]:
            item["root_cause_type"] = anchor["root_cause_type"]

        if _norm_location_id(preferred_location) != normalized_location:
            item["original_location"] = raw_location
            item["location"] = preferred_location
            rewritten += 1
        else:
            unchanged += 1

        item["location_disambiguation"] = {
            "policy": "type_conditioned_human_anchor",
            "anchor_role": anchor["anchor_role"],
            "reason": anchor["rule_reason"],
            "fault_origin": record["chain"].get("fault_origin"),
            "fault_usage": record["chain"].get("fault_usage"),
            "final_failure": record["chain"].get("final_failure"),
            "causal_path": record["chain"].get("causal_path", []),
            "causal_origin_location": origin_location,
            "human_preferred_location": preferred_location,
            "root_cause_type": anchor["root_cause_type"],
        }

    result["location_disambiguation"] = {
        "policy": "type_conditioned_human_anchor",
        "rewritten_root_causes": rewritten,
        "unchanged_root_causes": unchanged,
    }
    return result


def parse_json_from_llm(text):
    def _clean_common_artifacts(raw):
        s = (raw or "").strip()
        # Keep smart quotes untouched. Replacing them can corrupt otherwise valid JSON
        # when they appear inside a normal JSON string value.
        s = s.replace("\ufeff", "")
        return s

    def _extract_code_block(s):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
        return m.group(1).strip() if m else s

    def _extract_balanced_json_obj(s):
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for idx in range(start, len(s)):
            ch = s[idx]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : idx + 1]
        return None

    def _json_or_literal_eval(s):
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            return obj if isinstance(obj, (dict, list)) else None
        except Exception:
            return None

    text = _clean_common_artifacts(text)
    candidates = []
    candidates.append(text)
    candidates.append(_extract_code_block(text))

    balanced = _extract_balanced_json_obj(text)
    if balanced:
        candidates.append(balanced)

    regex_match = re.search(r"\{[\s\S]*\}", text)
    if regex_match:
        candidates.append(regex_match.group(0))

    for cand in candidates:
        if not cand:
            continue
        cand = _clean_common_artifacts(cand)
        parsed = _json_or_literal_eval(cand)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("LLM output is not valid JSON")


def _extract_usage(resp):
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    def _get(v, key):
        if isinstance(v, dict):

            
            return v.get(key)
        return getattr(v, key, None)

    return {
        "prompt_tokens": _get(usage, "prompt_tokens"),
        "completion_tokens": _get(usage, "completion_tokens"),
        "total_tokens": _get(usage, "total_tokens"),
    }


def _merge_usage(*usages):
    merged = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        merged["prompt_tokens"] += usage.get("prompt_tokens") or 0
        merged["completion_tokens"] += usage.get("completion_tokens") or 0
        merged["total_tokens"] += usage.get("total_tokens") or 0
    return merged


def _extract_text_content(resp):
    try:
        msg = resp.choices[0].message
        content = getattr(msg, "content", "")
    except Exception:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(parts).strip()

    return str(content).strip()


def _chat_completion_with_optional_max_tokens(client, **request):
    import time
    max_tokens = request.pop("max_tokens", None)
    thinking_mode = request.pop("thinking_mode", DEEPSEEK_THINKING_MODE)
    if max_tokens and max_tokens > 0:
        request["max_tokens"] = max_tokens
    base_url = str(getattr(client, "base_url", "") or "")
    model = str(request.get("model") or "")
    if "api.deepseek.com" in base_url and model.startswith("deepseek"):
        extra_body = dict(request.get("extra_body") or {})
        extra_body["thinking"] = {"type": thinking_mode}
        request["extra_body"] = extra_body
    for attempt in range(6):
        try:
            return client.chat.completions.create(**request)
        except Exception as e:
            if "429" in str(e) and attempt < 5:
                time.sleep(2 ** attempt * 5)
                continue
            raise


def _repair_json_with_llm(client, model, bad_text, repair_max_tokens=DEFAULT_JSON_REPAIR_MAX_TOKENS):
    repair_prompt = f"""
请把下面文本改写成严格 JSON 对象，仅输出 JSON，不要解释。
要求：
- 保持原有字段语义；
- 若缺字段可补空值；
- 使用双引号；
- 不要 markdown 代码块。

原文本：
{bad_text[:12000]}
""".strip()

    resp = _chat_completion_with_optional_max_tokens(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "You are a JSON repair tool. Output only valid JSON object."},
            {"role": "user", "content": repair_prompt},
        ],
        temperature=0.0,
        max_tokens=repair_max_tokens,
    )
    return _extract_text_content(resp)


class JudgeOutputParseError(Exception):
    def __init__(self, source_type, raw_text, repaired_text, first_error, second_error):
        super().__init__("LLM output is not valid JSON")
        self.source_type = source_type
        self.raw_text = raw_text
        self.repaired_text = repaired_text
        self.first_error = str(first_error)
        self.second_error = str(second_error)


def _parse_judge_json_with_repair(client, model, source_type, text, repair_max_tokens=DEFAULT_JSON_REPAIR_MAX_TOKENS):
    try:
        return parse_json_from_llm(text)
    except Exception as e1:
        repaired = _repair_json_with_llm(client, model, text, repair_max_tokens=repair_max_tokens)
        try:
            return parse_json_from_llm(repaired)
        except Exception as e2:
            raise JudgeOutputParseError(source_type, text, repaired, e1, e2)


def call_deepseek_judge(
    client,
    model,
    source_type,
    trace_id,
    digest,
    max_input_chars=28000,
    enable_review=True,
    judge_max_tokens=DEFAULT_JUDGE_MAX_TOKENS,
    judge_review_max_tokens=DEFAULT_JUDGE_REVIEW_MAX_TOKENS,
    json_repair_max_tokens=DEFAULT_JSON_REPAIR_MAX_TOKENS,
    review_first_pass_max_chars=DEFAULT_REVIEW_FIRST_PASS_MAX_CHARS,
    prompt_variant="default",
):
    digest_text = safe_dump(digest, max_chars=max_input_chars)
    _prompt_fn = make_judge_prompt_swe if prompt_variant == "swe" else make_judge_prompt
    prompt = _prompt_fn(source_type, trace_id, digest_text)

    resp = _chat_completion_with_optional_max_tokens(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict evaluator. Output only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=judge_max_tokens,
    )
    text = _extract_text_content(resp)
    parsed = _parse_judge_json_with_repair(
        client,
        model,
        source_type,
        text,
        repair_max_tokens=json_repair_max_tokens,
    )
    usage = _extract_usage(resp)

    if enable_review:
        first_pass_json = safe_dump(parsed, max_chars=review_first_pass_max_chars)
        review_prompt = make_judge_review_prompt(source_type, trace_id, digest_text, first_pass_json)
        review_resp = _chat_completion_with_optional_max_tokens(
            client,
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict diagnosis reviewer. Output only valid JSON."},
                {"role": "user", "content": review_prompt},
            ],
            temperature=0.0,
            max_tokens=judge_review_max_tokens,
        )
        review_text = _extract_text_content(review_resp)
        try:
            parsed = _parse_judge_json_with_repair(
                client,
                model,
                source_type,
                review_text,
                repair_max_tokens=json_repair_max_tokens,
            )
        except JudgeOutputParseError:
            pass
        usage = _merge_usage(usage, _extract_usage(review_resp))

    usage["input_chars"] = len(digest_text)
    usage["estimated_input_tokens"] = max(1, len(digest_text) // 4)
    return parsed, usage


def find_graph_file(graph_dir: Path, trace_id: str):
    exact = graph_dir / f"local_{trace_id}_graph.json"
    if exact.exists():
        return exact
    candidates = sorted(graph_dir.glob(f"*{trace_id}*_graph.json"))
    return candidates[0] if candidates else None


def find_pruned_graph_file(graph_dir: Path, trace_id: str, pruning_strategy: str):
    exact = graph_dir / f"local_{trace_id}_pruning" / f"{pruning_strategy}.json"
    if exact.exists():
        return exact

    candidates = sorted(graph_dir.glob(f"*{trace_id}*_pruning/{pruning_strategy}.json"))
    return candidates[0] if candidates else None


def find_agenttrace_graph_file(graph_dir: Path, trace_id: str):
    return find_pruned_graph_file(graph_dir, trace_id, "agenttrace_suspicious_paths")


def find_graph_input_file(graph_dir: Path, trace_id: str, graph_source: str, pruning_strategy: str):
    if graph_source == "agenttrace_first":
        agenttrace_file = find_agenttrace_graph_file(graph_dir, trace_id)
        if agenttrace_file:
            return agenttrace_file, "pruning", "agenttrace_suspicious_paths"
        full_file = find_graph_file(graph_dir, trace_id)
        if full_file:
            return full_file, "full", None
        return None, graph_source, None
    if graph_source == "pruning":
        return find_pruned_graph_file(graph_dir, trace_id, pruning_strategy), "pruning", pruning_strategy
    return find_graph_file(graph_dir, trace_id), "full", None


def normalize_judge_result(result, trace_id, source_type, node_to_span=None):
    if not isinstance(result, dict):
        result = {}
    result.setdefault("trace_id", trace_id)
    result.setdefault("source", source_type)
    if not isinstance(result.get("errors"), list):
        result["errors"] = []
    normalized_errors = []
    for item in result["errors"]:
        if not isinstance(item, dict):
            continue
        item["location"] = str(item.get("location") or "unknown").strip() or "unknown"
        if item["location"].lower() == "unknown":
            fallback = extract_best_visible_location(
                f"{item.get('evidence') or ''} {item.get('description') or ''} {item.get('note') or ''}",
                node_to_span,
            )
            if fallback != "unknown":
                item["location"] = fallback
        normalized_errors.append(item)
    result["errors"] = normalized_errors

    if not isinstance(result.get("scores"), list):
        result["scores"] = []
    if not isinstance(result.get("root_causes"), list):
        result["root_causes"] = []
    normalized_root_causes = []
    for item in result["root_causes"]:
        if isinstance(item, dict):
            normalized = dict(item)
        else:
            normalized = {"module": "unknown", "reason": str(item or ""), "impact": "UNKNOWN", "confidence": 0}
        normalized["location"] = str(normalized.get("location") or "unknown").strip() or "unknown"
        if normalized["location"].lower() == "unknown":
            fallback = extract_best_visible_location(
                f"{normalized.get('reason') or ''} {normalized.get('evidence') or ''} {normalized.get('description') or ''}",
                node_to_span,
            )
            if fallback != "unknown":
                normalized["location"] = fallback
        normalized_root_causes.append(normalized)
    result["root_causes"] = normalized_root_causes

    if not isinstance(result.get("taxonomy_hits"), list):
        result["taxonomy_hits"] = []
    result = normalize_causal_chains(result, node_to_span=node_to_span)
    result = _apply_origin_preferred_root_cause_locations(result, node_to_span=node_to_span)
    coverage_checks = result.get("coverage_checks")
    if not isinstance(coverage_checks, list):
        coverage_checks = []
    checks_by_id = {}
    for item in coverage_checks:
        if not isinstance(item, dict):
            continue
        check_id = str(item.get("check_id") or "").strip()
        if not check_id:
            continue
        normalized = {
            "check_id": check_id,
            "label": str(item.get("label") or "").strip(),
            "status": str(item.get("status") or "uncertain").strip().lower(),
            "mapped_category": str(item.get("mapped_category") or "Other").strip(),
            "location": str(item.get("location") or "unknown").strip(),
            "evidence": str(item.get("evidence") or "").strip(),
            "note": str(item.get("note") or "").strip(),
        }
        if normalized["status"] not in {"present", "absent", "uncertain"}:
            normalized["status"] = "uncertain"
        checks_by_id[check_id] = normalized
    result["coverage_checks"] = [
        checks_by_id.get(default_item["check_id"], default_item)
        for default_item in _empty_coverage_checklist()
    ]
    result.setdefault("final_diagnosis", "")
    if source_type in {"trace_graph", "pure_trace_graph"}:
        result = remap_error_locations_to_span_id(result, node_to_span)
        result["coverage_checks"] = remap_location_list_to_span_id(result["coverage_checks"], node_to_span)
        result["root_causes"] = remap_location_list_to_span_id(result["root_causes"], node_to_span)
    result["diagnostic_report"] = build_diagnostic_report(result)
    if source_type in {"trace_graph", "pure_trace_graph"}:
        report = result["diagnostic_report"]
        report["root_cause_details"] = remap_location_list_to_span_id(
            report.get("root_cause_details", []), node_to_span
        )
        report["failure_chain"] = remap_location_list_to_span_id(
            report.get("failure_chain", []), node_to_span
        )
    return result


def make_openai_client(api_key, base_url):
    return OpenAI(api_key=api_key, base_url=base_url)


def update_summary_token_usage(summary, raw_usage, graph_usage, pure_graph_usage):
    summary["token_usage"]["raw_prompt_tokens"] += raw_usage.get("prompt_tokens") or 0
    summary["token_usage"]["raw_completion_tokens"] += raw_usage.get("completion_tokens") or 0
    summary["token_usage"]["raw_total_tokens"] += raw_usage.get("total_tokens") or 0
    summary["token_usage"]["graph_prompt_tokens"] += graph_usage.get("prompt_tokens") or 0
    summary["token_usage"]["graph_completion_tokens"] += graph_usage.get("completion_tokens") or 0
    summary["token_usage"]["graph_total_tokens"] += graph_usage.get("total_tokens") or 0
    summary["token_usage"]["pure_graph_prompt_tokens"] += pure_graph_usage.get("prompt_tokens") or 0
    summary["token_usage"]["pure_graph_completion_tokens"] += pure_graph_usage.get("completion_tokens") or 0
    summary["token_usage"]["pure_graph_total_tokens"] += pure_graph_usage.get("total_tokens") or 0
    summary["token_usage"]["raw_total_input_chars"] += raw_usage.get("input_chars") or 0
    summary["token_usage"]["graph_total_input_chars"] += graph_usage.get("input_chars") or 0
    summary["token_usage"]["pure_graph_total_input_chars"] += pure_graph_usage.get("input_chars") or 0


def judge_one_source(
    api_key,
    base_url,
    model,
    source_type,
    trace_id,
    digest,
    max_input_chars,
    enable_review,
    judge_max_tokens,
    judge_review_max_tokens,
    json_repair_max_tokens,
    review_first_pass_max_chars,
    prompt_variant="default",
):
    client = make_openai_client(api_key, base_url)
    return call_deepseek_judge(
        client,
        model,
        source_type,
        trace_id,
        digest,
        max_input_chars=max_input_chars,
        enable_review=enable_review,
        judge_max_tokens=judge_max_tokens,
        judge_review_max_tokens=judge_review_max_tokens,
        json_repair_max_tokens=json_repair_max_tokens,
        review_first_pass_max_chars=review_first_pass_max_chars,
        prompt_variant=prompt_variant,
    )


def process_trace_file(
    gaia_file,
    graph_dir,
    raw_out_dir,
    graph_out_dir,
    merged_out_dir,
    debug_dir,
    model,
    base_url,
    api_key,
    max_spans,
    max_input_chars,
    raw_input_mode,
    raw_full_trace_ids,
    raw_full_max_input_chars,
    graph_source,
    pruning_strategy,
    judge_enable_review,
    judge_max_tokens,
    judge_review_max_tokens,
    json_repair_max_tokens,
    review_first_pass_max_chars,
    prompt_variant="default",
):
    trace_id = gaia_file.stem
    row = {"trace_id": trace_id, "status": "ok", "issues": []}
    raw_usage = None
    graph_usage = None
    pure_graph_usage = None

    try:
        raw_trace = load_json(gaia_file)
        graph_file, resolved_graph_source, resolved_pruning_strategy = find_graph_input_file(
            graph_dir, trace_id, graph_source, pruning_strategy
        )
        if graph_file is None:
            if graph_source == "pruning":
                raise RuntimeError(f"未找到对应 pruning 图文件: trace_id={trace_id}, strategy={pruning_strategy}")
            if graph_source == "agenttrace_first":
                raise RuntimeError(
                    f"未找到可用图文件: trace_id={trace_id}, 已尝试 agenttrace_suspicious_paths.json 和 full graph"
                )
            raise RuntimeError(f"未找到对应 graph 文件: {trace_id}")
        graph_obj = load_json(graph_file)
        node_to_span = build_node_to_span_map(graph_obj)

        effective_raw_mode = resolve_raw_input_mode(trace_id, raw_input_mode, raw_full_trace_ids)
        raw_payload = build_raw_trace_payload(raw_trace, effective_raw_mode, max_spans=max_spans)
        raw_input_char_limit = max_input_chars
        if effective_raw_mode == "full":
            raw_input_char_limit = raw_full_max_input_chars if raw_full_max_input_chars and raw_full_max_input_chars > 0 else None

        graph_digest = build_graph_digest(graph_obj, representation_mode=GRAPH_REPRESENTATION_HYBRID)
        pure_graph_digest = build_graph_digest(graph_obj, representation_mode=GRAPH_REPRESENTATION_PURE)

        pure_graph_out_dir = graph_out_dir.parent / "pure_graph_judgements"
        pure_graph_out_dir.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=3) as inner_executor:
            raw_future = inner_executor.submit(
                judge_one_source,
                api_key, base_url, model, "raw_trace", trace_id,
                raw_payload, raw_input_char_limit, judge_enable_review,
                judge_max_tokens, judge_review_max_tokens,
                json_repair_max_tokens, review_first_pass_max_chars, prompt_variant,
            )
            graph_future = inner_executor.submit(
                judge_one_source,
                api_key, base_url, model, "trace_graph", trace_id,
                graph_digest, max_input_chars, judge_enable_review,
                judge_max_tokens, judge_review_max_tokens,
                json_repair_max_tokens, review_first_pass_max_chars, prompt_variant,
            )
            pure_graph_future = inner_executor.submit(
                judge_one_source,
                api_key, base_url, model, "pure_trace_graph", trace_id,
                pure_graph_digest, max_input_chars, judge_enable_review,
                judge_max_tokens, judge_review_max_tokens,
                json_repair_max_tokens, review_first_pass_max_chars, prompt_variant,
            )
            raw_j, raw_usage = raw_future.result()
            graph_j, graph_usage = graph_future.result()
            pure_graph_j, pure_graph_usage = pure_graph_future.result()

        raw_j = normalize_judge_result(raw_j, trace_id, "raw_trace")
        graph_j = normalize_judge_result(graph_j, trace_id, "trace_graph", node_to_span=node_to_span)
        pure_graph_j = normalize_judge_result(pure_graph_j, trace_id, "pure_trace_graph", node_to_span=node_to_span)

        with open(raw_out_dir / f"{trace_id}.json", "w", encoding="utf-8") as f:
            json.dump(raw_j, f, ensure_ascii=False, indent=2)
        with open(graph_out_dir / f"{trace_id}.json", "w", encoding="utf-8") as f:
            json.dump(graph_j, f, ensure_ascii=False, indent=2)
        with open(pure_graph_out_dir / f"{trace_id}.json", "w", encoding="utf-8") as f:
            json.dump(pure_graph_j, f, ensure_ascii=False, indent=2)

        merged = {
            "trace_id": trace_id,
            "raw_trace_judgement": raw_j,
            "graph_judgement": graph_j,
            "pure_graph_judgement": pure_graph_j,
            "token_usage": {
                "raw": raw_usage,
                "graph": graph_usage,
                "pure_graph": pure_graph_usage,
                "delta_total_tokens": (
                    (raw_usage.get("total_tokens") or 0) - (graph_usage.get("total_tokens") or 0)
                ),
                "graph_token_saving_ratio": (
                    round(
                        ((raw_usage.get("total_tokens") or 0) - (graph_usage.get("total_tokens") or 0))
                        / (raw_usage.get("total_tokens") or 1),
                        4,
                    )
                    if raw_usage.get("total_tokens")
                    else None
                ),
                "delta_total_tokens_raw_minus_pure_graph": (
                    (raw_usage.get("total_tokens") or 0) - (pure_graph_usage.get("total_tokens") or 0)
                ),
                "pure_graph_token_saving_ratio": (
                    round(
                        ((raw_usage.get("total_tokens") or 0) - (pure_graph_usage.get("total_tokens") or 0))
                        / (raw_usage.get("total_tokens") or 1),
                        4,
                    )
                    if raw_usage.get("total_tokens")
                    else None
                ),
            },
            "meta": {
                "gaia_file": str(gaia_file),
                "graph_file": str(graph_file),
                "requested_graph_source": graph_source,
                "resolved_graph_source": resolved_graph_source,
                "graph_source": resolved_graph_source,
                "requested_pruning_strategy": pruning_strategy if graph_source in {"pruning", "agenttrace_first"} else None,
                "pruning_strategy": resolved_pruning_strategy,
                "judge_review_enabled": judge_enable_review,
                "raw_input_mode": effective_raw_mode,
                "raw_input_char_limit": raw_input_char_limit,
                "graph_representation_modes": [GRAPH_REPRESENTATION_HYBRID, GRAPH_REPRESENTATION_PURE],
            },
        }
        with open(merged_out_dir / f"{trace_id}.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        row.update(
            {
                "raw_errors": len(raw_j.get("errors", [])),
                "graph_errors": len(graph_j.get("errors", [])),
                "pure_graph_errors": len(pure_graph_j.get("errors", [])),
                "raw_overall": (raw_j.get("scores") or [{}])[0].get("overall"),
                "graph_overall": (graph_j.get("scores") or [{}])[0].get("overall"),
                "pure_graph_overall": (pure_graph_j.get("scores") or [{}])[0].get("overall"),
                "raw_total_tokens": raw_usage.get("total_tokens"),
                "graph_total_tokens": graph_usage.get("total_tokens"),
                "pure_graph_total_tokens": pure_graph_usage.get("total_tokens"),
                "raw_input_mode": effective_raw_mode,
                "requested_graph_source": graph_source,
                "graph_source": resolved_graph_source,
                "pruning_strategy": resolved_pruning_strategy,
                "graph_file": str(graph_file),
                "raw_input_chars": raw_usage.get("input_chars"),
                "token_delta_raw_minus_graph": (
                    (raw_usage.get("total_tokens") or 0) - (graph_usage.get("total_tokens") or 0)
                ),
                "graph_token_saving_ratio": (
                    round(
                        ((raw_usage.get("total_tokens") or 0) - (graph_usage.get("total_tokens") or 0))
                        / (raw_usage.get("total_tokens") or 1),
                        4,
                    )
                    if raw_usage.get("total_tokens")
                    else None
                ),
                "token_delta_raw_minus_pure_graph": (
                    (raw_usage.get("total_tokens") or 0) - (pure_graph_usage.get("total_tokens") or 0)
                ),
                "pure_graph_token_saving_ratio": (
                    round(
                        ((raw_usage.get("total_tokens") or 0) - (pure_graph_usage.get("total_tokens") or 0))
                        / (raw_usage.get("total_tokens") or 1),
                        4,
                    )
                    if raw_usage.get("total_tokens")
                    else None
                ),
            }
        )
    except Exception as e:
        row["status"] = "failed"
        row["issues"].append(str(e))
        if isinstance(e, JudgeOutputParseError):
            debug_payload = {
                "trace_id": trace_id,
                "source_type": e.source_type,
                "error": str(e),
                "first_parse_error": e.first_error,
                "second_parse_error": e.second_error,
                "raw_llm_output": e.raw_text,
                "repaired_llm_output": e.repaired_text,
            }
            debug_file = debug_dir / f"{trace_id}_{e.source_type}_parse_error.json"
            with open(debug_file, "w", encoding="utf-8") as f:
                json.dump(debug_payload, f, ensure_ascii=False, indent=2)
            row["debug_file"] = str(debug_file)

    return {
        "trace_id": trace_id,
        "row": row,
        "raw_usage": raw_usage,
        "graph_usage": graph_usage,
        "pure_graph_usage": pure_graph_usage,
    }


def run(args):
    global DEEPSEEK_THINKING_MODE
    if OpenAI is None:
        raise RuntimeError("未安装 openai 包，请先安装：pip install openai")

    DEEPSEEK_THINKING_MODE = args.thinking_mode

    api_key = (
        args.api_key
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("SILICONFLOW_API_KEY")
        or DEEPSEEK_API_KEY
    )
    if not api_key:
        raise RuntimeError(
            "缺少 API Key。请通过 --api-key 或环境变量 SILICONFLOW_API_KEY / DEEPSEEK_API_KEY 提供。"
        )

    gaia_dir = Path(args.gaia_dir)
    graph_dir = Path(args.graph_dir)
    out_dir = Path(args.out_dir)
    raw_out_dir = out_dir / "raw_trace_judgements"
    graph_out_dir = out_dir / "graph_judgements"
    merged_out_dir = out_dir / "merged"
    debug_dir = out_dir / "debug_llm"
    export_markdown = bool(args.export_md or args.markdown_dir)
    markdown_out_dir = Path(args.markdown_dir) if args.markdown_dir else out_dir / "markdown_reports"
    for p in [raw_out_dir, graph_out_dir, merged_out_dir, debug_dir]:
        p.mkdir(parents=True, exist_ok=True)
    if export_markdown:
        markdown_out_dir.mkdir(parents=True, exist_ok=True)

    markdown_exported = 0

    def export_markdown_for_trace(trace_id):
        nonlocal markdown_exported
        output_path = markdown_out_dir / f"{trace_id}_diagnostic_report.md"
        export_merged_diagnostic_markdown(
            merged_out_dir / f"{trace_id}.json",
            output_path,
            judgement_source=args.markdown_source,
        )
        markdown_exported += 1
        return output_path

    files = sorted(gaia_dir.glob("*.json"))
    if args.trace_id:
        files = [p for p in files if p.stem == args.trace_id]
    if args.max_samples and args.max_samples > 0:
        files = files[: args.max_samples]

    skipped_existing = 0
    if args.resume:
        pure_graph_out_dir = graph_out_dir.parent / "pure_graph_judgements"
        pending_files = []
        for gaia_file in files:
            trace_id = gaia_file.stem
            merged_file = merged_out_dir / f"{trace_id}.json"
            raw_file = raw_out_dir / f"{trace_id}.json"
            graph_file = graph_out_dir / f"{trace_id}.json"
            pure_graph_file = pure_graph_out_dir / f"{trace_id}.json"
            if merged_file.exists() and raw_file.exists() and graph_file.exists() and pure_graph_file.exists():
                if export_markdown:
                    markdown_file = markdown_out_dir / f"{trace_id}_diagnostic_report.md"
                    if not markdown_file.exists():
                        export_markdown_for_trace(trace_id)
                    else:
                        markdown_exported += 1
                skipped_existing += 1
                continue
            pending_files.append(gaia_file)
        files = pending_files

    if not files:
        if args.resume and skipped_existing > 0:
            print(f"No pending traces: skipped {skipped_existing} completed items.")
            if export_markdown:
                print(f"Markdown reports: {markdown_out_dir} ({markdown_exported} available)")
            return
        raise RuntimeError("No GAIA trace files to process.")

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "thinking_mode": args.thinking_mode,
        "raw_input_mode": args.raw_input_mode,
        "requested_graph_source": args.graph_source,
        "graph_source": args.graph_source,
        "pruning_strategy": args.pruning_strategy if args.graph_source in {"pruning", "agenttrace_first"} else None,
        "judge_review_enabled": not args.disable_judge_review,
        "judge_max_tokens": args.judge_max_tokens,
        "judge_review_max_tokens": args.judge_review_max_tokens,
        "json_repair_max_tokens": args.json_repair_max_tokens,
        "review_first_pass_max_chars": args.review_first_pass_max_chars,
        "raw_full_trace_ids": sorted(args.raw_full_trace_ids),
        "raw_full_max_input_chars": args.raw_full_max_input_chars,
        "resume": args.resume,
        "skipped_existing": skipped_existing,
        "processed": 0,
        "failed": 0,
        "token_usage": {
            "raw_prompt_tokens": 0,
            "raw_completion_tokens": 0,
            "raw_total_tokens": 0,
            "graph_prompt_tokens": 0,
            "graph_completion_tokens": 0,
            "graph_total_tokens": 0,
            "pure_graph_prompt_tokens": 0,
            "pure_graph_completion_tokens": 0,
            "pure_graph_total_tokens": 0,
            "raw_total_input_chars": 0,
            "graph_total_input_chars": 0,
            "pure_graph_total_input_chars": 0,
        },
        "rows": [],
    }

    tasks = [
        (idx, gaia_file)
        for idx, gaia_file in enumerate(files, start=1)
    ]
    rows_by_index = {}

    if args.workers <= 1:
        for idx, gaia_file in tasks:
            result = process_trace_file(
                gaia_file,
                graph_dir,
                raw_out_dir,
                graph_out_dir,
                merged_out_dir,
                debug_dir,
                args.model,
                args.base_url,
                api_key,
                args.max_spans,
                args.max_input_chars,
                args.raw_input_mode,
                args.raw_full_trace_ids,
                args.raw_full_max_input_chars,
                args.graph_source,
                args.pruning_strategy,
                not args.disable_judge_review,
                args.judge_max_tokens,
                args.judge_review_max_tokens,
                args.json_repair_max_tokens,
                args.review_first_pass_max_chars,
                args.prompt_variant,
            )
            row = result["row"]
            rows_by_index[idx] = row
            if row["status"] == "ok":
                if export_markdown:
                    row["markdown_file"] = str(export_markdown_for_trace(result["trace_id"]))
                summary["processed"] += 1
                update_summary_token_usage(summary, result["raw_usage"], result["graph_usage"], result["pure_graph_usage"])
            else:
                summary["failed"] += 1
            print(f"[{idx}/{len(files)}] {result['trace_id']}: {row['status']}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_index = {
                executor.submit(
                    process_trace_file,
                    gaia_file,
                    graph_dir,
                    raw_out_dir,
                    graph_out_dir,
                    merged_out_dir,
                    debug_dir,
                    args.model,
                    args.base_url,
                    api_key,
                    args.max_spans,
                    args.max_input_chars,
                    args.raw_input_mode,
                    args.raw_full_trace_ids,
                    args.raw_full_max_input_chars,
                    args.graph_source,
                    args.pruning_strategy,
                    not args.disable_judge_review,
                    args.judge_max_tokens,
                    args.judge_review_max_tokens,
                    args.json_repair_max_tokens,
                    args.review_first_pass_max_chars,
                    args.prompt_variant,
                ): idx
                for idx, gaia_file in tasks
            }

            completed = 0
            for future in as_completed(future_to_index):
                completed += 1
                idx = future_to_index[future]
                result = future.result()
                row = result["row"]
                rows_by_index[idx] = row
                if row["status"] == "ok":
                    if export_markdown:
                        row["markdown_file"] = str(export_markdown_for_trace(result["trace_id"]))
                    summary["processed"] += 1
                    update_summary_token_usage(summary, result["raw_usage"], result["graph_usage"], result["pure_graph_usage"])
                else:
                    summary["failed"] += 1
                print(f"[{completed}/{len(files)}] {result['trace_id']}: {row['status']}")

    summary["rows"] = [rows_by_index[idx] for idx, _ in tasks]
    summary["markdown_exported"] = markdown_exported
    summary["markdown_dir"] = str(markdown_out_dir) if export_markdown else None

    with open(out_dir / "judge_summary.json", "w", encoding="utf-8") as f:
        raw_total = summary["token_usage"]["raw_total_tokens"]
        graph_total = summary["token_usage"]["graph_total_tokens"]
        pure_graph_total = summary["token_usage"]["pure_graph_total_tokens"]
        summary["token_usage"]["delta_total_tokens_raw_minus_graph"] = raw_total - graph_total
        summary["token_usage"]["graph_token_saving_ratio"] = (
            round((raw_total - graph_total) / raw_total, 4) if raw_total else None
        )
        summary["token_usage"]["delta_total_tokens_raw_minus_pure_graph"] = raw_total - pure_graph_total
        summary["token_usage"]["pure_graph_token_saving_ratio"] = (
            round((raw_total - pure_graph_total) / raw_total, 4) if raw_total else None
        )
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n完成：")
    print(f"- processed: {summary['processed']}")
    print(f"- failed: {summary['failed']}")
    print(f"- token(raw): {summary['token_usage']['raw_total_tokens']}")
    print(f"- token(graph): {summary['token_usage']['graph_total_tokens']}")
    print(f"- token(pure_graph): {summary['token_usage']['pure_graph_total_tokens']}")
    print(f"- token差值(raw-graph): {summary['token_usage']['delta_total_tokens_raw_minus_graph']}")
    print(f"- token差值(raw-pure_graph): {summary['token_usage']['delta_total_tokens_raw_minus_pure_graph']}")
    print(f"- summary: {out_dir / 'judge_summary.json'}")
    if export_markdown:
        print(f"- markdown: {markdown_out_dir} ({markdown_exported} reports)")


def build_cli():
    p = argparse.ArgumentParser(description="DeepSeek LLM-as-a-Judge for raw trace and trace graph root-cause analysis")
    p.add_argument("--gaia-dir", default="trail_data/GAIA", help="Directory of raw GAIA trace json files")
    p.add_argument("--graph-dir", default="output_graphs/trail_gaia_local_local_batch", help="目录：full graph 在 *_graph.json；pruning 模式下在 *_pruning/<strategy>.json")
    p.add_argument("--graph-source", choices=["full", "pruning", "agenttrace_first"], default="agenttrace_first", help="评判使用的图来源：full、pruning，或优先 agenttrace_suspicious_paths 并在缺失时回退到 full")
    p.add_argument("--pruning-strategy", default="weighted_k_shortest_paths", help="当 --graph-source=pruning 时使用的策略文件名；agenttrace_first 模式下会优先尝试 agenttrace_suspicious_paths")
    p.add_argument("--out-dir", default="output_graphs/deepseek_judge", help="Output directory")
    p.add_argument("--export-md", action="store_true", help="Export each diagnosis as a Markdown report")
    p.add_argument("--markdown-dir", default=None, help="Markdown output directory; also enables --export-md")
    p.add_argument(
        "--markdown-source",
        choices=["raw", "graph", "pure_graph"],
        default="graph",
        help="Judgement variant used in the Markdown report",
    )
    p.add_argument("--trace-id", default=None, help="Run one trace_id only")
    p.add_argument("--max-samples", type=int, default=0, help="Max samples to process, 0 means all")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Judge model name (OpenAI-compatible)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Judge API base URL")
    p.add_argument(
        "--thinking-mode",
        choices=["disabled", "enabled"],
        default="disabled",
        help="DeepSeek V4 thinking mode; disabled matches the former deepseek-chat setup",
    )
    p.add_argument("--api-key", default=None, help="Judge API key (or use DEEPSEEK_API_KEY / OPENAI_API_KEY env)")
    p.add_argument("--max-spans", type=int, default=80, help="Max timeline spans to include in raw digest")
    p.add_argument("--max-input-chars", type=int, default=28000, help="Max chars of digest sent to judge")
    p.add_argument("--raw-input-mode", choices=["digest", "full"], default="digest", help="How to send raw trace to judge: digest summary or full JSON")
    p.add_argument("--raw-full-trace-ids", default="", help="Comma-separated trace_ids that bypass digest and send full raw trace JSON")
    p.add_argument("--raw-full-max-input-chars", type=int, default=0, help="Max chars for full raw trace mode; 0 means no truncation")
    p.add_argument("--workers", type=int, default=4, help="Number of traces to process in parallel; set to 1 for serial")
    p.add_argument("--resume", action="store_true", help="Skip traces whose merged/raw/graph outputs already exist in out-dir")
    p.add_argument("--prompt-variant", choices=["default", "swe"], default="default", help="Judge prompt variant: default or swe (SWE-specific execution-span anchoring)")
    p.add_argument("--disable-judge-review", action="store_true", help="Disable the second-pass checklist review in judge")
    p.add_argument(
        "--judge-max-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_TOKENS,
        help="Max completion tokens for first-pass judge calls; 0 means do not send an explicit max_tokens cap",
    )
    p.add_argument(
        "--judge-review-max-tokens",
        type=int,
        default=DEFAULT_JUDGE_REVIEW_MAX_TOKENS,
        help="Max completion tokens for review judge calls; 0 means do not send an explicit max_tokens cap",
    )
    p.add_argument(
        "--json-repair-max-tokens",
        type=int,
        default=DEFAULT_JSON_REPAIR_MAX_TOKENS,
        help="Max completion tokens for JSON repair calls; 0 means do not send an explicit max_tokens cap",
    )
    p.add_argument(
        "--review-first-pass-max-chars",
        type=int,
        default=DEFAULT_REVIEW_FIRST_PASS_MAX_CHARS,
        help="Max chars of first-pass JSON included in the review prompt; 0 means no truncation",
    )
    return p


if __name__ == "__main__":
    args = build_cli().parse_args()
    args.raw_full_trace_ids = parse_trace_id_csv(args.raw_full_trace_ids)
    run(args)
