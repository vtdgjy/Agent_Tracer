from __future__ import annotations

"""TG-SPE: final-outcome-conditioned suspicious causal path extraction.

The method is not a generic context compressor. It extracts a small set of
typed causal paths that can propagate a suspicious upstream state or action to
the final answer/failure:

1. identify terminal diagnostic targets;
2. trace backwards through typed dependency edges;
3. rank suspicious origins and their target-reaching paths;
4. reject paths dominated by temporal/control-flow edges;
5. retain diverse, non-duplicate causal explanations.

Character and node limits are optional experiment controls and are disabled by
default; they do not participate in the causal-path score.
"""

from collections import Counter, deque
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx


TOOL_PATTERN = re.compile(r"(?:Tool|Function):\s*([A-Za-z0-9_.:/-]+)")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_\-/]{3,}")

TARGET_NODE_TYPES = {"Conclusion", "Error"}
PATH_BRIDGE_NODE_TYPES = {"Planning", "Action", "Fact", "Error", "Conclusion", "Intent"}
CONTEXT_IMPORTANT_TYPES = {"Planning", "Action", "Fact", "Error", "Conclusion"}

EDGE_CAUSAL_CONFIDENCE = {
    "answer_dependency": 1.0,
    "reasoning_dependency": 0.9,
    "tool_observation_dependency": 0.85,
    "data_dependency": 0.8,
    "Cognitive": 0.9,
    "Observation": 0.85,
    "Call": 0.75,
    "AgentSpawn": 0.92,
    "SubagentReturn": 0.95,
    "Retry": 0.7,
    "SemanticValidation": 0.98,
    "SemanticPropagation": 0.95,
    "SemanticOrigin": 0.98,
    "SemanticRejection": 0.92,
    "EvidenceExposure": 0.72,
    "control_flow": 0.35,
    "Temporal": 0.25,
    "same_step_neighbor": 0.2,
}

ROOT_CAUSE_SCORE_WEIGHTS = {
    "anomaly_score": 0.24,
    "final_answer_dependency": 0.20,
    "causal_path_strength": 0.16,
    "semantic_error_score": 0.22,
    "propagation_position_score": 0.09,
    "origin_preference_score": 0.09,
}

TYPE_BONUS = {
    "Error": 5.0,
    "Conclusion": 4.2,
    "Action": 3.3,
    "Planning": 3.0,
    "Fact": 1.8,
    "Intent": 1.4,
}

ERROR_KEYWORDS = {
    "error",
    "failed",
    "failure",
    "invalid",
    "timeout",
    "timed out",
    "exception",
    "forbidden",
    "unauthorized",
    "authentication",
    "permission",
    "service unavailable",
    "404",
    "403",
    "500",
    "insufficient",
    "not found",
}

TOOL_DIAGNOSTIC_KEYWORDS = {
    "tool",
    "function",
    "api",
    "call",
    "args",
    "params",
    "parameter",
    "format",
    "json",
    "schema",
    "browser",
    "search",
    "bash",
    "python",
    "final_answer",
}

PLANNING_DRIFT_KEYWORDS = {
    "need to",
    "should",
    "let me",
    "next",
    "then",
    "instead",
    "verify",
    "double-check",
    "rethink",
    "plan",
}

RETRY_KEYWORDS = {"retry", "again", "re-run", "rerun", "try another", "another attempt"}

COVERAGE_ANCHOR_FAILURE_PATTERN = re.compile(
    r"\b(?:error|fail(?:ed|ure)?|incorrect|invalid|exception|traceback|timeout|"
    r"unable|cannot|contradict(?:ion|ory)?|unsupported|missing|not\s+found|"
    r"no\s+evidence|hallucinat(?:e|ed|ion)|wrong|mismatch|unsuccessful)\b|"
    r"exit\s*code\s*[:=]?\s*[1-9]",
    re.IGNORECASE,
)

RCA_PRODUCER_KEYWORDS = {
    "write_text",
    "write_bytes",
    ".write(",
    ".write_",
    "create",
    "write",
    "store",
    "patch",
}

RCA_CONSUMER_KEYWORDS = {
    "read_text",
    ".read(",
    "json.loads",
    "pickle.load",
    "subprocess.run",
    "connect",
    "print(",
    "open(",
}

RCA_BAD_STATE_KEYWORDS = {
    "missing",
    "nonexistent",
    "stale",
    "invalid",
    "malformed",
    "unsafe",
    "wrong",
    "low",
    "closed",
    "unset",
    "empty",
    "bad",
    "impossible",
    "over-escaped",
    "fallback",
    "required",
}

RCA_DISTRACTOR_HINTS = {
    "readme.md",
    "pyproject.toml",
    "stream_utils.py",
    "chat.py",
    "get-content",
    "select-object",
    "list server",
    "repository",
}

RCA_CONTROL_ARTIFACT_HINTS = {
    "base",
    "command",
    "config",
    "date",
    "env",
    "key",
    "manifest",
    "model",
    "payload",
    "policy",
    "pointer",
    "port",
    "rate",
    "record",
    "regex",
    "route",
}

RCA_RECOVERY_HINTS = {
    "fallback",
    "patch",
    "repair",
    "retry",
    "recovery",
    "replace",
    "unsafe default",
    "now succeeds",
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "then",
    "have",
    "has",
    "had",
    "will",
    "need",
    "tool",
    "args",
    "params",
    "output",
    "input",
    "message",
    "step",
    "final",
    "answer",
}


def _text(node: dict) -> str:
    return str(node.get("content", "") or "")


def _lower_text(node: dict) -> str:
    return _text(node).lower()


def _node_type(node: dict) -> str:
    return str(node.get("type", "") or "")


def _node_chars(node: dict) -> int:
    return len(_text(node))


def _extract_tool_name(text: str) -> Optional[str]:
    match = TOOL_PATTERN.search(text or "")
    if not match:
        return None
    return match.group(1).strip().lower()


def _lexical_tokens(text: str) -> Set[str]:
    tokens = {t.lower() for t in TOKEN_PATTERN.findall(text or "")}
    return {t for t in tokens if t not in STOPWORDS}


def _node_order_key(nid: str) -> Tuple[int, str]:
    match = re.search(r"_(\d+)$", str(nid or ""))
    return (int(match.group(1)) if match else 10**9, str(nid or ""))


def _coverage_anchor_anomaly(node: Dict[str, Any]) -> float:
    """Gold-independent anomaly prior used only for temporal coverage anchors."""
    content = str(node.get("content") or "")
    cue_count = min(4, len(COVERAGE_ANCHOR_FAILURE_PATTERN.findall(content)))
    explicit_error = 1.0 if str(node.get("type") or "") == "Error" else 0.0
    tool_status = node.get("tool_result_status") or {}
    tool_error = 1.0 if isinstance(tool_status, dict) and tool_status.get("is_error") else 0.0
    try:
        saliency = float(node.get("saliency") or 0.0)
    except (TypeError, ValueError):
        saliency = 0.0
    normalized_saliency = max(0.0, min(1.0, saliency / 6.0))
    return min(
        1.0,
        0.35 * normalized_saliency
        + 0.35 * max(explicit_error, tool_error)
        + 0.30 * (cue_count / 4.0),
    )


def select_temporal_coverage_anchor_ids(
    nodes: Sequence[Dict[str, Any]],
    retained_ids: Iterable[str],
    budget: int,
    anomaly_weight: float = 0.65,
) -> List[str]:
    """Select one representative from uncovered dialogue steps.

    The greedy objective combines a label-free anomaly prior with normalized
    distance from steps already represented by causal paths.  It is a safety
    net for step-localization benchmarks, not an additional causal path.
    """
    budget = max(0, int(budget))
    if budget == 0:
        return []
    anomaly_weight = max(0.0, min(1.0, float(anomaly_weight)))
    retained_ids = {str(node_id) for node_id in retained_ids}
    representatives: Dict[int, Dict[str, Any]] = {}
    retained_steps: Set[int] = set()
    for node in nodes:
        step = node.get("source_step")
        if not isinstance(step, int):
            continue
        node_id = str(node.get("id") or "")
        if node_id in retained_ids:
            retained_steps.add(step)
        current = representatives.get(step)
        if current is None:
            representatives[step] = node
            continue
        current_key = (
            float(current.get("saliency") or 0.0),
            len(str(current.get("content") or "")),
            str(current.get("id") or ""),
        )
        node_key = (
            float(node.get("saliency") or 0.0),
            len(str(node.get("content") or "")),
            str(node.get("id") or ""),
        )
        if node_key > current_key:
            representatives[step] = node

    candidates = set(representatives) - retained_steps
    chosen_steps: List[int] = []
    selected_steps = set(retained_steps)
    maximum_step = max(representatives, default=0)
    for _ in range(min(budget, len(candidates))):
        def score(step: int) -> Tuple[float, float, int]:
            anomaly = _coverage_anchor_anomaly(representatives[step])
            if selected_steps:
                diversity = min(abs(step - prior) for prior in selected_steps) / max(1, maximum_step)
            else:
                diversity = 1.0
            combined = anomaly_weight * anomaly + (1.0 - anomaly_weight) * diversity
            return combined, anomaly, -step

        winner = max(candidates, key=score)
        chosen_steps.append(winner)
        selected_steps.add(winner)
        candidates.remove(winner)
    return [str(representatives[step].get("id")) for step in chosen_steps]


class AgentTraceSuspiciousPathPruner:
    """
    面向最终结果的 Agent trace 可疑因果路径提取。

    输出只由最终诊断目标和能够通过类型化依赖边到达该目标的候选路径组成。
    时间相邻但缺少依赖证据的路径会被过滤，高度重叠的解释链会被去重。
    """

    strategy_name = "agenttrace_suspicious_paths"

    def __init__(
        self,
        max_suspicious_sources: int = 12,
        max_nodes: int = 0,
        char_budget: int = 0,
        local_hops: int = 1,
        max_paths_per_target: int = 4,
        max_candidate_chains: int = 5,
        min_dependency_confidence: float = 0.65,
        max_targets: int = 4,
        min_final_dependency: float = 0.55,
        min_path_confidence: float = 0.45,
        max_path_hops: int = 18,
        max_path_overlap: float = 0.85,
        include_path_context: bool = False,
        coverage_anchor_budget: int = 0,
        coverage_anchor_anomaly_weight: float = 0.65,
    ):
        self.max_suspicious_sources = max_suspicious_sources
        self.max_nodes = max(0, max_nodes)
        self.char_budget = max(0, char_budget)
        self.local_hops = local_hops
        self.max_paths_per_target = max_paths_per_target
        self.max_candidate_chains = max_candidate_chains
        self.min_dependency_confidence = min_dependency_confidence
        self.max_targets = max(1, max_targets)
        self.min_final_dependency = max(0.0, min(1.0, min_final_dependency))
        self.min_path_confidence = max(0.0, min(1.0, min_path_confidence))
        self.max_path_hops = max(2, max_path_hops)
        self.max_path_overlap = max(0.0, min(1.0, max_path_overlap))
        self.include_path_context = include_path_context
        self.coverage_anchor_budget = max(0, int(coverage_anchor_budget))
        self.coverage_anchor_anomaly_weight = max(
            0.0, min(1.0, float(coverage_anchor_anomaly_weight))
        )

    def select(self, trace: Any) -> Tuple[Set[str], Dict[str, Any]]:
        if not getattr(trace, "graph", None):
            return set(), {
                "strategy_rationale": "empty_graph",
                "diagnosis_targets": [],
                "candidate_pool_size": 0,
                "suspicious_sources": [],
                "selected_paths": [],
                "selected_edge_pairs": [],
                "root_cause_ranking": [],
                "candidate_chains": [],
                "chain_schema_version": "tg_spe_causal_path_v3_final_outcome_conditioned",
            }

        targets = self._select_targets(trace)
        failure_slice_ids, failure_slice_meta = self._backward_failure_slice(trace, targets)
        fallback_candidate_ids = self._collect_candidate_ids(trace, targets)
        repeated_tools = self._tool_frequency(trace, fallback_candidate_ids)
        fallback_scores = {
            nid: self._score_node(trace, nid, targets, repeated_tools)
            for nid in fallback_candidate_ids
        }
        rescue_ids = self._select_rescue_candidates(
            trace,
            fallback_candidate_ids - failure_slice_ids,
            fallback_scores,
        )
        candidate_ids = failure_slice_ids | rescue_ids
        if not candidate_ids:
            candidate_ids = fallback_candidate_ids
        scores = {
            nid: self._score_node(trace, nid, targets, repeated_tools)
            for nid in candidate_ids
        }

        self._ensure_causal_edge_costs(trace)
        root_cause_ranking = self._rank_root_cause_candidates(trace, candidate_ids, targets, scores)
        selected_chain_candidates = self._select_diverse_causal_chains(root_cause_ranking)
        rca_candidate_table = self._build_rca_candidate_table(trace, targets, root_cause_ranking, scores)
        suspicious_sources = [
            row["root_cause_node"]
            for row in selected_chain_candidates[: self.max_suspicious_sources]
            if row.get("root_cause_node")
        ]
        if not suspicious_sources:
            suspicious_sources = self._select_suspicious_sources(trace, candidate_ids, scores)
        kept: Set[str] = set(targets)
        protected: Set[str] = set(targets)
        selected_paths: List[dict] = []
        skipped_paths_for_limits = 0

        if self.include_path_context:
            for nid in list(targets):
                context = self._collect_local_context(trace, nid, hops=self.local_hops)
                kept.update(context)
            kept = self._trim(trace, kept, protected, scores)

        per_target_counts = Counter()
        accepted_chains: List[dict] = []
        for chain in selected_chain_candidates:
            src = chain.get("root_cause_node")
            dst = chain.get("final_failure")
            path = chain.get("causal_path")
            if not src or not dst or not isinstance(path, list):
                continue
            if per_target_counts[dst] >= self.max_paths_per_target:
                continue
            if not path or len(path) < 2:
                continue

            path_nodes = set(path)
            required = protected | path_nodes
            if not self._fits_limits(trace, required):
                skipped_paths_for_limits += 1
                continue

            path_context = set(path_nodes)
            if self.include_path_context:
                for pid in path:
                    path_context.update(self._collect_chain_context(trace, pid))

            tentative_protected = protected | path_nodes
            tentative = self._trim(
                trace,
                kept | path_context,
                tentative_protected,
                scores,
            )
            if (
                not path_nodes.issubset(tentative)
                or not self._fits_limits(trace, tentative)
            ):
                skipped_paths_for_limits += 1
                continue

            kept = tentative
            protected = tentative_protected
            per_target_counts[dst] += 1
            accepted_chains.append(chain)
            selected_paths.append(
                {
                    "source": src,
                    "target": dst,
                    "score": round(chain.get("root_cause_score", scores.get(src, 0.0)), 4),
                    "path": path,
                    "path_len": len(path),
                    "root_cause_type": chain.get("root_cause_type"),
                    "score_components": chain.get("score_components", {}),
                    "dependency_metrics": chain.get("dependency_metrics", {}),
                    "diversity": chain.get("diversity", {}),
                }
            )

        coverage_anchor_ids = select_temporal_coverage_anchor_ids(
            trace.nodes,
            kept,
            self.coverage_anchor_budget,
            self.coverage_anchor_anomaly_weight,
        )
        kept.update(coverage_anchor_ids)
        kept_node_count_before_trim = len(kept)
        kept = self._trim(trace, kept, protected, scores)
        retained_coverage_anchor_ids = [node_id for node_id in coverage_anchor_ids if node_id in kept]
        used_chars = self._estimated_chars(trace, kept)
        # Path diversity and root-cause ranking serve different purposes.  The
        # diverse paths determine graph coverage, while the ranked candidates
        # should still consider every retained node that can causally reach a
        # target.  Reusing only the diverse paths as the ranking suppresses
        # plausible upstream origins that appear inside another selected path.
        retained_root_cause_ranking = [
            row
            for row in root_cause_ranking
            if row.get("root_cause_node") in kept
            and set(row.get("causal_path") or []).issubset(kept)
        ][:15]
        if not retained_root_cause_ranking:
            retained_root_cause_ranking = accepted_chains[:15]
        candidate_chains = retained_root_cause_ranking[: self.max_candidate_chains]
        accepted_root_ids = {
            row.get("root_cause_node") for row in accepted_chains if row.get("root_cause_node")
        }
        retained_rca_candidate_table = [
            row
            for row in rca_candidate_table
            if row.get("node_id") in accepted_root_ids
            and (not row.get("consumer_node") or row.get("consumer_node") in kept)
            and (
                not row.get("symptom_or_error_node")
                or row.get("symptom_or_error_node") in kept
            )
        ]
        selected_edge_pairs = sorted(
            {
                (source, target)
                for chain in accepted_chains
                for source, target in zip(
                    chain.get("causal_path", []),
                    chain.get("causal_path", [])[1:],
                )
            }
        )

        meta = {
            "strategy_rationale": "final-outcome-conditioned suspicious causal path extraction",
            "source_mapping": {
                "query_specific_pruning": "diagnosis_targets + target-conditioned extraction",
                "snap": "candidate pool restricted mostly to ancestors of targets",
                "arrow": "coarse suspicious scoring before path-level extraction",
                "microservice_rca_transfer": [
                    "backward failure slicing from final answer / error targets",
                    "typed-edge causal confidence for dependency-aware pruning",
                    "graph-ranked candidate propagation chains before LLM verification",
                ],
                "agenttrace_adaptation": [
                    "extract only paths that reach a final diagnostic target",
                    "filter paths by typed dependency strength and causal confidence",
                    "deduplicate highly overlapping paths while preserving distinct causes",
                ],
            },
            "diagnosis_targets": sorted(targets),
            "target_selection_policy": "final-answer/terminal-conclusion first; otherwise recent terminal errors",
            "max_targets": self.max_targets,
            "failure_slice": failure_slice_meta,
            "candidate_pool_size": len(candidate_ids),
            "fallback_candidate_pool_size": len(fallback_candidate_ids),
            "rescued_high_anomaly_nodes": sorted(rescue_ids),
            "suspicious_sources": [
                {"node_id": nid, "score": round(scores.get(nid, 0.0), 4)}
                for nid in suspicious_sources
            ],
            "selected_paths": selected_paths,
            "selected_edge_pairs": [
                {"source": source, "target": target}
                for source, target in selected_edge_pairs
            ],
            "skipped_paths_for_limits": skipped_paths_for_limits,
            "path_selection_policy": "final-target reachability + dependency threshold + overlap-aware diversity",
            "root_ranking_policy": "rank every retained final-reaching origin independently of path diversity",
            "root_cause_score_weights": ROOT_CAUSE_SCORE_WEIGHTS,
            "rca_candidate_table": retained_rca_candidate_table[:12],
            "root_cause_ranking": retained_root_cause_ranking[:15],
            "candidate_chains": candidate_chains,
            "metadata_filtered_to_retained_nodes": True,
            "chain_schema_version": "tg_spe_causal_path_v3_final_outcome_conditioned",
            "protected_node_count": len(protected),
            "kept_node_count_before_trim": kept_node_count_before_trim,
            "used_chars": used_chars,
            "limits_enabled": self._limits_enabled(),
            "limits_satisfied": self._fits_limits(trace, kept),
            "budget_satisfied": self._fits_limits(trace, kept),
            "char_budget": self.char_budget or None,
            "max_nodes": self.max_nodes or None,
            "max_candidate_chains": self.max_candidate_chains,
            "min_dependency_confidence": self.min_dependency_confidence,
            "min_final_dependency": self.min_final_dependency,
            "min_path_confidence": self.min_path_confidence,
            "max_path_hops": self.max_path_hops,
            "max_path_overlap": self.max_path_overlap,
            "include_path_context": self.include_path_context,
            "coverage_anchors": {
                "policy": "0.65 anomaly + 0.35 normalized temporal diversity"
                if self.coverage_anchor_anomaly_weight == 0.65
                else "weighted anomaly + normalized temporal diversity",
                "requested_budget": self.coverage_anchor_budget,
                "anomaly_weight": self.coverage_anchor_anomaly_weight,
                "retained_node_ids": retained_coverage_anchor_ids,
                "retained_count": len(retained_coverage_anchor_ids),
                "participates_in_causal_path_score": False,
                "annotation_used": False,
            },
        }
        return kept, meta

    def _select_targets(self, trace: Any) -> Set[str]:
        conclusions: List[str] = []
        final_actions: List[str] = []
        errors: List[str] = []
        order: Dict[str, int] = {}
        for index, node in enumerate(getattr(trace, "nodes", [])):
            nid = str(node.get("id", "") or "")
            if not nid:
                continue
            order[nid] = index
            ntype = _node_type(node)
            text = _lower_text(node)
            if ntype == "Conclusion":
                conclusions.append(nid)
            elif ntype == "Error":
                errors.append(nid)
            elif ntype == "Action" and "final_answer" in text:
                final_actions.append(nid)

        graph = getattr(trace, "graph", None)

        def is_terminal(nid: str, peers: Set[str]) -> bool:
            if graph is None or nid not in graph:
                return True
            return not any(other != nid and nx.has_path(graph, nid, other) for other in peers)

        targets: List[str] = []
        conclusion_peers = set(conclusions) | set(final_actions)
        if conclusion_peers:
            terminal_conclusions = [nid for nid in conclusions if is_terminal(nid, conclusion_peers)]
            terminal_final_actions = [nid for nid in final_actions if is_terminal(nid, conclusion_peers)]
            if terminal_conclusions:
                explicit_final_conclusions = [
                    nid
                    for nid in terminal_conclusions
                    if any(
                        cue in _lower_text(trace.node_by_id.get(nid, {}))
                        for cue in ("final answer", "final response", "final_answer")
                    )
                ]
                ranked = sorted(
                    explicit_final_conclusions or terminal_conclusions,
                    key=lambda nid: order.get(nid, -1),
                    reverse=True,
                )
            elif terminal_final_actions:
                ranked = sorted(terminal_final_actions, key=lambda nid: order.get(nid, -1), reverse=True)
            else:
                ranked = sorted(conclusion_peers, key=lambda nid: order.get(nid, -1), reverse=True)[:1]
            for nid in ranked:
                if nid not in targets:
                    targets.append(nid)
                if len(targets) >= self.max_targets:
                    break
        elif errors:
            error_peers = set(errors)
            terminal_errors = [nid for nid in errors if is_terminal(nid, error_peers)]
            ranked = sorted(
                terminal_errors or errors[-1:],
                key=lambda nid: order.get(nid, -1),
                reverse=True,
            )
            for nid in ranked:
                if nid not in targets:
                    targets.append(nid)
                if len(targets) >= self.max_targets:
                    break

        if not targets and getattr(trace, "nodes", None):
            fallback = str(trace.nodes[-1].get("id", "") or "")
            if fallback:
                targets.append(fallback)
        return set(targets)

    def _collect_candidate_ids(self, trace: Any, targets: Set[str]) -> Set[str]:
        candidate_ids: Set[str] = set(targets)
        graph = trace.graph

        for nid in targets:
            if nid not in graph:
                continue
            candidate_ids.update(nx.ancestors(graph, nid))
            candidate_ids.update(self._collect_local_context(trace, nid, hops=1))

        if not candidate_ids:
            candidate_ids = {
                str(node.get("id", "") or "")
                for node in getattr(trace, "nodes", [])
                if node.get("id")
            }

        return {nid for nid in candidate_ids if nid in graph}

    def _backward_failure_slice(self, trace: Any, targets: Set[str]) -> Tuple[Set[str], Dict[str, Any]]:
        graph = trace.graph
        kept: Set[str] = {nid for nid in targets if nid in graph}
        q = deque((nid, 0) for nid in kept)
        best_depth = {nid: 0 for nid in kept}
        kept_by_reason: Counter = Counter()
        dropped_low_dependency_edges = 0
        inspected_edges = 0

        while q:
            cur, depth = q.popleft()
            for pred in graph.predecessors(cur):
                inspected_edges += 1
                edge = self._edge_payload(trace, pred, cur)
                confidence = self._edge_causal_confidence(edge)
                reason = None

                if confidence >= self.min_dependency_confidence:
                    reason = "typed_dependency"
                elif depth < self.local_hops:
                    reason = "target_local_context"
                elif self._is_failure_neighborhood(trace, pred):
                    reason = "failure_neighborhood"
                elif self._is_retry_node(trace, pred):
                    reason = "retry_context"
                elif self._is_planning_action_bridge(trace, pred) and depth <= 4:
                    reason = "planning_action_bridge"

                if reason is None:
                    dropped_low_dependency_edges += 1
                    continue

                if pred not in kept:
                    kept.add(pred)
                    kept_by_reason[reason] += 1

                should_expand = (
                    confidence >= self.min_dependency_confidence
                    or reason in {"failure_neighborhood", "retry_context", "planning_action_bridge"}
                )
                next_depth = depth + 1
                if should_expand and next_depth < best_depth.get(pred, 10**9):
                    best_depth[pred] = next_depth
                    q.append((pred, next_depth))

        if not kept and getattr(trace, "nodes", None):
            fallback = str(trace.nodes[-1].get("id", "") or "")
            if fallback:
                kept.add(fallback)

        meta = {
            "strategy": "backward_failure_slicing",
            "target_count": len(targets),
            "targets": sorted(targets),
            "sliced_node_count": len(kept),
            "inspected_backward_edges": inspected_edges,
            "dropped_low_dependency_edges": dropped_low_dependency_edges,
            "kept_by_reason": dict(kept_by_reason),
            "edge_confidence_policy": EDGE_CAUSAL_CONFIDENCE,
            "min_dependency_confidence": self.min_dependency_confidence,
        }
        return kept, meta

    def _select_rescue_candidates(
        self,
        trace: Any,
        node_ids: Set[str],
        scores: Dict[str, float],
        max_rescues: int = 6,
    ) -> Set[str]:
        ranked = sorted(
            node_ids,
            key=lambda nid: (
                scores.get(nid, 0.0),
                self._is_failure_neighborhood(trace, nid),
                self._is_retry_node(trace, nid),
                -_node_chars(trace.node_by_id.get(nid, {})),
            ),
            reverse=True,
        )
        rescued: Set[str] = set()
        for nid in ranked:
            if len(rescued) >= max_rescues:
                break
            score = scores.get(nid, 0.0)
            if score < 5.0 and not self._is_failure_neighborhood(trace, nid):
                break
            rescued.add(nid)
        return rescued

    def _edge_payload(self, trace: Any, src: str, dst: str) -> dict:
        try:
            payload = trace.graph[src][dst]
        except Exception:
            return {}
        return dict(payload or {})

    def _edge_causal_confidence(self, edge: dict) -> float:
        edge_type = str(edge.get("edge_type") or edge.get("type") or "")
        base = EDGE_CAUSAL_CONFIDENCE.get(edge_type, 0.5)
        if edge_type in {"Temporal", "control_flow", "same_step_neighbor"}:
            return base
        try:
            raw_score = float(edge.get("score", 0.0) or 0.0)
        except Exception:
            raw_score = 0.0
        if raw_score <= 0:
            return base
        return max(0.05, min(1.0, 0.75 * base + 0.25 * max(0.0, min(1.0, raw_score))))

    def _edge_dependency_class(self, trace: Any, src: str, dst: str, edge: dict) -> str:
        edge_type = str(edge.get("edge_type") or edge.get("type") or "")
        if edge_type in {
            "answer_dependency",
            "reasoning_dependency",
            "tool_observation_dependency",
            "data_dependency",
            "validation_dependency",
            "control_flow",
            "same_step_neighbor",
        }:
            return edge_type
        if edge_type == "Cognitive":
            dst_type = _node_type(trace.node_by_id.get(dst, {}))
            return "answer_dependency" if dst_type == "Conclusion" else "reasoning_dependency"
        if edge_type == "Observation":
            return "tool_observation_dependency"
        if edge_type == "Call":
            return "data_dependency"
        if edge_type == "AgentSpawn":
            return "data_dependency"
        if edge_type == "SubagentReturn":
            return "reasoning_dependency"
        if edge_type == "Retry":
            return "reasoning_dependency"
        if edge_type == "SemanticOrigin":
            return "data_dependency"
        if edge_type == "SemanticValidation":
            return "validation_dependency"
        if edge_type == "SemanticPropagation":
            dst_type = _node_type(trace.node_by_id.get(dst, {}))
            return "answer_dependency" if dst_type == "Conclusion" else "reasoning_dependency"
        if edge_type in {"SemanticRejection", "EvidenceExposure"}:
            return "control_flow"
        if edge_type == "Temporal":
            return "same_step_neighbor"
        return "control_flow"

    def _causal_edge_cost(self, trace: Any, src: str, dst: str, edge: dict) -> float:
        dependency_class = self._edge_dependency_class(trace, src, dst, edge)
        dependency_cost = {
            "answer_dependency": 0.25,
            "validation_dependency": 0.3,
            "reasoning_dependency": 0.45,
            "tool_observation_dependency": 0.55,
            "data_dependency": 0.6,
            "control_flow": 1.35,
            "same_step_neighbor": 1.8,
        }.get(dependency_class, 1.2)
        confidence = self._edge_causal_confidence(edge)
        return round(max(0.05, dependency_cost + 1.25 * (1.0 - confidence)), 6)

    def _ensure_causal_edge_costs(self, trace: Any) -> None:
        for src, dst, edge in trace.graph.edges(data=True):
            edge["tg_spe_causal_cost"] = self._causal_edge_cost(trace, src, dst, edge)

    def _best_path_to_target(
        self,
        trace: Any,
        src: str,
        targets: Set[str],
    ) -> Tuple[Optional[List[str]], Optional[str], Dict[str, float]]:
        if src not in trace.graph:
            return None, None, {}
        if src in targets:
            return [src], src, self._path_dependency_components(trace, [src], src)

        best_path = None
        best_target = None
        best_components: Dict[str, float] = {}
        best_quality = (-1.0, 0.0, 0.0, 0.0, 0)
        for dst in targets:
            if dst not in trace.graph or not nx.has_path(trace.graph, src, dst):
                continue
            try:
                path_generator = nx.shortest_simple_paths(
                    trace.graph,
                    src,
                    dst,
                    weight="tg_spe_causal_cost",
                )
                inspected = 0
                for path in path_generator:
                    inspected += 1
                    if inspected > 8:
                        break
                    if len(path) - 1 > self.max_path_hops:
                        continue
                    components = self._path_dependency_components(trace, path, dst)
                    if components["final_answer_dependency"] < self.min_final_dependency:
                        continue
                    if components["causal_path_strength"] < self.min_path_confidence:
                        continue
                    quality = (
                        components["final_answer_dependency"],
                        components["causal_path_strength"],
                        components["dependency_edge_ratio"],
                        -components["temporal_edge_ratio"],
                        -len(path),
                    )
                    if quality > best_quality:
                        best_quality = quality
                        best_path = path
                        best_target = dst
                        best_components = components
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        return best_path, best_target, best_components

    def _path_edge_chain(self, trace: Any, path: Sequence[str]) -> List[dict]:
        rows: List[dict] = []
        for src, dst in zip(path, path[1:]):
            edge = self._edge_payload(trace, src, dst)
            rows.append(
                {
                    "source": src,
                    "target": dst,
                    "edge_type": str(edge.get("edge_type") or edge.get("type") or "unknown"),
                    "dependency_class": self._edge_dependency_class(trace, src, dst, edge),
                    "causal_confidence": round(self._edge_causal_confidence(edge), 4),
                    "score": edge.get("score"),
                    "method": edge.get("method"),
                }
            )
        return rows

    def _path_dependency_components(self, trace: Any, path: Sequence[str], target: Optional[str]) -> Dict[str, float]:
        edge_chain = self._path_edge_chain(trace, path)
        if not edge_chain:
            return {
                "final_answer_dependency": 0.1 if target in path else 0.0,
                "causal_path_strength": 0.0,
                "dependency_edge_ratio": 0.0,
                "temporal_edge_ratio": 0.0,
                "minimum_edge_confidence": 0.0,
                "path_hops": 0.0,
            }

        confidences = [float(edge.get("causal_confidence") or 0.0) for edge in edge_chain]
        dependency_edges = [
            edge
            for edge in edge_chain
            if edge.get("dependency_class") not in {"same_step_neighbor", "control_flow"}
        ]
        dependency_confidences = [
            float(edge.get("causal_confidence") or 0.0) for edge in dependency_edges
        ]
        avg_confidence = sum(confidences) / len(confidences)
        avg_dependency_confidence = (
            sum(dependency_confidences) / len(dependency_confidences)
            if dependency_confidences
            else 0.0
        )
        min_dependency_confidence = min(dependency_confidences) if dependency_confidences else 0.0
        dependency_ratio = len(dependency_edges) / len(edge_chain)
        temporal_ratio = sum(
            edge.get("dependency_class") == "same_step_neighbor" for edge in edge_chain
        ) / len(edge_chain)
        target_bonus = 0.1 if target and _node_type(trace.node_by_id.get(target, {})) == "Conclusion" else 0.0
        final_dependency = (
            0.5 * dependency_ratio
            + 0.35 * avg_dependency_confidence
            + target_bonus
            - 0.2 * temporal_ratio
        )
        path_strength = (
            0.65 * avg_dependency_confidence
            + 0.35 * min_dependency_confidence
            if dependency_confidences
            else 0.35 * avg_confidence
        )
        return {
            "final_answer_dependency": round(max(0.0, min(1.0, final_dependency)), 4),
            "causal_path_strength": round(max(0.0, min(1.0, path_strength)), 4),
            "dependency_edge_ratio": round(dependency_ratio, 4),
            "temporal_edge_ratio": round(temporal_ratio, 4),
            "minimum_edge_confidence": round(min_dependency_confidence, 4),
            "path_hops": float(len(edge_chain)),
        }

    def _semantic_error_score(self, trace: Any, nid: str, targets: Set[str]) -> float:
        node = trace.node_by_id.get(nid, {})
        explicit_score = float(node.get("semantic_error_score") or 0.0)
        violations = node.get("semantic_violations") or []
        if node.get("semantic_violation"):
            violations = list(violations) + [node["semantic_violation"]]
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            severity_weight = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.25}.get(
                str(violation.get("severity") or "").upper(),
                0.5,
            )
            reaches_final = bool(violation.get("reaches_final_answer", violation.get("used_by_final")))
            reach_probability = float(violation.get("final_reach_probability") or 0.0)
            usage_weight = max(0.65, reach_probability) if reaches_final else 0.35
            explicit_score = max(
                explicit_score,
                float(violation.get("confidence") or 0.0) * severity_weight * usage_weight,
            )
        text = _lower_text(node)
        keyword_hits = sum(
            1
            for kw in [
                "error",
                "failed",
                "invalid",
                "wrong",
                "incorrect",
                "unsupported",
                "hallucinat",
                "irrelevant",
                "not found",
                "missing",
                "no evidence",
                "contradict",
                "timeout",
                "exception",
            ]
            if kw in text
        )
        keyword_score = min(1.0, keyword_hits * 0.2)

        node_tokens = _lexical_tokens(_text(node))
        overlap_score = 0.0
        for target in targets:
            target_tokens = _lexical_tokens(_text(trace.node_by_id.get(target, {})))
            if not target_tokens:
                continue
            overlap = len(node_tokens & target_tokens) / max(1, min(len(node_tokens), len(target_tokens)))
            overlap_score = max(overlap_score, min(1.0, overlap))
        lexical_score = min(1.0, keyword_score + 0.35 * overlap_score)
        return round(min(1.0, max(explicit_score, lexical_score)), 4)

    def _anomaly_component(self, trace: Any, nid: str, score: float) -> float:
        node = trace.node_by_id.get(nid, {})
        ntype = _node_type(node)
        value = max(0.0, min(1.0, (score - 3.0) / 6.0))
        if ntype == "Error":
            value = max(value, 0.85)
        if self._is_failure_neighborhood(trace, nid):
            value = max(value, 0.7)
        if self._is_retry_node(trace, nid):
            value = max(value, 0.65)
        if ntype == "Conclusion":
            value = min(value, 0.35)
        return round(value, 4)

    def _propagation_position_score(self, trace: Any, nid: str, path: Sequence[str]) -> float:
        if not path or nid == path[-1]:
            return 0.05
        distance = max(1, len(path) - 1)
        value = 0.2 + min(0.7, distance * 0.12)
        if _node_type(trace.node_by_id.get(nid, {})) in {"Error", "Fact", "Action", "Planning"}:
            value += 0.1
        return round(min(1.0, value), 4)

    def _origin_preference_score(self, trace: Any, nid: str, path: Sequence[str]) -> float:
        if not path or nid != path[0]:
            return 0.0

        ntype = _node_type(trace.node_by_id.get(nid, {}))
        if ntype == "Conclusion":
            return 0.0

        distance = max(0, len(path) - 1)
        value = 0.25 + min(0.45, distance * 0.08)

        if ntype in {"Error", "Fact", "Action"}:
            value += 0.18
        elif ntype == "Planning":
            value += 0.10

        if len(path) >= 2:
            first_edge = self._edge_payload(trace, path[0], path[1])
            dep_class = self._edge_dependency_class(trace, path[0], path[1], first_edge)
            if dep_class not in {"same_step_neighbor", "control_flow"}:
                value += 0.14

        if self._is_failure_neighborhood(trace, nid) or self._semantic_error_score(trace, nid, {path[-1]}) >= 0.3:
            value += 0.10

        # A direct predecessor of the final failure is often a usage/symptom node,
        # not the earliest human-annotated origin.
        if distance <= 1 and ntype not in {"Error", "Action"}:
            value -= 0.15

        return round(max(0.0, min(1.0, value)), 4)

    def _select_fault_usage(self, trace: Any, path: Sequence[str]) -> str:
        if len(path) <= 1:
            return path[0] if path else "unknown"
        target = path[-1]
        for nid in path[1:-1]:
            if _node_type(trace.node_by_id.get(nid, {})) in {"Planning", "Action", "Conclusion"}:
                return nid
        return path[1] if len(path) > 1 else target

    def _root_cause_type(self, trace: Any, nid: str) -> str:
        node = trace.node_by_id.get(nid, {})
        ntype = _node_type(node)
        text = _lower_text(node)
        semantic_types = {str(node.get("semantic_violation_type") or "")}
        semantic_types.update(
            str(item.get("violation_type") or "")
            for item in (node.get("semantic_violations") or [])
            if isinstance(item, dict)
        )
        if "entity_mismatch" in semantic_types:
            return "entity_resolution_mismatch"
        if "ungrounded_tool_argument" in semantic_types:
            return "unverified_tool_argument"
        if "argument_result_mismatch" in semantic_types:
            return "tool_contract_mismatch"
        if "temporal_mismatch" in semantic_types:
            return "temporal_constraint_mismatch"
        if any(kw in text for kw in ["format", "schema", "invalid", "missing required tag", "kwargs"]):
            return "format_or_tool_argument_error"
        if self._is_retry_node(trace, nid):
            return "retry_or_recovery_failure"
        if ntype == "Error" or any(kw in text for kw in ERROR_KEYWORDS):
            return "tool_or_execution_failure"
        if ntype == "Fact" and any(kw in text for kw in ["search", "result", "evidence", "retrieved", "google", "web"]):
            return "wrong_or_low_quality_evidence_used"
        if ntype == "Planning":
            return "unsupported_or_invalid_reasoning"
        if ntype == "Action":
            return "wrong_action_or_tool_choice"
        if ntype == "Conclusion":
            return "final_answer_symptom"
        return "agent_trace_failure"

    def _brief_node(self, trace: Any, nid: str, max_chars: int = 180) -> str:
        node = trace.node_by_id.get(nid, {})
        return (_text(node) or str(node.get("display", "") or ""))[:max_chars]

    def _extract_artifact_keys(self, text: str) -> Set[str]:
        keys: Set[str] = set()
        for match in re.findall(r"cx_\d{3}_[A-Za-z0-9_.-]+(?:[/\\][A-Za-z0-9_.-]+)?", text or ""):
            keys.add(match.replace("\\", "/"))
        for match in re.findall(r"[A-Za-z0-9_.-]+\.(?:json|txt|pkl|md|csv|yaml|yml)", text or ""):
            if match.lower() not in {"readme.md", "pyproject.toml"}:
                keys.add(match.replace("\\", "/"))
        return keys

    def _is_rca_producer_action(self, trace: Any, nid: str) -> bool:
        node = trace.node_by_id.get(nid, {})
        text = _lower_text(node)
        if _node_type(node) != "Action":
            return False
        return any(kw in text for kw in RCA_PRODUCER_KEYWORDS) and bool(self._extract_artifact_keys(_text(node)))

    def _is_rca_consumer_node(self, trace: Any, nid: str) -> bool:
        node = trace.node_by_id.get(nid, {})
        text = _lower_text(node)
        ntype = _node_type(node)
        if ntype == "Error":
            return True
        if ntype not in {"Action", "Fact", "Conclusion"}:
            return False
        return any(kw in text for kw in RCA_CONSUMER_KEYWORDS) or any(kw in text for kw in ERROR_KEYWORDS)

    def _next_error_after(self, trace: Any, nid: str) -> str:
        start_order = _node_order_key(nid)
        for node in sorted(getattr(trace, "nodes", []), key=lambda n: _node_order_key(str(n.get("id", "")))):
            other = str(node.get("id", "") or "")
            if _node_order_key(other) <= start_order:
                continue
            if _node_type(node) == "Error":
                return other
        return ""

    def _first_reachable_target(self, trace: Any, nid: str, targets: Set[str]) -> str:
        reachable = self._reachable_targets(trace, nid, targets)
        return reachable[0] if reachable else ""

    def _consumer_for_producer(self, trace: Any, producer_id: str, artifacts: Set[str]) -> Tuple[str, str]:
        producer_order = _node_order_key(producer_id)
        best_consumer = ""
        best_error = ""
        for node in sorted(getattr(trace, "nodes", []), key=lambda n: _node_order_key(str(n.get("id", "")))):
            nid = str(node.get("id", "") or "")
            if not nid or _node_order_key(nid) <= producer_order:
                continue
            node_artifacts = self._extract_artifact_keys(_text(node))
            if not artifacts & node_artifacts:
                continue
            if not self._is_rca_consumer_node(trace, nid):
                continue
            if not best_consumer:
                best_consumer = nid
            if _node_type(node) == "Error":
                best_error = nid
                break
            next_error = self._next_error_after(trace, nid)
            if next_error:
                best_error = next_error
                break
        return best_consumer, best_error

    def _build_rca_candidate_table(
        self,
        trace: Any,
        targets: Set[str],
        root_cause_ranking: List[dict],
        scores: Dict[str, float],
    ) -> List[dict]:
        ranking_by_node = {
            row.get("root_cause_node"): row
            for row in root_cause_ranking
            if row.get("root_cause_node")
        }
        rows: List[dict] = []
        producer_artifacts: Dict[str, Set[str]] = {}
        for node in getattr(trace, "nodes", []):
            nid = str(node.get("id", "") or "")
            if nid and self._is_rca_producer_action(trace, nid):
                producer_artifacts[nid] = self._extract_artifact_keys(_text(node))

        for node in getattr(trace, "nodes", []):
            nid = str(node.get("id", "") or "")
            if not nid or not self._is_rca_producer_action(trace, nid):
                continue
            text = _text(node)
            lower = text.lower()
            artifacts = self._extract_artifact_keys(text)
            consumer, error_node = self._consumer_for_producer(trace, nid, artifacts)
            target = error_node or self._first_reachable_target(trace, nid, targets) or (next(iter(targets)) if targets else "")
            chain = ranking_by_node.get(nid, {})
            bad_hits = sum(1 for kw in RCA_BAD_STATE_KEYWORDS if kw in lower)
            distractor_hits = sum(1 for kw in RCA_DISTRACTOR_HINTS if kw in lower)
            recovery_hits = sum(1 for kw in RCA_RECOVERY_HINTS if kw in lower)
            artifact_hint_hits = sum(
                1
                for artifact in artifacts
                for hint in RCA_CONTROL_ARTIFACT_HINTS
                if hint in artifact.lower()
            )
            has_earlier_same_artifact = any(
                other != nid
                and _node_order_key(other) < _node_order_key(nid)
                and bool(artifacts & other_artifacts)
                for other, other_artifacts in producer_artifacts.items()
            )
            has_consumer = bool(consumer)
            has_error = bool(error_node)
            root_score = (
                1.0
                + min(2.0, bad_hits * 0.35)
                + (0.9 if has_consumer else 0.0)
                + (0.5 if has_error else 0.0)
                + min(0.45, artifact_hint_hits * 0.15)
                + min(1.0, float(chain.get("root_cause_score") or 0.0))
                - min(0.8, distractor_hits * 0.25)
                - min(0.9, recovery_hits * 0.3)
                - (0.55 if has_earlier_same_artifact else 0.0)
            )
            role = "upstream_state_producer" if has_consumer else "possible_state_producer"
            why = []
            if artifacts:
                why.append(f"produces artifact(s): {', '.join(sorted(artifacts)[:3])}")
            if consumer:
                why.append(f"later consumed by {consumer}")
            if error_node:
                why.append(f"before symptom/error {error_node}")
            if bad_hits:
                why.append("producer text contains bad-state cue words")
            if has_earlier_same_artifact:
                why.append("later producer for an artifact that already had an earlier producer")
            rows.append(
                {
                    "rank": 0,
                    "node_id": nid,
                    "span_id": node.get("span_id"),
                    "role": role,
                    "artifacts": sorted(artifacts),
                    "consumer_node": consumer,
                    "symptom_or_error_node": error_node or target,
                    "suggested_root_cause_type": "upstream_configuration_or_state_fault",
                    "candidate_score": round(root_score, 4),
                    "why_candidate": "; ".join(why) or "action appears to produce state consumed later",
                    "evidence": self._brief_node(trace, nid, max_chars=260),
                    "consumer_evidence": self._brief_node(trace, consumer, max_chars=180) if consumer else "",
                    "symptom_evidence": self._brief_node(trace, error_node, max_chars=180) if error_node else "",
                    "counterfactual_hint": "If this produced artifact/value were corrected, downstream consumer should not fail or return the same wrong value.",
                }
            )

        rows.sort(
            key=lambda row: (
                row["candidate_score"],
                row["role"] == "upstream_state_producer",
                -_node_order_key(row["node_id"])[0],
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows[:12], start=1):
            row["rank"] = rank
        return rows[:12]

    def _rank_root_cause_candidates(
        self,
        trace: Any,
        candidate_ids: Set[str],
        targets: Set[str],
        scores: Dict[str, float],
    ) -> List[dict]:
        rows: List[dict] = []
        for nid in candidate_ids:
            path, target, dependency_components = self._best_path_to_target(trace, nid, targets)
            if not path or not target or len(path) < 2:
                continue
            components = {
                "anomaly_score": self._anomaly_component(trace, nid, scores.get(nid, 0.0)),
                "final_answer_dependency": dependency_components["final_answer_dependency"],
                "causal_path_strength": dependency_components["causal_path_strength"],
                "semantic_error_score": self._semantic_error_score(trace, nid, targets),
                "propagation_position_score": self._propagation_position_score(trace, nid, path),
                "origin_preference_score": self._origin_preference_score(trace, nid, path),
            }
            root_score = sum(
                ROOT_CAUSE_SCORE_WEIGHTS[name] * components[name]
                for name in ROOT_CAUSE_SCORE_WEIGHTS
            )
            fault_usage = self._select_fault_usage(trace, path)
            edge_chain = self._path_edge_chain(trace, path)
            rows.append(
                {
                    "root_cause_node": nid,
                    "fault_origin": nid,
                    "fault_usage": fault_usage,
                    "final_failure": target,
                    "causal_path": list(path),
                    "path_valid": bool(len(path) >= 2),
                    "root_cause_type": self._root_cause_type(trace, nid),
                    "root_cause_score": round(root_score, 4),
                    "score_components": components,
                    "dependency_metrics": dependency_components,
                    "edge_chain": edge_chain,
                    "fault_origin_type": _node_type(trace.node_by_id.get(nid, {})),
                    "fault_usage_type": _node_type(trace.node_by_id.get(fault_usage, {})),
                    "final_failure_type": _node_type(trace.node_by_id.get(target, {})),
                    "fault_origin_span_id": trace.node_by_id.get(nid, {}).get("span_id"),
                    "fault_usage_span_id": trace.node_by_id.get(fault_usage, {}).get("span_id"),
                    "final_failure_span_id": trace.node_by_id.get(target, {}).get("span_id"),
                    "fault_origin_evidence": self._brief_node(trace, nid),
                    "fault_usage_evidence": self._brief_node(trace, fault_usage),
                    "final_failure_evidence": self._brief_node(trace, target),
                    "counterfactual_question": "Without this origin evidence/action, would the same final failure likely still occur?",
                    "location_policy": "type_conditioned_human_anchor",
                }
            )

        rows.sort(
            key=lambda row: (
                row.get("root_cause_score", 0.0),
                row.get("score_components", {}).get("origin_preference_score", 0.0),
                row.get("path_valid", False),
                len(row.get("causal_path", [])),
                scores.get(row.get("root_cause_node"), 0.0),
            ),
            reverse=True,
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        return rows

    def _select_diverse_causal_chains(self, ranking: List[dict]) -> List[dict]:
        """Greedily retain high-impact paths while suppressing near-duplicate chains."""
        remaining = [dict(row) for row in ranking if len(row.get("causal_path") or []) >= 2]
        selected: List[dict] = []
        covered_edges: Dict[str, Set[Tuple[str, str]]] = {}
        per_target_counts: Counter = Counter()

        while remaining and len(selected) < self.max_candidate_chains:
            best_index = None
            best_key = None
            best_overlap = 0.0
            best_new_edges = 0

            for index, row in enumerate(remaining):
                target = str(row.get("final_failure") or "")
                if per_target_counts[target] >= self.max_paths_per_target:
                    continue
                path = row.get("causal_path") or []
                path_edges = set(zip(path, path[1:]))
                already_covered = covered_edges.get(target, set())
                overlap = (
                    len(path_edges & already_covered) / len(path_edges)
                    if path_edges
                    else 1.0
                )
                if selected and overlap > self.max_path_overlap:
                    continue
                new_edges = len(path_edges - already_covered)
                metrics = row.get("dependency_metrics", {})
                adjusted_score = (
                    float(row.get("root_cause_score") or 0.0)
                    + 0.12 * float(metrics.get("final_answer_dependency") or 0.0)
                    + 0.08 * float(metrics.get("causal_path_strength") or 0.0)
                    - 0.25 * overlap
                )
                key = (adjusted_score, new_edges, -len(path))
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = index
                    best_overlap = overlap
                    best_new_edges = new_edges

            if best_index is None:
                break

            chosen = remaining.pop(best_index)
            target = str(chosen.get("final_failure") or "")
            path = chosen.get("causal_path") or []
            path_edges = set(zip(path, path[1:]))
            covered_edges.setdefault(target, set()).update(path_edges)
            per_target_counts[target] += 1
            chosen["diversity"] = {
                "edge_overlap_with_selected": round(best_overlap, 4),
                "new_edge_count": best_new_edges,
            }
            chosen["selection_rank"] = len(selected) + 1
            selected.append(chosen)

        return selected

    def _tool_frequency(self, trace: Any, node_ids: Iterable[str]) -> Counter:
        counter: Counter = Counter()
        for nid in node_ids:
            node = trace.node_by_id.get(nid, {})
            tool = _extract_tool_name(_text(node))
            if tool:
                counter[tool] += 1
        return counter

    def _score_node(
        self,
        trace: Any,
        nid: str,
        targets: Set[str],
        repeated_tools: Counter,
    ) -> float:
        node = trace.node_by_id.get(nid, {})
        ntype = _node_type(node)
        text = _lower_text(node)
        score = float(node.get("saliency", 0.0) or 0.0) + TYPE_BONUS.get(ntype, 0.8)

        error_hits = sum(1 for kw in ERROR_KEYWORDS if kw in text)
        tool_hits = sum(1 for kw in TOOL_DIAGNOSTIC_KEYWORDS if kw in text)
        drift_hits = sum(1 for kw in PLANNING_DRIFT_KEYWORDS if kw in text)
        retry_hits = sum(1 for kw in RETRY_KEYWORDS if kw in text)

        score += min(3.0, error_hits * 0.65)
        score += min(1.8, tool_hits * 0.18)
        score += min(1.2, drift_hits * 0.25)
        score += min(1.0, retry_hits * 0.4)

        tool = _extract_tool_name(text)
        if tool and repeated_tools.get(tool, 0) > 1:
            score += min(1.6, 0.45 * (repeated_tools[tool] - 1))

        if self._is_retry_node(trace, nid):
            score += 1.8
        if self._is_failure_neighborhood(trace, nid):
            score += 1.2
        if self._is_planning_action_bridge(trace, nid):
            score += 0.8

        dist = self._distance_to_targets(trace, nid, targets)
        if dist >= 0:
            score += 1.5 / (1.0 + dist)

        # 很长的被动观察文本通常噪声较大；
        # 除非它明确提到了失败、工具或规划线索，否则适当降权。
        if ntype == "Fact" and _node_chars(node) > 1200 and error_hits == 0 and tool_hits <= 1:
            score -= 0.9

        return score

    def _select_suspicious_sources(
        self,
        trace: Any,
        candidate_ids: Set[str],
        scores: Dict[str, float],
    ) -> List[str]:
        ranked = sorted(
            candidate_ids,
            key=lambda nid: (
                scores.get(nid, 0.0),
                _node_chars(trace.node_by_id.get(nid, {})) < 1600,
            ),
            reverse=True,
        )

        selected: List[str] = []
        seen_signatures: Set[Tuple[str, str]] = set()
        for nid in ranked:
            node = trace.node_by_id.get(nid, {})
            ntype = _node_type(node)
            if ntype not in CONTEXT_IMPORTANT_TYPES:
                continue

            signature = (ntype, (_extract_tool_name(_text(node)) or "")[:80])
            if signature in seen_signatures and scores.get(nid, 0.0) < 6.0:
                continue

            selected.append(nid)
            seen_signatures.add(signature)
            if len(selected) >= self.max_suspicious_sources:
                break
        return selected

    def _reachable_targets(self, trace: Any, src: str, targets: Set[str]) -> List[str]:
        rows: List[Tuple[float, str]] = []
        for dst in targets:
            if src == dst or src not in trace.graph or dst not in trace.graph:
                continue
            if not nx.has_path(trace.graph, src, dst):
                continue
            rows.append((self._distance(src, dst, trace.graph), dst))
        rows.sort()
        return [dst for _, dst in rows]

    def _shortest_path(self, trace: Any, src: str, dst: str) -> Optional[List[str]]:
        if src not in trace.graph or dst not in trace.graph:
            return None
        try:
            return nx.shortest_path(trace.graph, src, dst, weight="cost")
        except Exception:
            return None

    def _collect_local_context(self, trace: Any, nid: str, hops: int) -> Set[str]:
        kept: Set[str] = {nid}
        if nid not in trace.graph:
            return kept

        q = deque([(nid, 0)])
        seen = {nid}
        while q:
            cur, depth = q.popleft()
            if depth >= hops:
                continue

            neighbors = list(trace.graph.predecessors(cur)) + list(trace.graph.successors(cur))
            for nxt in neighbors:
                if nxt in seen:
                    continue
                seen.add(nxt)
                node = trace.node_by_id.get(nxt, {})
                if _node_type(node) in CONTEXT_IMPORTANT_TYPES:
                    kept.add(nxt)
                q.append((nxt, depth + 1))
        return kept

    def _collect_chain_context(self, trace: Any, nid: str) -> Set[str]:
        """
        保留节点周围的短解释链。

        这是这套方法针对 Agent trace 的核心适配之一：
        单个可疑节点通常不足以支撑根因判断，judge 至少需要看到本地 plan、
        实际执行的 action，以及对应的 fact/error，才能较稳定地做诊断。
        """
        kept = {nid}
        if nid not in trace.graph:
            return kept

        node = trace.node_by_id.get(nid, {})
        ntype = _node_type(node)

        if ntype == "Error":
            kept.update(self._nearest_of_types(trace, nid, direction="backward", wanted={"Action", "Planning", "Fact"}))
            kept.update(self._nearest_of_types(trace, nid, direction="forward", wanted={"Planning", "Conclusion"}))
        elif ntype == "Action":
            kept.update(self._nearest_of_types(trace, nid, direction="backward", wanted={"Planning", "Intent", "Fact", "Error"}))
            kept.update(self._nearest_of_types(trace, nid, direction="forward", wanted={"Fact", "Error", "Planning", "Conclusion"}))
        elif ntype == "Planning":
            kept.update(self._nearest_of_types(trace, nid, direction="backward", wanted={"Fact", "Error", "Intent"}))
            kept.update(self._nearest_of_types(trace, nid, direction="forward", wanted={"Action", "Conclusion", "Error"}))
        elif ntype == "Fact":
            kept.update(self._nearest_of_types(trace, nid, direction="backward", wanted={"Action", "Planning", "Error"}))
            kept.update(self._nearest_of_types(trace, nid, direction="forward", wanted={"Planning", "Conclusion", "Error"}))
        elif ntype == "Conclusion":
            kept.update(self._nearest_of_types(trace, nid, direction="backward", wanted={"Planning", "Fact", "Action", "Error"}))

        for ctx_id in list(kept):
            if self._is_retry_node(trace, ctx_id) or self._is_failure_neighborhood(trace, ctx_id):
                kept.update(self._collect_local_context(trace, ctx_id, hops=1))
        return {x for x in kept if x in trace.graph}

    def _nearest_of_types(
        self,
        trace: Any,
        nid: str,
        direction: str,
        wanted: Set[str],
        max_depth: int = 2,
    ) -> Set[str]:
        found: Set[str] = set()
        q = deque([(nid, 0)])
        seen = {nid}
        while q:
            cur, depth = q.popleft()
            if depth >= max_depth:
                continue

            if direction == "backward":
                neighbors = list(trace.graph.predecessors(cur))
            else:
                neighbors = list(trace.graph.successors(cur))

            for nxt in neighbors:
                if nxt in seen:
                    continue
                seen.add(nxt)
                ntype = _node_type(trace.node_by_id.get(nxt, {}))
                if ntype in wanted:
                    found.add(nxt)
                q.append((nxt, depth + 1))
        return found

    def _protected_subset(self, trace: Any, node_ids: Iterable[str]) -> Set[str]:
        protected: Set[str] = set()
        for nid in node_ids:
            node = trace.node_by_id.get(nid, {})
            ntype = _node_type(node)
            if ntype in {"Error", "Conclusion"}:
                protected.add(nid)
            elif self._is_retry_node(trace, nid) or self._is_failure_neighborhood(trace, nid):
                protected.add(nid)
            elif self._is_planning_action_bridge(trace, nid):
                protected.add(nid)
        return protected

    def _is_retry_node(self, trace: Any, nid: str) -> bool:
        if nid not in trace.graph:
            return False
        node = trace.node_by_id.get(nid, {})
        text = _lower_text(node)
        if any(kw in text for kw in RETRY_KEYWORDS):
            return True

        for pred in trace.graph.predecessors(nid):
            if str(trace.graph[pred][nid].get("edge_type", "") or "") == "Retry":
                return True
        for succ in trace.graph.successors(nid):
            if str(trace.graph[nid][succ].get("edge_type", "") or "") == "Retry":
                return True
        return False

    def _is_failure_neighborhood(self, trace: Any, nid: str) -> bool:
        node = trace.node_by_id.get(nid, {})
        if _node_type(node) == "Error":
            return True
        if any(kw in _lower_text(node) for kw in ERROR_KEYWORDS):
            return True

        if nid not in trace.graph:
            return False
        for other in list(trace.graph.predecessors(nid)) + list(trace.graph.successors(nid)):
            other_node = trace.node_by_id.get(other, {})
            if _node_type(other_node) == "Error":
                return True
            if any(kw in _lower_text(other_node) for kw in ERROR_KEYWORDS):
                return True
        return False

    def _is_planning_action_bridge(self, trace: Any, nid: str) -> bool:
        if nid not in trace.graph:
            return False
        node = trace.node_by_id.get(nid, {})
        ntype = _node_type(node)
        if ntype == "Planning":
            return any(_node_type(trace.node_by_id.get(s, {})) == "Action" for s in trace.graph.successors(nid))
        if ntype == "Action":
            return any(_node_type(trace.node_by_id.get(p, {})) == "Planning" for p in trace.graph.predecessors(nid))
        return False

    def _distance_to_targets(self, trace: Any, nid: str, targets: Set[str]) -> float:
        distances = []
        for dst in targets:
            if nid == dst or nid not in trace.graph or dst not in trace.graph:
                continue
            if not nx.has_path(trace.graph, nid, dst):
                continue
            distances.append(self._distance(nid, dst, trace.graph))
        return min(distances) if distances else -1.0

    def _distance(self, src: str, dst: str, graph: nx.DiGraph) -> float:
        try:
            return nx.shortest_path_length(graph, src, dst, weight="cost")
        except Exception:
            return float("inf")

    def _estimated_chars(self, trace: Any, node_ids: Iterable[str]) -> int:
        return sum(_node_chars(trace.node_by_id.get(nid, {})) for nid in node_ids)

    def _limits_enabled(self) -> bool:
        return self.max_nodes > 0 or self.char_budget > 0

    def _fits_limits(self, trace: Any, node_ids: Iterable[str]) -> bool:
        node_set = set(node_ids)
        if self.max_nodes > 0 and len(node_set) > self.max_nodes:
            return False
        if self.char_budget > 0 and self._estimated_chars(trace, node_set) > self.char_budget:
            return False
        return True

    def _trim(
        self,
        trace: Any,
        kept: Set[str],
        protected: Set[str],
        scores: Dict[str, float],
    ) -> Set[str]:
        if self._fits_limits(trace, kept):
            return kept

        removable = sorted(
            [nid for nid in kept if nid not in protected],
            key=lambda nid: (
                scores.get(nid, 0.0),
                -_node_chars(trace.node_by_id.get(nid, {})),
            ),
        )

        trimmed = set(kept)
        for nid in removable:
            if self._fits_limits(trace, trimmed):
                break
            trimmed.remove(nid)

        return trimmed
