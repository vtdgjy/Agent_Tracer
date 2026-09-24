import argparse
import hashlib
import json
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx

from agenttrace_target_pruning import AgentTraceSuspiciousPathPruner


# =====================================
# 统一图结构与工具函数
# =====================================

NODE_TYPE_BONUS = {
    "Conclusion": 5.0,
    "Error": 4.8,
    "Intent": 3.5,
    "Fact": 3.2,
    "Action": 2.8,
    "Planning": 2.4,
}

EDGE_BASE_COST = {
    # Temporal 成本更高，避免仅靠时序边形成的冗长链路。
    "Temporal": 1.3,
    "Call": 0.95,
    "AgentSpawn": 0.7,
    "SubagentReturn": 0.55,
    "Retry": 0.85,
    "Observation": 0.65,
    "Cognitive": 0.5,
}


@dataclass
class PruneResult:
    strategy: str
    nodes: List[dict]
    edges: List[dict]
    summary: dict
    meta: dict


class TraceGraph:
    """轻量封装：兼容 OTL_trace.py 导出的 nodes / edges 结构。"""

    def __init__(self, nodes: Sequence[dict], edges: Sequence[dict]):
        self.nodes = list(nodes)
        self.edges = list(edges)

        self.node_by_id: Dict[str, dict] = {}
        for n in self.nodes:
            nid = str(n.get("id", ""))
            if nid:
                self.node_by_id[nid] = n

        self.graph = nx.DiGraph()
        for nid, n in self.node_by_id.items():
            self.graph.add_node(nid, **n)

        for e in self.edges:
            s = e.get("source")
            t = e.get("target")
            if s not in self.node_by_id or t not in self.node_by_id:
                continue
            edge_type = e.get("edge_type", "Temporal")
            score = float(e.get("score", 0.0) or 0.0)
            cost = edge_cost(edge_type, score)
            payload = dict(e)
            payload["cost"] = cost
            self.graph.add_edge(s, t, **payload)

    @classmethod
    def from_json(cls, path: Path) -> "TraceGraph":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data.get("nodes", []), data.get("edges", []))

    def to_json_payload(self, node_ids: Set[str], strategy: str, meta: dict) -> dict:
        kept_nodes = [n for n in self.nodes if n.get("id") in node_ids]
        kept_ids = {n.get("id") for n in kept_nodes}
        kept_edges = [
            e for e in self.edges if e.get("source") in kept_ids and e.get("target") in kept_ids
        ]
        return {
            "strategy": strategy,
            "summary": summarize_subgraph(self, kept_nodes, kept_edges),
            "meta": meta,
            "nodes": kept_nodes,
            "edges": kept_edges,
        }


# =====================================
# 通用特征函数
# =====================================

def edge_cost(edge_type: str, score: float) -> float:
    base = EDGE_BASE_COST.get(edge_type or "Temporal", 1.1)
    # score 越高，成本越低。
    return max(0.05, base + (1.0 - max(0.0, min(1.0, score))) * 0.8)


def node_saliency(node: dict) -> float:
    base = float(node.get("saliency", 0.0) or 0.0)
    bonus = NODE_TYPE_BONUS.get(node.get("type", ""), 1.0)
    return base + bonus


def select_anchor_ids(trace: TraceGraph) -> Set[str]:
    anchors: Set[str] = set()
    for n in trace.nodes:
        nid = n.get("id")
        if not nid:
            continue
        ntype = n.get("type")
        text = str(n.get("content", "")).lower()
        if ntype == "Conclusion":
            anchors.add(nid)
        elif ntype == "Action" and ("final_answer" in text or "tool: final_answer" in text):
            anchors.add(nid)

    if not anchors and trace.nodes:
        anchors.add(trace.nodes[-1].get("id"))
    return {a for a in anchors if a}


def select_anomaly_ids(trace: TraceGraph, topk: int = 10) -> List[str]:
    scored = []
    for n in trace.nodes:
        nid = n.get("id")
        if not nid:
            continue
        txt = str(n.get("content", "")).lower()
        is_anomaly = n.get("type") == "Error" or any(
            k in txt for k in ["error", "failed", "exception", "timeout", "invalid", "formattingerror"]
        )
        if is_anomaly:
            scored.append((node_saliency(n) + 2.0, nid))
        else:
            scored.append((node_saliency(n), nid))
    scored.sort(reverse=True)
    return [nid for _, nid in scored[:topk]]


def lexical_token_set(text: str) -> Set[str]:
    toks = re.findall(r"[a-zA-Z0-9_\-]{3,}", (text or "").lower())
    stop = {
        "the", "and", "for", "with", "this", "that", "from", "into", "were", "been",
        "have", "has", "had", "you", "your", "not", "but", "are", "was", "all",
        "code", "tool", "task", "step", "final", "answer", "output", "input", "message",
    }
    return {t for t in toks if t not in stop}


def lexical_tokens(text: str) -> List[str]:
    """Tokenize text for dependency-free lexical retrieval baselines."""
    stop = {
        "the", "and", "for", "with", "this", "that", "from", "into", "were", "been",
        "have", "has", "had", "you", "your", "not", "but", "are", "was", "all",
        "code", "tool", "task", "step", "final", "answer", "output", "input", "message",
    }
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9_\-]{3,}", (text or "").lower())
        if token not in stop
    ]


def node_char_count(trace: TraceGraph, node_ids: Iterable[str]) -> int:
    return sum(len(str(trace.node_by_id.get(nid, {}).get("content", "") or "")) for nid in node_ids)


def fits_budget(
    trace: TraceGraph,
    node_ids: Set[str],
    char_budget: int,
    max_nodes: int,
) -> bool:
    return len(node_ids) <= max_nodes and node_char_count(trace, node_ids) <= char_budget


def diagnostic_query(trace: TraceGraph, anchor_ids: Optional[Set[str]] = None) -> str:
    """Build the shared target-conditioned query used by retrieval baselines."""
    anchors = anchor_ids if anchor_ids is not None else select_anchor_ids(trace)
    target_text = "\n".join(
        str(trace.node_by_id.get(nid, {}).get("content", "") or "")
        for nid in anchors
    )
    return (
        target_text
        + "\nerror failure failed invalid incorrect wrong exception timeout missing "
        "unsupported contradiction retry root cause"
    ).strip()


def select_ranked_under_budget(
    trace: TraceGraph,
    ranked_ids: Sequence[str],
    mandatory_ids: Set[str],
    char_budget: int,
    max_nodes: int,
) -> Tuple[Set[str], int, List[str]]:
    """Greedily pack nodes in rank order while enforcing a shared input budget."""
    order = {str(node.get("id", "")): idx for idx, node in enumerate(trace.nodes)}

    def mandatory_key(nid: str) -> Tuple[int, int]:
        node = trace.node_by_id.get(nid, {})
        text = str(node.get("content", "") or "").lower()
        priority = 3 if node.get("type") == "Action" and "final_answer" in text else 0
        priority = max(priority, 2 if node.get("type") == "Conclusion" else 0)
        priority = max(priority, 1 if node.get("type") == "Error" else 0)
        return priority, order.get(nid, -1)

    selected: Set[str] = set()
    skipped: List[str] = []
    used_chars = 0
    mandatory_order = sorted(
        (nid for nid in mandatory_ids if nid in trace.node_by_id),
        key=mandatory_key,
        reverse=True,
    )
    candidates = mandatory_order + [nid for nid in ranked_ids if nid not in mandatory_ids]

    for nid in candidates:
        if nid in selected or nid not in trace.node_by_id:
            continue
        chars = len(str(trace.node_by_id[nid].get("content", "") or ""))
        if len(selected) >= max_nodes or used_chars + chars > char_budget:
            skipped.append(nid)
            continue
        selected.add(nid)
        used_chars += chars

    return selected, used_chars, skipped


def summarize_subgraph(trace: TraceGraph, nodes: Sequence[dict], edges: Sequence[dict]) -> dict:
    full_nodes = max(1, len(trace.nodes))
    full_edges = max(1, len(trace.edges))

    node_types = Counter(n.get("type", "Unknown") for n in nodes)
    edge_types = Counter(e.get("edge_type", "Unknown") for e in edges)

    chars = sum(len(str(n.get("content", ""))) for n in nodes)
    est_tokens = max(1, chars // 4)

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_ratio": round(len(nodes) / full_nodes, 4),
        "edge_ratio": round(len(edges) / full_edges, 4),
        "estimated_tokens": est_tokens,
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
    }


def extract_edges_for_node_ids(trace: TraceGraph, node_ids: Set[str]) -> List[dict]:
    return [
        e for e in trace.edges if e.get("source") in node_ids and e.get("target") in node_ids
    ]


def extract_edges_for_selected_paths(trace: TraceGraph, meta: Dict[str, Any]) -> List[dict]:
    """Return only original trace edges traversed by an accepted TG-SPE path."""
    selected_pairs = {
        (row.get("source"), row.get("target"))
        for row in meta.get("selected_edge_pairs", [])
        if row.get("source") and row.get("target")
    }
    if not selected_pairs:
        return []
    return [
        edge
        for edge in trace.edges
        if (edge.get("source"), edge.get("target")) in selected_pairs
    ]


def path_cost_in_graph(g: nx.Graph, path: Sequence[str]) -> float:
    if not path or len(path) < 2:
        return 0.0
    cost = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        cost += float(g[u][v].get("weight", 1.0) or 1.0)
    return cost


# =====================================
# 策略1：加权 K 最短因果路径
# =====================================

class WeightedKShortestPathPruner:
    """
    核心思路：
    1) 从疑似异常源节点出发，找到到 anchor 的 K 条低成本路径；
    2) 保留这些路径节点并集，形成可解释的“因果候选链”。
    """

    strategy_name = "weighted_k_shortest_paths"

    def __init__(
        self,
        k_paths: int = 2,
        max_sources: int = 12,
        max_nodes: int = 80,
        char_budget: int = 14000,
    ):
        self.k_paths = k_paths
        self.max_sources = max_sources
        self.max_nodes = max_nodes
        self.char_budget = char_budget

    def run(self, trace: TraceGraph) -> PruneResult:
        anchors = select_anchor_ids(trace)
        sources = select_anomaly_ids(trace, topk=self.max_sources)

        kept: Set[str] = set(anchors)
        selected_path_meta = []

        for src in sources:
            if src not in trace.graph:
                continue
            for dst in anchors:
                if dst not in trace.graph or src == dst:
                    continue
                if not nx.has_path(trace.graph, src, dst):
                    continue

                # shortest_simple_paths 以 weight 最小优先返回路径。
                gen = nx.shortest_simple_paths(trace.graph, src, dst, weight="cost")
                for rank in range(self.k_paths):
                    try:
                        path = next(gen)
                    except StopIteration:
                        break
                    tentative = kept | set(path)
                    if not fits_budget(trace, tentative, self.char_budget, self.max_nodes):
                        continue
                    kept = tentative
                    selected_path_meta.append({
                        "source": src,
                        "target": dst,
                        "rank": rank + 1,
                        "path": path,
                    })
                    if len(kept) >= self.max_nodes:
                        break
                if len(kept) >= self.max_nodes:
                    break
            if len(kept) >= self.max_nodes:
                break

        kept_nodes = [n for n in trace.nodes if n.get("id") in kept]
        kept_edges = extract_edges_for_node_ids(trace, kept)

        summary = summarize_subgraph(trace, kept_nodes, kept_edges)
        meta = {
            "k_paths": self.k_paths,
            "max_sources": self.max_sources,
            "char_budget": self.char_budget,
            "used_chars": node_char_count(trace, kept),
            "anchors": sorted(anchors),
            "sources": sources,
            "selected_paths": selected_path_meta,
        }
        return PruneResult("weighted_k_shortest_paths", kept_nodes, kept_edges, summary, meta)


# =====================================
# 策略2：Dominator 必经链剪枝
# =====================================

class DominatorPruner:
    """
    在反图上以 anchor 为起点计算 immediate dominators：
    - 若节点 d 支配节点 n，则 n->anchor 的所有路径都必经 d。
    """

    strategy_name = "dominator_chain"

    def __init__(self, max_sources: int = 12, max_nodes: int = 80, char_budget: int = 14000):
        self.max_sources = max_sources
        self.max_nodes = max_nodes
        self.char_budget = char_budget

    def run(self, trace: TraceGraph) -> PruneResult:
        anchors = select_anchor_ids(trace)
        sources = select_anomaly_ids(trace, topk=self.max_sources)

        rev = trace.graph.reverse(copy=True)
        kept: Set[str] = set(anchors)
        chains = []

        for anchor in anchors:
            if anchor not in rev:
                continue
            # 只在 anchor 可达子图上做 dominator，避免不连通噪声。
            reachable = nx.descendants(rev, anchor) | {anchor}
            sub = rev.subgraph(reachable).copy()
            if len(sub) <= 1:
                continue

            try:
                idom = nx.immediate_dominators(sub, anchor)
            except Exception:
                continue

            for src in sources:
                if src not in sub or src == anchor:
                    continue
                chain = [src]
                cur = src
                guard = 0
                while cur in idom and idom[cur] != cur and guard < len(sub):
                    cur = idom[cur]
                    chain.append(cur)
                    if cur == anchor:
                        break
                    guard += 1

                if anchor in chain:
                    tentative = kept | set(chain)
                    if not fits_budget(trace, tentative, self.char_budget, self.max_nodes):
                        continue
                    kept = tentative
                    chains.append({"anchor": anchor, "source": src, "dominator_chain": chain})

                if len(kept) >= self.max_nodes:
                    break
            if len(kept) >= self.max_nodes:
                break

        kept_nodes = [n for n in trace.nodes if n.get("id") in kept]
        kept_edges = extract_edges_for_node_ids(trace, kept)
        summary = summarize_subgraph(trace, kept_nodes, kept_edges)
        meta = {
            "anchors": sorted(anchors),
            "sources": sources,
            "dominator_chains": chains,
            "char_budget": self.char_budget,
            "used_chars": node_char_count(trace, kept),
        }
        return PruneResult("dominator_chain", kept_nodes, kept_edges, summary, meta)


# =====================================
# 策略3：Prize-Collecting Steiner（启发式）
# =====================================

class PrizeCollectingSteinerPruner:
    """
    启发式近似：
    - anchors 作为初始树；
    - 反复选择“奖赏增益 - 路径成本”最大的候选节点并接入。
    """

    strategy_name = "pcst_heuristic"

    def __init__(
        self,
        max_candidates: int = 30,
        lambda_cost: float = 0.9,
        min_gain: float = 0.2,
        max_nodes: int = 90,
        char_budget: int = 14000,
    ):
        self.max_candidates = max_candidates
        self.lambda_cost = lambda_cost
        self.min_gain = min_gain
        self.max_nodes = max_nodes
        self.char_budget = char_budget

    def run(self, trace: TraceGraph) -> PruneResult:
        anchors = select_anchor_ids(trace)
        if not anchors:
            return PruneResult(
                "pcst_heuristic",
                [],
                [],
                {"node_count": 0, "edge_count": 0},
                {"error": "no anchors"},
            )

        # 为避免方向性导致不可连，Steiner 近似在无向图上构图。
        ug = nx.Graph()
        for n in trace.nodes:
            nid = n.get("id")
            if nid:
                ug.add_node(nid)
        for e in trace.edges:
            s = e.get("source")
            t = e.get("target")
            if s in ug and t in ug:
                c = edge_cost(e.get("edge_type", "Temporal"), float(e.get("score", 0.0) or 0.0))
                if ug.has_edge(s, t):
                    ug[s][t]["weight"] = min(ug[s][t]["weight"], c)
                else:
                    ug.add_edge(s, t, weight=c)

        scored_nodes = sorted(
            [n for n in trace.nodes if n.get("id")],
            key=lambda n: node_saliency(n),
            reverse=True,
        )
        candidates = [n.get("id") for n in scored_nodes[: self.max_candidates] if n.get("id")]

        tree_nodes: Set[str] = set(anchors)
        accepted = []

        improved = True
        while improved and len(tree_nodes) < self.max_nodes:
            improved = False
            best = None

            for cand in candidates:
                if cand in tree_nodes or cand not in ug:
                    continue

                prize = node_saliency(trace.node_by_id.get(cand, {}))
                best_path = None
                best_cost = math.inf

                for t in list(tree_nodes):
                    if t not in ug or not nx.has_path(ug, cand, t):
                        continue
                    try:
                        p = nx.shortest_path(ug, cand, t, weight="weight")
                    except Exception:
                        continue
                    c = path_cost_in_graph(ug, p)
                    if c < best_cost:
                        best_cost = c
                        best_path = p

                if best_path is None:
                    continue

                if not fits_budget(
                    trace,
                    tree_nodes | set(best_path),
                    self.char_budget,
                    self.max_nodes,
                ):
                    continue

                gain = prize - self.lambda_cost * best_cost
                if best is None or gain > best[0]:
                    best = (gain, cand, best_path, best_cost, prize)

            if best and best[0] >= self.min_gain:
                gain, cand, path, p_cost, prize = best
                tree_nodes.update(path)
                accepted.append(
                    {
                        "candidate": cand,
                        "path": path,
                        "path_cost": round(p_cost, 4),
                        "prize": round(prize, 4),
                        "gain": round(gain, 4),
                    }
                )
                improved = True

        kept_nodes = [n for n in trace.nodes if n.get("id") in tree_nodes]
        kept_edges = extract_edges_for_node_ids(trace, tree_nodes)
        summary = summarize_subgraph(trace, kept_nodes, kept_edges)
        meta = {
            "anchors": sorted(anchors),
            "accepted_candidates": accepted,
            "lambda_cost": self.lambda_cost,
            "min_gain": self.min_gain,
            "char_budget": self.char_budget,
            "used_chars": node_char_count(trace, tree_nodes),
        }
        return PruneResult("pcst_heuristic", kept_nodes, kept_edges, summary, meta)


# =====================================
# 策略4：子模近似预算选择
# =====================================

class SubmodularBudgetPruner:
    """
    在字符预算下做贪心：最大化边际收益/成本。
    收益由 saliency、锚点可达、连接奖励、冗余惩罚组成。
    """

    strategy_name = "submodular_budget"

    def __init__(
        self,
        char_budget: int = 18000,
        max_nodes: int = 80,
        redundancy_weight: float = 0.7,
    ):
        self.char_budget = char_budget
        self.max_nodes = max_nodes
        self.redundancy_weight = redundancy_weight

    def run(self, trace: TraceGraph) -> PruneResult:
        anchors = select_anchor_ids(trace)
        selected: Set[str] = set(anchors)
        used_chars = sum(len(str(trace.node_by_id.get(a, {}).get("content", ""))) for a in anchors)

        token_cache = {
            n.get("id"): lexical_token_set(str(n.get("content", "")))
            for n in trace.nodes if n.get("id")
        }

        steps = []

        while len(selected) < self.max_nodes:
            best = None
            for n in trace.nodes:
                nid = n.get("id")
                if not nid or nid in selected:
                    continue

                c = len(str(n.get("content", "")))
                if used_chars + c > self.char_budget:
                    continue

                gain = self._marginal_gain(trace, nid, selected, token_cache)
                ratio = gain / max(20, c)
                if best is None or ratio > best[0]:
                    best = (ratio, nid, gain, c)

            if best is None or best[2] <= 0:
                break

            _, nid, gain, c = best
            selected.add(nid)
            used_chars += c
            steps.append({"node_id": nid, "gain": round(gain, 4), "chars": c})

        kept_nodes = [n for n in trace.nodes if n.get("id") in selected]
        kept_edges = extract_edges_for_node_ids(trace, selected)
        summary = summarize_subgraph(trace, kept_nodes, kept_edges)
        meta = {
            "anchors": sorted(anchors),
            "char_budget": self.char_budget,
            "used_chars": used_chars,
            "selection_steps": steps,
        }
        return PruneResult("submodular_budget", kept_nodes, kept_edges, summary, meta)

    def _marginal_gain(
        self,
        trace: TraceGraph,
        nid: str,
        selected: Set[str],
        token_cache: Dict[str, Set[str]],
    ) -> float:
        node = trace.node_by_id.get(nid, {})
        gain = node_saliency(node)

        # 连通奖励：与已选节点有边连接可提升可解释性。
        conn_bonus = 0.0
        for s in selected:
            if trace.graph.has_edge(s, nid) or trace.graph.has_edge(nid, s):
                conn_bonus += 0.4
        gain += min(2.0, conn_bonus)

        # 锚点可达奖励：到任何 anchor 可达则奖励。
        anchors = [a for a in selected if trace.node_by_id.get(a, {}).get("type") == "Conclusion"]
        for a in anchors[:3]:
            if nid in trace.graph and a in trace.graph and nx.has_path(trace.graph, nid, a):
                gain += 1.0
                break

        # 冗余惩罚：与已选文本 token 集合重叠越大，收益越低。
        toks = token_cache.get(nid, set())
        if toks:
            max_sim = 0.0
            for s in selected:
                st = token_cache.get(s, set())
                if not st:
                    continue
                inter = len(toks & st)
                union = len(toks | st)
                sim = inter / union if union else 0.0
                max_sim = max(max_sim, sim)
            gain -= self.redundancy_weight * max_sim

        return gain


# =====================================
# Budget-matched non-graph baselines
# =====================================

class BM25RetrieverPruner:
    """Retrieve target-relevant nodes with Okapi BM25 and no graph features."""

    strategy_name = "bm25_retrieval"

    def __init__(
        self,
        char_budget: int = 14000,
        max_nodes: int = 80,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.char_budget = char_budget
        self.max_nodes = max_nodes
        self.k1 = k1
        self.b = b

    def run(self, trace: TraceGraph) -> PruneResult:
        anchors = select_anchor_ids(trace)
        query_terms = lexical_tokens(diagnostic_query(trace, anchors))
        query_counts = Counter(query_terms)
        documents = {
            nid: lexical_tokens(str(node.get("content", "") or ""))
            for nid, node in trace.node_by_id.items()
        }
        doc_count = max(1, len(documents))
        avg_doc_len = sum(len(tokens) for tokens in documents.values()) / doc_count
        doc_frequency = Counter()
        for tokens in documents.values():
            doc_frequency.update(set(tokens))

        scores: Dict[str, float] = {}
        for nid, tokens in documents.items():
            term_frequency = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term, query_tf in query_counts.items():
                tf = term_frequency.get(term, 0)
                if tf <= 0:
                    continue
                df = doc_frequency.get(term, 0)
                idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
                norm = tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(1.0, avg_doc_len))
                score += idf * (tf * (self.k1 + 1.0) / norm) * (1.0 + math.log(query_tf))
            scores[nid] = score

        order = {str(node.get("id", "")): idx for idx, node in enumerate(trace.nodes)}
        ranked = sorted(scores, key=lambda nid: (scores[nid], order.get(nid, -1)), reverse=True)
        selected, used_chars, skipped = select_ranked_under_budget(
            trace, ranked, anchors, self.char_budget, self.max_nodes
        )
        kept_nodes = [node for node in trace.nodes if node.get("id") in selected]
        kept_edges = extract_edges_for_node_ids(trace, selected)
        return PruneResult(
            "bm25_retrieval",
            kept_nodes,
            kept_edges,
            summarize_subgraph(trace, kept_nodes, kept_edges),
            {
                "query_source": "diagnosis target text plus fixed failure terms",
                "anchors": sorted(anchors),
                "k1": self.k1,
                "b": self.b,
                "char_budget": self.char_budget,
                "used_chars": used_chars,
                "skipped_for_budget": len(skipped),
                "ranking": [
                    {"node_id": nid, "score": round(scores[nid], 6)} for nid in ranked[:20]
                ],
            },
        )


class TfidfRetrieverPruner:
    """Retrieve nodes by sparse TF-IDF cosine similarity to the diagnosis target."""

    strategy_name = "tfidf_retrieval"

    def __init__(self, char_budget: int = 14000, max_nodes: int = 80):
        self.char_budget = char_budget
        self.max_nodes = max_nodes

    def run(self, trace: TraceGraph) -> PruneResult:
        anchors = select_anchor_ids(trace)
        query_counts = Counter(lexical_tokens(diagnostic_query(trace, anchors)))
        documents = {
            nid: Counter(lexical_tokens(str(node.get("content", "") or "")))
            for nid, node in trace.node_by_id.items()
        }
        doc_count = max(1, len(documents))
        doc_frequency = Counter()
        for counts in documents.values():
            doc_frequency.update(counts.keys())

        def idf(term: str) -> float:
            return math.log((1.0 + doc_count) / (1.0 + doc_frequency.get(term, 0))) + 1.0

        query_vector = {term: (1.0 + math.log(tf)) * idf(term) for term, tf in query_counts.items()}
        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
        scores: Dict[str, float] = {}
        for nid, counts in documents.items():
            vector = {term: (1.0 + math.log(tf)) * idf(term) for term, tf in counts.items()}
            norm = math.sqrt(sum(weight * weight for weight in vector.values()))
            dot = sum(query_vector.get(term, 0.0) * weight for term, weight in vector.items())
            scores[nid] = dot / (query_norm * norm) if query_norm and norm else 0.0

        order = {str(node.get("id", "")): idx for idx, node in enumerate(trace.nodes)}
        ranked = sorted(scores, key=lambda nid: (scores[nid], order.get(nid, -1)), reverse=True)
        selected, used_chars, skipped = select_ranked_under_budget(
            trace, ranked, anchors, self.char_budget, self.max_nodes
        )
        kept_nodes = [node for node in trace.nodes if node.get("id") in selected]
        kept_edges = extract_edges_for_node_ids(trace, selected)
        return PruneResult(
            "tfidf_retrieval",
            kept_nodes,
            kept_edges,
            summarize_subgraph(trace, kept_nodes, kept_edges),
            {
                "query_source": "diagnosis target text plus fixed failure terms",
                "anchors": sorted(anchors),
                "char_budget": self.char_budget,
                "used_chars": used_chars,
                "skipped_for_budget": len(skipped),
                "ranking": [
                    {"node_id": nid, "score": round(scores[nid], 6)} for nid in ranked[:20]
                ],
            },
        )


class TemporalWindowPruner:
    """Keep chronology-local windows around final/error anchors without graph ranking."""

    strategy_name = "temporal_window"

    def __init__(
        self,
        char_budget: int = 14000,
        max_nodes: int = 80,
        window_size: int = 8,
    ):
        self.char_budget = char_budget
        self.max_nodes = max_nodes
        self.window_size = window_size

    def run(self, trace: TraceGraph) -> PruneResult:
        anchors = select_anchor_ids(trace)
        node_ids = [str(node.get("id", "")) for node in trace.nodes if node.get("id")]
        index = {nid: idx for idx, nid in enumerate(node_ids)}
        anchor_positions = [index[nid] for nid in anchors if nid in index]
        ranked_pairs = []
        for nid in node_ids:
            distance = min((abs(index[nid] - pos) for pos in anchor_positions), default=len(node_ids))
            if nid in anchors or distance <= self.window_size:
                ranked_pairs.append((distance, -index[nid], nid))
        ranked_pairs.sort()
        ranked = [nid for _, _, nid in ranked_pairs]
        selected, used_chars, skipped = select_ranked_under_budget(
            trace, ranked, anchors, self.char_budget, self.max_nodes
        )
        kept_nodes = [node for node in trace.nodes if node.get("id") in selected]
        kept_edges = extract_edges_for_node_ids(trace, selected)
        return PruneResult(
            "temporal_window",
            kept_nodes,
            kept_edges,
            summarize_subgraph(trace, kept_nodes, kept_edges),
            {
                "anchors": sorted(anchors),
                "window_size": self.window_size,
                "char_budget": self.char_budget,
                "used_chars": used_chars,
                "skipped_for_budget": len(skipped),
            },
        )


class RandomBudgetPruner:
    """Deterministic random-node control with the same anchor and character budget."""

    strategy_name = "random_budget"

    def __init__(self, char_budget: int = 14000, max_nodes: int = 80, seed: int = 42):
        self.char_budget = char_budget
        self.max_nodes = max_nodes
        self.seed = seed

    def run(self, trace: TraceGraph) -> PruneResult:
        anchors = select_anchor_ids(trace)
        node_ids = [str(node.get("id", "")) for node in trace.nodes if node.get("id")]
        signature = "\n".join(node_ids).encode("utf-8")
        trace_seed = self.seed ^ int.from_bytes(hashlib.sha256(signature).digest()[:8], "big")
        rng = random.Random(trace_seed)
        ranked = [nid for nid in node_ids if nid not in anchors]
        rng.shuffle(ranked)
        selected, used_chars, skipped = select_ranked_under_budget(
            trace, ranked, anchors, self.char_budget, self.max_nodes
        )
        kept_nodes = [node for node in trace.nodes if node.get("id") in selected]
        kept_edges = extract_edges_for_node_ids(trace, selected)
        return PruneResult(
            "random_budget",
            kept_nodes,
            kept_edges,
            summarize_subgraph(trace, kept_nodes, kept_edges),
            {
                "anchors": sorted(anchors),
                "seed": self.seed,
                "trace_seed": trace_seed,
                "char_budget": self.char_budget,
                "used_chars": used_chars,
                "skipped_for_budget": len(skipped),
            },
        )


# =====================================
# 策略5：异常子图 + 根因子图（组合封装）
# =====================================

class AnomalyRootCausePruner:
    """
    组合视图：
    - anomaly_subgraph: 围绕异常点的局部邻域；
    - rootcause_subgraph: 使用加权最短路径策略定位到结论的根因链。
    """

    def __init__(
        self,
        neighborhood_hops: int = 2,
        anomaly_topk: int = 6,
        char_budget: int = 14000,
        max_nodes: int = 80,
    ):
        self.neighborhood_hops = neighborhood_hops
        self.anomaly_topk = anomaly_topk
        self.char_budget = char_budget
        self.max_nodes = max_nodes
        self.rootcause_delegate = WeightedKShortestPathPruner(
            k_paths=2,
            max_sources=10,
            max_nodes=min(70, max_nodes),
            char_budget=char_budget,
        )

    def run(self, trace: TraceGraph) -> Dict[str, PruneResult]:
        anomaly_ids = select_anomaly_ids(trace, topk=self.anomaly_topk)
        kept = self._collect_neighborhood(trace, anomaly_ids, self.neighborhood_hops)
        ranked = sorted(
            kept,
            key=lambda nid: node_saliency(trace.node_by_id.get(nid, {})),
            reverse=True,
        )
        kept, used_chars, skipped = select_ranked_under_budget(
            trace,
            ranked,
            set(anomaly_ids),
            self.char_budget,
            self.max_nodes,
        )

        anomaly_nodes = [n for n in trace.nodes if n.get("id") in kept]
        anomaly_edges = extract_edges_for_node_ids(trace, kept)
        anomaly_result = PruneResult(
            strategy="anomaly_neighborhood",
            nodes=anomaly_nodes,
            edges=anomaly_edges,
            summary=summarize_subgraph(trace, anomaly_nodes, anomaly_edges),
            meta={
                "anomaly_ids": anomaly_ids,
                "neighborhood_hops": self.neighborhood_hops,
                "char_budget": self.char_budget,
                "used_chars": used_chars,
                "skipped_for_budget": len(skipped),
            },
        )

        root_result = self.rootcause_delegate.run(trace)
        root_result = PruneResult(
            strategy="rootcause_subgraph",
            nodes=root_result.nodes,
            edges=root_result.edges,
            summary=root_result.summary,
            meta={**root_result.meta, "delegate_strategy": root_result.strategy},
        )
        return {
            "anomaly_subgraph": anomaly_result,
            "rootcause_subgraph": root_result,
        }

    def _collect_neighborhood(self, trace: TraceGraph, seed_ids: List[str], hops: int) -> Set[str]:
        kept = set(seed_ids)
        if not seed_ids:
            return kept

        undirected = trace.graph.to_undirected()
        for s in seed_ids:
            if s not in undirected:
                continue
            q = deque([(s, 0)])
            seen = {s}
            while q:
                cur, d = q.popleft()
                if d >= hops:
                    continue
                for nxt in undirected.neighbors(cur):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    kept.add(nxt)
                    q.append((nxt, d + 1))
        return kept


# =====================================
# 批量运行与评估
# =====================================

def evaluate_against_anchor_and_anomaly(trace: TraceGraph, node_ids: Set[str]) -> dict:
    anchors = select_anchor_ids(trace)
    anomalies = set(select_anomaly_ids(trace, topk=10))

    anchor_coverage = len(node_ids & anchors) / max(1, len(anchors))
    anomaly_coverage = len(node_ids & anomalies) / max(1, len(anomalies))

    edges = extract_edges_for_node_ids(trace, node_ids)
    edge_type_count = Counter(e.get("edge_type", "Unknown") for e in edges)

    return {
        "anchor_coverage": round(anchor_coverage, 4),
        "anomaly_coverage": round(anomaly_coverage, 4),
        "retained_edge_types": dict(edge_type_count),
    }


def save_prune_result(path: Path, result: PruneResult, trace: TraceGraph):
    node_ids = {n.get("id") for n in result.nodes if n.get("id")}
    payload = {
        "strategy": result.strategy,
        "summary": result.summary,
        "meta": result.meta,
        "evaluation": evaluate_against_anchor_and_anomaly(trace, node_ids),
        "nodes": result.nodes,
        "edges": result.edges,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


AVAILABLE_STRATEGIES = (
    "bm25_retrieval",
    "tfidf_retrieval",
    "temporal_window",
    "random_budget",
    "weighted_k_shortest_paths",
    "dominator_chain",
    "pcst_heuristic",
    "submodular_budget",
    "agenttrace_suspicious_paths",
    "anomaly_neighborhood",
    "rootcause_subgraph",
)


def run_all_strategies(
    trace: TraceGraph,
    char_budget: int = 14000,
    max_nodes: int = 80,
    strategy_names: Optional[Set[str]] = None,
    random_seed: int = 42,
    tg_spe_max_targets: int = 4,
    tg_spe_char_budget: int = 0,
    tg_spe_max_nodes: int = 0,
    tg_spe_min_final_dependency: float = 0.55,
    tg_spe_min_path_confidence: float = 0.45,
    tg_spe_max_path_hops: int = 18,
    tg_spe_max_path_overlap: float = 0.85,
) -> Dict[str, PruneResult]:
    results: Dict[str, PruneResult] = {}
    requested = set(strategy_names or AVAILABLE_STRATEGIES)
    if "tg_spe" in requested:
        requested.remove("tg_spe")
        requested.add("agenttrace_suspicious_paths")
    unknown = requested - set(AVAILABLE_STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown pruning strategies: {', '.join(sorted(unknown))}")

    strategies = [
        BM25RetrieverPruner(char_budget=char_budget, max_nodes=max_nodes),
        TfidfRetrieverPruner(char_budget=char_budget, max_nodes=max_nodes),
        TemporalWindowPruner(char_budget=char_budget, max_nodes=max_nodes, window_size=8),
        RandomBudgetPruner(char_budget=char_budget, max_nodes=max_nodes, seed=random_seed),
        WeightedKShortestPathPruner(
            k_paths=2,
            max_sources=12,
            max_nodes=max_nodes,
            char_budget=char_budget,
        ),
        DominatorPruner(max_sources=12, max_nodes=max_nodes, char_budget=char_budget),
        PrizeCollectingSteinerPruner(
            max_candidates=30,
            lambda_cost=0.9,
            min_gain=0.2,
            max_nodes=max_nodes,
            char_budget=char_budget,
        ),
        SubmodularBudgetPruner(
            char_budget=char_budget,
            max_nodes=max_nodes,
            redundancy_weight=0.7,
        ),
    ]

    for strategy in strategies:
        if strategy.strategy_name not in requested:
            continue
        result = strategy.run(trace)
        results[result.strategy] = result

    if "agenttrace_suspicious_paths" in requested:
        agenttrace_selector = AgentTraceSuspiciousPathPruner(
            max_suspicious_sources=12,
            max_nodes=tg_spe_max_nodes,
            char_budget=tg_spe_char_budget,
            local_hops=1,
            max_paths_per_target=4,
            max_candidate_chains=5,
            min_dependency_confidence=0.65,
            max_targets=tg_spe_max_targets,
            min_final_dependency=tg_spe_min_final_dependency,
            min_path_confidence=tg_spe_min_path_confidence,
            max_path_hops=tg_spe_max_path_hops,
            max_path_overlap=tg_spe_max_path_overlap,
        )
        agenttrace_node_ids, agenttrace_meta = agenttrace_selector.select(trace)
        agenttrace_nodes = [n for n in trace.nodes if n.get("id") in agenttrace_node_ids]
        agenttrace_edges = extract_edges_for_selected_paths(trace, agenttrace_meta)
        agenttrace_summary = summarize_subgraph(trace, agenttrace_nodes, agenttrace_edges)
        results[agenttrace_selector.strategy_name] = PruneResult(
            strategy=agenttrace_selector.strategy_name,
            nodes=agenttrace_nodes,
            edges=agenttrace_edges,
            summary=agenttrace_summary,
            meta=agenttrace_meta,
        )

    combo_names = {"anomaly_neighborhood", "rootcause_subgraph"}
    if requested & combo_names:
        combo = AnomalyRootCausePruner(
            neighborhood_hops=2,
            anomaly_topk=6,
            char_budget=char_budget,
            max_nodes=max_nodes,
        ).run(trace)
        if "anomaly_neighborhood" in requested:
            results[combo["anomaly_subgraph"].strategy] = combo["anomaly_subgraph"]
        if "rootcause_subgraph" in requested:
            results[combo["rootcause_subgraph"].strategy] = combo["rootcause_subgraph"]

    return results


def _iter_graph_files(input_dir: Path) -> Iterable[Path]:
    # 仅处理 *_graph.json，自动排除 prompt_pack 等文件。
    for p in sorted(input_dir.glob("*_graph.json")):
        if p.name.endswith("_pruned_graph.json"):
            continue
        yield p


def main():
    parser = argparse.ArgumentParser(description="Run multiple pruning strategies on trace graph JSON files")
    parser.add_argument("--input-dir", required=True, help="目录：包含 *_graph.json")
    parser.add_argument("--output-dir", required=True, help="输出目录：保存各策略子图结果")
    parser.add_argument("--char-budget", type=int, default=14000, help="所有策略共享的节点文本字符预算")
    parser.add_argument("--max-nodes", type=int, default=80, help="所有策略共享的最大节点数")
    parser.add_argument(
        "--strategies",
        default="all",
        help="逗号分隔的策略名，默认 all；TG-SPE 可写为 tg_spe",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="随机预算基线的固定种子")
    parser.add_argument("--tg-spe-max-targets", type=int, default=4, help="TG-SPE 最多保留的诊断目标数")
    parser.add_argument("--tg-spe-char-budget", type=int, default=0, help="TG-SPE 可选字符上限；0 表示关闭")
    parser.add_argument("--tg-spe-max-nodes", type=int, default=0, help="TG-SPE 可选节点上限；0 表示关闭")
    parser.add_argument("--tg-spe-min-final-dependency", type=float, default=0.55)
    parser.add_argument("--tg-spe-min-path-confidence", type=float, default=0.45)
    parser.add_argument("--tg-spe-max-path-hops", type=int, default=18)
    parser.add_argument("--tg-spe-max-path-overlap", type=float, default=0.85)
    parser.add_argument(
        "--layout",
        choices=["judge", "stem"],
        default="judge",
        help="judge 生成 local_<trace>_pruning；stem 保留旧的 <graph-stem> 目录布局",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="处理样本上限，0 表示全量")
    parser.add_argument("--workers", type=int, default=8, help="并行 worker 数")
    args = parser.parse_args()

    strategy_names = None
    if args.strategies.strip().lower() != "all":
        strategy_names = {
            item.strip() for item in args.strategies.split(",") if item.strip()
        }

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_files = list(_iter_graph_files(input_dir))
    if args.max_samples:
        graph_files = graph_files[: args.max_samples]

    summary_rows = []

    def _process_graph(graph_path: Path):
        trace = TraceGraph.from_json(graph_path)
        results = run_all_strategies(
            trace,
            char_budget=args.char_budget,
            max_nodes=args.max_nodes,
            strategy_names=strategy_names,
            random_seed=args.random_seed,
            tg_spe_max_targets=args.tg_spe_max_targets,
            tg_spe_char_budget=args.tg_spe_char_budget,
            tg_spe_max_nodes=args.tg_spe_max_nodes,
            tg_spe_min_final_dependency=args.tg_spe_min_final_dependency,
            tg_spe_min_path_confidence=args.tg_spe_min_path_confidence,
            tg_spe_max_path_hops=args.tg_spe_max_path_hops,
            tg_spe_max_path_overlap=args.tg_spe_max_path_overlap,
        )

        sample_name = graph_path.stem
        if args.layout == "judge" and sample_name.endswith("_graph"):
            sample_name = sample_name[: -len("_graph")] + "_pruning"
        sample_out = output_dir / sample_name
        sample_out.mkdir(parents=True, exist_ok=True)

        rows = []
        for name, result in results.items():
            out_file = sample_out / f"{name}.json"
            save_prune_result(out_file, result, trace)
            rows.append(
                {
                    "sample": graph_path.stem,
                    "strategy": name,
                    "node_count": result.summary.get("node_count", 0),
                    "edge_count": result.summary.get("edge_count", 0),
                    "node_ratio": result.summary.get("node_ratio", 0),
                    "edge_ratio": result.summary.get("edge_ratio", 0),
                    "estimated_tokens": result.summary.get("estimated_tokens", 0),
                    "used_chars": result.meta.get("used_chars", sum(len(str(n.get("content", "") or "")) for n in result.nodes)),
                }
            )
        return rows

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_process_graph, graph_path) for graph_path in graph_files]
            for fut in as_completed(futures):
                summary_rows.extend(fut.result())
    else:
        for graph_path in graph_files:
            summary_rows.extend(_process_graph(graph_path))

    count = len(graph_files)

    summary_payload = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "processed_samples": count,
        "char_budget": args.char_budget,
        "max_nodes": args.max_nodes,
        "strategies": sorted(strategy_names or AVAILABLE_STRATEGIES),
        "random_seed": args.random_seed,
        "tg_spe_max_targets": args.tg_spe_max_targets,
        "tg_spe_char_budget": args.tg_spe_char_budget or None,
        "tg_spe_max_nodes": args.tg_spe_max_nodes or None,
        "layout": args.layout,
        "rows": summary_rows,
    }
    (output_dir / "pruning_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
