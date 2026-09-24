import ast
import json
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
import os
from trace_input_adapter import normalize_trace_for_parser, find_first_trace_obj
from semantic_trace_validator import analyze_semantic_trace, classify_tool_result

try:
    from pruning_strategies import (
        TraceGraph as ExternalPruningTraceGraph,
        run_all_strategies as run_external_pruning_strategies,
        save_prune_result as save_external_prune_result,
    )
except Exception:
    ExternalPruningTraceGraph = None
    run_external_pruning_strategies = None
    save_external_prune_result = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# python OTL_trace.py --local-mode --local-trace-dir .\gaia\trail_data\GAIA --local-annotations-dir .\gaia\trail_data\processed_annotations_gaia --out-prefix trail_gaia_local --output-dir .\gaia\output_graphs_useLLMv2 --max-samples 1

# ==================== 代码内开关（可直接修改） ====================
USE_LLM_IN_CODE = True
DEFAULT_LLM_BACKEND = "deepseek"  # 可选: deepseek / local_openai
DEFAULT_LLM_MODEL = "deepseek-chat"
DEFAULT_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_LOCAL_LLM_MODEL = "qwen2.5-7b-instruct"
DEFAULT_LOCAL_API_KEY = "EMPTY"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8000/v1"

# 图构建策略参数（统一放在代码内，减少命令行复杂度）
ALLOW_LLM_IN_BATCH_IN_CODE = True  # 是否允许在批处理模式下启用 LLM 依赖边判断
MAX_PAIRS_IN_CODE = 24
PREFILTER_TOPK_IN_CODE = 3
MAX_SALIENT_NODES_IN_CODE = 40
MAX_PROMPT_CHARS_IN_CODE = 18000
SIMILARITY_THRESHOLD_IN_CODE = 0.06
LAYERED_REVIEW_MARGIN_IN_CODE = 0.02
LAYERED_DIRECT_LINK_MARGIN_IN_CODE = 0.08
PRUNED_MAX_BACKTRACK_HOPS_IN_CODE = 4
PRUNED_MAX_TEMPORAL_HOPS_IN_CODE = 2
USE_EXTERNAL_PRUNING_MODULE_IN_CODE = True


def _get_default_llm_runtime():
    if DEFAULT_LLM_BACKEND == "local_openai":
        return {
            "llm_backend": DEFAULT_LLM_BACKEND,
            "llm_model": DEFAULT_LOCAL_LLM_MODEL,
            "api_key": DEFAULT_LOCAL_API_KEY,
            "base_url": DEFAULT_LOCAL_BASE_URL,
        }
    return {
        "llm_backend": DEFAULT_LLM_BACKEND,
        "llm_model": DEFAULT_LLM_MODEL,
        "api_key": DEFAULT_API_KEY,
        "base_url": DEFAULT_BASE_URL,
    }


class OTelTraceParser:
    def __init__(self, trace_data, enable_llm=False, llm_model="deepseek-chat", api_key=None, base_url=None, llm_backend="deepseek"):
        self.trace_data = trace_data
        self.nodes = []
        self.edges = []
        self.counts = {"I": 0, "P": 0, "A": 0, "F": 0, "C": 0, "E": 0}

        self.span_meta = {}
        self.span_nodes = defaultdict(list)
        self._edge_set = set()
        self.semantic_analysis = analyze_semantic_trace(trace_data)
        self.final_answer_span_ids = set(self.semantic_analysis.get("final_answer_span_ids", []))
        self.agent_scopes, self.span_agent_meta = self._infer_agent_scopes()

        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.api_key = self._normalize_api_key(api_key)
        self.base_url = base_url
        self.enable_llm = enable_llm and OpenAI is not None and self._llm_config_ready()
        self.client = None
        if self.enable_llm:
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = OpenAI(**client_kwargs)

    def _normalize_api_key(self, api_key):
        if api_key:
            return api_key
        if self.llm_backend == "local_openai":
            return DEFAULT_LOCAL_API_KEY
        return None

    def _infer_agent_scopes(self):
        """Infer agent boundaries from spawn-tool ancestry without Agent changes."""

        spans = []

        def walk(items):
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                spans.append(item)
                walk(item.get("child_spans", []))

        walk(self.trace_data.get("spans", []))
        span_by_id = {
            str(span.get("span_id")): span
            for span in spans
            if span.get("span_id")
        }

        def ancestors(span):
            result = []
            parent_id = str(span.get("parent_span_id") or "")
            while parent_id and parent_id not in result:
                result.append(parent_id)
                parent = span_by_id.get(parent_id)
                if not parent:
                    break
                parent_id = str(parent.get("parent_span_id") or "")
            return result

        def is_agent_run(span):
            name = str(span.get("span_name") or "").lower()
            return "agent" in name and (name.endswith(".run") or name.endswith("agent.run"))

        def is_spawn(span):
            attrs = span.get("span_attributes", {}) or {}
            name = f"{attrs.get('tool.name', '')} {span.get('span_name', '')}".lower()
            return bool(
                re.search(
                    r"(?:spawn|delegate|create).{0,20}(?:sub.?agent|agent)|"
                    r"(?:sub.?agent|agent).{0,20}(?:spawn|delegate)",
                    name,
                )
            )

        def relaxed_mapping(value):
            if isinstance(value, dict):
                return value
            if not isinstance(value, str) or not value.strip():
                return {}
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(value)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    return parsed
            return {}

        def task_from_span(span):
            attrs = span.get("span_attributes", {}) or {}
            parsed = relaxed_mapping(attrs.get("input.value"))
            task = parsed.get("objective") or parsed.get("task") or parsed.get("prompt")
            if not task:
                task = attrs.get("input.value") or span.get("span_name") or "Agent task"
            task = str(task).replace("user:", "", 1).strip()
            return task[:240]

        agent_runs = [span for span in spans if is_agent_run(span)]
        spawn_spans = [span for span in spans if is_spawn(span)]
        spawn_ids = {str(span.get("span_id")) for span in spawn_spans}

        # A trace can contain nested Agent.run wrappers inside the same agent.
        # Only the closest Agent.run below each spawn tool starts a new scope.
        boundary_roots = {}
        for spawn in spawn_spans:
            spawn_id = str(spawn.get("span_id") or "")
            candidates = []
            for agent_span in agent_runs:
                chain = ancestors(agent_span)
                if spawn_id in chain:
                    candidates.append((chain.index(spawn_id), agent_span))
            if candidates:
                _, closest = min(candidates, key=lambda item: item[0])
                boundary_roots[str(closest.get("span_id"))] = spawn_id

        agent_run_ids = {str(span.get("span_id")) for span in agent_runs}

        def scope_for_span(span):
            chain = [str(span.get("span_id") or "")] + ancestors(span)
            for span_id in chain:
                if span_id in boundary_roots:
                    return span_id
            root_runs = [span_id for span_id in chain if span_id in agent_run_ids]
            return root_runs[-1] if root_runs else "root"

        scope_parent = {}
        for agent_id, spawn_id in boundary_roots.items():
            spawn_span = span_by_id.get(spawn_id, {})
            scope_parent[agent_id] = scope_for_span(spawn_span)

        root_agent_ids = {
            scope_for_span(span)
            for span in spans
            if scope_for_span(span) not in boundary_roots
        }
        root_agent_ids.discard("root")
        if not root_agent_ids:
            root_agent_ids.add("root")

        scopes = {}
        for root_id in sorted(root_agent_ids):
            root_span = span_by_id.get(root_id, {})
            scopes[root_id] = {
                "agent_id": root_id,
                "agent_label": "MainAgent",
                "agent_role": "main",
                "agent_depth": 0,
                "parent_agent_id": None,
                "spawn_span_id": None,
                "agent_task": task_from_span(root_span) if root_span else "Main agent",
            }

        ordered_subagents = sorted(
            boundary_roots,
            key=lambda agent_id: str(span_by_id.get(agent_id, {}).get("timestamp") or ""),
        )
        for index, agent_id in enumerate(ordered_subagents, 1):
            spawn_id = boundary_roots[agent_id]
            parent_id = scope_parent.get(agent_id) or next(iter(root_agent_ids))
            spawn_span = span_by_id.get(spawn_id, {})
            scopes[agent_id] = {
                "agent_id": agent_id,
                "agent_label": f"Subagent {index}",
                "agent_role": "subagent",
                "agent_depth": 1,
                "parent_agent_id": parent_id,
                "spawn_span_id": spawn_id,
                "agent_task": task_from_span(spawn_span),
            }

        def depth_for(agent_id, visited=None):
            visited = set(visited or [])
            if agent_id in visited:
                return 0
            visited.add(agent_id)
            parent_id = (scopes.get(agent_id) or {}).get("parent_agent_id")
            if not parent_id or parent_id not in scopes:
                return 0
            return 1 + depth_for(parent_id, visited)

        def path_for(agent_id):
            path = []
            current = agent_id
            visited = set()
            while current and current not in visited:
                visited.add(current)
                path.append(current)
                current = (scopes.get(current) or {}).get("parent_agent_id")
            return list(reversed(path))

        for agent_id, scope in scopes.items():
            scope["agent_depth"] = depth_for(agent_id)
            scope["agent_path"] = path_for(agent_id)

        span_agent_meta = {}
        for span in spans:
            span_id = str(span.get("span_id") or "")
            agent_id = scope_for_span(span)
            scope = scopes.get(agent_id)
            if span_id and scope:
                span_agent_meta[span_id] = dict(scope)

        # Dialogue benchmarks such as Who&When provide the speaker directly
        # instead of representing agents through nested spawn spans. Prefer
        # this explicit identity when present so every graph node remains
        # attributable to the original agent.
        for span in spans:
            span_id = str(span.get("span_id") or "")
            attrs = span.get("span_attributes", {}) or {}
            explicit_agent = str(
                attrs.get("agent.id")
                or attrs.get("agent.name")
                or attrs.get("chat.name")
                or ""
            ).strip()
            if not span_id or not explicit_agent:
                continue
            scope_key = f"dialogue::{explicit_agent}"
            explicit_scope = scopes.setdefault(
                scope_key,
                {
                    "agent_id": explicit_agent,
                    "agent_label": explicit_agent,
                    "agent_role": "participant",
                    "agent_depth": 0,
                    "parent_agent_id": None,
                    "spawn_span_id": None,
                    "agent_task": "conversation participant",
                    "agent_path": [explicit_agent],
                },
            )
            span_agent_meta[span_id] = dict(explicit_scope)
        return scopes, span_agent_meta

    def _llm_config_ready(self):
        if self.llm_backend == "local_openai":
            return bool(self.base_url)
        return bool(self.api_key)

    def parse(self):
        spans = self.trace_data.get("spans", [])
        for span in spans:
            self._traverse_span(span)

        self._add_semantic_violation_nodes()
        self._deduplicate_nodes()
        self._build_temporal_edges()
        self._build_dialogue_edges()
        self._build_call_edges()
        self._build_agent_edges()
        self._build_observation_edges()
        self._build_semantic_edges()
        self._build_retry_edges()
        self._compute_saliency()
        return self.nodes, self.edges

    def _traverse_span(self, span):
        before_node_count = len(self.nodes)
        span_id = span.get("span_id")
        parent_span_id = span.get("parent_span_id")
        span_name = span.get("span_name", "")
        attrs = span.get("span_attributes", {})
        span_kind = attrs.get("openinference.span.kind", "")
        timestamp = span.get("timestamp", "")

        if span_id:
            self.span_meta[span_id] = {
                "span_name": span_name,
                "parent_span_id": parent_span_id,
                "timestamp": timestamp,
                "source_step": attrs.get("conversation.step_index"),
                "source_step_number": attrs.get("conversation.step_number"),
                "source_agent": attrs.get("agent.id") or attrs.get("agent.name") or attrs.get("chat.name"),
                "source_raw_agent": attrs.get("conversation.raw_agent") or attrs.get("chat.name"),
                "source_role": attrs.get("chat.role"),
                **self.span_agent_meta.get(str(span_id), {}),
            }

        status_code = str(span.get("status_code", "")).lower()
        status_message = span.get("status_message", "")
        if status_code == "error" or self._looks_like_error(status_message):
            err_text = status_message if status_message else f"SpanError: {span_name}"
            self._add_node("E", "Error", err_text, span_id, parent_span_id, timestamp)

        if span_name == "CodeAgent.run":
            input_val = attrs.get("input.value", "")
            if input_val:
                task_content = self._extract_task_from_input(input_val)
                self._add_node("I", "Intent", task_content, span_id, parent_span_id, timestamp)

        elif span_name == "LiteLLMModel.__call__":
            input_prompt = attrs.get("llm.input_messages.0.message.content", "")
            input_val = attrs.get("input.value", "")
            output_content = attrs.get("llm.output_messages.0.message.content", "")
            if output_content:
                node_type = (
                    "Conclusion"
                    if span_id in self.final_answer_span_ids
                    else self._classify_llm_output(output_content)
                )
                prefix = self._type_to_prefix(node_type)
                self._add_node(prefix, node_type, output_content, span_id, parent_span_id, timestamp)

                format_issue = self._infer_missing_required_tag(input_prompt, input_val, output_content)
                if format_issue:
                    self._add_node("E", "Error", format_issue, span_id, parent_span_id, timestamp)

        elif span_name.startswith("Step "):
            output_val = attrs.get("output.value", "")
            if output_val:
                self._add_node("F", "Fact", output_val, span_id, parent_span_id, timestamp)

        if "tool.name" in attrs or span_kind == "TOOL" or span_name.endswith("Tool"):
            tool_name = attrs.get("tool.name", span_name)
            tool_args = attrs.get("input.value", "")
            tool_output = attrs.get("output.value", "")
            tool_desc = attrs.get("tool.description", "")
            tool_params = attrs.get("tool.parameters", "")
            action_content = f"Tool: {tool_name} | Args: {tool_args} | Params: {tool_params}"
            self._add_node("A", "Action", action_content, span_id, parent_span_id, timestamp)

            arg_issue = self._infer_tool_arg_issue(tool_args, tool_params, tool_name)
            if arg_issue:
                self._add_node("E", "Error", arg_issue, span_id, parent_span_id, timestamp)

            if tool_desc:
                self._add_node("F", "Fact", f"ToolSpec: {tool_name} | {tool_desc}", span_id, parent_span_id, timestamp)

            if not self._is_empty_output(tool_output):
                result_status = classify_tool_result(tool_output, span.get("status_code", ""))
                self._add_node(
                    "F",
                    "Fact",
                    f"ToolResult: {tool_output}",
                    span_id,
                    parent_span_id,
                    timestamp,
                    extra={"tool_result_status": result_status},
                )
                if result_status["is_error"]:
                    error_text = result_status.get("error") or self._to_text(tool_output, max_chars=1200)
                    self._add_node(
                        "E",
                        "Error",
                        f"ToolResultError: {error_text}",
                        span_id,
                        parent_span_id,
                        timestamp,
                        extra={"tool_result_status": result_status},
                    )

        for ev in span.get("events", []):
            if not isinstance(ev, dict):
                continue
            ev_name = ev.get("name", "")
            ev_attrs = ev.get("attributes", {})
            ev_ts = ev.get("timestamp", timestamp)
            ev_text = self._to_text({"event": ev_name, "attributes": ev_attrs}, max_chars=1000)
            if self._looks_like_error(ev_name) or self._looks_like_error(ev_text):
                self._add_node("E", "Error", f"EventError: {ev_text}", span_id, parent_span_id, ev_ts)

        for log in span.get("logs", []):
            body = log.get("body", {})
            if not isinstance(body, dict):
                continue

            log_ts = log.get("timestamp", timestamp)
            function_name = body.get("function.name", "")
            function_args = body.get("function.arguments", None)
            function_output = body.get("function.output", None)

            if function_name and function_name not in {"main"}:
                args_text = self._to_text(function_args, max_chars=1200)
                self._add_node("A", "Action", f"Function: {function_name} | Args: {args_text}", span_id, parent_span_id, log_ts)

            if not self._is_empty_output(function_output):
                out_text = self._to_text(function_output, max_chars=2200)
                self._add_node("F", "Fact", f"FunctionOutput: {out_text}", span_id, parent_span_id, log_ts)
                result_status = classify_tool_result(function_output, span.get("status_code", ""))
                if result_status["is_error"]:
                    self._add_node(
                        "E",
                        "Error",
                        f"FunctionError: {out_text[:1200]}",
                        span_id,
                        parent_span_id,
                        log_ts,
                        extra={"tool_result_status": result_status},
                    )

        # 通用兜底：兼容 OTLP/对话类 span 中的文本字段，避免因字段名差异导致空图。
        if len(self.nodes) == before_node_count:
            role = str(attrs.get("chat.role", "")).lower()
            content = (
                attrs.get("chat.content")
                or attrs.get("output.value")
                or attrs.get("llm.output_messages.0.message.content")
                or attrs.get("message.content")
                or ""
            )
            if content:
                if role == "assistant":
                    node_type = (
                        "Conclusion"
                        if span_id in self.final_answer_span_ids
                        else self._classify_llm_output(str(content))
                    )
                elif role == "system":
                    node_type = "Planning"
                else:
                    node_type = "Fact"
                self._add_node(
                    self._type_to_prefix(node_type),
                    node_type,
                    str(content),
                    span_id,
                    parent_span_id,
                    timestamp,
                )

        for child in span.get("child_spans", []):
            self._traverse_span(child)

    def _extract_task_from_input(self, input_val):
        try:
            data = json.loads(input_val)
            if isinstance(data, dict):
                return data.get("task", input_val)
            return input_val
        except Exception:
            return input_val

    def _classify_llm_output(self, text):
        upper = text.upper()
        if "FINAL ANSWER:" in upper or "final_answer(" in text:
            return "Conclusion"
        if "[PLAN]" in text or "### 1. FACTS" in upper or "FACTS GIVEN" in upper:
            return "Planning"
        if "OBSERVATION" in upper or "EXECUTION LOGS" in upper:
            return "Fact"
        return "Planning"

    def _type_to_prefix(self, node_type):
        mapping = {
            "Intent": "I",
            "Planning": "P",
            "Action": "A",
            "Fact": "F",
            "Conclusion": "C",
            "Error": "E",
        }
        return mapping.get(node_type, "F")

    def _looks_like_error(self, text):
        if not isinstance(text, str) or not text.strip():
            return False
        t = re.sub(
            r"\berror\b\s*[:=]\s*(?:none|null|false|['\"]{2})",
            "",
            text.lower(),
        )
        keys = ["error", "exception", "traceback", "failed", "invalid", "unexpected keyword", "timeout"]
        return any(k in t for k in keys)

    def _infer_tool_arg_issue(self, tool_args, tool_params, tool_name):
        args_obj = _safe_json_parse(tool_args) if isinstance(tool_args, str) else None
        params_obj = _safe_json_parse(tool_params) if isinstance(tool_params, str) else None
        if not isinstance(args_obj, dict):
            return None

        kwargs = args_obj.get("kwargs", {})
        if isinstance(kwargs, dict) and "" in kwargs:
            return f"ToolArgError: {tool_name} called with empty argument key in kwargs"

        if params_obj == {} and isinstance(kwargs, dict) and len(kwargs) > 0:
            return f"ToolArgError: {tool_name} expects empty params but received kwargs={self._to_text(kwargs, max_chars=220)}"
        return None

    def _infer_missing_required_tag(self, input_prompt, input_value, output_content):
        required_tags = self._extract_required_tags(input_prompt, input_value)
        if not required_tags or not isinstance(output_content, str):
            return None

        missing_tags = [tag for tag in required_tags if tag not in output_content]
        if not missing_tags:
            return None

        return (
            "FormattingError: missing required tag(s) "
            f"{', '.join(missing_tags)} in LLM output"
        )

    def _extract_required_tags(self, input_prompt, input_value):
        tags = set()

        prompt_patterns = [
            r"write the ['\"](?:\\n)?(<[^>\s]+>)['\"] tag and stop there",
            r"write the (?:['\"])?(<[^>\s]+>)(?:['\"])? tag and stop there",
        ]

        if isinstance(input_prompt, str) and input_prompt:
            for pattern in prompt_patterns:
                for match in re.findall(pattern, input_prompt, flags=re.IGNORECASE):
                    tags.add(match)

        parsed_input = _safe_json_parse(input_value) if isinstance(input_value, str) else None
        if isinstance(parsed_input, dict):
            for seq in parsed_input.get("stop_sequences", []) or []:
                if isinstance(seq, str):
                    seq = seq.strip()
                    if re.fullmatch(r"<[^>\s]+>", seq):
                        tags.add(seq)

        return sorted(tags)

    def _add_node(self, node_prefix, node_type, content, span_id, parent_span_id, timestamp, extra=None):
        self.counts[node_prefix] += 1
        node_id = f"{node_prefix}_{self.counts[node_prefix]}"

        raw_content = self._to_text(content, max_chars=5000)
        display_content = raw_content.replace("\n", " ").strip()
        if len(display_content) > 160:
            display_content = display_content[:157] + "..."

        node = {
            "id": node_id,
            "type": node_type,
            "content": raw_content,
            "display": display_content,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "timestamp": timestamp,
            "saliency": 0.0,
        }
        agent_meta = self.span_agent_meta.get(str(span_id), {}) if span_id else {}
        if agent_meta:
            node.update(agent_meta)
        source_meta = self.span_meta.get(str(span_id), {}) if span_id else {}
        for key in (
            "source_step",
            "source_step_number",
            "source_agent",
            "source_raw_agent",
            "source_role",
        ):
            if source_meta.get(key) is not None:
                node[key] = source_meta[key]
        if isinstance(extra, dict):
            node.update(extra)
        self.nodes.append(node)
        if span_id:
            self.span_nodes[span_id].append(node_id)
        return node_id

    def _add_semantic_violation_nodes(self):
        for observation in self.semantic_analysis.get("semantic_observations", []):
            for node_id in self.span_nodes.get(observation.get("span_id"), []):
                node = next((item for item in self.nodes if item.get("id") == node_id), None)
                if node is not None:
                    node["semantic_observation"] = observation

        violations = self.semantic_analysis.get("semantic_violations", [])
        for violation in violations:
            source_span_id = violation.get("source_span_id")
            meta = self.span_meta.get(source_span_id, {})
            score = float(violation.get("confidence") or 0.0)
            if violation.get("severity") == "MEDIUM":
                score *= 0.6
            elif violation.get("severity") == "LOW":
                score *= 0.25
            reach_probability = float(violation.get("final_reach_probability") or 0.0)
            if violation.get("reaches_final_answer"):
                score *= max(0.65, reach_probability)
            else:
                score *= 0.35

            for node_id in self.span_nodes.get(source_span_id, []):
                node = next((item for item in self.nodes if item.get("id") == node_id), None)
                if node is not None:
                    node.setdefault("semantic_violations", []).append(violation)
                    node["semantic_error_score"] = max(node.get("semantic_error_score", 0.0), round(score, 4))

            content = (
                f"SemanticViolation[{violation.get('violation_type')}]: "
                f"expected={violation.get('expected')} observed={violation.get('observed')} | "
                f"propagation={violation.get('propagation_state')} "
                f"reaches_final={violation.get('reaches_final_answer')} | "
                f"evidence={violation.get('evidence', '')}"
            )
            self._add_node(
                "E",
                "Error",
                content,
                source_span_id,
                meta.get("parent_span_id"),
                meta.get("timestamp", ""),
                extra={
                    "semantic_violation": violation,
                    "semantic_violation_type": violation.get("violation_type"),
                    "semantic_source_span_id": source_span_id,
                    "semantic_error_score": round(score, 4),
                },
            )

    def _add_edge(self, source, target, edge_type, score=1.0, method="rule", extra=None):
        if not source or not target or source == target:
            return
        key = (source, target, edge_type)
        if key in self._edge_set:
            return
        payload = {
            "source": source,
            "target": target,
            "edge_type": edge_type,
            "score": score,
            "method": method,
        }
        if isinstance(extra, dict):
            payload.update(extra)
        self.edges.append(payload)
        self._edge_set.add(key)

    def _to_text(self, value, max_chars=2000):
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False)
            except Exception:
                text = str(value)
        text = text.strip()
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text

    def _is_empty_output(self, output):
        return (
            output is None
            or output == "<null>"
            or output == []
            or output == {}
            or (isinstance(output, str) and output.strip() == "")
        )

    def _deduplicate_nodes(self):
        seen = {}
        deduped = []
        old_to_new = {}
        retained_by_id = {}

        for node in self.nodes:
            fingerprint = self._fingerprint(node)
            if fingerprint in seen:
                retained_id = seen[fingerprint]
                old_to_new[node["id"]] = retained_id
                retained = retained_by_id[retained_id]
                retained_step = retained.get("source_step")
                duplicate_step = node.get("source_step")
                # Semantic deduplication must not erase temporal provenance.
                # Preserve every distinct source occurrence on the retained
                # node while keeping one graph vertex and therefore the same
                # causal topology/path score as the original implementation.
                if isinstance(retained_step, int) and isinstance(duplicate_step, int):
                    occurrences = retained.setdefault(
                        "source_occurrences",
                        [
                            {
                                "source_step": retained_step,
                                "source_step_number": retained.get("source_step_number"),
                                "source_agent": retained.get("source_agent"),
                                "source_raw_agent": retained.get("source_raw_agent"),
                                "source_role": retained.get("source_role"),
                                "span_id": retained.get("span_id"),
                            }
                        ],
                    )
                    occurrence = {
                        "source_step": duplicate_step,
                        "source_step_number": node.get("source_step_number"),
                        "source_agent": node.get("source_agent"),
                        "source_raw_agent": node.get("source_raw_agent"),
                        "source_role": node.get("source_role"),
                        "span_id": node.get("span_id"),
                    }
                    if not any(
                        item.get("source_step") == duplicate_step
                        and item.get("span_id") == occurrence["span_id"]
                        for item in occurrences
                    ):
                        occurrences.append(occurrence)
                    retained["source_steps"] = sorted(
                        {
                            int(item["source_step"])
                            for item in occurrences
                            if isinstance(item.get("source_step"), int)
                        }
                    )
                continue
            deduped.append(node)
            seen[fingerprint] = node["id"]
            retained_by_id[node["id"]] = node
            old_to_new[node["id"]] = node["id"]

        self.nodes = deduped
        valid_ids = {node["id"] for node in self.nodes}
        remapped_span_nodes = defaultdict(list)
        for span_id, node_ids in self.span_nodes.items():
            for node_id in node_ids:
                mapped = old_to_new.get(node_id, node_id)
                if mapped in valid_ids and mapped not in remapped_span_nodes[span_id]:
                    remapped_span_nodes[span_id].append(mapped)
        self.span_nodes = remapped_span_nodes

    def _fingerprint(self, node):
        text = re.sub(r"\s+", " ", node["content"].lower())
        text = text[:800]
        if node["type"] == "Error":
            return f"{node['type']}|{node.get('span_id')}|{text}"
        return f"{node['type']}|{node.get('agent_id')}|{text}"

    def _build_temporal_edges(self):
        for idx in range(len(self.nodes) - 1):
            self._add_edge(
                self.nodes[idx]["id"],
                self.nodes[idx + 1]["id"],
                "Temporal",
                score=1.0,
                method="sequence",
            )

    def _build_call_edges(self):
        for span_id, meta in self.span_meta.items():
            parent_span_id = meta.get("parent_span_id")
            if not parent_span_id:
                continue
            parent_nodes = self.span_nodes.get(parent_span_id, [])
            child_nodes = self.span_nodes.get(span_id, [])
            if not parent_nodes or not child_nodes:
                continue
            self._add_edge(
                parent_nodes[-1],
                child_nodes[0],
                "Call",
                score=1.0,
                method="span_parent",
            )

    def _build_dialogue_edges(self):
        """Link consecutive dialogue turns as explicit reply dependencies."""
        step_nodes = defaultdict(list)
        for node in self.nodes:
            step = node.get("source_step")
            if isinstance(step, int):
                step_nodes[step].append(node["id"])
        ordered_steps = sorted(step_nodes)
        for previous_step, current_step in zip(ordered_steps, ordered_steps[1:]):
            if current_step != previous_step + 1:
                continue
            source_id = step_nodes[previous_step][-1]
            target_id = step_nodes[current_step][0]
            source_node = self._node_by_id(source_id) or {}
            target_node = self._node_by_id(target_id) or {}
            self._add_edge(
                source_id,
                target_id,
                "reasoning_dependency",
                score=0.85,
                method="dialogue_turn_reply",
                extra={
                    "source_step": previous_step,
                    "target_step": current_step,
                    "source_agent": source_node.get("source_agent"),
                    "target_agent": target_node.get("source_agent"),
                },
            )

    def _build_agent_edges(self):
        node_by_id = {node["id"]: node for node in self.nodes}
        for scope in self.agent_scopes.values():
            if scope.get("agent_role") != "subagent":
                continue
            spawn_span_id = scope.get("spawn_span_id")
            agent_id = scope.get("agent_id")
            spawn_nodes = self.span_nodes.get(spawn_span_id, [])
            entry_nodes = self.span_nodes.get(agent_id, [])
            if spawn_nodes and entry_nodes:
                source = next(
                    (
                        node_id
                        for node_id in spawn_nodes
                        if node_by_id.get(node_id, {}).get("type") == "Action"
                    ),
                    spawn_nodes[0],
                )
                target = next(
                    (
                        node_id
                        for node_id in entry_nodes
                        if node_by_id.get(node_id, {}).get("type") == "Intent"
                    ),
                    entry_nodes[0],
                )
                self._add_edge(
                    source,
                    target,
                    "AgentSpawn",
                    score=1.0,
                    method="spawn_tool_ancestry",
                    extra={
                        "agent_id": agent_id,
                        "parent_agent_id": scope.get("parent_agent_id"),
                        "spawn_span_id": spawn_span_id,
                    },
                )

        for link in self.semantic_analysis.get("delivery_links", []):
            if link.get("delivery_method") != "subagent_return":
                continue
            source_span_id = link.get("source_span_id")
            target_span_id = link.get("consumer_span_id")
            source_nodes = self.span_nodes.get(source_span_id, [])
            target_nodes = self.span_nodes.get(target_span_id, [])
            if not source_nodes or not target_nodes:
                continue
            target = next(
                (
                    node_id
                    for node_id in reversed(target_nodes)
                    if node_by_id.get(node_id, {}).get("type") == "Fact"
                ),
                target_nodes[-1],
            )
            source_agent = self.span_agent_meta.get(str(source_span_id), {})
            self._add_edge(
                source_nodes[-1],
                target,
                "SubagentReturn",
                score=float(link.get("confidence") or 0.95),
                method="subagent_return",
                extra={
                    "agent_id": source_agent.get("agent_id"),
                    "parent_agent_id": source_agent.get("parent_agent_id"),
                    "tool_call_id": link.get("tool_call_id"),
                },
            )

    def _build_observation_edges(self):
        for idx, node in enumerate(self.nodes):
            if node["type"] != "Action":
                continue
            for j in range(idx + 1, min(len(self.nodes), idx + 4)):
                nxt = self.nodes[j]
                if nxt["type"] not in {"Fact", "Error"}:
                    continue
                if node.get("span_id") and nxt.get("span_id") and node.get("span_id") != nxt.get("span_id"):
                    continue
                self._add_edge(
                    node["id"],
                    nxt["id"],
                    "Observation",
                    score=0.95,
                    method="action_followed_by_observation",
                )
                break

    def _build_semantic_edges(self):
        for violation_node in self.nodes:
            if not violation_node.get("semantic_violation"):
                continue
            violation = violation_node["semantic_violation"]
            source_span_id = violation.get("source_span_id")
            origin_span_id = violation.get("origin_span_id")
            source_nodes = [
                node_id
                for node_id in self.span_nodes.get(source_span_id, [])
                if node_id != violation_node["id"]
            ]
            if source_nodes:
                self._add_edge(
                    source_nodes[0],
                    violation_node["id"],
                    "SemanticValidation",
                    score=float(violation.get("confidence") or 0.9),
                    method=violation.get("violation_type", "semantic_contract"),
                )
            origin_nodes = self.span_nodes.get(origin_span_id, []) if origin_span_id else []
            if origin_nodes and source_nodes:
                self._add_edge(
                    origin_nodes[0],
                    source_nodes[0],
                    "SemanticOrigin",
                    score=float(violation.get("confidence") or 0.9),
                    method="tool_argument_producer",
                )
        violation_nodes_by_span = defaultdict(list)
        for node in self.nodes:
            if node.get("semantic_violation") and node.get("span_id"):
                violation_nodes_by_span[node["span_id"]].append(node["id"])

        edge_type_by_state = {
            "PROPAGATED": "SemanticPropagation",
            "TRANSFORMED": "SemanticPropagation",
            "REJECTED": "SemanticRejection",
            "EXPOSED": "EvidenceExposure",
        }
        for link in self.semantic_analysis.get("delivery_links", []):
            state = str(link.get("state") or "UNKNOWN")
            edge_type = edge_type_by_state.get(state)
            if not edge_type:
                continue
            source_span_id = link.get("source_span_id")
            target_span_id = link.get("consumer_span_id")
            source_nodes = violation_nodes_by_span.get(source_span_id) or self.span_nodes.get(source_span_id, [])
            target_nodes = self.span_nodes.get(target_span_id, [])
            if not source_nodes or not target_nodes:
                continue
            for source_node_id in source_nodes:
                self._add_edge(
                    source_node_id,
                    target_nodes[0],
                    edge_type,
                    score=float(link.get("confidence") or 0.5),
                    method=str(link.get("delivery_method") or "semantic_delivery"),
                    extra={
                        "propagation_state": state,
                        "tool_call_id": link.get("tool_call_id"),
                        "matched_signatures": link.get("matched_signatures", []),
                    },
                )

    def _extract_tool_name_from_action(self, text):
        if not isinstance(text, str):
            return ""
        m = re.search(r"Tool:\s*([^|]+)", text)
        return m.group(1).strip().lower() if m else ""

    def _build_retry_edges(self):
        last_tool_node = {}
        for node in self.nodes:
            if node["type"] != "Action":
                continue
            tool = self._extract_tool_name_from_action(node.get("content", ""))
            if not tool:
                continue
            prev_id = last_tool_node.get(tool)
            if prev_id:
                self._add_edge(
                    prev_id,
                    node["id"],
                    "Retry",
                    score=0.9,
                    method="same_tool_reinvocation",
                    extra={"tool": tool},
                )
            last_tool_node[tool] = node["id"]

    def _compute_saliency(self):
        type_weight = {"Intent": 4.0, "Planning": 2.5, "Action": 3.0, "Fact": 3.2, "Conclusion": 5.0, "Error": 5.2}
        for idx, node in enumerate(self.nodes):
            score = type_weight.get(node["type"], 1.0)
            text = node["content"].lower()

            if "final answer" in text or "final_answer(" in text:
                score += 2.5
            if "observation" in text or "execution logs" in text:
                score += 0.8
            if "error" in text or "failed" in text:
                score += 0.8
            if node.get("semantic_violation"):
                score += 2.0
            if node.get("semantic_error_score"):
                score += float(node.get("semantic_error_score"))

            recency = (idx + 1) / max(1, len(self.nodes))
            score += 0.4 * recency
            node["saliency"] = round(score, 4)

    def select_salient_nodes(self, max_nodes=40, max_prompt_chars=18000):
        ranked = sorted(self.nodes, key=lambda n: (n["saliency"], n["id"]), reverse=True)
        selected = []
        total_chars = 0

        for node in ranked:
            candidate_chars = len(node["content"])
            if len(selected) >= max_nodes:
                break
            if total_chars + candidate_chars > max_prompt_chars:
                continue
            selected.append(node)
            total_chars += candidate_chars

        selected_ids = {n["id"] for n in selected}
        selected = sorted(selected, key=lambda n: self._node_index(n["id"]))
        selected_edges = [
            e for e in self.edges if e["source"] in selected_ids and e["target"] in selected_ids
        ]
        return selected, selected_edges

    def _edge_priority(self, edge_type):
        priority = {
            "Cognitive": 5,
            "SubagentReturn": 5,
            "AgentSpawn": 4,
            "Observation": 4,
            "Call": 3,
            "Retry": 2,
            "Temporal": 1,
        }
        return priority.get(edge_type, 0)

    def _select_outcome_anchor_ids(self):
        anchor_ids = set()
        anchor_span_ids = set()

        for node in self.nodes:
            text = str(node.get("content", "")).lower()
            if node["type"] == "Conclusion":
                anchor_ids.add(node["id"])
            elif node["type"] == "Action" and ("final_answer" in text or "tool: final_answer" in text):
                anchor_ids.add(node["id"])

        for node_id in anchor_ids:
            node = self._node_by_id(node_id)
            if node and node.get("span_id"):
                anchor_span_ids.add(node["span_id"])

        # 保留终局附近的关键错误，避免格式错误/工具调用错误在压缩后丢失。
        for node in self.nodes[-12:]:
            text = str(node.get("content", "")).lower()
            if node["type"] == "Error" and any(
                keyword in text
                for keyword in [
                    "formattingerror",
                    "agentexecutionerror",
                    "toolargerror",
                    "typeerror",
                    "exception",
                    "timeout",
                    "failed",
                ]
            ):
                anchor_ids.add(node["id"])
            elif anchor_span_ids and node.get("span_id") in anchor_span_ids and node["type"] in {"Planning", "Action", "Fact"}:
                anchor_ids.add(node["id"])

        if not anchor_ids and self.nodes:
            anchor_ids.add(self.nodes[-1]["id"])
        return anchor_ids

    def _node_by_id(self, node_id):
        for node in self.nodes:
            if node["id"] == node_id:
                return node
        return None

    def _summarize_view(self, nodes, edges):
        edge_type_count = defaultdict(int)
        for edge in edges:
            edge_type_count[edge["edge_type"]] += 1
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "counts": {
                "Intent": sum(1 for n in nodes if n["type"] == "Intent"),
                "Planning": sum(1 for n in nodes if n["type"] == "Planning"),
                "Action": sum(1 for n in nodes if n["type"] == "Action"),
                "Fact": sum(1 for n in nodes if n["type"] == "Fact"),
                "Conclusion": sum(1 for n in nodes if n["type"] == "Conclusion"),
                "Error": sum(1 for n in nodes if n["type"] == "Error"),
            },
            "edge_types": dict(edge_type_count),
        }

    def select_outcome_relevant_subgraph(
        self,
        max_nodes=40,
        max_prompt_chars=18000,
        max_backtrack_hops=4,
        max_temporal_hops=2,
    ):
        anchor_ids = self._select_outcome_anchor_ids()
        reverse_adj = defaultdict(list)
        for edge in self.edges:
            reverse_adj[edge["target"]].append(edge)

        frontier = []
        visited_state = set()
        kept_ids = set(anchor_ids)
        best_distance = {node_id: 0 for node_id in anchor_ids}

        for node_id in anchor_ids:
            frontier.append((node_id, 0, 0))

        while frontier:
            current_id, hops, temporal_hops = frontier.pop(0)
            if hops >= max_backtrack_hops:
                continue

            incoming_edges = sorted(
                reverse_adj.get(current_id, []),
                key=lambda edge: (
                    self._edge_priority(edge.get("edge_type")),
                    float(edge.get("score", 0.0) or 0.0),
                ),
                reverse=True,
            )

            for edge in incoming_edges:
                source_id = edge.get("source")
                edge_type = edge.get("edge_type")
                next_temporal_hops = temporal_hops + 1 if edge_type == "Temporal" else 0
                if edge_type == "Temporal" and next_temporal_hops > max_temporal_hops:
                    continue

                state = (source_id, hops + 1, next_temporal_hops)
                if state in visited_state:
                    continue
                visited_state.add(state)

                kept_ids.add(source_id)
                best_distance[source_id] = min(best_distance.get(source_id, 10 ** 9), hops + 1)
                frontier.append((source_id, hops + 1, next_temporal_hops))

        candidate_nodes = [node for node in self.nodes if node["id"] in kept_ids]
        ranked_candidates = sorted(
            candidate_nodes,
            key=lambda node: (
                0 if node["id"] in anchor_ids else 1,
                best_distance.get(node["id"], 10 ** 9),
                -node.get("saliency", 0.0),
                self._node_index(node["id"]),
            ),
        )

        selected = []
        selected_ids = set()
        total_chars = 0
        for node in ranked_candidates:
            candidate_chars = len(node.get("content", ""))
            must_keep = node["id"] in anchor_ids
            if len(selected) >= max_nodes and not must_keep:
                continue
            if not must_keep and total_chars + candidate_chars > max_prompt_chars:
                continue
            if node["id"] in selected_ids:
                continue
            selected.append(node)
            selected_ids.add(node["id"])
            total_chars += candidate_chars

        selected = sorted(selected, key=lambda node: self._node_index(node["id"]))
        selected_edges = [
            edge for edge in self.edges
            if edge.get("source") in selected_ids and edge.get("target") in selected_ids
        ]
        meta = {
            "anchor_ids": sorted(anchor_ids, key=self._node_index),
            "backtrack_hops": max_backtrack_hops,
            "temporal_hops": max_temporal_hops,
        }
        return selected, selected_edges, meta

    def _node_index(self, node_id):
        for i, n in enumerate(self.nodes):
            if n["id"] == node_id:
                return i
        return 10**9

    def build_cognitive_edges_budgeted(self, max_pairs=30, prefilter_topk=3, similarity_threshold=0.06):
        sources = [n for n in self.nodes if n["type"] in {"Fact", "Action", "Error", "Planning"}]
        targets = [n for n in self.nodes if n["type"] in {"Planning", "Action", "Conclusion", "Fact"}]

        candidates = []
        for tgt in targets:
            tgt_idx = self._node_index(tgt["id"])
            scored = []
            for src in sources:
                src_idx = self._node_index(src["id"])
                if src_idx >= tgt_idx:
                    continue
                if src["id"] == tgt["id"]:
                    continue
                sim = self._lexical_similarity(src["content"], tgt["content"])
                if src["type"] == "Error":
                    sim += 0.10
                if src.get("span_id") and tgt.get("span_id") and src.get("span_id") == tgt.get("span_id"):
                    sim += 0.08
                if sim > 0:
                    scored.append((sim, src, tgt))

            scored.sort(key=lambda x: x[0], reverse=True)
            candidates.extend(scored[:prefilter_topk])

        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:max_pairs]

        for sim, src, tgt in candidates:
            heuristic_threshold = self._heuristic_link_threshold(src, tgt, similarity_threshold)
            review_floor = self._review_floor_threshold(src, tgt, heuristic_threshold)
            direct_link_threshold = self._direct_link_threshold(src, tgt, heuristic_threshold)

            if sim < review_floor:
                continue

            should_link = False
            method = "heuristic_fallback"
            llm_confidence = None
            review_band = "gray"

            if sim >= direct_link_threshold:
                should_link = True
                method = "heuristic_strong"
                review_band = "strong"
            elif self.enable_llm and self.client:
                should_link, llm_confidence = self._llm_dependency_judge(src["content"], tgt["content"])
                method = "llm_gray_zone"
                review_band = "gray"
            else:
                should_link = sim >= heuristic_threshold
                method = "heuristic_fallback"
                review_band = "gray_no_model"

            if should_link:
                self._add_edge(
                    src["id"],
                    tgt["id"],
                    "Cognitive",
                    score=round(sim, 4),
                    method=method,
                    extra={
                        "source_type": src["type"],
                        "target_type": tgt["type"],
                        "llm_confidence": llm_confidence,
                        "review_band": review_band,
                        "llm_backend": self.llm_backend if method.startswith("llm") else None,
                    },
                )

    def _heuristic_link_threshold(self, src, tgt, similarity_threshold):
        threshold = similarity_threshold
        if src["type"] == "Error" and tgt["type"] in {"Action", "Conclusion", "Planning"}:
            threshold = max(0.03, similarity_threshold - 0.02)
        return threshold

    def _review_floor_threshold(self, src, tgt, heuristic_threshold):
        margin = LAYERED_REVIEW_MARGIN_IN_CODE
        if src["type"] == "Error" and tgt["type"] in {"Action", "Conclusion", "Planning"}:
            margin = min(margin, 0.01)
        return max(0.0, heuristic_threshold - margin)

    def _direct_link_threshold(self, src, tgt, heuristic_threshold):
        margin = LAYERED_DIRECT_LINK_MARGIN_IN_CODE
        if src["type"] == "Error" and tgt["type"] in {"Action", "Conclusion", "Planning"}:
            margin = max(0.05, margin - 0.03)
        return heuristic_threshold + margin

    def _lexical_similarity(self, a, b):
        ta = self._tokenize(a)
        tb = self._tokenize(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / union if union else 0.0

    def _tokenize(self, text):
        tokens = re.findall(r"[a-zA-Z0-9_\-]{3,}", text.lower())
        stop = {
            "the", "and", "for", "with", "this", "that", "from", "into", "were", "been",
            "have", "has", "had", "you", "your", "not", "but", "are", "was", "all",
            "code", "tool", "task", "step", "final", "answer", "output", "input", "message",
        }
        return {t for t in tokens if t not in stop}

    def _llm_dependency_judge(self, fact_content, target_content):
        prompt = f"""
你是一个专门用于分析 Agent 日志逻辑依赖的专家。
请判断下面【Target】是否依赖【Fact】。
如果 Target 明显使用了 Fact 的信息、结论或观察，请回答 YES；如果相关性较弱但可能存在依赖，回答 WEAK；否则回答 NO。
只允许回答 YES、WEAK 或 NO 之一。

【Fact】:
{fact_content[:700]}

【Target】:
{target_content[:700]}
"""
        try:
            resp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
            )
            text = (resp.choices[0].message.content or "").strip().upper()
            if "YES" in text:
                return True, 1.0
            if "WEAK" in text:
                return True, 0.6
            return False, 0.0
        except Exception:
            return False, 0.0

    def summary(self):
        edge_type_count = defaultdict(int)
        for e in self.edges:
            edge_type_count[e["edge_type"]] += 1

        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "counts": {
                "Intent": sum(1 for n in self.nodes if n["type"] == "Intent"),
                "Planning": sum(1 for n in self.nodes if n["type"] == "Planning"),
                "Action": sum(1 for n in self.nodes if n["type"] == "Action"),
                "Fact": sum(1 for n in self.nodes if n["type"] == "Fact"),
                "Conclusion": sum(1 for n in self.nodes if n["type"] == "Conclusion"),
                "Error": sum(1 for n in self.nodes if n["type"] == "Error"),
            },
            "edge_types": dict(edge_type_count),
        }

    def estimate_prompt_tokens(self, nodes):
        chars = sum(len(n["content"]) for n in nodes)
        return max(1, chars // 4)

    def _content_flags_for_prompt_view(self, node):
        content = str(node.get("content", ""))
        text = content.lower()
        flags = []
        if re.search(r"<[^>\s]+>", content):
            flags.append("has_tag")
        if re.search(r"\b\d+(?:\.\d+)?\b", content):
            flags.append("has_number")
        if any(keyword in text for keyword in ["error", "exception", "failed", "timeout"]):
            flags.append("has_error_terms")
        if "final answer" in text or "final_answer(" in text:
            flags.append("has_final_answer")
        if node.get("type") == "Action":
            match = re.search(r"Tool:\s*([^|]+)", content)
            if match:
                flags.append(f"tool:{match.group(1).strip()[:80]}")
        return flags

    def _project_node_for_prompt_view(self, node, representation_mode="hybrid"):
        projected = {
            "id": node["id"],
            "type": node["type"],
            "span_id": node.get("span_id"),
            "saliency": node.get("saliency", 0.0),
        }
        for key in [
            "semantic_violation_type",
            "semantic_error_score",
            "semantic_violation",
            "semantic_observation",
        ]:
            if key in node:
                projected[key] = node[key]
        if representation_mode == "pure":
            projected.update(
                {
                    "content_chars": len(str(node.get("content", ""))),
                    "content_flags": self._content_flags_for_prompt_view(node),
                }
            )
        else:
            projected.update(
                {
                    "display": node.get("display", ""),
                    "content": node.get("content", ""),
                }
            )
        return projected

    def _project_edge_for_prompt_view(self, edge, node_map, representation_mode="hybrid"):
        projected = {
            "source": edge.get("source"),
            "target": edge.get("target"),
            "edge_type": edge.get("edge_type"),
            "score": edge.get("score"),
            "method": edge.get("method"),
        }
        if representation_mode == "pure":
            projected.update(
                {
                    "source_type": node_map.get(edge.get("source"), {}).get("type"),
                    "target_type": node_map.get(edge.get("target"), {}).get("type"),
                }
            )
        return projected

    def export_graph_json(self, output_path):
        payload = {
            "summary": self.summary(),
            "nodes": self.nodes,
            "edges": self.edges,
            "semantic_analysis": self.semantic_analysis,
            "source_metadata": self.trace_data.get("source_metadata", {}),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def export_subgraph_json(self, output_path, nodes, edges, view_name="pruned", extra_summary=None):
        payload = {
            "summary": self._summarize_view(nodes, edges),
            "view": view_name,
            "nodes": nodes,
            "edges": edges,
            "semantic_analysis": self.semantic_analysis,
        }
        if isinstance(extra_summary, dict):
            payload["summary"].update(extra_summary)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def export_prompt_pack(self, output_path, max_nodes=40, max_prompt_chars=18000, representation_mode="hybrid"):
        selected_nodes, selected_edges = self.select_salient_nodes(max_nodes=max_nodes, max_prompt_chars=max_prompt_chars)
        node_map = {node["id"]: node for node in selected_nodes}
        payload = {
            "goal": "Token-efficient, high-fidelity trace context for downstream LLM reasoning",
            "summary": {
                "representation_mode": representation_mode,
                "selected_nodes": len(selected_nodes),
                "selected_edges": len(selected_edges),
                "estimated_tokens": self.estimate_prompt_tokens(selected_nodes),
            },
            "nodes": [
                self._project_node_for_prompt_view(node, representation_mode=representation_mode)
                for node in selected_nodes
            ],
            "edges": [
                self._project_edge_for_prompt_view(edge, node_map, representation_mode=representation_mode)
                for edge in selected_edges
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def export_outcome_prompt_pack(
        self,
        output_path,
        max_nodes=40,
        max_prompt_chars=18000,
        representation_mode="hybrid",
        max_backtrack_hops=4,
        max_temporal_hops=2,
    ):
        selected_nodes, selected_edges, meta = self.select_outcome_relevant_subgraph(
            max_nodes=max_nodes,
            max_prompt_chars=max_prompt_chars,
            max_backtrack_hops=max_backtrack_hops,
            max_temporal_hops=max_temporal_hops,
        )
        node_map = {node["id"]: node for node in selected_nodes}
        payload = {
            "goal": "Outcome-oriented causal subgraph for downstream LLM reasoning",
            "summary": {
                "representation_mode": representation_mode,
                "view": "outcome_pruned",
                "selected_nodes": len(selected_nodes),
                "selected_edges": len(selected_edges),
                "estimated_tokens": self.estimate_prompt_tokens(selected_nodes),
                "anchor_ids": meta["anchor_ids"],
                "backtrack_hops": meta["backtrack_hops"],
                "temporal_hops": meta["temporal_hops"],
            },
            "nodes": [
                self._project_node_for_prompt_view(node, representation_mode=representation_mode)
                for node in selected_nodes
            ],
            "edges": [
                self._project_edge_for_prompt_view(edge, node_map, representation_mode=representation_mode)
                for edge in selected_edges
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def export_mermaid(self, output_path, use_salient=False, max_nodes=60, max_prompt_chars=24000):
        if use_salient:
            nodes, edges = self.select_salient_nodes(max_nodes=max_nodes, max_prompt_chars=max_prompt_chars)
        else:
            nodes, edges = self.nodes, self.edges

        lines = ["flowchart LR"]
        for node in nodes:
            label = f"{node['id']}|{node['type']}"
            lines.append(f'    {node["id"]}["{label}"]')

        for e in edges:
            if e["edge_type"] == "Cognitive":
                lines.append(f'    {e["source"]} -.-> {e["target"]}')
            elif e["edge_type"] == "Call":
                lines.append(f'    {e["source"]} ==> {e["target"]}')
            else:
                lines.append(f'    {e["source"]} --> {e["target"]}')

        lines.extend([
            "",
            "    classDef intent fill:#d1e9ff,stroke:#0d6efd,stroke-width:1px;",
            "    classDef planning fill:#fff3cd,stroke:#ff9800,stroke-width:1px;",
            "    classDef action fill:#e2f0d9,stroke:#2e7d32,stroke-width:1px;",
            "    classDef fact fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px;",
            "    classDef conclusion fill:#ffe0e0,stroke:#d32f2f,stroke-width:2px;",
            "    classDef error fill:#ffd7d7,stroke:#b71c1c,stroke-width:2px;",
        ])

        for node in nodes:
            type_class = {
                "Intent": "intent",
                "Planning": "planning",
                "Action": "action",
                "Fact": "fact",
                "Conclusion": "conclusion",
                "Error": "error",
            }.get(node["type"], "fact")
            lines.append(f'    class {node["id"]} {type_class};')

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def export_outcome_mermaid(
        self,
        output_path,
        max_nodes=40,
        max_prompt_chars=18000,
        max_backtrack_hops=4,
        max_temporal_hops=2,
    ):
        nodes, edges, _ = self.select_outcome_relevant_subgraph(
            max_nodes=max_nodes,
            max_prompt_chars=max_prompt_chars,
            max_backtrack_hops=max_backtrack_hops,
            max_temporal_hops=max_temporal_hops,
        )

        lines = ["flowchart LR"]
        for node in nodes:
            label = f"{node['id']}|{node['type']}"
            lines.append(f'    {node["id"]}["{label}"]')

        for edge in edges:
            if edge["edge_type"] == "Cognitive":
                lines.append(f'    {edge["source"]} -.-> {edge["target"]}')
            elif edge["edge_type"] == "Call":
                lines.append(f'    {edge["source"]} ==> {edge["target"]}')
            else:
                lines.append(f'    {edge["source"]} --> {edge["target"]}')

        lines.extend([
            "",
            "    classDef intent fill:#d1e9ff,stroke:#0d6efd,stroke-width:1px;",
            "    classDef planning fill:#fff3cd,stroke:#ff9800,stroke-width:1px;",
            "    classDef action fill:#e2f0d9,stroke:#2e7d32,stroke-width:1px;",
            "    classDef fact fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px;",
            "    classDef conclusion fill:#ffe0e0,stroke:#d32f2f,stroke-width:2px;",
            "    classDef error fill:#ffd7d7,stroke:#b71c1c,stroke-width:2px;",
        ])

        for node in nodes:
            type_class = {
                "Intent": "intent",
                "Planning": "planning",
                "Action": "action",
                "Fact": "fact",
                "Conclusion": "conclusion",
                "Error": "error",
            }.get(node["type"], "fact")
            lines.append(f'    class {node["id"]} {type_class};')

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def export_markdown_report(self, output_path, max_salient_nodes=40, max_prompt_chars=18000):
        s = self.summary()
        salient_nodes, salient_edges = self.select_salient_nodes(
            max_nodes=max_salient_nodes,
            max_prompt_chars=max_prompt_chars,
        )
        pruned_nodes, pruned_edges, pruned_meta = self.select_outcome_relevant_subgraph(
            max_nodes=max_salient_nodes,
            max_prompt_chars=max_prompt_chars,
            max_backtrack_hops=PRUNED_MAX_BACKTRACK_HOPS_IN_CODE,
            max_temporal_hops=PRUNED_MAX_TEMPORAL_HOPS_IN_CODE,
        )
        est_tokens = self.estimate_prompt_tokens(salient_nodes)
        pruned_tokens = self.estimate_prompt_tokens(pruned_nodes)
        node_ratio = (len(pruned_nodes) / len(self.nodes)) if self.nodes else 0.0
        edge_ratio = (len(pruned_edges) / len(self.edges)) if self.edges else 0.0

        lines = [
            "# OTel Trace 优化解析报告",
            "",
            "## 全量图汇总",
            f"- 节点总数: {s['node_count']}",
            f"- 边总数: {s['edge_count']}",
            f"- 节点类型统计: {s['counts']}",
            f"- 边类型统计: {s['edge_types']}",
            "",
            "## Token 优化视图（给 LLM 的压缩上下文）",
            f"- 选中节点数: {len(salient_nodes)}",
            f"- 选中边数: {len(salient_edges)}",
            f"- 预估 token: ~{est_tokens}",
            "",
            "## 面向最终结果的剪枝子图",
            f"- 锚点节点: {pruned_meta['anchor_ids']}",
            f"- 反向追溯跳数: {pruned_meta['backtrack_hops']}",
            f"- Temporal 补桥跳数: {pruned_meta['temporal_hops']}",
            f"- 剪枝后节点数: {len(pruned_nodes)} (占全量 {node_ratio:.2%})",
            f"- 剪枝后边数: {len(pruned_edges)} (占全量 {edge_ratio:.2%})",
            f"- 剪枝后预估 token: ~{pruned_tokens}",
            "",
            "## 高显著节点预览",
        ]

        for node in salient_nodes[:30]:
            lines.append(f"- [{node['id']}] {node['type']} (saliency={node['saliency']}): {node['display']}")

        lines.extend([
            "",
            "## 剪枝后关键节点预览",
        ])

        for node in pruned_nodes[:30]:
            lines.append(f"- [{node['id']}] {node['type']} (saliency={node['saliency']}): {node['display']}")

        lines.extend([
            "",
            "## 论文式设计要点（本实现对应）",
            "- 结构先行：先用 span 树与规则抽取构建基础图，再引入 LLM（降低无效调用）。",
            "- 分层判边：高分候选直接连边，低分候选直接丢弃，仅灰区候选送模型判别。",
            "- 预算化推理：仅对高相关候选对做认知边判断（max_pairs 约束）。",
            "- 分层压缩：全量图用于追溯，salient 子图用于下游 LLM 提示，兼顾保真与成本。",
            "- 目标导向剪枝：从最终结论和终局错误反向回溯，只保留 outcome-relevant 子图。",
            "- 去重与字段投影：减少重复 Observation / 输出冗余，控制上下文长度。",
        ])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def _safe_json_parse(text):
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _run_external_pruning_for_graph(graph_path, sample_out_dir):
    """将剪枝统一委托到 pruning_strategies.py，避免主流程内双重剪枝。"""
    if not USE_EXTERNAL_PRUNING_MODULE_IN_CODE:
        return {"enabled": False, "reason": "disabled_by_config"}

    if (
        ExternalPruningTraceGraph is None
        or run_external_pruning_strategies is None
        or save_external_prune_result is None
    ):
        return {"enabled": False, "reason": "pruning_module_not_available"}

    try:
        trace = ExternalPruningTraceGraph.from_json(Path(graph_path))
        results = run_external_pruning_strategies(trace, char_budget=MAX_PROMPT_CHARS_IN_CODE)

        out_dir = Path(sample_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        output_files = {}
        for strategy_name, result in results.items():
            out_file = out_dir / f"{strategy_name}.json"
            save_external_prune_result(out_file, result, trace)
            output_files[strategy_name] = str(out_file)

        return {
            "enabled": True,
            "strategy_count": len(output_files),
            "output_dir": str(out_dir),
            "output_files": output_files,
        }
    except Exception as exc:
        return {"enabled": False, "reason": f"external_pruning_failed: {exc}"}


def _extract_trace_from_row(row):
    priority_keys = [
        "trace", "trace_json", "otel_trace", "gaia_trace", "raw_trace", "trace_content", "output"
    ]
    for key in priority_keys:
        if key in row:
            candidate = find_first_trace_obj(row[key])
            if candidate is not None:
                return candidate

    if "spans" in row and isinstance(row["spans"], list):
        trace = {"spans": row["spans"]}
        if "trace_id" in row:
            trace["trace_id"] = row["trace_id"]
        return trace

    return find_first_trace_obj(row)


def _extract_annotation_from_row(row):
    annotation_keys = [
        "processed_annotations_gaia",
        "processed_annotations_swe_bench",
        "processed_annotation",
        "annotations",
        "annotation",
        "error",
        "errors",
    ]
    for key in annotation_keys:
        if key in row:
            return row[key]
    return None


def _extract_sample_id(row, fallback_idx):
    for key in ["trace_id", "task_id", "id", "example_id", "uuid"]:
        if key in row and row[key] not in [None, ""]:
            return str(row[key])
    return f"sample_{fallback_idx:05d}"


def _sanitize_filename(text):
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", text)


def _extract_final_answer(nodes):
    for node in reversed(nodes):
        if node["type"] != "Conclusion":
            continue
        content = node["content"]
        m1 = re.search(r"FINAL ANSWER:\s*(.+)", content, flags=re.IGNORECASE)
        if m1:
            return m1.group(1).strip()
        m2 = re.search(r'final_answer\(["\'](.+?)["\']\)', content, flags=re.IGNORECASE)
        if m2:
            return m2.group(1).strip()
    return None


def _extract_true_answer_from_nodes(nodes):
    for node in nodes:
        if node["type"] != "Fact":
            continue
        m = re.search(r'"true_answer"\s*:\s*"([^"]+)"', node["content"])
        if m:
            return m.group(1).strip()
    return None


def _diagnose_graph(nodes):
    plan_text = "\n".join([n["content"] for n in nodes if n["type"] == "Planning"]).lower()
    action_text = "\n".join([n["content"] for n in nodes if n["type"] == "Action"]).lower()

    plan_mentions_search = any(k in plan_text for k in ["search_agent", "usgs", "web", "browse"])
    action_has_search = any(k in action_text for k in ["search_agent", "web_search", "search", "inspect_file_as_text"])
    plan_action_gap = bool(plan_mentions_search and not action_has_search)

    final_answer = _extract_final_answer(nodes)
    true_answer = _extract_true_answer_from_nodes(nodes)
    answer_mismatch = bool(final_answer and true_answer and final_answer != true_answer)

    return {
        "plan_action_gap": plan_action_gap,
        "final_answer": final_answer,
        "true_answer": true_answer,
        "answer_mismatch": answer_mismatch,
    }


def _iter_dataset_rows(ds_obj):
    if hasattr(ds_obj, "items"):
        for split_name, split_ds in ds_obj.items():
            for idx, row in enumerate(split_ds):
                yield split_name, idx, row
    else:
        for idx, row in enumerate(ds_obj):
            yield "default", idx, row


def run_dataset_batch(args):
    try:
        from datasets import load_dataset as hf_load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets 不可用，请在当前运行环境安装后重试。"
            "建议执行: conda run -n agent_eval pip install datasets\n"
            f"原始错误: {exc}"
        )

    if args.dataset_subset:
        ds = hf_load_dataset(args.dataset_name, args.dataset_subset)
    else:
        ds = hf_load_dataset(args.dataset_name)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{args.out_prefix}_batch"
    run_dir.mkdir(parents=True, exist_ok=True)

    llm_enabled_for_batch = USE_LLM_IN_CODE and ALLOW_LLM_IN_BATCH_IN_CODE
    llm_runtime = _get_default_llm_runtime()
    if USE_LLM_IN_CODE and not ALLOW_LLM_IN_BATCH_IN_CODE:
        print("[提示] 当前为代码内配置：批处理默认禁用 LLM 依赖边（可改 ALLOW_LLM_IN_BATCH_IN_CODE）。")

    work_items = []
    for split_name, idx, row in _iter_dataset_rows(ds):
        if args.dataset_split and split_name != args.dataset_split:
            continue
        work_items.append((split_name, idx, row))
        if args.max_samples and len(work_items) >= args.max_samples:
            break

    summary_rows = []
    processed = 0
    skipped = 0

    def _process_dataset_item(item):
        split_name, idx, row = item

        trace_obj = _extract_trace_from_row(row)
        trace_obj = normalize_trace_for_parser(trace_obj)
        if trace_obj is None:
            return {"skipped": True, "row_summary": None}

        sample_id = _sanitize_filename(_extract_sample_id(row, idx))
        sample_prefix = f"{split_name}_{sample_id}"

        parser = OTelTraceParser(
            trace_obj,
            enable_llm=llm_enabled_for_batch,
            **llm_runtime,
        )
        nodes, _ = parser.parse()
        parser.build_cognitive_edges_budgeted(
            max_pairs=MAX_PAIRS_IN_CODE,
            prefilter_topk=PREFILTER_TOPK_IN_CODE,
            similarity_threshold=SIMILARITY_THRESHOLD_IN_CODE,
        )

        graph_path = run_dir / f"{sample_prefix}_graph.json"
        mermaid_path = run_dir / f"{sample_prefix}_graph.mmd"
        pruning_dir = run_dir / f"{sample_prefix}_pruning"

        parser.export_graph_json(graph_path)
        parser.export_mermaid(mermaid_path, use_salient=False)
        pruning_result = _run_external_pruning_for_graph(graph_path, pruning_dir)

        diag = _diagnose_graph(nodes)
        ann = _extract_annotation_from_row(row)
        row_summary = {
            "split": split_name,
            "sample_id": sample_id,
            "node_count": len(parser.nodes),
            "edge_count": len(parser.edges),
            "counts": parser.summary()["counts"],
            "edge_types": parser.summary()["edge_types"],
            "diagnosis": diag,
            "annotation": ann,
            "external_pruning": pruning_result,
            "output_files": {
                "graph": str(graph_path),
                "mermaid": str(mermaid_path),
                "pruning_dir": str(pruning_dir),
            },
        }
        return {"skipped": False, "row_summary": row_summary}

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_process_dataset_item, item) for item in work_items]
            for i, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                if result["skipped"]:
                    skipped += 1
                else:
                    summary_rows.append(result["row_summary"])
                    processed += 1
                if i % 20 == 0:
                    print(f"已处理 {i}/{len(work_items)} 条 trace...")
    else:
        for i, item in enumerate(work_items, start=1):
            result = _process_dataset_item(item)
            if result["skipped"]:
                skipped += 1
            else:
                summary_rows.append(result["row_summary"])
                processed += 1
            if i % 20 == 0:
                print(f"已处理 {i}/{len(work_items)} 条 trace...")

    summary_file = run_dir / "batch_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset_name,
                "subset": args.dataset_subset,
                "split_filter": args.dataset_split,
                "processed": processed,
                "skipped": skipped,
                "llm_enabled_for_batch": llm_enabled_for_batch,
                "rows": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n批处理完成。")
    print(f"- 处理条数: {processed}")
    print(f"- 跳过条数: {skipped}")
    print(f"- 汇总文件: {summary_file}")
    print(f"- 样本输出目录: {run_dir}")


def _load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def run_local_batch(args):
    trace_dir = Path(args.local_trace_dir or args.local_gaia_dir)
    ann_dir = Path(args.local_annotations_dir) if args.local_annotations_dir else None

    if not trace_dir.exists() or not trace_dir.is_dir():
        raise RuntimeError(f"Local trace directory not found: {trace_dir}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"{args.out_prefix}_local_batch"
    run_dir.mkdir(parents=True, exist_ok=True)

    llm_enabled_for_batch = USE_LLM_IN_CODE and ALLOW_LLM_IN_BATCH_IN_CODE
    llm_runtime = _get_default_llm_runtime()
    if USE_LLM_IN_CODE and not ALLOW_LLM_IN_BATCH_IN_CODE:
        print("[提示] 当前为代码内配置：批处理默认禁用 LLM 依赖边（可改 ALLOW_LLM_IN_BATCH_IN_CODE）。")

    ann_map = {}
    if ann_dir and ann_dir.exists() and ann_dir.is_dir():
        for ann_path in ann_dir.glob("*.json"):
            ann_map[ann_path.stem] = _load_json_file(ann_path)

    trace_files = sorted(trace_dir.glob("*.json"))
    if args.max_samples:
        trace_files = trace_files[: args.max_samples]

    summary_rows = []
    processed = 0
    skipped = 0

    def _process_local_trace(trace_path):
        raw_trace = _load_json_file(trace_path)
        trace_obj = normalize_trace_for_parser(raw_trace)
        if trace_obj is None:
            return {"skipped": True, "row_summary": None}

        sample_id = _sanitize_filename(trace_path.stem)
        sample_prefix = f"local_{sample_id}"

        # 提前定义 graph_path，用于判断文件是否已经存在
        graph_path = run_dir / f"{sample_prefix}_graph.json"

        # 断点续传逻辑
        if graph_path.exists():
            print(f"⏩ 发现已处理文件，跳过: {sample_prefix}")
            return {"skipped": True, "row_summary": None}


        parser = OTelTraceParser(
            trace_obj,
            enable_llm=llm_enabled_for_batch,
            **llm_runtime,
        )
        nodes, _ = parser.parse()
        parser.build_cognitive_edges_budgeted(
            max_pairs=MAX_PAIRS_IN_CODE,
            prefilter_topk=PREFILTER_TOPK_IN_CODE,
            similarity_threshold=SIMILARITY_THRESHOLD_IN_CODE,
        )

        graph_path = run_dir / f"{sample_prefix}_graph.json"
        mermaid_path = run_dir / f"{sample_prefix}_graph.mmd"
        pruning_dir = run_dir / f"{sample_prefix}_pruning"

        parser.export_graph_json(graph_path)
        parser.export_mermaid(mermaid_path, use_salient=False)
        pruning_result = _run_external_pruning_for_graph(graph_path, pruning_dir)

        diag = _diagnose_graph(nodes)
        ann = ann_map.get(trace_path.stem)

        row_summary = {
            "source": "local",
            "sample_id": sample_id,
            "trace_file": str(trace_path),
            "annotation_file": str((ann_dir / f"{trace_path.stem}.json")) if ann_dir else None,
            "node_count": len(parser.nodes),
            "edge_count": len(parser.edges),
            "counts": parser.summary()["counts"],
            "edge_types": parser.summary()["edge_types"],
            "diagnosis": diag,
            "annotation": ann,
            "external_pruning": pruning_result,
            "output_files": {
                "graph": str(graph_path),
                "mermaid": str(mermaid_path),
                "pruning_dir": str(pruning_dir),
            },
        }
        return {"skipped": False, "row_summary": row_summary}

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_process_local_trace, trace_path) for trace_path in trace_files]
            for i, fut in enumerate(as_completed(futures), start=1):
                result = fut.result()
                if result["skipped"]:
                    skipped += 1
                else:
                    summary_rows.append(result["row_summary"])
                    processed += 1
                if i % 20 == 0:
                    print(f"已处理 {i}/{len(trace_files)} 条本地 trace...")
    else:
        for i, trace_path in enumerate(trace_files, start=1):
            result = _process_local_trace(trace_path)
            if result["skipped"]:
                skipped += 1
            else:
                summary_rows.append(result["row_summary"])
                processed += 1
            if i % 20 == 0:
                print(f"已处理 {i}/{len(trace_files)} 条本地 trace...")

    summary_file = run_dir / "batch_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "local",
                "trace_dir": str(trace_dir),
                "annotation_dir": str(ann_dir) if ann_dir else None,
                "processed": processed,
                "skipped": skipped,
                "llm_enabled_for_batch": llm_enabled_for_batch,
                "rows": summary_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n本地批处理完成。")
    print(f"- 处理条数: {processed}")
    print(f"- 跳过条数: {skipped}")
    print(f"- 汇总文件: {summary_file}")
    print(f"- 样本输出目录: {run_dir}")

def main():
    cli = argparse.ArgumentParser(description="Parse OTel-style trace with token-efficient graph extraction.")
    cli.add_argument("--input", default="trace.json", help="Input trace json file")
    cli.add_argument("--out-prefix", default="otl", help="Output file prefix")
    cli.add_argument("--output-dir", default="output_graphs", help="Output directory (single or batch mode)")
    cli.add_argument("--dataset-mode", action="store_true", help="Batch mode: load TRAIL dataset and process traces automatically")
    cli.add_argument("--local-mode", action="store_true", help="Batch mode: process local trace/annotations folders without HF online loading")
    cli.add_argument("--local-trace-dir", default=None, help="Local trace folder path")
    cli.add_argument("--local-gaia-dir", default="gaia/trail_data/GAIA", help=argparse.SUPPRESS)
    cli.add_argument("--local-annotations-dir", default="gaia/trail_data/processed_annotations_gaia", help="Local annotations folder path")
    cli.add_argument("--dataset-name", default="PatronusAI/TRAIL", help="Hugging Face dataset name")
    cli.add_argument("--dataset-subset", default=None, help="Optional dataset config/subset name")
    cli.add_argument("--dataset-split", default=None, help="Optional split filter (e.g., train/validation/test)")
    cli.add_argument("--max-samples", type=int, default=0, help="Max number of samples in batch mode (0 means all)")
    cli.add_argument("--workers", type=int, default=8, help="并行 worker 数，仅批处理模式有效")
    args = cli.parse_args()

    if args.local_mode:
        run_local_batch(args)
        return

    if args.dataset_mode:
        run_dataset_batch(args)
        return

    input_path = Path(args.input)
    out_prefix = args.out_prefix

    try:
        with open(input_path, "r", encoding="utf-8-sig") as f:
            trace_data = json.load(f)
        trace_data = normalize_trace_for_parser(trace_data)
        if trace_data is None:
            raise RuntimeError("输入文件中未识别到可解析的 trace 结构（支持 GAIA spans / OTLP resourceSpans / history 对话）。")

        llm_runtime = _get_default_llm_runtime()

        parser = OTelTraceParser(
            trace_data,
            enable_llm=USE_LLM_IN_CODE,
            **llm_runtime,
        )
        nodes, edges = parser.parse()

        parser.build_cognitive_edges_budgeted(
            max_pairs=MAX_PAIRS_IN_CODE,
            prefilter_topk=PREFILTER_TOPK_IN_CODE,
            similarity_threshold=SIMILARITY_THRESHOLD_IN_CODE,
        )

        summary = parser.summary()

        print(f"LLM辅助开关(USE_LLM_IN_CODE): {USE_LLM_IN_CODE}")
        print(f"LLM后端(DEFAULT_LLM_BACKEND): {DEFAULT_LLM_BACKEND}")
        print(f"LLM模型: {llm_runtime['llm_model']}")
        print(f"成功提取节点 {len(nodes)} 个，边 {len(parser.edges)} 条。")
        print("节点类型统计:", summary["counts"])
        print("边类型统计:", summary["edge_types"])
        print(f"全图预估token: ~{parser.estimate_prompt_tokens(nodes)}")

        for node in nodes[:40]:
            print(f"[{node['id']}] {node['type'].ljust(10)} | s={node['saliency']:.2f} | {node['display']}")

        output_root = Path(args.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        json_out = output_root / f"{out_prefix}_graph.json"
        mmd_out = output_root / f"{out_prefix}_graph.mmd"
        pruning_out_dir = output_root / f"{out_prefix}_pruning"

        parser.export_graph_json(json_out)
        parser.export_mermaid(mmd_out, use_salient=False)
        pruning_result = _run_external_pruning_for_graph(json_out, pruning_out_dir)

        print("\n已导出文件:")
        print(f"- {json_out}")
        print(f"- {mmd_out}")
        print(f"- 剪枝输出: {pruning_result}")

    except FileNotFoundError:
        print(f"未找到文件，请检查路径: {input_path}")


if __name__ == "__main__":
    main()
