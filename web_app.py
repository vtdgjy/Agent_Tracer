import json
import sys
import os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

sys.path.insert(0, str(Path(__file__).parent))
from trace_input_adapter import normalize_trace_for_parser
from OTL_trace import OTelTraceParser
from deepseek_judge_trace import build_graph_digest, judge_one_source, normalize_judge_result
from multilayer_monitor import build_live_system_snapshot, build_multilayer_monitoring
from monitoring_store import MONITORING_LAYERS, monitoring_event_store

LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

app = Flask(__name__, static_folder="web_static")

NODE_COLORS = {
    "Intent": "#4e79a7", "Planning": "#f28e2b", "Action": "#59a14f",
    "Fact": "#76b7b2", "Conclusion": "#edc948", "Error": "#e15759",
}
EDGE_COLORS = {
    "Temporal": "#70777d", "Call": "#59a14f", "Observation": "#4e79a7",
    "Retry": "#e15759", "Cognitive": "#b07aa1",
    "AgentSpawn": "#f59e0b", "SubagentReturn": "#8b5cf6",
    "SemanticOrigin": "#f28e2b", "SemanticValidation": "#ff6b81",
    "SemanticPropagation": "#2dd4a8", "SemanticRejection": "#ef4444",
    "EvidenceExposure": "#38bdf8",
}

# span_name pattern -> component metadata
COMPONENT_MAP = {
    "LiteLLMModel.__call__": {
        "component": "LiteLLMModel",
        "file": "smolagents/models.py",
        "layer": "LLM调用层",
        "fixable_aspects": ["输出格式约束", "prompt模板", "temperature/参数", "输出解析逻辑"],
    },
    "CodeAgent.run": {
        "component": "CodeAgent",
        "file": "smolagents/agents.py",
        "layer": "Agent主循环",
        "fixable_aspects": ["系统prompt", "最大步数限制", "停止条件", "任务分解策略"],
    },
    "ToolCallingAgent.run": {
        "component": "ToolCallingAgent",
        "file": "smolagents/agents.py",
        "layer": "工具调用Agent",
        "fixable_aspects": ["工具选择策略", "工具调用格式", "错误恢复逻辑"],
    },
    "FinalAnswerTool": {
        "component": "FinalAnswerTool",
        "file": "smolagents/default_tools.py",
        "layer": "答案收敛组件",
        "fixable_aspects": ["输出格式校验", "答案提取逻辑", "格式约束强制"],
    },
}

def _resolve_component(span_name: str) -> dict | None:
    if not span_name:
        return None
    # exact match
    if span_name in COMPONENT_MAP:
        return COMPONENT_MAP[span_name]
    # suffix match for *Tool
    if span_name.endswith("Tool"):
        return {
            "component": span_name,
            "file": "smolagents/tools/",
            "layer": "工具组件",
            "fixable_aspects": ["参数校验", "返回值格式", "错误处理", "超时处理"],
        }
    # Step N
    if span_name.startswith("Step "):
        return {
            "component": "AgentStep",
            "file": "smolagents/agents.py",
            "layer": "单步执行",
            "fixable_aspects": ["步骤重试逻辑", "错误恢复", "中间结果验证"],
        }
    return None


@app.route("/")
def index():
    return send_from_directory("web_static", "index.html")


@app.route("/api/monitor/snapshot")
def monitor_snapshot():
    """Return a live four-layer baseline before a trace is uploaded."""
    return jsonify(build_live_system_snapshot())


@app.route("/api/v1/monitor/events:batch", methods=["POST"])
def ingest_monitoring_events():
    """Ingest monitoring events independently from the trace upload path."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("events"), list):
        return jsonify({"error": "body.events must be a JSON array"}), 400
    if len(body["events"]) > 1_000:
        return jsonify({"error": "a batch may contain at most 1000 events"}), 413

    accepted, rejected = monitoring_event_store.append_batch(body["events"])
    status = 202 if accepted else 400
    return jsonify({
        "accepted": len(accepted),
        "rejected": len(rejected),
        "event_ids": [event["event_id"] for event in accepted],
        "errors": rejected,
    }), status


@app.route("/api/v1/monitor/systems/<system_id>/events")
def query_monitoring_events(system_id: str):
    layer = request.args.get("layer")
    if layer and layer not in MONITORING_LAYERS:
        return jsonify({"error": f"unknown layer: {layer}"}), 400
    try:
        events = monitoring_event_store.query(
            system_id,
            layer=layer,
            start=request.args.get("start"),
            end=request.args.get("end"),
            limit=int(request.args.get("limit", "500")),
        )
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"system_id": system_id, "count": len(events), "events": events})


@app.route("/api/v1/monitor/systems/<system_id>/snapshot")
def query_monitoring_snapshot(system_id: str):
    return jsonify(monitoring_event_store.snapshot(system_id))


def _build_span_index(normalized):
    index = {}
    def walk(spans):
        for s in (spans or []):
            sid = s.get("span_id")
            if sid:
                attrs = s.get("span_attributes", {})
                index[sid] = {
                    "span_name": s.get("span_name", ""),
                    "status_code": s.get("status_code", ""),
                    "timestamp": s.get("timestamp", ""),
                    "duration": s.get("duration", ""),
                    "input": (attrs.get("input.value") or attrs.get("llm.input_messages.0.message.content") or "")[:600],
                    "output": (attrs.get("output.value") or attrs.get("llm.output_messages.0.message.content") or "")[:600],
                    "tool_name": attrs.get("tool.name", ""),
                }
            walk(s.get("child_spans") or [])
    walk(normalized.get("spans", []))
    return index


def _build_span_tree(normalized):
    """Return spans as a nested tree for raw trace visualization."""
    def serialize(span, depth=0):
        attrs = span.get("span_attributes", {})
        node = {
            "span_id": span.get("span_id", ""),
            "span_name": span.get("span_name", ""),
            "status_code": span.get("status_code", ""),
            "timestamp": span.get("timestamp", ""),
            "duration": span.get("duration", ""),
            "tool_name": attrs.get("tool.name", ""),
            "input": (attrs.get("input.value") or attrs.get("llm.input_messages.0.message.content") or "")[:300],
            "output": (attrs.get("output.value") or attrs.get("llm.output_messages.0.message.content") or "")[:300],
            "component": _resolve_component(span.get("span_name", "")),
            "children": [serialize(c) for c in (span.get("child_spans") or [])],
        }
        return node
    return [serialize(s) for s in normalized.get("spans", [])]


def _parse_trace_payload(raw, run_judge=False):
    """Build the visualization and optional diagnosis payload for one trace."""
    if not isinstance(raw, dict):
        raise ValueError("trace JSON 须为对象")
    try:
        normalized = normalize_trace_for_parser(raw)
        if normalized is None:
            raise RuntimeError("输入文件中未识别到可解析的 trace 结构")
        parser = OTelTraceParser(normalized, enable_llm=False)
        nodes, edges = parser.parse()
        parser.build_cognitive_edges_budgeted()
    except Exception as e:
        raise RuntimeError(f"Graph build failed: {e}") from e

    vis_nodes = [
        {"id": n["id"], "label": n["id"], "title": (n.get("display") or n.get("content", ""))[:300],
         "group": n["type"], "color": NODE_COLORS.get(n["type"], "#ccc"),
         "saliency": n.get("saliency", 1.0), "span_id": n.get("span_id", ""),
         "parent_span_id": n.get("parent_span_id", ""),
         "agent_id": n.get("agent_id"), "agent_label": n.get("agent_label"),
         "agent_role": n.get("agent_role"), "agent_depth": n.get("agent_depth", 0),
         "parent_agent_id": n.get("parent_agent_id"), "spawn_span_id": n.get("spawn_span_id"),
         "agent_task": n.get("agent_task"), "agent_path": n.get("agent_path", [])}
        for n in nodes
    ]
    vis_edges = [
        {"from": e["source"], "to": e["target"], "label": e["edge_type"],
         "color": EDGE_COLORS.get(e["edge_type"], "#aaa"),
         "score": e.get("score"), "method": e.get("method"),
         "propagation_state": e.get("propagation_state"),
         "tool_call_id": e.get("tool_call_id")}
        for e in edges
    ]
    summary = {"node_count": len(nodes), "edge_count": len(edges), "node_types": {}, "edge_types": {}}
    for n in nodes:
        summary["node_types"][n["type"]] = summary["node_types"].get(n["type"], 0) + 1
    for e in edges:
        summary["edge_types"][e["edge_type"]] = summary["edge_types"].get(e["edge_type"], 0) + 1

    agents = []
    for scope in parser.agent_scopes.values():
        agent_id = scope.get("agent_id")
        agent_nodes = [node for node in nodes if node.get("agent_id") == agent_id]
        agent_errors = [node for node in agent_nodes if node.get("type") == "Error"]
        agents.append({
            **scope,
            "node_count": len(agent_nodes),
            "error_count": len(agent_errors),
        })
    agents.sort(key=lambda item: (item.get("agent_depth", 0), item.get("agent_label", "")))

    judge_result = None
    judge_error = None
    if run_judge:
        if not LLM_API_KEY:
            judge_error = "未配置 LLM_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY，已跳过 LLM 诊断"
        else:
            try:
                graph_obj = {
                    "nodes": nodes,
                    "edges": edges,
                    "summary": summary,
                    "meta": {},
                    "semantic_analysis": parser.semantic_analysis,
                }
                digest = build_graph_digest(graph_obj)
                trace_id = str(normalized.get("trace_id") or raw.get("trace_id") or "unknown")
                result, _ = judge_one_source(
                    api_key=LLM_API_KEY, base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL,
                    source_type="trace_graph", trace_id=trace_id, digest=digest,
                    max_input_chars=28000, enable_review=False,
                    judge_max_tokens=0, judge_review_max_tokens=0,
                    json_repair_max_tokens=0, review_first_pass_max_chars=0,
                    prompt_variant="swe",
                )
                node_to_span = {
                    str(node.get("id")): str(node.get("span_id"))
                    for node in nodes
                    if node.get("id") and node.get("span_id")
                }
                result = normalize_judge_result(result, trace_id, "trace_graph", node_to_span=node_to_span)
                judge_result = result
            except Exception as e:
                judge_error = str(e)

    span_index = _build_span_index(normalized)
    # attach component info to each span
    for sid, sp in span_index.items():
        comp = _resolve_component(sp["span_name"])
        sp["component"] = comp

    multilayer_monitoring = build_multilayer_monitoring(
        normalized,
        nodes,
        edges,
        agents,
    )

    return {
        "nodes": vis_nodes, "edges": vis_edges, "summary": summary,
        "agents": agents,
        "span_index": span_index, "span_tree": _build_span_tree(normalized),
        "multilayer_monitoring": multilayer_monitoring,
        "judge": judge_result, "judge_error": judge_error,
        "run_judge": run_judge,
        "trace_id": str(normalized.get("trace_id") or raw.get("trace_id") or "unknown"),
    }


@app.route("/api/parse", methods=["POST"])
def parse_trace():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    run_judge = str(request.form.get("run_judge", "")).lower() in {"1", "true", "yes", "on"}
    try:
        raw = json.loads(f.read().decode("utf-8"))
        return jsonify(_parse_trace_payload(raw, run_judge))
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Invalid JSON: {exc}"}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/parse-batch", methods=["POST"])
def parse_trace_batch():
    """Parse multiple trace files and return independent results for each file."""
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400
    if len(files) > 50:
        return jsonify({"error": "一次最多上传 50 个 trace 文件"}), 413

    run_judge = str(request.form.get("run_judge", "")).lower() in {"1", "true", "yes", "on"}
    results = []
    for index, uploaded in enumerate(files):
        name = uploaded.filename or f"trace-{index + 1}.json"
        item = {"index": index, "filename": name}
        try:
            raw = json.loads(uploaded.read().decode("utf-8"))
            item["result"] = _parse_trace_payload(raw, run_judge)
            item["ok"] = True
            item["trace_id"] = item["result"].get("trace_id")
        except json.JSONDecodeError as exc:
            item["ok"] = False
            item["error"] = f"Invalid JSON: {exc}"
        except Exception as exc:
            item["ok"] = False
            item["error"] = str(exc)
        results.append(item)

    succeeded = [item for item in results if item["ok"]]
    diagnosed = sum(1 for item in succeeded if item["result"].get("judge"))
    return jsonify({
        "results": results,
        "summary": {
            "total": len(results),
            "succeeded": len(succeeded),
            "failed": len(results) - len(succeeded),
            "diagnosed": diagnosed,
            "run_judge": run_judge,
        },
    })


@app.route("/api/fix", methods=["POST"])
def suggest_fix():
    """Given span_id + root_cause context, return LLM fix suggestion."""
    body = request.get_json()
    span_id = body.get("span_id", "")
    span_info = body.get("span", {})
    root_cause = body.get("root_cause", "")
    fault_types = body.get("fault_types", [])
    component = span_info.get("component") or {}

    comp_desc = ""
    if component:
        comp_desc = (f"组件: {component.get('component')} ({component.get('file')})\n"
                     f"层级: {component.get('layer')}\n"
                     f"可修复点: {', '.join(component.get('fixable_aspects', []))}")

    prompt = f"""你是一名 AI Agent 系统工程师，正在分析一个 smolagents 框架的 Agent 执行 trace。

## 根因信息
{root_cause}

## 故障分类
{', '.join(fault_types) if fault_types else '未知'}

## 定位到的 Span
- span_name: {span_info.get('span_name', '')}
- status: {span_info.get('status_code', '')}
- input (截断): {str(span_info.get('input', ''))[:400]}
- output (截断): {str(span_info.get('output', ''))[:400]}

## 对应 Harness 组件
{comp_desc if comp_desc else '未能映射到已知组件'}

## 任务
请给出针对该组件的**具体修复建议**，包含：
1. 问题根源（一句话）
2. 修复方向（2-3条可操作的建议）
3. 伪代码或修改示例（如果适用）

请用中文回答，简洁、可操作。"""

    try:
        client = OpenAI(api_key=LLM_API_KEY, base_url=DEFAULT_BASE_URL)
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        suggestion = resp.choices[0].message.content
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "component": component,
        "suggestion": suggestion,
    })


if __name__ == "__main__":
    os.makedirs("web_static", exist_ok=True)
    app.run(debug=True, port=5000)
