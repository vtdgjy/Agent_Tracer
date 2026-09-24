import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from diagnostic_report import export_merged_diagnostic_markdown
from trace_input_adapter import normalize_trace_for_parser


DEFAULT_STAGES = ("adapt", "graph", "judge")


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def sanitize_trace_id(value: str):
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._-")
    return text[:120]


def count_spans(spans):
    total = 0
    stack = list(spans or [])
    while stack:
        span = stack.pop()
        if not isinstance(span, dict):
            continue
        total += 1
        children = span.get("child_spans")
        if isinstance(children, list):
            stack.extend(children)
    return total


def parse_stages(text: str):
    if not text or text.lower() == "all":
        return list(DEFAULT_STAGES)
    requested = []
    valid = set(DEFAULT_STAGES)
    for part in text.split(","):
        item = part.strip().lower()
        if not item:
            continue
        if item not in valid:
            raise ValueError(f"Unknown stage: {item}. Valid stages: all,{','.join(DEFAULT_STAGES)}")
        requested.append(item)
    if not requested:
        raise ValueError("No stages selected.")
    return requested


def run_cmd(cmd, cwd: Path, env, dry_run=False):
    printable = []
    hide_next = False
    for item in cmd:
        text = str(item)
        if hide_next:
            printable.append("***")
            hide_next = False
            continue
        printable.append(text)
        if text in {"--api-key", "--key", "--token"}:
            hide_next = True
    print(f"\n[{now_str()}] RUN: {' '.join(printable)}")
    if dry_run:
        return True
    completed = subprocess.run([str(x) for x in cmd], cwd=str(cwd), env=env)
    return completed.returncode == 0


def resolve_api_key(args, env):
    if args.api_key:
        return args.api_key
    base_url = (args.base_url or "").lower()
    if "deepseek" in base_url:
        return env.get("DEEPSEEK_API_KEY") or env.get("OPENAI_API_KEY") or env.get("SILICONFLOW_API_KEY")
    return env.get("SILICONFLOW_API_KEY") or env.get("DEEPSEEK_API_KEY") or env.get("OPENAI_API_KEY")


def prepare_paths(args, trace_id):
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_name = args.run_name or f"{trace_id}_{timestamp_str()}"
        run_dir = Path(args.output_root) / run_name

    paths = {
        "run_dir": run_dir,
        "raw_dir": run_dir / "raw",
        "adapted_dir": run_dir / "adapted",
        "graph_dir": run_dir / "graph",
        "judge_dir": run_dir / "judge",
        "markdown_dir": Path(args.markdown_dir) if args.markdown_dir else run_dir / "reports",
        "summary_file": run_dir / "pipeline_summary.json",
    }
    return paths


def stage_adapt(args, trace_path: Path, trace_id_hint: str, paths, summary):
    stage = {"name": "adapt", "started_at": now_str(), "status": "running"}
    try:
        raw_obj = load_json(trace_path)
        normalized = normalize_trace_for_parser(raw_obj)
        if normalized is None:
            raise RuntimeError("trace_input_adapter did not recognize a supported trace structure.")

        trace_id = sanitize_trace_id(args.trace_id or normalized.get("trace_id") or trace_id_hint)
        if not trace_id:
            trace_id = sanitize_trace_id(trace_path.stem) or "trace"
        normalized["trace_id"] = normalized.get("trace_id") or trace_id

        raw_copy = paths["raw_dir"] / f"{trace_id}.json"
        adapted_file = paths["adapted_dir"] / f"{trace_id}.json"
        raw_copy.parent.mkdir(parents=True, exist_ok=True)
        adapted_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trace_path, raw_copy)
        dump_json(adapted_file, normalized)

        stage.update(
            {
                "status": "ok",
                "ended_at": now_str(),
                "trace_id": trace_id,
                "raw_file": str(raw_copy),
                "adapted_file": str(adapted_file),
                "root_span_count": len(normalized.get("spans") or []),
                "total_span_count": count_spans(normalized.get("spans") or []),
            }
        )
        summary["trace_id"] = trace_id
        summary["paths"]["raw_file"] = str(raw_copy)
        summary["paths"]["adapted_file"] = str(adapted_file)
        return True, trace_id, adapted_file
    except Exception as exc:
        stage.update({"status": "failed", "ended_at": now_str(), "error": str(exc)})
        return False, None, None
    finally:
        summary["stages"].append(stage)


def find_existing_adapted(paths, trace_id):
    adapted_file = paths["adapted_dir"] / f"{trace_id}.json"
    return adapted_file if adapted_file.exists() else None


def stage_graph(args, trace_id, adapted_file: Path, paths, env, summary):
    stage = {"name": "graph", "started_at": now_str(), "status": "running"}
    graph_file = paths["graph_dir"] / f"{trace_id}_graph.json"
    pruning_file = paths["graph_dir"] / f"{trace_id}_pruning" / "agenttrace_suspicious_paths.json"
    if graph_file.exists() and pruning_file.exists() and not args.force:
        stage.update(
            {
                "status": "skipped",
                "reason": "graph and agenttrace pruning already exist; use --force to regenerate",
                "ended_at": now_str(),
                "graph_file": str(graph_file),
                "pruning_file": str(pruning_file),
            }
        )
        summary["stages"].append(stage)
        summary["paths"]["graph_file"] = str(graph_file)
        summary["paths"]["pruning_file"] = str(pruning_file)
        return True

    cmd = [
        sys.executable,
        "OTL_trace.py",
        "--input",
        adapted_file,
        "--out-prefix",
        trace_id,
        "--output-dir",
        paths["graph_dir"],
    ]
    ok = run_cmd(cmd, cwd=args.project_root, env=env, dry_run=args.dry_run)
    stage.update(
        {
            "status": "ok" if ok else "failed",
            "ended_at": now_str(),
            "graph_file": str(graph_file),
            "pruning_file": str(pruning_file),
        }
    )
    summary["stages"].append(stage)
    summary["paths"]["graph_file"] = str(graph_file)
    summary["paths"]["pruning_file"] = str(pruning_file)
    return ok


def stage_judge(args, trace_id, paths, env, summary):
    stage = {"name": "judge", "started_at": now_str(), "status": "running"}
    merged_file = paths["judge_dir"] / "merged" / f"{trace_id}.json"
    if merged_file.exists() and not args.force:
        markdown_file = None
        if args.export_md:
            markdown_file = export_merged_diagnostic_markdown(
                merged_file,
                paths["markdown_dir"] / f"{trace_id}_diagnostic_report.md",
                judgement_source=args.markdown_source,
            )
        stage.update(
            {
                "status": "skipped",
                "reason": "merged judgement already exists; use --force to regenerate",
                "ended_at": now_str(),
                "merged_file": str(merged_file),
                "markdown_file": str(markdown_file) if markdown_file else None,
            }
        )
        summary["stages"].append(stage)
        summary["paths"]["judge_merged_file"] = str(merged_file)
        if markdown_file:
            summary["paths"]["diagnostic_markdown"] = str(markdown_file)
        return True

    api_key = resolve_api_key(args, env)
    judge_env = dict(env)
    if api_key:
        judge_env["DEEPSEEK_API_KEY"] = api_key
        judge_env["OPENAI_API_KEY"] = api_key

    cmd = [
        sys.executable,
        "deepseek_judge_trace.py",
        "--gaia-dir",
        paths["raw_dir"],
        "--graph-dir",
        paths["graph_dir"],
        "--graph-source",
        args.graph_source,
        "--pruning-strategy",
        args.pruning_strategy,
        "--out-dir",
        paths["judge_dir"],
        "--trace-id",
        trace_id,
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--max-spans",
        str(args.max_spans),
        "--max-input-chars",
        str(args.max_input_chars),
        "--workers",
        str(args.judge_workers),
        "--judge-max-tokens",
        str(args.judge_max_tokens),
        "--judge-review-max-tokens",
        str(args.judge_review_max_tokens),
        "--json-repair-max-tokens",
        str(args.json_repair_max_tokens),
        "--review-first-pass-max-chars",
        str(args.review_first_pass_max_chars),
        "--thinking-mode",
        args.thinking_mode,
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])
    if args.disable_judge_review:
        cmd.append("--disable-judge-review")
    if args.raw_input_mode:
        cmd.extend(["--raw-input-mode", args.raw_input_mode])
    if args.export_md:
        cmd.extend(
            [
                "--export-md",
                "--markdown-dir",
                paths["markdown_dir"],
                "--markdown-source",
                args.markdown_source,
            ]
        )

    ok = run_cmd(cmd, cwd=args.project_root, env=judge_env, dry_run=args.dry_run)
    markdown_file = paths["markdown_dir"] / f"{trace_id}_diagnostic_report.md"
    stage.update(
        {
            "status": "ok" if ok else "failed",
            "ended_at": now_str(),
            "merged_file": str(merged_file),
            "judge_summary_file": str(paths["judge_dir"] / "judge_summary.json"),
            "markdown_file": str(markdown_file) if args.export_md else None,
        }
    )
    summary["stages"].append(stage)
    summary["paths"]["judge_merged_file"] = str(merged_file)
    summary["paths"]["judge_summary_file"] = str(paths["judge_dir"] / "judge_summary.json")
    if args.export_md:
        summary["paths"]["diagnostic_markdown"] = str(markdown_file)
    return ok


def build_cli():
    p = argparse.ArgumentParser(
        description="Run one trace through adaptation, AgentTrace graph construction, pruning, and LLM root-cause diagnosis."
    )
    p.add_argument("--trace-path", required=True, help="Input raw trace JSON path.")
    p.add_argument("--trace-id", default=None, help="Optional stable trace id; defaults to normalized trace_id or file stem.")
    p.add_argument("--project-root", default=".", help="Project root containing OTL_trace.py and deepseek_judge_trace.py.")
    p.add_argument("--output-root", default="pipeline_runs/single_trace", help="Root directory for pipeline run outputs.")
    p.add_argument("--run-name", default=None, help="Optional run directory name under --output-root.")
    p.add_argument("--run-dir", default=None, help="Explicit run directory; overrides --output-root and --run-name.")
    p.add_argument("--export-md", action="store_true", help="Export the diagnosis as a Markdown report.")
    p.add_argument("--markdown-dir", default=None, help="Markdown output directory; also enables --export-md.")
    p.add_argument(
        "--markdown-source",
        choices=["raw", "graph", "pure_graph"],
        default="graph",
        help="Judgement variant used in the Markdown report.",
    )

    p.add_argument("--stages", default="all", help="Master switch: all or comma list from adapt,graph,judge.")
    p.add_argument("--force", action="store_true", help="Regenerate existing graph/judge outputs.")
    p.add_argument("--resume", action="store_true", help="Continue later stages even if a prior stage fails.")
    p.add_argument("--dry-run", action="store_true", help="Print commands and planned paths without running subprocess stages.")

    p.add_argument("--graph-source", choices=["full", "pruning", "agenttrace_first"], default="agenttrace_first")
    p.add_argument("--pruning-strategy", default="weighted_k_shortest_paths")

    p.add_argument("--model", default="deepseek-v4-flash", help="Judge model.")
    p.add_argument("--base-url", default="https://api.deepseek.com", help="Judge API base URL.")
    p.add_argument("--thinking-mode", choices=["disabled", "enabled"], default="disabled", help="DeepSeek V4 thinking mode")
    p.add_argument("--api-key", default=None, help="Judge API key; otherwise uses DEEPSEEK_API_KEY/OPENAI_API_KEY env.")
    p.add_argument("--judge-workers", type=int, default=1)
    p.add_argument("--max-spans", type=int, default=120)
    p.add_argument("--max-input-chars", type=int, default=50000)
    p.add_argument("--raw-input-mode", choices=["digest", "full"], default="digest")
    p.add_argument("--judge-max-tokens", type=int, default=0)
    p.add_argument("--judge-review-max-tokens", type=int, default=0)
    p.add_argument("--json-repair-max-tokens", type=int, default=0)
    p.add_argument("--review-first-pass-max-chars", type=int, default=0)
    p.add_argument("--disable-judge-review", action="store_true")
    return p


def main():
    args = build_cli().parse_args()
    if args.markdown_dir:
        args.export_md = True
    args.project_root = Path(args.project_root).resolve()
    trace_path = Path(args.trace_path).resolve()
    if not trace_path.exists():
        raise RuntimeError(f"Trace file not found: {trace_path}")

    stages = parse_stages(args.stages)
    trace_id_hint = sanitize_trace_id(args.trace_id or trace_path.stem) or "trace"
    initial_paths = prepare_paths(args, trace_id_hint)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    summary = {
        "started_at": now_str(),
        "status": "running",
        "project_root": str(args.project_root),
        "input_trace_path": str(trace_path),
        "requested_stages": stages,
        "graph_source": args.graph_source,
        "pruning_strategy": args.pruning_strategy if args.graph_source in {"pruning", "agenttrace_first"} else None,
        "model": args.model,
        "base_url": args.base_url,
        "paths": {"run_dir": str(initial_paths["run_dir"])},
        "stages": [],
    }

    paths = initial_paths
    trace_id = trace_id_hint
    adapted_file = None
    ok_all = True

    if "adapt" in stages:
        ok, adapted_trace_id, adapted_path = stage_adapt(args, trace_path, trace_id_hint, paths, summary)
        ok_all = ok_all and ok
        if ok:
            trace_id = adapted_trace_id
            adapted_file = adapted_path
        elif not args.resume:
            summary["status"] = "failed"
            summary["ended_at"] = now_str()
            dump_json(paths["summary_file"], summary)
            print(f"\nPipeline failed at adapt stage. summary: {paths['summary_file']}")
            return
    else:
        if args.trace_id:
            trace_id = sanitize_trace_id(args.trace_id)
        adapted_file = find_existing_adapted(paths, trace_id)
        if adapted_file is None and ("graph" in stages):
            raise RuntimeError("Cannot run graph without adapt stage unless adapted/<trace_id>.json already exists.")

    if "graph" in stages and adapted_file is not None:
        ok = stage_graph(args, trace_id, adapted_file, paths, env, summary)
        ok_all = ok_all and ok
        if not ok and not args.resume:
            summary["status"] = "failed"
            summary["ended_at"] = now_str()
            dump_json(paths["summary_file"], summary)
            print(f"\nPipeline failed at graph stage. summary: {paths['summary_file']}")
            return

    if "judge" in stages:
        ok = stage_judge(args, trace_id, paths, env, summary)
        ok_all = ok_all and ok
        if not ok and not args.resume:
            summary["status"] = "failed"
            summary["ended_at"] = now_str()
            dump_json(paths["summary_file"], summary)
            print(f"\nPipeline failed at judge stage. summary: {paths['summary_file']}")
            return

    summary["status"] = "ok" if ok_all else "failed"
    summary["ended_at"] = now_str()
    dump_json(paths["summary_file"], summary)

    print("\n=== Single Trace Pipeline Done ===")
    print(f"status: {summary['status']}")
    print(f"trace_id: {trace_id}")
    print(f"run_dir: {paths['run_dir']}")
    print(f"summary: {paths['summary_file']}")
    if "adapted_file" in summary["paths"]:
        print(f"adapted: {summary['paths']['adapted_file']}")
    if "graph_file" in summary["paths"]:
        print(f"graph: {summary['paths']['graph_file']}")
    if "pruning_file" in summary["paths"]:
        print(f"pruning: {summary['paths']['pruning_file']}")
    if "judge_merged_file" in summary["paths"]:
        print(f"judge: {summary['paths']['judge_merged_file']}")
    if "diagnostic_markdown" in summary["paths"]:
        print(f"markdown: {summary['paths']['diagnostic_markdown']}")


if __name__ == "__main__":
    main()
