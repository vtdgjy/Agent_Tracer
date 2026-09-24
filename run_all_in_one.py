import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(cmd, cwd, env):
    print(f"\n[{now_str()}] RUN: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(cwd), env=env)
    return completed.returncode == 0


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_trace_ids(gaia_dir: Path):
    return sorted([p.stem for p in gaia_dir.glob("*.json")])


def dump_json(path: Path, obj):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stage_build_graph(args, env, summary):
    stage = {"name": "stage1_build_graph", "started_at": now_str(), "status": "running"}

    cmd = [
        sys.executable,
        "OTL_trace.py",
        "--local-mode",
        "--local-gaia-dir",
        str(args.gaia_dir),
        "--local-annotations-dir",
        str(args.annotations_dir),
        "--out-prefix",
        args.out_prefix,
        "--output-dir",
        str(args.graph_output_root),
        "--max-samples",
        str(args.max_samples),
        "--workers",
        str(args.graph_workers),
    ]

    ok = run_cmd(cmd, cwd=args.project_root, env=env)
    stage["ended_at"] = now_str()
    stage["status"] = "ok" if ok else "failed"
    summary["stages"].append(stage)
    return ok


def stage_pruning_export(args, env, summary):
    stage = {"name": "stage1b_export_agenttrace_pruning", "started_at": now_str(), "status": "running"}

    cmd = [
        sys.executable,
        "export_agenttrace_pruning_for_judge.py",
        "--input-dir",
        str(args.graph_dir),
        "--max-samples",
        str(args.max_samples),
        "--workers",
        str(args.export_workers),
        "--max-candidate-chains",
        str(args.max_candidate_chains),
        "--min-dependency-confidence",
        str(args.min_dependency_confidence),
        "--char-budget",
        str(args.pruning_char_budget),
        "--max-targets",
        str(args.tg_spe_max_targets),
        "--min-final-dependency",
        str(args.tg_spe_min_final_dependency),
        "--min-path-confidence",
        str(args.tg_spe_min_path_confidence),
    ]

    ok = run_cmd(cmd, cwd=args.project_root, env=env)
    stage["ended_at"] = now_str()
    stage["status"] = "ok" if ok else "failed"
    summary["stages"].append(stage)
    return ok


def stage_judge(args, env, summary):
    stage = {"name": "stage2_judge", "started_at": now_str(), "status": "running"}

    judge_model = args.judge_model or args.model
    judge_base_url = args.judge_base_url or args.base_url
    judge_api_key = (
        args.judge_api_key
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("SILICONFLOW_API_KEY")
    )

    judge_env = dict(env)
    if judge_api_key:
        judge_env["DEEPSEEK_API_KEY"] = judge_api_key
    else:
        judge_env.pop("DEEPSEEK_API_KEY", None)
        judge_env.pop("OPENAI_API_KEY", None)

    cmd = [
        sys.executable,
        "deepseek_judge_trace.py",
        "--gaia-dir", str(args.gaia_dir),
        "--graph-dir", str(args.graph_dir),
        "--graph-source", args.graph_source,
        "--pruning-strategy", args.pruning_strategy,
        "--out-dir", str(args.judge_out_dir),
        "--model", judge_model,
        "--base-url", judge_base_url,
        "--max-samples", str(args.max_samples),
        "--workers", str(args.judge_workers),
        "--max-input-chars", str(args.judge_max_input_chars),
        "--judge-max-tokens", str(args.judge_max_tokens),
        "--judge-review-max-tokens", str(args.judge_review_max_tokens),
        "--json-repair-max-tokens", str(args.json_repair_max_tokens),
        "--review-first-pass-max-chars", str(args.review_first_pass_max_chars),
        "--thinking-mode", args.judge_thinking_mode,
    ]

    if args.trace_id:
        cmd.extend(["--trace-id", args.trace_id])

    if judge_api_key:
        cmd.extend(["--api-key", judge_api_key])

    ok = run_cmd(cmd, cwd=args.project_root, env=judge_env)
    stage["ended_at"] = now_str()
    stage["status"] = "ok" if ok else "failed"
    summary["stages"].append(stage)
    return ok


def stage_alignment(args, env, summary):
    stage = {"name": "stage3_alignment", "started_at": now_str(), "status": "running"}

    cmd = [
        sys.executable,
        "deepseek_alignment_eval.py",
        "--merged-dir",
        str(args.judge_out_dir / "merged"),
        "--annotations-dir",
        str(args.annotations_dir),
        "--out-dir",
        str(args.align_out_dir),
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--max-samples",
        str(args.max_samples),
        "--max-workers",
        str(args.align_workers),
        "--compare-max-tokens",
        str(args.align_compare_max_tokens),
        "--thinking-mode",
        args.align_thinking_mode,
    ]

    if args.trace_id:
        cmd.extend(["--trace-id", args.trace_id])

    if args.api_key:
        cmd.extend(["--api-key", args.api_key])

    ok = run_cmd(cmd, cwd=args.project_root, env=env)
    stage["ended_at"] = now_str()
    stage["status"] = "ok" if ok else "failed"
    summary["stages"].append(stage)
    return ok


def build_cli():
    p = argparse.ArgumentParser(description="One-click full pipeline: build graph -> judge -> alignment")
    p.add_argument("--project-root", default=".", help="Project root")
    p.add_argument("--gaia-dir", default="trail_data/GAIA", help="GAIA trace directory")
    p.add_argument("--annotations-dir", default="trail_data/processed_annotations_gaia", help="Human annotation directory")

    p.add_argument("--graph-output-root", default="output_graphs_useLLM0319", help="OTL_trace output root")
    p.add_argument("--out-prefix", default="trail_gaia_local", help="OTL_trace out prefix")

    p.add_argument("--judge-out-dir", default="output_graphs/deepseek_judge", help="Judge output directory")
    p.add_argument("--align-out-dir", default="output_graphs/deepseek_alignment", help="Alignment output directory")
    p.add_argument("--graph-source", choices=["full", "pruning", "agenttrace_first"], default="agenttrace_first", help="Judge阶段读取的图来源：full、pruning，或优先 agenttrace_suspicious_paths 再回退 full")
    p.add_argument("--pruning-strategy", default="weighted_k_shortest_paths", help="当 --graph-source=pruning 时使用的策略名；agenttrace_first 会优先尝试 agenttrace_suspicious_paths")

    p.add_argument("--model", default="deepseek-v4-flash", help="Default model for alignment stage")
    p.add_argument("--base-url", default="https://api.deepseek.com", help="Default API base URL for alignment stage")
    p.add_argument("--api-key", default=None, help="Default API key (optional, can use env vars)")
    p.add_argument("--judge-model", default="deepseek-v4-flash", help="Model for judge (root-cause analysis) stage")
    p.add_argument("--judge-base-url", default="https://api.deepseek.com", help="API base URL for judge stage")
    p.add_argument("--judge-api-key", default=None, help="API key for judge stage; falls back to SILICONFLOW_API_KEY env var")
    p.add_argument("--judge-thinking-mode", choices=["disabled", "enabled"], default="disabled", help="DeepSeek V4 thinking mode for judge calls")

    p.add_argument("--trace-id", default=None, help="Only run one trace_id")
    p.add_argument("--max-samples", type=int, default=0, help="Max samples (0 means all)")

    p.add_argument("--graph-workers", type=int, default=8, help="Workers for graph stage")
    p.add_argument("--export-workers", type=int, default=8, help="Workers for agenttrace pruning export stage")
    p.add_argument("--judge-workers", type=int, default=8, help="Workers for judge stage")
    p.add_argument("--align-workers", type=int, default=8, help="Workers for alignment stage")
    p.add_argument("--max-candidate-chains", type=int, default=5, help="Top-K causal chains exported for graph judge")
    p.add_argument("--min-dependency-confidence", type=float, default=0.65, help="Minimum typed-edge confidence for backward failure slicing")
    p.add_argument("--pruning-char-budget", type=int, default=0, help="Optional TG-SPE character limit; 0 disables it")
    p.add_argument("--tg-spe-max-targets", type=int, default=4, help="Maximum terminal diagnosis targets retained by TG-SPE")
    p.add_argument("--tg-spe-min-final-dependency", type=float, default=0.55, help="Minimum path dependency on the final target")
    p.add_argument("--tg-spe-min-path-confidence", type=float, default=0.45, help="Minimum typed-edge causal path confidence")
    p.add_argument("--judge-max-input-chars", type=int, default=28000, help="Max chars of each judge digest; 0 means no truncation")
    p.add_argument("--judge-max-tokens", type=int, default=0, help="Judge first-pass max completion tokens; 0 means no explicit cap")
    p.add_argument("--judge-review-max-tokens", type=int, default=0, help="Judge review max completion tokens; 0 means no explicit cap")
    p.add_argument("--json-repair-max-tokens", type=int, default=0, help="JSON repair max completion tokens; 0 means no explicit cap")
    p.add_argument("--review-first-pass-max-chars", type=int, default=0, help="Max chars of first-pass JSON included in review prompt; 0 means no truncation")
    p.add_argument("--align-compare-max-tokens", type=int, default=0, help="Alignment compare max completion tokens; 0 means no explicit cap")
    p.add_argument("--align-thinking-mode", choices=["disabled", "enabled"], default="disabled", help="DeepSeek V4 thinking mode for alignment calls")

    p.add_argument("--skip-graph", action="store_true", help="Skip graph stage")
    p.add_argument("--skip-pruning-export", action="store_true", help="Skip regenerating agenttrace_suspicious_paths pruning files")
    p.add_argument("--skip-judge", action="store_true", help="Skip judge stage")
    p.add_argument("--skip-align", action="store_true", help="Skip alignment stage")

    p.add_argument("--resume", action="store_true", help="Continue next stages even if prior stage failed")
    p.add_argument("--summary-file", default="output_graphs/one_click_pipeline_summary.json", help="Pipeline summary output")
    return p


def main():
    args = build_cli().parse_args()

    args.project_root = Path(args.project_root).resolve()
    args.gaia_dir = Path(args.gaia_dir)
    args.annotations_dir = Path(args.annotations_dir)
    args.graph_output_root = Path(args.graph_output_root)
    args.judge_out_dir = Path(args.judge_out_dir)
    args.align_out_dir = Path(args.align_out_dir)

    args.graph_dir = args.graph_output_root / f"{args.out_prefix}_local_batch"

    if not args.gaia_dir.exists():
        raise RuntimeError(f"GAIA 目录不存在: {args.gaia_dir}")

    trace_ids = list_trace_ids(args.gaia_dir)
    if not trace_ids:
        raise RuntimeError(f"GAIA 目录没有 json trace: {args.gaia_dir}")

    env = os.environ.copy()
    if args.api_key:
        env["DEEPSEEK_API_KEY"] = args.api_key
        env["OPENAI_API_KEY"] = args.api_key

    summary = {
        "started_at": now_str(),
        "project_root": str(args.project_root),
        "trace_total": len(trace_ids),
        "trace_id": args.trace_id,
        "max_samples": args.max_samples,
        "graph_source": args.graph_source,
        "pruning_strategy": args.pruning_strategy if args.graph_source in {"pruning", "agenttrace_first"} else None,
        "workers": {
            "graph": args.graph_workers,
            "export": args.export_workers,
            "judge": args.judge_workers,
            "align": args.align_workers,
        },
        "chain_export": {
            "max_candidate_chains": args.max_candidate_chains,
            "min_dependency_confidence": args.min_dependency_confidence,
        },
        "stages": [],
    }

    ok_graph = True
    if not args.skip_graph:
        ok_graph = stage_build_graph(args, env, summary)
        if not ok_graph and not args.resume:
            summary["status"] = "failed"
            summary["ended_at"] = now_str()
            dump_json(Path(args.summary_file), summary)
            print(f"\nPipeline failed at graph stage. summary: {args.summary_file}")
            return

    ok_export = True
    if not args.skip_pruning_export:
        ok_export = stage_pruning_export(args, env, summary)
        if not ok_export and not args.resume:
            summary["status"] = "failed"
            summary["ended_at"] = now_str()
            dump_json(Path(args.summary_file), summary)
            print(f"\nPipeline failed at pruning export stage. summary: {args.summary_file}")
            return

    ok_judge = True
    if not args.skip_judge:
        ok_judge = stage_judge(args, env, summary)
        if not ok_judge and not args.resume:
            summary["status"] = "failed"
            summary["ended_at"] = now_str()
            dump_json(Path(args.summary_file), summary)
            print(f"\nPipeline failed at judge stage. summary: {args.summary_file}")
            return

    ok_align = True
    if not args.skip_align:
        ok_align = stage_alignment(args, env, summary)

    summary["status"] = "ok" if (ok_graph and ok_export and ok_judge and ok_align) else "failed"
    summary["ended_at"] = now_str()

    dump_json(Path(args.summary_file), summary)

    print("\n=== One-click Pipeline Done ===")
    print(f"status: {summary['status']}")
    print(f"graph_dir: {args.graph_dir}")
    print(f"judge_out_dir: {args.judge_out_dir}")
    print(f"align_out_dir: {args.align_out_dir}")
    print(f"summary: {args.summary_file}")


if __name__ == "__main__":
    main()
