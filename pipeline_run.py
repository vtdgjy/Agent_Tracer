import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_cmd(cmd, cwd=None, env=None):
    print(f"\n[{now_str()}] RUN: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=cwd, env=env)
    return completed.returncode == 0


def list_trace_ids(gaia_dir: Path):
    return sorted([p.stem for p in gaia_dir.glob("*.json")])


def progress_iter(items, desc):
    if tqdm is None:
        print(f"{desc}: total={len(items)}")
        return items
    return tqdm(items, total=len(items), desc=desc)


def stage1_build_graph(args, trace_ids, summary, env):
    stage_name = "stage1_build_graph"
    graph_dir = Path(args.graph_dir)

    existing = 0
    for tid in trace_ids:
        if (graph_dir / f"local_{tid}_graph.json").exists():
            existing += 1

    stage_result = {
        "name": stage_name,
        "started_at": now_str(),
        "total": len(trace_ids),
        "existing_before": existing,
        "status": "skipped" if existing == len(trace_ids) else "running",
    }

    if existing == len(trace_ids):
        stage_result["ended_at"] = now_str()
        stage_result["existing_after"] = existing
        summary["stages"].append(stage_result)
        print(f"[{stage_name}] skipped (all {existing}/{len(trace_ids)} graphs already exist)")
        return True

    cmd = [
        sys.executable,
        "OTL_trace.py",
        "--local-mode",
        "--local-gaia-dir", str(args.gaia_dir),
        "--local-annotations-dir", str(args.annotations_dir),
        "--out-prefix", args.out_prefix,
        "--output-dir", str(args.graph_output_root),
        "--max-samples", "0",
        "--workers", str(args.graph_workers),
    ]
    ok = run_cmd(cmd, cwd=str(args.project_root), env=env)

    existing_after = 0
    for tid in trace_ids:
        if (graph_dir / f"local_{tid}_graph.json").exists():
            existing_after += 1

    stage_result["ended_at"] = now_str()
    stage_result["existing_after"] = existing_after
    stage_result["status"] = "ok" if ok else "failed"
    summary["stages"].append(stage_result)
    return ok


def stage1b_export_agenttrace_pruning(args, trace_ids, summary, env):
    stage_name = "stage1b_export_agenttrace_pruning"
    stage_result = {
        "name": stage_name,
        "started_at": now_str(),
        "total": len(trace_ids),
        "status": "running",
    }

    cmd = [
        sys.executable,
        "export_agenttrace_pruning_for_judge.py",
        "--input-dir", str(args.graph_dir),
        "--max-samples", "0",
        "--workers", str(args.export_workers),
        "--max-candidate-chains", str(args.max_candidate_chains),
        "--min-dependency-confidence", str(args.min_dependency_confidence),
        "--char-budget", str(args.pruning_char_budget),
        "--max-targets", str(args.tg_spe_max_targets),
        "--min-final-dependency", str(args.tg_spe_min_final_dependency),
        "--min-path-confidence", str(args.tg_spe_min_path_confidence),
    ]
    ok = run_cmd(cmd, cwd=str(args.project_root), env=env)

    stage_result["ended_at"] = now_str()
    stage_result["status"] = "ok" if ok else "failed"
    summary["stages"].append(stage_result)
    return ok


def stage2_judge(args, trace_ids, summary, env):
    stage_name = "stage2_judge"
    merged_dir = Path(args.judge_out_dir) / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    pending = [tid for tid in trace_ids if not (merged_dir / f"{tid}.json").exists()]
    stage_result = {
        "name": stage_name,
        "started_at": now_str(),
        "total": len(trace_ids),
        "pending_before": len(pending),
        "processed_now": 0,
        "failed_now": 0,
        "status": "running",
    }

    if not pending:
        stage_result["status"] = "skipped"
        stage_result["ended_at"] = now_str()
        summary["stages"].append(stage_result)
        print(f"[{stage_name}] skipped (all merged files already exist)")
        return True

    ok_all = True
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

    for tid in progress_iter(pending, "Stage2 Judge"):
        cmd = [
            sys.executable,
            "deepseek_judge_trace.py",
            "--gaia-dir", str(args.gaia_dir),
            "--graph-dir", str(args.graph_dir),
            "--out-dir", str(args.judge_out_dir),
            "--model", judge_model,
            "--base-url", judge_base_url,
            "--trace-id", tid,
            "--max-samples", "1",
            "--workers", str(args.judge_workers),
            "--max-input-chars", str(args.judge_max_input_chars),
            "--judge-max-tokens", str(args.judge_max_tokens),
            "--judge-review-max-tokens", str(args.judge_review_max_tokens),
            "--json-repair-max-tokens", str(args.json_repair_max_tokens),
            "--review-first-pass-max-chars", str(args.review_first_pass_max_chars),
            "--thinking-mode", args.judge_thinking_mode,
        ]
        if judge_api_key:
            cmd.extend(["--api-key", judge_api_key])

        ok = run_cmd(cmd, cwd=str(args.project_root), env=judge_env)
        if ok and (merged_dir / f"{tid}.json").exists():
            stage_result["processed_now"] += 1
        else:
            stage_result["failed_now"] += 1
            ok_all = False
            if not args.keep_going:
                break

    stage_result["status"] = "ok" if ok_all else "failed"
    stage_result["ended_at"] = now_str()
    summary["stages"].append(stage_result)
    return ok_all or args.keep_going


def stage3_alignment(args, trace_ids, summary, env):
    stage_name = "stage3_alignment"
    ann_dir = Path(args.annotations_dir)

    if not ann_dir.exists():
        stage_result = {
            "name": stage_name,
            "started_at": now_str(),
            "ended_at": now_str(),
            "status": "failed",
            "error": f"annotations dir not found: {ann_dir}",
        }
        summary["stages"].append(stage_result)
        return False

    missing_ann = [tid for tid in trace_ids if not (ann_dir / f"{tid}.json").exists()]
    if missing_ann:
        stage_result = {
            "name": stage_name,
            "started_at": now_str(),
            "ended_at": now_str(),
            "status": "failed",
            "error": f"missing annotations: {len(missing_ann)}",
            "missing_examples": missing_ann[:10],
        }
        summary["stages"].append(stage_result)
        return False

    raw_vs_human = Path(args.align_out_dir) / "raw_vs_human"
    graph_vs_human = Path(args.align_out_dir) / "graph_vs_human"
    raw_vs_human.mkdir(parents=True, exist_ok=True)
    graph_vs_human.mkdir(parents=True, exist_ok=True)

    pending = []
    for tid in trace_ids:
        raw_ok = (raw_vs_human / f"{tid}.json").exists()
        graph_ok = (graph_vs_human / f"{tid}.json").exists()
        if not (raw_ok and graph_ok):
            pending.append(tid)

    stage_result = {
        "name": stage_name,
        "started_at": now_str(),
        "total": len(trace_ids),
        "pending_before": len(pending),
        "processed_now": 0,
        "failed_now": 0,
        "status": "running",
    }

    if not pending:
        stage_result["status"] = "skipped"
        stage_result["ended_at"] = now_str()
        summary["stages"].append(stage_result)
        print(f"[{stage_name}] skipped (all alignment files already exist)")
        return True

    ok_all = True
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _run_align_one(tid):
        cmd = [
            sys.executable,
            "deepseek_alignment_eval.py",
            "--merged-dir", str(Path(args.judge_out_dir) / "merged"),
            "--annotations-dir", str(args.annotations_dir),
            "--out-dir", str(args.align_out_dir),
            "--model", args.model,
            "--base-url", args.base_url,
            "--trace-id", tid,
            "--max-samples", "1",
            "--max-workers", "1",
            "--compare-max-tokens", str(args.align_compare_max_tokens),
            "--thinking-mode", args.align_thinking_mode,
        ]
        if args.api_key:
            cmd.extend(["--api-key", args.api_key])
        ok = run_cmd(cmd, cwd=str(args.project_root), env=env)
        raw_ok = (raw_vs_human / f"{tid}.json").exists()
        graph_ok = (graph_vs_human / f"{tid}.json").exists()
        return tid, ok and raw_ok and graph_ok

    with ThreadPoolExecutor(max_workers=args.align_workers) as ex:
        futures = {ex.submit(_run_align_one, tid): tid for tid in pending}
        done = 0
        for fut in as_completed(futures):
            done += 1
            tid, success = fut.result()
            if success:
                stage_result["processed_now"] += 1
            else:
                stage_result["failed_now"] += 1
                ok_all = False
            print(f"  [{done}/{len(pending)}] {tid}: {'ok' if success else 'failed'}")

    stage_result["status"] = "ok" if ok_all else "failed"
    stage_result["ended_at"] = now_str()
    summary["stages"].append(stage_result)
    return ok_all or args.keep_going


def build_parser():
    p = argparse.ArgumentParser(description="End-to-end pipeline runner with resume support")
    p.add_argument("--project-root", default=".", help="Project root")
    p.add_argument("--gaia-dir", default="trail_data/GAIA", help="GAIA trace directory")
    p.add_argument("--annotations-dir", default="trail_data/processed_annotations_gaia", help="Human annotation directory")

    p.add_argument("--graph-output-root", default="output_graphs_useLLMv2", help="Root output for OTL_trace")
    p.add_argument("--out-prefix", default="trail_gaia_local", help="OTL_trace out-prefix")
    p.add_argument("--graph-dir", default="output_graphs_useLLMv2/trail_gaia_local_local_batch", help="Graph json directory")

    p.add_argument("--judge-out-dir", default="output_graphs/deepseek_judge", help="Judge output directory")
    p.add_argument("--align-out-dir", default="output_graphs/deepseek_alignment", help="Alignment output directory")

    p.add_argument("--model", default="deepseek-v4-flash", help="Default model for stages without an override (currently used by alignment)")
    p.add_argument("--base-url", default="https://api.deepseek.com", help="Default API base url for stages without an override (currently used by alignment)")
    p.add_argument("--api-key", default=None, help="Default API key for stages without an override (optional; can use env vars)")
    p.add_argument("--judge-model", default="deepseek-v4-flash", help="Override model used by the judge (root-cause analysis) stage")
    p.add_argument("--judge-base-url", default="https://api.deepseek.com", help="Override API base url used by the judge stage")
    p.add_argument("--judge-api-key", default=None, help="Override API key for the judge stage; falls back to --api-key / env vars (SILICONFLOW_API_KEY, DEEPSEEK_API_KEY)")
    p.add_argument("--judge-thinking-mode", choices=["disabled", "enabled"], default="disabled", help="DeepSeek V4 thinking mode for judge calls")
    p.add_argument("--judge-max-input-chars", type=int, default=28000, help="Max chars of each judge digest; 0 means no truncation")
    p.add_argument("--judge-max-tokens", type=int, default=0, help="Judge first-pass max completion tokens; 0 means no explicit cap")
    p.add_argument("--judge-review-max-tokens", type=int, default=0, help="Judge review max completion tokens; 0 means no explicit cap")
    p.add_argument("--json-repair-max-tokens", type=int, default=0, help="JSON repair max completion tokens; 0 means no explicit cap")
    p.add_argument("--review-first-pass-max-chars", type=int, default=0, help="Max chars of first-pass JSON included in review prompt; 0 means no truncation")
    p.add_argument("--align-compare-max-tokens", type=int, default=0, help="Alignment compare max completion tokens; 0 means no explicit cap")
    p.add_argument("--align-thinking-mode", choices=["disabled", "enabled"], default="disabled", help="DeepSeek V4 thinking mode for alignment calls")
    p.add_argument("--graph-workers", type=int, default=8, help="Workers for graph construction")
    p.add_argument("--export-workers", type=int, default=8, help="Workers for agenttrace pruning export")
    p.add_argument("--judge-workers", type=int, default=8, help="Workers passed to each judge command")
    p.add_argument("--align-workers", type=int, default=8, help="Workers passed to each alignment command")
    p.add_argument("--max-candidate-chains", type=int, default=5, help="Top-K causal chains exported for graph judge")
    p.add_argument("--min-dependency-confidence", type=float, default=0.65, help="Minimum typed-edge confidence for backward failure slicing")
    p.add_argument("--pruning-char-budget", type=int, default=0, help="Optional TG-SPE character limit; 0 disables it")
    p.add_argument("--tg-spe-max-targets", type=int, default=4, help="Maximum terminal diagnosis targets retained by TG-SPE")
    p.add_argument("--tg-spe-min-final-dependency", type=float, default=0.55, help="Minimum path dependency on the final target")
    p.add_argument("--tg-spe-min-path-confidence", type=float, default=0.45, help="Minimum typed-edge causal path confidence")
    p.add_argument("--skip-pruning-export", action="store_true", help="Skip regenerating agenttrace_suspicious_paths pruning files")
    p.add_argument("--keep-going", action="store_true", help="Continue remaining traces when one trace fails")
    p.add_argument("--summary-file", default="output_graphs/pipeline_run_summary.json", help="Final summary file")
    return p


def main():
    args = build_parser().parse_args()

    args.project_root = Path(args.project_root).resolve()
    args.gaia_dir = Path(args.gaia_dir)
    args.annotations_dir = Path(args.annotations_dir)
    args.graph_output_root = Path(args.graph_output_root)
    args.graph_dir = Path(args.graph_dir)
    args.judge_out_dir = Path(args.judge_out_dir)
    args.align_out_dir = Path(args.align_out_dir)

    env = os.environ.copy()
    if args.api_key:
        env["DEEPSEEK_API_KEY"] = args.api_key
        env["OPENAI_API_KEY"] = args.api_key

    trace_ids = list_trace_ids(args.gaia_dir)
    if not trace_ids:
        raise RuntimeError(f"No traces found in {args.gaia_dir}")

    summary = {
        "started_at": now_str(),
        "project_root": str(args.project_root),
        "trace_total": len(trace_ids),
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

    stage_total = 4 if not args.skip_pruning_export else 3
    stage_done = 0

    print(f"Pipeline start: traces={len(trace_ids)}")
    if tqdm is not None:
        stage_bar = tqdm(total=stage_total, desc="Pipeline Stages")
    else:
        stage_bar = None

    ok1 = stage1_build_graph(args, trace_ids, summary, env)
    stage_done += 1
    if stage_bar is not None:
        stage_bar.update(1)
    if not ok1 and not args.keep_going:
        summary["ended_at"] = now_str()
        summary["status"] = "failed"
        dump_json(Path(args.summary_file), summary)
        return

    ok1b = True
    if not args.skip_pruning_export:
        ok1b = stage1b_export_agenttrace_pruning(args, trace_ids, summary, env)
        stage_done += 1
        if stage_bar is not None:
            stage_bar.update(1)
        if not ok1b and not args.keep_going:
            summary["ended_at"] = now_str()
            summary["status"] = "failed"
            dump_json(Path(args.summary_file), summary)
            return

    ok2 = stage2_judge(args, trace_ids, summary, env)
    stage_done += 1
    if stage_bar is not None:
        stage_bar.update(1)
    if not ok2 and not args.keep_going:
        summary["ended_at"] = now_str()
        summary["status"] = "failed"
        dump_json(Path(args.summary_file), summary)
        return

    ok3 = stage3_alignment(args, trace_ids, summary, env)
    stage_done += 1
    if stage_bar is not None:
        stage_bar.update(1)
        stage_bar.close()

    summary["ended_at"] = now_str()
    summary["status"] = "ok" if (ok1 and ok1b and ok2 and ok3) else "failed"

    judge_summary = load_json(args.judge_out_dir / "judge_summary.json", default={})
    align_summary = load_json(args.align_out_dir / "alignment_summary.json", default={})
    summary["judge_summary_snapshot"] = {
        "processed": judge_summary.get("processed"),
        "failed": judge_summary.get("failed"),
        "token_usage": judge_summary.get("token_usage"),
    }
    summary["alignment_summary_snapshot"] = {
        "processed": align_summary.get("processed"),
        "failed": align_summary.get("failed"),
        "aggregates": align_summary.get("aggregates"),
    }

    dump_json(Path(args.summary_file), summary)

    print("\n=== Pipeline Done ===")
    print(f"- status: {summary['status']}")
    print(f"- traces: {len(trace_ids)}")
    print(f"- summary: {args.summary_file}")


if __name__ == "__main__":
    main()
