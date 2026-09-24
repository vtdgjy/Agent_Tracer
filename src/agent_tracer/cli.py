from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_root_script(script_name: str) -> None:
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find script: {script_path}")

    original_argv = sys.argv[:]
    original_sys_path = sys.path[:]
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.argv[0] = str(script_path)
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv[:] = original_argv
        sys.path[:] = original_sys_path


def otl_trace() -> None:
    _run_root_script("OTL_trace.py")


def pruning_strategies() -> None:
    _run_root_script("pruning_strategies.py")


def deepseek_judge() -> None:
    _run_root_script("deepseek_judge_trace.py")


def run_single_trace() -> None:
    _run_root_script("run_single_trace_pipeline.py")


def deepseek_align() -> None:
    _run_root_script("deepseek_alignment_eval.py")


def deepseek_unprocessed_eval() -> None:
    _run_root_script("deepseek_unprocessed_eval.py")


def pipeline_run() -> None:
    _run_root_script("pipeline_run.py")


def run_all_in_one() -> None:
    _run_root_script("run_all_in_one.py")


def graph_trace_cluster() -> None:
    _run_root_script("graph_trace_clustering.py")


def graph_trace_cluster_pipeline() -> None:
    _run_root_script("trace_cluster_pipeline.py")


def trace_review_queue() -> None:
    _run_root_script("trace_review_queue.py")


def trace_cluster_benchmark() -> None:
    _run_root_script("trace_cluster_benchmark.py")


def trace_cluster_trail_benchmark() -> None:
    _run_root_script("trace_cluster_trail_benchmark.py")


def trace_cluster_inheritance_eval() -> None:
    _run_root_script("trace_cluster_inheritance_eval.py")


def trace_cluster_batch_judge() -> None:
    _run_root_script("trace_cluster_batch_judge.py")


def trace_cluster_live_experiment() -> None:
    _run_root_script("trace_cluster_live_experiment.py")


def trace_cluster_compression_experiment() -> None:
    _run_root_script("trace_cluster_compression_experiment.py")


def trace_cluster_targeted_expansion() -> None:
    _run_root_script("trace_cluster_targeted_expansion.py")


def trace_cluster_hypothesis_pipeline() -> None:
    _run_root_script("trace_cluster_hypothesis_pipeline.py")


def trace_cluster_hypothesis_readout() -> None:
    _run_root_script("trace_cluster_hypothesis_readout.py")


def compare_pruning_methods() -> None:
    _run_root_script("compare_pruning_methods.py")


def compare_alignment_per_trace() -> None:
    _run_root_script("compare_alignment_per_trace.py")


def summarize_missed_root_causes() -> None:
    _run_root_script("summarize_missed_root_causes.py")


def analyze_semantic_location_mismatch() -> None:
    _run_root_script("analyze_semantic_location_mismatch.py")


def analyze_human_annotation_location_policy() -> None:
    _run_root_script("analyze_human_annotation_location_policy.py")


def who_when_adapter() -> None:
    _run_root_script("who_when_adapter.py")


def who_when_judge() -> None:
    _run_root_script("who_when_judge.py")


def who_when_evaluate() -> None:
    _run_root_script("who_when_evaluate.py")


def anomaly_detection_eval() -> None:
    _run_root_script("anomaly_detection_eval.py")


def jiuwenswarm_normal_replay() -> None:
    _run_root_script("integrations/jiuwenswarm/normal_replay_pipeline.py")


def trace_anomaly_detect() -> None:
    _run_root_script("trace_anomaly_detector.py")
