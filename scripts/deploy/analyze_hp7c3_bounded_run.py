#!/usr/bin/env python3
"""Read-only summary of one HP-7c3 bounded persistent workflow."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _elapsed_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return float(value)


def parse_time_v(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    text = path.read_text(errors="replace")
    elapsed = re.search(r"^\s*Elapsed \(wall clock\) time.*\):\s*(\S+)\s*$", text, re.MULTILINE)
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    status = re.search(r"Exit status:\s*(\d+)", text)
    result.update(
        {
            "elapsed_text": elapsed.group(1).strip() if elapsed else None,
            "elapsed_seconds": _elapsed_seconds(elapsed.group(1)) if elapsed else None,
            "maximum_resident_kbytes": int(rss.group(1)) if rss else None,
            "exit_status": int(status.group(1)) if status else None,
        }
    )
    return result


def parse_gpu_csv(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    rows = []
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 4:
                continue
            memory_text = row[3].strip()
            match = re.search(r"\d+", memory_text)
            rows.append(
                {
                    "timestamp": row[0].strip(),
                    "pid": row[1].strip(),
                    "gpu_uuid": row[2].strip(),
                    "used_gpu_memory_mib": int(match.group()) if match else None,
                }
            )
    memories = [
        row["used_gpu_memory_mib"] for row in rows if row["used_gpu_memory_mib"] is not None
    ]
    result.update(
        {
            "sample_count": len(rows),
            "pids": sorted({row["pid"] for row in rows}),
            "gpu_uuids": sorted({row["gpu_uuid"] for row in rows}),
            "peak_used_gpu_memory_mib": max(memories) if memories else None,
            "mean_used_gpu_memory_mib": sum(memories) / len(memories) if memories else None,
        }
    )
    return result


def analyze_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    records = payload.get("records", [])
    stage_seconds: dict[str, float] = defaultdict(float)
    stage_counts: Counter[str] = Counter()
    scenario_seconds: dict[str, float] = defaultdict(float)
    failures = []
    worker_pids = set()
    weight_versions = set()
    outer_iterations = set()
    execution_modes = set()

    for record in records:
        stage = str(record.get("stage"))
        duration = float(record.get("duration_seconds", 0.0))
        identity = record.get("identity", {})
        stage_seconds[stage] += duration
        stage_counts[stage] += 1
        scenario_seconds[str(identity.get("scenario"))] += duration
        if not record.get("success", False):
            failures.append(record)
        if identity.get("worker_pid") is not None:
            worker_pids.add(identity["worker_pid"])
        if identity.get("weight_version") is not None:
            weight_versions.add(identity["weight_version"])
        if identity.get("outer_iteration") is not None:
            outer_iterations.add(identity["outer_iteration"])
        if identity.get("execution_mode") is not None:
            execution_modes.add(identity["execution_mode"])

    ordered_stages = sorted(stage_seconds, key=stage_seconds.get, reverse=True)
    total_stage_seconds = sum(stage_seconds.values())
    return {
        "path": str(path),
        "record_count": len(records),
        "failure_count": len(failures),
        "stage_seconds": {stage: stage_seconds[stage] for stage in ordered_stages},
        "stage_counts": dict(stage_counts),
        "stage_share_of_summed_durations": {
            stage: stage_seconds[stage] / total_stage_seconds if total_stage_seconds else None
            for stage in ordered_stages
        },
        "scenario_seconds": dict(sorted(scenario_seconds.items())),
        "summed_stage_seconds": total_stage_seconds,
        "worker_pids": sorted(worker_pids),
        "weight_versions": sorted(weight_versions),
        "outer_iterations": sorted(outer_iterations),
        "execution_modes": sorted(execution_modes),
    }


def analyze(root: Path) -> dict[str, Any]:
    run_dir = root / "logs/distill_workflow/hp7c3_bounded_persistent_20260717_r1"
    metrics_path = run_dir / "distillation_metrics.json"
    manifest_path = run_dir / "run_manifest.json"
    oracle_path = root / "hp7c3_bounded_persistent_oracle_result_r1.json"
    manifest = json.loads(manifest_path.read_text())
    oracle = json.loads(oracle_path.read_text())
    iteration = manifest["dagger_iterations"][0]
    metrics = analyze_metrics(metrics_path)
    timing = parse_time_v(root / "hp7c3_bounded_persistent_r1.time")
    gpu = parse_gpu_csv(root / "hp7c3_bounded_persistent_r1.nvidia.csv")
    elapsed = timing.get("elapsed_seconds")
    staging = metrics["stage_seconds"].get("learner_batch_staging", 0.0)
    return {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "oracle": {
            "accepted": oracle.get("accepted"),
            "failures": oracle.get("failures"),
            "freeze_sha256": oracle.get("freeze_sha256"),
        },
        "manifest": {
            "completed_dagger_iterations": manifest.get("completed_dagger_iterations"),
            "updates": iteration.get("updates"),
            "aggregate_num_samples": iteration.get("aggregate_num_samples"),
            "input_weight_version": iteration.get("input_weight_version"),
            "checkpoint_path": iteration.get("checkpoint_path"),
            "cleanup_state": manifest.get("performance_cleanup", {}).get("state"),
        },
        "metrics": metrics,
        "time_v": timing,
        "gpu": gpu,
        "derived": {
            "staging_seconds_per_update": staging / iteration["updates"],
            "staging_share_of_wall_time": staging / elapsed if elapsed else None,
            "updates_per_wall_second": iteration["updates"] / elapsed if elapsed else None,
        },
        "claim_boundary": {
            "bounded_run_valid": oracle.get("accepted") is True,
            "end_to_end_speedup_claim_authorized": False,
            "promotion_authorized": False,
            "default_on_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/ssd1/cyx/UniLab"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root.resolve())
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
