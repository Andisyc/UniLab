#!/usr/bin/env python3
"""Validate an existing formal DAgger run without executing or repairing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_file_identity(failures: list[str], *, name: str, identity: dict[str, Any]) -> None:
    path = Path(identity["path"])
    if not path.is_file() or file_sha256(path) != identity["sha256"]:
        failures.append(f"frozen input drift: {name}")


def validate_postflight(
    freeze_path: Path,
    *,
    observed_head: str,
    observed_dependency: dict[str, Any],
    observed_gpu: dict[str, Any],
) -> dict[str, Any]:
    """Apply the complete v2 oracle to one immutable formal run."""

    freeze = json.loads(freeze_path.read_text())
    failures = list(freeze.get("failures", []))
    warnings: list[str] = []
    if observed_head != freeze["repo"]["head"]:
        warnings.append("HEAD drift after oracle repair; frozen runtime bytes are checked below")
    for group in ("source_identity", "hard_artifacts"):
        for name, identity in freeze[group].items():
            _check_file_identity(failures, name=f"{group}:{name}", identity=identity)
    if observed_dependency != freeze["dependency_identity"]:
        failures.append("dependency/import identity drift")
    if observed_gpu.get("returncode") != 0 or (
        observed_gpu.get("stdout") != freeze["gpu_query"].get("stdout")
    ):
        failures.append("GPU identity drift")

    workload = freeze["command"]["workload"]
    expected_count = int(workload["dagger_iterations"])
    expected_updates = list(workload["effective_updates_by_iteration"])
    expected_mode = str(workload["execution_mode"])
    expected_samples = int(workload["samples_per_role"])
    expected_scenarios = ["walk_flat", "static_stand", "walk_to_stop"]
    run_dir = Path(freeze["output_paths"]["run_dir"])
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "distillation_metrics.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    if not manifest:
        failures.append("formal run manifest missing")
    if not metrics:
        failures.append("formal metrics missing")

    iterations = manifest.get("dagger_iterations", [])
    if manifest.get("completed_dagger_iterations") != expected_count:
        failures.append("completed DAgger iteration count mismatch")
    if len(iterations) != expected_count:
        failures.append("manifest iteration count mismatch")

    lineage_source = freeze["command"].get("lineage", {}).get("source")
    if lineage_source == "fresh_teacher_bootstrap":
        bootstrap_path = Path(str(manifest.get("bootstrap_checkpoint_path", "")))
        if not bootstrap_path.is_file():
            failures.append("fresh bootstrap checkpoint missing")
            expected_input_sha = ""
        else:
            expected_input_sha = file_sha256(bootstrap_path)
            if manifest.get("bootstrap_checkpoint_sha256") != expected_input_sha:
                failures.append("fresh bootstrap checkpoint hash mismatch")
    else:
        expected_input_sha = freeze["hard_artifacts"]["parent_checkpoint"]["sha256"]
    previous_weight_version: int | None = None
    final_checkpoint_sha: str | None = None
    final_checkpoint_path: str | None = None
    for offset, iteration in enumerate(iterations, start=1):
        if (
            offset > len(expected_updates)
            or iteration.get("updates") != expected_updates[offset - 1]
        ):
            failures.append(f"iteration {offset} effective update count mismatch")
        if iteration.get("collection_execution_mode") != expected_mode:
            failures.append(f"iteration {offset} execution mode mismatch")
        if iteration.get("input_checkpoint_sha256") != expected_input_sha:
            failures.append(f"iteration {offset} input checkpoint lineage mismatch")
        weight_version = iteration.get("input_weight_version")
        if not isinstance(weight_version, int):
            failures.append(f"iteration {offset} input weight version missing")
        elif previous_weight_version is not None and weight_version != previous_weight_version + 1:
            failures.append(f"iteration {offset} weight version is not monotonic")
        scenarios = iteration.get("scenario_artifacts", [])
        if [item.get("scenario") for item in scenarios] != expected_scenarios:
            failures.append(f"iteration {offset} scenario order mismatch")
        if any(item.get("num_samples") != expected_samples for item in scenarios):
            failures.append(f"iteration {offset} scenario sample count mismatch")
        if any(item.get("input_weight_version") != weight_version for item in scenarios):
            failures.append(f"iteration {offset} scenario weight-version mismatch")
        worker_pids = {item.get("collector_worker_pid") for item in scenarios}
        if len(worker_pids) != 1 or None in worker_pids:
            failures.append(f"iteration {offset} persistent worker identity mismatch")
        for key in ("aggregate_dataset_path", "checkpoint_path"):
            artifact = Path(str(iteration.get(key, "")))
            label = f"iteration {offset} {key}"
            if not artifact.is_file():
                failures.append(f"missing output artifact: {label}")
            else:
                observed_sha = file_sha256(artifact)
                if iteration.get(key.replace("_path", "_sha256")) != observed_sha:
                    failures.append(f"output artifact hash mismatch: {label}")
                if key == "checkpoint_path":
                    final_checkpoint_sha = observed_sha
                    final_checkpoint_path = str(artifact)
                    expected_input_sha = observed_sha
        if isinstance(weight_version, int):
            previous_weight_version = weight_version

    records = metrics.get("records", [])
    if not records:
        failures.append("distillation metrics missing or empty")
    elif any(not record.get("success", False) for record in records):
        failures.append("one or more metric stages failed")
    required_stages = {
        "cumulative_aggregation",
        "learner_batch_staging",
        "learner_forward",
        "learner_backward",
        "optimizer_step",
        "checkpoint_save",
    }
    for offset, iteration in enumerate(iterations, start=1):
        iteration_records = [
            record
            for record in records
            if record.get("identity", {}).get("outer_iteration") == offset
        ]
        observed_stages = {record.get("stage") for record in iteration_records}
        if not required_stages.issubset(observed_stages):
            failures.append(f"iteration {offset} workflow metric stages missing")
        scenario_records = [
            record
            for record in iteration_records
            if record.get("identity", {}).get("scenario") in expected_scenarios
        ]
        if {record["identity"]["scenario"] for record in scenario_records} != set(
            expected_scenarios
        ):
            failures.append(f"iteration {offset} scenario metrics missing")
        for record in scenario_records:
            identity = record["identity"]
            if identity.get("execution_mode") != expected_mode:
                failures.append(f"iteration {offset} metric execution mode mismatch")
            if identity.get("weight_version") != iteration.get("input_weight_version"):
                failures.append(f"iteration {offset} metric weight-version mismatch")
            if identity.get("checkpoint_sha256") != iteration.get("input_checkpoint_sha256"):
                failures.append(f"iteration {offset} metric checkpoint lineage mismatch")

    cleanup_records = [record for record in records if record.get("stage") == "cleanup"]
    if len(cleanup_records) != 1 or cleanup_records[0].get("cleanup_state") != "complete":
        failures.append("cleanup metric contract mismatch")
    if manifest.get("performance_cleanup", {}).get("state") != "complete":
        failures.append("manifest cleanup state incomplete")
    if metrics_path.is_file():
        if manifest.get("distillation_metrics_sha256") != file_sha256(metrics_path):
            failures.append("metrics hash mismatch")
        if manifest.get("distillation_metrics_record_count") != len(records):
            failures.append("metrics record count mismatch")
    for name in ("log", "time", "gpu_telemetry"):
        path = Path(freeze["output_paths"][name])
        if not path.is_file() or path.stat().st_size <= 0:
            failures.append(f"telemetry artifact missing or empty: {name}")

    return {
        "mode": "postflight-v2",
        "accepted": not failures,
        "failures": failures,
        "warnings": warnings,
        "training_executed": True,
        "freeze_path": str(freeze_path),
        "freeze_sha256": file_sha256(freeze_path),
        "oracle_v2_path": str(Path(__file__).resolve()),
        "oracle_v2_sha256": file_sha256(Path(__file__).resolve()),
        "manifest_path": str(manifest_path),
        "metrics_path": str(metrics_path),
        "validated_iterations": len(iterations),
        "final_checkpoint_path": final_checkpoint_path,
        "final_checkpoint_sha256": final_checkpoint_sha,
    }


def _runtime_identity() -> tuple[str, dict[str, Any], dict[str, Any]]:
    import mujoco
    import torch

    import unilab

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dependency = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_path": str(Path(torch.__file__).resolve()),
        "mujoco_version": getattr(mujoco, "__version__", None),
        "mujoco_path": str(Path(mujoco.__file__).resolve()),
        "unilab_path": str(Path(unilab.__file__).resolve()),
        "uv_version": subprocess.check_output(["uv", "--version"], text=True).strip(),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR"),
        "UV_PROJECT_ENVIRONMENT": os.environ.get("UV_PROJECT_ENVIRONMENT"),
    }
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        head,
        dependency,
        {
            "returncode": gpu.returncode,
            "stdout": gpu.stdout.strip(),
            "stderr": gpu.stderr,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    head, dependency, gpu = _runtime_identity()
    result = validate_postflight(
        args.freeze.resolve(),
        observed_head=head,
        observed_dependency=dependency,
        observed_gpu=gpu,
    )
    if args.result.exists():
        raise FileExistsError(f"refusing to overwrite acceptance: {args.result}")
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
