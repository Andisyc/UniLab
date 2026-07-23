#!/usr/bin/env python3
"""Materialize the HP-7c3 no-training Gate 0 freeze and oracle.

This tool reads and hashes the frozen server identity. It never invokes the
training CLI, constructs an environment, or creates the workflow run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

RUNTIME_ANCHOR_HEAD = "4fd2f67c08bb5372221ee1347561145b27238a75"
EXPECTED_COMPOSE_SHA256 = "741676aca03cbed11f9ad6e37105216b3acb545b35ebc86690202b2c0798798d"
EXPECTED_UPDATES = 12320
RUNTIME_SCOPE = (
    "src",
    "scripts/train_distill.py",
    "conf/distill",
    "pyproject.toml",
    "uv.lock",
)

SOURCE_HASHES = {
    "src/unilab/algos/torch/distill/offline.py": "24d2230e98673625bc3202e600692b6eafe67ef88f5c99e2f345d3c41301d76f",
    "src/unilab/algos/torch/distill/workflow.py": "22896114219d5e08df9893f158c38c7470675ac6546feac9ae0d74351f86d47c",
    "src/unilab/algos/torch/distill/async_runtime.py": "69cf2a5ebc516c718454a75a96745534504c927efdcfecf5c5c6f44756aad7ae",
    "src/unilab/algos/torch/distill/persistent_runtime.py": "4e88ee8af7cf09fbc8b30cbee45cd354886ba34dec8ba84bf7950f5b5d23f442",
    "src/unilab/algos/torch/distill/persistent_resources.py": "7f2c936bfbb7d84a6bc09505801917536aa67785a734e75060e03edfb8d1e463",
    "src/unilab/algos/torch/distill/g1_persistent_worker.py": "77af161718248f7e046bfcb3717cf68d8ffeda304fa81b17c9c0fdd6ae37bd7f",
    "scripts/train_distill.py": "b0e3f1f6d5760a7318acd5ba694f52992397bcf4a8e2852448444e8441eb273b",
    "conf/distill/config.yaml": "64de26d85ffa058e09cf0344b7545bf8153704cf5f76216402a7758e1c9234da",
    "conf/distill/workflow/g1_walk_stand.yaml": "8e64ab659f1eaae169ecb6dd8b4059e5cc172464e78956b03d07fd954539e4ba",
}

ARTIFACTS = {
    "walk_teacher": (
        "model/G1WalkFlat/model_5000.pt",
        "db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291",
    ),
    "stand_teacher": (
        "model/G1StandStill/model_5000.pt",
        "91e18d3d1f469b2bead350cd41b33494c39c8ec8d26f2daf802e0273afa2c6da",
    ),
    "walk_dataset": (
        "model/teacher/walk_flat_teacher_policy.pt",
        "efa0bec38f43b2ef3e811e1d35fc1f54a40d0d7377aafaa47e113b74aa5be027",
    ),
    "stand_dataset": (
        "model/teacher/stand_teacher_policy.pt",
        "f0e37612a74a355e429518e1241cc4c991111deb5e6483bb592b5732dc085b59",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(root: Path, *args: str) -> str:
    return subprocess.check_output(args, cwd=root, text=True).strip()


def artifact_identity(path: Path, expected: str) -> dict[str, Any]:
    observed = file_sha256(path) if path.is_file() else None
    return {
        "path": str(path),
        "expected_sha256": expected,
        "observed_sha256": observed,
        "size": path.stat().st_size if path.is_file() else None,
        "matches": observed == expected,
    }


def supervisor_source(root: Path, run_dir: Path) -> str:
    """Return the frozen Gate 1 launcher text without executing it."""
    log = root / "hp7c3_bounded_persistent_r1.log"
    timing = root / "hp7c3_bounded_persistent_r1.time"
    gpu_csv = root / "hp7c3_bounded_persistent_r1.nvidia.csv"
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {root}
test ! -e {run_dir}
nvidia-smi --query-compute-apps=timestamp,pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits --loop-ms=250 > {gpu_csv} &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true; wait "$SAMPLER_PID" 2>/dev/null || true' EXIT
CUDA_VISIBLE_DEVICES=0 \\
UNILAB_G1_WALK_TEACHER={root}/model/G1WalkFlat/model_5000.pt \\
UNILAB_G1_STAND_TEACHER={root}/model/G1StandStill/model_5000.pt \\
UNILAB_G1_WALK_DATASET={root}/model/teacher/walk_flat_teacher_policy.pt \\
UNILAB_G1_STAND_DATASET={root}/model/teacher/stand_teacher_policy.pt \\
HYDRA_FULL_ERROR=1 PYTHONWARNINGS=ignore \\
/usr/bin/time -v -o {timing} \\
uv run --no-sync train --algo distill --task g1_walk_flat --sim mujoco \\
  workflow=g1_walk_stand algo.seed=0 training.device=cuda:0 \\
  training.workflow.mode=fork \\
  training.workflow.parent_run_dir={root}/logs/distill_workflow/g1_walk_stand_persistent_test01 \\
  training.workflow.run_dir={run_dir} \\
  training.workflow.execution_mode=persistent_async \\
  training.workflow.collect_num_envs=16 \\
  training.workflow.dagger_samples_per_role=512 \\
  training.workflow.dagger_iterations=1 \\
  training.workflow.dagger_batch_size=512 \\
  training.workflow.dagger_updates_per_iteration=512 > {log} 2>&1
"""


def oracle_source() -> str:
    return """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    freeze_path = Path(args.freeze)
    freeze = json.loads(freeze_path.read_text())
    failures = list(freeze.get("failures", []))
    root = Path(freeze["repo"]["root"])

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if head != freeze["repo"]["head"]:
        failures.append(f"HEAD drift: {head}")

    for identity in freeze["repo"]["source_identity"].values():
        path = Path(identity["path"])
        if not path.is_file() or sha256(path) != identity["observed_sha256"]:
            failures.append(f"source drift: {path}")

    compose = Path(freeze["compose"]["path"])
    if not compose.is_file() or sha256(compose) != freeze["compose"]["observed_sha256"]:
        failures.append("compose drift")

    for name, identity in freeze["hard_artifacts"].items():
        path = Path(identity["path"])
        if not path.is_file() or sha256(path) != identity["observed_sha256"]:
            failures.append(f"artifact drift: {name}")

    supervisor = Path(freeze["supervisor"]["path"])
    if not supervisor.is_file() or sha256(supervisor) != freeze["supervisor"]["sha256"]:
        failures.append("supervisor drift")
    if not freeze.get("command", {}).get("argv") or not freeze.get("command", {}).get("env"):
        failures.append("frozen command identity missing")

    import mujoco
    import torch
    import unilab

    observed_dependency = {
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
    if observed_dependency != freeze["dependency_identity"]:
        failures.append("dependency/import identity drift")

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
         "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    if gpu.returncode or gpu.stdout.strip() != freeze["gpu_query"]["stdout"]:
        failures.append("GPU identity drift")

    if args.preflight:
        existing = [path for path in freeze["output_paths"] if Path(path).exists()]
        if existing:
            failures.append(f"output paths already exist: {existing}")
        result = {
            "mode": "preflight",
            "accepted": not failures,
            "failures": failures,
            "training_executed": False,
            "freeze_path": str(freeze_path),
            "freeze_sha256": sha256(freeze_path),
            "output_paths_absent": not existing,
        }
    else:
        run_dir = Path(freeze["run_dir"])
        manifest_path = run_dir / "run_manifest.json"
        metrics_path = run_dir / "distillation_metrics.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if not manifest:
            failures.append("run manifest missing")
        iterations = manifest.get("dagger_iterations", [])
        if manifest.get("completed_dagger_iterations") != 1 or len(iterations) != 1:
            failures.append("expected exactly one completed DAgger iteration")
        iteration = iterations[0] if len(iterations) == 1 else {}
        if iteration.get("updates") != freeze["workload"]["effective_updates"]:
            failures.append("effective update count mismatch")
        if iteration.get("collection_execution_mode") != "persistent_async":
            failures.append("execution mode mismatch")
        scenarios = iteration.get("scenario_artifacts", [])
        expected_scenarios = freeze["workload"]["scenario_order"]
        if [item.get("scenario") for item in scenarios] != expected_scenarios:
            failures.append("scenario order mismatch")
        if any(item.get("num_samples") != freeze["workload"]["rows_per_scenario"] for item in scenarios):
            failures.append("scenario sample count mismatch")
        weight_versions = {item.get("input_weight_version") for item in scenarios}
        if len(weight_versions) != 1 or weight_versions != {iteration.get("input_weight_version")}:
            failures.append("scenario weight-version identity mismatch")
        worker_pids = {item.get("collector_worker_pid") for item in scenarios}
        if len(worker_pids) != 1 or None in worker_pids:
            failures.append("persistent collector worker identity mismatch")
        if iteration.get("input_checkpoint_sha256") != freeze["hard_artifacts"]["parent_checkpoint"]["observed_sha256"]:
            failures.append("input checkpoint lineage mismatch")
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        records = metrics.get("records", [])
        if not records:
            failures.append("distillation metrics missing or empty")
        elif any(not record.get("success", False) for record in records):
            failures.append("one or more metric stages failed")
        required_stages = {
            "cumulative_aggregation", "learner_batch_staging", "learner_forward",
            "learner_backward", "optimizer_step", "checkpoint_save", "cleanup",
        }
        observed_stages = {record.get("stage") for record in records}
        if not required_stages.issubset(observed_stages):
            failures.append("required workflow metric stages missing")
        cleanup_records = [record for record in records if record.get("stage") == "cleanup"]
        if len(cleanup_records) != 1 or cleanup_records[0].get("cleanup_state") != "complete":
            failures.append("cleanup metric contract mismatch")
        if manifest.get("performance_cleanup", {}).get("state") != "complete":
            failures.append("manifest cleanup state incomplete")
        if metrics_path.is_file():
            if manifest.get("distillation_metrics_sha256") != sha256(metrics_path):
                failures.append("metrics hash mismatch")
            if manifest.get("distillation_metrics_record_count") != len(records):
                failures.append("metrics record count mismatch")
        for key in ("aggregate_dataset_path", "checkpoint_path"):
            artifact = Path(str(iteration.get(key, "")))
            if not artifact.is_file():
                failures.append(f"missing output artifact: {key}")
            elif iteration.get(key.replace("_path", "_sha256")) != sha256(artifact):
                failures.append(f"output artifact hash mismatch: {key}")
        for name, path_text in freeze["telemetry"].items():
            path = Path(path_text)
            if not path.is_file() or path.stat().st_size <= 0:
                failures.append(f"telemetry artifact missing or empty: {name}")
        result = {
            "mode": "postrun",
            "accepted": not failures,
            "failures": failures,
            "training_executed": True,
            "freeze_path": str(freeze_path),
            "freeze_sha256": sha256(freeze_path),
            "manifest_path": str(manifest_path),
            "metrics_path": str(metrics_path),
            "observed_updates": iteration.get("updates"),
        }

    Path(args.result).write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""


def materialize(root: Path) -> dict[str, Any]:
    parent = root / "logs/distill_workflow/g1_walk_stand_persistent_test01"
    run_dir = root / "logs/distill_workflow/hp7c3_bounded_persistent_20260717_r1"
    compose = root / "hp7c3_gate0_compose_r2.yaml"
    compose_stderr = root / "hp7c3_gate0_compose_r2.stderr"
    probe_path = root / "hp7c3_gate0_identity_probe_r1.json"
    freeze_path = root / "hp7c3_bounded_persistent_freeze_r6.json"
    oracle_path = root / "hp7c3_bounded_persistent_oracle_v6.py"
    supervisor_path = root / "hp7c3_bounded_persistent_supervisor_r6.sh"
    preflight_path = root / "hp7c3_bounded_persistent_oracle_preflight_r6.json"
    output_paths = [
        run_dir,
        root / "hp7c3_bounded_persistent_oracle_result_r1.json",
        root / "hp7c3_bounded_persistent_r1.log",
        root / "hp7c3_bounded_persistent_r1.time",
        root / "hp7c3_bounded_persistent_r1.nvidia.csv",
    ]
    for path in (freeze_path, oracle_path, supervisor_path, preflight_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Gate 0 artifact: {path}")

    failures: list[str] = []
    head = command_output(root, "git", "rev-parse", "HEAD")
    committed_runtime_diff = subprocess.run(
        ["git", "diff", "--quiet", RUNTIME_ANCHOR_HEAD, head, "--", *RUNTIME_SCOPE],
        cwd=root,
        check=False,
    ).returncode
    if committed_runtime_diff:
        failures.append(
            f"runtime commit diff from anchor: anchor={RUNTIME_ANCHOR_HEAD} observed={head}"
        )
    worktree_runtime_diff = subprocess.run(
        ["git", "diff", "--quiet", "--", *RUNTIME_SCOPE],
        cwd=root,
        check=False,
    ).returncode
    if worktree_runtime_diff:
        failures.append("runtime worktree diff is not clean")

    source_identity = {
        relative: artifact_identity(root / relative, expected)
        for relative, expected in SOURCE_HASHES.items()
    }
    failures.extend(
        f"source mismatch: {relative}"
        for relative, identity in source_identity.items()
        if not identity["matches"]
    )

    compose_identity = artifact_identity(compose, EXPECTED_COMPOSE_SHA256)
    if not compose_identity["matches"]:
        failures.append("compose identity mismatch")
    if not compose_stderr.is_file() or compose_stderr.stat().st_size:
        failures.append("compose stderr missing or non-empty")

    probe = json.loads(probe_path.read_text()) if probe_path.is_file() else {}
    observed_updates = probe.get("workload", {}).get("effective_updates")
    if observed_updates != EXPECTED_UPDATES:
        failures.append(
            f"effective updates mismatch: expected={EXPECTED_UPDATES} observed={observed_updates}"
        )

    parent_manifest_path = parent / "run_manifest.json"
    parent_manifest = (
        json.loads(parent_manifest_path.read_text()) if parent_manifest_path.is_file() else {}
    )
    iterations = parent_manifest.get("dagger_iterations", [])
    latest = iterations[-1] if iterations else {}
    if not latest:
        failures.append("parent has no completed DAgger iteration")

    hard_artifacts = {
        name: artifact_identity(root / relative, expected)
        for name, (relative, expected) in ARTIFACTS.items()
    }
    for name, path_key, hash_key in (
        ("parent_aggregate", "aggregate_dataset_path", "aggregate_dataset_sha256"),
        ("parent_checkpoint", "checkpoint_path", "checkpoint_sha256"),
    ):
        path = Path(str(latest.get(path_key, "")))
        hard_artifacts[name] = artifact_identity(path, str(latest.get(hash_key, "")))
    failures.extend(
        f"hard artifact mismatch: {name}"
        for name, identity in hard_artifacts.items()
        if not identity["matches"]
    )

    metrics_path = parent / "distillation_metrics.json"
    existing_outputs = [str(path) for path in output_paths if path.exists()]
    if existing_outputs:
        failures.append(f"final paths already exist: {existing_outputs}")

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
    if gpu.returncode:
        failures.append("nvidia-smi query failed")

    import mujoco
    import torch

    import unilab

    dependency_identity = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_path": str(Path(torch.__file__).resolve()),
        "mujoco_version": getattr(mujoco, "__version__", None),
        "mujoco_path": str(Path(mujoco.__file__).resolve()),
        "unilab_path": str(Path(unilab.__file__).resolve()),
        "uv_version": command_output(root, "uv", "--version"),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR"),
        "UV_PROJECT_ENVIRONMENT": os.environ.get("UV_PROJECT_ENVIRONMENT"),
    }
    command_argv = [
        "uv",
        "run",
        "--no-sync",
        "train",
        "--algo",
        "distill",
        "--task",
        "g1_walk_flat",
        "--sim",
        "mujoco",
        "workflow=g1_walk_stand",
        "algo.seed=0",
        "training.device=cuda:0",
        "training.workflow.mode=fork",
        f"training.workflow.parent_run_dir={parent}",
        f"training.workflow.run_dir={run_dir}",
        "training.workflow.execution_mode=persistent_async",
        "training.workflow.collect_num_envs=16",
        "training.workflow.dagger_samples_per_role=512",
        "training.workflow.dagger_iterations=1",
        "training.workflow.dagger_batch_size=512",
        "training.workflow.dagger_updates_per_iteration=512",
    ]
    command_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "UNILAB_G1_WALK_TEACHER": str(root / ARTIFACTS["walk_teacher"][0]),
        "UNILAB_G1_STAND_TEACHER": str(root / ARTIFACTS["stand_teacher"][0]),
        "UNILAB_G1_WALK_DATASET": str(root / ARTIFACTS["walk_dataset"][0]),
        "UNILAB_G1_STAND_DATASET": str(root / ARTIFACTS["stand_dataset"][0]),
        "HYDRA_FULL_ERROR": "1",
        "PYTHONWARNINGS": "ignore",
    }
    supervisor = supervisor_source(root, run_dir)

    freeze = {
        "schema_version": 2,
        "gate": "HP-7c3 Gate 0",
        "training_authorized": False,
        "accepted": not failures,
        "failures": failures,
        "repo": {
            "root": str(root),
            "runtime_anchor_head": RUNTIME_ANCHOR_HEAD,
            "head": head,
            "committed_runtime_diff_clean": committed_runtime_diff == 0,
            "worktree_runtime_diff_clean": worktree_runtime_diff == 0,
            "runtime_scope": list(RUNTIME_SCOPE),
            "source_identity": source_identity,
        },
        "compose": {
            **compose_identity,
            "stderr_empty": compose_stderr.is_file() and not compose_stderr.stat().st_size,
        },
        "parent": {
            "run_dir": str(parent),
            "manifest_path": str(parent_manifest_path),
            "manifest_sha256": file_sha256(parent_manifest_path)
            if parent_manifest_path.is_file()
            else None,
            "latest_iteration": latest.get("iteration"),
            "aggregate_num_samples": latest.get("aggregate_num_samples"),
        },
        "hard_artifacts": hard_artifacts,
        "parent_metrics": {
            "role": "audit_only_non_fork_input",
            "path": str(metrics_path),
            "manifest_sha256": parent_manifest.get("distillation_metrics_sha256"),
            "observed_sha256": file_sha256(metrics_path) if metrics_path.is_file() else None,
            "acceptance_effect": "recorded_non_blocking",
        },
        "workload": {
            "seed": 0,
            "device": "cuda:0",
            "execution_mode": "persistent_async",
            "scenario_order": ["walk_flat", "static_stand", "walk_to_stop"],
            "rows_per_scenario": 512,
            "collect_num_envs": 16,
            "batch_size": 512,
            "configured_update_floor": 512,
            "required_updates": EXPECTED_UPDATES,
            "effective_updates": EXPECTED_UPDATES,
            "outer_iterations": 1,
        },
        "gpu_query": {
            "returncode": gpu.returncode,
            "stdout": gpu.stdout.strip(),
            "stderr": gpu.stderr.strip(),
        },
        "dependency_identity": dependency_identity,
        "command": {"argv": command_argv, "env": command_env},
        "supervisor": {
            "path": str(supervisor_path),
            "sha256": hashlib.sha256(supervisor.encode()).hexdigest(),
        },
        "telemetry": {
            "console_log": str(root / "hp7c3_bounded_persistent_r1.log"),
            "time_log": str(root / "hp7c3_bounded_persistent_r1.time"),
            "gpu_csv": str(root / "hp7c3_bounded_persistent_r1.nvidia.csv"),
        },
        "run_dir": str(run_dir),
        "output_paths": [str(path) for path in output_paths],
        "output_paths_absent": not existing_outputs,
    }
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    oracle_path.write_text(oracle_source())
    oracle_path.chmod(0o755)
    supervisor_path.write_text(supervisor)
    supervisor_path.chmod(0o755)

    subprocess.run(
        ["uv", "run", "--no-sync", "python", "-m", "py_compile", str(oracle_path)],
        cwd=root,
        check=True,
    )
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            str(oracle_path),
            "--preflight",
            "--freeze",
            str(freeze_path),
            "--result",
            str(preflight_path),
        ],
        cwd=root,
        check=False,
    )
    summary = {
        "materializer_training_executed": False,
        "freeze_path": str(freeze_path),
        "freeze_sha256": file_sha256(freeze_path),
        "oracle_path": str(oracle_path),
        "oracle_sha256": file_sha256(oracle_path),
        "supervisor_path": str(supervisor_path),
        "supervisor_sha256": file_sha256(supervisor_path),
        "preflight_path": str(preflight_path),
        "preflight_sha256": file_sha256(preflight_path) if preflight_path.is_file() else None,
        "preflight_returncode": result.returncode,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/ssd1/cyx/UniLab"))
    args = parser.parse_args()
    summary = materialize(args.root.resolve())
    return int(summary["preflight_returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
