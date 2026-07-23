"""Pure owner for a formal DAgger command and output identity.

This module materializes no files and executes no commands.  It converts one
explicitly reviewed formal-training specification into the argv, environment,
lineage, workload, and output-path identity consumed by the deploy-side Gate 0
materializer.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FormalDaggerIdentitySpec:
    """Human-approved inputs required before formal Gate 0 materialization."""

    repo_root: Path
    parent_run_dir: Path | None
    run_dir: Path
    parent_iteration: int
    dagger_iterations: int
    configured_update_floor: int
    effective_updates_by_iteration: tuple[int, ...]
    seed: int
    device: str
    collect_num_envs: int
    samples_per_role: int
    batch_size: int
    execution_mode: str
    mode: str = "fork"
    artifact_dir: Path | None = None
    bootstrap_updates: int = 0
    adopt_legacy_artifacts: bool = False
    transition_max_env_steps: int | None = None


@dataclass(frozen=True)
class FormalDaggerAutoOutputIdentity:
    """One Gate 0-resolved, time-sorted formal output identity."""

    run_name: str
    timestamp: str
    stem: str
    run_dir: Path
    artifact_dir: Path | None


_FORMAL_RUN_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def resolve_time_sorted_formal_output_identity(
    *,
    repo_root: Path,
    run_name: str,
    mode: str,
    now: datetime,
    run_root: Path | None = None,
    artifact_root: Path | None = None,
) -> FormalDaggerAutoOutputIdentity:
    """将 human `run_name` 解析为 Gate 0 一次性冻结的输出身份.

    函数名说明:
        这是 formal output identity owner, 只决定路径和排序语义, 不创建目录或启动训练.

    主链路:
        上游: Gate 0 materialization spec.
        下游: FormalDaggerIdentitySpec 的 run_dir/artifact_dir, freeze 和 supervisor.

    语义:
        `timestamp` 在 Gate 0 解析一次. 同一 supervisor 只能使用已冻结路径,
        不会在每次启动时生成新时间戳.
    """

    if mode not in {"fork", "fresh"}:
        raise ValueError(f"unsupported formal workflow mode: {mode!r}")
    if not _FORMAL_RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "run_name must contain only lowercase letters, digits, underscores, and hyphens "
            "and start with a letter or digit"
        )

    # B1: 将一次 Gate 0 时钟解析为可排序且人可读的 identity stem.
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    stem = f"{timestamp}_{run_name}"

    # B2: 只在 cold path 派生独立的 workflow 和 fresh role-artifact 根目录.
    resolved_repo_root = repo_root.resolve()
    resolved_run_root = (
        resolved_repo_root / "logs" / "distill_workflow"
        if run_root is None
        else (run_root if run_root.is_absolute() else resolved_repo_root / run_root).resolve()
    )
    resolved_artifact_root = (
        resolved_repo_root / "logs" / "distill_role_artifacts"
        if artifact_root is None
        else (
            artifact_root if artifact_root.is_absolute() else resolved_repo_root / artifact_root
        ).resolve()
    )

    # B3: 返回纯数据 identity, 由 deploy connector 写入 freeze 而非此 owner 写文件.
    return FormalDaggerAutoOutputIdentity(
        run_name=run_name,
        timestamp=timestamp,
        stem=stem,
        run_dir=resolved_run_root / stem,
        artifact_dir=(resolved_artifact_root / stem if mode == "fresh" else None),
    )


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size": path.stat().st_size}


def _reject_sentinel_path(path: Path) -> None:
    normalized = str(path).lower()
    if "r6" in normalized or "hp7c3" in normalized:
        raise ValueError(f"r6 sentinel path is forbidden in formal lineage: {path}")


def _validate_spec(spec: FormalDaggerIdentitySpec) -> None:
    if spec.mode not in {"fork", "fresh"}:
        raise ValueError(f"unsupported formal workflow mode: {spec.mode!r}")
    if spec.mode == "fork":
        if spec.parent_iteration != 3 or spec.parent_run_dir is None:
            raise ValueError("formal fork lineage must start from parent iteration 3")
        _reject_sentinel_path(spec.parent_run_dir)
    else:
        if spec.parent_iteration != 0 or spec.parent_run_dir is not None:
            raise ValueError("formal fresh lineage must not name a parent run")
        if spec.artifact_dir is None:
            raise ValueError("formal fresh identity requires a new artifact_dir")
        _reject_sentinel_path(spec.artifact_dir)
        if spec.bootstrap_updates <= 0:
            raise ValueError("formal fresh bootstrap_updates must be positive")
        if spec.adopt_legacy_artifacts:
            raise ValueError("formal fresh identity forbids legacy artifact adoption")
    if spec.transition_max_env_steps is not None and spec.transition_max_env_steps <= 0:
        raise ValueError("transition_max_env_steps must be positive when configured")
    _reject_sentinel_path(spec.run_dir)
    if spec.dagger_iterations <= 0:
        raise ValueError("dagger_iterations must be positive and explicit")
    if spec.configured_update_floor <= 0:
        raise ValueError("configured_update_floor must be positive")
    if len(spec.effective_updates_by_iteration) != spec.dagger_iterations:
        raise ValueError(
            "effective_updates_by_iteration must contain one value per DAgger iteration"
        )
    if any(value <= 0 for value in spec.effective_updates_by_iteration):
        raise ValueError("effective_updates_by_iteration must contain positive values")
    if spec.collect_num_envs <= 0 or spec.samples_per_role <= 0 or spec.batch_size <= 0:
        raise ValueError("environment, sample, and batch counts must be positive")
    if spec.execution_mode != "persistent_async":
        raise ValueError("formal identity requires explicit persistent_async execution")
    if not spec.device.startswith("cuda:"):
        raise ValueError("formal identity requires an explicit cuda:<index> device")


def _identity_paths(spec: FormalDaggerIdentitySpec) -> tuple[dict[str, str], dict[str, str]]:
    stem = spec.run_dir.name
    outputs = {
        "run_dir": str(spec.run_dir),
        "log": str(spec.repo_root / f"{stem}.log"),
        "time": str(spec.repo_root / f"{stem}.time"),
        "gpu_telemetry": str(spec.repo_root / f"{stem}.nvidia.csv"),
        "acceptance": str(spec.repo_root / f"{stem}.acceptance.json"),
    }
    if spec.mode == "fresh" and spec.artifact_dir is not None:
        outputs["artifact_dir"] = str(spec.artifact_dir)
    materialization = {
        "compose": str(spec.repo_root / f"{stem}.compose.yaml"),
        "freeze": str(spec.repo_root / f"{stem}.freeze.json"),
        "supervisor": str(spec.repo_root / f"{stem}.supervisor.sh"),
        "oracle": str(spec.repo_root / f"{stem}.oracle.py"),
        "preflight": str(spec.repo_root / f"{stem}.preflight.json"),
    }
    return outputs, materialization


def build_formal_command_identity(spec: FormalDaggerIdentitySpec) -> dict[str, Any]:
    """Build one fail-closed formal command identity without executing it."""

    _validate_spec(spec)
    outputs, materialization = _identity_paths(spec)
    existing = [
        path for path in (*outputs.values(), *materialization.values()) if Path(path).exists()
    ]
    if existing:
        raise FileExistsError(f"formal output already exists: {existing}")

    device_index = spec.device.split(":", maxsplit=1)[1]
    argv = [
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
        f"algo.seed={spec.seed}",
        f"training.device={spec.device}",
        f"training.workflow.mode={spec.mode}",
        f"training.workflow.run_dir={spec.run_dir}",
        f"training.workflow.execution_mode={spec.execution_mode}",
        f"training.workflow.collect_num_envs={spec.collect_num_envs}",
        f"training.workflow.dagger_samples_per_role={spec.samples_per_role}",
        f"training.workflow.dagger_iterations={spec.dagger_iterations}",
        f"training.workflow.dagger_batch_size={spec.batch_size}",
        f"training.workflow.dagger_updates_per_iteration={spec.configured_update_floor}",
    ]
    if spec.mode == "fork":
        argv.append(f"training.workflow.parent_run_dir={spec.parent_run_dir}")
    else:
        argv.extend(
            [
                f"training.workflow.artifact_dir={spec.artifact_dir}",
                f"training.workflow.bootstrap_updates={spec.bootstrap_updates}",
                "training.workflow.adopt_legacy_artifacts=false",
            ]
        )
    if spec.transition_max_env_steps is not None:
        argv.append(f"training.workflow.transition_max_env_steps={spec.transition_max_env_steps}")
    return {
        "schema_version": 1,
        "training_executed": False,
        "repo_root": str(spec.repo_root),
        "lineage": {
            "parent_iteration": spec.parent_iteration if spec.mode == "fork" else None,
            "source": (
                "original_parent_iteration_3" if spec.mode == "fork" else "fresh_teacher_bootstrap"
            ),
            "r6_sentinel_promoted": False,
        },
        "workload": {
            "dagger_iterations": spec.dagger_iterations,
            "configured_update_floor": spec.configured_update_floor,
            "effective_updates_by_iteration": list(spec.effective_updates_by_iteration),
            "total_effective_updates": sum(spec.effective_updates_by_iteration),
            "collect_num_envs": spec.collect_num_envs,
            "samples_per_role": spec.samples_per_role,
            "batch_size": spec.batch_size,
            "seed": spec.seed,
            "device": spec.device,
            "execution_mode": spec.execution_mode,
            "mode": spec.mode,
            "bootstrap_updates": spec.bootstrap_updates,
            "transition_max_env_steps": spec.transition_max_env_steps,
        },
        "argv": argv,
        "env": {
            "CUDA_VISIBLE_DEVICES": device_index,
            "HYDRA_FULL_ERROR": "1",
            "PYTHONWARNINGS": "ignore",
        },
        "output_paths": outputs,
        "materialization_paths": materialization,
    }


def build_formal_freeze_document(
    command_identity: dict[str, Any],
    *,
    repo_root: Path,
    head: str,
    source_paths: dict[str, Path],
    hard_artifact_paths: dict[str, Path],
    runtime_diff_clean: bool,
) -> dict[str, Any]:
    """Hash the reviewed formal inputs into a no-training freeze document."""

    failures: list[str] = []
    if not runtime_diff_clean:
        failures.append("runtime source diff is dirty")

    source_identity: dict[str, dict[str, Any]] = {}
    for name, path in source_paths.items():
        if not path.is_file():
            failures.append(f"missing source: {name}")
            continue
        source_identity[name] = _file_identity(path)

    hard_artifacts: dict[str, dict[str, Any]] = {}
    for name, path in hard_artifact_paths.items():
        if not path.is_file():
            failures.append(f"missing hard artifact: {name}")
            continue
        hard_artifacts[name] = _file_identity(path)

    return {
        "schema_version": 1,
        "accepted": not failures,
        "failures": failures,
        "training_executed": False,
        "repo": {
            "root": str(repo_root),
            "head": head,
            "runtime_diff_clean": runtime_diff_clean,
        },
        "source_identity": source_identity,
        "hard_artifacts": hard_artifacts,
        "command": command_identity,
        "output_paths": command_identity["output_paths"],
        "materialization_paths": command_identity["materialization_paths"],
    }


def build_formal_supervisor_source(identity: dict[str, Any]) -> str:
    """Render the one-shot supervisor for an already frozen command identity."""

    outputs = identity["output_paths"]
    env = " ".join(f"{name}={shlex.quote(str(value))}" for name, value in identity["env"].items())
    argv = shlex.join(identity["argv"])
    absence_checks = "\n".join(f"test ! -e {shlex.quote(path)}" for path in outputs.values())
    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(identity["repo_root"])}
{absence_checks}
nvidia-smi --query-compute-apps=timestamp,pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits --loop-ms=250 > {shlex.quote(outputs["gpu_telemetry"])} &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" 2>/dev/null || true; wait "$SAMPLER_PID" 2>/dev/null || true' EXIT
{env} /usr/bin/time -v -o {shlex.quote(outputs["time"])} {argv} > {shlex.quote(outputs["log"])} 2>&1
"""


def build_formal_oracle_source() -> str:
    """Render a fail-closed pre/post oracle that never launches training."""

    return """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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

    for group in ("source_identity", "hard_artifacts"):
        for name, identity in freeze[group].items():
            path = Path(identity["path"])
            if not path.is_file() or sha256(path) != identity["sha256"]:
                failures.append(f"{group} drift: {name}")

    command = freeze.get("command", {})
    if not command.get("argv") or not command.get("env"):
        failures.append("frozen command identity missing")

    output_paths = [Path(path) for path in freeze["output_paths"].values()]
    if args.preflight:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            failures.append(f"output paths already exist: {existing}")
        result = {
            "mode": "preflight",
            "accepted": not failures,
            "failures": failures,
            "training_executed": False,
            "freeze_sha256": sha256(freeze_path),
            "output_paths_absent": not existing,
        }
    else:
        run_dir = Path(freeze["output_paths"]["run_dir"])
        manifest_path = run_dir / "run_manifest.json"
        metrics_path = run_dir / "distillation_metrics.json"
        if not manifest_path.is_file():
            failures.append("formal run manifest missing")
        if not metrics_path.is_file():
            failures.append("formal metrics missing")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        iterations = manifest.get("dagger_iterations", [])
        expected_iterations = freeze["command"]["workload"]["dagger_iterations"]
        if len(iterations) != expected_iterations:
            failures.append(
                f"iteration count mismatch: {len(iterations)} != {expected_iterations}"
            )
        expected_updates = freeze["command"]["workload"][
            "effective_updates_by_iteration"
        ]
        for index, (iteration, expected) in enumerate(
            zip(iterations, expected_updates, strict=False), start=1
        ):
            observed_updates = iteration.get("updates")
            if observed_updates != expected:
                failures.append(
                    f"iteration {index} updates mismatch: "
                    f"{observed_updates} != {expected}"
                )
        if any(not record.get("success", False) for record in metrics.get("records", [])):
            failures.append("unsuccessful metrics record")
        result = {
            "mode": "postflight",
            "accepted": not failures,
            "failures": failures,
            "training_executed": True,
            "freeze_sha256": sha256(freeze_path),
            "manifest_path": str(manifest_path),
            "metrics_path": str(metrics_path),
        }

    Path(args.result).write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""
