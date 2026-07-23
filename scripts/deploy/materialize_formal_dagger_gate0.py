#!/usr/bin/env python3
"""Materialize a formal DAgger FT-0 identity without executing training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from unilab.algos.torch.distill.data import load_distillation_dataset
from unilab.algos.torch.distill.formal_identity import (
    FormalDaggerIdentitySpec,
    build_formal_command_identity,
    build_formal_freeze_document,
    build_formal_oracle_source,
    build_formal_supervisor_source,
    resolve_time_sorted_formal_output_identity,
)
from unilab.algos.torch.distill.offline import (
    required_balanced_replay_updates_for_labels,
)
from unilab.cli import build_command

FORK_HARD_ARTIFACTS = frozenset(
    {
        "parent_manifest",
        "parent_checkpoint",
        "parent_aggregate",
        "walk_teacher",
        "stand_teacher",
        "walk_dataset",
        "stand_dataset",
    }
)
FRESH_HARD_ARTIFACTS = frozenset({"walk_teacher", "stand_teacher", "walk_dataset", "stand_dataset"})
REQUIRED_HARD_ARTIFACTS = FORK_HARD_ARTIFACTS
RUNTIME_SCOPE = ("src", "scripts/train_distill.py", "conf/distill", "pyproject.toml", "uv.lock")


@dataclass(frozen=True)
class Gate0Observations:
    """No-training observations captured before materializing the freeze."""

    head: str
    runtime_diff_clean: bool
    compose_returncode: int
    compose_stdout: str
    compose_stderr: str
    dependency_identity: dict[str, Any]
    gpu_query: dict[str, Any]
    workload_identity: dict[str, Any]


@dataclass(frozen=True)
class MaterializationSpec:
    identity: FormalDaggerIdentitySpec
    source_paths: dict[str, Path]
    hard_artifact_paths: dict[str, Path]
    auto_output_identity: dict[str, str] | None = None


def load_materialization_spec(path: Path, *, now: datetime | None = None) -> MaterializationSpec:
    """Parse the reviewed JSON spec and reject incomplete formal identities."""

    payload = json.loads(path.read_text())
    hard_artifact_paths = {
        name: Path(value).resolve() for name, value in payload.pop("hard_artifact_paths").items()
    }
    required = (
        FRESH_HARD_ARTIFACTS if payload.get("mode", "fork") == "fresh" else FORK_HARD_ARTIFACTS
    )
    missing = sorted(required - hard_artifact_paths.keys())
    if missing:
        raise ValueError(f"missing hard artifact identities: {missing}")
    source_paths = {
        name: Path(value).resolve() for name, value in payload.pop("source_paths").items()
    }
    auto_output_identity: dict[str, str] | None = None
    run_name = payload.pop("run_name", None)
    run_root = payload.pop("run_root", None)
    artifact_root = payload.pop("artifact_root", None)
    if run_name is not None:
        conflicting = [field for field in ("run_dir", "artifact_dir") if field in payload]
        if conflicting:
            raise ValueError(
                "run_name cannot be combined with manual output paths: " + ", ".join(conflicting)
            )
        resolved_repo_root = Path(payload["repo_root"]).resolve()
        generated = resolve_time_sorted_formal_output_identity(
            repo_root=resolved_repo_root,
            run_name=str(run_name),
            mode=str(payload.get("mode", "fork")),
            now=now or datetime.now(),
            run_root=None if run_root is None else Path(str(run_root)),
            artifact_root=None if artifact_root is None else Path(str(artifact_root)),
        )
        payload["run_dir"] = generated.run_dir
        if generated.artifact_dir is not None:
            payload["artifact_dir"] = generated.artifact_dir
        auto_output_identity = {
            "run_name": generated.run_name,
            "timestamp": generated.timestamp,
            "stem": generated.stem,
        }
    elif run_root is not None or artifact_root is not None:
        raise ValueError("run_root and artifact_root require run_name")
    payload["effective_updates_by_iteration"] = tuple(
        int(value) for value in payload["effective_updates_by_iteration"]
    )
    for field in ("repo_root", "run_dir"):
        payload[field] = Path(payload[field]).resolve()
    payload["parent_run_dir"] = (
        None if payload.get("parent_run_dir") is None else Path(payload["parent_run_dir"]).resolve()
    )
    if payload.get("artifact_dir") is not None:
        payload["artifact_dir"] = Path(payload["artifact_dir"]).resolve()
    return MaterializationSpec(
        identity=FormalDaggerIdentitySpec(**payload),
        source_paths=source_paths,
        hard_artifact_paths=hard_artifact_paths,
        auto_output_identity=auto_output_identity,
    )


def _compose_argv(command_identity: dict[str, Any]) -> list[str]:
    public_argv = list(command_identity["argv"])
    override_start = public_argv.index("workflow=g1_walk_stand")
    routed = build_command(
        mode="train",
        algo="distill",
        task="g1_walk_flat",
        sim="mujoco",
        overrides=public_argv[override_start:],
    )
    return [*routed[:2], "--cfg", "job", "--resolve", *routed[2:]]


def bind_hard_artifact_environment(
    command_identity: dict[str, Any], spec: MaterializationSpec
) -> dict[str, Any]:
    """Bind manifest-reviewed teacher/data artifacts to config environment keys."""

    command_identity["env"].update(
        {
            "UNILAB_G1_WALK_TEACHER": str(spec.hard_artifact_paths["walk_teacher"]),
            "UNILAB_G1_STAND_TEACHER": str(spec.hard_artifact_paths["stand_teacher"]),
            "UNILAB_G1_WALK_DATASET": str(spec.hard_artifact_paths["walk_dataset"]),
            "UNILAB_G1_STAND_DATASET": str(spec.hard_artifact_paths["stand_dataset"]),
        }
    )
    return command_identity


def compute_observed_workload(spec: MaterializationSpec, compose_stdout: str) -> dict[str, Any]:
    """Recompute each formal replay budget from the real parent aggregate."""

    cfg = OmegaConf.create(compose_stdout)
    scenarios = OmegaConf.select(cfg, "training.workflow.scenarios")
    if not scenarios:
        raise ValueError("composed workflow has no scenarios")
    scenario_names = [str(item["name"]) for item in scenarios]
    quotas = {str(item["name"]): float(item["quota"]) for item in scenarios}
    batch_size = int(OmegaConf.select(cfg, "training.workflow.dagger_batch_size"))
    replay_passes = int(
        OmegaConf.select(cfg, "training.workflow.dagger_min_transition_replay_passes")
    )
    replay_labels = tuple(
        str(label)
        for label in OmegaConf.select(cfg, "training.workflow.dagger_min_transition_replay_labels")
    )
    balance_key = str(OmegaConf.select(cfg, "training.workflow.dagger_balance_key"))
    if balance_key != "scenario":
        raise ValueError(f"formal replay balance key must be scenario, got {balance_key!r}")
    if batch_size != spec.identity.batch_size:
        raise ValueError(
            f"composed batch size mismatch: {batch_size} != {spec.identity.batch_size}"
        )

    if spec.identity.mode == "fresh":
        walk = load_distillation_dataset(spec.hard_artifact_paths["walk_dataset"], device="cpu")
        stand = load_distillation_dataset(spec.hard_artifact_paths["stand_dataset"], device="cpu")
        labels = ("walk_flat",) * walk.num_samples + ("static_stand",) * stand.num_samples
        parent_rows = walk.num_samples + stand.num_samples
    else:
        aggregate = load_distillation_dataset(
            spec.hard_artifact_paths["parent_aggregate"], device="cpu"
        )
        if aggregate.scenario_labels is None:
            raise ValueError("parent aggregate has no scenario_labels")
        labels = tuple(aggregate.scenario_labels)
        parent_rows = aggregate.num_samples
    aggregate_rows: list[int] = []
    required_updates: list[int] = []
    effective_updates: list[int] = []
    for _ in range(spec.identity.dagger_iterations):
        labels += tuple(
            scenario for scenario in scenario_names for _ in range(spec.identity.samples_per_role)
        )
        required = required_balanced_replay_updates_for_labels(
            labels,
            batch_size=batch_size,
            balanced_labels=scenario_names,
            balance_quotas=quotas,
            replay_labels=replay_labels,
            replay_passes=replay_passes,
        )
        aggregate_rows.append(len(labels))
        required_updates.append(required)
        effective_updates.append(max(spec.identity.configured_update_floor, required))
    return {
        "parent_rows": parent_rows,
        "aggregate_rows_by_iteration": aggregate_rows,
        "required_updates_by_iteration": required_updates,
        "effective_updates_by_iteration": effective_updates,
        "total_effective_updates": sum(effective_updates),
    }


def validate_observed_workload(
    *, expected_schedule: list[int], expected_total: int, observed: dict[str, Any]
) -> list[str]:
    """Return fail-closed differences between spec and observed workload."""

    observed_schedule = observed.get("effective_updates_by_iteration")
    observed_total = observed.get("total_effective_updates")
    failures: list[str] = []
    if observed_schedule != expected_schedule:
        failures.append(
            "effective update schedule mismatch: "
            f"observed={observed_schedule} expected={expected_schedule}"
        )
    if observed_total != expected_total:
        failures.append(
            f"total effective updates mismatch: observed={observed_total} expected={expected_total}"
        )
    return failures


def observe_gate0(spec: MaterializationSpec) -> Gate0Observations:
    """Collect Git, compose, dependency, and GPU facts without training."""

    root = spec.identity.repo_root
    command_identity = bind_hard_artifact_environment(
        build_formal_command_identity(spec.identity), spec
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    runtime_status = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *RUNTIME_SCOPE,
        ],
        cwd=root,
        text=True,
    )
    compose = subprocess.run(
        _compose_argv(command_identity),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **command_identity["env"]},
    )

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
    try:
        workload_identity = compute_observed_workload(spec, compose.stdout)
    except Exception as error:  # fail closed into the freeze artifact
        workload_identity = {"error": f"{type(error).__name__}: {error}"}
    return Gate0Observations(
        head=head,
        runtime_diff_clean=not runtime_status.strip(),
        compose_returncode=compose.returncode,
        compose_stdout=compose.stdout,
        compose_stderr=compose.stderr,
        dependency_identity=dependency_identity,
        gpu_query={
            "returncode": gpu.returncode,
            "stdout": gpu.stdout.strip(),
            "stderr": gpu.stderr,
        },
        workload_identity=workload_identity,
    )


def materialize_from_spec(
    spec_path: Path,
    *,
    observations: Gate0Observations | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write FT-0 artifacts and run only the generated oracle preflight."""

    materialization = load_materialization_spec(spec_path, now=now)
    identity = bind_hard_artifact_environment(
        build_formal_command_identity(materialization.identity), materialization
    )
    observed = observations or observe_gate0(materialization)
    paths = {name: Path(path) for name, path in identity["materialization_paths"].items()}

    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite FT-0 artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    paths["compose"].write_text(observed.compose_stdout)
    paths["supervisor"].write_text(build_formal_supervisor_source(identity))
    paths["oracle"].write_text(build_formal_oracle_source())

    sources = {
        **materialization.source_paths,
        "formal_spec": spec_path.resolve(),
        "generated_compose": paths["compose"],
        "generated_supervisor": paths["supervisor"],
        "generated_oracle": paths["oracle"],
    }
    freeze = build_formal_freeze_document(
        identity,
        repo_root=materialization.identity.repo_root,
        head=observed.head,
        source_paths=sources,
        hard_artifact_paths=materialization.hard_artifact_paths,
        runtime_diff_clean=observed.runtime_diff_clean,
    )
    extra_failures: list[str] = []
    if observed.compose_returncode != 0:
        extra_failures.append(f"Hydra compose failed: {observed.compose_returncode}")
    if observed.compose_stderr:
        extra_failures.append("Hydra compose stderr is not empty")
    if not observed.compose_stdout.strip():
        extra_failures.append("Hydra compose output is empty")
    if observed.gpu_query.get("returncode") != 0:
        extra_failures.append("GPU identity query failed")
    extra_failures.extend(
        validate_observed_workload(
            expected_schedule=identity["workload"]["effective_updates_by_iteration"],
            expected_total=identity["workload"]["total_effective_updates"],
            observed=observed.workload_identity,
        )
    )
    freeze["failures"].extend(extra_failures)
    freeze["accepted"] = not freeze["failures"]
    freeze["compose"] = {
        "path": str(paths["compose"]),
        "returncode": observed.compose_returncode,
        "stderr": observed.compose_stderr,
    }
    freeze["dependency_identity"] = observed.dependency_identity
    freeze["gpu_query"] = observed.gpu_query
    freeze["observed_workload"] = observed.workload_identity
    if materialization.auto_output_identity is not None:
        freeze["auto_output_identity"] = materialization.auto_output_identity
    paths["freeze"].write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")

    preflight = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            str(paths["oracle"]),
            "--freeze",
            str(paths["freeze"]),
            "--result",
            str(paths["preflight"]),
            "--preflight",
        ],
        cwd=materialization.identity.repo_root,
        check=False,
    )
    preflight_payload = (
        json.loads(paths["preflight"].read_text()) if paths["preflight"].is_file() else {}
    )
    return {
        "accepted": preflight.returncode == 0 and bool(preflight_payload.get("accepted")),
        "training_executed": False,
        "freeze_path": str(paths["freeze"]),
        "preflight_path": str(paths["preflight"]),
        "preflight_returncode": preflight.returncode,
        "run_dir": str(materialization.identity.run_dir),
        "artifact_dir": (
            None
            if materialization.identity.artifact_dir is None
            else str(materialization.identity.artifact_dir)
        ),
        "auto_output_identity": materialization.auto_output_identity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_from_spec(args.spec.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
