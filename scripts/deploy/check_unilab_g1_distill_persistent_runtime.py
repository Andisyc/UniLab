"""Bounded live sentinel for the persistent G1 distillation collector."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.algos.torch.distill.async_runtime import DaggerCollectRequest
from unilab.algos.torch.distill.data import load_distillation_dataset
from unilab.algos.torch.distill.g1_persistent_worker import (
    build_persistent_g1_distillation_runtime,
)
from unilab.algos.torch.distill.workflow import RoleArtifactSpec

ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_train_distill_module() -> Any:
    path = ROOT_DIR / "scripts" / "train_distill.py"
    spec = importlib.util.spec_from_file_location("persistent_runtime_train_distill", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load train_distill entrypoint: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--walking-checkpoint", type=Path, required=True)
    parser.add_argument("--standing-checkpoint", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--pre-switch-steps", type=int, default=2)
    parser.add_argument("--min-post-switch-steps", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--worker-lifecycle",
        choices=("persistent", "restart_each_request"),
        default="persistent",
    )
    return parser.parse_args()


def _validate_summary(summary: dict[str, Any]) -> None:
    sequence = summary.get("sequence")
    repetitions = int(summary.get("repetitions", 1))
    worker_lifecycle = str(summary.get("worker_lifecycle", "persistent"))
    expected_cycle = ["walk_flat", "static_stand", "walk_to_stop", "walk_flat"]
    expected_scenarios = expected_cycle * repetitions
    if (
        not isinstance(sequence, list)
        or [row.get("scenario") for row in sequence] != expected_scenarios
    ):
        raise RuntimeError("persistent sentinel scenario sequence mismatch")

    worker_pids = {row.get("worker_pid") for row in sequence}
    weight_versions = {row.get("weight_version") for row in sequence}
    if None in worker_pids:
        raise RuntimeError(f"persistent sentinel worker identity missing: {worker_pids}")
    if worker_lifecycle == "persistent" and len(worker_pids) != 1:
        raise RuntimeError(f"persistent sentinel worker identity mismatch: {worker_pids}")
    if worker_lifecycle not in {"persistent", "restart_each_request"}:
        raise RuntimeError(f"unsupported sentinel worker lifecycle: {worker_lifecycle}")
    if len(weight_versions) != 1 or None in weight_versions:
        raise RuntimeError(f"persistent sentinel weight version mismatch: {weight_versions}")

    for transition in sequence[2::4]:
        if transition.get("command_intents") != ["active", "active", "inactive", "inactive"]:
            raise RuntimeError("persistent sentinel transition intent mismatch")
        if transition.get("role_labels") != ["walk_flat", "walk_flat", "stand", "stand"]:
            raise RuntimeError("persistent sentinel transition role mismatch")
        if transition.get("transition_ages") != [-1, -1, 0, 1]:
            raise RuntimeError("persistent sentinel transition age mismatch")

    close_reports = summary.get("close_reports")
    if close_reports is None:
        close_report = summary.get("close_report")
        close_reports = [close_report] if isinstance(close_report, dict) else None
    if not isinstance(close_reports, list) or not all(
        isinstance(report, dict) for report in close_reports
    ):
        raise RuntimeError("persistent sentinel close reports missing")
    expected_report_count = 1 if worker_lifecycle == "persistent" else len(sequence)
    if len(close_reports) != expected_report_count:
        raise RuntimeError(
            "persistent sentinel close report count mismatch: "
            f"expected={expected_report_count} actual={len(close_reports)}"
        )
    for close_report in close_reports:
        if close_report.get("worker_pid") not in worker_pids:
            raise RuntimeError("persistent sentinel close report worker mismatch")
        if close_report.get("student_init_count") != 1:
            raise RuntimeError("persistent sentinel resident student was reinitialized")
        counters = close_report.get("resource_counters")
        if not isinstance(counters, dict):
            raise RuntimeError("persistent sentinel resource counters missing")
        expected_request_count = len(sequence) if worker_lifecycle == "persistent" else 1
        expected_counters = {
            "request_count": expected_request_count,
            "reset_count": expected_request_count,
            "request_error_count": 0,
        }
        mismatches = {
            key: {"expected": expected, "actual": counters.get(key)}
            for key, expected in expected_counters.items()
            if counters.get(key) != expected
        }
        for resource in ("teacher", "env"):
            init_count = counters.get(f"{resource}_init_count")
            close_count = counters.get(f"{resource}_close_count")
            if init_count != close_count:
                mismatches[f"{resource}_close_count"] = {
                    "expected": init_count,
                    "actual": close_count,
                }
        if worker_lifecycle == "persistent":
            for key in ("teacher_init_count", "env_init_count"):
                if counters.get(key) != 2:
                    mismatches[key] = {"expected": 2, "actual": counters.get(key)}
        if mismatches:
            raise RuntimeError(f"persistent sentinel lifecycle mismatch: {mismatches}")


def main() -> None:
    args = _parse_args()
    for path in (
        args.walking_checkpoint,
        args.standing_checkpoint,
        args.student_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ["UNILAB_G1_WALK_TEACHER"] = str(args.walking_checkpoint.resolve())
    os.environ["UNILAB_G1_STAND_TEACHER"] = str(args.standing_checkpoint.resolve())
    os.environ["UNILAB_G1_WALK_DATASET"] = ""
    os.environ["UNILAB_G1_STAND_DATASET"] = ""
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "conf" / "distill"),
        version_base="1.3",
    ):
        cfg = compose(
            "config",
            overrides=[
                "workflow=g1_walk_stand",
                "training.workflow.execution_mode=persistent_async",
                f"training.workflow.collect_num_envs={int(args.num_envs)}",
                f"training.workflow.dagger_samples_per_role={int(args.samples)}",
                f"training.workflow.transition_pre_switch_steps={int(args.pre_switch_steps)}",
                "training.workflow.transition_min_post_switch_steps="
                f"{int(args.min_post_switch_steps)}",
                f"training.device={args.device}",
            ],
        )
    train_distill = _load_train_distill_module()
    entries = train_distill._workflow_role_entries(cfg)
    role_cfgs = {
        str(entry["role"]): train_distill._workflow_role_cfg(cfg, entry) for entry in entries
    }
    role_specs = tuple(
        RoleArtifactSpec(
            role=role,
            task=str(entry["task"]),
            teacher_checkpoint_path=Path(str(role_cfgs[role].teacher.checkpoint_path)),
            dataset_path=args.work_dir / f"unused-{role}.pt",
            schema_version=1,
            student_obs_dim=int(role_cfgs[role].student.obs_dim),
            teacher_obs_dim=int(role_cfgs[role].teacher.obs_dim),
            teacher_action_dim=int(role_cfgs[role].teacher.action_dim),
            teacher_obs_key=str(role_cfgs[role].training.collect_teacher_obs_key),
            teacher_projection=str(role_cfgs[role].training.collect_teacher_projection),
            student_projection=str(role_cfgs[role].training.collect_student_projection),
            student_drop_index=OmegaConf.select(
                role_cfgs[role], "training.collect_student_drop_index"
            ),
            command_sample_filter=str(role_cfgs[role].training.collect_command_sample_filter),
            command_info_key=str(role_cfgs[role].training.collect_command_info_key),
            command_xy_threshold=float(role_cfgs[role].training.collect_command_xy_threshold),
            command_yaw_threshold=float(role_cfgs[role].training.collect_command_yaw_threshold),
            owner_config=train_distill._workflow_owner_fingerprint_cfg(role_cfgs[role]),
        )
        for entry in entries
        for role in (str(entry["role"]),)
    )
    scenario_specs = train_distill._workflow_scenario_specs(
        cfg,
        set(role_cfgs),
    )
    if int(args.repetitions) <= 0:
        raise ValueError(f"repetitions must be positive, got {args.repetitions}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "probe": "g1_distill_persistent_runtime",
        "student_checkpoint": str(args.student_checkpoint.resolve()),
        "worker_lifecycle": args.worker_lifecycle,
        "repetitions": int(args.repetitions),
        "sequence": [],
        "close_reports": [],
    }

    def build_runtime():
        return build_persistent_g1_distillation_runtime(
            cfg=cfg,
            role_cfgs=role_cfgs,
            role_specs=role_specs,
            scenario_specs=scenario_specs,
        )

    scenarios = ("walk_flat", "static_stand", "walk_to_stop", "walk_flat") * int(args.repetitions)

    def collect(runtime, *, index: int, scenario: str, version: int) -> None:
        output_path = args.work_dir / f"{index:04d}-{scenario}.pt"
        result = runtime.collect(
            DaggerCollectRequest(
                request_id=f"live-{index}-{scenario}",
                scenario=scenario,
                iteration=1 + index // 4,
                checkpoint_path=str(args.student_checkpoint.resolve()),
                output_path=str(output_path.resolve()),
                expected_weight_version=version,
            )
        )
        dataset = load_distillation_dataset(output_path)
        summary["sequence"].append(
            {
                "scenario": scenario,
                "worker_pid": result.worker_pid,
                "weight_version": result.observed_weight_version,
                "num_samples": dataset.num_samples,
                "role_labels": list(dataset.role_labels or ()),
                "command_intents": list(dataset.command_intents or ()),
                "scenario_labels": list(dataset.scenario_labels or ()),
                "transition_ages": (
                    None if dataset.transition_ages is None else dataset.transition_ages.tolist()
                ),
                "metrics": dict(result.metrics),
                "metadata": dict(result.metadata),
                "artifact_path": str(output_path.resolve()),
            }
        )

    if args.worker_lifecycle == "persistent":
        runtime = build_runtime()
        try:
            version = runtime.activate_checkpoint(args.student_checkpoint)
            for index, scenario in enumerate(scenarios):
                collect(runtime, index=index, scenario=scenario, version=version)
        finally:
            runtime.close()
        summary["close_report"] = runtime.close_report
        summary["close_reports"].append(runtime.close_report)
    else:
        for index, scenario in enumerate(scenarios):
            runtime = build_runtime()
            try:
                version = runtime.activate_checkpoint(args.student_checkpoint)
                collect(runtime, index=index, scenario=scenario, version=version)
            finally:
                runtime.close()
            summary["close_reports"].append(runtime.close_report)
    _validate_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
