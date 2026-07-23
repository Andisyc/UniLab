#!/usr/bin/env python3
"""Exercise the saved-data distillation owner path without launching a simulator."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
from collections import Counter
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.train_distill import run_offline_dataset_update  # noqa: E402

from unilab.algos.torch.distill import (  # noqa: E402
    build_multitask_distillation_dataset,
    load_distillation_dataset,
    load_distillation_student_policy,
    save_distillation_dataset,
)
from unilab.algos.torch.distill.async_runtime import (  # noqa: E402
    DaggerCollectRequest,
    DaggerCollectResult,
)
from unilab.algos.torch.distill.persistent_runtime import (  # noqa: E402
    PersistentDistillationRuntime,
)
from unilab.ipc import SharedWeightSync  # noqa: E402

CONFIG_DIR = ROOT_DIR / "conf" / "distill"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_counts(labels: Sequence[str] | None) -> dict[str, int] | None:
    if labels is None:
        return None
    return dict(sorted(Counter(str(label) for label in labels).items()))


def _tensor_signature(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    tensor = value.detach().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(tensor).all()) if tensor.is_floating_point() else True,
        "sum": float(tensor.double().sum().item()),
    }


def _dataset_signature(dataset: Any) -> dict[str, Any]:
    return {
        "num_samples": int(dataset.num_samples),
        "student_obs": _tensor_signature(dataset.student_obs),
        "teacher_obs": _tensor_signature(dataset.teacher_obs),
        "teacher_actions": _tensor_signature(dataset.teacher_actions),
        "commands": _tensor_signature(dataset.commands),
        "transition_ages": _tensor_signature(dataset.transition_ages),
        "command_before": _tensor_signature(dataset.command_before),
        "command_after": _tensor_signature(dataset.command_after),
        "role_counts": _label_counts(dataset.role_labels),
        "command_intent_counts": _label_counts(dataset.command_intents),
        "scenario_counts": _label_counts(dataset.scenario_labels),
    }


def _assert_same_dataset(expected: Any, actual: Any) -> None:
    tensor_fields = (
        "student_obs",
        "teacher_obs",
        "teacher_actions",
        "commands",
        "transition_ages",
        "command_before",
        "command_after",
    )
    label_fields = ("role_labels", "command_intents", "scenario_labels")
    mismatches: list[str] = []
    for name in tensor_fields:
        expected_value = getattr(expected, name)
        actual_value = getattr(actual, name)
        if (expected_value is None) != (actual_value is None):
            mismatches.append(name)
        elif expected_value is not None and not torch.equal(
            expected_value.detach().cpu(), actual_value.detach().cpu()
        ):
            mismatches.append(name)
    for name in label_fields:
        if getattr(expected, name) != getattr(actual, name):
            mismatches.append(name)
    if mismatches:
        raise AssertionError(f"rebuilt aggregate differs from r10 seed: {sorted(mismatches)}")


def _sources_from_seed_aggregate(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"aggregate payload must be a mapping, got {type(payload).__name__}")
    metadata = dict(payload.get("metadata") or {})
    paths = list(metadata.get("source_paths") or ())
    roles = list(metadata.get("source_roles") or ())
    scenarios = list(metadata.get("source_scenarios") or ())
    if not paths or len(paths) != len(roles) or len(paths) != len(scenarios):
        raise ValueError(
            "r10 aggregate metadata must contain aligned source_paths/source_roles/source_scenarios"
        )
    sources: list[dict[str, Any]] = []
    for source_path, role, scenario in zip(paths, roles, scenarios, strict=True):
        resolved = Path(str(source_path)).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"aggregate source is missing: {resolved}")
        source: dict[str, Any] = {"path": str(resolved), "role": str(role)}
        if scenario not in (None, ""):
            source.update(
                {
                    "scenario": str(scenario),
                    "preserve_row_role_labels": True,
                }
            )
        sources.append(source)
    dimensions = {
        "student_obs": int(payload["student_obs_dim"]),
        "teacher_obs": int(payload["teacher_obs_dim"]),
        "teacher_action": int(payload["teacher_action_dim"]),
    }
    return sources, dimensions


def run_aggregate_assembly(
    *,
    seed_aggregate: Path,
    output: Path,
    device: str,
    keep_output: bool,
) -> dict[str, Any]:
    """Rebuild the r10 aggregate through the production data owner."""

    sources, dimensions = _sources_from_seed_aggregate(seed_aggregate)
    expected = load_distillation_dataset(seed_aggregate, device="cpu")
    assembled = build_multitask_distillation_dataset(
        sources,
        expected_student_obs_dim=dimensions["student_obs"],
        expected_teacher_obs_dim=dimensions["teacher_obs"],
        expected_teacher_action_dim=dimensions["teacher_action"],
        device=device,
    )
    _assert_same_dataset(expected, assembled)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_distillation_dataset(output, assembled)
    reloaded = load_distillation_dataset(
        output,
        expected_student_obs_dim=dimensions["student_obs"],
        expected_teacher_obs_dim=dimensions["teacher_obs"],
        expected_teacher_action_dim=dimensions["teacher_action"],
        device="cpu",
    )
    _assert_same_dataset(expected, reloaded)
    output_identity = {
        "path": str(output),
        "size": output.stat().st_size,
        "sha256": _file_sha256(output),
    }
    if not keep_output:
        output.unlink()
    return {
        "status": "PASS",
        "owner_path": "build_multitask_distillation_dataset/save/load",
        "device": device,
        "source_count": len(sources),
        "sources": sources,
        "semantic_match_to_seed": True,
        "signature": _dataset_signature(reloaded),
        "output_identity": output_identity,
        "output_retained": keep_output,
    }


def _load_student_module(path: Path, *, device: str) -> torch.nn.Module:
    return load_distillation_student_policy(path, device=device).policy


def _model_weight_sum(model: torch.nn.Module) -> float:
    return float(
        sum(parameter.detach().double().sum().cpu().item() for parameter in model.parameters())
    )


class _CheckpointWeightService:
    """Offline collector service using the production checkpoint and weight-sync owners."""

    def __init__(
        self,
        *,
        initial_checkpoint_path: str,
        device: str,
        weight_sync_name: str,
        weight_sync_lock: Any,
        weight_param_shapes: dict[str, torch.Size],
    ) -> None:
        self.device = str(device)
        self.policy = _load_student_module(Path(initial_checkpoint_path), device=self.device)
        self.weight_sync = SharedWeightSync(
            weight_param_shapes,
            create=False,
            shm_name=weight_sync_name,
            lock=weight_sync_lock,
        )

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        version = self.weight_sync.read_weights_into(self.policy.state_dict())
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.device))
        weight_sum = _model_weight_sum(self.policy)
        output = Path(request.output_path)
        _write_json(
            output,
            {
                "worker_pid": os.getpid(),
                "observed_weight_version": version,
                "weight_sum": weight_sum,
            },
        )
        return DaggerCollectResult(
            request_id=request.request_id,
            scenario=request.scenario,
            iteration=request.iteration,
            checkpoint_path=request.checkpoint_path,
            output_path=request.output_path,
            expected_weight_version=request.expected_weight_version,
            observed_weight_version=version,
            num_samples=1,
            worker_pid=os.getpid(),
            metrics={"weight_sum": weight_sum},
            metadata={"owner_path": "PersistentDistillationRuntime/SharedWeightSync"},
        )

    def close(self) -> None:
        self.weight_sync.close()


def _build_checkpoint_weight_service(**kwargs: Any) -> _CheckpointWeightService:
    return _CheckpointWeightService(**kwargs)


def _shared_memory_exists(name: str) -> bool:
    try:
        handle = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return False
    else:
        handle.close()
        return True


class _PersistentCheckpointExercise:
    def __init__(self, *, checkpoint: Path, device: str, output_dir: Path) -> None:
        self.device = str(device)
        self.output_dir = output_dir
        self.runtime = PersistentDistillationRuntime(
            student_loader=functools.partial(_load_student_module, device=self.device),
            worker_factory=_build_checkpoint_weight_service,
            worker_kwargs={
                "initial_checkpoint_path": str(checkpoint.resolve()),
                "device": self.device,
            },
            request_timeout_seconds=600.0,
        )
        self.shared_memory_name: str | None = None
        self.worker_pid: int | None = None
        self.activations: list[dict[str, Any]] = []

    def activate(self, checkpoint: Path, *, label: str, iteration: int) -> dict[str, Any]:
        version = self.runtime.activate_checkpoint(checkpoint)
        if self.runtime._weight_sync is None:
            raise RuntimeError("persistent runtime did not materialize SharedWeightSync")
        self.shared_memory_name = self.runtime._weight_sync.name
        sentinel_path = (self.output_dir / f"{label}-weight-sync.json").resolve()
        result = self.runtime.collect(
            DaggerCollectRequest(
                request_id=label,
                scenario="offline_owner_path",
                iteration=int(iteration),
                checkpoint_path=str(checkpoint.resolve()),
                output_path=str(sentinel_path),
                expected_weight_version=version,
            )
        )
        expected_model = _load_student_module(checkpoint, device=self.device)
        expected_sum = _model_weight_sum(expected_model)
        observed_sum = float(result.metrics["weight_sum"])
        tolerance = max(1e-4, abs(expected_sum) * 1e-6)
        if abs(observed_sum - expected_sum) > tolerance:
            raise AssertionError(
                "SharedWeightSync weight mismatch: "
                f"expected_sum={expected_sum} observed_sum={observed_sum}"
            )
        if self.worker_pid is not None and result.worker_pid != self.worker_pid:
            raise AssertionError(
                f"persistent worker PID changed: {self.worker_pid} -> {result.worker_pid}"
            )
        self.worker_pid = result.worker_pid
        record = {
            "label": label,
            "checkpoint": str(checkpoint.resolve()),
            "version": version,
            "worker_pid": result.worker_pid,
            "weight_sum": observed_sum,
            "weight_match": True,
        }
        self.activations.append(record)
        return record

    def close(self) -> dict[str, Any]:
        name = self.shared_memory_name
        self.runtime.close()
        return {
            "shared_memory_name": name,
            "shared_memory_unlinked": True if name is None else not _shared_memory_exists(name),
            "worker_pid": self.worker_pid,
            "activation_count": len(self.activations),
        }


def _compose_distill_cfg(
    *,
    device: str,
    aggregate: Path,
    init_checkpoint: Path,
    batch_size: int,
    max_updates: int,
) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["task=g1_walk_flat/mujoco", "workflow=g1_walk_stand"],
        )
    cfg.training.workflow.enabled = False
    cfg.training.device = str(device)
    cfg.training.offline_dataset_path = str(aggregate.resolve())
    cfg.training.offline_init_checkpoint = str(init_checkpoint.resolve())
    cfg.training.offline_resume_optimizer = True
    cfg.training.offline_repeat_dataset = True
    cfg.training.offline_shuffle = True
    cfg.training.offline_balance_key = "scenario"
    cfg.training.offline_balanced_labels = ["static_stand", "walk_flat", "walk_to_stop"]
    cfg.training.offline_balance_quotas = {
        "walk_flat": 0.5,
        "static_stand": 0.25,
        "walk_to_stop": 0.25,
    }
    cfg.training.offline_min_balanced_replay_passes = 8
    cfg.training.offline_min_balanced_replay_labels = ["walk_to_stop"]
    cfg.training.offline_batch_size = int(batch_size)
    cfg.training.offline_max_updates = int(max_updates)
    return cfg


def run_offline_owner_sequence(
    *,
    aggregate: Path,
    init_checkpoint: Path,
    teacher_checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    max_updates: int,
    rounds: int,
    keep_checkpoints: bool,
) -> dict[str, Any]:
    """Run real MoE updates while one production weight-sync runtime stays resident."""

    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _compose_distill_cfg(
        device=device,
        aggregate=aggregate,
        init_checkpoint=init_checkpoint,
        batch_size=batch_size,
        max_updates=max_updates,
    )
    _write_json(
        output_dir / "resolved_config.json",
        OmegaConf.to_container(cfg, resolve=True),
    )
    persistent = _PersistentCheckpointExercise(
        checkpoint=init_checkpoint,
        device=device,
        output_dir=output_dir,
    )
    round_results: list[dict[str, Any]] = []
    cleanup: dict[str, Any] | None = None
    try:
        persistent.activate(init_checkpoint, label="initial", iteration=0)
        for round_index in range(1, rounds + 1):
            output_checkpoint = output_dir / f"round-{round_index:02d}.pt"
            result = run_offline_dataset_update(
                cfg,
                teacher_checkpoint=teacher_checkpoint,
                dataset_path=aggregate,
                batch_size=batch_size,
                max_updates=max_updates,
                checkpoint_path=output_checkpoint,
                device=device,
            )
            if int(result["update_count"]) != int(max_updates):
                raise AssertionError(
                    f"offline update count mismatch: {result['update_count']} != {max_updates}"
                )
            activation = persistent.activate(
                output_checkpoint,
                label=f"round-{round_index:02d}",
                iteration=round_index,
            )
            checkpoint_identity = {
                "path": str(output_checkpoint.resolve()),
                "size": output_checkpoint.stat().st_size,
                "sha256": _file_sha256(output_checkpoint),
            }
            round_results.append(
                {
                    "round": round_index,
                    "update_count": int(result["update_count"]),
                    "samples_seen": int(result["samples_seen"]),
                    "loss": float(result["loss"]),
                    "student_grad_norm": float(result["student_grad_norm"]),
                    "checkpoint": checkpoint_identity,
                    "checkpoint_reload": activation,
                }
            )
            if not keep_checkpoints:
                output_checkpoint.unlink()
    finally:
        cleanup = persistent.close()
    return {
        "status": "PASS",
        "owner_path": (
            "load_distillation_dataset/run_offline_distillation_updates/"
            "BehaviorDistillationTrainer.update/checkpoint reload/"
            "PersistentDistillationRuntime/SharedWeightSync/cleanup"
        ),
        "device": device,
        "rounds": rounds,
        "batch_size": batch_size,
        "max_updates_per_round": max_updates,
        "round_results": round_results,
        "persistent_activations": persistent.activations,
        "cleanup": cleanup,
        "output_checkpoints_retained": keep_checkpoints,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--seed-aggregate", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--device", required=True)
    assemble.add_argument("--report", type=Path, required=True)
    assemble.add_argument("--keep-output", action="store_true")

    offline = subparsers.add_parser("offline")
    offline.add_argument("--aggregate", type=Path, required=True)
    offline.add_argument("--init-checkpoint", type=Path, required=True)
    offline.add_argument("--teacher-checkpoint", type=Path, required=True)
    offline.add_argument("--output-dir", type=Path, required=True)
    offline.add_argument("--device", required=True)
    offline.add_argument("--batch-size", type=int, default=512)
    offline.add_argument("--max-updates", type=int, required=True)
    offline.add_argument("--rounds", type=int, default=1)
    offline.add_argument("--report", type=Path, required=True)
    offline.add_argument("--keep-checkpoints", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "assemble":
        result = run_aggregate_assembly(
            seed_aggregate=args.seed_aggregate.resolve(),
            output=args.output.resolve(),
            device=str(args.device),
            keep_output=bool(args.keep_output),
        )
    else:
        result = run_offline_owner_sequence(
            aggregate=args.aggregate.resolve(),
            init_checkpoint=args.init_checkpoint.resolve(),
            teacher_checkpoint=args.teacher_checkpoint.resolve(),
            output_dir=args.output_dir.resolve(),
            device=str(args.device),
            batch_size=int(args.batch_size),
            max_updates=int(args.max_updates),
            rounds=int(args.rounds),
            keep_checkpoints=bool(args.keep_checkpoints),
        )
    _write_json(args.report.resolve(), result)
    print(json.dumps({"status": result["status"], "report": str(args.report.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
