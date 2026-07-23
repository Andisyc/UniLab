from __future__ import annotations

import builtins
import json
import os
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import torch

from .trainer import DistillationBatch, _distill_runtime_debug_enabled

_ORIGINAL_CALLABLE = callable
_ORIGINAL_ISINSTANCE = isinstance
_ORIGINAL_LIST = list
_ORIGINAL_REPR = repr
_ORIGINAL_STR = str
_ORIGINAL_TUPLE = tuple
_ORIGINAL_TYPE = type


def _validate_obs_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    expected_dim: int | None,
) -> int:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {tuple(tensor.shape)}")
    obs_dim = int(tensor.shape[-1])
    if expected_dim is not None and obs_dim != int(expected_dim):
        raise ValueError(f"{name} dim mismatch: expected {int(expected_dim)}, got {obs_dim}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return obs_dim


def _validate_action_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    expected_dim: int | None,
) -> int:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {tuple(tensor.shape)}")
    action_dim = int(tensor.shape[-1])
    if expected_dim is not None and action_dim != int(expected_dim):
        raise ValueError(f"{name} dim mismatch: expected {int(expected_dim)}, got {action_dim}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return action_dim


def _validate_role_labels(
    role_labels: list[str] | tuple[str, ...] | None,
    *,
    num_samples: int,
) -> tuple[str, ...] | None:
    if role_labels is None:
        return None
    if len(role_labels) != int(num_samples):
        raise ValueError(
            f"role_labels length mismatch: labels={len(role_labels)} samples={int(num_samples)}"
        )
    labels = tuple(str(label) for label in role_labels)
    if any(label == "" for label in labels):
        raise ValueError("role_labels must not contain empty labels")
    return labels


def _validate_commands(
    commands: torch.Tensor | None,
    *,
    num_samples: int,
) -> torch.Tensor | None:
    return _validate_command_tensor("commands", commands, num_samples=num_samples)


def _validate_command_tensor(
    name: str,
    commands: torch.Tensor | None,
    *,
    num_samples: int,
) -> torch.Tensor | None:
    if commands is None:
        return None
    if commands.ndim != 2 or int(commands.shape[-1]) != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(commands.shape)}")
    if int(commands.shape[0]) != int(num_samples):
        raise ValueError(
            f"{name} batch size mismatch: "
            f"{name}={int(commands.shape[0])} samples={int(num_samples)}"
        )
    if not torch.isfinite(commands).all():
        raise ValueError(f"{name} must contain only finite values")
    return commands


def _validate_command_intents(
    command_intents: list[str] | tuple[str, ...] | None,
    *,
    num_samples: int,
) -> tuple[str, ...] | None:
    if command_intents is None:
        return None
    if len(command_intents) != int(num_samples):
        raise ValueError(
            "command_intents length mismatch: "
            f"intents={len(command_intents)} samples={int(num_samples)}"
        )
    intents = tuple(str(intent) for intent in command_intents)
    allowed = {"active", "inactive"}
    invalid_indices = [index for index, intent in enumerate(intents) if intent not in allowed]
    if invalid_indices:
        invalid_head = [
            {
                "index": index,
                "raw_type": _ORIGINAL_TYPE(command_intents[index]).__name__,
                "raw_repr": _safe_runtime_repr(command_intents[index]),
                "normalized": intents[index],
            }
            for index in invalid_indices[:10]
        ]
        abort_requested = os.environ.get("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "0") == "1"
        _emit_data_runtime(
            "command_intent_validation/corruption_detected",
            num_samples=int(num_samples),
            command_intents_type=_ORIGINAL_TYPE(command_intents).__name__,
            command_intents_length=len(command_intents),
            invalid_count=len(invalid_indices),
            invalid_head=invalid_head,
            native_abort_requested=abort_requested,
        )
        if abort_requested:
            _abort_for_native_capture()
        raise ValueError(
            "command_intents must contain only active/inactive labels; "
            f"invalid_head={invalid_head!r}"
        )
    return intents


def _command_intents_from_commands(
    commands: torch.Tensor,
    *,
    xy_threshold: float,
    yaw_threshold: float,
) -> tuple[str, ...]:
    xy_threshold = float(xy_threshold)
    yaw_threshold = float(yaw_threshold)
    if xy_threshold < 0.0:
        raise ValueError(f"command_xy_threshold must be non-negative, got {xy_threshold}")
    if yaw_threshold < 0.0:
        raise ValueError(f"command_yaw_threshold must be non-negative, got {yaw_threshold}")
    xy_norm = torch.linalg.norm(commands[:, :2], dim=1)
    yaw_abs = commands[:, 2].abs()
    active = (xy_norm > xy_threshold) | (yaw_abs > yaw_threshold)
    return tuple("active" if bool(value) else "inactive" for value in active.detach().cpu())


def _command_intents_from_role_labels(
    role_labels: tuple[str, ...],
) -> tuple[str, ...] | None:
    intents: list[str] = []
    for role in role_labels:
        normalized = role.lower()
        if "stand" in normalized:
            intents.append("inactive")
        elif "walk" in normalized:
            intents.append("active")
        else:
            return None
    return tuple(intents)


def _label_counts(labels: tuple[str, ...]) -> dict[str, int]:
    return {label: labels.count(label) for label in sorted(set(labels))}


def _command_intent_debug_snapshot(
    command_intents: Sequence[Any],
) -> dict[str, Any]:
    normalized = tuple(_ORIGINAL_STR(intent) for intent in command_intents)
    return {
        "type": type(command_intents).__name__,
        "length": len(normalized),
        "command_intent_counts": _label_counts(normalized),
        "invalid_head": [
            {
                "index": index,
                "type": type(command_intents[index]).__name__,
                "repr": repr(command_intents[index]),
                "normalized": intent,
            }
            for index, intent in enumerate(normalized)
            if intent not in {"active", "inactive"}
        ][:10],
    }


def _expected_command_intent_for_scenario(scenario: str | None) -> str | None:
    if scenario == "walk_flat":
        return "active"
    if scenario == "static_stand":
        return "inactive"
    return None


def _command_intent_contract_debug_snapshot(
    command_intents: Sequence[Any] | None,
    *,
    expected_intent: str | None,
) -> dict[str, Any] | None:
    if command_intents is None:
        return None
    normalized = tuple(_ORIGINAL_STR(intent) for intent in command_intents)
    snapshot = _command_intent_debug_snapshot(command_intents)
    snapshot["expected_intent"] = expected_intent
    snapshot["expected_mismatch_head"] = (
        []
        if expected_intent is None
        else [
            {
                "index": index,
                "type": _ORIGINAL_TYPE(command_intents[index]).__name__,
                "repr": _safe_runtime_repr(command_intents[index]),
                "normalized": intent,
            }
            for index, intent in enumerate(normalized)
            if intent != expected_intent
        ][:10]
    )
    return snapshot


def _multitask_source_debug_snapshot(
    *,
    source_index: int,
    path: Path,
    role: str,
    scenario: str | None,
    dataset: DistillationTensorDataset,
    error: BaseException | None = None,
) -> dict[str, Any]:
    metadata = dict(dataset.metadata)
    metadata_keys = (
        "source",
        "scenario_annotation",
        "workflow_scenario",
        "command_sample_filter",
        "command_seen_samples",
        "command_selected_samples",
        "command_intent_counts",
        "scenario_counts",
        "role_label_counts",
        "num_samples",
    )
    expected_intent = _expected_command_intent_for_scenario(scenario)
    snapshot: dict[str, Any] = {
        "source_index": int(source_index),
        "path": str(path),
        "role": role,
        "requested_scenario": scenario,
        "num_samples": dataset.num_samples,
        "student_obs_shape": tuple(dataset.student_obs.shape),
        "teacher_obs_shape": tuple(dataset.teacher_obs.shape),
        "teacher_actions_shape": (
            None if dataset.teacher_actions is None else tuple(dataset.teacher_actions.shape)
        ),
        "commands_shape": None if dataset.commands is None else tuple(dataset.commands.shape),
        "command_intents": _command_intent_contract_debug_snapshot(
            dataset.command_intents,
            expected_intent=expected_intent,
        ),
        "scenario_labels": (
            None
            if dataset.scenario_labels is None
            else _scenario_label_debug_snapshot(dataset.scenario_labels)
        ),
        "metadata": {key: metadata[key] for key in metadata_keys if key in metadata},
    }
    if error is not None:
        snapshot["error_type"] = _ORIGINAL_TYPE(error).__name__
        snapshot["error"] = _ORIGINAL_STR(error)
        snapshot["error_repr"] = _safe_runtime_repr(error)
    return snapshot


def _metadata_workflow_scenario(dataset: DistillationTensorDataset) -> str | None:
    value = dataset.metadata.get("workflow_scenario")
    if value in (None, ""):
        return None
    scenario = _ORIGINAL_STR(value)
    if scenario not in _TRANSITION_SCENARIOS:
        raise ValueError(f"dataset metadata workflow_scenario is invalid: {scenario!r}")
    return scenario


def _safe_runtime_repr(value: Any) -> str:
    try:
        return _ORIGINAL_REPR(value)
    except BaseException as error:  # pragma: no cover - defensive runtime probe
        return f"<repr-error type={_ORIGINAL_TYPE(error).__name__} repr={_ORIGINAL_REPR(error)}>"


def _scenario_label_debug_snapshot(
    scenario_labels: Sequence[Any],
    *,
    source_ranges: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    invalid_head: list[dict[str, Any]] = []
    boundary_entries: list[dict[str, Any]] = []
    length = len(scenario_labels)
    boundary_indices = sorted({0, 1, max(0, length - 2), max(0, length - 1)})

    for index, raw_label in enumerate(scenario_labels):
        try:
            normalized_label = _ORIGINAL_STR(raw_label)
        except BaseException as error:  # pragma: no cover - defensive runtime probe
            normalized_label = (
                "<str-error "
                f"type={_ORIGINAL_TYPE(error).__name__} "
                f"repr={_safe_runtime_repr(error)}>"
            )
        label_counts[normalized_label] = label_counts.get(normalized_label, 0) + 1
        entry = {
            "index": index,
            "raw_type": _ORIGINAL_TYPE(raw_label).__name__,
            "raw_repr": _safe_runtime_repr(raw_label),
            "normalized": normalized_label,
        }
        if index in boundary_indices:
            boundary_entries.append(entry)
        if (
            _ORIGINAL_TYPE(raw_label) is not _ORIGINAL_STR
            or normalized_label not in _TRANSITION_SCENARIOS
        ) and len(invalid_head) < 10:
            if source_ranges:
                provenance = next(
                    (
                        source_range
                        for source_range in source_ranges
                        if source_range["global_start"] <= index < source_range["global_stop"]
                    ),
                    None,
                )
                enriched = {
                    "global_index": index,
                    "raw_type": entry["raw_type"],
                    "raw_repr": entry["raw_repr"],
                    "normalized": normalized_label,
                }
                if provenance is not None:
                    enriched.update(
                        {
                            "source_index": provenance["source_index"],
                            "source_row_index": index - provenance["global_start"],
                            "path": provenance["path"],
                            "role": provenance["role"],
                            "scenario": provenance["scenario"],
                        }
                    )
                invalid_head.append(enriched)
            else:
                invalid_head.append(entry)

    return {
        "type": _ORIGINAL_TYPE(scenario_labels).__name__,
        "length": length,
        "label_counts": dict(sorted(label_counts.items())),
        "boundary_entries": boundary_entries,
        "invalid_head": invalid_head,
    }


def _emit_data_runtime(stage: str, **fields: Any) -> None:
    if not _distill_runtime_debug_enabled():
        return
    is_storage = torch.is_storage
    current_int = builtins.int
    current_isinstance = builtins.isinstance
    current_str = builtins.str
    current_type = builtins.type
    current_tuple = builtins.tuple
    current_list = builtins.list
    trace = sys.gettrace()
    profile = sys.getprofile()
    snapshot = {
        "stage": stage,
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "torch_is_storage_type": _ORIGINAL_TYPE(is_storage).__name__,
        "torch_is_storage_repr": _safe_runtime_repr(is_storage),
        "torch_is_storage_callable": _ORIGINAL_CALLABLE(is_storage),
        "builtins_int_type": _ORIGINAL_TYPE(current_int).__name__,
        "builtins_int_repr": _safe_runtime_repr(current_int),
        "builtins_int_callable": _ORIGINAL_CALLABLE(current_int),
        "builtins_isinstance_type": _ORIGINAL_TYPE(current_isinstance).__name__,
        "builtins_isinstance_repr": _safe_runtime_repr(current_isinstance),
        "builtins_isinstance_callable": _ORIGINAL_CALLABLE(current_isinstance),
        "builtins_isinstance_is_original": current_isinstance is _ORIGINAL_ISINSTANCE,
        "builtins_str_type": _ORIGINAL_TYPE(current_str).__name__,
        "builtins_str_repr": _safe_runtime_repr(current_str),
        "builtins_str_callable": _ORIGINAL_CALLABLE(current_str),
        "builtins_str_is_original": current_str is _ORIGINAL_STR,
        "builtins_type_type": _ORIGINAL_TYPE(current_type).__name__,
        "builtins_type_repr": _safe_runtime_repr(current_type),
        "builtins_type_callable": _ORIGINAL_CALLABLE(current_type),
        "builtins_type_is_original": current_type is _ORIGINAL_TYPE,
        "builtins_tuple_type": _ORIGINAL_TYPE(current_tuple).__name__,
        "builtins_tuple_repr": _safe_runtime_repr(current_tuple),
        "builtins_tuple_callable": _ORIGINAL_CALLABLE(current_tuple),
        "builtins_tuple_is_original": current_tuple is _ORIGINAL_TUPLE,
        "builtins_list_type": _ORIGINAL_TYPE(current_list).__name__,
        "builtins_list_repr": _safe_runtime_repr(current_list),
        "builtins_list_callable": _ORIGINAL_CALLABLE(current_list),
        "builtins_list_is_original": current_list is _ORIGINAL_LIST,
        "sys_trace_type": None if trace is None else _ORIGINAL_TYPE(trace).__name__,
        "sys_trace_repr": None if trace is None else _safe_runtime_repr(trace),
        "sys_profile_type": None if profile is None else _ORIGINAL_TYPE(profile).__name__,
        "sys_profile_repr": None if profile is None else _safe_runtime_repr(profile),
        **fields,
    }
    print(f"[distill-data-runtime] {snapshot!r}", flush=True)


def _native_abort_for_impossible_callable_error_requested(error: BaseException) -> bool:
    return (
        os.environ.get("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "0") == "1"
        and _ORIGINAL_ISINSTANCE(error, TypeError)
        and "object is not callable" in _ORIGINAL_STR(error)
    )


def _abort_for_native_capture() -> None:
    # 仅用于诊断: 在 Apport core 中保留当前 learner 进程状态.
    sys.stdout.flush()
    sys.stderr.flush()
    os.abort()


_TRANSITION_SCENARIOS = {"static_stand", "walk_flat", "walk_to_stop"}


def _validate_scenario_labels(
    scenario_labels: list[str] | tuple[str, ...] | None,
    *,
    num_samples: int,
) -> tuple[str, ...] | None:
    if scenario_labels is None:
        return None
    entry_snapshot = _scenario_label_debug_snapshot(scenario_labels)
    _emit_data_runtime(
        "scenario_validation/entry",
        num_samples=num_samples,
        scenario_labels=entry_snapshot,
    )
    if len(scenario_labels) != int(num_samples):
        _emit_data_runtime(
            "scenario_validation/failure",
            reason="length_mismatch",
            num_samples=num_samples,
            scenario_labels=entry_snapshot,
        )
        raise ValueError(
            "scenario_labels length mismatch: "
            f"labels={len(scenario_labels)} samples={int(num_samples)}"
        )
    labels = tuple(str(label) for label in scenario_labels)
    if any(label == "" for label in labels):
        _emit_data_runtime(
            "scenario_validation/failure",
            reason="empty_label",
            num_samples=num_samples,
            scenario_labels=_scenario_label_debug_snapshot(scenario_labels),
        )
        raise ValueError("scenario_labels must not contain empty labels")
    unknown = sorted(set(labels) - _TRANSITION_SCENARIOS)
    if unknown:
        _emit_data_runtime(
            "scenario_validation/failure",
            reason="unknown_label",
            num_samples=num_samples,
            unknown=unknown,
            scenario_labels=_scenario_label_debug_snapshot(scenario_labels),
        )
        raise ValueError(
            f"scenario_labels must contain only static_stand/walk_flat/walk_to_stop, got {unknown}"
        )
    _emit_data_runtime(
        "scenario_validation/success",
        num_samples=num_samples,
        scenario_labels=_scenario_label_debug_snapshot(scenario_labels),
    )
    return labels


def _validate_transition_ages(
    transition_ages: torch.Tensor | None,
    *,
    num_samples: int,
) -> torch.Tensor | None:
    if transition_ages is None:
        return None
    if transition_ages.ndim != 1:
        raise ValueError(
            f"transition_ages must have shape (N,), got {tuple(transition_ages.shape)}"
        )
    if int(transition_ages.shape[0]) != int(num_samples):
        raise ValueError(
            "transition_ages batch size mismatch: "
            f"transition_ages={int(transition_ages.shape[0])} samples={int(num_samples)}"
        )
    if transition_ages.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError(f"transition_ages must have an integer dtype, got {transition_ages.dtype}")
    if torch.any(transition_ages < -1):
        raise ValueError("transition_ages must be -1 or non-negative")
    return transition_ages


def _validate_transition_fields(
    *,
    scenario_labels: list[str] | tuple[str, ...] | None,
    transition_ages: torch.Tensor | None,
    command_before: torch.Tensor | None,
    command_after: torch.Tensor | None,
    num_samples: int,
) -> tuple[
    tuple[str, ...] | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    has_extra = any(value is not None for value in (transition_ages, command_before, command_after))
    validated_labels = _validate_scenario_labels(scenario_labels, num_samples=num_samples)
    if validated_labels is None:
        if has_extra:
            raise ValueError("transition fields require scenario_labels")
        return None, None, None, None
    if transition_ages is None:
        raise ValueError("scenario_labels require transition_ages")
    validated_ages = _validate_transition_ages(transition_ages, num_samples=num_samples)
    if validated_ages is None:
        raise RuntimeError("validated transition_ages unexpectedly missing")
    if (command_before is None) != (command_after is None):
        raise ValueError("command_before and command_after must be provided together")
    validated_before = _validate_command_tensor(
        "command_before",
        command_before,
        num_samples=num_samples,
    )
    validated_after = _validate_command_tensor(
        "command_after",
        command_after,
        num_samples=num_samples,
    )
    transition_mask = torch.tensor(
        [label == "walk_to_stop" for label in validated_labels],
        dtype=torch.bool,
        device=validated_ages.device,
    )
    static_mask = ~transition_mask
    if torch.any(validated_ages[static_mask] != -1):
        raise ValueError("static_stand/walk_flat rows must use transition_age=-1")
    if bool(transition_mask.any()):
        if validated_before is None or validated_after is None:
            raise ValueError("walk_to_stop rows require command_before and command_after")
        post_switch = transition_mask & (validated_ages >= 0)
        if torch.any(validated_after[post_switch].abs() > 1e-6):
            raise ValueError("walk_to_stop post-switch command_after must be zero")
    return validated_labels, validated_ages, validated_before, validated_after


@dataclass(frozen=True)
class DistillationTensorDataset:
    """In-memory offline distillation observations with explicit shape contracts."""

    student_obs: torch.Tensor
    teacher_obs: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    role_labels: tuple[str, ...] | None = None
    teacher_actions: torch.Tensor | None = None
    commands: torch.Tensor | None = None
    command_intents: tuple[str, ...] | None = None
    scenario_labels: tuple[str, ...] | None = None
    transition_ages: torch.Tensor | None = None
    command_before: torch.Tensor | None = None
    command_after: torch.Tensor | None = None

    @property
    def num_samples(self) -> int:
        return int(self.student_obs.shape[0])

    @property
    def student_obs_dim(self) -> int:
        return int(self.student_obs.shape[-1])

    @property
    def teacher_obs_dim(self) -> int:
        return int(self.teacher_obs.shape[-1])

    @property
    def teacher_action_dim(self) -> int | None:
        if self.teacher_actions is None:
            return None
        return int(self.teacher_actions.shape[-1])

    def to(self, device: str | torch.device) -> DistillationTensorDataset:
        """Move every tensor field to one learner device while preserving labels."""

        return replace(
            self,
            student_obs=self.student_obs.to(device),
            teacher_obs=self.teacher_obs.to(device),
            teacher_actions=(
                None if self.teacher_actions is None else self.teacher_actions.to(device)
            ),
            commands=None if self.commands is None else self.commands.to(device),
            transition_ages=(
                None if self.transition_ages is None else self.transition_ages.to(device)
            ),
            command_before=(
                None if self.command_before is None else self.command_before.to(device)
            ),
            command_after=(None if self.command_after is None else self.command_after.to(device)),
        )

    def as_batch(self, *, start: int = 0, batch_size: int | None = None) -> DistillationBatch:
        if start < 0 or start >= self.num_samples:
            raise ValueError(f"start must be in [0, {self.num_samples}), got {start}")
        end = self.num_samples if batch_size is None else min(self.num_samples, start + batch_size)
        if end <= start:
            raise ValueError(f"batch_size must select at least one sample, got {batch_size}")
        return DistillationBatch(
            student_obs=self.student_obs[start:end],
            teacher_obs=self.teacher_obs[start:end],
            role_labels=None if self.role_labels is None else self.role_labels[start:end],
            teacher_actions=(
                None if self.teacher_actions is None else self.teacher_actions[start:end]
            ),
            commands=None if self.commands is None else self.commands[start:end],
            command_intents=(
                None if self.command_intents is None else self.command_intents[start:end]
            ),
            scenario_labels=(
                None if self.scenario_labels is None else self.scenario_labels[start:end]
            ),
            transition_ages=(
                None if self.transition_ages is None else self.transition_ages[start:end]
            ),
            command_before=(
                None if self.command_before is None else self.command_before[start:end]
            ),
            command_after=(None if self.command_after is None else self.command_after[start:end]),
        )


def build_distillation_dataset(
    student_obs: torch.Tensor,
    teacher_obs: torch.Tensor,
    *,
    expected_student_obs_dim: int | None = None,
    expected_teacher_obs_dim: int | None = None,
    expected_teacher_action_dim: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    role_labels: list[str] | tuple[str, ...] | None = None,
    teacher_actions: torch.Tensor | None = None,
    commands: torch.Tensor | None = None,
    command_intents: list[str] | tuple[str, ...] | None = None,
    scenario_labels: list[str] | tuple[str, ...] | None = None,
    transition_ages: torch.Tensor | None = None,
    command_before: torch.Tensor | None = None,
    command_after: torch.Tensor | None = None,
) -> DistillationTensorDataset:
    """Validate and package offline student/teacher observations for distillation."""

    _validate_obs_tensor(
        "student_obs",
        student_obs,
        expected_dim=expected_student_obs_dim,
    )
    _validate_obs_tensor(
        "teacher_obs",
        teacher_obs,
        expected_dim=expected_teacher_obs_dim,
    )
    if student_obs.shape[0] != teacher_obs.shape[0]:
        raise ValueError(
            "student/teacher dataset batch size mismatch: "
            f"student={student_obs.shape[0]} teacher={teacher_obs.shape[0]}"
        )
    if teacher_actions is not None:
        _validate_action_tensor(
            "teacher_actions",
            teacher_actions,
            expected_dim=expected_teacher_action_dim,
        )
        if student_obs.shape[0] != teacher_actions.shape[0]:
            raise ValueError(
                "student/teacher action dataset batch size mismatch: "
                f"student={student_obs.shape[0]} teacher_actions={teacher_actions.shape[0]}"
            )
    validated_commands = _validate_commands(
        commands,
        num_samples=int(student_obs.shape[0]),
    )
    metadata_dict = dict(metadata or {})
    metadata_role_labels = metadata_dict.get("role_labels")
    if role_labels is None and metadata_role_labels is not None:
        if not isinstance(metadata_role_labels, list | tuple):
            raise ValueError("metadata role_labels must be a list or tuple")
        role_labels = [str(label) for label in metadata_role_labels]
    metadata_command_intents = metadata_dict.get("command_intents")
    if command_intents is None and metadata_command_intents is not None:
        if not isinstance(metadata_command_intents, list | tuple):
            raise ValueError("metadata command_intents must be a list or tuple")
        command_intents = [str(intent) for intent in metadata_command_intents]
    validated_role_labels = _validate_role_labels(
        role_labels,
        num_samples=int(student_obs.shape[0]),
    )
    if command_intents is None and validated_commands is not None:
        command_intents = _command_intents_from_commands(
            validated_commands,
            xy_threshold=float(metadata_dict.get("command_xy_threshold", 0.05)),
            yaw_threshold=float(metadata_dict.get("command_yaw_threshold", 0.05)),
        )
        metadata_dict["command_intent_inference_source"] = "commands"
    if command_intents is None and validated_role_labels is not None:
        command_intents = _command_intents_from_role_labels(validated_role_labels)
        if command_intents is not None:
            metadata_dict["command_intent_inference_source"] = "role_labels"
    validated_command_intents = _validate_command_intents(
        command_intents,
        num_samples=int(student_obs.shape[0]),
    )
    (
        validated_scenario_labels,
        validated_transition_ages,
        validated_command_before,
        validated_command_after,
    ) = _validate_transition_fields(
        scenario_labels=scenario_labels,
        transition_ages=transition_ages,
        command_before=command_before,
        command_after=command_after,
        num_samples=int(student_obs.shape[0]),
    )
    if validated_role_labels is not None:
        metadata_dict["role_labels"] = list(validated_role_labels)
    if validated_command_intents is not None:
        metadata_dict["command_intents"] = list(validated_command_intents)
        metadata_dict["command_intent_counts"] = _label_counts(validated_command_intents)
    if validated_scenario_labels is not None:
        metadata_dict["scenario_labels"] = list(validated_scenario_labels)
        metadata_dict["scenario_counts"] = _label_counts(validated_scenario_labels)
        metadata_dict["transition_schema"] = "DISTILL-TRAIN-v002"
    return DistillationTensorDataset(
        student_obs=student_obs,
        teacher_obs=teacher_obs,
        metadata=metadata_dict,
        role_labels=validated_role_labels,
        teacher_actions=teacher_actions,
        commands=validated_commands,
        command_intents=validated_command_intents,
        scenario_labels=validated_scenario_labels,
        transition_ages=validated_transition_ages,
        command_before=validated_command_before,
        command_after=validated_command_after,
    )


def make_fake_distillation_dataset(
    *,
    num_samples: int,
    student_obs_dim: int,
    teacher_obs_dim: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> DistillationTensorDataset:
    """Create a deterministic shape-valid dataset for offline connectivity probes."""

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    student_obs = torch.randn(
        int(num_samples),
        int(student_obs_dim),
        generator=generator,
    ).to(device)
    teacher_obs = torch.randn(
        int(num_samples),
        int(teacher_obs_dim),
        generator=generator,
    ).to(device)
    return build_distillation_dataset(
        student_obs,
        teacher_obs,
        expected_student_obs_dim=int(student_obs_dim),
        expected_teacher_obs_dim=int(teacher_obs_dim),
        metadata={"source": "fake_probe", "seed": int(seed)},
    )


def annotate_distillation_dataset_scenario(
    dataset: DistillationTensorDataset,
    scenario_label: str,
) -> DistillationTensorDataset:
    """Explicitly bind a legacy role dataset to a workflow scenario.

    This is a workflow annotation, not a teacher or action rewrite. It makes
    the scenario fields required by transition-aware aggregation explicit while
    preserving row-level role labels and cached targets.
    """

    scenario = str(scenario_label)
    if scenario not in _TRANSITION_SCENARIOS:
        raise ValueError(
            f"workflow scenario must be static_stand/walk_flat/walk_to_stop, got {scenario!r}"
        )
    if dataset.scenario_labels is not None:
        if any(label != scenario for label in dataset.scenario_labels):
            raise ValueError(
                f"dataset scenario labels do not match requested scenario {scenario!r}"
            )
        return dataset
    if scenario == "walk_to_stop":
        raise ValueError("walk_to_stop source must already contain transition fields")

    commands = dataset.commands
    if commands is None:
        if scenario == "walk_flat":
            raise ValueError("walk_flat scenario annotation requires dataset.commands")
        commands = torch.zeros(
            (dataset.num_samples, 3),
            dtype=dataset.student_obs.dtype,
            device=dataset.student_obs.device,
        )
    expected_intent = "active" if scenario == "walk_flat" else "inactive"
    if dataset.command_intents is not None and any(
        intent != expected_intent for intent in dataset.command_intents
    ):
        raise ValueError(f"{scenario} scenario annotation conflicts with command_intents")
    metadata = dict(dataset.metadata)
    metadata["scenario_annotation"] = "workflow_explicit"
    return build_distillation_dataset(
        dataset.student_obs,
        dataset.teacher_obs,
        expected_student_obs_dim=dataset.student_obs_dim,
        expected_teacher_obs_dim=dataset.teacher_obs_dim,
        expected_teacher_action_dim=dataset.teacher_action_dim,
        metadata=metadata,
        role_labels=dataset.role_labels,
        teacher_actions=dataset.teacher_actions,
        commands=commands,
        command_intents=dataset.command_intents,
        scenario_labels=(scenario,) * dataset.num_samples,
        transition_ages=torch.full(
            (dataset.num_samples,),
            -1,
            dtype=torch.int64,
            device=dataset.student_obs.device,
        ),
        command_before=commands.clone(),
        command_after=commands.clone(),
    )


def _source_value(source: Mapping[str, Any], key: str) -> Any:
    value = source.get(key)
    if value in (None, ""):
        raise ValueError(f"multitask source must define non-empty {key!r}")
    return value


def build_multitask_distillation_dataset(
    sources: Sequence[Mapping[str, Any]],
    *,
    expected_student_obs_dim: int | None = None,
    expected_teacher_obs_dim: int | None = None,
    expected_teacher_action_dim: int | None = None,
    device: str | torch.device = "cpu",
    preserve_source_role_labels: bool = False,
) -> DistillationTensorDataset:
    """Merge saved role-specific datasets into one cached-target dataset."""

    if not sources:
        raise ValueError("multitask distillation dataset requires at least one source")

    _emit_data_runtime(
        "multitask/entry",
        source_count=len(sources),
        expected_student_obs_dim=expected_student_obs_dim,
        expected_teacher_obs_dim=expected_teacher_obs_dim,
        expected_teacher_action_dim=expected_teacher_action_dim,
        device=str(device),
    )

    datasets: list[DistillationTensorDataset] = []
    source_roles: list[str] = []
    source_paths: list[str] = []
    source_sample_counts: list[int] = []
    source_metadata: list[dict[str, Any]] = []
    source_preserve_role_labels: list[bool] = []
    source_scenarios: list[str | None] = []
    source_student_obs_dim: int | None = None
    source_teacher_obs_dim: int | None = None
    source_teacher_action_dim: int | None = None
    source_has_commands: bool | None = None
    source_has_command_intents: bool | None = None
    source_transition_presence: dict[str, bool] | None = None
    for loop_source_index, source in enumerate(sources):
        source_index = int(source.get("source_index", loop_source_index))
        path = Path(_source_value(source, "path"))
        role = str(_source_value(source, "role"))
        scenario = source.get("scenario")
        requested_scenario = None if scenario in (None, "") else str(scenario)
        _emit_data_runtime(
            "multitask/before_source_load",
            source_index=source_index,
            path=str(path),
            role=role,
            scenario=requested_scenario,
        )
        dataset = load_distillation_dataset(
            path,
            expected_student_obs_dim=expected_student_obs_dim,
            expected_teacher_obs_dim=expected_teacher_obs_dim,
            expected_teacher_action_dim=expected_teacher_action_dim,
            device=device,
        )
        metadata_scenario = _metadata_workflow_scenario(dataset)
        if requested_scenario is None:
            requested_scenario = metadata_scenario
        elif metadata_scenario is not None and metadata_scenario != requested_scenario:
            snapshot = {
                "stage": "multitask/source_scenario_contract_mismatch",
                "pid": os.getpid(),
                **_multitask_source_debug_snapshot(
                    source_index=source_index,
                    path=path,
                    role=role,
                    scenario=requested_scenario,
                    dataset=dataset,
                ),
                "metadata_workflow_scenario": metadata_scenario,
            }
            print(
                "[distill-source-contract-sentinel] " + json.dumps(snapshot, sort_keys=True),
                flush=True,
            )
            raise ValueError(
                "multitask source scenario contract mismatch: "
                + json.dumps(snapshot, sort_keys=True)
            )
        if requested_scenario is not None:
            try:
                dataset = annotate_distillation_dataset_scenario(
                    dataset,
                    requested_scenario,
                )
            except ValueError as error:
                snapshot = {
                    "stage": "multitask/source_annotation_failure",
                    "pid": os.getpid(),
                    **_multitask_source_debug_snapshot(
                        source_index=source_index,
                        path=path,
                        role=role,
                        scenario=requested_scenario,
                        dataset=dataset,
                        error=error,
                    ),
                }
                _emit_data_runtime(
                    "multitask/source_annotation_failure",
                    **{key: value for key, value in snapshot.items() if key != "stage"},
                )
                print(
                    "[distill-source-annotation-sentinel] " + json.dumps(snapshot, sort_keys=True),
                    flush=True,
                )
                raise ValueError(
                    "multitask source scenario annotation failed: "
                    + json.dumps(snapshot, sort_keys=True)
                ) from error
        _emit_data_runtime(
            "multitask/after_source_annotation",
            source_index=source_index,
            path=str(path),
            role=role,
            scenario=requested_scenario,
            num_samples=dataset.num_samples,
            student_obs_shape=tuple(dataset.student_obs.shape),
            teacher_obs_shape=tuple(dataset.teacher_obs.shape),
            teacher_actions_shape=(
                None if dataset.teacher_actions is None else tuple(dataset.teacher_actions.shape)
            ),
            command_intents=(
                None
                if dataset.command_intents is None
                else _command_intent_debug_snapshot(dataset.command_intents)
            ),
            scenario_labels=(
                None
                if dataset.scenario_labels is None
                else _scenario_label_debug_snapshot(dataset.scenario_labels)
            ),
        )
        preserve_row_labels = bool(
            source.get("preserve_row_role_labels", preserve_source_role_labels)
        )
        if preserve_row_labels and dataset.role_labels is None:
            raise ValueError(
                f"multitask source {path} requires row role_labels when preservation is enabled"
            )
        if dataset.teacher_actions is None:
            raise ValueError(f"multitask source {path} must contain cached teacher_actions")
        has_commands = dataset.commands is not None
        if source_has_commands is None:
            source_has_commands = has_commands
        elif has_commands != source_has_commands:
            raise ValueError("multitask sources must either all include commands or none")
        has_command_intents = dataset.command_intents is not None
        if source_has_command_intents is None:
            source_has_command_intents = has_command_intents
        elif has_command_intents != source_has_command_intents:
            raise ValueError("multitask sources must either all include command_intents or none")
        transition_presence = {
            "scenario_labels": dataset.scenario_labels is not None,
            "transition_ages": dataset.transition_ages is not None,
            "command_before": dataset.command_before is not None,
            "command_after": dataset.command_after is not None,
        }
        if source_transition_presence is None:
            source_transition_presence = transition_presence
        elif transition_presence != source_transition_presence:
            mismatched_field = next(
                field_name
                for field_name in transition_presence
                if transition_presence[field_name] != source_transition_presence[field_name]
            )
            raise ValueError(
                f"multitask sources must either all include {mismatched_field} or none"
            )
        if source_student_obs_dim is None:
            source_student_obs_dim = dataset.student_obs_dim
        elif dataset.student_obs_dim != source_student_obs_dim:
            raise ValueError(
                f"multitask source {path} role={role!r} student_obs dim mismatch: "
                f"expected {source_student_obs_dim}, got {dataset.student_obs_dim}"
            )
        if source_teacher_obs_dim is None:
            source_teacher_obs_dim = dataset.teacher_obs_dim
        elif dataset.teacher_obs_dim != source_teacher_obs_dim:
            raise ValueError(
                f"multitask source {path} role={role!r} teacher_obs dim mismatch: "
                f"expected {source_teacher_obs_dim}, got {dataset.teacher_obs_dim}"
            )
        if source_teacher_action_dim is None:
            source_teacher_action_dim = dataset.teacher_action_dim
        elif dataset.teacher_action_dim != source_teacher_action_dim:
            raise ValueError(
                f"multitask source {path} role={role!r} teacher_actions dim mismatch: "
                f"expected {source_teacher_action_dim}, got {dataset.teacher_action_dim}"
            )
        datasets.append(dataset)
        source_roles.append(role)
        source_paths.append(str(path))
        source_sample_counts.append(dataset.num_samples)
        source_metadata.append(dict(dataset.metadata))
        source_preserve_role_labels.append(preserve_row_labels)
        source_scenarios.append(requested_scenario)

    if source_transition_presence is None:
        raise RuntimeError("multitask source transition presence was not initialized")
    validated_transition_presence = source_transition_presence

    student_obs = torch.cat([dataset.student_obs for dataset in datasets], dim=0)
    teacher_obs = torch.cat([dataset.teacher_obs for dataset in datasets], dim=0)
    teacher_actions = torch.cat(
        [dataset.teacher_actions for dataset in datasets if dataset.teacher_actions is not None],
        dim=0,
    )
    commands = (
        torch.cat([dataset.commands for dataset in datasets if dataset.commands is not None], dim=0)
        if source_has_commands
        else None
    )
    command_intents = (
        tuple(
            intent
            for dataset in datasets
            if dataset.command_intents is not None
            for intent in dataset.command_intents
        )
        if source_has_command_intents
        else None
    )
    scenario_source_ranges: list[dict[str, Any]] = []
    if validated_transition_presence["scenario_labels"]:
        global_start = 0
        for source_index, (source_path, role, scenario, dataset) in enumerate(
            zip(
                source_paths,
                source_roles,
                source_scenarios,
                datasets,
                strict=True,
            )
        ):
            assert dataset.scenario_labels is not None
            global_stop = global_start + len(dataset.scenario_labels)
            source_range = {
                "source_index": source_index,
                "path": source_path,
                "role": role,
                "scenario": scenario,
                "global_start": global_start,
                "global_stop": global_stop,
            }
            scenario_source_ranges.append(source_range)
            _emit_data_runtime(
                "multitask/scenario_source_ready",
                **source_range,
                num_samples=dataset.num_samples,
                scenario_labels=_scenario_label_debug_snapshot(dataset.scenario_labels),
            )
            global_start = global_stop
    scenario_labels = (
        tuple(
            label
            for dataset in datasets
            if dataset.scenario_labels is not None
            for label in dataset.scenario_labels
        )
        if validated_transition_presence["scenario_labels"]
        else None
    )
    if scenario_labels is not None:
        for source_range, dataset in zip(
            scenario_source_ranges,
            datasets,
            strict=True,
        ):
            assert dataset.scenario_labels is not None
            global_start = cast(int, source_range["global_start"])
            global_stop = cast(int, source_range["global_stop"])
            aggregate_slice = scenario_labels[global_start:global_stop]
            _emit_data_runtime(
                "multitask/scenario_concat_chunk",
                **source_range,
                observation_timing="post_flatten_slice_check",
                source_scenario_labels=_scenario_label_debug_snapshot(dataset.scenario_labels),
                aggregate_slice=_scenario_label_debug_snapshot(aggregate_slice),
                source_matches_aggregate_slice=(dataset.scenario_labels == aggregate_slice),
            )
        _emit_data_runtime(
            "multitask/scenario_concat_complete",
            source_count=len(datasets),
            scenario_labels=_scenario_label_debug_snapshot(
                scenario_labels,
                source_ranges=scenario_source_ranges,
            ),
        )
    transition_ages = (
        torch.cat(
            [
                dataset.transition_ages
                for dataset in datasets
                if dataset.transition_ages is not None
            ],
            dim=0,
        )
        if validated_transition_presence["transition_ages"]
        else None
    )
    command_before = (
        torch.cat(
            [dataset.command_before for dataset in datasets if dataset.command_before is not None],
            dim=0,
        )
        if validated_transition_presence["command_before"]
        else None
    )
    command_after = (
        torch.cat(
            [dataset.command_after for dataset in datasets if dataset.command_after is not None],
            dim=0,
        )
        if validated_transition_presence["command_after"]
        else None
    )
    role_label_chunks: list[tuple[str, ...]] = []
    for role, dataset, preserve in zip(
        source_roles,
        datasets,
        source_preserve_role_labels,
        strict=True,
    ):
        if preserve:
            assert dataset.role_labels is not None
            role_label_chunks.append(dataset.role_labels)
        else:
            role_label_chunks.append((role,) * dataset.num_samples)
    role_labels = tuple(label for labels in role_label_chunks for label in labels)
    _emit_data_runtime(
        "multitask/after_concat",
        source_count=len(datasets),
        student_obs_shape=tuple(student_obs.shape),
        teacher_obs_shape=tuple(teacher_obs.shape),
        teacher_actions_shape=tuple(teacher_actions.shape),
        commands_shape=None if commands is None else tuple(commands.shape),
        command_intents=(
            None if command_intents is None else _command_intent_debug_snapshot(command_intents)
        ),
        scenario_labels=(
            None
            if scenario_labels is None
            else _scenario_label_debug_snapshot(
                scenario_labels,
                source_ranges=scenario_source_ranges,
            )
        ),
        role_labels_length=len(role_labels),
    )
    metadata = {
        "source": "multitask_adapter",
        "source_count": len(datasets),
        "source_paths": source_paths,
        "source_roles": source_roles,
        "source_sample_counts": source_sample_counts,
        "source_metadata": source_metadata,
        "source_scenarios": source_scenarios,
    }
    if command_intents is not None:
        metadata["command_intent_counts"] = _label_counts(command_intents)
    before_final_validation = (
        None if command_intents is None else _command_intent_debug_snapshot(command_intents)
    )
    before_final_scenario_validation = (
        None
        if scenario_labels is None
        else _scenario_label_debug_snapshot(
            scenario_labels,
            source_ranges=scenario_source_ranges,
        )
    )
    _emit_data_runtime(
        "multitask/before_final_validation",
        source_count=len(datasets),
        command_intents=before_final_validation,
        scenario_labels=before_final_scenario_validation,
        student_obs_shape=tuple(student_obs.shape),
        teacher_obs_shape=tuple(teacher_obs.shape),
        role_labels_length=len(role_labels),
    )
    try:
        result = build_distillation_dataset(
            student_obs,
            teacher_obs,
            expected_student_obs_dim=expected_student_obs_dim,
            expected_teacher_obs_dim=expected_teacher_obs_dim,
            expected_teacher_action_dim=expected_teacher_action_dim,
            metadata=metadata,
            role_labels=role_labels,
            teacher_actions=teacher_actions,
            commands=commands,
            command_intents=command_intents,
            scenario_labels=scenario_labels,
            transition_ages=transition_ages,
            command_before=command_before,
            command_after=command_after,
        )
        _emit_data_runtime(
            "multitask/after_final_validation",
            source_count=len(datasets),
            num_samples=result.num_samples,
            command_intents=(
                None
                if result.command_intents is None
                else _command_intent_debug_snapshot(result.command_intents)
            ),
            scenario_labels=(
                None
                if result.scenario_labels is None
                else _scenario_label_debug_snapshot(
                    result.scenario_labels,
                    source_ranges=scenario_source_ranges,
                )
            ),
        )
        return result
    except ValueError as error:
        error_text = _ORIGINAL_STR(error)
        if scenario_labels is not None and "scenario_labels" in error_text:
            after_failure = _scenario_label_debug_snapshot(
                scenario_labels,
                source_ranges=scenario_source_ranges,
            )
            _emit_data_runtime(
                "multitask/final_validation_failure",
                source_count=len(datasets),
                error_type=_ORIGINAL_TYPE(error).__name__,
                error_repr=_safe_runtime_repr(error),
                scenario_labels_before=before_final_scenario_validation,
                scenario_labels_after=after_failure,
            )
            scenario_sources_snapshot = []
            for source_range, dataset in zip(
                scenario_source_ranges,
                datasets,
                strict=True,
            ):
                assert dataset.scenario_labels is not None
                scenario_sources_snapshot.append(
                    {
                        **source_range,
                        "num_samples": dataset.num_samples,
                        "scenario_labels": _scenario_label_debug_snapshot(dataset.scenario_labels),
                    }
                )
            scenario_snapshot = {
                "stage": "multitask/final_validation_failure",
                "pid": os.getpid(),
                "error_type": _ORIGINAL_TYPE(error).__name__,
                "error": error_text,
                "source_count": len(datasets),
                "sources": scenario_sources_snapshot,
                "aggregate": after_failure,
                "before_final_validation": before_final_scenario_validation,
            }
            print(
                "[distill-scenario-label-sentinel] "
                + json.dumps(scenario_snapshot, sort_keys=True),
                flush=True,
            )
            raise
        if command_intents is None or "command_intents" not in error_text:
            raise
        _emit_data_runtime(
            "multitask/final_validation_failure",
            source_count=len(datasets),
            error_type=_ORIGINAL_TYPE(error).__name__,
            error_repr=_safe_runtime_repr(error),
            command_intents_before=before_final_validation,
            command_intents_after=_command_intent_debug_snapshot(command_intents),
        )
        sources_snapshot = []
        for source_path, role, scenario, dataset in zip(
            source_paths,
            source_roles,
            source_scenarios,
            datasets,
            strict=True,
        ):
            assert dataset.command_intents is not None
            source_intents = _command_intent_debug_snapshot(dataset.command_intents)
            sources_snapshot.append(
                {
                    "path": source_path,
                    "role": role,
                    "scenario": scenario,
                    "num_samples": dataset.num_samples,
                    "command_intent_counts": source_intents["command_intent_counts"],
                    "invalid_head": source_intents["invalid_head"],
                }
            )
        snapshot = {
            "stage": "multitask/final_validation_failure",
            "pid": os.getpid(),
            "error_type": _ORIGINAL_TYPE(error).__name__,
            "error": error_text,
            "source_count": len(datasets),
            "sources": sources_snapshot,
            "before_final_validation": before_final_validation,
            "after_final_validation_failure": _command_intent_debug_snapshot(command_intents),
        }
        print("[distill-command-intent-sentinel] " + json.dumps(snapshot, sort_keys=True))
        raise


def save_distillation_dataset(path: str | Path, dataset: DistillationTensorDataset) -> None:
    """Persist an offline distillation observation dataset."""

    payload = {
        "student_obs": dataset.student_obs.detach().cpu(),
        "teacher_obs": dataset.teacher_obs.detach().cpu(),
        "metadata": dict(dataset.metadata),
        "role_labels": None if dataset.role_labels is None else list(dataset.role_labels),
        "teacher_actions": (
            None if dataset.teacher_actions is None else dataset.teacher_actions.detach().cpu()
        ),
        "commands": None if dataset.commands is None else dataset.commands.detach().cpu(),
        "command_intents": (
            None if dataset.command_intents is None else list(dataset.command_intents)
        ),
        "scenario_labels": (
            None if dataset.scenario_labels is None else list(dataset.scenario_labels)
        ),
        "transition_ages": (
            None if dataset.transition_ages is None else dataset.transition_ages.detach().cpu()
        ),
        "command_before": (
            None if dataset.command_before is None else dataset.command_before.detach().cpu()
        ),
        "command_after": (
            None if dataset.command_after is None else dataset.command_after.detach().cpu()
        ),
        "student_obs_dim": dataset.student_obs_dim,
        "teacher_obs_dim": dataset.teacher_obs_dim,
        "teacher_action_dim": dataset.teacher_action_dim,
        "num_samples": dataset.num_samples,
    }
    resolved_path = Path(path)
    _emit_data_runtime(
        "serialization/before_torch_save",
        path=str(resolved_path),
        payload_keys=sorted(payload),
        payload_value_types={key: type(value).__name__ for key, value in payload.items()},
        num_samples=dataset.num_samples,
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = resolved_path.with_name(
        f".{resolved_path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(resolved_path)
    except Exception as error:
        native_abort_requested = _native_abort_for_impossible_callable_error_requested(error)
        _emit_data_runtime(
            "serialization/torch_save_failure",
            path=str(resolved_path),
            tmp_path=str(tmp_path),
            error_type=type(error).__name__,
            error_repr=repr(error),
            native_abort_requested=native_abort_requested,
        )
        if native_abort_requested:
            _abort_for_native_capture()
        raise
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    _emit_data_runtime(
        "serialization/after_torch_save",
        path=str(resolved_path),
        payload_keys=sorted(payload),
        file_exists=resolved_path.is_file(),
        file_size=resolved_path.stat().st_size if resolved_path.is_file() else None,
    )


def load_distillation_dataset(
    path: str | Path,
    *,
    expected_student_obs_dim: int | None = None,
    expected_teacher_obs_dim: int | None = None,
    expected_teacher_action_dim: int | None = None,
    device: str | torch.device = "cpu",
) -> DistillationTensorDataset:
    """Load and validate an offline distillation observation dataset."""

    resolved_path = Path(path)
    _emit_data_runtime(
        "serialization/before_torch_load",
        path=str(resolved_path),
        device=str(device),
        file_exists=resolved_path.is_file(),
        file_size=resolved_path.stat().st_size if resolved_path.is_file() else None,
    )
    try:
        payload = torch.load(resolved_path, map_location=device, weights_only=False)
    except Exception as error:
        native_abort_requested = _native_abort_for_impossible_callable_error_requested(error)
        _emit_data_runtime(
            "serialization/torch_load_failure",
            path=str(resolved_path),
            device=str(device),
            error_type=type(error).__name__,
            error_repr=repr(error),
            native_abort_requested=native_abort_requested,
        )
        if native_abort_requested:
            _abort_for_native_capture()
        raise
    _emit_data_runtime(
        "serialization/after_torch_load",
        path=str(resolved_path),
        device=str(device),
        payload_type=type(payload).__name__,
        payload_keys=sorted(payload),
        payload_value_types={key: type(value).__name__ for key, value in payload.items()},
    )
    teacher_actions = payload.get("teacher_actions")
    commands = payload.get("commands")
    transition_ages = payload.get("transition_ages")
    command_before = payload.get("command_before")
    command_after = payload.get("command_after")
    dataset = build_distillation_dataset(
        payload["student_obs"].to(device),
        payload["teacher_obs"].to(device),
        expected_student_obs_dim=expected_student_obs_dim,
        expected_teacher_obs_dim=expected_teacher_obs_dim,
        expected_teacher_action_dim=expected_teacher_action_dim,
        metadata=payload.get("metadata", {}),
        role_labels=payload.get("role_labels"),
        teacher_actions=None if teacher_actions is None else teacher_actions.to(device),
        commands=None if commands is None else commands.to(device),
        command_intents=payload.get("command_intents"),
        scenario_labels=payload.get("scenario_labels"),
        transition_ages=(None if transition_ages is None else transition_ages.to(device)),
        command_before=(None if command_before is None else command_before.to(device)),
        command_after=(None if command_after is None else command_after.to(device)),
    )
    expected_count = payload.get("num_samples")
    if expected_count is not None and int(expected_count) != dataset.num_samples:
        raise ValueError(
            "distillation dataset num_samples mismatch: "
            f"payload={expected_count} tensors={dataset.num_samples}"
        )
    return dataset
