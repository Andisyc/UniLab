"""Bounded rollout staging for APPO learners."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from unilab.ipc.rollout_ring_buffer import RolloutFieldSpec

_LAST_FIELDS = frozenset({"last_obs", "last_critic"})
_FIELD_ALIASES = {
    "obs": "observations",
    "log_probs": "actions_log_prob",
}


class RolloutStagingPool:
    """Preallocated learner-device storage for a bounded set of rollouts.

    Raw IPC rollout slots are env-major: [N, T, ...].  Learners consume
    time-major combined batches: [T, K*N, ...].  The pool owns the destination
    tensors and exposes active views without rebuilding them via torch.cat.
    """

    def __init__(
        self,
        *,
        capacity: int,
        num_envs: int,
        device: str | torch.device,
        slot_shapes: Mapping[str, tuple[int, ...]] | None = None,
        field_specs: Mapping[str, RolloutFieldSpec] | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("RolloutStagingPool capacity must be >= 1")
        if num_envs < 1:
            raise ValueError("RolloutStagingPool num_envs must be >= 1")

        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._next_slot = 0
        self._active_count = 0
        self._writes = 0
        self._slot_versions = [-1] * self.capacity
        self._raw_to_batch_field: dict[str, str] = {}
        self._buffers: dict[str, torch.Tensor] = {}
        self._time_axis_fields: set[str] = set()
        self._field_dtypes: dict[str, np.dtype] = {}

        if (slot_shapes is None) == (field_specs is None):
            raise ValueError("provide exactly one of slot_shapes or field_specs")
        if field_specs is not None:
            for raw_field, spec in field_specs.items():
                if not isinstance(spec, RolloutFieldSpec):
                    raise TypeError(f"field spec for {raw_field!r} must use RolloutFieldSpec")
                self._allocate_field(
                    raw_field,
                    spec.shape,
                    dtype=np.dtype(spec.dtype),
                    time_axis=spec.time_axis,
                )
        else:
            assert slot_shapes is not None
            for raw_field, slot_shape in slot_shapes.items():
                self._allocate_field(
                    raw_field,
                    tuple(slot_shape),
                    dtype=np.dtype(np.float32),
                    time_axis=raw_field not in _LAST_FIELDS,
                )

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def slot_versions(self) -> tuple[int, ...]:
        return tuple(self._slot_versions)

    def _allocate_field(
        self,
        raw_field: str,
        slot_shape: tuple[int, ...],
        *,
        dtype: np.dtype,
        time_axis: bool,
    ) -> None:
        if not slot_shape or slot_shape[0] != self.num_envs:
            raise ValueError(
                f"rollout field {raw_field!r} must start with num_envs={self.num_envs}; "
                f"got shape {slot_shape}"
            )

        batch_field = _FIELD_ALIASES.get(raw_field, raw_field)
        self._raw_to_batch_field[raw_field] = batch_field
        self._field_dtypes[raw_field] = dtype
        if time_axis:
            self._time_axis_fields.add(raw_field)

        if time_axis:
            if len(slot_shape) < 2:
                raise ValueError(f"rollout field {raw_field!r} must include a time dimension")
            combined_shape = (
                slot_shape[1],
                self.capacity * self.num_envs,
                *slot_shape[2:],
            )
        else:
            combined_shape = (self.capacity * self.num_envs, *slot_shape[1:])
        self._buffers[batch_field] = torch.empty(
            combined_shape,
            dtype=torch.from_numpy(np.empty((), dtype=dtype)).dtype,
            device=self.device,
        )

    def _slot_view(self, raw_field: str, slot: int) -> torch.Tensor:
        batch_field = self._raw_to_batch_field[raw_field]
        storage = self._buffers[batch_field]
        start = slot * self.num_envs
        end = start + self.num_envs
        if raw_field in self._time_axis_fields:
            return storage[:, start:end, ...]
        return storage[start:end]

    def stage_numpy_views(self, raw_views: Mapping[str, np.ndarray]) -> int:
        """Copy one raw shared-memory rollout into the next staging slot."""
        missing = self._raw_to_batch_field.keys() - raw_views.keys()
        if missing:
            raise KeyError(f"missing rollout fields for staging: {sorted(missing)}")

        slot = self._next_slot
        for raw_field, raw_view in raw_views.items():
            if raw_field not in self._raw_to_batch_field:
                raise KeyError(f"unexpected rollout field {raw_field!r}")
            expected_dtype = self._field_dtypes[raw_field]
            if raw_view.dtype != expected_dtype:
                raise TypeError(
                    f"rollout field {raw_field!r} must be {expected_dtype}, got {raw_view.dtype}"
                )

            src = torch.from_numpy(raw_view)
            if raw_field in self._time_axis_fields:
                src = src.transpose(0, 1)

            dst = self._slot_view(raw_field, slot)
            if tuple(src.shape) != tuple(dst.shape):
                raise ValueError(
                    f"rollout field {raw_field!r} shape mismatch: "
                    f"expected {tuple(dst.shape)}, got {tuple(src.shape)}"
                )
            dst.copy_(src, non_blocking=False)

        self._slot_versions[slot] = self._writes
        self._writes += 1
        self._active_count = min(self._active_count + 1, self.capacity)
        self._next_slot = (slot + 1) % self.capacity
        return slot

    def batch(self) -> dict[str, torch.Tensor]:
        """Return active learner-ready views backed by the staging pool."""
        if self._active_count == 0:
            raise RuntimeError("RolloutStagingPool has no active rollouts")

        active_envs = self._active_count * self.num_envs
        out: dict[str, torch.Tensor] = {}
        for raw_field, batch_field in self._raw_to_batch_field.items():
            storage = self._buffers[batch_field]
            if raw_field in self._time_axis_fields:
                out[batch_field] = storage[:, :active_envs, ...]
            else:
                out[batch_field] = storage[:active_envs]
        return out
