"""Shared rollout IPC ring buffer for APPO / async PPO."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict

import numpy as np

_SPAWN_CTX = mp.get_context("spawn")

_FIELD_SHAPES = {
    "obs": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns, od),
    "critic": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns, cd),
    "actions": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns, ad),
    "log_probs": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "rewards": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "dones": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "truncated": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, ns),
    "last_obs": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, od),
    "last_critic": lambda ns_slots, ne, ns, od, ad, cd: (ns_slots, ne, cd),
}


@dataclass(frozen=True)
class RolloutFieldSpec:
    """Per-slot layout contract for one rollout payload field."""

    shape: tuple[int, ...]
    dtype: str = "float32"
    time_axis: bool = True

    def __post_init__(self) -> None:
        shape = tuple(int(dim) for dim in self.shape)
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"rollout field shape must contain positive dimensions, got {shape}")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(f"unsupported rollout field dtype {self.dtype!r}") from exc
        if dtype.kind not in "biuf":
            raise ValueError(f"rollout field dtype must be numeric or bool, got {dtype}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", dtype.name)


class RolloutRingBuffer:
    """N-slot shared-memory ring buffer for raw rollout payloads."""

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        *,
        critic_dim: int = 0,
        num_slots: int = 4,
        create: bool = True,
        shm_name_prefix: Dict[str, str] | None = None,
        extra_fields: Mapping[str, RolloutFieldSpec] | None = None,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.critic_dim = critic_dim
        self.num_slots = num_slots
        self._owns_shm = bool(create)
        self._handle_closed = False
        self._unlinked = False

        self._shm_blocks: Dict[str, shared_memory.SharedMemory] = {}
        self._arrays: Dict[str, np.ndarray] = {}

        fields_to_allocate = {k: v for k, v in _FIELD_SHAPES.items()}
        if critic_dim == 0:
            fields_to_allocate.pop("critic", None)
            fields_to_allocate.pop("last_critic", None)

        field_specs: dict[str, RolloutFieldSpec] = {}
        for field, shape_fn in fields_to_allocate.items():
            full_shape = shape_fn(
                num_slots,
                num_envs,
                num_steps,
                obs_dim,
                action_dim,
                critic_dim,
            )
            field_specs[field] = RolloutFieldSpec(
                shape=tuple(full_shape[1:]),
                time_axis=field not in {"last_obs", "last_critic"},
            )

        for field, spec in (extra_fields or {}).items():
            if field in field_specs:
                raise ValueError(f"extra rollout field {field!r} collides with a built-in field")
            if not isinstance(spec, RolloutFieldSpec):
                raise TypeError(f"extra rollout field {field!r} must use RolloutFieldSpec")
            if spec.shape[0] != num_envs:
                raise ValueError(
                    f"extra rollout field {field!r} must start with num_envs={num_envs}; "
                    f"got {spec.shape}"
                )
            if spec.time_axis and (len(spec.shape) < 2 or spec.shape[1] != num_steps):
                raise ValueError(
                    f"time-axis rollout field {field!r} must have shape "
                    f"({num_envs}, {num_steps}, ...); got {spec.shape}"
                )
            field_specs[field] = spec

        if not create:
            assert shm_name_prefix is not None, "shm_name_prefix required when create=False"
            expected = set(field_specs)
            provided = set(shm_name_prefix)
            if expected != provided:
                raise ValueError(
                    "shared-memory field names do not match rollout contract: "
                    f"missing={sorted(expected - provided)}, unexpected={sorted(provided - expected)}"
                )

        self._field_specs = field_specs
        for field, spec in field_specs.items():
            shape = (num_slots, *spec.shape)
            dtype = np.dtype(spec.dtype)
            nbytes = int(np.prod(shape)) * dtype.itemsize

            if create:
                shm = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
            else:
                assert shm_name_prefix is not None, "shm_name_prefix required when create=False"
                shm = shared_memory.SharedMemory(name=shm_name_prefix[field], create=False)

            self._shm_blocks[field] = shm
            self._arrays[field] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        if create:
            self._write_ptr = _SPAWN_CTX.Value("l", 0)
            self._read_ptr = _SPAWN_CTX.Value("l", 0)

    @property
    def name(self) -> Dict[str, str]:
        return {field: shm.name for field, shm in self._shm_blocks.items()}

    @property
    def slot_shapes(self) -> Dict[str, tuple[int, ...]]:
        return {field: tuple(arr.shape[1:]) for field, arr in self._arrays.items()}

    @property
    def field_specs(self) -> dict[str, RolloutFieldSpec]:
        return dict(self._field_specs)

    def attach_sync_primitives(self, write_ptr, read_ptr) -> None:
        self._write_ptr = write_ptr
        self._read_ptr = read_ptr

    def _clamp_read_ptr_to_valid_window(self) -> None:
        wp = int(self._write_ptr.value)
        oldest_available = max(0, wp - self.num_slots)
        if int(self._read_ptr.value) >= oldest_available:
            return
        with self._read_ptr.get_lock():
            if int(self._read_ptr.value) < oldest_available:
                self._read_ptr.value = oldest_available

    @property
    def write_slot(self) -> int:
        return int(self._write_ptr.value) % self.num_slots

    @property
    def write_buffer(self) -> Dict[str, np.ndarray]:
        s = self.write_slot
        return {field: arr[s] for field, arr in self._arrays.items()}

    def signal_write_done(self) -> None:
        with self._write_ptr.get_lock():
            self._write_ptr.value += 1

    def available(self) -> int:
        self._clamp_read_ptr_to_valid_window()
        return min(max(0, int(self._write_ptr.value) - int(self._read_ptr.value)), self.num_slots)

    def wait_for_data(self, timeout: float = 60.0) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while self.available() == 0:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.001)
        return True

    @property
    def read_slot(self) -> int:
        self._clamp_read_ptr_to_valid_window()
        return int(self._read_ptr.value) % self.num_slots

    def read_numpy_views(self) -> dict[str, np.ndarray]:
        """Return shared-memory views for the current read slot.

        The returned arrays are borrowed views. Consumers must copy them into
        owned storage before calling advance_read().
        """
        s = self.read_slot
        return {field: arr[s] for field, arr in self._arrays.items()}

    def copy_read_slot_to_torch(self, destination: dict) -> None:
        import torch

        s = self.read_slot
        for field, arr in self._arrays.items():
            if field not in destination:
                raise KeyError(f"missing destination tensor for rollout field {field!r}")
            dst = destination[field]
            src_view = arr[s]
            if tuple(dst.shape) != tuple(src_view.shape):
                raise ValueError(
                    f"destination shape mismatch for {field!r}: "
                    f"expected {tuple(src_view.shape)}, got {tuple(dst.shape)}"
                )
            expected_dtype = torch.from_numpy(np.empty((), dtype=src_view.dtype)).dtype
            if dst.dtype != expected_dtype:
                raise TypeError(
                    f"destination tensor for {field!r} must be {expected_dtype}, got {dst.dtype}"
                )
            dst.copy_(torch.from_numpy(src_view), non_blocking=False)

    def read_torch(self, device: str) -> dict:
        import torch

        result = {
            field: torch.empty(
                tuple(arr.shape[1:]),
                dtype=torch.from_numpy(np.empty((), dtype=arr.dtype)).dtype,
                device=device,
            )
            for field, arr in self._arrays.items()
        }
        self.copy_read_slot_to_torch(result)
        return result

    def advance_read(self) -> None:
        with self._read_ptr.get_lock():
            wp = int(self._write_ptr.value)
            rp = min(int(self._read_ptr.value) + 1, wp)
            oldest_available = max(0, wp - self.num_slots)
            self._read_ptr.value = max(rp, oldest_available)

    def cleanup(self) -> None:
        self._release(unlink=self._owns_shm)

    def _release(self, *, unlink: bool) -> None:
        errors: list[str] = []
        if not self._handle_closed:
            self._arrays.clear()
            for field, shm in self._shm_blocks.items():
                try:
                    shm.close()
                except Exception as exc:
                    errors.append(f"{field}.close: {type(exc).__name__}: {exc}")
            self._handle_closed = True
        if unlink and not self._unlinked:
            for field, shm in self._shm_blocks.items():
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    errors.append(f"{field}.unlink: {type(exc).__name__}: {exc}")
            self._unlinked = True
        if errors:
            raise RuntimeError("RolloutRingBuffer release failed: " + "; ".join(errors))

    def close(self) -> None:
        self._release(unlink=False)
