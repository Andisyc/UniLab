"""Fail-closed forward-walk expert dataset for AMP training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unilab.algos.torch.amp.spec import (
    AMP_ANCHOR_BODY_INDEX,
    AMP_ANCHOR_BODY_NAME,
    AMP_BODY_INDICES,
    AMP_BODY_NAMES,
    build_amp_observation,
)

_REQUIRED_ARRAYS = (
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


@dataclass(frozen=True)
class AMPTransitionBatch:
    current: np.ndarray
    next: np.ndarray
    motion_index: np.ndarray
    frame_index: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_contract(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("AMP walk manifest schema_version must be 1")
    if manifest.get("motion_class") != "forward_walk":
        raise ValueError("AMP walk manifest motion_class must be forward_walk")
    if tuple(manifest.get("body_names", ())) != AMP_BODY_NAMES:
        raise ValueError("AMP walk manifest body_names do not match the canonical contract")
    if tuple(manifest.get("body_indices", ())) != AMP_BODY_INDICES:
        raise ValueError("AMP walk manifest body_indices do not match the canonical contract")
    if manifest.get("anchor_body_name") != AMP_ANCHOR_BODY_NAME:
        raise ValueError("AMP walk manifest anchor_body_name does not match the canonical contract")
    if manifest.get("anchor_body_index") != AMP_ANCHOR_BODY_INDEX:
        raise ValueError("AMP walk manifest anchor_body_index does not match the canonical contract")


class WalkMotionDataset:
    """Precomputed adjacent expert transitions from an explicit file manifest."""

    def __init__(self, *, names: tuple[str, ...], features: tuple[np.ndarray, ...]) -> None:
        if not names or len(names) != len(features):
            raise ValueError("walk motion dataset requires matching non-empty names and features")
        self.motion_names = names
        self.motion_features = features
        self._transition_counts = np.asarray(
            [motion.shape[0] - 1 for motion in features], dtype=np.int64
        )
        if np.any(self._transition_counts < 1):
            raise ValueError("every walk motion must contain at least two frames")
        self._transition_ends = np.cumsum(self._transition_counts)
        self.num_transitions = int(self._transition_ends[-1])
        self._current = np.ascontiguousarray(
            np.concatenate([motion[:-1] for motion in features], axis=0)
        )
        self._next = np.ascontiguousarray(
            np.concatenate([motion[1:] for motion in features], axis=0)
        )
        self._motion_index = np.repeat(
            np.arange(len(features), dtype=np.int64), self._transition_counts
        )
        self._frame_index = np.concatenate(
            [np.arange(count, dtype=np.int64) for count in self._transition_counts]
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        asset_root: str | Path | None = None,
    ) -> "WalkMotionDataset":
        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require_exact_contract(manifest)
        root = Path(asset_root).resolve() if asset_root is not None else manifest_path.parent
        entries = manifest.get("motions")
        if not isinstance(entries, list) or not entries:
            raise ValueError("AMP walk manifest motions must be a non-empty list")

        names: list[str] = []
        features: list[np.ndarray] = []
        seen_files: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("AMP walk manifest motion entries must be objects")
            filename = entry.get("file")
            if not isinstance(filename, str) or not filename.startswith("walk_forward_"):
                raise ValueError(
                    f"motion file {filename!r} violates the forward-walk naming contract"
                )
            if Path(filename).name != filename or not filename.endswith(".npz"):
                raise ValueError(f"motion file must be a direct .npz filename, got {filename!r}")
            if filename in seen_files:
                raise ValueError(f"duplicate motion file in manifest: {filename}")
            seen_files.add(filename)
            path = root / filename
            if not path.is_file():
                raise ValueError(f"manifest motion file does not exist: {path}")
            actual_sha = _sha256(path)
            if actual_sha != entry.get("sha256"):
                raise ValueError(
                    f"SHA-256 mismatch for {filename}: expected {entry.get('sha256')}, "
                    f"got {actual_sha}"
                )
            motion_features = cls._load_motion(path, entry)
            names.append(path.stem)
            features.append(motion_features)
        return cls(names=tuple(names), features=tuple(features))

    @staticmethod
    def _load_motion(path: Path, entry: dict[str, Any]) -> np.ndarray:
        with np.load(path, allow_pickle=False) as source:
            missing = (set(_REQUIRED_ARRAYS) | {"fps"}) - set(source.files)
            if missing:
                raise ValueError(f"motion {path.name} is missing arrays: {sorted(missing)}")
            frames = int(source["body_pos_w"].shape[0])
            if frames != entry.get("frames"):
                raise ValueError(
                    f"frame count mismatch for {path.name}: expected {entry.get('frames')}, got {frames}"
                )
            fps_values = np.asarray(source["fps"]).reshape(-1)
            if fps_values.size != 1 or float(fps_values[0]) != float(entry.get("fps")):
                raise ValueError(f"FPS mismatch for {path.name}")
            arrays = {name: np.asarray(source[name]) for name in _REQUIRED_ARRAYS}
        return build_amp_observation(**arrays)

    def sample(self, num_samples: int, *, seed: int | None = None) -> AMPTransitionBatch:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        rng = np.random.default_rng(seed)
        flat_indices = rng.integers(0, self.num_transitions, size=num_samples, dtype=np.int64)
        return AMPTransitionBatch(
            current=np.ascontiguousarray(self._current[flat_indices]),
            next=np.ascontiguousarray(self._next[flat_indices]),
            motion_index=self._motion_index[flat_indices],
            frame_index=self._frame_index[flat_indices],
        )

    @property
    def current_transitions(self) -> np.ndarray:
        return self._current

    @property
    def next_transitions(self) -> np.ndarray:
        return self._next
