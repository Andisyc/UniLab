"""CPU-owned atomic checkpoint IO for APPO runtimes."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch


def cpu_owned_checkpoint_value(value: Any) -> Any:
    """Detach every tensor in a checkpoint tree and move it to CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: cpu_owned_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_owned_checkpoint_value(item) for item in value)
    if isinstance(value, list):
        return [cpu_owned_checkpoint_value(item) for item in value]
    return value


def save_appo_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Serialize one CPU-owned payload and atomically replace the destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(cpu_owned_checkpoint_value(payload), temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    if not destination.is_file():
        raise FileNotFoundError(f"APPO checkpoint was not saved: {destination}")


def load_appo_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load an APPO checkpoint once onto the requested owner device."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"APPO checkpoint must contain a mapping: {path}")
    return cast(dict[str, Any], checkpoint)
