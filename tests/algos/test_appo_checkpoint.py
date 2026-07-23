from __future__ import annotations

from pathlib import Path

import pytest
import torch

import unilab.algos.torch.appo.checkpoint as checkpoint_module
from unilab.algos.torch.appo.checkpoint import load_appo_checkpoint, save_appo_checkpoint


def test_appo_checkpoint_is_cpu_owned_atomic_and_exactly_reloadable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "model_7.pt"
    observed: dict[str, object] = {}
    real_torch_save = torch.save

    def inspect_save(payload, path) -> None:
        resolved = Path(path)
        observed["path"] = resolved
        observed["actor_device"] = payload["actor"]["weight"].device.type
        observed["requires_grad"] = payload["actor"]["weight"].requires_grad
        real_torch_save(payload, resolved)

    monkeypatch.setattr(checkpoint_module.torch, "save", inspect_save)
    source = torch.tensor([1.0, 2.0], requires_grad=True)

    save_appo_checkpoint(
        destination,
        {
            "actor": {"weight": source},
            "optimizer": {"state": {0: {"step": torch.tensor(3)}}},
            "iteration": 7,
        },
    )

    saved_path = observed["path"]
    assert isinstance(saved_path, Path)
    assert saved_path.parent == destination.parent
    assert saved_path != destination
    assert saved_path.name.startswith(f".{destination.name}.tmp.")
    assert observed["actor_device"] == "cpu"
    assert observed["requires_grad"] is False
    assert destination.is_file()
    assert not list(tmp_path.glob(f".{destination.name}.tmp.*"))

    loaded = load_appo_checkpoint(destination, device="cpu")
    torch.testing.assert_close(loaded["actor"]["weight"], source.detach())
    assert loaded["optimizer"]["state"][0]["step"].item() == 3
    assert loaded["iteration"] == 7


def test_appo_checkpoint_failure_preserves_existing_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"existing-checkpoint")

    def fail_save(*_args, **_kwargs) -> None:
        raise RuntimeError("synthetic save failure")

    monkeypatch.setattr(checkpoint_module.torch, "save", fail_save)

    with pytest.raises(RuntimeError, match="synthetic save failure"):
        save_appo_checkpoint(destination, {"actor": {"weight": torch.ones(1)}})

    assert destination.read_bytes() == b"existing-checkpoint"
    assert not list(tmp_path.glob(f".{destination.name}.tmp.*"))
