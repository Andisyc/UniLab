from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.algos.torch.amp.runner import AMPAPPORunner
from unilab.algos.torch.amp.runtime import resolve_amp_appo_runtime
from unilab.algos.torch.amp.worker import write_amp_rollout_payload
from unilab.algos.torch.appo.runtime import resolve_appo_runtime

_ROOT = Path(__file__).resolve().parents[2]


def test_amp_runtime_resolver_uses_amp_runner_and_generic_actor_only_play() -> None:
    def play_fn(*args, **kwargs):
        del args, kwargs
        return None

    runtime = resolve_appo_runtime(
        {
            "runtime_impl": "amp_appo",
            "runtime_resolver": "unilab.algos.torch.amp.runtime:resolve_amp_appo_runtime",
        },
        default_play_fn=play_fn,
    )

    assert runtime.runner_cls is AMPAPPORunner
    assert runtime.play_fn is play_fn


def test_amp_runtime_resolver_rejects_missing_owner_marker() -> None:
    with pytest.raises(ValueError, match="runtime_impl='amp_appo'"):
        resolve_amp_appo_runtime({}, default_play_fn=lambda: None)


def test_amp_runtime_resolution_does_not_import_distillation_modules() -> None:
    before = {name for name in sys.modules if name.startswith("unilab.algos.torch.distill")}

    resolve_amp_appo_runtime(
        {"runtime_impl": "amp_appo"}, default_play_fn=lambda *args, **kwargs: None
    )

    after = {name for name in sys.modules if name.startswith("unilab.algos.torch.distill")}
    assert after == before


def test_amp_runner_declares_only_two_typed_transition_fields() -> None:
    runner = AMPAPPORunner.__new__(AMPAPPORunner)
    runner.num_envs = 8
    runner.steps_per_env = 4

    specs = runner._extra_rollout_field_specs()

    assert set(specs) == {"amp_state", "amp_next_state"}
    assert specs["amp_state"].shape == (8, 4, 195)
    assert specs["amp_next_state"].shape == (8, 4, 195)
    assert specs["amp_state"].dtype == "float32"
    assert runner._collector_runtime_kwargs() == {
        "rollout_payload_writer": "unilab.algos.torch.amp.worker:write_amp_rollout_payload"
    }


def test_amp_runner_resume_uses_full_learner_state_loader() -> None:
    loaded: list[dict] = []
    learner = SimpleNamespace(load_state_dict=lambda checkpoint: loaded.append(checkpoint))
    checkpoint = {"actor": {}, "amp": {"discriminator_version": 7}}
    runner = AMPAPPORunner.__new__(AMPAPPORunner)

    runner._restore_learner_checkpoint(learner, checkpoint)

    assert loaded == [checkpoint]


def test_amp_payload_writer_preserves_partial_terminal_identity() -> None:
    current = np.full((3, 195), 1.0, dtype=np.float32)
    actor_next = np.full((3, 195), 2.0, dtype=np.float32)
    terminal = np.stack([np.full(195, value, dtype=np.float32) for value in (3, 4, 5)])
    write_buffer = {
        "amp_state": np.zeros((3, 2, 195), dtype=np.float32),
        "amp_next_state": np.zeros((3, 2, 195), dtype=np.float32),
    }
    state = SimpleNamespace(
        obs={"amp": actor_next},
        terminated=np.array([False, True, False]),
        truncated=np.array([False, False, True]),
        final_observation={"amp": terminal},
        info={},
    )

    write_amp_rollout_payload(
        write_buffer=write_buffer,
        step=1,
        current_observation={"amp": current},
        state=state,
    )

    np.testing.assert_array_equal(write_buffer["amp_state"][:, 1], current)
    np.testing.assert_array_equal(write_buffer["amp_next_state"][0, 1], actor_next[0])
    np.testing.assert_array_equal(write_buffer["amp_next_state"][1, 1], terminal[1])
    np.testing.assert_array_equal(write_buffer["amp_next_state"][2, 1], terminal[2])


def test_g1_amp_walk_owner_yaml_composes_without_gait_or_distillation() -> None:
    with initialize_config_dir(config_dir=str(_ROOT / "conf" / "appo"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=g1_amp_walk/mujoco"])

    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    assert resolved["training"]["task_name"] == "G1AMPWalk"
    assert resolved["algo"]["runtime_impl"] == "amp_appo"
    assert resolved["algo"]["runtime_resolver"].endswith(":resolve_amp_appo_runtime")
    assert resolved["env"]["commands"]["vel_limit"] == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    assert not any("feet_phase" in key for key in resolved["reward"]["scales"])
    assert "distill" not in str(resolved).lower()
