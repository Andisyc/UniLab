from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from unilab.algos.torch.amp.spec import (
    AMP_BODY_NAMES,
    AMP_OBSERVATION_DIM,
    build_amp_observation_from_selected,
)
from unilab.algos.torch.amp.transition import resolve_amp_transition_next
from unilab.envs.locomotion.g1.amp_walk import G1AMPWalkCfg, G1AMPWalkEnv
from unilab.envs.locomotion.g1.joystick import GaitConstraintConfig


class _NoNoise:
    level = 0.0
    scale_gyro = 0.0
    scale_gravity = 0.0
    scale_joint_angle = 0.0
    scale_joint_vel = 0.0


class _BodyStateBackend:
    def __init__(self) -> None:
        rng = np.random.default_rng(17)
        self.pos = rng.normal(size=(2, 14, 3)).astype(np.float32)
        raw_quat = rng.normal(size=(2, 14, 4)).astype(np.float32)
        self.quat = raw_quat / np.linalg.norm(raw_quat, axis=-1, keepdims=True)
        self.lin = rng.normal(size=(2, 14, 3)).astype(np.float32)
        self.ang = rng.normal(size=(2, 14, 3)).astype(np.float32)
        self.requested_ids: np.ndarray | None = None

    def get_body_state_w(self, body_ids: np.ndarray):
        self.requested_ids = body_ids.copy()
        return self.pos, self.quat, self.lin, self.ang


def _unit_env() -> G1AMPWalkEnv:
    env = cast(Any, object.__new__(G1AMPWalkEnv))
    env._num_envs = 2
    env._num_action = 29
    env.default_angles = np.zeros((1, 29), dtype=np.float32)
    env._cfg = SimpleNamespace(
        noise_config=_NoNoise(),
        curriculum=SimpleNamespace(enabled=False),
        mode_observation=False,
        commands=SimpleNamespace(observe_height_command=False),
    )
    env._reward_cfg = SimpleNamespace(scales={}, gait_constraint=GaitConstraintConfig())
    env._obs_noise = lambda data, scale: data
    env._amp_body_ids_with_anchor = np.arange(14, dtype=np.int32)
    env._backend = _BodyStateBackend()
    return cast(G1AMPWalkEnv, env)


def test_amp_walk_cfg_is_fixed_nonzero_forward_without_standing() -> None:
    cfg = G1AMPWalkCfg()

    assert cfg.commands.vel_limit == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert cfg.commands.rel_standing_envs == 0.0
    assert cfg.commands.rel_transition_envs == 0.0
    assert cfg.commands.heading_command is False
    assert cfg.mode_observation is False
    assert cfg.commands.observe_height_command is False
    assert cfg.add_body_sensors is True


def test_amp_walk_rejects_default_pose_reward_authority() -> None:
    cfg = G1AMPWalkCfg()
    cfg.reward_config = SimpleNamespace(scales={"pose": -0.1})

    with pytest.raises(ValueError, match="default-pose reward"):
        G1AMPWalkEnv._validate_phase1_cfg(cfg)


def test_amp_walk_observation_contract_has_no_gait_phase_authority() -> None:
    env = _unit_env()
    common = {
        "commands": np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
        "current_actions": np.zeros((2, 29), dtype=np.float32),
    }
    kinematics = {
        "linvel": np.zeros((2, 3), dtype=np.float32),
        "gyro": np.zeros((2, 3), dtype=np.float32),
        "gravity": np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (2, 1)),
        "dof_pos": np.zeros((2, 29), dtype=np.float32),
        "dof_vel": np.zeros((2, 29), dtype=np.float32),
    }

    first = env._compute_obs(
        {**common, "gait_phase": np.zeros((2, 2), dtype=np.float32)}, **kinematics
    )
    second = env._compute_obs(
        {**common, "gait_phase": np.full((2, 2), 9.0, dtype=np.float32)}, **kinematics
    )

    assert env.obs_groups_spec == {"obs": 96, "critic": 99, "amp": 195}
    assert set(first) == {"obs", "critic", "amp"}
    for key, width in env.obs_groups_spec.items():
        assert first[key].shape == (2, width)
        np.testing.assert_array_equal(first[key], second[key])


def test_amp_walk_reads_one_cached_public_backend_body_state_contract() -> None:
    env = _unit_env()
    backend = cast(_BodyStateBackend, env._backend)

    actual = env._compute_amp_observation()
    expected = build_amp_observation_from_selected(
        body_pos_w=backend.pos[:, : len(AMP_BODY_NAMES)],
        body_quat_w=backend.quat[:, : len(AMP_BODY_NAMES)],
        body_lin_vel_w=backend.lin[:, : len(AMP_BODY_NAMES)],
        body_ang_vel_w=backend.ang[:, : len(AMP_BODY_NAMES)],
        anchor_pos_w=backend.pos[:, -1],
        anchor_quat_w=backend.quat[:, -1],
    )

    assert backend.requested_ids is not None
    np.testing.assert_array_equal(backend.requested_ids, np.arange(14, dtype=np.int32))
    assert actual.shape == (2, AMP_OBSERVATION_DIM)
    np.testing.assert_allclose(actual, expected)


def test_amp_walk_partial_reset_reads_only_requested_body_state_rows() -> None:
    env = _unit_env()

    selected = env._compute_amp_observation(np.array([1], dtype=np.int32))
    full = env._compute_amp_observation()

    assert selected.shape == (1, 195)
    np.testing.assert_array_equal(selected[0], full[1])


def test_amp_terminal_transition_uses_final_observation_for_only_done_rows() -> None:
    actor_next = np.full((3, 195), 10.0, dtype=np.float32)
    final_amp = np.stack(
        [
            np.full(195, 1.0, dtype=np.float32),
            np.full(195, 2.0, dtype=np.float32),
            np.full(195, 3.0, dtype=np.float32),
        ]
    )
    done = np.array([False, True, True])

    transition_next, terminal_mask = resolve_amp_transition_next(
        actor_next,
        done=done,
        final_observation={"amp": final_amp},
    )

    np.testing.assert_array_equal(terminal_mask, done)
    np.testing.assert_array_equal(transition_next[0], actor_next[0])
    np.testing.assert_array_equal(transition_next[1], final_amp[1])
    np.testing.assert_array_equal(transition_next[2], final_amp[2])
    np.testing.assert_array_equal(actor_next, 10.0)


def test_amp_terminal_transition_fails_closed_without_terminal_amp() -> None:
    with pytest.raises(ValueError, match=r"final_observation\['amp'\]"):
        resolve_amp_transition_next(
            np.zeros((2, 195), dtype=np.float32),
            done=np.array([True, False]),
            final_observation={"obs": np.zeros((2, 96), dtype=np.float32)},
        )


@pytest.mark.slow
def test_g1_amp_walk_live_reset_and_timeout_final_observation() -> None:
    from unilab.base import registry
    from unilab.base.registry import ensure_registries
    from unilab.envs.locomotion.g1.joystick import G1WalkRewardConfig

    ensure_registries()
    reward = G1WalkRewardConfig(
        scales={"tracking_lin_vel": 2.0, "alive": 1.0},
        tracking_sigma=0.25,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        close_feet_threshold=0.15,
        pose_weights=[0.01] * 29,
    )
    env = registry.make(
        "G1AMPWalk",
        num_envs=2,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": reward, "max_episode_seconds": 0.02},
    )
    try:
        obs, info = env.reset(np.arange(2, dtype=np.int32))
        assert env.obs_groups_spec == {"obs": 96, "critic": 99, "amp": 195}
        assert {key: value.shape for key, value in obs.items()} == {
            "obs": (2, 96),
            "critic": (2, 99),
            "amp": (2, 195),
        }
        np.testing.assert_array_equal(info["commands"], [[1.0, 0.0, 0.0]] * 2)

        state = env.step(np.zeros((2, 29), dtype=np.float32))
        assert np.all(state.truncated)
        assert state.final_observation is not None
        assert set(state.final_observation) == {"obs", "critic", "amp"}
        assert state.final_observation["amp"].shape == (2, 195)
        assert np.isfinite(state.final_observation["amp"]).all()
        np.testing.assert_array_equal(state.info["_final_observation"], [True, True])
    finally:
        env.close()
