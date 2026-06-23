from types import SimpleNamespace

import numpy as np

from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.common.commands import zero_small_xy_commands
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.g1.joystick import (
    G1WalkEnv,
    G1WalkRewardConfig,
    GaitConstraintConfig,
    compute_command_active_mask,
    compute_feet_phase_contact_targets,
    compute_feet_phase_height_targets,
    compute_gait_phase_contact_violation,
    compute_gait_phase_contrast_violation,
    compute_gait_phase_height_violation,
)


def _reward_config(**constraint_overrides) -> G1WalkRewardConfig:
    gait_constraint = GaitConstraintConfig(enabled=True, **constraint_overrides)
    return G1WalkRewardConfig(
        scales={},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_constraint=gait_constraint,
        pose_weights=[0.01] * 29,
    )


class _FakeBackend:
    def __init__(self, values):
        self._values = values

    def get_sensor_data(self, name):
        return self._values[name]


def _fake_env(reward_cfg: G1WalkRewardConfig, *, num_envs: int = 1) -> G1WalkEnv:
    env = object.__new__(G1WalkEnv)
    env._num_envs = num_envs
    env._reward_cfg = reward_cfg
    env._cfg = SimpleNamespace(
        ctrl_dt=0.02,
        control_config=SimpleNamespace(action_scale=1.0),
    )
    env._enable_reward_log = False
    env._gait_phase_delta = 0.1
    env.default_angles = np.zeros((29,), dtype=np.float32)
    env._backend = _FakeBackend(
        {
            "left_foot_pos": np.tile(
                np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32), (num_envs, 1)
            ),
            "right_foot_pos": np.tile(
                np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32), (num_envs, 1)
            ),
            "left_foot_contact_0": np.zeros((num_envs,), dtype=np.float32),
            "left_foot_contact_1": np.zeros((num_envs,), dtype=np.float32),
            "left_foot_contact_2": np.zeros((num_envs,), dtype=np.float32),
            "left_foot_contact_3": np.zeros((num_envs,), dtype=np.float32),
            "right_foot_contact_0": np.zeros((num_envs,), dtype=np.float32),
            "right_foot_contact_1": np.zeros((num_envs,), dtype=np.float32),
            "right_foot_contact_2": np.zeros((num_envs,), dtype=np.float32),
            "right_foot_contact_3": np.zeros((num_envs,), dtype=np.float32),
        }
    )
    return env


def _ctx(commands: np.ndarray, *, linvel_x: float) -> RewardContext:
    return RewardContext(
        info={
            "commands": commands,
            "gait_phase": np.asarray([[0.0, np.pi]], dtype=np.float32),
        },
        linvel=np.asarray([[linvel_x, 0.0, 0.0]], dtype=np.float32),
        gyro=np.zeros((1, 3), dtype=np.float32),
        dof_pos=np.zeros((1, 29), dtype=np.float32),
        num_envs=1,
    )


def test_command_active_mask_is_external_command_based() -> None:
    commands = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.04, 0.0, 0.0],
            [0.06, 0.0, 0.0],
            [0.0, 0.0, 0.06],
        ],
        dtype=np.float32,
    )

    mask = compute_command_active_mask(commands, xy_threshold=0.05, yaw_threshold=0.05)

    np.testing.assert_array_equal(mask, np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32))


def test_small_xy_threshold_keeps_low_speed_command_samples() -> None:
    commands = np.asarray(
        [[0.04, 0.0, 0.0], [0.1, 0.0, 0.0]],
        dtype=np.float32,
    )

    zero_small_xy_commands(commands, threshold=0.05)

    np.testing.assert_allclose(commands[0], np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(commands[1], np.asarray([0.1, 0.0, 0.0], dtype=np.float32))


def test_gait_phase_violation_zero_when_feet_match_generator() -> None:
    gait_phase = np.asarray([[0.0, np.pi], [np.pi, 2.0 * np.pi]], dtype=np.float32)
    swing_height = 0.09
    left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
    left_contact, right_contact = compute_feet_phase_contact_targets(gait_phase, swing_height)

    height = compute_gait_phase_height_violation(
        left_target, right_target, gait_phase, swing_height
    )
    contrast = compute_gait_phase_contrast_violation(
        left_target, right_target, gait_phase, swing_height
    )
    contact = compute_gait_phase_contact_violation(
        left_contact, right_contact, gait_phase, swing_height
    )

    np.testing.assert_allclose(height, 0.0)
    np.testing.assert_allclose(contrast, 0.0)
    np.testing.assert_allclose(contact, 0.0)


def test_reward_config_converts_gait_constraint_dict() -> None:
    cfg = G1WalkRewardConfig(
        scales={},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_constraint={"enabled": True, "penalty_scale": 0.5},
        pose_weights=[0.01] * 29,
    )

    assert isinstance(cfg.gait_constraint, GaitConstraintConfig)
    assert cfg.gait_constraint.enabled is True
    assert cfg.gait_constraint.penalty_scale == 0.5


def test_zero_command_drift_does_not_open_gait_constraint_gate() -> None:
    reward_cfg = _reward_config(epsilon=0.0, penalty_scale=1.0)
    env = _fake_env(reward_cfg)
    reward = np.asarray([1.0], dtype=np.float32)
    ctx = _ctx(np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32), linvel_x=0.2)

    components = env._compute_gait_constraint_components(ctx, reward_cfg.gait_constraint)
    bridged_reward = env._apply_gait_constraint_bridge(ctx, reward)

    assert components["total"][0] > 0.0
    assert components["command_active"][0] == 0.0
    assert components["gate"][0] == 0.0
    np.testing.assert_allclose(bridged_reward, reward)


def test_stand_phase_replaces_observation_phase_for_inactive_command() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[0.0, np.pi],
    )
    env = _fake_env(reward_cfg, num_envs=2)
    info = {
        "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
        "gait_phase": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }

    phase = env._gait_phase_for_observation(info)

    np.testing.assert_allclose(phase[0], np.asarray([0.0, np.pi], dtype=np.float32))
    np.testing.assert_allclose(phase[1], np.asarray([3.0, 4.0], dtype=np.float32))


def test_apply_action_freezes_stand_phase_and_advances_active_phase() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[0.0, np.pi],
    )
    env = _fake_env(reward_cfg, num_envs=2)
    state = NpEnvState(
        obs={},
        reward=np.zeros((2,), dtype=np.float32),
        terminated=np.zeros((2,), dtype=bool),
        truncated=np.zeros((2,), dtype=bool),
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "gait_phase": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        },
    )

    ctrl = env.apply_action(np.zeros((2, 29), dtype=np.float32), state)

    np.testing.assert_allclose(ctrl, np.zeros((2, 29), dtype=np.float32))
    np.testing.assert_allclose(state.info["gait_phase"][0], np.asarray([0.0, np.pi]))
    np.testing.assert_allclose(state.info["gait_phase"][1], np.asarray([3.1, 4.1]))


def test_stand_rewards_only_apply_when_command_inactive() -> None:
    reward_cfg = _reward_config()
    env = _fake_env(reward_cfg, num_envs=2)
    ctx = RewardContext(
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "current_actions": np.ones((2, 29), dtype=np.float32),
        },
        linvel=np.zeros((2, 3), dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.ones((2, 29), dtype=np.float32),
        dof_vel=np.ones((2, 29), dtype=np.float32),
        num_envs=2,
    )

    np.testing.assert_allclose(env._reward_stand_still(ctx), np.asarray([29.0, 0.0]))
    np.testing.assert_allclose(env._reward_stand_action_l2(ctx), np.asarray([29.0, 0.0]))
    np.testing.assert_allclose(env._reward_stand_dof_vel_l2(ctx), np.asarray([29.0, 0.0]))


def test_nonzero_command_applies_gait_constraint_cost() -> None:
    reward_cfg = _reward_config(epsilon=0.0, penalty_scale=1.0)
    env = _fake_env(reward_cfg)
    reward = np.asarray([1.0], dtype=np.float32)
    ctx = _ctx(np.asarray([[0.2, 0.0, 0.0]], dtype=np.float32), linvel_x=0.0)

    bridged_reward = env._apply_gait_constraint_bridge(ctx, reward)

    assert bridged_reward[0] < reward[0]
