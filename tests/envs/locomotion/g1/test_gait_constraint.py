from types import SimpleNamespace

import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf

from unilab.base.np_env import NpEnvState
from unilab.envs.locomotion.common.commands import zero_small_xy_commands
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.g1.joystick import (
    G1WalkDomainRandomizationProvider,
    G1WalkEnv,
    G1WalkRewardConfig,
    GaitConstraintConfig,
    RewardModeConfig,
    compute_command_active_mask,
    compute_external_command_mask,
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

    def get_base_pos(self):
        return self._values["base_pos"]


def _fake_env(reward_cfg: G1WalkRewardConfig, *, num_envs: int = 1) -> G1WalkEnv:
    env = object.__new__(G1WalkEnv)
    env._num_envs = num_envs
    env._num_action = 29
    env._reward_cfg = reward_cfg
    env._cfg = SimpleNamespace(
        ctrl_dt=0.02,
        control_config=SimpleNamespace(action_scale=1.0),
        mode_observation=False,
        noise_config=SimpleNamespace(
            level=0.0,
            scale_gyro=0.0,
            scale_gravity=0.0,
            scale_joint_angle=0.0,
            scale_joint_vel=0.0,
        ),
        stand_action_authority=False,
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
            "left_foot_quat": np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (num_envs, 1)
            ),
            "right_foot_quat": np.tile(
                np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (num_envs, 1)
            ),
            "left_foot_contact_0": np.zeros((num_envs,), dtype=np.float32),
            "left_foot_contact_1": np.zeros((num_envs,), dtype=np.float32),
            "left_foot_contact_2": np.zeros((num_envs,), dtype=np.float32),
            "left_foot_contact_3": np.zeros((num_envs,), dtype=np.float32),
            "right_foot_contact_0": np.zeros((num_envs,), dtype=np.float32),
            "right_foot_contact_1": np.zeros((num_envs,), dtype=np.float32),
            "right_foot_contact_2": np.zeros((num_envs,), dtype=np.float32),
            "right_foot_contact_3": np.zeros((num_envs,), dtype=np.float32),
            "base_pos": np.tile(
                np.asarray([[0.0, 0.0, 0.754]], dtype=np.float32), (num_envs, 1)
            ),
        }
    )
    env._pose_weights = np.ones((29,), dtype=np.float32)
    env._upper_body_pose_weights = np.ones((29,), dtype=np.float32)
    env._init_reward_functions()
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


def test_external_command_mask_is_discrete_nonzero_signal() -> None:
    commands = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.0, -0.01],
        ],
        dtype=np.float32,
    )

    mask = compute_external_command_mask(commands)

    np.testing.assert_array_equal(mask, np.asarray([0.0, 1.0, 1.0], dtype=np.float32))


def test_g1_reset_info_writes_gait_enabled_from_sampled_command() -> None:
    provider = G1WalkDomainRandomizationProvider()
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            commands=SimpleNamespace(heading_command=False),
            gait_phase_init_mode="offset_phase",
        )
    )
    commands = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.0, -0.01]], dtype=np.float32
    )

    updates = provider._build_extra_info_updates_for_commands(env, 3, commands)

    assert updates["gait_phase"].shape == (3, 2)
    np.testing.assert_array_equal(
        updates["gait_enabled"], np.asarray([0.0, 1.0, 1.0], dtype=np.float32)
    )


def test_common_small_xy_threshold_zeroes_low_speed_xy_commands() -> None:
    commands = np.asarray(
        [[0.04, 0.0, 0.0], [0.1, 0.0, 0.0]],
        dtype=np.float32,
    )

    zero_small_xy_commands(commands, threshold=0.05)

    np.testing.assert_allclose(commands[0], np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(commands[1], np.asarray([0.1, 0.0, 0.0], dtype=np.float32))


def test_g1_low_speed_nonzero_command_stays_walk_mode() -> None:
    provider = G1WalkDomainRandomizationProvider()
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            commands=SimpleNamespace(
                vel_limit=[[0.03, 0.0, 0.0], [0.03, 0.0, 0.0]],
                small_xy_threshold=0.0,
                rel_standing_envs=0.0,
                heading_command=False,
            ),
            gait_phase_init_mode="offset_phase",
        )
    )

    commands = provider._sample_commands(env, 4)
    updates = provider._build_extra_info_updates_for_commands(env, 4, commands)

    np.testing.assert_allclose(commands, np.asarray([[0.03, 0.0, 0.0]] * 4, dtype=np.float32))
    np.testing.assert_array_equal(updates["gait_enabled"], np.ones((4,), dtype=np.float32))


def test_g1_transition_command_distribution_stays_walk_mode() -> None:
    provider = G1WalkDomainRandomizationProvider()
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            commands=SimpleNamespace(
                vel_limit=[[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]],
                transition_vel_limit=[[0.12, 0.0, 0.0], [0.12, 0.0, 0.0]],
                small_xy_threshold=0.0,
                rel_standing_envs=0.0,
                rel_transition_envs=1.0,
                heading_command=False,
            ),
            gait_phase_init_mode="offset_phase",
        )
    )

    commands = provider._sample_commands(env, 4)
    updates = provider._build_extra_info_updates_for_commands(env, 4, commands)

    np.testing.assert_allclose(commands, np.asarray([[0.12, 0.0, 0.0]] * 4, dtype=np.float32))
    np.testing.assert_array_equal(updates["gait_enabled"], np.ones((4,), dtype=np.float32))


def test_g1_standing_reset_info_uses_stand_phase() -> None:
    provider = G1WalkDomainRandomizationProvider()
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            commands=SimpleNamespace(heading_command=False),
            gait_phase_init_mode="offset_phase",
            reward_config=SimpleNamespace(
                gait_constraint=GaitConstraintConfig(
                    enabled=True,
                    freeze_phase_in_stand_mode=True,
                    stand_phase=[np.pi, np.pi],
                )
            ),
        )
    )
    commands = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)

    updates = provider._build_extra_info_updates_for_commands(env, 2, commands)

    np.testing.assert_allclose(updates["gait_phase"][0], np.asarray([np.pi, np.pi]))
    assert not np.allclose(updates["gait_phase"][1], np.asarray([np.pi, np.pi]))
    np.testing.assert_array_equal(updates["gait_enabled"], np.asarray([0.0, 1.0], dtype=np.float32))


def test_g1_standing_reset_zeros_base_qvel_without_touching_walk_samples() -> None:
    class _Spawn:
        def apply_spawn(self, env_ids, qpos_xyz, *, yaw=None):
            return qpos_xyz

        def record_episode_start(self, env_ids, qpos_xyz) -> None:
            pass

    provider = G1WalkDomainRandomizationProvider()
    commands = np.asarray(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    provider._sample_commands = lambda env, num_reset: commands.copy()  # type: ignore[method-assign]
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            commands=SimpleNamespace(heading_command=False),
            gait_phase_init_mode="offset_phase",
            reset_base_qvel_limit=0.5,
            standing_reset_base_qvel_limit=0.0,
            domain_rand=None,
        ),
        _init_qpos=np.zeros((36,), dtype=np.float32),
        _init_qvel=np.zeros((35,), dtype=np.float32),
        _spawn=_Spawn(),
        _num_action=29,
    )

    plan = provider.build_reset_plan(env, np.asarray([0, 1, 2], dtype=np.int32))

    np.testing.assert_allclose(plan.qvel[[0, 2], 0:6], 0.0)
    assert not np.allclose(plan.qvel[1, 0:6], 0.0)
    np.testing.assert_array_equal(
        plan.info_updates["gait_enabled"], np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    )


def test_standing_reset_observation_is_idle_consistent() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[np.pi, np.pi],
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._cfg.mode_observation = True
    info = {
        "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
        "current_actions": np.zeros((2, 29), dtype=np.float32),
        "last_actions": np.zeros((2, 29), dtype=np.float32),
        "gait_enabled": np.asarray([0.0, 1.0], dtype=np.float32),
        "gait_phase": np.asarray([[np.pi, np.pi], [1.0, 2.0]], dtype=np.float32),
    }

    obs = env._compute_obs(
        info,
        linvel=np.zeros((2, 3), dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        gravity=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        dof_pos=np.zeros((2, 29), dtype=np.float32),
        dof_vel=np.zeros((2, 29), dtype=np.float32),
    )

    stand_obs = obs["obs"][0]
    np.testing.assert_allclose(stand_obs[64:93], 0.0)
    np.testing.assert_allclose(stand_obs[93:96], 0.0)
    np.testing.assert_allclose(stand_obs[96:98], np.asarray([np.pi, np.pi]))
    assert stand_obs[98] == 0.0


def test_mode_observation_appends_external_gait_signal_to_obs() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[np.pi, np.pi],
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._cfg.mode_observation = True
    info = {
        "commands": np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32),
        "current_actions": np.zeros((2, 29), dtype=np.float32),
        "gait_phase": np.asarray([[0.0, np.pi], [1.0, 2.0]], dtype=np.float32),
    }

    obs = env._compute_obs(
        info,
        linvel=np.zeros((2, 3), dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        gravity=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        dof_pos=np.zeros((2, 29), dtype=np.float32),
        dof_vel=np.zeros((2, 29), dtype=np.float32),
    )

    assert env.obs_groups_spec == {"obs": 99, "critic": 102}
    assert obs["obs"].shape == (2, 99)
    assert obs["critic"].shape == (2, 102)
    np.testing.assert_array_equal(obs["obs"][:, -1], np.asarray([0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(
        obs["critic"][:, -4], np.asarray([0.0, 1.0], dtype=np.float32)
    )


def test_mode_observation_disabled_preserves_legacy_obs_dims() -> None:
    reward_cfg = _reward_config()
    env = _fake_env(reward_cfg, num_envs=1)

    assert env.obs_groups_spec == {"obs": 98, "critic": 101}
    assert ("mode", 1) not in env._actor_symmetry_obs_layout()


def test_mode_observation_updates_symmetry_layout() -> None:
    reward_cfg = _reward_config()
    env = _fake_env(reward_cfg, num_envs=1)
    env._cfg.mode_observation = True

    actor_layout = env._actor_symmetry_obs_layout()

    assert actor_layout[-1] == ("mode", 1)
    assert sum(dim for _, dim in actor_layout) == 99


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


def test_stand_phase_is_double_stance() -> None:
    cfg = GaitConstraintConfig()
    gait_phase = np.asarray([cfg.stand_phase], dtype=np.float32)
    swing_height = 0.09

    left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
    left_contact, right_contact = compute_feet_phase_contact_targets(gait_phase, swing_height)

    np.testing.assert_allclose(left_target, 0.0, atol=1.0e-6)
    np.testing.assert_allclose(right_target, 0.0, atol=1.0e-6)
    np.testing.assert_array_equal(left_contact, np.asarray([True]))
    np.testing.assert_array_equal(right_contact, np.asarray([True]))


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


def test_reward_config_converts_reward_mode_dict() -> None:
    cfg = G1WalkRewardConfig(
        scales={},
        tracking_sigma=0.25,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        mode={
            "enabled": True,
            "balance_common_terms": ["base_height"],
            "stand_terms": ["stand_lin_vel_xy_l2"],
            "walk_terms": ["tracking_lin_vel"],
        },
        pose_weights=[0.01] * 29,
    )

    assert isinstance(cfg.mode, RewardModeConfig)
    assert cfg.mode.enabled is True
    assert cfg.mode.balance_common_terms == ["base_height"]
    assert cfg.mode.stand_terms == ["stand_lin_vel_xy_l2"]
    assert cfg.mode.walk_terms == ["tracking_lin_vel"]


def test_current_command_syncs_stale_gait_enabled_mode() -> None:
    reward_cfg = _reward_config()
    env = _fake_env(reward_cfg, num_envs=2)
    info = {
        "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
        "gait_enabled": np.asarray([1.0, 0.0], dtype=np.float32),
    }

    gait_enabled = env._gait_enabled_mask(info)
    stand_mask = env._stand_mode_mask(
        RewardContext(
            info=info,
            linvel=np.zeros((2, 3), dtype=np.float32),
            gyro=np.zeros((2, 3), dtype=np.float32),
            dof_pos=np.zeros((2, 29), dtype=np.float32),
            num_envs=2,
        )
    )

    np.testing.assert_array_equal(gait_enabled, np.asarray([0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(info["gait_enabled"], np.asarray([0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(stand_mask, np.asarray([1.0, 0.0], dtype=np.float32))


def test_reset_observation_accepts_partial_gait_enabled_batch() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[np.pi, np.pi],
    )
    env = _fake_env(reward_cfg, num_envs=2048)
    info = {
        "commands": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.0, -0.01],
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "gait_enabled": np.asarray([0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "gait_phase": np.asarray(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.5, 0.6], [1.5, 1.6]],
            dtype=np.float32,
        ),
    }

    phase = env._gait_phase_for_observation(info)

    assert phase.shape == (5, 2)
    np.testing.assert_allclose(phase[0], np.asarray([np.pi, np.pi], dtype=np.float32))
    np.testing.assert_allclose(phase[1], np.asarray([3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(phase[3], np.asarray([np.pi, np.pi], dtype=np.float32))


def test_reward_mode_dispatch_separates_stand_and_walk_terms() -> None:
    reward_cfg = G1WalkRewardConfig(
        scales={
            "alive": 2.0,
            "stand_lin_vel_xy_l2": -30.0,
            "tracking_lin_vel": 2.0,
        },
        tracking_sigma=0.12,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_constraint=GaitConstraintConfig(enabled=False),
        mode=RewardModeConfig(
            enabled=True,
            balance_common_terms=["alive"],
            stand_terms=["stand_lin_vel_xy_l2"],
            walk_terms=["tracking_lin_vel"],
        ),
        pose_weights=[0.01] * 29,
    )
    env = _fake_env(reward_cfg, num_envs=2)
    ctx = RewardContext(
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "gait_enabled": np.asarray([1.0, 0.0], dtype=np.float32),
        },
        linvel=np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.zeros((2, 29), dtype=np.float32),
        dof_vel=np.zeros((2, 29), dtype=np.float32),
        num_envs=2,
        default_angles=np.zeros((29,), dtype=np.float32),
        tracking_sigma=reward_cfg.tracking_sigma,
        base_height_target=reward_cfg.base_height_target,
        base_height=np.full((2,), reward_cfg.base_height_target, dtype=np.float32),
        gravity=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        pose_weights=np.ones((29,), dtype=np.float32),
    )

    reward = env._compute_mode_reward(ctx, reward_cfg)

    np.testing.assert_array_equal(ctx.info["gait_enabled"], np.asarray([0.0, 1.0], dtype=np.float32))
    assert reward[0] > 0.0
    assert reward[1] > 0.0
    np.testing.assert_allclose(
        reward[0],
        (2.0 - 30.0 * 0.01) * env._cfg.ctrl_dt,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        reward[1],
        (2.0 + 2.0) * env._cfg.ctrl_dt,
        rtol=1.0e-6,
    )


def test_walking_reward_is_hard_gated_out_of_standing_samples() -> None:
    reward_cfg = G1WalkRewardConfig(
        scales={"tracking_lin_vel": 2.0},
        tracking_sigma=0.12,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_constraint=GaitConstraintConfig(enabled=False),
        mode=RewardModeConfig(
            enabled=True,
            stand_terms=[],
            walk_terms=["tracking_lin_vel"],
        ),
        pose_weights=[0.01] * 29,
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._enable_reward_log = True
    ctx = RewardContext(
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "steps": np.zeros((2,), dtype=np.uint32),
        },
        linvel=np.asarray([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.zeros((2, 29), dtype=np.float32),
        dof_vel=np.zeros((2, 29), dtype=np.float32),
        num_envs=2,
        default_angles=np.zeros((29,), dtype=np.float32),
        tracking_sigma=reward_cfg.tracking_sigma,
        base_height_target=reward_cfg.base_height_target,
        base_height=np.full((2,), reward_cfg.base_height_target, dtype=np.float32),
        gravity=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        pose_weights=np.ones((29,), dtype=np.float32),
    )

    reward = env._compute_mode_reward(ctx, reward_cfg)

    assert reward[0] == 0.0
    assert reward[1] > 0.0
    assert ctx.info["log"]["reward/mode_stand_frac"] == 0.5
    assert ctx.info["log"]["reward/mode_walk_frac"] == 0.5
    assert ctx.info["log"]["reward/stand_total"] == 0.0
    assert ctx.info["log"]["reward/walk_total"] > 0.0


def test_standing_reward_is_hard_gated_out_of_walking_samples() -> None:
    reward_cfg = G1WalkRewardConfig(
        scales={"stand_lin_vel_xy_l2": -30.0},
        tracking_sigma=0.12,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_constraint=GaitConstraintConfig(enabled=False),
        mode=RewardModeConfig(
            enabled=True,
            stand_terms=["stand_lin_vel_xy_l2"],
            walk_terms=[],
        ),
        pose_weights=[0.01] * 29,
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._enable_reward_log = True
    ctx = RewardContext(
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "steps": np.zeros((2,), dtype=np.uint32),
        },
        linvel=np.asarray([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.zeros((2, 29), dtype=np.float32),
        dof_vel=np.zeros((2, 29), dtype=np.float32),
        num_envs=2,
        default_angles=np.zeros((29,), dtype=np.float32),
        tracking_sigma=reward_cfg.tracking_sigma,
        base_height_target=reward_cfg.base_height_target,
        base_height=np.full((2,), reward_cfg.base_height_target, dtype=np.float32),
        gravity=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        pose_weights=np.ones((29,), dtype=np.float32),
    )

    reward = env._compute_mode_reward(ctx, reward_cfg)

    assert reward[0] < 0.0
    assert reward[1] == 0.0
    assert ctx.info["log"]["reward/stand_total"] < 0.0
    assert ctx.info["log"]["reward/walk_total"] == 0.0


def test_common_base_height_reward_applies_to_stand_and_walk() -> None:
    reward_cfg = G1WalkRewardConfig(
        scales={"base_height": -80.0},
        tracking_sigma=0.12,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_constraint=GaitConstraintConfig(enabled=False),
        mode=RewardModeConfig(
            enabled=True,
            balance_common_terms=["base_height"],
            stand_terms=[],
            walk_terms=[],
        ),
        pose_weights=[0.01] * 29,
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._enable_reward_log = True
    ctx = RewardContext(
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "steps": np.zeros((2,), dtype=np.uint32),
        },
        linvel=np.zeros((2, 3), dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.zeros((2, 29), dtype=np.float32),
        dof_vel=np.zeros((2, 29), dtype=np.float32),
        num_envs=2,
        default_angles=np.zeros((29,), dtype=np.float32),
        tracking_sigma=reward_cfg.tracking_sigma,
        base_height_target=reward_cfg.base_height_target,
        base_height=np.asarray([0.3, 0.3], dtype=np.float32),
        gravity=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        pose_weights=np.ones((29,), dtype=np.float32),
    )

    reward = env._compute_mode_reward(ctx, reward_cfg)

    assert reward[0] < 0.0
    assert reward[1] < 0.0
    np.testing.assert_allclose(reward[0], reward[1])
    assert ctx.info["log"]["reward/base_height"] < 0.0
    assert ctx.info["log"]["reward/stand_total"] < 0.0
    assert ctx.info["log"]["reward/walk_total"] < 0.0


def test_active_g1_common_base_height_reward_numeric_effect() -> None:
    with initialize(config_path="../../../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_flat/mujoco"])

    assert cfg.reward.scales.base_height == -80.0
    assert "base_height" in cfg.reward.mode.balance_common_terms
    assert "base_height" not in cfg.reward.mode.stand_terms
    assert "base_height" not in cfg.reward.mode.walk_terms

    reward_cfg = G1WalkRewardConfig(
        scales={"base_height": float(cfg.reward.scales.base_height)},
        tracking_sigma=float(cfg.reward.tracking_sigma),
        gait_frequency=float(cfg.reward.gait_frequency),
        feet_phase_swing_height=float(cfg.reward.feet_phase_swing_height),
        feet_phase_tracking_sigma=float(cfg.reward.feet_phase_tracking_sigma),
        base_height_target=float(cfg.reward.base_height_target),
        min_base_height=float(cfg.reward.min_base_height),
        max_tilt_deg=float(cfg.reward.max_tilt_deg),
        gait_constraint=GaitConstraintConfig(enabled=False),
        mode=RewardModeConfig(
            enabled=True,
            balance_common_terms=["base_height"],
            stand_terms=[],
            walk_terms=[],
        ),
        pose_weights=list(OmegaConf.to_container(cfg.reward.pose_weights, resolve=True)),
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._enable_reward_log = True
    low_height = 0.300
    target_height = float(cfg.reward.base_height_target)
    expected_raw = float(cfg.reward.scales.base_height) * (low_height - target_height) ** 2
    expected_step = expected_raw * env._cfg.ctrl_dt

    ctx = RewardContext(
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "steps": np.zeros((2,), dtype=np.uint32),
        },
        linvel=np.zeros((2, 3), dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.zeros((2, 29), dtype=np.float32),
        dof_vel=np.zeros((2, 29), dtype=np.float32),
        num_envs=2,
        default_angles=np.zeros((29,), dtype=np.float32),
        tracking_sigma=reward_cfg.tracking_sigma,
        base_height_target=target_height,
        base_height=np.asarray([low_height, low_height], dtype=np.float32),
        gravity=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        pose_weights=np.ones((29,), dtype=np.float32),
    )

    reward = env._compute_mode_reward(ctx, reward_cfg)

    np.testing.assert_allclose(
        reward, np.asarray([expected_step, expected_step]), rtol=1e-6
    )
    np.testing.assert_allclose(ctx.info["log"]["reward/base_height"], expected_raw, rtol=1e-6)
    np.testing.assert_allclose(ctx.info["log"]["reward/stand_total"], expected_step / 2.0, rtol=1e-6)
    np.testing.assert_allclose(ctx.info["log"]["reward/walk_total"], expected_step / 2.0, rtol=1e-6)


def test_active_g1_standing_reward_prefers_balanced_residual_over_quiet_fall() -> None:
    with initialize(config_path="../../../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_flat/mujoco"])

    common_terms = list(OmegaConf.to_container(cfg.reward.mode.balance_common_terms, resolve=True))
    stand_terms = list(OmegaConf.to_container(cfg.reward.mode.stand_terms, resolve=True))
    assert "upright" in common_terms
    assert "stand_action_l2" in stand_terms
    assert cfg.reward.scales.upright > 0.0
    assert abs(float(cfg.reward.scales.stand_action_l2)) < 0.1

    reward_cfg = G1WalkRewardConfig(
        scales=dict(OmegaConf.to_container(cfg.reward.scales, resolve=True)),
        tracking_sigma=float(cfg.reward.tracking_sigma),
        gait_frequency=float(cfg.reward.gait_frequency),
        feet_phase_swing_height=float(cfg.reward.feet_phase_swing_height),
        feet_phase_tracking_sigma=float(cfg.reward.feet_phase_tracking_sigma),
        base_height_target=float(cfg.reward.base_height_target),
        min_base_height=float(cfg.reward.min_base_height),
        max_tilt_deg=float(cfg.reward.max_tilt_deg),
        gait_constraint=GaitConstraintConfig(enabled=False),
        mode=RewardModeConfig(
            enabled=True,
            balance_common_terms=common_terms,
            stand_terms=stand_terms,
            walk_terms=[],
        ),
        pose_weights=list(OmegaConf.to_container(cfg.reward.pose_weights, resolve=True)),
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._enable_reward_log = True
    tilt_rad = np.deg2rad(67.0)
    residual_action = np.full((29,), 0.25, dtype=np.float32)

    ctx = RewardContext(
        info={
            "commands": np.zeros((2, 3), dtype=np.float32),
            "current_actions": np.stack(
                [residual_action, np.zeros((29,), dtype=np.float32)], axis=0
            ),
            "last_actions": np.zeros((2, 29), dtype=np.float32),
            "steps": np.zeros((2,), dtype=np.uint32),
        },
        linvel=np.zeros((2, 3), dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.zeros((2, 29), dtype=np.float32),
        dof_vel=np.zeros((2, 29), dtype=np.float32),
        num_envs=2,
        default_angles=np.zeros((29,), dtype=np.float32),
        tracking_sigma=reward_cfg.tracking_sigma,
        base_height_target=float(cfg.reward.base_height_target),
        base_height=np.asarray([float(cfg.reward.base_height_target), 0.35], dtype=np.float32),
        gravity=np.asarray(
            [[0.0, 0.0, 1.0], [np.sin(tilt_rad), 0.0, np.cos(tilt_rad)]],
            dtype=np.float32,
        ),
        pose_weights=np.ones((29,), dtype=np.float32),
    )

    reward = env._compute_mode_reward(ctx, reward_cfg)

    assert reward[0] > 0.0
    assert reward[1] < 0.0
    assert reward[0] > reward[1] + 0.5
    assert ctx.info["log"]["reward/upright"] > 0.0
    assert ctx.info["log"]["reward/stand_action_l2"] > -0.01


def test_mode_reward_logs_shared_terms_without_overwrite() -> None:
    reward_cfg = G1WalkRewardConfig(
        scales={"alive": 2.0},
        tracking_sigma=0.12,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_constraint=GaitConstraintConfig(enabled=False),
        mode=RewardModeConfig(
            enabled=True,
            stand_terms=["alive"],
            walk_terms=["alive"],
        ),
        pose_weights=[0.01] * 29,
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._enable_reward_log = True
    ctx = RewardContext(
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "steps": np.zeros((2,), dtype=np.uint32),
        },
        linvel=np.zeros((2, 3), dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.zeros((2, 29), dtype=np.float32),
        dof_vel=np.zeros((2, 29), dtype=np.float32),
        num_envs=2,
        default_angles=np.zeros((29,), dtype=np.float32),
        tracking_sigma=reward_cfg.tracking_sigma,
        base_height_target=reward_cfg.base_height_target,
        base_height=np.full((2,), reward_cfg.base_height_target, dtype=np.float32),
        gravity=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        pose_weights=np.ones((29,), dtype=np.float32),
    )

    reward = env._compute_mode_reward(ctx, reward_cfg)

    np.testing.assert_allclose(reward, np.full((2,), 2.0 * env._cfg.ctrl_dt, dtype=np.float32))
    assert ctx.info["log"]["reward/alive"] == 2.0
    assert ctx.info["log"]["reward/stand_total"] > 0.0
    assert ctx.info["log"]["reward/walk_total"] > 0.0


def test_reward_mode_logs_reward_prefixed_live_path_diagnostics() -> None:
    reward_cfg = G1WalkRewardConfig(
        scales={
            "stand_lin_vel_xy_l2": -30.0,
            "tracking_lin_vel": 2.0,
        },
        tracking_sigma=0.12,
        gait_frequency=1.5,
        feet_phase_swing_height=0.09,
        feet_phase_tracking_sigma=0.04,
        base_height_target=0.754,
        min_base_height=0.3,
        max_tilt_deg=65.0,
        gait_constraint=GaitConstraintConfig(enabled=False),
        mode=RewardModeConfig(
            enabled=True,
            stand_terms=["stand_lin_vel_xy_l2"],
            walk_terms=["tracking_lin_vel"],
        ),
        pose_weights=[0.01] * 29,
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._enable_reward_log = True
    ctx = RewardContext(
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "gait_enabled": np.asarray([0.0, 1.0], dtype=np.float32),
            "steps": np.zeros((2,), dtype=np.uint32),
        },
        linvel=np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        dof_pos=np.zeros((2, 29), dtype=np.float32),
        dof_vel=np.zeros((2, 29), dtype=np.float32),
        num_envs=2,
        default_angles=np.zeros((29,), dtype=np.float32),
        tracking_sigma=reward_cfg.tracking_sigma,
        base_height_target=reward_cfg.base_height_target,
        base_height=np.full((2,), reward_cfg.base_height_target, dtype=np.float32),
        gravity=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        pose_weights=np.ones((29,), dtype=np.float32),
    )

    env._compute_mode_reward(ctx, reward_cfg)

    log = ctx.info["log"]
    assert log["reward/mode_stand_frac"] == 0.5
    assert log["reward/mode_walk_frac"] == 0.5
    assert log["reward/stand_total"] < 0.0
    assert log["reward/walk_total"] > 0.0


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


def test_stand_mode_can_apply_double_stance_gait_cost() -> None:
    reward_cfg = _reward_config(
        apply_in_stand_mode=True,
        epsilon=0.0,
        penalty_scale=1.0,
    )
    env = _fake_env(reward_cfg)
    reward = np.asarray([1.0], dtype=np.float32)
    ctx = RewardContext(
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            "gait_phase": np.asarray([[np.pi, np.pi]], dtype=np.float32),
        },
        linvel=np.zeros((1, 3), dtype=np.float32),
        gyro=np.zeros((1, 3), dtype=np.float32),
        dof_pos=np.zeros((1, 29), dtype=np.float32),
        num_envs=1,
    )

    components = env._compute_gait_constraint_components(ctx, reward_cfg.gait_constraint)
    bridged_reward = env._apply_gait_constraint_bridge(ctx, reward)

    assert components["command_active"][0] == 0.0
    assert components["gate"][0] == 1.0
    assert components["total"][0] > 0.0
    assert bridged_reward[0] < reward[0]


def test_stand_phase_replaces_observation_phase_for_inactive_command() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[np.pi, np.pi],
    )
    env = _fake_env(reward_cfg, num_envs=2)
    info = {
        "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
        "gait_phase": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    }

    phase = env._gait_phase_for_observation(info)

    np.testing.assert_allclose(phase[0], np.asarray([np.pi, np.pi], dtype=np.float32))
    np.testing.assert_allclose(phase[1], np.asarray([3.0, 4.0], dtype=np.float32))


def test_apply_action_freezes_stand_phase_and_advances_active_phase() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[np.pi, np.pi],
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
    np.testing.assert_allclose(state.info["gait_phase"][0], np.asarray([np.pi, np.pi]))
    np.testing.assert_allclose(state.info["gait_phase"][1], np.asarray([3.1, 4.1]))


def test_apply_action_freezes_phase_after_external_command_is_zeroed() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[np.pi, np.pi],
    )
    env = _fake_env(reward_cfg, num_envs=1)
    state = NpEnvState(
        obs={},
        reward=np.zeros((1,), dtype=np.float32),
        terminated=np.zeros((1,), dtype=bool),
        truncated=np.zeros((1,), dtype=bool),
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            "gait_enabled": np.asarray([1.0], dtype=np.float32),
            "gait_phase": np.asarray([[1.0, 2.0]], dtype=np.float32),
        },
    )

    env.apply_action(np.zeros((1, 29), dtype=np.float32), state)

    np.testing.assert_array_equal(state.info["gait_enabled"], np.asarray([0.0], dtype=np.float32))
    np.testing.assert_allclose(state.info["gait_phase"][0], np.asarray([np.pi, np.pi]))


def test_apply_action_removes_stand_action_authority_for_inactive_command() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[np.pi, np.pi],
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._cfg.stand_action_authority = True
    actions = np.ones((2, 29), dtype=np.float32)
    state = NpEnvState(
        obs={},
        reward=np.zeros((2,), dtype=np.float32),
        terminated=np.zeros((2,), dtype=bool),
        truncated=np.zeros((2,), dtype=bool),
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "gait_phase": np.asarray([[np.pi, np.pi], [3.0, 4.0]], dtype=np.float32),
        },
    )

    ctrl = env.apply_action(actions, state)

    np.testing.assert_allclose(state.info["current_actions"], actions)
    np.testing.assert_allclose(state.info["executed_actions"][0], np.zeros((29,), dtype=np.float32))
    np.testing.assert_allclose(state.info["executed_actions"][1], np.ones((29,), dtype=np.float32))
    np.testing.assert_allclose(ctrl[0], env.default_angles)
    np.testing.assert_allclose(ctrl[1], np.ones((29,), dtype=np.float32))


def test_apply_action_preserves_policy_actions_when_stand_authority_disabled() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[np.pi, np.pi],
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._cfg.stand_action_authority = False
    actions = np.ones((2, 29), dtype=np.float32)
    state = NpEnvState(
        obs={},
        reward=np.zeros((2,), dtype=np.float32),
        terminated=np.zeros((2,), dtype=bool),
        truncated=np.zeros((2,), dtype=bool),
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "gait_phase": np.asarray([[np.pi, np.pi], [3.0, 4.0]], dtype=np.float32),
        },
    )

    ctrl = env.apply_action(actions, state)

    np.testing.assert_allclose(state.info["current_actions"], actions)
    np.testing.assert_allclose(state.info["executed_actions"], actions)
    np.testing.assert_allclose(ctrl, actions)
    assert state.info["gait_enabled"][0] == 0.0
    assert state.info["gait_enabled"][1] == 1.0


def test_apply_action_logs_policy_action_diagnostics_when_stand_authority_disabled() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[np.pi, np.pi],
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._cfg.stand_action_authority = False
    env._enable_reward_log = True
    actions = np.ones((2, 29), dtype=np.float32)
    state = NpEnvState(
        obs={},
        reward=np.zeros((2,), dtype=np.float32),
        terminated=np.zeros((2,), dtype=bool),
        truncated=np.zeros((2,), dtype=bool),
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "gait_phase": np.asarray([[np.pi, np.pi], [3.0, 4.0]], dtype=np.float32),
            "steps": np.zeros((2,), dtype=np.uint32),
        },
    )

    env.apply_action(actions, state)

    log = state.info["log"]
    assert log["reward/action_authority_stand_frac"] == 0.5
    assert log["reward/stand_raw_action_l1"] == 29.0
    assert log["reward/stand_executed_action_l1"] == 29.0


def test_apply_action_logs_stand_action_authority_live_path() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[np.pi, np.pi],
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._cfg.stand_action_authority = True
    env._enable_reward_log = True
    actions = np.ones((2, 29), dtype=np.float32)
    state = NpEnvState(
        obs={},
        reward=np.zeros((2,), dtype=np.float32),
        terminated=np.zeros((2,), dtype=bool),
        truncated=np.zeros((2,), dtype=bool),
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "gait_phase": np.asarray([[np.pi, np.pi], [3.0, 4.0]], dtype=np.float32),
            "steps": np.zeros((2,), dtype=np.uint32),
        },
    )

    env.apply_action(actions, state)

    log = state.info["log"]
    assert log["reward/action_authority_stand_frac"] == 0.5
    assert log["reward/stand_raw_action_l1"] == 29.0
    assert log["reward/stand_executed_action_l1"] == 0.0
    assert log["reward/executed_action_l1"] == 14.5


def test_reward_logging_preserves_action_authority_after_dispatch() -> None:
    reward_cfg = _reward_config(
        freeze_phase_in_stand_mode=True,
        stand_phase=[np.pi, np.pi],
    )
    env = _fake_env(reward_cfg, num_envs=2)
    env._cfg.stand_action_authority = True
    env._enable_reward_log = True
    state = NpEnvState(
        obs={},
        reward=np.zeros((2,), dtype=np.float32),
        terminated=np.zeros((2,), dtype=bool),
        truncated=np.zeros((2,), dtype=bool),
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            "gait_phase": np.asarray([[np.pi, np.pi], [3.0, 4.0]], dtype=np.float32),
            "steps": np.zeros((2,), dtype=np.uint32),
        },
    )
    env.apply_action(np.ones((2, 29), dtype=np.float32), state)

    env._compute_reward(
        state.info,
        linvel=np.zeros((2, 3), dtype=np.float32),
        gyro=np.zeros((2, 3), dtype=np.float32),
        gravity=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        dof_pos=np.zeros((2, 29), dtype=np.float32),
        dof_vel=np.zeros((2, 29), dtype=np.float32),
    )

    log = state.info["log"]
    assert log["reward/action_authority_stand_frac"] == 0.5
    assert log["reward/stand_raw_action_l1"] == 29.0
    assert log["reward/stand_executed_action_l1"] == 0.0


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


def test_stand_drift_rewards_only_apply_when_command_inactive() -> None:
    reward_cfg = _reward_config()
    env = _fake_env(reward_cfg, num_envs=2)
    ctx = RewardContext(
        info={
            "commands": np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
        },
        linvel=np.asarray([[0.1, 0.2, 0.0], [0.1, 0.2, 0.0]], dtype=np.float32),
        gyro=np.asarray([[0.0, 0.0, 0.3], [0.0, 0.0, 0.3]], dtype=np.float32),
        dof_pos=np.zeros((2, 29), dtype=np.float32),
        num_envs=2,
    )

    np.testing.assert_allclose(
        env._reward_stand_lin_vel_xy_l2(ctx),
        np.asarray([0.05, 0.0], dtype=np.float32),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        env._reward_stand_yaw_vel_l2(ctx),
        np.asarray([0.09, 0.0], dtype=np.float32),
        rtol=1.0e-6,
    )


def test_nonzero_command_applies_gait_constraint_cost() -> None:
    reward_cfg = _reward_config(epsilon=0.0, penalty_scale=1.0)
    env = _fake_env(reward_cfg)
    reward = np.asarray([1.0], dtype=np.float32)
    ctx = _ctx(np.asarray([[0.2, 0.0, 0.0]], dtype=np.float32), linvel_x=0.0)

    bridged_reward = env._apply_gait_constraint_bridge(ctx, reward)

    assert bridged_reward[0] < reward[0]
