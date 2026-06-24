"""Test reward config injection system."""

from typing import Any, cast

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf


def test_reward_config_loading_g1():
    """Test G1 SAC reward config loads correctly."""
    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_flat/mujoco"])
        assert hasattr(cfg, "reward")
        assert cfg.reward.scales.tracking_lin_vel == 2.0
        assert cfg.reward.scales.feet_phase == 0.0
        assert cfg.reward.scales.feet_phase_contrast == 0.0
        assert cfg.reward.scales.feet_phase_contact == 0.0
        assert cfg.reward.scales.alive == 2.0
        assert cfg.reward.scales.stand_still == -2.0
        assert cfg.reward.scales.stand_action_l2 == -2.0
        assert cfg.reward.scales.stand_dof_vel_l2 == -0.15
        assert cfg.reward.scales.stand_lin_vel_xy_l2 == -30.0
        assert cfg.reward.scales.stand_yaw_vel_l2 == -10.0
        assert cfg.reward.tracking_sigma == 0.12
        assert cfg.reward.base_height_target == 0.754
        assert cfg.reward.gait_constraint.enabled is True
        assert cfg.reward.gait_constraint.freeze_phase_in_stand_mode is True
        assert cfg.reward.gait_constraint.apply_in_stand_mode is True
        assert cfg.reward.gait_constraint.contrast_weight == 2.0
        assert cfg.reward.gait_constraint.contact_weight == 1.0
        assert cfg.reward.gait_constraint.epsilon == 0.0
        assert cfg.reward.gait_constraint.penalty_scale == 2.0
        assert cfg.reward.gait_constraint.stand_phase == [
            3.141592653589793,
            3.141592653589793,
        ]
        assert cfg.env.commands.vel_limit[0] == [-0.3, -0.2, -0.4]
        assert cfg.env.commands.small_xy_threshold == 0.05
        assert cfg.env.commands.rel_standing_envs == 0.4
        assert cfg.env.stand_action_authority is False
        assert cfg.reward.mode.enabled is True
        assert "tracking_lin_vel" not in cfg.reward.mode.stand_terms
        assert "stand_lin_vel_xy_l2" in cfg.reward.mode.stand_terms
        assert "stand_lin_vel_xy_l2" not in cfg.reward.mode.walk_terms
        assert "tracking_lin_vel" in cfg.reward.mode.walk_terms
        assert "stand_action_l2" not in cfg.reward.mode.walk_terms
        assert cfg.reward.pose_weights[2] == 0.05
        assert cfg.reward.pose_weights[8] == 0.05


def test_reward_config_loading_g1_motrix():
    """Test G1 Motrix reward config loads correctly."""
    with initialize(config_path="../../conf/offpolicy", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["task=sac/g1_walk_flat/motrix"])
        assert hasattr(cfg, "reward")
        assert cfg.reward.scales.tracking_lin_vel == 2.2
        assert cfg.reward.scales.alive == 12.0


def test_resolve_reward_dict_reads_task_reward():
    """Task-backend configs should expose the final reward mapping directly."""
    from unilab.training.reward import resolve_reward_dict

    with initialize(config_path="../../conf/ppo", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["task=go2_joystick_flat/motrix"],
        )

    reward_dict = resolve_reward_dict(cfg)

    assert reward_dict["scales"]["tracking_lin_vel"] == 1.0
    assert reward_dict["scales"]["tracking_ang_vel"] == 0.2


def test_reward_config_conversion():
    """Test reward config converts to dataclasses via registry."""
    from unilab.base import registry
    from unilab.base.registry import ensure_registries

    ensure_registries()

    # Test G1 walk config - registry auto-converts dict to G1WalkRewardConfig
    g1_dict = {
        "scales": {"tracking_lin_vel": 2.0, "alive": 10.0},
        "tracking_sigma": 0.25,
        "base_height_target": 0.754,
        "gait_frequency": 1.5,
        "feet_phase_swing_height": 0.09,
        "feet_phase_tracking_sigma": 0.008,
        "min_base_height": 0.3,
        "max_tilt_deg": 65.0,
        "close_feet_threshold": 0.15,
        "pose_weights": [0.01] * 29,
    }
    env = cast(
        Any,
        registry.make(
            "G1WalkFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": g1_dict},
        ),
    )
    assert hasattr(env._cfg.reward_config, "scales")
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 2.0
    env.close()

    # Test Go1 config - registry auto-converts dict to RewardConfig
    go1_dict = {
        "scales": {"tracking_lin_vel": 1.0, "base_height": -100.0},
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
    }
    env = cast(
        Any,
        registry.make(
            "Go1JoystickFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": go1_dict},
        ),
    )
    assert hasattr(env._cfg.reward_config, "scales")
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 1.0
    env.close()
