#!/usr/bin/env python3
"""Minimal Reward Lab for G1 standing/walking preference diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from hydra import compose, initialize_config_dir

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from unilab.training import (  # noqa: E402
    BackendAdapter,
    assert_offpolicy_task_choice_matches_algo,
    create_env,
    ensure_registries,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    action_fn: Callable[[Any, dict[str, Any], int], np.ndarray]
    intent: str


LEG_PITCH_JOINTS = [
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
]
HIP_ROLL_JOINTS = ["left_hip_roll_joint", "right_hip_roll_joint"]
ANKLE_ROLL_JOINTS = ["left_ankle_roll_joint", "right_ankle_roll_joint"]


def _action_scale(env: Any) -> np.ndarray:
    scale = np.asarray(env.cfg.control_config.action_scale, dtype=np.float32)
    if scale.ndim == 0:
        scale = np.full((env.action_space.shape[0],), float(scale), dtype=np.float32)
    return np.where(np.abs(scale) < 1.0e-6, 1.0, scale).astype(np.float32)


def _set_joint_offsets(env: Any, action: np.ndarray, updates: dict[str, Any]) -> np.ndarray:
    indices = _joint_indices(env, list(updates))
    scale = _action_scale(env)
    for name, value in updates.items():
        action[:, indices[name]] = np.asarray(value, dtype=np.float32) / scale[indices[name]]
    return action


def _compose_cfg(stage: str):
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=["task=sac/g1_walk_flat/mujoco", f"+g1_walk_stage={stage}"],
        )


def _create_env(stage: str, num_envs: int):
    cfg = _compose_cfg(stage)
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name="sac")
    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="sac")
    env_override = adapter.build_task_env_cfg_override()
    ensure_registries()
    env = create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=env_override,
        sim_backend="mujoco",
        task_name="G1WalkFlat",
    )
    env.set_autoreset(False)
    return env, env_override


def _joint_indices(env: Any, names: list[str]) -> dict[str, int]:
    indices = env._backend.get_joint_dof_pos_indices(names)
    return {name: int(index) for name, index in zip(names, indices, strict=True)}


def _zero_action(env: Any) -> np.ndarray:
    return np.zeros((env.num_envs, env.action_space.shape[0]), dtype=np.float32)


def _constant_leg_pitch(
    env: Any,
    *,
    left_hip: float,
    left_knee: float,
    left_ankle: float,
    right_hip: float | None = None,
    right_knee: float | None = None,
    right_ankle: float | None = None,
) -> np.ndarray:
    action = _zero_action(env)
    return _set_joint_offsets(env, action, {
        "left_hip_pitch_joint": left_hip,
        "left_knee_joint": left_knee,
        "left_ankle_pitch_joint": left_ankle,
        "right_hip_pitch_joint": left_hip if right_hip is None else right_hip,
        "right_knee_joint": left_knee if right_knee is None else right_knee,
        "right_ankle_pitch_joint": left_ankle if right_ankle is None else right_ankle,
    })


def _hip_roll_action(env: Any, left: float, right: float) -> np.ndarray:
    action = _zero_action(env)
    return _set_joint_offsets(
        env,
        action,
        {
            "left_hip_roll_joint": left,
            "right_hip_roll_joint": right,
        },
    )


def _standing_brace_action(env: Any) -> np.ndarray:
    return _constant_leg_pitch(env, left_hip=-0.08, left_knee=0.18, left_ankle=-0.10)


def _soft_leg_oscillation_action(env: Any, step: int) -> np.ndarray:
    knee = 0.12 + 0.08 * np.sin(0.35 * step)
    return _constant_leg_pitch(env, left_hip=-0.05, left_knee=knee, left_ankle=-0.06)


def _feedback_standing_recovery_action(env: Any) -> np.ndarray:
    action = _zero_action(env)
    base_delta = _base_feet_center_delta(env)
    linvel = env.get_local_linvel()
    sagittal = np.clip(-0.20 * base_delta[:, 0] - 0.08 * linvel[:, 0], -0.10, 0.10)
    lateral = np.clip(-0.30 * base_delta[:, 1] - 0.08 * linvel[:, 1], -0.12, 0.12)
    return _set_joint_offsets(
        env,
        action,
        {
            "left_hip_pitch_joint": -0.04 + sagittal,
            "right_hip_pitch_joint": -0.04 + sagittal,
            "left_knee_joint": 0.12,
            "right_knee_joint": 0.12,
            "left_ankle_pitch_joint": -0.08 - sagittal,
            "right_ankle_pitch_joint": -0.08 - sagittal,
            "left_hip_roll_joint": lateral,
            "right_hip_roll_joint": lateral,
            "left_ankle_roll_joint": -0.5 * lateral,
            "right_ankle_roll_joint": -0.5 * lateral,
        },
    )


def _standing_candidates() -> list[Candidate]:
    return [
        Candidate(
            "standing_deployable_high",
            lambda env, info, step: _zero_action(env),
            "default stand keyframe; should be the clean deployable standing anchor",
        ),
        Candidate(
            "standing_low_crouch",
            lambda env, info, step: _constant_leg_pitch(
                env, left_hip=-0.18, left_knee=0.38, left_ankle=-0.20
            ),
            "symmetric knee-flexed target; should be lower than default stand",
        ),
        Candidate(
            "standing_rearward_com",
            lambda env, info, step: _constant_leg_pitch(
                env, left_hip=0.18, left_knee=-0.08, left_ankle=-0.20
            ),
            "symmetric lean target; should move the base/feet center away from neutral",
        ),
        Candidate(
            "standing_wide_feet",
            lambda env, info, step: _hip_roll_action(env, 0.55, -0.55),
            "symmetric hip-roll target; should widen stance without changing command mode",
        ),
    ]


def _standing_static_authority_candidates() -> list[Candidate]:
    return [
        Candidate(
            "standing_static_zero_action",
            lambda env, info, step: _zero_action(env),
            "static Standing should prefer quiet default action",
        ),
        Candidate(
            "standing_static_brace_action",
            lambda env, info, step: _standing_brace_action(env),
            "static Standing should not require a continuous bracing action",
        ),
        Candidate(
            "standing_static_soft_leg_oscillation",
            lambda env, info, step: _soft_leg_oscillation_action(env, step),
            "static Standing should penalize continuous soft-leg balancing",
        ),
    ]


def _walking_phase_action(env: Any, info: dict[str, Any], *, lift_scale: float) -> np.ndarray:
    action = _zero_action(env)
    phase = np.asarray(info.get("gait_phase"), dtype=np.float32)
    left_swing = np.maximum(np.sin(phase[:, 0]), 0.0)
    right_swing = np.maximum(np.sin(phase[:, 1]), 0.0)
    left_stance = 1.0 - np.clip(left_swing, 0.0, 1.0)
    right_stance = 1.0 - np.clip(right_swing, 0.0, 1.0)

    action = _set_joint_offsets(
        env,
        action,
        {
            "left_hip_pitch_joint": (-0.18 * left_swing + 0.08 * left_stance) * lift_scale,
            "left_knee_joint": (0.72 * left_swing + 0.02 * left_stance) * lift_scale,
            "left_ankle_pitch_joint": (-0.34 * left_swing - 0.02 * left_stance) * lift_scale,
            "right_hip_pitch_joint": (-0.18 * right_swing + 0.08 * right_stance) * lift_scale,
            "right_knee_joint": (0.72 * right_swing + 0.02 * right_stance) * lift_scale,
            "right_ankle_pitch_joint": (-0.34 * right_swing - 0.02 * right_stance) * lift_scale,
        },
    )
    return action.astype(np.float32)


def _walking_candidates() -> list[Candidate]:
    return [
        Candidate(
            "walking_0p1_static_stand",
            lambda env, info, step: _zero_action(env),
            "walk command but default stand action; should underperform real stepping",
        ),
        Candidate(
            "walking_0p1_forward_step",
            lambda env, info, step: _walking_phase_action(env, info, lift_scale=1.0),
            "alternating sagittal leg action; should create swing clearance and forward motion",
        ),
        Candidate(
            "walking_0p1_dragging_feet",
            lambda env, info, step: _walking_phase_action(env, info, lift_scale=0.25),
            "same gait phase with suppressed lift; should have lower clearance",
        ),
        Candidate(
            "walking_0p1_in_place_open_close",
            lambda env, info, step: _hip_roll_action(
                env,
                left=0.25 * np.sin(0.25 * step),
                right=-0.25 * np.sin(0.25 * step),
            ),
            "lateral open-close motion; should change width without meaningful forward step",
        ),
    ]


def _standing_recovery_authority_candidates() -> list[Candidate]:
    return [
        Candidate(
            "standing_recovery_zero_action",
            lambda env, info, step: _zero_action(env),
            "after a zero-command disturbance, return directly to default standing action",
        ),
        Candidate(
            "standing_recovery_brace_action",
            lambda env, info, step: _standing_brace_action(env),
            "after a zero-command disturbance, use a short bracing standing action",
        ),
        Candidate(
            "standing_recovery_feedback_action",
            lambda env, info, step: _feedback_standing_recovery_action(env),
            "after a zero-command disturbance, use a bounded feedback standing recovery action",
        ),
        Candidate(
            "standing_recovery_open_close",
            lambda env, info, step: _hip_roll_action(
                env,
                left=0.25 * np.sin(0.25 * step),
                right=-0.25 * np.sin(0.25 * step),
            ),
            "after a zero-command disturbance, use lateral opening/closing correction",
        ),
        Candidate(
            "standing_recovery_keep_stepping",
            lambda env, info, step: _walking_phase_action(env, info, lift_scale=0.5),
            "after a zero-command disturbance, keep a gait-like stepping action",
        ),
    ]


def _transition_recovery_candidates() -> list[Candidate]:
    return [
        Candidate(
            "transition_walk_to_stand_zero_action",
            lambda env, info, step: _zero_action(env),
            "after command drops to zero, instantly return to default standing action",
        ),
        Candidate(
            "transition_walk_to_stand_taper_gait",
            lambda env, info, step: _walking_phase_action(
                env, info, lift_scale=max(0.0, 1.0 - float(step - 1) / 8.0)
            ),
            "after command drops to zero, taper the walking recovery action back to stand",
        ),
        Candidate(
            "transition_walk_to_stand_continue_stepping",
            lambda env, info, step: _walking_phase_action(env, info, lift_scale=1.0),
            "after command drops to zero, keep walking gait action active",
        ),
        Candidate(
            "transition_walk_to_stand_open_close",
            lambda env, info, step: _hip_roll_action(
                env,
                left=0.25 * np.sin(0.25 * step),
                right=-0.25 * np.sin(0.25 * step),
            ),
            "after command drops to zero, keep lateral open-close correction active",
        ),
        Candidate(
            "transition_walk_to_stand_dragging",
            lambda env, info, step: _walking_phase_action(env, info, lift_scale=0.25),
            "after command drops to zero, keep low-clearance dragging gait active",
        ),
    ]


def _force_command(state: Any, command: np.ndarray) -> None:
    num_envs = state.info["commands"].shape[0]
    state.info["commands"] = np.broadcast_to(command.astype(np.float32), (num_envs, 3)).copy()


def _sync_envs_to_reference_state(env: Any, state: Any, ref_index: int = 0) -> None:
    backend = env._backend
    qpos = np.repeat(backend._qpos_view[ref_index : ref_index + 1], env.num_envs, axis=0)
    qvel_view = backend._physics_state[
        :, backend._idx_qvel : backend._idx_qvel + backend.nv
    ]
    qvel = np.repeat(qvel_view[ref_index : ref_index + 1], env.num_envs, axis=0)
    backend.set_state(np.arange(env.num_envs, dtype=np.int32), qpos, qvel)

    for key, value in list(state.info.items()):
        if not isinstance(value, np.ndarray) or value.shape[:1] != (env.num_envs,):
            continue
        state.info[key] = np.repeat(value[ref_index : ref_index + 1], env.num_envs, axis=0)
    state.terminated[:] = False
    state.truncated[:] = False


def _apply_reference_qvel_disturbance(
    env: Any,
    state: Any,
    *,
    lin_xy: tuple[float, float],
    yaw: float,
    ref_index: int = 0,
) -> None:
    backend = env._backend
    qpos = np.repeat(backend._qpos_view[ref_index : ref_index + 1], env.num_envs, axis=0)
    qvel_view = backend._physics_state[
        :, backend._idx_qvel : backend._idx_qvel + backend.nv
    ]
    qvel = np.repeat(qvel_view[ref_index : ref_index + 1], env.num_envs, axis=0)
    qvel[:, 0] = float(lin_xy[0])
    qvel[:, 1] = float(lin_xy[1])
    qvel[:, 5] = float(yaw)
    backend.set_state(np.arange(env.num_envs, dtype=np.int32), qpos, qvel)
    state.terminated[:] = False
    state.truncated[:] = False


def _sensor(env: Any, name: str) -> np.ndarray:
    return np.asarray(env._backend.get_sensor_data(name), dtype=np.float32)


def _tilt_deg(env: Any) -> np.ndarray:
    gravity = _sensor(env, env.cfg.sensor.upvector)
    return np.rad2deg(np.arccos(np.clip(gravity[:, 2], -1.0, 1.0))).astype(np.float32)


def _base_feet_center_delta(env: Any) -> np.ndarray:
    return env._base_delta_from_feet_center_in_base_yaw_frame()


def _foot_metrics(env: Any) -> dict[str, np.ndarray]:
    left = _sensor(env, "left_foot_pos")
    right = _sensor(env, "right_foot_pos")
    delta = env._feet_delta_in_base_yaw_frame(left, right)
    stance_z = np.minimum(left[:, 2], right[:, 2])
    return {
        "foot_width": np.abs(delta[:, 1]),
        "foot_sagittal_abs": np.abs(delta[:, 0]),
        "left_clearance": left[:, 2] - stance_z,
        "right_clearance": right[:, 2] - stance_z,
    }


def _swing_clearance(row: dict[str, Any]) -> float:
    return max(float(row["max_left_clearance"]), float(row["max_right_clearance"]))


def _reward_context(env: Any, state: Any):
    linvel = env.get_local_linvel()
    gyro = env.get_gyro()
    gravity = env._backend.get_sensor_data(env.cfg.sensor.upvector)
    dof_pos = env.get_dof_pos()
    dof_vel = env.get_dof_vel()
    return env._build_reward_context(state.info, linvel, gyro, gravity, dof_pos, dof_vel)


def _mode_masks(env: Any, ctx: Any) -> dict[str, np.ndarray]:
    walk = env._gait_enabled_mask(ctx.info)
    stand = np.asarray(1.0 - walk, dtype=np.float32)
    recovery = env._stand_recovery_mask(ctx, stand)
    static = np.asarray(stand - recovery, dtype=np.float32)
    return {"stand_static": static, "stand_recovery": recovery, "walk": walk}


def _section_contributions(env: Any, state: Any) -> dict[str, Any]:
    ctx = _reward_context(env, state)
    mode_cfg = env._reward_mode_cfg()
    masks = _mode_masks(env, ctx)
    sections = {
        "standing": env._combine_mode_terms(mode_cfg.balance_common_terms, mode_cfg.stand_terms),
        "standing_recovery": env._combine_mode_terms(
            mode_cfg.balance_common_terms, mode_cfg.stand_recovery_terms
        ),
        "walking": env._combine_mode_terms(mode_cfg.balance_common_terms, mode_cfg.walk_terms),
    }
    section_scale_overrides = {
        "standing": getattr(mode_cfg, "stand_scale_overrides", {}),
        "standing_recovery": getattr(mode_cfg, "stand_recovery_scale_overrides", {}),
        "walking": getattr(mode_cfg, "walk_scale_overrides", {}),
    }
    section_masks = {
        "standing": masks["stand_static"],
        "standing_recovery": masks["stand_recovery"],
        "walking": masks["walk"],
    }
    out: dict[str, Any] = {"sections": {}, "terms": {}}
    for section, terms in sections.items():
        total = np.zeros((env.num_envs,), dtype=np.float32)
        mask = section_masks[section]
        scale_overrides = section_scale_overrides[section]
        section_terms: dict[str, list[float]] = {}
        for name in terms:
            scale = float(scale_overrides.get(name, env.cfg.reward_config.scales.get(name, 0.0)))
            if scale == 0.0 or name not in env._reward_fns:
                continue
            values = np.asarray(env._reward_fns[name](ctx) * scale * mask * env.cfg.ctrl_dt, dtype=np.float32)
            total += values
            section_terms[name] = [float(v) for v in values]
        out["sections"][section] = [float(v) for v in total]
        out["terms"][section] = section_terms

    gait_cfg = env._gait_constraint_cfg()
    components = env._compute_gait_constraint_components(ctx, gait_cfg)
    excess = np.maximum(components["total"] - gait_cfg.epsilon, 0.0)
    gait_cost = -float(gait_cfg.penalty_scale) * excess * components["gate"] * env.cfg.ctrl_dt
    out["sections"]["gait_constraint"] = [float(v) for v in gait_cost]
    out["terms"]["gait_constraint"] = {"gait_constraint": [float(v) for v in gait_cost]}
    return out


def _candidate_term_contributions(contributions: dict[str, Any], index: int) -> dict[str, dict[str, float]]:
    return {
        section: {name: float(values[index]) for name, values in terms.items()}
        for section, terms in contributions["terms"].items()
    }


def _zero_contributions_like(contributions: dict[str, Any]) -> dict[str, Any]:
    return {
        "sections": {
            section: [0.0 for _ in values] for section, values in contributions["sections"].items()
        },
        "terms": {
            section: {name: [0.0 for _ in values] for name, values in terms.items()}
            for section, terms in contributions["terms"].items()
        },
    }


def _add_contributions(total: dict[str, Any], step: dict[str, Any]) -> None:
    for section, values in step["sections"].items():
        total["sections"].setdefault(section, [0.0 for _ in values])
        total["sections"][section] = [
            float(lhs) + float(rhs) for lhs, rhs in zip(total["sections"][section], values, strict=True)
        ]
    for section, terms in step["terms"].items():
        section_total = total["terms"].setdefault(section, {})
        for name, values in terms.items():
            section_total.setdefault(name, [0.0 for _ in values])
            section_total[name] = [
                float(lhs) + float(rhs) for lhs, rhs in zip(section_total[name], values, strict=True)
            ]


def _run_counterfactual(
    *,
    section: str,
    stage: str,
    command: np.ndarray,
    candidates: list[Candidate],
    steps: int,
    seed: int,
) -> dict[str, Any]:
    np.random.seed(seed)
    env, env_override = _create_env(stage, len(candidates))
    try:
        state = env.init_state()
        _force_command(state, command)
        initial_base_x = np.asarray(env._backend.get_base_pos()[:, 0], dtype=np.float32).copy()
        cumulative = np.zeros((len(candidates),), dtype=np.float64)
        min_height = np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32).copy()
        max_tilt = _tilt_deg(env).astype(np.float64)
        max_left_clearance = np.zeros((len(candidates),), dtype=np.float64)
        max_right_clearance = np.zeros((len(candidates),), dtype=np.float64)
        initial_feet = _foot_metrics(env)
        min_foot_width = np.asarray(initial_feet["foot_width"], dtype=np.float32).copy()
        max_foot_width = np.asarray(initial_feet["foot_width"], dtype=np.float32).copy()
        first_terminated = np.full((len(candidates),), steps + 1, dtype=np.int32)
        cumulative_contributions: dict[str, Any] | None = None

        for step in range(1, steps + 1):
            actions = np.zeros((len(candidates), env.action_space.shape[0]), dtype=np.float32)
            for index, candidate in enumerate(candidates):
                action = candidate.action_fn(env, state.info, step)
                actions[index] = action[index]
            state = env.step(actions)
            _force_command(state, command)

            cumulative += np.asarray(state.reward, dtype=np.float64)
            step_contributions = _section_contributions(env, state)
            if cumulative_contributions is None:
                cumulative_contributions = _zero_contributions_like(step_contributions)
            _add_contributions(cumulative_contributions, step_contributions)
            height = np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32)
            min_height = np.minimum(min_height, height)
            max_tilt = np.maximum(max_tilt, _tilt_deg(env))
            feet = _foot_metrics(env)
            max_left_clearance = np.maximum(max_left_clearance, feet["left_clearance"])
            max_right_clearance = np.maximum(max_right_clearance, feet["right_clearance"])
            min_foot_width = np.minimum(min_foot_width, feet["foot_width"])
            max_foot_width = np.maximum(max_foot_width, feet["foot_width"])
            newly_done = np.asarray(state.terminated, dtype=bool) & (first_terminated > steps)
            first_terminated[newly_done] = step

        final_base_x = np.asarray(env._backend.get_base_pos()[:, 0], dtype=np.float32)
        final_height = np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32)
        base_delta = _base_feet_center_delta(env)
        feet = _foot_metrics(env)
        contributions = cumulative_contributions or _section_contributions(env, state)
        ctx = _reward_context(env, state)
        masks = _mode_masks(env, ctx)

        rows = []
        for index, candidate in enumerate(candidates):
            rows.append(
                {
                    "name": candidate.name,
                    "intent": candidate.intent,
                    "total_reward": float(cumulative[index]),
                    "standing": contributions["sections"]["standing"][index],
                    "standing_recovery": contributions["sections"]["standing_recovery"][index],
                    "walking": contributions["sections"]["walking"][index],
                    "gait_constraint": contributions["sections"]["gait_constraint"][index],
                    "term_contributions": _candidate_term_contributions(contributions, index),
                    "x_displacement": float(final_base_x[index] - initial_base_x[index]),
                    "base_feet_center_x": float(base_delta[index, 0]),
                    "base_feet_center_y": float(base_delta[index, 1]),
                    "foot_width": float(feet["foot_width"][index]),
                    "foot_width_range": float(max_foot_width[index] - min_foot_width[index]),
                    "foot_sagittal_abs": float(feet["foot_sagittal_abs"][index]),
                    "max_left_clearance": float(max_left_clearance[index]),
                    "max_right_clearance": float(max_right_clearance[index]),
                    "min_height": float(min_height[index]),
                    "final_height": float(final_height[index]),
                    "max_tilt_deg": float(max_tilt[index]),
                    "first_terminated_step": None
                    if int(first_terminated[index]) > steps
                    else int(first_terminated[index]),
                    "stand_static_mask": float(masks["stand_static"][index]),
                    "stand_recovery_mask": float(masks["stand_recovery"][index]),
                    "walk_mask": float(masks["walk"][index]),
                }
            )

        return {
            "section": section,
            "stage": stage,
            "command": [float(v) for v in command],
            "steps": steps,
            "env_override": {
                "rel_standing_envs": env_override.get("commands", {}).get("rel_standing_envs"),
                "rel_transition_envs": env_override.get("commands", {}).get("rel_transition_envs"),
                "mode_observation": env_override.get("mode_observation"),
            },
            "rows": rows,
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _run_switch_counterfactual(
    *,
    section: str,
    stage: str,
    pre_command: np.ndarray,
    post_command: np.ndarray,
    pre_action: Callable[[Any, dict[str, Any], int], np.ndarray],
    candidates: list[Candidate],
    pre_steps: int,
    post_steps: int,
    seed: int,
) -> dict[str, Any]:
    np.random.seed(seed)
    env, env_override = _create_env(stage, len(candidates))
    try:
        state = env.init_state()
        _force_command(state, pre_command)
        for step in range(1, pre_steps + 1):
            action = pre_action(env, state.info, step)
            state = env.step(action)
            _force_command(state, pre_command)

        _sync_envs_to_reference_state(env, state)
        switch_base_x = np.asarray(env._backend.get_base_pos()[:, 0], dtype=np.float32).copy()
        switch_height = np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32).copy()
        switch_tilt = _tilt_deg(env).astype(np.float32)
        _force_command(state, post_command)
        initial_recovery_mask = _mode_masks(env, _reward_context(env, state))["stand_recovery"]
        cumulative = np.zeros((len(candidates),), dtype=np.float64)
        cumulative_contributions: dict[str, Any] | None = None
        min_height = np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32).copy()
        max_tilt = _tilt_deg(env).astype(np.float64)
        max_left_clearance = np.zeros((len(candidates),), dtype=np.float64)
        max_right_clearance = np.zeros((len(candidates),), dtype=np.float64)
        initial_feet = _foot_metrics(env)
        min_foot_width = np.asarray(initial_feet["foot_width"], dtype=np.float32).copy()
        max_foot_width = np.asarray(initial_feet["foot_width"], dtype=np.float32).copy()
        first_terminated = np.full((len(candidates),), post_steps + 1, dtype=np.int32)

        for step in range(1, post_steps + 1):
            actions = np.zeros((len(candidates), env.action_space.shape[0]), dtype=np.float32)
            for index, candidate in enumerate(candidates):
                action = candidate.action_fn(env, state.info, step)
                actions[index] = action[index]
            state = env.step(actions)
            _force_command(state, post_command)
            cumulative += np.asarray(state.reward, dtype=np.float64)
            step_contributions = _section_contributions(env, state)
            if cumulative_contributions is None:
                cumulative_contributions = _zero_contributions_like(step_contributions)
            _add_contributions(cumulative_contributions, step_contributions)
            height = np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32)
            min_height = np.minimum(min_height, height)
            max_tilt = np.maximum(max_tilt, _tilt_deg(env))
            feet = _foot_metrics(env)
            max_left_clearance = np.maximum(max_left_clearance, feet["left_clearance"])
            max_right_clearance = np.maximum(max_right_clearance, feet["right_clearance"])
            min_foot_width = np.minimum(min_foot_width, feet["foot_width"])
            max_foot_width = np.maximum(max_foot_width, feet["foot_width"])
            newly_done = np.asarray(state.terminated, dtype=bool) & (first_terminated > post_steps)
            first_terminated[newly_done] = step

        final_base_x = np.asarray(env._backend.get_base_pos()[:, 0], dtype=np.float32)
        final_height = np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32)
        base_delta = _base_feet_center_delta(env)
        feet = _foot_metrics(env)
        contributions = cumulative_contributions or _section_contributions(env, state)
        ctx = _reward_context(env, state)
        masks = _mode_masks(env, ctx)
        rows = []
        for index, candidate in enumerate(candidates):
            rows.append(
                {
                    "name": candidate.name,
                    "intent": candidate.intent,
                    "total_reward": float(cumulative[index]),
                    "standing": contributions["sections"]["standing"][index],
                    "standing_recovery": contributions["sections"]["standing_recovery"][index],
                    "walking": contributions["sections"]["walking"][index],
                    "gait_constraint": contributions["sections"]["gait_constraint"][index],
                    "x_displacement": float(final_base_x[index] - switch_base_x[index]),
                    "base_feet_center_x": float(base_delta[index, 0]),
                    "base_feet_center_y": float(base_delta[index, 1]),
                    "foot_width": float(feet["foot_width"][index]),
                    "foot_width_range": float(max_foot_width[index] - min_foot_width[index]),
                    "foot_sagittal_abs": float(feet["foot_sagittal_abs"][index]),
                    "max_left_clearance": float(max_left_clearance[index]),
                    "max_right_clearance": float(max_right_clearance[index]),
                    "min_height": float(min_height[index]),
                    "final_height": float(final_height[index]),
                    "max_tilt_deg": float(max_tilt[index]),
                    "first_terminated_step": None
                    if int(first_terminated[index]) > post_steps
                    else int(first_terminated[index]),
                    "switch_height": float(switch_height[index]),
                    "switch_tilt_deg": float(switch_tilt[index]),
                    "initial_stand_recovery_mask": float(initial_recovery_mask[index]),
                    "stand_static_mask": float(masks["stand_static"][index]),
                    "stand_recovery_mask": float(masks["stand_recovery"][index]),
                    "walk_mask": float(masks["walk"][index]),
                    "term_contributions": _candidate_term_contributions(contributions, index),
                }
            )

        return {
            "section": section,
            "stage": stage,
            "pre_command": [float(v) for v in pre_command],
            "post_command": [float(v) for v in post_command],
            "pre_steps": pre_steps,
            "steps": post_steps,
            "env_override": {
                "rel_standing_envs": env_override.get("commands", {}).get("rel_standing_envs"),
                "rel_transition_envs": env_override.get("commands", {}).get("rel_transition_envs"),
                "mode_observation": env_override.get("mode_observation"),
            },
            "rows": rows,
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _run_standing_disturbance_counterfactual(
    *,
    section: str,
    candidates: list[Candidate],
    steps: int,
    seed: int,
    lin_xy: tuple[float, float],
    yaw: float,
) -> dict[str, Any]:
    np.random.seed(seed)
    env, env_override = _create_env("standing_sanity", len(candidates))
    try:
        state = env.init_state()
        command = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        _force_command(state, command)
        _sync_envs_to_reference_state(env, state)
        _apply_reference_qvel_disturbance(env, state, lin_xy=lin_xy, yaw=yaw)
        _force_command(state, command)

        initial_base_x = np.asarray(env._backend.get_base_pos()[:, 0], dtype=np.float32).copy()
        initial_recovery_mask = _mode_masks(env, _reward_context(env, state))["stand_recovery"]
        cumulative = np.zeros((len(candidates),), dtype=np.float64)
        min_height = np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32).copy()
        max_tilt = _tilt_deg(env).astype(np.float64)
        max_left_clearance = np.zeros((len(candidates),), dtype=np.float64)
        max_right_clearance = np.zeros((len(candidates),), dtype=np.float64)
        initial_feet = _foot_metrics(env)
        min_foot_width = np.asarray(initial_feet["foot_width"], dtype=np.float32).copy()
        max_foot_width = np.asarray(initial_feet["foot_width"], dtype=np.float32).copy()
        first_terminated = np.full((len(candidates),), steps + 1, dtype=np.int32)
        cumulative_contributions: dict[str, Any] | None = None

        for step in range(1, steps + 1):
            actions = np.zeros((len(candidates), env.action_space.shape[0]), dtype=np.float32)
            for index, candidate in enumerate(candidates):
                action = candidate.action_fn(env, state.info, step)
                actions[index] = action[index]
            state = env.step(actions)
            _force_command(state, command)
            cumulative += np.asarray(state.reward, dtype=np.float64)
            step_contributions = _section_contributions(env, state)
            if cumulative_contributions is None:
                cumulative_contributions = _zero_contributions_like(step_contributions)
            _add_contributions(cumulative_contributions, step_contributions)
            height = np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32)
            min_height = np.minimum(min_height, height)
            max_tilt = np.maximum(max_tilt, _tilt_deg(env))
            feet = _foot_metrics(env)
            max_left_clearance = np.maximum(max_left_clearance, feet["left_clearance"])
            max_right_clearance = np.maximum(max_right_clearance, feet["right_clearance"])
            min_foot_width = np.minimum(min_foot_width, feet["foot_width"])
            max_foot_width = np.maximum(max_foot_width, feet["foot_width"])
            newly_done = np.asarray(state.terminated, dtype=bool) & (first_terminated > steps)
            first_terminated[newly_done] = step

        final_base_x = np.asarray(env._backend.get_base_pos()[:, 0], dtype=np.float32)
        final_height = np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32)
        base_delta = _base_feet_center_delta(env)
        feet = _foot_metrics(env)
        contributions = cumulative_contributions or _section_contributions(env, state)
        ctx = _reward_context(env, state)
        masks = _mode_masks(env, ctx)
        rows = []
        for index, candidate in enumerate(candidates):
            rows.append(
                {
                    "name": candidate.name,
                    "intent": candidate.intent,
                    "total_reward": float(cumulative[index]),
                    "standing": contributions["sections"]["standing"][index],
                    "standing_recovery": contributions["sections"]["standing_recovery"][index],
                    "walking": contributions["sections"]["walking"][index],
                    "gait_constraint": contributions["sections"]["gait_constraint"][index],
                    "term_contributions": _candidate_term_contributions(contributions, index),
                    "x_displacement": float(final_base_x[index] - initial_base_x[index]),
                    "base_feet_center_x": float(base_delta[index, 0]),
                    "base_feet_center_y": float(base_delta[index, 1]),
                    "foot_width": float(feet["foot_width"][index]),
                    "foot_width_range": float(max_foot_width[index] - min_foot_width[index]),
                    "foot_sagittal_abs": float(feet["foot_sagittal_abs"][index]),
                    "max_left_clearance": float(max_left_clearance[index]),
                    "max_right_clearance": float(max_right_clearance[index]),
                    "min_height": float(min_height[index]),
                    "final_height": float(final_height[index]),
                    "max_tilt_deg": float(max_tilt[index]),
                    "first_terminated_step": None
                    if int(first_terminated[index]) > steps
                    else int(first_terminated[index]),
                    "initial_stand_recovery_mask": float(initial_recovery_mask[index]),
                    "stand_static_mask": float(masks["stand_static"][index]),
                    "stand_recovery_mask": float(masks["stand_recovery"][index]),
                    "walk_mask": float(masks["walk"][index]),
                }
            )

        return {
            "section": section,
            "stage": "standing_sanity",
            "command": [0.0, 0.0, 0.0],
            "disturbance": {"lin_xy": [float(v) for v in lin_xy], "yaw": float(yaw)},
            "steps": steps,
            "env_override": {
                "rel_standing_envs": env_override.get("commands", {}).get("rel_standing_envs"),
                "rel_transition_envs": env_override.get("commands", {}).get("rel_transition_envs"),
                "mode_observation": env_override.get("mode_observation"),
            },
            "rows": rows,
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _row(results: dict[str, Any], name: str) -> dict[str, Any]:
    for row in results["rows"]:
        if row["name"] == name:
            return row
    raise KeyError(name)


def _preference(results: dict[str, Any], better: str, worse: str, metric: str = "total_reward") -> dict[str, Any]:
    lhs = _row(results, better)
    rhs = _row(results, worse)
    return {
        "kind": "reward_preference",
        "name": f"{better}_{metric}_gt_{worse}",
        "better": better,
        "worse": worse,
        "metric": metric,
        "better_value": float(lhs[metric]),
        "worse_value": float(rhs[metric]),
        "pass": float(lhs[metric]) > float(rhs[metric]),
    }


def _diagnostic_preference(
    results: dict[str, Any], better: str, worse: str, metric: str = "total_reward"
) -> dict[str, Any]:
    check = _preference(results, better, worse, metric=metric)
    check["kind"] = "diagnostic_preference"
    check["gating"] = False
    return check


def _term_delta(
    results: dict[str, Any],
    *,
    lhs: str,
    rhs: str,
    section: str,
    top_k: int = 8,
) -> dict[str, Any]:
    lhs_row = _row(results, lhs)
    rhs_row = _row(results, rhs)
    lhs_terms = lhs_row["term_contributions"].get(section, {})
    rhs_terms = rhs_row["term_contributions"].get(section, {})
    names = sorted(set(lhs_terms) | set(rhs_terms))
    deltas = [
        {
            "term": name,
            "lhs": float(lhs_terms.get(name, 0.0)),
            "rhs": float(rhs_terms.get(name, 0.0)),
            "delta": float(lhs_terms.get(name, 0.0) - rhs_terms.get(name, 0.0)),
        }
        for name in names
    ]
    deltas.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return {
        "kind": "term_delta",
        "name": f"{lhs}_minus_{rhs}_{section}_terms",
        "lhs": lhs,
        "rhs": rhs,
        "section": section,
        "top": deltas[:top_k],
    }


def _quality(name: str, passed: bool, *, value: float, expected: str) -> dict[str, Any]:
    return {
        "kind": "candidate_quality",
        "name": name,
        "value": float(value),
        "expected": expected,
        "pass": bool(passed),
    }


def _diagnostic_quality(
    name: str, passed: bool, *, value: float, expected: str
) -> dict[str, Any]:
    check = _quality(name, passed, value=value, expected=expected)
    check["kind"] = "diagnostic_quality"
    check["gating"] = False
    return check


def _standing_quality_checks(results: dict[str, Any]) -> list[dict[str, Any]]:
    deployable = _row(results, "standing_deployable_high")
    crouch = _row(results, "standing_low_crouch")
    rearward = _row(results, "standing_rearward_com")
    wide = _row(results, "standing_wide_feet")
    deployable_width = float(deployable["foot_width"])
    deployable_com_x = abs(float(deployable["base_feet_center_x"]))
    return [
        _quality(
            "standing_deployable_high_clean",
            deployable["first_terminated_step"] is None and float(deployable["max_tilt_deg"]) < 15.0,
            value=float(deployable["max_tilt_deg"]),
            expected="default stand should stay upright without early termination",
        ),
        _quality(
            "standing_low_crouch_lower_than_deployable",
            float(crouch["min_height"]) < float(deployable["min_height"]) - 0.02,
            value=float(crouch["min_height"] - deployable["min_height"]),
            expected="low crouch should reduce base height",
        ),
        _quality(
            "standing_rearward_com_changes_base_feet_center",
            abs(float(rearward["base_feet_center_x"])) > deployable_com_x + 0.02,
            value=float(rearward["base_feet_center_x"]),
            expected="rearward candidate should move base/feet-center x away from neutral",
        ),
        _quality(
            "standing_wide_feet_wider_than_deployable",
            float(wide["foot_width"]) > deployable_width + 0.03,
            value=float(wide["foot_width"] - deployable_width),
            expected="wide-feet candidate should increase final stance width",
        ),
    ]


def _walking_quality_checks(results: dict[str, Any]) -> list[dict[str, Any]]:
    static = _row(results, "walking_0p1_static_stand")
    forward = _row(results, "walking_0p1_forward_step")
    dragging = _row(results, "walking_0p1_dragging_feet")
    open_close = _row(results, "walking_0p1_in_place_open_close")
    return [
        _quality(
            "walking_forward_step_has_more_clearance_than_dragging",
            _swing_clearance(forward) > _swing_clearance(dragging) + 0.01,
            value=_swing_clearance(forward) - _swing_clearance(dragging),
            expected="forward-step archetype should lift feet more than dragging archetype",
        ),
        _quality(
            "walking_open_close_wider_than_static",
            float(open_close["foot_width"]) > float(static["foot_width"]) + 0.05,
            value=float(open_close["foot_width"] - static["foot_width"]),
            expected="open-close archetype should end with visibly wider feet than static standing",
        ),
        _quality(
            "walking_forward_step_moves_more_than_static",
            float(forward["x_displacement"]) > float(static["x_displacement"]) + 0.02,
            value=float(forward["x_displacement"] - static["x_displacement"]),
            expected="forward-step archetype should move forward more than static standing",
        ),
    ]


def _transition_recovery_quality_checks(results: dict[str, Any]) -> list[dict[str, Any]]:
    zero = _row(results, "transition_walk_to_stand_zero_action")
    taper = _row(results, "transition_walk_to_stand_taper_gait")
    stepping = _row(results, "transition_walk_to_stand_continue_stepping")
    return [
        _quality(
            "transition_switch_state_not_fallen",
            float(zero["switch_height"]) > 0.45 and float(zero["switch_tilt_deg"]) < 45.0,
            value=float(zero["switch_tilt_deg"]),
            expected="switch should happen before the open-loop warmup has already fallen",
        ),
        _quality(
            "transition_initial_recovery_mask_active",
            float(zero["initial_stand_recovery_mask"]) > 0.5,
            value=float(zero["initial_stand_recovery_mask"]),
            expected="walking-to-zero switch should start in standing recovery when residual motion exists",
        ),
        _diagnostic_quality(
            "transition_taper_gait_stops_clearance_before_continuing",
            _swing_clearance(taper) < _swing_clearance(stepping),
            value=_swing_clearance(stepping) - _swing_clearance(taper),
            expected="tapered recovery should reduce swing clearance relative to continuing gait",
        ),
        _diagnostic_quality(
            "transition_taper_gait_more_stable_than_instant_zero",
            float(taper["max_tilt_deg"]) < float(zero["max_tilt_deg"]),
            value=float(zero["max_tilt_deg"] - taper["max_tilt_deg"]),
            expected="tapered recovery should reduce peak tilt relative to instant zero action",
        ),
    ]


def _standing_recovery_authority_quality_checks(results: dict[str, Any]) -> list[dict[str, Any]]:
    zero = _row(results, "standing_recovery_zero_action")
    brace = _row(results, "standing_recovery_brace_action")
    feedback = _row(results, "standing_recovery_feedback_action")
    open_close = _row(results, "standing_recovery_open_close")
    stepping = _row(results, "standing_recovery_keep_stepping")
    return [
        _quality(
            "standing_recovery_mask_active",
            float(zero["stand_recovery_mask"]) > 0.5,
            value=float(zero["stand_recovery_mask"]),
            expected="disturbed zero-command state should route to Standing recovery",
        ),
        _diagnostic_quality(
            "standing_recovery_feedback_reduces_tilt_vs_zero",
            float(feedback["max_tilt_deg"]) < float(zero["max_tilt_deg"]),
            value=float(zero["max_tilt_deg"] - feedback["max_tilt_deg"]),
            expected="feedback action should reduce peak tilt if it is a useful recovery archetype",
        ),
        _diagnostic_quality(
            "standing_recovery_brace_is_not_feedback_substitute",
            float(brace["max_tilt_deg"]) >= float(feedback["max_tilt_deg"]),
            value=float(brace["max_tilt_deg"] - feedback["max_tilt_deg"]),
            expected="fixed brace should not be treated as equivalent to feedback recovery",
        ),
        _diagnostic_quality(
            "standing_recovery_open_close_not_best_stability",
            float(open_close["max_tilt_deg"])
            > min(float(zero["max_tilt_deg"]), float(feedback["max_tilt_deg"])),
            value=float(open_close["max_tilt_deg"]),
            expected="lateral open-close should not be the cleanest recovery archetype",
        ),
        _diagnostic_quality(
            "standing_recovery_keep_stepping_has_clearance",
            _swing_clearance(stepping) > _swing_clearance(zero) + 0.005,
            value=_swing_clearance(stepping) - _swing_clearance(zero),
            expected="keep-stepping archetype should remain visibly different from quiet recovery",
        ),
    ]


def _run_standing(args: argparse.Namespace) -> dict[str, Any]:
    results = _run_counterfactual(
        section="standing",
        stage="standing_sanity",
        command=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        candidates=_standing_candidates(),
        steps=args.steps,
        seed=args.seed,
    )
    results["checks"] = [
        *_standing_quality_checks(results),
        _preference(results, "standing_deployable_high", "standing_low_crouch"),
        _preference(results, "standing_deployable_high", "standing_rearward_com"),
        _preference(results, "standing_deployable_high", "standing_wide_feet"),
    ]
    return results


def _run_standing_recovery(args: argparse.Namespace) -> dict[str, Any]:
    static = _run_counterfactual(
        section="standing_static_authority",
        stage="standing_sanity",
        command=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        candidates=_standing_static_authority_candidates(),
        steps=args.steps,
        seed=args.seed,
    )
    static["checks"] = [
        _preference(static, "standing_static_zero_action", "standing_static_brace_action"),
        _preference(static, "standing_static_zero_action", "standing_static_soft_leg_oscillation"),
    ]
    static["term_deltas"] = [
        _term_delta(
            static,
            lhs="standing_static_zero_action",
            rhs="standing_static_brace_action",
            section="standing",
        ),
        _term_delta(
            static,
            lhs="standing_static_zero_action",
            rhs="standing_static_soft_leg_oscillation",
            section="standing",
        ),
    ]

    recovery = _run_standing_disturbance_counterfactual(
        section="standing_recovery_authority",
        candidates=_standing_recovery_authority_candidates(),
        steps=args.steps,
        seed=args.seed,
        lin_xy=(0.35, 0.08),
        yaw=0.25,
    )
    recovery["checks"] = [
        *_standing_recovery_authority_quality_checks(recovery),
        _diagnostic_preference(
            recovery,
            "standing_recovery_feedback_action",
            "standing_recovery_zero_action",
        ),
        _diagnostic_preference(
            recovery,
            "standing_recovery_feedback_action",
            "standing_recovery_brace_action",
        ),
        _diagnostic_preference(
            recovery,
            "standing_recovery_feedback_action",
            "standing_recovery_open_close",
        ),
        _diagnostic_preference(
            recovery,
            "standing_recovery_feedback_action",
            "standing_recovery_keep_stepping",
        ),
    ]
    recovery["term_deltas"] = [
        _term_delta(
            recovery,
            lhs="standing_recovery_feedback_action",
            rhs="standing_recovery_zero_action",
            section="standing_recovery",
        ),
        _term_delta(
            recovery,
            lhs="standing_recovery_feedback_action",
            rhs="standing_recovery_brace_action",
            section="standing_recovery",
        ),
        _term_delta(
            recovery,
            lhs="standing_recovery_feedback_action",
            rhs="standing_recovery_open_close",
            section="standing_recovery",
        ),
        _term_delta(
            recovery,
            lhs="standing_recovery_feedback_action",
            rhs="standing_recovery_keep_stepping",
            section="standing_recovery",
        ),
    ]
    return {
        "section": "standing_recovery",
        "steps": args.steps,
        "static": static,
        "recovery": recovery,
        "checks": [],
    }


def _run_transition_recovery(args: argparse.Namespace) -> dict[str, Any]:
    results = _run_switch_counterfactual(
        section="transition_recovery",
        stage="mixed_mode",
        pre_command=np.asarray([0.2, 0.0, 0.0], dtype=np.float32),
        post_command=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        pre_action=lambda env, info, step: _walking_phase_action(env, info, lift_scale=1.0),
        candidates=_transition_recovery_candidates(),
        pre_steps=max(args.steps // 8, 1),
        post_steps=args.steps,
        seed=args.seed,
    )
    results["checks"] = [
        *_transition_recovery_quality_checks(results),
        _diagnostic_preference(
            results,
            "transition_walk_to_stand_taper_gait",
            "transition_walk_to_stand_zero_action",
        ),
        _diagnostic_preference(
            results,
            "transition_walk_to_stand_taper_gait",
            "transition_walk_to_stand_continue_stepping",
        ),
        _diagnostic_preference(
            results,
            "transition_walk_to_stand_taper_gait",
            "transition_walk_to_stand_open_close",
        ),
        _diagnostic_preference(
            results,
            "transition_walk_to_stand_taper_gait",
            "transition_walk_to_stand_dragging",
        ),
    ]
    results["term_deltas"] = [
        _term_delta(
            results,
            lhs="transition_walk_to_stand_taper_gait",
            rhs="transition_walk_to_stand_zero_action",
            section="standing_recovery",
        ),
        _term_delta(
            results,
            lhs="transition_walk_to_stand_taper_gait",
            rhs="transition_walk_to_stand_continue_stepping",
            section="standing_recovery",
        ),
        _term_delta(
            results,
            lhs="transition_walk_to_stand_taper_gait",
            rhs="transition_walk_to_stand_open_close",
            section="standing_recovery",
        ),
        _term_delta(
            results,
            lhs="transition_walk_to_stand_taper_gait",
            rhs="transition_walk_to_stand_dragging",
            section="standing_recovery",
        ),
    ]
    return results


def _run_walking_0p1(args: argparse.Namespace) -> dict[str, Any]:
    results = _run_counterfactual(
        section="walking_0p1",
        stage="walking_sanity",
        command=np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
        candidates=_walking_candidates(),
        steps=args.steps,
        seed=args.seed,
    )
    results["checks"] = [
        *_walking_quality_checks(results),
        _preference(results, "walking_0p1_forward_step", "walking_0p1_static_stand"),
        _preference(results, "walking_0p1_forward_step", "walking_0p1_in_place_open_close"),
        _preference(results, "walking_0p1_forward_step", "walking_0p1_dragging_feet"),
        _diagnostic_preference(
            results,
            "walking_0p1_forward_step",
            "walking_0p1_in_place_open_close",
            metric="x_displacement",
        ),
    ]
    results["term_deltas"] = [
        _term_delta(
            results,
            lhs="walking_0p1_forward_step",
            rhs="walking_0p1_static_stand",
            section="walking",
        ),
        _term_delta(
            results,
            lhs="walking_0p1_forward_step",
            rhs="walking_0p1_in_place_open_close",
            section="walking",
        ),
        _term_delta(
            results,
            lhs="walking_0p1_forward_step",
            rhs="walking_0p1_dragging_feet",
            section="walking",
        ),
        _term_delta(
            results,
            lhs="walking_0p1_forward_step",
            rhs="walking_0p1_in_place_open_close",
            section="gait_constraint",
            top_k=3,
        ),
        _term_delta(
            results,
            lhs="walking_0p1_forward_step",
            rhs="walking_0p1_dragging_feet",
            section="gait_constraint",
            top_k=3,
        ),
    ]
    return results


def _run_isolation(args: argparse.Namespace) -> dict[str, Any]:
    standing = _run_counterfactual(
        section="isolation_standing",
        stage="standing_sanity",
        command=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        candidates=[
            Candidate(
                "standing_zero_command",
                lambda env, info, step: _zero_action(env),
                "stand-command sample for routing isolation",
            )
        ],
        steps=max(args.steps // 4, 1),
        seed=args.seed,
    )
    walking = _run_counterfactual(
        section="isolation_walking",
        stage="walking_sanity",
        command=np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
        candidates=[
            Candidate(
                "walking_0p1_command",
                lambda env, info, step: _walking_phase_action(env, info, lift_scale=1.0),
                "walk-command sample for routing isolation",
            )
        ],
        steps=max(args.steps // 4, 1),
        seed=args.seed,
    )
    stand_row = standing["rows"][0]
    walk_row = walking["rows"][0]
    checks = [
        {
            "name": "standing_sample_walk_mask_zero",
            "value": stand_row["walk_mask"],
            "pass": abs(float(stand_row["walk_mask"])) < 1.0e-6,
        },
        {
            "name": "walking_sample_walk_mask_one",
            "value": walk_row["walk_mask"],
            "pass": abs(float(walk_row["walk_mask"]) - 1.0) < 1.0e-6,
        },
        {
            "name": "standing_sample_walking_subtotal_zero",
            "value": stand_row["walking"],
            "pass": abs(float(stand_row["walking"])) < 1.0e-6,
        },
        {
            "name": "walking_sample_standing_subtotal_zero",
            "value": walk_row["standing"] + walk_row["standing_recovery"],
            "pass": abs(float(walk_row["standing"] + walk_row["standing_recovery"])) < 1.0e-6,
        },
    ]
    return {
        "section": "isolation",
        "steps": max(args.steps // 4, 1),
        "standing": standing,
        "walking": walking,
        "checks": checks,
    }


def _child_sections(section: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        value
        for value in section.values()
        if isinstance(value, dict) and isinstance(value.get("section"), str)
    ]


def _print_section_summary(section: dict[str, Any], *, indent: str = "") -> None:
    print(f"{indent}section={section['section']}")
    rows = section.get("rows", [])
    for row in rows:
        print(
            indent
            + "  "
            + f"{row['name']}: total={row['total_reward']:.3f}, "
            f"standing={row['standing']:.3f}, recovery={row['standing_recovery']:.3f}, "
            f"walking={row['walking']:.3f}, gait={row['gait_constraint']:.3f}, "
            f"x={row['x_displacement']:.3f}, tilt={row['max_tilt_deg']:.2f}, "
            f"height={row['min_height']:.3f}->{row['final_height']:.3f}, "
            f"width={row['foot_width']:.3f}, width_range={row['foot_width_range']:.3f}, "
            f"clearance=({row['max_left_clearance']:.3f},{row['max_right_clearance']:.3f})"
        )
        print(f"{indent}    intent={row['intent']}")
    for check in section.get("checks", []):
        status = "PASS" if check["pass"] else "FAIL"
        print(f"{indent}  [{status}] {check}")
    for delta in section.get("term_deltas", []):
        print(
            f"{indent}  term_delta {delta['name']} "
            f"(negative means {delta['rhs']} is favored by that term)"
        )
        for item in delta["top"]:
            print(
                indent
                + "    "
                + f"{item['term']}: lhs={item['lhs']:.6f}, "
                f"rhs={item['rhs']:.6f}, delta={item['delta']:.6f}"
            )
    for child in _child_sections(section):
        _print_section_summary(child, indent=indent + "  ")


def _iter_checks(section: dict[str, Any]):
    yield from section.get("checks", [])
    for child in _child_sections(section):
        yield from _iter_checks(child)


def _print_summary(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nReward Lab summary")
    for section in report["sections"]:
        _print_section_summary(section)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        action="append",
        choices=(
            "all",
            "standing",
            "standing_recovery",
            "walking_0p1",
            "transition_recovery",
            "isolation",
        ),
        default=None,
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    requested = args.section or ["all"]
    sections = (
        ["standing", "walking_0p1", "isolation"]
        if "all" in requested
        else requested
    )
    runners = {
        "standing": _run_standing,
        "standing_recovery": _run_standing_recovery,
        "walking_0p1": _run_walking_0p1,
        "transition_recovery": _run_transition_recovery,
        "isolation": _run_isolation,
    }
    report = {"steps": args.steps, "seed": args.seed, "sections": [runners[name](args) for name in sections]}
    _print_summary(report)
    failed = [
        check
        for section in report["sections"]
        for check in _iter_checks(section)
        if not bool(check["pass"])
        and bool(check.get("gating", True))
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
