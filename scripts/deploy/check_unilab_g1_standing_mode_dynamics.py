#!/usr/bin/env python3
"""Numerical MuJoCo standing-mode dynamics check for G1.

This is intentionally stronger than a reward helper test. It creates the real
G1WalkFlat MuJoCo env through the training owner config, forces the
``standing_sanity`` stage, runs with autoreset disabled, and checks whether the
robot remains upright while the standing reward path is active.

The ``reward-search`` mode is the strongest local diagnostic in this file: it
parallel-rolls out deterministic standing residual-action candidates and asks
whether the action with the highest standing reward is also dynamically able to
stand better than the zero-action baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def _compose_cfg():
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                "task=sac/g1_walk_flat/mujoco",
                "+g1_walk_stage=standing_sanity",
            ],
        )


def _stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _height(env: Any) -> np.ndarray:
    return np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32)


def _tilt_deg(env: Any) -> np.ndarray:
    gravity = np.asarray(env._backend.get_sensor_data(env.cfg.sensor.upvector), dtype=np.float32)
    return np.rad2deg(np.arccos(np.clip(gravity[:, 2], -1.0, 1.0))).astype(np.float32)


def _create_env(cfg: Any, env_override: dict[str, Any], num_envs: int):
    return create_env(
        cfg,
        num_envs=num_envs,
        env_cfg_override=env_override,
        sim_backend="mujoco",
        task_name="G1WalkFlat",
    )


def _create_policy_session(cfg: Any, env_override: dict[str, Any], *, num_envs: int, device: str):
    from unilab.visualization.interactive_playback import (
        RslRlPlaybackConfig,
        create_sac_playback_session,
    )

    playback_cfg = RslRlPlaybackConfig(
        task="G1WalkFlat",
        load_run=str(cfg.algo.load_run),
        checkpoint=None,
        action_mode="policy",
        policy_obs_mode="actor",
        algo_log_name=str(cfg.algo.algo_log_name),
        log_root=None,
        num_envs=num_envs,
    )
    return create_sac_playback_session(
        playback_cfg=playback_cfg,
        cfg=cfg,
        env_factory=lambda n: _create_env(cfg, env_override, n),
        root_dir=ROOT_DIR,
        device=device,
        algo_name="sac",
        log=lambda message: print(f"[standing_dynamics] {message}"),
    )


def _joint_action_indices(env: Any, names: list[str]) -> dict[str, int]:
    indices = env._backend.get_joint_dof_pos_indices(names)
    return {name: int(index) for name, index in zip(names, indices, strict=True)}


def _build_standing_reward_search_actions(
    env: Any, *, num_envs: int, seed: int
) -> tuple[np.ndarray, list[str]]:
    action_dim = int(env.action_space.shape[0])
    actions: list[np.ndarray] = []
    labels: list[str] = []

    def add(label: str, updates: dict[str, float]) -> None:
        action = np.zeros((action_dim,), dtype=np.float32)
        for name, value in updates.items():
            action[joint_indices[name]] = float(value)
        actions.append(action)
        labels.append(label)

    leg_names = [
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
    ]
    joint_indices = _joint_action_indices(env, leg_names)

    add("zero", {})
    for hip in (-0.30, -0.15, 0.0, 0.15, 0.30):
        for knee in (-0.20, 0.0, 0.20, 0.40):
            for ankle in (-0.30, -0.15, 0.0, 0.15, 0.30):
                add(
                    f"symmetric_pitch hip={hip:+.2f} knee={knee:+.2f} ankle={ankle:+.2f}",
                    {
                        "left_hip_pitch_joint": hip,
                        "right_hip_pitch_joint": hip,
                        "left_knee_joint": knee,
                        "right_knee_joint": knee,
                        "left_ankle_pitch_joint": ankle,
                        "right_ankle_pitch_joint": ankle,
                    },
                )

    rng = np.random.default_rng(seed)
    while len(actions) < num_envs:
        values = rng.uniform(-0.35, 0.35, size=(6,)).astype(np.float32)
        add(
            "random_leg_pitch",
            {
                "left_hip_pitch_joint": float(values[0]),
                "left_knee_joint": float(values[1]),
                "left_ankle_pitch_joint": float(values[2]),
                "right_hip_pitch_joint": float(values[3]),
                "right_knee_joint": float(values[4]),
                "right_ankle_pitch_joint": float(values[5]),
            },
        )

    return np.stack(actions[:num_envs], axis=0).astype(np.float32), labels[:num_envs]


def _first_step_array(values: np.ndarray, steps: int) -> list[int | None]:
    return [None if int(value) > steps else int(value) for value in values]


def run_check(
    *, num_envs: int, steps: int, seed: int, action_mode: str, device: str
) -> tuple[list[str], dict[str, Any]]:
    np.random.seed(seed)
    cfg = _compose_cfg()
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name="sac")
    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="sac")
    env_override = adapter.build_task_env_cfg_override()

    ensure_registries()
    session = None
    if action_mode == "policy":
        session_tuple = _create_policy_session(cfg, env_override, num_envs=num_envs, device=device)
        session = session_tuple[0]
        env = session.env
        checkpoint_path = session_tuple[2]
    else:
        env = _create_env(cfg, env_override, num_envs)
        checkpoint_path = None
    failures: list[str] = []
    details: dict[str, Any] = {
        "num_envs": num_envs,
        "steps": steps,
        "seed": seed,
        "action_mode": action_mode,
        "checkpoint": checkpoint_path,
        "env_override": {
            "mode_observation": env_override.get("mode_observation"),
            "stand_action_authority": env_override.get("stand_action_authority"),
            "rel_standing_envs": env_override.get("commands", {}).get("rel_standing_envs"),
            "standing_reset_base_qvel_limit": env_override.get("standing_reset_base_qvel_limit"),
            "reward_base_height_scale": env_override.get("reward_config", {})
            .get("scales", {})
            .get("base_height"),
            "balance_common_terms": env_override.get("reward_config", {})
            .get("mode", {})
            .get("balance_common_terms"),
            "stand_terms": env_override.get("reward_config", {}).get("mode", {}).get("stand_terms"),
        },
    }
    try:
        env.set_autoreset(False)
        state = env.init_state() if session is None else None
        if session is not None:
            session.reset()
            state = env.state
        if state is None:
            raise RuntimeError("standing dynamics check could not initialize env state")
        action_dim = int(env.action_space.shape[0])
        if action_mode == "reward-search":
            actions, action_labels = _build_standing_reward_search_actions(
                env, num_envs=num_envs, seed=seed
            )
        else:
            actions = np.zeros((num_envs, action_dim), dtype=np.float32)
            action_labels = ["zero"] * num_envs

        initial_obs = np.asarray(state.obs["obs"], dtype=np.float32)
        initial_mode = initial_obs[:, -1]
        initial_commands = np.asarray(state.info["commands"], dtype=np.float32)
        initial_gait_enabled = np.asarray(state.info["gait_enabled"], dtype=np.float32)
        details["initial"] = {
            "height": _stats(_height(env)),
            "tilt_deg": _stats(_tilt_deg(env)),
            "commands_max_abs": float(np.max(np.abs(initial_commands))),
            "mode_signal": _stats(initial_mode),
            "gait_enabled": _stats(initial_gait_enabled),
        }

        if np.max(np.abs(initial_commands)) > 1.0e-7:
            failures.append("standing_sanity did not reset with zero commands")
        if np.max(np.abs(initial_gait_enabled)) > 1.0e-7 or np.max(np.abs(initial_mode)) > 1.0e-7:
            failures.append("standing_sanity did not enter STAND mode")

        min_height = float(np.min(_height(env)))
        max_tilt = float(np.max(_tilt_deg(env)))
        per_env_min_height = _height(env).astype(np.float64)
        per_env_max_tilt = _tilt_deg(env).astype(np.float64)
        per_env_cumulative_reward = np.zeros((num_envs,), dtype=np.float64)
        per_env_first_terminated = np.full((num_envs,), steps + 1, dtype=np.int32)
        per_env_first_low_height = np.full((num_envs,), steps + 1, dtype=np.int32)
        per_env_first_large_tilt = np.full((num_envs,), steps + 1, dtype=np.int32)
        first_terminated_step: int | None = None
        first_low_height_step: int | None = None
        first_large_tilt_step: int | None = None
        reward_base_height_values: list[float] = []
        reward_stand_total_values: list[float] = []

        for step in range(1, steps + 1):
            if session is None:
                state = env.step(actions)
            else:
                session.step_once()
                state = env.state
                if state is None:
                    raise RuntimeError("policy playback step did not leave env.state available")
            heights = _height(env)
            tilts = _tilt_deg(env)
            min_height = min(min_height, float(np.min(heights)))
            max_tilt = max(max_tilt, float(np.max(tilts)))
            per_env_min_height = np.minimum(per_env_min_height, heights)
            per_env_max_tilt = np.maximum(per_env_max_tilt, tilts)
            per_env_cumulative_reward += np.asarray(state.reward, dtype=np.float64)
            log = state.info.get("log", {})
            if "reward/base_height" in log:
                reward_base_height_values.append(float(log["reward/base_height"]))
            if "reward/stand_total" in log:
                reward_stand_total_values.append(float(log["reward/stand_total"]))
            if first_low_height_step is None and np.any(
                heights < float(env.cfg.reward_config.min_base_height)
            ):
                first_low_height_step = step
            newly_low = (heights < float(env.cfg.reward_config.min_base_height)) & (
                per_env_first_low_height > steps
            )
            per_env_first_low_height[newly_low] = step
            if first_large_tilt_step is None and np.any(
                tilts > float(env.cfg.reward_config.max_tilt_deg)
            ):
                first_large_tilt_step = step
            newly_tilted = (tilts > float(env.cfg.reward_config.max_tilt_deg)) & (
                per_env_first_large_tilt > steps
            )
            per_env_first_large_tilt[newly_tilted] = step
            if first_terminated_step is None and np.any(state.terminated):
                first_terminated_step = step
            newly_terminated = np.asarray(state.terminated, dtype=bool) & (
                per_env_first_terminated > steps
            )
            per_env_first_terminated[newly_terminated] = step
            if first_terminated_step is not None and action_mode != "reward-search":
                break

        details["rollout"] = {
            "completed_steps": step,
            "min_height": min_height,
            "max_tilt_deg": max_tilt,
            "first_terminated_step": first_terminated_step,
            "first_low_height_step": first_low_height_step,
            "first_large_tilt_step": first_large_tilt_step,
            "final_height": _stats(_height(env)),
            "final_tilt_deg": _stats(_tilt_deg(env)),
            "reward_base_height_min": (
                float(np.min(reward_base_height_values)) if reward_base_height_values else None
            ),
            "reward_stand_total_min": (
                float(np.min(reward_stand_total_values)) if reward_stand_total_values else None
            ),
        }

        if action_mode == "reward-search":
            zero_idx = 0
            best_idx = int(np.argmax(per_env_cumulative_reward))
            first_terminated = _first_step_array(per_env_first_terminated, steps)
            best_summary = {
                "index": best_idx,
                "label": action_labels[best_idx],
                "cumulative_reward": float(per_env_cumulative_reward[best_idx]),
                "first_terminated_step": first_terminated[best_idx],
                "min_height": float(per_env_min_height[best_idx]),
                "max_tilt_deg": float(per_env_max_tilt[best_idx]),
            }
            zero_summary = {
                "index": zero_idx,
                "label": action_labels[zero_idx],
                "cumulative_reward": float(per_env_cumulative_reward[zero_idx]),
                "first_terminated_step": first_terminated[zero_idx],
                "min_height": float(per_env_min_height[zero_idx]),
                "max_tilt_deg": float(per_env_max_tilt[zero_idx]),
            }
            details["reward_search"] = {
                "zero": zero_summary,
                "best_by_cumulative_reward": best_summary,
                "top5_by_cumulative_reward": [
                    {
                        "index": int(idx),
                        "label": action_labels[int(idx)],
                        "cumulative_reward": float(per_env_cumulative_reward[int(idx)]),
                        "first_terminated_step": first_terminated[int(idx)],
                        "min_height": float(per_env_min_height[int(idx)]),
                        "max_tilt_deg": float(per_env_max_tilt[int(idx)]),
                    }
                    for idx in np.argsort(per_env_cumulative_reward)[-5:][::-1]
                ],
            }
            zero_survival = (
                steps + 1
                if zero_summary["first_terminated_step"] is None
                else int(zero_summary["first_terminated_step"])
            )
            best_survival = (
                steps + 1
                if best_summary["first_terminated_step"] is None
                else int(best_summary["first_terminated_step"])
            )
            if best_idx == zero_idx:
                failures.append("standing reward search still prefers zero action")
            if best_survival <= zero_survival:
                failures.append(
                    "standing reward search did not find an action that survives longer than zero"
                )
            if best_summary["first_terminated_step"] is not None:
                failures.append(
                    "standing reward search did not find a candidate that stands for the full horizon"
                )
        elif first_terminated_step is not None:
            failures.append(f"{action_mode} standing terminated at step {first_terminated_step}")
        if action_mode != "reward-search":
            if min_height < float(env.cfg.reward_config.min_base_height):
                failures.append(
                    f"base height dropped below min_base_height: {min_height:.4f} "
                    f"< {float(env.cfg.reward_config.min_base_height):.4f}"
                )
            if max_tilt > float(env.cfg.reward_config.max_tilt_deg):
                failures.append(
                    f"tilt exceeded max_tilt_deg: {max_tilt:.2f} "
                    f"> {float(env.cfg.reward_config.max_tilt_deg):.2f}"
                )
            if reward_base_height_values and min(reward_base_height_values) >= -1.0e-6:
                failures.append("base_height reward did not become negative during standing rollout")
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    return failures, details


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--action-mode", choices=("zero", "policy", "reward-search"), default="zero"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    failures, details = run_check(
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
        action_mode=args.action_mode,
        device=args.device,
    )
    print("G1 standing-mode dynamics check")
    print(json.dumps(details, indent=2, sort_keys=True))
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    if args.action_mode == "reward-search":
        print("[PASS] reward-search found a reward-preferred candidate that stands better than zero")
    else:
        print(f"[PASS] {args.action_mode} STAND mode stayed upright and standing reward path was active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
