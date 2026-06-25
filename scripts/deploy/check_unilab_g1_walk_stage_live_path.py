#!/usr/bin/env python3
"""Live-path sentinel for G1 standing/walking stage configs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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


@dataclass(frozen=True)
class StageExpectation:
    stand: str
    reset_qvel: float
    curriculum_enabled: bool


@dataclass
class Check:
    level: str
    name: str
    detail: str


STAGES: dict[str, StageExpectation] = {
    "standing_sanity": StageExpectation("all", 0.0, False),
    "walking_sanity": StageExpectation("none", 0.5, True),
    "mixed_mode": StageExpectation("mixed", 0.5, True),
}


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level, name, detail))


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _compose_stage(stage: str):
    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                "task=sac/g1_walk_flat/mujoco",
                f"+g1_walk_stage={stage}",
            ],
        )


def _mean(value: np.ndarray) -> float:
    return float(np.mean(np.asarray(value, dtype=np.float64)))


def _max_abs(value: np.ndarray) -> float:
    arr = np.asarray(value, dtype=np.float64)
    return float(np.max(np.abs(arr))) if arr.size else 0.0


def _check_stand_distribution(
    checks: list[Check],
    *,
    stage: str,
    expected: StageExpectation,
    gait_enabled: np.ndarray,
    mode_signal: np.ndarray,
) -> None:
    stand_mask = gait_enabled <= 0.5
    stand_frac = float(np.mean(stand_mask))
    if expected.stand == "all":
        if np.all(stand_mask) and _max_abs(mode_signal) == 0.0:
            _add(checks, "PASS", f"{stage}: standing mode", f"stand_frac={stand_frac:.3f}")
        else:
            _add(checks, "FAIL", f"{stage}: standing mode", f"stand_frac={stand_frac:.3f}")
    elif expected.stand == "none":
        if not np.any(stand_mask) and np.all(mode_signal > 0.5):
            _add(checks, "PASS", f"{stage}: walking mode", f"stand_frac={stand_frac:.3f}")
        else:
            _add(checks, "FAIL", f"{stage}: walking mode", f"stand_frac={stand_frac:.3f}")
    else:
        if np.any(stand_mask) and np.any(~stand_mask):
            _add(checks, "PASS", f"{stage}: mixed mode", f"stand_frac={stand_frac:.3f}")
        else:
            _add(checks, "FAIL", f"{stage}: mixed mode", f"stand_frac={stand_frac:.3f}")


def audit_stage(stage: str, *, num_envs: int, seed: int, steps: int) -> tuple[list[Check], dict[str, Any]]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose one of {sorted(STAGES)}")

    expected = STAGES[stage]
    np.random.seed(seed)
    cfg = _compose_stage(stage)
    assert_offpolicy_task_choice_matches_algo(cfg, algo_name="sac")
    adapter = BackendAdapter(cfg, root_dir=ROOT_DIR, algo_name="sac")
    env_override = adapter.build_task_env_cfg_override()

    checks: list[Check] = []
    details: dict[str, Any] = {
        "stage": stage,
        "num_envs": num_envs,
        "seed": seed,
        "steps": steps,
        "env_override": {
            "mode_observation": env_override.get("mode_observation"),
            "stand_action_authority": env_override.get("stand_action_authority"),
            "reset_base_qvel_limit": env_override.get("reset_base_qvel_limit"),
            "standing_reset_base_qvel_limit": env_override.get("standing_reset_base_qvel_limit"),
            "rel_standing_envs": env_override.get("commands", {}).get("rel_standing_envs"),
            "curriculum_enabled": env_override.get("curriculum", {}).get("enabled"),
        },
    }

    env = None
    try:
        ensure_registries()
        env = create_env(
            cfg,
            num_envs=num_envs,
            env_cfg_override=env_override,
            sim_backend="mujoco",
            task_name="G1WalkFlat",
        )
        state = env.init_state()
        if "steps" in state.info:
            state.info["steps"].fill(0)

        actions = np.ones(env.action_space.shape, dtype=np.float32)
        actions = np.tile(actions[None, :], (num_envs, 1))
        for _ in range(steps):
            state = env.step(actions)

        obs = state.obs["obs"]
        critic = state.obs["critic"]
        gait_enabled = np.asarray(state.info["gait_enabled"], dtype=np.float32)
        mode_signal = np.asarray(obs[:, -1], dtype=np.float32)
        log = state.info.get("log", {})

        details.update(
            {
                "obs_shape": list(obs.shape),
                "critic_shape": list(critic.shape),
                "mode_signal_mean": _mean(mode_signal),
                "gait_enabled_mean": _mean(gait_enabled),
                "commands_max_abs": _max_abs(state.info["commands"]),
                "log": {key: float(value) for key, value in log.items() if isinstance(value, int | float)},
            }
        )

        if env.obs_groups_spec == {"obs": 99, "critic": 102} and obs.shape[1] == 99 and critic.shape[1] == 102:
            _add(checks, "PASS", f"{stage}: obs contract", str(env.obs_groups_spec))
        else:
            _add(
                checks,
                "FAIL",
                f"{stage}: obs contract",
                f"spec={env.obs_groups_spec}, obs={obs.shape}, critic={critic.shape}",
            )

        if env_override.get("stand_action_authority") is False:
            _add(checks, "PASS", f"{stage}: action authority ablation", "stand_action_authority=false")
        else:
            _add(checks, "FAIL", f"{stage}: action authority ablation", "expected false")

        if env_override.get("reset_base_qvel_limit") == expected.reset_qvel:
            _add(checks, "PASS", f"{stage}: reset qvel stage", str(expected.reset_qvel))
        else:
            _add(
                checks,
                "FAIL",
                f"{stage}: reset qvel stage",
                f"got {env_override.get('reset_base_qvel_limit')}, expected {expected.reset_qvel}",
            )

        if env_override.get("curriculum", {}).get("enabled") is expected.curriculum_enabled:
            _add(
                checks,
                "PASS",
                f"{stage}: curriculum stage",
                f"enabled={expected.curriculum_enabled}",
            )
        else:
            _add(
                checks,
                "FAIL",
                f"{stage}: curriculum stage",
                f"got {env_override.get('curriculum', {}).get('enabled')}",
            )

        _check_stand_distribution(
            checks,
            stage=stage,
            expected=expected,
            gait_enabled=gait_enabled,
            mode_signal=mode_signal,
        )

        for key in ("reward/mode_stand_frac", "reward/mode_walk_frac", "reward/stand_raw_action_l1"):
            if key in log:
                _add(checks, "PASS", f"{stage}: log {key}", f"{float(log[key]):.6f}")
            else:
                _add(checks, "FAIL", f"{stage}: log {key}", "missing")

        if expected.stand != "none":
            raw = float(log.get("reward/stand_raw_action_l1", -1.0))
            executed = float(log.get("reward/stand_executed_action_l1", -1.0))
            if raw > 0.0 and abs(raw - executed) < 1e-6:
                _add(
                    checks,
                    "PASS",
                    f"{stage}: ungated standing action diagnostics",
                    f"raw={raw:.6f}, executed={executed:.6f}",
                )
            else:
                _add(
                    checks,
                    "FAIL",
                    f"{stage}: ungated standing action diagnostics",
                    f"raw={raw:.6f}, executed={executed:.6f}",
                )
    finally:
        if env is not None:
            _close_env(env)

    return checks, details


def print_report(all_checks: list[Check], details: dict[str, Any]) -> None:
    print("UniLab G1 stage live-path sentinel")
    print(f"details: {details}")
    for check in all_checks:
        print(f"[{check.level}] {check.name}: {check.detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        action="append",
        choices=sorted(STAGES),
        help="Stage to audit. Defaults to all stages.",
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stages = args.stage or list(STAGES)
    all_checks: list[Check] = []
    details: dict[str, Any] = {}
    for index, stage in enumerate(stages):
        checks, stage_details = audit_stage(
            stage,
            num_envs=args.num_envs,
            seed=args.seed + index,
            steps=args.steps,
        )
        all_checks.extend(checks)
        details[stage] = stage_details
    print_report(all_checks, details)
    return 1 if any(check.level == "FAIL" for check in all_checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
