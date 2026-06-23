#!/usr/bin/env python3
"""Section 3 sensor-source checker for UniLab -> RoboJudo_Real.

The original Section 3 checker validated observation order and first-frame
values. Section 9 showed the official UniLab env is stable while the RoboJudo
deployment still falls, so the remaining high-risk boundary is observation
source, not observation order.

This checker verifies that UniLab G1WalkFlat actor obs uses torso IMU sensors:

    torso_gyro * 0.25
    -torso_upvector

and compares ONNX actions when those first 6 obs entries are replaced by
pelvis/base-like sensors.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from unilab.base import registry
from unilab.base.observations import split_obs_dict
from unilab.base.registry import ensure_registries
from unilab.training.seed import apply_training_seed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOJUDO_ROOT = REPO_ROOT.parent / "RoboJudo_Real"
DEFAULT_RUN_DIR = (
    DEFAULT_ROBOJUDO_ROOT
    / "assets/models/g1/unilab/g1_walk_flat/2026-06-12_15-46-01_mujoco"
)
DEFAULT_ONNX = DEFAULT_ROBOJUDO_ROOT / "assets/models/g1/unilab/g1_walk_flat/policy.onnx"
DEFAULT_ROBOJUDO_POLICY = DEFAULT_ROBOJUDO_ROOT / "robojudo/policy/unilab_policy.py"
DEFAULT_ROBOJUDO_BASE_ENV = DEFAULT_ROBOJUDO_ROOT / "robojudo/environment/base_env.py"

EXPECTED_OBS_DIM = 98
TOL = 1.0e-5


@dataclass
class Check:
    level: str
    name: str
    detail: str


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level, name, detail))


def stats(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "mean": float(arr.mean()) if arr.size else None,
        "std": float(arr.std()) if arr.size else None,
        "max_abs": float(np.max(np.abs(arr))) if arr.size else None,
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_env_override(run_cfg: dict[str, Any]) -> dict[str, Any]:
    config = run_cfg["config"]
    env_override = dict(config.get("env", {}))
    env_override["reward_config"] = dict(config["reward"])
    return env_override


def run_onnx(session: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    fixed_batch = input_shape[0] if input_shape and isinstance(input_shape[0], int) else None
    obs = np.asarray(obs, dtype=np.float32)
    if fixed_batch in (None, obs.shape[0]):
        return np.asarray(session.run([output_name], {input_name: obs})[0], dtype=np.float32)
    if fixed_batch == 1:
        return np.concatenate(
            [
                np.asarray(session.run([output_name], {input_name: obs[i : i + 1]})[0], dtype=np.float32)
                for i in range(obs.shape[0])
            ],
            axis=0,
        )
    raise ValueError(f"ONNX fixed batch {fixed_batch} incompatible with obs batch {obs.shape[0]}")


def inspect_robojudo_unilab_policy(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    source = ast.unparse(tree)
    return {
        "uses_torso_ang_vel": "torso_ang_vel" in source,
        "uses_torso_quat": "torso_quat" in source,
        "fallbacks_to_base": "base_ang_vel" in source and "base_quat" in source,
    }


def inspect_robojudo_base_env(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    return {
        "fk_passes_joint_vel": "joint_vel=self.dof_vel" in source,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--robojudo-policy", type=Path, default=DEFAULT_ROBOJUDO_POLICY)
    parser.add_argument("--robojudo-base-env", type=Path, default=DEFAULT_ROBOJUDO_BASE_ENV)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--probe-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    run_config_path = args.run_dir.resolve() / "run_config.json"
    onnx_path = args.onnx.resolve()
    robojudo_policy_path = args.robojudo_policy.resolve()
    robojudo_base_env_path = args.robojudo_base_env.resolve()
    checks: list[Check] = []
    for name, path in [
        ("run_config_exists", run_config_path),
        ("onnx_exists", onnx_path),
        ("robojudo_policy_exists", robojudo_policy_path),
        ("robojudo_base_env_exists", robojudo_base_env_path),
    ]:
        _add(checks, "PASS" if path.is_file() else "FAIL", name, path.as_posix())
    if any(c.level == "FAIL" for c in checks):
        for check in checks:
            print(f"[{check.level}] {check.name}: {check.detail}")
        return 1

    run_cfg = load_json(run_config_path)
    env_override = build_env_override(run_cfg)
    task_name = str(run_cfg["config"]["training"]["task_name"])
    sim_backend = str(run_cfg["config"]["training"]["sim_backend"])

    ensure_registries()
    apply_training_seed(args.seed, torch_runtime=False, cuda=False)
    env = registry.make(
        task_name,
        num_envs=args.num_envs,
        sim_backend=sim_backend,
        env_cfg_override=env_override,
    )
    try:
        state = env.init_state()
        obs, _ = split_obs_dict(state.obs)
        obs = np.asarray(obs, dtype=np.float32)
        session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])

        for _ in range(max(args.probe_steps, 0)):
            actions = run_onnx(session, obs)
            state = env.step(actions)
            obs, _ = split_obs_dict(state.obs)
            obs = np.asarray(obs, dtype=np.float32)

        torso_gyro = np.asarray(env._backend.get_sensor_data("torso_gyro"), dtype=np.float32)
        torso_upvector = np.asarray(env._backend.get_sensor_data("torso_upvector"), dtype=np.float32)
        pelvis_gyro = np.asarray(env._backend.get_sensor_data("pelvis_gyro"), dtype=np.float32)
        pelvis_upvector = np.asarray(env._backend.get_sensor_data("pelvis_upvector"), dtype=np.float32)

        torso_obs = obs.copy()
        pelvis_obs = obs.copy()
        pelvis_obs[:, 0:3] = pelvis_gyro * 0.25
        pelvis_obs[:, 3:6] = -pelvis_upvector

        torso_action = run_onnx(session, torso_obs)
        pelvis_action = run_onnx(session, pelvis_obs)
        action_diff = pelvis_action - torso_action

        gyro_match = float(np.max(np.abs(obs[:, 0:3] - torso_gyro * 0.25)))
        gravity_match = float(np.max(np.abs(obs[:, 3:6] + torso_upvector)))
        pelvis_gyro_gap = float(np.max(np.abs(torso_gyro - pelvis_gyro)))
        pelvis_upvector_gap = float(np.max(np.abs(torso_upvector - pelvis_upvector)))
        action_gap = float(np.max(np.abs(action_diff)))

        source_info = inspect_robojudo_unilab_policy(robojudo_policy_path)
        base_env_info = inspect_robojudo_base_env(robojudo_base_env_path)

        print("== Section 3: Sensor Source ==")
        print(f"run_config: {run_config_path}")
        print(f"onnx: {onnx_path}")
        print(f"robojudo_policy: {robojudo_policy_path}")
        print(f"robojudo_base_env: {robojudo_base_env_path}")
        print(f"probe_steps: {args.probe_steps}")
        print(f"obs_stats: {stats(obs)}")
        print(f"torso_gyro_stats: {stats(torso_gyro)}")
        print(f"pelvis_gyro_stats: {stats(pelvis_gyro)}")
        print(f"torso_upvector_stats: {stats(torso_upvector)}")
        print(f"pelvis_upvector_stats: {stats(pelvis_upvector)}")
        print(f"official_obs_vs_torso_gyro_scaled_max_abs: {gyro_match:.8f}")
        print(f"official_obs_vs_minus_torso_upvector_max_abs: {gravity_match:.8f}")
        print(f"torso_vs_pelvis_gyro_max_abs: {pelvis_gyro_gap:.8f}")
        print(f"torso_vs_pelvis_upvector_max_abs: {pelvis_upvector_gap:.8f}")
        print(f"torso_action_stats: {stats(torso_action)}")
        print(f"pelvis_replaced_action_stats: {stats(pelvis_action)}")
        print(f"pelvis_replaced_minus_torso_action_stats: {stats(action_diff)}")
        print(f"robojudo_source_info: {source_info}")
        print(f"robojudo_base_env_info: {base_env_info}")
        print()

        _add(
            checks,
            "PASS" if obs.shape == (args.num_envs, EXPECTED_OBS_DIM) else "FAIL",
            "official_obs_dim",
            str(obs.shape),
        )
        _add(
            checks,
            "PASS" if gyro_match <= TOL else "FAIL",
            "official_obs_uses_torso_gyro",
            f"max_abs={gyro_match:.3e}",
        )
        _add(
            checks,
            "PASS" if gravity_match <= TOL else "FAIL",
            "official_obs_uses_minus_torso_upvector",
            f"max_abs={gravity_match:.3e}",
        )
        _add(
            checks,
            "WARN" if action_gap > 0.05 else "PASS",
            "pelvis_replacement_changes_action",
            f"max_abs={action_gap:.3f}",
        )
        _add(
            checks,
            "PASS" if source_info["uses_torso_ang_vel"] and source_info["uses_torso_quat"] else "FAIL",
            "robojudo_unilab_policy_uses_torso_source",
            str(source_info),
        )
        _add(
            checks,
            "PASS" if source_info["fallbacks_to_base"] else "FAIL",
            "robojudo_unilab_policy_has_base_fallback",
            str(source_info),
        )
        _add(
            checks,
            "PASS" if base_env_info["fk_passes_joint_vel"] else "FAIL",
            "robojudo_fk_passes_joint_vel",
            str(base_env_info),
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    fail_count = sum(c.level == "FAIL" for c in checks)
    warn_count = sum(c.level == "WARN" for c in checks)
    pass_count = sum(c.level == "PASS" for c in checks)
    for check in checks:
        print(f"[{check.level}] {check.name}: {check.detail}")
    print(f"\nsummary: {fail_count} fail(s), {warn_count} warning(s), {pass_count} pass(es)")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
