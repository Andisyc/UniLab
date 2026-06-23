#!/usr/bin/env python3
"""Section 11: test whether the UniLab policy is stable under zero commands.

Section 9 verified that the checkpoint/ONNX is stable in the official UniLab
env with the env's sampled commands. RoboJudo's joystick idle path feeds
commands=[0, 0, 0], so this checker forces zero commands inside the official
UniLab env and compares the rollout height.
"""

from __future__ import annotations

import argparse
import json
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


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _stats(x: Any) -> dict[str, Any]:
    arr = np.asarray(x, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr)) if arr.size else 0.0,
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
    }


def _build_env_override(run_cfg: dict[str, Any]) -> dict[str, Any]:
    config = run_cfg["config"]
    env_override = dict(config.get("env", {}))
    env_override["reward_config"] = dict(config["reward"])
    return env_override


def _onnx_policy(onnx_path: Path):
    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    fixed_batch = input_shape[0] if input_shape and isinstance(input_shape[0], int) else None

    def _policy(obs_np: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_np, dtype=np.float32)
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

    return _policy


def _actor_obs(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    actor_obs, _ = split_obs_dict(obs_dict)
    return np.asarray(actor_obs, dtype=np.float32)


def _height(env: Any) -> np.ndarray:
    return np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32)


def _force_command(info: dict[str, Any], command: np.ndarray) -> None:
    if "commands" in info:
        info["commands"][:] = command[None, :]


def _run_case(
    *,
    label: str,
    force_zero_command: bool,
    command: np.ndarray,
    run_cfg: dict[str, Any],
    onnx_path: Path,
    num_envs: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    apply_training_seed(seed, torch_runtime=False, cuda=False)
    env_override = _build_env_override(run_cfg)
    task_name = str(run_cfg["config"]["training"]["task_name"])
    sim_backend = str(run_cfg["config"]["training"]["sim_backend"])
    env = registry.make(
        task_name,
        num_envs=num_envs,
        sim_backend=sim_backend,
        env_cfg_override=env_override,
    )
    policy = _onnx_policy(onnx_path)
    try:
        env._autoreset = True
        state = env.init_state()
        if force_zero_command:
            _force_command(state.info, command)
        obs = _actor_obs(state.obs)
        if force_zero_command:
            obs[:, 93:96] = command[None, :]
        initial_commands = np.asarray(state.info.get("commands"), dtype=np.float32).copy()
        min_height = float(np.min(_height(env)))
        max_height = float(np.max(_height(env)))
        done_counts = np.zeros(num_envs, dtype=np.int32)
        first_action = None
        for _ in range(steps):
            if force_zero_command and env.state is not None:
                _force_command(env.state.info, command)
                obs[:, 93:96] = command[None, :]
            action = policy(obs)
            if first_action is None:
                first_action = action.copy()
            state = env.step(action)
            if force_zero_command:
                _force_command(state.info, command)
            obs = _actor_obs(state.obs)
            if force_zero_command:
                obs[:, 93:96] = command[None, :]
            h = _height(env)
            min_height = min(min_height, float(np.min(h)))
            max_height = max(max_height, float(np.max(h)))
            done = np.asarray(state.terminated | state.truncated, dtype=bool)
            done_counts += done.astype(np.int32)
        assert first_action is not None
        final_commands = np.asarray(env.state.info.get("commands"), dtype=np.float32) if env.state is not None else initial_commands
        return {
            "label": label,
            "force_zero_command": force_zero_command,
            "initial_command_stats": _stats(initial_commands),
            "final_command_stats": _stats(final_commands),
            "first_action_stats": _stats(first_action),
            "min_height": min_height,
            "max_height": max_height,
            "final_height_stats": _stats(_height(env)),
            "done_total": int(np.sum(done_counts)),
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    run_cfg = _load_json(args.run_dir / "run_config.json")
    ensure_registries()
    command = np.zeros(3, dtype=np.float32)
    sampled = _run_case(
        label="sampled_commands",
        force_zero_command=False,
        command=command,
        run_cfg=run_cfg,
        onnx_path=args.onnx,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
    )
    zero = _run_case(
        label="forced_zero_commands",
        force_zero_command=True,
        command=command,
        run_cfg=run_cfg,
        onnx_path=args.onnx,
        num_envs=args.num_envs,
        steps=args.steps,
        seed=args.seed,
    )
    print("section11_results:")
    print(json.dumps([sampled, zero], indent=2, sort_keys=True))
    fail = 0
    warn = 0
    if sampled["min_height"] < 0.45:
        fail += 1
        print(f"[FAIL] sampled_commands_stable: min_height={sampled['min_height']:.6g}")
    else:
        print(f"[PASS] sampled_commands_stable: min_height={sampled['min_height']:.6g}")
    if zero["min_height"] < 0.45:
        warn += 1
        print(f"[WARN] forced_zero_commands_stable: min_height={zero['min_height']:.6g}")
    else:
        print(f"[PASS] forced_zero_commands_stable: min_height={zero['min_height']:.6g}")
    print(f"summary: {fail} fail(s), {warn} warning(s), {2 - fail - warn} pass(es)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
