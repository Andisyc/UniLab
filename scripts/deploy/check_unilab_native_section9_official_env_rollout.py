#!/usr/bin/env python3
"""Section 9 checker: run the official UniLab env.step/play semantics.

Section 8 showed that hand-written MuJoCo rollouts apply non-zero force but
still fall. This checker moves one layer up: it constructs the UniLab
G1WalkFlat environment through the registry, loads the same FastSAC checkpoint
and ONNX, and rolls out through ``env.step()``.

Two lifecycle modes are intentionally tested:

* collector_init_state: matches the off-policy collector path.
* play_reset_call: matches scripts/train_offpolicy.py play_offpolicy's
  initialize/reset call before the first policy step.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import onnxruntime as ort
import torch

from unilab.algos.torch.fast_sac.learner import SACActor
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


@dataclass
class Check:
    level: str
    name: str
    detail: str


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level, name, detail))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def find_checkpoint(run_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        last = summary.get("last_checkpoint")
        if isinstance(last, str):
            local = run_dir / Path(last).name
            if local.is_file():
                return local.resolve()
    candidates = sorted(run_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No model_*.pt found in {run_dir}")
    return candidates[-1].resolve()


def build_env_override(run_cfg: dict[str, Any]) -> dict[str, Any]:
    config = run_cfg["config"]
    env_override = dict(config.get("env", {}))
    env_override["reward_config"] = dict(config["reward"])
    return env_override


def build_actor(checkpoint_path: Path, run_cfg: dict[str, Any], device: str) -> SACActor:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    actor_state = checkpoint["actor"]
    obs_dim = int(actor_state["net.0.weight"].shape[1])
    action_dim = int(actor_state["fc_mu.weight"].shape[0])
    algo_cfg = run_cfg["config"]["algo"]
    actor = SACActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(algo_cfg.get("actor_hidden_dim", 512)),
        use_layer_norm=bool(algo_cfg.get("use_layer_norm", True)),
        device=device,
    )
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


def actor_policy(actor: SACActor, device: str) -> Callable[[np.ndarray], np.ndarray]:
    def _policy(obs_np: np.ndarray) -> np.ndarray:
        obs_torch = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).to(device)
        with torch.inference_mode():
            return actor.explore(obs_torch, deterministic=True).cpu().numpy()

    return _policy


def onnx_policy(onnx_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    fixed_batch = input_shape[0] if input_shape and isinstance(input_shape[0], int) else None

    def _policy(obs_np: np.ndarray) -> np.ndarray:
        obs_arr = np.asarray(obs_np, dtype=np.float32)
        if fixed_batch in (None, obs_arr.shape[0]):
            return np.asarray(
                session.run([output_name], {input_name: obs_arr})[0],
                dtype=np.float32,
            )
        if fixed_batch == 1:
            outputs = [
                session.run([output_name], {input_name: obs_arr[i : i + 1]})[0]
                for i in range(obs_arr.shape[0])
            ]
            return np.asarray(np.concatenate(outputs, axis=0), dtype=np.float32)
        raise ValueError(
            f"ONNX fixed batch {fixed_batch} is incompatible with obs batch {obs_arr.shape[0]}"
        )

    return _policy


def extract_actor_obs(obs_dict: dict[str, np.ndarray]) -> np.ndarray:
    actor_obs, _ = split_obs_dict(obs_dict)
    return np.asarray(actor_obs, dtype=np.float32)


def backend_base_height(env: Any) -> np.ndarray:
    return np.asarray(env._backend.get_base_pos()[:, 2], dtype=np.float32)


def maybe_close(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def make_env(
    *,
    task_name: str,
    sim_backend: str,
    num_envs: int,
    env_override: dict[str, Any],
) -> Any:
    return registry.make(
        task_name,
        num_envs=num_envs,
        sim_backend=sim_backend,
        env_cfg_override=env_override,
    )


def run_rollout(
    *,
    label: str,
    lifecycle: str,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    task_name: str,
    sim_backend: str,
    num_envs: int,
    steps: int,
    env_override: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    apply_training_seed(seed, torch_runtime=True, cuda=False)
    env = make_env(
        task_name=task_name,
        sim_backend=sim_backend,
        num_envs=num_envs,
        env_override=env_override,
    )
    try:
        env._autoreset = True
        if lifecycle == "collector_init_state":
            state = env.init_state()
            obs = extract_actor_obs(state.obs)
            info = state.info
            state_was_none_after_init = env.state is None
        elif lifecycle == "play_reset_call":
            obs_dict, info = env.reset(np.arange(num_envs, dtype=np.int32))
            obs = extract_actor_obs(obs_dict)
            state_was_none_after_init = env.state is None
        else:
            raise ValueError(f"unknown lifecycle: {lifecycle}")

        commands = np.asarray(info.get("commands"), dtype=np.float32)
        gait_phase = np.asarray(info.get("gait_phase"), dtype=np.float32)
        current_actions = np.asarray(info.get("current_actions"), dtype=np.float32)
        reset_height = backend_base_height(env)

        episode_lengths: list[int] = []
        active_lengths = np.zeros(num_envs, dtype=np.int32)
        done_counts = np.zeros(num_envs, dtype=np.int32)
        min_height = float(np.min(reset_height))
        max_height = float(np.max(reset_height))
        reward_sum = np.zeros(num_envs, dtype=np.float64)
        first_action: np.ndarray | None = None
        first_step_state_was_none = env.state is None

        for _ in range(steps):
            actions = np.asarray(policy_fn(obs), dtype=np.float32)
            if first_action is None:
                first_action = actions.copy()
            state = env.step(actions)
            obs = extract_actor_obs(state.obs)
            active_lengths += 1
            reward_sum += np.asarray(state.reward, dtype=np.float64)
            height = backend_base_height(env)
            min_height = min(min_height, float(np.min(height)))
            max_height = max(max_height, float(np.max(height)))

            done = np.asarray(state.terminated | state.truncated, dtype=bool)
            if np.any(done):
                episode_lengths.extend(active_lengths[done].astype(int).tolist())
                done_counts[done] += 1
                active_lengths[done] = 0

        assert first_action is not None
        final_info = env.state.info if env.state is not None else {}
        final_commands = np.asarray(final_info.get("commands", commands), dtype=np.float32)
        final_gait_phase = np.asarray(final_info.get("gait_phase", gait_phase), dtype=np.float32)
        result = {
            "label": label,
            "lifecycle": lifecycle,
            "state_was_none_after_init": state_was_none_after_init,
            "first_step_state_was_none": first_step_state_was_none,
            "reset_height_stats": stats(reset_height),
            "command_stats": stats(commands),
            "gait_phase_stats": stats(gait_phase),
            "current_actions_stats": stats(current_actions),
            "first_action_stats": stats(first_action),
            "final_command_stats": stats(final_commands),
            "final_gait_phase_stats": stats(final_gait_phase),
            "min_height": min_height,
            "max_height": max_height,
            "done_total": int(np.sum(done_counts)),
            "done_counts": done_counts.tolist(),
            "episode_lengths": episode_lengths,
            "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else None,
            "episode_length_min": int(np.min(episode_lengths)) if episode_lengths else None,
            "episode_length_max": int(np.max(episode_lengths)) if episode_lengths else None,
            "active_length_mean": float(np.mean(active_lengths)),
            "reward_sum_mean": float(np.mean(reward_sum)),
        }
        return result
    finally:
        maybe_close(env)


def print_result(result: dict[str, Any]) -> None:
    print(f"-- {result['label']} / {result['lifecycle']} --")
    for key in [
        "state_was_none_after_init",
        "first_step_state_was_none",
        "reset_height_stats",
        "command_stats",
        "gait_phase_stats",
        "current_actions_stats",
        "first_action_stats",
        "final_command_stats",
        "final_gait_phase_stats",
        "min_height",
        "max_height",
        "done_total",
        "episode_length_mean",
        "episode_length_min",
        "episode_length_max",
        "active_length_mean",
        "reward_sum_mean",
    ]:
        print(f"{key}: {result[key]}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--runtime",
        choices=["checkpoint", "onnx", "both"],
        default="both",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    run_config_path = run_dir / "run_config.json"
    checkpoint_path = find_checkpoint(run_dir, args.checkpoint)
    onnx_path = args.onnx.resolve()

    checks: list[Check] = []
    for name, path in [
        ("run_config_exists", run_config_path),
        ("checkpoint_exists", checkpoint_path),
        ("onnx_exists", onnx_path),
    ]:
        _add(checks, "PASS" if path.is_file() else "FAIL", name, path.as_posix())
    if any(c.level == "FAIL" for c in checks):
        for check in checks:
            print(f"[{check.level}] {check.name}: {check.detail}")
        return 1

    run_cfg = load_json(run_config_path)
    config = run_cfg["config"]
    task_name = str(config["training"]["task_name"])
    sim_backend = str(config["training"]["sim_backend"])
    seed = int(args.seed if args.seed is not None else run_cfg["run"].get("effective_seed", 1))
    env_override = build_env_override(run_cfg)

    ensure_registries()
    print("== Section 9: Official UniLab Env Rollout ==")
    print(f"run_dir: {run_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"onnx: {onnx_path}")
    print(f"task_name: {task_name}")
    print(f"sim_backend: {sim_backend}")
    print(f"num_envs: {args.num_envs}")
    print(f"steps: {args.steps}")
    print(f"seed: {seed}")
    print(f"env_override_keys: {sorted(env_override.keys())}")
    print()

    policies: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
    if args.runtime in {"checkpoint", "both"}:
        actor = build_actor(checkpoint_path, run_cfg, args.device)
        policies.append(("checkpoint_actor", actor_policy(actor, args.device)))
    if args.runtime in {"onnx", "both"}:
        policies.append(("onnx_actor", onnx_policy(onnx_path)))

    results: list[dict[str, Any]] = []
    for label, policy_fn in policies:
        for lifecycle in ("collector_init_state", "play_reset_call"):
            result = run_rollout(
                label=label,
                lifecycle=lifecycle,
                policy_fn=policy_fn,
                task_name=task_name,
                sim_backend=sim_backend,
                num_envs=args.num_envs,
                steps=args.steps,
                env_override=env_override,
                seed=seed,
            )
            results.append(result)
            print_result(result)

    stable_results = [
        r
        for r in results
        if r["done_total"] == 0 and (r["min_height"] is None or float(r["min_height"]) > 0.3)
    ]
    if stable_results:
        _add(checks, "PASS", "official_env_stable_candidate", ", ".join(f"{r['label']}/{r['lifecycle']}" for r in stable_results))
    else:
        _add(checks, "WARN", "official_env_stable_candidate", "No rollout stayed termination-free above min height.")

    for r in results:
        if r["lifecycle"] == "play_reset_call" and r["state_was_none_after_init"]:
            _add(
                checks,
                "WARN",
                "play_reset_lifecycle_state",
                "env.reset(...) did not populate env.state before first env.step(); first step may call init_state().",
            )
            break

    if len(results) >= 2:
        actor_results = [r for r in results if r["label"] == "checkpoint_actor"]
        onnx_results = [r for r in results if r["label"] == "onnx_actor"]
        if actor_results and onnx_results:
            max_gap = max(
                abs(float(a["min_height"]) - float(o["min_height"]))
                for a in actor_results
                for o in onnx_results
                if a["lifecycle"] == o["lifecycle"]
            )
            if max_gap > 0.05:
                _add(checks, "WARN", "checkpoint_onnx_rollout_gap", f"max_min_height_gap={max_gap:.3f}")
            else:
                _add(checks, "PASS", "checkpoint_onnx_rollout_gap", f"max_min_height_gap={max_gap:.3f}")

    fail_count = sum(c.level == "FAIL" for c in checks)
    warn_count = sum(c.level == "WARN" for c in checks)
    pass_count = sum(c.level == "PASS" for c in checks)
    for check in checks:
        print(f"[{check.level}] {check.name}: {check.detail}")
    print(f"\nsummary: {fail_count} fail(s), {warn_count} warning(s), {pass_count} pass(es)")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
