#!/usr/bin/env python3
"""Section 2 checker for UniLab -> RoboJudo_Real observation normalization.

This script checks only the obs-normalization contract:

* run_config says whether training requested observation normalization.
* checkpoint says whether deployable normalizer statistics are available.
* ONNX is compared against the PyTorch actor reconstructed from checkpoint.
* If normalizer stats exist, the script compares raw vs normalized feeds.

It deliberately does not start RoboJudo_Real, MuJoCo, or a controller.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from unilab.algos.torch.fast_sac.learner import SACActor


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOJUDO_ROOT = REPO_ROOT.parent / "RoboJudo_Real"
DEFAULT_POLICY_REL = Path("assets/models/g1/unilab/g1_walk_flat/policy.onnx")
EXPECTED_OBS_DIM = 98
EXPECTED_ACTION_DIM = 29


@dataclass
class Check:
    level: str
    name: str
    detail: str


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level=level, name=name, detail=detail))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def find_run_configs(policy_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    direct = policy_dir / "run_config.json"
    if direct.is_file():
        candidates.append(direct)
    candidates.extend(sorted(policy_dir.glob("*/run_config.json")))
    return sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)


def find_checkpoint(run_config: Path, requested: Path | None) -> Path | None:
    if requested is not None:
        return requested.resolve()

    run_dir = run_config.parent
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        last = summary.get("last_checkpoint")
        if isinstance(last, str):
            local = run_dir / Path(last).name
            if local.is_file():
                return local.resolve()

    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0].resolve() if checkpoints else None


def summarize_tensor(name: str, value: torch.Tensor | np.ndarray) -> dict[str, Any]:
    arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def infer_actor_dims(actor_state: dict[str, torch.Tensor]) -> tuple[int | None, int | None]:
    obs_dim = None
    action_dim = None
    first_weight = actor_state.get("net.0.weight")
    mu_weight = actor_state.get("fc_mu.weight")
    if isinstance(first_weight, torch.Tensor) and first_weight.ndim == 2:
        obs_dim = int(first_weight.shape[1])
    if isinstance(mu_weight, torch.Tensor) and mu_weight.ndim == 2:
        action_dim = int(mu_weight.shape[0])
    return obs_dim, action_dim


def build_actor(
    actor_state: dict[str, torch.Tensor],
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
    use_layer_norm: bool,
) -> SACActor:
    actor = SACActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        use_layer_norm=use_layer_norm,
        device="cpu",
    )
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor


def extract_normalizer_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    state = checkpoint.get("obs_normalizer")
    if isinstance(state, dict):
        return {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    return None


def normalize_obs(obs: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = state.get("_mean")
    std = state.get("_std")
    if mean is None and "mean" in state:
        mean = state["mean"]
    if std is None and "std" in state:
        std = state["std"]
    if mean is None or std is None:
        raise ValueError(f"Unsupported obs_normalizer state keys: {sorted(state.keys())}")
    eps = 1e-2
    return (obs - mean.to(obs.device)) / (std.to(obs.device) + eps)


def make_probe_obs(obs_dim: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    zero = np.zeros((1, obs_dim), dtype=np.float32)
    randn = rng.normal(0.0, 1.0, size=(1, obs_dim)).astype(np.float32)
    stand_like = np.zeros((1, obs_dim), dtype=np.float32)
    if obs_dim == 98:
        # gyro(3), -gravity(3), dof_rel(29), dof_vel(29), action(29), cmd(3), phase(2)
        stand_like[0, 3:6] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        stand_like[0, 96:98] = np.array([0.0, np.pi], dtype=np.float32)
    return {"zero": zero, "randn": randn, "stand_like": stand_like}


def run_onnx(session: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return np.asarray(session.run([output_name], {input_name: obs.astype(np.float32)})[0])


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def section2_checks(
    run_config: Path,
    checkpoint_path: Path | None,
    policy_path: Path,
) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    details: dict[str, Any] = {}

    cfg = load_json(run_config)
    obs_normalization = get_nested(cfg, ["config", "algo", "obs_normalization"])
    algo = get_nested(cfg, ["run", "algo"])
    hidden_dim = int(get_nested(cfg, ["config", "algo", "actor_hidden_dim"], 512))
    use_layer_norm = bool(get_nested(cfg, ["config", "algo", "use_layer_norm"], True))

    details["run_config"] = run_config.as_posix()
    details["algo"] = algo
    details["obs_normalization"] = obs_normalization
    details["actor_hidden_dim"] = hidden_dim
    details["use_layer_norm"] = use_layer_norm

    if obs_normalization is True:
        _add(checks, "WARN", "run_config_obs_normalization", "true")
    elif obs_normalization is False:
        _add(checks, "PASS", "run_config_obs_normalization", "false")
    else:
        _add(checks, "WARN", "run_config_obs_normalization", str(obs_normalization))

    if checkpoint_path is None or not checkpoint_path.is_file():
        _add(checks, "FAIL", "checkpoint", f"not found: {checkpoint_path}")
        return checks, details

    details["checkpoint"] = checkpoint_path.as_posix()
    checkpoint = load_checkpoint(checkpoint_path)
    details["checkpoint_top_keys"] = sorted(checkpoint.keys())
    _add(checks, "PASS", "checkpoint", checkpoint_path.as_posix())

    actor_state = checkpoint.get("actor")
    if not isinstance(actor_state, dict):
        _add(checks, "FAIL", "actor_state", "checkpoint has no dict key 'actor'")
        return checks, details

    obs_dim, action_dim = infer_actor_dims(actor_state)
    details["actor_obs_dim"] = obs_dim
    details["actor_action_dim"] = action_dim
    if obs_dim == EXPECTED_OBS_DIM:
        _add(checks, "PASS", "actor_obs_dim", str(obs_dim))
    else:
        _add(checks, "FAIL", "actor_obs_dim", f"expected {EXPECTED_OBS_DIM}, got {obs_dim}")
    if action_dim == EXPECTED_ACTION_DIM:
        _add(checks, "PASS", "actor_action_dim", str(action_dim))
    else:
        _add(
            checks,
            "FAIL",
            "actor_action_dim",
            f"expected {EXPECTED_ACTION_DIM}, got {action_dim}",
        )
    if obs_dim is None or action_dim is None:
        return checks, details

    normalizer_state = extract_normalizer_state(checkpoint)
    details["checkpoint_has_obs_normalizer"] = normalizer_state is not None
    if normalizer_state is None:
        if obs_normalization is True:
            _add(
                checks,
                "FAIL",
                "obs_normalizer_state",
                "run_config requests obs_normalization=true, but checkpoint has no obs_normalizer",
            )
        else:
            _add(checks, "PASS", "obs_normalizer_state", "not required and not present")
    else:
        details["obs_normalizer_keys"] = sorted(normalizer_state.keys())
        for key, value in normalizer_state.items():
            details[f"obs_normalizer_{key}"] = summarize_tensor(key, value)
        _add(checks, "PASS", "obs_normalizer_state", ",".join(sorted(normalizer_state.keys())))

    if not policy_path.is_file():
        _add(checks, "FAIL", "policy_onnx", f"not found: {policy_path}")
        return checks, details
    session = ort.InferenceSession(policy_path.as_posix(), providers=["CPUExecutionProvider"])
    details["onnx_input"] = {
        "name": session.get_inputs()[0].name,
        "shape": list(session.get_inputs()[0].shape),
        "type": session.get_inputs()[0].type,
    }
    details["onnx_output"] = {
        "name": session.get_outputs()[0].name,
        "shape": list(session.get_outputs()[0].shape),
        "type": session.get_outputs()[0].type,
    }
    _add(checks, "PASS", "policy_onnx", policy_path.as_posix())

    actor = build_actor(actor_state, obs_dim, action_dim, hidden_dim, use_layer_norm)
    probes = make_probe_obs(obs_dim)
    comparisons: dict[str, Any] = {}
    with torch.inference_mode():
        for name, obs_np in probes.items():
            obs_t = torch.from_numpy(obs_np).float()
            pt_raw = actor(obs_t)[0].cpu().numpy()
            onnx_raw = run_onnx(session, obs_np)
            item: dict[str, Any] = {
                "raw_obs": summarize_tensor(f"{name}_raw_obs", obs_np),
                "onnx_raw_action": summarize_tensor(f"{name}_onnx_raw_action", onnx_raw),
                "pt_raw_action": summarize_tensor(f"{name}_pt_raw_action", pt_raw),
                "onnx_vs_pt_raw_max_abs": max_abs(onnx_raw, pt_raw),
            }
            if normalizer_state is not None:
                norm_t = normalize_obs(obs_t, normalizer_state)
                norm_np = norm_t.cpu().numpy().astype(np.float32)
                onnx_norm = run_onnx(session, norm_np)
                pt_norm = actor(norm_t)[0].cpu().numpy()
                item.update(
                    {
                        "normalized_obs": summarize_tensor(f"{name}_normalized_obs", norm_np),
                        "onnx_normalized_action": summarize_tensor(
                            f"{name}_onnx_normalized_action", onnx_norm
                        ),
                        "pt_normalized_action": summarize_tensor(
                            f"{name}_pt_normalized_action", pt_norm
                        ),
                        "onnx_vs_pt_normalized_max_abs": max_abs(onnx_norm, pt_norm),
                        "onnx_raw_vs_normalized_action_max_abs": max_abs(onnx_raw, onnx_norm),
                    }
                )
            comparisons[name] = item

    details["probe_comparisons"] = comparisons

    raw_diffs = [item["onnx_vs_pt_raw_max_abs"] for item in comparisons.values()]
    max_raw_diff = max(raw_diffs)
    if max_raw_diff <= 1e-4:
        _add(
            checks,
            "PASS",
            "onnx_matches_bare_actor_on_raw_input",
            f"max_abs={max_raw_diff:.3e}",
        )
    else:
        _add(
            checks,
            "FAIL",
            "onnx_matches_bare_actor_on_raw_input",
            f"max_abs={max_raw_diff:.3e}",
        )

    if obs_normalization is True and normalizer_state is None:
        _add(
            checks,
            "FAIL",
            "normalized_obs_replayable",
            "false: no checkpoint normalizer stats to apply before ONNX",
        )
    elif obs_normalization is True and normalizer_state is not None:
        norm_action_diffs = [
            item["onnx_raw_vs_normalized_action_max_abs"] for item in comparisons.values()
        ]
        _add(
            checks,
            "PASS",
            "normalized_obs_replayable",
            f"true: max raw-vs-normalized action diff={max(norm_action_diffs):.3e}",
        )
    else:
        _add(checks, "PASS", "normalized_obs_replayable", "not required by run_config")

    return checks, details


def print_report(checks: list[Check], details: dict[str, Any]) -> None:
    print("# Section 2: UniLab policy obs-normalization contract")
    print("details:")
    print(json.dumps(details, indent=2, ensure_ascii=False))
    print()
    print("checks:")
    for check in checks:
        print(f"  [{check.level}] {check.name}: {check.detail}")
    fail_count = sum(1 for c in checks if c.level == "FAIL")
    warn_count = sum(1 for c in checks if c.level == "WARN")
    print()
    print(f"summary: {fail_count} fail(s), {warn_count} warning(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robojudo-root",
        type=Path,
        default=DEFAULT_ROBOJUDO_ROOT,
        help="Path to RoboJudo_Real. Defaults to UniLab sibling directory.",
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=None,
        help="Override policy.onnx path. Relative paths are resolved from --robojudo-root.",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="Override run_config.json path. Relative paths are resolved from --robojudo-root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint path. Relative paths are resolved from --robojudo-root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robojudo_root = args.robojudo_root.expanduser().resolve()

    policy_path = args.policy_path or DEFAULT_POLICY_REL
    if not policy_path.is_absolute():
        policy_path = robojudo_root / policy_path
    policy_path = policy_path.resolve()

    if args.run_config is not None:
        run_config = args.run_config
        if not run_config.is_absolute():
            run_config = robojudo_root / run_config
        run_configs = [run_config.resolve()]
    else:
        run_configs = find_run_configs(policy_path.parent)
    if not run_configs:
        print(f"No run_config.json found under {policy_path.parent}")
        return 1
    run_config = run_configs[0]

    checkpoint = args.checkpoint
    if checkpoint is not None and not checkpoint.is_absolute():
        checkpoint = robojudo_root / checkpoint
    checkpoint_path = find_checkpoint(run_config, checkpoint)

    checks, details = section2_checks(run_config, checkpoint_path, policy_path)
    print_report(checks, details)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
