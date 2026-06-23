#!/usr/bin/env python3
"""Audit whether FastSAC observation normalization is actually wired.

This is a Section 2 follow-up diagnostic. It answers one narrow question:

    If run_config says algo.obs_normalization=true for FastSAC, did the
    FastSAC learner actually own, update, save, and export normalizer stats?

The script combines runtime introspection with checkpoint evidence. It does not
train or run simulation.
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.fast_sac.learner import FastSACLearner


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOJUDO_ROOT = REPO_ROOT.parent / "RoboJudo_Real"
DEFAULT_RUN_CONFIG_REL = Path(
    "assets/models/g1/unilab/g1_walk_flat/2026-06-12_15-46-01_mujoco/run_config.json"
)


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


def audit(run_config: Path, checkpoint: Path | None) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    details: dict[str, Any] = {}

    cfg = load_json(run_config)
    algo = get_nested(cfg, ["run", "algo"])
    obs_norm_cfg = get_nested(cfg, ["config", "algo", "obs_normalization"])
    details["run_config"] = run_config.as_posix()
    details["algo"] = algo
    details["run_config_obs_normalization"] = obs_norm_cfg

    if algo != "sac":
        _add(checks, "WARN", "algo", f"expected sac for this audit, got {algo}")
    else:
        _add(checks, "PASS", "algo", "sac")

    init_sig = inspect.signature(FastSACLearner.__init__)
    details["fast_sac_init_params"] = list(init_sig.parameters)
    if "obs_normalization" in init_sig.parameters:
        _add(checks, "PASS", "learner_accepts_obs_normalization", "FastSACLearner.__init__")
    else:
        _add(
            checks,
            "FAIL",
            "learner_accepts_obs_normalization",
            "FastSACLearner.__init__ has no obs_normalization parameter",
        )

    learner = FastSACLearner(
        obs_dim=98,
        action_dim=29,
        critic_obs_dim=98,
        device="cpu",
        use_compile=False,
        use_amp=False,
        use_symmetry=False,
    )
    has_attr = hasattr(learner, "obs_normalizer")
    details["fresh_learner_has_obs_normalizer"] = has_attr
    if has_attr:
        _add(checks, "PASS", "fresh_learner_obs_normalizer", type(learner.obs_normalizer).__name__)
    else:
        _add(checks, "FAIL", "fresh_learner_obs_normalizer", "missing")

    state = learner.get_state_dict()
    details["fresh_state_keys"] = sorted(state.keys())
    if "obs_normalizer" in state:
        _add(checks, "PASS", "fresh_state_saves_obs_normalizer", "present")
    else:
        _add(checks, "FAIL", "fresh_state_saves_obs_normalizer", "missing")

    if checkpoint is None or not checkpoint.is_file():
        _add(checks, "FAIL", "checkpoint", f"not found: {checkpoint}")
    else:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        details["checkpoint"] = checkpoint.as_posix()
        details["checkpoint_keys"] = sorted(ckpt.keys())
        _add(checks, "PASS", "checkpoint", checkpoint.as_posix())
        if "obs_normalizer" in ckpt:
            _add(checks, "PASS", "checkpoint_saves_obs_normalizer", "present")
        else:
            _add(checks, "FAIL", "checkpoint_saves_obs_normalizer", "missing")

    if obs_norm_cfg is True:
        _add(checks, "WARN", "run_config_requests_obs_normalization", "true")
        if not has_attr:
            _add(
                checks,
                "FAIL",
                "effective_training_obs_normalization",
                "false: config requests it, but FastSACLearner has no normalizer",
            )
    elif obs_norm_cfg is False:
        _add(checks, "PASS", "run_config_requests_obs_normalization", "false")
    else:
        _add(checks, "WARN", "run_config_requests_obs_normalization", str(obs_norm_cfg))

    return checks, details


def print_report(checks: list[Check], details: dict[str, Any]) -> None:
    print("# FastSAC obs-normalization implementation audit")
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
    parser.add_argument("--robojudo-root", type=Path, default=DEFAULT_ROBOJUDO_ROOT)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG_REL)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robojudo_root = args.robojudo_root.expanduser().resolve()
    run_config = args.run_config
    if not run_config.is_absolute():
        run_config = robojudo_root / run_config
    run_config = run_config.resolve()

    checkpoint = args.checkpoint
    if checkpoint is not None and not checkpoint.is_absolute():
        checkpoint = robojudo_root / checkpoint
    checkpoint = find_checkpoint(run_config, checkpoint)

    checks, details = audit(run_config, checkpoint)
    print_report(checks, details)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
