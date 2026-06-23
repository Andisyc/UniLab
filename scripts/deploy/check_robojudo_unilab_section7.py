#!/usr/bin/env python3
"""Section 7 checker for UniLab -> RoboJudo_Real pipeline state pollution.

This script checks whether RoboJudo_Real dry-run lifecycle calls pollute the
UniLabPolicy state that enters the actor observation:

* self_check() dry-runs are followed by RlPipeline.__init__ reset().
* dry_run freezes gait_phase.
* dry_run still calls get_pd_target(), which calls get_action() and updates
  UniLabPolicy.last_action.
* prepare() resets at t == 900, then performs more dry-runs before returning.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOJUDO_ROOT = REPO_ROOT.parent / "RoboJudo_Real"
DEFAULT_ROBOJUDO_XML_REL = Path("assets/robots/g1/g1_29dof_rev_1_0.xml")
DEFAULT_POLICY_REL = Path("assets/models/g1/unilab/g1_walk_flat/policy.onnx")
DEFAULT_ROBOJUDO_UNILAB_CFG_REL = Path("robojudo/config/g1/policy/g1_unilab_policy_cfg.py")
DEFAULT_ROBOJUDO_PIPELINE_REL = Path("robojudo/pipeline/rl_pipeline.py")
DEFAULT_ROBOJUDO_POLICY_REL = Path("robojudo/policy/unilab_policy.py")

EXPECTED_DOF = 29
EXPECTED_OBS_DIM = 98
PREPARE_TRAJ_LEN = 1000
PREPARE_RESET_STEP = 900
TOL = 1.0e-6


@dataclass
class Check:
    level: str
    name: str
    detail: str


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level=level, name=name, detail=detail))


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


def eval_literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = eval_literal(node.operand)
        if isinstance(value, (int, float)):
            return -value
    if isinstance(node, ast.List):
        values: list[Any] = []
        for elt in node.elts:
            if isinstance(elt, ast.Starred):
                values.extend(eval_literal(elt.value))
            else:
                values.append(eval_literal(elt))
        return values
    raise ValueError(f"Unsupported literal node: {ast.dump(node)}")


def find_class(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise KeyError(f"class {class_name} not found")


def find_method(cls: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise KeyError(f"method {cls.name}.{method_name} not found")


def extract_class_assignment(path: Path, class_name: str, attr_name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    cls = find_class(tree, class_name)
    for stmt in cls.body:
        value = None
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    value = stmt.value
                    break
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id == attr_name:
                value = stmt.value
        if value is not None:
            return eval_literal(value)
    raise KeyError(f"{class_name}.{attr_name} not found")


def inspect_pipeline_source(path: Path, policy_path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    cls = find_class(tree, "RlPipeline")
    init_src = ast.unparse(find_method(cls, "__init__"))
    step_src = ast.unparse(find_method(cls, "step"))
    prepare_src = ast.unparse(find_method(cls, "prepare"))
    wrapper_cls = find_class(tree, "PolicyWrapper")
    get_pd_target_src = ast.unparse(find_method(wrapper_cls, "get_pd_target"))
    policy_tree = ast.parse(policy_path.read_text(encoding="utf-8"), filename=policy_path.as_posix())
    policy_cls = find_class(policy_tree, "UniLabPolicy")
    policy_methods = {node.name for node in policy_cls.body if isinstance(node, ast.FunctionDef)}
    return {
        "init_calls_self_check_then_reset": init_src.find("self.self_check()") < init_src.find("self.reset()"),
        "dry_run_adds_freeze_phase": "[UNILAB_FREEZE_PHASE]" in step_src,
        "dry_run_skips_env_step": "if not dry_run:" in step_src and "self.env.step" in step_src,
        "dry_run_snapshots_policy_state": "snapshot_state()" in step_src,
        "dry_run_restores_policy_state": "restore_state(policy_state)" in step_src,
        "step_always_calls_get_pd_target": "pd_target = self.policy.get_pd_target(obs)" in step_src,
        "get_pd_target_calls_get_action": "action = self.policy.get_action(obs)" in get_pd_target_src,
        "prepare_calls_step_dry_run": "self.step(dry_run=True)" in prepare_src,
        "prepare_has_reset_at_900": "if t == 0.9 * traj_len:" in prepare_src and "self.reset()" in prepare_src,
        "prepare_has_final_reset_after_loop": prepare_src.rstrip().endswith("self.reset()"),
        "unilab_policy_has_snapshot_restore": {"snapshot_state", "restore_state"}.issubset(policy_methods),
    }


def get_gravity_orientation_xyzw(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quat
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2.0 * (-qz * qx + qw * qy)
    gravity[1] = -2.0 * (qz * qy + qw * qx)
    gravity[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity


def build_obs(
    model: mujoco.MjModel,
    default_angles: np.ndarray,
    last_action: np.ndarray,
    gait_phase: np.ndarray,
) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = np.concatenate(
        [
            np.asarray([0.0, 0.0, 0.754, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            default_angles.astype(np.float64),
        ]
    )
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    base_quat_xyzw = data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
    return np.concatenate(
        [
            data.qvel.astype(np.float32)[3:6] * 0.25,
            -get_gravity_orientation_xyzw(base_quat_xyzw),
            data.qpos.astype(np.float32)[-EXPECTED_DOF:] - default_angles,
            data.qvel.astype(np.float32)[-EXPECTED_DOF:] * 0.05,
            last_action.astype(np.float32),
            np.zeros(3, dtype=np.float32),
            gait_phase.astype(np.float32),
        ],
        dtype=np.float32,
    )


def run_onnx(policy_path: Path, obs: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(policy_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return np.asarray(session.run([output_name], {input_name: obs[None, :].astype(np.float32)})[0]).squeeze()


def simulate_dry_runs(
    policy_path: Path,
    model: mujoco.MjModel,
    default_angles: np.ndarray,
    count: int,
    freeze_phase: bool,
    preserve_policy_state: bool,
    gait_frequency: float,
    dt: float,
    initial_gait_phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    last_action = np.zeros(EXPECTED_DOF, dtype=np.float32)
    gait_phase = np.asarray(initial_gait_phase, dtype=np.float32).copy()
    for _ in range(count):
        obs = build_obs(model, default_angles, last_action, gait_phase)
        next_action = run_onnx(policy_path, obs).astype(np.float32)
        if not preserve_policy_state:
            last_action = next_action
        if not freeze_phase:
            gait_phase = (gait_phase + 2.0 * np.pi * gait_frequency * dt) % (2.0 * np.pi)
    return last_action, gait_phase


def audit(robojudo_root: Path) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    policy_path = robojudo_root / DEFAULT_POLICY_REL
    policy_cfg = robojudo_root / DEFAULT_ROBOJUDO_UNILAB_CFG_REL
    pipeline_path = robojudo_root / DEFAULT_ROBOJUDO_PIPELINE_REL
    policy_source_path = robojudo_root / DEFAULT_ROBOJUDO_POLICY_REL
    model = mujoco.MjModel.from_xml_path((robojudo_root / DEFAULT_ROBOJUDO_XML_REL).as_posix())
    default_angles = np.asarray(
        extract_class_assignment(policy_cfg, "G1UniLabDoF", "default_pos"), dtype=np.float32
    )
    gait_frequency = float(extract_class_assignment(policy_cfg, "G1UniLabPolicyCfg", "gait_frequency"))
    freq = int(extract_class_assignment(policy_cfg, "G1UniLabPolicyCfg", "freq"))
    freeze_phase_during_dry_run = bool(
        extract_class_assignment(policy_cfg, "G1UniLabPolicyCfg", "freeze_phase_during_dry_run")
    )
    initial_gait_phase = np.asarray(
        extract_class_assignment(policy_cfg, "G1UniLabPolicyCfg", "initial_gait_phase"),
        dtype=np.float32,
    )
    dt = 1.0 / freq
    source = inspect_pipeline_source(pipeline_path, policy_source_path)
    preserve_policy_state = bool(
        source["dry_run_snapshots_policy_state"]
        and source["dry_run_restores_policy_state"]
        and source["unilab_policy_has_snapshot_restore"]
    )

    one_dry_last_action, one_dry_phase = simulate_dry_runs(
        policy_path,
        model,
        default_angles,
        1,
        freeze_phase_during_dry_run,
        preserve_policy_state,
        gait_frequency,
        dt,
        initial_gait_phase,
    )
    self_check_last_action, self_check_phase = simulate_dry_runs(
        policy_path,
        model,
        default_angles,
        10,
        freeze_phase_during_dry_run,
        preserve_policy_state,
        gait_frequency,
        dt,
        initial_gait_phase,
    )
    post_prepare_dry_runs = PREPARE_TRAJ_LEN - PREPARE_RESET_STEP - 1
    prepare_last_action, prepare_phase = simulate_dry_runs(
        policy_path,
        model,
        default_angles,
        post_prepare_dry_runs,
        freeze_phase_during_dry_run,
        preserve_policy_state,
        gait_frequency,
        dt,
        initial_gait_phase,
    )
    clean_obs = build_obs(
        model,
        default_angles,
        np.zeros(EXPECTED_DOF, dtype=np.float32),
        initial_gait_phase,
    )
    polluted_obs = build_obs(model, default_angles, prepare_last_action, prepare_phase)

    details: dict[str, Any] = {
        "robojudo_root": robojudo_root.as_posix(),
        "pipeline": pipeline_path.as_posix(),
        "policy": policy_path.as_posix(),
        "policy_source": policy_source_path.as_posix(),
        "source": source,
        "preserve_policy_state_during_dry_run": preserve_policy_state,
        "freq": freq,
        "dt": dt,
        "gait_frequency": gait_frequency,
        "initial_gait_phase": initial_gait_phase.tolist(),
        "freeze_phase_during_dry_run": freeze_phase_during_dry_run,
        "prepare_traj_len": PREPARE_TRAJ_LEN,
        "prepare_reset_step": PREPARE_RESET_STEP,
        "post_prepare_dry_runs_after_reset": post_prepare_dry_runs,
        "one_dry_run_last_action_stats": stats(one_dry_last_action),
        "one_dry_run_gait_phase": one_dry_phase.tolist(),
        "self_check_dry_run_last_action_stats_before_constructor_reset": stats(self_check_last_action),
        "self_check_dry_run_gait_phase_before_constructor_reset": self_check_phase.tolist(),
        "prepare_end_last_action_stats": stats(prepare_last_action),
        "prepare_end_gait_phase": prepare_phase.tolist(),
        "clean_first_obs_last_action_stats": stats(clean_obs[64:93]),
        "polluted_first_obs_last_action_stats": stats(polluted_obs[64:93]),
        "clean_vs_polluted_obs_max_abs": float(np.max(np.abs(clean_obs - polluted_obs))),
        "clean_vs_polluted_last_action_max_abs": float(
            np.max(np.abs(clean_obs[64:93] - polluted_obs[64:93]))
        ),
    }

    for key, ok in source.items():
        if key == "prepare_has_final_reset_after_loop":
            continue
        _add(checks, "PASS" if ok else "FAIL", f"source_{key}", str(ok))

    if source["prepare_has_final_reset_after_loop"]:
        _add(checks, "PASS", "source_prepare_final_reset", "prepare resets after dry-run loop")
    else:
        _add(
            checks,
            "PASS" if preserve_policy_state else "FAIL",
            "source_prepare_final_reset",
            "not required because dry-run restores policy state" if preserve_policy_state else "prepare has no final policy reset after loop",
        )

    one_last = float(details["one_dry_run_last_action_stats"]["max_abs"])
    if one_last <= TOL:
        _add(checks, "PASS", "dry_run_does_not_update_last_action", f"max_abs={one_last:.3e}")
    else:
        _add(checks, "FAIL", "dry_run_updates_last_action", f"max_abs={one_last:.3e}")

    phase_after_one = float(np.max(np.abs(one_dry_phase - initial_gait_phase)))
    if phase_after_one <= TOL:
        _add(checks, "PASS", "dry_run_freezes_gait_phase", f"max_delta={phase_after_one:.3e}")
    else:
        _add(checks, "FAIL", "dry_run_freezes_gait_phase", f"max_delta={phase_after_one:.3e}")

    if source["init_calls_self_check_then_reset"]:
        _add(
            checks,
            "PASS",
            "self_check_pollution_cleared_by_constructor_reset",
            "RlPipeline.__init__ calls reset() after self_check()",
        )
    else:
        _add(checks, "FAIL", "self_check_pollution_cleared_by_constructor_reset", "missing reset")

    prepare_last = float(details["prepare_end_last_action_stats"]["max_abs"])
    if prepare_last <= TOL:
        _add(checks, "PASS", "prepare_end_last_action_zero", f"max_abs={prepare_last:.3e}")
    else:
        _add(checks, "FAIL", "prepare_end_last_action_nonzero", f"max_abs={prepare_last:.3e}")

    prepare_phase_delta = float(np.max(np.abs(prepare_phase - initial_gait_phase)))
    if prepare_phase_delta <= TOL:
        _add(checks, "PASS", "prepare_end_gait_phase_clean", f"max_delta={prepare_phase_delta:.3e}")
    else:
        _add(checks, "FAIL", "prepare_end_gait_phase_polluted", f"max_delta={prepare_phase_delta:.3e}")

    obs_last = details["clean_vs_polluted_last_action_max_abs"]
    if obs_last <= TOL:
        _add(checks, "PASS", "first_real_step_last_action_segment_zero", f"max_abs={obs_last:.3e}")
    else:
        _add(checks, "FAIL", "first_real_step_last_action_segment_polluted", f"max_abs={obs_last:.3e}")

    return checks, details


def print_report(checks: list[Check], details: dict[str, Any], json_out: bool) -> None:
    if json_out:
        print(
            json.dumps(
                {
                    "checks": [check.__dict__ for check in checks],
                    "details": details,
                    "summary": {
                        "fail": sum(check.level == "FAIL" for check in checks),
                        "warn": sum(check.level == "WARN" for check in checks),
                        "pass": sum(check.level == "PASS" for check in checks),
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    print("== Section 7: Pipeline Lifecycle State Pollution ==")
    print(f"pipeline: {details['pipeline']}")
    print(f"post_prepare_dry_runs_after_reset: {details['post_prepare_dry_runs_after_reset']}")
    print(f"freeze_phase_during_dry_run: {details['freeze_phase_during_dry_run']}")
    print(f"preserve_policy_state_during_dry_run: {details['preserve_policy_state_during_dry_run']}")
    print()
    for name in [
        "one_dry_run_last_action_stats",
        "self_check_dry_run_last_action_stats_before_constructor_reset",
        "prepare_end_last_action_stats",
        "clean_first_obs_last_action_stats",
        "polluted_first_obs_last_action_stats",
    ]:
        s = details[name]
        print(
            f"{name}: shape={s['shape']} min={s['min']:.6g} max={s['max']:.6g} "
            f"mean={s['mean']:.6g} std={s['std']:.6g} max_abs={s['max_abs']:.6g}"
        )
    print(f"one_dry_run_gait_phase: {details['one_dry_run_gait_phase']}")
    print(f"prepare_end_gait_phase: {details['prepare_end_gait_phase']}")
    print(f"clean_vs_polluted_last_action_max_abs: {details['clean_vs_polluted_last_action_max_abs']:.6g}")
    print()
    for check in checks:
        print(f"[{check.level}] {check.name}: {check.detail}")
    failures = sum(check.level == "FAIL" for check in checks)
    warnings = sum(check.level == "WARN" for check in checks)
    print(f"\nsummary: {failures} fail(s), {warnings} warning(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robojudo-root", type=Path, default=DEFAULT_ROBOJUDO_ROOT)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks, details = audit(args.robojudo_root.resolve())
    print_report(checks, details, args.json)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
