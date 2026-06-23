#!/usr/bin/env python3
"""Check UniLab-native Section 6 control semantics against RoboJudo_Real.

This follow-up answers two questions raised by Section 6:

1. Does UniLab's own checkpoint/actor produce the same first-frame action and
   out-of-joint-limit target as the deployed ONNX?
2. Does UniLab's MuJoCo control layer apply clipping/saturation semantics that
   RoboJudo_Real has not reproduced?

The test avoids a full viewer/pipeline run. It reconstructs the same standing
first-frame observation, compares ONNX to the UniLab FastSAC actor checkpoint,
then compares UniLab position-actuator forces with RoboJudo's external PD
torques for the same action target.
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
import torch

from unilab.algos.torch.fast_sac.learner import SACActor


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOJUDO_ROOT = REPO_ROOT.parent / "RoboJudo_Real"
DEFAULT_UNILAB_SCENE = REPO_ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"
DEFAULT_POLICY_REL = Path("assets/models/g1/unilab/g1_walk_flat/policy.onnx")
DEFAULT_ROBOJUDO_XML_REL = Path("assets/robots/g1/g1_29dof_rev_1_0.xml")
DEFAULT_ROBOJUDO_UNILAB_CFG_REL = Path("robojudo/config/g1/policy/g1_unilab_policy_cfg.py")
DEFAULT_ROBOJUDO_ENV_CFG_REL = Path("robojudo/config/g1/env/g1_env_cfg.py")
DEFAULT_ROBOJUDO_G1_CFG_REL = Path("robojudo/config/g1/g1_cfg.py")

ROOT_QPOS_DIM = 7
EXPECTED_DOF = 29
EXPECTED_OBS_DIM = 98
EXPECTED_ACTION_DIM = 29
TOL = 1.0e-5


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


def extract_class_assignment(path: Path, class_name: str, attr_name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    return extract_class_assignment_from_tree(tree, class_name, attr_name)


def extract_class_assignment_from_tree(
    tree: ast.Module,
    class_name: str,
    attr_name: str,
    seen: set[str] | None = None,
) -> Any:
    if seen is None:
        seen = set()
    if class_name in seen:
        raise KeyError(f"cyclic inheritance while looking for {class_name}.{attr_name}")
    seen.add(class_name)
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
    for base in cls.bases:
        if isinstance(base, ast.Name):
            try:
                return extract_class_assignment_from_tree(tree, base.id, attr_name, seen)
            except KeyError:
                pass
    raise KeyError(f"{class_name}.{attr_name} not found")


def extract_g1_unilab_env_dof_class(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    cls = find_class(tree, "g1_unilab")
    for stmt in cls.body:
        value = None
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "env":
                    value = stmt.value
                    break
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id == "env":
                value = stmt.value
        if not isinstance(value, ast.Call):
            continue
        for kw in value.keywords:
            if kw.arg == "dof" and isinstance(kw.value, ast.Call):
                if isinstance(kw.value.func, ast.Name):
                    return kw.value.func.id
    return "G1_29DoF"


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


def find_run_config(policy_dir: Path) -> Path:
    candidates: list[Path] = []
    direct = policy_dir / "run_config.json"
    if direct.is_file():
        candidates.append(direct)
    candidates.extend(sorted(policy_dir.glob("*/run_config.json")))
    if not candidates:
        raise FileNotFoundError(f"No run_config.json found below {policy_dir}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def find_checkpoint(run_config: Path) -> Path:
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
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt found in {run_dir}")
    return checkpoints[0].resolve()


def infer_actor_dims(actor_state: dict[str, torch.Tensor]) -> tuple[int, int]:
    first_weight = actor_state["net.0.weight"]
    mu_weight = actor_state["fc_mu.weight"]
    return int(first_weight.shape[1]), int(mu_weight.shape[0])


def load_actor(checkpoint_path: Path, run_config: Path) -> SACActor:
    cfg = load_json(run_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actor_state = checkpoint["actor"]
    obs_dim, action_dim = infer_actor_dims(actor_state)
    hidden_dim = int(get_nested(cfg, ["config", "algo", "actor_hidden_dim"], 512))
    use_layer_norm = bool(get_nested(cfg, ["config", "algo", "use_layer_norm"], True))
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


def get_gravity_orientation_xyzw(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quat
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2.0 * (-qz * qx + qw * qy)
    gravity[1] = -2.0 * (qz * qy + qw * qx)
    gravity[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity


def build_obs(model: mujoco.MjModel, default_angles: np.ndarray, gait_phase: np.ndarray) -> np.ndarray:
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
            np.zeros(EXPECTED_DOF, dtype=np.float32),
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


def run_actor(actor: SACActor, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = actor.as_export_module()(torch.from_numpy(obs[None, :]).float())
    return out.detach().cpu().numpy().squeeze()


def joint_limit_violations(
    target: np.ndarray,
    joint_names: list[str],
    model: mujoco.MjModel,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for i, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = model.jnt_range[jid]
        if target[i] < low - TOL or target[i] > high + TOL:
            violations.append(
                {
                    "index": i,
                    "joint": name,
                    "target": float(target[i]),
                    "low": float(low),
                    "high": float(high),
                }
            )
    return violations


def actuator_forces(
    model: mujoco.MjModel,
    default_angles: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = np.concatenate(
        [
            np.asarray([0.0, 0.0, 0.754, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            default_angles.astype(np.float64),
        ]
    )
    data.qvel[:] = 0.0
    data.ctrl[:] = target.astype(np.float64)
    mujoco.mj_forward(model, data)
    return data.actuator_force.astype(np.float32).copy()


def audit(robojudo_root: Path, unilab_scene: Path) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    policy_path = robojudo_root / DEFAULT_POLICY_REL
    policy_dir = policy_path.parent
    run_config = find_run_config(policy_dir)
    checkpoint = find_checkpoint(run_config)
    actor = load_actor(checkpoint, run_config)

    unilab_cfg = robojudo_root / DEFAULT_ROBOJUDO_UNILAB_CFG_REL
    env_cfg = robojudo_root / DEFAULT_ROBOJUDO_ENV_CFG_REL
    g1_cfg = robojudo_root / DEFAULT_ROBOJUDO_G1_CFG_REL
    env_dof_class = extract_g1_unilab_env_dof_class(g1_cfg)
    default_angles = np.asarray(
        extract_class_assignment(unilab_cfg, "G1UniLabDoF", "default_pos"),
        dtype=np.float32,
    )
    joint_names = extract_class_assignment(unilab_cfg, "G1UniLabDoF", "joint_names")
    stiffness = np.asarray(extract_class_assignment(env_cfg, env_dof_class, "stiffness"), dtype=np.float32)
    damping = np.asarray(extract_class_assignment(env_cfg, env_dof_class, "damping"), dtype=np.float32)
    torque_limits = np.asarray(
        extract_class_assignment(env_cfg, env_dof_class, "torque_limits"), dtype=np.float32
    )

    unilab_model = mujoco.MjModel.from_xml_path(unilab_scene.as_posix())
    robojudo_model = mujoco.MjModel.from_xml_path((robojudo_root / DEFAULT_ROBOJUDO_XML_REL).as_posix())
    obs = build_obs(unilab_model, default_angles, gait_phase=np.zeros(2, dtype=np.float32))
    onnx_action = run_onnx(policy_path, obs).astype(np.float32)
    actor_action = run_actor(actor, obs).astype(np.float32)
    target = onnx_action + default_angles
    target_violations = joint_limit_violations(target, joint_names, unilab_model)
    native_force = actuator_forces(unilab_model, default_angles, target)
    robojudo_torque = np.clip(onnx_action * stiffness - 0.0 * damping, -torque_limits, torque_limits)
    force_diff = robojudo_torque - native_force
    abs_ratio = np.abs(robojudo_torque) / (np.abs(native_force) + 1.0e-9)

    common: list[dict[str, Any]] = []
    for i, name in enumerate(joint_names):
        common.append(
            {
                "index": i,
                "joint": name,
                "action": float(onnx_action[i]),
                "target": float(target[i]),
                "unilab_kp": float(unilab_model.actuator_gainprm[i, 0]),
                "unilab_kd": float(-unilab_model.actuator_biasprm[i, 2]),
                "unilab_force": float(native_force[i]),
                "unilab_forcelimited": bool(unilab_model.actuator_forcelimited[i]),
                "unilab_forcerange": unilab_model.actuator_forcerange[i].tolist(),
                "unilab_ctrllimited": bool(unilab_model.actuator_ctrllimited[i]),
                "robojudo_stiffness": float(stiffness[i]),
                "robojudo_damping": float(damping[i]),
                "robojudo_torque_limit": float(torque_limits[i]),
                "robojudo_torque": float(robojudo_torque[i]),
                "torque_minus_unilab_force": float(force_diff[i]),
                "abs_torque_to_force_ratio": float(abs_ratio[i]),
            }
        )

    details: dict[str, Any] = {
        "robojudo_root": robojudo_root.as_posix(),
        "unilab_scene": unilab_scene.as_posix(),
        "policy_path": policy_path.as_posix(),
        "run_config": run_config.as_posix(),
        "checkpoint": checkpoint.as_posix(),
        "effective_env_dof_class": env_dof_class,
        "obs_stats": stats(obs),
        "onnx_action_stats": stats(onnx_action),
        "actor_action_stats": stats(actor_action),
        "onnx_vs_actor_max_abs": float(np.max(np.abs(onnx_action - actor_action))),
        "target_stats": stats(target),
        "target_joint_limit_violations": target_violations,
        "unilab_actuator_force_stats": stats(native_force),
        "robojudo_first_frame_torque_stats": stats(robojudo_torque),
        "torque_minus_unilab_force_stats": stats(force_diff),
        "abs_torque_to_force_ratio_stats": stats(abs_ratio),
        "top_force_mismatches": sorted(
            common,
            key=lambda item: abs(item["torque_minus_unilab_force"]),
            reverse=True,
        )[:8],
        "watched_joints": [
            item
            for item in common
            if item["joint"]
            in {
                "left_ankle_pitch_joint",
                "right_ankle_pitch_joint",
                "waist_pitch_joint",
                "left_hip_pitch_joint",
                "right_hip_pitch_joint",
                "left_knee_joint",
                "right_knee_joint",
            }
        ],
        "all_unilab_actuators_ctrllimited": bool(np.all(unilab_model.actuator_ctrllimited)),
        "any_unilab_actuators_ctrllimited": bool(np.any(unilab_model.actuator_ctrllimited)),
        "all_unilab_actuators_forcelimited": bool(np.all(unilab_model.actuator_forcelimited)),
        "any_robojudo_xml_actuator_forcelimited": bool(np.any(robojudo_model.actuator_forcelimited)),
    }

    if obs.shape == (EXPECTED_OBS_DIM,):
        _add(checks, "PASS", "obs_dim", str(obs.shape[0]))
    else:
        _add(checks, "FAIL", "obs_dim", str(obs.shape))

    if onnx_action.shape == (EXPECTED_ACTION_DIM,):
        _add(checks, "PASS", "onnx_action_dim", str(onnx_action.shape[0]))
    else:
        _add(checks, "FAIL", "onnx_action_dim", str(onnx_action.shape))

    max_actor_diff = details["onnx_vs_actor_max_abs"]
    if max_actor_diff <= TOL:
        _add(checks, "PASS", "unilab_checkpoint_actor_matches_onnx", f"max_abs={max_actor_diff:.3e}")
    else:
        _add(checks, "FAIL", "unilab_checkpoint_actor_matches_onnx", f"max_abs={max_actor_diff:.3e}")

    if target_violations:
        _add(
            checks,
            "PASS",
            "unilab_native_target_limit_violations_reproduced",
            json.dumps(target_violations),
        )
    else:
        _add(checks, "WARN", "unilab_native_target_limit_violations_reproduced", "none")

    if not details["any_unilab_actuators_ctrllimited"]:
        _add(checks, "PASS", "unilab_no_ctrlrange_clip", "actuator_ctrllimited is false for all actuators")
    else:
        _add(checks, "FAIL", "unilab_no_ctrlrange_clip", "some actuators have ctrlrange clipping")

    if details["all_unilab_actuators_forcelimited"]:
        _add(checks, "PASS", "unilab_force_saturation", "all actuators have forcerange saturation")
    else:
        _add(checks, "WARN", "unilab_force_saturation", "not all actuators are force-limited")

    max_force_diff = float(np.max(np.abs(force_diff)))
    if max_force_diff <= 1.0:
        _add(checks, "PASS", "robojudo_torque_matches_unilab_actuator_force", f"max_abs={max_force_diff:.3e}")
    else:
        _add(
            checks,
            "FAIL",
            "robojudo_torque_matches_unilab_actuator_force",
            f"max_abs={max_force_diff:.3e}; control gains/limits are not reproduced",
        )

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

    print("== UniLab Native Section 6 Control Check ==")
    print(f"run_config: {details['run_config']}")
    print(f"checkpoint: {details['checkpoint']}")
    print(f"effective_env_dof_class: {details['effective_env_dof_class']}")
    print(f"onnx_vs_actor_max_abs: {details['onnx_vs_actor_max_abs']:.3e}")
    print(f"target_limit_violations: {len(details['target_joint_limit_violations'])}")
    for item in details["target_joint_limit_violations"]:
        print(
            f"  {item['joint']}: target={item['target']:.6g}, "
            f"limit=[{item['low']:.6g}, {item['high']:.6g}]"
        )
    print()
    for name in [
        "onnx_action_stats",
        "target_stats",
        "unilab_actuator_force_stats",
        "robojudo_first_frame_torque_stats",
        "torque_minus_unilab_force_stats",
        "abs_torque_to_force_ratio_stats",
    ]:
        s = details[name]
        print(
            f"{name}: shape={s['shape']} min={s['min']:.6g} max={s['max']:.6g} "
            f"mean={s['mean']:.6g} std={s['std']:.6g} max_abs={s['max_abs']:.6g}"
        )
    print()
    print("Watched joints:")
    for item in details["watched_joints"]:
        print(
            f"  {item['joint']}: action={item['action']:.6g}, target={item['target']:.6g}, "
            f"UniLab kp/kd={item['unilab_kp']:.6g}/{item['unilab_kd']:.6g}, "
            f"UniLab force={item['unilab_force']:.6g}, "
            f"RoboJudo kp/kd={item['robojudo_stiffness']:.6g}/{item['robojudo_damping']:.6g}, "
            f"RoboJudo torque={item['robojudo_torque']:.6g}, "
            f"ratio={item['abs_torque_to_force_ratio']:.3g}"
        )
    print()
    for check in checks:
        print(f"[{check.level}] {check.name}: {check.detail}")
    failures = sum(check.level == "FAIL" for check in checks)
    warnings = sum(check.level == "WARN" for check in checks)
    print(f"\nsummary: {failures} fail(s), {warnings} warning(s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robojudo-root", type=Path, default=DEFAULT_ROBOJUDO_ROOT)
    parser.add_argument("--unilab-scene", type=Path, default=DEFAULT_UNILAB_SCENE)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks, details = audit(args.robojudo_root.resolve(), args.unilab_scene.resolve())
    print_report(checks, details, args.json)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
