#!/usr/bin/env python3
"""Section 6 checker for UniLab -> RoboJudo_Real action-to-PD contract.

This script validates the boundary where a UniLab ONNX action becomes a
RoboJudo PD target:

* ONNX raw action has 29 dimensions.
* RoboJudo's UniLab policy config keeps action_scale = 1.0.
* PolicyWrapper computes pd_target = action + UniLab default pose.
* DoFAdapter maps all 29 policy joints to the env joints.
* The first-frame pd_target stays within G1 joint position limits.
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
DEFAULT_ROBOJUDO_ENV_CFG_REL = Path("robojudo/config/g1/env/g1_env_cfg.py")
DEFAULT_ROBOJUDO_G1_CFG_REL = Path("robojudo/config/g1/g1_cfg.py")
DEFAULT_ROBOJUDO_PIPELINE_REL = Path("robojudo/pipeline/rl_pipeline.py")
DEFAULT_ROBOJUDO_POLICY_REL = Path("robojudo/policy/unilab_policy.py")

ROOT_QPOS_DIM = 7
EXPECTED_DOF = 29
EXPECTED_OBS_DIM = 98
EXPECTED_ACTION_DIM = 29
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


def g1_unilab_uses_stand_init_qpos(path: Path) -> bool:
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
            if kw.arg == "init_qpos":
                return isinstance(kw.value, ast.Name) and kw.value.id == "UNILAB_G1_STAND_QPOS"
    return False


def get_gravity_orientation_xyzw(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quat
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2.0 * (-qz * qx + qw * qy)
    gravity[1] = -2.0 * (qz * qy + qw * qx)
    gravity[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity


def build_first_frame_obs(
    model: mujoco.MjModel,
    default_angles: np.ndarray,
    gait_phase: np.ndarray,
) -> np.ndarray:
    init_root = np.asarray([0.0, 0.0, 0.754, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    init_qpos = np.concatenate([init_root, default_angles.astype(np.float64)])
    data = mujoco.MjData(model)
    data.qpos[:] = init_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    dof_pos = data.qpos.astype(np.float32)[-EXPECTED_DOF:].copy()
    dof_vel = data.qvel.astype(np.float32)[-EXPECTED_DOF:].copy()
    base_quat_xyzw = data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
    base_ang_vel = data.qvel.astype(np.float32)[3:6].copy()
    gravity = -get_gravity_orientation_xyzw(base_quat_xyzw)
    return np.concatenate(
        [
            base_ang_vel * 0.25,
            gravity,
            np.asarray(dof_pos - default_angles, dtype=np.float32),
            dof_vel * 0.05,
            np.zeros(EXPECTED_DOF, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.asarray(gait_phase, dtype=np.float32),
        ],
        dtype=np.float32,
    )


def fit_dof(
    data: np.ndarray,
    src_joint_names: list[str],
    tar_joint_names: list[str],
    template: np.ndarray | None = None,
) -> np.ndarray:
    if data.shape[-1] != len(src_joint_names):
        raise ValueError(f"data dim {data.shape[-1]} != src joint count {len(src_joint_names)}")
    out = np.zeros(len(tar_joint_names), dtype=data.dtype) if template is None else template.copy()
    for src_idx, name in enumerate(src_joint_names):
        if name in tar_joint_names:
            out[tar_joint_names.index(name)] = data[src_idx]
    return out


def inspect_pipeline_source(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    cls = find_class(tree, "PolicyWrapper")
    method = find_method(cls, "get_pd_target")
    assigns: dict[str, str] = {}
    returns: list[str] = []
    for node in method.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigns[target.id] = ast.unparse(node.value)
        elif isinstance(node, ast.Return):
            returns.append(ast.unparse(node.value))
    return {"assigns": assigns, "returns": returns}


def inspect_policy_source(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    cls = find_class(tree, "UniLabPolicy")
    method = find_method(cls, "get_action")
    source = ast.unparse(method)
    return {
        "updates_last_action": "self.last_action = actions.copy()" in source,
        "applies_action_scale": "actions = actions * self.action_scale" in source,
        "applies_action_clip": "np.clip(actions, -self.action_clip, self.action_clip)" in source,
    }


def run_onnx(policy_path: Path, obs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    session = ort.InferenceSession(policy_path.as_posix(), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise ValueError(f"expected one ONNX input, got {len(inputs)}")
    action = session.run([outputs[0].name], {inputs[0].name: obs[None, :].astype(np.float32)})[0]
    return np.asarray(action).squeeze().astype(np.float32), {
        "input_name": inputs[0].name,
        "input_shape": list(inputs[0].shape),
        "output_name": outputs[0].name,
        "output_shape": list(outputs[0].shape),
        "providers": session.get_providers(),
    }


def audit(robojudo_root: Path) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    paths = {
        "robojudo_xml": robojudo_root / DEFAULT_ROBOJUDO_XML_REL,
        "policy_onnx": robojudo_root / DEFAULT_POLICY_REL,
        "unilab_cfg": robojudo_root / DEFAULT_ROBOJUDO_UNILAB_CFG_REL,
        "env_cfg": robojudo_root / DEFAULT_ROBOJUDO_ENV_CFG_REL,
        "g1_cfg": robojudo_root / DEFAULT_ROBOJUDO_G1_CFG_REL,
        "pipeline": robojudo_root / DEFAULT_ROBOJUDO_PIPELINE_REL,
        "policy_source": robojudo_root / DEFAULT_ROBOJUDO_POLICY_REL,
    }
    details: dict[str, Any] = {"robojudo_root": robojudo_root.as_posix()}

    for name, path in paths.items():
        details[name] = path.as_posix()
        if path.exists():
            _add(checks, "PASS", f"{name}_exists", path.as_posix())
        else:
            _add(checks, "FAIL", f"{name}_exists", path.as_posix())

    default_pos = np.asarray(
        extract_class_assignment(paths["unilab_cfg"], "G1UniLabDoF", "default_pos"),
        dtype=np.float32,
    )
    policy_joint_names = extract_class_assignment(paths["unilab_cfg"], "G1UniLabDoF", "joint_names")
    env_dof_class = extract_g1_unilab_env_dof_class(paths["g1_cfg"])
    env_default_pos = np.asarray(
        extract_class_assignment(paths["env_cfg"], env_dof_class, "default_pos"),
        dtype=np.float32,
    )
    env_joint_names = extract_class_assignment(paths["env_cfg"], env_dof_class, "joint_names")
    position_limits = np.asarray(
        extract_class_assignment(paths["env_cfg"], env_dof_class, "position_limits"),
        dtype=np.float32,
    )
    action_scale = float(extract_class_assignment(paths["unilab_cfg"], "G1UniLabPolicyCfg", "action_scale"))
    action_clip = extract_class_assignment(paths["unilab_cfg"], "G1UniLabPolicyCfg", "action_clip")
    action_beta = float(extract_class_assignment(paths["unilab_cfg"], "G1UniLabPolicyCfg", "action_beta"))
    expected_action_dim = int(
        extract_class_assignment(paths["unilab_cfg"], "G1UniLabPolicyCfg", "expected_action_dim")
    )
    uses_stand_init = g1_unilab_uses_stand_init_qpos(paths["g1_cfg"])
    pipeline_source = inspect_pipeline_source(paths["pipeline"])
    policy_source = inspect_policy_source(paths["policy_source"])

    model = mujoco.MjModel.from_xml_path(paths["robojudo_xml"].as_posix())
    obs = build_first_frame_obs(model, default_pos, gait_phase=np.zeros(2, dtype=np.float32))
    raw_action, onnx_info = run_onnx(paths["policy_onnx"], obs)
    obs_offset_phase = build_first_frame_obs(
        model,
        default_pos,
        gait_phase=np.asarray([0.0, np.pi], dtype=np.float32),
    )
    raw_action_offset_phase, _ = run_onnx(paths["policy_onnx"], obs_offset_phase)
    scaled_action = raw_action * action_scale
    scaled_action_offset_phase = raw_action_offset_phase * action_scale
    pd_target_policy_order = scaled_action + default_pos
    pd_target_offset_phase_policy_order = scaled_action_offset_phase + default_pos
    default_pos_env_order = fit_dof(
        default_pos,
        src_joint_names=policy_joint_names,
        tar_joint_names=env_joint_names,
        template=env_default_pos,
    )
    pd_target_env_order = fit_dof(
        pd_target_policy_order,
        src_joint_names=policy_joint_names,
        tar_joint_names=env_joint_names,
        template=env_default_pos,
    )
    pd_target_offset_phase_env_order = fit_dof(
        pd_target_offset_phase_policy_order,
        src_joint_names=policy_joint_names,
        tar_joint_names=env_joint_names,
        template=env_default_pos,
    )
    action_delta_env_order = fit_dof(
        scaled_action,
        src_joint_names=policy_joint_names,
        tar_joint_names=env_joint_names,
    )
    limit_low = position_limits[:, 0]
    limit_high = position_limits[:, 1]
    lower_violation = pd_target_env_order < (limit_low - TOL)
    upper_violation = pd_target_env_order > (limit_high + TOL)
    violation = lower_violation | upper_violation
    offset_lower_violation = pd_target_offset_phase_env_order < (limit_low - TOL)
    offset_upper_violation = pd_target_offset_phase_env_order > (limit_high + TOL)
    offset_violation = offset_lower_violation | offset_upper_violation

    details.update(
        {
            "uses_g1_unilab_stand_init_qpos": uses_stand_init,
            "onnx": onnx_info,
            "obs_stats": stats(obs),
            "raw_action": raw_action.tolist(),
            "raw_action_stats": stats(raw_action),
            "offset_phase_action": raw_action_offset_phase.tolist(),
            "offset_phase_action_stats": stats(raw_action_offset_phase),
            "scaled_action_stats": stats(scaled_action),
            "offset_phase_scaled_action_stats": stats(scaled_action_offset_phase),
            "default_pos_stats": stats(default_pos),
            "default_pos_env_order": default_pos_env_order.tolist(),
            "pd_target_policy_order": pd_target_policy_order.tolist(),
            "pd_target_policy_stats": stats(pd_target_policy_order),
            "pd_target_env_order": pd_target_env_order.tolist(),
            "pd_target_env_stats": stats(pd_target_env_order),
            "offset_phase_pd_target_env_order": pd_target_offset_phase_env_order.tolist(),
            "offset_phase_pd_target_env_stats": stats(pd_target_offset_phase_env_order),
            "pd_target_minus_default_env_stats": stats(action_delta_env_order),
            "action_scale": action_scale,
            "action_clip": action_clip,
            "action_beta": action_beta,
            "pipeline_source": pipeline_source,
            "policy_source": policy_source,
            "dof_mapping_identity": policy_joint_names == env_joint_names,
            "effective_env_dof_class": env_dof_class,
            "mapped_joint_count": sum(name in env_joint_names for name in policy_joint_names),
            "position_limit_violations": [
                {
                    "index": int(i),
                    "joint": env_joint_names[i],
                    "pd_target": float(pd_target_env_order[i]),
                    "low": float(limit_low[i]),
                    "high": float(limit_high[i]),
                }
                for i in np.where(violation)[0]
            ],
            "offset_phase_position_limit_violations": [
                {
                    "index": int(i),
                    "joint": env_joint_names[i],
                    "pd_target": float(pd_target_offset_phase_env_order[i]),
                    "low": float(limit_low[i]),
                    "high": float(limit_high[i]),
                }
                for i in np.where(offset_violation)[0]
            ],
        }
    )

    if uses_stand_init:
        _add(checks, "PASS", "init_qpos_source", "g1_unilab uses UNILAB_G1_STAND_QPOS")
    else:
        _add(checks, "FAIL", "init_qpos_source", "g1_unilab does not use stand init qpos")

    if obs.shape == (EXPECTED_OBS_DIM,):
        _add(checks, "PASS", "obs_dim_for_onnx", str(obs.shape[0]))
    else:
        _add(checks, "FAIL", "obs_dim_for_onnx", str(obs.shape))

    if raw_action.shape == (EXPECTED_ACTION_DIM,):
        _add(checks, "PASS", "raw_action_dim", str(raw_action.shape[0]))
    else:
        _add(checks, "FAIL", "raw_action_dim", str(raw_action.shape))

    if expected_action_dim == EXPECTED_ACTION_DIM:
        _add(checks, "PASS", "cfg_expected_action_dim", str(expected_action_dim))
    else:
        _add(checks, "FAIL", "cfg_expected_action_dim", str(expected_action_dim))

    if abs(action_scale - 1.0) <= TOL:
        _add(checks, "PASS", "action_scale", f"{action_scale}")
    else:
        _add(checks, "FAIL", "action_scale", f"expected 1.0, got {action_scale}")

    if action_clip is None:
        _add(checks, "PASS", "action_clip", "None")
    else:
        _add(checks, "WARN", "action_clip", f"{action_clip}")

    if abs(action_beta - 1.0) <= TOL:
        _add(checks, "PASS", "action_beta", f"{action_beta}")
    else:
        _add(checks, "WARN", "action_beta", f"{action_beta}")

    assigns = pipeline_source["assigns"]
    returns = pipeline_source["returns"]
    if assigns.get("pd_target") == "action + self.policy.default_pos":
        _add(checks, "PASS", "source_pd_target_formula", "pd_target = action + default_pos")
    else:
        _add(checks, "FAIL", "source_pd_target_formula", str(assigns.get("pd_target")))

    expected_return = "self.actions_adapter.fit(pd_target, template=self.env_dof_cfg.default_pos)"
    if expected_return in returns:
        _add(checks, "PASS", "source_pd_target_adapter", expected_return)
    else:
        _add(checks, "FAIL", "source_pd_target_adapter", str(returns))

    if policy_source["applies_action_scale"]:
        _add(checks, "PASS", "source_action_scale_applied_once", "UniLabPolicy.get_action applies self.action_scale")
    else:
        _add(checks, "FAIL", "source_action_scale_applied_once", "action_scale application not found")

    if details["mapped_joint_count"] == EXPECTED_DOF:
        _add(checks, "PASS", "dof_adapter_maps_all_joints", str(details["mapped_joint_count"]))
    else:
        _add(checks, "FAIL", "dof_adapter_maps_all_joints", str(details["mapped_joint_count"]))

    if details["dof_mapping_identity"]:
        _add(checks, "PASS", "dof_adapter_mapping_identity", "policy and env joint orders match")
    else:
        _add(checks, "WARN", "dof_adapter_mapping_identity", "mapping is complete but order differs")

    max_pd_diff = float(np.max(np.abs((pd_target_env_order - default_pos_env_order) - action_delta_env_order)))
    if max_pd_diff <= TOL:
        _add(checks, "PASS", "pd_target_minus_default_equals_action", f"max_abs={max_pd_diff:.3e}")
    else:
        _add(checks, "FAIL", "pd_target_minus_default_equals_action", f"max_abs={max_pd_diff:.3e}")

    if len(details["position_limit_violations"]) == 0:
        _add(checks, "PASS", "position_limit_violation", "0")
    else:
        _add(
            checks,
            "WARN",
            "position_limit_violation",
            json.dumps(details["position_limit_violations"]),
        )

    if len(details["offset_phase_position_limit_violations"]) == 0:
        _add(checks, "PASS", "offset_phase_position_limit_violation", "0")
    else:
        _add(
            checks,
            "WARN",
            "offset_phase_position_limit_violation",
            json.dumps(details["offset_phase_position_limit_violations"]),
        )

    return checks, details


def print_report(checks: list[Check], details: dict[str, Any], *, json_out: bool) -> None:
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

    print("== Section 6: Action To PD Target ==")
    print(f"robojudo_root: {details['robojudo_root']}")
    print(f"policy_onnx: {details['policy_onnx']}")
    print(f"onnx_input: {details['onnx']['input_name']} {details['onnx']['input_shape']}")
    print(f"onnx_output: {details['onnx']['output_name']} {details['onnx']['output_shape']}")
    print()
    for name in [
        "raw_action_stats",
        "offset_phase_action_stats",
        "scaled_action_stats",
        "pd_target_policy_stats",
        "pd_target_env_stats",
        "offset_phase_pd_target_env_stats",
        "pd_target_minus_default_env_stats",
    ]:
        s = details[name]
        print(
            f"{name}: shape={s['shape']} min={s['min']:.6g} max={s['max']:.6g} "
            f"mean={s['mean']:.6g} std={s['std']:.6g} max_abs={s['max_abs']:.6g}"
        )
    print(f"action_scale: {details['action_scale']}")
    print(f"action_clip: {details['action_clip']}")
    print(f"action_beta: {details['action_beta']}")
    print(f"effective_env_dof_class: {details['effective_env_dof_class']}")
    print(f"dof_mapping_identity: {details['dof_mapping_identity']}")
    print(f"position_limit_violations: {len(details['position_limit_violations'])}")
    if details["position_limit_violations"]:
        for item in details["position_limit_violations"]:
            print(
                f"  {item['joint']}: target={item['pd_target']:.6g}, "
                f"limit=[{item['low']:.6g}, {item['high']:.6g}]"
            )
    print(f"offset_phase_position_limit_violations: {len(details['offset_phase_position_limit_violations'])}")
    if details["offset_phase_position_limit_violations"]:
        for item in details["offset_phase_position_limit_violations"]:
            print(
                f"  {item['joint']}: target={item['pd_target']:.6g}, "
                f"limit=[{item['low']:.6g}, {item['high']:.6g}]"
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
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks, details = audit(args.robojudo_root.resolve())
    print_report(checks, details, json_out=args.json)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
