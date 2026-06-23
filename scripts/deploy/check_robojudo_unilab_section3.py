#!/usr/bin/env python3
"""Section 3 checker for UniLab -> RoboJudo_Real observation construction.

This script checks the first-frame observation contract without starting the
full RoboJudo pipeline or viewer. It reconstructs the same input fields used by
RoboJudo_Real's UniLabPolicy from a MuJoCo reset state:

    ang_vel * 0.25
    gravity
    dof_pos - default_angles
    dof_vel * 0.05
    last_action
    command
    raw gait_phase

The goal is to catch segment boundary, initial value, and unit/scale mistakes
before running a closed-loop policy.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOJUDO_ROOT = REPO_ROOT.parent / "RoboJudo_Real"
DEFAULT_ROBOJUDO_XML_REL = Path("assets/robots/g1/g1_29dof_rev_1_0.xml")
DEFAULT_ROBOJUDO_UNILAB_CFG_REL = Path("robojudo/config/g1/policy/g1_unilab_policy_cfg.py")
DEFAULT_ROBOJUDO_UNILAB_POLICY_REL = Path("robojudo/policy/unilab_policy.py")
DEFAULT_ROBOJUDO_G1_CFG_REL = Path("robojudo/config/g1/g1_cfg.py")

ROOT_QPOS_DIM = 7
EXPECTED_DOF = 29
EXPECTED_OBS_DIM = 98
TOL = 1.0e-6


SEGMENTS: tuple[tuple[str, int], ...] = (
    ("gyro", 3),
    ("gravity", 3),
    ("dof_pos_rel", EXPECTED_DOF),
    ("dof_vel", EXPECTED_DOF),
    ("last_action", EXPECTED_DOF),
    ("command", 3),
    ("gait_phase", 2),
)


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
    raise KeyError(f"{class_name}.{attr_name} not found in {path}")


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


def inspect_unilab_policy_source(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    cls = find_class(tree, "UniLabPolicy")
    method = find_method(cls, "get_observation")

    gravity_uses_getter = False
    concat_terms: list[str] = []
    for node in ast.walk(method):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "gravity" for target in node.targets):
                value_src = ast.unparse(node.value)
                gravity_uses_getter = (
                    not value_src.startswith("-")
                    and "get_gravity_orientation" in value_src
                    and ".astype(np.float32)" in value_src
                )
        if isinstance(node, ast.Call):
            func = node.func
            is_concat = (
                isinstance(func, ast.Attribute)
                and func.attr == "concatenate"
                and isinstance(func.value, ast.Name)
                and func.value.id == "np"
            )
            if is_concat and node.args and isinstance(node.args[0], ast.List):
                concat_terms = [ast.unparse(elt) for elt in node.args[0].elts]
                break

    return {
        "path": path.as_posix(),
        "gravity_uses_getter": gravity_uses_getter,
        "concat_terms": concat_terms,
    }


def get_gravity_orientation_xyzw(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quat
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2.0 * (-qz * qx + qw * qy)
    gravity[1] = -2.0 * (qz * qy + qw * qx)
    gravity[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity


def segment_obs(obs: np.ndarray) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    start = 0
    for name, width in SEGMENTS:
        end = start + width
        values = obs[start:end]
        out[name] = {
            "range": [start, end],
            "values": values.tolist(),
            "stats": stats(values),
        }
        start = end
    return out


def build_first_frame_obs(
    *,
    model: mujoco.MjModel,
    init_qpos: np.ndarray,
    default_angles: np.ndarray,
    initial_gait_phase: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    data = mujoco.MjData(model)
    data.qpos[:] = init_qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    dof_pos = data.qpos.astype(np.float32)[-EXPECTED_DOF:].copy()
    dof_vel = data.qvel.astype(np.float32)[-EXPECTED_DOF:].copy()
    base_quat_xyzw = data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
    base_ang_vel = data.qvel.astype(np.float32)[3:6].copy()
    gravity = get_gravity_orientation_xyzw(base_quat_xyzw)
    last_action = np.zeros(EXPECTED_DOF, dtype=np.float32)
    command = np.zeros(3, dtype=np.float32)
    gait_phase = np.asarray(initial_gait_phase, dtype=np.float32).copy()

    obs = np.concatenate(
        [
            base_ang_vel * 0.25,
            gravity,
            np.asarray(dof_pos - default_angles, dtype=np.float32),
            dof_vel * 0.05,
            last_action,
            command,
            gait_phase,
        ],
        dtype=np.float32,
    )
    fields = {
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "base_quat_xyzw": base_quat_xyzw.tolist(),
        "base_ang_vel": base_ang_vel.tolist(),
        "gravity_segment": gravity.tolist(),
        "dof_pos": dof_pos.tolist(),
        "dof_vel": dof_vel.tolist(),
        "default_angles": default_angles.tolist(),
        "last_action": last_action.tolist(),
        "command": command.tolist(),
        "gait_phase": gait_phase.tolist(),
    }
    return obs, fields


def audit(
    *,
    robojudo_root: Path,
    robojudo_xml: Path,
    unilab_cfg: Path,
    unilab_policy: Path,
    g1_cfg: Path,
) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    details: dict[str, Any] = {
        "robojudo_root": robojudo_root.as_posix(),
        "robojudo_xml": robojudo_xml.as_posix(),
        "robojudo_unilab_cfg": unilab_cfg.as_posix(),
        "robojudo_unilab_policy": unilab_policy.as_posix(),
        "robojudo_g1_cfg": g1_cfg.as_posix(),
    }

    for name, path in [
        ("robojudo_xml", robojudo_xml),
        ("robojudo_unilab_cfg", unilab_cfg),
        ("robojudo_unilab_policy", unilab_policy),
        ("robojudo_g1_cfg", g1_cfg),
    ]:
        if path.exists():
            _add(checks, "PASS", f"{name}_exists", path.as_posix())
        else:
            _add(checks, "FAIL", f"{name}_exists", path.as_posix())

    model = mujoco.MjModel.from_xml_path(robojudo_xml.as_posix())
    source_contract = inspect_unilab_policy_source(unilab_policy)
    qpos0 = model.qpos0.copy()
    default_angles = np.asarray(
        extract_class_assignment(unilab_cfg, "G1UniLabDoF", "default_pos"),
        dtype=np.float32,
    )
    initial_gait_phase = np.asarray(
        extract_class_assignment(unilab_cfg, "G1UniLabPolicyCfg", "initial_gait_phase"),
        dtype=np.float32,
    )
    uses_stand_init = g1_unilab_uses_stand_init_qpos(g1_cfg)
    init_root = np.asarray([0.0, 0.0, 0.754, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    init_qpos = (
        np.concatenate([init_root, default_angles.astype(np.float64)])
        if uses_stand_init
        else qpos0.copy()
    )

    obs, fields = build_first_frame_obs(
        model=model,
        init_qpos=init_qpos,
        default_angles=default_angles,
        initial_gait_phase=initial_gait_phase,
    )
    segments = segment_obs(obs)

    details.update(
        {
            "model_nq": int(model.nq),
            "model_nv": int(model.nv),
            "uses_g1_unilab_stand_init_qpos": uses_stand_init,
            "init_source": "g1_unilab.env.init_qpos" if uses_stand_init else "model.qpos0",
            "source_contract": source_contract,
            "initial_gait_phase": initial_gait_phase.tolist(),
            "obs": obs.tolist(),
            "obs_stats": stats(obs),
            "segments": segments,
            "fields": fields,
        }
    )

    if model.nq == ROOT_QPOS_DIM + EXPECTED_DOF:
        _add(checks, "PASS", "model_qpos_dim", str(model.nq))
    else:
        _add(checks, "FAIL", "model_qpos_dim", f"expected 36, got {model.nq}")

    if model.nv == 6 + EXPECTED_DOF:
        _add(checks, "PASS", "model_qvel_dim", str(model.nv))
    else:
        _add(checks, "FAIL", "model_qvel_dim", f"expected 35, got {model.nv}")

    if uses_stand_init:
        _add(checks, "PASS", "init_qpos_source", "g1_unilab uses UNILAB_G1_STAND_QPOS")
    else:
        _add(checks, "FAIL", "init_qpos_source", "g1_unilab falls back to RoboJudo qpos0")

    expected_terms = [
        "ang_vel * 0.25",
        "gravity",
        "dof_pos_rel",
        "dof_vel * 0.05",
        "np.asarray(self.last_action, dtype=np.float32)",
        "commands",
        "self.gait_phase.astype(np.float32)",
    ]
    if source_contract["gravity_uses_getter"]:
        _add(checks, "PASS", "source_gravity_sign", "gravity = get_gravity_orientation(...)")
    else:
        _add(checks, "FAIL", "source_gravity_sign", "gravity sign does not match UniLab actor obs")

    if source_contract["concat_terms"] == expected_terms:
        _add(checks, "PASS", "source_obs_concat_order", "matches UniLab G1WalkFlat actor obs")
    else:
        _add(
            checks,
            "FAIL",
            "source_obs_concat_order",
            f"expected={expected_terms}, got={source_contract['concat_terms']}",
        )

    expected_width = sum(width for _, width in SEGMENTS)
    if obs.shape == (EXPECTED_OBS_DIM,) and expected_width == EXPECTED_OBS_DIM:
        _add(checks, "PASS", "obs_dim", str(obs.shape[0]))
    else:
        _add(checks, "FAIL", "obs_dim", f"obs={obs.shape}, segment_sum={expected_width}")

    for name, width in SEGMENTS:
        seg_shape = segments[name]["stats"]["shape"]
        if seg_shape == [width]:
            _add(checks, "PASS", f"{name}_segment_dim", f"{segments[name]['range']}")
        else:
            _add(checks, "FAIL", f"{name}_segment_dim", f"expected {width}, got {seg_shape}")

    zero_like_segments = [
        "gyro",
        "dof_pos_rel",
        "dof_vel",
        "last_action",
        "command",
    ]
    for name in zero_like_segments:
        max_abs = float(segments[name]["stats"]["max_abs"])
        if max_abs <= TOL:
            _add(checks, "PASS", f"{name}_first_frame_zero", f"max_abs={max_abs:.3e}")
        else:
            _add(checks, "FAIL", f"{name}_first_frame_zero", f"max_abs={max_abs:.3e}")

    gait_phase = np.asarray(segments["gait_phase"]["values"], dtype=np.float64)
    gait_phase_diff = float(np.max(np.abs(gait_phase - initial_gait_phase)))
    if gait_phase_diff <= TOL:
        _add(
            checks,
            "PASS",
            "gait_phase_first_frame_initial",
            f"values={gait_phase.tolist()}",
        )
    else:
        _add(
            checks,
            "FAIL",
            "gait_phase_first_frame_initial",
            f"values={gait_phase.tolist()}, expected={initial_gait_phase.tolist()}",
        )

    gravity = np.asarray(segments["gravity"]["values"], dtype=np.float64)
    expected_gravity = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    gravity_diff = float(np.max(np.abs(gravity - expected_gravity)))
    if gravity_diff <= TOL:
        _add(checks, "PASS", "gravity_first_frame_upright", f"values={gravity.tolist()}")
    else:
        _add(
            checks,
            "FAIL",
            "gravity_first_frame_upright",
            f"values={gravity.tolist()}, expected={expected_gravity.tolist()}",
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

    print("== Section 3: Observation Construction ==")
    print(f"robojudo_root: {details['robojudo_root']}")
    print(f"init_source: {details['init_source']}")
    print(f"obs_dim: {details['obs_stats']['shape']}")
    print(f"source_policy: {details['source_contract']['path']}")
    print()
    print("Segments:")
    for name, width in SEGMENTS:
        seg = details["segments"][name]
        seg_stats = seg["stats"]
        print(
            f"  {name:12s} range={seg['range']} width={width:2d} "
            f"min={seg_stats['min']:.6g} max={seg_stats['max']:.6g} "
            f"mean={seg_stats['mean']:.6g} std={seg_stats['std']:.6g} "
            f"max_abs={seg_stats['max_abs']:.6g}"
        )
        if name in {"gravity", "command", "gait_phase"}:
            print(f"    values={seg['values']}")
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
    robojudo_root = args.robojudo_root.resolve()
    checks, details = audit(
        robojudo_root=robojudo_root,
        robojudo_xml=robojudo_root / DEFAULT_ROBOJUDO_XML_REL,
        unilab_cfg=robojudo_root / DEFAULT_ROBOJUDO_UNILAB_CFG_REL,
        unilab_policy=robojudo_root / DEFAULT_ROBOJUDO_UNILAB_POLICY_REL,
        g1_cfg=robojudo_root / DEFAULT_ROBOJUDO_G1_CFG_REL,
    )
    print_report(checks, details, json_out=args.json)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
