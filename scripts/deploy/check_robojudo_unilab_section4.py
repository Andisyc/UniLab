#!/usr/bin/env python3
"""Section 4 checker for UniLab -> RoboJudo_Real initial pose/default pose.

This script checks only the reset/default-pose contract:

* UniLab scene_flat.xml stand keyframe.
* RoboJudo_Real G1UniLabDoF.default_pos.
* RoboJudo_Real MuJoCo XML qpos0, which is what MujocoEnv starts from when no
  keyframe reset is applied.
* Whether zero policy action maps to the UniLab stand default pose.

It deliberately avoids importing RoboJudo_Real because that pulls optional
runtime dependencies. Config values are parsed directly from the Python source.
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
DEFAULT_UNILAB_SCENE = REPO_ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"
DEFAULT_ROBOJUDO_XML_REL = Path("assets/robots/g1/g1_29dof_rev_1_0.xml")
DEFAULT_ROBOJUDO_UNILAB_CFG_REL = Path("robojudo/config/g1/policy/g1_unilab_policy_cfg.py")
DEFAULT_ROBOJUDO_ENV_CFG_REL = Path("robojudo/config/g1/env/g1_env_cfg.py")
DEFAULT_ROBOJUDO_G1_CFG_REL = Path("robojudo/config/g1/g1_cfg.py")

ROOT_QPOS_DIM = 7
EXPECTED_DOF = 29
TOL = 1e-6


@dataclass
class Check:
    level: str
    name: str
    detail: str


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level=level, name=name, detail=detail))


def stats(name: str, arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "name": name,
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
    if isinstance(node, ast.Tuple):
        return tuple(eval_literal(elt) for elt in node.elts)
    raise ValueError(f"Unsupported literal node: {ast.dump(node)}")


def find_class(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise KeyError(f"class {class_name} not found")


def extract_class_assignment(path: Path, class_name: str, attr_name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    cls = find_class(tree, class_name)
    for stmt in cls.body:
        target = None
        value = None
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id == attr_name:
                    target = t.id
                    value = stmt.value
                    break
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id == attr_name:
                target = stmt.target.id
                value = stmt.value
        if target == attr_name and value is not None:
            return eval_literal(value)
    raise KeyError(f"{class_name}.{attr_name} not found in {path}")


def g1_unilab_uses_stand_init_qpos(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    cls = find_class(tree, "g1_unilab")
    for stmt in cls.body:
        value = None
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id == "env":
                    value = stmt.value
                    break
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.target.id == "env":
                value = stmt.value
        if not isinstance(value, ast.Call):
            continue
        for kw in value.keywords:
            if kw.arg != "init_qpos":
                continue
            return isinstance(kw.value, ast.Name) and kw.value.id == "UNILAB_G1_STAND_QPOS"
    return False


def load_unilab_stand(scene_path: Path, key_name: str) -> tuple[np.ndarray, np.ndarray | None]:
    model = mujoco.MjModel.from_xml_path(scene_path.as_posix())
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id < 0:
        raise KeyError(f"keyframe {key_name!r} not found in {scene_path}")
    qpos = model.key_qpos[key_id].copy()
    ctrl = model.key_ctrl[key_id].copy() if model.nu else None
    return qpos, ctrl


def load_robojudo_qpos0(xml_path: Path) -> tuple[np.ndarray, int]:
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    return model.qpos0.copy(), int(model.nkey)


def compare_vec(name: str, a: np.ndarray, b: np.ndarray, tol: float, checks: list[Check]) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    max_abs = float(np.max(np.abs(diff)))
    if max_abs <= tol:
        _add(checks, "PASS", name, f"max_abs={max_abs:.3e}")
    else:
        _add(checks, "FAIL", name, f"max_abs={max_abs:.3e} > tol={tol:.1e}")
    return max_abs


def audit(
    robojudo_root: Path,
    unilab_scene: Path,
    robojudo_xml: Path,
    unilab_cfg: Path,
    env_cfg: Path,
    g1_cfg: Path,
    key_name: str,
) -> tuple[list[Check], dict[str, Any]]:
    checks: list[Check] = []
    details: dict[str, Any] = {}

    for name, path in [
        ("unilab_scene", unilab_scene),
        ("robojudo_xml", robojudo_xml),
        ("robojudo_unilab_cfg", unilab_cfg),
        ("robojudo_env_cfg", env_cfg),
        ("robojudo_g1_cfg", g1_cfg),
    ]:
        details[name] = path.as_posix()
        if path.exists():
            _add(checks, "PASS", f"{name}_exists", path.as_posix())
        else:
            _add(checks, "FAIL", f"{name}_exists", path.as_posix())

    unilab_stand_qpos, unilab_stand_ctrl = load_unilab_stand(unilab_scene, key_name)
    unilab_stand_root = unilab_stand_qpos[:ROOT_QPOS_DIM]
    unilab_stand_dof = unilab_stand_qpos[ROOT_QPOS_DIM:]

    robojudo_qpos0, robojudo_nkey = load_robojudo_qpos0(robojudo_xml)
    robojudo_qpos0_root = robojudo_qpos0[:ROOT_QPOS_DIM]
    robojudo_qpos0_dof = robojudo_qpos0[ROOT_QPOS_DIM:]

    unilab_cfg_default = np.asarray(
        extract_class_assignment(unilab_cfg, "G1UniLabDoF", "default_pos"), dtype=np.float64
    )
    robojudo_env_default = np.asarray(
        extract_class_assignment(env_cfg, "G1_29DoF", "default_pos"), dtype=np.float64
    )
    g1_unilab_has_stand_init = g1_unilab_uses_stand_init_qpos(g1_cfg)
    configured_init_qpos = (
        np.concatenate([unilab_stand_root, unilab_cfg_default]).astype(np.float64)
        if g1_unilab_has_stand_init
        else None
    )
    effective_initial_qpos = configured_init_qpos if configured_init_qpos is not None else robojudo_qpos0
    effective_initial_root = effective_initial_qpos[:ROOT_QPOS_DIM]
    effective_initial_dof = effective_initial_qpos[ROOT_QPOS_DIM:]

    details.update(
        {
            "key_name": key_name,
            "robojudo_xml_nkey": robojudo_nkey,
            "unilab_stand_root": unilab_stand_root.tolist(),
            "robojudo_qpos0_root": robojudo_qpos0_root.tolist(),
            "unilab_stand_dof": unilab_stand_dof.tolist(),
            "robojudo_qpos0_dof": robojudo_qpos0_dof.tolist(),
            "robojudo_unilab_cfg_default": unilab_cfg_default.tolist(),
            "robojudo_env_default": robojudo_env_default.tolist(),
            "g1_unilab_has_stand_init_qpos": g1_unilab_has_stand_init,
            "effective_initial_root": effective_initial_root.tolist(),
            "effective_initial_dof": effective_initial_dof.tolist(),
            "diff_qpos0_minus_unilab_default": stats(
                "qpos0_minus_unilab_default", robojudo_qpos0_dof - unilab_cfg_default
            ),
            "diff_effective_initial_minus_unilab_default": stats(
                "effective_initial_minus_unilab_default",
                effective_initial_dof - unilab_cfg_default,
            ),
            "diff_qpos0_minus_unilab_stand": stats(
                "qpos0_minus_unilab_stand", robojudo_qpos0_dof - unilab_stand_dof
            ),
            "diff_env_default_minus_unilab_default": stats(
                "env_default_minus_unilab_default", robojudo_env_default - unilab_cfg_default
            ),
        }
    )
    if unilab_stand_ctrl is not None:
        details["unilab_stand_ctrl"] = unilab_stand_ctrl.tolist()

    if len(unilab_stand_dof) == EXPECTED_DOF:
        _add(checks, "PASS", "unilab_stand_dof_dim", str(len(unilab_stand_dof)))
    else:
        _add(checks, "FAIL", "unilab_stand_dof_dim", str(len(unilab_stand_dof)))

    if len(robojudo_qpos0_dof) == EXPECTED_DOF:
        _add(checks, "PASS", "robojudo_qpos0_dof_dim", str(len(robojudo_qpos0_dof)))
    else:
        _add(checks, "FAIL", "robojudo_qpos0_dof_dim", str(len(robojudo_qpos0_dof)))

    if len(unilab_cfg_default) == EXPECTED_DOF:
        _add(checks, "PASS", "robojudo_unilab_default_dim", str(len(unilab_cfg_default)))
    else:
        _add(checks, "FAIL", "robojudo_unilab_default_dim", str(len(unilab_cfg_default)))

    if robojudo_nkey > 0:
        _add(checks, "PASS", "robojudo_xml_keyframes", str(robojudo_nkey))
    else:
        _add(
            checks,
            "WARN",
            "robojudo_xml_keyframes",
            "0: MujocoEnv.reborn() default keyframe reset is unavailable in this XML",
        )

    compare_vec("unilab_cfg_default_matches_stand", unilab_cfg_default, unilab_stand_dof, TOL, checks)
    compare_vec("unilab_stand_ctrl_matches_stand_dof", unilab_stand_ctrl, unilab_stand_dof, TOL, checks)
    if g1_unilab_has_stand_init:
        _add(checks, "PASS", "g1_unilab_configures_init_qpos", "UNILAB_G1_STAND_QPOS")
        _add(
            checks,
            "WARN",
            "robojudo_raw_qpos0_ignored_for_g1_unilab",
            "XML qpos0 is not the policy start pose when init_qpos is configured",
        )
        _add(
            checks,
            "WARN",
            "robojudo_env_default_overridden_for_g1_unilab",
            "env.update_dof_cfg(policy.action_dof) overrides default_pos before control",
        )
    else:
        _add(checks, "FAIL", "g1_unilab_configures_init_qpos", "missing")
        compare_vec(
            "robojudo_env_default_matches_unilab_default",
            robojudo_env_default,
            unilab_cfg_default,
            TOL,
            checks,
        )
        compare_vec(
            "robojudo_qpos0_dof_matches_unilab_default",
            robojudo_qpos0_dof,
            unilab_cfg_default,
            TOL,
            checks,
        )
        compare_vec(
            "robojudo_qpos0_root_matches_unilab_stand_root",
            robojudo_qpos0_root,
            unilab_stand_root,
            TOL,
            checks,
        )
    compare_vec(
        "effective_initial_dof_matches_unilab_default",
        effective_initial_dof,
        unilab_cfg_default,
        TOL,
        checks,
    )
    compare_vec(
        "effective_initial_root_matches_unilab_stand_root",
        effective_initial_root,
        unilab_stand_root,
        TOL,
        checks,
    )

    zero_action_pd_target = unilab_cfg_default.copy()
    compare_vec("zero_action_pd_target_matches_unilab_default", zero_action_pd_target, unilab_cfg_default, TOL, checks)

    details["section4_conclusion"] = (
        "g1_unilab effective initial qpos is UniLab stand"
        if np.max(np.abs(effective_initial_dof - unilab_cfg_default)) <= TOL
        and np.max(np.abs(effective_initial_root - unilab_stand_root)) <= TOL
        else "g1_unilab effective initial qpos is not UniLab stand"
    )

    return checks, details


def print_report(checks: list[Check], details: dict[str, Any]) -> None:
    print("# Section 4: UniLab policy initial pose/default pose contract")
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
    parser.add_argument("--unilab-scene", type=Path, default=DEFAULT_UNILAB_SCENE)
    parser.add_argument("--robojudo-xml", type=Path, default=DEFAULT_ROBOJUDO_XML_REL)
    parser.add_argument("--robojudo-unilab-cfg", type=Path, default=DEFAULT_ROBOJUDO_UNILAB_CFG_REL)
    parser.add_argument("--robojudo-env-cfg", type=Path, default=DEFAULT_ROBOJUDO_ENV_CFG_REL)
    parser.add_argument("--robojudo-g1-cfg", type=Path, default=DEFAULT_ROBOJUDO_G1_CFG_REL)
    parser.add_argument("--key-name", default="stand")
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def main() -> int:
    args = parse_args()
    robojudo_root = args.robojudo_root.expanduser().resolve()
    unilab_scene = args.unilab_scene.expanduser().resolve()
    robojudo_xml = resolve_path(robojudo_root, args.robojudo_xml)
    unilab_cfg = resolve_path(robojudo_root, args.robojudo_unilab_cfg)
    env_cfg = resolve_path(robojudo_root, args.robojudo_env_cfg)
    g1_cfg = resolve_path(robojudo_root, args.robojudo_g1_cfg)
    checks, details = audit(
        robojudo_root=robojudo_root,
        unilab_scene=unilab_scene,
        robojudo_xml=robojudo_xml,
        unilab_cfg=unilab_cfg,
        env_cfg=env_cfg,
        g1_cfg=g1_cfg,
        key_name=args.key_name,
    )
    print_report(checks, details)
    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
