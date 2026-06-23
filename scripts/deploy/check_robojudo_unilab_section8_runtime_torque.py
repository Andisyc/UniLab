#!/usr/bin/env python3
"""Section 8 checker for closed-loop torque in RoboJudo UniLab deployment.

This script does not import RoboJudo or start the viewer. It reconstructs the
runtime path that matters for the "zero torque" symptom:

    obs -> ONNX action -> pd_target -> external PD torque -> MuJoCo motor ctrl

If this trace produces non-zero torque while the full RoboJudo sim looks like
zero torque, the remaining bug is in the live pipeline/config path rather than
the UniLab policy math.
"""

from __future__ import annotations

import argparse
import ast
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
DEFAULT_UNILAB_SCENE = REPO_ROOT / "src/unilab/assets/robots/g1/scene_flat.xml"

ROOT_QPOS_DIM = 7
EXPECTED_DOF = 29
EXPECTED_OBS_DIM = 98


@dataclass
class Check:
    level: str
    name: str
    detail: str


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level, name, detail))


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


def extract_class_assignment(path: Path, class_name: str, attr_name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    return extract_class_assignment_from_tree(tree, class_name, attr_name)


def stats(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "max_abs": float(np.max(np.abs(arr))),
        "l2": float(np.linalg.norm(arr)),
    }


def get_gravity_orientation_xyzw(quat: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = quat
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2.0 * (-qz * qx + qw * qy)
    gravity[1] = -2.0 * (qz * qy + qw * qx)
    gravity[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity


def build_obs(
    data: mujoco.MjData,
    default_angles: np.ndarray,
    last_action: np.ndarray,
    gait_phase: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    base_quat_xyzw = data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
    obs = np.concatenate(
        [
            data.qvel.astype(np.float32)[3:6] * 0.25,
            -get_gravity_orientation_xyzw(base_quat_xyzw),
            data.qpos.astype(np.float32)[-EXPECTED_DOF:] - default_angles,
            data.qvel.astype(np.float32)[-EXPECTED_DOF:] * 0.05,
            last_action.astype(np.float32),
            command.astype(np.float32),
            gait_phase.astype(np.float32),
        ],
        dtype=np.float32,
    )
    if obs.shape != (EXPECTED_OBS_DIM,):
        raise ValueError(f"obs shape {obs.shape} != ({EXPECTED_OBS_DIM},)")
    return obs


def run_onnx(session: ort.InferenceSession, obs: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return np.asarray(
        session.run([output_name], {input_name: obs[None, :].astype(np.float32)})[0]
    ).squeeze().astype(np.float32)


def step_robojudo_motor(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    session: ort.InferenceSession,
    steps: int,
    default_angles: np.ndarray,
    initial_gait_phase: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
    torque_limits: np.ndarray,
    command: np.ndarray,
) -> dict[str, float]:
    sim_decimation = 10
    policy_dt = model.opt.timestep * sim_decimation
    gait_frequency = 1.5
    last_action = np.zeros(EXPECTED_DOF, dtype=np.float32)
    gait_phase = initial_gait_phase.copy()

    max_torque_l2 = 0.0
    max_ctrl_l2 = 0.0
    min_base_z = float(data.qpos[2])
    first_torque: np.ndarray | None = None
    first_pd_target: np.ndarray | None = None

    print("-- RoboJudo motor XML trace --")
    for step in range(steps):
        obs = build_obs(data, default_angles, last_action, gait_phase, command)
        action = run_onnx(session, obs)
        pd_target = action + default_angles
        torque = (pd_target - data.qpos[-EXPECTED_DOF:].astype(np.float32)) * stiffness
        torque -= data.qvel[-EXPECTED_DOF:].astype(np.float32) * damping
        torque = np.clip(torque, -torque_limits, torque_limits)
        if first_torque is None:
            first_torque = torque.copy()
            first_pd_target = pd_target.copy()

        for _ in range(sim_decimation):
            data.ctrl[:] = torque.astype(np.float64)
            mujoco.mj_step(model, data)

        last_action = action.copy()
        gait_phase = (gait_phase + 2.0 * np.pi * gait_frequency * policy_dt) % (2.0 * np.pi)
        max_torque_l2 = max(max_torque_l2, float(np.linalg.norm(torque)))
        max_ctrl_l2 = max(max_ctrl_l2, float(np.linalg.norm(data.ctrl)))
        min_base_z = min(min_base_z, float(data.qpos[2]))

        print(
            f"step={step:03d} "
            f"base_z={data.qpos[2]:.6f} "
            f"action_l2={np.linalg.norm(action):.6f} "
            f"target_delta_l2={np.linalg.norm(pd_target - data.qpos[-EXPECTED_DOF:]):.6f} "
            f"torque_l2={np.linalg.norm(torque):.6f} "
            f"ctrl_l2={np.linalg.norm(data.ctrl):.6f} "
            f"torque_max_abs={np.max(np.abs(torque)):.6f}"
        )

    assert first_torque is not None and first_pd_target is not None
    print()
    print(f"robojudo_first_pd_target_stats: {stats(first_pd_target)}")
    print(f"robojudo_first_torque_stats: {stats(first_torque)}")
    print(f"robojudo_max_torque_l2: {max_torque_l2:.6f}")
    print(f"robojudo_max_ctrl_l2: {max_ctrl_l2:.6f}")
    print(f"robojudo_min_base_z: {min_base_z:.6f}")
    print()
    return {
        "max_torque_l2": max_torque_l2,
        "max_ctrl_l2": max_ctrl_l2,
        "min_base_z": min_base_z,
    }


def step_unilab_position_scene(
    *,
    scene_path: Path,
    session: ort.InferenceSession,
    steps: int,
    default_angles: np.ndarray,
    initial_gait_phase: np.ndarray,
    command: np.ndarray,
) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(scene_path.as_posix())
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    data.qpos[:] = np.concatenate(
        [
            np.asarray([0.0, 0.0, 0.754, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            default_angles.astype(np.float64),
        ]
    )
    data.qvel[:] = 0.0
    data.ctrl[:] = default_angles.astype(np.float64)
    mujoco.mj_forward(model, data)

    sim_decimation = 10
    policy_dt = model.opt.timestep * sim_decimation
    gait_frequency = 1.5
    last_action = np.zeros(EXPECTED_DOF, dtype=np.float32)
    gait_phase = initial_gait_phase.copy()

    max_force_l2 = 0.0
    max_ctrl_delta_l2 = 0.0
    min_base_z = float(data.qpos[2])

    print("-- UniLab position-actuator scene trace --")
    for step in range(steps):
        obs = build_obs(data, default_angles, last_action, gait_phase, command)
        action = run_onnx(session, obs)
        pd_target = action + default_angles
        for _ in range(sim_decimation):
            data.ctrl[:] = pd_target.astype(np.float64)
            mujoco.mj_step(model, data)

        force_l2 = float(np.linalg.norm(data.qfrc_actuator[-EXPECTED_DOF:]))
        ctrl_delta_l2 = float(np.linalg.norm(pd_target - default_angles))
        max_force_l2 = max(max_force_l2, force_l2)
        max_ctrl_delta_l2 = max(max_ctrl_delta_l2, ctrl_delta_l2)
        min_base_z = min(min_base_z, float(data.qpos[2]))
        last_action = action.copy()
        gait_phase = (gait_phase + 2.0 * np.pi * gait_frequency * policy_dt) % (2.0 * np.pi)

        print(
            f"step={step:03d} "
            f"base_z={data.qpos[2]:.6f} "
            f"action_l2={np.linalg.norm(action):.6f} "
            f"ctrl_delta_l2={ctrl_delta_l2:.6f} "
            f"actuator_force_l2={force_l2:.6f}"
        )

    print()
    print(f"unilab_max_actuator_force_l2: {max_force_l2:.6f}")
    print(f"unilab_max_ctrl_delta_l2: {max_ctrl_delta_l2:.6f}")
    print(f"unilab_min_base_z: {min_base_z:.6f}")
    print()
    return {
        "max_force_l2": max_force_l2,
        "max_ctrl_delta_l2": max_ctrl_delta_l2,
        "min_base_z": min_base_z,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robojudo-root", type=Path, default=DEFAULT_ROBOJUDO_ROOT)
    parser.add_argument("--unilab-scene", type=Path, default=DEFAULT_UNILAB_SCENE)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--initial-gait-phase", type=float, nargs=2, default=None)
    args = parser.parse_args()

    robojudo_root = args.robojudo_root.resolve()
    xml_path = robojudo_root / DEFAULT_ROBOJUDO_XML_REL
    policy_path = robojudo_root / DEFAULT_POLICY_REL
    policy_cfg_path = robojudo_root / DEFAULT_ROBOJUDO_UNILAB_CFG_REL
    env_cfg_path = robojudo_root / DEFAULT_ROBOJUDO_ENV_CFG_REL
    unilab_scene = args.unilab_scene.resolve()

    checks: list[Check] = []
    for name, path in [
        ("robojudo_xml_exists", xml_path),
        ("policy_onnx_exists", policy_path),
        ("policy_cfg_exists", policy_cfg_path),
        ("env_cfg_exists", env_cfg_path),
        ("unilab_scene_exists", unilab_scene),
    ]:
        _add(checks, "PASS" if path.is_file() else "FAIL", name, path.as_posix())
    if any(c.level == "FAIL" for c in checks):
        for check in checks:
            print(f"[{check.level}] {check.name}: {check.detail}")
        return 1

    default_angles = np.asarray(
        extract_class_assignment(policy_cfg_path, "G1UniLabDoF", "default_pos"),
        dtype=np.float32,
    )
    initial_gait_phase = np.asarray(
        extract_class_assignment(policy_cfg_path, "G1UniLabPolicyCfg", "initial_gait_phase"),
        dtype=np.float32,
    )
    if args.initial_gait_phase is not None:
        initial_gait_phase = np.asarray(args.initial_gait_phase, dtype=np.float32)
    stiffness = np.asarray(
        extract_class_assignment(env_cfg_path, "G1UniLabMujocoDoF", "stiffness"),
        dtype=np.float32,
    )
    damping = np.asarray(
        extract_class_assignment(env_cfg_path, "G1UniLabMujocoDoF", "damping"),
        dtype=np.float32,
    )
    torque_limits = np.asarray(
        extract_class_assignment(env_cfg_path, "G1UniLabMujocoDoF", "torque_limits"),
        dtype=np.float32,
    )
    command = np.asarray(args.command, dtype=np.float32)
    if command.shape != (3,):
        raise ValueError(f"--command must contain 3 values, got {command.shape}")

    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    data.qpos[:] = np.concatenate(
        [
            np.asarray([0.0, 0.0, 0.754, 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            default_angles.astype(np.float64),
        ]
    )
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    session = ort.InferenceSession(policy_path.as_posix(), providers=["CPUExecutionProvider"])
    last_action = np.zeros(EXPECTED_DOF, dtype=np.float32)
    gait_phase = initial_gait_phase.copy()

    print("== Section 8: Runtime Torque Trace ==")
    print(f"robojudo_root: {robojudo_root}")
    print(f"xml: {xml_path}")
    print(f"policy: {policy_path}")
    print(f"unilab_scene: {unilab_scene}")
    print(f"initial_base_z: {data.qpos[2]:.6f}")
    print(f"command: {command.tolist()}")
    print(f"initial_gait_phase: {initial_gait_phase.tolist()}")
    print(f"stiffness_stats: {stats(stiffness)}")
    print(f"damping_stats: {stats(damping)}")
    print(f"torque_limit_stats: {stats(torque_limits)}")
    print()

    robojudo_trace = step_robojudo_motor(
        model=model,
        data=data,
        session=session,
        steps=args.steps,
        default_angles=default_angles,
        initial_gait_phase=initial_gait_phase,
        stiffness=stiffness,
        damping=damping,
        torque_limits=torque_limits,
        command=command,
    )
    unilab_trace = step_unilab_position_scene(
        scene_path=unilab_scene,
        session=session,
        steps=args.steps,
        default_angles=default_angles,
        initial_gait_phase=initial_gait_phase,
        command=command,
    )

    if robojudo_trace["max_torque_l2"] < 1.0e-5 or robojudo_trace["max_ctrl_l2"] < 1.0e-5:
        _add(checks, "FAIL", "runtime_torque_nonzero", f"max_torque_l2={robojudo_trace['max_torque_l2']:.3e}, max_ctrl_l2={robojudo_trace['max_ctrl_l2']:.3e}")
    else:
        _add(checks, "PASS", "runtime_torque_nonzero", f"max_torque_l2={robojudo_trace['max_torque_l2']:.3f}, max_ctrl_l2={robojudo_trace['max_ctrl_l2']:.3f}")

    if unilab_trace["max_force_l2"] < 1.0e-5:
        _add(checks, "FAIL", "unilab_native_force_nonzero", f"max_force_l2={unilab_trace['max_force_l2']:.3e}")
    else:
        _add(checks, "PASS", "unilab_native_force_nonzero", f"max_force_l2={unilab_trace['max_force_l2']:.3f}")

    height_gap = robojudo_trace["min_base_z"] - unilab_trace["min_base_z"]
    if abs(height_gap) > 0.05:
        _add(checks, "WARN", "robojudo_unilab_height_gap", f"robojudo_min_z={robojudo_trace['min_base_z']:.3f}, unilab_min_z={unilab_trace['min_base_z']:.3f}, gap={height_gap:.3f}")
    else:
        _add(checks, "PASS", "robojudo_unilab_height_gap", f"robojudo_min_z={robojudo_trace['min_base_z']:.3f}, unilab_min_z={unilab_trace['min_base_z']:.3f}, gap={height_gap:.3f}")

    print()
    fail_count = sum(c.level == "FAIL" for c in checks)
    warn_count = sum(c.level == "WARN" for c in checks)
    pass_count = sum(c.level == "PASS" for c in checks)
    for check in checks:
        print(f"[{check.level}] {check.name}: {check.detail}")
    print(f"\nsummary: {fail_count} fail(s), {warn_count} warning(s), {pass_count} pass(es)")
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
