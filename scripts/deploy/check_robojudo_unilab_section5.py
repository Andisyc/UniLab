#!/usr/bin/env python3
"""Section 5 checker for UniLab -> RoboJudo_Real command and gait phase.

This script checks the deploy-time contract for:

* command range and joystick axis mapping,
* optional keyboard command saturation risk,
* gait frequency and phase delta,
* dry-run phase freeze,
* reset-time gait phase compatibility with UniLab offset_phase training.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOJUDO_ROOT = REPO_ROOT.parent / "RoboJudo_Real"
DEFAULT_POLICY_CFG_REL = Path("robojudo/config/g1/policy/g1_unilab_policy_cfg.py")
DEFAULT_POLICY_SRC_REL = Path("robojudo/policy/unilab_policy.py")
DEFAULT_PIPELINE_SRC_REL = Path("robojudo/pipeline/rl_pipeline.py")
DEFAULT_UTIL_SRC_REL = Path("robojudo/utils/util_func.py")
DEFAULT_RUN_CONFIG_GLOB = (
    "assets/models/g1/unilab/g1_walk_flat/*_mujoco/run_config.json"
)
DEFAULT_FLASH_SAC_YAML = Path("conf/offpolicy/task/flashsac/g1_walk_flat/mujoco.yaml")
DEFAULT_COMMANDS_SRC = Path("src/unilab/envs/locomotion/common/commands.py")

TOL = 1.0e-6


@dataclass
class Check:
    level: str
    name: str
    detail: str


def add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level=level, name=name, detail=detail))


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
    if isinstance(node, ast.Call):
        for keyword in node.keywords:
            if keyword.arg == "default_factory" and isinstance(keyword.value, ast.Lambda):
                return eval_literal(keyword.value.body)
    raise ValueError(f"cannot evaluate literal AST node: {ast.dump(node)}")


def extract_class_assignment(source: str, class_name: str, attr_name: str) -> Any:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            target: ast.AST | None = None
            value: ast.AST | None = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                value = stmt.value
            elif isinstance(stmt, ast.AnnAssign):
                target = stmt.target
                value = stmt.value
            if isinstance(target, ast.Name) and target.id == attr_name and value is not None:
                return eval_literal(value)
    raise KeyError(f"{class_name}.{attr_name} not found")


def extract_commands_vel_limit(source: str) -> list[list[float]]:
    return extract_class_assignment(source, "Commands", "vel_limit")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def nested_get(data: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def find_run_config(robojudo_root: Path) -> Path | None:
    paths = sorted(robojudo_root.glob(DEFAULT_RUN_CONFIG_GLOB))
    return paths[-1] if paths else None


def read_training_snapshot(robojudo_root: Path, fallback_yaml: Path) -> dict[str, Any]:
    run_config = find_run_config(robojudo_root)
    if run_config is not None:
        data = read_json(run_config)
        return {
            "source": str(run_config),
            "gait_phase_init_mode": nested_get(data, ("config", "env", "gait_phase_init_mode")),
            "gait_frequency": nested_get(data, ("config", "reward", "gait_frequency")),
            "ctrl_dt": nested_get(data, ("config", "env", "ctrl_dt")),
            "commands_vel_limit": nested_get(data, ("config", "env", "commands", "vel_limit")),
        }

    text = fallback_yaml.read_text()
    snapshot: dict[str, Any] = {"source": str(fallback_yaml)}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("gait_phase_init_mode:"):
            snapshot["gait_phase_init_mode"] = stripped.split(":", 1)[1].strip().strip('"')
        if stripped.startswith("gait_frequency:"):
            snapshot["gait_frequency"] = float(stripped.split(":", 1)[1].strip())
    return snapshot


def command_remap(command: float, new_range: list[float], old_range: list[float] | None = None) -> float:
    if old_range is None:
        old_range = [-1.0, 0.0, 1.0]
    old_min, old_mid, old_max = old_range
    new_min, new_mid, new_max = new_range
    if abs((command - old_mid) / (old_max - old_min)) < 0.02:
        return float(new_mid)
    scale_neg = (new_mid - new_min) / (old_mid - old_min)
    scale_pos = (new_max - new_mid) / (old_max - old_mid)
    if command < old_mid:
        return float(new_mid + (command - old_mid) * scale_neg)
    return float(new_mid + (command - old_mid) * scale_pos)


def joystick_command(command_maps: list[list[float]], *, left_x: float = 0.0, left_y: float = 0.0, right_x: float = 0.0) -> np.ndarray:
    return np.asarray(
        [
            command_remap(left_y, command_maps[0]),
            command_remap(left_x, command_maps[1]),
            command_remap(right_x, command_maps[2]),
        ],
        dtype=np.float64,
    )


def keyboard_command(command_maps: list[list[float]], key: str, pressed: bool = True) -> np.ndarray:
    commands = np.zeros(3, dtype=np.float64)
    value = float(pressed) * 1.5
    if key == "w":
        commands[0] = command_remap(value, command_maps[0])
    elif key == "s":
        commands[0] = command_remap(-value, command_maps[0])
    elif key == "a":
        commands[1] = command_remap(-value, command_maps[1])
    elif key == "d":
        commands[1] = command_remap(value, command_maps[1])
    elif key == "e":
        commands[2] = command_remap(value, command_maps[2])
    elif key == "q":
        commands[2] = command_remap(-value, command_maps[2])
    else:
        raise ValueError(f"unsupported key: {key}")
    return commands


def wrap_to_pi(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def format_vec(values: np.ndarray | list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    return "[" + ", ".join(f"{v:.6g}" for v in arr.tolist()) + "]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robojudo-root", type=Path, default=DEFAULT_ROBOJUDO_ROOT)
    args = parser.parse_args()

    robojudo_root = args.robojudo_root.resolve()
    policy_cfg_path = robojudo_root / DEFAULT_POLICY_CFG_REL
    policy_src_path = robojudo_root / DEFAULT_POLICY_SRC_REL
    pipeline_src_path = robojudo_root / DEFAULT_PIPELINE_SRC_REL
    util_src_path = robojudo_root / DEFAULT_UTIL_SRC_REL

    checks: list[Check] = []

    policy_cfg_src = policy_cfg_path.read_text()
    policy_src = policy_src_path.read_text()
    pipeline_src = pipeline_src_path.read_text()
    util_src = util_src_path.read_text()
    commands_src = (REPO_ROOT / DEFAULT_COMMANDS_SRC).read_text()

    freq = float(extract_class_assignment(policy_cfg_src, "G1UniLabPolicyCfg", "freq"))
    gait_frequency = float(
        extract_class_assignment(policy_cfg_src, "G1UniLabPolicyCfg", "gait_frequency")
    )
    initial_gait_phase = np.asarray(
        extract_class_assignment(policy_cfg_src, "G1UniLabPolicyCfg", "initial_gait_phase"),
        dtype=np.float64,
    )
    command_maps = extract_class_assignment(policy_cfg_src, "G1UniLabPolicyCfg", "command_maps")
    freeze_phase = bool(
        extract_class_assignment(policy_cfg_src, "G1UniLabPolicyCfg", "freeze_phase_during_dry_run")
    )
    unilab_vel_limit = extract_commands_vel_limit(commands_src)

    snapshot = read_training_snapshot(robojudo_root, REPO_ROOT / DEFAULT_FLASH_SAC_YAML)
    snapshot_vel_limit = snapshot.get("commands_vel_limit") or unilab_vel_limit
    training_gait_phase_mode = snapshot.get("gait_phase_init_mode")
    training_gait_frequency = snapshot.get("gait_frequency")

    map_ranges = [[min(m[0], m[2]), max(m[0], m[2])] for m in command_maps]
    expected_ranges = [
        [float(snapshot_vel_limit[0][i]), float(snapshot_vel_limit[1][i])] for i in range(3)
    ]
    if np.allclose(np.asarray(map_ranges), np.asarray(expected_ranges), atol=TOL):
        add(checks, "PASS", "command_map_ranges_match_unilab", f"ranges={map_ranges}")
    else:
        add(
            checks,
            "FAIL",
            "command_map_ranges_match_unilab",
            f"robojudo={map_ranges}, unilab={expected_ranges}",
        )

    zero = joystick_command(command_maps)
    forward = joystick_command(command_maps, left_y=1.0)
    backward = joystick_command(command_maps, left_y=-1.0)
    left_x_pos = joystick_command(command_maps, left_x=1.0)
    left_x_neg = joystick_command(command_maps, left_x=-1.0)
    right_x_pos = joystick_command(command_maps, right_x=1.0)
    right_x_neg = joystick_command(command_maps, right_x=-1.0)

    if np.allclose(zero, np.zeros(3), atol=TOL):
        add(checks, "PASS", "zero_joystick_maps_to_zero_command", format_vec(zero))
    else:
        add(checks, "FAIL", "zero_joystick_maps_to_zero_command", format_vec(zero))

    expected_samples = {
        "forward_left_y_positive": (forward, np.asarray([1.0, 0.0, 0.0])),
        "backward_left_y_negative": (backward, np.asarray([-0.6, 0.0, 0.0])),
        "left_x_positive_maps_negative_vy": (left_x_pos, np.asarray([0.0, -0.4, 0.0])),
        "left_x_negative_maps_positive_vy": (left_x_neg, np.asarray([0.0, 0.4, 0.0])),
        "right_x_positive_maps_negative_yaw": (right_x_pos, np.asarray([0.0, 0.0, -0.8])),
        "right_x_negative_maps_positive_yaw": (right_x_neg, np.asarray([0.0, 0.0, 0.8])),
    }
    for name, (actual, expected) in expected_samples.items():
        if np.allclose(actual, expected, atol=TOL):
            add(checks, "PASS", name, format_vec(actual))
        else:
            add(checks, "FAIL", name, f"actual={format_vec(actual)}, expected={format_vec(expected)}")

    deadzone = joystick_command(command_maps, left_y=0.03)
    outside_deadzone = joystick_command(command_maps, left_y=0.05)
    if np.allclose(deadzone, np.zeros(3), atol=TOL) and outside_deadzone[0] > 0.0:
        add(
            checks,
            "PASS",
            "joystick_deadzone_matches_command_remap",
            f"0.03->{format_vec(deadzone)}, 0.05->{format_vec(outside_deadzone)}",
        )
    else:
        add(
            checks,
            "FAIL",
            "joystick_deadzone_matches_command_remap",
            f"0.03->{format_vec(deadzone)}, 0.05->{format_vec(outside_deadzone)}",
        )

    if "def command_remap" in util_src and "np.where" in util_src and "clip" not in util_src:
        add(checks, "PASS", "command_remap_has_no_hidden_clip", "matches current RoboJudo util_func.py")
    else:
        add(checks, "WARN", "command_remap_has_no_hidden_clip", "please inspect util_func.py manually")

    keyboard_samples = {key: keyboard_command(command_maps, key) for key in "wsadeq"}
    keyboard_exceeds: list[str] = []
    lo = np.asarray(expected_ranges, dtype=np.float64)[:, 0]
    hi = np.asarray(expected_ranges, dtype=np.float64)[:, 1]
    for key, cmd in keyboard_samples.items():
        if np.any(cmd < lo - TOL) or np.any(cmd > hi + TOL):
            keyboard_exceeds.append(f"{key}->{format_vec(cmd)}")
    if keyboard_exceeds:
        add(
            checks,
            "WARN",
            "keyboard_command_can_exceed_training_range",
            "; ".join(keyboard_exceeds),
        )
    else:
        add(checks, "PASS", "keyboard_command_within_training_range", str(keyboard_samples))

    if training_gait_frequency is not None and abs(float(training_gait_frequency) - gait_frequency) < TOL:
        add(
            checks,
            "PASS",
            "gait_frequency_matches_training_snapshot",
            f"robojudo={gait_frequency}, training={training_gait_frequency}",
        )
    else:
        add(
            checks,
            "FAIL",
            "gait_frequency_matches_training_snapshot",
            f"robojudo={gait_frequency}, training={training_gait_frequency}",
        )

    dt = 1.0 / freq
    phase_delta = 2.0 * math.pi * gait_frequency * dt
    expected_delta = 2.0 * math.pi * 1.5 * 0.02
    if abs(phase_delta - expected_delta) < TOL:
        add(checks, "PASS", "phase_delta_matches_unilab_formula", f"{phase_delta:.9f}")
    else:
        add(
            checks,
            "FAIL",
            "phase_delta_matches_unilab_formula",
            f"actual={phase_delta:.9f}, expected={expected_delta:.9f}",
        )

    dry_phase_before = initial_gait_phase.copy()
    dry_phase_after = dry_phase_before.copy() if freeze_phase else (dry_phase_before + phase_delta) % (2.0 * math.pi)
    if freeze_phase and "[UNILAB_FREEZE_PHASE]" in pipeline_src and np.allclose(dry_phase_after, dry_phase_before):
        add(checks, "PASS", "dry_run_freezes_gait_phase", format_vec(dry_phase_after))
    else:
        add(
            checks,
            "FAIL",
            "dry_run_freezes_gait_phase",
            f"freeze_cfg={freeze_phase}, sentinel_present={'[UNILAB_FREEZE_PHASE]' in pipeline_src}",
        )

    normal_phase_after = (dry_phase_before + phase_delta) % (2.0 * math.pi)
    expected_normal_phase_after = (initial_gait_phase + phase_delta) % (2.0 * math.pi)
    if np.allclose(normal_phase_after, expected_normal_phase_after, atol=TOL):
        add(checks, "PASS", "normal_step_advances_raw_left_right_phase", format_vec(normal_phase_after))
    else:
        add(checks, "FAIL", "normal_step_advances_raw_left_right_phase", format_vec(normal_phase_after))

    reset_uses_initial = "self.gait_phase = self.initial_gait_phase.copy()" in policy_src
    reset_phase = initial_gait_phase.copy() if reset_uses_initial else np.asarray([math.nan, math.nan])
    if training_gait_phase_mode == "offset_phase":
        phase_offset = wrap_to_pi(float(reset_phase[1] - reset_phase[0]))
        if abs(abs(phase_offset) - math.pi) < 1.0e-5:
            add(
                checks,
                "PASS",
                "reset_gait_phase_matches_unilab_offset_phase",
                f"reset={format_vec(reset_phase)}, offset={phase_offset:.6f}",
            )
        else:
            add(
                checks,
                "FAIL",
                "reset_gait_phase_matches_unilab_offset_phase",
                f"training=offset_phase, robojudo_reset={format_vec(reset_phase)}, offset={phase_offset:.6f}",
            )
    else:
        add(
            checks,
            "WARN",
            "reset_gait_phase_matches_training_mode",
            f"training mode is {training_gait_phase_mode!r}; manual inspection needed",
        )

    print("Section 5: command and gait phase")
    print(f"RoboJudo root: {robojudo_root}")
    print(f"Training snapshot: {snapshot['source']}")
    print(f"command_maps: {command_maps}")
    print(f"unilab_vel_limit: {snapshot_vel_limit}")
    print(f"joystick zero: {format_vec(zero)}")
    print(f"joystick forward/backward: {format_vec(forward)} / {format_vec(backward)}")
    print(f"joystick lateral +LeftX/-LeftX: {format_vec(left_x_pos)} / {format_vec(left_x_neg)}")
    print(f"joystick yaw +RightX/-RightX: {format_vec(right_x_pos)} / {format_vec(right_x_neg)}")
    print(f"phase_delta: {phase_delta:.9f}")
    print(f"normal_phase_after_one_step: {format_vec(normal_phase_after)}")
    print(f"reset_phase_observed_from_source: {format_vec(reset_phase)}")
    print()

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for check in checks:
        counts[check.level] += 1
        print(f"[{check.level}] {check.name}: {check.detail}")

    print()
    print(
        f"summary: {counts['FAIL']} fail(s), {counts['WARN']} warning(s), "
        f"{counts['PASS']} pass(es)"
    )
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
