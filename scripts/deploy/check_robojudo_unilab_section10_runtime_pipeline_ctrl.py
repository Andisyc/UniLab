#!/usr/bin/env python3
"""Section 10: trace RoboJudo's real g1_unilab pipeline ctrl path.

This checker intentionally exercises RoboJudo_Real's RlPipeline instead of a
hand-written rollout. It replaces the GUI viewer with a dummy object, then
captures obs/action/pd_target/torque/data.ctrl around pipeline.step().
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROBOJUDO_ROOT = Path("/Users/chengyuxuan/ArtiIntComVis/RoboJudo_Real")


@dataclass
class Check:
    status: str
    name: str
    detail: str


def _stats(x: Any) -> dict[str, Any]:
    arr = np.asarray(x, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr)) if arr.size else 0.0,
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "l2": float(np.linalg.norm(arr)) if arr.size else 0.0,
    }


class _DummyCam:
    def __init__(self) -> None:
        self.lookat = np.zeros(3, dtype=np.float64)
        self.distance = 0.0
        self.elevation = 0.0
        self.azimuth = 0.0


class _DummyViewer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.cam = _DummyCam()
        self.is_alive = False
        self._paused = False

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.is_alive = False

    def add_marker(self, *args: Any, **kwargs: Any) -> None:
        return None


class _DummyMessage:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _DummyCallable:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any):
        return _DummyMessage()

    def __getattr__(self, name: str):
        return _DummyCallable()


class _DummyModule(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        value = _DummyCallable()
        setattr(self, name, value)
        return value


def _install_module_stub(name: str, **attrs: Any) -> None:
    module = _DummyModule(name)
    module.__file__ = f"<stub {name}>"
    module.__package__ = name.rpartition(".")[0]
    if "." not in name:
        module.__path__ = []  # type: ignore[attr-defined]
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _install_runtime_stubs() -> None:
    _install_module_stub("mujoco_viewer", MujocoViewer=_DummyViewer)

    unitree_modules = [
        "unitree_sdk2py",
        "unitree_sdk2py.core",
        "unitree_sdk2py.core.channel",
        "unitree_sdk2py.idl",
        "unitree_sdk2py.idl.default",
        "unitree_sdk2py.idl.unitree_go",
        "unitree_sdk2py.idl.unitree_go.msg",
        "unitree_sdk2py.idl.unitree_go.msg.dds_",
        "unitree_sdk2py.idl.unitree_hg",
        "unitree_sdk2py.idl.unitree_hg.msg",
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "unitree_sdk2py.utils",
        "unitree_sdk2py.utils.crc",
        "unitree_sdk2py.utils.thread",
    ]
    for module_name in unitree_modules:
        _install_module_stub(module_name)
    sys.modules["unitree_sdk2py.core.channel"].ChannelFactoryInitialize = lambda *a, **k: None
    sys.modules["unitree_sdk2py.core.channel"].ChannelPublisher = _DummyCallable
    sys.modules["unitree_sdk2py.core.channel"].ChannelSubscriber = _DummyCallable
    sys.modules["unitree_sdk2py.utils.crc"].CRC = _DummyCallable
    sys.modules["unitree_sdk2py.utils.thread"].RecurrentThread = _DummyCallable

    _install_module_stub(
        "unitree_cpp",
        RobotState=_DummyMessage,
        SportState=_DummyMessage,
        UnitreeController=_DummyCallable,
    )


def _import_robojudo(robojudo_root: Path):
    sys.path.insert(0, str(robojudo_root))
    _install_runtime_stubs()

    import robojudo  # noqa: F401
    import robojudo.pipeline
    from robojudo.config.config_manager import ConfigManager

    return robojudo, ConfigManager


def _write_position_actuator_xml(source_xml: str, dof_cfg: Any) -> str:
    source_path = Path(source_xml)
    text = source_path.read_text(encoding="utf-8")
    joints = list(dof_cfg.joint_names)
    kp = list(dof_cfg.stiffness)
    kd = list(dof_cfg.damping)
    force = list(dof_cfg.torque_limits)
    lines = ["  <actuator>"]
    for name, kpi, kdi, limit in zip(joints, kp, kd, force, strict=True):
        lines.append(
            f'    <position name="{name}" joint="{name}" '
            f'kp="{float(kpi):.12g}" kv="{float(kdi):.12g}" '
            f'forcerange="{-float(limit):.12g} {float(limit):.12g}"/>'
        )
    lines.append("  </actuator>")
    new_text, count = re.subn(
        r"\s*<actuator>.*?</actuator>",
        "\n" + "\n".join(lines),
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"Expected exactly one actuator block in {source_path}, replaced {count}")
    target = source_path.with_name(source_path.stem + "_unilab_position_test.xml")
    target.write_text(new_text, encoding="utf-8")
    return target.as_posix()


def _install_position_control_step(env: Any) -> None:
    import mujoco  # type: ignore

    def position_step(pd_target, hand_pose=None):
        if hand_pose is not None:
            print("Hand pose-->", hand_pose)
        env.viewer.cam.lookat = env.data.qpos.astype(np.float32)[:3]
        if env.viewer.is_alive:
            env.viewer.render()
        ctrl = np.asarray(pd_target, dtype=np.float64)
        for _ in range(env.sim_decimation):
            env.data.ctrl[:] = ctrl
            mujoco.mj_step(env.model, env.data)
            env.update(simple=True)
        env.update(simple=False)

    env.step = position_step


def _build_pipeline(
    robojudo_root: Path,
    config_name: str,
    initial_phase: float | None = None,
    position_actuator_test: bool = False,
    position_control_step: bool = False,
    override_xml: Path | None = None,
):
    robojudo, ConfigManager = _import_robojudo(robojudo_root)
    cfg = ConfigManager(config_name=config_name).get_cfg()
    cfg.ctrl = []
    if initial_phase is not None:
        cfg.policy.initial_gait_phase = [float(initial_phase), float(initial_phase + np.pi)]
    if override_xml is not None:
        cfg.env.xml = override_xml.expanduser().resolve().as_posix()
    if position_actuator_test:
        cfg.env.xml = _write_position_actuator_xml(cfg.env.xml, cfg.env.dof)
    pipeline_cls = getattr(robojudo.pipeline, cfg.pipeline_type)
    pipeline = pipeline_cls(cfg=cfg)
    if position_actuator_test or position_control_step:
        _install_position_control_step(pipeline.env)
    return cfg, pipeline


def _capture_manual_forward(pipeline) -> dict[str, Any]:
    pipeline.env.update()
    env_data = pipeline.env.get_data()
    ctrl_data = pipeline.ctrl_manager.get_ctrl_data(env_data)
    obs, extras = pipeline.policy.get_observation(env_data, ctrl_data)
    pd_target = pipeline.policy.get_pd_target(obs)
    torque = (pd_target - pipeline.env.dof_pos) * pipeline.env.stiffness - pipeline.env.dof_vel * pipeline.env.damping
    torque = np.clip(torque, -pipeline.env.torque_limits, pipeline.env.torque_limits)
    sensor_compare: dict[str, Any] = {}
    try:
        import mujoco  # type: ignore

        gyro_id = mujoco.mj_name2id(pipeline.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_gyro")
        up_id = mujoco.mj_name2id(pipeline.env.model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_upvector")
        if gyro_id >= 0:
            adr = int(pipeline.env.model.sensor_adr[gyro_id])
            dim = int(pipeline.env.model.sensor_dim[gyro_id])
            torso_gyro_sensor = np.asarray(pipeline.env.data.sensordata[adr : adr + dim], dtype=np.float32)
            sensor_compare["torso_gyro_sensor"] = torso_gyro_sensor.copy()
            sensor_compare["obs_vs_torso_gyro_sensor_scaled_max_abs"] = float(
                np.max(np.abs(np.asarray(obs[0:3], dtype=np.float32) - torso_gyro_sensor * 0.25))
            )
        if up_id >= 0:
            adr = int(pipeline.env.model.sensor_adr[up_id])
            dim = int(pipeline.env.model.sensor_dim[up_id])
            torso_upvector_sensor = np.asarray(pipeline.env.data.sensordata[adr : adr + dim], dtype=np.float32)
            sensor_compare["torso_upvector_sensor"] = torso_upvector_sensor.copy()
            sensor_compare["obs_vs_minus_torso_upvector_sensor_max_abs"] = float(
                np.max(np.abs(np.asarray(obs[3:6], dtype=np.float32) + torso_upvector_sensor))
            )
    except Exception as exc:
        sensor_compare["sensor_compare_error"] = str(exc)

    return {
        "env_data_obs_source": extras.get("obs_source"),
        "obs": np.asarray(obs).copy(),
        "pd_target": np.asarray(pd_target).copy(),
        "dof_pos": np.asarray(pipeline.env.dof_pos).copy(),
        "dof_vel": np.asarray(pipeline.env.dof_vel).copy(),
        "torque": np.asarray(torque).copy(),
        "data_ctrl_before_step": np.asarray(pipeline.env.data.ctrl).copy(),
        "base_z": float(pipeline.env.base_pos[2]),
        "torso_ang_vel_is_none": env_data.torso_ang_vel is None,
        "torso_quat_is_none": env_data.torso_quat is None,
        "sensor_compare": sensor_compare,
    }


def _run_pipeline_trace(args: argparse.Namespace, root: Path, initial_phase: float | None = None) -> tuple[list[Check], dict[str, float]]:
    checks: list[Check] = []
    cfg, pipeline = _build_pipeline(
        root,
        args.config,
        initial_phase=initial_phase,
        position_actuator_test=args.position_actuator_test,
        position_control_step=args.position_control_step,
        override_xml=args.override_xml,
    )
    phase_detail = "default" if initial_phase is None else f"{initial_phase:.6g}"
    checks.append(
        Check(
            "PASS",
            "pipeline_constructed",
            (
                f"{args.config} -> {type(pipeline).__name__}, "
                f"initial_phase={phase_detail}, "
                f"position_actuator_test={args.position_actuator_test}, "
                f"position_control_step={args.position_control_step}, "
                f"override_xml={args.override_xml}"
            ),
        )
    )

    init_delta = np.asarray(pipeline.env.dof_pos) - np.asarray(pipeline.policy.policy.default_dof_pos)
    checks.append(
        Check(
            "PASS" if np.max(np.abs(init_delta)) < args.pose_tol else "FAIL",
            "post_pipeline_reset_matches_unilab_default_pose",
            f"max_abs={np.max(np.abs(init_delta)):.3e}",
        )
    )

    manual = _capture_manual_forward(pipeline)
    checks.append(
        Check(
            "PASS" if manual["env_data_obs_source"] == "torso" else "FAIL",
            "runtime_obs_source_is_torso",
            str(manual["env_data_obs_source"]),
        )
    )
    checks.append(
        Check(
            "PASS" if not manual["torso_ang_vel_is_none"] and not manual["torso_quat_is_none"] else "FAIL",
            "runtime_torso_fields_present",
            f"torso_ang_vel_is_none={manual['torso_ang_vel_is_none']}, torso_quat_is_none={manual['torso_quat_is_none']}",
        )
    )
    checks.append(
        Check(
            "PASS" if _stats(manual["torque"])["max_abs"] > args.nonzero_tol else "FAIL",
            "manual_pipeline_forward_torque_nonzero",
            json.dumps(_stats(manual["torque"]), sort_keys=True),
        )
    )
    sensor_compare = manual.get("sensor_compare", {})
    if "obs_vs_torso_gyro_sensor_scaled_max_abs" in sensor_compare:
        checks.append(
            Check(
                "PASS"
                if float(sensor_compare["obs_vs_torso_gyro_sensor_scaled_max_abs"]) <= args.sensor_tol
                else "FAIL",
                "runtime_obs_matches_torso_gyro_sensor",
                f"max_abs={float(sensor_compare['obs_vs_torso_gyro_sensor_scaled_max_abs']):.6g}",
            )
        )
    if "obs_vs_minus_torso_upvector_sensor_max_abs" in sensor_compare:
        checks.append(
            Check(
                "PASS"
                if float(sensor_compare["obs_vs_minus_torso_upvector_sensor_max_abs"]) <= args.sensor_tol
                else "FAIL",
                "runtime_obs_matches_minus_torso_upvector_sensor",
                f"max_abs={float(sensor_compare['obs_vs_minus_torso_upvector_sensor_max_abs']):.6g}",
            )
        )

    captured: list[dict[str, np.ndarray]] = []
    original_step = pipeline.env.step

    def wrapped_step(pd_target, hand_pose=None):
        pre_torque = (pd_target - pipeline.env.dof_pos) * pipeline.env.stiffness - pipeline.env.dof_vel * pipeline.env.damping
        pre_torque = np.clip(pre_torque, -pipeline.env.torque_limits, pipeline.env.torque_limits)
        ret = original_step(pd_target, hand_pose)
        captured.append(
            {
                "pd_target": np.asarray(pd_target).copy(),
                "pre_torque": np.asarray(pre_torque).copy(),
                "data_ctrl_after_step": np.asarray(pipeline.env.data.ctrl).copy(),
                "base_z_after_step": np.asarray([pipeline.env.base_pos[2]], dtype=np.float64),
            }
        )
        return ret

    pipeline.env.step = wrapped_step
    for _ in range(args.steps):
        pipeline.step(dry_run=False)

    history_summary: dict[str, float] = {}
    if not captured:
        checks.append(Check("FAIL", "pipeline_step_called_env_step", "captured=0"))
    else:
        height_history = np.asarray([float(item["base_z_after_step"][0]) for item in captured], dtype=np.float64)
        ctrl_l2_history = np.asarray([_stats(item["data_ctrl_after_step"])["l2"] for item in captured], dtype=np.float64)
        history_summary = {
            "base_z_min": float(np.min(height_history)),
            "base_z_last": float(height_history[-1]),
            "base_z_max": float(np.max(height_history)),
            "ctrl_l2_min": float(np.min(ctrl_l2_history)),
            "ctrl_l2_last": float(ctrl_l2_history[-1]),
            "ctrl_l2_max": float(np.max(ctrl_l2_history)),
        }
        checks.append(Check("PASS", "pipeline_step_called_env_step", f"captured={len(captured)}"))
        first = captured[0]
        last = captured[-1]
        checks.append(
            Check(
                "PASS" if _stats(first["pre_torque"])["max_abs"] > args.nonzero_tol else "FAIL",
                "first_pipeline_step_pre_torque_nonzero",
                json.dumps(_stats(first["pre_torque"]), sort_keys=True),
            )
        )
        checks.append(
            Check(
                "PASS" if _stats(first["data_ctrl_after_step"])["max_abs"] > args.nonzero_tol else "FAIL",
                "first_pipeline_step_data_ctrl_nonzero_after_step",
                json.dumps(_stats(first["data_ctrl_after_step"]), sort_keys=True),
            )
        )
        checks.append(
            Check(
                "PASS" if _stats(last["data_ctrl_after_step"])["max_abs"] > args.nonzero_tol else "FAIL",
                "last_pipeline_step_data_ctrl_nonzero_after_step",
                json.dumps(_stats(last["data_ctrl_after_step"]), sort_keys=True),
            )
        )
        checks.append(
            Check(
                "PASS" if float(np.min(ctrl_l2_history)) > args.nonzero_tol else "FAIL",
                "all_pipeline_steps_data_ctrl_nonzero",
                f"ctrl_l2_min={float(np.min(ctrl_l2_history)):.6g}, ctrl_l2_max={float(np.max(ctrl_l2_history)):.6g}",
            )
        )
        checks.append(
            Check(
                "PASS" if float(np.min(height_history)) > args.min_height else "WARN",
                "pipeline_rollout_height",
                (
                    f"min_base_z={float(np.min(height_history)):.6g}, "
                    f"last_base_z={float(height_history[-1]):.6g}, "
                    f"max_base_z={float(np.max(height_history)):.6g}"
                ),
            )
        )
        print("first_pipeline_step:")
        print(json.dumps({k: _stats(v) for k, v in first.items() if k != "base_z_after_step"}, indent=2, sort_keys=True))
        print("last_pipeline_step:")
        print(json.dumps({k: _stats(v) for k, v in last.items()}, indent=2, sort_keys=True))
        print("rollout_history:")
        print(
            json.dumps(
                {
                    "steps": len(captured),
                    **history_summary,
                },
                indent=2,
                sort_keys=True,
            )
        )

    print("manual_forward:")
    printable_manual = {
        "obs_source": manual["env_data_obs_source"],
        "obs": _stats(manual["obs"]),
        "pd_target": _stats(manual["pd_target"]),
        "torque": _stats(manual["torque"]),
        "data_ctrl_before_step": _stats(manual["data_ctrl_before_step"]),
        "base_z": manual["base_z"],
        "sensor_compare": {
            key: (_stats(value) if isinstance(value, np.ndarray) else value)
            for key, value in manual.get("sensor_compare", {}).items()
        },
    }
    print(json.dumps(printable_manual, indent=2, sort_keys=True))

    return checks, history_summary


def run(args: argparse.Namespace) -> int:
    root = Path(args.robojudo_root).expanduser().resolve()
    if not root.exists():
        print(f"RoboJudo root does not exist: {root}")
        return 2

    if args.scan_phases > 0:
        phase_values = np.linspace(0.0, 2.0 * np.pi, num=args.scan_phases, endpoint=False)
        rows = []
        any_fail = False
        for phase in phase_values:
            checks, summary = _run_pipeline_trace(args, root, initial_phase=float(phase))
            any_fail = any(check.status == "FAIL" for check in checks) or any_fail
            rows.append({"phase": float(phase), **summary})
        print("phase_scan_summary:")
        print(json.dumps(rows, indent=2, sort_keys=True))
        stable = [row for row in rows if row.get("base_z_min", 0.0) > args.min_height]
        print(
            f"phase_scan_result: stable={len(stable)}/{len(rows)}, "
            f"best_min_height={max((row.get('base_z_min', 0.0) for row in rows), default=0.0):.6g}"
        )
        return 1 if any_fail else 0

    checks, _ = _run_pipeline_trace(args, root, initial_phase=args.initial_phase)

    fail = 0
    warn = 0
    for check in checks:
        if check.status == "FAIL":
            fail += 1
        elif check.status == "WARN":
            warn += 1
        print(f"[{check.status}] {check.name}: {check.detail}")
    print(f"summary: {fail} fail(s), {warn} warning(s), {len(checks) - fail - warn} pass(es)")
    return 1 if fail else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robojudo-root", default=str(DEFAULT_ROBOJUDO_ROOT))
    parser.add_argument("--config", default="g1_unilab")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--nonzero-tol", type=float, default=1e-6)
    parser.add_argument("--pose-tol", type=float, default=1e-6)
    parser.add_argument("--sensor-tol", type=float, default=1e-5)
    parser.add_argument("--min-height", type=float, default=0.45)
    parser.add_argument("--initial-phase", type=float, default=None)
    parser.add_argument("--scan-phases", type=int, default=0)
    parser.add_argument("--position-actuator-test", action="store_true")
    parser.add_argument("--position-control-step", action="store_true")
    parser.add_argument("--override-xml", type=Path, default=None)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
