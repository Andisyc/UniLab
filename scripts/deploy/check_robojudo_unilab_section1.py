#!/usr/bin/env python3
"""Section 1 checker for UniLab policy deployment into RoboJudo_Real.

This script checks only the artifact/loading contract:

* Which ONNX file RoboJudo_Real is expected to load.
* Whether the file exists and can be inspected.
* ONNX input/output names, shapes, and last dimensions.
* Which UniLab run_config is associated with the artifact.
* Whether that run_config implies raw or normalized policy input.

It deliberately does not construct RoboJudo observations or run MuJoCo. Those
belong to later control-list sections.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOJUDO_ROOT = REPO_ROOT.parent / "RoboJudo_Real"
DEFAULT_POLICY_REL = Path("assets/models/g1/unilab/g1_walk_flat/policy.onnx")
EXPECTED_INPUT_DIM = 98
EXPECTED_OUTPUT_DIM = 29


@dataclass
class Check:
    level: str
    name: str
    detail: str


def _add(checks: list[Check], level: str, name: str, detail: str) -> None:
    checks.append(Check(level=level, name=name, detail=detail))


def _last_dim(shape: Any) -> int | None:
    if not shape:
        return None
    dim = shape[-1]
    if isinstance(dim, int):
        return dim
    if isinstance(dim, str) and dim.isdigit():
        return int(dim)
    return None


def _inspect_with_onnxruntime(path: Path) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
    return {
        "inspector": "onnxruntime",
        "inputs": [
            {"name": i.name, "shape": list(i.shape), "type": i.type} for i in session.get_inputs()
        ],
        "outputs": [
            {"name": o.name, "shape": list(o.shape), "type": o.type} for o in session.get_outputs()
        ],
        "providers": session.get_providers(),
    }


def _tensor_type_shape(value_info: Any) -> list[Any]:
    tensor_type = value_info.type.tensor_type
    dims: list[Any] = []
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def _inspect_with_onnx(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(path.as_posix())
    return {
        "inspector": "onnx",
        "inputs": [
            {"name": i.name, "shape": _tensor_type_shape(i), "type": "tensor"}
            for i in model.graph.input
        ],
        "outputs": [
            {"name": o.name, "shape": _tensor_type_shape(o), "type": "tensor"}
            for o in model.graph.output
        ],
        "providers": [],
    }


def inspect_onnx(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return _inspect_with_onnxruntime(path), None
    except Exception as ort_exc:  # pragma: no cover - depends on local deps
        try:
            return _inspect_with_onnx(path), None
        except Exception as onnx_exc:  # pragma: no cover - depends on local deps
            return None, (
                "Cannot inspect ONNX. onnxruntime error: "
                f"{type(ort_exc).__name__}: {ort_exc}; onnx error: "
                f"{type(onnx_exc).__name__}: {onnx_exc}"
            )


def find_run_configs(policy_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    direct = policy_dir / "run_config.json"
    if direct.is_file():
        candidates.append(direct)
    candidates.extend(sorted(policy_dir.glob("*/run_config.json")))
    return sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_nested(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def summarize_run_config(path: Path) -> dict[str, Any]:
    data = load_json(path)
    run_summary_path = path.with_name("run_summary.json")
    summary_data = load_json(run_summary_path) if run_summary_path.is_file() else {}
    return {
        "path": path.as_posix(),
        "run_summary": run_summary_path.as_posix() if run_summary_path.is_file() else None,
        "algo": _get_nested(data, ["run", "algo"]),
        "task": _get_nested(data, ["run", "task"]),
        "sim_backend": _get_nested(data, ["run", "sim_backend"]),
        "checkpoint": _get_nested(data, ["run", "checkpoint"]),
        "last_checkpoint": _get_nested(data, ["run", "last_checkpoint"])
        or _get_nested(data, ["summary", "last_checkpoint"])
        or summary_data.get("last_checkpoint"),
        "obs_normalization": _get_nested(data, ["config", "algo", "obs_normalization"]),
        "export_onnx": _get_nested(data, ["config", "training", "export_onnx"]),
        "action_scale": _get_nested(data, ["config", "env", "control_config", "action_scale"]),
        "gait_frequency": _get_nested(data, ["config", "reward", "gait_frequency"]),
        "git_commit": _get_nested(data, ["run", "git", "commit"]),
        "git_branch": _get_nested(data, ["run", "git", "branch"]),
        "dirty": _get_nested(data, ["run", "git", "dirty"]),
    }


def infer_input_space(run_summary: dict[str, Any] | None) -> str:
    if not run_summary:
        return "unknown_no_run_config"
    obs_norm = run_summary.get("obs_normalization")
    algo = str(run_summary.get("algo") or "").lower()
    if obs_norm is True and algo in {"sac", "flashsac", "td3"}:
        return "likely_normalized_obs"
    if obs_norm is True:
        return "normalization_enabled_check_export_path"
    if obs_norm is False:
        return "likely_raw_obs"
    return "unknown_obs_normalization_missing"


def check_contract(
    policy_path: Path,
    onnx_info: dict[str, Any] | None,
    run_summary: dict[str, Any] | None,
) -> list[Check]:
    checks: list[Check] = []

    if policy_path.is_file():
        _add(checks, "PASS", "policy_file_exists", policy_path.as_posix())
    else:
        _add(checks, "FAIL", "policy_file_missing", policy_path.as_posix())
        return checks

    if onnx_info is None:
        _add(checks, "FAIL", "onnx_inspection", "No ONNX inspector succeeded.")
    else:
        inputs = onnx_info["inputs"]
        outputs = onnx_info["outputs"]
        _add(checks, "PASS", "onnx_inspector", str(onnx_info["inspector"]))

        if len(inputs) == 1:
            _add(checks, "PASS", "onnx_input_count", "1")
        else:
            _add(checks, "FAIL", "onnx_input_count", f"expected 1, got {len(inputs)}")

        if outputs:
            _add(checks, "PASS", "onnx_output_count", str(len(outputs)))
        else:
            _add(checks, "FAIL", "onnx_output_count", "expected at least 1, got 0")

        if inputs:
            in_dim = _last_dim(inputs[0].get("shape"))
            if in_dim == EXPECTED_INPUT_DIM:
                _add(checks, "PASS", "onnx_input_last_dim", str(in_dim))
            else:
                _add(
                    checks,
                    "FAIL",
                    "onnx_input_last_dim",
                    f"expected {EXPECTED_INPUT_DIM}, got {in_dim}",
                )

        if outputs:
            out_dim = _last_dim(outputs[0].get("shape"))
            if out_dim == EXPECTED_OUTPUT_DIM:
                _add(checks, "PASS", "onnx_output_last_dim", str(out_dim))
            else:
                _add(
                    checks,
                    "FAIL",
                    "onnx_output_last_dim",
                    f"expected {EXPECTED_OUTPUT_DIM}, got {out_dim}",
                )

    if run_summary is None:
        _add(checks, "WARN", "run_config", "No run_config.json found beside policy artifact.")
    else:
        _add(checks, "PASS", "run_config", run_summary["path"])
        if run_summary.get("task") == "G1WalkFlat":
            _add(checks, "PASS", "run_task", "G1WalkFlat")
        else:
            _add(checks, "WARN", "run_task", str(run_summary.get("task")))

        input_space = infer_input_space(run_summary)
        if input_space == "likely_normalized_obs":
            _add(
                checks,
                "WARN",
                "onnx_input_space",
                "run_config has obs_normalization=true; Section 2 must verify normalized obs feed.",
            )
        elif input_space == "likely_raw_obs":
            _add(checks, "PASS", "onnx_input_space", "obs_normalization=false")
        else:
            _add(checks, "WARN", "onnx_input_space", input_space)

    return checks


def print_report(
    policy_path: Path,
    policy_dir: Path,
    run_configs: list[Path],
    onnx_info: dict[str, Any] | None,
    onnx_error: str | None,
    run_summary: dict[str, Any] | None,
    checks: list[Check],
) -> None:
    print("# Section 1: UniLab policy artifact/loading contract")
    print(f"policy_path: {policy_path}")
    print(f"policy_dir:  {policy_dir}")
    print()

    if run_configs:
        print("run_config candidates newest-first:")
        for path in run_configs:
            print(f"  - {path}")
    else:
        print("run_config candidates newest-first: <none>")
    print()

    if onnx_info is not None:
        print(f"onnx_inspector: {onnx_info['inspector']}")
        print(f"onnx_inputs:  {json.dumps(onnx_info['inputs'], ensure_ascii=False)}")
        print(f"onnx_outputs: {json.dumps(onnx_info['outputs'], ensure_ascii=False)}")
        if onnx_info.get("providers"):
            print(f"onnx_providers: {onnx_info['providers']}")
    else:
        print("onnx_inspector: <failed>")
        print(f"onnx_error: {onnx_error}")
    print()

    if run_summary is not None:
        print("selected_run_config_summary:")
        print(json.dumps(run_summary, indent=2, ensure_ascii=False))
        print(f"inferred_input_space: {infer_input_space(run_summary)}")
    else:
        print("selected_run_config_summary: <none>")
        print("inferred_input_space: unknown_no_run_config")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robojudo_root = args.robojudo_root.expanduser().resolve()

    policy_path = args.policy_path or DEFAULT_POLICY_REL
    if not policy_path.is_absolute():
        policy_path = robojudo_root / policy_path
    policy_path = policy_path.resolve()
    policy_dir = policy_path.parent

    onnx_info, onnx_error = (None, None)
    if policy_path.is_file():
        onnx_info, onnx_error = inspect_onnx(policy_path)

    if args.run_config is not None:
        run_config = args.run_config
        if not run_config.is_absolute():
            run_config = robojudo_root / run_config
        run_configs = [run_config.resolve()] if run_config.exists() else []
    else:
        run_configs = find_run_configs(policy_dir)

    run_summary = summarize_run_config(run_configs[0]) if run_configs else None
    checks = check_contract(policy_path, onnx_info, run_summary)
    print_report(policy_path, policy_dir, run_configs, onnx_info, onnx_error, run_summary, checks)

    return 1 if any(check.level == "FAIL" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
