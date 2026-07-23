#!/usr/bin/env python3
"""Replay one DAgger iteration aggregate boundary without launching a simulator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.algos.torch.distill.data import (  # noqa: E402
    annotate_distillation_dataset_scenario,
    build_multitask_distillation_dataset,
    load_distillation_dataset,
    save_distillation_dataset,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _label_counts(labels: Sequence[str] | None) -> dict[str, int] | None:
    if labels is None:
        return None
    return dict(sorted(Counter(str(label) for label in labels).items()))


def _bad_label_head(
    labels: Sequence[str] | None,
    *,
    expected: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    if labels is None:
        return []
    out: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        value = str(label)
        if value != expected:
            out.append(
                {
                    "index": index,
                    "type": type(label).__name__,
                    "repr": repr(label),
                    "value": value,
                }
            )
            if len(out) >= limit:
                break
    return out


def _tensor_signature(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        return {"type": type(value).__name__, "repr": repr(value)[:256]}
    tensor = value.detach().cpu()
    finite = True
    min_value = None
    max_value = None
    mean_value = None
    if tensor.numel() > 0 and tensor.is_floating_point():
        finite = bool(torch.isfinite(tensor).all())
        min_value = float(tensor.min().item())
        max_value = float(tensor.max().item())
        mean_value = float(tensor.double().mean().item())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(value.device),
        "finite": finite,
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
    }


def _payload_snapshot(payload: Mapping[str, Any], scenario: str | None) -> dict[str, Any]:
    command_intents = payload.get("command_intents")
    scenario_labels = payload.get("scenario_labels")
    metadata = dict(payload.get("metadata") or {})
    expected_intent = None
    if scenario == "walk_flat":
        expected_intent = "active"
    elif scenario == "static_stand":
        expected_intent = "inactive"
    return {
        "num_samples": payload.get("num_samples"),
        "keys": sorted(str(key) for key in payload),
        "student_obs": _tensor_signature(payload.get("student_obs")),
        "teacher_obs": _tensor_signature(payload.get("teacher_obs")),
        "teacher_actions": _tensor_signature(payload.get("teacher_actions")),
        "commands": _tensor_signature(payload.get("commands")),
        "transition_ages": _tensor_signature(payload.get("transition_ages")),
        "command_intent_counts": _label_counts(command_intents),
        "scenario_counts": _label_counts(scenario_labels),
        "role_counts": _label_counts(payload.get("role_labels")),
        "metadata_command_intent_counts": metadata.get("command_intent_counts"),
        "metadata_scenario_counts": metadata.get("scenario_counts"),
        "metadata_workflow_scenario": metadata.get("workflow_scenario"),
        "metadata_filter": metadata.get("command_sample_filter"),
        "bad_command_intent_head": (
            []
            if expected_intent is None
            else _bad_label_head(command_intents, expected=expected_intent)
        ),
    }


def _dataset_snapshot(dataset: Any, scenario: str | None) -> dict[str, Any]:
    expected_intent = None
    if scenario == "walk_flat":
        expected_intent = "active"
    elif scenario == "static_stand":
        expected_intent = "inactive"
    return {
        "num_samples": int(dataset.num_samples),
        "student_obs": _tensor_signature(dataset.student_obs),
        "teacher_obs": _tensor_signature(dataset.teacher_obs),
        "teacher_actions": _tensor_signature(dataset.teacher_actions),
        "commands": _tensor_signature(dataset.commands),
        "transition_ages": _tensor_signature(dataset.transition_ages),
        "command_intent_counts": _label_counts(dataset.command_intents),
        "scenario_counts": _label_counts(dataset.scenario_labels),
        "role_counts": _label_counts(dataset.role_labels),
        "bad_command_intent_head": (
            []
            if expected_intent is None
            else _bad_label_head(dataset.command_intents, expected=expected_intent)
        ),
    }


def _git_identity() -> dict[str, Any]:
    def run(*cmd: str) -> str | None:
        result = subprocess.run(
            list(cmd),
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    return {
        "root_dir": str(ROOT_DIR),
        "head": run("git", "rev-parse", "HEAD"),
        "status_short": run("git", "status", "--short"),
    }


def _manifest_sources_before_iteration(
    manifest: Mapping[str, Any],
    *,
    iteration: int,
) -> list[dict[str, Any]]:
    sources = [dict(item) for item in manifest.get("bootstrap_sources", [])]
    for item in manifest.get("dagger_iterations", []):
        current = int(item.get("iteration", 0))
        if current >= iteration:
            continue
        scenario_artifacts = item.get("scenario_artifacts")
        if scenario_artifacts:
            for artifact in scenario_artifacts:
                source_roles = list(artifact.get("source_roles") or ())
                sources.append(
                    {
                        "path": str(artifact["dataset_path"]),
                        "role": str(source_roles[0]),
                        "scenario": str(artifact["scenario"]),
                        "preserve_row_role_labels": True,
                    }
                )
            continue
        for artifact in item.get("role_artifacts", []):
            sources.append(
                {
                    "path": str(artifact["dataset_path"]),
                    "role": str(artifact["role"]),
                }
            )
    return sources


def _pending_iteration_sources(
    run_dir: Path,
    manifest: Mapping[str, Any],
    *,
    iteration: int,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    scenario_specs = list(manifest.get("scenario_specs") or ())
    if not scenario_specs:
        raise ValueError("manifest has no scenario_specs; cannot rebuild scenario iteration")
    iteration_dir = run_dir / "datasets" / f"dagger_iteration_{iteration}"
    for scenario in scenario_specs:
        source_roles = list(scenario.get("source_roles") or ())
        if not source_roles:
            raise ValueError(f"scenario source_roles missing: {scenario}")
        sources.append(
            {
                "path": str((iteration_dir / f"{scenario['name']}.pt").resolve()),
                "role": str(source_roles[0]),
                "scenario": str(scenario["name"]),
                "preserve_row_role_labels": True,
            }
        )
    return sources


def _source_exists_or_raise(sources: Sequence[Mapping[str, Any]]) -> None:
    missing = [str(source["path"]) for source in sources if not Path(str(source["path"])).is_file()]
    if missing:
        raise FileNotFoundError(f"missing replay source(s): {missing}")


def _first_dimensions(sources: Sequence[Mapping[str, Any]], *, device: str) -> dict[str, int]:
    dataset = load_distillation_dataset(str(sources[0]["path"]), device=device)
    if dataset.teacher_action_dim is None:
        raise ValueError(f"first source has no teacher_actions: {sources[0]['path']}")
    return {
        "student_obs": int(dataset.student_obs_dim),
        "teacher_obs": int(dataset.teacher_obs_dim),
        "teacher_action": int(dataset.teacher_action_dim),
    }


def _inspect_sources(
    sources: Sequence[Mapping[str, Any]],
    *,
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        path = Path(str(source["path"]))
        scenario = None if source.get("scenario") in (None, "") else str(source["scenario"])
        row: dict[str, Any] = {
            "index": index,
            "path": str(path),
            "role": source.get("role"),
            "scenario": scenario,
            "preserve_row_role_labels": bool(source.get("preserve_row_role_labels", False)),
            "exists": path.is_file(),
            "raw": None,
            "loaded": None,
            "annotate": None,
        }
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, Mapping):
                raise TypeError(f"payload must be mapping, got {type(payload).__name__}")
            row["raw"] = _payload_snapshot(payload, scenario)
            dataset = load_distillation_dataset(path, device=device)
            row["loaded"] = _dataset_snapshot(dataset, scenario)
            if scenario is not None:
                annotated = annotate_distillation_dataset_scenario(dataset, scenario)
                row["annotate"] = {
                    "status": "PASS",
                    "result": _dataset_snapshot(annotated, scenario),
                }
            else:
                row["annotate"] = {"status": "SKIP_NO_SCENARIO"}
        except BaseException as error:
            row["annotate"] = {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error_repr": repr(error),
                "traceback": traceback.format_exc(),
            }
        rows.append(row)
    return rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sources = _manifest_sources_before_iteration(manifest, iteration=args.iteration)
    pending_sources = _pending_iteration_sources(run_dir, manifest, iteration=args.iteration)
    sources = [*manifest_sources, *pending_sources]
    _source_exists_or_raise(sources)

    report: dict[str, Any] = {
        "status": "UNKNOWN",
        "run_dir": str(run_dir),
        "iteration": int(args.iteration),
        "device": str(args.device),
        "git": _git_identity(),
        "manifest_completed_dagger_iterations": manifest.get("completed_dagger_iterations"),
        "source_count": len(sources),
        "manifest_source_count": len(manifest_sources),
        "pending_source_count": len(pending_sources),
        "sources": sources,
        "source_inspection": [],
        "dimensions": None,
        "aggregate": None,
    }

    source_inspection = _inspect_sources(sources, device=str(args.device))
    report["source_inspection"] = source_inspection
    failed_sources = [
        row
        for row in source_inspection
        if isinstance(row.get("annotate"), Mapping) and row["annotate"].get("status") == "FAIL"
    ]
    if failed_sources:
        report["status"] = "SOURCE_ANNOTATE_FAILED"
        report["first_failed_source"] = failed_sources[0]
        _write_json(args.report.resolve(), report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(args.report.resolve()),
                    "first_failed_index": failed_sources[0]["index"],
                    "first_failed_path": failed_sources[0]["path"],
                    "first_failed_scenario": failed_sources[0]["scenario"],
                    "first_failed_error": failed_sources[0]["annotate"]["error_repr"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1

    dimensions = _first_dimensions(sources, device=str(args.device))
    report["dimensions"] = dimensions
    try:
        aggregate = build_multitask_distillation_dataset(
            sources,
            expected_student_obs_dim=dimensions["student_obs"],
            expected_teacher_obs_dim=dimensions["teacher_obs"],
            expected_teacher_action_dim=dimensions["teacher_action"],
            device=str(args.device),
        )
        aggregate_snapshot = _dataset_snapshot(aggregate, None)
        output_identity = None
        if args.output is not None:
            output = args.output.resolve()
            save_distillation_dataset(output, aggregate)
            reloaded = load_distillation_dataset(
                output,
                expected_student_obs_dim=dimensions["student_obs"],
                expected_teacher_obs_dim=dimensions["teacher_obs"],
                expected_teacher_action_dim=dimensions["teacher_action"],
                device="cpu",
            )
            output_identity = {
                "path": str(output),
                "num_samples": int(reloaded.num_samples),
                "signature": _dataset_snapshot(reloaded, None),
            }
        report["status"] = "PASS"
        report["aggregate"] = {
            "status": "PASS",
            "snapshot": aggregate_snapshot,
            "output": output_identity,
        }
        _write_json(args.report.resolve(), report)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "report": str(args.report.resolve()),
                    "source_count": len(sources),
                    "num_samples": aggregate_snapshot["num_samples"],
                    "command_intent_counts": aggregate_snapshot["command_intent_counts"],
                    "scenario_counts": aggregate_snapshot["scenario_counts"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        report["status"] = "AGGREGATE_FAILED"
        report["aggregate"] = {
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error_repr": repr(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(args.report.resolve(), report)
        print(
            json.dumps(
                {
                    "status": "AGGREGATE_FAILED",
                    "report": str(args.report.resolve()),
                    "error_type": type(error).__name__,
                    "error_repr": repr(error),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
