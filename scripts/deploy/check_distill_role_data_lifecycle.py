#!/usr/bin/env python3
"""Stress the production distillation Role Data lifecycle without a simulator."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from unilab.algos.torch.distill import (
    DistillationTensorDataset,
    build_multitask_distillation_dataset,
    load_distillation_dataset,
    save_distillation_dataset,
)

_PREFIX = "[distill-native-lifecycle] "
_LABEL_FIELDS = ("role_labels", "command_intents", "scenario_labels")
_TENSOR_FIELDS = (
    "student_obs",
    "teacher_obs",
    "teacher_actions",
    "commands",
    "transition_ages",
    "command_before",
    "command_after",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _current_rss_bytes() -> int | None:
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


def _label_fingerprint(
    dataset: DistillationTensorDataset,
    field_name: str,
) -> tuple[dict[str, int], str | None]:
    labels = getattr(dataset, field_name)
    if labels is None:
        return {}, None
    digest = hashlib.sha256()
    for index, value in enumerate(labels):
        if type(value) is not str:
            raise TypeError(
                f"{field_name}[{index}] must be exact str before normalization, "
                f"got type={type(value).__name__} repr={value!r}"
            )
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    metadata_value = dataset.metadata.get(field_name)
    if metadata_value is not None:
        if not isinstance(metadata_value, list | tuple):
            raise TypeError(
                f"metadata[{field_name!r}] must be list or tuple, "
                f"got {type(metadata_value).__name__}"
            )
        for index, value in enumerate(metadata_value):
            if type(value) is not str:
                raise TypeError(
                    f"metadata[{field_name!r}][{index}] must be exact str, "
                    f"got type={type(value).__name__} repr={value!r}"
                )
        if tuple(metadata_value) != labels:
            raise ValueError(f"top-level {field_name} differs from metadata duplicate")
    return dict(sorted(Counter(labels).items())), digest.hexdigest()


def _tensor_sample_digest(tensor: torch.Tensor) -> str:
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    if flat.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    sample_count = min(int(flat.numel()), 96)
    indices = torch.linspace(0, int(flat.numel()) - 1, steps=sample_count, dtype=torch.int64)
    sample = flat.index_select(0, indices).contiguous()
    raw = sample.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def fingerprint_dataset(dataset: DistillationTensorDataset) -> dict[str, Any]:
    """Return a compact semantic fingerprint without normalizing label objects."""

    label_counts: dict[str, dict[str, int]] = {}
    label_sha256: dict[str, str | None] = {}
    for field_name in _LABEL_FIELDS:
        counts, digest = _label_fingerprint(dataset, field_name)
        label_counts[field_name] = counts
        label_sha256[field_name] = digest

    tensor_fingerprints: dict[str, Any] = {}
    for field_name in _TENSOR_FIELDS:
        tensor = getattr(dataset, field_name)
        if tensor is None:
            tensor_fingerprints[field_name] = None
            continue
        tensor_fingerprints[field_name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sample_sha256": _tensor_sample_digest(tensor),
        }
    return {
        "num_samples": dataset.num_samples,
        "student_obs_dim": dataset.student_obs_dim,
        "teacher_obs_dim": dataset.teacher_obs_dim,
        "teacher_action_dim": dataset.teacher_action_dim,
        "label_counts": label_counts,
        "label_sha256": label_sha256,
        "tensors": tensor_fingerprints,
    }


def _load_sources(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("source manifest must contain a non-empty 'sources' list")
    sources: list[dict[str, Any]] = []
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict):
            raise TypeError(f"sources[{index}] must be a mapping")
        source = dict(raw_source)
        source_path = Path(str(source.get("path", "")))
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source["path"] = str(source_path.resolve())
        if not str(source.get("role", "")):
            raise ValueError(f"sources[{index}] requires a non-empty role")
        sources.append(source)
    return sources


def _build_cycle_dataset(
    *,
    dataset_path: Path | None,
    sources: Sequence[Mapping[str, Any]] | None,
) -> DistillationTensorDataset:
    if dataset_path is not None:
        # Re-enter the production aggregation path even for an already merged
        # incident artifact. This repeats tuple/list flattening and metadata
        # duplication instead of testing only torch.load/torch.save.
        return build_multitask_distillation_dataset(
            [
                {
                    "path": str(dataset_path),
                    "role": "incident_aggregate",
                    "preserve_row_role_labels": True,
                }
            ],
            device="cpu",
        )
    if sources is None:
        raise RuntimeError("dataset path or source manifest is required")
    return build_multitask_distillation_dataset(sources, device="cpu")


def run_lifecycle(
    *,
    work_dir: str | Path,
    cycles: int,
    source_manifest: str | Path | None,
    dataset_path: str | Path | None,
    keep_cycle_outputs: bool,
    report_every: int,
    inject_failure_cycle: int | None = None,
) -> dict[str, Any]:
    """Run repeated production data lifecycle cycles and assert semantic stability."""

    if int(cycles) <= 0:
        raise ValueError(f"cycles must be positive, got {cycles}")
    if int(report_every) <= 0:
        raise ValueError(f"report_every must be positive, got {report_every}")
    if (source_manifest is None) == (dataset_path is None):
        raise ValueError("exactly one of source_manifest or dataset_path is required")

    resolved_work_dir = Path(work_dir).resolve()
    resolved_work_dir.mkdir(parents=True, exist_ok=True)
    resolved_dataset_path = None if dataset_path is None else Path(dataset_path).resolve()
    if resolved_dataset_path is not None and not resolved_dataset_path.is_file():
        raise FileNotFoundError(resolved_dataset_path)
    resolved_manifest = None if source_manifest is None else Path(source_manifest).resolve()
    sources = None if resolved_manifest is None else _load_sources(resolved_manifest)

    started = time.monotonic()
    baseline: dict[str, Any] | None = None
    cycles_completed = 0
    last_output_path: str | None = None
    rss_samples: list[int] = []
    for cycle in range(1, int(cycles) + 1):
        dataset = _build_cycle_dataset(dataset_path=resolved_dataset_path, sources=sources)
        before_save = fingerprint_dataset(dataset)
        if baseline is None:
            baseline = before_save
        elif before_save != baseline:
            raise AssertionError(
                f"cycle {cycle} pre-save fingerprint drifted from baseline: "
                f"before={before_save!r} baseline={baseline!r}"
            )

        cycle_path = resolved_work_dir / f"cycle-{cycle:06d}.pt"
        save_distillation_dataset(cycle_path, dataset)
        del dataset
        gc.collect()

        restored = load_distillation_dataset(cycle_path, device="cpu")
        after_load = fingerprint_dataset(restored)
        if after_load != before_save:
            raise AssertionError(
                f"cycle {cycle} roundtrip fingerprint mismatch: "
                f"before={before_save!r} after={after_load!r}"
            )
        if inject_failure_cycle is not None and cycle == int(inject_failure_cycle):
            raise RuntimeError(f"synthetic lifecycle failure at cycle {cycle}")

        del restored
        gc.collect()
        cycles_completed = cycle
        last_output_path = str(cycle_path)
        rss = _current_rss_bytes()
        if rss is not None:
            rss_samples.append(rss)
        if not keep_cycle_outputs:
            cycle_path.unlink()
            last_output_path = None
        if cycle == 1 or cycle == int(cycles) or cycle % int(report_every) == 0:
            print(
                _PREFIX
                + json.dumps(
                    {
                        "stage": "cycle_complete",
                        "cycle": cycle,
                        "cycles": int(cycles),
                        "num_samples": before_save["num_samples"],
                        "rss_bytes": rss,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    assert baseline is not None
    result = {
        "status": "completed",
        "pid": os.getpid(),
        "cycles_requested": int(cycles),
        "cycles_completed": cycles_completed,
        "duration_seconds": time.monotonic() - started,
        "source_manifest": None if resolved_manifest is None else str(resolved_manifest),
        "dataset_path": None if resolved_dataset_path is None else str(resolved_dataset_path),
        "baseline_fingerprint": baseline,
        "rss_bytes": {
            "first": None if not rss_samples else rss_samples[0],
            "last": None if not rss_samples else rss_samples[-1],
            "peak": None if not rss_samples else max(rss_samples),
        },
        "last_output_path": last_output_path,
    }
    _write_json(resolved_work_dir / "result.json", result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset", type=Path)
    inputs.add_argument("--source-manifest", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=16)
    parser.add_argument("--report-every", type=int, default=8)
    parser.add_argument("--keep-cycle-outputs", action="store_true")
    parser.add_argument(
        "--test-inject-failure-cycle", type=int, default=None, help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run_lifecycle(
            work_dir=args.work_dir,
            cycles=args.cycles,
            source_manifest=args.source_manifest,
            dataset_path=args.dataset,
            keep_cycle_outputs=bool(args.keep_cycle_outputs),
            report_every=args.report_every,
            inject_failure_cycle=args.test_inject_failure_cycle,
        )
    except BaseException as error:
        failure = {
            "status": "failed",
            "pid": os.getpid(),
            "error_type": type(error).__name__,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(Path(args.work_dir).resolve() / "result.json", failure)
        print(_PREFIX + json.dumps(failure, sort_keys=True), file=sys.stderr, flush=True)
        return 2
    print(_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
