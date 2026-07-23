#!/usr/bin/env python3
"""Isolate host, CUDA, and same-GPU multiprocess corruption with numeric canaries."""

from __future__ import annotations

import argparse
import faulthandler
import json
import multiprocessing as mp
import os
import platform
import queue
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class WorkerConfig:
    """One isolated numerical worker configuration."""

    stage: str
    device: str
    worker_index: int
    seconds: float
    iterations: int
    matrix_size: int
    batch_size: int
    allocation_mib: int
    canary_size: int
    seed: int
    serialization_interval: int
    inject_failure_iteration: int | None = None


@dataclass(frozen=True)
class PythonCanary:
    """Process-private Python references that no numerical operation may mutate."""

    token: str
    labels: tuple[str, ...]
    callback: Any
    callbacks: tuple[Any, ...]
    builtin_identities: tuple[Any, ...]


class PythonObjectIntegrityError(RuntimeError):
    """Raised when a process-private Python object identity changes."""


def _make_python_canary(*, size: int, seed: int, worker_index: int) -> PythonCanary:
    if int(size) <= 0:
        raise ValueError(f"canary size must be positive, got {size}")
    token = f"walk_to_stop::{os.getpid()}::{seed}::{worker_index}::{time.time_ns()}"

    def callback(value: int) -> int:
        return value + 1

    return PythonCanary(
        token=token,
        labels=(token,) * int(size),
        callback=callback,
        callbacks=(callback,) * 32,
        builtin_identities=(str, int, list, tuple, type, isinstance, callable),
    )


def assert_python_canary(canary: PythonCanary, *, checkpoint: str) -> None:
    """Fail at the first changed object slot instead of normalizing it with ``str``."""

    if type(canary.token) is not str:
        raise PythonObjectIntegrityError(
            f"{checkpoint}: token type changed to {type(canary.token).__name__}"
        )
    for index, label in enumerate(canary.labels):
        if type(label) is not str or label is not canary.token:
            raise PythonObjectIntegrityError(
                f"{checkpoint}: labels[{index}] changed; "
                f"type={type(label).__name__} repr={label!r} "
                f"identity_match={label is canary.token}"
            )
    for index, callback in enumerate(canary.callbacks):
        if callback is not canary.callback or not callable(callback):
            raise PythonObjectIntegrityError(
                f"{checkpoint}: callbacks[{index}] changed; "
                f"type={type(callback).__name__} repr={callback!r}"
            )
    expected_builtins = (str, int, list, tuple, type, isinstance, callable)
    if canary.builtin_identities != expected_builtins:
        raise PythonObjectIntegrityError(
            f"{checkpoint}: builtin identity tuple changed; "
            f"observed={[type(value).__name__ for value in canary.builtin_identities]}"
        )
    if canary.callback(41) != 42:
        raise PythonObjectIntegrityError(f"{checkpoint}: callback result changed")


def _device_integer_canary(device: torch.device, *, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    cpu = torch.arange(int(size), dtype=torch.int64)
    expected = ((cpu * 17 + 23) ^ (cpu << 2)).contiguous()
    return expected.to(device), expected


def _assert_device_integer_canary(
    observed: torch.Tensor,
    expected_cpu: torch.Tensor,
    *,
    checkpoint: str,
) -> None:
    actual_cpu = observed.detach().cpu()
    if not torch.equal(actual_cpu, expected_cpu):
        mismatch = torch.nonzero(actual_cpu != expected_cpu, as_tuple=False).flatten()
        head = mismatch[:8].tolist()
        raise RuntimeError(
            f"{checkpoint}: device integer canary mismatch count={int(mismatch.numel())} "
            f"head={head}"
        )


def _serialization_roundtrip(
    *,
    device: torch.device,
    python_canary: PythonCanary,
    integer_canary: torch.Tensor,
    checkpoint: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="unilab-native-numeric-") as temporary_dir:
        path = Path(temporary_dir) / "payload.pt"
        torch.save(
            {
                "scenario_labels": list(python_canary.labels),
                "integer_canary": integer_canary,
            },
            path,
        )
        payload = torch.load(path, map_location=device, weights_only=False)
    labels = payload["scenario_labels"]
    if len(labels) != len(python_canary.labels):
        raise PythonObjectIntegrityError(
            f"{checkpoint}: serialized label length changed from "
            f"{len(python_canary.labels)} to {len(labels)}"
        )
    if any(type(label) is not str or label != python_canary.token for label in labels):
        bad = next(
            index
            for index, label in enumerate(labels)
            if type(label) is not str or label != python_canary.token
        )
        raise PythonObjectIntegrityError(
            f"{checkpoint}: serialized labels[{bad}] changed; "
            f"type={type(labels[bad]).__name__} repr={labels[bad]!r}"
        )
    if not torch.equal(payload["integer_canary"].detach().cpu(), integer_canary.detach().cpu()):
        raise RuntimeError(f"{checkpoint}: serialized integer canary changed")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_numeric_worker(config: WorkerConfig) -> dict[str, Any]:
    """Run deterministic optimizer, allocation, transfer, and serialization stress."""

    faulthandler.enable(all_threads=True)
    if config.matrix_size <= 0 or config.batch_size <= 0:
        raise ValueError("matrix_size and batch_size must be positive")
    if config.seconds <= 0 and config.iterations <= 0:
        raise ValueError("seconds or iterations must be positive")
    if config.allocation_mib <= 0:
        raise ValueError("allocation_mib must be positive")

    torch.manual_seed(int(config.seed) + int(config.worker_index))
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA stage requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(device)

    python_canary = _make_python_canary(
        size=config.canary_size,
        seed=config.seed,
        worker_index=config.worker_index,
    )
    integer_canary, expected_integer_cpu = _device_integer_canary(
        device,
        size=max(1024, min(config.canary_size, 65536)),
    )
    weight = torch.nn.Parameter(
        torch.randn(config.matrix_size, config.matrix_size, device=device) * 0.01
    )
    optimizer = torch.optim.Adam((weight,), lr=1e-4)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.seed) + int(config.worker_index) * 1009)
    allocation_elements = max(1, config.allocation_mib * 1024 * 1024 // 4)

    assert_python_canary(python_canary, checkpoint="worker/start")
    _assert_device_integer_canary(
        integer_canary,
        expected_integer_cpu,
        checkpoint="worker/start",
    )
    _serialization_roundtrip(
        device=device,
        python_canary=python_canary,
        integer_canary=integer_canary,
        checkpoint="worker/start",
    )

    started = time.monotonic()
    completed = 0
    last_loss = 0.0
    while True:
        iteration = completed + 1
        if config.iterations > 0 and iteration > config.iterations:
            break
        if config.iterations <= 0 and time.monotonic() - started >= config.seconds:
            break
        if config.inject_failure_iteration == iteration:
            raise RuntimeError(f"synthetic numeric failure at iteration {iteration}")

        assert_python_canary(python_canary, checkpoint=f"iteration_{iteration}/before_native")
        _assert_device_integer_canary(
            integer_canary,
            expected_integer_cpu,
            checkpoint=f"iteration_{iteration}/before_native",
        )

        batch = torch.randn(
            config.batch_size,
            config.matrix_size,
            generator=generator,
            device=device,
        )
        target = torch.tanh(batch * 0.25)
        optimizer.zero_grad(set_to_none=True)
        prediction = torch.tanh(batch @ weight)
        loss = torch.nn.functional.mse_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"iteration_{iteration}: non-finite loss {loss!r}")
        loss.backward()
        optimizer.step()

        churn = torch.empty(allocation_elements, dtype=torch.float32, device=device)
        churn.fill_(float((iteration % 17) + 1))
        expected_edge = float((iteration % 17) + 1)
        edge = churn[[0, -1]].detach().cpu()
        if not torch.equal(edge, torch.full((2,), expected_edge, dtype=torch.float32)):
            raise RuntimeError(
                f"iteration_{iteration}: allocation edge mismatch observed={edge.tolist()}"
            )
        del churn, batch, target, prediction
        _synchronize(device)

        if config.serialization_interval > 0 and iteration % config.serialization_interval == 0:
            _serialization_roundtrip(
                device=device,
                python_canary=python_canary,
                integer_canary=integer_canary,
                checkpoint=f"iteration_{iteration}/serialization",
            )

        assert_python_canary(python_canary, checkpoint=f"iteration_{iteration}/after_native")
        _assert_device_integer_canary(
            integer_canary,
            expected_integer_cpu,
            checkpoint=f"iteration_{iteration}/after_native",
        )
        last_loss = float(loss.detach().cpu())
        completed = iteration

    _synchronize(device)
    return {
        "status": "completed",
        "stage": config.stage,
        "device": str(device),
        "worker_index": config.worker_index,
        "pid": os.getpid(),
        "iterations_completed": completed,
        "duration_seconds": time.monotonic() - started,
        "last_loss": last_loss,
    }


def _worker_entry(config: WorkerConfig, result_queue: Any) -> None:
    try:
        result_queue.put(run_numeric_worker(config))
    except BaseException as error:
        result_queue.put(
            {
                "status": "failed",
                "stage": config.stage,
                "device": config.device,
                "worker_index": config.worker_index,
                "pid": os.getpid(),
                "error_type": type(error).__name__,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def run_stage(
    *,
    name: str,
    device: str,
    workers: int,
    base_config: WorkerConfig,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run one fresh-process stage and preserve fatal-signal exit codes."""

    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}")
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=workers * 2)
    processes = []
    for worker_index in range(workers):
        config = WorkerConfig(
            **{
                **asdict(base_config),
                "stage": name,
                "device": device,
                "worker_index": worker_index,
            }
        )
        process = context.Process(target=_worker_entry, args=(config, result_queue))
        process.start()
        processes.append(process)

    deadline = time.monotonic() + float(timeout_seconds)
    timed_out = False
    for process in processes:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            process.join(timeout=remaining)
        if process.is_alive():
            timed_out = True
            process.terminate()
    for process in processes:
        process.join(timeout=10)

    results: list[dict[str, Any]] = []
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            break
    result_queue.close()
    exitcodes = [process.exitcode for process in processes]
    status = "completed"
    if (
        timed_out
        or any(code != 0 for code in exitcodes)
        or any(result.get("status") != "completed" for result in results)
    ):
        status = "failed"
    if len(results) != workers:
        status = "failed"
    return {
        "name": name,
        "status": status,
        "device": device,
        "workers": workers,
        "timed_out": timed_out,
        "exitcodes": exitcodes,
        "worker_results": sorted(results, key=lambda item: int(item["worker_index"])),
    }


def stage_matrix(device: str) -> tuple[tuple[str, str, int], ...]:
    """Return matched stages ordered from broad host baseline to concurrency pressure."""

    return (
        ("cpu_single", "cpu", 1),
        ("gpu_single", device, 1),
        ("gpu_dual", device, 2),
    )


def run_campaign(
    *,
    device: str,
    base_config: WorkerConfig,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run stages sequentially and stop at the first contradicted boundary."""

    stages: list[dict[str, Any]] = []
    for name, stage_device, workers in stage_matrix(device):
        if stage_device.startswith("cuda") and not torch.cuda.is_available():
            stages.append(
                {
                    "name": name,
                    "status": "skipped",
                    "device": stage_device,
                    "workers": workers,
                    "reason": "cuda-unavailable",
                }
            )
            continue
        result = run_stage(
            name=name,
            device=stage_device,
            workers=workers,
            base_config=base_config,
            timeout_seconds=timeout_seconds,
        )
        stages.append(result)
        if result["status"] == "failed":
            break

    first_failure = next(
        (stage["name"] for stage in stages if stage["status"] == "failed"),
        None,
    )
    skipped = any(stage["status"] == "skipped" for stage in stages)
    status = "failed" if first_failure is not None else ("partial" if skipped else "completed")
    return {
        "status": status,
        "first_failure": first_failure,
        "evidence_level": (
            "runtime-confirmed"
            if first_failure is not None
            else ("partial-not-reproduced" if skipped else "not-reproduced")
        ),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "config": asdict(base_config),
        "stages": stages,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seconds-per-stage", type=float, default=60.0)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--matrix-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--allocation-mib", type=int, default=128)
    parser.add_argument("--canary-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--serialization-interval", type=int, default=25)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    base_config = WorkerConfig(
        stage="template",
        device="cpu",
        worker_index=0,
        seconds=float(args.seconds_per_stage),
        iterations=int(args.iterations),
        matrix_size=int(args.matrix_size),
        batch_size=int(args.batch_size),
        allocation_mib=int(args.allocation_mib),
        canary_size=int(args.canary_size),
        seed=int(args.seed),
        serialization_interval=int(args.serialization_interval),
    )
    report = run_campaign(
        device=str(args.device),
        base_config=base_config,
        timeout_seconds=float(args.timeout_seconds),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
