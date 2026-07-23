#!/usr/bin/env python3
"""Run an isolated one-shot diagnostic campaign for distillation native corruption."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
LIFECYCLE_SCRIPT = ROOT_DIR / "scripts" / "deploy" / "check_distill_role_data_lifecycle.py"
_ACTIVE_DIAGNOSTIC_ENV = (
    "PYTHONMALLOC",
    "PYTHONFAULTHANDLER",
    "MALLOC_CHECK_",
    "MALLOC_PERTURB_",
    "CUDA_LAUNCH_BLOCKING",
    "PYTORCH_NO_CUDA_MEMORY_CACHING",
    "ASAN_OPTIONS",
    "TSAN_OPTIONS",
    "UNILAB_NATIVE_HEAP_DEBUG",
    "UNILAB_NATIVE_ABORT_ON_CORRUPTION",
)
_FIRST_INVALID_MARKERS = (
    "ERROR: AddressSanitizer",
    "Invalid read of size",
    "Invalid write of size",
    "Invalid free()",
    "Invalid __global__",
    "Invalid __local__",
    "Invalid __shared__",
    "Race reported between",
    "Uninitialized __global__",
    "Barrier error detected",
)
_NATIVE_SYMPTOM_MARKERS = (
    "double free",
    "heap corruption",
    "corrupted size",
    "Py_FatalError",
    "Fatal Python error",
    "bad trailing pad byte",
    "bad leading pad byte",
    "Segmentation fault",
    "SIGABRT",
    "'cell' object is not callable",
    "\"<class 'frame'>\"",
    "<class 'frame'>",
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    command: tuple[str, ...]
    env_overrides: Mapping[str, str] = field(default_factory=dict)
    env_unset: tuple[str, ...] = _ACTIVE_DIAGNOSTIC_ENV
    timeout_seconds: float = 3600.0
    method: str = "plain"
    cwd: str = str(ROOT_DIR)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_paths(dataset_path: Path | None, source_manifest: Path | None) -> list[Path]:
    paths: list[Path] = []
    if dataset_path is not None:
        paths.append(dataset_path)
    if source_manifest is not None:
        paths.append(source_manifest)
        payload = json.loads(source_manifest.read_text(encoding="utf-8"))
        for source in payload.get("sources", []):
            source_path = Path(str(source["path"]))
            if not source_path.is_absolute():
                source_path = source_manifest.parent / source_path
            paths.append(source_path.resolve())
    return list(dict.fromkeys(path.resolve() for path in paths))


def _identity(paths: Sequence[Path]) -> list[dict[str, Any]]:
    identity: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        identity.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _file_sha256(path),
            }
        )
    return identity


def _best_effort_command(command: Sequence[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as error:
        return {"command": list(command), "error": repr(error)}
    return {
        "command": list(command),
        "returncode": result.returncode,
        "stdout": result.stdout[-20000:],
        "stderr": result.stderr[-20000:],
    }


def collect_preflight(paths: Sequence[Path]) -> dict[str, Any]:
    core_pattern = Path("/proc/sys/kernel/core_pattern")
    ptrace_scope = Path("/proc/sys/kernel/yama/ptrace_scope")
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "cwd": str(ROOT_DIR),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "core_limit": list(resource.getrlimit(resource.RLIMIT_CORE)),
        "core_pattern": core_pattern.read_text().strip() if core_pattern.is_file() else None,
        "ptrace_scope": ptrace_scope.read_text().strip() if ptrace_scope.is_file() else None,
        "tools": {
            name: shutil.which(name)
            for name in (
                "uv",
                "gdb",
                "valgrind",
                "rr",
                "compute-sanitizer",
                "nvidia-smi",
                "coredumpctl",
            )
        },
        "git_head": _best_effort_command(("git", "rev-parse", "HEAD")),
        "git_status": _best_effort_command(("git", "status", "--short", "--branch")),
        "nvidia_smi": _best_effort_command(
            (
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total,ecc.errors.uncorrected.aggregate.total",
                "--format=csv,noheader",
            )
        )
        if shutil.which("nvidia-smi")
        else {"status": "unavailable"},
        "inputs": _identity(paths),
    }


def _linux_process_tree(root_pid: int) -> list[dict[str, Any]]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return [{"pid": root_pid, "telemetry": "procfs-unavailable"}]
    rows: dict[int, dict[str, Any]] = {}
    for status_path in proc_root.glob("[0-9]*/status"):
        try:
            values: dict[str, str] = {}
            for line in status_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition(":")
                if separator and key in {"Name", "State", "Pid", "PPid", "Threads", "VmRSS"}:
                    values[key] = value.strip()
            pid = int(values["Pid"])
            rows[pid] = {
                "pid": pid,
                "ppid": int(values.get("PPid", "0")),
                "name": values.get("Name"),
                "state": values.get("State"),
                "threads": int(values.get("Threads", "0")),
                "rss_kib": int(values.get("VmRSS", "0 kB").split()[0]),
            }
        except (FileNotFoundError, KeyError, PermissionError, ProcessLookupError, ValueError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, row in rows.items():
            if pid not in descendants and row["ppid"] in descendants:
                descendants.add(pid)
                changed = True
    return [rows[pid] for pid in sorted(descendants) if pid in rows]


def _scan_evidence(stage_dir: Path) -> tuple[str, list[str]]:
    snippets: list[str] = []
    level = "unconfirmed"
    for log_path in sorted(stage_dir.glob("*.log")):
        try:
            with log_path.open("r", encoding="utf-8", errors="replace") as stream:
                for line in stream:
                    stripped = line.rstrip()
                    if any(marker in stripped for marker in _FIRST_INVALID_MARKERS):
                        level = "first-invalid-operation-confirmed"
                        if len(snippets) < 20:
                            snippets.append(f"{log_path.name}: {stripped}")
                    elif level != "first-invalid-operation-confirmed" and any(
                        marker in stripped for marker in _NATIVE_SYMPTOM_MARKERS
                    ):
                        level = "native-symptom-confirmed"
                        if len(snippets) < 20:
                            snippets.append(f"{log_path.name}: {stripped}")
        except OSError:
            continue
    return level, snippets


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def run_stage(
    spec: StageSpec,
    stage_dir: str | Path,
    *,
    monitor_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    """Run one active diagnostic in one isolated child identity."""

    resolved_stage_dir = Path(stage_dir).resolve()
    resolved_stage_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        resolved_stage_dir / "command.json",
        {
            "name": spec.name,
            "method": spec.method,
            "command": list(spec.command),
            "cwd": spec.cwd,
            "env_overrides": dict(spec.env_overrides),
            "env_unset": list(spec.env_unset),
            "timeout_seconds": spec.timeout_seconds,
        },
    )
    env = os.environ.copy()
    for key in spec.env_unset:
        env.pop(key, None)
    env.update({key: str(value) for key, value in spec.env_overrides.items()})
    stdout_path = resolved_stage_dir / "stdout.log"
    stderr_path = resolved_stage_dir / "stderr.log"
    telemetry_path = resolved_stage_dir / "process_telemetry.jsonl"
    started = time.monotonic()
    started_at_epoch = time.time()
    timed_out = False
    launch_error: str | None = None
    returncode: int | None = None
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
        telemetry_path.open("w", encoding="utf-8") as telemetry,
    ):
        try:
            process = subprocess.Popen(
                list(spec.command),
                cwd=spec.cwd,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
        except Exception as error:
            launch_error = repr(error)
        else:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                telemetry.write(
                    json.dumps(
                        {
                            "elapsed_seconds": elapsed,
                            "processes": _linux_process_tree(process.pid),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                telemetry.flush()
                if elapsed >= float(spec.timeout_seconds):
                    timed_out = True
                    _terminate_process_group(process)
                    break
                time.sleep(max(0.01, float(monitor_interval_seconds)))
            returncode = process.poll()

    evidence_level, evidence_snippets = _scan_evidence(resolved_stage_dir)
    termination_signal: str | None = None
    if returncode is not None and returncode < 0 and not timed_out:
        try:
            termination_signal = signal.Signals(-returncode).name
        except ValueError:
            termination_signal = f"SIGNAL_{-returncode}"
        if evidence_level == "unconfirmed":
            evidence_level = "native-symptom-confirmed"
            evidence_snippets.append(f"process terminated by {termination_signal}")
    if evidence_level == "unconfirmed" and returncode == 0:
        evidence_level = "runtime-confirmed"
    if launch_error is not None:
        status = "unavailable"
    elif timed_out:
        status = "timeout"
    elif returncode == 0:
        status = "completed"
    else:
        status = "failed"
    result = {
        "name": spec.name,
        "method": spec.method,
        "status": status,
        "returncode": returncode,
        "termination_signal": termination_signal,
        "timed_out": timed_out,
        "duration_seconds": time.monotonic() - started,
        "started_at_epoch": started_at_epoch,
        "ended_at_epoch": time.time(),
        "evidence_level": evidence_level,
        "evidence_snippets": evidence_snippets,
        "launch_error": launch_error,
        "stage_dir": str(resolved_stage_dir),
    }
    _write_json(resolved_stage_dir / "stage_result.json", result)
    return result


def _lifecycle_command(
    runtime: Sequence[str],
    *,
    lifecycle_script: Path,
    stage_data_dir: Path,
    dataset_path: Path | None,
    source_manifest: Path | None,
    cycles: int,
) -> tuple[str, ...]:
    command = [
        *runtime,
        str(lifecycle_script),
        "--work-dir",
        str(stage_data_dir),
        "--cycles",
        str(cycles),
    ]
    if dataset_path is not None:
        command.extend(("--dataset", str(dataset_path)))
    else:
        assert source_manifest is not None
        command.extend(("--source-manifest", str(source_manifest)))
    return tuple(command)


def build_lifecycle_stages(
    *,
    python_executable: str,
    lifecycle_script: Path,
    campaign_dir: Path,
    dataset_path: Path | None,
    source_manifest: Path | None,
    cycles: int,
    timeout_seconds: float,
    valgrind_path: str | None,
    rr_path: str | None,
    uv_executable: str | None = None,
) -> list[StageSpec]:
    """Build isolated host stages; no child stacks two active diagnostics."""

    uv_path = uv_executable or shutil.which("uv")
    if uv_path is None:
        raise FileNotFoundError("uv is required by the UniLab execution contract")

    def data_dir(name: str) -> Path:
        return campaign_dir / "stages" / name / "data"

    plain_command = _lifecycle_command(
        (uv_path, "run"),
        lifecycle_script=lifecycle_script,
        stage_data_dir=data_dir("host_plain"),
        dataset_path=dataset_path,
        source_manifest=source_manifest,
        cycles=cycles,
    )
    allocator_command = _lifecycle_command(
        (uv_path, "run"),
        lifecycle_script=lifecycle_script,
        stage_data_dir=data_dir("host_allocator_debug"),
        dataset_path=dataset_path,
        source_manifest=source_manifest,
        cycles=cycles,
    )
    stages = [
        StageSpec(
            name="host_plain",
            command=plain_command,
            timeout_seconds=timeout_seconds,
            method="plain-lifecycle",
        ),
        StageSpec(
            name="host_allocator_debug",
            command=allocator_command,
            env_overrides={
                "PYTHONMALLOC": "debug",
                "PYTHONFAULTHANDLER": "1",
                "MALLOC_CHECK_": "3",
                "MALLOC_PERTURB_": "165",
            },
            timeout_seconds=timeout_seconds,
            method="python-glibc-allocator-debug",
        ),
    ]
    if valgrind_path is not None:
        target = _lifecycle_command(
            (python_executable,),
            lifecycle_script=lifecycle_script,
            stage_data_dir=data_dir("host_memcheck"),
            dataset_path=dataset_path,
            source_manifest=source_manifest,
            cycles=max(1, min(cycles, 16)),
        )
        stages.append(
            StageSpec(
                name="host_memcheck",
                command=(
                    valgrind_path,
                    "--tool=memcheck",
                    "--track-origins=yes",
                    "--leak-check=full",
                    "--error-exitcode=86",
                    *target,
                ),
                timeout_seconds=timeout_seconds,
                method="memcheck",
            )
        )
    if rr_path is not None:
        target = _lifecycle_command(
            (python_executable,),
            lifecycle_script=lifecycle_script,
            stage_data_dir=data_dir("host_rr"),
            dataset_path=dataset_path,
            source_manifest=source_manifest,
            cycles=max(1, min(cycles, 64)),
        )
        stages.append(
            StageSpec(
                name="host_rr",
                command=(rr_path, "record", "-o", str(campaign_dir / "rr-trace"), *target),
                timeout_seconds=timeout_seconds,
                method="rr-record",
            )
        )
    return stages


def _resolve_optional_tool(mode: str, name: str) -> tuple[str | None, dict[str, Any] | None]:
    if mode == "off":
        return None, {"name": name, "status": "skipped", "reason": "disabled"}
    path = shutil.which(name)
    if path is not None:
        return path, None
    if mode == "on":
        raise FileNotFoundError(f"required diagnostic tool is unavailable: {name}")
    return None, {"name": name, "status": "skipped", "reason": "tool-unavailable"}


def _offline_replay_command(
    *,
    uv_executable: str,
    dataset_path: Path,
    init_checkpoint: Path,
    teacher_checkpoint: Path,
    output_checkpoint: Path,
    device: str,
    max_updates: int,
    batch_size: int,
) -> tuple[str, ...]:
    return (
        uv_executable,
        "run",
        "scripts/train_distill.py",
        "task=g1_walk_flat/mujoco",
        "workflow=g1_walk_stand",
        "training.workflow.enabled=false",
        f"training.device={device}",
        f"teacher.checkpoint_path={teacher_checkpoint}",
        f"training.offline_dataset_path={dataset_path}",
        f"training.offline_init_checkpoint={init_checkpoint}",
        f"training.offline_checkpoint={output_checkpoint}",
        "training.offline_repeat_dataset=true",
        "training.offline_shuffle=true",
        "training.offline_balance_key=scenario",
        "training.offline_balanced_labels=[static_stand,walk_flat,walk_to_stop]",
        "++training.offline_balance_quotas={walk_flat:0.5,static_stand:0.25,walk_to_stop:0.25}",
        "training.offline_min_balanced_replay_passes=8",
        "training.offline_min_balanced_replay_labels=[walk_to_stop]",
        f"training.offline_batch_size={batch_size}",
        f"training.offline_max_updates={max_updates}",
    )


def _build_gpu_stages(
    args: argparse.Namespace, uv_executable: str
) -> tuple[list[StageSpec], list[dict[str, Any]]]:
    if args.offline_init_checkpoint is None or args.teacher_checkpoint is None:
        return [], [
            {
                "name": "gpu-replay",
                "status": "skipped",
                "reason": "offline init and teacher checkpoints were not both supplied",
            }
        ]
    dataset_path = args.offline_dataset or args.dataset
    if dataset_path is None:
        return [], [{"name": "gpu-replay", "status": "skipped", "reason": "no dataset"}]
    for path in (dataset_path, args.offline_init_checkpoint, args.teacher_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    stages: list[StageSpec] = []
    sync_command = _offline_replay_command(
        uv_executable=uv_executable,
        dataset_path=dataset_path.resolve(),
        init_checkpoint=args.offline_init_checkpoint.resolve(),
        teacher_checkpoint=args.teacher_checkpoint.resolve(),
        output_checkpoint=(args.work_dir / "stages/gpu_sync_replay/model.pt").resolve(),
        device=args.device,
        max_updates=args.gpu_sync_updates,
        batch_size=args.offline_batch_size,
    )
    stages.append(
        StageSpec(
            name="gpu_sync_replay",
            command=sync_command,
            env_overrides={"CUDA_LAUNCH_BLOCKING": "1", "UNILAB_NATIVE_ABORT_ON_CORRUPTION": "1"},
            timeout_seconds=args.timeout_seconds,
            method="cuda-launch-blocking",
        )
    )
    compute_path, skipped = _resolve_optional_tool(args.compute_sanitizer, "compute-sanitizer")
    skipped_rows: list[dict[str, Any]] = [] if skipped is None else [skipped]
    if compute_path is not None:
        compute_command = _offline_replay_command(
            uv_executable=uv_executable,
            dataset_path=dataset_path.resolve(),
            init_checkpoint=args.offline_init_checkpoint.resolve(),
            teacher_checkpoint=args.teacher_checkpoint.resolve(),
            output_checkpoint=(args.work_dir / "stages/gpu_memcheck_replay/model.pt").resolve(),
            device=args.device,
            max_updates=args.gpu_memcheck_updates,
            batch_size=args.offline_batch_size,
        )
        stages.append(
            StageSpec(
                name="gpu_memcheck_replay",
                command=(
                    compute_path,
                    "--tool",
                    "memcheck",
                    "--target-processes",
                    "all",
                    "--error-exitcode",
                    "87",
                    *compute_command,
                ),
                env_overrides={"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"},
                timeout_seconds=args.timeout_seconds,
                method="compute-sanitizer-memcheck",
            )
        )
    return stages, skipped_rows


def _build_persistent_differential_stages(
    args: argparse.Namespace,
    uv_executable: str,
) -> tuple[list[StageSpec], list[dict[str, Any]]]:
    if args.persistent_differential_repetitions <= 0:
        return [], [
            {
                "name": "persistent-lifecycle-differential",
                "status": "skipped",
                "reason": "repetitions-disabled",
            }
        ]
    student_checkpoint = args.student_checkpoint or args.offline_init_checkpoint
    if args.walk_teacher is None or args.stand_teacher is None or student_checkpoint is None:
        return [], [
            {
                "name": "persistent-lifecycle-differential",
                "status": "skipped",
                "reason": "walking, standing, and student checkpoints were not all supplied",
            }
        ]
    for path in (args.walk_teacher, args.stand_teacher, student_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    probe = ROOT_DIR / "scripts/deploy/check_unilab_g1_distill_persistent_runtime.py"
    stages: list[StageSpec] = []
    for lifecycle, name, method in (
        ("persistent", "collector_persistent", "persistent-worker"),
        ("restart_each_request", "collector_restart_each_request", "restart-each-request"),
    ):
        command = (
            uv_executable,
            "run",
            str(probe),
            "--walking-checkpoint",
            str(args.walk_teacher.resolve()),
            "--standing-checkpoint",
            str(args.stand_teacher.resolve()),
            "--student-checkpoint",
            str(student_checkpoint.resolve()),
            "--work-dir",
            str((args.work_dir / "stages" / name / "probe").resolve()),
            "--num-envs",
            str(args.persistent_num_envs),
            "--samples",
            str(args.persistent_samples),
            "--device",
            args.device,
            "--repetitions",
            str(args.persistent_differential_repetitions),
            "--worker-lifecycle",
            lifecycle,
        )
        stages.append(
            StageSpec(
                name=name,
                command=command,
                timeout_seconds=args.timeout_seconds,
                method=method,
            )
        )
    return stages, []


def _build_formal_stages(args: argparse.Namespace) -> list[StageSpec]:
    if args.formal_attempts <= 0:
        return []
    required = {
        "walk_teacher": args.walk_teacher,
        "stand_teacher": args.stand_teacher,
        "walk_dataset": args.walk_dataset,
        "stand_dataset": args.stand_dataset,
    }
    missing = [name for name, path in required.items() if path is None]
    if missing:
        raise ValueError(f"formal attempts require paths: {missing}")
    for path in required.values():
        assert path is not None
        if not path.is_file():
            raise FileNotFoundError(path)
    stages: list[StageSpec] = []
    campaign_token = re.sub(r"[^a-z0-9-]+", "-", args.work_dir.name.lower()).strip("-")
    campaign_token = (campaign_token or "run")[-24:]
    for attempt in range(1, args.formal_attempts + 1):
        run_name = f"native-campaign-{campaign_token}-a{attempt}"
        command = (
            str(ROOT_DIR / "train.sh"),
            "--workflow-mode",
            "fresh",
            "--run-name",
            run_name,
            "--execution-mode",
            "persistent_async",
            f"training.workflow.collect_num_envs={args.formal_num_envs}",
            f"training.workflow.dagger_iterations={args.formal_dagger_iterations}",
        )
        stages.append(
            StageSpec(
                name=f"formal_native_attempt_{attempt}",
                command=command,
                env_overrides={
                    "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
                    "HYDRA_FULL_ERROR": "1",
                    "PYTHONWARNINGS": "ignore",
                    "UNILAB_G1_WALK_TEACHER": str(args.walk_teacher.resolve()),
                    "UNILAB_G1_STAND_TEACHER": str(args.stand_teacher.resolve()),
                    "UNILAB_G1_WALK_DATASET": str(args.walk_dataset.resolve()),
                    "UNILAB_G1_STAND_DATASET": str(args.stand_dataset.resolve()),
                    "UNILAB_NATIVE_HEAP_DEBUG": "1",
                    "UNILAB_NATIVE_ABORT_ON_CORRUPTION": "1",
                },
                timeout_seconds=args.formal_timeout_seconds,
                method="formal-native-capture",
            )
        )
    return stages


def classify_campaign(stages: Sequence[Mapping[str, Any]]) -> str:
    levels = [str(stage.get("evidence_level", "unconfirmed")) for stage in stages]
    if "first-invalid-operation-confirmed" in levels:
        return "FIRST_INVALID_OPERATION_CAPTURED"
    failed_methods = {
        str(stage.get("method"))
        for stage in stages
        if stage.get("status") == "failed"
        or stage.get("evidence_level") == "native-symptom-confirmed"
    }
    completed_methods = {
        str(stage.get("method")) for stage in stages if stage.get("status") == "completed"
    }
    if failed_methods and completed_methods:
        return "ROOT_CAUSE_BOUNDARY_ISOLATED"
    return "INCONCLUSIVE_NOT_REPRODUCED"


def create_archive(work_dir: Path) -> Path:
    archive = work_dir.with_name(work_dir.name + "-RETURN_ME.tar.gz")
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(work_dir, arcname=work_dir.name, recursive=True)
    return archive


def _gdb_command_file(path: Path) -> None:
    """Write a CPython-aware core script that preserves the handled exception."""

    path.write_text(
        """set pagination off
set print pretty on
set print elements 256
info threads
thread apply all bt full
python
import gdb

def run_optional(label, command):
    print(f"### {label}")
    try:
        gdb.execute(command)
    except Exception as error:
        print(f"{label}_ERROR: {error!r}")

run_optional("PY_BT", "thread apply all py-bt")
run_optional("INFO_SHAREDLIBRARY", "info sharedlibrary")
run_optional("INFO_FILES", "info files")

expressions = {
    "TSTATE": "(PyThreadState*)_PyRuntime.gilstate.tstate_current._value",
    "CUREXC_TYPE": "((PyThreadState*)_PyRuntime.gilstate.tstate_current._value)->curexc_type",
    "CUREXC_VALUE": "((PyThreadState*)_PyRuntime.gilstate.tstate_current._value)->curexc_value",
    "CUREXC_TRACEBACK": "((PyThreadState*)_PyRuntime.gilstate.tstate_current._value)->curexc_traceback",
    "HANDLED_EXCEPTION_TYPE": "((PyThreadState*)_PyRuntime.gilstate.tstate_current._value)->exc_info->exc_type",
    "HANDLED_EXCEPTION": "((PyThreadState*)_PyRuntime.gilstate.tstate_current._value)->exc_info->exc_value",
    "HANDLED_TRACEBACK": "((PyThreadState*)_PyRuntime.gilstate.tstate_current._value)->exc_info->exc_traceback",
}
helper = globals().get("PyObjectPtr")
for label, expression in expressions.items():
    try:
        value = gdb.parse_and_eval(expression)
        if int(value) == 0:
            print(f"{label}: NULL")
        elif helper is None:
            print(f"{label}: pointer={value}")
        else:
            obj = helper.from_pyobject_ptr(value)
            print(f"{label}: {obj.get_truncated_repr(8192)}")
    except Exception as error:
        print(f"{label}_ERROR: {error!r}")
end
""",
        encoding="utf-8",
    )


def analyze_native_core_artifact(
    *,
    artifact: Path,
    capture_dir: Path,
    gdb_path: str | None,
    apport_unpack_path: str | None,
) -> dict[str, Any]:
    """Analyze raw cores directly and Apport reports only after ``apport-unpack``."""

    artifact = artifact.resolve()
    capture_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "path": str(artifact),
        "exists": artifact.is_file(),
        "readable": os.access(artifact, os.R_OK),
        "artifact_kind": "apport-report" if artifact.suffix == ".crash" else "raw-core",
        "gdb_status": "skipped",
    }
    if not artifact.is_file():
        record["status"] = "missing"
        return record
    stat = artifact.stat()
    record.update(
        {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "mode": oct(stat.st_mode & 0o777),
        }
    )
    if not record["readable"]:
        record["status"] = "unreadable"
        return record

    core_path = artifact
    executable = Path("/usr/bin/python3.10")
    unpack_dir: Path | None = None
    if artifact.suffix == ".crash":
        if apport_unpack_path is None:
            record.update({"status": "unavailable", "reason": "apport-unpack unavailable"})
            return record
        unpack_dir = capture_dir / "apport-unpacked"
        unpack_dir.mkdir(parents=True, exist_ok=False)
        unpack_result = subprocess.run(
            [apport_unpack_path, str(artifact), str(unpack_dir)],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        (capture_dir / "apport-unpack.log").write_text(
            unpack_result.stdout + unpack_result.stderr,
            encoding="utf-8",
        )
        record["apport_unpack_returncode"] = unpack_result.returncode
        record["apport_unpack_dir"] = str(unpack_dir)
        core_path = unpack_dir / "CoreDump"
        executable_path = unpack_dir / "ExecutablePath"
        if executable_path.is_file():
            candidate = Path(executable_path.read_text(encoding="utf-8").strip())
            if candidate.is_file():
                executable = candidate
        if unpack_result.returncode != 0 or not core_path.is_file():
            record.update(
                {
                    "status": "unpack-failed",
                    "core_path_passed_to_gdb": None,
                }
            )
            return record
    elif not executable.is_file():
        executable = Path(sys.executable).resolve()

    record["executable"] = str(executable)
    record["core_path_passed_to_gdb"] = str(core_path)
    if gdb_path is None:
        record.update({"status": "unpacked", "reason": "gdb unavailable"})
    else:
        command_file = capture_dir / "gdb-core-commands.txt"
        _gdb_command_file(command_file)
        output_path = capture_dir / "gdb-all-threads-and-exception.txt"
        command = (
            gdb_path,
            "-q",
            str(executable),
            str(core_path),
            "-batch",
            "-x",
            str(command_file),
        )
        record["gdb_command"] = list(command)
        try:
            result = subprocess.run(
                list(command),
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                timeout=1200,
                check=False,
            )
        except Exception as error:
            record.update({"status": "gdb-error", "gdb_status": "error", "error": repr(error)})
        else:
            output_path.write_text(result.stdout + result.stderr, encoding="utf-8")
            record.update(
                {
                    "status": "analyzed" if result.returncode == 0 else "gdb-failed",
                    "gdb_status": "completed" if result.returncode == 0 else "failed",
                    "gdb_returncode": result.returncode,
                    "gdb_output": str(output_path),
                }
            )

    # The original Apport report remains untouched. Only the multi-GB extracted
    # copy is removed so the retrieval archive contains metadata and GDB evidence.
    if unpack_dir is not None and core_path.is_file():
        record["unpacked_core_size"] = core_path.stat().st_size
        core_path.unlink()
        record["unpacked_core_removed_after_analysis"] = True
    return record


def harvest_native_cores(
    *,
    work_dir: Path,
    since_epoch: float,
    gdb_path: str | None,
) -> dict[str, Any]:
    """Locate new system cores and symbolize them without copying multi-GB dumps."""

    candidates: list[Path] = []
    for root in (Path("/var/lib/apport/coredump"), Path("/var/crash")):
        if not root.is_dir():
            continue
        try:
            entries = tuple(root.iterdir())
        except PermissionError:
            continue
        for path in entries:
            try:
                if path.is_file() and path.stat().st_mtime >= float(since_epoch) - 1.0:
                    candidates.append(path)
            except (FileNotFoundError, PermissionError):
                continue
    capture_dir = work_dir / "native_cores"
    capture_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, path in enumerate(sorted(set(candidates), key=lambda item: item.stat().st_mtime)):
        records.append(
            analyze_native_core_artifact(
                artifact=path,
                capture_dir=capture_dir / f"core-{index:02d}",
                gdb_path=gdb_path,
                apport_unpack_path=shutil.which("apport-unpack"),
            )
        )
    result = {
        "since_epoch": since_epoch,
        "candidate_count": len(records),
        "candidates": records,
    }
    _write_json(capture_dir / "core_candidates.json", result)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--dataset", type=Path)
    inputs.add_argument("--source-manifest", type=Path)
    parser.add_argument("--cycles", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--monitor-interval-seconds", type=float, default=1.0)
    parser.add_argument("--valgrind", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--rr", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--compute-sanitizer", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--offline-dataset", type=Path)
    parser.add_argument("--offline-init-checkpoint", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offline-batch-size", type=int, default=512)
    parser.add_argument("--gpu-sync-updates", type=int, default=6000)
    parser.add_argument("--gpu-memcheck-updates", type=int, default=32)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--persistent-differential-repetitions", type=int, default=0)
    parser.add_argument("--persistent-num-envs", type=int, default=1)
    parser.add_argument("--persistent-samples", type=int, default=4)
    parser.add_argument("--formal-attempts", type=int, default=0)
    parser.add_argument("--formal-timeout-seconds", type=float, default=21600.0)
    parser.add_argument("--formal-num-envs", type=int, default=32)
    parser.add_argument("--formal-dagger-iterations", type=int, default=8)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--walk-teacher", type=Path)
    parser.add_argument("--stand-teacher", type=Path)
    parser.add_argument("--walk-dataset", type=Path)
    parser.add_argument("--stand-dataset", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    campaign_started_epoch = time.time()
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=False)
    args.dataset = None if args.dataset is None else args.dataset.resolve()
    args.source_manifest = None if args.source_manifest is None else args.source_manifest.resolve()
    paths = _input_paths(args.dataset, args.source_manifest)
    optional_inputs = (
        args.offline_dataset,
        args.offline_init_checkpoint,
        args.teacher_checkpoint,
        args.student_checkpoint,
        args.walk_teacher,
        args.stand_teacher,
        args.walk_dataset,
        args.stand_dataset,
    )
    paths.extend(path.resolve() for path in optional_inputs if path is not None and path.is_file())
    paths = list(dict.fromkeys(paths))
    preflight = collect_preflight(paths)
    _write_json(args.work_dir / "preflight.json", preflight)
    before_identity = preflight["inputs"]

    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise FileNotFoundError("uv is required by the UniLab execution contract")
    valgrind_path, valgrind_skipped = _resolve_optional_tool(args.valgrind, "valgrind")
    rr_path, rr_skipped = _resolve_optional_tool(args.rr, "rr")
    stages = build_lifecycle_stages(
        python_executable=sys.executable,
        lifecycle_script=LIFECYCLE_SCRIPT,
        campaign_dir=args.work_dir,
        dataset_path=args.dataset,
        source_manifest=args.source_manifest,
        cycles=args.cycles,
        timeout_seconds=args.timeout_seconds,
        valgrind_path=valgrind_path,
        rr_path=rr_path,
        uv_executable=uv_executable,
    )
    gpu_stages, gpu_skipped = _build_gpu_stages(args, uv_executable)
    persistent_stages, persistent_skipped = _build_persistent_differential_stages(
        args, uv_executable
    )
    stages.extend(persistent_stages)
    stages.extend(gpu_stages)
    stages.extend(_build_formal_stages(args))
    skipped = [
        row
        for row in (valgrind_skipped, rr_skipped, *persistent_skipped, *gpu_skipped)
        if row is not None
    ]

    stage_results: list[dict[str, Any]] = []
    decisive = False
    for spec in stages:
        if decisive:
            stage_results.append(
                {
                    "name": spec.name,
                    "method": spec.method,
                    "status": "skipped",
                    "reason": "earlier first-invalid-operation evidence",
                    "evidence_level": "unconfirmed",
                }
            )
            continue
        result = run_stage(
            spec,
            args.work_dir / "stages" / spec.name,
            monitor_interval_seconds=args.monitor_interval_seconds,
        )
        stage_results.append(result)
        decisive = result["evidence_level"] == "first-invalid-operation-confirmed"

    after_identity = _identity(paths)
    input_identity_unchanged = before_identity == after_identity
    core_capture = harvest_native_cores(
        work_dir=args.work_dir,
        since_epoch=campaign_started_epoch,
        gdb_path=shutil.which("gdb"),
    )
    summary: dict[str, Any] = {
        "campaign_status": "completed" if input_identity_unchanged else "error",
        "verdict": classify_campaign(stage_results),
        "root_cause_owner": "unconfirmed",
        "input_identity_unchanged": input_identity_unchanged,
        "inputs_before": before_identity,
        "inputs_after": after_identity,
        "stages": stage_results,
        "skipped_capabilities": skipped,
        "native_core_capture": core_capture,
        "retrieval_archive": str(args.work_dir.with_name(args.work_dir.name + "-RETURN_ME.tar.gz")),
    }
    _write_json(args.work_dir / "campaign_summary.json", summary)
    (args.work_dir / "RETURN_ME.txt").write_text(
        f"verdict={summary['verdict']}\narchive={summary['retrieval_archive']}\n",
        encoding="utf-8",
    )
    archive = create_archive(args.work_dir)
    print(
        json.dumps(
            {
                "campaign_status": summary["campaign_status"],
                "verdict": summary["verdict"],
                "retrieval_archive": str(archive),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if summary["campaign_status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
