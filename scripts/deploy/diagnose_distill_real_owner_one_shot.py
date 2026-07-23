#!/usr/bin/env python3
"""Run one offline campaign across the real distillation owner path."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.deploy.check_distill_real_owner_path import (  # noqa: E402
    _sources_from_seed_aggregate,
)
from scripts.deploy.diagnose_distill_native_corruption import (  # noqa: E402
    StageSpec,
    _file_sha256,
    _identity,
    _write_json,
    analyze_native_core_artifact,
    collect_preflight,
    create_archive,
    harvest_native_cores,
    run_stage,
)

OWNER_WORKER = ROOT_DIR / "scripts" / "deploy" / "check_distill_real_owner_path.py"


def _best_effort(command: Sequence[str], *, timeout: float = 60.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception as error:
        return {"command": list(command), "error": repr(error)}
    return {
        "command": list(command),
        "returncode": result.returncode,
        "stdout": result.stdout[-200000:],
        "stderr": result.stderr[-200000:],
    }


def _read_optional(path: Path, *, limit: int = 200000) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as error:
        return {"path": str(path), "error": repr(error)}
    return {"path": str(path), "text": text[-limit:]}


def collect_health_snapshot(output: Path, *, kernel_since_epoch: float | None) -> dict[str, Any]:
    """Capture GPU, RAM, Xid, MCE, and EDAC facts without requiring root."""

    commands: dict[str, Sequence[str]] = {
        "nvidia_smi_query": (
            "nvidia-smi",
            "--query-gpu=timestamp,index,uuid,name,driver_version,pstate,temperature.gpu,"
            "memory.total,memory.used,memory.free,ecc.mode.current,"
            "ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total,"
            "retired_pages.single_bit_ecc.count,retired_pages.double_bit.count",
            "--format=csv,noheader,nounits",
        ),
        "nvidia_smi_full": ("nvidia-smi", "-q"),
        "free": ("free", "-b"),
        "vmstat": ("vmstat", "-s"),
        "edac_util": ("edac-util", "-v"),
        "ras_mc_ctl": ("ras-mc-ctl", "--errors"),
        "mcelog": ("mcelog", "--client"),
    }
    if kernel_since_epoch is not None:
        commands["kernel_journal"] = (
            "journalctl",
            "-k",
            "--no-pager",
            "--since",
            f"@{int(kernel_since_epoch)}",
        )
    results = {
        name: _best_effort(command)
        if shutil.which(command[0]) is not None
        else {"command": list(command), "status": "unavailable"}
        for name, command in commands.items()
    }
    results["proc_meminfo"] = _read_optional(Path("/proc/meminfo"))
    results["proc_memory_pressure"] = _read_optional(Path("/proc/pressure/memory"))
    results["mce_log"] = _read_optional(Path("/var/log/mcelog"))
    edac_counters: list[dict[str, Any]] = []
    edac_root = Path("/sys/devices/system/edac/mc")
    if edac_root.is_dir():
        for counter in sorted(edac_root.glob("mc*/**/*_count")):
            if counter.is_file():
                edac_counters.append(_read_optional(counter, limit=4096))
    results["edac_counters"] = edac_counters
    try:
        results["dev_shm"] = [
            {
                "name": entry.name,
                "size": entry.stat().st_size,
                "mtime_ns": entry.stat().st_mtime_ns,
            }
            for entry in sorted(Path("/dev/shm").iterdir(), key=lambda item: item.name)
            if entry.name.startswith(("psm_", "sem."))
        ]
    except Exception as error:
        results["dev_shm"] = {"error": repr(error)}
    _write_json(output, results)
    return results


def _owner_command(
    uv: str,
    *,
    mode: str,
    stage_dir: Path,
    aggregate: Path,
    checkpoint: Path,
    teacher_checkpoint: Path,
    device: str,
    batch_size: int,
    max_updates: int,
    rounds: int,
) -> tuple[str, ...]:
    if mode == "assemble":
        return (
            uv,
            "run",
            str(OWNER_WORKER),
            "assemble",
            "--seed-aggregate",
            str(aggregate),
            "--output",
            str(stage_dir / "rebuilt-aggregate.pt"),
            "--device",
            device,
            "--report",
            str(stage_dir / "owner-report.json"),
        )
    return (
        uv,
        "run",
        str(OWNER_WORKER),
        "offline",
        "--aggregate",
        str(aggregate),
        "--init-checkpoint",
        str(checkpoint),
        "--teacher-checkpoint",
        str(teacher_checkpoint),
        "--output-dir",
        str(stage_dir / "owner-output"),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--max-updates",
        str(max_updates),
        "--rounds",
        str(rounds),
        "--report",
        str(stage_dir / "owner-report.json"),
    )


def build_stage_matrix(
    *,
    uv: str,
    work_dir: Path,
    aggregate: Path,
    checkpoint: Path,
    teacher_checkpoint: Path,
    role_env: Mapping[str, str],
    gpu_device: str,
    batch_size: int,
    fresh_updates: int,
    lifecycle_updates: int,
    lifecycle_rounds: int,
    timeout_seconds: float,
    native_abort_on_corruption: bool = False,
) -> dict[str, list[StageSpec]]:
    """Build matched controls; each group changes only its named variable."""

    common_env = {
        "HYDRA_FULL_ERROR": "1",
        "PYTHONFAULTHANDLER": "1",
        "UNILAB_NATIVE_ABORT_ON_CORRUPTION": ("1" if native_abort_on_corruption else "0"),
    }
    common_env.update({key: str(value) for key, value in role_env.items()})

    def spec(
        name: str,
        *,
        mode: str,
        device: str,
        updates: int,
        rounds: int,
        method: str,
    ) -> StageSpec:
        stage_dir = work_dir / "stages" / name
        return StageSpec(
            name=name,
            command=_owner_command(
                uv,
                mode=mode,
                stage_dir=stage_dir,
                aggregate=aggregate,
                checkpoint=checkpoint,
                teacher_checkpoint=teacher_checkpoint,
                device=device,
                batch_size=batch_size,
                max_updates=updates,
                rounds=rounds,
            ),
            env_overrides=common_env,
            timeout_seconds=timeout_seconds,
            method=method,
        )

    restart = [
        spec(
            f"gpu_restart_round_{index:02d}",
            mode="offline",
            device=gpu_device,
            updates=lifecycle_updates,
            rounds=1,
            method="real-owner-gpu-restart-each-round",
        )
        for index in range(1, lifecycle_rounds + 1)
    ]
    dual = [
        spec(
            f"gpu_dual_resident_{index:02d}",
            mode="offline",
            device=gpu_device,
            updates=lifecycle_updates,
            rounds=lifecycle_rounds,
            method="real-owner-gpu-dual-resident",
        )
        for index in (1, 2)
    ]
    return {
        "assembly_device": [
            spec(
                "aggregate_cpu_fresh",
                mode="assemble",
                device="cpu",
                updates=1,
                rounds=1,
                method="real-aggregate-fresh-process",
            ),
            spec(
                "aggregate_gpu_fresh",
                mode="assemble",
                device=gpu_device,
                updates=1,
                rounds=1,
                method="real-aggregate-fresh-process",
            ),
        ],
        "offline_device": [
            spec(
                "offline_cpu_fresh",
                mode="offline",
                device="cpu",
                updates=fresh_updates,
                rounds=1,
                method="real-offline-fresh-process",
            ),
            spec(
                "offline_gpu_fresh",
                mode="offline",
                device=gpu_device,
                updates=fresh_updates,
                rounds=1,
                method="real-offline-fresh-process",
            ),
        ],
        "gpu_continuous": [
            spec(
                "gpu_continuous",
                mode="offline",
                device=gpu_device,
                updates=lifecycle_updates,
                rounds=lifecycle_rounds,
                method="real-owner-gpu-continuous",
            )
        ],
        "gpu_restart_each_round": restart,
        "gpu_dual_resident": dual,
    }


def _run_one_stage(spec: StageSpec, work_dir: Path, *, kernel_since: float) -> dict[str, Any]:
    stage_dir = work_dir / "stages" / spec.name
    collect_health_snapshot(stage_dir / "health-before.json", kernel_since_epoch=kernel_since)
    result = run_stage(spec, stage_dir, monitor_interval_seconds=1.0)
    collect_health_snapshot(stage_dir / "health-after.json", kernel_since_epoch=kernel_since)
    owner_report = stage_dir / "owner-report.json"
    result["owner_report_exists"] = owner_report.is_file()
    stderr_path = stage_dir / "stderr.log"
    if stderr_path.is_file():
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        if "InterpolationResolutionError" in stderr or "Environment variable" in stderr:
            result["configuration_error"] = True
            result["configuration_error_tail"] = stderr[-4000:]
    return result


def _run_dual_group(
    specs: Sequence[StageSpec],
    work_dir: Path,
    *,
    kernel_since: float,
) -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_run_one_stage, spec, work_dir, kernel_since=kernel_since)
            for spec in specs
        ]
        return [future.result() for future in futures]


def _core_inventory() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for root in (Path("/var/crash"), Path("/var/lib/apport/coredump")):
        try:
            entries = tuple(root.iterdir())
        except Exception:
            continue
        for path in entries:
            try:
                stat = path.stat()
            except Exception:
                continue
            if path.is_file():
                rows[str(path)] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return rows


def _prune_generated_models(work_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((work_dir / "stages").glob("**/*.pt")):
        if not path.is_file():
            continue
        records.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
        path.unlink()
    _write_json(work_dir / "generated_model_manifest.json", {"removed_after_hash": records})
    return records


def _verdict(stage_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name = {str(row["name"]): row for row in stage_results}
    cpu = by_name.get("offline_cpu_fresh", {})
    gpu = by_name.get("offline_gpu_fresh", {})
    failed = [str(row["name"]) for row in stage_results if row.get("status") != "completed"]
    config_failed = [str(row["name"]) for row in stage_results if row.get("configuration_error")]
    owner_path_failed = [
        str(row["name"])
        for row in stage_results
        if row.get("status") != "completed"
        and row.get("evidence_level")
        in {"native-symptom-confirmed", "first-invalid-operation-confirmed"}
    ]
    if config_failed:
        boundary = "CAMPAIGN_CONFIGURATION_FAILED"
    elif cpu.get("status") != "completed" and str(cpu.get("name")) in owner_path_failed:
        boundary = "REAL_CPU_OFFLINE_OWNER_PATH_REPRODUCED"
    elif gpu.get("status") != "completed" and str(gpu.get("name")) in owner_path_failed:
        boundary = "REAL_GPU_OFFLINE_OWNER_PATH_REPRODUCED"
    elif any(name in owner_path_failed for name in failed):
        boundary = "GPU_LIFECYCLE_DIFFERENTIAL_REPRODUCED"
    elif failed:
        boundary = "OWNER_PATH_FAILED_WITHOUT_NATIVE_EVIDENCE"
    else:
        boundary = "OFFLINE_OWNER_PATH_NOT_REPRODUCED"
    return {
        "boundary": boundary,
        "failed_stages": failed,
        "configuration_failed_stages": config_failed,
        "native_evidence_failed_stages": owner_path_failed,
        "root_cause_owner": "unconfirmed until first-invalid-operation evidence",
        "formal_live_training_recommendation": (
            "NOT_AUTHORIZED_AUTOMATICALLY; a clean offline campaign alone does not prove "
            "simulator/persistent-live ownership"
        ),
    }


def _selected_groups(raw_groups: str, known_groups: Iterable[str]) -> list[str]:
    known = list(known_groups)
    if raw_groups.strip() == "all":
        return known
    selected = [item.strip() for item in raw_groups.split(",") if item.strip()]
    unknown = sorted(set(selected).difference(known))
    if unknown:
        raise ValueError(
            "unknown stage group(s): " + ",".join(unknown) + "; known groups: " + ",".join(known)
        )
    if not selected:
        raise ValueError("--groups must be 'all' or a comma-separated non-empty group list")
    return selected


def _selected_stage_names(raw_stage_names: str | None) -> set[str] | None:
    if raw_stage_names is None or raw_stage_names.strip() in ("", "all"):
        return None
    selected = {item.strip() for item in raw_stage_names.split(",") if item.strip()}
    if not selected:
        raise ValueError("--stage-names must be 'all' or a comma-separated stage list")
    return selected


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--stand-teacher-checkpoint",
        type=Path,
        default=Path("/ssd1/cyx/UniLab/model/G1StandStill/model_5000.pt"),
    )
    parser.add_argument("--existing-apport", type=Path, required=True)
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--fresh-updates", type=int, default=6000)
    parser.add_argument("--lifecycle-updates", type=int, default=2048)
    parser.add_argument("--lifecycle-rounds", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=10800.0)
    parser.add_argument(
        "--groups",
        default="all",
        help=(
            "Comma-separated stage groups to run. Use 'all' for the full campaign. "
            "Known groups: assembly_device,offline_device,gpu_continuous,"
            "gpu_restart_each_round,gpu_dual_resident."
        ),
    )
    parser.add_argument(
        "--stage-names",
        default=None,
        help="Optional comma-separated stage-name filter inside selected groups.",
    )
    parser.add_argument(
        "--native-abort-on-corruption",
        action="store_true",
        help="Set UNILAB_NATIVE_ABORT_ON_CORRUPTION=1 for new stages.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    campaign_started = time.time()
    token = datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = (args.work_root.resolve() / f"{token}_distill-real-owner-one-shot").resolve()
    work_dir.mkdir(parents=True, exist_ok=False)

    aggregate = args.aggregate.resolve()
    checkpoint = args.checkpoint.resolve()
    teacher_checkpoint = args.teacher_checkpoint.resolve()
    stand_teacher_checkpoint = args.stand_teacher_checkpoint.resolve()
    existing_apport = args.existing_apport.resolve()
    for path in (
        aggregate,
        checkpoint,
        teacher_checkpoint,
        stand_teacher_checkpoint,
        existing_apport,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    sources, _dimensions = _sources_from_seed_aggregate(aggregate)
    walk_dataset = next(
        (
            Path(source["path"]).resolve()
            for source in sources
            if source.get("scenario") == "walk_flat" or source.get("role") == "walk_flat"
        ),
        None,
    )
    stand_dataset = next(
        (
            Path(source["path"]).resolve()
            for source in sources
            if source.get("scenario") == "static_stand" or source.get("role") == "static_stand"
        ),
        None,
    )
    if walk_dataset is None or stand_dataset is None:
        raise ValueError("seed aggregate must expose walk_flat and static_stand source paths")
    role_env = {
        "UNILAB_G1_WALK_TEACHER": str(teacher_checkpoint),
        "UNILAB_G1_STAND_TEACHER": str(stand_teacher_checkpoint),
        "UNILAB_G1_WALK_DATASET": str(walk_dataset),
        "UNILAB_G1_STAND_DATASET": str(stand_dataset),
    }
    input_paths = [
        aggregate,
        checkpoint,
        teacher_checkpoint,
        stand_teacher_checkpoint,
        *(Path(source["path"]) for source in sources),
    ]
    input_paths = list(dict.fromkeys(path.resolve() for path in input_paths))
    preflight = collect_preflight(input_paths)
    preflight["synthetic_numeric_probe_classification"] = (
        "platform-pressure auxiliary only; it does not cross aggregate, MoE update, "
        "checkpoint reload, or persistent runtime owners"
    )
    preflight["native_abort_policy"] = (
        "UNILAB_NATIVE_ABORT_ON_CORRUPTION=1 for selected new stages"
        if bool(args.native_abort_on_corruption)
        else "UNILAB_NATIVE_ABORT_ON_CORRUPTION=0 for all new stages so original exceptions survive"
    )
    preflight["role_environment"] = role_env
    _write_json(work_dir / "preflight.json", preflight)
    identity_before = preflight["inputs"]
    core_before = _core_inventory()
    _write_json(work_dir / "core-inventory-before.json", core_before)
    collect_health_snapshot(work_dir / "health-before.json", kernel_since_epoch=campaign_started)

    existing_core = analyze_native_core_artifact(
        artifact=existing_apport,
        capture_dir=work_dir / "native_cores" / "existing-gpu-sync-replay",
        gdb_path=shutil.which("gdb"),
        apport_unpack_path=shutil.which("apport-unpack"),
    )
    _write_json(work_dir / "existing-apport-analysis.json", existing_core)

    uv = shutil.which("uv")
    if uv is None:
        raise FileNotFoundError("uv is required by the UniLab execution contract")
    matrix = build_stage_matrix(
        uv=uv,
        work_dir=work_dir,
        aggregate=aggregate,
        checkpoint=checkpoint,
        teacher_checkpoint=teacher_checkpoint,
        role_env=role_env,
        gpu_device=str(args.gpu_device),
        batch_size=int(args.batch_size),
        fresh_updates=int(args.fresh_updates),
        lifecycle_updates=int(args.lifecycle_updates),
        lifecycle_rounds=int(args.lifecycle_rounds),
        timeout_seconds=float(args.timeout_seconds),
        native_abort_on_corruption=bool(args.native_abort_on_corruption),
    )
    selected_groups = _selected_groups(str(args.groups), matrix.keys())
    selected_stage_names = _selected_stage_names(args.stage_names)
    _write_json(
        work_dir / "differential-contract.json",
        {
            "selected_groups": selected_groups,
            "selected_stage_names": (
                None if selected_stage_names is None else sorted(selected_stage_names)
            ),
            "native_abort_on_corruption": bool(args.native_abort_on_corruption),
            "assembly_device": "CPU fresh vs GPU fresh; device only",
            "offline_device": "CPU fresh vs GPU fresh; device only; 6000 updates crosses r10 failure 4915",
            "gpu_continuous_vs_restart": (
                "same aggregate/checkpoint/config/update budget/round count; process lifetime only"
            ),
            "gpu_continuous_vs_dual": (
                "same per-process workload and persistent owner chain; concurrent owner count only"
            ),
            "groups": {
                name: [
                    {
                        "name": spec.name,
                        "method": spec.method,
                        "command": list(spec.command),
                        "env_overrides": dict(spec.env_overrides),
                    }
                    for spec in specs
                ]
                for name, specs in matrix.items()
            },
        },
    )

    stage_results: list[dict[str, Any]] = []
    for group_name in selected_groups:
        if group_name == "gpu_dual_resident":
            continue
        for spec in matrix[group_name]:
            if selected_stage_names is not None and spec.name not in selected_stage_names:
                continue
            stage_results.append(_run_one_stage(spec, work_dir, kernel_since=campaign_started))
    if "gpu_dual_resident" in selected_groups:
        dual_specs = matrix["gpu_dual_resident"]
        if selected_stage_names is not None:
            dual_specs = [spec for spec in dual_specs if spec.name in selected_stage_names]
        stage_results.extend(_run_dual_group(dual_specs, work_dir, kernel_since=campaign_started))
    if not stage_results:
        raise ValueError("no stages selected; check --groups and --stage-names")

    collect_health_snapshot(work_dir / "health-after.json", kernel_since_epoch=campaign_started)
    core_after = _core_inventory()
    _write_json(work_dir / "core-inventory-after.json", core_after)
    new_cores = harvest_native_cores(
        work_dir=work_dir,
        since_epoch=campaign_started,
        gdb_path=shutil.which("gdb"),
    )
    identity_after = _identity(input_paths)
    inputs_unchanged = identity_before == identity_after
    generated_models = _prune_generated_models(work_dir)
    verdict = _verdict(stage_results)
    summary = {
        "campaign_status": "completed" if inputs_unchanged else "input-identity-changed",
        "work_dir": str(work_dir),
        "owner_path": (
            "aggregate assembly -> torch.load/map_location -> offline MoE update -> "
            "checkpoint reload -> PersistentDistillationRuntime -> SharedWeightSync -> cleanup"
        ),
        "existing_gpu_sync_apport": existing_core,
        "stage_results": stage_results,
        "verdict": verdict,
        "inputs_unchanged": inputs_unchanged,
        "inputs_before": identity_before,
        "inputs_after": identity_after,
        "new_native_cores": new_cores,
        "generated_models_removed_after_hash": generated_models,
        "retrieval_archive": str(work_dir.with_name(work_dir.name + "-RETURN_ME.tar.gz")),
    }
    _write_json(work_dir / "campaign-summary.json", summary)
    (work_dir / "RETURN_ME.txt").write_text(
        f"verdict={verdict['boundary']}\narchive={summary['retrieval_archive']}\n",
        encoding="utf-8",
    )
    archive = create_archive(work_dir)
    print(
        json.dumps(
            {
                "campaign_status": summary["campaign_status"],
                "verdict": verdict["boundary"],
                "retrieval_archive": str(archive),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if inputs_unchanged else 2


if __name__ == "__main__":
    raise SystemExit(main())
