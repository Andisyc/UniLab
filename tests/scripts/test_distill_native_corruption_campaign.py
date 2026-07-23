from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from unilab.algos.torch.distill import build_distillation_dataset, save_distillation_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> Any:
    path = ROOT_DIR / "scripts" / "deploy" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_sources(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stand = build_distillation_dataset(
        torch.zeros(2, 4),
        torch.zeros(2, 4),
        teacher_actions=torch.zeros(2, 3),
        commands=torch.zeros(2, 3),
        command_intents=("inactive", "inactive"),
        role_labels=("stand", "stand"),
    )
    walk = build_distillation_dataset(
        torch.ones(2, 4),
        torch.ones(2, 4),
        teacher_actions=torch.ones(2, 3),
        commands=torch.full((2, 3), 0.2),
        command_intents=("active", "active"),
        role_labels=("walk_flat", "walk_flat"),
    )
    transition = build_distillation_dataset(
        torch.full((4, 4), 2.0),
        torch.full((4, 4), 2.0),
        teacher_actions=torch.full((4, 3), 2.0),
        commands=torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        command_intents=("active", "active", "inactive", "inactive"),
        role_labels=("walk_flat", "walk_flat", "stand", "stand"),
        scenario_labels=("walk_to_stop",) * 4,
        transition_ages=torch.tensor([-1, -1, 0, 1], dtype=torch.int64),
        command_before=torch.full((4, 3), 0.2),
        command_after=torch.zeros(4, 3),
    )
    paths = {
        "stand": root / "stand.pt",
        "walk": root / "walk.pt",
        "transition": root / "transition.pt",
    }
    save_distillation_dataset(paths["stand"], stand)
    save_distillation_dataset(paths["walk"], walk)
    save_distillation_dataset(paths["transition"], transition)
    manifest = {
        "sources": [
            {
                "path": str(paths["stand"]),
                "role": "stand",
                "scenario": "static_stand",
                "preserve_row_role_labels": True,
            },
            {
                "path": str(paths["walk"]),
                "role": "walk_flat",
                "scenario": "walk_flat",
                "preserve_row_role_labels": True,
            },
            {
                "path": str(paths["transition"]),
                "role": "walk_to_stop",
                "scenario": "walk_to_stop",
                "preserve_row_role_labels": True,
            },
        ]
    }
    manifest_path = root / "sources.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_role_data_lifecycle_crosses_real_owner_path_and_releases_outputs(tmp_path: Path) -> None:
    lifecycle = _load_script("check_distill_role_data_lifecycle")
    manifest_path = _write_sources(tmp_path / "sources")

    result = lifecycle.run_lifecycle(
        work_dir=tmp_path / "lifecycle",
        cycles=3,
        source_manifest=manifest_path,
        dataset_path=None,
        keep_cycle_outputs=False,
        report_every=10,
    )

    assert result["status"] == "completed"
    assert result["cycles_completed"] == 3
    assert result["baseline_fingerprint"]["num_samples"] == 8
    assert result["baseline_fingerprint"]["label_counts"] == {
        "command_intents": {"active": 4, "inactive": 4},
        "role_labels": {"stand": 4, "walk_flat": 4},
        "scenario_labels": {"static_stand": 2, "walk_flat": 2, "walk_to_stop": 4},
    }
    assert list((tmp_path / "lifecycle").glob("cycle-*.pt")) == []


def test_stage_capture_keeps_environment_and_failure_evidence_isolated(tmp_path: Path) -> None:
    campaign = _load_script("diagnose_distill_native_corruption")
    stage = campaign.StageSpec(
        name="synthetic_failure",
        command=(
            sys.executable,
            "-c",
            "import os,sys; print(os.environ['ONLY_THIS_STAGE']); "
            "print('Invalid write of size 8', file=sys.stderr); raise SystemExit(86)",
        ),
        env_overrides={"ONLY_THIS_STAGE": "yes", "PYTHONMALLOC": "debug"},
        timeout_seconds=10.0,
        method="memcheck",
    )

    result = campaign.run_stage(stage, tmp_path / "stage", monitor_interval_seconds=0.01)

    assert result["status"] == "failed"
    assert result["returncode"] == 86
    assert result["evidence_level"] == "first-invalid-operation-confirmed"
    assert "Invalid write" in (tmp_path / "stage" / "stderr.log").read_text()
    command_record = json.loads((tmp_path / "stage" / "command.json").read_text())
    assert command_record["env_overrides"] == {
        "ONLY_THIS_STAGE": "yes",
        "PYTHONMALLOC": "debug",
    }


def test_lifecycle_stage_matrix_never_stacks_active_diagnostics(tmp_path: Path) -> None:
    campaign = _load_script("diagnose_distill_native_corruption")
    dataset_path = tmp_path / "dataset.pt"
    dataset_path.write_bytes(b"identity-only")
    stages = campaign.build_lifecycle_stages(
        python_executable=sys.executable,
        lifecycle_script=ROOT_DIR / "scripts/deploy/check_distill_role_data_lifecycle.py",
        campaign_dir=tmp_path / "campaign",
        dataset_path=dataset_path,
        source_manifest=None,
        cycles=2,
        timeout_seconds=30.0,
        valgrind_path="/usr/bin/valgrind",
        rr_path="/usr/bin/rr",
    )

    assert [stage.name for stage in stages] == [
        "host_plain",
        "host_allocator_debug",
        "host_memcheck",
        "host_rr",
    ]
    by_name = {stage.name: stage for stage in stages}
    assert by_name["host_plain"].env_overrides == {}
    assert by_name["host_allocator_debug"].env_overrides["PYTHONMALLOC"] == "debug"
    assert "valgrind" in by_name["host_memcheck"].command[0]
    assert by_name["host_memcheck"].env_overrides == {}
    assert by_name["host_rr"].command[:2] == ("/usr/bin/rr", "record")
    assert by_name["host_rr"].env_overrides == {}


def test_campaign_main_creates_one_summary_and_retrieval_archive(tmp_path: Path) -> None:
    lifecycle = _load_script("check_distill_role_data_lifecycle")
    campaign = _load_script("diagnose_distill_native_corruption")
    manifest_path = _write_sources(tmp_path / "sources")
    aggregate = lifecycle.run_lifecycle(
        work_dir=tmp_path / "seed",
        cycles=1,
        source_manifest=manifest_path,
        dataset_path=None,
        keep_cycle_outputs=True,
        report_every=10,
    )
    dataset_path = Path(aggregate["last_output_path"])
    work_dir = tmp_path / "campaign"

    exit_code = campaign.main(
        [
            "--work-dir",
            str(work_dir),
            "--dataset",
            str(dataset_path),
            "--cycles",
            "1",
            "--timeout-seconds",
            "60",
            "--valgrind",
            "off",
            "--rr",
            "off",
        ]
    )

    assert exit_code == 0
    summary = json.loads((work_dir / "campaign_summary.json").read_text())
    assert summary["verdict"] == "INCONCLUSIVE_NOT_REPRODUCED"
    assert summary["stages"][0]["name"] == "host_plain"
    assert summary["stages"][1]["name"] == "host_allocator_debug"
    archive = Path(summary["retrieval_archive"])
    assert archive.is_file()
    assert archive.parent == work_dir.parent
    assert archive.name.endswith("-RETURN_ME.tar.gz")


def test_persistent_differential_builds_matched_isolated_lifecycles(tmp_path: Path) -> None:
    campaign = _load_script("diagnose_distill_native_corruption")
    checkpoints = [tmp_path / name for name in ("walk.pt", "stand.pt", "student.pt")]
    for path in checkpoints:
        path.write_bytes(b"identity")
    args = SimpleNamespace(
        persistent_differential_repetitions=3,
        student_checkpoint=checkpoints[2],
        offline_init_checkpoint=None,
        walk_teacher=checkpoints[0],
        stand_teacher=checkpoints[1],
        work_dir=tmp_path / "campaign",
        persistent_num_envs=2,
        persistent_samples=4,
        device="cuda:0",
        timeout_seconds=60.0,
    )

    stages, skipped = campaign._build_persistent_differential_stages(args, "/usr/bin/uv")

    assert skipped == []
    assert [stage.name for stage in stages] == [
        "collector_persistent",
        "collector_restart_each_request",
    ]
    assert [stage.method for stage in stages] == [
        "persistent-worker",
        "restart-each-request",
    ]
    assert stages[0].env_overrides == stages[1].env_overrides == {}
    assert stages[0].command[-1] == "persistent"
    assert stages[1].command[-1] == "restart_each_request"


def test_gpu_replay_stages_do_not_stack_sync_and_compute_sanitizer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign = _load_script("diagnose_distill_native_corruption")
    dataset, init_checkpoint, teacher = [
        tmp_path / name for name in ("dataset.pt", "init.pt", "teacher.pt")
    ]
    for path in (dataset, init_checkpoint, teacher):
        path.write_bytes(b"identity")
    args = SimpleNamespace(
        offline_init_checkpoint=init_checkpoint,
        teacher_checkpoint=teacher,
        offline_dataset=dataset,
        dataset=None,
        work_dir=tmp_path / "campaign",
        device="cuda:0",
        gpu_sync_updates=6000,
        gpu_memcheck_updates=32,
        offline_batch_size=512,
        timeout_seconds=60.0,
        compute_sanitizer="auto",
    )
    monkeypatch.setattr(
        campaign,
        "_resolve_optional_tool",
        lambda _mode, _name: ("/usr/bin/compute-sanitizer", None),
    )

    stages, skipped = campaign._build_gpu_stages(args, "/usr/bin/uv")

    assert skipped == []
    assert [stage.name for stage in stages] == ["gpu_sync_replay", "gpu_memcheck_replay"]
    assert stages[0].env_overrides["CUDA_LAUNCH_BLOCKING"] == "1"
    assert "PYTORCH_NO_CUDA_MEMORY_CACHING" not in stages[0].env_overrides
    assert stages[1].env_overrides == {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
    assert "CUDA_LAUNCH_BLOCKING" not in stages[1].env_overrides
    assert "--target-processes" in stages[1].command
