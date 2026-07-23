from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch
from scripts.deploy import diagnose_distill_native_corruption, diagnose_distill_real_owner_one_shot
from scripts.deploy.check_distill_real_owner_path import (
    _PersistentCheckpointExercise,
    run_aggregate_assembly,
)
from scripts.deploy.diagnose_distill_native_corruption import analyze_native_core_artifact
from scripts.deploy.diagnose_distill_real_owner_one_shot import build_stage_matrix

from unilab.algos.torch.distill import (
    MoEStudentPolicy,
    build_distillation_dataset,
    build_multitask_distillation_dataset,
    save_distillation_checkpoint,
    save_distillation_dataset,
)


def _write_real_seed_aggregate(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stand = build_distillation_dataset(
        torch.zeros(2, 4),
        torch.zeros(2, 4),
        teacher_actions=torch.zeros(2, 2),
        commands=torch.zeros(2, 3),
        command_intents=("inactive", "inactive"),
        role_labels=("stand", "stand"),
    )
    walk = build_distillation_dataset(
        torch.ones(2, 4),
        torch.ones(2, 4),
        teacher_actions=torch.ones(2, 2),
        commands=torch.full((2, 3), 0.2),
        command_intents=("active", "active"),
        role_labels=("walk_flat", "walk_flat"),
    )
    paths = {"stand": root / "stand.pt", "walk": root / "walk.pt"}
    save_distillation_dataset(paths["stand"], stand)
    save_distillation_dataset(paths["walk"], walk)
    sources = [
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
    ]
    aggregate = build_multitask_distillation_dataset(
        sources,
        expected_student_obs_dim=4,
        expected_teacher_obs_dim=4,
        expected_teacher_action_dim=2,
    )
    path = root / "seed-aggregate.pt"
    save_distillation_dataset(path, aggregate)
    return path


def _save_moe_checkpoint(path: Path) -> Path:
    policy = MoEStudentPolicy(
        obs_dim=4,
        action_dim=2,
        num_experts=2,
        expert_hidden_dims=(8,),
        router_hidden_dims=(8,),
        routing_mode="soft",
    )
    save_distillation_checkpoint(
        path,
        student=policy,
        agent_steps=7,
        distill_runtime_cfg={
            "student_model_type": "moe",
            "student_obs_dim": 4,
            "student_action_dim": 2,
            "student_activation": "elu",
            "student_squash_action": True,
            "student_num_experts": 2,
            "student_expert_hidden_dims": [8],
            "student_router_hidden_dims": [8],
            "student_routing_mode": "soft",
            "student_router_temperature": 1.0,
        },
    )
    return path


def test_real_aggregate_worker_rebuilds_seed_through_production_owner(tmp_path: Path) -> None:
    seed = _write_real_seed_aggregate(tmp_path / "inputs")
    output = tmp_path / "output" / "rebuilt.pt"

    result = run_aggregate_assembly(
        seed_aggregate=seed,
        output=output,
        device="cpu",
        keep_output=False,
    )

    assert result["status"] == "PASS"
    assert result["semantic_match_to_seed"] is True
    assert result["source_count"] == 2
    assert result["signature"]["scenario_counts"] == {
        "static_stand": 2,
        "walk_flat": 2,
    }
    assert not output.exists()


def test_real_checkpoint_exercise_uses_persistent_runtime_and_unlinks_shared_memory(
    tmp_path: Path,
) -> None:
    checkpoint = _save_moe_checkpoint(tmp_path / "student.pt")
    exercise = _PersistentCheckpointExercise(
        checkpoint=checkpoint,
        device="cpu",
        output_dir=tmp_path / "runtime",
    )
    try:
        first = exercise.activate(checkpoint, label="first", iteration=1)
        second = exercise.activate(checkpoint, label="second", iteration=2)
    finally:
        cleanup = exercise.close()

    assert (first["version"], second["version"]) == (1, 2)
    assert first["worker_pid"] == second["worker_pid"]
    assert first["weight_match"] is second["weight_match"] is True
    assert cleanup["shared_memory_unlinked"] is True


def test_stage_matrix_keeps_real_cpu_gpu_and_lifecycle_controls_matched(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("aggregate.pt", "student.pt", "teacher.pt")]
    role_env = {
        "UNILAB_G1_WALK_TEACHER": str(paths[2]),
        "UNILAB_G1_STAND_TEACHER": str(tmp_path / "stand_teacher.pt"),
        "UNILAB_G1_WALK_DATASET": str(tmp_path / "walk_dataset.pt"),
        "UNILAB_G1_STAND_DATASET": str(tmp_path / "stand_dataset.pt"),
    }
    matrix = build_stage_matrix(
        uv="/usr/bin/uv",
        work_dir=tmp_path / "campaign",
        aggregate=paths[0],
        checkpoint=paths[1],
        teacher_checkpoint=paths[2],
        role_env=role_env,
        gpu_device="cuda:0",
        batch_size=512,
        fresh_updates=6000,
        lifecycle_updates=2048,
        lifecycle_rounds=3,
        timeout_seconds=60.0,
    )

    assembly = matrix["assembly_device"]
    offline = matrix["offline_device"]
    assert [spec.command[spec.command.index("--device") + 1] for spec in assembly] == [
        "cpu",
        "cuda:0",
    ]
    assert [spec.command[spec.command.index("--max-updates") + 1] for spec in offline] == [
        "6000",
        "6000",
    ]
    assert [spec.command[spec.command.index("--device") + 1] for spec in offline] == [
        "cpu",
        "cuda:0",
    ]
    assert (
        matrix["gpu_continuous"][0].command[
            matrix["gpu_continuous"][0].command.index("--rounds") + 1
        ]
        == "3"
    )
    assert len(matrix["gpu_restart_each_round"]) == 3
    assert all(
        spec.command[spec.command.index("--rounds") + 1] == "1"
        for spec in matrix["gpu_restart_each_round"]
    )
    assert len(matrix["gpu_dual_resident"]) == 2
    assert all(
        spec.env_overrides["UNILAB_NATIVE_ABORT_ON_CORRUPTION"] == "0"
        for specs in matrix.values()
        for spec in specs
    )
    assert all(
        spec.env_overrides["UNILAB_G1_WALK_TEACHER"] == str(paths[2])
        and spec.env_overrides["UNILAB_G1_STAND_DATASET"].endswith("stand_dataset.pt")
        for specs in matrix.values()
        for spec in specs
    )
    abort_matrix = build_stage_matrix(
        uv="/usr/bin/uv",
        work_dir=tmp_path / "abort-campaign",
        aggregate=paths[0],
        checkpoint=paths[1],
        teacher_checkpoint=paths[2],
        role_env=role_env,
        gpu_device="cuda:0",
        batch_size=512,
        fresh_updates=6000,
        lifecycle_updates=2048,
        lifecycle_rounds=3,
        timeout_seconds=60.0,
        native_abort_on_corruption=True,
    )
    assert all(
        spec.env_overrides["UNILAB_NATIVE_ABORT_ON_CORRUPTION"] == "1"
        for specs in abort_matrix.values()
        for spec in specs
    )


def test_real_owner_verdict_separates_config_failure_from_native_reproduction() -> None:
    verdict = diagnose_distill_real_owner_one_shot._verdict(
        [
            {
                "name": "offline_cpu_fresh",
                "status": "failed",
                "evidence_level": "unconfirmed",
                "configuration_error": True,
            }
        ]
    )

    assert verdict["boundary"] == "CAMPAIGN_CONFIGURATION_FAILED"
    assert verdict["configuration_failed_stages"] == ["offline_cpu_fresh"]


def test_real_owner_selected_groups_allow_offline_fresh_only() -> None:
    groups = diagnose_distill_real_owner_one_shot._selected_groups(
        "offline_device",
        [
            "assembly_device",
            "offline_device",
            "gpu_continuous",
            "gpu_restart_each_round",
            "gpu_dual_resident",
        ],
    )

    assert groups == ["offline_device"]


def test_real_owner_selected_groups_reject_unknown_group() -> None:
    with pytest.raises(ValueError, match="unknown stage group"):
        diagnose_distill_real_owner_one_shot._selected_groups(
            "offline_device,gpu_lifecycle_all",
            ["assembly_device", "offline_device"],
        )


def test_real_owner_selected_stage_names_allow_single_cpu_capture() -> None:
    assert diagnose_distill_real_owner_one_shot._selected_stage_names("offline_cpu_fresh") == {
        "offline_cpu_fresh"
    }
    assert diagnose_distill_real_owner_one_shot._selected_stage_names("all") is None


def test_real_owner_verdict_requires_native_evidence_for_reproduced_boundary() -> None:
    verdict = diagnose_distill_real_owner_one_shot._verdict(
        [
            {
                "name": "offline_cpu_fresh",
                "status": "failed",
                "evidence_level": "native-symptom-confirmed",
            }
        ]
    )

    assert verdict["boundary"] == "REAL_CPU_OFFLINE_OWNER_PATH_REPRODUCED"
    assert verdict["native_evidence_failed_stages"] == ["offline_cpu_fresh"]


def test_apport_report_is_unpacked_before_gdb_receives_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crash = tmp_path / "gpu_sync_replay.crash"
    crash.write_text("ProblemType: Crash\n", encoding="utf-8")
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        command = list(command)
        commands.append(command)
        if command[0] == "/usr/bin/apport-unpack":
            unpack_dir = Path(command[2])
            (unpack_dir / "CoreDump").write_bytes(b"raw-core")
            (unpack_dir / "ExecutablePath").write_text("/bin/sh\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "unpacked\n", "")
        assert str(crash) not in command
        core_arg = next(value for value in command if value.endswith("/CoreDump"))
        assert Path(core_arg).read_bytes() == b"raw-core"
        return subprocess.CompletedProcess(
            command,
            0,
            "HANDLED_EXCEPTION: TypeError('cell object is not callable')\n",
            "",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    record = analyze_native_core_artifact(
        artifact=crash,
        capture_dir=tmp_path / "capture",
        gdb_path="/usr/bin/gdb",
        apport_unpack_path="/usr/bin/apport-unpack",
    )

    assert record["artifact_kind"] == "apport-report"
    assert record["status"] == "analyzed"
    assert record["core_path_passed_to_gdb"].endswith("/CoreDump")
    assert record["core_path_passed_to_gdb"] != str(crash)
    assert record["unpacked_core_removed_after_analysis"] is True
    assert "HANDLED_EXCEPTION" in Path(record["gdb_output"]).read_text()
    assert commands[0][0] == "/usr/bin/apport-unpack"


def test_gdb_command_file_keeps_python_backtrace_best_effort(tmp_path: Path) -> None:
    command_file = tmp_path / "gdb.txt"

    diagnose_distill_native_corruption._gdb_command_file(command_file)
    text = command_file.read_text()

    assert 'run_optional("PY_BT", "thread apply all py-bt")' in text
    assert "thread apply all py-bt\ninfo sharedlibrary" not in text
