from datetime import datetime
from pathlib import Path

import pytest

from unilab.algos.torch.distill.formal_identity import (
    FormalDaggerIdentitySpec,
    build_formal_command_identity,
    build_formal_freeze_document,
    build_formal_oracle_source,
    build_formal_supervisor_source,
    resolve_time_sorted_formal_output_identity,
)


def _spec(tmp_path: Path, **overrides: object) -> FormalDaggerIdentitySpec:
    values: dict[str, object] = {
        "repo_root": tmp_path / "repo",
        "parent_run_dir": tmp_path / "parent_iteration_3",
        "run_dir": tmp_path / "formal_run_r1",
        "parent_iteration": 3,
        "dagger_iterations": 4,
        "configured_update_floor": 512,
        "effective_updates_by_iteration": (12320, 12352, 12384, 12416),
        "seed": 0,
        "device": "cuda:0",
        "collect_num_envs": 16,
        "samples_per_role": 512,
        "batch_size": 512,
        "execution_mode": "persistent_async",
    }
    values.update(overrides)
    return FormalDaggerIdentitySpec(**values)  # type: ignore[arg-type]


def test_formal_identity_builds_owner_cli_command_from_original_parent(
    tmp_path: Path,
) -> None:
    identity = build_formal_command_identity(_spec(tmp_path))

    assert identity["training_executed"] is False
    assert identity["lineage"] == {
        "parent_iteration": 3,
        "source": "original_parent_iteration_3",
        "r6_sentinel_promoted": False,
    }
    assert identity["workload"]["dagger_iterations"] == 4
    assert identity["workload"]["effective_updates_by_iteration"] == [
        12320,
        12352,
        12384,
        12416,
    ]
    assert identity["workload"]["total_effective_updates"] == 49472
    assert identity["argv"][:8] == [
        "uv",
        "run",
        "--no-sync",
        "train",
        "--algo",
        "distill",
        "--task",
        "g1_walk_flat",
    ]
    assert "workflow=g1_walk_stand" in identity["argv"]
    assert "training.workflow.dagger_iterations=4" in identity["argv"]
    assert "training.workflow.dagger_updates_per_iteration=512" in identity["argv"]
    assert identity["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert identity["output_paths"]["run_dir"] == str(tmp_path / "formal_run_r1")
    assert "freeze" not in identity["output_paths"]
    assert identity["materialization_paths"]["freeze"].endswith(".freeze.json")


def test_formal_identity_materializes_iteration_aware_effective_update_schedule(
    tmp_path: Path,
) -> None:
    spec = FormalDaggerIdentitySpec(
        repo_root=tmp_path / "repo",
        parent_run_dir=tmp_path / "parent_iteration_3",
        run_dir=tmp_path / "formal_run_r1",
        parent_iteration=3,
        dagger_iterations=2,
        configured_update_floor=512,
        effective_updates_by_iteration=(12320, 12352),
        seed=0,
        device="cuda:0",
        collect_num_envs=16,
        samples_per_role=512,
        batch_size=512,
        execution_mode="persistent_async",
    )

    identity = build_formal_command_identity(spec)

    assert identity["workload"]["effective_updates_by_iteration"] == [12320, 12352]
    assert identity["workload"]["total_effective_updates"] == 24672


def test_fresh_formal_identity_owns_bootstrap_and_has_no_parent_lineage(
    tmp_path: Path,
) -> None:
    spec = FormalDaggerIdentitySpec(
        repo_root=tmp_path / "repo",
        parent_run_dir=None,
        run_dir=tmp_path / "formal_fresh_r1",
        parent_iteration=0,
        dagger_iterations=8,
        configured_update_floor=128,
        effective_updates_by_iteration=tuple(4096 * i for i in range(1, 9)),
        seed=0,
        device="cuda:0",
        collect_num_envs=64,
        samples_per_role=65536,
        batch_size=512,
        execution_mode="persistent_async",
        mode="fresh",
        artifact_dir=tmp_path / "formal_role_artifacts_r1",
        bootstrap_updates=20000,
        adopt_legacy_artifacts=False,
        transition_max_env_steps=24576,
    )

    identity = build_formal_command_identity(spec)

    assert identity["lineage"] == {
        "parent_iteration": None,
        "source": "fresh_teacher_bootstrap",
        "r6_sentinel_promoted": False,
    }
    assert "training.workflow.mode=fresh" in identity["argv"]
    assert not any("parent_run_dir=" in arg for arg in identity["argv"])
    assert "training.workflow.bootstrap_updates=20000" in identity["argv"]
    assert "training.workflow.adopt_legacy_artifacts=false" in identity["argv"]
    assert "training.workflow.transition_max_env_steps=24576" in identity["argv"]
    assert identity["workload"]["total_effective_updates"] == 147456
    assert identity["output_paths"]["artifact_dir"] == str(tmp_path / "formal_role_artifacts_r1")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"parent_iteration": 2}, "parent iteration 3"),
        ({"parent_run_dir": Path("/tmp/r6_sentinel")}, "r6 sentinel"),
        ({"run_dir": Path("/tmp/hp7c3_bounded_persistent_r6")}, "r6 sentinel"),
        ({"dagger_iterations": 0}, "dagger_iterations"),
        (
            {"effective_updates_by_iteration": (12320, 0, 12384, 12416)},
            "effective_updates_by_iteration",
        ),
        ({"execution_mode": "legacy"}, "persistent_async"),
    ],
)
def test_formal_identity_rejects_unfrozen_or_sentinel_lineage(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_formal_command_identity(_spec(tmp_path, **override))


def test_formal_identity_requires_fresh_outputs(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec.run_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="formal output already exists"):
        build_formal_command_identity(spec)


def test_time_sorted_output_identity_derives_fresh_paths_from_run_name(tmp_path: Path) -> None:
    identity = resolve_time_sorted_formal_output_identity(
        repo_root=tmp_path,
        run_name="g1_walk_stand_fresh_oom_r2",
        mode="fresh",
        now=datetime(2026, 7, 20, 9, 8, 7),
    )

    assert identity.stem == "20260720-090807_g1_walk_stand_fresh_oom_r2"
    assert identity.run_dir == (tmp_path / "logs" / "distill_workflow" / identity.stem)
    assert identity.artifact_dir == (tmp_path / "logs" / "distill_role_artifacts" / identity.stem)


@pytest.mark.parametrize("run_name", ["", "../escape", "name/child", "has space"])
def test_time_sorted_output_identity_rejects_unsafe_run_names(
    tmp_path: Path, run_name: str
) -> None:
    with pytest.raises(ValueError, match="run_name"):
        resolve_time_sorted_formal_output_identity(
            repo_root=tmp_path,
            run_name=run_name,
            mode="fresh",
            now=datetime(2026, 7, 20, 9, 8, 7),
        )


def test_formal_supervisor_executes_only_the_frozen_argv(tmp_path: Path) -> None:
    identity = build_formal_command_identity(_spec(tmp_path))

    source = build_formal_supervisor_source(identity)

    assert f"cd {tmp_path / 'repo'}" in source
    for output_path in identity["output_paths"].values():
        assert f"test ! -e {output_path}" in source
    assert "nvidia-smi" in source
    assert "/usr/bin/time -v" in source
    assert "uv run --no-sync train --algo distill" in source
    assert "training.workflow.dagger_iterations=4" in source
    assert "hp7c3" not in source.lower()
    assert "r6" not in source.lower()


def test_formal_oracle_is_syntax_valid_and_preflight_never_trains(tmp_path: Path) -> None:
    source = build_formal_oracle_source()

    compile(source, "formal_oracle.py", "exec")
    assert '"training_executed": False' in source
    assert 'parser.add_argument("--preflight"' in source
    assert "output paths already exist" in source
    assert 'subprocess.run(freeze["command"]' not in source
    assert "effective_updates_by_iteration" in source
    assert 'iteration.get("updates")' in source


def test_formal_freeze_hashes_runtime_and_training_inputs(tmp_path: Path) -> None:
    source = tmp_path / "workflow.py"
    artifact = tmp_path / "parent_iteration_3.pt"
    source.write_text("runtime-owner\n")
    artifact.write_bytes(b"formal-parent")
    identity = build_formal_command_identity(_spec(tmp_path))

    freeze = build_formal_freeze_document(
        identity,
        repo_root=tmp_path,
        head="a" * 40,
        source_paths={"workflow": source},
        hard_artifact_paths={"parent_checkpoint": artifact},
        runtime_diff_clean=True,
    )

    assert freeze["accepted"] is True
    assert freeze["failures"] == []
    assert freeze["training_executed"] is False
    assert freeze["repo"]["head"] == "a" * 40
    assert freeze["repo"]["runtime_diff_clean"] is True
    assert freeze["source_identity"]["workflow"]["size"] == len("runtime-owner\n")
    assert freeze["hard_artifacts"]["parent_checkpoint"]["size"] == len(b"formal-parent")
    assert freeze["command"]["lineage"]["r6_sentinel_promoted"] is False


def test_formal_freeze_fails_closed_on_dirty_or_missing_inputs(tmp_path: Path) -> None:
    identity = build_formal_command_identity(_spec(tmp_path))

    freeze = build_formal_freeze_document(
        identity,
        repo_root=tmp_path,
        head="b" * 40,
        source_paths={"missing_source": tmp_path / "missing.py"},
        hard_artifact_paths={"missing_parent": tmp_path / "missing.pt"},
        runtime_diff_clean=False,
    )

    assert freeze["accepted"] is False
    assert "runtime source diff is dirty" in freeze["failures"]
    assert "missing source: missing_source" in freeze["failures"]
    assert "missing hard artifact: missing_parent" in freeze["failures"]
