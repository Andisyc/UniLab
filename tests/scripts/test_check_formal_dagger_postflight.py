from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_oracle() -> ModuleType:
    path = Path("scripts/deploy/check_formal_dagger_postflight.py")
    spec = importlib.util.spec_from_file_location("formal_postflight", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path.as_posix()


def _fixture(tmp_path: Path, mod: ModuleType) -> tuple[Path, dict, dict, dict]:
    root = tmp_path / "repo"
    run = root / "run"
    source = Path(_write(root / "workflow.py", b"source"))
    parent_checkpoint = Path(_write(root / "parent.pt", b"parent"))
    parent_aggregate = Path(_write(root / "parent_aggregate.pt", b"aggregate"))
    input_sha = mod.file_sha256(parent_checkpoint)
    iterations = []
    metric_records = []
    for index, (updates, version) in enumerate(((2, 3), (4, 4)), start=1):
        checkpoint = Path(_write(run / f"checkpoint_{index}.pt", f"ckpt{index}".encode()))
        aggregate = Path(_write(run / f"aggregate_{index}.pt", f"agg{index}".encode()))
        scenarios = []
        for scenario in ("walk_flat", "static_stand", "walk_to_stop"):
            scenarios.append(
                {
                    "scenario": scenario,
                    "num_samples": 2,
                    "input_weight_version": version,
                    "collector_worker_pid": 100,
                }
            )
            metric_records.append(
                {
                    "identity": {
                        "outer_iteration": index,
                        "execution_mode": "persistent_async",
                        "scenario": scenario,
                        "worker_pid": 100,
                        "weight_version": version,
                        "checkpoint_sha256": input_sha,
                    },
                    "stage": "collector_total",
                    "success": True,
                }
            )
        for stage in (
            "cumulative_aggregation",
            "learner_batch_staging",
            "learner_forward",
            "learner_backward",
            "optimizer_step",
            "checkpoint_save",
        ):
            metric_records.append(
                {
                    "identity": {
                        "outer_iteration": index,
                        "execution_mode": "persistent_async",
                        "scenario": "**workflow**",
                        "worker_pid": 200,
                        "weight_version": version,
                        "checkpoint_sha256": input_sha,
                    },
                    "stage": stage,
                    "success": True,
                }
            )
        checkpoint_sha = mod.file_sha256(checkpoint)
        iterations.append(
            {
                "iteration": index,
                "updates": updates,
                "collection_execution_mode": "persistent_async",
                "input_weight_version": version,
                "input_checkpoint_sha256": input_sha,
                "scenario_artifacts": scenarios,
                "aggregate_dataset_path": str(aggregate),
                "aggregate_dataset_sha256": mod.file_sha256(aggregate),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
            }
        )
        input_sha = checkpoint_sha
    metric_records.append(
        {
            "identity": {
                "outer_iteration": 2,
                "execution_mode": "persistent_async",
                "scenario": "**workflow**",
                "worker_pid": 200,
                "weight_version": 4,
                "checkpoint_sha256": input_sha,
            },
            "stage": "cleanup",
            "success": True,
            "cleanup_state": "complete",
        }
    )
    metrics_path = run / "distillation_metrics.json"
    metrics_path.write_text(json.dumps({"records": metric_records}))
    manifest = {
        "completed_dagger_iterations": 2,
        "dagger_iterations": iterations,
        "performance_cleanup": {"state": "complete"},
        "distillation_metrics_path": str(metrics_path),
        "distillation_metrics_sha256": mod.file_sha256(metrics_path),
        "distillation_metrics_record_count": len(metric_records),
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest))
    outputs = {
        "run_dir": str(run),
        "log": _write(root / "formal.log", b"log"),
        "time": _write(root / "formal.time", b"time"),
        "gpu_telemetry": _write(root / "formal.csv", b"gpu"),
        "acceptance": str(root / "acceptance.json"),
    }
    dependency = {"python": "test"}
    gpu = {"returncode": 0, "stdout": "GPU-test", "stderr": ""}
    freeze = {
        "accepted": True,
        "failures": [],
        "repo": {"root": str(root), "head": "a" * 40},
        "source_identity": {
            "workflow": {
                "path": str(source),
                "sha256": mod.file_sha256(source),
                "size": source.stat().st_size,
            }
        },
        "hard_artifacts": {
            "parent_checkpoint": {
                "path": str(parent_checkpoint),
                "sha256": mod.file_sha256(parent_checkpoint),
                "size": parent_checkpoint.stat().st_size,
            },
            "parent_aggregate": {
                "path": str(parent_aggregate),
                "sha256": mod.file_sha256(parent_aggregate),
                "size": parent_aggregate.stat().st_size,
            },
        },
        "command": {
            "lineage": {
                "parent_iteration": 3,
                "source": "original_parent_iteration_3",
                "r6_sentinel_promoted": False,
            },
            "workload": {
                "dagger_iterations": 2,
                "effective_updates_by_iteration": [2, 4],
                "samples_per_role": 2,
                "execution_mode": "persistent_async",
            },
        },
        "output_paths": outputs,
        "dependency_identity": dependency,
        "gpu_query": gpu,
    }
    freeze_path = root / "freeze.json"
    freeze_path.write_text(json.dumps(freeze))
    return freeze_path, freeze, dependency, gpu


def test_v2_accepts_fresh_bootstrap_to_iteration_chain(tmp_path: Path) -> None:
    mod = _load_oracle()
    freeze_path, freeze, dependency, gpu = _fixture(tmp_path, mod)
    parent = freeze["hard_artifacts"].pop("parent_checkpoint")
    freeze["command"]["lineage"] = {
        "parent_iteration": None,
        "source": "fresh_teacher_bootstrap",
        "r6_sentinel_promoted": False,
    }
    manifest_path = Path(freeze["output_paths"]["run_dir"]) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bootstrap_checkpoint_path"] = parent["path"]
    manifest["bootstrap_checkpoint_sha256"] = parent["sha256"]
    manifest_path.write_text(json.dumps(manifest))
    freeze_path.write_text(json.dumps(freeze))

    result = mod.validate_postflight(
        freeze_path,
        observed_head="a" * 40,
        observed_dependency=dependency,
        observed_gpu=gpu,
    )

    assert result["accepted"] is True
    assert result["failures"] == []


def test_v2_accepts_complete_two_iteration_artifact_chain(tmp_path: Path) -> None:
    mod = _load_oracle()
    freeze_path, _freeze, dependency, gpu = _fixture(tmp_path, mod)

    result = mod.validate_postflight(
        freeze_path,
        observed_head="a" * 40,
        observed_dependency=dependency,
        observed_gpu=gpu,
    )

    assert result["accepted"] is True
    assert result["failures"] == []
    assert result["validated_iterations"] == 2
    assert result["final_checkpoint_sha256"]


def test_v2_rejects_lineage_cleanup_and_runtime_identity_drift(tmp_path: Path) -> None:
    mod = _load_oracle()
    freeze_path, freeze, dependency, gpu = _fixture(tmp_path, mod)
    manifest_path = Path(freeze["output_paths"]["run_dir"]) / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dagger_iterations"][1]["input_checkpoint_sha256"] = "0" * 64
    manifest["performance_cleanup"]["state"] = "pending"
    manifest_path.write_text(json.dumps(manifest))

    result = mod.validate_postflight(
        freeze_path,
        observed_head="b" * 40,
        observed_dependency={"python": "drift"},
        observed_gpu={"returncode": 0, "stdout": "GPU-other", "stderr": ""},
    )

    assert result["accepted"] is False
    assert result["warnings"] == [
        "HEAD drift after oracle repair; frozen runtime bytes are checked below"
    ]
    assert "dependency/import identity drift" in result["failures"]
    assert "GPU identity drift" in result["failures"]
    assert "iteration 2 input checkpoint lineage mismatch" in result["failures"]
    assert "manifest cleanup state incomplete" in result["failures"]


def test_v2_rejects_missing_checkpoint_and_cleanup_metric(tmp_path: Path) -> None:
    mod = _load_oracle()
    freeze_path, freeze, dependency, gpu = _fixture(tmp_path, mod)
    run = Path(freeze["output_paths"]["run_dir"])
    manifest = json.loads((run / "run_manifest.json").read_text())
    Path(manifest["dagger_iterations"][1]["checkpoint_path"]).unlink()
    metrics_path = run / "distillation_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["records"] = [r for r in metrics["records"] if r["stage"] != "cleanup"]
    metrics_path.write_text(json.dumps(metrics))

    result = mod.validate_postflight(
        freeze_path,
        observed_head="a" * 40,
        observed_dependency=dependency,
        observed_gpu=gpu,
    )

    assert result["accepted"] is False
    assert "missing output artifact: iteration 2 checkpoint_path" in result["failures"]
    assert "cleanup metric contract mismatch" in result["failures"]
    assert "metrics hash mismatch" in result["failures"]
