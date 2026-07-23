from __future__ import annotations

import copy

import pytest
from scripts.deploy.check_unilab_g1_distill_persistent_runtime import (
    _validate_summary,
)


def _valid_summary() -> dict[str, object]:
    sequence = [
        {"scenario": "walk_flat", "worker_pid": 7, "weight_version": 1},
        {"scenario": "static_stand", "worker_pid": 7, "weight_version": 1},
        {
            "scenario": "walk_to_stop",
            "worker_pid": 7,
            "weight_version": 1,
            "command_intents": ["active", "active", "inactive", "inactive"],
            "role_labels": ["walk_flat", "walk_flat", "stand", "stand"],
            "transition_ages": [-1, -1, 0, 1],
        },
        {"scenario": "walk_flat", "worker_pid": 7, "weight_version": 1},
    ]
    return {
        "sequence": sequence,
        "close_report": {
            "worker_pid": 7,
            "student_init_count": 1,
            "resource_counters": {
                "teacher_init_count": 2,
                "env_init_count": 2,
                "request_count": 4,
                "reset_count": 4,
                "request_error_count": 0,
                "teacher_close_count": 2,
                "env_close_count": 2,
            },
        },
    }


def test_persistent_runtime_probe_accepts_exact_lifecycle_summary() -> None:
    _validate_summary(_valid_summary())


def test_persistent_runtime_probe_rejects_cleanup_mismatch() -> None:
    summary = copy.deepcopy(_valid_summary())
    summary["close_report"]["resource_counters"]["env_close_count"] = 1

    with pytest.raises(RuntimeError, match="env_close_count"):
        _validate_summary(summary)


def test_persistent_runtime_probe_accepts_restart_each_request_summary() -> None:
    summary = _valid_summary()
    summary["worker_lifecycle"] = "restart_each_request"
    summary["sequence"] = [
        {**row, "worker_pid": index + 10} for index, row in enumerate(summary["sequence"])
    ]
    summary.pop("close_report")
    summary["close_reports"] = [
        {
            "worker_pid": index + 10,
            "student_init_count": 1,
            "resource_counters": {
                "teacher_init_count": 1,
                "env_init_count": 1,
                "request_count": 1,
                "reset_count": 1,
                "request_error_count": 0,
                "teacher_close_count": 1,
                "env_close_count": 1,
            },
        }
        for index in range(4)
    ]

    _validate_summary(summary)
