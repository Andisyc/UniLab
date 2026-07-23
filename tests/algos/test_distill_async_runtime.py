from __future__ import annotations

import os
from pathlib import Path

import pytest

from unilab.algos.torch.distill.async_runtime import (
    DaggerCollectRequest,
    DaggerCollectResult,
    PersistentDaggerCollectorRunner,
)


class _FakeCollectorService:
    def __init__(self, *, rows_per_request: int = 7):
        self.rows_per_request = rows_per_request

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        if request.scenario == "explode":
            raise RuntimeError("fake collector exploded")
        return DaggerCollectResult(
            request_id=request.request_id,
            scenario=request.scenario,
            iteration=request.iteration,
            checkpoint_path=request.checkpoint_path,
            output_path=request.output_path,
            expected_weight_version=request.expected_weight_version,
            observed_weight_version=request.expected_weight_version,
            num_samples=self.rows_per_request,
            worker_pid=os.getpid(),
            metrics={"collect_seconds": 0.01},
        )

    def close(self) -> None:
        pass


def _build_fake_collector_service(*, rows_per_request: int = 7):
    return _FakeCollectorService(rows_per_request=rows_per_request)


def _request(tmp_path: Path, *, request_id: str, scenario: str = "walk_flat"):
    return DaggerCollectRequest(
        request_id=request_id,
        scenario=scenario,
        iteration=3,
        checkpoint_path=str(tmp_path / "student_2.pt"),
        output_path=str(tmp_path / f"{scenario}.pt"),
        expected_weight_version=9,
    )


def test_persistent_runner_reuses_one_spawned_worker_for_sequential_requests(tmp_path):
    runner = PersistentDaggerCollectorRunner(
        worker_factory=_build_fake_collector_service,
        worker_kwargs={"rows_per_request": 11},
    )
    try:
        first = runner.collect(_request(tmp_path, request_id="req-1"))
        second = runner.collect(_request(tmp_path, request_id="req-2", scenario="stand"))
    finally:
        runner.close()

    assert first.num_samples == 11
    assert second.num_samples == 11
    assert first.worker_pid == second.worker_pid
    assert first.worker_pid != os.getpid()
    assert first.observed_weight_version == 9
    assert second.request_id == "req-2"


def test_persistent_runner_propagates_worker_exception_with_async_runner_diagnostics(tmp_path):
    runner = PersistentDaggerCollectorRunner(worker_factory=_build_fake_collector_service)
    try:
        with pytest.raises(RuntimeError, match="fake collector exploded"):
            runner.collect(_request(tmp_path, request_id="req-boom", scenario="explode"))
    finally:
        runner.close()


def test_persistent_runner_fails_closed_on_result_weight_version_mismatch(tmp_path):
    class _WrongVersionService(_FakeCollectorService):
        def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
            result = super().collect(request)
            return DaggerCollectResult(
                **{
                    **result.__dict__,
                    "observed_weight_version": request.expected_weight_version + 1,
                }
            )

    # Local classes cannot be pickled under spawn, so exercise the parent-side
    # result contract directly without starting a subprocess.
    request = _request(tmp_path, request_id="req-version")
    result = _WrongVersionService().collect(request)

    with pytest.raises(ValueError, match="weight version mismatch"):
        PersistentDaggerCollectorRunner.validate_result(request, result)


def test_persistent_runner_close_is_idempotent_and_reaps_worker(tmp_path):
    runner = PersistentDaggerCollectorRunner(worker_factory=_build_fake_collector_service)
    runner.start()
    process = runner._collector_process

    first_report = runner.close()
    second_report = runner.close()

    assert process is not None
    assert not process.is_alive()
    assert first_report == second_report
    assert first_report["state"] == "complete"
    assert first_report["resource_count"] == 2


def test_persistent_runner_rejects_request_outside_activated_checkpoint_barrier(tmp_path):
    runner = PersistentDaggerCollectorRunner(
        worker_factory=_build_fake_collector_service,
        checkpoint_activator=lambda _path: 4,
    )
    try:
        checkpoint = tmp_path / "student_2.pt"
        assert runner.activate_checkpoint(checkpoint) == 4
        request = DaggerCollectRequest(
            request_id="req-stale-version",
            scenario="walk_flat",
            iteration=3,
            checkpoint_path=str(checkpoint.resolve()),
            output_path=str((tmp_path / "walk.pt").resolve()),
            expected_weight_version=5,
        )

        with pytest.raises(ValueError, match="request version.*active version"):
            runner.collect(request)
    finally:
        runner.close()
