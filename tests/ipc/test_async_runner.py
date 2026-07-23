"""Tests for AsyncRunner base class."""

from __future__ import annotations

import multiprocessing as mp
import signal
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from unilab.ipc import async_runner as async_runner_module
from unilab.ipc.async_runner import AsyncRunner
from unilab.ipc.collector_error import (
    ExceptionWrapper,
    create_error_pipe,
    format_collector_death,
)

_SPAWN_CTX = mp.get_context("spawn")


# ---------------------------------------------------------------------------
# Minimal concrete implementation for testing
# ---------------------------------------------------------------------------


class _StubRunner(AsyncRunner):
    """Minimal concrete AsyncRunner — used only for unit tests."""

    def _get_default_device(self) -> str:
        return "cpu"

    def _build_learner(self) -> Any:
        return None

    def _collector_fn(self, stop_event: Any, **kwargs) -> None:
        pass

    def learn(self, max_iterations: int, save_interval: int = 50, log_dir: str = "logs") -> None:
        pass


def _make_runner(rl_cfg=None, **kwargs) -> _StubRunner:
    return _StubRunner(
        env_name="DummyEnv",
        env_cfg_overrides={},
        rl_cfg=rl_cfg or {},
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_init_stores_env_name():
    r = _make_runner()
    assert r.env_name == "DummyEnv"


def test_init_stores_rl_cfg():
    cfg = {"gamma": 0.99}
    r = _make_runner(rl_cfg=cfg)
    assert r.rl_cfg == cfg


def test_init_device_explicit():
    r = _make_runner(device="cpu")
    assert r.device == "cpu"


def test_init_device_default():
    """When device=None, uses _get_default_device() which returns 'cpu' for stub."""
    r = _make_runner()
    assert r.device == "cpu"


def test_init_collector_device_defaults_to_device():
    r = _make_runner(device="cpu")
    assert r.collector_device == "cpu"


def test_init_collector_device_explicit():
    r = _make_runner(device="cpu", collector_device="cpu")
    assert r.collector_device == "cpu"


def test_init_sim_backend_explicit():
    r = _make_runner(sim_backend="motrix")
    assert r.sim_backend == "motrix"


def test_init_num_envs():
    r = _make_runner(num_envs=64)
    assert r.num_envs == 64


def test_init_shared_resources_empty():
    r = _make_runner()
    assert r._shared_resources == []


def test_init_collector_process_none():
    r = _make_runner()
    assert r._collector_process is None


# ---------------------------------------------------------------------------
# close() — no collector
# ---------------------------------------------------------------------------


def test_close_with_no_collector_does_not_raise():
    r = _make_runner()
    r.close()  # _collector_process is None → should be a no-op


def test_close_is_idempotent():
    r = _make_runner()
    r.close()
    r.close()


# ---------------------------------------------------------------------------
# close() — resource cleanup
# ---------------------------------------------------------------------------


def test_close_calls_cleanup_on_resources():
    """Resources with cleanup() method must be cleaned up on close()."""
    r = _make_runner()
    mock_res = MagicMock(spec=["cleanup"])
    r._shared_resources.append(mock_res)
    r.close()
    mock_res.cleanup.assert_called_once()


def test_close_calls_close_on_resources_without_cleanup():
    """Resources without cleanup() but with close() must have close() called."""
    r = _make_runner()
    mock_res = MagicMock(spec=["close"])
    r._shared_resources.append(mock_res)
    r.close()
    mock_res.close.assert_called_once()


def test_close_handles_multiple_resources():
    r = _make_runner()
    resources = [MagicMock(spec=["cleanup"]) for _ in range(3)]
    r._shared_resources.extend(resources)
    r.close()
    for res in resources:
        res.cleanup.assert_called_once()


def test_close_closes_and_joins_queue_resources():
    r = _make_runner()
    queue_resource = MagicMock(spec=["close", "join_thread"])
    r._shared_resources.append(queue_resource)

    report = r.close()

    queue_resource.close.assert_called_once()
    queue_resource.join_thread.assert_called_once()
    assert report["state"] == "complete"
    assert report["resource_count"] == 1


def test_close_reports_resource_failure_after_attempting_all_cleanup():
    r = _make_runner()
    failing = MagicMock(spec=["cleanup"])
    failing.cleanup.side_effect = RuntimeError("synthetic cleanup failure")
    following = MagicMock(spec=["cleanup"])
    r._shared_resources.extend([failing, following])

    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        r.close()

    following.cleanup.assert_called_once()
    assert r.last_close_report["state"] == "failed"
    assert r.last_close_report["resource_count"] == 2
    assert len(r.last_close_report["errors"]) == 1


# ---------------------------------------------------------------------------
# close() — with live collector process
# ---------------------------------------------------------------------------


def _worker_wait_for_stop(stop_event) -> None:
    """Cooperative worker: exits as soon as stop_event is set."""
    stop_event.wait(timeout=30)


def test_close_joins_running_collector():
    """close() must signal the stop event and reap the collector process.

    Under heavy CI load a spawned process may still be importing the test module
    when close() hits its timeout path, so SIGTERM is also an acceptable outcome
    as long as the collector does not leak.
    """
    r = _make_runner()
    r._collector_process = _SPAWN_CTX.Process(
        target=_worker_wait_for_stop,
        args=(r._stop_event,),
        daemon=True,
    )
    r._collector_process.start()
    assert r._collector_process.is_alive()

    r.close()

    assert r._stop_event.is_set()
    assert not r._collector_process.is_alive()
    assert r._collector_process.exitcode in (0, -signal.SIGTERM)


# ---------------------------------------------------------------------------
# _start_collector
# ---------------------------------------------------------------------------


def _noop_collector(stop_event) -> None:
    stop_event.wait(timeout=30)


def _collector_report_kwargs(
    stop_event,
    report_queue,
    token: str,
    sim_backend: str = "missing",
) -> None:
    report_queue.put({"sim_backend": sim_backend, "token": token})
    stop_event.wait(timeout=30)


def _collector_raise_runtime_error() -> None:
    raise RuntimeError("collector sentinel failure")


def test_start_collector_spawns_process():
    """_start_collector() must create and start a subprocess."""
    r = _make_runner()
    r._start_collector(target_fn=_noop_collector, kwargs={"stop_event": r._stop_event})
    assert r._collector_process is not None
    assert r._collector_process.is_alive()
    r.close()


def test_start_collector_does_not_merge_runner_runtime_fields():
    r = _make_runner(sim_backend="motrix")
    report_queue = _SPAWN_CTX.Queue()
    r._start_collector(
        target_fn=_collector_report_kwargs,
        kwargs={
            "stop_event": r._stop_event,
            "report_queue": report_queue,
            "token": "ok",
        },
    )
    payload = report_queue.get(timeout=5)
    assert payload == {"sim_backend": "missing", "token": "ok"}
    r.close()


def test_collector_entry_native_fail_stop_reports_error_before_abort(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    error_conn = MagicMock()
    error_conn.send.side_effect = lambda _wrapper: events.append("pipe_send")
    abort = MagicMock(side_effect=lambda: events.append("abort"))
    monkeypatch.setattr(async_runner_module.os, "abort", abort)
    monkeypatch.setenv("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "1")

    with pytest.raises(RuntimeError, match="collector sentinel failure"):
        async_runner_module._collector_entry_wrapper(
            _collector_raise_runtime_error,
            error_conn,
            {},
        )

    assert events == ["pipe_send", "abort"]


def test_collector_entry_native_fail_stop_disabled_preserves_original_error(
    monkeypatch: pytest.MonkeyPatch,
):
    error_conn = MagicMock()
    abort = MagicMock()
    monkeypatch.setattr(async_runner_module.os, "abort", abort)
    monkeypatch.setenv("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "0")

    with pytest.raises(RuntimeError, match="collector sentinel failure"):
        async_runner_module._collector_entry_wrapper(
            _collector_raise_runtime_error,
            error_conn,
            {},
        )

    error_conn.send.assert_called_once()
    abort.assert_not_called()


def test_collector_entry_native_fail_stop_aborts_real_spawned_collector(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "1")
    error_recv, error_send = create_error_pipe()
    process = _SPAWN_CTX.Process(
        target=async_runner_module._collector_entry_wrapper,
        args=(_collector_raise_runtime_error, error_send, {}),
    )

    process.start()
    error_send.close()
    assert error_recv.poll(timeout=10)
    wrapper = error_recv.recv()
    process.join(timeout=10)

    assert isinstance(wrapper, ExceptionWrapper)
    assert "collector sentinel failure" in wrapper.exc_msg
    assert process.exitcode == -signal.SIGABRT
    error_recv.close()


def test_format_collector_death_reports_shell_style_sigbus():
    report = format_collector_death(135)

    assert "shell-style signal 7" in report
    assert "SIGBUS" in report
    assert "shared memory" in report


def test_format_collector_death_reports_negative_sigbus():
    report = format_collector_death(-7)

    assert "signal 7" in report
    assert "SIGBUS" in report


def test_format_collector_death_reports_sigkill_oom_hint():
    report = format_collector_death(137)

    assert "SIGKILL" in report
    assert "OOM killer" in report


# ---------------------------------------------------------------------------
# __del__ exception handling
# ---------------------------------------------------------------------------


def test_del_does_not_raise_even_if_close_fails():
    """__del__ must swallow exceptions from close()."""
    r = _make_runner()
    # Force close() to fail by corrupting internal state
    r._shared_resources = None  # type: ignore[assignment]  # will raise in close()
    # __del__ calls close() and must not propagate the exception
    r.__del__()  # noqa: PLC2801  (explicit __del__ call for test)
