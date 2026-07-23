"""Base async runner for multi-process RL training."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable

from unilab.ipc.collector_error import (
    ExceptionWrapper,
    collector_error_guard,
    create_error_pipe,
    format_collector_death,
)

_SPAWN_CTX = mp.get_context("spawn")


def _collector_entry_wrapper(
    target_fn: Callable,
    error_conn: Any,
    kwargs: dict,
) -> None:
    """Top-level wrapper for collector subprocess entry point.

    Ensures ordinary Python exceptions (including import errors and env creation
    failures) are captured and sent to the parent via the error pipe.

    Status: active. 诊断开关启用时, collector 异常先写入错误通道, 随后在
    collector 本进程 abort, 以便 core dump 保留真实 collector native 状态.
    """
    label = kwargs.pop("_error_label", "collector")
    try:
        with collector_error_guard(
            error_conn=error_conn,
            metrics_queue=kwargs.get("metrics_queue"),
            stop_event=kwargs.get("stop_event"),
            label=label,
        ):
            target_fn(**kwargs)
    except BaseException:
        if os.environ.get("UNILAB_NATIVE_ABORT_ON_CORRUPTION", "0") == "1":
            # 对普通 Exception, guard 已先发送 traceback. 此处只制造 collector-owned core.
            sys.stdout.flush()
            sys.stderr.flush()
            os.abort()
        raise


class AsyncRunner(ABC):
    """Base class for async RL algorithms.

    Manages:
    - Shared memory allocation/cleanup
    - Collector process lifecycle
    - Error propagation from collector subprocess
    - Training loop skeleton
    """

    def __init__(
        self,
        env_name: str,
        env_cfg_overrides: dict,
        rl_cfg: dict,
        *,
        device: str | None = None,
        collector_device: str | None = None,
        sim_backend: str = "mujoco",
        num_envs: int = 4096,
    ):
        self.env_name = env_name
        self.env_cfg_overrides = env_cfg_overrides
        self.rl_cfg = rl_cfg
        self.device = device or self._get_default_device()
        self.collector_device = collector_device or self.device
        self.sim_backend = sim_backend
        self.num_envs = num_envs

        self._collector_process: Any = None
        self._stop_event = _SPAWN_CTX.Event()
        self._shared_resources: list = []
        self._error_recv: Any = None
        self._error_send: Any = None
        self._closed = False
        self.last_close_report: dict[str, Any] = {
            "state": "pending",
            "collector_exitcode": None,
            "collector_terminated": False,
            "resource_count": 0,
            "errors": [],
        }

    @abstractmethod
    def _get_default_device(self) -> str:
        """Get default device (backend-specific)."""
        ...

    @abstractmethod
    def _build_learner(self) -> Any: ...

    @abstractmethod
    def _collector_fn(self, stop_event: Any, **kwargs) -> None: ...

    @abstractmethod
    def learn(
        self, max_iterations: int, save_interval: int = 50, log_dir: str = "logs"
    ) -> None: ...

    def _start_collector(self, target_fn: Callable, kwargs: dict) -> None:
        self._error_recv, self._error_send = create_error_pipe()

        self._collector_process = _SPAWN_CTX.Process(
            target=_collector_entry_wrapper,
            args=(target_fn, self._error_send, kwargs),
            daemon=True,
        )
        self._collector_process.start()
        self._error_send.close()
        self._error_send = None

    def _check_collector_alive(self) -> bool:
        """Check if collector is alive. Prints full diagnostic if dead."""
        if self._collector_process is None:
            return True
        if self._collector_process.is_alive():
            return True

        death_info = self._read_collector_error()
        print(f"\n{death_info}\n", file=sys.stderr, flush=True)
        return False

    def _read_collector_error(self) -> str:
        """Read error info from dead collector — pipe first, then exit code."""
        traceback_text = None
        if self._error_recv is not None:
            try:
                if self._error_recv.poll(timeout=0.1):
                    obj = self._error_recv.recv()
                    if isinstance(obj, ExceptionWrapper):
                        traceback_text = obj.exc_msg
            except (EOFError, OSError):
                pass

        exitcode = getattr(self._collector_process, "exitcode", None)
        return format_collector_death(exitcode, traceback_text)

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.last_close_report
        self._closed = True
        errors: list[str] = []
        collector_terminated = False
        self._stop_event.set()
        if self._collector_process is not None and self._collector_process.is_alive():
            self._collector_process.join(timeout=10)
            if self._collector_process.is_alive():
                self._collector_process.terminate()
                self._collector_process.join(timeout=5)
                collector_terminated = True

        if self._collector_process is not None:
            exitcode = getattr(self._collector_process, "exitcode", None)
            # -15 (SIGTERM) is expected during normal close()
            if exitcode is not None and exitcode != 0 and exitcode != -15:
                death_info = self._read_collector_error()
                print(
                    f"\n[AsyncRunner] Collector exited with code {exitcode}:\n{death_info}\n",
                    file=sys.stderr,
                    flush=True,
                )

        resources = list(self._shared_resources)
        for resource in resources:
            try:
                if hasattr(resource, "cleanup"):
                    resource.cleanup()
                elif hasattr(resource, "close"):
                    resource.close()
                    join_thread = getattr(resource, "join_thread", None)
                    if callable(join_thread):
                        join_thread()
            except Exception as exc:
                errors.append(f"{type(resource).__name__}: {type(exc).__name__}: {exc}")
        self._shared_resources.clear()

        if self._error_recv is not None:
            try:
                self._error_recv.close()
            except Exception as exc:
                errors.append(f"error_recv: {type(exc).__name__}: {exc}")
            self._error_recv = None
        if self._error_send is not None:
            try:
                self._error_send.close()
            except Exception as exc:
                errors.append(f"error_send: {type(exc).__name__}: {exc}")
            self._error_send = None

        self.last_close_report = {
            "state": "failed" if errors else "complete",
            "collector_exitcode": (
                getattr(self._collector_process, "exitcode", None)
                if self._collector_process is not None
                else None
            ),
            "collector_terminated": collector_terminated,
            "resource_count": len(resources),
            "errors": errors,
        }
        if errors:
            raise RuntimeError("AsyncRunner cleanup failed: " + "; ".join(errors))
        return self.last_close_report

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
