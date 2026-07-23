"""Persistent collector runtime protocol for DAgger workflows.

DAgger owns request and result semantics. Process lifecycle and collector
failure propagation remain owned by :class:`unilab.ipc.async_runner.AsyncRunner`.

状态: active OFF-default persistent collector runtime, legacy 默认路径不变.
上游: ``run_multirole_dagger_workflow(execution_mode="persistent_async")``.
下游: persistent worker request/result 与 parent-side checkpoint activator.
证据: S1/S2 contract-confirmed; E61/E65/E67 S4 bounded G1 lifecycle, timing,
and legacy/persistent A/B runtime-confirmed.
缺口: end-to-end stable speedup 未证明(``NO_STABLE_SPEEDUP``); HP-6 production
gate 与 physical policy acceptance 仍未完成.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, cast

from unilab.ipc.async_runner import AsyncRunner

_SPAWN_CTX = mp.get_context("spawn")


@dataclass(frozen=True)
class DaggerCollectRequest:
    """One scenario collection request at a fixed outer-iteration barrier."""

    request_id: str
    scenario: str
    iteration: int
    checkpoint_path: str
    output_path: str
    expected_weight_version: int


@dataclass(frozen=True)
class DaggerCollectResult:
    """Identity and structured metrics returned by a persistent collector."""

    request_id: str
    scenario: str
    iteration: int
    checkpoint_path: str
    output_path: str
    expected_weight_version: int
    observed_weight_version: int
    num_samples: int
    worker_pid: int
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class DaggerCollectorService(Protocol):
    """Worker-local service initialized once and reused across requests."""

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult: ...

    def close(self) -> None: ...


def validate_dagger_collect_result(
    request: DaggerCollectRequest,
    result: DaggerCollectResult,
) -> None:
    """Fail closed when a worker result crosses its request barrier."""

    identity_fields = (
        "request_id",
        "scenario",
        "iteration",
        "checkpoint_path",
        "output_path",
        "expected_weight_version",
    )
    mismatches = [
        name for name in identity_fields if getattr(request, name) != getattr(result, name)
    ]
    if mismatches:
        raise ValueError(f"persistent DAgger collector result identity mismatch: {mismatches}")
    if result.observed_weight_version != request.expected_weight_version:
        raise ValueError(
            "persistent DAgger collector weight version mismatch: "
            f"expected {request.expected_weight_version}, "
            f"observed {result.observed_weight_version}"
        )
    if result.num_samples <= 0:
        raise ValueError(
            "persistent DAgger collector must return a positive sample count, "
            f"got {result.num_samples}"
        )


def _persistent_dagger_collector_entry(
    *,
    stop_event: Any,
    request_queue: Any,
    result_queue: Any,
    worker_factory: Callable[..., DaggerCollectorService],
    worker_kwargs: Mapping[str, Any],
) -> None:
    """Spawn-safe service loop; exceptions escape to AsyncRunner's guard."""

    service = worker_factory(**dict(worker_kwargs))
    try:
        while not stop_event.is_set():
            try:
                request = request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                break
            result_queue.put(service.collect(request))
    finally:
        service.close()


class PersistentDaggerCollectorRunner(AsyncRunner):
    """Synchronous request facade backed by one persistent spawned worker.

    Collection remains sequential here to preserve the DAgger outer barrier.
    The performance gain targeted by this layer is persistent runtime state,
    not unsound overlap between collection and student publication.
    """

    def __init__(
        self,
        *,
        worker_factory: Callable[..., DaggerCollectorService],
        worker_kwargs: Mapping[str, Any] | None = None,
        checkpoint_activator: Callable[[Path], int] | None = None,
        request_timeout_seconds: float = 300.0,
    ) -> None:
        super().__init__(
            env_name="DAggerPersistentCollector",
            env_cfg_overrides={},
            rl_cfg={},
            device="cpu",
            collector_device="cpu",
            num_envs=1,
        )
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._worker_factory = worker_factory
        self._worker_kwargs = dict(worker_kwargs or {})
        self._checkpoint_activator = checkpoint_activator
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._request_queue = _SPAWN_CTX.Queue(maxsize=1)
        self._result_queue = _SPAWN_CTX.Queue(maxsize=1)
        self._shared_resources.extend([self._request_queue, self._result_queue])
        self._closed = False
        self._active_checkpoint_path: str | None = None
        self._active_weight_version: int | None = None

    def _get_default_device(self) -> str:
        return "cpu"

    def _build_learner(self) -> None:
        return None

    def _collector_fn(self, stop_event: Any, **kwargs: Any) -> None:
        _persistent_dagger_collector_entry(stop_event=stop_event, **kwargs)

    def learn(
        self,
        max_iterations: int,
        save_interval: int = 50,
        log_dir: str = "logs",
    ) -> None:
        raise NotImplementedError("DAgger workflow owns the outer learning loop")

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("cannot restart a closed persistent DAgger collector")
        if self._collector_process is not None:
            if self._collector_process.is_alive():
                return
            raise RuntimeError(self._read_collector_error())
        self._start_collector(
            target_fn=_persistent_dagger_collector_entry,
            kwargs={
                "stop_event": self._stop_event,
                "request_queue": self._request_queue,
                "result_queue": self._result_queue,
                "worker_factory": self._worker_factory,
                "worker_kwargs": self._worker_kwargs,
                "_error_label": "persistent DAgger collector",
            },
        )

    def activate_checkpoint(self, checkpoint_path: Path) -> int:
        """Publish one checkpoint through the configured parent-side owner."""

        if self._checkpoint_activator is None:
            raise RuntimeError("persistent DAgger collector has no checkpoint_activator")
        resolved_path = str(Path(checkpoint_path).resolve())
        version = int(self._checkpoint_activator(Path(resolved_path)))
        if version < 0:
            raise ValueError(f"checkpoint_activator returned negative version {version}")
        self._active_checkpoint_path = resolved_path
        self._active_weight_version = version
        return version

    def collect(self, request: DaggerCollectRequest) -> DaggerCollectResult:
        if self._checkpoint_activator is not None:
            if self._active_checkpoint_path is None or self._active_weight_version is None:
                raise RuntimeError("activate_checkpoint must run before collect")
            if request.checkpoint_path != self._active_checkpoint_path:
                raise ValueError(
                    "persistent DAgger request checkpoint does not match active checkpoint: "
                    f"request={request.checkpoint_path!r}, "
                    f"active={self._active_checkpoint_path!r}"
                )
            if request.expected_weight_version != self._active_weight_version:
                raise ValueError(
                    "persistent DAgger request version does not match active version: "
                    f"request={request.expected_weight_version}, "
                    f"active={self._active_weight_version}"
                )
        self.start()
        self._request_queue.put(request)
        deadline = time.monotonic() + self._request_timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"persistent DAgger collector timed out for request {request.request_id!r}"
                )
            try:
                result = cast(
                    DaggerCollectResult,
                    self._result_queue.get(timeout=min(0.1, remaining)),
                )
            except queue.Empty:
                if self._collector_process is not None and not self._collector_process.is_alive():
                    raise RuntimeError(self._read_collector_error())
                continue
            validate_dagger_collect_result(request, result)
            return result

    @staticmethod
    def validate_result(
        request: DaggerCollectRequest,
        result: DaggerCollectResult,
    ) -> None:
        validate_dagger_collect_result(request, result)

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.last_close_report
        try:
            if self._collector_process is not None and self._collector_process.is_alive():
                try:
                    self._request_queue.put_nowait(None)
                except queue.Full:
                    pass
        finally:
            close_report = super().close()
        return close_report
