from __future__ import annotations

import queue

import numpy as np
import pytest

import unilab.algos.torch.appo.worker as worker_module
from unilab.algos.torch.appo.worker import (
    _close_appo_collector_resources,
    appo_collector_fn,
    compute_timeout_bootstrap_correction,
    put_latest_metrics,
)


class _FakeCritic:
    def __call__(self, obs):
        policy = obs["policy"]
        return policy.sum(dim=1, keepdim=True)


def test_compute_timeout_bootstrap_correction_uses_final_observation_value():
    correction = compute_timeout_bootstrap_correction(
        critic=_FakeCritic(),
        collector_device="cpu",
        gamma=0.5,
        timeout_mask=np.array([True, False]),
        final_obs=np.array([[2.0, 3.0], [9.0, 9.0]], dtype=np.float32),
        final_critic=np.array([[2.0, 3.0], [9.0, 9.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(correction, np.array([2.5, 0.0], dtype=np.float32))


def test_compute_timeout_bootstrap_correction_prefers_explicit_final_critic():
    correction = compute_timeout_bootstrap_correction(
        critic=_FakeCritic(),
        collector_device="cpu",
        gamma=0.5,
        timeout_mask=np.array([True, False]),
        final_obs=np.array([[2.0, 3.0], [9.0, 9.0]], dtype=np.float32),
        final_critic=np.array([[11.0, 13.0], [0.0, 0.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(correction, np.array([12.0, 0.0], dtype=np.float32))


def test_put_latest_metrics_replaces_stale_item_when_queue_is_full(capsys):
    metrics_queue = queue.Queue(maxsize=1)
    metrics_queue.put_nowait({"total_steps": 1})

    put_latest_metrics(metrics_queue, {"total_steps": 2}, worker_name="APPOWorker")

    assert metrics_queue.get_nowait() == {"total_steps": 2}
    captured = capsys.readouterr()
    assert captured.err == ""


def test_close_appo_collector_resources_attempts_every_owner_in_reverse_order():
    events: list[str] = []

    class _Resource:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            events.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} close failed")

    resources = [
        _Resource("ring"),
        _Resource("actor_sync", fail=True),
        _Resource("critic_sync"),
        _Resource("env"),
    ]

    with pytest.raises(RuntimeError, match="actor_sync close failed"):
        _close_appo_collector_resources(resources)

    assert events == ["env", "critic_sync", "actor_sync", "ring"]


def test_appo_collector_wrapper_closes_registered_resources_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class _StopEvent:
        def __init__(self) -> None:
            self.was_set = False

        def set(self) -> None:
            self.was_set = True

    class _Resource:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            events.append(self.name)

    def fail_body(*, resources, **_kwargs) -> None:
        resources.extend([_Resource("ring"), _Resource("env")])
        raise RuntimeError("synthetic collector failure")

    monkeypatch.setattr(worker_module, "_run_appo_collector", fail_body)
    stop_event = _StopEvent()

    with pytest.raises(RuntimeError, match="synthetic collector failure"):
        appo_collector_fn(
            stop_event=stop_event,
            env_name="DummyEnv",
            rl_cfg={},
            num_envs=1,
            steps_per_env=1,
            shm_rollout_ring_buffer_name={},
            sync_primitives=(),
            obs_dim=1,
            action_dim=1,
            critic_dim=1,
            actor_weight_sync_name="actor",
            actor_weight_param_shapes={},
            critic_weight_sync_name="critic",
            critic_weight_param_shapes={},
            metrics_queue=None,
        )

    assert stop_event.was_set is True
    assert events == ["env", "ring"]
