"""Owner-configured AMP APPO runtime bundle."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from unilab.algos.torch.amp.runner import AMPAPPORunner
from unilab.algos.torch.appo.runtime import APPORuntime


def resolve_amp_appo_runtime(
    rl_cfg: dict[str, Any], *, default_play_fn: Callable[..., str | None]
) -> APPORuntime:
    if rl_cfg.get("runtime_impl") != "amp_appo":
        raise ValueError("AMP runtime resolver requires runtime_impl='amp_appo'")
    return APPORuntime(runner_cls=AMPAPPORunner, play_fn=default_play_fn)
