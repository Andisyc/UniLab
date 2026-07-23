"""Collector-side AMP payload writer with no training/scoring state."""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.algos.torch.amp.spec import AMP_OBSERVATION_DIM
from unilab.algos.torch.amp.transition import resolve_amp_transition_next


def write_amp_rollout_payload(
    *,
    write_buffer: dict[str, np.ndarray],
    step: int,
    current_observation: dict[str, Any],
    state: Any,
) -> None:
    current_amp = np.asarray(current_observation["amp"], dtype=np.float32)
    actor_next_amp = np.asarray(state.obs["amp"], dtype=np.float32)
    expected = (actor_next_amp.shape[0], AMP_OBSERVATION_DIM)
    if current_amp.shape != expected or actor_next_amp.shape != expected:
        raise ValueError(
            f"AMP collector observations must have shape {expected}; "
            f"got {current_amp.shape} and {actor_next_amp.shape}"
        )
    done = np.asarray(state.terminated | state.truncated, dtype=bool)
    transition_next, _ = resolve_amp_transition_next(
        actor_next_amp,
        done=done,
        final_observation=state.final_observation,
        info=state.info,
    )
    write_buffer["amp_state"][:, step, :] = current_amp
    write_buffer["amp_next_state"][:, step, :] = transition_next
