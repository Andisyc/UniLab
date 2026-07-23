"""Policy AMP transition identity helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.base.final_observation import resolve_terminal_observation_contract


def resolve_amp_transition_next(
    actor_next_amp: np.ndarray,
    *,
    done: np.ndarray | None = None,
    final_observation: dict[str, Any] | None = None,
    info: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Patch reset-after-step AMP rows with their true terminal observation."""
    actor_next_amp = np.asarray(actor_next_amp)
    if actor_next_amp.ndim != 2:
        raise ValueError(f"actor_next_amp must be rank 2, got {actor_next_amp.shape}")
    contract = resolve_terminal_observation_contract(
        next_obs_batch_size=actor_next_amp.shape[0],
        final_observation=final_observation,
        done=done,
        info=info,
    )
    if not np.any(contract.terminal_mask):
        return actor_next_amp, contract.terminal_mask

    resolved_final = final_observation
    if resolved_final is None and isinstance(info, dict):
        candidate = info.get("final_observation")
        if isinstance(candidate, dict):
            resolved_final = candidate
    terminal_amp = None if resolved_final is None else resolved_final.get("amp")
    if terminal_amp is None:
        raise ValueError("terminal AMP transition requires final_observation['amp']")
    terminal_amp = np.asarray(terminal_amp, dtype=actor_next_amp.dtype)
    if terminal_amp.shape != actor_next_amp.shape:
        raise ValueError(
            f"terminal AMP shape must be {actor_next_amp.shape}, got {terminal_amp.shape}"
        )
    transition_next = actor_next_amp.copy()
    transition_next[contract.terminal_mask] = terminal_amp[contract.terminal_mask]
    return transition_next, contract.terminal_mask
