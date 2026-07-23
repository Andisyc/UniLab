"""Probe the production offline balanced-staging path without learner updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from unilab.algos.torch.distill import offline
from unilab.algos.torch.distill.data import DistillationBatch, load_distillation_dataset


class _NoOpTrainer:
    """Consume production batches while intentionally performing no learning."""

    def __init__(self) -> None:
        self.update_count = 0

    def update(self, batch: DistillationBatch, *, performance: object = None) -> SimpleNamespace:
        del performance
        self.update_count += 1
        batch_size = int(batch.student_obs.shape[0])
        action_shape = (
            (batch_size, 0) if batch.teacher_actions is None else tuple(batch.teacher_actions.shape)
        )
        return SimpleNamespace(
            update_count=self.update_count,
            loss=0.0,
            student_grad_norm=0.0,
            student_action_shape=action_shape,
            teacher_action_shape=action_shape,
            teacher_action_requires_grad=False,
            teacher_action_source="cached" if batch.teacher_actions is not None else "none",
            behavior_loss=0.0,
            behavior_action_shape=action_shape,
            behavior_action_source="no_op_probe",
            behavior_target_count=batch_size,
            aux_loss=0.0,
            role_loss=0.0,
            role_target_count=0 if batch.role_labels is None else len(batch.role_labels),
            command_intent_loss=0.0,
            command_intent_target_count=(
                0 if batch.command_intents is None else len(batch.command_intents)
            ),
            expert_usage=None,
            route_entropy=None,
        )


def _parse_quotas(values: list[str]) -> dict[str, float] | None:
    if not values:
        return None
    quotas: dict[str, float] = {}
    for value in values:
        label, separator, weight = value.partition("=")
        if not separator or not label:
            raise ValueError(f"quota must use LABEL=WEIGHT, got {value!r}")
        quotas[label] = float(weight)
    return quotas


def _digest_update(digest: Any, indices: torch.Tensor) -> None:
    digest.update(indices.detach().cpu().contiguous().numpy().tobytes())


def run_production_path_probe(
    *,
    dataset_path: Path,
    device: torch.device,
    batch_size: int,
    updates: int,
    seed: int,
    balance_key: str,
    balanced_labels: tuple[str, ...],
    balance_quotas: dict[str, float] | None,
    shuffle: bool,
) -> dict[str, Any]:
    dataset = load_distillation_dataset(dataset_path, device=device)
    labels = offline._labels_for_balance_key(dataset, balance_key)
    if labels is None:
        raise ValueError(f"production probe requires labels for balance_key={balance_key!r}")

    production_digest = hashlib.sha256()
    production_build_count = 0
    production_final_rng_state: torch.Tensor | None = None
    original_builder = offline._build_balanced_label_pools
    original_sampler = offline._sample_balanced_batch_indices_from_pools

    def counted_builder(*args: Any, **kwargs: Any) -> offline.BalancedLabelIndexPools:
        nonlocal production_build_count
        production_build_count += 1
        return original_builder(*args, **kwargs)

    def observed_sampler(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[str, int]]:
        nonlocal production_final_rng_state
        indices, counts = original_sampler(*args, **kwargs)
        _digest_update(production_digest, indices)
        generator = kwargs["generator"]
        production_final_rng_state = generator.get_state().clone()
        return indices, counts

    def synchronized_clock() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    offline._build_balanced_label_pools = counted_builder
    offline._sample_balanced_batch_indices_from_pools = observed_sampler
    try:
        result = offline.run_offline_distillation_updates(
            _NoOpTrainer(),  # type: ignore[arg-type]
            dataset,
            batch_size=batch_size,
            max_updates=updates,
            repeat_dataset=True,
            shuffle=shuffle,
            seed=seed,
            balance_key=balance_key,
            balanced_labels=balanced_labels,
            balance_quotas=balance_quotas,
            performance_clock=synchronized_clock,
        )
    finally:
        offline._build_balanced_label_pools = original_builder
        offline._sample_balanced_batch_indices_from_pools = original_sampler

    reference_generator = torch.Generator().manual_seed(seed)
    if shuffle:
        torch.randperm(dataset.num_samples, generator=reference_generator)
    reference_digest = hashlib.sha256()
    selected = offline._resolve_balanced_labels(
        labels,
        batch_size=batch_size,
        balanced_labels=balanced_labels,
    )
    for _ in range(updates):
        rebuilt = original_builder(labels, selected, balance_key=balance_key)
        indices, _ = original_sampler(
            rebuilt,
            batch_size=batch_size,
            balance_quotas=balance_quotas,
            generator=reference_generator,
        )
        _digest_update(reference_digest, indices)

    staging = next(
        observation
        for observation in result.performance_stage_observations
        if observation.stage == "learner_batch_staging"
    )
    final_rng_equal = production_final_rng_state is not None and torch.equal(
        production_final_rng_state, reference_generator.get_state()
    )
    digest_equal = production_digest.hexdigest() == reference_digest.hexdigest()
    passed = production_build_count == 1 and digest_equal and final_rng_equal
    return {
        "probe": "hp7c3_label_pool_production_path",
        "training_executed": False,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_rows": dataset.num_samples,
        "device": str(device),
        "batch_size": batch_size,
        "updates": updates,
        "seed": seed,
        "shuffle": shuffle,
        "balance_key": balance_key,
        "balanced_labels": list(balanced_labels),
        "balance_quotas": balance_quotas,
        "production_cache_build_count": production_build_count,
        "production_update_count": result.update_count,
        "production_staging_seconds": staging.duration_seconds,
        "production_staging_seconds_per_update": staging.duration_seconds / updates,
        "sampled_indices_digest": production_digest.hexdigest(),
        "reference_indices_digest": reference_digest.hexdigest(),
        "sampled_indices_digest_equal": digest_equal,
        "final_rng_state_equal": final_rng_equal,
        "pass": passed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--balance-key", choices=("role", "command_intent", "scenario"), required=True
    )
    parser.add_argument("--balanced-label", action="append", default=[])
    parser.add_argument("--balance-quota", action="append", default=[])
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_production_path_probe(
        dataset_path=args.dataset,
        device=torch.device(args.device),
        batch_size=args.batch_size,
        updates=args.updates,
        seed=args.seed,
        balance_key=args.balance_key,
        balanced_labels=tuple(args.balanced_label),
        balance_quotas=_parse_quotas(args.balance_quota),
        shuffle=args.shuffle,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    if not report["pass"]:
        raise SystemExit("HP-7c3 production-path sentinel failed")


if __name__ == "__main__":
    main()
