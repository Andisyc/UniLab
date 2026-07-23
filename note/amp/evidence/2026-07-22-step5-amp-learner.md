# Step 5 Evidence: Learner-Only AMP Ordering And Persistence

Date: 2026-07-22

Branch: `codex/amp-walk-async-migration`

## Learner Ownership

`AMPAPPOLearner` owns all training-only AMP state in the main learner process:

- discriminator and its separate optimizer;
- 195-D running normalizer;
- bounded policy transition replay;
- frozen forward-walk expert tensors;
- expert/replay sampling RNG; and
- monotonically increasing discriminator version.

The collector owns none of these objects and never scores AMP reward.

## Frozen Version Order

For every staged batch the enforced trace is:

```text
(score, D_k)
(vtrace, D_k)
(policy, D_k)
(discriminator, D_(k+1))
```

The normalizer is also held fixed through scoring, V-trace, policy update, and
the discriminator loss. It absorbs the sampled raw policy/expert states only
after the discriminator optimizer step, so the next batch sees one coherent
new scorer state.

## Reward And Discriminator Formula

The migrated AMP_mjlab reward is:

```text
r_amp = reward_coef * clamp(1 - 0.25 * (D(s,s') - 1)^2, min=0)
r_combined = task_reward_lerp * r_task + (1-task_reward_lerp) * r_amp
```

The discriminator uses MSE targets `expert=+1`, `policy=-1`, plus a configured
expert-input gradient penalty. The combined reward replaces the batch reward
exactly once before `APPOLearner.process_batch`, which is the V-trace owner.
Calling the preparation twice on one batch fails closed.

## Persistence

The existing CPU-owned atomic APPO checkpoint now contains actor, critic,
policy optimizer, discriminator, discriminator optimizer, normalizer, complete
zero-initialized replay storage and cursors, generator state, and discriminator
version. A second learner with a different initial seed restored every AMP
tensor, cursor, RNG byte state, and version exactly.

## Verification

```text
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/algos/test_amp_appo_learner.py \
  tests/algos/test_amp_motion_dataset.py \
  tests/algos/test_appo_learner.py \
  tests/algos/test_appo_learner_metrics.py \
  tests/algos/test_appo_checkpoint.py -q

13 passed in 0.38s
```

Ruff passed.

## Verdict

Step 5 is `PASS`. Reward value, one-time V-trace connectivity, frozen-D order,
separate discriminator update, finite replay, and full learner state roundtrip
are directly tested.
