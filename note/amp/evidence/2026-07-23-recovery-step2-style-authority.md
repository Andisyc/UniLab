# Recovery Step 2 Style-Authority Evidence

Date: 2026-07-23

Classification: `pass` for the v003 short-sentinel gate, with minimal logit
margin. This is not a policy-quality or human-like-walking acceptance result.

## Frozen Boundary

- active method: `AMP-WALK-METHOD-v002`;
- active training contract: `AMP-WALK-TRAIN-v003`;
- fresh initialization: `algo.load_run=null`;
- fixed forward command: `[1.0, 0.0, 0.0]`;
- no default-pose reward, gait-phase reward/observation, motion reset, standing,
  running, fall recovery, Motrix, playback, or distillation;
- official UniLab APPO async runner, collector, typed rollout ring, staging pool,
  learner, and lifecycle owners remain unchanged.

Repository HEAD at execution: `553c5790188328be4fc3b982d02dc5a2ff776ac7`.
The recovery changes were present as an uncommitted working-tree diff at this
identity.

## Owner Repair

- the AMP owner YAML no longer enables inherited `pose` reward authority;
- `G1AMPWalkEnv` rejects any nonzero `pose` scale, while `G1WalkFlat` is not
  modified;
- `WalkMotionDataset` exposes two motion clips separately from 935 unique
  adjacent transitions;
- the learner emits frozen-scoring-batch policy logit quantiles, zero-style
  fraction, weighted task/style means, and expert clip/transition/draw counts;
- the existing `D_k -> V-trace -> actor/critic -> D_(k+1)` order is unchanged.

## Deterministic Verification

Failing-first tests were observed for the missing pose exclusion, env guard,
motion count, and policy-health diagnostics. After the owner repair:

```text
4 passed in 0.66s
24 passed, 1 deselected in 0.51s
1 slow live MuJoCo reset/timeout test passed in 0.67s
ruff: All checks passed
```

The composed Hydra job had `reward.scales.pose` absent, no enabled
`feet_phase*` reward, `load_run: null`, 2048 environments, 24 steps per env,
and 20 iterations.

## Live Execution

Two sandboxed preflights stopped before iteration zero and closed cleanly:

- `r1`: sandbox could not expose the macOS MPS backend;
- `r2`: sandbox denied POSIX shared-memory creation.

These are execution-boundary observations, not AMP/runtime failures. The formal
run was therefore executed outside the sandbox with the frozen MPS command:

```bash
UV_CACHE_DIR=/private/tmp/unilab-uv-cache uv run --no-sync train \
  --algo appo --task g1_amp_walk --sim mujoco \
  training.device=mps training.collector_device=mps training.no_play=true \
  algo.num_envs=2048 algo.steps_per_env=24 algo.max_iterations=20 \
  algo.save_interval=0 algo.load_run=null \
  training.log_dir=/private/tmp/unilab_amp_recovery_step2_sentinel_r3
```

Result:

- 20/20 learner iterations;
- 983,040 environment steps;
- 64.73 s outer wall time, 63.08 s training wall time;
- collector exit code 0, no forced termination, no lifecycle errors;
- `resource_count=4` closed under lifecycle state `complete`;
- effective/configured seed 1;
- expert identity: 2 clips, 935 unique transitions, 4096 draws per update.

Although `save_interval=0`, the runner's final-save behavior emitted
`model_20.pt`. It is sentinel evidence only and is not an authorized Step 3
resume source.

## Frozen Tail-Five Gate

| Metric | v003 gate | Last-five mean | Result |
| --- | ---: | ---: | --- |
| `amp/policy_logit_p50` | `> -0.95` | `-0.94973273` | PASS |
| `amp/policy_zero_style_fraction` | `< 0.50` | `0.35026855` | PASS |
| `amp/style_reward_mean` | `> 0.005` | `0.00832356` | PASS |

Supporting weighted means were `amp/task_weighted_mean=-0.00872293` and
`amp/style_weighted_mean=0.00208089`.

The logit gate clears by only about `0.00027`. The final individual iteration
also regressed to policy-logit median `-1.04390`, zero-style fraction `0.62683`,
and style reward `0.00355`. The contract classifies the frozen tail-five result
as `pass`, but the narrow/non-monotonic margin remains a Step 3 quality risk.

## Artifact Identity

| Artifact | SHA-256 |
| --- | --- |
| `run_config.json` | `0b2f8d6e648817bbbb2a4d090d9219ed722a8ba13ddf6e0adea6e0f3db984913` |
| `run_summary.json` | `8962a914ef7d190029d97e63f052d58e84b47615375d560afda30524441d29b2` |
| TensorBoard event | `fe898f5613aaa8cd08e4f5a9dacc987229ece4c1449a114e55b74dc883499b48` |
| `model_20.pt` | `979e8ac11aad39789fc05d913f7745058112494d83d320975e9d10c46f0ddba4` |

Artifacts remain under
`/private/tmp/unilab_amp_recovery_step2_sentinel_r3` and are not promoted into
the repository.

## Decision

Recovery Step 2 is complete as `pass`: pose authority is removed, support and
style health are observable, and the official async sentinel crossed the
predeclared non-collapse thresholds with a clean lifecycle. Recovery Step 3 is
ready for a separate human authorization but was not started. No claim is made
that the 20-iteration actor already walks with acceptable human-like style.
