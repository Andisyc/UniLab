# Symmetric Self-Collision Repair Steps 1-2 Evidence

Date: 2026-07-23

Classification: `runtime-pass / style-health-pass /
collision-connectivity-pass` for Engineering Steps 1-2 under
`AMP-WALK-METHOD-v003` and `AMP-WALK-TRAIN-v004`. This is not a repaired-policy
quality or visible hand-leg-contact acceptance result.

## Frozen Boundary

- Whole-body symmetric `pelvis` subtree against itself.
- Contact force, one slot, `reduce="none"`, four physics-substep samples.
- Strict force magnitude threshold `> 10.0 N` and reward scale `-0.1`.
- AMP-only task XML and Hydra config; robot XML and `G1WalkFlat` remain unchanged.
- No right-hand-specific pose/distance term, gait phase, standing, running,
  recovery, motion reset, discriminator tuning, Motrix, or distillation.
- Existing learner-only `D_k -> V-trace -> actor/critic -> D_(k+1)` and official
  APPO async lifecycle remain unchanged.

Repo HEAD at execution: `ef0e2c229f798e061fffa4aedc554d3c11e9d1f8`.
The repair was present as an uncommitted working-tree diff.

## Runtime-Probing Correction

The first backend implementation split one native `BatchEnvPool.step(nstep=4)`
call into four `nstep=1` calls to capture final-step sensor values. A new
same-model/same-control differential test rejected that implementation: after
80 physics substeps the state differed by up to `3.0994415e-06`. The associated
20-iteration probe is invalid acceptance evidence even though it completed
cleanly; it also failed the logit gate narrowly.

MjLab v1.2.0 defines contact history across decimation substeps. MuJoCo 3.8
stores a sensor ring in the `mjSTATE_HISTORY` component of
`mjSTATE_FULLPHYSICS`. The final implementation therefore declares
`nsample="4"` on the task-level contact sensor and reads the native ring
oldest-to-newest after the unchanged single native `nstep=N` call. No hot-path
XML parsing or backend-private env access is used.

## Implemented Owners

- `SimBackend`: public fail-closed configure/read sensor-history contract.
- `MuJoCoBackend`: cold-path native history metadata resolution, ring-order
  reconstruction, and reset-row refresh from FULLPHYSICS state.
- AMP task XML: symmetric source contact sensor with native four-sample history.
- `G1AMPWalkEnv`: exact source reducer plus raw hit-count/rate diagnostics.
- AMP owner YAML: isolated `self_collisions: -0.1` coefficient.

The default backend route still performs one `BatchEnvPool.step(nstep=N)` call.
The enabled route performs that same one call and reads the native state ring.

## Deterministic And Regression Evidence

- Failing-first import/config/reducer tests were observed before implementation.
- The physical-trajectory differential and APPO diagnostic-log test both failed
  against the split-step implementation, then passed after the native-ring fix.
- Native ring ordering, strict threshold, multi-slot reduction, nonzero forced
  contact, partial-reset isolation, AMP-only config, and OFF fast path pass.
- Three slow MuJoCo boundary tests: `3 passed`.
- Focused impact suite: `288 passed, 226 deselected`.
- Ruff: `All checks passed`; `git diff --check` passed.

At 512 environments, 30 measured four-substep calls after warmup gave:

| Route | Median env step |
| --- | ---: |
| OFF | `16.3900 ms` |
| AMP native history ON | `17.2635 ms` |

The observed median ratio is `1.05329` (about `5.33%` overhead). This is a
diagnostic, not a fixed performance gate. Default tasks do not configure the
sensor or history.

## Formal Step 2 Sentinel

Command:

```bash
uv run train --algo appo --task g1_amp_walk --sim mujoco \
  training.device=mps training.collector_device=mps training.no_play=true \
  algo.num_envs=2048 algo.steps_per_env=24 algo.max_iterations=20 \
  algo.save_interval=0 algo.load_run=null \
  training.log_dir=/private/tmp/unilab_amp_self_collision_step2_sentinel_native
```

The fresh run completed 20/20 iterations and 983,040 environment steps in 60
seconds. Lifecycle report: collector exit code 0, no forced termination, no
errors, state `complete`.

### Frozen Tail-Five Gate

| Metric | Gate | Tail-five mean | Result |
| --- | ---: | ---: | --- |
| `amp/policy_logit_p50` | `> -0.95` | `-0.93973405` | PASS |
| `amp/policy_zero_style_fraction` | `< 0.50` | `0.31870117` | PASS |
| `amp/style_reward_mean` | `> 0.005` | `0.00881476` | PASS |

Supporting tail-five means:

- `amp/task_weighted_mean=-0.00913823`;
- `amp/style_weighted_mean=0.00220369`;
- `reward/tracking_lin_vel=0.15175867`;
- `reward/self_collisions=-0.02904297`;
- `reward/self_collision_hit_count_mean=0.29042969`;
- `reward/self_collision_hit_rate_mean=0.07260742`.

Collision values are finite and structurally nonzero. This proves
sensor-to-reducer-to-task-reward connectivity, not that collision quality has
already improved after 20 iterations.

## Artifact Identity

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `run_config.json` | 7,449 | `835a93bf8bae01a65886698f8fbc3ab213314eac41a2ae615c9583d4db86eef7` |
| `run_summary.json` | 912 | `42a794b647e8914aef469676a839bb6d5b92ccf0f570592ccedae189e1cd49d3` |
| TensorBoard event | 77,868 | `655dfb7e911d98f56c37350073929ef1401b792ef4ecdcd6e2c8af03d03272c5` |
| `model_20.pt` | 69,036,299 | `d16074316f616ce87b971d01ad6a9b8727ba8cbc478db371c6f9f4409411ffb7` |

Artifacts remain under
`/private/tmp/unilab_amp_self_collision_step2_sentinel_native`. The checkpoint
is sentinel evidence only and is not an authorized Step 3 resume source.

## Decision

Engineering Steps 1 and 2 are complete. Step 3 remains blocked on separate
human authorization. It must train from scratch and perform matched old/new
collision evaluation plus human playback review; no 20-iteration quality claim
is promoted.
