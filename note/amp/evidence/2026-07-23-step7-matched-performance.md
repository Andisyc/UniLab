# Step 7 Matched AMP Performance Evidence

Date: 2026-07-23

Contract: `AMP-WALK-TRAIN-v001`

Classification: `performance-fail`

## Decision

The full AMP route does not meet the accepted 30% overhead bound on this local
MPS machine. After excluding two warm-up iterations, reconstructed end-to-end
time overhead is 96.6% and throughput falls from 22,107 to 11,245 consumed env
steps/s. The 10-20 minute useful-policy projection is therefore rejected for
this machine identity. This is performance evidence, not policy-quality evidence.

Step 7 is nevertheless complete under its explicit alternative stop condition:
the matched artifact, failure classification, and owner-level bottleneck verdict
are persisted. Step 8 remains unopened.

Machine-readable values are stored in
`2026-07-23-step7-matched-performance.json`.

## Matched Identity

All three runs used MuJoCo, 2048 envs, 24 steps/env, 10 learner iterations,
seed 1, MPS learner and collector devices, staging capacity 3, and the same
512/256/128 actor/critic hidden dimensions. The first two iterations were
excluded from timing aggregation as warm-up. Each run consumed one rollout per
logged iteration and ended with a clean lifecycle report.

The comparison has three arms:

1. `G1WalkFlat` plus generic APPO: unchanged baseline.
2. `G1AMPWalk` plus generic APPO: diagnostic env-only ablation selected by
   `algo.runtime_impl=null algo.runtime_resolver=null`; no formal owner config
   was changed.
3. `G1AMPWalk` plus `amp_appo`: complete Phase 1 AMP route.

## Commands

Each command used `UV_CACHE_DIR=.uv-cache`, `training.device=mps`,
`training.collector_device=mps`, `algo.num_envs=2048`,
`algo.steps_per_env=24`, `algo.max_iterations=10`, `algo.save_interval=0`,
`algo.seed=1`, `training.no_play=true`, and TensorBoard logging.

```bash
uv run train --algo appo --task g1_walk_flat --sim mujoco \
  training.log_dir=/private/tmp/unilab_amp_step7_baseline_r1

uv run train --algo appo --task g1_amp_walk --sim mujoco \
  algo.runtime_impl=null algo.runtime_resolver=null \
  training.log_dir=/private/tmp/unilab_amp_step7_envonly_r1

uv run train --algo appo --task g1_amp_walk --sim mujoco \
  training.log_dir=/private/tmp/unilab_amp_step7_amp_r1
```

The actual launches were wrapped by `/usr/bin/time -l` for peak RSS. MPS is
unified memory on this host, so a separate VRAM measurement does not exist.

## Results

| metric, warm mean unless noted | baseline | AMP env + generic APPO | full AMP |
| --- | ---: | ---: | ---: |
| reconstructed end-to-end env steps/s | 22,107 | 12,667 | 11,245 |
| training wall time, s | 26.003 | 39.803 | 48.380 |
| learner wait, ms | 234.1 | 1246.0 | 659.7 |
| learner H2D, ms | 20.6 | 18.1 | 28.0 |
| learner train, ms | 1954.9 | 2606.6 | 3656.0 |
| collector MLP infer, ms | 2.87 | 3.33 | 4.92 |
| collector env step, ms | 93.9 | 148.6 | 189.3 |
| peak RSS, bytes | 775,831,552 | 1,274,560,512 | 1,284,587,520 |
| staging occupancy after warm-up | 3/3 | 3/3 | 3/3 |

The end-to-end reconstruction uses actual `train/rollouts_read` and sums each
post-warm-up iteration's learner wait plus H2D/train/weight-sync pipeline time.
The displayed terminal `steps/s` is not used because its implementation excludes
learner wait.

Full AMP versus baseline:

- 96.6% reconstructed time overhead, above the 30% limit;
- 49.1% throughput loss;
- 86.1% training-wall-time overhead;
- 65.6% peak-RSS overhead;
- 87.0% learner-train-time overhead;
- 101.5% collector-env-step-time overhead.

## Bottleneck Verdict

The primary owner is the MuJoCo tracked-body sensor contract used by
`G1AMPWalk`, not the 195-D feature math itself.

- The env-only ablation already causes 74.5% end-to-end time overhead and 58.3%
  collector-env-step overhead relative to `G1WalkFlat`.
- `G1AMPWalkCfg.add_body_sensors=True` activates
  `inject_mujoco_tracking_sensors`, which injects position, quaternion, linear
  velocity, and angular velocity sensors in both world and pelvis-relative
  frames for every G1 body.
- The AMP task consumes only 14 world-frame body states: 13 feature bodies plus
  the pelvis anchor.
- A secondary 2048-row micro-probe measured the canonical 195-D NumPy transform
  at 1.58 ms/batch, far below the 54.7 ms env-step increase in the env-only arm.

The secondary owner is `AMPAPPOLearner`: full AMP adds 12.6% reconstructed
end-to-end time over the env-only arm, and learner train time increases another
40.3%. This arm includes AMP payload/replay/reward/discriminator work. H2D rises
to only 28.0 ms, so float16 transport is not the first justified optimization.
The current collector timer surrounds `env.step()` and does not separately time
the post-step payload writer; no stronger payload attribution is claimed.

The next performance design should first narrow the public backend body-state
contract to the requested body IDs and required frame(s), while preserving
cold-path sensor materialization and backend isolation. Only after a new matched
A/B should discriminator or transport compaction be considered.

## Lifecycle And Scope

All three runs completed 10/10 iterations with `collector_exitcode=0`,
`collector_terminated=false`, four resources, and `errors=[]`. No distillation
owner, active method semantic, reward coefficient, gait mechanism, standing,
running, recovery, or Step 8 live training was changed.

The evidence is S3 local-machine evidence. It does not establish CUDA/S4 target
performance. A CUDA target-machine A/B is still required before reviving the
10-20 minute projection.
