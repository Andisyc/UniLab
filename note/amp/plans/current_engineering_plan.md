# AMP-WALK Phase 1 Style-Authority Recovery Plan

Status: `paused before Step 3`

Date: 2026-07-23

Contracts:

- `AMP-WALK-METHOD-v002`
- `AMP-WALK-TRAIN-v003`

Concept Figure:

- `note/architecture/concept/04_amp_walk_async_method.data.json`

Plan cursor: Recovery Step 2 / 3 passed with minimal logit margin. Stop before
Step 3 material GPU training; Step 3 requires separate human authorization.

## Terminal Outcome

Produce a fresh-training-ready Phase 1 AMP route in which task reward owns only
fixed-forward motion and minimum physical viability, AMP exclusively owns
human-like posture/style, and one 20-iteration official async sentinel proves
that the style signal is not already collapsed. This plan does not execute the
material Step 3 GPU quality run.

## Human Method Map

| design ID | canonical human name | active contract + section | Concept Figure block ID | current gap |
| --- | --- | --- | --- | --- |
| `AMP-WALK-DP-01` | Walk Expert Transitions | `AMP-WALK-METHOD-v002#walk-expert-transitions` | `AW-M-01` | expose two clips/935 unique transitions separately from draws |
| `AMP-WALK-DP-02` | Policy Walk Transitions | `AMP-WALK-METHOD-v002#policy-walk-transitions` | `AW-M-02` | unchanged, regression only |
| `AMP-WALK-DP-03` | AMP Style Discriminator | `AMP-WALK-METHOD-v002#amp-style-discriminator` | `AW-M-03` | add quantile/zero-plateau diagnostics and pass sentinel |
| `AMP-WALK-DP-04` | AMP-Regularized Walking Policy | `AMP-WALK-METHOD-v002#amp-regularized-walking-policy` | `AW-M-04` | retire default-pose task authority and verify V-trace connectivity |

## Semantic Source Of Truth

| semantic object | active owner | consumers | legacy path | isolation rule | implementation evidence | live gap |
| --- | --- | --- | --- | --- | --- | --- |
| task posture authority | AMP owner YAML | G1 reward owner -> AMP combined reward | inherited `G1WalkFlat` pose scale | AMP `pose` must be absent/zero; legacy task unchanged | composed-config and reward connectivity tests | fresh sentinel |
| expert support | manifest + `WalkMotionDataset` | learner expert sampler | preload/draw count | report clips=2, unique transitions=935, draws separately | dataset fixture and learner metrics | no extra forward clips exist |
| style health | `AMPAPPOLearner.process_batch` | diagnostics and readiness evaluator | mean-only metrics | log p10/p50/p90, zero fraction, weighted components | deterministic batch test | tail-five sentinel gate |
| score/update order | `AMPAPPOLearner` | APPO V-trace/update | none | preserve `D_k -> V-trace -> policy -> D_(k+1)` | existing order/connectivity regression | formal sentinel |
| recovery identity | runner/config | checkpoint and tracker | `model_1850.pt` | fresh initialization only | no-load effective config and checkpoint identity | Step 3 blocked |

## Step Map

### Step 1 / 3: Activate Task/Style Authority

Objective: version the human-confirmed training-signal authority before code.

Scope: activate method/training contracts, preserve four design/block IDs,
synchronize Concept Figure, registry, plan, checklist, and canvas.

Non-scope: code, tests, simulator, or training.

Expected evidence: atlas validator resolves all four design points to active
contracts; history contracts are excluded from default recall.

Stop condition: `AMP-WALK-METHOD-v002` and `AMP-WALK-TRAIN-v003` are the only
active AMP contracts and the four-block figure matches them.

Status: complete.

### Step 2 / 3: Engineering And Short-Sentinel Closure

Objective: make AMP style measurable before any material training run.

Scope:

- set AMP task default-joint `pose` reward to zero without modifying
  `G1WalkFlat`;
- expose expert support and sampling identities;
- expose frozen-scoring-batch policy logit quantiles, zero-style fraction, and
  weighted task/style reward contributions;
- add failing-first deterministic tests, focused regressions, composed config
  verification, and one fresh 20-iteration official async sentinel;
- inspect the sentinel event/log/lifecycle identity and persist evidence;
- synchronize checklist, canvas, contracts, and Architecture current state.

Non-scope: motion reset, motion-reset curriculum, new motion generation,
discriminator sweeps, checkpoint resume, long GPU training, standing, running,
recovery, gait control, Motrix, or distillation.

Owner files/modules:

- `conf/appo/task/g1_amp_walk/mujoco.yaml`: task reward identity;
- `src/unilab/algos/torch/amp/motion_dataset.py`: support identity;
- `src/unilab/algos/torch/amp/learner.py`: scoring-batch diagnostics;
- AMP dataset/learner/runtime tests: deterministic and connector evidence;
- formal `uv run train --algo appo --task g1_amp_walk --sim mujoco`: live sentinel.

Core parameter path:

```text
AMP owner YAML pose=0
-> G1 reward component map
-> task reward batch
-> frozen D_k policy transition score
-> style reward and zero plateau
-> weighted combined reward
-> V-trace/APPO update
-> TensorBoard diagnostics
```

Test classes:

- core param path: reward/style metrics and combined-reward consumer;
- secondary contract path: Hydra config and expert support identity;
- live sentinel path: real MuJoCo collector/IPC/learner/lifecycle.

Commands:

```bash
uv run pytest -q tests/algos/test_amp_motion_dataset.py \
  tests/algos/test_amp_appo_learner.py \
  tests/algos/test_amp_appo_runtime.py \
  tests/envs/locomotion/g1/test_amp_walk.py

uv run train --algo appo --task g1_amp_walk --sim mujoco \
  training.device=mps training.collector_device=mps training.no_play=true \
  algo.num_envs=2048 algo.steps_per_env=24 algo.max_iterations=20 \
  algo.save_interval=0 algo.load_run=null \
  training.log_dir=/private/tmp/unilab_amp_recovery_step2_sentinel
```

Expected result: focused tests pass; fresh sentinel completes 20/20 with clean
lifecycle and final-five means satisfying the v003 style-health gate.

Stop condition: classify Step 2 as `pass`, `expert-support-blocker`,
`style-saturation-blocker`, `lifecycle-fail`, or `capacity-fail`. On any failure,
do not enter Step 3 and do not tune another mechanism in the same run.

Status: complete as `pass`. The formal MPS run completed 20/20 with clean
lifecycle. All three frozen final-five gates passed, but policy-logit median
cleared its threshold by only about 0.00027 and the final point regressed. See
`evidence/2026-07-23-recovery-step2-style-authority.md`.

### Step 3 / 3: Fresh Bounded GPU Quality Acceptance

Objective: train and judge human-like fixed-forward AMP walking.

Scope: one fresh frozen target-GPU run, lifecycle postflight, artifacts,
actor-only playback, and physical-quality judgment.

Non-scope: resume from `model_1850.pt`, repeated tuning, expanded locomotion,
or Phase 2 distillation.

Expected evidence: frozen commit/config/data identity, TensorBoard curves,
checkpoint hash, lifecycle report, and human-like walking playback.

Stop condition: one terminal quality/lifecycle classification.

Status: blocked on separate human authorization. Step 2 passed; this execution
stopped before starting Step 3.

## Conditional Escalation

- The source repository has only two compatible forward-walk clips. No extra
  support may be invented or inferred from non-forward filenames.
- Source AMP uses motion-frame reset, but v002 forbids it. If the sentinel
  saturates, return this no-RSI boundary to the human owner instead of silently
  adding motion reset.
- Healthy AMP diagnostics with non-human playback would reopen the 195-D style
  representation, not task reward coefficients.
- Native/lifecycle anomalies return to the existing native owner-boundary
  campaign without changing AMP semantics.

## Authority Boundary

Recovery Steps 1-2 are authorized in one closure. Step 3 is a material remote
GPU training boundary and remains unauthorized.
