# AMP-WALK Phase 1 Engineering Plan

Status: `active`

Date: 2026-07-22

Contracts:

- `AMP-WALK-METHOD-v001`
- `AMP-WALK-TRAIN-v002`

Concept Figure:

- `note/architecture/concept/04_amp_walk_async_method.data.json`

Plan cursor: Steps 1-7 complete. Measured Step 7 overhead is accepted under
`AMP-WALK-TRAIN-v002`; stopped before Step 8 bounded live acceptance.

## Terminal Outcome

One fixed-nonzero-forward G1 AMP policy trains through UniLab's official
asynchronous APPO route, produces an atomically reloadable actor checkpoint,
exits without process/shared-memory residue, and is evaluated in one bounded
live run, nominally 20-30 minutes under training contract v002. Standing,
running, recovery, gait control, Motrix, and
AMP/distillation integration remain outside Phase 1.

## Human Method Map

| design ID | canonical human name | active contract + section | Concept Figure block ID | code/evidence gap |
| --- | --- | --- | --- | --- |
| `AMP-WALK-DP-01` | Walk Expert Transitions | `AMP-WALK-METHOD-v001#walk-expert-transitions` | `AW-M-01` | Forward-only manifest, 195-D transform, and sampler implemented |
| `AMP-WALK-DP-02` | Policy Walk Transitions | `AMP-WALK-METHOD-v001#policy-walk-transitions` | `AW-M-02` | Isolated G1 AMP producer and terminal transition owner implemented |
| `AMP-WALK-DP-03` | AMP Style Discriminator | `AMP-WALK-METHOD-v001#amp-style-discriminator` | `AW-M-03` | Learner-local D/normalizer/replay/update implemented |
| `AMP-WALK-DP-04` | AMP-Regularized Walking Policy | `AMP-WALK-METHOD-v001#amp-regularized-walking-policy` | `AW-M-04` | Formal runtime verified; local MPS performance failed with owner verdict; live quality pending |

## Semantic Source Of Truth

| semantic object | active owner | active consumers | legacy/source path | retirement/isolation rule | implementation test | integration test | live gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 1 forward-walk manifest | `amp/motion_dataset.py` plus frozen asset manifest | expert sampler | AMP_mjlab recursive motion loading | two SHA-locked forward clips; non-forward clips fail closed | manifest oracle PASS | formal compose/load | source diversity |
| 195-D AMP state | `amp/spec.py` plus `G1AMPWalkEnv` | payload, discriminator | AMP_mjlab observation manager | one 13-body order/transform; existing task unchanged | source-frame parity PASS | live env reset/step PASS | sensor cost |
| policy AMP transition | typed APPO payload plus `resolve_amp_transition_next` | learner replay/scoring | AMP_mjlab runner-local transition | exact final observation; no collector scoring | partial-terminal order PASS | spawned 195-D IPC PASS | termination mix |
| AMP style reward | `AMPAPPOLearner` | V-trace reward owner, diagnostics | AMP_mjlab discriminator formula | frozen scorer `D_k` per batch | deterministic value/order PASS | reward-to-V-trace PASS | learning scale |
| combined reward | `AMPAPPOLearner.process_batch` | APPO V-trace/update | current task reward only | task/AMP/final values logged; duplicate apply fails | numeric composition PASS | consumer connectivity PASS | policy quality |
| checkpoint | generic atomic APPO owner plus AMP learner state | resume, playback/export | direct APPO `torch.save` | CPU-owned atomic save; one startup resume | exact full-state roundtrip PASS | spawned resume and actor-only playback PASS | server filesystem |

There is one active owner per semantic object. Source/legacy routes remain
references or isolated existing behavior; they do not participate in the AMP
formal route unless explicitly adapted and covered by the named tests.

## Execution Units And True Boundaries

The eight steps are audit-visible causality units, not eight mandatory user
approvals. The one-shot deletion test produces four execution boundaries:

1. **Foundation boundary, Step 1:** lifecycle residue or unstable baseline would
   invalidate all later work.
2. **Main engineering closure, Steps 2-6:** implement payload, data/env, learner,
   and formal runtime with embedded local/integration verification in one
   authorized closure.
3. **Performance observation boundary, Step 7:** optimization is allowed only
   after float32 correctness and requires measured A/B evidence. A fixed
   overhead percentage is not a Phase 1 acceptance gate under training v002.
4. **Material live boundary, Step 8:** the bounded GPU run has external cost and
   requires explicit authorization.

Different owner files and test tiers inside Steps 2-6 do not create additional
approval gates.

## Step Map

### Step 1 / 8: Freeze The APPO Baseline And Lifecycle Floor

Objective: establish stable throughput ownership and clean process/resource
lifecycle before introducing AMP state.

Scope: unchanged `g1_walk_flat/mujoco` at 2048 environments; normal/failure
collector close; queue/pipe/shared-memory ownership; CPU-owned atomic APPO
checkpoint; generic APPO/IPC owner repairs when required.

Non-scope: AMP semantics, AMP reward, distillation code, or policy tuning.

Owner files/modules: `scripts/train_appo.py`, `src/unilab/algos/torch/appo/`
runner/worker, `src/unilab/ipc/async_runner.py`, ring buffer, weight sync,
checkpoint owner, and focused tests.

Expected evidence: S2-S4 / T-order,T-persist,T-diff,T-performance proving
bounded child exit, explicit resource closure/unlink, atomic checkpoint reload,
unchanged APPO behavior, and a reproducible timing baseline.

Stop condition: baseline and lifecycle report pass with no child/shared-memory
residue. Any missing cleanup identity or unstable baseline blocks Step 2.

### Step 2 / 8: Add Reusable APPO Payload Extension Hooks

Objective: allow algorithm-owned rollout fields without copying the APPO runner
or lifecycle protocol.

Scope: typed extra ring fields/dtypes, collector payload hook, staging support,
attached-child cleanup, and complete memory-budget accounting.

Non-scope: AMP feature meaning, reward, or discriminator.

Owner files/modules: rollout ring buffer, APPO runner/worker/staging, memory
budget, and fake spawned-payload tests.

Expected evidence: S1-S3 / T-shape,T-connect,T-order,T-diff; a fake 195-D
current/next payload crosses collector -> ring -> staging -> learner while
default APPO/HORA field behavior stays unchanged.

Stop condition: optional payload survives spawned IPC with exact shape/order and
no copied runner branch.

### Step 3 / 8: Establish Walk Expert And AMP-State Owners

Objective: implement one forward-only expert distribution and one canonical
195-D AMP feature transform.

Scope: fail-closed manifest, 13-body ordering, deterministic transition
sampling, vectorized cold-path conversion, normalizer input, and AMP_mjlab
feature parity.

Non-scope: policy rollout, task reward, or learning.

Owner files/modules: planned `src/unilab/algos/torch/amp/spec.py`,
`motion_dataset.py`, motion manifest/assets, and rotation helpers.

Expected evidence: S0-S1 / T-oracle,T-shape,T-value,T-diff; non-forward clips
cannot enter and sampled 195-D/195-D transitions match AMP_mjlab within tolerance.

Stop condition: deterministic expert transition sampling and manifest identity
are reproducible.

### Step 4 / 8: Add AMP-Only G1 Environment Contract

Objective: emit exact policy AMP transitions without gait or standing ownership.

Scope: isolated AMP task/config, init-time tracked-body setup, cached body IDs,
`amp` observation group, exact terminal/final AMP state, partial reset, fixed
nonzero forward objective, and no gait-phase input/reward.

Non-scope: mutation of existing `G1WalkFlat`, Motrix, recovery, or motion reset.

Owner files/modules: G1 locomotion environment adapter, declared backend calls,
new AMP owner YAML, registry, and env contract tests.

Expected evidence: S1-S2 / T-shape,T-order,T-diff,T-connect for actor/critic/AMP
groups, full rollout, done rows, subset reset, and legacy-task isolation.

Stop condition: exact current/next/terminal AMP transitions are correct and the
existing G1 task is unchanged.

### Step 5 / 8: Implement AMP APPO Learner

Objective: make learner-only AMP reward and discriminator training first-class
owners.

Scope: discriminator, normalizer, policy replay, expert sampling, frozen `D_k`
scoring, combined reward, discriminator optimizer/version, and learner state.

Non-scope: collector-side discriminator, deployment AMP, or in-loop checkpoint
reload.

Owner files/modules: planned AMP discriminator/replay/APPO learner modules and
deterministic learner tests.

Expected evidence: S1-S2 / T-value,T-order,T-connect,T-persist proving exact
formula, one score/version per batch, reward-to-V-trace connectivity,
actor/critic-before-discriminator ordering, and state roundtrip.

Stop condition: a fake async batch changes V-trace through AMP reward and
advances exactly one discriminator version after the policy update.

### Step 6 / 8: Connect The Formal Async Runtime

Objective: compose the AMP task, payload, and learner through the existing APPO
entrypoint and runtime resolver.

Scope: runtime bundle, Hydra task/config, formal spawned route, checkpoint/
logging, normal/failure lifecycle report, resume, and actor-only playback.

Non-scope: second training script, copied synchronization protocol,
distillation integration, or policy-quality tuning.

Owner files/modules: planned AMP APPO runtime adapters, generic APPO resolver,
`scripts/train_appo.py` assembly only, config/registry, and spawned integration
tests.

Expected evidence: S2-S3 / T-connect,T-order,T-persist,T-diff; a formal
two-iteration run produces finite AMP/APPO metrics, atomically reloadable state,
clean injected-failure shutdown, and actor-only playback.

Stop condition: the official command completes/resumes/plays without resource
residue and without touching legacy G1 or distillation paths.

### Step 7 / 8: Measure And Optimize The AMP Async Path

Objective: bound AMP overhead against the Step 1 baseline.

Scope: matched timing, queue/staging occupancy, memory/RSS/VRAM, discriminator
cost, and only then optional float16 transport or payload compaction.

Non-scope: reward/method tuning used to hide a throughput defect.

Owner files/modules: AMP runtime metrics, IPC dtype extension when justified,
benchmark command, and performance evidence ledger.

Expected evidence: S3-S4 / T-performance,T-diff with one matched APPO-versus-AMP
A/B artifact.

Stop condition: matched overhead and owner-level bottlenecks are persisted;
forward progress, bounded completion, memory capacity, and lifecycle remain
valid. Overhead alone does not block Phase 1.

Observed result: the matched local MPS identity has 96.6% reconstructed
end-to-end time overhead. The primary owner is the
over-broad MuJoCo all-body/world-and-base tracking sensor route; the secondary
owner is AMP learner work. See
`evidence/2026-07-23-step7-matched-performance.md`. The stop condition is met by
the persisted observation and three clean completed runs. Under
`AMP-WALK-TRAIN-v002`, the human owner accepts this extra training time and the
result does not block Step 8.

### Step 8 / 8: Bounded Live Acceptance

Objective: test the wall-clock and fixed-forward walking hypothesis.

Scope: one frozen command/config/data/checkpoint identity, nominally 20-30
minutes of training, postflight lifecycle check, diagnostics, and playback.

Non-scope: standing, stopping, running, recovery, gait control, indefinite
tuning, Motrix, or Phase 2 distillation.

Owner files/modules: formal launch/postflight owner, experiment tracker,
playback route, and AMP evidence ledger.

Expected evidence: S4 / T-live,T-performance containing wall time, env steps,
checkpoint hash, task/AMP/combined reward, logits/loss/version curves,
termination rate, lifecycle report, and playback.

Stop condition: classify the result as runtime-pass/quality-pass,
runtime-pass/quality-fail, lifecycle-fail, or capacity-fail. Extra bounded
training time alone is not failure. Do not infer a useful policy from a clean
process or finite checkpoint alone.

## Conditional Escalation

- Formal route not reached or owner forwarding contradicts the contract:
  activate only the smallest relevant formal-runtime audit.
- Formal route runs but policy is no-op or poor: activate policy-quality audit
  after preserving the run identity.
- Moving impossible-object/native symptoms recur: stop business-logic patching
  and return to the native/lifecycle campaign.
- A new behavior objective or AMP/distillation interaction is requested: stop
  Phase 1 and create a new contract proposal/version.

## Current Authority Boundary

Steps 1-7 are complete under their accepted contracts. Step 8 simulator startup,
bounded training, and policy-quality acceptance remain unauthorized until the
user explicitly continues. Performance redesign and CUDA A/B are optional
optimization work, not prerequisites for Step 8.
