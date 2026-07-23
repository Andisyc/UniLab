---
contract_id: AMP-WALK-TRAIN-v003
status: active
effective_date: 2026-07-23
updated_date: 2026-07-23
supersedes: AMP-WALK-TRAIN-v002
scope: Phase 1 AMP on UniLab asynchronous APPO with diagnostic-only throughput overhead
method_contract: AMP-WALK-METHOD-v002
concept_figure: note/architecture/concept/04_amp_walk_async_method.data.json
implementation_status: style_authority_recovery
promotion_status: recovery_step_2
---

# AMP Walk Asynchronous Training Contract

## Recovery Decision

Step 8 under v002 completed as `runtime-pass / quality-fail`. Recovery starts
from random actor/critic/discriminator initialization. `model_1850.pt` is
evidence/playback-only and cannot initialize or normalize the recovery run.

The AMP owner YAML must set default-joint `pose` reward to zero. The remaining
task terms own fixed-forward motion and minimum physical viability only. AMP is
the only owner of human-like joint posture and transition style.

The expert owner must report source clip count, unique adjacent-transition
count, and sampled draw count separately. The current accepted support is two
clips and 935 unique transitions; a 200,000-draw preload or repeated sampling
does not increase that support identity.

## Short Sentinel Gate

Before the material Step 3 GPU run, one fresh official async sentinel must run
for 20 learner iterations at 2048 environments and 24 steps per environment.
It must use the active owner config, no checkpoint load, no motion reset, and
no playback.

The learner must emit these batch-owner diagnostics:

- `amp/policy_logit_p10`, `amp/policy_logit_p50`, and
  `amp/policy_logit_p90` from the frozen `D_k` scoring batch;
- `amp/policy_zero_style_fraction`, where style reward equals the clamped zero
  plateau;
- `amp/task_weighted_mean` and `amp/style_weighted_mean` after applying the
  configured reward mixture;
- `amp/expert_motion_count`, `amp/expert_transition_count`, and
  `amp/expert_draw_count`.

The final five iterations pass the style-health gate only when all diagnostics
are finite and their means satisfy:

```text
amp/policy_logit_p50 > -0.95
amp/policy_zero_style_fraction < 0.50
amp/style_reward_mean > 0.005
```

The `-0.95` and `0.005` limits are coupled by the active AMP formula: a logit
of `-0.95` yields approximately `0.00494` reward with `reward_coef=0.1`, while
the zero plateau begins at `-1`. The fraction gate prevents a favorable mean
from hiding a collapsed majority. These are readiness invariants, not a policy-
quality claim and not thresholds that may be changed after observing the run.

Sentinel failure stops before Step 3 and is classified as
`expert-support-blocker`, `style-saturation-blocker`, `lifecycle-fail`, or
`capacity-fail`. It does not authorize coefficient sweeps, discriminator
regularizer changes, or motion reset.

## Decision

Phase 1 uses UniLab's existing spawned collector/learner APPO architecture. The
AMP discriminator remains learner-only. For each staged rollout batch, the
learner freezes `D_k`, scores every policy AMP transition exactly once, forms
the final reward before V-trace, updates actor/critic, then updates the
discriminator to `D_(k+1)`.

The acceptance priority is correct integration with UniLab's high-speed
asynchronous runtime, bounded forward progress, and clean process/resource
lifecycle. AMP throughput overhead is measured and retained as diagnostic
evidence, but a fixed percentage ceiling is not a Phase 1 blocker. An additional
training wait on the order of ten minutes is acceptable to the human owner.

## Formal Runtime Order

```text
spawn one collector under actor/critic version k
-> collect actor/critic/action/behavior-logp/task-reward/AMP-transition payload
-> stage owned float32 rollout batches on the learner device
-> freeze D_k for the staged batch
-> score policy AMP transitions once
-> combine task and AMP rewards
-> run V-trace and APPO actor/critic update
-> update discriminator to D_(k+1)
-> publish actor/critic weights only
```

## Runtime Ownership

- `AsyncRunner` owns the single collector process lifecycle and error channel.
- Generic APPO/IPC owners provide extensible rollout fields, shared actor/critic
  weights, bounded staging, V-trace, and liveness checks.
- The isolated G1 AMP task owns actor/critic/AMP observation groups and exact
  terminal transition construction.
- The AMP learner owns discriminator, normalizer, expert sampler, policy replay,
  reward composition, discriminator optimizer, and discriminator version.
- Hydra/registry/runtime resolver select owners. No second training script or
  copied collector/learner synchronization protocol is allowed.

## Process And Resource Lifecycle

The Phase 1 runtime inherits the steady-state lessons from
`note/distill_native_corruption/ownerfix_r4_resolution.md` without importing
DAgger semantics:

1. Use multiprocessing `spawn` and one collector process.
2. Collector initialization and hot loop must be covered by owner-local
   `finally` cleanup for environment, ring attachments, and weight attachments.
3. Parent close order is stop signal, bounded join, terminate only as fallback,
   verified worker exit, explicit queue/pipe close, then parent shared-memory
   close/unlink.
4. Cleanup failures must be visible in the formal lifecycle report; a normal
   formal gate must not rely on swallowed exceptions or `__del__`.
5. Discriminator/replay/optimizer state never crosses process boundaries.
6. No checkpoint is loaded repeatedly inside the update loop.

## Checkpoint Contract

- Checkpoint payloads are recursively detached to CPU before serialization.
- Save uses a same-directory temporary file and atomic replacement.
- Periodic training state may contain actor, critic, APPO optimizer,
  discriminator, discriminator optimizer, normalizer, replay identity, counters,
  and version metadata.
- An explicit resume loads once before collector startup and then synchronizes
  the resumed actor/critic identity to collector weights.
- Actor-only playback/export must not require expert data, replay, discriminator,
  or either optimizer.
- Exact reload and no-residue process cleanup are independent acceptance items.

## Data And IPC Contract

- Correctness begins with float32 current/next AMP states.
- At 2048 environments and 24 steps, memory budget checks must include all AMP
  ring slots and learner staging before allocation.
- Optional field extension must leave default APPO and HORA payload behavior
  unchanged when no extra fields are configured.
- Float16 transport or payload compaction is optional optimization after
  float32 feature, reward, and discriminator parity is established and measured.
  It is not required for Phase 1 acceptance.

## Phase 1 Task Contract

- MuJoCo G1 flat ground only for first acceptance.
- One fixed nonzero forward objective.
- Nominal upright reset as initial condition only.
- No zero-command standing, walk-to-stop, command distribution, gait phase,
  contact schedule, running, recovery, motion reset, Motrix, multi-GPU, or
  multiple collectors.
- Existing `G1WalkFlat` and all distillation routes remain isolated and unchanged.

## Required Evidence

### Implementation gate

- deterministic forward-only manifest and AMP feature parity;
- typed AMP payload shape/order and terminal-transition identity;
- exact AMP reward and combined-reward value tests;
- frozen-`D_k` update-order trace;
- CPU-owned atomic checkpoint roundtrip;
- normal and injected-failure lifecycle cleanup.

### Integration gate

- one formal `uv run train --algo appo --task g1_amp_walk ...` route selected by
  Hydra/runtime resolver;
- spawned collector -> ring -> staging -> learner connectivity;
- reward enters V-trace exactly once;
- collector death reaches the parent without a 60-second blind wait;
- actor-only playback loads the produced checkpoint;
- default APPO, HORA, existing G1 task, and distillation routes remain unchanged.

### Performance observation gate

- persist a matched APPO-versus-AMP timing identity with collector, IPC, H2D,
  learner, staging occupancy, memory, and wall time;
- preserve bottleneck findings for later optimization;
- do not reject Phase 1 solely because AMP exceeds a fixed overhead percentage
  or adds roughly ten minutes of training time;
- block only when throughput symptoms break forward progress, bounded completion,
  memory capacity, or the async process/resource lifecycle contract.

### Live gate

- one frozen bounded command/config/data/checkpoint identity;
- finite reward/logit/loss/version diagnostics and clean lifecycle postflight;
- playback evidence of fixed-forward walking;
- result classified as runtime-pass/quality-pass, runtime-pass/quality-fail,
  lifecycle-fail, or capacity-fail.

Standing and walk-to-stop are not live acceptance items. A clean run or
checkpoint alone is not policy-quality evidence.

## Forbidden Behavior

- Do not copy or bypass `AsyncRunner` lifecycle and APPO IPC contracts.
- Do not add a collector-side or separate-process discriminator.
- Do not let `D_(k+1)` rescore the batch used for the actor/critic update.
- Do not hide or relabel measured overhead; retain it as diagnostic evidence.
- Do not optimize throughput by changing AMP method semantics or reward identity.
- Do not parse assets/XML or probe backend-private features in hot paths.
- Do not mutate the current distillation method in Phase 1.
- Do not begin Phase 2 without a new contract proposal and Concept Figure gate.

## Current Acceptance Status

Training semantics and ownership are active under `AMP-WALK-TRAIN-v003`.
Steps 1-6 verify the official spawned async route, typed AMP payload,
learner-only frozen-`D_k` order, atomic checkpoint/resume, actor-only playback,
and clean lifecycle. Step 7 measured 96.6% local-MPS end-to-end time overhead;
that observation remains valid but is accepted as non-blocking under the human
owner's revised priority.

The async route and lifecycle evidence from v002 remain accepted. Recovery
Steps 1-2 are complete. The fresh 20-iteration sentinel passed all three frozen
tail-five health thresholds with a clean lifecycle, although the policy-logit
median margin is only about 0.00027 and the final point regressed. Step 3 live
quality training remains blocked on separate human authorization; no
policy-quality claim follows from the short sentinel.
