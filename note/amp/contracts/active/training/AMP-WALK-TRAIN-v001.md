---
contract_id: AMP-WALK-TRAIN-v001
status: active
effective_date: 2026-07-22
updated_date: 2026-07-22
supersedes: null
scope: Phase 1 AMP on UniLab asynchronous APPO
method_contract: AMP-WALK-METHOD-v001
concept_figure: note/architecture/concept/04_amp_walk_async_method.data.json
implementation_status: planned
promotion_status: not_started
---

# AMP Walk Asynchronous Training Contract

## Decision

Phase 1 uses UniLab's existing spawned collector/learner APPO architecture. The
AMP discriminator remains learner-only. For each staged rollout batch, the
learner freezes `D_k`, scores every policy AMP transition exactly once, forms
the final reward before V-trace, updates actor/critic, then updates the
discriminator to `D_(k+1)`.

This active contract fixes training ownership and ordering. It does not claim
that the route is implemented, that asynchronous execution is faster, or that a
useful policy can be trained within 10-20 minutes.

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
- Float16 transport or payload compaction is allowed only after float32 feature,
  reward, and discriminator parity is established and measured.

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

### Performance gate

- unchanged APPO baseline on the target machine;
- matched AMP A/B timing with collector, IPC, H2D, learner, discriminator,
  staging occupancy, RSS, VRAM, and wall time;
- AMP overhead at or below 30%, or an explicit bottleneck and rejection of the
  10-20 minute projection.

### Live gate

- one frozen 10-20 minute command/config/data/checkpoint identity;
- finite reward/logit/loss/version diagnostics and clean lifecycle postflight;
- playback evidence of fixed-forward walking;
- result classified as runtime-pass/quality-pass, runtime-pass/quality-fail,
  lifecycle-fail, or performance-fail.

Standing and walk-to-stop are not live acceptance items. A clean run or
checkpoint alone is not policy-quality evidence.

## Forbidden Behavior

- Do not copy or bypass `AsyncRunner` lifecycle and APPO IPC contracts.
- Do not add a collector-side or separate-process discriminator.
- Do not let `D_(k+1)` rescore the batch used for the actor/critic update.
- Do not infer speedup from persistence/asynchrony without matched evidence.
- Do not parse assets/XML or probe backend-private features in hot paths.
- Do not mutate the current distillation method in Phase 1.
- Do not begin Phase 2 without a new contract proposal and Concept Figure gate.

## Current Acceptance Status

Training semantics and ownership are active. Implementation status is
`performance-fail`: Steps 1-6 include a spawned AMP route, full atomic
checkpoint/resume, actor-only playback, and clean lifecycle evidence. Step 7
persisted a matched local-MPS bottleneck verdict after measuring 96.6% overhead,
above the 30% gate. Bounded live-policy evidence has not been accepted.
