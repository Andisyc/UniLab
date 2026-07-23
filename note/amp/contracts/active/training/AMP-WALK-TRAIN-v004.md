---
contract_id: AMP-WALK-TRAIN-v004
status: active
effective_date: 2026-07-23
updated_date: 2026-07-23
supersedes: AMP-WALK-TRAIN-v003
scope: Phase 1 AMP on UniLab asynchronous APPO with diagnostic-only throughput overhead
method_contract: AMP-WALK-METHOD-v003
concept_figure: note/architecture/concept/04_amp_walk_async_method.data.json
implementation_status: engineering_steps_1_2_complete
promotion_status: awaiting_step_3_authorization
---

# AMP Walk Asynchronous Training Contract

## Quality-Run Decision

The fresh 2000-iteration target-GPU run completed 98,304,000 environment steps
in 33m41s with a clean collector lifecycle. Its final-100 AMP diagnostics remain
non-collapsed, and playback shows arm swing. The remaining visible defect is
persistent hand-leg self-contact. The human owner reports that the source
author's video has a similar overall style ceiling, so this repair does not
reopen the AMP representation or add a new pose owner.

The source G1 AMP task includes a symmetric whole-body self-collision cost that
is absent from UniLab. The human owner confirms that source signal as the only
new Phase 1 objective. `model_2000.pt` is a baseline/evaluation artifact only;
the repaired run must start from random actor/critic/discriminator state.

## Self-Collision Contract

- Sensor selector: `pelvis` subtree against the same `pelvis` subtree.
- Sensor payload: contact force, no pair reduction, one slot, four-entry physics
  history.
- Reducer: for each history entry, mark a hit when any contact force magnitude
  is greater than `10.0 N`; sum hits per environment.
- Reward scale: `self_collisions: -0.1`, then the existing standard task-reward
  `ctrl_dt` scaling and AMP task/style mixture apply exactly once.
- Symmetry: all eligible full-body contacts use one shared rule. No right-hand,
  hand-leg distance, joint-pose, or side-specific term is allowed.
- Ownership: the AMP task fragment declares the sensor; `SimBackend` declares
  the public history interface; MuJoCo owns native sensor collection and reset
  isolation; the G1 AMP task owns the reducer and Hydra scale.
- Isolation: the sensor and reward are enabled only for `G1AMPWalk`; existing
  G1 tasks, Motrix routes, and distillation routes remain unchanged.

The current `BatchEnvPool.step(nstep=N)` exposes only final-step sensor data.
Exact history therefore requires a bounded backend-owned substep collection
path when sensor history is explicitly configured. It must keep the batched
pool and official async collector/learner lifecycle; it may not read private
MuJoCo data from the env or parse XML in `step()`.

## Inherited Style-Health Gate

The repaired short sentinel and material run retain the v003 non-collapse
invariants: all diagnostics are finite and the relevant tail mean satisfies
`amp/policy_logit_p50 > -0.95`, `amp/policy_zero_style_fraction < 0.50`, and
`amp/style_reward_mean > 0.005`. These protect AMP authority but do not prove
that self-contact improved.

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
- Source-parity symmetric full-body self-collision cost exactly as specified
  above.
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
- exact self-collision reducer, four-entry history, and partial-reset isolation;
- frozen-`D_k` update-order trace;
- CPU-owned atomic checkpoint roundtrip;
- normal and injected-failure lifecycle cleanup.

### Integration gate

- one formal `uv run train --algo appo --task g1_amp_walk ...` route selected by
  Hydra/runtime resolver;
- spawned collector -> ring -> staging -> learner connectivity;
- task-fragment sensor -> public backend history -> task reward connectivity;
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
- finite self-collision diagnostics and matched old/new collision evaluation;
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
- Do not replace the full-body source signal with a right-hand-only pose,
  distance, or collision rule.
- Do not mutate the current distillation method in Phase 1.
- Do not begin Phase 2 without a new contract proposal and Concept Figure gate.

## Current Acceptance Status

Training semantics and ownership are active under `AMP-WALK-TRAIN-v004`.
The official spawned async route, typed AMP payload, learner-only frozen-`D_k`
order, checkpoint/playback path, and clean lifecycle remain accepted. The fresh
2000-iteration v003 run is now classified `runtime-pass / style-health-pass /
self-collision-gap`: it is baseline evidence, not a resumable checkpoint.

Engineering Steps 1-2 are complete: the public native sensor-history contract,
source-parity reward, raw collision diagnostics, and official 20-iteration
sentinel pass their frozen gates. See
`evidence/2026-07-23-self-collision-steps1-2.md`. A new material 2000-iteration
run remains a separate Step 3 authorization boundary and must start fresh.
