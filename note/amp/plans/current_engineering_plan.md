# AMP-WALK Phase 1 Symmetric Self-Collision Repair Plan

Status: `Engineering Steps 1-2 complete; paused before Step 3`

Date: 2026-07-23

Contracts:

- `AMP-WALK-METHOD-v003`
- `AMP-WALK-TRAIN-v004`

Concept Figure:

- `note/architecture/concept/04_amp_walk_async_method.data.json`

Plan cursor: Engineering Steps 1-2 implemented the source-parity signal and
passed the frozen 20-iteration async sentinel. Step 3 material GPU training and
matched playback evaluation remain a separate human authorization boundary.

## Terminal Outcome

Run one fresh fixed-forward AMP experiment through UniLab's official async APPO
route with the AMP_mjlab full-body symmetric self-collision cost. Preserve AMP
as the sole human-style owner, preserve the existing lifecycle repair, and
determine whether the missing physical-viability signal removes persistent
hand-leg contact without sacrificing forward locomotion or AMP health.

## Confirmed Semantic Boundary

- Whole-body selector: `pelvis` subtree against itself.
- Source parameters: four-entry force history, `10.0 N` threshold, `-0.1`
  reward scale, one sensor slot, and no contact-pair reduction.
- No right-hand-specific pose, distance, collision, or action constraint.
- No default pose, gait phase, standing, running, fall recovery, motion reset,
  discriminator tuning, AMP coefficient tuning, Motrix, or distillation.
- `model_2000.pt` is an old-policy comparison artifact, never a resume source.

## Owner Map

| Semantic object | Owner | Consumer | Required boundary |
| --- | --- | --- | --- |
| self-contact declaration | AMP-only task XML fragment | MuJoCo model materialization | task-level fragment; do not modify `g1.xml` |
| sensor-history capability | `SimBackend` public API | G1 AMP task | no env access to MuJoCo-private data |
| native force history | `MuJoCoBackend` | public backend API | cache sensor address at init; no XML parsing in `step()` |
| collision reducer | `G1AMPWalkEnv` | standard reward dispatch | exact source threshold/count semantics |
| reward coefficient | `conf/appo/task/g1_amp_walk/mujoco.yaml` | Hydra -> task reward | `self_collisions: -0.1` only for AMP task |
| task/style combination | existing AMP learner | V-trace/APPO | unchanged, exactly once, frozen `D_k` order |
| process lifecycle | existing AsyncRunner/APPO owners | parent/collector | preserve spawn, close order, and lifecycle report |

## Backend Feasibility Finding

The installed `mujoco-uni==3.8.0` contact sensor directly supports
`subtree1`, `subtree2`, force data, slot count, and reduction mode, so the
source full-body selector can be represented without enumerating geom pairs.
`BatchEnvPool.step(nstep=N)` returns only final-step `sensordata`, but its
FULLPHYSICS state contains MuJoCo's native sensor history. The implemented path
declares `nsample=4` in the AMP task sensor and reconstructs the native ring
oldest-to-newest after the unchanged single `nstep=N` call. A split-step draft
was rejected by a physical-trajectory differential test. Default tasks retain
their current single-call fast path and do not configure this sensor/history.

## Difficulty And Workload

Difficulty is medium-high, not because the reward formula is complex, but
because exact four-entry force history crosses the shared backend boundary and
must survive partial resets without slowing unrelated tasks. The task/reward
portion is small; backend history, regression coverage, and live evidence own
most of the work.

- Step 1 is the shared-backend risk boundary: roughly 3-5 focused engineering
  hours including deterministic and regression tests.
- Step 2 is the AMP-only connector and bounded-live boundary: roughly 2-4
  focused engineering hours plus the 20-iteration sentinel.
- Step 3 is primarily machine time: approximately the observed 34-minute
  training duration plus artifact transfer, matched evaluation, and human
  playback review. Allow about 1-2 hours of wall time without tuning.

## Step Map

### Step 1 / 3: Backend Sensor-History Contract

Objective: add the reusable backend capability without changing any task reward
or starting training.

Scope:

- add a public, backend-neutral sensor-history contract to `SimBackend` and a
  MuJoCo implementation with cold-path sensor lookup, four-entry history, and
  per-row reset clearing;
- keep the default backend fast path unchanged when history is not configured;
- write failing-first history ordering, sensor-address, forced-contact,
  partial-reset, disabled-fast-path, and backend-interface tests;
- measure the opt-in native sensor-history overhead with a minimal batched
  backend probe;
- persist backend evidence and update the Architecture map only if the public
  implemented interface changes the current runtime ownership map.

Core connector:

```text
configured contact sensor with `nsample=4`
-> MuJoCo FULLPHYSICS native history
-> backend-owned oldest-to-newest four-entry view
-> public SimBackend history value
```

Expected evidence:

- history has stable shape/order, clears only reset rows, and cannot leak an
  old episode collision into a new episode;
- default `BatchEnvPool.step(nstep=N)` behavior remains unchanged when history
  is disabled;
- forced contact produces nonzero force history through the public API;
- timing delta is recorded without imposing a new percentage gate.

Stop conditions:

- source-equivalent history cannot be produced without env-private backend
  access, XML parsing in the hot path, or bypassing the batched pool;
- history ordering or partial-reset isolation is unproven;
- sensor forces stay structurally zero under a deterministic forced-contact
  probe;
- default backend behavior or existing backend tests regress.

Status: `COMPLETE`; see
`evidence/2026-07-23-self-collision-steps1-2.md`.

### Step 2 / 3: AMP Integration And Short Sentinel

Objective: connect the verified backend capability to the isolated AMP task and
prove the official async route remains healthy.

Scope:

- add an AMP-only task fragment with the symmetric `pelvis`-subtree contact
  force sensor;
- add the exact source collision reducer and `self_collisions: -0.1` owner YAML;
- expose raw hit-count/rate diagnostics without adding a right-side objective;
- write failing-first reducer, XML/sensor, config-isolation, legacy G1,
  reward-connectivity, and async lifecycle tests;
- run focused tests, config compose, unchanged-path regressions, and one fresh
  20-iteration 2048-env official async sentinel;
- persist implementation, timing, lifecycle, and diagnostic evidence, then
  synchronize contracts/checklist/canvas/Architecture current state.

Core connector:

```text
AMP task fragment contact sensor
-> public backend four-entry history
-> G1AMPWalk source reducer (>10 N, count history hits)
-> self_collisions * -0.1 * ctrl_dt
-> existing task/style mixture
-> V-trace/APPO update
```

Expected evidence:

- deterministic reducer matches the AMP_mjlab oracle;
- composed `G1AMPWalk` config contains the term while legacy G1 configs do not;
- reward reaches task/V-trace exactly once;
- sentinel emits finite collision/AMP metrics, completes 20/20, and reports a
  clean collector lifecycle;
- timing delta is recorded against the existing matched async evidence.

Stop conditions:

- task fragment changes robot XML or non-AMP G1 behavior;
- reward reaches task/V-trace more than once;
- collision metrics are structurally zero in the live route;
- AMP health gate, capacity, forward progress, or lifecycle fails.

Status: `COMPLETE`; see
`evidence/2026-07-23-self-collision-steps1-2.md`.

### Step 3 / 3: Fresh 2000-Iteration GPU Acceptance

Objective: determine whether the source-parity repair resolves the observed
self-contact while preserving the already working async AMP walk.

Scope:

- freeze commit, composed config, motion manifest, command, seed set, and fresh
  initialization identity;
- run one 2000-iteration target-GPU training through the official detached
  async route, with no resume and no parameter sweep;
- collect console, TensorBoard event, checkpoint, hashes, and lifecycle report;
- evaluate old `model_2000.pt` and the repaired checkpoint under the same
  collision-enabled simulator and fixed-forward multi-seed playback manifest;
- compare mean and p95 count of history entries above `10.0 N`, fixed-forward
  tracking, termination/timeout behavior, and AMP health;
- return the paired replays to the human owner for the final visible
  hand-leg-contact judgment.

Acceptance:

- run completes 2000/2000 with clean lifecycle and finite diagnostics;
- tail AMP health retains `policy_logit_p50 > -0.95`,
  `policy_zero_style_fraction < 0.50`, and `style_reward_mean > 0.005`;
- repaired policy is lower than the old policy on both mean and p95 symmetric
  self-collision hit count under the matched evaluation manifest;
- fixed-forward locomotion remains functional;
- human playback review finds no persistent hand-leg catch.

Stop conditions:

- lifecycle/capacity failure, AMP collapse, or loss of fixed-forward walking;
- collision metrics fail to improve, or playback still shows persistent
  contact. In that case classify `runtime-pass / collision-quality-fail` and
  return to human design; do not add a hand-specific pose/distance term.

Status: `BLOCKED` pending separate human authorization of the material GPU run.

## Authority Boundary

Engineering Steps 1-2 were authorized together and are complete. Material Step
3 is not authorized. The next safe action is human review of the Step 1-2
evidence followed by explicit Step 3 authorization or a stop decision.
