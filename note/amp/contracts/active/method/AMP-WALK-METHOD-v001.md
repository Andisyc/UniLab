---
contract_id: AMP-WALK-METHOD-v001
status: active
effective_date: 2026-07-22
updated_date: 2026-07-22
supersedes: null
scope: Phase 1 fixed-forward walk-only AMP on MuJoCo G1
concept_figure: note/architecture/concept/04_amp_walk_async_method.data.json
---

# AMP Walk-Only Method Contract

## Problem

UniLab must reproduce the human-like walking signal from AMP_mjlab without
importing its legacy synchronous runner or expanding the first milestone into a
general locomotion controller. Phase 1 is a migration-correctness experiment:
learn one fixed nonzero forward-walking policy through UniLab's asynchronous
APPO architecture.

Standing still, walk-to-stop, command diversity, gait control, running, and fall
recovery are not part of the learned behavior or acceptance boundary. A nominal
upright reset is only an initial condition.

## Design Point Register

| design_id | Canonical name | block_id | Contract section | Current code/evidence gap |
| --- | --- | --- | --- | --- |
| AMP-WALK-DP-01 | Walk Expert Transitions | AW-M-01 | [Walk Expert Transitions](#walk-expert-transitions) | Implemented and source-parity verified; policy-quality diversity remains untested |
| AMP-WALK-DP-02 | Policy Walk Transitions | AW-M-02 | [Policy Walk Transitions](#policy-walk-transitions) | Implemented through env, terminal owner, spawned IPC, and staging |
| AMP-WALK-DP-03 | AMP Style Discriminator | AW-M-03 | [AMP Style Discriminator](#amp-style-discriminator) | Learner-only D/normalizer/replay implemented and checkpointed |
| AMP-WALK-DP-04 | AMP-Regularized Walking Policy | AW-M-04 | [AMP-Regularized Walking Policy](#amp-regularized-walking-policy) | Formal reward-to-V-trace route passes; performance and live quality remain pending |

## Method Closure

```text
forward-walk expert transitions --------------------+
                                                    |
policy AMP transitions -> AMP Style Discriminator -> D_k style reward
policy task transition -> fixed-forward task reward -> combined reward
                                                    |
                                                    v
                                               V-trace/APPO
                                                    |
                                                    v
                                      fixed-forward walking actor
                                                    |
                                                    +-> next policy transitions
```

## Observation, Action, And Training Signal

- Actor observation: the isolated G1 AMP task's actor observation with one fixed
  nonzero forward command and no gait-phase input.
- Critic observation: the isolated task's declared critic group. It must not
  cause AMP state to be concatenated into actor input implicitly.
- AMP observation: one 195-D state from 13 tracked bodies. Each body contributes
  root-relative position 3, root-relative orientation representation 6, local
  linear velocity 3, and local angular velocity 3.
- AMP transition: `(amp_state_t, amp_state_t_plus_1)`, 390 floats before any
  transport optimization.
- Action: the existing 29-D G1 policy action through the normal environment and
  backend control contract.
- Task reward: fixed-forward velocity tracking plus explicitly configured basic
  physical-viability regularization. It contains no standing, stop-transition,
  gait-phase, contact-schedule, running, or recovery objective.
- AMP style reward:
  `r_amp = amp_coef * clamp(1 - 0.25 * (D_k(x) - 1)^2, min=0)`.
- Combined reward: one Hydra-owned nonnegative combination of `r_task` and
  `r_amp`. The two components and final reward must be logged separately, and
  no fallback calculation may bypass this owner.

Numeric coefficients are configuration choices inside this contract. Changing
the scoring object, adding a new task objective, or changing the reward owner is
a semantic change and requires a new proposal/version.

## Walk Expert Transitions

- `design_id`: `AMP-WALK-DP-01`
- `block_id`: `AW-M-01`
- Meaning: forward-walking motion clips define the positive transition
  distribution for human-like style.
- Inputs: an explicit fail-closed manifest over AMP_mjlab-compatible motion
  artifacts and the accepted 13-body ordering.
- Output: sampled expert `(s_t, s_t+1)` pairs at 195-D per state, with source,
  clip, frame, FPS, and body-order identity.
- Assumptions: source and UniLab G1 topology/body order remain compatible; the
  transform is deterministic and cold-path asset access is sufficient.
- Ownership boundary: the expert dataset/spec owner classifies clips and builds
  AMP states; the environment and learner may consume but not reinterpret the
  manifest.
- Interaction: expert transitions provide the positive samples consumed by
  `AMP-WALK-DP-03`.
- Forbidden: recursive directory admission, backward/sideways/arc/jog/idle/
  recovery clips, inferred labels from filenames without manifest validation,
  or hot-path asset/XML parsing.
- Required evidence: manifest oracle, shape/value parity against AMP_mjlab, and
  deterministic transition sampling.

## Policy Walk Transitions

- `design_id`: `AMP-WALK-DP-02`
- `block_id`: `AW-M-02`
- Meaning: policy-generated body-state transitions are the negative/current
  distribution compared with expert walking.
- Inputs: the isolated G1 AMP environment state before and after one policy
  action, including the exact terminal/final observation on done rows.
- Output: policy `(s_t, s_t+1)` pairs matching the expert 195-D/195-D feature
  semantics, plus transition/done identity required by the learner.
- Assumptions: tracked-body IDs and sensors are resolved and cached during
  initialization; partial resets preserve row identity.
- Ownership boundary: the environment owns AMP state construction; the
  collector owns ordered transition capture and transport, not style scoring.
- Interaction: policy transitions feed `AMP-WALK-DP-03`; the updated policy from
  `AMP-WALK-DP-04` produces the next distribution.
- Forbidden: gait-phase/contact features, using post-reset observations as
  terminal state, collector-side discriminator inference, or mutating existing
  `G1WalkFlat` semantics.
- Required evidence: feature parity, reset/terminal ordering tests, and spawned
  collector-to-learner transition identity.

## AMP Style Discriminator

- `design_id`: `AMP-WALK-DP-03`
- `block_id`: `AW-M-03`
- Meaning: distinguish expert forward-walk transitions from policy transitions
  and convert the frozen discriminator output into the AMP style reward.
- Inputs: normalized expert transitions and learner-owned policy replay samples.
- Output: discriminator logits/loss, versioned `D_k`, and one `r_amp` value for
  each policy transition in a staged batch.
- Assumptions: the discriminator is unconditional; no command, gait phase, or
  contact label is available in the expert data.
- Ownership boundary: discriminator, normalizer, expert sampler, replay, and
  optimizer are learner-local and training-only.
- Interaction: receives both transition distributions and supplies the frozen
  `D_k` reward consumed by `AMP-WALK-DP-04`.
- Forbidden: a separate discriminator process, collector weight sync/inference,
  deployment dependency, scoring one staged batch with multiple discriminator
  versions, or scoring with `D_(k+1)` before the policy update.
- Required evidence: deterministic reward formula, expert/policy loss,
  one-score-per-batch version trace, finite update, and checkpoint roundtrip.

## AMP-Regularized Walking Policy

- `design_id`: `AMP-WALK-DP-04`
- `block_id`: `AW-M-04`
- Meaning: APPO learns a deployable actor from the fixed-forward task signal and
  the learner-owned AMP style reward.
- Inputs: actor/critic rollout payload, behavior log probability, task reward,
  done/truncated/final state, and `D_k` style reward.
- Output: updated actor/critic weights and an actor-only deployment artifact.
- Assumptions: V-trace consumes the final combined reward; asynchronous behavior
  policy staleness remains owned by APPO.
- Ownership boundary: the AMP learner prepares the final reward and update
  order; generic APPO owns V-trace/actor/critic mechanics; the collector only
  receives actor/critic weights.
- Interaction: consumes `AMP-WALK-DP-03` reward and generates the next
  `AMP-WALK-DP-02` policy transition distribution.
- Required order: freeze `D_k` for the staged batch, score every transition once,
  form the combined reward, run V-trace/APPO actor/critic update, update the
  discriminator to `D_(k+1)`, then publish actor/critic weights.
- Forbidden: zero-command standing acceptance, gait ownership, synchronous PPO
  fallback, discriminator deployment, or claiming policy quality from finite
  losses/checkpoint creation alone.
- Required evidence: reward-to-V-trace consumer connectivity, update-order
  trace, actor/critic and discriminator version identity, formal async smoke,
  and bounded walking playback.

## Phase Boundary

Phase 1 ends with a bounded fixed-forward AMP walking result and lifecycle/
performance evidence. AMP plus the current distillation/DAgger architecture is
Phase 2. It must begin with a new proposal and must not silently modify this
contract, the distillation contracts, or the Phase 1 runtime.

## Current Acceptance Status

Method semantics are active and human-confirmed. Steps 1-6 implementation and
formal async integration are accepted by the current checklist. Step 7 produced
a matched local-MPS overhead measurement and owner-level bottleneck verdict
without changing method semantics. `AMP-WALK-TRAIN-v002` accepts the additional
training time as non-blocking. Bounded live-policy quality remains pending in
Step 8.
