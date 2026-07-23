---
contract_id: AMP-WALK-METHOD-v003
status: active
effective_date: 2026-07-23
updated_date: 2026-07-23
supersedes: AMP-WALK-METHOD-v002
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
| AMP-WALK-DP-01 | Walk Expert Transitions | AW-M-01 | [Walk Expert Transitions](#walk-expert-transitions) | Two source forward clips and 935 unique adjacent transitions verified; draw count is reported separately from support |
| AMP-WALK-DP-02 | Policy Walk Transitions | AW-M-02 | [Policy Walk Transitions](#policy-walk-transitions) | Implemented through env, terminal owner, spawned IPC, and staging |
| AMP-WALK-DP-03 | AMP Style Discriminator | AW-M-03 | [AMP Style Discriminator](#amp-style-discriminator) | Learner-only D/normalizer/replay and 2000-iteration non-collapse evidence verified |
| AMP-WALK-DP-04 | AMP-Regularized Walking Policy | AW-M-04 | [AMP-Regularized Walking Policy](#amp-regularized-walking-policy) | Source-parity full-body self-collision signal confirmed; implementation pending |

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
  physical-viability regularization. It may stabilize vertical/base motion,
  orientation, angular velocity, action rate, termination, and survival, but it
  must not penalize deviation from default joint posture. It contains no
  standing, stop-transition, gait-phase, contact-schedule, running, or recovery
  objective.
- Self-collision cost: the task owner symmetrically observes contact force
  between the whole G1 subtree rooted at `pelvis` and itself. It counts entries
  in a four-physics-substep history for which any contact force magnitude is
  greater than `10.0 N`, then applies source weight `-0.1` through the standard
  task-reward dispatch. This is a minimum-physical-viability signal, not a
  posture target.
- AMP style reward:
  `r_amp = amp_coef * clamp(1 - 0.25 * (D_k(x) - 1)^2, min=0)`.
- Combined reward: one Hydra-owned nonnegative combination of `r_task` and
  `r_amp`. The two components and final reward must be logged separately, and
  no fallback calculation may bypass this owner.

The self-collision threshold, history length, and weight above are frozen
source-parity semantics for this repair. Other numeric coefficients are
configuration choices inside this contract. A
nonzero default-joint-pose reward is forbidden because it gives the task owner
direct authority over walking posture. Changing the scoring object, adding a
new task objective, or changing the reward owner is a semantic change and
requires a new proposal/version.

## Walk Expert Transitions

- `design_id`: `AMP-WALK-DP-01`
- `block_id`: `AW-M-01`
- Meaning: forward-walking motion clips define the positive transition
  distribution for human-like style.
- Inputs: an explicit fail-closed manifest over AMP_mjlab-compatible motion
  artifacts and the accepted 13-body ordering.
- Output: sampled expert `(s_t, s_t+1)` pairs at 195-D per state, with source,
  clip, frame, FPS, and body-order identity.
- Current support identity: exactly two source forward-walk clips, 937 frames,
  and 935 unique adjacent transitions. Sampling with replacement changes draw
  count only; it must not be reported as new diversity.
- Assumptions: source and UniLab G1 topology/body order remain compatible; the
  transform is deterministic and cold-path asset access is sufficient.
- Ownership boundary: the expert dataset/spec owner classifies clips and builds
  AMP states; the environment and learner may consume but not reinterpret the
  manifest.
- Interaction: expert transitions provide the positive samples consumed by
  `AMP-WALK-DP-03`.
- Forbidden: recursive directory admission, backward/sideways/arc/jog/idle/
  recovery clips, inferred labels from filenames without manifest validation,
  hot-path asset/XML parsing, or hiding limited support behind a preload count.
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
  one-score-per-batch version trace, finite update, checkpoint roundtrip,
  expert clip/unique-transition counts, policy logit quantiles, zero-style-
  reward fraction, and weighted task/style reward contributions.

## AMP-Regularized Walking Policy

- `design_id`: `AMP-WALK-DP-04`
- `block_id`: `AW-M-04`
- Meaning: APPO learns a deployable actor from the fixed-forward task signal and
  the learner-owned AMP style reward.
- Inputs: actor/critic rollout payload, behavior log probability, task reward,
  done/truncated/final state, symmetric full-body self-collision cost, and
  `D_k` style reward.
- Output: updated actor/critic weights and an actor-only deployment artifact.
- Assumptions: V-trace consumes the final combined reward; asynchronous behavior
  policy staleness remains owned by APPO.
- Style authority invariant: task reward may demand fixed-forward motion and
  minimum physical viability, including self-collision avoidance, but AMP is
  the sole owner of human-like joint posture and transition style. No body-side,
  hand, joint-pose, or target-distance exception is allowed.
- Ownership boundary: the AMP learner prepares the final reward and update
  order; generic APPO owns V-trace/actor/critic mechanics; the collector only
  receives actor/critic weights.
- Interaction: consumes `AMP-WALK-DP-03` reward and generates the next
  `AMP-WALK-DP-02` policy transition distribution.
- Required order: freeze `D_k` for the staged batch, score every transition once,
  form the combined reward, run V-trace/APPO actor/critic update, update the
  discriminator to `D_(k+1)`, then publish actor/critic weights.
- Forbidden: default-joint-pose reward, right-hand-only pose/distance/contact
  reward, zero-command standing acceptance, gait
  ownership, synchronous PPO fallback, discriminator deployment, checkpoint
  resume from the v001 quality-fail run, or claiming policy quality from finite
  losses/checkpoint creation alone.
- Required evidence: source-equivalent collision reducer values, backend-owned
  four-substep history and reset isolation, reward-to-V-trace consumer
  connectivity, update-order trace, actor/critic and discriminator version
  identity, formal async smoke, and bounded walking playback.

## Phase Boundary

Phase 1 ends with a bounded fixed-forward AMP walking result and lifecycle/
performance evidence. AMP plus the current distillation/DAgger architecture is
Phase 2. It must begin with a new proposal and must not silently modify this
contract, the distillation contracts, or the Phase 1 runtime.

## Current Acceptance Status

This version is active and human-confirmed. It supersedes v002 after the fresh
2000-iteration run completed with a clean async lifecycle and a non-collapsed
AMP signal, while playback exposed persistent hand-leg self-contact. The source
repository uses a whole-body symmetric self-collision cost with `-0.1` weight,
`10.0 N` threshold, and four-entry force history. UniLab now implements and
connects that signal through the public backend and AMP task owners; the frozen
short sentinel passes. The completed `model_2000.pt` and new `model_20.pt`
remain evidence only and must not seed the material repaired run.

The source AMP task resets environments from motion frames. Phase 1 continues
to forbid motion reset, motion-reset curriculum, recovery reset, and delayed
termination. This is an explicit experimental boundary rather than an assumed
source-parity fact: if the fresh short sentinel still collapses without motion
reset, Step 2 stops and the no-RSI method boundary returns to human decision.
