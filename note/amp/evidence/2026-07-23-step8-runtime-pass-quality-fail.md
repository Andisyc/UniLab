# Step 8 Live Acceptance Evidence

Date: 2026-07-23

Contracts: `AMP-WALK-METHOD-v001`, `AMP-WALK-TRAIN-v002`

Classification: `runtime-pass / quality-fail`

## Scope

One fixed-forward MuJoCo G1 AMP/APPO run through the official asynchronous
collector/learner route, followed by checkpoint, TensorBoard, playback, and
live AMP-observation audits. Standing, running, recovery, gait control,
Motrix, and distillation were not introduced.

## Raw Artifact Identity

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `console.log` | 6,227 | `45a075e9531a845f9a977a1fee05593a3f333c204f1a7a91a00a5b584688a93c` |
| `events.out.tfevents.1784790097.ubuntu-msi.226531.0` | 5,941,954 | `8f8f81dd4053b2fbe6bd08c7c69713aa5f954738c9d18e8af1f3d5aa0e7a120f` |
| `model_1850.pt` | 69,036,811 | `ed883fc955e4cf4837a481a0a8dd892b25e7141c2a9f8c3981991f84f6daf4cc` |

The event file contains the resumed iterations from discriminator version 151
through 2000. The checkpoint records discriminator version 2000.

## Runtime Facts

- The resumed run completed 1850/1850 iterations and 90,931,200 resumed
  environment steps in about 30 minutes 15 seconds.
- Collector exit code was zero, the lifecycle report was complete, and the
  error list was empty.
- The checkpoint loaded with `weights_only=True`; actor, critic, optimizer,
  discriminator, discriminator optimizer, normalizer, replay, generator state,
  and discriminator version were present and finite.
- Tail-100 timeout rate was 0.9646 and terminated rate was 0.0354. The learned
  actor therefore achieved sustained fixed-forward locomotion.

## Policy-Quality Facts

TensorBoard tail-100 means:

| Metric | Tail-100 mean |
| --- | ---: |
| `amp/policy_logit_mean` | -0.99662 |
| `amp/expert_logit_mean` | 0.99646 |
| `amp/style_reward_mean` | 0.001418 |
| `amp/task_reward_mean` | 0.046224 |
| `amp/combined_reward_mean` | 0.035022 |
| `reward/tracking_lin_vel` | 1.81169 |

Under the active `0.75 * task + 0.25 * style` combination, the tail means
contribute about 0.03467 task reward and 0.00035 style reward. AMP style is
therefore only about one percent of the final mean reward at the end of this
run.

Offline replay/expert scoring with the checkpoint produced:

- expert logit median `0.990870`;
- policy logit median `-1.006128`;
- policy style-reward median `0.0` and p90 `0.003920`;
- standardized expert/policy mean gaps: position `0.5150`, rotation-6D
  `0.8418`, linear velocity `0.5503`, angular velocity `0.0192`.

The discriminator cleanly separates the two distributions, and most policy
transitions lie on the zero-reward plateau. The AMP signal is not an effective
style-learning authority in this run.

## Gait And Observation Exclusions

- The composed AMP task has no nonzero `feet_phase*` reward scale.
- `G1AMPWalkEnv` removes the two gait-phase dimensions from actor and critic
  observations and rejects nonzero gait-phase reward terms.
- The gait-constraint bridge is disabled by configuration.
- A live one-environment MuJoCo oracle compared the AMP sensor route with
  independent `xquat`/`xmat` values. Quaternion sign-invariant maximum error
  was `2.57e-8`, rotation-6D maximum error was `5.96e-8`, and reconstructed AMP
  observation maximum error was `1.19e-7`.
- Three focused config/observation contract tests passed. The slow pytest live
  case was deselected by the repository marker policy; the independent live
  oracle reached the real MuJoCo boundary instead.

These facts reject active gait-phase authority and quaternion/frame-order
corruption as explanations for the observed playback.

## First Invalid Authority Boundary

The active AMP task config retains the inherited default-pose reward and large
upper-body pose weights, while the AMP style reward collapses near zero. The
policy can therefore satisfy the fixed-forward task through the inherited task
shape without matching the expert walking style. The source AMP task does not
use a default-joint-pose reward.

The current expert manifest contains two clips and 935 adjacent transitions.
This is source-transform-correct but does not establish sufficient expert
distribution support. Repeating the same transitions cannot be claimed as
additional diversity.

## Decision

Step 8 closes as `runtime-pass / quality-fail`. The asynchronous migration and
lifecycle route are valid; Phase 1 human-like AMP walking is not accepted.
Further training from `model_1850.pt` under the same reward/data identity is
blocked because it would preserve the failed style-authority mechanism.

Next: Recovery Step 2 in `../plans/current_engineering_plan.md` under
`AMP-WALK-METHOD-v002` and `AMP-WALK-TRAIN-v003`.
