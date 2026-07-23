# Fresh 2000-Iteration Baseline And Self-Collision Gap

Date: 2026-07-23

Classification: `runtime-pass / style-health-pass / self-collision-gap`

Active successor contracts:

- `AMP-WALK-METHOD-v003`
- `AMP-WALK-TRAIN-v004`

## Artifact Identity

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `console.log` | 7,308 bytes | `73c04cee6d20d43400a7d6584f586e32c7e5fc27a217e53a2f486857bdbf7552` |
| `events.out.tfevents.1784800252.ubuntu-msi.227607.0` | 7,469,568 bytes | `300df5921375edd2437597d5c69b89216fcbc9f7533b4c7b28e2fc650cd5bd25` |
| `model_2000.pt` | 69,036,811 bytes | `f4c4d3b4c62419cd51fa6b790ed132427bd9d9209b4aa7eb13f6320f4a75f0a8` |

The console identifies `logs/amp_step3_fresh_20260723_175050`, APPO,
`G1AMPWalk`, 2000/2000 iterations, 98,304,000 environment steps, and 33m41s
wall time. The lifecycle report is:

```json
{"collector_exitcode": 0, "collector_terminated": false, "errors": [], "resource_count": 4, "state": "complete"}
```

Evidence class: S4 / T-live,T-persist,T-order.

## TensorBoard Tail

All listed series contain 2000 points.

| Metric | Last | Final-100 mean |
| --- | ---: | ---: |
| `amp/policy_logit_p50` | -0.902429 | -0.885149 |
| `amp/policy_zero_style_fraction` | 0.240112 | 0.220438 |
| `amp/style_reward_mean` | 0.013504 | 0.014698 |
| `amp/task_reward_mean` | 0.052028 | 0.052455 |
| `amp/task_weighted_mean` | 0.039021 | 0.039341 |
| `amp/style_weighted_mean` | 0.003376 | 0.003675 |
| `reward/tracking_lin_vel` | 1.774911 | 1.780735 |

The frozen non-collapse limits remain satisfied over the final 100 points. This
supports AMP signal health and fixed-forward learning; it does not prove
human-like quality or collision avoidance.

Evidence class: S4 / T-value,T-live.

## Human Playback Observation

The human owner observed arm swing and fixed-forward walking. The right hand
can remain too close to the right leg and become caught during playback. The
human owner also compared the result with the source author's video and judged
the overall style to be similar to the source result. Therefore the next repair
targets the missing source physical-viability signal, not a new style target.

Evidence class: S4 / human visual review.

## Source-Parity Gap

AMP_mjlab declares a `self_collision` contact sensor over the full `pelvis`
subtree against itself, with force data, no reduction, one slot, and history
length 4. Its `self_collisions` reward has weight `-0.1` and force threshold
`10.0`. The reducer computes force magnitude, marks each history entry where
any contact exceeds the threshold, and sums the hits.

Local source pointers:

- `/Users/chengyuxuan/ArtiIntComVis/AMP_mjlab/src/tasks/amp_loco/config/g1/env_cfgs.py`
- `/Users/chengyuxuan/ArtiIntComVis/AMP_mjlab/src/tasks/amp_loco/mdp/rewards.py`

UniLab's `conf/appo/task/g1_amp_walk/mujoco.yaml` has no self-collision term.
The active AMP task forbids pose and gait-phase reward authority, so the
source-parity collision cost fits the existing minimum-physical-viability owner
without changing AMP's human-style authority.

Evidence class: S0-S1 / T-contract,T-diff.

## Geometry Proxy

A deterministic capsule-segment proxy over the two accepted expert clips and
the source/UniLab G1 collision geometry found same-side hand-thigh overlaps in
the reference motion:

| Clip | Right hand/right thigh overlap frames | Minimum proxy clearance |
| --- | ---: | ---: |
| `A022` | 40.22% | -0.03395 m |
| `A024` | 41.49% | -0.03842 m |

The corresponding left-side rates were 22.86% and 15.35%; cross-side pairs did
not overlap in this proxy. These values are geometry evidence only, not MuJoCo
contact-force measurements. They explain why AMP can favor close hand-leg poses
and why a separate physical collision cost is needed.

Evidence class: S1 / T-proxy,T-value.

## Backend Feasibility And Risk

The installed `mujoco-uni==3.8.0` XML schema supports contact sensors with
`subtree1`, `subtree2`, `data`, `num`, and `reduce`, so the full-body selector
does not require hand-specific pair enumeration. The current
`BatchEnvPool.step(nstep=N)` returns final-step sensor data only. Reproducing the
source four-entry history therefore requires an opt-in backend-owned substep
history path and reset-row clearing.

This is the main implementation risk. It must remain inside `SimBackend` and
`MuJoCoBackend`, retain batched simulation and the official async lifecycle,
and leave the default fast path unchanged when history is disabled.

Evidence class: S1 / T-contract,T-shape,T-performance-risk.

## Decision

Migrate the source full-body symmetric self-collision reward with its exact
parameters. Do not add right-hand-specific pose, distance, contact, or action
constraints. Do not resume `model_2000.pt`. The next authorized boundary, if
granted, is Engineering Step 1 / 3 only.
