# AMP Performance Gate Acceptance Revision

Date: 2026-07-23

Decision owner: human project owner

Supersedes acceptance boundary: `AMP-WALK-TRAIN-v001`

Activates: `AMP-WALK-TRAIN-v002`

## Human Decision

The Phase 1 priority is successful integration with UniLab's high-speed
asynchronous architecture. The human owner explicitly accepts waiting roughly
ten additional minutes for AMP training and does not require the previous fixed
30% overhead ceiling.

## Evidence Interpretation

The Step 7 matched measurement remains unchanged: the local MPS run observed
96.6% reconstructed end-to-end time overhead and identified the broad MuJoCo
tracking-sensor route plus AMP learner work as bottleneck owners. It remains a
valid performance observation and optimization backlog item.

The observation no longer means that Phase 1 migration failed. Under v002,
performance blocks only if it prevents forward progress, bounded completion,
memory capacity, or clean lifecycle. The three Step 7 runs completed normally,
so Step 7 is accepted and Step 8 is ready for separate explicit authorization.

## Unchanged Boundaries

- No method, reward, observation, discriminator-order, or runtime route changed.
- No standing, running, recovery, gait control, Motrix, or Phase 2 distillation
  scope was added.
- No training or policy-quality claim was produced by this decision.
