# AMP-Only Async Walking Proposal

Status: `accepted-and-extracted`

Accepted: 2026-07-22

The human confirmed:

- Phase 1 trains one fixed nonzero forward-walking policy and does not require
  standing;
- the AMP discriminator is learner-only;
- for each staged batch, frozen `D_k` scores AMP reward before V-trace,
  actor/critic update next, and discriminator update produces `D_(k+1)` last;
- Phase 1 uses UniLab asynchronous APPO and excludes AMP/distillation
  integration.

The accepted semantics have been extracted into:

- `../contracts/active/method/AMP-WALK-METHOD-v001.md`
- `../contracts/active/training/AMP-WALK-TRAIN-v001.md`
- `../../architecture/concept/04_amp_walk_async_method.data.json`

The current implementation route and acceptance state are now owned by:

- `current_engineering_plan.md`
- `../checklists/current.md`

This file is no longer a current design or execution authority. New semantic
changes require a new replaceable proposal rather than editing the active
contracts silently.

