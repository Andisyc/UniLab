# AMP-Only Async Walking Control Room

Status: symmetric self-collision Engineering Steps 1-2 are
`runtime-pass / style-health-pass / collision-connectivity-pass`; work is
paused before the separately authorized material Step 3 / 3.

This directory governs the proposed migration of walk-only AMP training from
`/Users/chengyuxuan/ArtiIntComVis/AMP_mjlab` into UniLab's asynchronous APPO
runtime. It is separate from `note/distillation/`; AMP is not a distillation
role or a hidden extension of the active multi-teacher method.

## Confirmed Human Scope

- migrate human-like **walking** AMP only;
- use UniLab's asynchronous collector/learner architecture;
- make Phase 1 a migration-correctness sentinel with one fixed nonzero forward
  walking objective; standing still is not a required skill or acceptance item;
- exclude running, fall recovery, delayed termination, recovery reset, and
  motion-reset curriculum;
- exclude explicit gait-phase observation, gait-phase reward, contact schedule,
  and other gait-controller ownership;
- retain only the minimum task reward needed to specify commanded locomotion and
  physical viability;
- include AMP_mjlab's full-body symmetric self-collision viability signal with
  source parameters, without any right-hand-specific pose or distance term;
- train a deployable actor; the discriminator remains training-only.
- defer AMP plus the current multi-teacher distillation workflow to Phase 2;
  that integration requires a separate design/contract pass and must not enter
  the Phase 1 runtime.

## Default Read Path

1. [Active contract registry](contracts/README.md)
2. [AMP Walk Concept Figure](../architecture/concept/04_amp_walk_async_method.data.json)
3. [Current engineering plan](plans/current_engineering_plan.md)
4. [Current acceptance checklist](checklists/current.md)
5. [Static migration evidence](evidence/2026-07-21-static-migration-baseline.md)
6. [Distillation lifecycle impact review](evidence/2026-07-22-distill-lifecycle-impact-review.md)
7. [Current task canvas](task_canvas.md)
8. [Step 8 quality-failure evidence](evidence/2026-07-23-step8-runtime-pass-quality-fail.md)
9. [Recovery Step 2 evidence](evidence/2026-07-23-recovery-step2-style-authority.md)
10. [Fresh baseline and self-collision gap evidence](evidence/2026-07-23-step3-baseline-self-collision-gap.md)
11. [Self-collision Steps 1-2 evidence](evidence/2026-07-23-self-collision-steps1-2.md)
12. [Current repair engineering plan](plans/current_engineering_plan.md)

## Governance State

The AMP method now has its own active `AMP-WALK` method/training contracts and
Concept Figure. It remains separate from the active multi-teacher distillation
method. Planned implementation owners stay in the engineering plan until code
exists; no AMP Method-to-Code or Runtime Atlas is presented as implemented.

The original eight-step migration plan reached Step 8 and closed as
`runtime-pass / quality-fail`. The async runtime, lifecycle, checkpoint, and
fixed-forward locomotion routes passed; human-like AMP style did not. Recovery
Step 1 activated `AMP-WALK-METHOD-v002` and `AMP-WALK-TRAIN-v003`. Recovery
Step 2 removed default-pose authority, added support/style diagnostics, and
passed a fresh 20-iteration official async sentinel. The subsequently
authorized 2000-iteration run completed with a clean lifecycle and healthy AMP
tail metrics. Playback reached the source author's approximate style ceiling
but exposed persistent hand-leg contact.

The human owner confirmed the AMP_mjlab whole-body symmetric self-collision
term as a missing minimum-viability signal. This activated
`AMP-WALK-METHOD-v003` and `AMP-WALK-TRAIN-v004` without adding a Concept Figure
block or reopening gait/style authority. Engineering Steps 1-2 now pass their
backend, isolation, connectivity, AMP-health, and lifecycle gates. Material
Step 3 remains unauthorized.

No repaired-policy quality claim is active. `model_1850.pt` and `model_2000.pt`
are retained as evidence/playback identities and must not seed the proposed
repair run.
