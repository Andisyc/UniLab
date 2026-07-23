# AMP-Only Async Walking Control Room

Status: `Steps 1-7 complete`; measured AMP overhead is accepted under training
contract v002 and Step 8 awaits explicit authorization.

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

## Governance State

The AMP method now has its own active `AMP-WALK` method/training contracts and
Concept Figure. It remains separate from the active multi-teacher distillation
method. Planned implementation owners stay in the engineering plan until code
exists; no AMP Method-to-Code or Runtime Atlas is presented as implemented.

The current cursor is Step 1 / 8, pending explicit implementation authority.
The accepted document activation does not authorize code changes, tests,
simulator startup, checkpoint IO, or training.

No speedup or policy-quality claim is active. Step 8 will test useful walking
within one bounded run; a roughly ten-minute increase over the original
10-20-minute estimate is explicitly acceptable.
