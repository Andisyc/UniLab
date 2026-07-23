# AMP-WALK Contract Registry

This registry is the only default contract entrypoint for Phase 1 AMP walking.

## Active

| Contract | Status | Scope | Supersedes |
| --- | --- | --- | --- |
| [AMP-WALK-METHOD-v001](active/method/AMP-WALK-METHOD-v001.md) | active | Fixed-forward walk-only AMP method semantics | none |
| [AMP-WALK-TRAIN-v002](active/training/AMP-WALK-TRAIN-v002.md) | active | Learner-only discriminator on UniLab asynchronous APPO; measured overhead is diagnostic | AMP-WALK-TRAIN-v001 |

## History

| Contract | Status | Reason |
| --- | --- | --- |
| [AMP-WALK-TRAIN-v001](history/training/AMP-WALK-TRAIN-v001.md) | superseded | Replaced fixed 30% overhead gate with async-integration-first acceptance |

## Recall Rule

Read only the active contract required by the task. Do not infer Phase 2
AMP/distillation integration from these Phase 1 contracts. Future semantic
changes belong in a proposal first and require a new contract version.
