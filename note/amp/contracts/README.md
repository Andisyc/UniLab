# AMP-WALK Contract Registry

This registry is the only default contract entrypoint for Phase 1 AMP walking.

## Active

| Contract | Status | Scope | Supersedes |
| --- | --- | --- | --- |
| [AMP-WALK-METHOD-v003](active/method/AMP-WALK-METHOD-v003.md) | active | Fixed-forward AMP with source-parity symmetric self-collision viability | AMP-WALK-METHOD-v002 |
| [AMP-WALK-TRAIN-v004](active/training/AMP-WALK-TRAIN-v004.md) | active | Self-collision owner repair on UniLab asynchronous APPO | AMP-WALK-TRAIN-v003 |

## History

| Contract | Status | Reason |
| --- | --- | --- |
| [AMP-WALK-TRAIN-v001](history/training/AMP-WALK-TRAIN-v001.md) | superseded | Replaced fixed 30% overhead gate with async-integration-first acceptance |
| [AMP-WALK-METHOD-v001](history/method/AMP-WALK-METHOD-v001.md) | superseded | Step 8 exposed inherited pose authority and a collapsed AMP style signal |
| [AMP-WALK-TRAIN-v002](history/training/AMP-WALK-TRAIN-v002.md) | superseded | Replaced by fresh non-saturation sentinel and no-resume recovery gate |
| [AMP-WALK-METHOD-v002](history/method/AMP-WALK-METHOD-v002.md) | superseded | 2000-iteration playback exposed the missing source self-collision objective |
| [AMP-WALK-TRAIN-v003](history/training/AMP-WALK-TRAIN-v003.md) | superseded | Short-sentinel recovery completed; replaced by source-parity collision repair |

## Recall Rule

Read only the active contract required by the task. Do not infer Phase 2
AMP/distillation integration from these Phase 1 contracts. Future semantic
changes belong in a proposal first and require a new contract version.
