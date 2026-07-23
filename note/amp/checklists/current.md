# AMP-WALK Phase 1 Acceptance Checklist

Status values: `PASS`, `PARTIAL`, `PENDING`, `BLOCKED`.

Current cursor: original Step 8 is `runtime-pass / quality-fail`. Recovery
Steps 1-2 / 3 are complete; the short sentinel passed with minimal logit
margin. Recovery Step 3 is waiting for separate human authorization.

## Governance Gate

| Item | Owner | S tier / T kind | Status | Evidence |
| --- | --- | --- | --- | --- |
| Active method semantics | AMP contract registry | S0 / T-contract | PASS | `contracts/active/method/AMP-WALK-METHOD-v002.md` |
| Active async training semantics | AMP contract registry | S0 / T-contract | PASS | `contracts/active/training/AMP-WALK-TRAIN-v003.md` |
| Four design points map one-to-one to Concept Figure | docs governance | S0-S1 / T-contract | PASS | `concept/04_amp_walk_async_method.data.json`; atlas checker |
| Phase 2 distillation isolation | AMP contracts/plan | S0 / T-contract | PASS | Phase boundary in both active contracts |

## Step Acceptance

| Step | Acceptance item | Owner | S tier / T kind | Status | Evidence pointer |
| --- | --- | --- | --- | --- | --- |
| 1 | Normal collector exit closes env, queue, pipe, attachments, and parent shared memory | AsyncRunner/APPO worker | S2-S3 / T-order,T-persist | PASS | `evidence/2026-07-22-step1-appo-foundation.md`; spawned smoke + lifecycle report |
| 1 | Injected collector failure propagates and leaves no process/shared-memory residue | AsyncRunner/APPO worker | S2-S4 / T-order,T-diff | PASS | `evidence/2026-07-22-step1-appo-foundation.md`; wrapper failure and unlink regressions |
| 1 | APPO checkpoint is CPU-owned, atomic, and exactly reloadable | APPO checkpoint owner | S1-S3 / T-persist,T-order | PASS | `tests/algos/test_appo_checkpoint.py`; Step 1 evidence |
| 1 | Unchanged APPO 2048-env throughput baseline is persisted | APPORunner/metrics | S4 / T-performance | PASS | `/private/tmp/unilab_appo_step1_baseline_r2`; Step 1 evidence; stable A/B remains Step 7 |
| 2 | Optional typed current/next 195-D payload crosses spawned IPC | ring/staging/collector | S1-S3 / T-shape,T-connect,T-order | PASS | `evidence/2026-07-22-step2-typed-appo-payload.md`; spawned 195-D probe |
| 2 | Default APPO/HORA field and memory behavior is unchanged | generic APPO/IPC | S2-S3 / T-diff | PASS | Step 2 evidence; 57-test APPO/HORA regression suite |
| 3 | Manifest admits forward walking only | AMP dataset owner | S1 / T-oracle,T-value | PASS | `evidence/2026-07-22-step3-forward-expert.md`; SHA-locked two-file manifest |
| 3 | Expert transition feature values match AMP_mjlab | AMP spec/dataset owner | S1 / T-shape,T-value,T-diff | PASS | Step 3 evidence; fixed source-frame oracle and deterministic adjacent sampler |
| 4 | Actor/critic/AMP observation groups have declared dimensions and no gait phase | G1 AMP env/config | S1-S2 / T-shape,T-contract | PASS | `evidence/2026-07-22-step4-g1-amp-env.md`; phase differential and live reset |
| 4 | Done/timeout/partial-reset rows preserve exact AMP transition identity | G1 AMP env/collector | S1-S2 / T-order,T-connect | PASS | Step 4 evidence; partial-row helper plus live one-step timeout final observation |
| 4 | Existing `G1WalkFlat` behavior remains unchanged | existing G1 task | S2 / T-diff | PASS | Step 4 evidence; 58-test legacy env suite; body tracking default OFF |
| 5 | AMP reward matches the active formula and enters combined reward once | AMP learner | S1-S2 / T-value,T-connect | PASS | `evidence/2026-07-22-step5-amp-learner.md`; formula and V-trace consumer test |
| 5 | One staged batch uses one frozen `D_k`; APPO updates before `D_(k+1)` | AMP learner | S1-S2 / T-order | PASS | Step 5 evidence; explicit version trace `score/vtrace/policy/D+1` |
| 5 | Discriminator/normalizer/replay/state roundtrip is exact and finite | AMP learner | S1-S2 / T-persist,T-value | PASS | Step 5 evidence; atomic second-learner exact roundtrip |
| 6 | Formal Hydra/runtime-resolver route reaches AMP collector and learner | entrypoint/runtime owner | S2-S3 / T-connect | PASS | `evidence/2026-07-22-step6-formal-async-runtime.md`; spawned two-iteration run |
| 6 | Formal normal/failure routes close cleanly and resume once before spawn | runtime/checkpoint owner | S2-S3 / T-order,T-persist | PASS | Step 6 evidence; failure close, success close, and `D_2 -> D_3` resume |
| 6 | Actor-only playback requires no AMP training objects | playback/export owner | S2-S3 / T-connect,T-persist | PASS | Step 6 evidence; generic play, ONNX parity, two-frame render |
| 6 | Distillation runtime is neither imported nor mutated by Phase 1 | AMP runtime resolver | S2-S3 / T-diff,T-connect | PASS | Runtime import-isolation regression; no distill owner files changed |
| 7 | Matched AMP overhead and bottleneck identity are persisted without lifecycle or capacity failure | performance owner | S3-S4 / T-performance,T-diff | PASS | Step 7 matched evidence plus `evidence/2026-07-23-performance-gate-acceptance-revision.md`; 96.6% local-MPS overhead accepted as diagnostic |
| 8 | Frozen bounded run completes with finite diagnostics and clean postflight | formal live owner | S4 / T-live,T-performance | PASS | `evidence/2026-07-23-step8-runtime-pass-quality-fail.md`; 1850/1850 resume, clean lifecycle, checkpoint/event hashes |
| 8 | Playback shows fixed-forward human-like AMP walking | playback/acceptance owner | S4 / T-live | BLOCKED | fixed-forward locomotion present, but visual style remains task-shape dominated; discriminator/style evidence classifies quality fail |

## Recovery Proposal Acceptance

| Recovery step | Acceptance item | Owner | S tier / T kind | Status | Evidence pointer |
| --- | --- | --- | --- | --- | --- |
| 1 / 3 | Human confirms task-versus-style authority and contract versioning | contracts/Concept Figure | S0 / T-contract | PASS | `AMP-WALK-METHOD-v002`; `AMP-WALK-TRAIN-v003`; four-block Concept Figure |
| 2 / 3 | AMP owner config has no default-pose authority while legacy G1Walk remains unchanged | task config/reward owner | S0-S2 / T-contract,T-diff,T-connect | PASS | failing-first owner/config tests; Recovery Step 2 evidence |
| 2 / 3 | Expert support and sampling identities are distinct and logged | dataset/learner | S1-S2 / T-oracle,T-value,T-connect | PASS | 2 clips, 935 unique transitions, 4096 draws/update in sentinel |
| 2 / 3 | Frozen scoring batch logs quantiles, zero plateau, and weighted reward components | AMP learner | S1-S2 / T-value,T-connect | PASS | deterministic regression and TensorBoard event |
| 2 / 3 | Fresh 20-iteration official async sentinel passes v003 health/lifecycle gate | formal runtime | S3 / T-live,T-order,T-performance | PASS | `evidence/2026-07-23-recovery-step2-style-authority.md`; narrow tail-five pass; clean 20/20 lifecycle |
| 3 / 3 | Fresh bounded target-GPU run produces lifecycle and human-like playback evidence | formal live/playback owner | S4 / T-live,T-performance | BLOCKED | Step 2 passed; separate human authorization required |

## S/T Matrix

| Layer | Meaning | Phase 1 use |
| --- | --- | --- |
| S0 | document/config semantic contract | active contracts, Concept Figure, Hydra ownership |
| S1 | deterministic owner behavior | manifest, shapes, values, formula, checkpoint |
| S2 | cross-owner connectivity | env/collector/IPC/learner/runtime routing |
| S3 | formal spawned route and persisted artifacts | lifecycle, resume, playback, performance identity |
| S4 | target-machine/live physical evidence | throughput observation, bounded live run, walking playback |

Required T kinds include `T-contract`, `T-oracle`, `T-shape`, `T-value`,
`T-order`, `T-connect`, `T-persist`, `T-diff`, `T-performance`, and `T-live`.

## Completion Rule

No step is complete until all rows owned by that step are `PASS` with concrete
evidence pointers. Planning documents alone cannot promote an implementation,
integration, performance, or live row beyond `PENDING`.
