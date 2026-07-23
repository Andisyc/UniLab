# Distillation Lifecycle Repair Impact On AMP/APPO

Date: 2026-07-22

Scope: read-only review of current repository code, recent committed changes,
and `note/distill_native_corruption/`. No test, training, collector, checkpoint,
or cleanup sentinel was run in this review.

## Evidence

- `E1`: `note/distill_native_corruption/ownerfix_r4_resolution.md` records the
  R4 owner-boundary repair and its remaining uncertainty.
- `E2`: `note/distill_native_corruption/real_owner_offline_campaign.md` records
  the persistent runtime, shared-weight, checkpoint, and cleanup risk audit.
- `E3`: commit `77cc4a3c` changes distillation data/checkpoint/offline/workflow
  owners, not APPO owners.
- `E4`: `scripts/train_appo.py` closes the runner in `finally`.
- `E5`: `src/unilab/ipc/async_runner.py` uses `spawn`, propagates collector
  errors, joins the collector before parent resource cleanup, and terminates
  only after a bounded cooperative wait.
- `E6`: `src/unilab/algos/torch/appo/worker.py` closes the environment and
  attached shared-memory objects only after its loop/exception block, not in a
  `finally` covering initialization and failure paths.
- `E7`: `src/unilab/algos/torch/appo/runner.py` creates a metrics queue without
  an explicit close/join owner and writes checkpoints directly with
  `torch.save(..., final_path)`.
- `E8`: `src/unilab/ipc/{rollout_ring_buffer,weight_sync}.py` suppress cleanup
  exceptions and do not emit a formal unlink/close report.

## Verified Facts

1. The R4 repair hardened five distillation boundaries: artifact scenario
   identity, exact aggregate source identity, atomic dataset save, CPU-owned
   atomic checkpoint save, and DAgger iteration checkpoint reload without
   optimizer state.
2. One formal persistent DAgger run completed through iteration 8 after R4,
   and the user reported that the pulled checkpoint appeared valid locally.
3. The first invalid native operation was not captured. R4 is evidence for an
   owner-boundary repair, not proof of PyTorch/CUDA/native memory safety.
4. The committed R4 code does not directly alter the APPO runner, worker,
   rollout ring, staging pool, or learner. The AMP method route is therefore
   still structurally valid.
5. Phase 1 AMP/APPO has a simpler process topology than DAgger: one collector,
   one learner, no repeated aggregate identity, and no need to reload a new
   policy checkpoint every iteration.
6. Current APPO already has several correct lifecycle foundations, but its
   child exception cleanup, metrics queue ownership, cleanup observability, and
   checkpoint atomicity are not yet at the R4 steady-state standard.

## Planning Decisions

- Treat the generic APPO lifecycle floor as a prerequisite acceptance boundary,
  not as an AMP-specific workaround.
- Keep the discriminator, expert sampler, replay, normalizer, and all optimizer
  state inside the learner process.
- Do not add another persistent service or synchronize discriminator weights to
  the collector.
- Require child `finally` cleanup, parent join-before-unlink ordering, explicit
  queue/pipe closure, and a no-residue lifecycle report on normal and injected
  failure paths.
- Require CPU-owned atomic checkpoints. Optimizer state may remain live during
  one uninterrupted learner run and may be loaded once for an explicit resume;
  it must not be repeatedly serialized/reloaded as an inner training phase.
- Keep the active distillation workflow unchanged in Phase 1. AMP plus
  distillation is a separate Phase 2 design.

## Open Risks

- No current runtime evidence proves APPO normal/failure cleanup leaves no
  child, queue feeder, or shared-memory residue.
- No measured baseline proves that lifecycle hardening or AMP payload transfer
  preserves the target throughput.
- The old native corruption's first invalid operation remains unknown, so
  recurrence with moving impossible-object symptoms must return to a native
  lifecycle campaign rather than victim-local business logic patches.

## Next Evidence

Before AMP payload work, run the generic APPO lifecycle and throughput gate:
normal completion, injected collector exception, bounded join/terminate,
explicit queue/pipe close, shared-memory unlink verification, CPU-owned atomic
checkpoint round trip, and matched throughput metrics.
