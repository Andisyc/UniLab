# Step 1 APPO Foundation Evidence

Date: 2026-07-22

Scope: generic APPO/IPC lifecycle, CPU-owned atomic checkpoint, and bounded
unchanged G1 APPO baseline. No AMP or distillation behavior was enabled.

## Commands And Results

- `uv run pytest tests/algos/test_appo_checkpoint.py tests/ipc/test_async_runner.py tests/ipc/test_rollout_ring_buffer.py tests/ipc/test_shared_weight_sync.py tests/algos/test_appo_worker.py tests/algos/test_appo_runner_unit.py -q`
  - result: `62 passed in 3.38s`.
- `uv run pytest tests/algos/test_appo_runner.py -q -m slow`
  - result: `3 passed in 8.33s`.
  - boundary: real spawned MuJoCo collector/learner, including a 128-env,
    two-iteration APPO run.
- `uv run pytest tests/scripts/test_train_scripts.py -q -k 'appo_runner_kwargs or build_appo_runner_kwargs or appo_hydra_default_algo_log_name or train_appo_get_log_root'`
  - result: `7 passed, 196 deselected`.
- `UV_CACHE_DIR=.uv-cache uv run train --algo appo --task g1_walk_flat --sim mujoco algo.max_iterations=1 algo.save_interval=0 training.no_play=true training.log_dir=/private/tmp/unilab_appo_step1_baseline_r2`
  - result: exit 0, 2048 envs, 1 iteration, 40,960 reported env steps,
    roughly 2 seconds training summary.
  - checkpoint: `/private/tmp/unilab_appo_step1_baseline_r2/model_1.pt`.
  - lifecycle: `collector_exitcode=0`, `collector_terminated=false`,
    `resource_count=4`, `errors=[]`, `state=complete`.
- focused Ruff over changed APPO/IPC/test owners
  - result: all checks passed.

## Facts

1. Collector resources are closed in reverse construction order from a wrapper
   `finally`, including a synthetic body failure path.
2. `AsyncRunner.close()` attempts every resource, closes and joins queue
   resources, reports cleanup state, and surfaces cleanup failures.
3. Ring and weight shared-memory owners distinguish handle close from unlink;
   cleanup remains idempotent and owner close followed by cleanup unlinks the
   POSIX shared-memory name.
4. APPO checkpoints recursively detach tensors to CPU, write a same-directory
   temporary artifact, atomically replace the destination, and preserve an
   existing destination on a synthetic save failure.
5. The unchanged 2048-env G1 APPO route completed and emitted a clean lifecycle
   report. The short run is throughput/lifecycle evidence, not policy-quality
   evidence.

## Limitations

- The local one-iteration baseline is too short for a stable performance claim.
- The displayed steps/s value was terminal-render truncated; Step 7 still owns
  a persisted matched A/B performance artifact.
- No AMP payload, reward, discriminator, or task was present in this evidence.

