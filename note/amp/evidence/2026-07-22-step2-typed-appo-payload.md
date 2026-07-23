# Step 2 Evidence: Typed APPO Payload Extension

Date: 2026-07-22

Branch: `codex/amp-walk-async-migration`

## Implemented Contract

- `RolloutFieldSpec` declares each optional per-slot shape, NumPy dtype, and
  whether the second dimension is rollout time.
- `RolloutRingBuffer` validates name collisions, environment/time dimensions,
  exact child attachment fields, and allocates each field with its declared
  dtype.
- `RolloutStagingPool` accepts either the legacy `slot_shapes` contract or the
  typed `field_specs` contract. It transposes only declared time-axis fields
  and preserves dtype.
- `APPORunner` exposes three specialization hooks: extra field specs, collector
  target, and collector runtime kwargs. The default implementations are empty
  or resolve to the existing APPO collector.
- The worker attaches using the same optional field contract. Shared-memory
  budget estimation includes every extra byte in every ring slot.

No AMP feature semantics, discriminator, reward, or task behavior was added in
this step.

## Runtime Boundary Probe

`test_spawned_collector_payload_reaches_learner_staging` starts a spawn-context
child process, attaches it to the owner ring, writes a constant
`[N, T, 195]` float32 field, and signals completion. The parent stages the
borrowed shared-memory view and observes the exact `[T, N, 195]` tensor and
values. This verifies collector process -> shared memory -> learner staging,
not merely two views in one process.

## Verification

```text
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/algos/test_appo_worker.py \
  tests/algos/test_appo_runner.py \
  tests/algos/test_appo_runner_unit.py \
  tests/algos/test_appo_staging.py \
  tests/algos/test_hora_contract.py \
  tests/algos/test_hora_imports.py \
  tests/ipc/test_rollout_ring_buffer.py \
  tests/ipc/test_memory_budget.py -q

57 passed, 3 deselected in 1.29s
```

The POSIX shared-memory tests were run outside the filesystem sandbox because
macOS denied `shm_open` inside it. The same initial command inside the sandbox
failed only with `PermissionError: Operation not permitted` at `shm_open`.

## Verdict

Step 2 is `PASS`. Optional typed payloads cross a real spawned IPC boundary;
default APPO declares no extra fields; HORA continues using the legacy staging
entry without behavioral changes.
