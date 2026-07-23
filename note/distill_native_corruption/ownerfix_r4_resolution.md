# Ownerfix R4 Resolution Record

Date: 2026-07-22

## Executive summary

The latest successful run suggests the failure was not a single bad Python line in
`scenario_labels`, `target_indices`, or `Module.train()`. Those were moving victims.
The practical repair was to harden the owner boundaries that repeatedly crossed
long-process GPU, checkpoint, and serialization lifecycles:

```text
Role/scenario artifact identity
-> aggregate source identity
-> dataset atomic save
-> checkpoint CPU-owned atomic save
-> DAgger learner checkpoint reload without optimizer state
```

After synchronizing the repair, `g1-walk-stand-ownerfix-r4` completed formal DAgger
training through iteration 8. The pulled `dagger_iteration_8.pt` was also tested
locally by the user in simulation and appeared valid. The final checkpoint size
around 1.99 MB is expected because R4 no longer stores optimizer state in DAgger
iteration checkpoints.

## Frozen probing paths

The following low-yield debugging paths are frozen unless new evidence makes them
high-information again:

- Do not keep adding print statements around `scenario_labels`, `target_indices`, or
  `self.student.train()`.
- Do not treat the Python traceback line as the writer/corrupter without first-invalid
  operation evidence.
- Do not keep using formal live training as the primary locator. Use it only as a
  bounded confirmation after structural fixes.

## Observed moving victims

Observed detector sites across the investigation included:

```text
data.py::_validate_scenario_labels()
data.py::_validate_command_intents()
trainer.py::BehaviorDistillationTrainer.update()
torch.nn.Module.train()
torch.save() / torch.serialization.persistent_id
```

The important pattern was that the detector moved while owner-local contracts often
remained valid. This is why the investigation switched from ordinary runtime prints
to owner-boundary structural fixes.

## R3 failure that changed the active boundary

`g1-walk-stand-ownerfix-r3` no longer failed at role/scenario aggregate annotation.
It reached offline student update and failed at:

```text
run_offline_distillation_updates()
-> trainer.update(batch)
-> self.student.train()
-> torch.nn.Module.__setattr__()
-> name in params
TypeError: argument of type '_ParameterMeta' is not iterable
```

Interpretation:

- Victim: PyTorch `Module.__setattr__`.
- Trigger: recursive `self.student.train()`.
- Owner: learner/offline update owns the student module and checkpoint reload lifecycle.
- Corrupter candidate: repeated DAgger checkpoint reload/save and optimizer-state
  ownership in a long persistent GPU process.

This specifically weakened the earlier role-data-only hypothesis. The scenario/role
artifact fix remained useful, but it was not sufficient to explain R3.

## Structural fixes applied before R4

### 1. Role/scenario artifact owner truth

Role/scenario collected artifacts now carry their scenario contract in dataset
metadata:

```text
metadata["workflow_scenario"]
```

Aggregate assembly treats this metadata as owner truth:

- If caller-side source scenario is missing, use artifact metadata.
- If caller-side source scenario conflicts with artifact metadata, fail closed.

This removes the class of bugs where a live mutable source list or reconstructed
manifest assigns a valid artifact to the wrong scenario.

### 2. Exact aggregate source snapshot

Before each live DAgger aggregate, the workflow writes the actual source list that
enters aggregate:

```text
<aggregate>.sources.json
```

This prevents post-failure reconstruction from silently differing from the real
in-process source list.

### 3. Dataset atomic save

`save_distillation_dataset()` now writes to a same-directory temporary file and then
atomically replaces the final `.pt`.

Purpose:

- Prevent readers from observing a partially written dataset artifact.
- Preserve cold-path dataset ownership as CPU/serialization IO, not live GPU state.

### 4. Checkpoint CPU-owned atomic save

`save_distillation_checkpoint()` now serializes a CPU-owned checkpoint payload and
writes through temp file plus atomic replace.

Purpose:

- Avoid carrying GPU tensor storage / allocator state into the cold checkpoint
  artifact.
- Prevent the next DAgger iteration from observing a partially written checkpoint.

### 5. DAgger iteration checkpoints no longer resume/save optimizer state

The R4 repair changes DAgger offline update to:

```text
offline_resume_optimizer = False
offline_save_optimizer = False
```

Rationale:

- Student weights are the semantic owner of the next DAgger iteration.
- Adam optimizer state is tied to parameter identity and update lifecycle.
- Reusing optimizer state across repeated aggregate identities and repeated
  long-process checkpoint reloads is unnecessary for DAgger correctness and was a
  plausible Heisenbug corrupter candidate.

Consequence:

- `dagger_iteration_8.pt` is expected to be small.
- A 1.99 MB checkpoint is normal for a deployable MoE student weight checkpoint
  without `optimizer_state_dict`.

## Local verification evidence

Focused local checks run after the R4 structural repair:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m py_compile \
  src/unilab/algos/torch/distill/checkpoint.py \
  src/unilab/algos/torch/distill/offline.py \
  scripts/train_distill.py

UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/algos/test_g1_distillation_contract.py -q \
  -k 'checkpoint_roundtrip or offline_distillation_checkpoint_can_omit_optimizer_state'

UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/scripts/test_train_scripts.py -q \
  -k 'dagger_update_does_not_resume_or_save_optimizer_state'

UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/algos/test_g1_distillation_contract.py -q \
  -k 'multitask_source_annotation_failure_reports_source_context or multitask_uses_dataset_workflow_scenario_metadata_as_owner_contract or multitask_rejects_source_scenario_metadata_drift or multitask_scenario_failure_emits_raw_source_provenance_snapshot or multitask_workflow_scenario_annotation_preserves_row_roles or offline_distillation_checkpoint_can_omit_optimizer_state'

UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/scripts/test_train_scripts.py -q \
  -k 'distill_script_collects_live_env_dataset_with_owner_projection or dagger_update_does_not_resume_or_save_optimizer_state'

UV_CACHE_DIR=.uv-cache uv run ruff check \
  src/unilab/algos/torch/distill/checkpoint.py \
  src/unilab/algos/torch/distill/offline.py \
  scripts/train_distill.py \
  tests/algos/test_g1_distillation_contract.py \
  tests/scripts/test_train_scripts.py
```

Observed local result:

```text
checkpoint/offline focused tests: pass
workflow DAgger optimizer-boundary test: pass
role/scenario owner-contract regression tests: pass
ruff: pass
```

## Server runtime evidence

Run identity:

```text
g1-walk-stand-ownerfix-r4
execution_mode=persistent_async
collect_num_envs=32
dagger_iterations=8
```

Observed result:

```text
formal DAgger training completed successfully through iteration 8
final checkpoint pulled locally
local simulation test by user appeared valid
```

Additional operational note:

- A foreground SSH-started run can be killed or orphaned by SSH disconnects.
- R4 should be launched with `nohup` or an equivalent detached session for future
  long runs.

## Current evidence classification

| Claim | Evidence | Class | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| R4 owner-boundary fixes are wired | Focused contract tests and Ruff passed | contract-confirmed | The intended checkpoint/data/workflow boundaries are represented in code | It does not prove native memory safety |
| R4 formal training completed | User observed successful formal training and tested checkpoint locally | runtime-confirmed by user | The previous moving failure did not recur in this run | It does not prove the first invalid native operation from older failures |
| Small checkpoint is expected | Checkpoint no longer stores optimizer state | code-confirmed / contract-confirmed | 1.99 MB is consistent with deployable student weights only | It does not measure policy quality |

## Remaining uncertainty

The first invalid native operation was never captured. Therefore the correct wording is:

```text
R4 fixed the owner-boundary class suspected to cause the Heisenbug.
It does not prove the underlying PyTorch/CUDA/native first invalid operation.
```

If future runs again show impossible Python objects, `_ParameterMeta`, `frame`, `cell`,
SIGSEGV, SIGABRT, or moving crash sites, the next step should not be another business
logic patch. It should be a native/lifecycle campaign:

```text
process isolation or restart-each-iteration differential
or Python-aware core/GDB capture
or CUDA synchronous/sanitizer run on the smallest owner path
```

## Recommended steady-state policy

Keep these boundaries unless there is a strong experimental reason to reopen them:

```text
Role Data aggregate/save: CPU cold path
Dataset save: atomic replace
Checkpoint save: CPU-owned atomic replace
DAgger iteration resume: student weights only
DAgger iteration checkpoint: no optimizer_state_dict by default
Artifact scenario identity: metadata owner truth, fail closed on mismatch
```

