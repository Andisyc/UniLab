# Role Data GPU Aggregate Save Bug

## Purpose

This document records the current answer for the distillation native crash investigation.
It is intentionally written as a white-box debugging artifact: what failed, what was
proved, what was ruled out, and what boundary should be fixed.

## Current Answer

The strongest current explanation is a device ownership bug in the Role Data
aggregate assembly path.

Formal training sets `training.device=cuda:0`. The current assembly code passes that
training device into `build_multitask_distillation_dataset()`, so the large Role Data
aggregate is constructed on GPU during a cold-path cache/IO phase. The subsequent
`save_distillation_dataset()` call detaches the tensors to CPU and calls `torch.save()`.
On the real r10 aggregate identity, this GPU-born aggregate save path reproduced a
native `SIGSEGV` inside PyTorch serialization.

The fix boundary should be:

```text
Role Data aggregate assembly/save: CPU-only cold path.
Offline training/load/update: still uses training.device.
```

This means the aggregate dataset should be built and saved on CPU, then later loaded
to the requested training device by the offline distillation update path.

## Observed Runtime Chain

```text
formal training config: training.device=cuda:0
-> scripts/train_distill.py::run_multitask_dataset_assembly()
-> build_multitask_distillation_dataset(..., device=_distill_device(cfg))
-> GPU-born aggregate with 1,048,576 samples
-> save_distillation_dataset()
-> tensor.detach().cpu()
-> torch.save(payload)
-> torch.serialization.persistent_id
-> SIGSEGV
```

The key code boundary is:

- `scripts/train_distill.py::run_multitask_dataset_assembly()`
  - currently passes `device=_distill_device(cfg)` into aggregate assembly.
- `src/unilab/algos/torch/distill/data.py::save_distillation_dataset()`
  - saves a CPU-detached payload with `torch.save()`.
- `src/unilab/algos/torch/distill/offline.py::run_offline_distillation_updates()`
  - later loads and trains on the configured device; this is not the same boundary.

## Evidence Table

| Claim | Evidence | Evidence class | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| CPU aggregate assembly/save is valid | `aggregate_cpu_fresh` rebuilt and saved/reloaded the r10 aggregate successfully | runtime-confirmed | The source role datasets, metadata, and aggregate semantics are not inherently corrupt | It does not prove GPU-born aggregate save is safe |
| GPU aggregate assembly/save reproduces the native symptom | `aggregate_gpu_fresh` exited with return code `139`, `Fatal Python error: Segmentation fault` | native-symptom-confirmed | The failure can occur before MoE update, in Role Data aggregate save | It does not identify the first invalid native operation |
| Python label values are not the writer | Before crash, `scenario_labels` were all legal strings, `invalid_head=[]`, builtins were normal | runtime-confirmed | The observed crash is not explained by bad `scenario_labels`, `target_indices`, or corrupted builtins | It does not prove PyTorch/CUDA native state is healthy |
| MoE/offline update was not reached in the latest campaign | `offline_cpu_fresh`, `offline_gpu_fresh`, and lifecycle stages stopped at the replay-budget guard | runtime-confirmed | The latest package cannot implicate MoE trainer/update/checkpoint reload | It does not prove those later stages are impossible to fail |
| No immediate hardware fault was observed | health snapshots showed no Xid/NVRM/kernel error and normal RAM/GPU memory state | runtime-confirmed | The current evidence does not point to a visible hardware/kernel GPU fault | It does not exclude a PyTorch/CUDA native memory bug |

## What Was Ruled Out

These hypotheses are currently weaker than the Role Data GPU aggregate boundary:

- `target_indices` stale list or class aliasing.
  - Prior bytecode evidence showed `target_indices` is initialized as a fresh list.
- Bad `scenario_labels` content.
  - The crash run validated all boundary entries and produced `invalid_head=[]`.
- MoE Student update as the first observed failing boundary.
  - The latest campaign did not reach offline update because the replay-budget guard fired first.
- Checkpoint reload or `PersistentDistillationRuntime` as the first observed failing boundary.
  - The latest native symptom appeared during aggregate save, before those stages.
- Simulator or formal live lifecycle as the required trigger.
  - The crash reproduced in a fresh offline owner-path aggregate assembly process.

## Important Correction From The Investigation

The phrase "directly saving CUDA tensors" is not precise enough.

`save_distillation_dataset()` already calls `.detach().cpu()` before `torch.save()`.
The problem boundary is more specific:

```text
large aggregate tensors are born and concatenated on GPU,
then detached/copied to CPU inside the save path,
then serialized by torch.save().
```

The CPU-born version of the same aggregate passes. The GPU-born version segfaults.
That single-variable contrast is the useful fact.

## Current Evidence Level

```text
root-cause class:
  native-symptom-confirmed

confirmed boundary:
  Role Data aggregate assembly/save with device=cuda:0

victim/detection site:
  torch.save -> torch.serialization.persistent_id

owner-confirmed:
  no

first-invalid-operation-confirmed:
  no
```

The investigation has not yet proved whether the first invalid operation is a PyTorch
serialization bug, CUDA allocator/lifetime issue, GPU-to-CPU detach/copy issue, or a
prior native state corruption. For repository repair, the actionable owner boundary is
already clear: Role Data aggregate assembly/save is cold-path IO and should not run on
the training GPU.

## Proposed Repair Boundary

Scope:

- Change `run_multitask_dataset_assembly()` so aggregate assembly/save uses CPU.
- Preserve offline distillation update behavior: loading and training still use
  `training.device`.
- Add a focused contract test proving that a CUDA training config does not make the
  aggregate assembly owner build the cached dataset on GPU.

Non-scope:

- Do not change MoE trainer/update logic.
- Do not change checkpoint reload or `PersistentDistillationRuntime`.
- Do not change role dataset semantics, labels, or balancing policy.
- Do not start formal DAgger training as part of this repair.

Expected invariant:

```text
training.device may be cuda:0,
but Role Data aggregate assembly/save uses CPU.
```

## Repair Applied

`scripts/train_distill.py::run_multitask_dataset_assembly()` now treats aggregate
assembly as a CPU-owned Role Data cold path:

```text
training.device=cuda:0
-> run_multitask_dataset_assembly()
-> build_multitask_distillation_dataset(..., device="cpu")
-> save_distillation_dataset()
-> later offline load/update still owns movement to training.device
```

This does not change role labels, balancing semantics, MoE trainer/update behavior,
checkpoint reload, or persistent runtime behavior. It only removes the training GPU
from the cache assembly/save owner boundary that reproduced the native PyTorch
serialization crash.

The focused contract test is:

```text
tests/scripts/test_train_scripts.py::test_distill_script_builds_multitask_dataset_from_saved_sources
```

It composes a `training.device=cuda:0` distillation config, runs multitask assembly on
small saved role datasets, and asserts the aggregate assembly device reported by the
probe is `cpu`.

## Repair Process Record

Date: 2026-07-22

Human decision:

```text
Accept the owner-boundary repair:
Role Data aggregate assembly/save is CPU-only cold-path IO;
offline training/load/update may still use training.device.
```

Reason for the decision:

- The existing one-shot owner-path package showed the single useful contrast:
  `aggregate_cpu_fresh` passed, while `aggregate_gpu_fresh` reproduced `SIGSEGV`
  inside `torch.save -> torch.serialization.persistent_id`.
- The same run showed valid labels and normal builtins before the crash, so the
  immediate failure was not explained by bad `scenario_labels`, bad
  `target_indices`, or Python builtin rebinding.
- The offline update / MoE / checkpoint / persistent lifecycle stages were not reached
  in that package, so the repair must not modify those semantics.
- `save_distillation_dataset()` already CPU-detaches tensors before serialization;
  therefore the repaired boundary is not "avoid saving CUDA tensors", but "avoid
  creating the large aggregate as GPU-born tensors in the cold cache path".

Code change:

```text
scripts/train_distill.py
  _ROLE_DATA_ASSEMBLY_DEVICE = "cpu"

  run_multitask_dataset_assembly()
    before: build_multitask_distillation_dataset(..., device=_distill_device(cfg))
    after:  build_multitask_distillation_dataset(..., device=_ROLE_DATA_ASSEMBLY_DEVICE)

  probe result now records:
    aggregate_assembly_device = "cpu"
```

Contract change:

```text
tests/scripts/test_train_scripts.py::test_distill_script_builds_multitask_dataset_from_saved_sources
  sets training.device=cuda:0
  runs real saved-source multitask assembly
  asserts probe["aggregate_assembly_device"] == "cpu"
  still verifies restored role labels and teacher actions
```

Local validation:

```bash
uv run pytest tests/scripts/test_train_scripts.py -q -k multitask_dataset
# 2 passed, 200 deselected

uv run ruff check scripts/train_distill.py tests/scripts/test_train_scripts.py
# All checks passed
```

Preserved semantics:

- No change to teacher role datasets.
- No change to role labels, scenario labels, or balancing semantics.
- No change to MoE Student architecture.
- No change to offline update device ownership after dataset load.
- No change to checkpoint reload.
- No change to `PersistentDistillationRuntime` or `SharedWeightSync`.
- No formal DAgger live training was launched as part of this repair.

Remaining uncertainty:

```text
The first invalid native operation inside PyTorch/CUDA is still not confirmed.
The repository-level fix is justified because the failing boundary was a cold-path
owner violation, and the CPU-owned version of the same aggregate identity passed.
```

Next verification:

```text
Sync this patch to the server.
Run the real-owner one-shot campaign once.
Expected result: aggregate assembly no longer allocates the large aggregate on GPU,
so the previous aggregate_gpu_fresh torch.save SIGSEGV should disappear.
Only after that should formal training be considered again.
```

## Minimal Verification Plan

1. Code-level verification:
   - Inspect `run_multitask_dataset_assembly()` and confirm it no longer passes
     `_distill_device(cfg)` to `build_multitask_distillation_dataset()`.

2. Contract test:
   - Build a config with `training.device=cuda:0`.
   - Run the aggregate assembly path on a small semantic source fixture.
   - Assert the saved dataset can be loaded and matches expected labels.
   - Assert assembly device ownership is CPU or is recorded as CPU in the result.

3. Server one-shot verification:
   - Re-run only the aggregate CPU/GPU differential or the full offline campaign after
     the code fix.
   - Expected result: the previous `aggregate_gpu_fresh` crash disappears because the
     aggregate assembly owner no longer uses GPU.

## Human Decision Point

The next decision is not whether to keep adding prints. The next decision is whether to
accept the owner-boundary fix:

```text
Role Data cache assembly is a CPU cold path,
not a training-GPU computation path.
```

If accepted, the repair should be small and local to the assembly owner plus its
contract test.

## 2026-07-22 Offline Fresh Evidence

Artifact:

```text
20260722-012517_distill-real-owner-one-shot-RETURN_ME.tar.gz
```

Selected groups:

```text
offline_device only
```

Result:

```text
offline_cpu_fresh: FAIL, 4.05s, native-symptom-confirmed
offline_gpu_fresh: PASS, 111.99s, runtime-confirmed
```

The CPU fresh failure happened before optimizer/update work:

```text
load_distillation_dataset()
-> torch.load(map_location=cpu)
-> build_distillation_dataset()
-> _validate_command_intents()
-> command_intent_validation/corruption_detected
```

Observed impossible object fact:

```text
index: 17957
raw_type: str
raw_repr: "'active'"
normalized: "<class 'frame'>"
```

Interpretation:

- The raw loaded value was a normal Python string.
- The validator's normalization step produced an impossible value for that raw object.
- `builtins.str`, `builtins.isinstance`, `builtins.type`, `builtins.tuple`, and
  `builtins.list` were reported as original and callable.
- No new native core was produced because the stage used
  `UNILAB_NATIVE_ABORT_ON_CORRUPTION=0`.
- The same aggregate/checkpoint identity completed the GPU fresh offline path for
  8192 updates, including checkpoint reload and `SharedWeightSync` cleanup.

This changes the active debugging boundary:

```text
previous useful boundary:
  GPU-born aggregate assembly/save -> torch.save SIGSEGV

current useful boundary:
  CPU fresh offline load -> command_intents validation impossible normalization
```

The current symptom is no longer tied to formal DAgger live training. It is available
through a short offline CPU owner path and should be captured there.

Current classification:

```text
root-cause class:
  native-symptom-confirmed / impossible Python object transformation

victim/detection site:
  src/unilab/algos/torch/distill/data.py::_validate_command_intents()

unconfirmed native component:
  still unknown; first invalid write/free/op not captured
```

Next plan:

```text
Run only offline_cpu_fresh.
Set UNILAB_NATIVE_ABORT_ON_CORRUPTION=1.
Abort at command_intent_validation/corruption_detected.
Harvest the new Apport/core artifact and inspect the active Python frame, locals,
globals, and native thread state at the exact detector site.
```

Non-scope for the next plan:

- Do not run formal training.
- Do not run GPU fresh again.
- Do not run aggregate/lifecycle/dual-resident stages.
- Do not change label/data semantics before the first-invalid-operation evidence is
  captured.

Implemented capture entry:

```text
scripts/deploy/run_distill_offline_cpu_abort_capture.sh
```

This wrapper sets:

```text
RUN_GROUPS=offline_device
STAGE_NAMES=offline_cpu_fresh
NATIVE_ABORT_ON_CORRUPTION=1
BATCH_SIZE=2048
FRESH_UPDATES=8192
```

Server command:

```bash
cd /ssd1/cyx/UniLab
bash scripts/deploy/run_distill_offline_cpu_abort_capture.sh
```

Expected evidence:

```text
offline_cpu_fresh aborts at command_intent_validation/corruption_detected.
The campaign harvests the new Apport/core candidate into RETURN_ME.tar.gz.
The next review reads the new core/GDB output instead of running another live test.
```

## 2026-07-22: formal-run diagnostic output gate

Observed fact:

```text
g1-walk-stand-ownerfix-r1-console.log reached workflow/iteration_4/after_aggregate
and update_number=14300 without Traceback/SIGABRT/SIGSEGV/KeyboardInterrupt in
the pulled log. The last lines were ordinary target_index diagnostic prints.
```

Boundary result:

```text
CPU-owner aggregate assembly/save/load passed through iteration 4.
The run was not a crash log; it was stopped or truncated during offline update.
```

Engineering correction:

```text
The temporary [distill-data-runtime], [distill-trainer-runtime], and
[distill-offline-runtime] breadcrumbs are now opt-in through
UNILAB_DISTILL_RUNTIME_DEBUG=1.
Default formal training no longer emits the high-volume target_index trace.
```

Verification:

```text
uv run pytest tests/algos/test_distill_runtime_debug.py -q
uv run pytest tests/scripts/test_train_scripts.py -q -k multitask_dataset
uv run ruff check src/unilab/algos/torch/distill/trainer.py \
  src/unilab/algos/torch/distill/data.py \
  src/unilab/algos/torch/distill/offline.py \
  tests/algos/test_distill_runtime_debug.py

## 2026-07-22: iteration-7 aggregate replay plan
Observed fact:
g1-walk-stand-ownerfix-r2 failed at:
run_multirole_dagger_workflow()
-> aggregate_datasets(tuple(cumulative_sources), aggregate_path)
-> run_multitask_dataset_assembly()
-> build_multitask_distillation_dataset()
-> annotate_distillation_dataset_scenario()
with:
ValueError: walk_flat scenario annotation conflicts with command_intents

The pulled manifest-source scan showed all manifest-recorded sources before iteration 7
were semantically valid:
- bootstrap walk source: scenario=walk_flat, command_intents=active only.
- bootstrap stand source: scenario=static_stand, command_intents=inactive only.
- iteration 1-6 walk_flat sources: active only.
- iteration 1-6 static_stand sources: inactive only.
- iteration 1-6 walk_to_stop sources: mixed active/inactive with scenario_labels=walk_to_stop.

Correction:
The previous bootstrap-source hypothesis is ruled out by this scan.
The missing boundary is the transient iteration-7 source list. Those files exist before
aggregate, but are not written into run_manifest.json until after aggregate/update
succeeds. Therefore manifest-only scans cannot prove the failing source.

New diagnostic entry:
scripts/deploy/replay_distill_iteration_aggregate.py
  Rebuilds the exact pre-aggregate source list:
  bootstrap_sources
  + completed iteration scenario_artifacts
  + pending datasets/dagger_iteration_N/{scenario}.pt
  Then runs:
  raw torch.load snapshot
  -> load_distillation_dataset()
  -> annotate_distillation_dataset_scenario()
  -> build_multitask_distillation_dataset()

Wrapper:
scripts/deploy/run_distill_iteration_aggregate_replay.sh
  Runs the replay on the server, captures precheck/console/report, and packages
  RETURN_ME.tar.gz. It does not save the rebuilt aggregate by default, so the
  returned archive stays small.

Local validation:
UV_CACHE_DIR=.uv-cache uv run python -m py_compile scripts/deploy/replay_distill_iteration_aggregate.py
UV_CACHE_DIR=.uv-cache uv run ruff check scripts/deploy/replay_distill_iteration_aggregate.py
bash -n scripts/deploy/run_distill_iteration_aggregate_replay.sh
Result: pass.

Semantic fixture validation:
A tiny fake run with valid bootstrap + pending iteration sources passed aggregate replay.
The same fixture with pending walk_flat command_intents=[active,inactive] failed with:
SOURCE_ANNOTATE_FAILED
first_failed_scenario=walk_flat
first_failed_error=ValueError('walk_flat scenario annotation conflicts with command_intents')

Next evidence boundary:
Run the wrapper on the server for ownerfix-r2 iteration 7.
Expected decisive outcomes:
- SOURCE_ANNOTATE_FAILED: identifies the concrete pending source path and label head.
- AGGREGATE_FAILED: source-level annotation passed, but full concat/build failed.
- PASS: files/schema are clean; the original failure then points to runtime mutation,
 server code identity drift, or a source-list mismatch in the live process.
```

## 2026-07-22: ownerfix-r4 resolution pointer

The later successful repair path is recorded separately in:

```text
note/distill_native_corruption/ownerfix_r4_resolution.md
```

Summary:

```text
Role/scenario artifact metadata became owner truth.
Dataset and checkpoint writes became atomic.
Checkpoints became CPU-owned cold artifacts.
DAgger iteration checkpoints stopped saving/resuming optimizer state.
g1-walk-stand-ownerfix-r4 completed formal DAgger iteration 8.
```
