# Step 6 Evidence: Formal Async AMP Runtime

Date: 2026-07-22

Branch: `codex/amp-walk-async-migration`

## Formal Route

The owner YAML `conf/appo/task/g1_amp_walk/mujoco.yaml` selects exactly one
runtime feature owner:

```text
algo.runtime_impl = amp_appo
algo.runtime_resolver = unilab.algos.torch.amp.runtime:resolve_amp_appo_runtime
```

The resolved path is:

```text
train_appo.py
  -> AMPAPPORunner (generic AsyncRunner lifecycle)
  -> appo_collector_fn + optional AMP payload writer
  -> typed RolloutRingBuffer amp_state/amp_next_state
  -> generic RolloutStagingPool
  -> AMPAPPOLearner
```

There is one collector and one learner. No discriminator process, alternate
queue, or distillation runtime exists on this route. The generic play function
is reused unchanged and loads only `checkpoint["actor"]`.

## Failure-To-Fix Evidence

Two formal attempts failed before the accepted run and exposed real boundary
defects:

1. MPS rejected a float64 normalizer count buffer before spawn. The owner fix
   made count float32, consistent with its statistics and Phase 1 scale.
2. A single terminated env caused partial-reset AMP shape `(128,195)` versus
   `(1,195)`. The owner fix added a G1 reset-row observation hook; the default
   task ignores row IDs, while G1 AMP indexes the public body-state view only on
   reset. A dedicated partial-row regression now covers this case.

The second failure still demonstrated error propagation and clean closure:
collector exit code 1, four parent resources closed, lifecycle error list empty.

## Accepted Spawned Run

Formal command family:

```text
UV_CACHE_DIR=.uv-cache uv run train --algo appo --task g1_amp_walk --sim mujoco \
  algo.num_envs=128 algo.steps_per_env=8 algo.max_iterations=2 \
  algo.save_interval=1 ... training.no_play=true \
  training.log_dir=/private/tmp/unilab_amp_step6_formal_r3
```

Persisted result:

- status `completed`, iterations `2/2`;
- 17,920 collector env steps observed by the logger;
- training wall time approximately 3.34 s;
- discriminator version `2`;
- replay size `2048`, normalizer count `2048`;
- all actor/critic/discriminator/normalizer floating tensors finite;
- checkpoint `/private/tmp/unilab_amp_step6_formal_r3/model_2.pt`;
- lifecycle: collector exit code 0, not terminated, four resources, no errors.

This is a connectivity smoke, not a policy-quality or 10-20 minute performance
claim.

## Resume And Actor-Only Playback

The produced checkpoint was resumed through the same formal training route for
one iteration. Its first reported version was `3`, and the persisted checkpoint
contains version `3` with normalizer count `3072`. This proves version `2` was
loaded before collector spawn rather than reinitialized.

The generic APPO play entrypoint then loaded the same actor checkpoint without
constructing learner/discriminator/replay state:

- ONNX exported to `policy.onnx`;
- ONNX versus PyTorch maximum difference `8.94e-08`;
- two-frame/two-env MuJoCo video persisted as `play_video.mp4`.

The two-frame video verifies playback connectivity only; it is not evidence of
learned walking quality.

## Regression Closure

```text
Steps 1-6 focused aggregate:
138 passed, 3 skipped, 1 deselected

Default APPO real spawned OFF path after hook integration:
3 passed

Ruff: PASS
git diff --check: PASS
Runtime Atlas JSON: PASS
AMP Concept Figure checker: PASS (4 design points, 4 nodes, 4 interactions)
```

## Verdict

Step 6 is `PASS`. The formal Hydra/runtime resolver route, spawned collector,
typed AMP payload, frozen-D learner, full resume, actor-only playback, failure
propagation, lifecycle cleanup, and distillation isolation are verified. Step 7
performance A/B and Step 8 bounded policy training remain explicitly pending.
