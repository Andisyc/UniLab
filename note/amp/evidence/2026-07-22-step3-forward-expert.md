# Step 3 Evidence: Forward-Walk Expert And 195-D State

Date: 2026-07-22

Branch: `codex/amp-walk-async-migration`

## Frozen Expert Identity

The Phase 1 manifest admits exactly two AMP_mjlab clips:

- `walk_forward_loop_002__A022.npz`: 455 frames at 50 Hz,
  SHA-256 `dc1eba2d7f124d99d058562072a2376bcf123ec52b7c2b565abd723c3f688e1f`.
- `walk_forward_loop_002__A024.npz`: 482 frames at 50 Hz,
  SHA-256 `25149b69debf5f015379ead46ed1e709cb052e96dbf974865ea2e2a66bf1be0f`.

`cmp` confirmed both UniLab copies are byte-identical to their AMP_mjlab
sources. The loader does not scan a directory. It rejects a non-forward name,
path traversal, duplicate file, hash mismatch, metadata mismatch, missing
array, or canonical body/anchor contract mismatch.

Jog, arc, backward, sideways, idle/turn, and recovery clips are not members of
the manifest and cannot enter Phase 1 expert sampling.

## Canonical Feature

`build_amp_observation` reproduces the AMP_mjlab observation order for these
13 bodies:

```text
[anchor-relative position (13x3),
 anchor-relative rotation first two matrix columns (13x6),
 body-local linear velocity (13x3),
 body-local angular velocity (13x3)] = 195
```

The anchor is `torso_link`. The body indices are frozen in the manifest and
code. A fixed first source frame is checked at the head and every feature-group
boundary against persisted AMP_mjlab-derived values.

## Transition And Runtime Probe

- Legal adjacent transitions: `(455 - 1) + (482 - 1) = 935`.
- Sampling uses precomputed contiguous current/next arrays, so no Python loop
  runs per sample and no transition can cross a clip boundary.
- Repeated seed `20260722` produces exact motion/frame/current/next identity.
- Cold load: approximately `0.0073 s` on the local machine.
- Vectorized 200,000-transition sample: approximately `0.0360 s`, shape
  `(200000, 195)`, all finite.

## Verification

```text
UV_CACHE_DIR=.uv-cache uv run pytest \
  tests/algos/test_amp_motion_dataset.py \
  tests/utils/test_math_utils.py -q

13 passed in 0.03s
```

Ruff passed for the AMP modules and tests.

## Verdict

Step 3 is `PASS`. Expert identity, source bytes, 195-D value/order, deterministic
adjacent sampling, and cold/hot ownership are explicit and reproducible.
