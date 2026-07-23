# Step 4 Evidence: Isolated G1 AMP Walk Environment

Date: 2026-07-22

Branch: `codex/amp-walk-async-migration`

## Isolated Task Contract

`G1AMPWalk` is a new MuJoCo-only registry entry. It does not replace or mutate
the `G1WalkFlat` registration. Its configuration fails closed unless:

- every command is exactly `[1.0, 0.0, 0.0]`;
- standing and transition sampling probabilities are zero;
- heading, mode, and height control/observation are disabled; and
- gait-phase reward terms have zero weight.

The observation contract is:

```text
obs:    96 = existing G1 actor observation minus 2-D gait phase
critic: 99 = existing G1 critic observation minus 2-D gait phase
amp:   195 = canonical 13-body AMP state
```

Changing the internal inherited `gait_phase` info value does not change any of
the three groups. The phase therefore has no policy, critic, AMP, reward, or
control authority in this task.

## Backend Boundary

`G1WalkEnvCfg.add_body_sensors` is an explicit cold-path option with default
`false`. `G1AMPWalkCfg` alone enables it. During env initialization, 13 body
names plus `torso_link` are resolved once through `SimBackend.get_body_ids`.
Each observation calls the declared `SimBackend.get_body_state_w` once and
never parses XML/assets or accesses a backend subclass/private array.

## Terminal Identity

`NpEnv` automatically stores every observation group before resetting done
rows, so `final_observation` now includes `amp`. The pure
`resolve_amp_transition_next` helper copies only terminal rows from
`final_observation["amp"]`, leaves collector actor-next input unchanged, and
fails closed if any terminal row lacks the AMP final group.

## Verification

```text
Focused unit plus legacy environment regression:
58 passed, 3 skipped

Real G1AMPWalk MuJoCo reset/one-step timeout:
1 passed, 5 deselected
```

The live test verified fixed commands, exact declared shapes, all-done timeout,
`final_observation` keys `obs/critic/amp`, terminal mask, and finite `(2,195)`
terminal AMP values. Ruff passed.

## Verdict

Step 4 is `PASS`. Policy current/next AMP observations have one owner, terminal
identity is explicit, gait phase is absent from all learning groups, and the
legacy task remains on its default no-body-tracking path.
