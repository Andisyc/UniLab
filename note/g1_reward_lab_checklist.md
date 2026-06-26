# G1 Reward Lab Checklist

## Current Status

Status: Core Reward Lab gating checks complete; transition recovery probe is
implemented as a diagnostic section; policy retraining and learned-policy
simulation are not complete.

Completed:

- [x] Reward Lab exists and runs on real MuJoCo G1 env.
- [x] Command routing/isolation checks pass for Standing and `vx = 0.1`
      Walking.
- [x] Standing candidate-quality checks pass.
- [x] Standing reward-preference checks pass.
- [x] Walking `vx = 0.1` candidate-quality checks pass.
- [x] Walking `vx = 0.1` reward-preference checks pass after per-mode reward
      scale overrides.
- [x] Per-term Walking contribution reporting identifies responsible terms.
- [x] Standing and Walking posture/stability terms are split by per-mode scale
      overrides.
- [x] Walking-to-Standing transition probe exists and uses same-state
      counterfactual candidate comparison.

Not complete:

- [ ] The hand-coded open-loop `forward_step` candidate still has worse
      x-displacement than open-close. This is diagnostic-only, not a Reward Lab
      gating failure.
- [ ] Walking-to-Standing transition recovery does not yet pass as a long-window
      stability claim; current open-loop candidates still fall over 40 steps.
- [ ] Long SAC retraining has not been launched from the corrected reward.
- [ ] Learned-policy simulation has not yet confirmed stable Standing,
      `vx = 0.1` forward stepping, and Walking-to-Standing recovery.

## 0. Purpose

This document is the working checklist for G1 standing/walking reward debugging.

The goal is not to prove that reward code runs. The goal is to prove that the
reward expresses the intended physical preference before launching long
training runs.

Every reward change must be checked in this order:

1. section ownership: which reward manifold owns the behavior;
2. mode isolation: whether the term is active only in the intended mode;
3. preference ordering: whether good archetypes score above bad archetypes;
4. live-path routing: whether Hydra/env/reset/step use the intended branch;
5. short rollout probe: whether reward-preferred candidates also behave better.

Implementation-only checks such as `pytest`, `ruff`, `py_compile`, and config
compose are necessary, but they are not reward validation.

## 0.1 Baseline Freeze

Status: frozen for diagnosis.

Current baseline scope:

- `conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml`
- `src/unilab/envs/locomotion/g1/joystick.py`
- `tests/config/test_reward_injection.py`
- `tests/envs/locomotion/g1/test_gait_constraint.py`
- `note/g1_reward_lab_checklist.md`

Freeze rule:

- [x] Stop ad-hoc reward tuning.
- [x] Do not change reward weights, reward terms, mode masks, command thresholds,
      or training distributions unless a Reward Lab preference check identifies
      the responsible section.
- [x] Treat current code as the diagnostic baseline, not as the final reward
      design.
- [x] Before the next reward code change, create or run a Reward Lab check that
      names the failing section and failing preference pair.
- [x] Every future reward change must update this document with the section,
      failed archetype, intervention, and validation result.

Known baseline questions:

- Does `vx = 0.1 m/s` Walking prefer real forward stepping over in-place
  foot opening/closing?
- Does Standing prefer deployable high stable stance over rearward-COM and
  low-crouch stances?
- Do shared terms such as `base_height`, `pose`, `penalty_feet_ori`, and
  `penalty_action_rate` suppress Walking swing or low-speed forward motion?

## 1. Core Design Contract

- [x] Treat Standing and Walking as two separate reward manifolds.
- [x] Do not assume Standing and Walking should share concrete posture terms.
- [x] Keep only minimal shared physical priors, such as catastrophic fall safety
      or broad torso sanity, when they do not suppress either manifold.
- [x] Separate action-related rewards by mode. Standing and Walking actions are
      not semantically equivalent.
- [x] Standing may reward quiet double-support stability.
- [x] Walking must reward step execution and forward displacement, including
      the first keyboard speed command `vx = 0.1 m/s`.
- [x] If a term affects both Standing and Walking, record why it is a shared
      invariant rather than a copied posture constraint.
- [x] If a term is useful only for diagnostics, do not silently keep it in the
      training reward.

Interpretation note:

The word "posture" in discussion means an abstract behavioral style or
locomotion manifold. It does not mean that Standing and Walking should have the
same joint-angle, foot, or action constraints.

## 2. Section 0: Command Contract

Intent:

- Keyboard commands are discrete velocity commands.
- One forward key press corresponds to `vx = 0.1 m/s`.
- Therefore `vx = 0.1 m/s` is a valid Walking command and must not be treated
  as Standing.

Checklist:

- [ ] `vx = 0.0` routes to Standing.
- [ ] `vx = 0.1` routes to Walking.
- [ ] `vx = 0.1` is expected to move forward, not merely show gait phase.
- [ ] `vx = 0.2+` routes to Walking.
- [ ] Yaw commands follow the same explicit command contract.
- [ ] Command thresholds do not erase valid keyboard increments.
- [ ] Interactive playback and training reset/resampling use the same command
      semantics.

Required preference tests:

- [ ] `walking_0p1_forward_step` scores above `walking_0p1_in_place_open_close`.
- [ ] `walking_0p1_forward_step` scores above `walking_0p1_dragging_feet`.
- [ ] `standing_zero_command` scores above `standing_zero_command_foot_lift`.

Evidence to record:

- command vector;
- `gait_enabled`;
- mode observation value;
- `stand_static_mask`, `stand_recovery_mask`, `walk_mask`;
- forward displacement over a short rollout.

## 3. Section 1: Standing Manifold

Intent:

Standing is a deployable, stable double-support manifold. It is not merely
"zero command" and not merely "high base height".

Target physical style:

- relatively high but not locked-out stance;
- knees slightly bent, not deep crouch;
- torso upright;
- base/COM centered inside the support region;
- feet at a deployable width and yaw;
- no stepping, no foot lifting, no constant soft-leg balancing;
- recovery returns to the same stable stance.

Checklist:

- [ ] Standing reward is owned by Standing-only terms or explicitly justified
      shared priors.
- [ ] Standing does not use Walking gait/action terms.
- [ ] Standing prefers deployable high stance over low crouch.
- [ ] Standing prefers centered COM over rearward COM.
- [ ] Standing prefers stable double support over foot-opening/foot-closing.
- [ ] Standing prefers quiet stability over continuous balance correction.
- [ ] Standing recovery can use action authority when needed, but static
      standing must not require continuous large actions.

Required archetypes:

- [ ] `standing_deployable_high`
- [ ] `standing_low_crouch`
- [ ] `standing_rearward_com`
- [ ] `standing_wide_feet`
- [ ] `standing_toe_in`
- [ ] `standing_soft_leg_oscillation`
- [ ] `standing_foot_lift`
- [ ] `standing_recovery_to_center`

Required preference tests:

- [ ] `standing_deployable_high > standing_low_crouch`
- [ ] `standing_deployable_high > standing_rearward_com`
- [ ] `standing_deployable_high > standing_wide_feet`
- [ ] `standing_deployable_high > standing_toe_in`
- [ ] `standing_deployable_high > standing_soft_leg_oscillation`
- [ ] `standing_recovery_to_center > standing_recovery_continues_drifting`

Evidence to record:

- base height;
- base/feet-center delta in base-yaw frame;
- foot width;
- foot sagittal offset;
- foot yaw;
- torso tilt;
- local base velocity;
- joint velocity;
- action L2/action rate;
- termination/fall signal;
- per-term reward contribution.

## 4. Section 2: Walking Manifold

Intent:

Walking is a locomotion manifold, not Standing plus a small velocity command.
It must produce forward step execution, including low-speed walking at
`vx = 0.1 m/s`.

Target physical style:

- tracks commanded velocity;
- lifts swing foot enough to avoid dragging;
- places feet to move the base forward;
- does not solve low-speed walking by opening/closing feet in place;
- keeps body style compatible with Walking, not static Standing.

Checklist:

- [ ] Walking reward does not include Standing-only quietness terms.
- [ ] Walking reward does not include Standing-only double-support terms.
- [ ] Walking can keep broad torso sanity only if it does not suppress swing.
- [ ] `vx = 0.1` has a valid forward-step preference.
- [ ] Swing-foot clearance is rewarded against dragging.
- [ ] Forward displacement is rewarded against in-place gait motion.
- [ ] High-speed Walking behavior remains compatible with the older successful
      Walking policy.

Required archetypes:

- [ ] `walking_0p1_forward_step`
- [ ] `walking_0p1_in_place_open_close`
- [ ] `walking_0p1_dragging_feet`
- [ ] `walking_0p1_static_stand`
- [ ] `walking_0p3_normal_gait`
- [ ] `walking_0p3_overconstrained_posture`

Required preference tests:

- [ ] `walking_0p1_forward_step > walking_0p1_static_stand`
- [ ] `walking_0p1_forward_step > walking_0p1_in_place_open_close`
- [ ] `walking_0p1_forward_step > walking_0p1_dragging_feet`
- [ ] `walking_0p3_normal_gait > walking_0p3_overconstrained_posture`

Evidence to record:

- commanded velocity;
- measured forward velocity;
- short-horizon forward displacement;
- left/right foot height relative to stance foot;
- foot contact sequence;
- gait phase;
- per-term reward contribution;
- action magnitude and action rate.

## 5. Section 3: Mode Isolation and Reward Pollution

Intent:

Standing and Walking can be learned by one network if the training distribution
and reward ownership are clear. The network can represent multiple manifolds,
but ambiguous reward gradients can still bias both behaviors.

Checklist:

- [ ] Standing-only reward terms are zero in Walking mode.
- [ ] Walking-only reward terms are zero in Standing mode.
- [ ] Shared terms are minimal and explicitly justified.
- [ ] Shared terms do not dominate mode-specific terms.
- [ ] Standing action penalties do not suppress Walking swing.
- [ ] Walking gait terms do not create stepping in Standing.
- [ ] Training batch contains enough clean Standing and clean Walking samples.
- [ ] Transition samples do not outnumber or blur the clean manifolds.

Required diagnostics:

- [ ] section reward totals by mode;
- [ ] term contribution table for Standing samples;
- [ ] term contribution table for Walking samples;
- [ ] gradient-risk review for terms shared by both modes;
- [ ] sample fraction report: Standing, Walking, Recovery, Transition.

Failure patterns:

- Walking has gait phase but no forward displacement: check Walking manifold and
  shared posture/action pollution.
- Standing keeps balance by moving feet: check Standing manifold and gait/action
  isolation.
- High-speed Walking works but `vx = 0.1` fails: check low-speed Walking
  archetypes, not command routing alone.

## 6. Section 4: Transition and Recovery

Intent:

Transition is not a separately trained policy. It is the same policy moving
between two clean manifolds under a clear command/mode observation.

Checklist:

- [ ] Standing to Walking starts from the deployable Standing manifold.
- [ ] Walking to Standing returns to the deployable Standing manifold.
- [ ] Recovery terms help return to center without becoming permanent walking.
- [ ] Transition samples do not redefine either Standing or Walking.
- [ ] Zero command after Walking freezes or returns gait phase appropriately.
- [ ] Walking command after Standing activates Walking reward immediately for
      `vx = 0.1`.

Required archetypes:

- [ ] `transition_stand_to_walk_0p1`
- [ ] `transition_stand_to_walk_0p3`
- [ ] `transition_walk_to_stand_recover`
- [ ] `transition_walk_to_stand_continues_stepping`
- [ ] `transition_walk_to_stand_falls`

Required preference tests:

- [ ] `transition_stand_to_walk_0p1_forward > transition_stand_to_walk_0p1_open_close`
- [ ] `transition_walk_to_stand_recover > transition_walk_to_stand_continues_stepping`
- [ ] `transition_walk_to_stand_recover > transition_walk_to_stand_falls`

Evidence to record:

- command switch time;
- mode masks before and after switch;
- base velocity decay after stop command;
- foot contacts after stop command;
- base/feet-center delta after stop command;
- termination/fall status.

## 7. Reward Lab Output Format

Every Reward Lab run should output:

- section name;
- archetype name;
- command;
- mode masks;
- total reward;
- section reward;
- per-term contribution;
- short-rollout behavior metrics;
- pass/fail preference checks.

Minimum output table:

```text
section | archetype | total | standing | walking | shared | x_disp | max_tilt | min_height | pass
```

Do not accept a run as reward validation if it only reports that code executed.

## 8. Current Open Questions

- [ ] Which current shared terms should remain shared between Standing and
      Walking?
- [ ] Should `base_height`, `pose`, `penalty_feet_ori`, and `penalty_action_rate`
      be split into Standing-specific and Walking-specific versions?
- [ ] What is the deployable Standing target height and knee bend from real
      deployment constraints?
- [ ] What minimum swing clearance is required for `vx = 0.1` without causing
      excessive stepping?
- [ ] What Standing/Walking sample ratio best preserves two clean manifolds?
- [ ] Should transition samples be a small bridge distribution rather than a
      large fraction of training?

## 9. Next Implementation Checklist

- [x] Implement Reward Lab section reporter.
- [x] Add real MuJoCo archetype constructors for Standing.
- [x] Add real MuJoCo archetype constructors for low-speed Walking.
- [ ] Add counterfactual short-rollout probes for Standing recovery.
- [ ] Add counterfactual short-rollout probes for Walking `vx = 0.1`.
- [ ] Add config/report output so failed preference checks identify the
      responsible reward terms.
- [ ] Only after these checks pass, launch long SAC training.

## 10. MVP Run Record

Command:

```bash
uv run scripts/deploy/check_unilab_g1_reward_lab.py --steps 40
```

Result:

- [x] Script compiles.
- [x] `ruff` passes.
- [x] Isolation checks pass:
  - Standing sample has `walk_mask = 0`.
  - Walking `vx = 0.1` sample has `walk_mask = 1`.
  - Standing sample has zero Walking subtotal.
  - Walking sample has zero Standing subtotal.
- [ ] Standing preference checks do not pass yet.
- [ ] Walking `vx = 0.1` preference checks do not pass yet.

Interpretation:

- This is a useful first microscope, but not yet a final Reward Lab.
- The current hand-coded `standing_deployable_high` candidate tilts too much in
  short rollout, so Standing archetype constructors must be improved before
  using this as a reward-design verdict.
- The current `walking_0p1_forward_step` candidate does not produce clean
  forward displacement, so low-speed Walking archetypes must be improved before
  using this as a reward-design verdict.
- The isolation result is already meaningful: direct Standing/Walking mode masks
  and mode-specific subtotals are separated in this probe.

## 11. Candidate Quality Run Record

Commands:

```bash
uv run python -m py_compile scripts/deploy/check_unilab_g1_reward_lab.py
uv run ruff check scripts/deploy/check_unilab_g1_reward_lab.py
uv run scripts/deploy/check_unilab_g1_reward_lab.py --section standing --steps 40
uv run scripts/deploy/check_unilab_g1_reward_lab.py --section walking_0p1 --steps 40
uv run scripts/deploy/check_unilab_g1_reward_lab.py --section isolation --steps 40
```

Result:

- [x] Script compiles.
- [x] `ruff` passes.
- [x] Standing candidate-quality checks pass:
  - default stand keyframe remains upright.
  - low crouch is lower than default stand.
  - rearward candidate moves base/feet-center x away from neutral.
  - wide-feet candidate increases stance width.
- [x] Standing reward-preference checks pass:
  - default/deployable standing is preferred over low crouch.
  - default/deployable standing is preferred over rearward COM.
  - default/deployable standing is preferred over wide feet.
- [x] Walking `vx = 0.1` candidate-quality checks pass:
  - forward-step has more swing clearance than dragging.
  - open-close ends wider than static standing.
  - forward-step moves more than static standing in this short rollout.
- [ ] Walking `vx = 0.1` reward-preference checks still fail:
  - forward-step is not preferred over static standing.
  - forward-step is not preferred over open-close.
  - forward-step is not preferred over dragging feet.
  - open-close still has better x displacement than forward-step in this
    uncontrolled short rollout.
- [x] Isolation checks pass:
  - Standing sample has `walk_mask = 0`.
  - Walking `vx = 0.1` sample has `walk_mask = 1`.
  - Standing sample has zero Walking subtotal.
  - Walking sample has zero Standing subtotal.

Interpretation:

- Reward Lab can now separate candidate-construction failure from reward
  preference failure.
- Standing is no longer the immediate Lab failure: the current reward prefers
  the default deployable standing keyframe over low crouch, rearward COM, and
  wide stance under this probe.
- Low-speed Walking is the concrete failure: once candidate labels are clean,
  current reward still prefers static/open-close/dragging alternatives over the
  forward-step archetype.
- The next Reward Lab step should report per-term Walking contributions, because
  the failure is now localized to low-speed Walking preference rather than mode
  routing or Standing candidate construction.

## 12. Walking Per-Term Contribution Run Record

Commands:

```bash
uv run python -m py_compile scripts/deploy/check_unilab_g1_reward_lab.py
uv run ruff check scripts/deploy/check_unilab_g1_reward_lab.py
uv run scripts/deploy/check_unilab_g1_reward_lab.py --section walking_0p1 --steps 40
```

Result:

- [x] Script compiles.
- [x] `ruff` passes.
- [x] Reward Lab now reports cumulative per-term contributions over the rollout,
      not only last-step terms.
- [x] Walking `vx = 0.1` candidate-quality checks still pass.
- [ ] Walking `vx = 0.1` reward-preference checks still fail.

Observed cumulative totals over 40 steps:

| Candidate | total | walking | gait | x displacement |
| --- | ---: | ---: | ---: | ---: |
| `walking_0p1_forward_step` | 5.357 | 6.303 | -0.946 | -0.450 |
| `walking_0p1_static_stand` | 7.984 | 8.924 | -0.940 | -0.671 |
| `walking_0p1_dragging_feet` | 9.035 | 9.857 | -0.822 | -0.575 |
| `walking_0p1_in_place_open_close` | 9.734 | 10.519 | -0.785 | 0.185 |

Main term deltas, shown as `forward_step - alternative`; negative means the
alternative is favored by that term:

| Alternative | main negative deltas |
| --- | --- |
| static stand | `pose=-1.774`, `upright=-0.650`, `penalty_orientation=-0.559`, `penalty_ang_vel_xy=-0.446` |
| open-close | `pose=-1.399`, `upright=-1.147`, `penalty_orientation=-0.832`, `tracking_ang_vel=-0.329`, `penalty_ang_vel_xy=-0.310` |
| dragging feet | `pose=-1.609`, `upright=-0.799`, `penalty_orientation=-0.650`, `penalty_ang_vel_xy=-0.419`, `tracking_ang_vel=-0.312` |

Important non-culprits:

- `feet_phase` is not the main cause in this probe. It slightly favors
  `forward_step` over open-close and dragging feet.
- `tracking_lin_vel` also favors `forward_step` over static stand and dragging,
  but its magnitude is too small to overcome the posture/stability penalties.
- `gait_constraint` hurts `forward_step` more than dragging by about `-0.124`,
  but this is much smaller than the `pose` and orientation/upright terms.

Interpretation:

- The low-speed Walking failure is now localized to shared Walking posture and
  stability terms dominating over forward-progress / gait evidence.
- `pose`, `upright`, `penalty_orientation`, and `penalty_ang_vel_xy` make
  conservative/open-close/dragging samples look better than the actual
  forward-step archetype.
- The next design step should not be to blindly increase gait reward. It should
  split or reweight the Walking posture/stability terms so they do not suppress
  low-speed stepping, while preserving Standing posture terms separately.

## 13. Walking/Standing Posture Split Run Record

Design change:

- Add per-mode reward scale overrides to `RewardModeConfig`:
  - `stand_scale_overrides`
  - `stand_recovery_scale_overrides`
  - `walk_scale_overrides`
- Keep old behavior as the default when overrides are empty.
- Configure SAC G1 MuJoCo so Standing keeps the strong common posture/stability
  scales, while Walking uses lighter posture/stability scales:
  - `pose: 0.0`
  - `penalty_action_rate: -0.5`
  - `upright: 0.25`
  - `penalty_orientation: -0.5`
  - `penalty_ang_vel_xy: -0.05`
  - `base_height: -5.0`
  - `penalty_feet_ori: -1.0`
  - `tracking_ang_vel: 0.3`
  - `feet_phase: 12.0`
- Keep `feet_phase` as the main low-speed Walking gait evidence.
- Do not use `feet_phase_contact` / `feet_phase_contrast` as direct Walking
  rewards in this probe, because they favored static/open-close/dragging
  candidates under the current open-loop archetypes.
- Reduce gait-constraint bridge penalty from `2.0` to `0.5`; it remains a light
  shape constraint instead of overpowering low-speed stepping.

Commands:

```bash
uv run pytest tests/config/test_reward_injection.py -q
uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py -q -k "mode_scale_overrides or common_base_height or reward_mode_dispatch"
uv run ruff check scripts/deploy/check_unilab_g1_reward_lab.py src/unilab/envs/locomotion/g1/joystick.py tests/config/test_reward_injection.py tests/envs/locomotion/g1/test_gait_constraint.py
uv run scripts/deploy/check_unilab_g1_reward_lab.py --steps 40
```

Result:

- [x] Config injection tests pass.
- [x] Mode reward dispatch / per-mode override tests pass.
- [x] Ruff passes for touched Python files.
- [x] Full Reward Lab gating checks pass.
- [x] Standing candidate-quality and reward-preference checks still pass.
- [x] Walking `vx = 0.1` candidate-quality checks still pass.
- [x] Walking reward-preference checks now pass:
  - `forward_step > static_stand`
  - `forward_step > in_place_open_close`
  - `forward_step > dragging_feet`
- [x] Isolation checks still pass.
- [ ] The open-loop x-displacement diagnostic still fails:
  - `forward_step` has lower x displacement than open-close under this hand-coded
    open-loop probe.
  - This diagnostic is no longer a gating Reward Lab failure because it measures
    open-loop archetype dynamics, not reward preference.

Observed Walking totals after the split:

| Candidate | total | walking | gait | x displacement |
| --- | ---: | ---: | ---: | ---: |
| `walking_0p1_forward_step` | 10.502 | 10.738 | -0.236 | -0.450 |
| `walking_0p1_static_stand` | 10.342 | 10.577 | -0.235 | -0.671 |
| `walking_0p1_dragging_feet` | 10.456 | 10.661 | -0.206 | -0.575 |
| `walking_0p1_in_place_open_close` | 10.456 | 10.652 | -0.196 | 0.185 |

Interpretation:

- The previous low-speed Walking reward pollution is fixed at the Reward Lab
  preference level.
- Walking is no longer dominated by default-pose/upright/orientation objectives.
- Standing still keeps the stronger posture and stability objective.
- The remaining risk is policy training, not immediate Reward Lab preference:
  after retraining, verify whether the learned controller converts the corrected
  reward preference into real low-speed forward stepping.

## 14. Walking-to-Standing Transition Probe Run Record

Design change:

- Add a `transition_recovery` Reward Lab section.
- Warm up with `vx = 0.2`, then switch the command to zero.
- Compare recovery candidates from the same copied switch state, rather than
  from independently randomized env states.
- Keep this section diagnostic-only for now; default `--steps` Reward Lab gating
  still covers mature Standing, Walking, and isolation sections.

Commands:

```bash
uv run python -m py_compile scripts/deploy/check_unilab_g1_reward_lab.py
uv run ruff check scripts/deploy/check_unilab_g1_reward_lab.py
uv run scripts/deploy/check_unilab_g1_reward_lab.py --steps 40
uv run scripts/deploy/check_unilab_g1_reward_lab.py --section transition_recovery --steps 40
```

Result:

- [x] Script compiles.
- [x] `ruff` passes.
- [x] Default Reward Lab gating still passes.
- [x] Transition probe confirms routing at switch time:
  - switch state is not fallen: `switch_tilt_deg = 3.18`.
  - zero command enters Standing recovery: `initial_stand_recovery_mask = 1.0`.
- [x] Candidate comparison now starts from a shared physical switch state.
- [ ] Long-window transition recovery is not solved by current open-loop
      candidates:
  - `transition_walk_to_stand_zero_action` terminates at step 39.
  - `transition_walk_to_stand_taper_gait` terminates at step 38.
  - all candidates reach about `70 deg` peak tilt over 40 post-switch steps.

Observed 40-step totals after same-state synchronization:

| Candidate | total | recovery | first terminated | max tilt |
| --- | ---: | ---: | ---: | ---: |
| `transition_walk_to_stand_zero_action` | -33.913 | -33.842 | 39 | 69.95 |
| `transition_walk_to_stand_taper_gait` | -39.105 | -39.044 | 38 | 70.76 |
| `transition_walk_to_stand_continue_stepping` | -37.946 | -37.935 | 38 | 70.70 |
| `transition_walk_to_stand_open_close` | -40.033 | -40.028 | 37 | 70.80 |
| `transition_walk_to_stand_dragging` | -37.962 | -37.952 | 38 | 70.71 |

Interpretation:

- The previous apparent preference for open-close during transition was partly
  caused by comparing candidates from different switch states.
- Routing is not the current transition failure: command switch and recovery
  mask activation both work.
- The remaining transition issue is a real recovery-control problem: current
  open-loop recovery candidates cannot keep the robot upright over 40 steps
  after a Walking-to-Standing switch.
- Next reward/debug step should target Standing recovery action authority and
  recovery archetype quality, not Walking reward routing.

## 15. Standing Recovery Authority Run Record

Design change:

- Add a dedicated `standing_recovery` Reward Lab section.
- Split the check into two sub-sections:
  - `standing_static_authority`: static Standing should prefer quiet zero action.
  - `standing_recovery_authority`: disturbed zero-command Standing should allow a
    bounded feedback recovery action.
- Add a same-state velocity disturbance probe with zero command:
  - `lin_xy = [0.35, 0.08]`
  - `yaw = 0.25`
- Add a feedback recovery archetype that uses base/feet-center and local velocity
  feedback only inside the Lab probe.
- Add Standing-recovery-only scale overrides:
  - `stand_lin_vel_xy_l2: -6.0`
  - `stand_yaw_vel_l2: -2.0`
  - `base_height: -35.0`
  - `pose: -0.25`
  - `penalty_feet_ori: -8.0`
  - `stand_tilt_l2: -55.0`
  - `stand_tilt_margin_l2: -160.0`

Commands:

```bash
uv run scripts/deploy/check_unilab_g1_reward_lab.py --section standing_recovery --steps 40
uv run scripts/deploy/check_unilab_g1_reward_lab.py --steps 40
uv run pytest tests/config/test_reward_injection.py -q
uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py -q -k "mode_scale_overrides or common_base_height or reward_mode_dispatch"
uv run ruff check scripts/deploy/check_unilab_g1_reward_lab.py src/unilab/envs/locomotion/g1/joystick.py tests/config/test_reward_injection.py tests/envs/locomotion/g1/test_gait_constraint.py
```

Result:

- [x] Static Standing still prefers quiet zero action:
  - `standing_static_zero_action = 9.078`
  - `standing_static_brace_action = 3.891`
  - `standing_static_soft_leg_oscillation = 3.384`
- [x] Disturbed Standing routes to Standing recovery:
  - `standing_recovery_mask_active = 1.0`
- [x] Feedback recovery is physically cleaner than zero action:
  - max tilt `18.35 deg` vs `23.77 deg`.
- [x] Standing recovery reward now prefers feedback recovery over bad recovery
      alternatives:
  - `feedback = 5.055`
  - `zero_action = 4.841`
  - `open_close = 3.687`
  - `brace = 1.467`
  - `keep_stepping = 1.095`
- [x] Default Reward Lab gating still passes.
- [x] Config injection tests pass.
- [x] Mode reward dispatch / override tests pass.
- [x] Ruff passes.
- [ ] Learned-policy retraining and simulation are still required.

Interpretation:

- Static Standing and Standing recovery now express different action authority:
  static Standing rewards quietness, while recovery can reward bounded corrective
  action.
- The recovery fix is scoped to `stand_recovery_scale_overrides`; it does not
  alter Walking reward routing or static Standing terms.
- The remaining question is whether SAC learns this distinction in policy space.
