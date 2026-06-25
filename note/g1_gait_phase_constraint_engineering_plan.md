# G1 Gait Phase Constraint Engineering Plan

日期：2026-06-23

目的：为 UniLab G1 locomotion 中 `gait_phase` / `feet_phase` 从正向 reward 迁移为步态约束提供工程方案。本文只设计 UniLab 内部改法，不涉及 RoboJudo_Real 部署层。

## 1. 核心判断

`gait_phase` 是 walking style prior，不是 locomotion task objective。

当前 G1 SAC 配置中 `feet_phase` 是正向 reward，且权重高于速度跟踪损失的实际支配力。policy 可以通过原地踏步或慢速漂移打开 gait reward 通道，获得稳定高 Q。这个 failure 不是简单阈值问题，而是 reward 的概念边界错位：

- task reward 应回答：是否响应 command，是否稳定，是否按目标速度/角速度运动。
- gait phase 应回答：在 walking mode 下，脚部摆动/接触结构是否像一个合理步态。
- gait phase 不应回答：机器人是否应该移动，移动多快，或者踏步本身是否值得奖励。

因此目标不是继续调 `feet_phase` 权重，而是把它从 reward owner 中移出，变成 command-conditioned 的 gait violation / gait cost。

## 2. UniLab 设计约束

本方案遵守 UniLab 当前设计原则：

- Contract first：先定义 env 输出和 learner 消费的 contract，再改算法。
- Fix at owner layer：G1 步态结构属于 `src/unilab/envs/locomotion/g1/`，不能塞进 `scripts/`。
- Config first：启用方式、阈值、权重、是否 constrained SAC 由 Hydra YAML 表达。
- Backend isolation：步态 violation 只使用 `SimBackend` 已有传感器访问和 G1 env 已声明的 sensor 名称，不在热路径解析 asset/XML。
- Validate near risk：在 G1 env、replay buffer、FastSAC learner 边界分别补测试，而不是只跑顶层训练。

## 3. 现有代码边界

当前相关 owner：

- `src/unilab/envs/locomotion/g1/joystick.py`
  - `compute_feet_phase_height_targets()`
  - `compute_feet_phase_contact_targets()`
  - `G1RewardConfig`
  - `_reward_feet_phase()`
  - `_reward_feet_phase_contrast()`
  - `_reward_feet_phase_contact()`
  - `_gait_reward_gate()`
- `src/unilab/envs/locomotion/common/rewards.py`
  - shared task reward functions and `RewardContext`
  - `run_reward_dispatch()`
- `src/unilab/base/np_env.py`
  - `NpEnvState` currently exposes scalar `reward`, no first-class `cost` field.
- `src/unilab/ipc/replay_buffer.py`
  - packed replay row stores obs, next_obs, actions, rewards, dones, truncated, critic obs.
  - no cost/constraint column.
- `src/unilab/algos/torch/offpolicy/worker.py`
  - collector extracts `state.reward` and writes only `rewards` into replay.
- `src/unilab/algos/torch/fast_sac/learner.py`
  - critic target and actor loss consume only scalar `rewards`.

This means a fully clean constraint implementation needs an algorithm-facing cost channel. A minimal compatible implementation can first write gait violation into `info["log"]` and subtract a gated cost from reward, but that is an implementation bridge, not the final concept.

## 4. Concept Contract

Define three distinct quantities:

### 4.1 Task reward

Task reward owns locomotion success:

- command tracking: xy velocity and yaw velocity tracking
- stability: upright, height, angular velocity, no early termination
- stand mode: when command is zero, remain still and near default pose

Task reward must not include positive gait phase reward.

### 4.2 Gait generator truth

The gait generator provides procedural ground truth:

- phase -> expected swing foot height
- phase -> expected left/right contact state
- phase -> expected left/right foot height contrast

This is not mocap ground truth like AMP. It is analytic ground truth, valid only under walking mode.

### 4.3 Gait violation

Gait violation measures deviation from the procedural gait truth:

```text
height_violation = squared foot height error relative to phase target
contact_violation = mismatch with phase contact schedule
contrast_violation = mismatch between actual and target left-right height delta
```

Gait violation is a non-negative cost. Lower is better. It never creates positive reward.

## 5. Walking Mode Contract

Walking mode must be driven by command, not by policy-generated body velocity.

Recommended gate:

```text
command_active =
  norm(command_xy) > command_xy_threshold
  or abs(command_yaw) > command_yaw_threshold
```

Optional stricter tracking gate for applying gait pressure:

```text
tracking_ok =
  exp(-||command_xy - linvel_xy||^2 / sigma) > tracking_threshold
  or warmup allows weak pressure
```

Important rule:

- `actual_speed > threshold` must not be the sole gait gate.
- Otherwise policy can create slow drift to unlock gait reward/cost dynamics.

Zero command mode:

- gait violation can be logged for diagnostics, but should not affect actor objective by default.
- stand-still penalty / pose regularization owns behavior.

Walking command mode:

- task reward drives motion.
- gait violation constrains style.

## 6. Phase 1: Env-Owned Gait Violation Refactor

Owner file: `src/unilab/envs/locomotion/g1/joystick.py`

Add a G1-local config object:

```python
@dataclass
class GaitConstraintConfig:
    enabled: bool = False
    command_xy_threshold: float = 0.05
    command_yaw_threshold: float = 0.05
    height_weight: float = 1.0
    contrast_weight: float = 1.0
    contact_weight: float = 1.0
    epsilon: float = 0.02
    penalty_scale: float = 1.0
    apply_when_tracking: bool = False
    tracking_threshold: float = 0.3
```

Attach it to `G1RewardConfig` or `G1WalkEnvCfg`. Preferred location:

- If the value affects environment objective semantics, put it under `reward.gait_constraint`.
- If it becomes a first-class constrained RL signal, later move the learner-specific lambda settings under `algo.constraint`.

Add helper functions in `g1/joystick.py`:

```python
compute_command_active_mask(commands, xy_threshold, yaw_threshold)
compute_gait_phase_height_violation(...)
compute_gait_phase_contrast_violation(...)
compute_gait_phase_contact_violation(...)
compute_gait_constraint_cost(ctx) -> dict[str, np.ndarray]
```

Keep these G1-local because they depend on biped foot sensors, foot names, and phase semantics.

Refactor existing positive reward functions:

- Keep old `_reward_feet_phase*` only for ablation compatibility.
- Introduce cost functions with names like `_cost_gait_phase_height`, `_cost_gait_phase_contact`.
- Default active G1 SAC configs should set old positive `feet_phase*` scales to `0.0`.

Logging:

```text
info["log"]["constraint/gait_height"] = mean(height_violation)
info["log"]["constraint/gait_contact"] = mean(contact_violation)
info["log"]["constraint/gait_total"] = mean(total_violation)
info["log"]["mode/command_active"] = mean(command_active)
```

## 7. Phase 2: Compatible Reward Bridge

This is the smallest runnable bridge before adding first-class constrained SAC.

In `G1WalkEnv._compute_reward()`:

```text
task_reward = run_reward_dispatch(task reward scales)
gait_cost = command_active * max(total_violation - epsilon, 0)
reward = task_reward - penalty_scale * gait_cost * ctrl_dt
```

Rules:

- `gait_cost` is non-negative.
- `gait_cost` is zero in zero-command mode unless explicitly configured otherwise.
- `gait_cost` has no positive counterpart.
- `penalty_scale` should start small because this bridge still folds cost into scalar reward.

This stage is not the final architecture, but it is useful for quick validation because it does not change `NpEnvState`, replay buffer, or learner.

## 8. Phase 3: First-Class Constraint Contract

Long-term clean design:

### 8.1 Env state

Extend `NpEnvState` with optional cost dict or scalar cost:

```python
cost: dict[str, np.ndarray] | None = None
```

or minimally:

```python
constraint_cost: np.ndarray | None = None
```

Prefer dict if UniLab expects future constraints beyond gait:

```text
cost["gait"] -> shape (num_envs,)
cost["energy"] -> optional future constraint
cost["safety"] -> optional future constraint
```

### 8.2 Replay buffer

Extend `ReplayBuffer` packed layout with optional cost columns:

```text
obs, next_obs, action, reward, cost, done, truncated, critic, next_critic
```

Make cost storage opt-in:

- default `cost_dim=0` preserves existing algorithms.
- constrained algorithms request `cost_dim > 0`.

### 8.3 Collector

In `offpolicy/worker.py`, extract cost from state:

```text
cost_np = state.cost["gait"] or zeros
replay_buffer.add(..., costs=cost_np)
```

### 8.4 Learner

Add a constrained SAC variant rather than silently changing standard FastSAC:

- file option: `src/unilab/algos/torch/constrained_sac/`
- or subclass/flag in `fast_sac` only if the code remains clean.

Learner owns:

```text
Q_task: reward critic
Q_cost: cost critic
lambda: Lagrange multiplier
```

Actor objective:

```text
maximize Q_task - lambda * Q_cost + entropy term
```

Lambda update:

```text
lambda <- max(0, lambda + lr * (mean_cost - cost_limit))
```

Config:

```yaml
algo:
  algo: constrained_sac
  constraint:
    names: ["gait"]
    cost_limit: 0.02
    lambda_init: 0.1
    lambda_lr: 3.0e-4
```

This keeps standard SAC behavior stable and makes constrained SAC an explicit algorithm choice.

## 9. Config Migration

Initial target config: `conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml`

Change active SAC G1 locomotion config toward:

```yaml
reward:
  scales:
    tracking_lin_vel: 2.0
    tracking_ang_vel: 1.5
    under_speed: -0.5
    penalty_ang_vel_xy: -1.0
    penalty_orientation: -10.0
    penalty_action_rate: -4.0
    pose: -0.5
    penalty_feet_ori: -20.0
    feet_phase: 0.0
    feet_phase_contrast: 0.0
    feet_phase_contact: 0.0
    alive: 2.0
  gait_constraint:
    enabled: true
    command_xy_threshold: 0.05
    command_yaw_threshold: 0.05
    epsilon: 0.02
    penalty_scale: 0.5
```

Command distribution should also be made explicit in the SAC task YAML:

```yaml
env:
  commands:
    vel_limit:
      - [0.2, -0.2, -0.4]
      - [0.8, 0.2, 0.4]
```

For joystick deploy policy, training should include the command range expected at deployment. Avoid relying on default `Commands.vel_limit`.

## 10. Tests

Add tests near the risk boundary:

### 10.1 Gait generator unit tests

Target: `tests/envs/locomotion/test_g1_gait_constraint.py`

Check:

- height targets are phase-periodic.
- offset phase `[phi, phi + pi]` produces alternating left/right targets.
- contact target matches height threshold.

### 10.2 Command gate tests

Check:

- zero command -> `command_active == false`.
- small drift in body velocity with zero command does not activate gait constraint.
- non-zero yaw command activates walking mode even if xy command is zero.

### 10.3 Env reward/cost tests

Instantiate `G1WalkFlat` with small `num_envs`.

Check:

- old positive gait reward can be disabled by config.
- `info["log"]["constraint/gait_total"]` appears when enabled.
- zero command does not produce gait cost pressure by default.
- non-zero command produces non-negative gait cost.

### 10.4 Replay and learner tests for Phase 3

Only after first-class cost channel is implemented:

- `ReplayBuffer(cost_dim=1)` stores and samples `costs`.
- constrained SAC actor loss includes `lambda * cost_q`.
- lambda increases when sampled cost is above limit and decreases/holds when below limit.

## 11. Rollout Diagnostics

During training, always log:

```text
reward/tracking_lin_vel
reward/tracking_ang_vel
reward/under_speed
constraint/gait_total
constraint/gait_height
constraint/gait_contact
mode/command_active
diagnostic/body_speed_when_command_zero
diagnostic/mean_abs_vx_error
diagnostic/zero_command_foot_height
```

The key anti-hacking diagnostic:

```text
zero command + nonzero body drift + foot swing
```

This should trend down early. If it appears early and persists, the gait constraint is still coupled to a policy-controllable gate.

## 12. Recommended Implementation Order

1. Add gait constraint note and tests for current helper behavior.
2. Add command-active gate and gait violation helpers in `g1/joystick.py`.
3. Add Phase 2 compatible reward bridge behind `reward.gait_constraint.enabled=false` by default.
4. Create a new SAC G1 config variant enabling the bridge and disabling positive `feet_phase*`.
5. Run short training/rollout to confirm zero-command no longer steps in place.
6. If the bridge works, implement Phase 3 first-class cost channel and constrained SAC.
7. Promote the constrained config to the default only after diagnostics show no early slow-drift hacking.

## 13. Success Criteria

The fix is successful only if all are true:

- Zero command produces standing behavior, not phase-following foot swing.
- Slow body drift under zero command does not unlock gait objective.
- Non-zero command produces movement because tracking reward is high, not because phase reward is high.
- Gait phase improves contact/height regularity only after walking is commanded.
- Deployment policy responds to joystick command without reverting to in-place stepping.

## 14. 2026-06-23 Implementation Status

Implemented Phase 1 and the Phase 2 compatible bridge:

- Added `GaitConstraintConfig` to the G1 reward config contract.
- Added command-driven walking-mode gate helpers. The gate is based on external command, not policy-generated body velocity.
- Added G1-local gait violation helpers for height, contrast, and contact schedule.
- Added a bridge in `G1WalkEnv._compute_reward()`:

```text
reward = task_reward - penalty_scale * command_gate * max(gait_violation - epsilon, 0) * ctrl_dt
```

- Updated `conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml` to disable positive `feet_phase*` rewards, reduce `alive`, add `under_speed`, make command range explicit, and enable the gait constraint bridge.
- Added tests for command gating, procedural gait truth, dict-to-dataclass conversion, zero-command drift resistance, and nonzero-command gait cost.

Not implemented yet:

- First-class `NpEnvState.cost`.
- Replay buffer cost columns.
- Constrained SAC cost critic and Lagrange multiplier.

Current implementation is therefore a validated bridge, not the final constrained RL architecture.

## 15. Non-Goals

- Do not change RoboJudo_Real for this problem.
- Do not solve natural locomotion style with AMP in this iteration.
- Do not introduce asset/XML parsing in hot path.
- Do not silently change standard FastSAC semantics for all tasks.
- Do not make `actual_speed` the only gate for gait pressure.

## 16. 2026-06-24 Stand/Low-Speed/Turn Follow-Up

Training observation after the Phase 2 bridge:

- In-place stepping is removed.
- Zero command still produces standing jitter.
- Visible stepping appears only for larger forward commands, around `0.5 m/s`.
- Turning has a waist/upper-body-first tendency.
- Backward walking is out of distribution and falls after a few steps.

Interpretation:

- The previous fix removed gait reward hacking but did not define a separate stand mode.
- The SAC command range `[0.2, 0.8]` makes low joystick increments near or outside the training distribution.
- G1 command sampling still zeroes small xy commands with the old `0.2` threshold, which removes low-speed walking examples.
- Hip yaw is part of the leg pose prior. Penalizing it too strongly can make yaw tracking avoid the intended leg yaw freedom and use upper-body/base alternatives.
- Backward commands require either an explicit "unsupported" deployment boundary or a training distribution that includes negative `vx`.

Engineering follow-up:

- Add a command sampling threshold field so G1 can keep low-speed command samples.
- Add stand-mode samples through `commands.rel_standing_envs`.
- Freeze gait phase while command is inactive so zero-command observations are not driven by a moving phase signal.
- Add stand-mode rewards/penalties for joint deviation, action magnitude, and joint velocity.
- Reduce hip yaw pose weights in the active SAC G1 config so turning can use leg yaw before waist/upper-body twist.
- Include modest negative `vx` samples for initial backward support, but treat backward stability as a separate diagnostic.

## 17. 2026-06-24 Low-Speed Gait Authority Follow-Up

Training observation after the stand/low-speed follow-up:

- Zero command no longer falls, but still has small standing jitter.
- Forward walking appears clearly only around `0.4 m/s`.
- Backward walking starts earlier, around `0.2 m/s`.
- Yaw commands also need around `0.2 rad/s` before visible turning.
- Forward/backward and yaw behaviors can lose the intended alternating gait: one foot steps and the other follows.
- Turning still has an upper-body-first tendency.
- No falling is observed in this run.

Interpretation:

- Stability is no longer the primary blocker; the remaining issue is mode authority.
- Stand mode is active, but the stand smoothness penalties are too weak to fully suppress small action and joint-velocity jitter.
- `tracking_sigma=0.25` is too broad for low-speed joystick commands. At `0.1 m/s`, staying still is still rewarded too well, so the policy has little incentive to leave the stand basin.
- The Phase 2 bridge made gait a negative constraint, which prevents in-place reward hacking, but it does not positively attract the policy toward a clean alternating gait.
- The proper next move is not to restore positive `feet_phase*` rewards. That would reopen the original hacking channel.
- Instead, strengthen the command-active gait violation cost, especially contrast/contact terms, so follow-step and upper-body-first turning become more expensive.
- Hip yaw still appears underused; the pose prior should continue to avoid penalizing hip yaw as a default-pose deviation in this command-active locomotion regime.

Engineering follow-up:

- Narrow velocity tracking with `tracking_sigma=0.12` so low-speed command errors have meaningful reward gradient.
- Strengthen stand smoothness penalties, especially action and joint velocity, while keeping stand pose deviation moderate to avoid suppressing takeoff.
- Increase `gait_constraint.contrast_weight`, `contact_weight`, and `penalty_scale`; set `epsilon=0.0` so command-active gait errors are not ignored.
- Keep positive `feet_phase*` rewards disabled.
- Further relax hip-yaw pose weights to near zero in the active SAC G1 config.
- Validate by config tests rather than a top-level script-only check, because this fix is an owner-YAML objective contract change.

## 18. 2026-06-24 Stand Phase Must Be Double Stance

Training observation after Section 17:

- Low-speed forward/backward/yaw behavior improves.
- Standing stability and no-fall behavior remain acceptable.
- In-place stepping returns, which violates the original stand-mode contract.

Diagnosis:

- The failure is not caused by restoring positive `feet_phase*` rewards; those remain disabled.
- The stand-mode phase itself is wrong.
- With the current generator, phase `0` maps to a swing-height target, while phase `pi` maps to a stance target.
- Therefore `stand_phase: [0, pi]` is not a neutral standing phase. It tells the actor that one foot is in swing even when command is zero.
- Freezing a walking phase is still a walking cue; stand mode needs a real double-stance phase.

Engineering follow-up:

- Change the default and active SAC G1 `stand_phase` to `[pi, pi]`, so both feet map to stance/contact targets.
- Add a unit test that proves the configured stand phase produces zero height targets and contact targets for both feet.
- Keep the Section 17 low-speed gait authority changes in place unless the next training run shows a separate regression.

## 19. 2026-06-24 Stand Mode Needs An Active Stance Constraint

Training observation after Section 18:

- The policy still learns in-place stepping after retraining.
- The behavior looks similar to the original hacking mode.
- This means the previous fix corrected the observation phase semantics, but did not remove the early-training oscillator attractor.

Diagnosis:

- A frozen double-stance observation is necessary but not sufficient.
- Once a cyclic stepping policy is discovered early, action-history feedback and body-state feedback can sustain the oscillator even when the command is zero.
- The stand-mode reward terms penalize action, joint velocity, and joint deviation, but they do not directly say "both feet must stay in stance/contact".
- The gait constraint was explicitly disabled in stand mode (`apply_in_stand_mode: false`), so zero-command stepping did not pay the double-stance contact/height violation cost.
- This is the missing boundary: gait is a locomotion structure in walk mode, but double-stance is a stance constraint in stand mode.

Engineering follow-up:

- Keep positive `feet_phase*` rewards disabled.
- Keep `stand_phase: [pi, pi]`.
- Enable `gait_constraint.apply_in_stand_mode` in the active SAC G1 owner config, so inactive-command samples enforce double stance/contact rather than merely hiding the walking phase.
- Increase zero-command sampling pressure so the learner sees enough stand-mode evidence before the stepping oscillator becomes a default attractor.
- Add tests showing that zero-command gait violation remains gated off by default, but the active SAC config enables stand-mode gait cost.

## 20. 2026-06-24 Drift Is Unauthorized Locomotion

Training observation after Section 19 reasoning:

- The in-place stepping is not purely in-place.
- The policy can add a small base drift while stepping, which makes the behavior resemble low-speed locomotion.
- This drift can collect reward or avoid penalties while still violating the zero-command intent.

Diagnosis:

- Command is the only legitimate behavior-authority signal.
- Actual base velocity is an outcome, not a mode selector.
- If zero command produces nonzero base velocity, the correct interpretation is not "the robot is walking"; it is "stand mode has been violated".
- Therefore actual velocity must never open the gait/walk gate. It can only serve as evidence for stand-mode anti-drift cost.
- Existing tracking reward is too smooth near zero command; small drift does not lose enough reward to prevent the early-training oscillator from becoming an attractor.

Engineering follow-up:

- Add stand-only anti-drift penalties for base xy velocity and yaw velocity.
- Gate these penalties only by command inactivity, not by measured speed.
- Keep `tracking_lin_vel`, `tracking_ang_vel`, and gait constraint command-gated semantics unchanged.
- Update the active SAC G1 owner config with strong enough anti-drift scales to make small zero-command drift more expensive than the tracking reward tolerance near zero.
- Add tests proving that zero-command drift is penalized while the same measured velocity under a nonzero command is not treated as a stand violation.

## 21. 2026-06-24 Stand Mode Must Remove Action Authority

Training observation after anti-drift and stand gait constraints:

- The policy still discovers the in-place stepping attractor during early training.
- The behavior includes slight drift, so the oscillator can still create a locomotion-like transition under zero command.
- Reward penalties after the transition are not enough to prevent the attractor from forming.

Diagnosis:

- The previous fixes treated zero-command stepping as a reward violation.
- That is too late in the causal chain: the actor can still execute locomotion actions, move the body, and let the critic discover that this default oscillator is useful across nearby low-speed commands.
- The missing boundary is action authority.
- Command is the only legitimate authorization signal for locomotion actions.
- When command is inactive, the policy may be penalized for nonzero raw action, but the env should not execute that action as locomotion control.

Engineering follow-up:

- Add an owner-configured `stand_action_authority` switch for G1 walking.
- When enabled, `apply_action()` keeps raw policy actions in `current_actions` for observation/reward accounting, but uses zero executed actions for inactive-command environments.
- Store `executed_actions` in `info` for diagnostics.
- Keep walk-mode action authority unchanged when command is active.
- Strengthen stand raw-action penalty so the actor learns to output zero under inactive commands, rather than relying only on the execution clamp.
- Add tests proving inactive-command actions are clamped at execution while active-command actions still pass through.

## 22. 2026-06-24 Mode-Specific Reward Contract

Concept correction:

- The current "locomotion" training task is actually training two behaviors in one network: standing and walking.
- Standing and walking are not two speeds of the same objective.
- Standing suppresses support transfer, drift, and periodic gait.
- Walking authorizes support transfer, drift, and periodic gait to track an external command.
- If both objectives are mixed in one reward table, the policy can discover an intersection: low-amplitude stepping with small drift.

Mode contract:

- `gait_enabled = false` is a discrete external mode: train standing only.
- `gait_enabled = true` is a discrete external mode: train walking only.
- Command magnitude is continuous only inside walking mode; it must not decide whether gaiting is authorized.
- Actual velocity is an outcome, not a mode selector.

Engineering ownership:

- Env owns mode interpretation, reward masking, action authority, gait phase freezing, and diagnostics.
- Owner YAML owns which reward terms belong to standing and walking for this task/backend.
- Training scripts remain orchestration only.

Minimal implementation:

- Add reward mode config under `reward.mode`.
- Dispatch stand reward terms and walk reward terms separately.
- Combine them as `stand_mask * R_stand + walk_mask * R_walk`.
- Write `info["gait_enabled"]` during G1 reset from the discrete nonzero external command event.
- Derive `gait_enabled` from `info["gait_enabled"]` when present; otherwise fall back to command activity for backward compatibility.
- Keep actor observation dimension unchanged in this iteration; command zero/nonzero remains the observable proxy until a separate deployment contract introduces an explicit `gait_enabled` observation.

Forbidden freedom:

- Do not use measured velocity to enable walking reward.
- Do not let stand reward and walk reward both shape the same transition.
- Do not restore positive `feet_phase*` rewards in stand mode.

Validation:

- Unit-test reward mode masks independently of raw command magnitude.
- Unit-test that stand terms do not contribute in walk mode.
- Unit-test that walk terms do not contribute in stand mode.
- Keep existing config tests as owner-YAML contract checks.

## 23. 2026-06-24 Reset-Time Mode Mask Batch Contract

Runtime failure:

- During async off-policy training, `_reset_done_envs()` can reset only the environments that just finished.
- In the observed crash, global `num_envs` was 2048 but the reset batch contained only 5 environments.
- `info["gait_enabled"]` correctly had shape `(5,)`, but `_gait_enabled_mask()` incorrectly required `(2048,)`.

Contract correction:

- Mode masks are per-info-batch tensors.
- During rollout step they usually match global `num_envs`.
- During reset observation construction they match `len(env_ids)`.
- Validation must compare `gait_enabled` against the current `commands` batch when commands are present, not against global env count.

Validation:

- Add a regression test where `env._num_envs` is larger than the reset info batch.
- Verify `_gait_phase_for_observation()` accepts partial reset batches and still applies stand-phase replacement correctly.

## 24. 2026-06-25 Low-Speed Command Must Not Become Stand Mode

Engineering audit after the first reward-mode implementation:

- The concept says mode is a discrete external event: command absent means standing, command present means walking.
- The implementation still used `zero_small_xy_commands()` in the G1 reset command sampler.
- With `small_xy_threshold > 0`, a low-speed nonzero command can be rewritten to zero before `gait_enabled` is computed.
- That silently reintroduces command magnitude as the mode switch, contradicting Section 22.

Contract correction:

- G1 walking should create standing samples through `rel_standing_envs`, which explicitly writes zero command.
- Low-speed nonzero commands must remain nonzero and must produce `gait_enabled = true`.
- `small_xy_threshold` should be zero for the active G1 SAC MuJoCo owner config.
- The G1 provider fallback threshold should also be zero so missing config does not reintroduce magnitude gating.

Validation:

- Config test must assert `env.commands.small_xy_threshold == 0.0`.
- Provider test must sample a low-speed nonzero command and verify it remains nonzero.
- Provider test must verify the same low-speed command writes `gait_enabled = true`.

## 25. 2026-06-25 Standing Mode Needs Live-Path Diagnostics

Engineering audit after repeated "same as original" behavior:

- The active checkout's Hydra + BackendAdapter + registry path does construct G1 with `rel_standing_envs = 0.4`, `small_xy_threshold = 0.0`, and `reward.mode.enabled = true`.
- A live reset sentinel shows zero-command standing samples are generated.
- However, off-policy collector only forwards `info["log"]` keys whose names start with `reward/`.
- The reward-mode implementation logged masks as `mode/*`, so the training log could not prove whether standing samples entered replay.
- Mode reward dispatch calls `run_reward_dispatch()` twice. Since that helper clears `info["log"]` on logging cadence, the second call can overwrite the first call's term logs. This makes stand-term diagnostics especially misleading even if the reward itself is computed.

Contract correction:

- Standing mode must have explicit `reward/*` live-path diagnostics.
- Diagnostics should include the sampled stand/walk fraction and the masked stand/walk reward totals after mode selection.
- These diagnostics are not new rewards; they only prove that the intended mode path is active in the collector/replay loop.

Validation:

- Unit-test that `_compute_mode_reward()` writes `reward/mode_stand_frac`, `reward/mode_walk_frac`, `reward/stand_total`, and `reward/walk_total` when reward logging is enabled.
- Keep the existing lifecycle tests for reset-time `gait_enabled` and low-speed nonzero command behavior.

## 26. 2026-06-25 Current Command Owns Gait Mode

Failure after simulation test:

- The policy still steps in place at zero command.
- Training reset already creates standing samples, and reward-mode diagnostics can show standing samples in the live path.
- However, play/deploy code can update `info["commands"]` after reset on every control step.
- `info["gait_enabled"]` was only written during reset, so it can remain `true` after the external command has been changed to zero.
- In that case, zero command is only zero in the observation vector; reward mode, gait phase, and action-authority logic can still treat the transition as walking.

Contract correction:

- The current external command is the owner of the current gait mode.
- `commands == 0` means standing.
- `commands != 0` means walking, independent of command magnitude.
- `info["gait_enabled"]` is a cached diagnostic/reset field, not an authority that can override the current command.
- Whenever env logic needs gait mode and `commands` is present, it must recompute and sync `gait_enabled` from the current command batch.

Validation:

- Unit-test that stale `gait_enabled = true` with current zero command is corrected to stand mode.
- Unit-test that stale `gait_enabled = false` with current nonzero command is corrected to walk mode.
- Unit-test that reward-mode dispatch follows current command rather than stale reset-time `gait_enabled`.
- Unit-test that `apply_action()` freezes stand phase after an external command is edited to zero.

## 27. 2026-06-25 Playback Must Reject Old Walking-Only Checkpoints

Failure after another simulation test:

- Local `load_run=-1` resolves to `logs/fast_sac/G1WalkFlat/2026-06-12_15-46-01_mujoco`.
- That run was trained at commit `9dee71386ed61b46f8ec93d74499dcdb4d06bb92`, before the standing/walking reward-mode contract.
- Its `run_config.json` has no `env.commands.rel_standing_envs`, no `reward.mode`, no `gait_constraint`, positive `feet_phase = 5.0`, and `alive = 10.0`.
- Therefore playback of that checkpoint should still step in place. It is evidence for an old policy, not evidence that the current two-mode env code failed.

Contract correction:

- Debugging standing behavior must inspect both current code and the checkpoint's `run_config.json`.
- A G1 SAC checkpoint is not standing-mode-compatible unless its run config records the two-mode reward contract.
- `load_run=-1` is unsafe during iterative reward redesign because it silently selects the latest available run under the configured log root, which may be an old walking-only run.

Validation:

- Add a playback-time warning for G1 SAC policy playback when the selected run config lacks `reward.mode.enabled`, required stand terms, `env.commands.rel_standing_envs`, frozen stand gait phase, or still has positive gait-phase reward.
- Add tests for compatible and incompatible run configs so this diagnosis survives future refactors.
