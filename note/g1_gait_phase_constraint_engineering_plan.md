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

## 28. 2026-06-25 Standing Mode Must Own Execution Authority

Failure after simulation feedback:

- The code can correctly enter standing mode, but the owner config still has `env.stand_action_authority = false`.
- With that setting, standing mode only changes reward accounting and gait phase observation.
- The policy's locomotion action is still executed by PD at zero command.
- Therefore a policy that learned stepping can remain visually stepping even when standing reward terms are active.

Concept correction:

- External command is not only a reward selector.
- It is the authority switch for whether locomotion actions may be executed.
- `command == 0` means the locomotion action channel is not authorized.
- In standing mode the env may still record the raw policy action for `stand_action_l2`, but the executed action must be the standing/default action.
- `command != 0` restores normal locomotion action authority immediately.

Engineering correction:

- Set `env.stand_action_authority: true` in the active G1 SAC MuJoCo owner YAML.
- Keep `current_actions` as the raw policy output so standing reward can still train the actor away from useless zero-command actions.
- Use `executed_actions` only for PD control.

Validation:

- Config test must assert `env.stand_action_authority is True`.
- Existing action-authority unit test must prove raw policy actions are kept while executed stand actions are zeroed.

## 29. 2026-06-25 Standing Fix Must Be Visible In Playback And Live Logs

Failure after retraining feedback:

- The robot still visually steps in place, matching the original failure.
- This rejects another round of reward tuning as the next step.
- The missing evidence is whether the live run actually entered the standing execution contract.

Concept correction:

- Standing mode is only proven if three live-path facts are simultaneously true:
  - the current command is exactly zero;
  - the reward mode reports stand samples;
  - the executed action for stand samples is zero even if the raw policy action is nonzero.
- Playback must not leave the command source ambiguous. For G1 SAC walking, interactive playback should default to policy actions controlled by keyboard commands.

Engineering correction:

- Set the G1 SAC MuJoCo owner YAML interactive default to `action_mode: policy` and `keyboard: true`.
- Log action-authority diagnostics from `G1WalkEnv.apply_action()`:
  - `reward/action_authority_stand_frac`;
  - `reward/raw_action_l1`;
  - `reward/executed_action_l1`;
  - `reward/stand_raw_action_l1`;
  - `reward/stand_executed_action_l1`.

Expected next-run sentinel:

- During training with `rel_standing_envs: 0.4`, logs should show `reward/mode_stand_frac` near `0.4`.
- For standing samples, `reward/stand_executed_action_l1` should be exactly `0.0`.
- If `reward/stand_raw_action_l1` remains large, the actor has not yet learned standing, but execution authority is still correctly preventing the standing samples from stepping.
- If `reward/stand_executed_action_l1` is nonzero under zero command, the standing authority contract is not live.

Logging lifecycle correction:

- `apply_action()` runs before reward computation.
- `rewards.run_reward_dispatch()` creates a fresh `info["log"]` on logging ticks.
- Therefore action-authority diagnostics written only inside `apply_action()` can be erased before the collector reads `info["log"]`.
- G1 must re-log action-authority diagnostics at the end of `_compute_reward()`, after reward dispatch and gait-constraint logging.

## 30. 2026-06-25 Walking Reward Must Be Hard-Gated By External Gait Authority

Concept correction:

- The core LeCAR-like idea is not to let reward teach the policy whether to walk.
- External command decides gait authority.
- When `gait_enabled == 0`, the locomotion policy has no gait/action authority and must not be able to receive Walking Reward.
- When `gait_enabled == 1`, Walking Reward is active and the locomotion policy is trained normally.

Failure mechanism:

- If zero-command samples can receive any tracking, gait, phase, or walking-style reward, the policy can hack the objective by stepping or drifting in place.
- A soft penalty on idle motion is not enough because it still leaves the walking reward channel visible.
- The important boundary is an explicit if:

```text
if gait_enabled:
    apply Walking Reward
else:
    apply Idle/Standing Reward only
```

Engineering correction:

- Replace the previous "compute stand and walk rewards for all envs, then multiply by masks" implementation with a masked mode dispatch.
- Each reward term must be multiplied by its mode mask at the point where the term contributes to the reward.
- Logged term values must also be mode-masked, so terminal/TensorBoard cannot show walking reward earned by standing samples.
- `reward/stand_total` and `reward/walk_total` remain the canonical mode totals.

Validation:

- Unit-test that a zero-command sample with high walking velocity receives zero walking reward contribution.
- Unit-test that reward component logs are mode-masked.
- Keep action-authority diagnostics: standing samples should show `reward/stand_executed_action_l1 == 0.0`.

## 31. 2026-06-25 Standing Reset Must Satisfy Idle Controller Assumption

Observation after failed training:

- With no external command, the robot can still drift.
- This makes the standing segment look like it never reaches a true idle state, even though `gait_enabled == 0`.

Concept correction:

- LeCAR-like hard gating assumes that when locomotion authority is off, the remaining idle/default controller is given an idle-compatible state.
- UniLab G1 reset previously sampled random base velocity for every env before command sampling.
- Therefore standing samples could start with nonzero base velocity while locomotion actions were explicitly blocked.
- That creates a contradiction: the policy is not allowed to correct the motion, but the reset state already contains motion.

Engineering correction:

- Add `env.standing_reset_base_qvel_limit`.
- For G1 walking, after commands are sampled, detect zero-command standing reset samples.
- Clamp/resample their base qvel independently from walking samples.
- The active MuJoCo owner config sets `standing_reset_base_qvel_limit: 0.0`.
- Walking samples keep `reset_base_qvel_limit: 0.5`, so walking robustness is not removed.

Validation:

- Unit-test that G1 reset plan zeros base qvel for standing samples but preserves nonzero qvel randomization for walking samples.
- Keep mode/action diagnostics to ensure standing samples still report `reward/mode_stand_frac > 0` and `reward/stand_executed_action_l1 == 0.0`.

## 32. 2026-06-25 Mode-Conditioned Single-Policy Plan

Goal:

- Do not rely on a deployment-only hard gate as the core solution.
- Train one policy that can genuinely represent two externally specified modes:
  - `STAND`: zero external gait command, quiet double-support standing.
  - `WALK`: nonzero external gait command, velocity tracking and gait control.
- Prevent walking behavior from smoothly generalizing into zero-command standing observations.

Design principle:

- Zero command is too weak as a mode signal because it is only a boundary point in continuous command space.
- Add an explicit mode/gait authority signal to the policy observation.
- Keep reward routing as a hard if:

```text
if gait_enabled:
    Walking reward is visible.
else:
    Standing reward is visible.
```

- But do not use action hard-gating as the main learning mechanism. The policy must see `gait_enabled` and learn different actions for different modes.

Module ownership:

- `src/unilab/envs/locomotion/g1/joystick.py`
  - Owns G1 mode signal construction.
  - Owns actor/critic observation layout and dimension changes.
  - Owns standing reset distribution.
  - Owns standing/walking reward routing and diagnostics.
- `conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml`
  - Owns active task hyperparameters and curriculum defaults.
  - Adds config switches for mode-conditioned observation and optional standing action authority ablations.
- `tests/envs/locomotion/g1/test_gait_constraint.py`
  - Owns unit/lifecycle tests for mode signal, observation dimension, reward routing, reset distribution, and action diagnostics.
- `tests/config/test_reward_injection.py`
  - Owns Hydra owner-config contract checks.
- Off-policy runner/learner
  - Should not be changed in the first step. SAC can train from the new observation once env exposes the correct contract.

Step A: explicit mode observation (implemented)

- Add a G1 env config field such as `mode_observation: true`.
- Append one scalar to actor and critic observations:

```text
mode_signal = gait_enabled
0.0 = STAND
1.0 = WALK
```

- Update `obs_groups_spec` from:

```text
obs=98, critic=101
```

to:

```text
obs=99, critic=102
```

- Update symmetry observation layout to include `("mode", 1)`.
- This intentionally invalidates old checkpoints because the policy input contract changed.

Step A tests:

- Unit-test that zero command gives `mode_signal = 0`.
- Unit-test that any nonzero command gives `mode_signal = 1`.
- Unit-test actor/critic observation dimensions.
- Config test that active G1 SAC MuJoCo owner YAML enables mode observation.
- Optional play/checkpoint guard should warn that old 98-dim checkpoints are incompatible with mode-conditioned policy.

Step B: standing reset distribution (implemented)

- Keep `standing_reset_base_qvel_limit: 0.0`.
- Ensure standing reset also starts with:
  - `current_actions = 0`;
  - `last_actions = 0`;
  - frozen stand gait phase;
  - zero command;
  - `gait_enabled = 0`.
- Walking reset may keep velocity perturbation and normal gait phase sampling.
- Standing reset writes stand phase into `info["gait_phase"]` directly, not only through observation-time freezing.

Step B tests:

- Reset-plan test for qvel split by mode.
- Reset-observation test for partial standing/walking batches.
- Test that standing reset actor obs contains zero command, stand phase, zero last action, and mode signal 0.

Step C: reward routing remains hard-if (implemented)

- Keep masked reward dispatch.
- Standing samples must not receive tracking/gait/walking-style reward.
- Walking samples must not receive standing-specific reward.
- Standing reward should focus on:
  - no xy drift;
  - no yaw drift;
  - low joint velocity;
  - posture/default pose;
  - foot contact/foot no-slip if needed.

Step C tests:

- Zero-command sample with high forward velocity cannot receive `tracking_lin_vel`.
- Nonzero-command sample does not receive `stand_lin_vel_xy_l2`.
- Logs for shared terms such as `alive` remain mode-masked and not overwritten.

Step D: ablation switch for action authority (implemented)

- Keep `stand_action_authority` as a safety/ablation switch, but no longer treat it as the core learning mechanism.
- Two recommended experiments:

```text
mode_observation=true, stand_action_authority=true
mode_observation=true, stand_action_authority=false
```

- If `stand_action_authority=false` succeeds only after mode observation is added, that proves the policy learned standing rather than relying on the hard gate.
- If it still fails, standing reward/reset distribution remains insufficient.

Step D tests:

- Existing action-authority tests remain.
- Add config-level ablation tests to ensure both switches are independently expressible.
- Add `apply_action` test that `stand_action_authority=false` preserves raw policy actions even for standing samples.

Step E: training curriculum (implemented)

- Stage 1: standing-heavy or standing-only sanity run.
  - Goal: `stand_raw_action_l1` decreases, xy/yaw drift remains near zero, episode length increases.
- Stage 2: walking-only or walking-heavy run.
  - Goal: locomotion still learns gait and tracking.
- Stage 3: mixed stand/walk.
  - Goal: mode switch separates behavior.
- Stage 4: transition curriculum if needed.
  - Add short stand-to-walk and walk-to-stand segments only after stand/walk are individually stable.

Implemented Hydra stage configs:

- `+g1_walk_stage=standing_sanity`
  - `rel_standing_envs=1.0`;
  - zero velocity command range;
  - `reset_base_qvel_limit=0.0`;
  - `stand_action_authority=false`.
- `+g1_walk_stage=walking_sanity`
  - `rel_standing_envs=0.0`;
  - reduced walking command range;
  - `reset_base_qvel_limit=0.5`;
  - `stand_action_authority=false`.
- `+g1_walk_stage=mixed_mode`
  - `rel_standing_envs=0.4`;
  - full G1 flat command range;
  - `reset_base_qvel_limit=0.5`;
  - `stand_action_authority=false`.

Recommended commands:

```bash
CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 PYTHONWARNINGS="ignore" \
  uv run train --algo sac --task g1_walk_flat --sim mujoco \
  +g1_walk_stage=standing_sanity \
  algo.max_iterations=800

CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 PYTHONWARNINGS="ignore" \
  uv run train --algo sac --task g1_walk_flat --sim mujoco \
  +g1_walk_stage=walking_sanity \
  algo.max_iterations=800

CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 PYTHONWARNINGS="ignore" \
  uv run train --algo sac --task g1_walk_flat --sim mujoco \
  +g1_walk_stage=mixed_mode \
  algo.max_iterations=5000
```

Expected live diagnostics:

- `reward/mode_stand_frac`
- `reward/mode_walk_frac`
- `reward/stand_raw_action_l1`
- `reward/stand_executed_action_l1`
- `reward/stand_lin_vel_xy_l2`
- `reward/stand_yaw_vel_l2`
- `reward/walk_total`
- `reward/stand_total`

Acceptance criteria before long training:

- Tests pass.
- Fresh run config records `obs=99`, `critic=102`, `mode_observation=true`.
- Terminal shows standing samples and standing action diagnostics.
- Short standing-heavy run shows `stand_raw_action_l1` trending down rather than remaining walking-like.

Step F: live-path stage sentinel (implemented)

- Add `scripts/deploy/check_unilab_g1_walk_stage_live_path.py`.
- The sentinel does not train and does not rely on checkpoint quality.
- It composes the same Hydra stage fragments used by training:
  - `+g1_walk_stage=standing_sanity`;
  - `+g1_walk_stage=walking_sanity`;
  - `+g1_walk_stage=mixed_mode`.
- It then constructs `G1WalkFlat` through `BackendAdapter -> create_env -> registry.make`, calls `init_state()`, runs a short zero-cost `step()` with fixed actions, and checks:
  - `obs=99`, `critic=102`;
  - `mode_signal` agrees with `gait_enabled`;
  - standing/walking fractions match the stage;
  - `stand_action_authority=false` reaches the env override;
  - reward-mode logs appear in the live path;
  - standing action diagnostics are visible even when action authority is disabled.

Important Step F correction:

- Action diagnostics must not depend on `stand_action_authority=true`.
- For ablation runs with `stand_action_authority=false`, we still need `reward/stand_raw_action_l1` and `reward/stand_executed_action_l1`.
- Otherwise the short standing-heavy run cannot tell whether the policy actually learned to reduce standing actions.

Step F validation command:

```bash
uv run scripts/deploy/check_unilab_g1_walk_stage_live_path.py --num-envs 16 --steps 1
```

Observed local sentinel result:

- `standing_sanity`: `reward/mode_stand_frac=1.0`, `reward/mode_walk_frac=0.0`.
- `walking_sanity`: `reward/mode_stand_frac=0.0`, `reward/mode_walk_frac=1.0`.
- `mixed_mode`: both stand and walk samples appear in the same live batch.

## 33. 2026-06-25 Playback Must Restore Checkpoint Env Contract

Observed failure after training:

```text
RuntimeError: size mismatch for net.0.weight:
copying a param with shape torch.Size([512, 99]) from checkpoint,
the shape in current model is torch.Size([512, 98])
```

Diagnosis:

- The checkpoint was trained with `env.mode_observation=true`, so actor input is 99.
- Playback rebuilt the env/actor with a 98-dimensional observation contract.
- This is not a bad checkpoint. It is a train/play contract mismatch.

Engineering correction:

- `scripts/play_interactive.py` now reads the selected checkpoint's sibling `run_config.json`.
- For off-policy playback, it replays the checkpoint's `config.env` and `config.reward` into the env override before creating the playback env.
- `src/unilab/visualization/interactive_playback.py` now checks the checkpoint actor first-layer input dimension before `load_state_dict`.
- If the checkpoint and playback env still disagree, the error now names:
  - `checkpoint=<dim>`;
  - `playback_env_obs=<dim>`;
  - the missing env contract restoration.

Validation:

- Unit-test that SAC/G1 playback env override restores `mode_observation=true` from checkpoint `run_config.json`.
- Unit-test that a 99-dim SAC actor checkpoint against a 98-dim playback env raises an explicit contract error.
- Existing G1 stage live-path sentinel still passes.

## 34. 2026-06-25 Standing Must Learn Residual Balance, Not Rely On Action Authority

Observed failure after fixing the playback observation contract:

- `./start.sh` can load the mode-conditioned checkpoint, but the robot falls
  directly and does not move.
- This is progress relative to zero-command stepping: the policy is no longer
  visibly exploiting the walking channel.
- The remaining symptom matches a standing controller that has no effective
  residual action in zero-command mode.

Diagnosis:

- The active G1 SAC MuJoCo owner YAML still had
  `env.stand_action_authority: true`.
- That setting is the older hard-gate experiment: standing samples record raw
  policy actions for diagnostics, but execute zero action through the PD target.
- If `default_angles` alone cannot stabilize the robot, the standing segment
  collapses while the actor never receives an execution path to learn balance.
- This contradicts the Step A-F design: `mode_observation` and hard-if reward
  routing should teach one policy two externally specified modes; action
  authority is only an ablation/safety switch.

Engineering correction:

- Set the active `conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml` default to
  `env.stand_action_authority: false`.
- Keep all `+g1_walk_stage=*` configs at `stand_action_authority=false`.
- Keep the action-authority unit tests, because the switch remains useful for
  diagnosis, but it must not be the default training path.
- Extend the G1 SAC playback checkpoint warning: a selected run with
  `env.stand_action_authority=true` is a hard-gated standing run and cannot
  prove that the policy learned residual standing.

Expected behavior after retraining:

- In zero-command samples, Walking Reward remains invisible.
- The actor's standing action is actually executed.
- `reward/stand_raw_action_l1` and `reward/stand_executed_action_l1` should
  match in standing samples.
- A successful standing run should reduce drift and termination through learned
  residual balance, not through env-side action suppression.

## 35. 2026-06-25 Keyboard Command Probe Must Not Sample Standing Reset

Observed playback failure:

- The first `./start.sh` launch could open the viewer and keyboard commands
  changed `vx/vyaw`.
- After closing and launching again, playback printed:

```text
[play_interactive] interactive.keyboard unavailable: policy obs does not contain the velocity command.
```

Diagnosis:

- This was a playback verification false negative, not a MuJoCo or policy load
  crash.
- `_policy_obs_contains_command()` verifies keyboard control by temporarily
  setting `commands.vel_limit` to a probe command and calling reset.
- For G1 mixed-mode configs, `rel_standing_envs=0.4` means reset can still
  sample a standing env and zero the command.
- If that happens during the probe reset, the checker sees a zero-command obs
  and incorrectly concludes that policy obs has no velocity command.
- That explains the random behavior: one launch can pass, another can fail.

Engineering correction:

- During the playback-only command-observation probe, temporarily set
  `commands.rel_standing_envs=0.0` together with the probe `vel_limit`.
- Restore both fields immediately after the probe.
- This does not affect training or normal rollout; it only makes the startup
  keyboard-safety check deterministic.

Validation:

- Unit-test that `_policy_obs_contains_command()` disables standing sampling
  during the probe and restores `rel_standing_envs` / `vel_limit` afterward.
- Re-run G1 mode/reward lifecycle tests and the stage live-path sentinel.

## 36. 2026-06-25 Standing Reward Needs A Dense Height Objective

Observed failure after startup became reliable:

- The viewer opens consistently.
- At zero command, the robot enters simulation and falls almost immediately.
- When a velocity command is given, the legs respond slightly, so policy load,
  action execution, and keyboard command injection are not the primary failure.

Diagnosis:

- This is a standing ability failure, not the old zero-command stepping hack.
- The STAND reward was mostly a collection of "do not move" penalties:
  drift, yaw drift, joint velocity, action magnitude, pose, orientation, and
  foot orientation.
- It did not include the existing `base_height` reward term even though the
  config already defines `base_height_target=0.754`.
- The termination condition catches a fall, but it is sparse. For SAC, that is
  a weak learning signal compared with the dense action/pose penalties that can
  push the policy toward small actions near `default_angles`.
- Therefore the policy can learn a quiet but dynamically insufficient standing
  behavior: no stepping, little action, then fall.

Engineering correction:

- Add `reward.scales.base_height: -80.0` to the active G1 SAC MuJoCo owner YAML.
- Add `base_height` to `reward.mode.stand_terms`.
- Do not add it to `walk_terms` in this step. Walking already has tracking and
  gait objectives; this fix targets zero-command standing only.

Expected effect:

- Zero-command standing samples still cannot receive Walking Reward.
- Standing samples receive a dense penalty when base height drops below the
  stand target.
- The actor is allowed to execute residual standing actions because
  `stand_action_authority=false`, so it can learn balance instead of relying on
  env-side action suppression.

Validation:

- Config tests assert that `base_height` is in STAND terms and not in WALK terms.
- Unit test proves low base height is penalized only for standing samples.
- Numeric test reads the active Hydra owner YAML and verifies the exact effect:
  with `base_height=0.300`, `base_height_target=0.754`, `scale=-80.0`, and
  `ctrl_dt=0.02`, the standing sample receives
  `-80.0 * (0.300 - 0.754)^2 * 0.02 ~= -0.3298` while the walking sample
  receives exactly `0.0` from this term.
- Add `scripts/deploy/check_unilab_g1_standing_mode_dynamics.py` for a true
  MuJoCo closed-loop check. It supports:
  - `--action-mode zero`: default/zero action standing dynamics;
  - `--action-mode policy`: selected SAC checkpoint standing dynamics.
- Observed local zero-action result:
  - initial standing state is clean: command `0`, mode signal `0`, height
    `0.754`, tilt `0`;
  - zero/default action terminates at step `72`;
  - max tilt reaches `67.84 deg`, exceeding `max_tilt_deg=65`;
  - min height remains `0.358`, so the failure is tilt-first rather than
    height-threshold-first;
  - `reward/base_height` becomes negative, proving the new reward path is active
    but not itself a controller.
- Stage live-path sentinel should still report mode separation and ungated
  standing action execution.

Interpretation:

- Reward routing and the dense height term are now testable and active.
- The current default standing controller is not stable.
- Therefore a trained standing residual policy must be validated with
  `--action-mode policy`; if that fails too, the next fix is standing controller
  design/training, not another proof that the reward function is wired.

## 37. 2026-06-25 Standing Reward Priority Must Prefer Balance Over Quietness

Concept correction:

- Because reward routing already separates STAND from WALK, the remaining
  standing problem is not walking reward leakage.
- The STAND reward should not mean "output the smallest action near
  `default_angles`."
- It should mean "remain upright and at standing height; then be quiet."
- Balance residuals are allowed in STAND mode. Penalizing all residual action
  too strongly creates a passive controller that looks calm and then falls.

Engineering correction:

- Add `upright` to the G1 reward function map and to STAND terms.
- Strengthen first-priority standing terms:
  - `penalty_orientation: -25.0`;
  - `penalty_ang_vel_xy: -2.0`;
  - `base_height: -80.0`;
  - `upright: 4.0`.
- Weaken quietness regularizers so they do not dominate balance:
  - `stand_action_l2: -0.01`;
  - `stand_still: -0.2`;
  - `stand_dof_vel_l2: -0.05`;
  - `pose: -0.1`;
  - `penalty_feet_ori: -5.0`.

Validation:

- Add a numeric reward-ordering test using the active Hydra owner YAML:
  - sample A is upright, at target height, but uses moderate residual action;
  - sample B is quiet, low, and tilted near failure;
  - sample A must receive positive STAND reward;
  - sample B must receive negative STAND reward;
  - sample A must exceed sample B by a large margin.
- The zero-action MuJoCo dynamics test is still expected to fail because reward
  changes do not turn the default PD target into a controller. The policy-mode
  version is the real standing-controller acceptance test after retraining.

## 38. 2026-06-25 Standing Is Zero-Command Locomotion, Not A Separate Static Pose Task

Concept correction:

- Standing and walking share the same first-order physical objective: keep the
  robot dynamically balanced, upright, at usable base height, and smooth enough
  for the PD/action interface to remain controllable.
- The mode routing should only remove the hacking channel: zero-command samples
  must not receive command-tracking or gait-progression reward.
- Routing must not make STAND a different task whose primary objective is
  "small action near default pose." That makes STAND weaker than WALK exactly
  where the robot needs active residual balance.
- Therefore STAND is better modeled as the zero-command simplification of the
  locomotion task:

```text
reward =
  balance_common_terms
  + stand_mode * zero_command_terms
  + walk_mode  * walking_tracking_terms
```

Engineering contract:

- `balance_common_terms` enter both STAND and WALK:
  - `penalty_orientation`;
  - `upright`;
  - `penalty_ang_vel_xy`;
  - `penalty_action_rate`;
  - `base_height`;
  - `pose`;
  - `penalty_feet_ori`;
  - `alive`.
- `stand_terms` only express zero-command behavior:
  - `stand_still`;
  - `stand_action_l2`;
  - `stand_dof_vel_l2`;
  - `stand_lin_vel_xy_l2`;
  - `stand_yaw_vel_l2`.
- `walk_terms` only express commanded locomotion:
  - `tracking_lin_vel`;
  - `tracking_ang_vel`;
  - `under_speed`.
- Positive gait-style terms remain disabled in this stage. Gait pressure still
  comes from the constraint bridge and only under its configured gate.
- `penalty_action_rate` is common but must be weaker than the previous walking
  value (`-1.0` instead of `-4.0` here). Otherwise a necessary first-step
  standing residual can be ranked below quiet falling, which recreates the
  passive-controller failure.

Validation:

- Config tests must prove active SAC G1 owner YAML contains
  `balance_common_terms`, and that `tracking_lin_vel` does not leak into common
  or STAND terms.
- Reward-dispatch tests must prove a common term contributes to both STAND and
  WALK samples while mode-specific terms remain isolated.
- Numeric tests must now expect `base_height` to penalize low height in both
  modes; the isolation boundary is no longer `base_height`, but walking command
  tracking and gait terms.

## 39. 2026-06-25 Standing Reward Must Be Tested As A Dynamics Objective

Problem:

- Static reward tests can prove wiring and scalar ordering, but they cannot
  prove that the Standing Reward provides an executable standing direction.
- The meaningful question is not whether `base_height` or `upright` is present.
  The meaningful question is whether the reward ranks a dynamically standing
  residual action above the zero/default action that falls.

Diagnostic:

- Extend `scripts/deploy/check_unilab_g1_standing_mode_dynamics.py` with
  `--action-mode reward-search`.
- This mode creates a real MuJoCo `G1WalkFlat` env under
  `+g1_walk_stage=standing_sanity`, disables autoreset, and rolls out many
  deterministic standing residual-action candidates in parallel.
- Candidate actions include zero action and symmetric leg-pitch residuals over
  hip/knee/ankle pitch joints.
- The acceptance condition is not that every candidate stands. Most candidates
  are deliberately bad. The acceptance condition is:
  - the best candidate by cumulative Standing Reward is not zero action;
  - it survives longer than zero action;
  - it survives the full tested horizon.

Observed local result:

- Command/mode reset is clean: command `0`, `gait_enabled=0`, mode signal `0`.
- Zero action terminates at step `72`.
- The reward-selected best candidate is
  `symmetric_pitch hip=+0.15 knee=+0.20 ankle=-0.15`.
- This candidate survives the full `120` step horizon with
  `max_tilt_deg=58.33` and `min_height=0.656`.
- Therefore the Standing Reward is now able to rank a dynamically standing
  residual above the zero/default falling action.

Interpretation:

- This does not prove that SAC will immediately learn the standing controller.
- It does prove that the reward itself is no longer only a wiring artifact or a
  static scalar trick: in real MuJoCo dynamics, it contains at least one
  executable standing optimum near the reset state.
- If training still falls at zero command, the next suspect becomes exploration,
  replay distribution, curriculum, or actor optimization, not basic reward
  direction.

## 40. 2026-06-26 Transition Is A Mixed-Distribution Bridge, Not A New Stage

Problem:

- After the common-balance reward fix, Standing can be learned without falling,
  but Walking can degrade into conservative balance-plus-tracking behavior.
- The failure is not that the system needs another standalone curriculum stage.
  The failure is that the active mixed distribution jumps directly from
  zero-command Standing to normal-command Walking.
- This creates two discontinuities:
  - a command discontinuity: `command=0` immediately becomes regular walking
    velocity;
  - a height/control discontinuity: the policy can learn different body-height
    transients around the mode boundary even though locomotion height should be
    a shared invariant.

Design delta:

- Add transition samples inside the same mixed training distribution:
  - `rel_standing_envs`: strict zero-command Standing samples;
  - `rel_transition_envs`: low-speed nonzero commands that keep `gait_enabled=1`;
  - remaining samples: normal Walking commands.
- Transition is not a third reward mode and not a new training phase.
- Transition samples use WALK reward routing because their command is nonzero.
  They only narrow the command range so the policy sees stand-to-walk starts
  during the same run.
- `base_height` must remain in `balance_common_terms`. There is one
  `base_height_target` shared by Standing, Transition, and Walking. Any
  standing/walking height split would make the visible height jump a learned
  objective artifact.

Implemented owner contract:

- Main SAC G1 MuJoCo mixed distribution:
  - `rel_standing_envs: 0.3`;
  - `rel_transition_envs: 0.2`;
  - transition command range:
    `vx in [0.05, 0.25]`, `vy in [-0.05, 0.05]`, `vyaw in [-0.15, 0.15]`.
- `standing_sanity` and `walking_sanity` keep `rel_transition_envs: 0.0` so
  they remain pure diagnostics.

Validation expectation:

- Config tests must prove the transition fraction and range reach env override.
- Reset/helper tests must prove transition commands are nonzero and therefore
  `gait_enabled=1`.
- Existing reward-mode tests must continue to prove `base_height` is common and
  command tracking remains WALK-only.

## 41. 2026-06-26 Walking Needs Positive Step Credit After The Transition Bridge

Observation after retraining with transition samples:

- The robot no longer behaves like it lacks a walking command.
- It visibly wants to move forward, but cannot commit to a real step.
- The observed behavior is a slow shuffle: small contact-preserving motions
  inch the body forward instead of producing clear alternating swing/stance.

Diagnosis:

- Section 40 fixed the distribution boundary between Standing and Walking.
- It did not provide a positive learning signal for how Walking should realize
  forward motion.
- The active WALK terms only contained velocity tracking and under-speed
  pressure. That can say "move forward", but it does not say "lift and exchange
  feet while moving forward".
- The gait constraint bridge is a negative pressure. It can punish bad phase
  structure, but by itself it is too weak and too indirect to create a first
  stepping behavior from a conservative balance solution.

Design correction:

- Keep Standing and Walking height unified through `balance_common_terms` and a
  single `base_height_target`.
- Keep transition samples as WALK samples, not a third reward mode.
- Re-enable the existing positive gait-style terms only in WALK routing:
  - `feet_phase`;
  - `feet_phase_contrast`;
  - `feet_phase_contact`.
- Do not add these terms to `balance_common_terms` or `stand_terms`. Zero-command
  Standing must not receive a foot-swing reward.

Implemented owner contract:

- Active SAC G1 MuJoCo owner YAML sets:
  - `feet_phase: 1.0`;
  - `feet_phase_contrast: 0.8`;
  - `feet_phase_contact: 0.5`.
- These terms are listed only under `reward.mode.walk_terms`.
- `reward.mode.stand_terms` remains limited to zero-command behavior, and
  `reward.mode.balance_common_terms` remains the shared balance/height owner.

Validation expectation:

- Config tests must prove gait-style terms have nonzero scales and are present
  only in WALK terms.
- Reward dispatch tests must prove a zero-command sample receives no gait-style
  positive reward even if its feet match the phase target, while a nonzero
  command sample can receive that reward.
- Existing transition tests must continue to prove transition commands keep
  `gait_enabled=1`.

## 42. 2026-06-26 Standing Must Cover Return-To-Stand Recovery

Observation after restoring WALK-only gait reward:

- Forward walking is now possible.
- Standing still has a soft-leg feel.
- The stand-to-walk switch is unstable, and walk-to-stand is more likely to
  fall than a fresh zero-command standing reset.

Diagnosis:

- This is not primarily a missing Walking gait signal anymore.
- The active training distribution still under-represents return-to-stand
  recovery.
- Before this change, G1 commands were sampled at reset, so one episode mostly
  stayed in one command mode.
- Standing reset samples also used `standing_reset_base_qvel_limit=0.0`, so
  zero-command Standing was trained from a quiet initial state.
- Interactive playback is different: keyboard commands change inside the same
  episode, and walk-to-stand inherits residual base velocity, foot placement,
  and body momentum from Walking.

Design correction:

- Add runtime command resampling to G1 using the same owner distribution as
  reset sampling:
  - Standing samples;
  - Transition samples;
  - normal Walking samples.
- Keep this in the G1 env owner, not in scripts.
- When a runtime resample changes a sample to zero command, immediately refresh
  `gait_enabled` and set the gait phase to the double-stance stand phase before
  reward and observation construction.
- Add a small standing reset velocity range in the active mixed config so
  Standing learns to absorb residual motion:
  `standing_reset_base_qvel_limit: 0.2`.

Implemented owner contract:

- Active SAC G1 MuJoCo owner YAML:
  - `env.commands.resampling_time: 2.0`;
  - `env.standing_reset_base_qvel_limit: 0.2`.
- Mixed-mode stage uses the same values.
- `standing_sanity` and `walking_sanity` keep their mode fractions and remain
  diagnostic stages.

Validation expectation:

- Config tests must prove runtime resampling and standing recovery qvel reach
  the env override.
- Env lifecycle tests must prove runtime resampling can switch an env from
  WALK to STAND and refresh `gait_enabled` plus double-stance phase.
- Live-path sentinel must still prove the observation contract and mode reward
  logs survive in standing, walking, and mixed stages.

## 43. 2026-06-26 Standing Should Share The Walking Posture Prior

Clarified observation:

- Standing is not merely unstable during mode transition.
- The standing posture itself is wrong.
- The policy keeps making balance corrections, the legs look soft, and the feet
  gradually become staggered front-to-back instead of staying aligned.

Diagnosis:

- The Standing objective already contains balance, height, velocity damping, and
  action regularization.
- The active Walking objective already has a posture prior through the shared
  `pose` term in `balance_common_terms`.
- Therefore the clean solution is not to add a separate STAND-only foot geometry
  reward. A separate Standing geometry can create a new discontinuity at the
  stand/walk boundary.
- The correct invariant is: Standing and Walking should share the same nominal
  body posture prior; Walking adds velocity tracking and gait reward on top.

Design correction:

- Keep `pose` in `reward.mode.balance_common_terms`, so both STAND and WALK use
  the same posture reward.
- Strengthen the shared `pose` scale instead of adding another STAND-only
  geometry term.
- Keep WALK-only terms limited to command tracking and gait-style rewards.
- Keep STAND-only terms limited to zero-command damping and drift suppression.

Implemented owner contract:

- Active SAC G1 MuJoCo owner YAML:
  - `pose: -0.3`;
  - `pose` remains in `balance_common_terms`;
  - no separate `stand_feet_sagittal_l2` live term.

Validation expectation:

- Config tests must prove `pose` is a shared common term and that no STAND-only
  foot geometry term is active.
- Existing reward-dispatch tests must continue to prove common terms apply to
  both STAND and WALK while walking tracking/gait terms remain WALK-only.
