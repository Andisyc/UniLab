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
