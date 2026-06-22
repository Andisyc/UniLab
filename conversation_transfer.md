# UniLab / RoboJudo_Real 工作交接记录

生成日期：2026-06-22

## 相关仓库

- UniLab：`C:\ArtiIntComVis\UniLab`
- RoboJudo_Real：`C:\ArtiIntComVis\RoboJudo_Real`
- Unitree RL Lab：`C:\ArtiIntComVis\unitree_rl_lab`
- 当前 Codex workspace：`C:\Users\ChengYuxuan\Documents\UniLab`

## 已写入 UniLab note 的文档

- `C:\ArtiIntComVis\UniLab\note\unilab_to_robojudo_real_plan.md`
- `C:\ArtiIntComVis\UniLab\note\checklist.md`

这些文档记录了 UniLab policy 迁移到 RoboJudo_Real 的设计方案、风险点和检查链路。

## UniLab G1 Locomotion 结论

G1 velocity locomotion 任务在：

- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py`

重要结论：

- 当前 G1WalkFlat 是速度跟踪，不是高度跟踪。
- 它有固定 base height penalty，但没有 height command tracking。
- 它带 gait control。
- observation 维度为 98。
- actor observation 顺序为：

```text
gyro * 0.25
-gravity
dof_pos - default_angles
dof_vel * 0.05
previous/current policy action delta
velocity command: vx, vy, yaw_rate
raw left/right gait_phase
```

关键代码锚点：

- `joystick.py:316`：obs 维度注释，98。
- `joystick.py:403`：actor observation 拼接开始。
- `joystick.py:405`：gyro 乘 `0.25`。
- `joystick.py:406`：gravity 使用负号。
- `joystick.py:410`：command 位于 action 后、gait phase 前。
- `joystick.py:411`：gait phase 直接进入 obs，不是 sin/cos。
- `joystick.py:622`：gait phase 更新。
- `joystick.py:629`：`ctrl = actions * action_scale + default_angles`。
- `joystick.py:647`：`action_scale = 1.0`。
- `conf\ppo\task\g1_walk_flat\mujoco.yaml:33`：`gait_frequency: 1.5`。

## RoboJudo_Real 迁移方案

核心设计：不要把 UniLab 作为新的 Environment 接进 RoboJudo_Real，而是作为新的 Policy adapter 接进去。

原因：

- UniLab 负责训练、导出和仿真验证。
- RoboJudo_Real 已经负责真机通信、控制器输入、DoF adapter、PD target 下发。
- 最自然的接口是：

```text
RoboJudo env_data + ctrl_data
-> UniLabPolicy 构造 UniLab obs
-> UniLab policy 推理 action delta
-> RoboJudo PolicyWrapper 加 default_pos
-> RoboJudo env.step(pd_target)
```

## 已落地到 RoboJudo_Real 的代码

新增：

- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\unilab_policy.py`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\policy\g1_unilab_policy_cfg.py`

修改：

- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\__init__.py`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\g1_cfg.py`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py`

新增配置：

- `g1_unilab`：RoboJudo MuJoCo sim2sim 测试。
- `g1_real_unilab`：Unitree 真机部署入口。

已落地内容：

- 新增 `UniLabPolicy`。
- 注册 `UniLabPolicy`。
- 新增 `G1UniLabPolicyCfg`。
- 新增 `G1UniLabDoF`，使用 29 DoF。
- `default_pos` 覆盖为 UniLab `stand` keyframe 后 29 维。
- `action_scale = 1.0`。
- command range 使用 UniLab G1WalkFlat 范围：

```text
vx: [-0.6, 1.0]
vy: [-0.4, 0.4]
yaw_rate: [-0.8, 0.8]
```

- ONNX input/output name 动态读取，不硬编码 `actor_obs`。
- 断言 obs dim = 98。
- 断言 action dim = 29。
- dry-run / prepare 阶段通过 `[UNILAB_FREEZE_PHASE]` 避免提前推进 gait phase。

已验证：

```bash
python -m py_compile C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\unilab_policy.py C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\policy\g1_unilab_policy_cfg.py C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\g1_cfg.py C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py
```

结果：通过。

未完成验证：

- 当前普通 Python 环境缺少 `box` 依赖，无法完整实例化 RoboJudo config。
- 需要在 RoboJudo_Real 正确运行环境中再次验证 `g1_unilab` / `g1_real_unilab`。
- gravity 段需要用 standing sample 数值验证，确认等价于 UniLab obs 中的 `-gravity`。

## DoF 与 Default Angle 对比

已确认：

- UniLab `g1.xml` 与 RoboJudo_Real `g1_29dof_rev_1_0.xml` 的 29 个 actuator/joint 顺序一致。

不能直接沿用：

- RoboJudo_Real `G1_29DoF.default_pos` 与 UniLab `stand` keyframe default 不一致。

重要差异：

```text
left_hip_pitch_joint: UniLab -0.312, RoboJudo -0.1
left_knee_joint: UniLab 0.669, RoboJudo 0.3
left_ankle_pitch_joint: UniLab -0.363, RoboJudo -0.2
right_hip_pitch_joint: UniLab -0.312, RoboJudo -0.1
right_knee_joint: UniLab 0.669, RoboJudo 0.3
right_ankle_pitch_joint: UniLab -0.363, RoboJudo -0.2
left_shoulder_pitch_joint: UniLab 0.2, RoboJudo 0
left_shoulder_roll_joint: UniLab 0.2, RoboJudo 0
left_elbow_joint: UniLab 0.6, RoboJudo 0
right_shoulder_pitch_joint: UniLab 0.2, RoboJudo 0
right_shoulder_roll_joint: UniLab -0.2, RoboJudo 0
right_elbow_joint: UniLab 0.6, RoboJudo 0
```

## UniLab play_interactive checkpoint 问题

用户命令：

```bash
uv run scripts/play_interactive.py --algo ppo --task go2_joystick_rough --sim mujoco \
  interactive.action_mode=policy interactive.keyboard=true
```

报错：

```text
Run not found for load_run=-1
```

原因：

`load_run=-1` 表示自动查找最新 run。对于 `go2_joystick_rough/mujoco`，UniLab 会找：

```text
<UniLab>/logs/rsl_rl_ppo/Go2JoystickRough/<run_dir>/model_*.pt
```

关键配置：

- `conf/ppo/config.yaml`
  - `algo.algo_log_name: rsl_rl_ppo`
  - `algo.load_run: "-1"`
  - `algo.checkpoint: -1`
- `conf/ppo/task/go2_joystick_rough/mujoco.yaml`
  - `training.task_name: Go2JoystickRough`

如果 logs 放成：

```text
UniLab/logs/logs/rsl_rl_ppo/Go2JoystickRough/...
```

会找不到。应为：

```text
UniLab/logs/rsl_rl_ppo/Go2JoystickRough/<run_dir>/model_*.pt
```

推荐命令 1：自动加载最新 run 最新 checkpoint。

```bash
uv run scripts/play_interactive.py --algo ppo --task go2_joystick_rough --sim mujoco \
  interactive.action_mode=policy interactive.keyboard=true
```

推荐命令 2：指定 run。

```bash
uv run scripts/play_interactive.py --algo ppo --task go2_joystick_rough --sim mujoco \
  interactive.action_mode=policy interactive.keyboard=true \
  algo.load_run=<run_dir_name>
```

推荐命令 3：指定 run 和 checkpoint。

```bash
uv run scripts/play_interactive.py --algo ppo --task go2_joystick_rough --sim mujoco \
  interactive.action_mode=policy interactive.keyboard=true \
  algo.load_run=<run_dir_name> \
  algo.checkpoint=<iteration>
```

例如：

```bash
uv run scripts/play_interactive.py --algo ppo --task go2_joystick_rough --sim mujoco \
  interactive.action_mode=policy interactive.keyboard=true \
  algo.load_run=2026-xx-xx_xx-xx-xx_mujoco \
  algo.checkpoint=2000
```

这会找：

```text
logs/rsl_rl_ppo/Go2JoystickRough/2026-xx-xx_xx-xx-xx_mujoco/model_2000.pt
```

最稳命令：直接给 checkpoint 绝对路径。

```bash
uv run scripts/play_interactive.py --algo ppo --task go2_joystick_rough --sim mujoco \
  interactive.action_mode=policy interactive.keyboard=true \
  algo.load_run=/absolute/path/to/model_2000.pt
```

需要检查：

```bash
ls logs/rsl_rl_ppo/Go2JoystickRough
ls logs/rsl_rl_ppo/Go2JoystickRough/<run_dir>
ls logs/rsl_rl_ppo/Go2JoystickRough/<run_dir>/model_*.pt
```

如果没有 `model_*.pt`，说明复制来的不是 `--algo ppo` 需要的 RSL-RL checkpoint。

## Carrying Locomotion 概念记录

当前研究概念：

- 用户只负责 locomotion 部分，manipulation 部分由他人负责。
- 目标是 carrying / affordance locomotion。
- 手臂姿态作为 condition，不是 locomotion command。
- 第一版选择固定手臂 DOF。
- carry mode 选择 bottom-supported carry mode。
- 弯腰行走不应直接作为 reward 或 command，而应作为在 carrying condition 下满足 locomotion + carrying relation 的 emergent adaptation。

核心抽象：

```text
robot-world relation:
  velocity tracking

object-robot relation:
  carrying relation tracking

policy behavior:
  bending, lowering, step shortening, balance recovery emerge
```

Reward 设计原则：

- 不直接奖励可见姿态。
- 先找任务关系。
- locomotion reward 追踪 robot-world motion relation。
- carrying reward 追踪 object-robot carrying relation。
- posture/action/contact penalty 只作为 boundary，不作为任务本身。

## 下一步建议

1. 在 RoboJudo_Real 正确环境中安装/确认依赖，至少包含 `box` / `python-box`。
2. 验证 `g1_unilab` config 能实例化。
3. 放置 UniLab 导出的 policy：

```text
C:\ArtiIntComVis\RoboJudo_Real\assets\models\g1\unilab\g1_walk_flat\policy.onnx
```

4. 运行 sim2sim：

```bash
python scripts/run_pipeline.py --config g1_unilab
```

5. 检查：

- obs dim = 98
- action dim = 29
- default_pos 是否为 UniLab stand pose
- zero action 时 PD target 是否等于 UniLab default_pos
- joystick command 是否在 UniLab command range
- gait phase 是否在 prepare/dry-run 阶段被冻结
- gravity 段是否和 UniLab `-gravity` 等价

6. sim2sim 成功后再考虑 `g1_real_unilab`。

