# UniLab Policy Migration to RoboJudo Real

## 目标边界

目标不是把 UniLab 重新写成 RoboJudo 的 Environment，而是把 UniLab 训练出的 G1 locomotion policy 接入 RoboJudo_Real 的既有部署链路。

这个判断的核心原因是：

- UniLab 已经负责训练、导出和 sim/prototype 验证。
- RoboJudo_Real 已经负责真机通信、控制器输入、关节映射、PD target 下发。
- 两者之间最自然的接口是 policy adapter：把 RoboJudo 的 `env_data + ctrl_data` 转换成 UniLab policy 期望的 observation，再把 policy 输出转换成 RoboJudo 期望的 action delta。

因此第一版方案只新增 UniLab policy 适配层，不改 RoboJudo 的 Environment 和 Controller 主结构。

## 代码锚点

下面的行号来自当前本地代码阅读结果。后续如果 RoboJudo_Real 或 UniLab 更新，应该先重新核对这些行号，再继续实现。

### RoboJudo_Real

- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:178`：根据配置创建 Environment。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:182`：创建 ControllerManager。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:185`：创建 PolicyWrapper。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:193`：用 policy action DoF 覆盖/适配 env DoF。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:259`：`RlPipeline.step()` 主循环开始。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:261`：`env.update()` 读取状态。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:264`：`env.get_data()` 取状态。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:267`：`ctrl_manager.get_ctrl_data(env_data)` 取控制命令。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:273`：`policy.get_observation(env_data, ctrl_data)` 构造 observation。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:276`：`policy.get_pd_target(obs)` 推理动作并转 PD target。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:280`：`env.step(pd_target, hand_pose)` 下发动作。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:50`：`PolicyWrapper.get_pd_target()` 中执行 `action + default_pos`。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\unitree_policy.py:92`：现有 UnitreePolicy 的 observation 构造入口。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\unitree_policy.py:42`：现有 UnitreePolicy 的 command 读取逻辑入口。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\environment\base_env.py:167`：Environment `get_data()` 返回状态字段。
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\environment\unitree_cpp_env.py:156`：`UnitreeCppEnv.step()` 真机/底层下发入口。

### UniLab

- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:316`：G1WalkFlat observation 维度注释，当前为 98。
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:386`：`_compute_obs()` 入口。
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:403`：actor observation 拼接顺序开始。
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:405`：gyro 乘 `0.25`。
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:406`：gravity 使用负号。
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:410`：command 位于 action 后、gait phase 前。
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:411`：gait phase 直接进入 obs，不是 sin/cos。
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:619`：`last_actions/current_actions` 更新位置。
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:622`：gait phase 更新位置。
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:629`：`ctrl = actions * action_scale + default_angles`。
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:647`：`G1WalkControlConfig.action_scale = 1.0`。
- `C:\ArtiIntComVis\UniLab\conf\ppo\task\g1_walk_flat\mujoco.yaml:33`：`gait_frequency: 1.5`。

## 当前静态对比结论

这些不是待讨论的猜测，而是已经由本地 UniLab 与 RoboJudo_Real 代码对比得到的实现约束。后续实现时应直接按这些结论修复方案，除非重新检查代码后发现仓库已更新。

### 1. DoF 顺序基本兼容，但默认角不兼容

UniLab `g1.xml` 与 RoboJudo_Real `g1_29dof_rev_1_0.xml` 的 29 个 actuator/joint 顺序一致。

但是 UniLab G1WalkFlat 使用 `stand` keyframe 作为 `default_angles`，RoboJudo_Real `G1_29DoF.default_pos` 是另一套默认角。它们不相等。

因此：

- UniLabPolicy 不能复用 RoboJudo_Real 的 `G1UnitreeDoF`，因为它只有 12 个腿部 DoF。
- UniLabPolicy 可以以 RoboJudo_Real 的 `G1_29DoF` joint order 为基础。
- UniLabPolicy 必须覆盖 `default_pos` 为 UniLab `stand` keyframe 的后 29 维。
- 否则 `pd_target = action + default_pos` 会把 policy 输出加到错误站姿上。

### 2. Command range 不兼容

UniLab G1WalkFlat 的训练 command range 是：

```text
vx: [-0.6, 1.0]
vy: [-0.4, 0.4]
yaw_rate: [-0.8, 0.8]
```

RoboJudo_Real 现有 `UnitreePolicyCfg.max_cmd` / `commands_map` 与这个范围不一致。

因此 UniLabPolicy 需要自己的 command map，不能继承 RoboJudo `UnitreePolicy` 的 command scale。

### 3. Gravity 语义需要实测对齐

RoboJudo_Real 的 `base_quat` 与工具函数按 `xyzw` 工作。UniLab G1WalkFlat 不是直接从四元数函数命名推断 gravity，而是从 sensor `torso_upvector` 取 `gravity`，actor obs 中拼接的是 `-gravity`。

因此实现时必须用 standing sample 或小姿态扰动验证最终进入 UniLab obs 的 3 维 gravity 段，而不是只看函数名。

### 4. ONNX 输入输出名不能硬编码

UniLab 不同训练脚本导出的 ONNX input/output 名称并不完全一致，可能是 `obs/action`，也可能是 `obs_history/actions`。RoboJudo_Real 的 `AsapPolicy` 硬编码 `actor_obs`，不能直接复制给 UniLabPolicy。

因此 UniLabPolicy 必须读取 ONNX session 的真实 input/output name，并断言：

- 输入最后一维为 98。
- 输出最后一维为 29。

### 5. Prepare 阶段可能推进 policy 内部时间

RoboJudo_Real 的 `prepare()` 会调用 `step(dry_run=True)`，而 `step()` 即使 dry run 也会执行 `post_step_callback()`。

如果 UniLabPolicy 在 `post_step_callback()` 中更新 gait phase，那么 prepare 阶段会提前推进 gait phase。实现时必须二选一：

- dry run 不推进 UniLab gait phase。
- prepare 结束后 reset UniLab gait phase。

## 设计不变量

### 1. Environment 不变量

RoboJudo_Real 的 Environment 继续拥有真机/仿真的状态读取与动作下发。

关键链路：

- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py`
- `RlPipeline.step()`
- `env.update()`
- `env.get_data()`
- `env.step(pd_target, hand_pose)`

第一版不新增 UniLabEnv，不把 UniLab 的 MuJoCo 环境搬进 RoboJudo_Real。

原因是 UniLab 的 MuJoCo 环境适合训练和仿真验证；RoboJudo_Real 的价值在于真实 Unitree 接口、控制器输入、DoF adapter 和部署节奏。

### 2. Controller 不变量

RoboJudo_Real 的 Controller 继续负责手柄、键盘或 Unitree 控制输入。

UniLab G1 velocity tracking policy 需要的是 3 维速度命令：

- `vx`
- `vy`
- `yaw_rate`

这些命令应从 RoboJudo 的 `ctrl_data` 读取，再按 UniLab 训练时的 command scale/range 转换。

第一版不重新实现手柄控制。

注意：这里的“转换”指把控制器输入变成 UniLab 训练时采样的物理速度命令范围。不要直接套用 RoboJudo `UnitreePolicy` 的 `commands * obs_scales.command * max_cmd`，除非已经确认 UniLab 导出的 policy 训练 observation 也使用同一缩放。

### 3. Policy 不变量

UniLab policy 输出应保持为相对默认关节角的 action delta。

RoboJudo_Real 的 `PolicyWrapper.get_pd_target()` 会执行：

```text
pd_target = policy_action + policy.default_pos
```

因此 UniLabPolicy 内部不能再次加默认关节角。否则会出现 double default，真机目标角会整体偏移。

### 4. Gait Phase 不变量

UniLab G1 locomotion policy 的 observation 中包含 gait phase。

因此 RoboJudo_Real 侧新增的 UniLabPolicy 必须自己维护 gait phase，而不能省略。

关键训练配置：

- UniLab `gait_frequency = 1.5`
- control frequency / dt 必须和导出 policy 的训练配置一致或明确换算。

如果 gait phase 更新不一致，policy 可能在仿真中仍能勉强动，但真机会表现出节奏错位、脚步不稳、周期性抖动。

## 建议新增文件

### 1. `robojudo/policy/unilab_policy.py`

职责：

- 加载 UniLab 导出的 ONNX 或 TorchScript policy。
- 从 `env_data` 构造 UniLab G1WalkFlat 的 98 维 observation。
- 从 `ctrl_data` 读取速度命令。
- 维护上一帧 policy action delta，用于填入 UniLab obs 中的 action 段。
- 维护 raw left/right `gait_phase`。
- 输出 action delta。
- 对 ONNX 输入输出名和维度做运行时断言。
- 处理 prepare/dry-run 阶段的 gait phase 冻结或 reset。

不负责：

- 不下发 PD。
- 不访问 Unitree SDK。
- 不直接改 Environment。
- 不覆盖 Controller 行为。

### 2. `robojudo/config/g1/policy/g1_unilab_policy_cfg.py`

职责：

- 声明 UniLab policy 文件路径。
- 声明 observation/action DoF 配置，使用 29 DoF joint order，并覆盖为 UniLab `stand` default angles。
- 声明 action scale、clip、command scale。
- 声明 gait frequency。
- 声明 policy frequency。
- 声明 UniLab command range，不能沿用 RoboJudo `UnitreePolicyCfg.max_cmd`。

### 3. `robojudo/config/g1/g1_cfg.py`

新增两个配置类即可：

- `g1_unilab`：RoboJudo MuJoCo sim2sim 测试。
- `g1_real_unilab`：Unitree 真机部署入口。

第一阶段必须先跑 `g1_unilab`，确认 observation、action、DoF mapping、gait phase 都正确，再进入 `g1_real_unilab`。

## Observation 对齐

UniLab G1WalkFlat 速度跟踪 policy 的 observation 结构是：

```text
gyro * 0.25
-gravity
dof_pos - default_angles
dof_vel * 0.05
previous/current policy action delta
velocity command: vx, vy, yaw_rate
raw left/right gait_phase
```

总维度为 98。

这里有一个容易写错的点：RoboJudo_Real 现有 `UnitreePolicy` 的 observation 顺序是 `ang_vel, gravity, command, dof_pos, dof_vel, last_action, sin_phase, cos_phase`。它只能作为 RoboJudo policy 写法参考，不能作为 UniLab G1WalkFlat observation 的目标顺序。

需要重点确认：

- RoboJudo 的 `base_quat` 是 `xyzw`。
- UniLab 计算 projected gravity 时使用的四元数顺序是否一致。
- `dof_pos` 的关节顺序是否经过 DoFAdapter 后与 UniLab policy 训练顺序一致。
- `default_pos` 必须覆盖为 UniLab 的 `default_angles`；当前 RoboJudo 默认角不相等，不能直接沿用。
- action 段是 policy 输出的 action delta，而不是加了 default 后的 target。
- command 不要套用 RoboJudo `UnitreePolicy` 的 command observation scale，除非已经确认 UniLab 导出的 policy 训练时也使用同一缩放。
- gait phase 不要套用 RoboJudo `UnitreePolicy` 的 `sin/cos` 表达；UniLab 当前代码直接拼接两路相位值。

## Action 对齐

UniLab G1WalkFlat 的控制逻辑可以抽象为：

```text
target = default_angles + action * action_scale
```

RoboJudo_Real 的 policy wrapper 已经执行：

```text
pd_target = action + default_pos
```

因此 UniLabPolicy 返回值应该是：

```text
action_delta = raw_policy_output * action_scale
```

其中第一版按 UniLab G1WalkFlat 配置使用：

```text
action_scale = 1.0
```

不能在 UniLabPolicy 内部返回完整 target。

## Command 对齐

UniLab 训练时的 command 是 policy 的任务语义输入。

RoboJudo_Real 的 Controller 输出是用户控制输入。

两者不是同一个概念，需要一层明确转换：

```text
controller axes / keyboard / unitree command
-> normalized user command
-> UniLab command range
-> policy observation command
```

第一版应只支持速度跟踪：

- `vx`
- `vy`
- `yaw_rate`

当前代码对比已经确认：UniLab G1WalkFlat 的训练 command range 是 `vx [-0.6, 1.0] / vy [-0.4, 0.4] / yaw_rate [-0.8, 0.8]`，而 RoboJudo_Real 现有 Unitree command range 与它不一致，所以 UniLabPolicy 需要自己的 command mapping。

不要在第一版混入高度、身体 pitch/roll、手臂姿态或负载信息。

这些属于后续 carrying locomotion policy 的 command/condition 设计，而不是当前 UniLab policy migration 的必要条件。

## 推荐实现顺序

### 阶段 1：只做 sim2sim policy adapter

目标是在 RoboJudo_Real 的 MuJoCo 环境中运行 UniLab 导出的 G1 locomotion policy。

验收标准：

- 场景中只有一个 G1。
- 手柄或键盘命令能改变 `vx / vy / yaw_rate`。
- observation 维度为 98。
- policy 输出维度等于 action DoF 数量。
- 站立、慢速前进、转向都没有明显关节偏置。

### 阶段 2：加入严格诊断输出

在进入真机前，需要输出或断言：

- obs shape
- action shape
- action min/max
- command values
- gait phase values
- default_pos 前几项
- dof_pos - default 的范围
- projected gravity 的范围

这些诊断应该只在 debug 模式打开，避免影响部署频率。

### 阶段 3：进入真机 dry run

目标不是立刻行走，而是验证真机链路。

检查：

- 不启动 policy 行走时，机器人能进入 policy default pose。
- action 为零时，PD target 等于 default_pos。
- 小幅速度命令时，action 不突变。
- 急停/退出逻辑可用。

### 阶段 4：真机低速行走

只测试低速：

- 小 `vx`
- 零 `vy`
- 小 `yaw_rate`

不要一开始测试侧向或大角速度。

## 与 Carrying Locomotion 的关系

这次迁移方案服务于更大的 carrying locomotion 目标，但不是 carrying locomotion 本身。

它的作用是建立一个可靠的部署基座：

```text
UniLab training/export
-> RoboJudo sim2sim adapter
-> RoboJudo real deployment
```

后续 carrying locomotion 才会引入：

- 固定手臂 DOF。
- bottom-supported carry mode。
- 物体尺寸。
- 物体重量。
- object-robot carrying relation reward。
- locomotion under carrying condition。

第一版迁移不要把这些概念混进 UniLab locomotion policy adapter，否则会把“部署适配问题”和“新任务设计问题”纠缠在一起。

## 后续研究接口

当 UniLab locomotion policy 能稳定接入 RoboJudo_Real 后，可以新增第二类 policy：

```text
UniLabCarryPolicy
```

它与 UniLabPolicy 的区别不是部署框架，而是 observation / reward / command 语义：

- locomotion command 仍然是速度。
- arm carry posture 是 condition，不是速度 command。
- object relation 是任务约束，不是单纯姿态奖励。
- 弯腰行走是 carrying relation 与稳定 locomotion 共同诱导出的结果。

这能保持研究叙事干净：

```text
先证明 UniLab policy 能进 RoboJudo_Real。
再证明 carrying condition 会改变 locomotion 稳定性。
最后证明 relation-centered reward 比姿态外观奖励更不容易被 hacking。
```
