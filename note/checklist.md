# UniLab to RoboJudo_Real Checklist

## 0. 文档原则

这个 checklist 用来检查迁移、调试和后续实现中的关键链路。

每次修改代码后，至少更新对应条目：

- 哪个链路被改动。
- 改动后需要重新验证什么。
- 如果出问题，应该优先读哪些文件。

## 0.5 已发现问题与修复要求

- [x] 修复 UniLabPolicy DoF 配置：不能使用 12 DoF 的 `G1UnitreeDoF`，必须使用 29 DoF。已落地到 `G1UniLabDoF`。
- [x] 修复 UniLabPolicy default_pos：不能沿用 RoboJudo_Real `G1_29DoF.default_pos`，必须覆盖为 UniLab `stand` keyframe 后 29 维。已落地到 `G1UniLabDoF.default_pos`。
- [x] 修复 UniLabPolicy command map：不能继承 RoboJudo `UnitreePolicyCfg.max_cmd` / `commands_map`。已落地到 `G1UniLabPolicyCfg.command_maps`。
- [x] 修复 UniLabPolicy observation：不能照搬 RoboJudo `UnitreePolicy` obs 顺序，必须按 UniLab `joystick.py:403` 拼接。已落地到 `UniLabPolicy.get_observation()`。
- [x] 修复 UniLabPolicy gait phase：不能使用 RoboJudo `sin/cos phase`，必须使用 UniLab raw left/right phase。已落地到 `UniLabPolicy.gait_phase`。
- [x] 修复 UniLabPolicy ONNX 推理：不能硬编码 input name 为 `actor_obs`。已落地为读取 `session.get_inputs()` / `session.get_outputs()`。
- [x] 修复 prepare 阶段时间推进问题：确认 dry run 是否推进 gait phase，并冻结或 reset。已通过 pipeline dry-run sentinel `[UNILAB_FREEZE_PHASE]` 冻结。
- [ ] 修复 gravity 对齐风险：代码已按 RoboJudo `xyzw` gravity 函数构造等价 `-gravity` 段；仍需在 RoboJudo 运行环境中用 standing sample 数值验证。
- [x] 修复 action_scale 风险：不能继承 RoboJudo `UnitreePolicyCfg.action_scale = 0.25`，当前 UniLab G1WalkFlat 是 `1.0`。已落地到 `G1UniLabPolicyCfg.action_scale`。
- [ ] 运行环境验证：当前普通 Python 环境缺少 `box` 依赖，配置实例化需要在 RoboJudo_Real 正确环境中重新执行。
- [ ] 每完成一个修复，必须在对应章节把检查项标记或补充新的代码锚点。

重点检查文件：

- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:403`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:629`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:647`
- `C:\ArtiIntComVis\UniLab\src\unilab\assets\robots\g1\scene_flat.xml:52`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\common\commands.py:13`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\env\g1_env_cfg.py:7`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\env\g1_env_cfg.py:46`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\policy_cfgs.py:76`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:325`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\asap_policy.py:142`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\unilab_policy.py`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\policy\g1_unilab_policy_cfg.py`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\g1_cfg.py`

## 1. 仓库与入口

- [ ] 确认 UniLab 仓库位置：`C:\ArtiIntComVis\UniLab`
- [ ] 确认 RoboJudo_Real 仓库位置：`C:\ArtiIntComVis\RoboJudo_Real`
- [ ] 确认 UniLab policy 导出文件路径。
- [ ] 确认 RoboJudo_Real 运行入口。
- [ ] 确认 sim2sim 配置入口。
- [ ] 确认真机配置入口。

重点检查文件：

- `C:\ArtiIntComVis\RoboJudo_Real\scripts\run_pipeline.py`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\g1_cfg.py:51`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\g1_cfg.py:77`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:178`

## 2. Policy 接入链路

- [ ] 新增 `UniLabPolicy` 后，确认它能被 RoboJudo policy registry/import 发现。
- [ ] 确认 config 中的 `policy_type` 与 class name 完全一致。
- [ ] 确认 policy 文件路径存在。
- [ ] 确认 ONNX/TorchScript 输入名和输出名正确。
- [ ] 确认 policy 推理设备与 RoboJudo_Real 当前运行方式兼容。

重点检查文件：

- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\__init__.py`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\base_policy.py:74`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\base_policy.py:137`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\asap_policy.py`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\unitree_policy.py:92`

## 3. Observation 维度与顺序

- [ ] 确认 UniLab G1WalkFlat policy observation 维度为 98。
- [ ] 确认 RoboJudo_Real 中构造的 observation 维度为 98。
- [ ] 确认 observation 拼接顺序与 UniLab 训练代码一致。
- [ ] 确认 observation 顺序是 `gyro * 0.25, -gravity, diff, dof_vel * 0.05, action_delta, command, gait_phase`。
- [ ] 确认 command 位于 action 段之后、gait phase 之前。
- [ ] 确认 gait phase 直接使用 raw left/right phase，不是 RoboJudo UnitreePolicy 的 `sin/cos`。
- [ ] 确认 action 段使用的是上一帧 policy action delta，不是加 default 后的 PD target。
- [ ] 确认没有照搬 RoboJudo UnitreePolicy 的 observation 顺序。

UniLab 重点检查文件：

- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:316`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:386`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:403`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:619`
- `C:\ArtiIntComVis\UniLab\conf\ppo\task\g1_walk_flat\mujoco.yaml:33`

RoboJudo_Real 重点检查文件：

- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\unitree_policy.py:92`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\asap_policy.py`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\environment\base_env.py:167`

## 4. 四元数与重力投影

- [ ] 确认 RoboJudo_Real 的 `base_quat` 顺序是 `xyzw`。
- [ ] 确认 UniLab projected gravity 计算函数期望的四元数顺序。
- [ ] 确认没有把 `xyzw` 直接传给期望 `wxyz` 的函数。
- [ ] 静止站立时 projected gravity 数值应接近训练中的站立分布。
- [ ] pitch/roll 变化时 projected gravity 方向应连续变化，不应跳变。
- [ ] 当前静态对比结果：RoboJudo_Real `base_quat` 初始化和工具函数都按 `xyzw` 读取。
- [ ] 当前静态对比结果：UniLab G1WalkFlat 从 sensor `torso_upvector` 读取 `gravity`，actor obs 中拼的是 `-gravity`。
- [ ] 实现 UniLabPolicy 时不能只按函数名选择 `get_gravity_orientation()`；必须用 standing sample 验证最终进入 obs 的 3 维 gravity 段是否等价于 UniLab 的 `-gravity`。

重点检查文件：

- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\environment\base_env.py:167`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\environment\base_env.py:34`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\utils\util_func.py:50`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\environment\unitree_cpp_env.py`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\base.py:31`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:352`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:406`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:386`

## 5. DoF 顺序与默认角

- [ ] 确认 UniLab policy 的 action DoF 数量。
- [ ] 确认 RoboJudo_Real policy config 的 action DoF 数量。
- [ ] 确认 joint name 顺序通过 DoFAdapter 后与 UniLab 训练顺序一致。
- [ ] 确认 `default_pos` 等于 UniLab `default_angles`。
- [ ] 确认手臂、腰部、腿部关节是否都在 action space 中。
- [ ] 如果某些关节在 UniLab 中固定，在 RoboJudo_Real 中也不能被 policy 意外控制。
- [ ] 不要复用 RoboJudo_Real 的 `G1UnitreeDoF` 作为 UniLab G1WalkFlat policy DoF；它只有 12 个腿部 DoF，而 UniLab G1WalkFlat 当前是 29 action DoF。
- [ ] 可以以 RoboJudo_Real 的 `G1_29DoF` 为起点，但必须覆盖 `default_pos` 为 UniLab `stand` keyframe 的后 29 维。
- [ ] 当前静态对比结果：UniLab 和 RoboJudo_Real 的 29 actuator/joint 顺序一致。
- [ ] 当前静态对比结果：UniLab `stand` default 与 RoboJudo_Real `G1_29DoF.default_pos` 不一致，不能直接沿用 RoboJudo 默认角。
- [ ] 需要特别检查这些默认角差异：腿部 `hip_pitch/knee/ankle_pitch`，左右肩 pitch/roll，左右 elbow。

重点检查文件：

- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\policy\g1_unitree_policy_cfg.py:105`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\policy\g1_unitree_policy_cfg.py:113`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\env\g1_env_cfg.py:7`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\env\g1_env_cfg.py:46`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:193`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\base.py:56`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\common\base.py:81`
- `C:\ArtiIntComVis\UniLab\src\unilab\assets\robots\g1\scene_flat.xml:52`
- `C:\ArtiIntComVis\UniLab\src\unilab\assets\robots\g1\g1.xml:328`
- `C:\ArtiIntComVis\RoboJudo_Real\assets\robots\g1\g1_29dof_rev_1_0.xml:243`

当前默认角差异记录：

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

## 6. Action 语义

- [ ] 确认 UniLabPolicy 返回 action delta。
- [ ] 确认 UniLabPolicy 内部没有加 default_pos。
- [ ] 确认 RoboJudo_Real `PolicyWrapper.get_pd_target()` 只加一次 default_pos。
- [ ] 确认 action scale 使用 UniLab 配置。
- [ ] 确认 action clip 不比训练时更激进。
- [ ] action 为零时，PD target 应等于 default_pos。
- [ ] 当前静态对比结果：UniLab G1WalkFlat `action_scale = 1.0`。
- [ ] 当前静态对比结果：RoboJudo_Real `UnitreePolicyCfg.action_scale = 0.25`，不能继承给 UniLabPolicy。
- [ ] 当前静态对比结果：RoboJudo_Real `BasePolicy.get_action()` 先更新 `last_action`，再 clip/scale；UniLab G1WalkFlat obs 中的 `current_actions` 是传给 `apply_action()` 的 action。UniLabPolicy 必须明确 observation 里的 action 段使用 raw action 还是 processed action。
- [ ] 对当前 G1WalkFlat 而言，因为 UniLab `action_scale = 1.0`，raw action 与 scaled action 暂时等价；但如果后续 action_scale 改动，这里会变成隐性 bug。

重点检查文件：

- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:50`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\base_policy.py`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\base_policy.py:77`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\base_policy.py:85`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\base_policy.py:91`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\policy_cfgs.py:76`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:629`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:647`

## 7. Gait Phase

- [ ] 确认 UniLab 训练配置中的 `gait_frequency`。
- [ ] 确认 RoboJudo_Real policy frequency。
- [ ] 确认 gait phase delta 是 `2 * pi * gait_frequency * ctrl_dt`。
- [ ] 确认 gait phase 每步按真实 control dt 更新。
- [ ] 确认 phase wrap 不产生数值跳变。
- [ ] 确认 observation 中的 phase 表达形式与训练一致：UniLab 当前是 raw phase，不是 `sin/cos`。
- [ ] 确认 reset 或 prepare 阶段 phase 初值合理。
- [ ] 确认 `prepare()` 阶段是否会推进 UniLabPolicy 的 gait phase；RoboJudo_Real 当前 `prepare()` 会调用 `step(dry_run=True)`，而 `step()` 即使 dry run 也会执行 `post_step_callback()`。
- [ ] 如果 UniLabPolicy 在 `post_step_callback()` 中更新 gait phase，必须决定 prepare 阶段是否冻结 phase，或在 prepare 结束后 reset phase。

重点检查文件：

- `C:\ArtiIntComVis\UniLab\conf\ppo\task\g1_walk_flat\mujoco.yaml:33`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:285`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py:622`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\unitree_policy.py:92`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:246`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:283`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\pipeline\rl_pipeline.py:325`

## 8. Command 输入

- [ ] 确认手柄输入能映射到 `vx / vy / yaw_rate`。
- [ ] 确认键盘输入能映射到 `vx / vy / yaw_rate`。
- [ ] 确认 Unitree remote 输入能映射到 `vx / vy / yaw_rate`。
- [ ] 确认命令范围与 UniLab 训练 command range 一致。
- [ ] 确认 zero command 会逐步停下，而不是触发原地异常节奏。
- [ ] 确认 command scale 没有被重复乘。
- [ ] 确认没有把 RoboJudo UnitreePolicy 的 `commands * obs_scales.command * max_cmd` 直接套到 UniLabPolicy。
- [ ] 当前静态对比结果：UniLab `Commands.vel_limit` 是 `vx [-0.6, 1.0] / vy [-0.4, 0.4] / yaw [-0.8, 0.8]`。
- [ ] 当前静态对比结果：RoboJudo_Real `UnitreePolicyCfg.max_cmd` 和 `commands_map` 与 UniLab 训练 command range 不一致，UniLabPolicy 需要自己的 command map。
- [ ] 确认 joystick 轴方向是否与 UniLab command 正方向一致，尤其是 `vy` 和 `yaw_rate`。

重点检查文件：

- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\controller\ctrl_manager.py:122`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\unitree_policy.py:42`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\policy_cfgs.py:61`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\policy_cfgs.py:82`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\policy_cfgs.py:83`
- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\common\commands.py:13`
- `C:\ArtiIntComVis\RoboJudo_Real\docs\controller.md`
- `C:\ArtiIntComVis\RoboJudo_Real\docs\policy.md`

## 8.5 Policy 文件与推理接口

- [ ] 确认 UniLab 导出的 policy 文件实际是 ONNX 还是 TorchScript。
- [ ] 如果是 ONNX，运行时必须读取 `session.get_inputs()` 和 `session.get_outputs()`，不要硬编码 `actor_obs` 或 `action`。
- [ ] 当前静态对比结果：UniLab 不同训练脚本导出的 ONNX 名称并不完全一致，可能是 `obs/action`，也可能是 `obs_history/actions`。
- [ ] 当前静态对比结果：RoboJudo_Real `AsapPolicy` 硬编码了 ONNX input `"actor_obs"`，不能直接复用为 UniLabPolicy。
- [ ] 确认 ONNX 输入最后一维等于 98。
- [ ] 确认 ONNX 输出最后一维等于 29。
- [ ] 确认 UniLabPolicy 的 `last_action` 保存的是 scale/clip 后用于 PD 的 action delta，还是 raw network output；该选择必须与 observation 训练语义一致。

重点检查文件：

- `C:\ArtiIntComVis\UniLab\scripts\train_mlx_ppo.py:287`
- `C:\ArtiIntComVis\UniLab\scripts\train_mlx_ppo.py:288`
- `C:\ArtiIntComVis\UniLab\src\unilab\algos\torch\him_ppo\runner.py:243`
- `C:\ArtiIntComVis\UniLab\src\unilab\algos\torch\him_ppo\runner.py:244`
- `C:\ArtiIntComVis\UniLab\scripts\deploy\sim_prototype.py:313`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\asap_policy.py:30`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\asap_policy.py:142`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\policy\asap_policy.py:146`

## 9. Sim2Sim 首轮验收

- [ ] RoboJudo_Real MuJoCo 场景中只有一个 G1。
- [ ] policy 能成功加载。
- [ ] prepare 阶段不会把机器人拉到奇怪姿态。
- [ ] zero command 下能稳定站立或保持训练预期行为。
- [ ] 小 `vx` 下能向前走。
- [ ] 小 `yaw_rate` 下能转向。
- [ ] action 不出现持续饱和。
- [ ] dof_pos 不出现持续偏置。

失败时优先检查：

- observation 顺序
- default_pos
- action 是否 double default
- gait phase
- command scale

## 10. 真机前安全检查

- [ ] 急停可用。
- [ ] 网络接口配置正确。
- [ ] Unitree SDK / UnitreeCppEnv 能正常读取状态。
- [ ] action 为零时不会产生非默认目标角。
- [ ] policy 输出异常时有 clip 或保护。
- [ ] 低速命令测试前，先完成 default pose dry run。
- [ ] 第一次真机测试只允许小速度。

重点检查文件：

- `C:\ArtiIntComVis\RoboJudo_Real\docs\unitree_setup.md`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\environment\unitree_env.py:393`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\environment\unitree_env.py:498`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\environment\unitree_cpp_env.py:156`
- `C:\ArtiIntComVis\RoboJudo_Real\robojudo\config\g1\g1_cfg.py:77`

## 11. Carrying Locomotion 后续链路

- [ ] 第一版不要把 carrying reward 混入 UniLab migration。
- [ ] 确认固定手臂 DOF 是训练任务设计的一部分，不是部署适配的一部分。
- [ ] 确认 bottom-supported carry mode 的手臂姿态来源。
- [ ] 确认物体质量、尺寸、接触关系进入训练环境的方式。
- [ ] 确认 carrying relation reward 与 locomotion reward 的边界。
- [ ] 确认弯腰姿态是 emergent adaptation，而不是直接被外观奖励硬推出来。

后续重点检查文件：

- `C:\ArtiIntComVis\UniLab\src\unilab\envs\locomotion\g1\joystick.py`
- 未来新增的 carrying locomotion task 文件。
- 未来新增的 object/carry reward config 文件。
