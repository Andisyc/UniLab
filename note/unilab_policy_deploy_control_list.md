# UniLab Policy Deploy Control List

生成日期：2026-06-23

目的：将 UniLab 训练得到的 G1 locomotion policy 部署到 RoboJudo_Real 时，把排查链路切成 7 个可独立验证的 section。后续测试必须按 section 逐项记录证据，避免只根据“机器人摔倒”反推原因。

当前最高优先级风险：

- 当前测试 run 是 `sac`，`run_config.json` 中 `algo.obs_normalization=true`。
- UniLab `scripts/train_offpolicy.py` 导出 ONNX 时会先对输入做 normalizer，再导出裸 actor。
- RoboJudo_Real 当前 `UniLabPolicy` 直接把 raw 98 维 obs 喂给 ONNX。
- 因此 Section 2 `Obs Normalization` 是第一优先级排查对象。

2026-06-23 Section 2 审计更新：

- `policy.onnx` 与 checkpoint 中的 FastSAC 裸 actor 在 raw input 上一致，max abs diff 约 `7e-7`。
- `FastSACLearner.__init__` 没有 `obs_normalization` 参数。
- fresh `FastSACLearner` 没有 `obs_normalizer` 成员。
- fresh `FastSACLearner.get_state_dict()` 不保存 `obs_normalizer`。
- 当前 `model_5000.pt` 也没有 `obs_normalizer`。
- 结论：这个 FastSAC run 的 `run_config.json` 中 `obs_normalization=true` 是误导性配置字段；当前训练/导出链路实际没有启用可部署的 obs normalization。Section 2 不是当前摔倒主因，应继续检查 Section 4 初始姿态与 Section 3 obs 数值。

2026-06-23 Section 4 审计更新：

- 新增并运行 `scripts/deploy/check_robojudo_unilab_section4.py`。
- `G1UniLabDoF.default_pos` 与 UniLab `scene_flat.xml` 的 `stand` keyframe 后 29 维完全一致。
- UniLab `stand` root qpos 为 `[0, 0, 0.754, 1, 0, 0, 0]`。
- RoboJudo `g1_29dof_rev_1_0.xml` 的 `qpos0` root 为 `[0, 0, 0.793, 1, 0, 0, 0]`。
- RoboJudo `qpos0` 的 29 个关节全为 0，与 UniLab default pose 最大偏差 `0.669` rad。
- RoboJudo XML 没有 keyframe，`MujocoEnv.reborn()` 的默认 keyframe reset 对该 XML 不可用。
- 结论：Section 4 当前 FAIL。RoboJudo 仿真必须在 policy 闭环前显式 reset/reborn 到 UniLab stand root+joint pose，否则第一帧 `dof_pos - default_angles` 已经严重偏离训练分布。

2026-06-23 Section 4 修复更新：

- 只修改 RoboJudo_Real，不修改 UniLab。
- 在 RoboJudo_Real `MujocoEnvCfg` 增加 `init_qpos`。
- 在 RoboJudo_Real `MujocoEnv` 的 `__init__`、`reset()`、`reborn()` 中支持配置式完整 qpos 初始化。
- 只给 `g1_unilab` sim config 设置 `UNILAB_G1_STAND_QPOS`，普通 `g1` 不使用该 init pose。
- 重跑 `scripts/deploy/check_robojudo_unilab_section4.py`：`summary: 0 fail(s), 3 warning(s)`。
- 当前有效初始 root 与 UniLab stand root 完全一致；有效初始 29 DoF 与 UniLab default pose 完全一致。

2026-06-23 Section 3 测试更新：

- 新增并运行 `scripts/deploy/check_robojudo_unilab_section3.py`。
- Section 3 结果：`summary: 0 fail(s), 0 warning(s)`。
- 测试脚本检查 RoboJudo_Real `UniLabPolicy.get_observation()` 源码中的 gravity 符号与 obs concat 顺序，并用 MuJoCo reset 到 `g1_unilab.env.init_qpos` 后构造第一帧 obs。
- 第一帧 obs 维度为 98。
- obs 分段为 `gyro[0:3]`、`gravity[3:6]`、`dof_pos_rel[6:35]`、`dof_vel[35:64]`、`last_action[64:93]`、`command[93:96]`、`gait_phase[96:98]`。
- 第一帧 `dof_pos - default_angles` 最大绝对值为 0。
- 第一帧 `dof_vel * 0.05`、`last_action`、`command` 均为 0。
- 修复 Section 5 后，第一帧 `gait_phase` 为 `[0, pi]`，与 RoboJudo `G1UniLabPolicyCfg.initial_gait_phase` 一致。
- Section 10 修复后，第一帧 gravity segment 为 `[0, 0, -1]`，与 UniLab actor obs 的 `-torso_upvector` 一致。
- 结论：Section 3 当前通过。初始姿态修复后，第一帧 obs 的顺序、维度、静态数值已经符合 UniLab G1WalkFlat actor obs contract。

2026-06-23 Section 6 测试更新：

- 新增并运行 `scripts/deploy/check_robojudo_unilab_section6.py`。
- Section 6 结果：`summary: 1 fail(s), 1 warning(s)`。
- 通过项：ONNX input/output 为 `obs [1,98] -> action [1,29]`；`action_scale=1.0`；`action_clip=None`；`action_beta=1.0`；`PolicyWrapper.get_pd_target()` 源码为 `pd_target = action + self.policy.default_pos`；DoFAdapter 映射 29/29 个 joint，且 UniLab policy joint order 与 RoboJudo env joint order 完全一致。
- 通过项：`pd_target - UniLab default_pos == action`，最大误差约 `5.96e-08`，说明 action 到 PD target 的加法和 DoF 映射没有错。
- 失败项：第一帧 zero phase obs 下，ONNX raw action 范围约 `[-0.837, 0.961]`，加上 UniLab default pose 后有 3 个 PD target 超出 G1 joint position limits：
  - `left_ankle_pitch_joint`: target `-1.20021`，limit `[-0.87267, 0.5236]`
  - `right_ankle_pitch_joint`: target `-1.12885`，limit `[-0.87267, 0.5236]`
  - `waist_pitch_joint`: target `0.534146`，limit `[-0.52, 0.52]`
- 额外对照：把 gait phase 改成 `[0, pi]` 后仍有同样 3 个限位问题，说明这不是单纯由 RoboJudo reset phase `[0,0]` 引起。
- 对照 UniLab `scene_flat.xml` 与 RoboJudo `g1_29dof_rev_1_0.xml`：上述 3 个 joint 的 position limit 一致。
- 结论：Section 6 当前 FAIL。wrapper 的 action-to-PD 公式和 DoF 映射正确，但 UniLab ONNX 在当前第一帧 obs 上输出的 action delta 会产生越界 PD target。下一步不应先盲目改 policy 或 clamp，而应确认 UniLab 原生 play/rollout 中同一 obs 是否也输出这个 action，以及 UniLab 后端是否在控制层有额外 clipping/saturation 语义 RoboJudo 没复现。

2026-06-23 Section 6 原生控制语义测试更新：

- 新增并运行 `scripts/deploy/check_unilab_native_section6_control.py`。
- 结果：`summary: 1 fail(s), 0 warning(s)`。
- 同一帧 obs 下，RoboJudo 部署 ONNX 与 UniLab checkpoint actor 输出一致，`onnx_vs_actor_max_abs = 7.078e-07`。
- 因此第一帧大 action 不是 ONNX 导出错误，也不是 RoboJudo ONNX runtime 特有错误。
- UniLab 原生 target 同样越过 3 个 joint limit：
  - `left_ankle_pitch_joint`: target `-1.20021`，limit `[-0.87267, 0.5236]`
  - `right_ankle_pitch_joint`: target `-1.12885`，limit `[-0.87267, 0.5236]`
  - `waist_pitch_joint`: target `0.534146`，limit `[-0.52, 0.52]`
- UniLab XML 中全部 actuator `ctrllimited=false`，所以没有 target ctrlrange clipping。
- UniLab XML 中全部 actuator `forcelimited=true`，存在 actuator force saturation；例如 ankle/waist force range 是 `[-50, 50]`。
- RoboJudo 当前不是复用 MuJoCo position actuator，而是在 `MujocoEnv.step()` 中用 `torque = (pd_target - dof_pos) * stiffness - dof_vel * damping` 自己算 torque，再按 `G1_29DoF.torque_limits` clip。
- 同一第一帧 action 下，UniLab actuator force 与 RoboJudo torque 差异很大：`torque_minus_unilab_force` 最大绝对值约 `91.6`。
- 典型差异：
  - `waist_pitch_joint`: UniLab `kp/kd=28.501/1.814`，force `15.2237`；RoboJudo `kp/kd=200/6`，torque `106.829`，约 `7.02x`。
  - `left_hip_pitch_joint`: UniLab force `30.0772`；RoboJudo torque `74.8579`，约 `2.49x`。
  - `left_knee_joint`: UniLab force `94.507`；RoboJudo torque `143.051`，约 `1.51x`。
- 结论修正：越界 target 是 UniLab 原生策略也会输出的现象，不是当前 deploy 的主差异；真正需要修复的是 RoboJudo 控制层没有复现 UniLab/MuJoCo actuator 的 `kp/kd/forcerange` 语义，导致相同 policy action 被转换成过大的 torque。

2026-06-23 Section 6 控制语义修复更新：

- 只修改 RoboJudo_Real，不修改 UniLab。
- 在 RoboJudo_Real `robojudo/config/g1/env/g1_env_cfg.py` 新增 `G1UniLabMujocoDoF`。
- `G1UniLabMujocoDoF` 继承 `G1_29DoF`，保留 joint order 与 joint position limits，只覆盖 UniLab stand `default_pos`、UniLab MuJoCo actuator `kp/kd`、以及 actuator `forcerange` 对应的 `torque_limits`。
- 在 RoboJudo_Real `robojudo/config/g1/g1_cfg.py` 中，仅让 `g1_unilab` 的 MuJoCo sim env 使用 `G1MujocoEnvCfg(dof=G1UniLabMujocoDoF(), init_qpos=UNILAB_G1_STAND_QPOS)`。
- `g1` 普通配置未使用 `G1UniLabMujocoDoF`。
- `g1_real_unilab` 仍显式覆盖为 `G1RealEnvCfg`，因此不把 UniLab MuJoCo actuator kp/kd 直接注入真机 env。
- 语法检查通过：`python -m py_compile robojudo/config/g1/env/g1_env_cfg.py robojudo/config/g1/g1_cfg.py`。
- 重跑 `scripts/deploy/check_robojudo_unilab_section4.py`：`summary: 0 fail(s), 3 warning(s)`，初始姿态修复仍保持有效。
- 重跑 `scripts/deploy/check_robojudo_unilab_section6.py`：`summary: 0 fail(s), 2 warning(s)`。剩余 warning 是 UniLab 原生也会输出的 target joint-limit violation，不再作为 RoboJudo deploy 差异。
- 重跑 `scripts/deploy/check_unilab_native_section6_control.py`：`summary: 0 fail(s), 0 warning(s)`。
- 修复后同一 first-frame action 下，RoboJudo torque 与 UniLab actuator force 对齐，`torque_minus_unilab_force` 最大绝对值约 `7.63e-06`。
- 典型修复结果：
  - `waist_pitch_joint`: UniLab force `15.2237`，RoboJudo torque `15.2237`，ratio `1.0`。
  - `left_hip_pitch_joint`: UniLab force `30.0772`，RoboJudo torque `30.0772`，ratio `1.0`。
  - `left_knee_joint`: UniLab force `94.507`，RoboJudo torque `94.507`，ratio `1.0`。
- 结论：Section 6 的控制语义差异已修复。RoboJudo `g1_unilab` 仿真现在在第一帧动作到力矩的边界上复现 UniLab MuJoCo actuator 的 kp/kd/forcerange 语义。

2026-06-23 Section 7 生命周期状态污染测试更新：

- 新增并运行 `scripts/deploy/check_robojudo_unilab_section7.py`。
- 结果：`summary: 4 fail(s), 0 warning(s)`。
- `RlPipeline.__init__` 中 `self_check()` 后会调用 `reset()`，因此构造阶段 self-check 造成的 policy state 污染会被清掉。
- `step(dry_run=True)` 会给 commands 加 `[UNILAB_FREEZE_PHASE]`，因此 dry-run 不推进 `gait_phase`；测试中 dry-run 后 `gait_phase` 仍为 `[0, 0]`。
- 失败点：`step(dry_run=True)` 仍然调用 `PolicyWrapper.get_pd_target(obs)`，而 `get_pd_target()` 内部调用 `policy.get_action(obs)`；`UniLabPolicy.get_action()` 会更新 `self.last_action`。
- 单次 dry-run 后 `last_action` 最大绝对值约 `0.961`。
- `prepare()` 在 `t == 900` 时调用 `self.reset()`，但循环之后还会继续执行 99 次 dry-run；`prepare()` 结束前没有 final policy reset。
- 因此 `prepare()` 结束时 `last_action` 最大绝对值约 `0.931`。
- 正式控制第一帧 obs 的 `last_action[64:93]` 会从全 0 变为 dry-run 残留 action，`clean_vs_polluted_last_action_max_abs = 0.931215`。
- 结论：Section 7 当前 FAIL。`prepare()` 后的正式第一帧不再是 UniLab reset 分布，因为 obs 中 action-history 段被 dry-run 推理污染。下一步应只修 RoboJudo 生命周期：让 dry-run 推理不更新 `last_action`，或在 `prepare()` 返回前显式 `policy.reset()`，同时确认不破坏 gait phase freeze 与 self-check 逻辑。

2026-06-23 Section 7 生命周期状态污染修复更新：

- 只修改 RoboJudo_Real，不修改 UniLab。
- 在 RoboJudo_Real `robojudo/policy/unilab_policy.py` 为 `UniLabPolicy` 增加 `snapshot_state()` / `restore_state()`，保存并恢复 `last_action`、`gait_phase`、`_last_obs`。
- 在 RoboJudo_Real `robojudo/pipeline/rl_pipeline.py` 的 `step(dry_run=True)` 中，如果 policy 支持 snapshot/restore，则 dry-run 推理前保存状态、推理和 callback 后恢复状态。
- 该修复不跳过 dry-run 推理，仍会计算 `pd_target` 供 self-check/prepare 使用；只是不让 dry-run 写入后续正式 obs 会使用的 policy internal state。
- 普通控制帧 `dry_run=False` 不走 snapshot/restore，正常更新 `last_action` 和 gait phase。
- 重跑 `scripts/deploy/check_robojudo_unilab_section7.py`：`summary: 0 fail(s), 0 warning(s)`。
- 修复后单次 dry-run 后 `last_action max_abs = 0`。
- 修复后 `prepare()` 结束时 `last_action max_abs = 0`，`gait_phase = [0, pi]`。
- 修复后正式控制第一帧 obs 的 `last_action[64:93]` 保持全 0，`clean_vs_polluted_last_action_max_abs = 0`。
- 回归 `scripts/deploy/check_robojudo_unilab_section3.py`：`summary: 0 fail(s), 0 warning(s)`。
- 回归 `scripts/deploy/check_robojudo_unilab_section6.py`：`summary: 0 fail(s), 2 warning(s)`，剩余 warning 仍是 UniLab 原生也会输出的 target joint-limit warning。
- 回归 `scripts/deploy/check_unilab_native_section6_control.py`：`summary: 0 fail(s), 0 warning(s)`。
- 结论：Section 7 的 dry-run state pollution 已修复。RoboJudo dry-run 现在不会污染 UniLab policy 的 `last_action` 观测段。

2026-06-23 Section 5 gait phase 初值修复更新：

- 只修改 RoboJudo_Real，不修改 UniLab。
- 在 RoboJudo_Real `robojudo/config/g1/policy/g1_unilab_policy_cfg.py` 为 `G1UniLabPolicyCfg` 增加 `initial_gait_phase = [0.0, pi]`。
- 在 RoboJudo_Real `robojudo/policy/unilab_policy.py` 中，`UniLabPolicy.__init__()` 读取并校验 `initial_gait_phase`，`reset()` 使用该配置恢复 gait phase，不再硬编码 `[0, 0]`。
- 同步更新 `scripts/deploy/check_robojudo_unilab_section3.py`、`scripts/deploy/check_robojudo_unilab_section5.py`、`scripts/deploy/check_robojudo_unilab_section7.py` 的测试预期。
- 重跑 Section 3：`summary: 0 fail(s), 0 warning(s)`；第一帧 `gait_phase[96:98] = [0, pi]`。
- 重跑 Section 5：`summary: 0 fail(s), 1 warning(s), 15 pass(es)`；reset gait phase 已符合 UniLab 训练 `offset_phase`，剩余 warning 是默认未启用的 KeyboardCtrl pressed value `1.5` 会超出训练 command range。
- 重跑 Section 7：`summary: 0 fail(s), 0 warning(s)`；dry-run 后 `last_action` 仍为 0，`gait_phase` 保持 `[0, pi]`。
- 结论：当前 Section 5 主失败项已修复。RoboJudo UniLab policy 的正式第一帧现在从 UniLab `offset_phase` 语义下的 `[0, pi]` 开始。

2026-06-23 Section 8 runtime torque trace 更新：

- 新增并运行 `scripts/deploy/check_robojudo_unilab_section8_runtime_torque.py`。
- 该脚本不启动 RoboJudo viewer，不 import RoboJudo runtime；它直接重构 `obs -> ONNX action -> pd_target -> torque/ctrl -> MuJoCo step`，用于确认“零力矩倒地”是否真的发生在 action-to-torque 数值链路。
- 结果显示 RoboJudo motor XML 路径并非零力矩：在 `--steps 20 --command 0 0 0` 下，第一帧 torque L2 约 `157.39`，后续最大 torque L2 约 `348.39`，`data.ctrl` L2 同步非零。
- 同一脚本新增 UniLab 原生 `scene_flat.xml` position-actuator 对照；在相同 ONNX、相同 obs 构造、相同初始 qpos 下，UniLab scene 的 actuator force 也非零。
- RoboJudo motor XML 与 UniLab scene 在手写闭环里都会掉高；`--steps 60 --command 0 0 0` 下 RoboJudo 最小 base z 约 `0.062`，UniLab scene 最小 base z 约 `0.249`。
- `--command 0.5 0 0` 以及多个初始 gait phase 扫描没有消除掉高，只改变掉高速度。
- 因此当前证据不支持“RoboJudo 已经把 policy 输出变成零力矩”；更像是手写 rollout 仍缺少 UniLab 官方 env reset/play 链路中的某个语义，或当前 ONNX/checkpoint 在该固定初始化条件下本身不可稳定 playback。
- 下一步不应继续盲修 RoboJudo 控制参数；应新增 Section 9，直接通过 UniLab 官方 `env.reset()` / `env.step()` / `play_offpolicy()` 链路加载同一 checkpoint/ONNX，比较官方 env state 中的 `commands`、`gait_phase`、`current_actions`、base height 与手写 Section 8 trace。

2026-06-23 Section 9 UniLab 官方 env rollout 更新：

- 新增并运行 `scripts/deploy/check_unilab_native_section9_official_env_rollout.py`。
- 脚本通过 UniLab registry 构造 `G1WalkFlat` MuJoCo env，使用 run 的 `run_config.json` 生成 `env_cfg_override`，并加载同一 `model_5000.pt` 与 `policy.onnx`。
- 同时测试两种官方生命周期：
  - `collector_init_state`：匹配 off-policy collector 的 `env.init_state()` 路径。
  - `play_reset_call`：匹配 `scripts/train_offpolicy.py -> play_offpolicy()` 中先调用 `env.reset(...)` 再进入 policy step 的路径。
- 运行命令：`python3 -m uv run scripts/deploy/check_unilab_native_section9_official_env_rollout.py --num-envs 16 --steps 800 --runtime both`。
- 结果：`summary: 0 fail(s), 1 warning(s), 5 pass(es)`。
- `checkpoint_actor / collector_init_state`：800 step 无 done，min height `0.67549`，reward_sum_mean `281.57`。
- `checkpoint_actor / play_reset_call`：800 step 无 done，min height `0.63865`，reward_sum_mean `286.23`。
- `onnx_actor / collector_init_state`：800 step 无 done，min height `0.67549`，reward_sum_mean `289.13`。
- `onnx_actor / play_reset_call`：800 step 无 done，min height `0.63866`，reward_sum_mean `290.22`。
- checkpoint actor 与 ONNX actor 的 rollout 高度几乎一致，说明 ONNX 导出不是当前摔倒主因。
- 官方 env 首帧 action 明显小于 Section 8 手写 obs：官方 `first_action max_abs` 约 `0.69` 或 `0.76`，而 Section 8 固定 stand 手写 obs 下 action L2 约 `2.6`、target 大幅越界。
- Section 9 warning：`play_reset_call` 中 `env.reset(...)` 不会填充 `env.state`，第一次 `env.step()` 会触发 `init_state()`。该 warning 是 UniLab play 生命周期细节，不是导致官方 rollout 摔倒的问题，因为两种生命周期都稳定。
- 关键结论：策略/checkpoint/ONNX 在 UniLab 官方 env 链路中是稳定的；RoboJudo 当前仍缺少 UniLab 官方 observation/state 语义。当前最高嫌疑从控制力矩转移到 observation 的 sensor source：UniLab G1 使用 `torso_gyro` 与 `torso_upvector`，RoboJudo 当前 `UniLabPolicy` 使用的是 base/pelvis `base_ang_vel` 与 `base_quat` 推导 gravity。

2026-06-23 Section 3 sensor source 修复更新：

- 新增并运行 `scripts/deploy/check_robojudo_unilab_section3_sensor_source.py`。
- UniLab 官方 `G1WalkFlat` actor obs 的前 6 维与 XML sensor 完全对齐：`gyro == torso_gyro * 0.25`，`gravity == -torso_upvector`，两者 `max_abs` 均为 0。
- UniLab 的 `torso_gyro` 与 `torso_upvector` 都绑定在 XML site `imu_in_torso`，不是 pelvis/base body source。
- 30 个官方 env step 后，torso 与 pelvis source 已明显分叉：`torso_vs_pelvis_gyro_max_abs = 1.44818580`，`torso_vs_pelvis_upvector_max_abs = 0.12273764`。
- 将同一帧 obs 的 torso source 替换为 pelvis source 会让 policy action 改变，`action_changed_by_pelvis_source_max_abs = 0.7139`。该项保留为 warning，用于说明 sensor source 差异足够大，不是新失败。
- 只修改 RoboJudo_Real，不修改 UniLab。
- 在 RoboJudo_Real `G1UniLabPolicyCfg` 中新增 `use_torso_obs_source = True`。
- 在 RoboJudo_Real `UniLabPolicy.get_observation()` 中优先使用 `env_data.torso_ang_vel` 与 `env_data.torso_quat` 构造 `ang_vel * 0.25` 和 gravity；缺失时才 fallback 到原来的 base source。
- 在 RoboJudo_Real `Environment.fk()` 中向 `MujocoKinematics.forward()` 传入 `joint_vel=self.dof_vel`，否则 torso angular velocity 会缺少 joint velocity 对 torso link motion 的贡献。
- 重跑 `scripts/deploy/check_robojudo_unilab_section3_sensor_source.py --num-envs 16 --probe-steps 30`：`summary: 0 fail(s), 1 warning(s), 10 pass(es)`。
- 重跑 `scripts/deploy/check_robojudo_unilab_section3.py`：`summary: 0 fail(s), 0 warning(s)`。
- 回归 `scripts/deploy/check_robojudo_unilab_section6.py`：`summary: 0 fail(s), 2 warning(s)`，剩余 warning 仍是 UniLab 原生 target joint-limit warning。
- 回归 `scripts/deploy/check_robojudo_unilab_section7.py`：`summary: 0 fail(s), 0 warning(s)`。
- 结论：Section 3 现在不仅 obs 顺序正确，而且 observation sensor source 也已对齐 UniLab 官方 `torso_gyro / torso_upvector` 语义。

2026-06-23 Section 10/11 runtime 控制链路与 gravity 符号修复更新：

- 新增并运行 `scripts/deploy/check_robojudo_unilab_section10_runtime_pipeline_ctrl.py`。
- 该脚本用 dummy viewer 跑 RoboJudo_Real 的真实 `RlPipeline(g1_unilab).step()`，直接记录 `obs -> action -> pd_target -> data.ctrl -> mj_step`。
- 修复前，真实 pipeline 并不是零力矩：200 step 内 `data.ctrl` 每步非零，`ctrl_l2_min = 21.2219`，但 base height 掉到 `0.125058`。
- 初始 gait phase 扫描 8 个 phase 后仍全部掉高，`stable=0/8`，说明固定 `[0, pi]` 不是主因。
- 新增并运行 `scripts/deploy/check_unilab_native_section11_zero_command_rollout.py`。
- UniLab 官方 env 在强制 `commands=[0,0,0]` 时仍稳定 800 step，`min_height = 0.677627`；因此 joystick idle 零命令不是主因。
- Section 10 在 UniLab 官方 `scene_flat.xml` 下直接比较 runtime obs 与 MuJoCo sensor，发现 `gyro` 完全匹配，但 gravity 与 `-torso_upvector` 相差 `2.0`。
- 根因：RoboJudo `get_gravity_orientation()` 已经返回 body frame 下的 gravity，即 UniLab actor obs 需要的 `-torso_upvector`；此前 `UniLabPolicy.get_observation()` 又额外取负号，导致站立首帧 gravity 从 `[0, 0, -1]` 反成 `[0, 0, 1]`。
- 只修改 RoboJudo_Real，不修改 UniLab：移除 `UniLabPolicy.get_observation()` 中 torso/base gravity 的额外负号。
- 同步将 `g1_unilab` 的 MuJoCo sim 配置对齐 UniLab G1 默认物理步长：`sim_dt = 0.02 / 3.0`，`sim_decimation = 3`；普通 `g1` 与真机 `g1_real_unilab` 不受影响。
- 修复后重跑默认 RoboJudo `g1_unilab` runtime pipeline 200 step：`summary: 0 fail(s), 0 warning(s)`，`min_base_z = 0.714963`，`last_base_z = 0.742533`，`data.ctrl` 全程非零。
- 修复后用 UniLab 官方 XML + position-control 对照 200 step：`summary: 0 fail(s), 0 warning(s)`，`min_base_z = 0.714518`。
- 回归 `scripts/deploy/check_robojudo_unilab_section3.py`：`summary: 0 fail(s), 0 warning(s)`，首帧 gravity 为 `[0, 0, -1]`。
- 回归 `scripts/deploy/check_robojudo_unilab_section3_sensor_source.py --num-envs 16 --probe-steps 30`：`summary: 0 fail(s), 1 warning(s), 10 pass(es)`，唯一 warning 仍是 pelvis 替换会显著改变 action 的诊断信号。
- 回归 `scripts/deploy/check_robojudo_unilab_section6.py`：`summary: 0 fail(s), 2 warning(s)`，剩余 warning 仍是 UniLab 原生 target joint-limit warning。
- 回归 `scripts/deploy/check_robojudo_unilab_section7.py`：`summary: 0 fail(s), 0 warning(s)`。
- 结论：用户在 viewer 中观察到的“零力矩倒下”实际不是 `data.ctrl=0`，而是 gravity obs 符号反转导致 policy 在错误姿态观测下输出无法稳定的控制。修复 gravity 符号后，RoboJudo 默认 `g1_unilab` pipeline 可在无 viewer 测试中稳定 200 step。

2026-06-23 Joystick axis 兼容修复更新：

- 症状：G1 已能在 MuJoCo 中正常站立，但使用 RoboJudo `JoystickCtrl` 时线程报错 `pygame.error: Invalid joystick axis`。
- 原因：RoboJudo 默认 joystick axis map 假设 Xbox 风格 6 轴设备：`LeftX=0, LeftY=1, LT=2, RightX=3, RightY=4, RT=5`。部分手柄/远程设备只有 4 或 5 个 SDL axes，读取不存在的 axis 5 会触发 pygame error。
- 只修改 RoboJudo_Real，不修改 UniLab。
- 在 RoboJudo_Real `robojudo/controller/utils/joystick.py` 中新增 axis map 自适应：当设备为 4/5 轴时，使用常见布局 `LeftX=0, LeftY=1, RightX=2, RightY=3`，并丢弃不可用的 `LT/RT` axes。
- 对 6 轴设备保留原 Xbox 风格映射，不影响现有正常手柄。
- UniLab policy 仍读取 `JoystickCtrl.axes` 中的 `LeftX/LeftY/RightX`，并通过 `G1UniLabPolicyCfg.command_maps` 映射到训练 command range。
- 新增并运行 `scripts/deploy/check_robojudo_unilab_joystick_axes.py`：`summary: 0 fail(s), 4 pass(es)`。
- 语法检查通过：`python -m py_compile robojudo/controller/utils/joystick.py`。
- 回归 `scripts/deploy/check_robojudo_unilab_section5.py`：`summary: 0 fail(s), 1 warning(s), 15 pass(es)`，剩余 warning 仍是 keyboard path 可超出训练 command range，与 joystick axis 修复无关。

相关仓库：

- UniLab：`/Users/chengyuxuan/ArtiIntComVis/UniLab`
- RoboJudo_Real：`/Users/chengyuxuan/ArtiIntComVis/RoboJudo_Real`

---

## Section 1. 模型 artifact 与加载契约

目标：确认 RoboJudo_Real 加载的是正确的 UniLab policy 文件，且模型输入/输出契约与部署代码一致。

代码锚点：

- RoboJudo_Real：`robojudo/config/g1/policy/g1_unilab_policy_cfg.py`
- RoboJudo_Real：`robojudo/policy/unilab_policy.py`
- UniLab run：`assets/models/g1/unilab/g1_walk_flat/2026-06-12_15-46-01_mujoco/run_config.json`

检查动作：

- [ ] 确认 `policy_file` 指向 `assets/models/g1/unilab/g1_walk_flat/policy.onnx`。
- [ ] 确认该 ONNX 文件来自要测试的 UniLab checkpoint，而不是旧导出。
- [ ] 读取 ONNX input name、input shape、output name、output shape。
- [ ] 确认 input dim = 98。
- [ ] 确认 output dim = 29。
- [ ] 确认 ONNX 是 raw obs 输入，还是 normalized obs 输入。

通过标准：

- [ ] RoboJudo_Real 加载的文件路径明确。
- [ ] ONNX 只有一个 policy obs 输入，除非 UniLab 导出明确使用 privileged input。
- [ ] ONNX 输出是 29 维 action delta。
- [ ] 对“ONNX 输入空间 raw / normalized”的判断有代码证据，不靠猜。

记录：

```text
policy_path:
onnx_input:
onnx_output:
raw_or_normalized_input:
evidence:
result:
```

---

## Section 2. Obs Normalization

目标：确认 UniLab 训练时的 observation normalization 是否被完整复现在 RoboJudo_Real 推理路径中。

代码锚点：

- UniLab：`scripts/train_offpolicy.py`
- UniLab：`src/unilab/algos/torch/fast_sac/learner.py`
- RoboJudo_Real：`robojudo/policy/unilab_policy.py`
- UniLab run：`run_config.json -> algo.obs_normalization`

检查动作：

- [ ] 确认 `run_config.json` 中 `algo.obs_normalization` 的值。
- [ ] 确认 checkpoint 中是否含 `obs_normalizer` state。
- [ ] 确认 ONNX 导出是否包含 normalizer。
- [ ] 如果 ONNX 不包含 normalizer，则在 RoboJudo_Real 中加载同一份 mean/std，并在 `session.run()` 前归一化 obs。
- [ ] 打印 raw obs 与 normalized obs 的 min/max/mean/std。
- [ ] 用同一帧 obs 比较 UniLab PyTorch actor 输出与 RoboJudo_Real ONNX 输出。

通过标准：

- [ ] 如果训练 `obs_normalization=false`，RoboJudo_Real 可直接喂 raw obs。
- [ ] 如果训练 `obs_normalization=true`，RoboJudo_Real 必须喂 normalized obs，或使用包含 normalizer 的 ONNX。
- [ ] 同一输入下，UniLab PyTorch actor 与 RoboJudo_Real ONNX action 差异应在可接受范围内。

记录：

```text
obs_normalization:
normalizer_in_onnx:
normalizer_source:
raw_obs_stats:
normalized_obs_stats:
pt_vs_onnx_diff:
result:
```

---

## Section 3. Observation 构造顺序与数值

目标：确认 RoboJudo_Real 构造的 98 维 obs 与 UniLab G1WalkFlat 训练时 actor obs 完全一致。

代码锚点：

- UniLab：`src/unilab/envs/locomotion/g1/joystick.py`
- RoboJudo_Real：`robojudo/policy/unilab_policy.py`
- RoboJudo_Real：`robojudo/pipeline/rl_pipeline.py -> PolicyWrapper.get_observation`

目标 obs 顺序：

```text
base_ang_vel * 0.25
-gravity
dof_pos - default_angles
dof_vel * 0.05
previous/current policy action delta
command: vx, vy, yaw_rate
raw left/right gait_phase
```

检查动作：

- [ ] 打印 obs 总维度，必须为 98。
- [ ] 分段打印每一段 shape 和数值范围。
- [ ] 静止站立第一帧，`dof_pos - default_angles` 应接近 0。
- [ ] 静止站立第一帧，`dof_vel * 0.05` 应接近 0。
- [ ] reset 后 action history 段应为 0。
- [ ] zero joystick 时 command 段应为 `[0, 0, 0]`。
- [ ] gait phase 段应是 raw phase，不是 sin/cos。

通过标准：

- [ ] 98 维 obs 分段边界正确。
- [ ] 每段数值量纲与 UniLab play 中同一场景匹配。
- [ ] 没有把 PD target、绝对关节角、sin/cos phase、错误 command scale 混入 obs。

记录：

```text
obs_dim:
gyro_stats:
gravity:
dof_pos_rel_stats:
dof_vel_stats:
last_action_stats:
command:
gait_phase:
result:
```

---

## Section 4. 初始姿态与 default pose

目标：确认仿真 reset 后机器人处在 UniLab policy 训练分布附近，而不是 RoboJudo_Real XML 默认 zero pose。

代码锚点：

- UniLab：`src/unilab/assets/robots/g1/scene_flat.xml -> keyframe stand`
- RoboJudo_Real：`robojudo/config/g1/policy/g1_unilab_policy_cfg.py -> G1UniLabDoF.default_pos`
- RoboJudo_Real：`robojudo/environment/mujoco_env.py -> reset/reborn`
- RoboJudo_Real：`robojudo/pipeline/rl_pipeline.py -> reset/prepare`

检查动作：

- [ ] 打印 RoboJudo_Real 仿真 reset 后的 `env.dof_pos`。
- [ ] 打印 `G1UniLabDoF.default_pos`。
- [ ] 计算 `env.dof_pos - G1UniLabDoF.default_pos`。
- [ ] 检查 MuJoCo 初始 qpos 是否使用 UniLab `stand` keyframe。
- [ ] 如果没有，先让仿真 reset/reborn 到 UniLab stand pose，再开 policy。
- [ ] zero action 时，PD target 应等于 UniLab stand pose。

通过标准：

- [ ] policy 开始推理前，机器人关节位置接近 UniLab stand pose。
- [ ] 第一帧 obs 中 `dof_pos - default_angles` 不能大幅偏离 0。
- [ ] zero action 对应的 PD target 是 UniLab default pose，而不是 RoboJudo default pose。

记录：

```text
env_reset_dof_pos:
unilab_default_pos:
dof_pos_minus_default_stats:
reborn_to_stand:
zero_action_pd_target:
result:
```

---

## Section 5. Command 与 gait phase

目标：确认遥控输入和 gait phase 与 UniLab 训练时的控制语义一致。

代码锚点：

- UniLab：`src/unilab/envs/locomotion/g1/joystick.py`
- RoboJudo_Real：`robojudo/policy/unilab_policy.py -> _get_commands/post_step_callback`
- RoboJudo_Real：`robojudo/pipeline/rl_pipeline.py -> step(dry_run)`

检查动作：

- [x] zero joystick 时 command = `[0, 0, 0]`。
- [x] 前进 joystick 映射到 UniLab `vx` 正方向。
- [x] 左右平移 joystick 映射到 UniLab `vy` 正负方向。
- [x] yaw joystick 映射到 UniLab `yaw_rate` 正负方向。
- [x] command range 为 `vx [-0.6, 1.0]`，`vy [-0.4, 0.4]`，`yaw_rate [-0.8, 0.8]`。
- [x] gait frequency = 1.5。
- [x] 每步 phase delta = `2*pi*1.5*0.02`。
- [x] dry-run 阶段不推进 gait phase。
- [x] 增加测试脚本：`scripts/deploy/check_robojudo_unilab_section5.py`。

通过标准：

- [x] command 方向、尺度、零点与 UniLab 一致。
- [x] gait phase 是 raw left/right phase。
- [x] dry-run 不改变 phase。
- [x] reset 初始 gait phase 与 UniLab 训练 `offset_phase` 一致。

记录：

```text
command_map_ranges: PASS, [[-0.6, 1.0], [-0.4, 0.4], [-0.8, 0.8]]
zero_command: PASS, [0, 0, 0]
forward_command: PASS, [1, 0, 0] / [-0.6, 0, 0]
lateral_command: PASS, LeftX +1 -> [0, -0.4, 0], LeftX -1 -> [0, 0.4, 0]
yaw_command: PASS, RightX +1 -> [0, 0, -0.8], RightX -1 -> [0, 0, 0.8]
gait_frequency: PASS, RoboJudo 1.5 == training snapshot 1.5
phase_delta: PASS, 0.188495559
dry_run_phase_changed: PASS, no change under [UNILAB_FREEZE_PHASE]
normal_step_phase_after: PASS, [0.188496, 3.33009]
reset_gait_phase: PASS, training offset_phase expects left/right pi offset, RoboJudo reset is [0, pi]
keyboard_command_range: WARN, KeyboardCtrl pressed value 1.5 can exceed training range
result: PASS, 0 fail(s), 1 warning(s), 15 pass(es)
```

---

## Section 6. Action 到 PD target

目标：确认 ONNX 输出被解释为 UniLab action delta，并通过 RoboJudo_Real wrapper 正确转成 PD target。

代码锚点：

- UniLab：`src/unilab/envs/locomotion/g1/joystick.py -> apply_action`
- RoboJudo_Real：`robojudo/policy/unilab_policy.py -> get_action`
- RoboJudo_Real：`robojudo/pipeline/rl_pipeline.py -> PolicyWrapper.get_pd_target`
- RoboJudo_Real：`robojudo/tools/dof.py -> DoFAdapter`

检查动作：

- [ ] 打印 ONNX raw action。
- [ ] 打印 action min/max/mean/std。
- [ ] 确认 action_scale = 1.0。
- [ ] 确认 action 没有被重复乘 scale。
- [ ] 确认 `pd_target = action + UniLab default_pos`。
- [ ] 确认 DoFAdapter 只做顺序映射，不丢失 29 个 action。
- [ ] 打印 `pd_target - default_pos`，应等于 action delta。
- [ ] 检查 PD target 是否超出 joint position limits。

通过标准：

- [ ] zero action -> PD target = UniLab default pose。
- [ ] policy action 范围合理，不出现异常大值。
- [ ] 29 个关节均正确映射到 RoboJudo env joint order。

记录：

```text
raw_action_stats:
action_scale:
pd_target_stats:
pd_target_minus_default_stats:
dof_adapter_mapping_ok:
position_limit_violation:
result:
```

---

## Section 7. Pipeline 生命周期与状态污染

目标：确认 RoboJudo_Real 的 self_check、dry-run、prepare、reset 不会污染 UniLab policy 的内部状态。

代码锚点：

- RoboJudo_Real：`robojudo/pipeline/rl_pipeline.py -> self_check/reset/prepare/step`
- RoboJudo_Real：`robojudo/policy/unilab_policy.py -> reset/post_step_callback/get_action`

检查动作：

- [ ] 检查 `self_check()` 期间是否调用 policy 推理。
- [ ] 检查 dry-run 是否更新 `last_action`。
- [ ] 检查 dry-run 是否推进 gait phase。
- [ ] 检查 `self_check()` 后是否调用 `policy.reset()`。
- [ ] 检查 `prepare()` 中 dry-run 是否污染 `last_action`。
- [ ] 真机 `prepare()` 到 900 step reset 后，确认 action history 和 gait phase 是否回到 0。
- [ ] 仿真测试前，确认 policy 内部状态为 reset 后状态。

通过标准：

- [ ] policy 真正闭环控制第一帧时，`last_action=0`。
- [ ] policy 真正闭环控制第一帧时，gait phase 符合预期初始化。
- [ ] dry-run 不应改变会进入 obs 的 policy state，或必须在正式控制前 reset。

记录：

```text
self_check_updates_action:
self_check_resets_policy:
prepare_updates_action:
prepare_resets_policy:
first_real_step_last_action:
first_real_step_gait_phase:
result:
```

---

## 推荐排查顺序

1. Section 2：Obs Normalization。
2. Section 4：初始姿态与 default pose。
3. Section 3：Observation 构造顺序与数值。
4. Section 6：Action 到 PD target。
5. Section 5：Command 与 gait phase。
6. Section 7：Pipeline 生命周期与状态污染。
7. Section 1：模型 artifact 与加载契约在每次换模型时重新检查。

每个 section 只有在“通过标准”都有证据后，才进入下一段。若某段失败，不继续调后面的控制参数。
