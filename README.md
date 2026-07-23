<h1 align="center"> UniLab </h1>

<h3 align="center">
A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms
</h3>

<p align="center">Languages: English | <a href="README_zh.md">简体中文</a></p>

<p align="center">
  <a href="https://unilabsim.github.io"><img src="https://img.shields.io/badge/project-page-brightgreen" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2605.30313"><img src="https://img.shields.io/badge/arxiv-2605.30313-red" alt="arXiv"></a>
  <a href="https://unilabsim.github.io/paper/"><img src="https://img.shields.io/badge/paper-UniLab-orange" alt="Paper"></a>
  <a href="https://unilabsim.github.io/UniLab-doc/"><img src="https://img.shields.io/badge/docs-UniLab--doc-blue" alt="Documentation"></a>
  <a href="https://unilabsim.github.io/paper/paper2gal.html"><img src="https://img.shields.io/badge/Galgame-play-ff69b4" alt="Galgame"></a>
  <a href="https://discord.gg/EPCuguRmGX"><img src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License"></a>
</p>

<p align="center">
  <img src="docs/sphinx/source/_static/assets/teaser.jpg" alt="UniLab Teaser" width="95%">
</p>

<p align="center"><em>Train robot RL without a GPU simulation backend. Teaser rendered with MotrixSim.</em></p>

Start with the `Quick Demo` below to run the primary training command. The recommended setup uses `uv`; Conda and pip users should still follow the `uv` workflow for now. Platform-specific notes and current boundaries are in the [installation guide](https://unilabsim.github.io/UniLab-doc/en/1-getting_started/2-installation.html).

## ✨ Highlights

```
┌───────────────────┐                            ┌─────────────────────────┐
│  CPU Physics Sim  │   Unified Shared Memory    │   GPU Policy Training   │
│   MuJoCo/Motrix   │ ─────────────────────────▶ │     PPO / SAC / TD3     │
│ Multithread Step  │    SharedReplayBuffer      │ CUDA / MPS / ROCm / XPU │
└───────────────────┘                            └─────────────────────────┘
```

- **Heterogeneous RL runtime:** CPU-parallel simulation streams transitions through shared memory while policy learning runs on GPU accelerators.
- **Two physics backends:** MuJoCoUni and MotrixSim are integrated through backend-specific adapters and task owner configs.
- **Unified training CLI:** `uv run train` and `uv run eval` cover PPO, MLX PPO, APPO, SAC, TD3, and FlashSAC; additional HORA and HIM-PPO paths are documented as script-level workflows.
- **Config-owned tasks:** Hydra owner YAML files select task, reward, backend, and algorithm settings together; backend switching is expressed as `task=<task>/<backend>`.
- **Cross-platform setup paths:** The repository tracks Linux CUDA, Linux ROCm, Linux XPU, and Apple Silicon / macOS setup flows.

## 🚀 Installation

```bash
# 0. If uv is not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Clone the repository
git clone https://github.com/unilabsim/UniLab.git
cd UniLab

# 2. Install dependencies
# Pick the setup command for your platform.

# Linux CUDA or macOS
make setup-motrix
# Without shell completion setup: uv sync --extra motrix
# If `make` is not installed: uv sync --extra motrix && uv run --no-sync unilab-complete install

# Linux AMD / ROCm
# make sync-rocm

# Linux Intel Arc / iGPU
# make sync-xpu

# 3. Pre-trained checkpoint playback (downloads from Hugging Face on first run)
uv run demo dance
```

## 🏃 Training

```bash
# vanilla command
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

## 🏃 Sim2Sim

```bash
# vanilla command
uv run scripts/play_interactive.py \
  --algo sac \
  --task g1_walk_flat \
  --sim mujoco interactive.action_mode=policy interactive.keyboard=true
```

```bash
# quick command
./start.sh --algo distill --task g1_walk_flat \
  --checkpoint-path /path/to/checkpoint/dagger_iteration_8.pt
```

```bash
# with diagnoisis
UNILAB_G1_ACTION_TRACE=1 UNILAB_G1_ACTION_TRACE_INTERVAL=20 \
./start.sh g1_stand_still 2026-07-09_22-02-36_mujoco
```

## 🏃 Sim2Real

```bash
# See RoboJuDo
```

## 📚 Documentation

Use the published [UniLab documentation](https://unilabsim.github.io/UniLab-doc/); start at the [English documentation index](https://unilabsim.github.io/UniLab-doc/en/0-index.html). High-signal entrypoints:

- [Getting Started](https://unilabsim.github.io/UniLab-doc/en/1-getting_started/0-index.html): installation, Docker runtime, dependency setup, and first-run commands
- [Training Guide](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/1-training/0-index.html): training, playback, resume flow, Hydra overrides, and W&B
- [Simulation Backends](https://unilabsim.github.io/UniLab-doc/en/2-user_guide/3-backends/0-index.html): generated MuJoCo / Motrix support matrix
- [Development Standard](https://unilabsim.github.io/UniLab-doc/en/4-developer_guide/0-index.html): contracts, layering, and validation boundaries
- [ADR Index](https://unilabsim.github.io/UniLab-doc/adr/ADR-0000-index.html): accepted architecture decisions

## 🧾 Citation

### UniLab

```bibtex
@article{jia2026unilab,
  title         = {UniLab: A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms},
  author        = {Yufei Jia and Zhanxiang Cao and Mingrui Yu and Heng Zhang and Shenyu Chen and Dixuan Jiang and Meng Li and Xiaofan Li and Yiyang Liu and Junzhe Wu and Zheng Li and XiLin Fang and Tingyu Cui and Shengcheng Fu and Haoyang Li and Anqi Wang and Zifan Wang and Dongjie Zhu and Chenyu Cao and Zhenbiao Huang and Ziang Zheng and Jie Lu and Xin Ma and Zhengyang Wei and Xiang Zhao and Tianyue Zhan and Ye He and Yuxiang Chen and Yizhou Jiang and Yue Li and Haizhou Ge and Yuhang Dong and Fan Jia and Ziheng Zhang and Meng Zhang and Xiwa Deng and Zhixing Chen and Hanyang Shao and Chenxin Dong and Yixuan Li and Yizhi Chen and Bokui Chen and Kaifeng Zhang and Hanqing Cui and Yusen Qin and Ruqi Huang and Lei Han and Tiancai Wang and Xiang Li and Yue Gao and Guyue Zhou},
  journal       = {arXiv preprint arXiv:2605.30313},
  year          = {2026},
  url           = {https://arxiv.org/abs/2605.30313}
}
```

### Physics Backends

```bibtex
@article{jia2026mujocouni,
  title  = {MuJoCoUni: Persistent Batched Runtime Primitives for MuJoCo},
  author = {Jia, Yufei and Wu, Junzhe},
  journal = {arXiv preprint arXiv:2605.24922},
  year   = {2026}
}

@software{motrixsim2026,
  title  = {MotrixSim: A Physics Simulation Engine for Robotics and Embodied AI},
  author = {{Motphys Team}},
  year   = {2026},
  url    = {https://motrixsim.readthedocs.io/},
  note   = {Python binary package}
}
```
