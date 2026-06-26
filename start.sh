#!/bin/bash

# 获取当前操作系统名称
OS="$(uname -s)"

# 将那些两边都一样的长参数提取出来，方便以后修改
ARGS="scripts/play_interactive.py --algo sac --task g1_walk_flat --sim mujoco interactive.action_mode=policy interactive.keyboard=true"

# 根据操作系统执行不同的命令
if [ "$OS" = "Darwin" ]; then
    # Mac OS 系统 (Darwin) 必须使用 mjpython
    echo "🍏 检测到 macOS，正在使用 mjpython 启动仿真..."
    uv run mjpython $ARGS
elif [ "$OS" = "Linux" ]; then
    # 4090 服务器 (Linux) 直接使用默认的 python 即可
    echo "🐧 检测到 Linux，正在使用常规 python 启动仿真..."
    uv run python $ARGS
else
    echo "⚠️ 未知的操作系统: $OS"
    exit 1
fi
