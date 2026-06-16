#!/bin/bash

# 获取执行脚本时传入的第一个参数。如果没有传入，则默认使用 "sync update"
COMMIT_MSG=${1:-"sync update"}

echo "🚀 开始同步代码到远程仓库..."

# 1. 暂存所有更改
echo "📦 正在执行: git add ."
git add .

# 2. 提交更改
echo "📝 正在执行: git commit -m \"$COMMIT_MSG\""
git commit -m "$COMMIT_MSG"

# 3. 推送到 main 分支
echo "☁️ 正在执行: git push origin main"
git push origin main

echo "✅ 同步完成！"
