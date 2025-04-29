#!/bin/bash

echo "================================="
echo "=== Monk Torch Music Training ==="
echo "================================="

# 安全起见，先激活环境
source ~/.bashrc
conda activate your_env_name  # <-- 记得改成你的env名字，比如 pytorch-env

# 进入你的项目目录
cd /workspace/your_project_directory  # <-- 改成你的monk_torch.py所在目录

# 询问用户选择
echo "What would you like to do?"
echo "1. Train the model"
echo "2. Generate music"
read -p "Enter your choice (1-2): " choice

if [ "$choice" == "1" ]; then
    echo "Starting training process..."
    python monk_torch.py --mode train
elif [ "$choice" == "2" ]; then
    echo "Starting music generation..."
    python monk_torch.py --mode generate
else
    echo "Invalid choice. Please enter 1 or 2."
fi

