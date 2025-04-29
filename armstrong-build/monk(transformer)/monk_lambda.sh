# 改造后的 monk_lambda.sh （适配Docker版，不再需要conda）

#!/bin/bash
set -e

# Color formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Welcome
echo -e "${PURPLE}================ Monk Lambda Setup (Docker Version) ================${NC}"

# 安装Python依赖
echo -e "${YELLOW}Installing required Python packages...${NC}"
pip install --upgrade pip
pip install music21 matplotlib tqdm

# 检查TensorFlow GPU是否可用
echo -e "${YELLOW}Verifying TensorFlow GPU availability...${NC}"
python -c "import tensorflow as tf; print('GPUs detected by TensorFlow:', tf.config.list_physical_devices('GPU'))"

# 进入项目目录（可选，如果你的路径不是/workspace就改）
cd /workspace

# 启动主程序
echo -e "${GREEN}Starting Monk Lambda Training/Generation...${NC}"
python monk_lambda.py

# 结束
echo -e "${PURPLE}================ Monk Lambda Process Complete ================${NC}"



