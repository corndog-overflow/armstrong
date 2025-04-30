# monk_docker.sh （一键拉镜像 + 启动Docker + 安装依赖 + sudo版）

#!/bin/bash
set -e

# Color formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Welcome
echo -e "${PURPLE}================ Monk Docker Setup (Sudo Version) ================${NC}"

# 1. 检查Docker
if ! command -v docker &> /dev/null
then
    echo -e "${YELLOW}Docker not found, please install Docker first.${NC}"
    exit 1
fi

# 2. 拉取官方TensorFlow GPU镜像
echo -e "${YELLOW}Pulling TensorFlow 2.15 GPU Docker image...${NC}"
sudo docker pull tensorflow/tensorflow:2.15.0-gpu

# 3. 定义本地代码路径（改成你的路径！）
LOCAL_PROJECT_PATH="/home/ubuntu/armstrong/armstrong-build/monk(transformer)"

# 4. 启动Docker容器并挂载代码
echo -e "${YELLOW}Starting Docker container...${NC}"
sudo docker run --gpus all -it \
  -v "$LOCAL_PROJECT_PATH":/workspace \
  tensorflow/tensorflow:2.15.0-gpu bash -c "\
    cd /workspace && \
    pip install --upgrade pip && \
    pip install music21 matplotlib tqdm && \
    bash monk_lambda_new_.sh"

# 5. 结束
echo -e "${PURPLE}================ Monk Docker Process Done ================${NC}"

