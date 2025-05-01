#!/bin/bash
set -e

# Color formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}=========================================${NC}"
echo -e "${PURPLE}=== Monk Lambda Transformer Setup ===${NC}"
echo -e "${PURPLE}=========================================${NC}"

# 安装Python依赖
echo -e "${YELLOW}Installing required Python packages...${NC}"
pip install --upgrade pip
pip install music21 matplotlib tqdm tensorflow==2.15.1 keras

# 创建必要的目录
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p ./data
mkdir -p ./jazz_and_stuff
mkdir -p ./outputs

# 检查TensorFlow GPU
echo -e "${YELLOW}Verifying TensorFlow GPU availability...${NC}"
python -c "import tensorflow as tf; print('GPUs detected by TensorFlow:', tf.config.list_physical_devices('GPU'))"

echo -e "${PURPLE}=================================${NC}"
echo -e "${PURPLE}=== Setup Complete ===${NC}"
echo -e "${PURPLE}=================================${NC}"

# 选择执行模式
echo -e "${YELLOW}What would you like to do?${NC}"
echo "1. Train the model"
echo "2. Generate music (now generates 5 songs to ./outputs/)"
echo "3. Both train and generate"
read -p "Enter your choice (1-3): " choice

case $choice in
  1)
    echo -e "${GREEN}Starting training process...${NC}"
    python monk_lambda_new_.py --mode train
    echo -e "${GREEN}Training complete!${NC}"
    ;;
  2)
    echo -e "${GREEN}Starting music generation (5 songs)...${NC}"
    python monk_lambda_new_.py --mode generate
    echo -e "${GREEN}Music generation complete! Outputs saved in './outputs/' folder.${NC}"
    ;;
  3)
    echo -e "${GREEN}Starting training process...${NC}"
    python monk_lambda_new_.py --mode train
    echo -e "${GREEN}Training complete!${NC}"
    sleep 2
    echo -e "${GREEN}Starting music generation (5 songs)...${NC}"
    python monk_lambda_new_.py --mode generate
    echo -e "${GREEN}Music generation complete! Outputs saved in './outputs/' folder.${NC}"
    ;;
  *)
    echo -e "${RED}Invalid choice. Exiting.${NC}"
    exit 1
    ;;
esac

echo -e "${PURPLE}=================================${NC}"
echo -e "${PURPLE}=== Monk Lambda Process Complete ===${NC}"
echo -e "${PURPLE}=================================${NC}"
