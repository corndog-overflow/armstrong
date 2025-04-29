# monk_lambda.sh (修正版：conda+pip混合安装，保证music21、tensorflow等正常)

#!/bin/bash
set -e

# Color formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Welcome message
echo -e "${PURPLE}==================== Monk Setup ===================${NC}"
echo -e "${YELLOW}Activating Conda Environment...${NC}"

# Setup Conda
eval "$(conda shell.bash hook)"
ENV_NAME=monktransformerenv
if ! conda info --envs | grep -q "$ENV_NAME"; then
  echo -e "${YELLOW}Creating Conda Environment: $ENV_NAME${NC}"
  conda create -y -n $ENV_NAME python=3.10
fi
conda activate $ENV_NAME

# Dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
conda install -y numpy
pip install music21 tensorflow==2.10.1 matplotlib tqdm

# Directory setup
mkdir -p ./data ./jazz_and_stuff

# GPU Check
echo -e "${GREEN}Available GPUs:${NC}"
nvidia-smi

# Train or Generate
echo -e "${PURPLE}Select Action:${NC}"
echo "1. Train Transformer (Multi-GPU)"
echo "2. Generate Music (Single-GPU)"
echo "3. Train and Generate"
echo "4. Exit"
read -p "Enter your choice: " choice

if [ "$choice" == "1" ]; then
  echo -e "${GREEN}Starting training...${NC}"
  python monk_lambda.py --mode train
elif [ "$choice" == "2" ]; then
  echo -e "${GREEN}Generating music...${NC}"
  python monk_lambda.py --mode generate
elif [ "$choice" == "3" ]; then
  echo -e "${GREEN}Training model...${NC}"
  python monk_lambda.py --mode train
  sleep 2
  echo -e "${GREEN}Generating music...${NC}"
  python monk_lambda.py --mode generate
else
  echo -e "${YELLOW}Exiting.${NC}"
  exit 0
fi

echo -e "${PURPLE}================== Done ==================${NC}"


