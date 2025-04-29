# monk_lambda.sh (升级版：自动TensorFlow升级到2.15并启用GPU)

#!/bin/bash
set -e

# Color formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Welcome message
echo -e "${PURPLE}==================== Monk Lambda Setup (Auto-Upgrade TF) ===================${NC}"
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
pip install --upgrade pip
pip install music21 matplotlib tqdm

# Install TensorFlow 2.15 to match CUDA 12.8
pip install --upgrade tensorflow==2.15

# Directory setup
mkdir -p ./data ./jazz_and_stuff

# GPU Check
echo -e "${GREEN}Available GPUs:${NC}"
nvidia-smi

# Confirm TensorFlow sees GPU
echo -e "${YELLOW}Verifying TensorFlow GPU availability...${NC}"
python -c "import tensorflow as tf; print('GPUs detected by TensorFlow:', tf.config.list_physical_devices('GPU'))"

# Train or Generate
echo -e "${PURPLE}Select Action:${NC}"
echo "1. Train Transformer"
echo "2. Generate Music"
echo "3. Train and Generate"
echo "4. Exit"
read -p "Enter your choice: " choice

if [ "$choice" == "1" ]; then
  echo -e "${GREEN}Starting training...${NC}"
  python monk_lambda.py --mode train
elif [ "$choice" == "2" ]; then
  echo -e "${GREEN}Starting music generation...${NC}"
  python monk_lambda.py --mode generate
elif [ "$choice" == "3" ]; then
  echo -e "${GREEN}Training...${NC}"
  python monk_lambda.py --mode train
  sleep 2
  echo -e "${GREEN}Generating music...${NC}"
  python monk_lambda.py --mode generate
else
  echo -e "${YELLOW}Exiting.${NC}"
  exit 0
fi

echo -e "${PURPLE}================== Monk Lambda Process Done ==================${NC}"


