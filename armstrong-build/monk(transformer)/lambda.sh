#!/bin/bash

# Exit immediately if any command fails
set -e

# Color codes for nice output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}=========================================${NC}"
echo -e "${PURPLE}=== Monk Music Transformer Setup Script for Lambda ===${NC}"
echo -e "${PURPLE}=========================================${NC}"

# Skip module loading, directly activate conda (Lambda has conda already installed)

# Initialize conda shell
eval "$(conda shell.bash hook)"

# Define environment name
ENV_NAME=monktransformerenv

# Create the conda environment if it doesn't exist
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo -e "${GREEN}Creating conda environment: $ENV_NAME...${NC}"
    conda create -y -n $ENV_NAME python=3.10
fi

# Activate the environment
conda activate $ENV_NAME

# Export library path for safety
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo -e "${BLUE}Installing dependencies...${NC}"

# Install essential libraries
conda install -y numpy=1.26 music21
pip install tensorflow==2.10.1 matplotlib tqdm

# Create necessary directories
mkdir -p ./data
mkdir -p ./jazz_and_stuff

# Verify GPU
echo -e "${YELLOW}Verifying TensorFlow GPU availability...${NC}"
python -c "import tensorflow as tf; print('GPU devices found:', tf.config.list_physical_devices('GPU'))"

echo -e "${PURPLE}=================================${NC}"
echo -e "${PURPLE}=== Setup Complete ===${NC}"
echo -e "${PURPLE}=================================${NC}"

# Ask user what to do
echo -e "${YELLOW}What would you like to do?${NC}"
echo "1. Train the model"
echo "2. Generate music (requires a trained model)"
echo "3. Both train and generate"
read -p "Enter your choice (1-3): " choice

case $choice in
  1)
    echo -e "${GREEN}Starting training process...${NC}"
    python monk.py --mode train
    echo -e "${GREEN}Training complete!${NC}"
    ;;
  2)
    echo -e "${GREEN}Starting music generation...${NC}"
    python monk.py --mode generate
    echo -e "${GREEN}Music generation complete! Output saved as 'transformer_generated.mid'${NC}"
    ;;
  3)
    echo -e "${GREEN}Starting training process...${NC}"
    python monk.py --mode train
    echo -e "${GREEN}Training complete!${NC}"
    sleep 2
    echo -e "${GREEN}Now generating music...${NC}"
    python monk.py --mode generate
    echo -e "${GREEN}Music generation complete! Output saved as 'transformer_generated.mid'${NC}"
    ;;
  *)
    echo -e "${RED}Invalid choice. Exiting.${NC}"
    exit 1
    ;;
esac

echo -e "${PURPLE}=================================${NC}"
echo -e "${PURPLE}=== Monk Music Transformer Process Complete ===${NC}"
echo -e "${PURPLE}=================================${NC}"
