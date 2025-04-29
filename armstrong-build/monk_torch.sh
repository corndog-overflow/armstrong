#!/bin/bash

# Exit immediately if a command fails
set -e

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}=========================================${NC}"
echo -e "${PURPLE}=== Monk Torch Transformer Setup Script ===${NC}"
echo -e "${PURPLE}=========================================${NC}"

# Make sure ~/.local/bin is in PATH (for --user installs)
export PATH=$HOME/.local/bin:$PATH

echo -e "${BLUE}Upgrading pip...${NC}"
python3 -m pip install --upgrade pip

echo -e "${BLUE}Installing Python dependencies...${NC}"
pip3 install --user numpy==1.26 music21 matplotlib tqdm

echo -e "${BLUE}Installing PyTorch (CUDA 11.8)...${NC}"
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create necessary directories
echo -e "${BLUE}Creating project directories...${NC}"
mkdir -p ./data
mkdir -p ./jazz_and_stuff
mkdir -p ./weights
mkdir -p ./outputs

# Verify CUDA is visible to PyTorch
echo -e "${YELLOW}Verifying PyTorch GPU availability...${NC}"
python3 -c "import torch; print('GPU devices found:', torch.cuda.device_count())"

echo -e "${PURPLE}=================================${NC}"
echo -e "${PURPLE}=== Monk Torch Transformer Setup Complete ===${NC}"
echo -e "${PURPLE}=================================${NC}"

# Prompt for action
echo -e "${YELLOW}What would you like to do?${NC}"
echo "1. Train the model"
echo "2. Generate music (requires a trained model)"
echo "3. Both train and generate"
read -p "Enter your choice (1-3): " choice

case $choice in
  1)
    echo -e "${GREEN}Beginning training process...${NC}"
    python3 monk_torch.py --mode train
    echo -e "${GREEN}Training complete!${NC}"
    ;;
  2)
    echo -e "${GREEN}Beginning music generation...${NC}"
    python3 monk_torch.py --mode generate
    echo -e "${GREEN}Music generation complete! Output saved as './outputs/transformer_generated_torch.mid'${NC}"
    ;;
  3)
    echo -e "${GREEN}Beginning training process...${NC}"
    python3 monk_torch.py --mode train
    echo -e "${GREEN}Training complete!${NC}"
    sleep 2
    echo -e "${GREEN}Now generating music with the trained model...${NC}"
    python3 monk_torch.py --mode generate
    echo -e "${GREEN}Music generation complete! Output saved as './outputs/transformer_generated_torch.mid'${NC}"
    ;;
  *)
    echo -e "${RED}Invalid choice. Exiting.${NC}"
    exit 1
    ;;
esac

echo -e "${PURPLE}=================================${NC}"
echo -e "${PURPLE}=== Monk Torch Transformer Process Complete ===${NC}"
echo -e "${PURPLE}=================================${NC}"
