#!/bin/bash
# Exit immediately if a command fails
set -e

echo "========================================"
echo "  Armstrong MIDI Generator - Inference"
echo "========================================"

# Load Miniconda module
echo "Loading Miniconda module..."
module load miniconda

# Initialize conda (important for non-login shells)
eval "$(conda shell.bash hook)"

# Define environment name
ENV_NAME=armstrongcondaenv

# Activate the conda environment
echo "Activating conda environment: $ENV_NAME..."
conda activate $ENV_NAME

# Set up library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Verify TensorFlow and GPU availability
echo "Verifying TensorFlow installation..."
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import tensorflow as tf; print('GPU Devices:', tf.config.list_physical_devices('GPU'))"

# Run inference
echo "Starting Armstrong inference process..."
python gen_armstrong.py

# Check if MIDI was generated
if [ -f "generated.mid" ]; then
    echo "========================================"
    echo "Success! MIDI file 'generated.mid' created."
    echo "========================================"
else
    echo "========================================"
    echo "Error: MIDI file was not generated."
    echo "========================================"
    exit 1
fi