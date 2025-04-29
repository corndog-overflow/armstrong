#!/bin/bash

# Exit immediately if a command fails
set -e

echo "Loading Miniconda and CUDA modules..."
module load miniconda
# module load cuda

# Initialize conda (important for non-login shells)
eval "$(conda shell.bash hook)"

# Define environment name
ENV_NAME=armstrongcondaenv

# Create the conda environment if it doesn't exist
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Creating conda environment: $ENV_NAME..."
    conda create -y -n $ENV_NAME python=3.10
fi


# Now properly activate the environment
conda activate $ENV_NAME
# conda install -y cudatoolkit=11.2 cudnn=8.1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


echo "Installing dependencies..."

# Install packages
conda install -y \
    numpy=1.26 \
    music21 \
    keras \
    cudnn=8.1.0 \
    cudatoolkit=11.2


# Install TensorFlow with correct CUDA linkage
pip install tensorflow==2.10.1

# Verify CUDA is visible
echo "Verifying TensorFlow GPU availability..."
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

echo "hello, world!"
sleep 3
echo "beginning training"
python train_armstrong.py
sleep 3
echo "training complete"
