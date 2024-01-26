#!/bin/bash

# Initialize Conda
source ~/anaconda3/etc/profile.d/conda.sh

# Create the LFE environment with specified dependencies
conda create -n LFE python=3.8 numpy pandas matplotlib scipy tqdm pillow seaborn pathlib wget -y -c pytorch -c defaults
conda activate LFE

# Install PyTorch and torchvision
conda install pytorch torchvision -c pytorch -y

# Install additional pip packages
pip install captum medmnist

echo "Environment setup is complete."
