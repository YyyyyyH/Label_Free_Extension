#!/bin/bash

# Initialize Conda
source ~/anaconda3/etc/profile.d/conda.sh

# Create the LFE environment with specified dependencies
conda create -n LFE python=3.8 
conda activate LFE

# Install PyTorch and torchvision
conda install numpy pandas matplotlib scipy tqdm pillow torch captum seaborn pathlib wget -y -c pytorch -y

# Install additional pip packages
pip install captum medmnist

echo "Environment setup is complete."
