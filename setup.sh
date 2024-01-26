#!/bin/bash

# Initialize Conda
source ~/anaconda3/etc/profile.d/conda.sh

# Create and activate the environment
conda env create -f environment.yml
conda activate LFE

# Install additional dependencies
conda install pytorch torchvision -c pytorch
