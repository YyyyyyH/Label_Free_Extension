#!/bin/bash

# 创建新的 Conda 环境
conda create -n LFE python=3.8 -y
conda activate LFE

# 安装 Conda 包
conda install numpy pandas matplotlib scipy tqdm pillow seaborn -y
conda install -c pytorch torch torchvision -y

# 安装 pip 包
pip install captum medmnist

echo "环境设置完成。"
