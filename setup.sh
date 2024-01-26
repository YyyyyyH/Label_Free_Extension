#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

# 创建并激活环境
conda env create -f environment.yml
conda activate LFE

# 安装其他依赖
conda install pytorch torchvision -c pytorch

