#!/bin/bash

conda env create -f environment.yml
conda activate LFE

conda install pytorch torchvision -c pytorch
