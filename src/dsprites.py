import argparse
import csv
import itertools
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import GradientShap
from utils.features import attribute_individual_dim
from torch.utils.data import random_split

from disentangeling_vae.vae import VAE
from disentangeling_vae.encoder import EncoderBurgess
from disentangeling_vae.decoder import DecoderBurgess
from disentangeling_vae import BetaHLoss, BtcvaeLoss
from lfxai.utils.datasets import DSprites
from lfxai.utils.metrics import (
    compute_metrics,
    cos_saliency,
    count_activated_neurons,
    entropy_saliency,
    pearson_saliency,
    spearman_saliency,
)
from lfxai.utils.visualize import plot_vae_saliencies, vae_box_plots
