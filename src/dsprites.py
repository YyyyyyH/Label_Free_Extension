import argparse
import csv
import itertools
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from captum.attr import GradientShap
from utils.features import attribute_individual_dim
from torch.utils.data import random_split

from disentangling.vae import VAE
from disentangling.encoders import EncoderBurgess
from disentangling.decoders import DecoderBurgess
from disentangling.losses import BetaHLoss, BtcvaeLoss, FactorKLoss
from utils.datasets import DSprites
from utils.metrics import (
    compute_metrics,
    cos_saliency,
    count_activated_neurons,
    entropy_saliency,
    pearson_saliency,
    spearman_saliency,
)
from utils.visualize import plot_vae_saliencies, vae_box_plots
import torch.nn as nn
import numpy as np
from disentangling.encoders import Encoder
from disentangling.quantizer import VectorQuantizer
from disentangling.decoders import Decoder

def disvae_feature_importance(
    random_seed: int = 1,
    batch_size: int = 500,
    n_plots: int = 10,
    n_runs: int = 2,
    dim_latent: int = 3,
    n_epochs: int = 2,
    gamma_list: list = [1, 5, 10],
    test_split=0.1,
) -> None:
    # Initialize seed and device
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load dsprites
    W = 64
    img_size = (1, W, W)
    data_dir = Path.cwd() / "data/dsprites"
    dsprites_dataset = DSprites(str(data_dir))
    test_size = int(test_split * len(dsprites_dataset))
    train_size = len(dsprites_dataset) - test_size
    train_dataset, test_dataset = random_split(
        dsprites_dataset, [train_size, test_size]
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Create saving directory
    save_dir = Path.cwd() / "results/dsprites/vae"
    if not save_dir.exists():
        os.makedirs(save_dir)

    # Define the computed metrics and create a csv file with appropriate headers
    # loss_list = [FactorKLoss(device=device), BetaHLoss(), BtcvaeLoss(is_mss=False, n_data=len(train_dataset))]
    loss_list = [FactorKLoss(device=device)]
    metric_list = [
        pearson_saliency,
        spearman_saliency,
        cos_saliency,
        entropy_saliency,
        count_activated_neurons,
    ]
    metric_names = [
        "Pearson Correlation",
        "Spearman Correlation",
        "Cosine",
        "Entropy",
        "Active Neurons",
    ]
    headers = ["Loss Type"] + metric_names
    csv_path = save_dir / "metrics.csv"
    if not csv_path.is_file():
        logging.info(f"Creating metrics csv in {csv_path}")
        with open(csv_path, "w") as csv_file:
            dw = csv.DictWriter(csv_file, delimiter=",", fieldnames=headers)
            dw.writeheader()

    for gamma, run in itertools.product(gamma_list, range(1, n_runs + 1)):
        # Initialize vaes
        encoder = EncoderBurgess(img_size, dim_latent)
        decoder = DecoderBurgess(img_size, dim_latent)
        loss = FactorKLoss(device=device, gamma=gamma)
        name = f"factorK-vae_run{run}"
        model = VAE(img_size, encoder, decoder, dim_latent, loss, name=name)
        logging.info(f"Now fitting {name}")
        model.fit(device, train_loader, test_loader, save_dir, n_epochs)
        model.load_state_dict(torch.load(save_dir / (name + ".pt")), strict=False)

        # Compute test-set saliency and associated metrics
        baseline_image = torch.zeros((1, 1, W, W), device=device)
        gradshap = GradientShap(encoder.mu)
        attributions = attribute_individual_dim(
            encoder.mu, dim_latent, test_loader, device, gradshap, baseline_image
        )
        metrics = compute_metrics(attributions, metric_list)
        results_str = "\t".join(
            [f"{metric_names[k]} {metrics[k]:.2g}" for k in range(len(metric_list))]
        )
        logging.info(f"Model {name} \t {results_str}")

        # Save the metrics
        with open(csv_path, "a", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow([str(loss), gamma] + metrics)

        # Plot a couple of examples
        plot_idx = [n for n in range(n_plots)]
        images_to_plot = [test_dataset[i][0].numpy().reshape(W, W) for i in plot_idx]
        fig = plot_vae_saliencies(images_to_plot, attributions[plot_idx])
        fig.savefig(save_dir / f"{name}.pdf")
        plt.close(fig)

    fig = vae_box_plots(pd.read_csv(csv_path), metric_names)
    fig.savefig(save_dir / "metric_box_plots.pdf")
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    disvae_feature_importance(
        n_runs=args.n_runs, batch_size=args.batch_size, random_seed=args.seed
    )

class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity
