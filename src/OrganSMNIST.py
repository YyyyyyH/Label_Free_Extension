import torch
from pathlib import Path
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.nn import MSELoss
from captum.attr import IntegratedGradients

from lfxai.models.images import AutoEncoderMnist, EncoderMnist, DecoderMnist
from lfxai.models.pretext import Identity
from lfxai.explanations.features import attribute_auxiliary
from lfxai.explanations.examples import SimplEx
import argparse
import csv
import itertools
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from captum.attr import GradientShap, IntegratedGradients, Saliency
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import transforms

from lfxai.explanations.examples import (
    InfluenceFunctions,
    NearestNeighbours,
    SimplEx,
    TracIn,
)
from lfxai.explanations.features import attribute_auxiliary, attribute_individual_dim
from lfxai.models.images import (
    VAE,
    AutoEncoderMnist,
    ClassifierMnist,
    DecoderBurgess,
    DecoderMnist,
    EncoderBurgess,
    EncoderMnist,
)
from lfxai.models.losses import BetaHLoss, BtcvaeLoss
from lfxai.models.pretext import Identity, Mask, RandomNoise
from lfxai.utils.datasets import MaskedMNIST
from lfxai.utils.feature_attribution import generate_masks
from lfxai.utils.metrics import (
    compute_metrics,
    cos_saliency,
    count_activated_neurons,
    entropy_saliency,
    pearson_saliency,
    similarity_rates,
    spearman_saliency,
)
from lfxai.utils.visualize import (
    correlation_latex_table,
    plot_pretext_saliencies,
    plot_pretext_top_example,
    plot_vae_saliencies,
    vae_box_plots,
)
# Load MedMNIST data
from medmnist import OrganSMNIST
from medmnist import INFO, Evaluator


# Select torch device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Select the MedMNIST subset you want to use
data_flag = 'organsmnist'  # for example, use 'pathmnist' for patholog images
info = INFO['organsmnist']
task = info['task']
n_channels = info['n_channels']
n_classes = 11

# Preprocessing transformations
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels)
])


data_dir = Path.cwd() / "data" / data_flag
#train_dataset = MedMNIST(root=data_dir, split='train', transform=transformations, download=True)
#test_dataset = MedMNIST(root=data_dir, split='test', transform=transformations, download=True)

train_dataset = OrganSMNIST(split='train', download=True, transform=transformations)
test_dataset = OrganSMNIST(split='test', download=True, transform=transformations)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


# Get a model
encoder = EncoderMnist(encoded_space_dim=10)
decoder = DecoderMnist(encoded_space_dim=10)
model = AutoEncoderMnist(encoder, decoder, latent_dim=10, input_pert=Identity())
model.to(device)

## Get label-free feature importance
#baseline = torch.zeros((1, 1, 28, 28)).to(device) # black image as baseline
#attr_method = IntegratedGradients(model)
#feature_importance = attribute_auxiliary(encoder, test_loader,
#                                         device, attr_method, baseline)
#
## Get label-free example importance
#train_subset = Subset(train_dataset, indices=list(range(500))) # Limit the number of training examples
#train_subloader = DataLoader(train_subset, batch_size=500)
#attr_method = SimplEx(model, loss_f=MSELoss())
#example_importance = attr_method.attribute_loader(device, train_subloader, test_loader)
def consistency_feature_importance(
    random_seed: int = 1,
    batch_size: int = 200,
    dim_latent: int = 4,
    n_epochs: int = 100,
) -> None:
    # Initialize seed and device
    torch.random.manual_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    W = 28  # Image width = height
    pert_percentages = [5, 10, 20, 50, 80, 100]

    # Load MNIST
    data_dir = Path.cwd() / "data/Medmnist"
    train_dataset = OrganSMNIST(split='train', download=True, transform=transformations)
    test_dataset = OrganSMNIST(split='test', download=True, transform=transformations)
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Initialize encoder, decoder and autoencoder wrapper
    pert = RandomNoise()
    encoder = EncoderMnist(encoded_space_dim=dim_latent)
    decoder = DecoderMnist(encoded_space_dim=dim_latent)
    autoencoder = AutoEncoderMnist(encoder, decoder, dim_latent, pert)
    encoder.to(device)
    decoder.to(device)

    # Train the denoising autoencoder
    save_dir = Path.cwd() / "results/mnist/consistency_features"
    if not save_dir.exists():
        os.makedirs(save_dir)
    autoencoder.fit(device, train_loader, test_loader, save_dir, n_epochs)
    autoencoder.load_state_dict(
        torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
    )

    attr_methods = {
        "Gradient Shap": GradientShap,
        "Integrated Gradients": IntegratedGradients,
        "Saliency": Saliency,
        "Random": None,
    }
    results_data = []
    baseline_features = torch.zeros((1, 1, W, W)).to(
        device
    )  # Baseline image for attributions
    for method_name in attr_methods:
        logging.info(f"Computing feature importance with {method_name}")
        results_data.append([method_name, 0, 0])
        attr_method = attr_methods[method_name]
        if attr_method is not None:
            attr = attribute_auxiliary(
                encoder, test_loader, device, attr_method(encoder), baseline_features
            )
        else:
            np.random.seed(random_seed)
            attr = np.random.randn(len(test_dataset), 1, W, W)

        for pert_percentage in pert_percentages:
            logging.info(
                f"Perturbing {pert_percentage}% of the features with {method_name}"
            )
            mask_size = int(pert_percentage * W**2 / 100)
            masks = generate_masks(attr, mask_size)
            for batch_id, (images, _) in enumerate(test_loader):
                mask = masks[
                    batch_id * batch_size : batch_id * batch_size + len(images)
                ].to(device)
                images = images.to(device)
                original_reps = encoder(images)
                images = mask * images
                pert_reps = encoder(images)
                rep_shift = torch.mean(
                    torch.sum((original_reps - pert_reps) ** 2, dim=-1)
                ).item()
                results_data.append([method_name, pert_percentage, rep_shift])

    logging.info("Saving the plot")
    results_df = pd.DataFrame(
        results_data, columns=["Method", "% Perturbed Pixels", "Representation Shift"]
    )
    sns.set(font_scale=1.3)
    sns.set_style("white")
    sns.set_palette("colorblind")
    sns.lineplot(
        data=results_df, x="% Perturbed Pixels", y="Representation Shift", hue="Method"
    )
    plt.tight_layout()
    plt.savefig(save_dir / "OrganSMNIST_consistency_features.pdf")
    plt.close()
#
#def consistency_feature_importance(
#    random_seed: int = 1,
#    batch_size: int = 200,
#    dim_latent: int = 4,
#    n_epochs: int = 100,
#) -> None:
#    # Initialize seed and device
#    torch.random.manual_seed(random_seed)
#    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#    W = 28  # Image width = height
#    pert_percentages = [5, 10, 20, 50, 80, 100]
#
#    # Load MNIST
#    data_flag = 'chestmnist'  # for example, use 'pathmnist' for pathology images
#    info = INFO['chestmnist']
#    task = info['task']
#    n_channels = info['n_channels']
#    #n_classes = len(info['label'])
#    n_classes = 14
#
#    # Preprocessing transformations
#    transformations = transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize(mean=[0.5] * n_channels, std=[0.5] * n_channels)
#    ])
#    data_dir = Path.cwd() / "data" / data_flag
#    train_dataset = ChestMNIST(split='train', download=True, transform=transformations)
#    test_dataset = ChestMNIST(split='test', download=True, transform=transformations)
#
## Create data loaders
#    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
#    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
##    print(*test_loader)
#
#
#    # Initialize encoder, decoder and autoencoder wrapper
#    pert = RandomNoise()
#    encoder = EncoderMnist(encoded_space_dim=dim_latent)
#    decoder = DecoderMnist(encoded_space_dim=dim_latent)
#    autoencoder = AutoEncoderMnist(encoder, decoder, dim_latent, pert)
#    encoder.to(device)
#    decoder.to(device)
##
#    # Train the denoising autoencoder
#    save_dir = Path.cwd() / "results/mnist/consistency_features"
#    if not save_dir.exists():
#        os.makedirs(save_dir)
#    autoencoder.fit(device, train_loader, test_loader, save_dir, n_epochs)
#    autoencoder.load_state_dict(
#        torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
#    )
#
#    attr_methods = {
#        "Gradient Shap": GradientShap,
#        "Integrated Gradients": IntegratedGradients,
#        "Saliency": Saliency,
#        "Random": None,
#    }
#    results_data = []
#    baseline_features = torch.zeros((1, 1, W, W)).to(
#        device
#    )  # Baseline image for attributions
#    for method_name in attr_methods:
#        logging.info(f"Computing feature importance with {method_name}")
#        results_data.append([method_name, 0, 0])
#        attr_method = attr_methods[method_name]
#        if attr_method is not None:
#            attr = attribute_auxiliary(
#                encoder, test_loader, device, attr_method(encoder), baseline_features
#            )
#        else:
#            np.random.seed(random_seed)
#            attr = np.random.randn(len(test_dataset), 1, W, W)
#
#        for pert_percentage in pert_percentages:
#            logging.info(
#                f"Perturbing {pert_percentage}% of the features with {method_name}"
#            )
#            mask_size = int(pert_percentage * W**2 / 100)
#            masks = generate_masks(attr, mask_size)
#            for batch_id, (images, _) in enumerate(test_loader):
#                current_batch_size = images.size(0)
#                mask = masks[batch_id * batch_size : batch_id * batch_size + current_batch_size].to(device)
#                images = images.to(device)
#                original_reps = encoder(images)
#                images = mask * images
#                pert_reps = encoder(images)
#                rep_shift = torch.mean(
#                    torch.sum((original_reps - pert_reps) ** 2, dim=-1)
#                ).item()
#                results_data.append([method_name, pert_percentage, rep_shift])
#
#    logging.info("Saving the plot")
#    results_df = pd.DataFrame(
#        results_data, columns=["Method", "% Perturbed Pixels", "Representation Shift"]
#    )
#    sns.set(font_scale=1.3)
#    sns.set_style("white")
#    sns.set_palette("colorblind")
#    sns.lineplot(
#        data=results_df, x="% Perturbed Pixels", y="Representation Shift", hue="Method"
#    )
#    plt.tight_layout()
#    plt.savefig(save_dir / "mnist_consistency_features.pdf")
#    plt.close()

consistency_feature_importance(
    random_seed=1,   # You can change these parameters as needed
    batch_size=200,
    dim_latent=4,   # Ensure this matches your model's latent dimension
    n_epochs=100
)

