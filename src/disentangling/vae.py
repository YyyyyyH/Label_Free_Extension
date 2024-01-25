"""
Module containing the main VAE class.
"""
import logging
import json
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import pathlib
from pathlib import Path
from tqdm import tqdm
from utils.initialization import weights_init
from .encoders import get_encoder
from .decoders import get_decoder
from disentangling.encoders import EncoderBurgess
from disentangling.decoders import DecoderBurgess
from disentangling.losses import BaseLoss, FactorKLoss
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder

MODELS = ["Burgess"]


def init_vae(img_size, latent_dim, loss_f, name):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    encoder = EncoderBurgess(img_size, latent_dim)
    decoder = DecoderBurgess(img_size, latent_dim)
    model = VAE(img_size, encoder, decoder, latent_dim, loss_f, name)
    return model


class VAE(nn.Module):
    def __init__(
        self,
        img_size: tuple,
        encoder: EncoderBurgess,
        decoder: DecoderBurgess,
        latent_dim: int,
        loss_f: BaseLoss,
        name: str = "model",
    ):
        """Class which defines model and forward pass.

        Parameters:
        -----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder
        self.decoder = decoder
        self.loss_f = loss_f
        self.name = name

    def reparameterize(self, mean, logvar):
        """Samples from a normal distribution using the reparameterization trick.

        Parameters:
        -----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """Forward pass of model.

        Parameters:
        -----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def sample_latent(self, x):
        """Returns a sample from the latent distribution.

        Parameters:
        -----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample

    def train_epoch(
        self,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> np.ndarray:
        self.train()
        train_loss = []
        for image_batch, _ in tqdm(dataloader, unit="batches", leave=False):
            image_batch = image_batch.to(device)
            recon_batch, latent_dist, latent_sample = self.forward(image_batch)
            
            # 由于使用FactorKLoss，不再调用self.loss_f直接计算loss
            # 而是使用call_optimize方法
            loss = self.loss_f.call_optimize(
                data=image_batch,
                model=self,
                optimizer=optimizer,
                storer=None
            )
            
            optimizer.step()
            train_loss.append(loss.item())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader):
        self.eval()
        test_loss = []
        with torch.no_grad():
            for image_batch, _ in dataloader:
                image_batch = image_batch.to(device)
                recon_batch, latent_dist, latent_batch = self.forward(image_batch)
                loss = self.loss_f.call_optimize(
                        data=image_batch,
                        model=self,
                        optimizer=None,  # 不需要优化器，因为我们不更新判别器
                        storer=None
                    )
                test_loss.append(loss.item())
        return np.mean(test_loss)

    def fit(
        self,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        save_dir: pathlib.Path,
        n_epoch: int = 30,
        patience: int = 10,
    ) -> None:
        self.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=1e-03, weight_decay=1e-05)
        
        # FactorKLoss的判别器也需要优化器
        optim_d = self.loss_f.optimizer_d
    
        waiting_epoch = 0
        best_test_loss = float("inf")
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss = self.test_epoch(device, test_loader)
            
            # 更新日志和保存逻辑
            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t "
            )
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                self.cpu()
                self.save(save_dir)
                self.to(device)
                waiting_epoch = 0
            else:
                waiting_epoch += 1
                if waiting_epoch >= patience:
                    break


    def save(self, directory: pathlib.Path) -> None:
        """Save a model and corresponding metadata.

        Parameters:
        -----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        model_name = self.name
        self.save_metadata(directory)
        path_to_model = directory / (model_name + ".pt")
        torch.save(self.state_dict(), path_to_model)

    def load_metadata(self, directory: pathlib.Path) -> dict:
        """Load the metadata of a training directory.

        Parameters:
        -----------
        directory : pathlib.Path
            Path to folder where model is saved. For example './experiments/mnist'.
        """
        path_to_metadata = directory / (self.name + ".json")

        with open(path_to_metadata) as metadata_file:
            metadata = json.load(metadata_file)
        return metadata

    def save_metadata(self, directory: pathlib.Path, **kwargs) -> None:
        """Load the metadata of a training directory.

        Parameters:
        -----------
        directory: string
            Path to folder where to save model. For example './experiments/mnist'.
        kwargs:
            Additional arguments to `json.dump`
        """
        path_to_metadata = directory / (self.name + ".json")
        metadata = {
            "latent_dim": self.latent_dim,
            "img_size": self.img_size,
            "num_pixels": self.num_pixels,
            "name": self.name,
        }
        with open(path_to_metadata, "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)

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
