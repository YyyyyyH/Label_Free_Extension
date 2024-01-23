"""
Module containing the main VAE class.
"""
import torch
from torch import nn, optim
from torch.nn import functional as F

from utils.initialization import weights_init
from .encoders import get_encoder
from .decoders import get_decoder

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
        loss_f: BaseVAELoss,
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
            recon_batch, latent_dist, latent_batch = self.forward(image_batch)
            loss = self.loss_f(
                image_batch,
                recon_batch,
                latent_dist,
                is_train=True,
                storer=None,
                latent_sample=latent_batch,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader):
        self.eval()
        test_loss = []
        with torch.no_grad():
            for image_batch, _ in dataloader:
                image_batch = image_batch.to(device)
                recon_batch, latent_dist, latent_batch = self.forward(image_batch)
                loss = self.loss_f(
                    image_batch,
                    recon_batch,
                    latent_dist,
                    is_train=True,
                    storer=None,
                    latent_sample=latent_batch,
                )
                test_loss.append(loss.cpu().numpy())
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
        waiting_epoch = 0
        best_test_loss = float("inf")
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(device, train_loader, optim)
            test_loss = self.test_epoch(device, test_loader)
            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t "
            )
            if test_loss >= best_test_loss:
                waiting_epoch += 1
                logging.info(
                    f"No improvement over the best epoch \t Patience {waiting_epoch} / {patience}"
                )
            else:
                logging.info(f"Saving the model in {save_dir}")
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_loss = test_loss.data
                waiting_epoch = 0
            if waiting_epoch == patience:
                logging.info("Early stopping activated")
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
