import torch
from azula.nn.layers import ConvNd
from torch import nn

from auto_cast.decoders import Decoder
from auto_cast.encoders import Encoder
from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.types import Batch, Tensor


class VAELoss(nn.Module):
    """Variational Autoencoder Loss Function."""

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, model: EncoderDecoder, batch: Batch) -> Tensor:
        decoded, encoded = model.forward_with_latent(batch)

        return self.beta * self.kl_divergence(encoded) + nn.functional.mse_loss(
            decoded, batch.output_fields
        )

    def kl_divergence(self, encoded: Tensor) -> Tensor:
        """Compute the KL divergence loss.

        Parameters
        ----------
        encoded: Tensor
            Encoded tensor containing mean and log variance.
            Shape: [B, 2*C, H, W, ...] for spatial or
            [B, 2*latent_dim] for flat.

        Returns
        -------
        Tensor
            KL divergence loss.
        """
        # Split along the appropriate dimension
        split_dim = 1 if encoded.dim() > 2 else -1
        mean, log_var = encoded.chunk(2, dim=split_dim)
        # Compute KL divergence, sum over all non-batch dimensions
        kl_div = -0.5 * torch.sum(
            1 + log_var - mean.pow(2) - log_var.exp(),
            dim=list(range(1, encoded.dim())),
        )
        return kl_div.mean()


class VAE(EncoderDecoder):
    """Variational Autoencoder Model.

    Supports both flat (B, latent_dim) and spatial (B, C, H, W, ...)
    latent representations.
    """

    encoder: Encoder
    decoder: Decoder
    fc_mean: nn.Module
    fc_log_var: nn.Module

    def __init__(self, encoder: Encoder, decoder: Decoder, spatial: int | None = None):
        """Initialize VAE.

        Parameters
        ----------
        encoder : Encoder
            Encoder network.
        decoder : Decoder
            Decoder network.
        spatial : int | None
            Number of spatial dimensions in latent space (e.g., 2 for images).
            If None, assumes flat 1D latent representation.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.spatial = spatial
        latent_dim = encoder.latent_dim
        if encoder.latent_dim != decoder.latent_dim:
            msg = "Encoder and Decoder latent dimensions must match for VAE."
            raise ValueError(msg)

        # For spatial latents, use 1x1 convolutions; for flat, use linear
        if spatial is not None:
            self.fc_mean = ConvNd(
                latent_dim,
                latent_dim,
                spatial=spatial,
                kernel_size=1,
            )
            self.fc_log_var = ConvNd(
                latent_dim,
                latent_dim,
                spatial=spatial,
                kernel_size=1,
            )
        else:
            self.fc_mean = nn.Linear(latent_dim, latent_dim)
            self.fc_log_var = nn.Linear(latent_dim, latent_dim)

        self.loss_func = VAELoss()

    def forward(self, batch: Batch) -> Tensor:
        return self.forward_with_latent(batch)[0]

    def forward_with_latent(self, batch: Batch) -> tuple[Tensor, Tensor]:
        encoded = self.encode(batch)
        # Split along channel dim (1) for spatial, feature dim (-1) for flat
        split_dim = 1 if self.spatial is not None else -1
        mean, log_var = encoded.chunk(2, dim=split_dim)
        z = self.reparametrize(mean, log_var)
        decoded = self.decode(z)
        return decoded, encoded

    def reparametrize(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterisation trick.

        Samples z ~ N(mean, sigma) during training, but returns the mean
        deterministically in evaluation mode. This makes `model.eval()` produce
        deterministic reconstructions while training remains stochastic.
        """
        if not self.training:
            return mean

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, batch: Batch) -> Tensor:
        h = self.encoder.encode(batch)  # not super().encode
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        concat_dim = 1 if self.spatial is not None else -1
        return torch.cat([mean, log_var], dim=concat_dim)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss = self.loss_func(self, batch)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss
