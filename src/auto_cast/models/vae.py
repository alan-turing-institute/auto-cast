import torch
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

        Returns
        -------
        Tensor
            KL divergence loss.
        """
        if (
            isinstance(encoded, Tensor)
            and encoded.dim() == 2
            and encoded.size(1) != 2 * (encoded.size(1) // 2)
        ):
            msg = "encoded must be [B, 2 * latent_dim]"
            raise ValueError(msg)
        mean, log_var = encoded.chunk(2, dim=-1)
        kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1)
        return kl_div.mean()


class VAE(EncoderDecoder):
    """Variational Autoencoder Model."""

    encoder: Encoder
    decoder: Decoder
    fc_mean: nn.Linear
    fc_log_var: nn.Linear

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc_mean = nn.Linear(encoder.latent_dim, encoder.latent_dim)
        self.fc_log_var = nn.Linear(encoder.latent_dim, encoder.latent_dim)
        self.loss_func = VAELoss()

    def forward(self, batch: Batch) -> Tensor:
        return self.forward_with_latent(batch)[0]

    def forward_with_latent(self, batch: Batch) -> tuple[Tensor, Tensor]:
        encoded = self.encode(batch)
        if (
            isinstance(encoded, Tensor)
            and encoded.dim() == 2
            and encoded.size(1) != 2 * self.encoder.latent_dim
        ):
            msg = "encoded must be [B, 2 * latent_dim]"
            raise ValueError(msg)
        mean, log_var = encoded.chunk(2, dim=-1)
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
        h = super().encode(batch)  # [B, latent_dim]
        mean = self.fc_mean(h)  # [B, latent_dim]
        log_var = self.fc_log_var(h)  # [B, latent_dim]
        return torch.cat([mean, log_var], dim=-1)  # [B, 2*latent_dim]

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss = self.loss_func(self, batch)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.input_fields.shape[0]
        )
        return loss
