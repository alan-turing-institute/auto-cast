from torch import nn
from abc import ABC
from auto_cast.types import Tensor


class Encoder(nn.Module, ABC):
    """Base encoder."""

    def __init__(self, latent_dim: int, input_channels: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_channels = input_channels

    def encode(self, x: Tensor) -> Tensor:
        """Encode the input tensor into the latent space.

        Args:
            x (Tensor): Input tensor to be encoded.

        Returns
        -------
            Tensor: Encoded tensor in the latent space.
        """
        raise NotImplementedError("The encode method must be implemented by subclasses.")  # noqa: E501, EM101
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass through the Encoder."""
        return self.encode(x)
