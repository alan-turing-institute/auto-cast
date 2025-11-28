from torch import nn

from auto_cast.types import Tensor
from abc import ABC

class Decoder(nn.Module, ABC):
    """Base Decoder."""

    def __init__(self, latent_dim: int, output_channels: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels

    def decode(self, z: Tensor) -> Tensor:
        """Decode the latent tensor back to the original space.

        Args:
            z (Tensor): Latent tensor to be decoded.

        Returns
        -------
            Tensor: Decoded tensor in the original space.
        """
        raise NotImplementedError("The decode method must be implemented by subclasses.")  # noqa: E501, EM101
    
    def forward(self, z: Tensor) -> Tensor:
        """Forward Pass through the Decoder."""
        return self.decode(z)
    