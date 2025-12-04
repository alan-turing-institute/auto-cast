from abc import ABC, abstractmethod

from torch import nn

from auto_cast.types import Tensor, TensorBMStarL, TensorBTSPlusC


class Decoder(nn.Module, ABC):
    """Base Decoder."""

    decoder_model: nn.Module
    latent_dim: int

    def __init__(self, latent_dim: int, output_channels: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels

    def postprocess(self, decoded: Tensor) -> TensorBTSPlusC:
        """Optionally transform the decoded tensor before returning.

        Subclasses can override to implement post-decoding steps. Default is
        identity.
        """
        return decoded

    @abstractmethod
    def decode(self, z: TensorBMStarL) -> TensorBTSPlusC:
        """Decode the latent tensor back to the original space.

        Parameters
        ----------
        z: TensorBMultiL
            Latent tensor to be decoded.

        Returns
        -------
            Tensor: Decoded tensor in the original space.
        """

    def __call__(self, z: TensorBMStarL) -> TensorBTSPlusC:
        return self.decode(z)
