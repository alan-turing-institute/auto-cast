from torch import nn

from auto_cast.types import Tensor


class Encoder(nn.Module):
    """Base encoder."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward Pass through the Encoder."""
        msg = "To implement."
        raise NotImplementedError(msg)
