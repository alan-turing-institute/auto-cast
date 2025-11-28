from torch import nn

from auto_cast.types import Batch, Tensor


class Decoder(nn.Module):
    """Base Decoder."""

    # Q: Should decoder handle all these input types
    def forward(self, x: Batch) -> Tensor: ...
