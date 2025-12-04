from einops import rearrange

from auto_cast.decoders.base import Decoder
from auto_cast.types import Tensor


class ChannelsLast(Decoder):
    """Decoder that splits merged (channel*time) back to (time, channel) and reorders to channels-last format."""  # noqa: E501

    def __init__(self, output_channels: int, time_steps: int = 1) -> None:
        """Initialize the ChannelsLast decoder.

        Parameters
        ----------
        output_channels: int
            Number of output channels (C).
        time_steps: int
            Number of time steps (T) that were merged with channels in encoding.
        """
        super().__init__()
        self.output_channels = output_channels
        self.time_steps = time_steps

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the ChannelsLast decoder.

        Expects input shape (B, C*T, W, H) and outputs (B, T, W, H, C).
        """
        # Split merged (C*T) dimension back into separate C and T
        # x: (B, C*T, W, H) -> (B, C, T, W, H)
        x = rearrange(
            x, "b (c t) w h -> b c t w h", c=self.output_channels, t=self.time_steps
        )
        # Rearrange to channels-last: (B, C, T, W, H) -> (B, T, W, H, C)
        return rearrange(x, "b c t w h -> b t w h c")
