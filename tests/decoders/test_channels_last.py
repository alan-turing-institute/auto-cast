import torch
from einops import rearrange

from auto_cast.decoders.channels_last import ChannelsLast


def test_channels_last_reorders_dimensions():
    """Test that ChannelsLast splits merged (C*T) and reorders to (B, T, W, H, C)."""
    batch_size, channels, time_steps, width, height = 2, 3, 4, 5, 6
    decoder = ChannelsLast(output_channels=channels, time_steps=time_steps)

    # Input shape: (B, C*T, W, H) - simulating encoder output
    x = torch.randn(batch_size, channels * time_steps, width, height)

    output = decoder(x)

    # Expected output shape: (B, T, W, H, C)
    assert output.shape == (batch_size, time_steps, width, height, channels)

    # Verify the transformation is correct by checking a round-trip
    # Create a known input in (B, T, W, H, C) format
    original = torch.randn(batch_size, time_steps, width, height, channels)
    # Simulate encoder: (B, T, W, H, C) -> (B, C, T, W, H) -> (B, C*T, W, H)
    encoded = rearrange(original, "b t w h c -> b (c t) w h")
    # Decode back
    decoded = decoder(encoded)
    # Should match original
    assert torch.allclose(decoded, original)
