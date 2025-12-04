import torch
from einops import rearrange

from auto_cast.encoders.permute_concat import PermuteConcat
from auto_cast.types import Batch


def _make_batch(
    batch_size: int = 1,
    t: int = 1,
    w: int = 2,
    h: int = 3,
    c: int = 2,
    const_c: int = 1,
    scalar_c: int = 1,
) -> Batch:
    input_fields = torch.arange(batch_size * t * w * h * c, dtype=torch.float32)
    input_fields = input_fields.view(batch_size, t, w, h, c)
    output_fields = torch.zeros(batch_size, t, w, h, c)
    constant_fields = torch.ones(batch_size, w, h, const_c)
    constant_scalars = torch.full((batch_size, scalar_c), 5.0)
    return Batch(
        input_fields=input_fields,
        output_fields=output_fields,
        constant_scalars=constant_scalars,
        constant_fields=constant_fields,
    )


def test_permute_concat_with_constants():
    encoder = PermuteConcat(with_constants=True)
    batch = _make_batch()

    encoded = encoder(batch)

    # After encoder: (B, T, W, H, C) -> (B, C, T, W, H) -> (B, C*T, W, H)
    base_channels = batch.input_fields.shape[-1]
    time_steps = batch.input_fields.shape[1]
    assert batch.constant_fields is not None
    assert batch.constant_scalars is not None
    const_channels = batch.constant_fields.shape[-1]
    scalar_channels = batch.constant_scalars.shape[-1]

    expected_channels = (
        base_channels * time_steps + const_channels + scalar_channels * time_steps
    )

    assert encoded.shape == (
        batch.input_fields.shape[0],
        expected_channels,
        batch.input_fields.shape[2],
        batch.input_fields.shape[3],
    )

    expected_base = rearrange(batch.input_fields, "b t w h c -> b (c t) w h")
    assert torch.allclose(encoded[:, : base_channels * time_steps, ...], expected_base)

    # Verify constant fields (no time dimension, just added as single channel)
    const_slice = encoded[
        :, base_channels * time_steps : base_channels * time_steps + const_channels, ...
    ]
    expected_const = rearrange(batch.constant_fields, "b w h c -> b c w h")
    assert torch.allclose(const_slice, expected_const)

    # Verify constant scalars (expanded across time and merged)
    scalar_slice = encoded[:, -scalar_channels * time_steps :, ...]
    # Each scalar value is expanded to:
    # (scalar_c, T, W, H)
    # then merged to (scalar_c*T, W, H)
    expected_scalars = batch.constant_scalars.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    expected_scalars = expected_scalars.expand(
        -1, -1, time_steps, batch.input_fields.shape[2], batch.input_fields.shape[3]
    )
    expected_scalars = rearrange(expected_scalars, "b c t w h -> b (c t) w h")
    assert torch.allclose(scalar_slice, expected_scalars)
