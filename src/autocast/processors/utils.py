from __future__ import annotations

from collections.abc import Sequence

import torch


def initialize_flow_matching_backbone(
    processor,
    n_steps_input: int | None,
    channel_count: int | None,
    spatial_shape: Sequence[int] | None,
) -> None:
    """Instantiate the flow-matching backbone before optimizers are created."""
    builder = getattr(processor, "_maybe_build_backbone", None)
    has_model = getattr(processor, "flow_matching_model", None) is not None
    if builder is None or has_model:
        return
    if n_steps_input is None or channel_count is None:
        return
    spatial = tuple(spatial_shape) if spatial_shape is not None else ()
    dummy = torch.zeros(
        (1, n_steps_input, *spatial, channel_count), dtype=torch.float32
    )
    builder(dummy)
