from abc import ABC, abstractmethod

import lightning as L
import torch
from torch import nn

from auto_cast.types import Batch, RolloutOutput, Tensor


class Preprocessor(nn.Module):
    """Base Preprocessor."""

    def forward(self, x: Batch) -> Tensor:
        """Forward Pass through the Preprocessor."""
        msg = "To implement."
        raise NotImplementedError(msg)


class Processor(L.LightningModule, ABC):
    """Processor Base Class."""

    teacher_forcing_ratio: float
    stride: int
    max_rolout_steps: int
    preprocessor: Preprocessor

    def __init__(self):
        pass

    # Option 1
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the Processor."""
        msg = "To implement."
        raise NotImplementedError(msg)

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        x = self.preprocessor(batch)
        return self(x)

    # # Option 2
    # def forward(self, x: Batch) -> Tensor:
    #     """Forward pass through the Processor."""
    #     msg = "To implement."
    #     raise NotImplementedError(msg)

    # def training_step(self, batch: Batch, batch_idx: int):
    #     self(batch)

    def configure_optmizers(self):
        pass

    def rollout(self, batch: Batch) -> RolloutOutput:
        """Rollout over multiple time steps."""
        pred_outs = []
        gt_outs = []
        for _time_step in range(0, self.max_rolout_steps, self.stride):
            x = self.preprocessor(batch)
            pred_outs.append(self(x))
            gt_outs.append(
                batch["output_fields"]
            )  # Q: this assumes we have output fields
        return torch.stack(pred_outs), torch.stack(gt_outs)


class DiscreteProcessor(Processor, ABC):
    """DiscreteProcessor."""

    @abstractmethod
    def map(self, x: Batch) -> Tensor:
        ...
        # Map input window of states/times to output window

    def rollout(self, batch: Batch) -> RolloutOutput:
        ...

        # Use self.map to generate trajectory


class FlowBasedGenerativeProcessor(DiscreteProcessor):
    """Flow-based generative processor."""

    def map(self, x: Batch) -> Tensor:
        ...
        # Sample generative model    def loss(self, ...):...
        # Flow matc
