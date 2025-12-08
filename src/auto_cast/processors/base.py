from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import torch
from torch import nn

# Assuming these are the correct imports for your types:
from auto_cast.processors.rollout import RolloutMixin
from auto_cast.types import EncodedBatch, Tensor, TensorBMStarL, Batch # <-- Added Batch import


class Processor(RolloutMixin[EncodedBatch], ABC, L.LightningModule):
    """Processor Base Class."""

    def __init__(
        self,
        *,
        stride: int = 1,
        teacher_forcing_ratio: float = 0.0,
        max_rollout_steps: int = 1,
        loss_func: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_rollout_steps = max_rollout_steps
        self.loss_func = loss_func or nn.MSELoss()
        for key, value in kwargs.items():
            setattr(self, key, value)

    learning_rate: float

    def forward(self, *args, **kwargs: Any) -> Any:
        """Forward pass through the Processor."""
        msg = "To implement."
        raise NotImplementedError(msg)
        

    def training_step(self, batch: EncodedBatch, batch_idx: int) -> Tensor: # <-- Changed type hint from EncodedBatch to Batch
        # Convert raw Batch to EncodedBatch before processing        
        output = self.map(batch.encoded_inputs)
        loss = self.loss_func(output, batch.encoded_output_fields)
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=batch.encoded_inputs.shape[0]
        )
        return loss

    @abstractmethod
    def map(self, x: Tensor) -> Tensor:
        '''
        Maps input states to output states.

        Args:
            x (Tensor): Input tensor of shape (B, T_in, ...)
        Returns:
            y (Tensor): Output tensor of shape (B, T_out, ...)
        '''

    def validation_step(self, batch: EncodedBatch, batch_idx: int) -> Tensor: # <-- Changed type hint from EncodedBatch to Batch
        # Convert raw Batch to EncodedBatch before processing
        
        output = self.map(batch.encoded_inputs)
        loss = self.loss_func(output, batch.encoded_output_fields)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=batch.encoded_inputs.shape[0]
        )
        return loss

    def configure_optimizers(self):
        """Configure optimizers for training.

        Returns Adam optimizer with learning_rate. Subclasses can override
        to use different optimizers or learning rate schedules.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def _clone_batch(self, batch: EncodedBatch) -> EncodedBatch:
        return EncodedBatch(
            encoded_inputs=batch.encoded_inputs.clone(),
            encoded_output_fields=batch.encoded_output_fields.clone(),
            encoded_info={
                key: value.clone() if hasattr(value, "clone") else value
                for key, value in batch.encoded_info.items()
            },
        )

    def _predict(self, batch: EncodedBatch) -> Tensor:
        return self.map(batch.encoded_inputs)

    def _true_slice(self, batch: EncodedBatch, stride: int) -> tuple[Tensor, bool]:
        if batch.encoded_output_fields.shape[1] >= stride:
            return batch.encoded_output_fields[:, :stride, ...], True
        return batch.encoded_output_fields, False

    def _advance_batch(
        self, batch: EncodedBatch, next_inputs: Tensor, stride: int
    ) -> EncodedBatch:
        next_inputs = torch.cat(
            [batch.encoded_inputs[:, stride:, ...], next_inputs[:, :stride, ...]],
            dim=1,
        )
        next_outputs = (
            batch.encoded_output_fields[:, stride:, ...]
            if batch.encoded_output_fields.shape[1] > stride
            else batch.encoded_output_fields[:, 0:0, ...]
        )
        return EncodedBatch(
            encoded_inputs=next_inputs,
            encoded_output_fields=next_outputs,
            encoded_info=batch.encoded_info,
        )


class FlowMatchingProcessor(Processor):
    """Processor that wraps a flow-matching generative model."""

    def __init__(
        self,
        flow_matching_model: nn.Module,
        *,
        stride: int = 1,
        teacher_forcing_ratio: float = 0.0,
        max_rollout_steps: int = 1,
        loss_func: nn.Module | None = None,
        learning_rate: float = 1e-3,
        flow_ode_steps: int = 1,
        output_shape: tuple[int, ...],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            stride=stride,
            teacher_forcing_ratio=teacher_forcing_ratio,
            max_rollout_steps=max_rollout_steps,
            loss_func=loss_func or nn.MSELoss(),
            **kwargs,
        )
        '''
        Args:
            flow_matching_model (nn.Module): The flow-matching vector field.
            learning_rate (float): Learning rate for the optimizer.
            flow_ode_steps (int): Number of steps to use in the flow ODE integration.
            output_shape (tuple[int, ...]): Shape of the output states (B, T_out, *S, C).
        '''
        self.flow_matching_model = flow_matching_model
        self.learning_rate = learning_rate
        self.flow_ode_steps = flow_ode_steps
        self.output_shape = output_shape

    def forward(self, z: Tensor, t: Tensor, x: Tensor) -> Tensor:
        '''
        The vector field over the tangent space of output states (z)
        conditioned on input states (x) at time (t).

        Args:
            z (Tensor): Current output states of shape (B, T_out, *S, C)
            t (Tensor): Time tensor of shape (B,)
            x (Tensor): Input states of shape (B, T_in, *S, C)
        Returns:
            dz (Tensor): Time derivative of output states of shape (B, T_out, *S, C)
        '''
        return self.flow_matching_model(z, t, x)

    def map(self, x: TensorBMStarL) -> TensorBMStarL:
        '''
        Maps inputs states (x) to output states (z) by integrating the flow ODE.

        Args:
            x (Tensor): Input states of shape (B, T_in, *S, C)
        Returns:
            z (Tensor): Output states of shape (B, T_out, *S, C)
        '''

        # Random noisy output states and interpolant time
        z = torch.randn(self.output_shape, device=x.device, dtype=x.dtype)
        t = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Integrate flow ODE from t=0 to t=1
        dt = torch.tensor(1.0 / max(self.flow_ode_steps, 1), device=x.device, dtype=x.dtype)
        for _ in range(self.flow_ode_steps):
            z = z + dt * self.forward(z, t, x)
            t = t + dt
        return z

    def training_step(self, batch: EncodedBatch, batch_idx: int) -> Tensor:  # noqa: ARG002
        '''
        Performs a single training step.

        Args:
            batch (EncodedBatch): Batch of encoded input and output states
                intput_fields: (B, T_in, *S, C)
                output_fields: (B, T_out, *S, C)
            batch_idx (int): Index of the batch
        Returns:
            loss (Tensor): Computed training loss ()
        '''

        # Shapes 
        input_states = batch.encoded_inputs
        target_states = batch.encoded_output_fields

        # Random noisy initial states and interpolant time
        z0 = torch.randn_like(target_states)
        expand_shape = (target_states.shape[0],) + (1,) * (target_states.ndim - 1)
        t = torch.rand(expand_shape, device=target_states.device, dtype=target_states.dtype)
        zt = (1 - t) * z0 + t * target_states

        # Target velocity, predicted velocity, and loss
        target_velocity = target_states - z0
        v_pred = self.forward(zt, t, input_states)
        flow_loss = torch.mean((v_pred - target_velocity) ** 2)

        # Log and return loss
        batch_size = batch.encoded_inputs.shape[0]
        self.log("train_loss", flow_loss, prog_bar=True, batch_size=batch_size)
        self.log(
            "train_flow_matching_loss",
            flow_loss,
            prog_bar=False,
            batch_size=batch_size,
        )
        return flow_loss

    def validation_step(self, batch: EncodedBatch, batch_idx: int) -> Tensor:
        '''
        Computes the test/validation metrics on the rollout of the processor.

        Args:
            batch (EncodedBatch): Batch of encoded input and output states
                intput_fields: (B, T_in, *S, C)
                output_fields: (B, T_out, *S, C)
            batch_idx (int): Index of the batch
        '''

        # Roll out predictions and compare against available ground-truth rollout slices.
        preds, gts = self.rollout(batch)
        loss = self.loss_func(preds, gts) if gts is not None else torch.tensor(0.0, device=preds.device)

        batch_size = batch.encoded_inputs.shape[0]
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("val_rollout_loss", loss, prog_bar=False, batch_size=batch_size)
        return loss
