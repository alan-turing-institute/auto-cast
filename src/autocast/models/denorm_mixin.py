from the_well.data.normalization import ZScoreNormalization

from autocast.types.batch import Batch
from autocast.types.types import Tensor


class DenormMixin:
    """
    Mixin class to provide denormalization functionality for models.

    Taken from The Well Trainer.denormalize(), see:
    https://github.com/PolymathicAI/the_well/blob/6cd3c44ef832855a5abae87d555bf0f0f52b1fa7/the_well/benchmark/trainer/training.py#L190
    """

    def denormalize(
        self,
        batch: Batch,
        prediction: Tensor,
        norm: ZScoreNormalization,
        delta=False,
    ) -> tuple[Batch, Tensor]:
        """
        Denormalize the input batch and model prediction.

        Parameters
        ----------
        batch : Batch
            The input batch containing normalized data.
        prediction : Tensor
            The model's prediction on the normalized batch.
        norm : type[ZScoreNormalization]
            The normalization class used for denormalization.
        delta : bool, optional
            Whether to apply delta denormalization. Default is False.

        Returns
        -------
        tuple[Batch, Tensor]
            A tuple containing the denormalized batch and prediction.
        """
        denorm_batch = Batch(
            input_fields=norm.denormalize_flattened(batch.input_fields, "variable"),
            output_fields=batch.output_fields,
            constant_scalars=batch.constant_scalars,
            constant_fields=(
                norm.denormalize_flattened(batch.constant_fields, "constant")
                if batch.constant_fields
                else None
            ),
        )
        if delta:
            denorm_pred = norm.delta_denormalize_flattened(prediction, "variable")
        else:
            denorm_pred = norm.denormalize_flattened(prediction, "variable")

        return denorm_batch, denorm_pred
