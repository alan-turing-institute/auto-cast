from the_well.data.normalization import ZScoreNormalization

from autocast.types.batch import Batch
from autocast.types.types import Tensor


class DenormMixin:
    """
    Mixin class to provide denormalization functionality for models.

    Taken from The Well Trainer.denormalize(), see:
    https://github.com/PolymathicAI/the_well/blob/6cd3c44ef832855a5abae87d555bf0f0f52b1fa7/the_well/benchmark/trainer/training.py#L190
    """

    def denormalize_batch(
        self,
        batch: Batch,
        norm: ZScoreNormalization,
    ) -> Batch:
        """
        Denormalize the input batch.

        Parameters
        ----------
        batch : Batch
            The input batch containing normalized data.
        norm : type[ZScoreNormalization]
            The normalization class used for denormalization.

        Returns
        -------
        Batch
            The denormalized batch.
        """
        return Batch(
            input_fields=norm.denormalize_flattened(batch.input_fields, "variable"),
            output_fields=batch.output_fields,
            constant_scalars=batch.constant_scalars,
            constant_fields=(
                norm.denormalize_flattened(batch.constant_fields, "constant")
                if batch.constant_fields
                else None
            ),
        )

    def denormalize_tensor(
        self,
        tensor: Tensor,
        norm: ZScoreNormalization,
        delta=False,
    ) -> Tensor:
        """
        Denormalize a tensor (e.g., a prediction).

        Parameters
        ----------
        tensor : Tensor
            The normalized tensor to be denormalized.
        norm : type[ZScoreNormalization]
            The normalization class used for denormalization.
        delta : bool, optional
            Whether to apply delta denormalization. Default is False.

        Returns
        -------
        Tensor
            The denormalized tensor.
        """
        if delta:
            denorm_tensor = norm.delta_denormalize_flattened(tensor, "variable")
        else:
            denorm_tensor = norm.denormalize_flattened(tensor, "variable")

        return denorm_tensor
