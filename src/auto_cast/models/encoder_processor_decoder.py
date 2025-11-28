import lightning as L

from auto_cast.models.encoder_decoder import EncoderDecoder
from auto_cast.processors.base import Processor
from auto_cast.types import Batch, Tensor


class EncoderProcessorDecoder(L.LightningModule):
    """Encoder-Processor-Decoder Model."""

    encoder_decoder: EncoderDecoder
    processor: Processor

    def __init__(self): ...

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder_decoder.decoder(
            self.processor(self.encoder_decoder.encoder(x))
        )

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor: ...

    def configure_optmizers(self): ...
