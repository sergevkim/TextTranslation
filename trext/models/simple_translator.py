from typing import Tuple

import einops
import torch
from torch import Tensor
from torch.nn import (
    Dropout,
    Embedding,
    GRU,
    Linear,
    Module,
)


class Encoder(Module):
    def __init__(
            self,
            input_dim: int,
        ):
        super().__init__()

        self.embedding = Embedding()
        self.rnn = GRU()
        self.fc = Linear()
        self.dropout = Dropout()

    def forward(
            self,
            x: Tensor,
        ) -> Tuple[Tensor]:

        embedded_x = self.dropout(self.embedding(x))

        outputs, hidden = self.rnn(embedded_x)

        hidden = torch.tanh()

        return outputs, hidden


class Decoder(Module):
    def __init__(
            self,
            output_dim: int,
            attention: Module,
        ):
        super().__init__()

    def forward(
            self,
            x: Tensor,
            decoder_hidden: Tensor,
            encoder_outputs: Tensor,
        ) -> Tuple[Tensor]:
        pass


class SimpleTranslator(Module):
    def __init__(
            self,
            encoder: Module,
            decoder: Module,
            device: torch.device,
        ):
        super().__init__()

        self.device = device
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            source: Tensor,
            target: Tensor,
        ):
        pass

    def training_step(
            self,
        ):
        pass

