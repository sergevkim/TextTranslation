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

        outputs, hidden = self.rnn(embedded_x

        hidden = torch.tanh()

        return outputs, hidden


class Attention(Module):
    def __init__(
            self,
        ):
        super().__init__()

    def forward(
            self,
            decoder_hidden: Tensor,
            encoder_outputs: Tensor,
        ) -> Tensor:

        energy = torch.tanh()

        attention = torch.sum(energy, dim=2)

        return torch.nn.functional.softmax(attention, dim=1)


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


class AttentionTranslator(Module):
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

    def training_step(
            self,
        ):

