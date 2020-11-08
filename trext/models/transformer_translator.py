import random
from typing import Tuple

#import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Dropout,
    Embedding,
    GRU,
    Linear,
    Module,
    CrossEntropyLoss,
)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

from .transformer_blocks import (
    TransformerDecoder,
    TransformerEncoder,
)


class TransformerTranslator(Module):
    def __init__(
            self,
            src_pad_idx, #TODO
            trg_pad_idx, #TODO
            learning_rate: float,
            input_dim: int,
            output_dim: int,
            hidden_dim: int,
            encoder_dropout_p: float,
            encoder_heads_num: int,
            encoder_layers_num: int,
            encoder_pf_dim: int,
            decoder_dropout_p: float,
            decoder_heads_num: int,
            decoder_layers_num: int,
            decoder_pf_dim: int,
            device: torch.device,
        ):
        super().__init__()

        self.device = device
        self.learning_rate = learning_rate
        self.criterion = CrossEntropyLoss(ignore_index=trg_pad_idx)

        self.encoder = TransformerEncoder(#TODO
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            layers_num=encoder_layers_num,
            heads_num=encoder_heads_num,
            pf_dim=encoder_pf_dim,
            dropout_p=encoder_dropout_p,
            device=device,
        )

        self.decoder = TransformerDecoder( #TODO
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            layers_num=decoder_layers_num,
            heads_num=decoder_heads_num,
            pf_dim=decoder_pf_dim,
            dropout_p=decoder_dropout_p,
            device=device,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(
            self,
            src: Tensor,
        ):
        src_mask = (
            src != self.src_pad_idx
        ).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(
            self,
            trg: Tensor,
        ):
        trg_pad_mask = (
            trg != self.trg_pad_idx
        ).unsqueeze(1).unsqueeze(2).to(self.device)

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(
            input=torch.ones(
                size=(trg_len, trg_len),
                device=self.device,
            ),
        ).bool()

        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(
            self,
            src: Tensor,
            trg: Tensor,
        ):
        src_mask = self.make_src_mask(src=src)
        trg_mask = self.make_trg_mask(trg=trg)

        enc_src = self.encoder(
            src,
            src_mask,
        )

        output, attention = self.decoder(
            trg,
            enc_src,
            trg_mask,
            src_mask,
        )

        return output, attention

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ):
        de_tags = batch.src.permute(1, 0).to(self.device)
        en_tags = batch.trg.permute(1, 0).to(self.device)

        pred_en_tags, _ = self(
            src=de_tags,
            trg=en_tags[:, :-1],
        )
        output_dim = pred_en_tags.shape[-1]

        pred_en_tags_2 = pred_en_tags.contiguous().view(-1, output_dim)

        en_tags_2 = en_tags[:, 1:].contiguous().view(-1)

        loss = self.criterion(
            input=pred_en_tags_2,
            target=en_tags_2,
        )

        return loss

    def training_step_end(self):
        pass

    def training_epoch_end(self, epoch_idx):
        print(f"Training epoch #{epoch_idx} is over.")

    def validation_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        de_tags = batch.src.permute(1, 0).to(self.device)
        en_tags = batch.trg.permute(1, 0).to(self.device)

        pred_en_tags, _ = self(
            src=de_tags,
            trg=en_tags[:, :-1],
        )
        output_dim = pred_en_tags.shape[-1]

        pred_en_tags_2 = pred_en_tags.contiguous().view(-1, output_dim)

        en_tags_2 = en_tags[:, 1:].contiguous().view(-1)

        loss = self.criterion(
            input=pred_en_tags_2,
            target=en_tags_2,
        )

        return loss

    def validation_step_end(self):
        pass

    def validation_epoch_end(self, epoch_idx):
        print(f"Validation epoch #{epoch_idx} is over.")

    def test_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        de_tags = batch.src.permute(1, 0).to(self.device)
        en_tags = batch.trg.permute(1, 0).to(self.device)

        pred_en_tags, _ = self(
            src=de_tags,
            trg=en_tags,
        )
        output_dim = pred_en_tags.shape[-1]

        pred_en_tags_2 = pred_en_tags.contiguous().view(-1, output_dim)

        return pred_en_tags

    def test_step_end(self):
        pass

    def test_epoch_end(self):
        pass

    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )

        return optimizer

