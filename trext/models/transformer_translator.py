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


class TransformerTranslator(Module):
    def __init__(
            self,
            encoder: Module,
            decoder: Module,
            source_pad_idx, #TODO
            target_pad_idx, #TODO
            learning_rate: float,
            device: torch.device,
        ):
        super().__init__()

        self.device = device
        self.learning_rate = learning_rate
        self.criterion = CrossEntropyLoss(ignore_index=0)

        self.encoder = encoder
        self.decoder = decoder
        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx

    def make_source_mask(
            self,
            source: Tensor,
        ):
        source_mask = (
            source != self.source_pad_idx
        ).unsqueeze(1).unsqueeze(2)

        return source_mask

    def make_target_mask(
            self,
            target: Tensor,
        ):
        target_pad_mask = (
            target != self.target_pad_idx
        ).unsqueeze(1).unsqueeze(2).to(self.device)

        target_len = target.shape[1]

        target_sub_mask = torch.tril(
            input=torch.ones(
                size=(target_len, target_len),
                device=self.device,
            ),
        ).bool()

        target_mask = target_pad_mask & target_sub_mask

        return target_mask

    def forward(
            self,
            source: Tensor,
            target: Tensor,
        ):
        source_mask = self.make_source_mask(source=source)
        target_mask = self.make_target_mask(target=target)

        enc_source = self.encoder(
            source,
            source_mask,
        )

        output, attention = self.decoder(
            target,
            enc_source,
            target_mask,
            source_mask,
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
            source=de_tags,
            target=en_tags[:, :-1],
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
            source=de_tags,
            target=en_tags[:, :-1],
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
        de_tags, en_tags = batch
        de_tags = de_tags.permute(1, 0).to(self.device)

        pred_en_tags = self(
            sources=de_tags,
            targets=en_tags,
        )

        output_dim = pred_en_tags.shape[-1]
        pred_en_tags_2 = pred_en_tags[1:].view(-1, output_dim)
        en_tags_2 = en_tags[1:].contiguous().view(-1) #TODO remove for GPU

        return pred_en_tags_2

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

