import random
from typing import Tuple

import einops
import torch
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


class Encoder(Module):
    def __init__(
            self,
            input_dim: int,
            embedding_dim: int,
            encoder_hidden_dim: int,
            decoder_hidden_dim: int,
            dropout_p: float,
        ):
        super().__init__()

        self.embedding = Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
        )
        self.rnn = GRU(
            input_size=embedding_dim,
            hidden_size=encoder_hidden_dim,
            bidirectional=True,
        )
        self.fc = Linear(
            in_features=encoder_hidden_dim * 2,
            out_features=decoder_hidden_dim,
        )
        self.dropout = Dropout(
            p=dropout_p,
            inplace=True,
        )

    def forward(
            self,
            x: Tensor,
        ) -> Tuple[Tensor]:

        x_1 = self.embedding(x)
        x_2 = self.dropout(x_1)

        outputs, hidden = self.rnn(x_2)

        hidden_1 = torch.cat(
            tensors=(hidden[-2, :, :], hidden[-1, :, :]),
            dim = 1,
        )
        hidden_2 = self.fc(hidden_1)
        hidden_3 = torch.tanh(hidden_2)

        return outputs, hidden_3


class Attention(Module):
    def __init__(
            self,
            encoder_hidden_dim: int,
            decoder_hidden_dim: int,
        ):
        super().__init__()

        self.attn = Linear(
            in_features=encoder_hidden_dim * 2 + decoder_hidden_dim,
            out_features=decoder_hidden_dim,
        )
        self.v = Linear(
            in_features=decoder_hidden_dim,
            out_features=1,
            bias=False,
        )

    def forward(
            self,
            hidden: Tensor,
            encoder_outputs: Tensor,
        ) -> Tensor:
        source_len, batch_size, _ = encoder_outputs.shape
        hidden = hidden.unsqueeze(1).repeat(1, source_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        cats = torch.cat(
            (hidden, encoder_outputs),
            dim=2,
        )
        energy = torch.tanh(self.attn(cats))
        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(Module):
    def __init__(
            self,
            output_dim: int,
            embedding_dim: int,
            encoder_hidden_dim: int,
            decoder_hidden_dim: int,
            dropout_p: float,
            attention: Module,
        ):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.embedding = Embedding(
            num_embeddings=output_dim,
            embedding_dim=embedding_dim,
        )
        self.rnn = GRU(
            input_size=encoder_hidden_dim * 2 + embedding_dim,
            hidden_size=decoder_hidden_dim,
        )
        self.fc = Linear(
            in_features=\
                encoder_hidden_dim * 2 + embedding_dim + decoder_hidden_dim,
            out_features=output_dim,
        )
        self.dropout = Dropout(
            p=dropout_p,
            inplace=True,
        )

    def forward(
            self,
            x: Tensor,
            decoder_hidden: Tensor,
            encoder_outputs: Tensor,
        ) -> Tuple[Tensor]:
        x_1 = x.unsqueeze(0)
        x_2 = self.embedding(x_1)
        x_3 = self.dropout(x_2)

        a = self.attention(
            hidden=decoder_hidden,
            encoder_outputs=encoder_outputs,
        )
        a = a.unsqueeze(1)

        #TODO
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat(
            (x_3, weighted),
            dim=2,
        )

        output, hidden = self.rnn(
            rnn_input,
            decoder_hidden.unsqueeze(0),
        )

        assert (output == hidden).all()

        x_4 = x_3.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        cats = torch.cat(
            (output, weighted, x_4),
            dim=1,
        )
        prediction = self.fc(cats)

        return prediction, hidden.squeeze(0)


class AttentionTranslator(Module):
    def __init__(
            self,
            encoder: Module,
            decoder: Module,
            teacher_forcing_ratio: float,
            learning_rate: float,
            device: torch.device,
        ):
        super().__init__()

        self.device = device
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.criterion = CrossEntropyLoss(ignore_index=0)

        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self,
            sources: Tensor,
            targets: Tensor,
        ):
        target_len, batch_size = targets.shape
        target_vocabulary_size = self.decoder.output_dim
        outputs = torch.zeros((
            target_len,
            batch_size,
            target_vocabulary_size,
        )).to(self.device)

        encoder_outputs, hidden = self.encoder(sources)
        decoder_input = targets[0, :]

        for t in range(1, target_len):
            output, hidden = self.decoder(
                x=decoder_input,
                decoder_hidden=hidden,
                encoder_outputs=encoder_outputs,
            )

            outputs[t] = output
            top_1 = output.argmax(1)

            #teacher_force = random.random() < self.teacher_forcing_ratio
            #if teacher_force:
            #    decoder_input = targets[t]
            #else:
            decoder_input = top_1

        return outputs

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ):
        de_tags, en_tags = batch
        de_tags = de_tags.permute(1, 0).to(self.device)
        en_tags = en_tags.permute(1, 0).to(self.device)

        pred_en_tags = self(
            sources=de_tags,
            targets=en_tags,
        )
        output_dim = pred_en_tags.shape[-1]

        pred_en_tags_2 = pred_en_tags[1:].view(-1, output_dim)

        en_tags_2 = en_tags[1:].contiguous().view(-1) #TODO remove for GPU

        loss = self.criterion(
            input=pred_en_tags_2,
            target=en_tags_2,
        )

        return loss

    def training_step_end(self):
        pass

    def training_epoch_end(self):
        pass

    def validation_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        pass

    def validation_step_end(self):
        pass

    def validation_epoch_end(self):
        pass

    def test_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        de_tags, en_tags = batch
        de_tags = de_tags.permute(1, 0).to(self.device)
        en_tags = en_tags.permute(1, 0).to(self.device)

        pred_en_tags = self(
            sources=de_tags,
            targets=en_tags,
        )

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

