import random
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    Dropout,
    Embedding,
    GRU,
    LayerNorm,
    Linear,
    Module,
    Sequential,
)


class PositionwiseFeedforwardLayer(Module):
    def __init__(
            self,
            hidden_dim: int,
            pf_dim: int,
            dropout_p: float,
        ):
        super().__init__()

        self.sequential = Sequential(
            Linear(
                in_features=hidden_dim,
                out_features=pf_dim,
            ),
            ReLU(),
            Dropout(p=dropout_p),
            Linear(
                in_features=pf_dim,
                out_features=hidden_dim,
            )
        )

    def forward(self, x):
        x_1 = self.sequential(x)

        return x_1


class MultiHeadAttentionLayer(Module):
    def __init__(
            self,
            hidden_dim: int,
            heads_num: int,
            dropout_p: float,
            device: torch.device,
        ):
        super().__init__()

        assert hidden_dim % heads_num == 0

        self.hidden_dim = hidden_dim
        self.heads_num = heads_num
        self.head_dim = hidden_dim // heads_num

        self.fc_q = Linear(hidden_dim, hidden_dim)
        self.fc_k = Linear(hidden_dim, hidden_dim)
        self.fc_v = Linear(hidden_dim, hidden_dim)

        self.fc_o = Linear(hidden_dim, hidden_dim)

        self.dropout = Dropout(p=dropout_p)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(
            self,
            query,
            key,
            value,
            mask=None,
        ):
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.heads_num, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.heads_num, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.heads_num, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hidden_dim)
        x = self.fc_o(x)

        return x, attention


class EncoderLayer(Module):
    def __init__(
            self,
            hidden_dim: int,
            heads_num: int,
            pf_dim: int,
            dropout_p: float,
            device: torch.device,
        ):
        super().__init__()

        self.self_attn_layer_norm = LayerNorm(normalized_shape=hidden_dim)
        self.ff_layer_norm = LayerNorm(hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hidden_dim=hidden_dim,
            heads_num=heads_num,
            dropout_p=dropout_p,
            device=device,
        )
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hidden_dim=hidden_dim,
            pf_dim=pf_dim,
            dropout_p=dropout_p,
        )
        self.dropout = Dropout(p=dropout_p)

    def forward(
            self,
            src,
            src_mask,
        ):
        _src, _ = self.self_attention(
            query=src,
            key=src,
            value=src,
            mask=src_mask,
        )

        src = self.self_attn_layer_norm(src + self.dropout(_src))

        _src = self.positionwise_feedforward(src)

        src = self.ff_layer_norm(src + self.dropout(_src))

        return src


class TransformerEncoder(Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            layers_num: int,
            heads_num: int,
            pf_dim: int,
            dropout_p: float,
            device: torch.device,
            max_length: int=100,
        ):
        super().__init__()
        self.device = device

        self.tok_embedding = Embedding(
            num_embeddings=input_dim,
            embedding_dim=hidden_dim,
        )
        self.pos_embedding = Embedding(
            num_embeddings=max_length,
            embedding_dim=hidden_dim,
        )

        self.layers = ModuleList([
            EncoderLayer(
                hidden_dim=hidden_dim,
                heads_num=heads_num,
                pf_dim=pf_dim,
                dropout_p=dropout_p,
                device=device,
            ).to(device) for _ in range(layers_num)
        ])

        self.dropout = Dropout(p=dropout_p)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(
            self,
            src,
            src_mask,
        ):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(
            0,
            src_len,
        ).unsqueeze(0).repeat(
            batch_size,
            1,
        ).to(self.device)

        a = self.tok_embedding(src)
        b = self.pos_embedding(pos)
        src = self.dropout(a * self.scale + b)

        for layer in self.layers:
            src = layer(
                src=src,
                src_mask=src_mask,
            )

        return src


class DecoderLayer(Module):
    def __init__(
            self,
            hidden_dim: int,
            heads_num: int,
            pf_dim: int,
            dropout_p: float,
            device: torch.device,
        ):
        super().__init__()

        self.self_attn_layer_norm = LayerNorm(normalized_shape=hidden_dim)
        self.enc_attn_layer_norm = LayerNorm(normalized_shape=hidden_dim)
        self.ff_layer_norm = LayerNorm(normalized_shape=hidden_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hidden_dim=hidden_dim,
            heads_num=heads_num,
            dropout_p=dropout_p,
            device=device,
        )
        self.encoder_attention = MultiHeadAttentionLayer(
            hidden_dim=hidden_dim,
            heads_num=heads_num,
            dropout_p=dropout_p,
            device=device,
        )
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hidden_dim=hidden_dim,
            pf_dim=pf_dim,
            dropout_p=dropout_p,
        )
        self.dropout = Dropout(p=dropout_p)

    def forward(
            self,
            trg,
            enc_src,
            trg_mask,
            src_mask,
        ):
        _trg, _ = self.self_attention(
            query=trg,
            key=trg,
            value=trg,
            mask=trg_mask,
        )

        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        _trg, attention = self.encoder_attention(
            query=trg,
            key=enc_src,
            value=enc_src,
            mask=src_mask,
        )

        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        _trg = self.positionwise_feedforward(trg)

        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention


class TransformerDecoder(Module):
    def __init__(
            self,
            output_dim: int,
            hidden_dim: int,
            layers_num: int,
            heads_num: int,
            pf_dim: int,
            dropout_p: float,
            device: torch.device,
            max_length: int=100,
        ):
        super().__init__()

        self.device = device

        self.tok_embedding = Embedding(
            num_embeddings=output_dim,
            embedding_dim=hidden_dim,
        )
        self.pos_embedding = Embedding(
            num_embeddings=max_length,
            embedding_dim=hidden_dim,
        )

        self.layers = ModuleList([
            DecoderLayer(
                hidden_dim=hidden_dim,
                heads_num=heads_num,
                pf_dim=pf_dim,
                dropout_p=dropout_p,
                device=device,
            ).to(device) for _ in range(layers_num)
        ])

        self.fc_out = Linear(
            in_features=hidden_dim,
            out_features=output_dim,
        )

        self.dropout = Dropout(p=dropout_p)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(
            0,
            trg_len,
        ).unsqueeze(0).repeat(
            batch_size,
            1,
        ).to(self.device)

        a = self.tok_embedding(trg)
        b = self.pos_embedding(pos)
        trg = self.dropout(a * self.scale + b)

        for layer in self.layers:
            trg, attention = layer(
                trg=trg,
                enc_src=enc_src,
                trg_mask=trg_mask,
                src_mask=src_mask,
            )

        output = self.fc_out(trg)

        return output, attention

