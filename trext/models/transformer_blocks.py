import random
from typing import Tuple

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


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            pf_dim: int,
            dropout_p: float,
        ):
        super().__init__()

        self.fc_1 = nn.Linear(hidden_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hidden_dim)

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)

        return x


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

        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

        self.fc_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(p=dropout_p)

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

        attention = torch.softmax(energy, dim = -1)

        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hidden_dim)
        x = self.fc_o(x)

        return x, attention


class EncoderLayer(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            heads_num: int,
            pf_dim: int,
            dropout_p: float,
            device: torch.device,
        ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
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
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
            self,
            src,
            src_mask,
        ):
        _src, _ = self.self_attention(
            src,
            src,
            src,
            src_mask,
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

        self.tok_embedding = nn.Embedding(
            input_dim,
            hidden_dim,
        )
        self.pos_embedding = nn.Embedding(
            max_length,
            hidden_dim,
        )

        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_dim=hidden_dim,
                heads_num=heads_num,
                pf_dim=pf_dim,
                dropout_p=dropout_p,
                device=device,
            ).to(device) for _ in range(layers_num)
        ])

        self.dropout = nn.Dropout(p=dropout_p)

        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(
            self,
            src,
            src_mask,
        ):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        a = self.tok_embedding(src)
        b = self.pos_embedding(pos)
        src = self.dropout(a * self.scale + b)

        for layer in self.layers:
            src = layer(src, src_mask)

        return src


class DecoderLayer(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            heads_num: int,
            pf_dim: int,
            dropout_p: float,
            device: torch.device,
        ):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hidden_dim)
        self.ff_layer_norm = nn.LayerNorm(hidden_dim)
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
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
            self,
            trg,
            enc_src,
            trg_mask,
            src_mask,
        ):
        _trg, _ = self.self_attention(
            trg,
            trg,
            trg,
            trg_mask,
        )

        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))

        _trg, attention = self.encoder_attention(
            trg,
            enc_src,
            enc_src,
            src_mask,
        )

        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))

        _trg = self.positionwise_feedforward(trg)

        trg = self.ff_layer_norm(trg + self.dropout(_trg))

        return trg, attention


class TransformerDecoder(nn.Module):
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

        self.tok_embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_embedding = nn.Embedding(max_length, hidden_dim)

        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_dim=hidden_dim,
                heads_num=heads_num,
                pf_dim=pf_dim,
                dropout_p=dropout_p,
                device=device,
            ).to(device) for _ in range(layers_num)
        ])

        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(p=dropout_p)

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

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)

        return output, attention

