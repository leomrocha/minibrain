import numpy as np
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.modules.normalization import LayerNorm
# based on Attention mechanism explained here:
#

from enum import IntEnum


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1)  # get the size of the key
        assert q.size(-1) == d_k

        # compute the dot product between queries and keys for
        # each batch and position in the sequence
        attn = torch.bmm(q, k.transpose(Dim.seq, Dim.feature))  # (Batch, Seq, Seq)
        # we get an attention score between each position in the sequence
        # for each batch

        # scale the dot products by the dimensionality (see the paper for why we do this!)
        attn = attn / np.sqrt(d_k)
        # normalize the weights across the sequence dimension
        # (Note that since we transposed, the sequence and feature dimensions are switched)
        attn = torch.exp(attn)
        # fill attention weights with 0s where padded
        if mask is not None:
            attn = attn.masked_fill(mask, 0)
        attn = attn / attn.sum(-1, keepdim=True)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)  # (Batch, Seq, Feature)
        return output


class AttentionHead(nn.Module):
    """A single attention head"""
    def __init__(self, d_model, d_feature, kernel_size=3, dropout=0.1):
        super().__init__()
        # We will assume the queries, keys, and values all have the same feature size
        self.attn = ScaledDotProductAttention(dropout)
        self.query_tfm = nn.Linear(d_model, d_feature)
        self.key_tfm = nn.Linear(d_model, d_feature)
        self.value_tfm = nn.Linear(d_model, d_feature)
         # self.query_tfm = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        # # nn.Linear(d_model, d_feature)
        # self.key_tfm = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        # self.value_tfm = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, queries, keys, values, mask=None):
        Q = self.query_tfm(queries)  # (Batch, Seq, Feature)
        K = self.key_tfm(keys)  # (Batch, Seq, Feature)
        V = self.value_tfm(values)  # (Batch, Seq, Feature)
        # compute multiple attention weighted sums
        x = self.attn(Q, K, V)
        return x


class MultiHeadAttention(nn.Module):
    """The full multihead attention block"""
    def __init__(self, d_model, d_feature, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_feature = d_feature
        self.n_heads = n_heads
        # in practice, d_model == d_feature * n_heads
        assert d_model == d_feature * n_heads

        # Note that this is very inefficient:
        # I am merely implementing the heads separately because it is
        # easier to understand this way
        self.attn_heads = nn.ModuleList([
            AttentionHead(d_model, d_feature, dropout) for _ in range(n_heads)
        ])
        self.projection = nn.Linear(d_feature * n_heads, d_model)

    def forward(self, queries, keys, values, mask=None):
        # log_size(queries, "Input queries")
        x = [attn(queries, keys, values, mask=mask)  # (Batch, Seq, Feature)
             for i, attn in enumerate(self.attn_heads)]

        # reconcatenate
        x = torch.cat(x, dim=Dim.feature)  # (Batch, Seq, D_Feature * n_heads)
        # log_size(x, "concatenated output")
        x = self.projection(x)  # (Batch, Seq, D_Model)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, d_feature=64,
                 d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.layer_norm1 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.layer_norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        att = self.attn_head(x, x, x, mask=mask)
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm1(att))
        # Apply position-wise feedforward network
        pos = self.position_wise_feed_forward(x)
        # Apply normalization and residual connection
        x = x + self.dropout(self.layer_norm2(pos))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_blocks=6, d_model=512,
                 n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoders = nn.ModuleList([
            EncoderBlock(d_model=d_model, d_feature=d_model // n_heads,
                         d_ff=d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.FloatTensor, mask=None):
        for encoder in self.encoders:
            x = encoder(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, d_feature=64,
                 d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.position_wise_feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

        self.layer_norm1 = LayerNorm(d_model)
        self.layer_norm2 = LayerNorm(d_model)
        self.layer_norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out,
                src_mask=None, tgt_mask=None):
        # Apply attention to inputs
        att = self.masked_attn_head(x, x, x, mask=src_mask)
        x = x + self.dropout(self.layer_norm1(att))
        # Apply attention to the encoder outputs and outputs of the previous layer
        att = self.attn_head(queries=att, keys=x, values=x, mask=tgt_mask)
        x = x + self.dropout(self.layer_norm2(att))
        # Apply position-wise feedforward network
        pos = self.position_wise_feed_forward(x)
        x = x + self.dropout(self.layer_norm3(pos))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, n_blocks=6, d_model=512, d_feature=64,
                 d_ff=2048, n_heads=8, dropout=0.1):
        super().__init__()
        self.position_embedding = PositionalEmbedding(d_model)
        self.decoders = nn.ModuleList([
            DecoderBlock(d_model=d_model, d_feature=d_model // n_heads,
                         d_ff=d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.FloatTensor,
                enc_out: torch.FloatTensor,
                src_mask=None, tgt_mask=None):
        for decoder in self.decoders:
            x = decoder(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(1), :]  # (1, Seq, Feature)

