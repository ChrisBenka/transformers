import torch
from torch import nn
from transformer.layers.MultiHeadAttention import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, d_model, heads, ff_hidden_units):
        super(EncoderBlock, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(d_model,heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, ff_hidden_units),
            nn.ReLU(),
            nn.Linear(ff_hidden_units, d_model)
        )

    def forward(self, k, v, q, src_mask):
        attention = self.multiHeadAttention(k, v, q, src_mask)
        norm_attention_q = self.norm1(attention + q)
        forward = self.fc(norm_attention_q)
        out = self.norm2(norm_attention_q + forward)
        return out
