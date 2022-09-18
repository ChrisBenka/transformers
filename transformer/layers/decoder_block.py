import torch
from torch import nn
from layers.MultiHeadAttention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, ff_hidden_units, device):
        super(DecoderBlock, self).__init__()
        self.multiHeadAttention1 = MultiHeadAttention(d_model,heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.multiHeadAttention2 = MultiHeadAttention(d_model,heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, ff_hidden_units),
            nn.ReLU(),
            nn.Linear(ff_hidden_units, d_model)
        )

    def forward(self, k, q, v, target_mask, encoder_out):
        attention = self.multiHeadAttention1(k, q, v, target_mask)
        attention_1_norm_q = self.norm1(attention + q)
        attention_2 = self.multiHeadAttention2(encoder_out, encoder_out, attention_1_norm_q)
        attention_2_norm_q = self.norm2(attention_2 + attention_1_norm_q)
        forward = self.fc(attention_2_norm_q)
        x = self.norm3(attention_2_norm_q + forward)
        return x
