import numpy as np
import torch
from torch import nn
from transformer.layers.SelfAttention import SelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert (self.d_model % num_heads == 0), "Dimension of model must be divisible by num heads"

        self.features = d_model // num_heads

        self.w_q = nn.Linear(d_model, num_heads * self.features, bias=False)
        self.w_k = nn.Linear(d_model, num_heads * self.features, bias=False)
        self.w_v = nn.Linear(d_model, num_heads * self.features, bias=False)
        self.w_o = nn.Linear(num_heads * self.features,d_model)

        self.attention = SelfAttention()

    def forward(self, k, q, v, mask):
        batch_sz = q.size(0)
        num_tokens_q, num_tokens_k, num_tokens_v = q.size(1), k.size(1), v.size(1)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        q_head_split = q.reshape(batch_sz, num_tokens_q, self.num_heads, self.features)
        k_head_split = k.reshape(batch_sz, num_tokens_k, self.num_heads, self.features)
        v_head_split = v.reshape(batch_sz, num_tokens_v, self.num_heads, self.features)
        out = self.attention(q_head_split, k_head_split, v_head_split,mask)
        out = out.reshape(batch_sz,num_tokens_q,self.num_heads * self.features)
        out = self.w_o(out)
        return out


if __name__ == '__main__':
    d_model = 4
    attention = MultiHeadAttention(d_model, 2)
    batch = torch.tensor([[(1, 2, 3, 4), (4, 5, 6, 7), (7, 8, 9, 10), (7, 8, 9, 10)]]).float()
    attention(batch, batch, batch, None)
