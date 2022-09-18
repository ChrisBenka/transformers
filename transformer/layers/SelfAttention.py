import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, k, q, v, mask):
        dim_k = k.size(-1)
        q_k_scaled = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(dim_k)
        if mask:
            q_k_scaled = torch.masked_fill(q_k_scaled,mask == 0, 1e-9)
        attention = F.softmax(q_k_scaled,dim=-1)
        return torch.matmul(attention,v)


if __name__ == '__main__':
    attention = SelfAttention()
    batch = torch.tensor([(1, 2, 3, 4), (4, 5, 6, 7), (7, 8, 9, 10), (7, 8, 9, 10)]).float()
    attention(batch, batch, batch, None)
