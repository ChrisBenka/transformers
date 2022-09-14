"""
Author: ChrisBenka
Date: 2022 - 09 - 13
"""
import math
import torch
from torch import nn, Tensor

'''
PE(pos,2i) = sin(pos / 10000 ^{2i/d_model})
PE(pos,2i+1) = cos(pos / 10000 ^{2i/d_model})

pos is the position and is the dimension

'''

device = torch.device("cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

if __name__ == '__main__':
    pe = PositionalEncoding(5,10)
    print(pe(torch.zeros((10,1,5)).to(device)))
