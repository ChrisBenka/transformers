import torch
from torch import nn

device = torch.device("mps")


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_encoder_layers, max_seq_len, num_heads, num_ff_hidden_layers,
                 num_ff_hidden_units, device):
        pass

    def forward(self):
        pass
