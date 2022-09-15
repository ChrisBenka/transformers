import torch
from torch import nn
from layers.encoder_block import EncoderBlock

device = torch.device("mps")


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_encoder_layers, max_seq_len, num_heads,num_ff_hidden_units, device):
        super(Encoder, self).__init__()
        self.word_embedding = None
        self.position_embedding = None
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(d_model,
                          num_heads,
                          num_ff_hidden_units,
                          device) for i in range(num_encoder_layers)]
        )

    def forward(self, src, src_mask):
        x = self.word_embedding(src) + self.position_embedding(src)
        for layer in self.encoder_layers:
            x = layer(x, x, x, src_mask)
        return x
