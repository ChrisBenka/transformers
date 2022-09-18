import torch
from torch import nn
from layers.encoder_block import EncoderBlock
from embeddings.positional_encoding import PositionalEncoding
device = torch.device("mps")


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, num_encoder_layers, max_seq_len, num_heads,num_ff_hidden_units, d_k,d_v, device):
        super(Encoder, self).__init__()
        self.word_embedding = nn.Embedding(src_vocab_size,d_model)
        self.position_embedding = PositionalEncoding(d_model,max_seq_len)
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(d_model,
                          num_heads,
                          num_ff_hidden_units,
                          d_k,
                          d_v,
                          device) for i in range(num_encoder_layers)]
        )

    def forward(self, src, src_mask):
        x = self.word_embedding(src) + self.position_embedding(src)
        for layer in self.encoder_layers:
            x = layer(x, x, x, src_mask)
        return x
