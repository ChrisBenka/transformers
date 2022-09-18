import torch
from torch import nn
from layers.decoder_block import DecoderBlock
from embeddings.positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, d_model, num_layers, max_seq_len, num_heads,
                 num_ff_hidden_units, device):
        self.word_embedding = nn.Embedding(target_vocab_size, d_model)
        self.position_embedding = PositionalEncoding(d_model, max_seq_len)
        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(d_model,
                          num_heads,
                          num_ff_hidden_units,
                          device) for i in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, target_vocab_size)

    def forward(self, target, target_mask, encoded):
        x = self.word_embedding(target) + self.position_embedding(target)
        for layer in self.decoder_layers:
            x = layer(x, x, x, target_mask, encoded)
        out = self.fc_out(x)
        return out
