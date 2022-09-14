import torch
from torch import nn
from encoder import Encoder
from decoder import Decoder

dev = torch.device("mps")


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            target_vocab_size,
            max_seq_len=250,
            num_encoder_layers=6,
            num_decoder_layers=6,
            num_attention_heads=8,
            d_model=512,
            encoder_ff_hidden_layers=2,
            decoder_ff_hidden_layers=2,
            ff_hidden_units=2048,
            device=dev
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            num_encoder_layers,
            max_seq_len,
            num_attention_heads,
            encoder_ff_hidden_layers,
            ff_hidden_units,
            device
        )
        self.decoder = Decoder(
            target_vocab_size,
            d_model,
            num_decoder_layers,
            max_seq_len,
            num_attention_heads,
            decoder_ff_hidden_layers,
            ff_hidden_units,
            device
        )

    def forward(self, src, target):
        src_mask = self.src_mask(src)
        target_mask = self.target_mask(target)
        encoded = self.encoder(src_mask, src)
        return encoded
