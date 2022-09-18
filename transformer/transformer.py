import torch
from torch import nn
from transformer.encoder import Encoder
from transformer.decoder import Decoder

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            src_pad_idx,
            target_vocab_size,
            trg_pad_idx,
            max_seq_len=250,
            num_encoder_layers=6,
            num_decoder_layers=6,
            num_attention_heads=8,
            d_model=512,
            ff_hidden_units=2048,
            device=torch.device("cpu")
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            d_model,
            num_encoder_layers,
            max_seq_len,
            num_attention_heads,
            ff_hidden_units,
        )
        self.decoder = Decoder(
            target_vocab_size,
            d_model,
            num_decoder_layers,
            max_seq_len,
            num_attention_heads,
            ff_hidden_units
        )
    def make_src_mask(self,src):
       pass
    def make_trg_mask(self,trg):
        pass

    def forward(self, src, target):
        src_mask = self.src_mask(src)
        target_mask = self.target_mask(target)
        encoded = self.encoder(src_mask, src)
        decoded = self.decoder(target,target_mask,encoded)
        return decoded

