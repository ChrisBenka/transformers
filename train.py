import torch
from transformer import Transformer

device =  torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

if __name__ == '__main__':
    src_pad_idx,trg_pad_idx = 0,0
    src_vocab_size,trgt_vocab_size = 10,10
    model = Transformer(
        src_vocab_size,
        src_pad_idx,
        trgt_vocab_size,
        trg_pad_idx,
        d_model=1,
        num_attention_heads=1,
        device=device
    ).to(device)
