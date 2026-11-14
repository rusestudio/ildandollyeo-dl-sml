import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):

    def __init__(self, head_dim, max_len=2048, base=10000.0):
        super().__init__()
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))   # (head_dim/2,)
        m = torch.arange(max_len).float()                                           # (max_len,)
        angles = torch.outer(m, theta)                                              # (max_len, head_dim/2)
        angles = torch.cat([angles, angles], dim=-1)                                # (max_len, head_dim)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, T):
        return self.cos[:T], self.sin[:T]                                           # (T, head_dim)


class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size, d_model, n_head, max_len=2048, pad_id=None):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.rope = RotaryEmbedding(d_model // n_head, max_len)

        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)
        if pad_id is not None:
            with torch.no_grad():
                self.tok.weight[pad_id].zero_()

    def forward(self, ids):                       # ids: (B, T)
        x = self.tok(ids)                         # (B, T, d_model)
        cos, sin = self.rope(ids.size(1))         # (T, head_dim)
        return x, (cos, sin)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)                   
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return (x * cos + rotate_half(x) * sin).type_as(x)