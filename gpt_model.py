from dataclasses import dataclass
import math

import torch
from torch import nn

from embedding import EmbeddingLayer, apply_rope


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 16_000 # 정현님 please remove n combine in full config 
    seq_len: int = 256
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    pad_id: int | None = 0

class FeedFowordBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class ResConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(features)

    def forward(self, x: torch.Tensor, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        if d_model % h:
            raise ValueError("d_model must be divisible by h")
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        if self.d_k % 2:
            raise ValueError("RoPE requires even-sized attention heads")
        # Keep the original separate, bias-free Q/K/V/output projections.
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        if C != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}; received {C}")
        if cos.shape != (T, self.d_k) or sin.shape != (T, self.d_k):
            raise ValueError("Invalid RoPE tables: expected (seq_len, head_dim)")
        q = self.w_q(x).reshape(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(x).reshape(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.w_v(x).reshape(B, T, self.h, self.d_k).transpose(1, 2)

        # embedding.py supplies real RoPE functions. Q/K must be rotated
        # AFTER reshaping into heads but BEFORE the scaled-dot-product.
        cos = cos[None, None, :, :].to(dtype=q.dtype)
        sin = sin[None, None, :, :].to(dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # True = permitted attention. No token may attend to future positions.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        out = attention @ v
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.w_o(out)


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedFowordBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResConnection(features, dropout) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](
            x, lambda normalized: self.self_attention_block(normalized, cos, sin, mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(features)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x)


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embed = EmbeddingLayer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_head=config.n_heads,
            max_len=config.seq_len,
            pad_id=config.pad_id,
        )
        blocks = nn.ModuleList([
            DecoderBlock(
                config.d_model,
                MultiHeadAttention(config.d_model, config.n_heads, config.dropout),
                FeedFowordBlock(config.d_model, config.d_ff, config.dropout),
                config.dropout,
            )
            for _ in range(config.n_layers)
        ])
        self.decoder = Decoder(config.d_model, blocks)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Scratch-only initialization for this model's new Linear weights;
        # existing EmbeddingLayer initializes its own token weights with std=.02.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be torch.long (int64)")
        B, T = input_ids.shape
        if B == 0 or T == 0 or T > self.config.seq_len:
            raise ValueError(f"Input shape incompatible with context length {self.config.seq_len}")
        x, (cos, sin) = self.embed(input_ids)  # teammate's embedding.py
        mask = torch.ones((T, T), device=input_ids.device, dtype=torch.bool).tril()
        mask = mask[None, None, :, :]  # [1,1,T,T] broadcasts over B and heads
        x = self.decoder(x, cos, sin, mask)
        return self.lm_head(x)  # [B,T,V] raw logits, no softmax


def build_language_model(vocab_size: int, seq_len: int, d_model: int = 256,
                         N: int = 6, h: int = 8, dropout: float = 0.1,
                         d_ff: int = 1024, pad_id: int | None = 0) -> GPTModel:
    return GPTModel(GPTConfig(vocab_size, seq_len, d_model, N, h, d_ff, dropout, pad_id))
