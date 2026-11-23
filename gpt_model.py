from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F
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
    ffn_type: str = "relu"  # relu, swiglu 테스트 옵션
    norm_type: str = "layernorm"  # layernorm, rmsnorm
    qk_norm: bool = False  # attention q, k 정규화


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()  # 학습은 bf16인데, norm 은 fp32로 계산
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def make_norm(norm_type: str, dim: int) -> nn.Module:
    return RMSNorm(dim) if norm_type == "rmsnorm" else nn.LayerNorm(dim)

class FeedFowordBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class SwiGLUBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))

class ResConnection(nn.Module):
    def __init__(self, features: int, dropout: float, norm_type: str = "layernorm"):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = make_norm(norm_type, features)

    def forward(self, x: torch.Tensor, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float, qk_norm: bool = False):
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
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.d_k)
            self.k_norm = RMSNorm(self.d_k)

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

        if self.qk_norm:
            q = self.q_norm(q)  # head 차원(d_k) 기준 정규화, rope 전에
            k = self.k_norm(k)

        # embedding.py supplies real RoPE functions. Q/K must be rotated
        # AFTER reshaping into heads but BEFORE the scaled-dot-product.
        cos = cos[None, None, :, :].to(dtype=q.dtype)
        sin = sin[None, None, :, :].to(dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.w_o(out)


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedFowordBlock, dropout: float,
                 norm_type: str = "layernorm"):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResConnection(features, dropout, norm_type) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](
            x, lambda normalized: self.self_attention_block(normalized, cos, sin, mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList, norm_type: str = "layernorm"):
        super().__init__()
        self.layers = layers
        self.norm = make_norm(norm_type, features)

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
                MultiHeadAttention(config.d_model, config.n_heads, config.dropout, config.qk_norm),
                (SwiGLUBlock if config.ffn_type == "swiglu" else FeedFowordBlock)
                (config.d_model, config.d_ff, config.dropout),
                config.dropout,
                config.norm_type,
            )
            for _ in range(config.n_layers)
        ])
        self.decoder = Decoder(config.d_model, blocks, config.norm_type)
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
