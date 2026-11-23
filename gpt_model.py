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
    ffn_type: str = "relu"  # relu, swiglu 테스트 옵션

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
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(features)  # LayerNorm보다 가볍고 RoPE/SwiGLU 계열과 맞춤임.

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
        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)  # Q/K/V를 행렬곱 한 번으로 계산함.
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        # head별 QK-norm으로 attention logit이 커지는 것을 막는 것을 의도임.
        self.q_norm = nn.RMSNorm(self.d_k)
        self.k_norm = nn.RMSNorm(self.d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.w_qkv(x).reshape(B, T, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)

        # cos/sin은 fp32 그대로 써서 회전 계산을 fp32로 함.
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True)
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](
            x, lambda normalized: self.self_attention_block(normalized, cos, sin)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.RMSNorm(features)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cos, sin)
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
            pad_id=None,  # packed 데이터라 PAD 안 씀. lm_head와 공유하므로 padding_idx 뺌.
        )
        blocks = nn.ModuleList([
            DecoderBlock(
                config.d_model,
                MultiHeadAttention(config.d_model, config.n_heads, config.dropout),
                (SwiGLUBlock if config.ffn_type == "swiglu" else FeedFowordBlock)
                (config.d_model, config.d_ff, config.dropout),
                config.dropout,
            )
            for _ in range(config.n_layers)
        ])
        self.decoder = Decoder(config.d_model, blocks)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.tok.weight  # 입력 임베딩과 가중치 공유(weight tying)임.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # 잔차로 더해지는 출력층은 std를 1/sqrt(2*n_layers)만큼 줄임 (GPT-2 방식)임.
        for name, p in self.named_parameters():
            if name.endswith(("w_o.weight", "w_down.weight", "linear_2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must be torch.long (int64)")
        B, T = input_ids.shape
        if B == 0 or T == 0 or T > self.config.seq_len:
            raise ValueError(f"Input shape incompatible with context length {self.config.seq_len}")
        x, (cos, sin) = self.embed(input_ids)  # teammate's embedding.py
        # causal은 SDPA is_causal로 처리해서 mask 안 만듦. cos/sin은 여기서 한 번만 [1,1,T,d]로 맞춤임.
        x = self.decoder(x, cos[None, None], sin[None, None])
        return self.lm_head(x)  # [B,T,V] raw logits, no softmax


def build_language_model(vocab_size: int, seq_len: int, d_model: int = 256,
                         N: int = 6, h: int = 8, dropout: float = 0.1,
                         d_ff: int = 1024, ffn_type: str = "relu") -> GPTModel:
    return GPTModel(GPTConfig(vocab_size, seq_len, d_model, N, h, d_ff, dropout, ffn_type))
