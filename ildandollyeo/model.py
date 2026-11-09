import torch
import torch.nn as nn
import torch.nn.functional as F


from dataclasses import dataclass


# Configuration
@dataclass
class configuration:
    vocab_size: int = 65
    max_seq_len: int = 256
    n_layers: int = 6  # 트랜스포머 블럭 개수
    n_heads: int = 6  # 어텐션 헤드 개수
    d_model: int = 384  # 임베딩 차원

    # vocab size 는 tokenizer 에서 결정됨
    # max seq len 는 모델이 한 번에 확인할 수 있는 최대 토큰 수를 의미하
    # d_model 는 토큰 하나를 나타내는 벡터의 길이 (모델의 폭) 모든 hiddne state는 이 크기의 벡터로 구성됨


### Embeddings
# token_embedding word token embedding maps each token ID to a learned vector. size : 65. 384
# position_embedding word position embeddign maps each position (0~255) to a learned vector : 256, 384
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                token_embedding=nn.Embedding(
                    config.vocab_size, config.d_model
                ),  # token embeddings
                position_embedding=nn.Embedding(
                    config.max_seq_len, config.d_model
                ),  # position embeddings
                h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f=nn.LayerNorm(config.d_model),
            )
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.transformer.token_embedding.weight = self.lm_head.weight

    ### Forward Pass
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)

        tok_emb = self.transformer.token_embedding(idx)  # B, T , d_model
        pos_emb = self.transformer.position_embedding(pos)  # T, d_model
        x = tok_emb + pos_emb  # B, T, d_model - broadcasting adds position information

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # B, T, vocab_size

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


### Self-Attention
# This is the mechanism that lets each token attend to look ar every previous token in the sequence


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.c_attn = nn.Linear(
            config.d_model, 3 * config.d_model
        )  #  Q, K, V projections
        self.down_proj = nn.Linear(config.d_model, config.d_model)  # ouptut projection
        self.n_heads = config.n_heads
        self.d_model = config.d_model

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # reshape for multi-head
        head_dim = d_model // self.n_heads
        q = q.view(batch, seq_len, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, head_dim).transpose(1, 2)

        # attention with causal mask (each token can only attend to previous tokens)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.down_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = nn.GELU(approximate="tanh")
        self.down_proj = nn.Linear(4 * config.d_model, config.d_model)

    def forward(self, x):
        x = self.up_proj(x)
        x = self.gelu(x)
        return self.down_proj(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


config = configuration()
model = GPT(config)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params / 1e6:.1f}M")  # ~10.8M

# 데이터 안 넣으면 forward 실행 안됨

x = torch.randint(0, config.vocab_size, (2, 16))  # 토큰 번호 2문장 x  16개
y = torch.randint(0, config.vocab_size, (2, 16))  # 정답 자리
logits, loss = model(x, y)
print(logits.shape, loss.item())  # torch.size([2, 16, 65]) 4.27~
# 2문장 마다 16위치마다, 65개 토큰에 대한 점수
