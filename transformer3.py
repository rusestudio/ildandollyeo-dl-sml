from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F


DATASET_NAME = 'HuggingFaceFW/fineweb-edu'
DATASET_CONFIG = 'sample-10BT'


def document_stream(limit, min_chars=100):
    """Fresh streaming iterator for each pass; no pretrained tokenizer/model."""
    dataset = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split='train', streaming=True)
    count = 0
    for example in dataset:
        text = example.get('text') or ''
        if len(text.strip()) < min_chars:
            continue
        yield text
        count += 1
        if limit is not None and count >= limit:
            break

#emdb
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_length):
        super().__init__()
        assert head_dim % 2 == 0
        frequencies = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(max_length).float(), frequencies)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, x):
        # x: [batch, heads, sequence, head_dim]
        seq = x.size(-2)
        a, b = x[..., 0::2], x[..., 1::2]
        cos = self.cos[:seq].to(dtype=x.dtype)
        sin = self.sin[:seq].to(dtype=x.dtype)
        return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)


#attentionnnn
class GroupedQueryAttention(nn.Module):
    def __init__(self, width, query_heads, kv_heads, dropout):
        super().__init__()
        assert width % query_heads == 0 and query_heads % kv_heads == 0
        self.q_heads, self.kv_heads = query_heads, kv_heads
        self.head_dim = width // query_heads
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(width, kv_heads * self.head_dim, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.dropout = dropout

    def forward(self, x, rope):
        batch, seq, width = x.shape
        q = self.q(x).reshape(batch, seq, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = rope(q), rope(k)
        # Repeat KV heads explicitly for compatibility with CPU, CUDA and Apple MPS.
        repeats = self.q_heads // self.kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                          dropout_p=self.dropout if self.training else 0.0)
        return self.output(attended.transpose(1, 2).contiguous().reshape(batch, seq, width))


#glu
class SwiGLU(nn.Module):
    def __init__(self, width, inner):
        super().__init__()
        self.gate = nn.Linear(width, inner, bias=False)
        self.up = nn.Linear(width, inner, bias=False)
        self.down = nn.Linear(inner, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


#moe
class MixtureOfExperts(nn.Module):
    def __init__(self, width, experts, top_k):
        super().__init__()
        assert 1 <= top_k <= experts
        self.router = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(width, 4 * width) for _ in range(experts)])
        self.top_k = top_k

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        scores, indices = self.router(flat).topk(self.top_k, dim=-1)
        weights = scores.softmax(dim=-1)
        result = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(indices == expert_id)
            if token_idx.numel():
                values = expert(flat[token_idx]) * weights[token_idx, slot_idx, None]
                result.index_add_(0, token_idx, values)
        return result.reshape(shape)


#rms
class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight

#dec
class DecoderBlock(nn.Module):
    def __init__(self, width, heads, kv_heads, experts, top_k, dropout):
        super().__init__()
        self.norm_attn = RMSNorm(width)
        self.attn = GroupedQueryAttention(width, heads, kv_heads, dropout)
        self.norm_ff = RMSNorm(width)
        self.ff = MixtureOfExperts(width, experts, top_k)

    def forward(self, x, rope):
        x = x + self.attn(self.norm_attn(x), rope)
        return x + self.ff(self.norm_ff(x))

#transfblock
class ScratchTransformer(nn.Module):
    def __init__(self, vocab_size, width=128, layers=2, heads=4, kv_heads=2,
                 experts=4, top_k=2, context=128, dropout=0.1):
        super().__init__()
        self.context = context
        self.embedding = nn.Embedding(vocab_size, width)
        self.rope = RotaryEmbedding(width // heads, context)
        self.blocks = nn.ModuleList([DecoderBlock(width, heads, kv_heads, experts, top_k, dropout)
                                     for _ in range(layers)])
        self.norm = RMSNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, ids):
        if ids.size(1) > self.context:
            raise ValueError('Input exceeds context window')
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x, self.rope)
        return self.lm_head(self.norm(x))