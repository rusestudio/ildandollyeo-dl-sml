from datasets import load_dataset
import torch


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
