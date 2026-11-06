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
