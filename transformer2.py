#positional encoding

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
 
def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)
 
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_seq_len).float()
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())
 
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)
 
sequence = torch.randn(1, 10, 4, 128)
rope = RotaryPositionalEncoding(128)
new_sequence = rope(sequence)


#attention~~
batch_size, seq_len, hidden_dim = x.shape
 
q_proj = nn.Linear(hidden_dim, num_heads * head_dim)
k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim)
v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim)
out_proj = nn.Linear(num_heads * head_dim, hidden_dim)
 
q = q_proj(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
k = k_proj(x).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
v = v_proj(x).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
output = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim).contiguous()
output = out_proj(q)
