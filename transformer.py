from dataclasses import dataclass
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Define the model architecture here based on the config parameters
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

@dataclass
class GPTConfig:
    vocab_size: int = 65   #from tokennizer    
    block_size: int = 256     
    n_layer: int = 6          
    n_head: int = 6            
    n_embd: int = 384 

class SelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Liner(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)  

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)   

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

def feedforward(self, idx, target=None):
    B, T = idx.shape
    pos = torch.arange(0,T, device=idx.device)

    tok_emb = self.transformer.wte(idx) 
    pos_emb = self.transformer.wpe(pos) 
    x = tok_emb + pos_emb

    for block in self.transformer.h:
        x = block(x)

    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)

    loss = None
    if target is not None:
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1)
        )
        return logits, loss