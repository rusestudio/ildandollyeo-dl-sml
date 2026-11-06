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

@dataclass
class GPTConfig:
    vocab_size: int = 65   #from tokennizer    
    block_size: int = 256     
    n_layer: int = 6          
    n_head: int = 6            
    n_embd: int = 384         

