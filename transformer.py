from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 65   #from tokennizer    
    block_size: int = 256     
    n_layer: int = 6          
    n_head: int = 6            
    n_embd: int = 384         