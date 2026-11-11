import torch
import torch.nn as nn
from model import (MultiHeadAttention, 
                   FeedFowordBlock, 
                   ResConnection, 
                   LayerNormalization , 
                   InputEmb, 
                   PositionalEncoding, )
 
#user model for llm style artchitectire trnasformer

#decoder block
class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedFowordBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Only 2 residual connections (self-attention + feed-forward)
        self.residual_connections = nn.ModuleList([ResConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        # Self-attention with causal mask
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        # Feed-forward
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


#decoder
class Decoder(nn.MOdule):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#gpt style : Input → Embeddings → Positional Encoding → Decoder Stack → Language Head → Logits
class Glm(nn.Module):
    def __init__(self, decoder: DecoderBlock, embed: InputEmb, 
                 pos: PositionalEncoding, lm_head: nn.Linear) -> None:
        super().__init__()
        self.decoder = decoder
        self.embed = embed
        self.pos = pos
        self.lm_head = lm_head  # Projects to vocabulary for next-token prediction

    def forward(self, x, mask=None):
        # Create causal mask if not provided
        if mask is None:
            mask = causal_mask(x.size(1)).to(x.device)
            
        # Embeddings + positional encoding
        x = self.embed(x)  # (batch, seq_len, d_model)
        x = self.pos(x)    # Add positional information
        
        # Through decoder stack
        x = self.decoder(x, mask)  # (batch, seq_len, d_model)
        
        # Project to vocabulary
        logits = self.lm_head(x)   # (batch, seq_len, vocab_size)
        
        return logits

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get logits for next token
                logits = self.forward(generated)[:, -1, :]  # (batch, vocab_size)
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, float('-inf'))
                    logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if we hit end token (you'd define this)
                # if next_token.item() == eos_token_id:
                #     break
                    
        return generated

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.bool)
    return ~mask  # Invert so True = can attend, False = cannot attend

def build_language_model (vocab_size: int, seq_len: int, d_model: int = 512, N: int = 12, h: int = 8, droput: float = 0.1, d_ff: int = 2048) -> glm # need to redefine in config?
    embed = InputEmb(d_model, vocab_size)
    pos = PositionalEncoding(d_model, seq_len, dropout) # kiv 
    
    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttention(d_model, h)
        feed_forward_block = FeedFowordBlock(d_model, d_ff)
        decoder_block = DecoderBlock(d_model, self_attention_block, feed_forward_block)
        decoder_blocks.append(decoder_block)
    
    decoder = DecoderBlock(d_model, nn.ModuleList(decoder_blocks))
    lm_head = nn.Linear(d_model, vocab_size)
    model = Glm(decoder, embed, pos, lm_head)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model


#refer reff!!
# Example usage and comparison
if __name__ == "__main__":
    print("🤖 Language Model (LLM) vs Translation Model")
    print("=" * 60)
    
    # Model parameters
    vocab_size = 10000
    seq_len = 512
    d_model = 512
    
    # Build LLM
    llm = build_language_model(vocab_size, seq_len, d_model, N=6, h=8)
    
    # Example input (batch_size=1, seq_len=10)
    input_ids = torch.randint(0, vocab_size, (1, 10))
    
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = llm(input_ids)
        
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in llm.parameters()):,}")
    
    print("\n🔍 Key Differences:")
    print("Translation Model: Encoder-Decoder, Cross-attention, Fixed I/O")
    print("Language Model: Decoder-only, Self-attention, Autoregressive")
    
    print("\n🎯 Next Steps:")
    print("1. Create training script for language modeling")
    print("2. Use text datasets (not translation pairs)")
    print("3. Implement text generation methods")
    print("4. Add training optimizations (gradient checkpointing, etc.)") 