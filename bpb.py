import math

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from gpt_model import GPTConfig, GPTModel

device = "cuda"
ckpt = torch.load(
    "checkpoints/d768L12_lr0.0006_do0_wd0.1_674244st/ckpt.pt", map_location=device
)
config = GPTConfig(**ckpt["gpt_config"])
model = GPTModel(config).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")
tokens = np.memmap("data/processed/val.bin", dtype=np.uint16, mode="r")

seq_len = config.seq_len
batch_size = 64
num_windows = (len(tokens) - 1) // seq_len

total_loss = 0.0  # nats
total_tokens = 0
total_bytes = 0

with torch.no_grad():
    for start in range(0, num_windows, batch_size):
        rows = range(start, min(start + batch_size, num_windows))
        x = np.stack(
            [tokens[i * seq_len : i * seq_len + seq_len] for i in rows]
        ).astype(np.int64)
        y = np.stack(
            [tokens[i * seq_len + 1 : i * seq_len + seq_len + 1] for i in rows]
        ).astype(np.int64)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum"
        )

        total_loss += loss.item()
        total_tokens += y.numel()
        for row in y.tolist():
            total_bytes += len(
                tokenizer.decode(row, skip_special_tokens=True).encode("utf-8")
            )

loss_per_token = total_loss / total_tokens
bpb = total_loss / math.log(2) / total_bytes

print(f"val tokens: {total_tokens:,}, bytes: {total_bytes:,}")
print(f"bytes per token: {total_bytes / total_tokens:.3f}")
print(f"val loss: {loss_per_token:.4f}, ppl: {math.exp(loss_per_token):.2f}")
print(f"bits-per-byte: {bpb:.4f}")

# data/processed/val.bin 을 그대로 쓰고, 같은 토큰을 tokenzier로 다시 문자열로 바꿔서 바이트 수를 셈
# Byte-level BPE는 원문 복원이 정확해서 이 방법이 정확함
