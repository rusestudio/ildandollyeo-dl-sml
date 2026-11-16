# best 체크포인트로 val / test 전체 평가
import os
import sys
import json
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PackedTokenDataset
from gpt_model import GPTConfig, GPTModel

ckpt_dir = sys.argv[1]
meta = json.load(open("data/processed/meta.json"))
checkpoint = torch.load(os.path.join(ckpt_dir, "ckpt.pt"), map_location="cuda")
model = GPTModel(GPTConfig(**checkpoint["gpt_config"])).cuda()
model.load_state_dict(checkpoint["model"])
model.eval()


@torch.no_grad()
def evaluate(split):
    ds = PackedTokenDataset(f"data/processed/{split}.bin", meta["seq_len"], meta["dtype"])
    losses = []
    for x, y in DataLoader(ds, batch_size=32):
        x, y = x.cuda(), y.cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    return sum(losses) / len(losses)


val_loss = evaluate("val")
test_loss = evaluate("test")
print(f"checkpoint: {ckpt_dir} (step {checkpoint['step']})")
print(f"val loss {val_loss:.4f}, val ppl {math.exp(val_loss):.2f}  (val 전체)")
print(f"test loss {test_loss:.4f}, test ppl {math.exp(test_loss):.2f}")
