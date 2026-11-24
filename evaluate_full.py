
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DataConfig
from dataset import load_packed_datasets
from gpt_model import GPTConfig, GPTModel

checkpoint_path = (
    "checkpoints/"
    "muon_muon0.02_adamw0.0003_674245st/"
    "final.pt"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False,
)

model_config = GPTConfig(**checkpoint["gpt_config"])
model = GPTModel(model_config).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

_, val_dataset, meta = load_packed_datasets(
    DataConfig().processed_dir
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
)

total_loss = 0.0
total_tokens = 0

with torch.inference_mode():
    for i, (x, y) in enumerate(val_loader):
        x = x.to(device)
        y = y.to(device)

        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
        else:
            logits = model(x)

        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )

        total_loss += loss.item()
        total_tokens += y.numel()

        if (i + 1) % 100 == 0:
            print(f"Evaluated {i + 1}/{len(val_loader)} batches")

val_loss = total_loss / total_tokens

print("\n=== FULL VALIDATION RESULTS ===")
print(f"Checkpoint step: {checkpoint['step']}")
print(f"Validation tokens: {total_tokens:,}")
print(f"Validation loss: {val_loss:.6f}")
print(f"Validation perplexity: {math.exp(val_loss):.4f}")
