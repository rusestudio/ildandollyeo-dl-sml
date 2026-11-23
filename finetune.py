import os
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from config import TokenizerConfig, FinetuneConfig
from gpt_model import GPTConfig, GPTModel

tokenizer_config = TokenizerConfig()
ft_config = FinetuneConfig()

torch.manual_seed(ft_config.random_seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs(ft_config.out_dir, exist_ok=True)

tokenizer = Tokenizer.from_file(tokenizer_config.tokenizer_path)
eos_id = tokenizer.token_to_id("[EOS]")
pad_id = tokenizer.token_to_id("[PAD]")

# model
checkpoint = torch.load(ft_config.pretrained_path, map_location=device)
gpt_config = GPTConfig(**{**checkpoint["gpt_config"], "dropout": ft_config.dropout})
model = GPTModel(gpt_config).to(device)
model.load_state_dict(checkpoint["model"])
del checkpoint


# data
def encode(example):
    prompt = tokenizer.encode(
        ft_config.prompt_template.format(**example), add_special_tokens=False
    ).ids
    response = tokenizer.encode(
        example[ft_config.output_column], add_special_tokens=False
    ).ids + [eos_id]
    # 출력 부분만 학습하기 위해 입력 부분 label은 -100으로 가림.
    return {"input_ids": prompt + response, "labels": [-100] * len(prompt) + response}


ds = load_dataset("parquet", data_files=ft_config.data_path, split="train")
ds = ds.map(encode, remove_columns=ds.column_names)
num_total = len(ds)
# context보다 긴 샘플은 잘리면 EOS가 사라지기 때문에 제외함.
ds = ds.filter(lambda e: len(e["input_ids"]) <= gpt_config.seq_len + 1)
print(f"samples: {len(ds)} / {num_total} (too long dropped)")
split = ds.train_test_split(test_size=ft_config.val_size, seed=ft_config.split_seed)
train_ds, val_ds = split["train"], split["test"]


def collate(batch):
    x = [torch.tensor(b["input_ids"][:-1]) for b in batch]
    y = [torch.tensor(b["labels"][1:]) for b in batch]
    return (
        pad_sequence(x, batch_first=True, padding_value=pad_id),
        pad_sequence(y, batch_first=True, padding_value=-100),
    )


train_loader = DataLoader(train_ds, batch_size=ft_config.batch_size, shuffle=True,
                          collate_fn=collate, drop_last=True, pin_memory=(device == "cuda"))
val_loader = DataLoader(val_ds, batch_size=ft_config.batch_size, shuffle=False,
                        collate_fn=collate)
max_steps = len(train_loader) * ft_config.num_epochs // ft_config.gradient_accumulation_steps
warmup_steps = int(max_steps * ft_config.warmup_ratio)
print(f"train: {len(train_ds)}, val: {len(val_ds)}, max_steps: {max_steps}")

# optimizer (train.py와 동일하게 임베딩, 1D 파라미터는 decay 제외)
named = list(model.named_parameters())
decay_params = [p for n, p in named if p.dim() >= 2 and not n.startswith("embed")]
nodecay_params = [p for n, p in named if p.dim() < 2 or n.startswith("embed")]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": ft_config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    lr=ft_config.learning_rate,
    betas=ft_config.adam_betas,
    fused=(device == "cuda"),
)
# batch마다 길이가 달라 재컴파일이 잦기 때문에 torch.compile은 안 씀.


# warmup -> min_lr까지 선형 감소
def get_lr(it):
    if it < warmup_steps:
        return ft_config.learning_rate * (it + 1) / warmup_steps
    progress = (it - warmup_steps) / (max_steps - warmup_steps)
    return ft_config.learning_rate + progress * (ft_config.min_lr - ft_config.learning_rate)


@torch.no_grad()
def estimate_loss():
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
        total_loss += F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=-100, reduction="sum",
        ).item()
        total_tokens += (y != -100).sum().item()
    model.train()
    return total_loss / total_tokens


@torch.no_grad()
def generate(ids):
    model.eval()
    idx = torch.tensor([ids], device=device)
    for _ in range(ft_config.max_new_tokens):
        logits = model(idx[:, -gpt_config.seq_len :])[:, -1, :] / ft_config.temperature
        v, _ = torch.topk(logits, ft_config.top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        if idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return tokenizer.decode(idx[0, len(ids) :].tolist())


# 학습 중 출력 확인용으로 val 첫 샘플의 입력 부분만 씀.
sample = val_ds[0]
sample_ids = sample["input_ids"][: sum(l == -100 for l in sample["labels"])]
sample_answer = tokenizer.decode([t for t in sample["labels"] if t != -100])


def infinite_batches():
    while True:
        for x, y in train_loader:
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# training loop
batches = infinite_batches()
best_val_loss = 1e9
model.train()
t0 = time.time()
for step in range(max_steps + 1):
    last_step = step == max_steps

    if step % ft_config.eval_steps == 0 or last_step:
        val_loss = estimate_loss()
        print(f"step {step}: val loss {val_loss:.4f}")
        # 과적합 시점 이후를 피하기 위해 val loss가 가장 낮은 모델만 저장함.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model": model.state_dict(),
                "gpt_config": asdict(gpt_config),
                "step": step,
                "best_val_loss": best_val_loss,
            }
            print(f"saving checkpoint to {ft_config.out_dir}")
            torch.save(checkpoint, os.path.join(ft_config.out_dir, "ckpt.pt"))
        print(f"[input] {tokenizer.decode(sample_ids)}")
        print(f"[pred] {generate(sample_ids)}")
        print(f"[answer] {sample_answer}")

    if last_step:
        break

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    loss_accum = torch.zeros((), device=device)
    for micro_step in range(ft_config.gradient_accumulation_steps):
        x, y = next(batches)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100
        ) / ft_config.gradient_accumulation_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ft_config.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if step % ft_config.log_steps == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(
            f"step {step}: loss {loss_accum.item():.4f}, lr {lr:.2e}, norm {norm:.2f}, "
            f"time {dt * 1000:.0f}ms"
        )
