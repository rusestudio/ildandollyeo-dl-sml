import os
import time
from dataclasses import asdict
from sklearn.metrics import roc_auc_score

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
bos_id = tokenizer.token_to_id("[BOS]")
pad_id = tokenizer.token_to_id("[PAD]")
choice_ids = [
    tokenizer.encode(o, add_special_tokens=False).ids[0] for o in ft_config.answer_options
]
choice_tensor = torch.tensor(choice_ids, device=device)                                     # 4지 선다끼리 비교

checkpoint = torch.load(ft_config.pretrained_path, map_location=device)
gpt_config = GPTConfig(**{**checkpoint["gpt_config"], "dropout": ft_config.dropout})
model = GPTModel(gpt_config).to(device)
model.load_state_dict(checkpoint["model"])
del checkpoint


def encode(example):
    prompt = [bos_id] + tokenizer.encode(
        ft_config.prompt_template.format(**example), add_special_tokens=False
    ).ids
    target = choice_ids[example["answer"]]
    return {"input_ids": prompt + [target], "labels": [-100] * len(prompt) + [target]}


ds = load_dataset("parquet", data_files=ft_config.data_path, split="train")
ds = ds.map(encode, remove_columns=ds.column_names)
num_total = len(ds)
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


def get_lr(it):
    if it < warmup_steps:
        return ft_config.learning_rate * (it + 1) / warmup_steps
    progress = (it - warmup_steps) / (max_steps - warmup_steps)
    return ft_config.learning_rate + progress * (ft_config.min_lr - ft_config.learning_rate)


def answer_loss(logits, y, reduction="mean"):
    mask = y != -100
    answer_logits, target = logits[mask].float(), y[mask]
    if ft_config.choice_loss:
        answer_logits = answer_logits[:, choice_tensor]
        target = (target[:, None] == choice_tensor).int().argmax(-1)
    return F.cross_entropy(answer_logits, target, reduction=reduction)

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss, probs, targets = 0.0, [], []
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
        mask = y != -100
        answer_logits = logits[mask].float()
        target = y[mask]
        total_loss += F.cross_entropy(answer_logits, target, reduction="sum").item()
        probs.append(answer_logits[:, choice_ids].softmax(-1).cpu())
        targets.append(torch.tensor([choice_ids.index(t) for t in target.tolist()]))
    model.train()
    probs, targets = torch.cat(probs), torch.cat(targets)
    acc = (probs.argmax(-1) == targets).float().mean().item()
    auc = roc_auc_score(targets.numpy(), probs.numpy(), multi_class="ovr")
    return total_loss / len(targets), acc, auc


def infinite_batches():
    while True:
        for x, y in train_loader:
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


batches = infinite_batches()
best_val_acc = -1.0
model.train()
t0 = time.time()
for step in range(max_steps + 1):
    last_step = step == max_steps

    if step % ft_config.eval_steps == 0 or last_step:
        val_loss, val_acc, val_auc = evaluate()
        print(f"step {step}: val loss {val_loss:.4f}, val acc {val_acc:.4f}, val auc {val_auc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model": model.state_dict(),
                "gpt_config": asdict(gpt_config),
                "step": step,
                "best_val_acc": best_val_acc,
            }
            print(f"saving checkpoint to {ft_config.out_dir}")
            torch.save(checkpoint, os.path.join(ft_config.out_dir, "ckpt.pt"))

    if last_step:
        break

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    loss_accum = torch.zeros((), device=device)
    train_correct, train_total = 0, 0
    for micro_step in range(ft_config.gradient_accumulation_steps):
        x, y = next(batches)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
        loss = answer_loss(logits, y) / ft_config.gradient_accumulation_steps
        loss_accum += loss.detach()
        loss.backward()
        mask = y != -100
        pred = torch.tensor(choice_ids, device=device)[logits.detach()[mask][:, choice_ids].argmax(-1)]
        train_correct += (pred == y[mask]).sum().item()
        train_total += mask.sum().item()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ft_config.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if step % ft_config.log_steps == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(
            f"step {step}: loss {loss_accum.item():.4f}, train acc {train_correct / train_total:.4f}, "
            f"lr {lr:.2e}, norm {norm:.2f}, time {dt * 1000:.0f}ms"
        )
