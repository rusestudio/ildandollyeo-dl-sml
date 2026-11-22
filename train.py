import os
import sys
import math
import time
from ast import literal_eval
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from config import DataConfig, TokenizerConfig, ModelConfig, PretrainConfig

# 팀원 코드
from dataset import load_packed_datasets  #  build_dataloaders
from gpt_model import GPTConfig, GPTModel

import wandb

data_config = DataConfig()
tokenizer_config = TokenizerConfig()
model_config = ModelConfig()
train_config = PretrainConfig()

# 터미널 명령줄에서 설정 덮어쓰기 위함
for arg in sys.argv[1:]:
    key, val = arg.split("=")
    try:
        val = literal_eval(val)
    except (ValueError, SyntaxError):
        pass
    for cfg in (data_config, model_config, train_config):
        if hasattr(cfg, key):
            setattr(cfg, key, val)


# warmup -> cosine decay
def get_lr(it):
    if it < train_config.warmup_steps:
        return train_config.learning_rate * (it + 1) / train_config.warmup_steps
    decay_ratio = (it - train_config.warmup_steps) / (
        train_config.max_steps - train_config.warmup_steps
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (
        train_config.learning_rate - train_config.min_lr
    )


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= train_config.eval_iters:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=train_config.max_new_tokens,
    temperature=train_config.temperature,
    top_k=train_config.top_k,
):
    raw_model.eval()
    idx = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False).ids], device=device
    )
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -gpt_config.seq_len :]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = raw_model(idx_cond)[:, -1, :]
        logits = logits.float() / temperature
        # logits = model(idx_cond)[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        if idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    raw_model.train()
    return tokenizer.decode(idx[0].tolist())


def infinite_batches():
    while True:
        for x, y in train_loader:
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def save_checkpoint(path, step):
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "gpt_config": asdict(gpt_config),
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )


if __name__ == "__main__":
    device = "cuda"

    run_name = train_config.wandb_run_name or (
        f"d{model_config.hidden_size}L{model_config.num_layers}_T{data_config.seq_len}_{train_config.optimizer}_lr{train_config.learning_rate}_{train_config.max_steps}st"
    )
    train_config.out_dir = os.path.join(train_config.out_dir, run_name)
    os.makedirs(train_config.out_dir, exist_ok=True)

    tokens_per_step = (
        train_config.batch_size
        * data_config.seq_len
        * train_config.gradient_accumulation_steps
    )
    print(f"run {run_name}, tokens per step {tokens_per_step:,}")

    if train_config.wandb_log:
        wandb.init(
            project=train_config.wandb_project,
            name=run_name,
            config={
                **asdict(data_config),
                **asdict(model_config),
                **asdict(train_config),
            },
        )

    torch.manual_seed(train_config.random_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_ds, val_ds, meta = load_packed_datasets(
        data_config.processed_dir, seq_len=data_config.seq_len
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=True,
    )
    tokenizer = Tokenizer.from_file(tokenizer_config.tokenizer_path)
    eos_id = meta["eos_id"]

    # model
    gpt_config = GPTConfig(
        vocab_size=meta["vocab_size"],
        seq_len=data_config.seq_len,
        d_model=model_config.hidden_size,
        n_layers=model_config.num_layers,
        n_heads=model_config.num_heads,
        d_ff=model_config.ffn_hidden_size,
        dropout=model_config.dropout,
        pad_id=meta["pad_id"],
        ffn_type=model_config.ffn_type,
        norm_type=model_config.norm_type,
        qk_norm=model_config.qk_norm,
    )
    raw_model = GPTModel(gpt_config).to(device)
    print(
        f"number of parameters: {sum(p.numel() for p in raw_model.parameters()) / 1e6:.2f}M"
    )

    # optimizer
    # Muon : attention /FFN의 2D 행렬만, embedding, lm_head, norm 은 AdamW
    lr0 = train_config.learning_rate
    wd = train_config.weight_decay
    named = list(raw_model.named_parameters())
    is_io = lambda n: n.startswith(("embed", "lm_head."))
    nodecay = [p for n, p in named if p.dim() < 2]

    if train_config.optimizer == "muon":
        hidden = [p for n, p in named if p.dim() == 2 and not is_io(n)]
        io = [p for n, p in named if p.dim() == 2 and is_io(n)]

        optimizers = [
            # match_rms_adamw: AdamW 와 같은 lr 값을 그대로 쓸 수 있게 스케일 맞춤
            torch.optim.Muon(
                hidden, lr=lr0, weight_decay=wd, adjust_lr_fn="match_rms_adamw"
            ),
            torch.optim.AdamW(
                [
                    {"params": io, "weight_decay": wd},
                    {"params": nodecay, "weight_decay": 0.0},
                ],
                lr=lr0,
                betas=train_config.adam_betas,
                fused=True,
            ),
        ]
    else:
        decay = [p for n, p in named if p.dim() >= 2]
        optimizers = [
            torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": wd},
                    {"params": nodecay, "weight_decay": 0.0},
                ],
                lr=lr0,
                betas=train_config.adam_betas,
                fused=True,
            ),
        ]

    start_step = 0
    best_val_loss = 1e9
    if train_config.resume_from_checkpoint:
        ckpt = torch.load(
            os.path.join(train_config.out_dir, "last.pt"), map_location=device
        )
        raw_model.load_state_dict(ckpt["model"])
        for opt, state in zip(optimizers, ckpt["optimizers"]):
            opt.load_state_dict(state)
        start_step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]

    model = torch.compile(raw_model) if train_config.compile else raw_model

    # training loop
    batches = infinite_batches()
    model.train()
    t0 = time.time()
    accum = train_config.gradient_accumulation_steps
    for step in range(start_step, train_config.max_steps + 1):
        last_step = step == train_config.max_steps

        # evaluate + checkpoint + sample
        if step % train_config.eval_steps == 0 or last_step:
            val_loss = estimate_loss()
            print(
                f"step {step}: val loss {val_loss:.4f}, val ppl {math.exp(val_loss):.2f}"
            )
            if train_config.wandb_log:
                wandb.log(
                    {"val/loss": val_loss, "val/ppl": math.exp(val_loss)}, step=step
                )
            if step > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(os.path.join(train_config.out_dir, "best.pt"), step)
                save_checkpoint(os.path.join(train_config.out_dir, "last.pt"), step)
            print(generate(train_config.sample_prompt))

        if last_step:
            break

        lr = get_lr(step)
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group["lr"] = lr

        loss_accum = 0.0
        for micro_step in range(accum):
            x, y = next(batches)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / accum
            loss_accum += loss.item()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), train_config.max_grad_norm
        )
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)

        if step % train_config.log_steps == 0:
            dt = time.time() - t0
            t0 = time.time()
            tok_per_sec = tokens_per_step * train_config.log_steps / dt
            print(
                f"step {step}: loss {loss_accum:.4f}, lr {lr:.2e}, norm {norm:.2f}, "
                f"time {dt * 1000:.0f}ms, {tok_per_sec:,.0f} tok/s"
            )
            if train_config.wandb_log:
                wandb.log(
                    {"train/loss": loss_accum, "lr": lr, "grad_norm": norm.item()},
                    step=step,
                )

    if train_config.wandb_log:
        wandb.finish()
