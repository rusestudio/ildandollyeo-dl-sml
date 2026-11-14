import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# 팀원의 실제 config.py에서 각 데이터클래스 임포트
from config import DataConfig, TokenizerConfig, ModelConfig, PretrainConfig
from dataset import build_dataloaders
from gpt_model import GPTConfig, GPTModel


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model, val_loader, vocab_size, eval_iters, device):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        losses.append(loss.item())

    model.train()
    avg_loss = sum(losses) / max(1, len(losses))
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def main():

    data_cfg = DataConfig()
    tok_cfg = TokenizerConfig()
    model_cfg = ModelConfig()
    pre_cfg = PretrainConfig()

    torch.manual_seed(pre_cfg.random_seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    train_loader, val_loader, meta = build_dataloaders(
        processed_dir=data_cfg.processed_dir,
        batch_size=pre_cfg.batch_size,
        seq_len=data_cfg.seq_len,
        num_workers=pre_cfg.num_workers,
    )

    
    gpt_conf = GPTConfig(
        vocab_size=int(tok_cfg.vocab_size),
        seq_len=int(data_cfg.seq_len),
        d_model=int(model_cfg.hidden_size),
        n_layers=int(model_cfg.num_layers),
        n_heads=int(model_cfg.num_heads),
        d_ff=int(model_cfg.ffn_hidden_size),
        dropout=float(model_cfg.dropout),
        pad_id=meta.get("pad_id", 0),
    )
    model = GPTModel(gpt_conf).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": float(pre_cfg.weight_decay)},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=float(pre_cfg.learning_rate),
        betas=pre_cfg.adam_betas,
    )

    start_step = 0
    best_val_loss = float("inf")
    ckpt_dir = Path(pre_cfg.out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    train_iter = iter(train_loader)
    max_steps = int(pre_cfg.max_steps)
    grad_accum = int(pre_cfg.gradient_accumulation_steps)

    print("pre-training")
    t0 = time.time()

    for step in range(start_step, max_steps):
        lr = get_lr(
            step,
            int(pre_cfg.warmup_steps),
            max_steps,
            float(pre_cfg.learning_rate),
            float(pre_cfg.min_lr),
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, gpt_conf.vocab_size), y.reshape(-1))
            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), float(pre_cfg.max_grad_norm))
        optimizer.step()

        # 평가 및 출력
        if (step + 1) % int(pre_cfg.eval_steps) == 0 or step == 0 or (step + 1) == max_steps:
            val_loss, val_ppl = evaluate(
                model, val_loader, gpt_conf.vocab_size, int(pre_cfg.eval_iters), device
            )
            dt = time.time() - t0
            print(
                f"step {step+1}: val loss {val_loss:.4f} | "
                f"Train Loss: {accum_loss*grad_accum:.4f} | PPL: {val_ppl:.2f} | "
                f"LR: {lr:.2e} | Time: {dt:.1f}s"
            )
            t0 = time.time()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "step": step + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                    },
                    ckpt_dir / "ckpt.pt",
                )

    print("Pre-training end")


if __name__ == "__main__":
    main()