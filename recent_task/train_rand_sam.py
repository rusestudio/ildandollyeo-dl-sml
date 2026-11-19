import os
import math
import time
import numpy as np
import random
from dataclasses import asdict

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import DataConfig, TokenizerConfig, ModelConfig, PretrainConfig

# 팀원 코드
from dataset import build_dataloaders
from gpt_model import GPTConfig, GPTModel

import wandb

MY_WANDB_ENTITY = "<wandb아이디>"
MY_WANDB_PROJECT = "<프로젝트명>"
MY_OUT_DIR = "checkpoints/my_server_test"

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

data_config = DataConfig()
tokenizer_config = TokenizerConfig()
model_config = ModelConfig()
train_config = PretrainConfig()


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
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
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
    model.eval()
    idx = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False).ids], device=device
    )
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -gpt_config.seq_len :]
        logits = model(idx_cond)[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        if idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return tokenizer.decode(idx[0].tolist())


def infinite_batches():
    while True:
        for x, y in train_loader:
            yield x.to(device), y.to(device)


if __name__ == "__main__":

    base_out_dir = MY_OUT_DIR
    wandb_project = MY_WANDB_PROJECT
    wandb_entity = MY_WANDB_ENTITY
    run_id = None

    if getattr(train_config, "wandb_log", False):
        default_run_name = train_config.wandb_run_name or (
            f"my_{train_config.learning_rate}_do{model_config.dropout}"
            f"_wd{train_config.weight_decay}_{train_config.max_steps}st"
        )
        candidate_ckpt = os.path.join(base_out_dir, default_run_name, "last_ckpt.pt")
        if train_config.resume_from_checkpoint and os.path.exists(candidate_ckpt):
            pre_ckpt = torch.load(candidate_ckpt, map_location="cpu", weights_only=False)
            run_id = pre_ckpt.get("wandb_run_id")
            del pre_ckpt

        init_kwargs = dict(
            project=wandb_project,
            name=default_run_name,
            id=run_id,
            resume="allow" if run_id else None,
            config={**asdict(model_config), **asdict(train_config)},
        )
        if wandb_entity:
            init_kwargs["entity"] = wandb_entity

        wandb.init(**init_kwargs)

        if getattr(wandb.run, "sweep-id", None):
          for k, v in wandb.config.items():
              if hasattr(model_config, k):
                  setattr(model_config, k, v)
              if hasattr(train_config, k):
                  setattr(train_config, k, v)
        else:
          wandb.config.update(
              {**asdict(model_config), **asdict(train_config)},
              allow_val_change=True
          )

        if not train_config.wandb_run_name:
            wandb.run.name = (
                f"my_{train_config.learning_rate}_do{model_config.dropout}"
                f"_wd{train_config.weight_decay}_{train_config.max_steps}st"
            )
        train_config.out_dir = os.path.join(base_out_dir, wandb.run.name)
    else:
        train_config.out_dir = base_out_dir

    set_seed(train_config.random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(train_config.out_dir, exist_ok=True)
    ckpt_path = os.path.join(train_config.out_dir, "ckpt.pt")

    start_step = 0
    best_val_loss = 1e9
    checkpoint = None

    if train_config.resume_from_checkpoint and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        start_step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]

    samples_per_step = train_config.batch_size * train_config.gradient_accumulation_steps
    consumed_samples = start_step * samples_per_step

    # data
    train_loader, val_loader, meta = build_dataloaders(
        data_config.processed_dir,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        pin_memory=(device == "cuda"),
        seed=train_config.random_seed,
        consumed_samples=consumed_samples
    )
    tokenizer = Tokenizer.from_file(tokenizer_config.tokenizer_path)
    eos_id = meta["eos_id"]

    # model
    gpt_config = GPTConfig(
        vocab_size=meta["vocab_size"],
        seq_len=meta["seq_len"],
        d_model=model_config.hidden_size,
        n_layers=model_config.num_layers,
        n_heads=model_config.num_heads,
        d_ff=model_config.ffn_hidden_size,
        dropout=model_config.dropout,
        pad_id=meta["pad_id"],
    )
    model = GPTModel(gpt_config).to(device)
    print(
        f"number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    print(
        f"number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    # optimizer
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=train_config.learning_rate,
        betas=train_config.adam_betas,
    )

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            if rng_state.get("torch") is not None:
                torch.set_rng_state(rng_state["torch"].cpu())
            if rng_state.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() for s in rng_state["cuda"]])
            if rng_state.get("numpy") is not None:
                np.random.set_state(rng_state["numpy"])
            if rng_state.get("python") is not None:
                random.setstate(rng_state["python"])

        del checkpoint
        if device == "cuda":
            torch.cuda.empty_cache()

    start_step = 0
    best_val_loss = 1e9
    if train_config.resume_from_checkpoint:
        checkpoint = torch.load(
            os.path.join(train_config.out_dir, "ckpt.pt"), map_location=device
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]

    # training loop
    batches = infinite_batches()
    model.train()
    t0 = time.time()
    for step in range(start_step, train_config.max_steps + 1):
        last_step = step == train_config.max_steps

        # evaluate + checkpoint + sample
        # 현재 config 를 수정하고 실행하여도 반영이 되지 않아서 하드코딩 함 주의
        # 바로 아래 주석 부분
        # train_config.eval_steps = 100
        if step % train_config.eval_steps == 0 or last_step:
            val_loss = estimate_loss()
            print(
                f"step {step}: val loss {val_loss:.4f}, val ppl {math.exp(val_loss):.2f}"
            )
            wandb.log(
                {"val/loss": val_loss, "val/ppl": math.exp(val_loss)}, step=step
            )
            if step > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "gpt_config": asdict(gpt_config),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "rng_state" : {
                        "torch" : torch.get_rng_state(),
                        "cuda" : (
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available()
                            else None
                        ),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                    },
                }
                print(f"saving checkpoint to {train_config.out_dir}")
                torch.save(checkpoint, os.path.join(train_config.out_dir, "ckpt.pt"))
            print(generate(train_config.sample_prompt))

        if last_step:
            break

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        loss_accum = 0.0
        for micro_step in range(train_config.gradient_accumulation_steps):
            x, y = next(batches)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / train_config.gradient_accumulation_steps
            loss_accum += loss.item()
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.max_grad_norm
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % train_config.log_steps == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"step {step}: loss {loss_accum:.4f}, lr {lr:.2e}, norm {norm:.2f}, time {dt * 1000:.0f}ms"
            )
            if train_config.wandb_log:
                wandb.log(
                    {"train/loss": loss_accum, "lr": lr, "grad_norm": norm.item()},
                    step=step,
                )
    if getattr(train_config, "wandb_log" , False):
        wandb.finish()
