import os
import math
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import DataConfig, TokenizerConfig, ModelConfig, PretrainConfig

# 팀원 코드
from dataset import build_dataloaders
from gpt_model import GPTConfig, GPTModel

import wandb

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
    if train_config.wandb_log:
        wandb.init(
            entity="ildandollyeo-sml-train",
            project=train_config.wandb_project,
            name=train_config.wandb_run_name or None,
            config={**asdict(model_config), **asdict(train_config)},
        )
        # sweep 값으로 덮어쓰기
        for k, v in wandb.config.items():
            if hasattr(model_config, k):
                setattr(model_config, k, v)
            if hasattr(train_config, k):
                setattr(train_config, k, v)
        wandb.run.name = (
            f"{train_config.optimizer_name}_"
            f"muon{train_config.muon_lr:g}_"
            f"adamw{train_config.muon_adamw_lr:g}_"
            f"{train_config.max_steps}st"
        )
        train_config.out_dir = os.path.join(train_config.out_dir, wandb.run.name)

    torch.manual_seed(train_config.random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(train_config.out_dir, exist_ok=True)

    # data
    train_loader, val_loader, meta = build_dataloaders(
        data_config.processed_dir,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
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
    # optimizer: AdamW or Muon + AdamW
    muon_optimizer = None

    if train_config.optimizer_name == "muon":
        # Muon only for hidden-layer weight matrices.
        muon_params = [
            p for name, p in model.named_parameters()
            if name.startswith("decoder.layers.") and p.ndim == 2
        ]

        muon_ids = {id(p) for p in muon_params}
        adamw_params = [
            p for p in model.parameters()
            if id(p) not in muon_ids
        ]

        if not muon_params:
            raise RuntimeError("No hidden-layer parameters for Muon")

        muon_optimizer = torch.optim.Muon(
            muon_params,
            lr=train_config.muon_lr,
            momentum=train_config.muon_momentum,
            weight_decay=train_config.muon_weight_decay,
        )

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for p in adamw_params if p.ndim >= 2],
                    "weight_decay": train_config.weight_decay,
                },
                {
                    "params": [p for p in adamw_params if p.ndim < 2],
                    "weight_decay": 0.0,
                },
            ],
            lr=train_config.muon_adamw_lr,
            betas=train_config.adam_betas,
        )

        print(f"Muon parameters: {sum(p.numel() for p in muon_params):,}")
        print(f"AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    elif train_config.optimizer_name == "adamw":
        decay_params = [p for p in model.parameters() if p.ndim >= 2]
        nodecay_params = [p for p in model.parameters() if p.ndim < 2]

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": train_config.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=train_config.learning_rate,
            betas=train_config.adam_betas,
        )

    else:
        raise ValueError(f"Unknown optimizer: {train_config.optimizer_name}")

    print(f"Optimizer: {train_config.optimizer_name}")

    start_step = 0
    best_val_loss = 1e9

    def save_checkpoint(filename, completed_steps):
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "muon_optimizer": (
                muon_optimizer.state_dict()
                if muon_optimizer is not None else None
            ),
            "optimizer_name": train_config.optimizer_name,
            "gpt_config": asdict(gpt_config),
            "step": completed_steps,
            "best_val_loss": best_val_loss,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if device == "cuda" else None
            ),
        }
        path = os.path.join(train_config.out_dir, filename)
        temp_path = path + ".tmp"

        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)

    if train_config.resume_from_checkpoint:
        checkpoint = torch.load(
            os.path.join(train_config.out_dir, "latest.pt"), map_location=device
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        saved_optimizer = checkpoint.get("optimizer_name", "adamw")

        if saved_optimizer != train_config.optimizer_name:
            raise RuntimeError("Checkpoint optimizer does not match configuration")

        if muon_optimizer is not None:
            if checkpoint.get("muon_optimizer") is None:
                raise RuntimeError("Missing Muon optimizer state")
            muon_optimizer.load_state_dict(checkpoint["muon_optimizer"])
            best_val_loss = checkpoint["best_val_loss"]
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

            if checkpoint.get("cuda_rng_state_all") is not None:
                torch.cuda.set_rng_state_all([
                    state.cpu()
                    for state in checkpoint["cuda_rng_state_all"]
                ])

    # training loop
    batches = infinite_batches()
    model.train()
    t0 = time.time()
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
            if step > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "gpt_config": asdict(gpt_config),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "optimizer_name": train_config.optimizer_name,
                    "muon_optimizer": (
                        muon_optimizer.state_dict()
                        if muon_optimizer is not None else None
                    ),
                }
            if train_config.save_best_model:
                save_checkpoint("best.pt", step)
            print(generate(train_config.sample_prompt))

        if last_step:
            break

        lr = get_lr(step)
        lr_scale = lr / train_config.learning_rate

        if muon_optimizer is None:
            for group in optimizer.param_groups:
                group["lr"] = lr
        else:
            for group in optimizer.param_groups:
                group["lr"] = train_config.muon_adamw_lr * lr_scale

            for group in muon_optimizer.param_groups:
                group["lr"] = train_config.muon_lr * lr_scale

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

        if muon_optimizer is not None:
            muon_optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if muon_optimizer is not None:
            muon_optimizer.zero_grad(set_to_none=True)

            completed_steps = step + 1

        if completed_steps % train_config.save_steps == 0:
            save_checkpoint("latest.pt", completed_steps)

        if train_config.wandb_log:
            wandb.log(
                {
                    "train/loss": loss_accum,
                    "train/grad_norm": norm.item(),
                    "train/muon_lr": muon_optimizer.param_groups[0]["lr"]
                        if muon_optimizer is not None else 0.0,
                    "train/adamw_lr": optimizer.param_groups[0]["lr"],
                    "train/tokens_processed_est": (
                        (step + 1)
                        * train_config.batch_size
                        * train_config.gradient_accumulation_steps
                        * gpt_config.seq_len
                    ),
                },
                 step=step,
            )

        save_checkpoint("latest.pt", train_config.max_steps)

    if train_config.save_final_model:
        save_checkpoint("final.pt", train_config.max_steps)

    if train_config.wandb_log:
        wandb.finish()
