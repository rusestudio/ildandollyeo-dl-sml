"""Continue the existing scratch-trained GPT for one additional FULL FineWeb-Edu 10BT epoch.

Run from the project root (next to config.py, dataset.py, gpt_model.py):
    python -u continue_train.py

No new tokenizer, pretrained weights, or dataset download. This script verifies the
full 14-file processed 10BT sample, restores BOTH Muon and AdamW optimizer states,
uses a conservative new LR schedule, evaluates the FULL validation split, and
saves checkpoints into a NEW directory (leaving the original run untouched).

If interrupted, run the same command to continue from this run's latest.pt.
The deterministic sampler resumes at the next UNTRAINED window.
"""

import argparse
import hashlib
import math
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from config import DataConfig, TokenizerConfig, PretrainConfig
from dataset import load_packed_datasets
from gpt_model import GPTConfig, GPTModel


OLD_CHECKPOINT = (
    "checkpoints/muon_muon0.02_adamw0.0003_674245st/final.pt"
)
RUN_NAME = "continue_full10BT_epoch2_muon005_adamw000075"
EXPECTED_FILES = [f"{i:03d}_00000.parquet" for i in range(14)]


class ResumePermutationSampler(Sampler):
    """One deterministic shuffled pass, with direct offset for safe restarts."""

    def __init__(self, size, seed, offset=0):
        self.size = int(size)
        self.seed = int(seed)
        self.offset = int(offset)
        if not 0 <= self.offset <= self.size:
            raise ValueError("Invalid sampler offset")

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        order = torch.randperm(self.size, generator=generator)
        # Avoid a very large .tolist() allocation for ~43 million indices.
        for start in range(self.offset, self.size, 65536):
            yield from order[start:start + 65536].tolist()

    def __len__(self):
        return self.size - self.offset


def full_validation(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            total_loss += F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            ).item()
            total_tokens += y.numel()
    model.train()
    return total_loss / total_tokens


def new_learning_rates(local_step, total, warmup, muon_peak, adamw_peak,
                       start_fraction=0.4, end_fraction=0.2):
    """Warm from 40% to peak, then cosine-decay to 20% of peak."""
    if local_step < warmup:
        t = (local_step + 1) / max(1, warmup)
        factor = start_fraction + (1 - start_fraction) * t
    else:
        t = min(1.0, (local_step - warmup) / max(1, total - warmup - 1))
        factor = end_fraction + (1 - end_fraction) * 0.5 * (1 + math.cos(math.pi * t))
    return muon_peak * factor, adamw_peak * factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=OLD_CHECKPOINT)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--muon-lr", type=float, default=0.005)
    parser.add_argument("--adamw-lr", type=float, default=0.000075)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=25000)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Run this on your GPU training server.")
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)

    cfg = PretrainConfig()
    data_cfg = DataConfig()
    tokenizer_cfg = TokenizerConfig()
    out_dir = Path(cfg.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "latest.pt"
    start_path = latest_path if latest_path.exists() else Path(args.checkpoint)
    if not start_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {start_path}")

    train_ds, val_ds, meta = load_packed_datasets(data_cfg.processed_dir)
    if sorted(meta.get("source_files", [])) != EXPECTED_FILES:
        raise RuntimeError("Data is NOT the full 14-file FineWeb-Edu sample/10BT")
    if meta.get("source") != "HuggingFaceFW/fineweb-edu/sample-10BT":
        raise RuntimeError(f"Unexpected source: {meta.get('source')}")
    if train_ds.num_tokens != meta["train"]["tokens_including_eos"]:
        raise RuntimeError("train.bin token count differs from metadata")
    if val_ds.num_tokens != meta["val"]["tokens_including_eos"]:
        raise RuntimeError("val.bin token count differs from metadata")
    if train_ds.num_tokens < 10_000_000_000:
        raise RuntimeError("Training token stream unexpectedly small")
    tokenizer_sha256 = hashlib.sha256(
        Path(tokenizer_cfg.tokenizer_path).read_bytes()
    ).hexdigest()
    if tokenizer_sha256 != meta["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer hash differs from the one used to pack data")

    # Do not change the model shape when loading the existing weights.
    ckpt = torch.load(start_path, map_location="cpu", weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["gpt_config"])
    if gpt_cfg.vocab_size != meta["vocab_size"] or gpt_cfg.seq_len != meta["seq_len"]:
        raise RuntimeError("Model vocab/context does not match the processed data")
    model = GPTModel(gpt_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    # This parameter partition MUST match the original training script.
    muon_params = [
        p for name, p in model.named_parameters()
        if name.startswith("decoder.layers.") and p.ndim == 2
    ]
    if not muon_params:
        raise RuntimeError("Muon parameter group empty: check GPTModel names")
    muon_ids = {id(p) for p in muon_params}
    adamw_params = [p for p in model.parameters() if id(p) not in muon_ids]
    muon = torch.optim.Muon(
        muon_params, lr=args.muon_lr,
        momentum=cfg.muon_momentum,
        weight_decay=cfg.muon_weight_decay,
    )
    adamw = torch.optim.AdamW(
        [
            {"params": [p for p in adamw_params if p.ndim >= 2],
             "weight_decay": cfg.weight_decay},
            {"params": [p for p in adamw_params if p.ndim < 2],
             "weight_decay": 0.0},
        ],
        lr=args.adamw_lr, betas=tuple(cfg.adam_betas),
    )
    if ckpt.get("optimizer_name") != "muon" or ckpt.get("muon_optimizer") is None:
        raise RuntimeError("Expected a checkpoint containing both Muon and AdamW states")
    adamw.load_state_dict(ckpt["optimizer"])
    muon.load_state_dict(ckpt["muon_optimizer"])

    # Exactly one complete pass over every training window.
    batches_per_epoch = math.ceil(len(train_ds) / cfg.batch_size)
    if batches_per_epoch % cfg.gradient_accumulation_steps:
        raise RuntimeError("Batch count not divisible by gradient accumulation")
    extra_steps = batches_per_epoch // cfg.gradient_accumulation_steps
    first_step = 674245
    if latest_path.exists():
        if ckpt.get("origin_step") != first_step:
            raise RuntimeError("Existing continuation checkpoint has unexpected origin")
        completed_extra = int(ckpt["extra_step"])
        best_val = float(ckpt["best_val_loss"])
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if ckpt.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng_state_all"]])
    else:
        if ckpt["step"] != first_step:
            raise RuntimeError(f"Expected original checkpoint step {first_step}; got {ckpt['step']}")
        completed_extra = 0
        best_val = float("inf")

    if not 0 <= completed_extra <= extra_steps:
        raise RuntimeError("Invalid continuation step count")

    def save_ckpt(filename, done):
        state = {
            "model": model.state_dict(),
            "optimizer": adamw.state_dict(),
            "muon_optimizer": muon.state_dict(),
            "optimizer_name": "muon",
            "gpt_config": asdict(gpt_cfg),
            "step": first_step + done,
            "origin_step": first_step,
            "extra_step": done,
            "best_val_loss": best_val,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        }
        final_path = out_dir / filename
        temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        torch.save(state, temp_path)
        os.replace(temp_path, final_path)

    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    # At any restart, resume at exactly the next untrained window index.
    start_offset = completed_extra * cfg.batch_size * cfg.gradient_accumulation_steps
    sampler = ResumePermutationSampler(len(train_ds), args.seed, start_offset)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    print("=== DATA COVERAGE CHECK ===", flush=True)
    print(f"Full 10BT source files: {len(meta['source_files'])}/14", flush=True)
    print(f"Train docs: {meta['train']['documents_encoded']:,}", flush=True)
    print(f"Train tokens: {train_ds.num_tokens:,}", flush=True)
    print(f"Train windows: {len(train_ds):,}", flush=True)
    print(f"Train batches: {batches_per_epoch:,}", flush=True)
    print(f"Additional steps for full epoch: {extra_steps:,}", flush=True)
    print(f"Continuing from global step: {first_step + completed_extra:,}", flush=True)
    print(f"Remaining windows: {len(sampler):,}", flush=True)
    print(f"Output checkpoints: {out_dir}", flush=True)

    run = None
    if not args.no_wandb:
        import wandb
        run_id_file = out_dir / "wandb_run_id.txt"
        existing_id = run_id_file.read_text().strip() if run_id_file.exists() else None
        run = wandb.init(
            entity="ildandollyeo-sml-train",
            project=cfg.wandb_project,
            name=args.run_name,
            id=existing_id,
            resume="allow" if existing_id else None,
            config={
                "optimizer_name": "muon",
                "origin_step": first_step,
                "additional_steps": extra_steps,
                "train_tokens": train_ds.num_tokens,
                "train_docs": meta["train"]["documents_encoded"],
                "source_files": meta["source_files"],
                "batch_size": cfg.batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "muon_peak_lr": args.muon_lr,
                "adamw_peak_lr": args.adamw_lr,
                "warmup_steps": args.warmup,
                "full_validation_batches": len(val_loader),
            },
        )
        if existing_id is None:
            run_id_file.write_text(run.id)

    model.train()
    if completed_extra == 0:
        baseline_loss = full_validation(model, val_loader, device)
        print(f"Baseline full validation: {baseline_loss:.6f} "
              f"(ppl={math.exp(baseline_loss):.2f})", flush=True)
        best_val = baseline_loss
        save_ckpt("best.pt", 0)
        save_ckpt("latest.pt", 0)
        if run:
            wandb.log({"val/full_loss": baseline_loss,
                       "val/full_ppl": math.exp(baseline_loss)}, step=first_step)

    if completed_extra == extra_steps:
        print("This continuation already completed one full extra epoch.", flush=True)
        if run:
            wandb.finish()
        return

    train_iterator = iter(train_loader)
    for local_step in range(completed_extra, extra_steps):
        muon_lr, adamw_lr = new_learning_rates(
            local_step, extra_steps, args.warmup, args.muon_lr, args.adamw_lr
        )
        for group in muon.param_groups:
            group["lr"] = muon_lr
        for group in adamw.param_groups:
            group["lr"] = adamw_lr

        adamw.zero_grad(set_to_none=True)
        muon.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(cfg.gradient_accumulation_steps):
            x, y = next(train_iterator)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
            ) / cfg.gradient_accumulation_steps
            loss_sum += loss.detach().item()
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm
        )
        adamw.step()
        muon.step()
        done = local_step + 1
        global_step = first_step + done
        metrics = {}
        if done % args.log_every == 0:
            metrics = {
                "train/loss": loss_sum,
                "train/grad_norm": grad_norm.item(),
                "train/muon_lr": muon_lr,
                "train/adamw_lr": adamw_lr,
                "train/tokens_processed_est": (
                    (first_step + done) * cfg.batch_size
                    * cfg.gradient_accumulation_steps * gpt_cfg.seq_len
                ),
            }
            print(f"step {global_step:,} | +{done:,}/{extra_steps:,} | "
                  f"train loss {loss_sum:.4f} | muon_lr {muon_lr:.5g}",
                  flush=True) if done % 1000 == 0 else None

        if done % args.eval_every == 0 or done == extra_steps:
            val_loss = full_validation(model, val_loader, device)
            print(f"step {global_step:,} | FULL val loss {val_loss:.6f} "
                  f"| ppl {math.exp(val_loss):.2f} | "
                  f"best {min(best_val, val_loss):.6f}", flush=True)
            metrics.update({"val/full_loss": val_loss,
                            "val/full_ppl": math.exp(val_loss)})
            if val_loss < best_val:
                best_val = val_loss
                save_ckpt("best.pt", done)

        if done % args.save_every == 0 or done == extra_steps:
            save_ckpt("latest.pt", done)
        if run and metrics:
            wandb.log(metrics, step=global_step)

    save_ckpt("final.pt", extra_steps)
    print(f"COMPLETE: trained across all {len(train_ds):,} windows. "
          f"Best FULL val loss: {best_val:.6f}", flush=True)
    if run:
        wandb.finish()


if __name__ == "__main__":
    main()
