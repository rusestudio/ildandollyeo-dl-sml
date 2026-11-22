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

data_config = DataConfig()
tokenizer_config = TokenizerConfig()
model_config = ModelConfig()
train_config = PretrainConfig()

torch.manual_seed(train_config.random_seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs(train_config.out_dir, exist_ok=True)

# data
train_loader, val_loader, meta = build_dataloaders(
    data_config.processed_dir,
    batch_size=train_config.batch_size,
    seq_len=data_config.seq_len,  # meta 값 대신 config 값 씀. bin 다시 안 만들고 길이 바꿀 수 있음.
    num_workers=train_config.num_workers,
    pin_memory=(device == "cuda"),  # GPU 전송을 비동기로 하기 위함임.
)
tokens_per_step = (
    train_config.batch_size * train_config.gradient_accumulation_steps * data_config.seq_len
)
print(
    f"train tokens: {len(train_loader.dataset) * data_config.seq_len / 1e9:.2f}B, "
    f"planned: {tokens_per_step * train_config.max_steps / 1e9:.2f}B"
)  # planned가 train tokens보다 크면 데이터를 여러 번 도는 것임.
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
    ffn_type=model_config.ffn_type,
    dropout=model_config.dropout,
)
model = GPTModel(gpt_config).to(device)
print(f"number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# optimizer (2D 이상 파라미터만 weight decay)
# 임베딩(lm_head와 공유)은 decay에서 빼서 임베딩 크기가 줄어드는 것을 막는 것을 의도임.
named = list(model.named_parameters())
decay_params = [p for n, p in named if p.dim() >= 2 and not n.startswith("embed")]
nodecay_params = [p for n, p in named if p.dim() < 2 or n.startswith("embed")]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ],
    lr=train_config.learning_rate,
    betas=train_config.adam_betas,
    fused=(device == "cuda"),
)

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

raw_model = model
if train_config.compile and device == "cuda":
    model = torch.compile(model)

# warmup -> stable -> 마지막 decay_ratio 구간 선형 감소 (WSD)
# 마감이 당겨지면 max_steps만 줄여 resume해서 감소 구간을 앞당기는 것을 의도임.
def get_lr(it):
    if it < train_config.warmup_steps:
        return train_config.learning_rate * (it + 1) / train_config.warmup_steps
    decay_start = int(train_config.max_steps * (1 - train_config.decay_ratio))
    if it < decay_start:
        return train_config.learning_rate
    progress = (it - decay_start) / (train_config.max_steps - decay_start)
    return train_config.learning_rate + progress * (
        train_config.min_lr - train_config.learning_rate
    )


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= train_config.eval_iters:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
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
        logits = raw_model(idx_cond)[:, -1, :] / temperature
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
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# training loop
batches = infinite_batches()
model.train()
t0 = time.time()
for step in range(start_step, train_config.max_steps + 1):
    last_step = step == train_config.max_steps

    # evaluate + checkpoint + sample
    if step % train_config.eval_steps == 0 or last_step:
        val_loss = estimate_loss()
        print(f"step {step}: val loss {val_loss:.4f}, val ppl {math.exp(val_loss):.2f}")
        # 중간에 끊겨도 이어서 학습할 수 있게 eval마다 저장함.
        if step > 0:
            best_val_loss = min(best_val_loss, val_loss)
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "gpt_config": asdict(gpt_config),
                "step": step,
                "best_val_loss": best_val_loss,
            }
            print(f"saving checkpoint to {train_config.out_dir}")
            torch.save(checkpoint, os.path.join(train_config.out_dir, "ckpt.pt"))
        print(generate(train_config.sample_prompt))

    if last_step:
        break

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    loss_accum = torch.zeros((), device=device)
    for micro_step in range(train_config.gradient_accumulation_steps):
        x, y = next(batches)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
        logits = logits.float()  # softmax 정밀도 위해 loss는 fp32로 계산함.
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        # z-loss로 출력 logit이 커지는 것을 막는 것을 의도임.
        z = train_config.z_loss * torch.logsumexp(logits, dim=-1).pow(2).mean()
        loss = (ce + z) / train_config.gradient_accumulation_steps
        # 로그는 z-loss 뺀 ce만. .item() 안 불러서 매 micro-step GPU 동기화 없앰.
        loss_accum += ce.detach() / train_config.gradient_accumulation_steps
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
            f"step {step}: loss {loss_accum.item():.4f}, lr {lr:.2e}, norm {norm:.2f}, "
            f"time {dt * 1000:.0f}ms, tok/s {tokens_per_step * train_config.log_steps / dt:,.0f}"
        )  # tok/s로 남은 시간 계산함.
