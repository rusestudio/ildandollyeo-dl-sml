
import torch
import torch.nn.functional as F

from dataset import build_dataloaders
from gpt_model import GPTConfig, GPTModel


BATCH_SIZE = 4 # 정현님 please remove n combine in full config 
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
D_FF = 1024
DROPOUT = 0.1


def test_causal_mask(model, device):
    was_training = model.training
    model.eval()

    try:
        length = min(8, model.config.seq_len)
        prefix_len = length // 2

        original = torch.randint(
            0, model.config.vocab_size, (1, length), device=device
        )
        changed = original.clone()
        changed[:, prefix_len:] = (
            changed[:, prefix_len:] + 1
        ) % model.config.vocab_size

        with torch.no_grad():
            logits_a = model(original)[:, :prefix_len, :]
            logits_b = model(changed)[:, :prefix_len, :]

        if not torch.allclose(logits_a, logits_b, atol=1e-5, rtol=1e-5):
            raise AssertionError("Causal masking failed")

    finally:
        model.train(was_training)


def main():
    torch.manual_seed(42)

    train_loader, val_loader, meta = build_dataloaders(
        processed_dir="data/processed",
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    config = GPTConfig(
        vocab_size=int(meta["vocab_size"]),
        seq_len=int(meta["seq_len"]),
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        pad_id=meta.get("pad_id")
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = GPTModel(config).to(device)

    # Check causal attention
    test_causal_mask(model, device)

    # Forward pass
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)

    assert x.shape == y.shape == (BATCH_SIZE, config.seq_len)

    model.train()
    logits = model(x)

    expected_shape = (BATCH_SIZE, config.seq_len, config.vocab_size)
    assert tuple(logits.shape) == expected_shape
    assert torch.isfinite(logits).all()

    # Calculate loss and gradients. no update. no optimizer  yet
    loss = F.cross_entropy( 
        logits.reshape(-1, config.vocab_size),
        y.reshape(-1)
    )

    assert torch.isfinite(loss)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise AssertionError(f"Invalid gradient in {name}")

    # Validation
    model.eval()

    with torch.no_grad():
        val_x, val_y = next(iter(val_loader))
        val_x, val_y = val_x.to(device), val_y.to(device)

        val_logits = model(val_x)
        val_loss = F.cross_entropy(
            val_logits.reshape(-1, config.vocab_size),
            val_y.reshape(-1)
        )

    return loss.item(), val_loss.item()


if __name__ == "__main__":
    main()
