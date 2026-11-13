from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_hidden_size: int = 1024
    dropout: float = 0.1


@dataclass
class PretrainConfig:
    data_dir: str = "data/processed"
    tokenizer_path: str = "data/tokenizer/tokenizer.json"
    out_dir: str = "checkpoints"

    batch_size: int = 32  # gpu 메모리에 맞게 조절
    gradient_accumulation_steps: int = 4
    max_steps: int = 5000
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 200
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0  # gradient clipping
    eval_steps: int = 250
    eval_iters: int = 50
    log_steps: int = 10
    num_workers: int = 2
    resume_from_checkpoint: bool = False
    random_seed: int = 42
