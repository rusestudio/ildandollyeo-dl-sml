from dataclasses import dataclass


@dataclass
class DataConfig:
    download_dir: str = "data/raw"
    download_pattern: str = "sample/10BT/013_00000.parquet"

    raw_dir: str = "data/raw/sample/10BT"
    val_size: float = 0.0005
    random_seed: int = 42
    num_proc: int = 2

    processed_dir: str = "data/processed"
    max_train_docs: int = 500
    max_val_docs: int = 0
    seq_len: int = 64
    encode_batch_docs: int = 32
    verify_batch_size: int = 4


@dataclass
class TokenizerConfig:
    tokenizer_path: str = "data/tokenizer/tokenizer.json"
    vocab_size: int = 16000
    max_docs: int = 500
    max_token_length: int = 16
    num_val_docs: int = 4


@dataclass
class ModelConfig:
    hidden_size: int = 64
    num_layers: int = 1
    num_heads: int = 2
    ffn_hidden_size: int = 256
    dropout: float = 0.0


@dataclass
class PretrainConfig:
    out_dir: str = "checkpoints"

    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 10
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 1
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    eval_steps: int = 5
    eval_iters: int = 2
    log_steps: int = 1
    num_workers: int = 0
    resume_from_checkpoint: bool = False
    random_seed: int = 42

    sample_prompt: str = "The meaning of education is"
    max_new_tokens: int = 10
    temperature: float = 0.8
    top_k: int = 50
