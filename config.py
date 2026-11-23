import sys
from dataclasses import dataclass

from conf.setting import load_config

cfg = load_config(sys.argv[1:])


@dataclass
class DataConfig:
    # download.py
    download_dir: str = cfg.paths.download
    download_pattern: str = cfg.data.download_pattern

    # split_dataset.py
    raw_dir: str = cfg.paths.raw
    val_size: float = cfg.data.val_size
    random_seed: int = cfg.data.split_seed
    num_proc: int = cfg.data.num_proc

    # preparetoken.py
    processed_dir: str = cfg.paths.processed
    max_train_docs: int = cfg.data.max_train_docs
    max_val_docs: int = cfg.data.max_val_docs
    seq_len: int = cfg.data.seq_len
    encode_batch_docs: int = cfg.data.encode_batch_docs
    verify_batch_size: int = cfg.data.verify_batch_size


@dataclass
class TokenizerConfig:
    tokenizer_path: str = cfg.paths.tokenizer
    vocab_size: int = cfg.tokenizer.vocab_size
    max_docs: int = cfg.tokenizer.max_docs
    max_token_length: int = cfg.tokenizer.max_token_length
    num_val_docs: int = cfg.tokenizer.num_val_docs


@dataclass
class ModelConfig:
    hidden_size: int = cfg.model.d_model
    num_layers: int = cfg.model.n_layers
    num_heads: int = cfg.model.n_heads
    ffn_hidden_size: int = cfg.model.d_ff
    dropout: float = cfg.model.dropout
    ffn_type: str = cfg.model.ffn_type


@dataclass
class PretrainConfig:
    out_dir: str = cfg.paths.checkpoint

    batch_size: int = cfg.train.batch_size
    gradient_accumulation_steps: int = cfg.train.gradient_accumulation_steps
    max_steps: int = cfg.train.max_steps
    learning_rate: float = cfg.train.learning_rate
    min_lr: float = cfg.train.min_lr
    warmup_steps: int = cfg.train.warmup_steps
    decay_ratio: float = cfg.train.decay_ratio
    z_loss: float = cfg.train.z_loss
    adam_betas: tuple[float, float] = tuple(cfg.train.adam_betas)
    weight_decay: float = cfg.train.weight_decay
    max_grad_norm: float = cfg.train.max_grad_norm
    compile: bool = cfg.train.compile

    eval_steps: int = cfg.train.eval_steps
    eval_iters: int = cfg.train.eval_iters
    log_steps: int = cfg.train.log_steps
    num_workers: int = cfg.train.num_workers
    resume_from_checkpoint: bool = cfg.train.resume_from_checkpoint
    random_seed: int = cfg.train.seed

    # 학습 중 샘플 생성
    sample_prompt: str = cfg.train.sample_prompt
    max_new_tokens: int = cfg.train.max_new_tokens
    temperature: float = cfg.train.temperature
    top_k: int = cfg.train.top_k


@dataclass
class FinetuneConfig:
    data_path: str = cfg.paths.finetune_data
    out_dir: str = cfg.paths.finetune_checkpoint
    pretrained_path: str = cfg.finetune.pretrained_path

    prompt_template: str = cfg.finetune.prompt_template
    output_column: str = cfg.finetune.output_column
    val_size: float = cfg.finetune.val_size
    split_seed: int = cfg.finetune.split_seed

    batch_size: int = cfg.finetune.batch_size
    gradient_accumulation_steps: int = cfg.finetune.gradient_accumulation_steps
    num_epochs: int = cfg.finetune.num_epochs
    learning_rate: float = cfg.finetune.learning_rate
    min_lr: float = cfg.finetune.min_lr
    warmup_ratio: float = cfg.finetune.warmup_ratio
    adam_betas: tuple[float, float] = tuple(cfg.finetune.adam_betas)
    weight_decay: float = cfg.finetune.weight_decay
    max_grad_norm: float = cfg.finetune.max_grad_norm
    dropout: float = cfg.finetune.dropout

    eval_steps: int = cfg.finetune.eval_steps
    log_steps: int = cfg.finetune.log_steps
    random_seed: int = cfg.finetune.seed

    # 학습 중 샘플 생성
    max_new_tokens: int = cfg.finetune.max_new_tokens
    temperature: float = cfg.finetune.temperature
    top_k: int = cfg.finetune.top_k
