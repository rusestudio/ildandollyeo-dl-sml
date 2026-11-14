from dataclasses import dataclass


@dataclass
class DataConfig:
    # download.py
    download_dir: str = "data/raw"
    download_pattern: str = "sample/10BT/013_00000.parquet"  # 우선 샘플 하나만

    # split_dataset.py
    raw_dir: str = "data/raw/sample/10BT"
    val_size: float = 0.0005
    random_seed: int = 42
    num_proc: int = 8

    # preparetoken.py
    processed_dir: str = "data/processed"
    max_train_docs: int = 20000  # 0 = 전체
    max_val_docs: int = 0  # 0 = 전체
    seq_len: int = 256  # 모델 context 길이
    encode_batch_docs: int = 128  # 토큰화할 때 한 번에 묶는 문서 수
    verify_batch_size: int = 4  # 데이터 확인용 배치 크기


@dataclass
class TokenizerConfig:
    tokenizer_path: str = "data/tokenizer/tokenizer.json"
    vocab_size: int = 16000
    max_docs: int = 20000  # 토크나이저 학습에 쓸 문서 수
    max_token_length: int = 16
    num_val_docs: int = 4  # 토크나이저 검증용 문서 수


@dataclass
class ModelConfig:
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_hidden_size: int = 1024
    dropout: float = 0.1


@dataclass
class PretrainConfig:
    out_dir: str = "checkpoints"

    batch_size: int = 32  # gpu 메모리에 맞게 조절
    gradient_accumulation_steps: int = 4
    max_steps: int = 200
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

    # 학습 중 샘플 생성
    sample_prompt: str = "The meaning of education is"
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
