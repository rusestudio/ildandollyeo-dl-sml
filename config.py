from dataclasses import dataclass


@dataclass
class DataConfig:
    # download.py
    download_dir: str = "data/raw"
    download_pattern: str = "sample/10BT/*.parquet"
    # 샘플 하나만 하면서 튜닝 후, 전체 파일

    # split_dataset.py
    raw_dir: str = "data/raw/sample/10BT"
    val_size: float = 0.0005
    test_size: float = 0.0005
    random_seed: int = 42
    num_proc: int = 8

    # preparetoken.py
    processed_dir: str = "data/processed"
    max_train_docs: int = 0  # 0 = 전체, 일부만 할때는 조절 필요
    max_val_docs: int = 0  # 0 = 전체
    seq_len: int = 1024  # 모델 context 길이
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
    hidden_size: int = 768  # 256
    num_layers: int = 12  # 6
    num_heads: int = 12  # 8
    ffn_hidden_size: int = 2048
    # SwiGLU는 Linear 3개라서 2048이어야 파라미터 수 동일 3072  # 1024
    dropout: float = 0.0
    ffn_type: str = "swiglu"  # relu, swiglu
    norm_type: str = "rmsnorm"  # layernorm ,rmsnorm
    qk_norm: bool = True


@dataclass
class PretrainConfig:
    out_dir: str = "checkpoints"

    batch_size: int = 16  # gpu 메모리에 맞게 조절
    gradient_accumulation_steps: int = 8
    # 현재 한 step에 16*8*1024 = 131072토큰 # 이전 case: 한 step = 32 x 2 x 256 = 16,384 토큰
    max_steps: int = 84280  # 110억 토큰 /131,072 = 1epoch # 674244  # 12600  # 1 epoch
    learning_rate: float = 1e-3
    # 실행 시 learning_rate= .. 덮어 쓸 수 있음# 6e-4  # 2e-3
    min_lr: float = 1e-4  # 6e-5  # 2e-4
    warmup_steps: int = 1000  # 2000
    adam_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0  # gradient clipping
    optimizer: str = "muon"
    compile: bool = True

    eval_steps: int = 2000  # 10000
    eval_iters: int = 50
    log_steps: int = 10
    num_workers: int = 4  # 2
    resume_from_checkpoint: bool = False
    random_seed: int = 42

    # 학습 중 샘플 생성
    sample_prompt: str = "The meaning of education is"
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50

    # wandb
    wandb_log: bool = True
    wandb_project: str = "ildandollyeo-dl-sml"
    wandb_run_name: str = ""
