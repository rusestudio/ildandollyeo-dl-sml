# Team 4 - ildandollyeo-dl-sml

## Overview
| 구분 | 내용 |
|-----|-----|
| 모델 | Decoder-only Transformer (GPT), 324.7M parameters |
| 토크나이저 | Byte-Level BPE, vocab 16,000 (직접 학습) |
| 사전학습 데이터 | FineWeb-Edu sample-10BT (외부 사전학습 모델·데이터 미사용) |
| 파인튜닝 데이터 | `/data/finetuning-dataset/train.parquet` (99,842문항, 추가 데이터 X) |
| 파인튜닝 결과 | val acc 0.5970, val AUC 0.8210 (train.parquet 중 5% 4,975문항 hold-out, 공지된 입력 형식 기준) |

## Features

- 모델 구조: d_model 1024, 24 layers, 16 heads (head_dim 64), context 1024 tokens
- 구성 요소: RoPE 위치 임베딩, SwiGLU FFN (d_ff 2816), RMSNorm (pre-norm), QK-norm, 입력 임베딩–출력층 weight tying
- 사전학습: AdamW, WSD 스케줄 (warmup 400 → 유지 → 마지막 20% 선형 감소), z-loss 1e-4, bf16
- 파인튜닝: `[BOS]` + 문제/보기 + `Answer: ` 뒤 정답 토큰 위치에서만 loss 계산, 총 4 epoch

## Directory Structure

```
models/
├── README.md                  # README 문서
├── load_model.py              # 체크포인트 로드 + 정답 예측 예제 (단독 실행 가능)
├── checkpoints/
│   ├── pretrained.pt          # 사전학습 모델 (8,000 step, val loss 2.684)
│   └── finetuned.pt           # 파인튜닝 모델 (최종 제출 모델, val acc 0.5970)
├── tokenizer/
│   ├── tokenizer.json         # 학습된 Byte-Level BPE 토크나이저 (HuggingFace tokenizers 형식)
│   └── tokenizer_meta.json    # 토크나이저 학습 정보 (데이터, special token id 등)
└── code/
    ├── gpt_model.py           # 모델 정의 (GPTConfig, GPTModel)
    ├── embedding.py           # 토큰 임베딩, RoPE
    ├── config.py              # Hydra 설정을 dataclass로 불러옴
    ├── conf/                  # Hydra 설정 (finetune/sft.yaml: 평가 형식, finetune/sft_stage1.yaml: 1단계)
    ├── download.py            # 1. FineWeb-Edu 다운로드
    ├── split_dataset.py       # 2. train / val / test 분할
    ├── tokenizer_train.py     # 3. 토크나이저 학습
    ├── preparetoken.py        # 4. 토큰화 후 .bin 저장
    ├── dataset.py             # 사전학습 DataLoader
    ├── train.py               # 5. 사전학습
    ├── finetune.py            # 6. 파인튜닝
    ├── finetune_adapt.py      # 7. 추가 파인튜닝 (기존 체크포인트에서 이어서 학습, 같은 train/val split)
    └── finetune_sweep.py      # 파인튜닝 하이퍼파라미터 sweep (Hydra multirun)
```

## 사전에 확인해야 하는 부분

- Python 3.10+, CUDA GPU (bf16 지원)
- `torch`, `tokenizers`, `datasets`, `hydra-core`, `omegaconf`, `scikit-learn`, `pandas`, `pyarrow`

```bash
pip install torch tokenizers datasets hydra-core omegaconf scikit-learn pandas pyarrow
```

## 사용 방법

### 1. Model Load

체크포인트는 `{"model": state_dict, "gpt_config": dict, ...}` 형식
가중치는 `gpt_config`로 모델을 만든 뒤에 넣습니다.

```python
import sys, torch
from tokenizers import Tokenizer

sys.path.insert(0, "code") # models/ 폴더 기준
from gpt_model import GPTConfig, GPTModel

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

ckpt = torch.load("checkpoints/finetuned.pt", map_location="cpu") # 파인튜닝 모델
# ckpt = torch.load("checkpoints/pretrained.pt", map_location="cpu") # 사전학습 모델
model = GPTModel(GPTConfig(**{**ckpt["gpt_config"], "dropout": 0.0}))
model.load_state_dict(ckpt["model"])
model = model.cuda().eval()

# model(input_ids: LongTensor[B, T]) -> logits: [B, T, vocab_size]
```

토크나이저 참고 사항:
- special token id: `[PAD]`=0, `[BOS]`=1, `[EOS]`=2
- `tokenizer.encode()`는 BOS/EOS를 자동으로 붙이지 않기 때문에 입력 맨 앞에 `[BOS]`를 직접 붙여야 합니다.

### 2. 입출력 형식

```python
ANSWER_OPTIONS = ["A", "B", "C", "D"]

def build_context(sample):
    question = sample["question"].strip()
    choices_str = " ".join(f"({ANSWER_OPTIONS[i]}) {c}" for i, c in enumerate(sample["choices"]))
    return f"{question}\n{choices_str}\nAnswer: "

input_ids = [bos_id] + tokenizer.encode(build_context(sample)).ids
# 마지막 위치의 logit에서 가장 높은 값의 토큰을 정답으로 사용함
```

### 3. exec

```bash
cd models
python load_model.py # finetuned.pt로 예시 5문항 예측
python load_model.py --ckpt checkpoints/pretrained.pt # 사전학습 모델
python load_model.py --eval 1000 # 앞 1000문항 정답률
```

### 4. 재현 (code/ 폴더에서 실행)

```bash
cd models/code
python download.py && python split_dataset.py # 데이터 준비
python tokenizer_train.py # 토크나이저
python preparetoken.py # 토큰화
python train.py # 사전학습
# 파인튜닝 1단계: 3 epoch
python finetune.py finetune=sft_stage1 paths.finetune_checkpoint=checkpoints/stage1
# 파인튜닝 2단계: +0.5 epoch
ADAPT_EPOCHS=0.5 python finetune_adapt.py finetune.pretrained_path=checkpoints/stage1/ckpt.pt finetune.learning_rate=2e-5 finetune.min_lr=2e-6 finetune.dropout=0.0 finetune.eval_steps=150 paths.finetune_checkpoint=checkpoints/stage2
# 파인튜닝 3단계: +0.5 epoch -> finetuned.pt
ADAPT_EPOCHS=0.5 python finetune_adapt.py finetune.pretrained_path=checkpoints/stage2/ckpt.pt finetune.learning_rate=1e-5 finetune.min_lr=1e-6 finetune.dropout=0.1 finetune.eval_steps=100 paths.finetune_checkpoint=checkpoints/stage3
```

Hydra 설정은 `code/conf/`에 있고, 명령줄에서 `key=value`로 덮어쓸 수 있습니다.
사전학습 데이터 경로(`paths.raw`, `paths.processed`)는 학습 서버 기준이므로 환경에 맞게 수정해야 합니다.

### 5. 소수 epoch 추가 학습 (finetune_adapt.py)

기존 체크포인트에서 이어서 원하는 epoch만큼 학습합니다. train/val split은 1단계와 같습니다.

```bash
cd models/code
ADAPT_EPOCHS=0.5 python finetune_adapt.py \
  finetune.pretrained_path=../checkpoints/finetuned.pt \
  finetune.learning_rate=1e-5 finetune.min_lr=1e-6 \
  paths.finetune_checkpoint=checkpoints/adapt
```

| 옵션 | 설명 |
|---|---|
| `ADAPT_EPOCHS` | 학습할 epoch 수 (소수 가능, 기본 0.5) |
| `finetune.pretrained_path` | 시작 체크포인트 |
| `paths.finetune_checkpoint` | 저장 폴더 (val acc 최고 시점의 `ckpt.pt` 저장) |

## Details

### Pretraining

| 항목 | 값 |
|---|---|
| 데이터 | FineWeb-Edu sample-10BT 중 parquet 14개 (`000`–`013_00000.parquet`), 토큰화 후 train 11.05B tokens |
| 학습량 | 8,000 step × 약 0.5M tokens/step 거의 4.2B tokens (train 의 약 38%) |
| batch | 16 × grad accum 32 × seq 1024 |
| optimizer | AdamW ($\beta$ 0.9/0.95, weight decay 0.1), grad clip 1.0 |
| learning rate | 3e-4 → 3e-5, warmup 400 step, WSD (마지막 20% 감소) |
| 결과 | best val loss 2.684 |

### Finetuning

| 항목 | 값 |
|---|---|
| 데이터 | train.parquet 99,842문항 → 95% 학습 / 5% 검증 (seed 42), 1024 토큰 초과 353문항 제외, 전 단계 같은 split |
| loss | 정답 토큰 위치의 cross-entropy (context 마스킹) |
| batch | 16 × grad accum 4 |
| optimizer | AdamW ($\beta$ 0.9/0.95, weight decay 0.1), 단계마다 warmup 3% 후 선형 감소 |
| 학습량 | 총 4 epoch (3 → +0.5 → +0.5), 이전 단계 체크포인트에서 이어서 학습 |

| 단계 | epoch | lr | dropout | 템플릿 | 사용 체크포인트 |
|---|---|---|---|---|---|
| 1 | 3 | 5e-5 → 5e-6 | 0.0 | `{question}\n(A) {choices[0]} (B) ... (D) {choices[3]}\nAnswer: `| step 3,700 |
| 2 | +0.5 | 2e-5 → 2e-6 | 0.0 | `{question}\n(A) {choices[0]} (B) ... (D) {choices[3]}\nAnswer: ` | step 300 |
| 3 | +0.5 | 1e-5 → 1e-6 | 0.1 | `{question}\n(A) {choices[0]} (B) ... (D) {choices[3]}\nAnswer: ` | step 700 → `finetuned.pt` |

- 2·3단계는 같은 seed라 0.5 epoch 동안 학습 데이터의 같은 절반을 사용
- 결과: val acc 0.5970, val AUC 0.8210 
