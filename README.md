# Team 4 - ildandollyeo-dl-sml

## Overview

| 구분 | 내용 |
|-----|-----|
| 모델 | Decoder-only Transformer (GPT), **324.7M** parameters |
| 토크나이저 | Byte-Level BPE, vocab 16,000 (직접 학습) |
| 사전학습 데이터 | FineWeb-Edu sample-10BT (외부 사전학습 모델·데이터 미사용) |
| 파인튜닝 데이터 | `/data/finetuning-dataset/train.parquet` (99,842문항, 추가 데이터 X) |
| 파인튜닝 결과 | val acc **0.5861**, val AUC 0.8149 (train.parquet 중 5% 4,975문항 hold-out) |

## Features

- **모델 구조**: d_model 1024, 24 layers, 16 heads (head_dim 64), context 1024 tokens
- **구성 요소**: RoPE 위치 임베딩, SwiGLU FFN (d_ff 2816), RMSNorm (pre-norm), QK-norm, 입력 임베딩–출력층 weight tying
- **사전학습**: AdamW, WSD 스케줄 (warmup 400 → 유지 → 마지막 20% 선형 감소), z-loss 1e-4, bf16
- **파인튜닝**: `[BOS]` + 문제/보기 + `Answer:` 뒤 정답 토큰(A/B/C/D) 위치에서만 loss 계산

## Directory Structure

```
models/
├── README.md                  # README 문서
├── load_model.py              # 체크포인트 로드 + 정답 예측 예제 (단독 실행 가능)
├── checkpoints/
│   ├── pretrained.pt          # 사전학습 모델 (8,000 step, val loss 2.684)
│   └── finetuned.pt           # 파인튜닝 모델 (최종 제출 모델, val acc 0.5861)
├── tokenizer/
│   ├── tokenizer.json         # 학습된 Byte-Level BPE 토크나이저 (HuggingFace tokenizers 형식)
│   └── tokenizer_meta.json    # 토크나이저 학습 정보 (데이터, special token id 등)
└── code/
    ├── gpt_model.py           # 모델 정의 (GPTConfig, GPTModel)
    ├── embedding.py           # 토큰 임베딩, RoPE
    ├── config.py              # Hydra 설정을 dataclass로 불러옴
    ├── conf/                  # Hydra 설정 (model / train / finetune / tokenizer / data)
    ├── download.py            # 1. FineWeb-Edu 다운로드
    ├── split_dataset.py       # 2. train / val / test 분할
    ├── tokenizer_train.py     # 3. 토크나이저 학습
    ├── preparetoken.py        # 4. 토큰화 후 .bin 저장
    ├── dataset.py             # 사전학습 DataLoader
    ├── train.py               # 5. 사전학습
    ├── finetune.py            # 6. 파인튜닝
    └── finetune_sweep.py      # 파인튜닝 하이퍼파라미터 sweep (Hydra multirun)
```

## Prerequisites

- Python 3.10+, CUDA GPU (bf16 지원)
- `torch`, `tokenizers`, `datasets`, `hydra-core`, `omegaconf`, `scikit-learn`, `pandas`, `pyarrow`

```bash
pip install torch tokenizers datasets hydra-core omegaconf scikit-learn pandas pyarrow
```

## Usage

### 1. 모델 로드

체크포인트는 `{"model": state_dict, "gpt_config": dict, ...}` 형식입니다. `gpt_config`로 모델을 만든 뒤 가중치를 넣습니다.

```python
import sys, torch
from tokenizers import Tokenizer

sys.path.insert(0, "code")                       # models/ 폴더 기준
from gpt_model import GPTConfig, GPTModel

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

ckpt = torch.load("checkpoints/finetuned.pt", map_location="cpu")   # 또는 pretrained.pt
model = GPTModel(GPTConfig(**{**ckpt["gpt_config"], "dropout": 0.0}))
model.load_state_dict(ckpt["model"])
model = model.cuda().eval()

# model(input_ids: LongTensor[B, T]) -> logits: [B, T, vocab_size]
```

토크나이저 참고 사항:
- special token id: `[PAD]`=0, `[BOS]`=1, `[EOS]`=2
- `tokenizer.encode()`는 BOS/EOS를 **자동으로 붙이지 않습니다**. 입력 맨 앞에 `[BOS]`를 직접 붙입니다.
- `"A"`, `"B"`, `"C"`, `"D"`는 각각 토큰 1개(id 41–44)입니다.

### 2. 입출력 형식 (공지 형식과 동일)

```python
ANSWER_OPTIONS = ["A", "B", "C", "D"]

def build_context(sample):
    question = sample["question"].strip()
    choices_str = " ".join(f"({ANSWER_OPTIONS[i]}) {c}" for i, c in enumerate(sample["choices"]))
    return f"{question}\n{choices_str}\nAnswer: "

input_ids = [bos_id] + tokenizer.encode(build_context(sample)).ids
# 마지막 위치의 logit에서 A/B/C/D 토큰 중 가장 높은 것을 정답으로 사용
```

### 3. 예측 실행

```bash
cd models
python load_model.py                                    # finetuned.pt로 예시 5문항 예측
python load_model.py --ckpt checkpoints/pretrained.pt   # 사전학습 모델
python load_model.py --eval 1000                        # 앞 1000문항 정답률
```

### 4. 재현 (code/ 폴더에서 실행)

```bash
cd models/code
python download.py && python split_dataset.py   # 데이터 준비
python tokenizer_train.py                        # 토크나이저
python preparetoken.py                           # 토큰화
python train.py                                  # 사전학습
python finetune.py finetune.learning_rate=5e-5 finetune.dropout=0.0 finetune.num_epochs=3 finetune.min_lr=5e-6   # 파인튜닝
```

Hydra 설정은 `code/conf/`에 있고, 명령줄에서 `key=value`로 덮어쓸 수 있습니다. 
사전학습 데이터 경로(`paths.raw`, `paths.processed`)는 학습 서버 기준이므로 환경에 맞게 수정해야 합니다.

## Training Details

### Pretraining

| 항목 | 값 |
|---|---|
| 데이터 | FineWeb-Edu sample-10BT 중 parquet 14개 (`000`–`013_00000.parquet`), 토큰화 후 train 11.05B tokens |
| 학습량 | 8,000 step × 약 0.5M tokens/step ≈ **4.2B tokens** (train 토큰의 약 38%) |
| batch | 16 × grad accum 32 × seq 1024 |
| optimizer | AdamW (β 0.9/0.95, weight decay 0.1), grad clip 1.0 |
| learning rate | 3e-4 → 3e-5, warmup 400 step, WSD (마지막 20% 감소) |
| 결과 | best val loss **2.684** |

### Finetuning

| 항목 | 값 |
|---|---|
| 데이터 | train.parquet 99,842문항 → 95% 학습 / 5% 검증 (seed 42), 1024 토큰 초과 353문항 제외 |
| loss | 정답 토큰 위치의 cross-entropy만 사용 (context는 마스킹) |
| 입력 템플릿 | `{question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:` |
| batch | 16 × grad accum 4, 3 epochs (4,430 step) |
| optimizer | AdamW (β 0.9/0.95, weight decay 0.1), warmup 3% 후 선형 감소 (5e-5 → 5e-6) |
| learning rate / dropout | 5e-5 / 0.0 |
| 선택 | lr {2e-5, 5e-5, 1e-4} × dropout {0.0, 0.1} sweep 중 val acc 최고 (step 3,700 체크포인트) |
| 결과 | val acc **0.5861**, val AUC **0.8149** |
