import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

VOCAB_SIZE = 16_000 #fill # 정현님 please remove n combine in full config 
MAX_DOCS = 20_000
MODEL_PATH = "data/tokenizer/tokenizer.json"
MUN_VAL_DOCS = 4  
MAX_TOKEN_LENGTH = 5 

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[CLS]"] + [
    f"[UNUSED_{i}]" for i in range(5)
]

def build_tokenizer(vocab_size=VOCAB_SIZE):
    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = Sequence([
        ByteLevel(add_prefix_space=False),
        Digits(individual_digits=True)
    ])
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer.decoder = ByteLevelDecoder()

    tokens = [
        AddedToken(token, special=True, normalized=False)
        for token in SPECIAL_TOKENS
    ]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=ByteLevel.alphabet(),
        max_token_length=MAX_TOKEN_LENGTH,
        special_tokens=tokens
    )

    return tokenizer, trainer


def verify_tokenizer(tokenizer, examples=None):
    vocab_size = tokenizer.get_vocab_size()

    # Check special tokens
    for token in SPECIAL_TOKENS:
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"missing st: {token}")

    if examples is None:
        examples = [
            "hello dep leanirng",
            "hello sml"
        ]

    # Check encoding and decoding
    for text in examples:
        if not text:
            continue

        encoded = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(
            encoded.ids, skip_special_tokens=False
        )

        if decoded != text:
            raise AssertionError(
                f"mismatch: {text!r} != {decoded!r}"
            )

        if any(idx < 0 or idx >= vocab_size for idx in encoded.ids):
            raise AssertionError("invalid id")

    print(f"ok. vocab size: {vocab_size:,}")


def get_validation_examples():
    texts = [
        "hi deep learning",
        "need training asap"
    ]

    for text in iter_texts(split="val", max_docs=NUM_VAL_DOCS):
        if text:
            texts.append(text[:200])

    return texts


def train_tokenizer(
    vocab_size=VOCAB_SIZE,
    max_docs=MAX_DOCS,
    model_path=MODEL_PATH,
    force=False
):
    if vocab_size < len(SPECIAL_TOKENS) + 256:
        raise ValueError("Vocabulary size must be at least 265")

    if max_docs < 0:
        raise ValueError("max_docs must be 0 or greater")

    output = Path(model_path)

    if output.exists() and not force:
        raise FileExistsError(
            f"{output} already exists. Set force=True to retrain."
        )

    source_files = list_files()
    if not source_files:
        raise FileNotFoundError("No Parquet files found. Prepare the dataset first.")

    splits = load_split()
    n_train = len(splits["train"])
    n_val = len(splits["val"])

    n_fit = n_train if max_docs == 0 else min(n_train, max_docs)

    if n_fit == 0 or n_val == 0:
        raise ValueError("Train and validation sets must not be empty")

    # Train BPE using training data only
    tokenizer, trainer = build_tokenizer(vocab_size)

    tokenizer.train_from_iterator(
        iter_texts(split="train", max_docs=n_fit),
        trainer=trainer,
        length=n_fit
    )

    # Save tokenizer
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))

    meta = {
        "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT",
        "source_files": [Path(p).name for p in source_files],
        "split_seed": SEED,
        "validation_fraction": VAL_SIZE,
        "train_documents_available": n_train,
        "validation_documents_available": n_val,
        "tokenizer_train_documents": n_fit,
        "tokenizer_algorithm": "Byte-Level BPE",
        "requested_vocab_size": vocab_size,
        "actual_vocab_size": tokenizer.get_vocab_size(),
        "max_token_length": MAX_TOKEN_LENGTH,
        "special_token_ids": {
            token: tokenizer.token_to_id(token)
            for token in SPECIAL_TOKENS
        },
        "bos_eos_automatically_inserted": False
    }

    metadata_path = output.with_name("tokenizer_meta.json")
    metadata_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    # Load saved tokenizer and verify
    loaded = Tokenizer.from_file(str(output))
    verify_tokenizer(loaded, get_validation_examples())

    print(f"Tokenizer saved to {output}")

    return loaded


if __name__ == "__main__":
    train_tokenizer()
