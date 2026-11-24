import hashlib
import json
import os
from itertools import islice
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from config import DataConfig, TokenizerConfig
from dataset import build_dataloaders, load_packed_datasets
from split_dataset import SEED, VAL_SIZE, list_files, load_split, iter_texts

data_config = DataConfig()
tokenizer_config = TokenizerConfig()

TOKENIZER_PATH = Path(tokenizer_config.tokenizer_path)
TOKENIZER_META_PATH = TOKENIZER_PATH.with_name("tokenizer_meta.json")
PROCESSED_DIR = Path(data_config.processed_dir)

MAX_TRAIN_DOCS = data_config.max_train_docs
MAX_VAL_DOCS = data_config.max_val_docs
SEQ_LEN = data_config.seq_len
BATCH_SIZE = data_config.verify_batch_size
BATCH_DOCS = data_config.encode_batch_docs


def file_sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def batches(texts, size):
    iterator = iter(texts)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch


def encode_to_bin(tokenizer, texts, output_path, eos_id, dtype, batch_docs=BATCH_DOCS):
    output = Path(output_path)
    temp = output.with_name(output.name + ".tmp")
    n_seen = n_written = n_empty = n_tokens = 0
    max_id = int(np.iinfo(dtype).max)

    try:
        with temp.open("wb") as file:
            for text_batch in batches(texts, batch_docs):
                n_seen += len(text_batch)
                valid_texts = [t for t in text_batch if isinstance(t, str) and t]
                n_empty += len(text_batch) - len(valid_texts)
                if not valid_texts:
                    continue

                encodings = tokenizer.encode_batch(
                    valid_texts, add_special_tokens=False
                )
                ids = []
                for encoded in encodings:
                    if not encoded.ids:
                        n_empty += 1
                        continue
                    ids.extend(encoded.ids)
                    ids.append(eos_id)
                    n_written += 1

                if ids:
                    if min(ids) < 0 or max(ids) > max_id:
                        raise ValueError("token id invalid")
                    np.asarray(ids, dtype=dtype).tofile(file)
                    n_tokens += len(ids)

        os.replace(temp, output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise

    return {
        "documents_seen": n_seen,
        "documents_encoded": n_written,
        "documents_empty": n_empty,
        "tokens_including_eos": n_tokens,
    }


def validate_source(tokenizer_meta, files, train_count, val_count):
    previous_files = tokenizer_meta.get("source_files")
    current_files = [Path(p).name for p in files]

    if previous_files is not None and sorted(previous_files) != sorted(current_files):
        raise RuntimeError("prev file change")

    for key, count in (
        ("train_documents_available", train_count),
        ("validation_documents_available", val_count),
    ):
        if key in tokenizer_meta and tokenizer_meta[key] != count:
            raise RuntimeError(f"datasplit key: {key}")


def verify_outputs(root=PROCESSED_DIR, batch_size=BATCH_SIZE, seq_len=None):
    root = Path(root)
    train_ds, val_ds, meta = load_packed_datasets(root, seq_len=seq_len)

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"misisng tokkennizer: {TOKENIZER_PATH}")
    if meta.get("tokenizer_sha256") != file_sha256(TOKENIZER_PATH):
        raise RuntimeError("not match")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    if tokenizer.get_vocab_size() != meta["vocab_size"]:
        raise RuntimeError("not match")
    if tokenizer.token_to_id("[EOS]") != meta["eos_id"]:
        raise RuntimeError("not match")

    for name, ds in (("train", train_ds), ("val", val_ds)):
        if ds.num_tokens != meta[name]["tokens_including_eos"]:
            raise RuntimeError(f"mismatch in {name}")

    train_loader, val_loader, _ = build_dataloaders(
        root, batch_size=batch_size, seq_len=seq_len, num_workers=0
    )

    for loader, ds in ((train_loader, train_ds), (val_loader, val_ds)):
        x, y = next(iter(loader))
        x0, y0 = ds[0]

        assert x0.shape == y0.shape == (ds.seq_len,)
        assert np.array_equal(x0.numpy()[1:], y0.numpy()[:-1]), "mismatch"
        assert x.min().item() >= 0 and y.max().item() < meta["vocab_size"]


def prepare(
    max_train_docs=MAX_TRAIN_DOCS,
    max_val_docs=MAX_VAL_DOCS,
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE,
    batch_docs=BATCH_DOCS,
    output_dir=PROCESSED_DIR,
    force=False,
):
    if seq_len < 2 or batch_size < 1 or batch_docs < 1:
        raise ValueError("seq, batch invalis")
    if max_train_docs < 0 or max_val_docs < 0:
        raise ValueError("limit")
    if not TOKENIZER_PATH.is_file() or not TOKENIZER_META_PATH.is_file():
        raise FileNotFoundError("missing tokennizer")

    output = Path(output_dir)
    if not force and any(
        (output / f).exists() for f in ("train.bin", "val.bin", "meta.json")
    ):
        raise FileExistsError(f"data exist at {output}")

    files = list_files()
    if not files:
        raise FileNotFoundError("no file found")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    tokenizer_meta = json.loads(TOKENIZER_META_PATH.read_text(encoding="utf-8"))
    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id("[EOS]")
    if eos_id is None:
        raise RuntimeError("missing [EOS] token")

    dtype = np.dtype("uint16" if vocab_size <= 65536 else "uint32")
    splits = load_split()
    train_count = len(splits["train"])
    val_count = len(splits["val"])
    validate_source(tokenizer_meta, files, train_count, val_count)

    train_limit = (
        train_count if max_train_docs == 0 else min(max_train_docs, train_count)
    )
    val_limit = val_count if max_val_docs == 0 else min(max_val_docs, val_count)
    output.mkdir(parents=True, exist_ok=True)

    train_info = encode_to_bin(
        tokenizer,
        iter_texts(split="train", max_docs=train_limit),
        output / "train.bin",
        eos_id,
        dtype,
        batch_docs,
    )
    val_info = encode_to_bin(
        tokenizer,
        iter_texts(split="val", max_docs=val_limit),
        output / "val.bin",
        eos_id,
        dtype,
        batch_docs,
    )

    meta = {
        "description": "Flat EOS-separated BPE ID streams for causal next-token LM",
        "source": "HuggingFaceFW/fineweb-edu/sample-10BT",
        "source_files": [Path(f).name for f in files],
        "split_seed": SEED,
        "validation_fraction": VAL_SIZE,
        "tokenizer_path": str(TOKENIZER_PATH).replace("\\", "/"),
        "tokenizer_sha256": file_sha256(TOKENIZER_PATH),
        "vocab_size": vocab_size,
        "pad_id": tokenizer.token_to_id("[PAD]"),
        "bos_id": tokenizer.token_to_id("[BOS]"),
        "eos_id": eos_id,
        "eos_after_each_document": True,
        "bos_inserted": False,
        "dtype": dtype.name,
        "seq_len": seq_len,
        "window_stride": seq_len,
        "target_shift": 1,
        "train": train_info,
        "val": val_info,
    }
    (output / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    verify_outputs(output, batch_size=batch_size, seq_len=seq_len)


if __name__ == "__main__":
    prepare()
