from pathlib import Path
import hashlib
from datasets import load_dataset

from config import DataConfig

data_config = DataConfig()

RAW_DIR = Path(data_config.raw_dir)
VAL_SIZE = data_config.val_size
TEST_SIZE = data_config.test_size
SEED = data_config.random_seed
NUM_PROC = data_config.num_proc

_split = None


def list_files():
    return sorted(str(p) for p in RAW_DIR.glob("*.parquet"))


def load_split():
    global _split
    if _split is None:
        files = list_files()

        if not files:
            raise FileNotFoundError("No Parquet files found.")

        _split = load_dataset(
            "parquet",
            data_files=files,
            split="train",
            streaming=True,
        )

        dataset = load_dataset("parquet", data_files=list_files(), num_proc=NUM_PROC)
        # train / holdout -> holdout 을 val / test 로
        split_dataset = dataset["train"].train_test_split(
            test_size=VAL_SIZE + TEST_SIZE, seed=SEED, shuffle=True
        )
        holdout = split_dataset["test"].train_test_split(
            test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), seed=SEED, shuffle=True
        )
        split_dataset["val"] = holdout["train"]
        split_dataset["test"] = holdout["test"]
        _split = split_dataset
    return _split

def assign_split(text):
    key = str(SEED).encode("utf-8")

    digest = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
        key=key,
    ).digest()

    bucket = int.from_bytes(digest, "big") % 1_000_000

    test_threshold = round(TEST_SIZE * 1_000_000)
    val_threshold = test_threshold + round(VAL_SIZE * 1_000_000)

    if bucket < test_threshold:
        return "test"
    if bucket < val_threshold:
        return "val"

    return "train"

def iter_texts(split="train", max_docs=None):
    ds = load_split()[split].select_columns(["text"])
    if max_docs:
        ds = ds.select(range(min(max_docs, len(ds))))
    for batch in ds.iter(batch_size=1000):
        for text in batch["text"]:
            yield text


if __name__ == "__main__":
    s = load_split()
    print(s)
    print("train:", len(s["train"]), "val:", len(s["val"]), "test:", len(s["test"]))
    print("HEAD 300:", s["train"][0]["text"][:300])
