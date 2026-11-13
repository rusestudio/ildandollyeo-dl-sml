import argparse
import os
import runpy
from pathlib import Path
from split_dataset import RAW_DIR, SEED, VAL_SIZE, list_files, load_split, iter_texts


ROOT = Path(__file__).resolve().parent

def main():
    os.chdir(ROOT)
    for filename in ("download.py", "split_dataset.py"): #태성님 다운 only 100k for test now later change when use prof gpu
        if not (ROOT / filename).is_file():
            raise FileNotFoundError(f"missing {filename} in {ROOT}")

    files = list_files()
    if not files:
        raise FileNotFoundError(f"no file {RAW_DIR}")

    split = load_split()
    train_count = len(split["train"])
    val_count = len(split["val"])
    total = train_count + val_count

    if train_count == 0 or val_count == 0:
        raise RuntimeError("trian or val empty")

    print(f"Total: {total:,} , Train: {train_count:,} ,Val: {val_count:,}")

    for name in ("train", "val"):
        text = next(iter_texts(split=name, max_docs=1), "")
        preview = str(text).replace("\n", " ")[:120]
        print(f"{name}: {preview!r}") #see data

if __name__ == "__main__":
    main()