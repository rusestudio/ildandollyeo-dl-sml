import sys
from functools import lru_cache
from pathlib import Path
from datasets import load_dataset
from config.setting import load_config

def list_files(raw_dir):
    return sorted(str(path) for path in Path(raw_dir).glob("*.parquet"))

@lru_cache(maxsize=1)
def load_split(raw_dir, val_size, seed, num_proc):
    dataset = load_dataset("parquet", data_files=list_files(raw_dir), num_proc=num_proc,)
    splits = dataset["train"].train_test_split(test_size=val_size, seed=seed, shuffle=True,)
    splits["val"] = splits.pop("test")
    return splits

def iter_texts(cfg, split="train"):
    splits = load_split(
        cfg.path.raw, cfg.data.val_size, 
        cfg.data.split_seed, cfg.data.num_proc,
        )
    texts = splits[split].select_columns(["text"])

    for batch in texts.iter(batch_size=cfg.data.read_batch_docs):
        yield from batch["text"]

if __name__ == "__main__":
    cfg = load_config(sys.argv[1:])
    splits = load_split(
        cfg.path.raw, cfg.data.val_size, 
        cfg.data.split_seed, cfg.data.num_proc,
        )
    print("train:", len(splits["train"]))
    print("val:", len(splits["val"]))