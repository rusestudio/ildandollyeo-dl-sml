from pathlib import Path
from datasets import load_dataset

RAW_DIR = Path("data/raw/sample/10BT") # 정현님 please remove n combine in full config 
VAL_SIZE = 0.0005
SEED = 42
NUM_PROC = 8

_split = None

def list_files():
    return sorted(str(p) for p in RAW_DIR.glob("*.parquet"))

def load_split():
    global _split
    if _split is None:
        dataset = load_dataset("parquet", data_files=list_files(), num_proc=NUM_PROC)
        split_dataset = dataset["train"].train_test_split(test_size=VAL_SIZE, seed=SEED, shuffle=True)
        split_dataset["val"] = split_dataset.pop("test")
        _split = split_dataset
    return _split

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
    print("train:", len(s["train"]), "val:", len(s["val"]))
    print("HEAD 300:", s["train"][0]["text"][:300])
