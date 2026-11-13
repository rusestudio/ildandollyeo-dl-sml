import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PackedTokenDataset(Dataset):
    def __init__(self, bin_path, seq_len, dtype="uint16"):
        self.bin_path = str(Path(bin_path).resolve())
        self.seq_len = int(seq_len)
        if self.seq_len <= 0:
            raise ValueError("seq len must be positive")
        if dtype not in ("uint16", "uint32"):
            raise ValueError("Only uint16/uint32 token storage is supported")
        self.dtype = np.dtype(dtype)
        path = Path(self.bin_path)
        if not path.is_file():
            raise FileNotFoundError(f"Token file not found: {path}")
        if path.stat().st_size % self.dtype.itemsize:
            raise ValueError(f"Invalid token file byte length: {path}")
        self.num_tokens = path.stat().st_size // self.dtype.itemsize
        self.num_samples = max(0, (self.num_tokens - 1) // self.seq_len)
        self._tokens = None  # Open separately inside each DataLoader worker.

    def __len__(self):
        return self.num_samples

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_tokens"] = None  # Safe when Windows DataLoader uses 'spawn'.
        return state

    def __getitem__(self, index):
        if not isinstance(index, int):
            index = int(index)
        if index < 0:
            index += self.num_samples
        if index < 0 or index >= self.num_samples:
            raise IndexError(index)
        if self._tokens is None:
            self._tokens = np.memmap(self.bin_path, dtype=self.dtype, mode="r")
        start = index * self.seq_len
        # Force writable copies before handing arrays to torch (readonly memmap).
        x = np.array(self._tokens[start : start + self.seq_len], dtype=np.int64, copy=True)
        y = np.array(self._tokens[start + 1 : start + self.seq_len + 1], dtype=np.int64, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_packed_datasets(processed_dir="data/processed", seq_len=None):
    root = Path(processed_dir)
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Run Step 3 first. Missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual_seq_len = int(seq_len) if seq_len is not None else int(meta["seq_len"])
    train = PackedTokenDataset(root / "train.bin", actual_seq_len, meta["dtype"])
    val = PackedTokenDataset(root / "val.bin", actual_seq_len, meta["dtype"])
    if len(train) == 0 or len(val) == 0:
        raise ValueError("Not enough packed tokens for at least one sample per split")
    return train, val, meta


def build_dataloaders(processed_dir="data/processed", batch_size=4,
                      seq_len=None, num_workers=0, pin_memory=False):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    train_ds, val_ds, meta = load_packed_datasets(processed_dir, seq_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, meta
