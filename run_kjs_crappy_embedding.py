import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MemmapDataset(Dataset):
    def __init__(self, shard_dir, seq_len=128):
        self.seq_len = seq_len
        shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.bin")))

        if not shard_files:
            raise FileNotFoundError(f"{shard_dir}경로에 binard shard 가 없음")

        self.mmap_data_list = []
        self.total_tokens = 0

        for fpath in shard_files:
            arr = np.memmap(fpath, dtype=np.uint32, mode='r')
            self.mmap_data_list.append(arr)
            self.total_tokens += len(arr)

        print(f"{len(shard_files)}샤드 로드됨, 총 토큰 수: {self.total_tokens}")

    def __len__(self):
        if self.total_tokens <= self.seq_len:
            return 0
        return (self.total_tokens - self.seq_len - 1) // self.seq_len + 1

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        chunk = self._get_tokens(start_idx, end_idx)

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)

        return x, y

    def _get_tokens(self, start, end):
        current_pos = 0
        result = []

        for mmap_arr in self.mmap_data_list:
            arr_len = len(mmap_arr)
            if (current_pos + arr_len > start) and (current_pos < end):
                local_start = max(0, start - current_pos)
                local_end = min(arr_len, end - current_pos)
                result.append(mmap_arr[local_start:local_end])

            current_pos += arr_len

            if current_pos >= end:
                break
        if not result:
            raise IndexError(f"Requested token range [{start}, {end}) is out of bounds for the available shards.")
        return np.concatenate(result)

class TokenPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embed, block_size, dropout=0.1, pad_id=0):
        super().__init__()
        self.token = nn.Embedding(vocab_size, n_embed, padding_idx=pad_id)
        self.position = nn.Embedding(block_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids):
        b, t = ids.shape
        positions = torch.arange(t, device=ids.device)
        return self.dropout(self.token(ids) + self.position(positions))

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("그냥 임베딩")

    vocab_size = 20_000 # 추후에 입력 파일 제공 예정
    block_size = 128
    n_embed = 256

    dataset = MemmapDataset(shard_dir="data/bin_shards", seq_len=block_size)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    embedding_layer = TokenPositionEmbedding(vocab_size, n_embed, block_size).to(device)
    optimizer = optim.AdamW(embedding_layer.parameters(), lr=1e-4)

    print("시작")

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        embedded_output = embedding_layer(x)
        print(f"입력 shape: {x.shape}")
        print(f"결과 shape: {embedded_output.shape}")
    print("임베딩 정상(?)종료")