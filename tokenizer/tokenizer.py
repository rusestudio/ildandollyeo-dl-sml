import os
import time
import heapq
import json
from collections import Counter, defaultdict
import regex as re

SPECIALS = ["<|endoftext|>", "<|pad|>"]
PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def count_words(texts):
    c = Counter()
    for t in texts:
        c.update(PAT.findall(t))
    return c


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train(word_counts, num_merges, log_step=100):
    words = {tuple(w.encode("utf-8")): n for w, n in word_counts.items()}
    merges = []
    start = time.time()
    for k in range(num_merges):
        pairs = Counter()
        for w, n in words.items():
            for p in zip(w, w[1:]):
                pairs[p] += n       # 연결 쌍 카운팅
        if not pairs:
            break

        best = min(pairs, key=lambda p: (-pairs[p], p)) 
        if pairs[best] < 2:
            break

        merges.append(best)         # merge
        words = {tuple(merge(w, best, 256 + k)): n for w, n in words.items()}

        if (k + 1) % log_step == 0:
            elapsed = time.time() - start
            remain = elapsed / (k + 1)*(num_merges - k - 1)
            print(f"{k + 1}/{num_merges} merges, {elapsed:.0f}초 경과, 약 {remain / 60:.0f}분 남음")
    return merges


# Tokenizer
class Tokenizer:
    def __init__(self, merges):
        self.merges = [tuple(p) for p in merges]
        self.rank = {p: i for i, p in enumerate(self.merges)}   # 먼저 배운 쌍에 작은 번호 할당
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        base = 256 + len(self.merges)
        self.special = {tok: base + i for i, tok in enumerate(SPECIALS)}   
        self.special_inv = {i: tok for tok, i in self.special.items()}
        self.vocab_size = base + len(SPECIALS)
        self.special_pat = re.compile("(" + "|".join(re.escape(t) for t in SPECIALS) + ")")

    @property
    def eot_id(self):
        return self.special["<|endoftext|>"]

    @property
    def pad_id(self):
        return self.special["<|pad|>"]

    # special token 없을 때
    def _encode_ordinary(self, text):
        ids_all = []
        for piece in PAT.findall(text):
            ids = list(piece.encode("utf-8"))
            while len(ids) >= 2:
                pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))
                if pair not in self.rank:
                    break
                ids = merge(ids, pair, 256 + self.rank[pair])
            ids_all.extend(ids)
        return ids_all

    # special token 있을 때
    def encode(self, text, allow_special=False):
        if not allow_special:
            return self._encode_ordinary(text)       
        out = []
        for part in self.special_pat.split(text):    
            if part in self.special:
                out.append(self.special[part])
            elif part:
                out.extend(self._encode_ordinary(part))
        return out

    def encode_doc(self, text):
        return self.encode(text) + [self.special["<|endoftext|>"]]          

    def decode(self, ids):
        data = b"".join(self.special_inv[i].encode() if i in self.special_inv else self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": self.merges, "specials": list(self.special)}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["merges"])