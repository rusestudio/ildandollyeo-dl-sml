import time
from pathlib import Path
from dataset.split_dataset import iter_texts
from tokenizer.tokenizer import count_words, train, Tokenizer, SPECIALS

VOCAB_SIZE = 4096
NUM_MERGES = VOCAB_SIZE - 256 - len(SPECIALS)      # 바이트 256개 + specail 토큰을 뺀 나머지
MAX_DOCS = 20_00 
LOG_STEP = 100
SAVE_PATH = "tokenizer/bpe.json"
NUM_VAL_DOCS = 500

if __name__ == "__main__":
    # 1) 단어 조각 만들기
    t = time.time()
    counts = count_words(iter_texts("train", max_docs=MAX_DOCS))
    print(f"Piece {len(counts):,}개 ({time.time() - t:.0f}초)")

    # 2) BPE 학습
    t = time.time()
    merges = train(counts, NUM_MERGES, log_step=LOG_STEP)
    tok = Tokenizer(merges)
    print(f"Finshed train: merges {len(merges):,}개, vocab {tok.vocab_size:,} ({time.time() - t:.0f}초)")

    # 3) 저장 후 다시 불러와서 같은지 확인
    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    tok.save(SAVE_PATH)
    tok = Tokenizer.load(SAVE_PATH)
    print(f"Save and load: {SAVE_PATH}")

    # 4) 특수 토큰 확인
    print(f"special token 개수: eot_id={tok.eot_id}, pad_id={tok.pad_id}")
    sample = "Hello world.<|endoftext|>Next doc"
    print("    allow_special=False:", tok.eot_id in tok.encode(sample))                      
    print("    allow_special=True :", tok.eot_id in tok.encode(sample, allow_special=True))   
    print("    encode_doc 끝 토큰:", tok.encode_doc("Hi")[-1] == tok.eot_id)                

    # 5) val 문서로 원문 복원 확인
    ok, n_docs, n_bytes, n_tok = 0, 0, 0, 0
    for text in iter_texts("val", max_docs=NUM_VAL_DOCS):
        ids = tok.encode(text)
        ok += tok.decode(ids) == text
        n_docs += 1
        n_bytes += len(text.encode("utf-8"))
        n_tok += len(ids)
    print(f"복원 성공 {ok}/{n_docs}, 토큰당 {n_bytes / n_tok:.2f} 바이트")