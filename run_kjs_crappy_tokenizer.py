import os
from datasets import load_dataset
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

LEN_OF_SUB_DATASET = 1_000_000

if __name__ == "__main__":
    print("huggingface fineweb edu 10 bt 중에서 일부 로드 하기")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )

    sub_dataset = dataset.take(LEN_OF_SUB_DATASET)

    #tokenizer setting 과정
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Sequence(
        [ByteLevel(add_prefix_space=False), #Byte 단위로 text 변환, 
         Digits(individual_digits=True)
        ])
    tokenizer.post_processor = ByteLevelProcessor(tirm_offsets=False)
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = [AddedToken(token, special=True, normalized=True) 
                      for token in ["[PAD]", "[BOS]", "[EOS]"]]

    special_tokens += [AddedToken(f"[UNUSED]_{i}", special=True, normalized=False) for i in range(10)]
    #UNUSED 저거에서 10은 예비토큰수임: 

    #token 학습하기
    trainer = BpeTrainer(
        vocab_size=20000,
        initial_alphabet=ByteLevel.alphabet(),
        max_token_length=10,
        special_tokens=special_tokens,
    )

    def batch_iterator(batch_size=1000, max_docs=100_000):
        batch = []
        doc_count = 0

        for ex in sub_dataset:
            batch.append(ex["text"])
            doc_count += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
            if doc_count >= max_docs:
                break
        if batch:
            yield batch

    print("BPE 토크나이저 ")
    tokenizer.train_from_iterator(batch_iterator(batch_size=1000, max_docs=100_000), trainer=trainer)

    model_path = "models/tokenizer.json"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    tokenizer.save(model_path)

    print("tokenizer 끝")