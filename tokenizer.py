import sys
from pathlib import Path

from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from config.setting import load_config
from split_dataset import iter_texts


if __name__ == "__main__":
    # QWen2 방식을 참고했음, 숫자를 한 자리씩 나누어서 학습. -> 다만 숫자가 많이 나오는 테스크에 불리함
    cfg = load_config(sys.argv[1:])

    tokenizer = Tokenizer(BPE())

    tokenizer.pre_tokenizer = Sequence(
        [Digits(individual_digits=True), ByteLevel(add_prefix_space=False)]
    )
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = [
        AddedToken(token, special=True, normalized=False) 
        for token in ["[PAD]", "[BOS]", "[EOS]", "[CLS]"]
        ]

    reserved_tokens = [
        AddedToken(f"[UNUSED_{i}]", special=True, normalized=False) 
        for i in range(5)
        ]

    trainer = BpeTrainer(
        vocab_size=cfg.tokenizer.vocab_size, min_frequency=cfg.tokenizer.min_frequency,
        max_token_length=cfg.tokenizer.max_token_length, initial_alphabet=ByteLevel.alphabet(),
        special_tokens=special_tokens + reserved_tokens,
    )

    tokenizer.train_from_iterator(iter_texts(cfg, "train"), trainer=trainer)

    output_path = Path(cfg.paths.tokenizer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Tokenzier path: {output_path}")