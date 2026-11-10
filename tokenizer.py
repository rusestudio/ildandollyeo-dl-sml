from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from split_dataset import iter_texts

VOCAB_SIZE = 
MAX_DOCS =
MODEL_PATH = "data/tokenizer/tokenizer.json"
MUN_VAL_DOCS = 


if __name__ == "__main__":
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Sequence(
        [ByteLevel(add_prefix_space=False), Digits(individual_digits=True)]
    )
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = [AddedToken(special=True, normalized=False) for token in ["[PAD]", "[BOS]", "[EOS]", "[CLS]"]]

    sparse_tokens = [AddedToken(f"[UNUSED_{i}]", special=True, normalized=False) for i in range(5)]

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE, initial_alphabet=ByteLevel.alphabet(), 
        max_token_length=5, special_tokens=special_tokens+sparse_tokens
        )
    
    tokenizer.train_from_iterator(batch, )