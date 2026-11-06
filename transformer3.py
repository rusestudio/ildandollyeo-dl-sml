from datasets import load_dataset


DATASET_NAME = 'HuggingFaceFW/fineweb-edu'
DATASET_CONFIG = 'sample-10BT'


def document_stream(limit, min_chars=100):
    """Fresh streaming iterator for each pass; no pretrained tokenizer/model."""
    dataset = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split='train', streaming=True)
    count = 0
    for example in dataset:
        text = example.get('text') or ''
        if len(text.strip()) < min_chars:
            continue
        yield text
        count += 1
        if limit is not None and count >= limit:
            break

