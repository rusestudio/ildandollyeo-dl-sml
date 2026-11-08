import os
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.base import PipelineStep
from tokenizers import Tokenizer
import numpy as np

LIMITS = 10_000



class CustomTokenizerStep(PipelineStep):
    def __init__(self, tokenizer_path: str, output_dir: str, chunk_size:int = 50_000):
        super().__init__()
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        os.makedirs(output_dir, exist_ok=True)

    def run(self, data, rank: int = 0, word_size: int = 1):
        #word_size 쟤는 worker 의 수와 같다보 보면 됨
        tokenizer = Tokenizer.from_file(self.tokenizer_path)

        all_tokens = []
        shard_counter = 0

        print(f"[Task {rank}] 토큰화 시작, 바이너리 샤딩")

        for doc in data:
            text = doc.text # 문서 text 가져옴
            if not text: # 근데 없으면 건너뜀
                continue

            
            token_ids = tokenizer.encode(text).ids
            token_ids.append(tokenizer.token_to_id("[EOS]")) # 문장 끝에 대해서 대비하기

            all_tokens.extend(token_ids) # 모든 token id 저장

            if len(all_tokens) >= self.chunk_size:
                arr = np.array(all_tokens, dtype=np.uint)
                shard_path = os.path.join(self.output_dir, f"shard_{rank}_{shard_counter:05d}.bin")
                arr.tofile(shard_path)

                del arr
                all_tokens = []
                shard_counter += 1

                

        if all_tokens:
            arr = np.array(all_tokens, dtype=np.uint32)
            shard_path = os.path.join(self.output_dir, f"shard_{rank}_{shard_counter:05d}.bin")
            arr.tofile(shard_path)
            
        print("샤딩 끝!!")
#저 if 문 부터 저게 뭐하는 애인지 설명하자면 모든 토큰은 한 청크로 잘라서 각 청크마다 다음과 같은 작업을 함
#그 후로는 샤딩한 경로를 따라 파일로 저장

if __name__=="__main__":

    os.environ["HF_DATASETS_CACHE"] = "./data/hf_cache"
    os.makedirs("./data/hf_cache", exist_ok=True)

    output_dir = "data/bin_shards"
    os.makedirs("./data/bin_shards", exist_ok=True)

    reader = HuggingFaceDatasetReader(
        dataset="HuggingFaceFW/fineweb-edu",
        dataset_options={"name": "sample-10BT", "split": "train"},
        text_key="text",
        streaming=True,
        limit=LIMITS
    )

    tokenizer_step = CustomTokenizerStep(
        tokenizer_path="models/tokenizer.json",
        output_dir=output_dir
    )

    executor = LocalPipelineExecutor(
        pipeline=[
            reader,
            tokenizer_step
        ],
        tasks=1,
        logging_dir="logs/datatrove"
    )

    print("샤딩 실행")
    executor.run()
    print("샤딩 끝")

        
