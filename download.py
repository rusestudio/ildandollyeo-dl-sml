from pathlib import Path
from huggingface_hub import snapshot_download

from config import DataConfig

data_config = DataConfig()

if __name__ == "__main__":
    snapshot_download(
        repo_id="HuggingFaceFW/fineweb-edu",
        repo_type="dataset",
        allow_patterns=[data_config.download_pattern],
        local_dir=data_config.download_dir,
    )
    files = sorted(Path(data_config.raw_dir).glob("*.parquet"))
    print(len(files), "files")
    for f in files:
        print(f.name, round(f.stat().st_size / 1e9, 2), "GB")
