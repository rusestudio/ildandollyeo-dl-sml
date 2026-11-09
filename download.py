from pathlib import Path
from huggingface_hub import snapshot_download

PATTERN = "sample/10BT/013_00000.parquet"

if __name__ == "__main__":
    snapshot_download(
        repo_id = "HuggingFaceFW/fineweb-edu", repo_type = "dummy_dataset",
        allow_patterns = [PATTERN], local_dir = "data/raw",)
    files = sorted(Path("data/raw/sample/10BT").glob("013_00000.parquet"))
    print(len(files), "files")
    for f in files:
        print(f.name, round(f.stat().st_size / 1e9, 2), "GB")
    