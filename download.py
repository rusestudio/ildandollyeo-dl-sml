from pathlib import Path
from huggingface_hub import snapshot_download

PATH = "sample/10BT/*.parquet"

if __name__ == "__main__":
    snapshot_download(
        repo_id = "HuggingFaceFW/fineweb-edu", repo_type = "dataset",
        allow_patterns = [PATH], local_dir = "data/raw",)
    files = sorted(Path("data/raw/sample/10BT").glob("*.parquet"))     # 우선 샘플 하나만
    print(len(files), "files")
    for f in files:
        print(f.name, round(f.stat().st_size / 1e9, 2), "GB")
    