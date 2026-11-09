import json
import time
from pathlib import Path


def downloda_dataset():
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id = "HuggingFaceFW/fineweb-edu",
        repo_type = "dataset",
        allow_paatterns=["sample/10BT/*"],
        local_dir = "data/raw",
    )
   








def main():
    files = sorted(Path("data/raw/sample/10BT").glob("*.paraquet"))
    print(len(files))
    for f in files:
        print(f.name, round(f.stat().st_size / 1e9, 2), "GB")
if __name__ == "__main__":
    main()
