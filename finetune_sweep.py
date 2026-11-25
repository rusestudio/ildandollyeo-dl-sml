import subprocess
import sys
import logging
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="conf", config_name="finetune_sweep")
def main(cfg: DictConfig) -> float:
    out_dir = HydraConfig.get().runtime.output_dir
    overrides = list(HydraConfig.get().overrides.task)
    # logging
    proc = subprocess.Popen(
        [sys.executable, "-u", "finetune.py", *overrides, f"paths.finetune_checkpoint={out_dir}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in proc.stdout:
        log.info(line.rstrip())
    if proc.wait():
        sys.exit(proc.returncode)
    
    best_val_acc = torch.load(f"{out_dir}/ckpt.pt", map_location="cpu")["best_val_acc"]
    print(f"{overrides} -> best val acc {best_val_acc:.4f}")
    return best_val_acc


if __name__ == "__main__":
    main()