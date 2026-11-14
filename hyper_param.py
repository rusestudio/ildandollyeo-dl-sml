import re
import sys
import subprocess
import itertools
import matplotlib.pyplot as plt

tuning_grid = {
    "learning_rate": [1e-4, 3e-4, 6e-4],
    "hidden_size": [64, 128],
    "dropout": [0.05, 0.1]
}

with open("config.py", "r", encoding="utf-8") as f:
    original_config = f.read()

def update_config_file(params_dict):
    modified_config = original_config
    for key, val in params_dict.items():
        pattern = rf"({key}\s*:\s*[^=]+=\s*)([^\n#]+)"
        modified_config = re.sub(pattern, rf"\g<1>{val}", modified_config)
    with open("config.py", "w", encoding="utf-8") as f:
        f.write(modified_config)

keys = list(tuning_grid.keys())
combinations = list(itertools.product(*tuning_grid.values()))
results = {}

try:
    for combo in combinations:
        current_params = dict(zip(keys, combo))
        exp_label = ", ".join([f"{k}={v}" for k, v in current_params.items()])
        print(f"하이퍼 파라미터 {exp_label}")

        update_config_file(current_params)

        process = subprocess.Popen(
            ["python", "-u", "kjs_tesk2.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1
        )

        steps = []
        val_losses = []

        for line in process.stdout:
            sys.stdout.write(line)
            
            match = re.search(r"step\s+(\d+):\s+val loss\s+([0-9.]+)", line)
            if match:
                steps.append(int(match.group(1)))
                val_losses.append(float(match.group(2)))

        process.wait()
        
        if process.returncode != 0:
            print("training 도중 오류 발생")
            break

        results[exp_label] = (steps, val_losses)

finally:
    with open("config.py", "w", encoding="utf-8") as f:
        f.write(original_config)


if results:
    plt.figure(figsize=(10, 6))
    for label, (steps, losses) in results.items():
        if steps and losses:
            plt.plot(steps, losses, marker="o", linewidth=2, label=label)

    plt.title("Hyperparameter Tuning (Validation Loss)", fontsize=14, fontweight="bold")
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("tuning_comparison.png", dpi=300)
    print("graph")
    plt.show()