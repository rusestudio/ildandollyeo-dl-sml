# 하이퍼파라미터 튜닝 자동 실행
# A: dropout x weight_decay  ->  B: 배치 크기(같은 토큰 수)  ->  C: 1 epoch 최종 학습  ->  test 평가
import os
import re
import sys
import json
import time
import subprocess

import wandb

ROOT = "/home/hyeeon02/LLMCOURSE/ildandollyeo-dl-sml"
PY = "/home/hyeeon02/miniconda3/bin/python"
WANDB = "/home/hyeeon02/miniconda3/bin/wandb"
PROJECT = "ildandollyeo-dl-sml"
ENTITY = "hyeeon02-kangwon-national-university"
GPUS = [0, 1, 2, 3]
TUNE_DIR = os.path.join(ROOT, "tuning")
LOG_DIR = os.path.join(TUNE_DIR, "logs")
RESULTS = os.path.join(TUNE_DIR, "results.md")

BASE = dict(
    learning_rate=0.003,
    min_lr=0.0003,
    dropout=0.1,
    weight_decay=0.1,
    gradient_accumulation_steps=4,
    max_steps=1000,
    warmup_steps=100,
    eval_steps=250,
)


def fmt(v):
    return f"{v:.8f}" if isinstance(v, float) else str(v)


def label_of(stage, c):
    return (
        f"{stage}_lr{c['learning_rate']:g}_do{c['dropout']:g}_wd{c['weight_decay']:g}"
        f"_acc{c['gradient_accumulation_steps']}_{c['max_steps']}st"
    )


def create_sweep(label, cfg):
    path = os.path.join(LOG_DIR, f"{label}.yaml")
    lines = [
        "program: train.py",
        "method: grid",
        f"project: {PROJECT}",
        "metric:",
        "  name: val/loss",
        "  goal: minimize",
        "parameters:",
    ]
    for k, v in cfg.items():
        lines += [f"  {k}:", f"    value: {fmt(v)}"]
    lines += ["command:", "  - ${env}", f"  - {PY}", "  - ${program}"]
    open(path, "w").write("\n".join(lines) + "\n")
    out = subprocess.run(
        [WANDB, "sweep", path], cwd=ROOT, capture_output=True, text=True
    )
    return re.search(r"wandb agent (\S+)", out.stdout + out.stderr).group(1)


def parse_log(path):
    text = open(path).read()
    vals = re.findall(r"step (\d+): val loss ([\d.]+)", text)
    run_id = re.findall(r"/runs/(\w+)", text)
    ckpt = re.findall(r"saving checkpoint to (\S+)", text)
    return {
        "val_curve": [(int(s), float(v)) for s, v in vals],
        "val_loss": float(vals[-1][1]) if vals else None,
        "run_id": run_id[-1] if run_id else None,
        "ckpt_dir": os.path.join(ROOT, ckpt[-1]) if ckpt else None,
    }


def run_stage(stage, configs):
    print(f"\n===== stage {stage}: {len(configs)} runs =====", flush=True)
    queue = [(label_of(stage, c), c) for c in configs]
    running = {}  # gpu -> (label, cfg, proc)
    results = []
    while queue or running:
        for g in GPUS:
            if g not in running and queue:
                label, cfg = queue.pop(0)
                sweep_id = create_sweep(label, cfg)
                log = open(os.path.join(LOG_DIR, f"{label}.log"), "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                proc = subprocess.Popen(
                    [WANDB, "agent", sweep_id, "--count", "1"],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                running[g] = (label, cfg, proc)
                print(f"[{time.strftime('%H:%M')}] gpu{g} start {label}", flush=True)
        time.sleep(30)
        for g in list(running):
            label, cfg, proc = running[g]
            if proc.poll() is not None:
                r = parse_log(os.path.join(LOG_DIR, f"{label}.log"))
                r.update(label=label, cfg=cfg)
                results.append(r)
                del running[g]
                print(f"[{time.strftime('%H:%M')}] gpu{g} done  {label}  val {r['val_loss']}", flush=True)
                if r["run_id"]:
                    run = wandb.Api().run(f"{ENTITY}/{PROJECT}/{r['run_id']}")
                    run.name = label
                    run.update()
    ok = [r for r in results if r["val_loss"] is not None]
    ok.sort(key=lambda r: r["val_loss"])
    write_results(stage, ok, [r for r in results if r["val_loss"] is None])
    return ok


def write_results(stage, ok, failed):
    lines = [f"\n## stage {stage}  ({time.strftime('%m/%d %H:%M')})\n",
             "| rank | run | val loss | val curve |", "|---|---|---|---|"]
    for i, r in enumerate(ok):
        curve = " ".join(f"{s}:{v:.3f}" for s, v in r["val_curve"])
        lines.append(f"| {i + 1} | {r['label']} | {r['val_loss']:.4f} | {curve} |")
    for r in failed:
        lines.append(f"| - | {r['label']} | FAILED | |")
    open(RESULTS, "a").write("\n".join(lines) + "\n")
    json.dump(ok, open(os.path.join(LOG_DIR, f"stage_{stage}.json"), "w"), indent=2)


def run_test(r):
    out = subprocess.run(
        [PY, os.path.join(TUNE_DIR, "eval_test.py"), r["ckpt_dir"]],
        cwd=ROOT, capture_output=True, text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONPATH=ROOT),
    )
    open(os.path.join(LOG_DIR, "test_eval.log"), "w").write(out.stdout + out.stderr)
    print(out.stdout, flush=True)
    m = re.search(r"test loss ([\d.]+), test ppl ([\d.]+)", out.stdout)
    run = wandb.Api().run(f"{ENTITY}/{PROJECT}/{r['run_id']}")
    run.summary["test/loss"] = float(m.group(1))
    run.summary["test/ppl"] = float(m.group(2))
    run.summary.update()
    open(RESULTS, "a").write(f"\n## test (best = {r['label']})\n\n```\n{out.stdout}```\n")


def smoke():
    c = dict(BASE, max_steps=20, eval_steps=10, warmup_steps=5)
    r = run_stage("smoke", [c])
    print(r, flush=True)


def main():
    open(RESULTS, "a").write(f"\n# tuning started {time.strftime('%m/%d %H:%M')}\n")

    # A: dropout x weight_decay (lr 0.003, 1000 step)
    a = run_stage("A", [
        dict(BASE, dropout=do, weight_decay=wd)
        for do in [0.0, 0.1] for wd in [0.0, 0.1]
    ])
    best_a, second_a = a[0]["cfg"], a[1]["cfg"]

    # B: 배치 크기 (토큰 수 동일 = 32.8M), accum 4 는 A 의 best
    b_cfgs = []
    for acc, lr in [(2, 0.002), (2, 0.003), (8, 0.003), (8, 0.004)]:
        steps = 1000 * 4 // acc
        b_cfgs.append(dict(best_a, gradient_accumulation_steps=acc, learning_rate=lr,
                           min_lr=lr / 10, max_steps=steps, warmup_steps=steps // 10))
    b = run_stage("B", b_cfgs)
    best_b = min([a[0]] + b, key=lambda r: r["val_loss"])["cfg"]

    # C: 1 epoch (약 206M 토큰) 최종 학습
    acc = best_b["gradient_accumulation_steps"]
    lr = best_b["learning_rate"]
    steps = 6300 * 4 // acc
    full = dict(max_steps=steps, warmup_steps=200 * 4 // acc, eval_steps=500 * 4 // acc)
    c_cfgs = [
        dict(best_b, **full, learning_rate=round(l, 6), min_lr=round(l / 10, 7))
        for l in [lr * 2 / 3, lr, lr * 4 / 3]
    ]
    c_cfgs.append(dict(best_b, **full, dropout=second_a["dropout"],
                       weight_decay=second_a["weight_decay"]))
    c = run_stage("C", c_cfgs)

    # test: C 에서 val 이 가장 좋은 체크포인트 1개만
    run_test(c[0])
    print("all done", flush=True)


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke()
    else:
        main()
