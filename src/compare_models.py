"""Final comparison table across all four models.  (Owner: Aanyaa Agarwwal)

Reads results/final/<model>_seed<k>_results.json for every model and seed and writes
results/final/comparison.json with mean and standard deviation over seeds.

Run (after run_experiments.py):  python src/compare_models.py
"""
import glob
import json
import os

import numpy as np

from text_utils import CLASSES

MODELS = ["mlp", "cnn", "bilstm", "transformer"]
METRICS = ["test_acc", "macro_precision", "macro_recall", "macro_f1", "best_val_acc"]


def main(final_dir="results/final"):
    selection = json.load(open(os.path.join(final_dir, "selection.json")))
    table = {}
    for m in MODELS:
        runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(final_dir, f"{m}_seed*_results.json")))]
        if not runs:
            continue
        row = {"n_seeds": len(runs), "seeds": [r["hyperparams"]["seed"] for r in runs],
               "selected_config": selection.get(m, {}).get("selected"),
               "params": runs[0]["params"], "model_hp": runs[0]["model_hp"],
               "lr": runs[0]["hyperparams"]["lr"], "warmup": runs[0]["hyperparams"]["warmup"]}
        for k in METRICS:
            v = np.array([r[k] for r in runs])
            row[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "values": v.tolist()}
        row["per_class_f1_mean"] = {c: float(np.mean([r["per_class"][c]["f1"] for r in runs])) for c in CLASSES}
        sec_per_epoch = [np.mean([h["epoch_sec"] for h in r["history"]]) for r in runs]
        row["sec_per_epoch_mean"] = float(np.mean(sec_per_epoch))
        row["epochs_run"] = [r["epochs_run"] for r in runs]
        row["best_epoch"] = [r["best_epoch"] for r in runs]
        row["train_sec_mean"] = float(np.mean([r["train_sec"] for r in runs]))
        row["test_infer_sec_mean"] = float(np.mean([r["test_infer_sec"] for r in runs]))
        table[m] = row
    json.dump(table, open(os.path.join(final_dir, "comparison.json"), "w"), indent=2)
    print(f"{'model':12s} {'acc':>15s} {'macro-F1':>15s} {'params':>8s} {'s/epoch':>8s}")
    for m, r in table.items():
        print(f"{m:12s} {r['test_acc']['mean']*100:7.2f} ± {r['test_acc']['std']*100:4.2f}"
              f" {r['macro_f1']['mean']:8.4f} ± {r['macro_f1']['std']:.4f}"
              f" {r['params']/1e6:7.2f}M {r['sec_per_epoch_mean']:8.1f}")


if __name__ == "__main__":
    main()
