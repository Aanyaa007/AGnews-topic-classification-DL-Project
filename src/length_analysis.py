"""Accuracy by input length for every model.  (Owner: Siya Srivastava)

Splits the test set into length buckets (number of tokens before truncation) and
reports each model's accuracy per bucket, using the seed-42 final run.

Run (after run_experiments.py):  python src/length_analysis.py
"""
import json
import os

import numpy as np
import pandas as pd

from text_utils import tokenize

MODELS = ["mlp", "cnn", "bilstm", "transformer"]
BUCKETS = [(0, 29), (30, 39), (40, 49), (50, 10_000)]


def main(data="processed", final_dir="results/final"):
    test = pd.read_csv(os.path.join(data, "test_clean.csv"))
    y = test["label"].to_numpy()
    lengths = np.array([len(tokenize(t)) for t in test["text"]])
    out = {"buckets": []}
    for lo, hi in BUCKETS:
        mask = (lengths >= lo) & (lengths <= hi)
        label = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
        row = {"bucket": label, "n": int(mask.sum())}
        for m in MODELS:
            p = os.path.join(final_dir, f"{m}_seed42_test_preds.npy")
            if os.path.exists(p):
                pred = np.load(p)
                row[m] = float((pred[mask] == y[mask]).mean())
        out["buckets"].append(row)
    json.dump(out, open(os.path.join(final_dir, "length_analysis.json"), "w"), indent=2)
    print(pd.DataFrame(out["buckets"]).to_string(index=False))


if __name__ == "__main__":
    main()
