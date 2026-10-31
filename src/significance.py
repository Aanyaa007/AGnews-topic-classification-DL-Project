"""Pairwise McNemar tests between models.  (Owner: Urvi Kedar Mapsenkar)

Tests whether two models' error rates on the same test samples differ more than
chance would explain (continuity-corrected McNemar chi-square, 1 d.o.f.),
using the seed-42 final run of each model.

Run (after run_experiments.py):  python src/significance.py
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import chi2

MODELS = ["mlp", "cnn", "bilstm", "transformer"]


def mcnemar(correct_a, correct_b):
    b = int((correct_a & ~correct_b).sum())     # A right, B wrong
    c = int((~correct_a & correct_b).sum())     # A wrong, B right
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return b, c, stat, float(chi2.sf(stat, 1))


def main(data="processed", final_dir="results/final"):
    y = pd.read_csv(os.path.join(data, "test_clean.csv"))["label"].to_numpy()
    correct = {m: np.load(os.path.join(final_dir, f"{m}_seed42_test_preds.npy")) == y
               for m in MODELS if os.path.exists(os.path.join(final_dir, f"{m}_seed42_test_preds.npy"))}
    rows = []
    for a, b in itertools.combinations(correct, 2):
        n01, n10, stat, p = mcnemar(correct[a], correct[b])
        rows.append({"model_a": a, "model_b": b, "a_right_b_wrong": n01, "a_wrong_b_right": n10,
                     "chi2": round(stat, 3), "p_value": p, "significant_0.05": p < 0.05})
    json.dump(rows, open(os.path.join(final_dir, "significance.json"), "w"), indent=2)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
