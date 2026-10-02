"""Error analysis across trained models.  (Owner: Nehal Rai)

Reads the saved test predictions of every trained model and reports:
  - the most frequent confusion pairs per model
  - how many test samples the models get right / wrong together
  - example texts that every model misclassifies

Run (after training):  python src/error_analysis.py
"""
import json
import os

import numpy as np
import pandas as pd

from text_utils import CLASSES


def main(data="processed", out="results", models=("mlp", "cnn", "bilstm", "transformer")):
    test = pd.read_csv(os.path.join(data, "test_clean.csv"))
    y = test["label"].to_numpy()
    preds = {m: np.load(os.path.join(out, f"{m}_test_preds.npy"))
             for m in models if os.path.exists(os.path.join(out, f"{m}_test_preds.npy"))}
    report = {"models": list(preds), "n_test": int(len(y)), "per_model": {}}

    for m, p in preds.items():
        wrong = p != y
        pairs = {}
        for t, q in zip(y[wrong], p[wrong]):
            key = f"{CLASSES[t]} -> {CLASSES[q]}"
            pairs[key] = pairs.get(key, 0) + 1
        report["per_model"][m] = {
            "errors": int(wrong.sum()),
            "top_confusions": dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:4]),
            "business_scitech_errors": int(((y == 2) & (p == 3)).sum() + ((y == 3) & (p == 2)).sum()),
        }

    correct = np.stack([p == y for p in preds.values()])
    report["all_correct_pct"] = float(correct.all(0).mean() * 100)
    report["all_wrong_pct"] = float((~correct).all(0).mean() * 100)
    report["disagree_pct"] = 100 - report["all_correct_pct"] - report["all_wrong_pct"]

    shared = test[(~correct).all(0)].sample(10, random_state=1)
    report["shared_error_examples"] = [
        {"true": CLASSES[r.label],
         "predicted": {m: CLASSES[p[i]] for m, p in preds.items()},
         "text": r.text[:160]}
        for i, r in shared.iterrows()]

    with open(os.path.join(out, "error_analysis.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
