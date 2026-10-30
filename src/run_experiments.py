"""Run the full Phase 3 experiment plan.  (Owner: Nehal Rai)

1. Tuning: every configuration in configs/tuning.json, seed 42, max 5 epochs.
2. Selection: for each model, the configuration with the highest VALIDATION accuracy.
3. Final runs: the selected configuration with seeds 42, 13 and 7.
Results go to results/tuning/ and results/final/. Finished runs are skipped, so the
script can be re-run after an interruption.

Run:  python src/run_experiments.py                 (all four models)
      python src/run_experiments.py --models bilstm (one model)
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(model, cfg, seed, out, tag):
    if os.path.exists(os.path.join(out, f"{tag}_results.json")):
        print(f"skip {tag} (done)", flush=True)
        return
    cmd = [sys.executable, os.path.join(HERE, "train.py"), "--model", model,
           "--hp", json.dumps(cfg["hp"]), "--lr", str(cfg["lr"]),
           "--warmup", str(cfg.get("warmup", 0)), "--epochs", "5", "--patience", "2",
           "--seed", str(seed), "--out", out, "--tag", tag]
    print("RUN", " ".join(cmd), flush=True)
    with open(os.path.join(out, f"{tag}.log"), "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mlp", "cnn", "bilstm", "transformer"])
    ap.add_argument("--config", default="configs/tuning.json")
    args = ap.parse_args()
    plan = json.load(open(args.config))
    os.makedirs("results/tuning", exist_ok=True)
    os.makedirs("results/final", exist_ok=True)

    selection = {}
    for model in args.models:
        for cfg in plan[model]:
            run(model, cfg, 42, "results/tuning", f"{model}_{cfg['name']}")
        scores = {c["name"]: json.load(open(f"results/tuning/{model}_{c['name']}_results.json"))["best_val_acc"]
                  for c in plan[model]}
        best = max(scores, key=scores.get)
        selection[model] = {"val_acc_by_config": scores, "selected": best}
        print(model, scores, "->", best, flush=True)
        cfg = next(c for c in plan[model] if c["name"] == best)
        for seed in plan["final_seeds"]:
            run(model, cfg, seed, "results/final", f"{model}_seed{seed}")
        path = "results/final/selection.json"
        old = json.load(open(path)) if os.path.exists(path) else {}
        old.update({model: selection[model]})
        json.dump(old, open(path, "w"), indent=2)


if __name__ == "__main__":
    main()
