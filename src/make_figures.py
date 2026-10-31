"""Figures for the final report.  (Owner: Aanyaa Agarwwal)

fig1_comparison.png   test accuracy (mean ± std over 3 seeds) and accuracy vs. training cost
fig2_curves.png       training / validation accuracy per epoch (seed 42)
fig3_confusion.png    test-set confusion matrices (seed 42)
fig4_length.png       test accuracy by input length

Run (after run_experiments.py, compare_models.py, length_analysis.py):
    python src/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

FINAL, OUT = "results/final", "figures"
MODELS = ["mlp", "cnn", "bilstm", "transformer"]
NAME = {"mlp": "MLP", "cnn": "1D CNN", "bilstm": "BiLSTM", "transformer": "Transformer"}
COL = {"mlp": "#2a78d6", "cnn": "#eb6834", "bilstm": "#1baf7a", "transformer": "#eda100"}
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def load(path):
    return json.load(open(path))


def fig_comparison(comp):
    ms = [m for m in MODELS if m in comp]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    acc = [comp[m]["test_acc"]["mean"] * 100 for m in ms]
    sd = [comp[m]["test_acc"]["std"] * 100 for m in ms]
    x = np.arange(len(ms))
    ax[0].bar(x, acc, width=0.6, color=[COL[m] for m in ms], edgecolor="white", linewidth=2)
    ax[0].errorbar(x, acc, yerr=sd, fmt="none", ecolor=INK, capsize=3, lw=1)
    for xi, a, s in zip(x, acc, sd):
        ax[0].text(xi, a + s + 0.08, f"{a:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    lo = min(acc) - 1.5
    ax[0].set_ylim(lo, max(acc) + 0.8)
    ax[0].set_xticks(x, [NAME[m] for m in ms])
    ax[0].set_ylabel("Test accuracy (%)")
    ax[0].set_title("(a) Test accuracy, mean ± s.d. over 3 seeds", fontsize=9)
    ax[0].grid(axis="x", visible=False)

    for m in ms:
        ax[1].scatter(comp[m]["sec_per_epoch_mean"], comp[m]["test_acc"]["mean"] * 100, s=60,
                      color=COL[m], edgecolor="white", linewidth=1.5, zorder=3)
        ax[1].annotate(NAME[m], (comp[m]["sec_per_epoch_mean"], comp[m]["test_acc"]["mean"] * 100),
                       textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Training time per epoch (s, log scale, 2-core CPU)")
    ax[1].set_ylabel("Test accuracy (%)")
    ax[1].set_title("(b) Accuracy vs. training cost", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_comparison.png"), dpi=220)


def fig_curves():
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=False)
    for m in MODELS:
        p = os.path.join(FINAL, f"{m}_seed42_results.json")
        if not os.path.exists(p):
            continue
        h = load(p)["history"]
        e = [r["epoch"] for r in h]
        ax[0].plot(e, [r["val_acc"] * 100 for r in h], "-o", color=COL[m], lw=2, ms=4, label=NAME[m])
        ax[1].plot(e, [r["train_acc"] * 100 for r in h], "--o", color=COL[m], lw=1.5, ms=3.5, label=NAME[m])
    ax[0].set(xlabel="Epoch", ylabel="Accuracy (%)")
    ax[1].set(xlabel="Epoch", ylabel="Accuracy (%)")
    ax[0].set_title("(a) Validation accuracy", fontsize=9)
    ax[1].set_title("(b) Training accuracy", fontsize=9)
    for a in ax:
        a.set_xticks(range(1, 6))
    ax[0].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_curves.png"), dpi=220)


def fig_confusion():
    cmap = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#104281"])
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.4))
    for a, m in zip(axes.ravel(), MODELS):
        p = os.path.join(FINAL, f"{m}_seed42_results.json")
        if not os.path.exists(p):
            a.axis("off"); continue
        r = load(p)
        cm = np.array(r["confusion_matrix"])
        a.imshow(cm, cmap=cmap, vmin=0, vmax=1900)
        a.grid(False)
        for i in range(4):
            for j in range(4):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                       color="white" if cm[i, j] > 900 else INK)
        a.set_xticks(range(4), CLASSES, fontsize=7.5)
        a.set_yticks(range(4), CLASSES, fontsize=7.5)
        a.set_xlabel("Predicted", fontsize=8)
        a.set_ylabel("True", fontsize=8)
        a.set_title(f"{NAME[m]} (accuracy {r['test_acc']*100:.2f}%)", fontsize=9)
        for s in a.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_confusion.png"), dpi=220)


def fig_length():
    p = os.path.join(FINAL, "length_analysis.json")
    if not os.path.exists(p):
        return
    b = load(p)["buckets"]
    labels = [f"{r['bucket']}\n(n={r['n']})" for r in b]
    x = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(6.0, 2.5))
    for m in MODELS:
        if m in b[0]:
            ax.plot(x, [r[m] * 100 for r in b], "-o", color=COL[m], lw=2, ms=4, label=NAME[m])
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_xlabel("Input length (tokens)")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(fontsize=7.5, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_length.png"), dpi=220)


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_comparison(load(os.path.join(FINAL, "comparison.json")))
    fig_curves()
    fig_confusion()
    fig_length()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
