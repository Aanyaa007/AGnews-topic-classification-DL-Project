"""
Common training / evaluation script - identical protocol for every model.  (Owner: Nehal Rai)

Run:  python src/train.py --model mlp --epochs 6
      python src/train.py --model cnn --epochs 5
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, TensorDataset

from models import MODELS
from text_utils import CLASSES

SEED = 42


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def loader(path, bs, shuffle):
    d = np.load(path)
    ds = TensorDataset(torch.from_numpy(d["X"]).long(), torch.from_numpy(d["L"]).long(),
                       torch.from_numpy(d["y"]).long())
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


@torch.no_grad()
def predict(model, dl):
    model.eval()
    preds, ys, loss_sum, n = [], [], 0.0, 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for X, L, y in dl:
        out = model(X, L)
        loss_sum += crit(out, y).item(); n += len(y)
        preds.append(out.argmax(1)); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy(), loss_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--data", default="processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    set_seed(SEED)
    os.makedirs(args.out, exist_ok=True)

    vocab_size = len(json.load(open(os.path.join(args.data, "vocab.json"))))
    tr = loader(os.path.join(args.data, "train.npz"), args.bs, True)
    va = loader(os.path.join(args.data, "val.npz"), 512, False)
    te = loader(os.path.join(args.data, "test.npz"), 512, False)

    model = MODELS[args.model](vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    history, best_val, best_state, t_start = [], -1, None, time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot, correct, n = 0.0, 0, 0
        for X, L, y in tr:
            opt.zero_grad()
            out = model(X, L)
            loss = crit(out, y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
        yv, pv, vloss = predict(model, va)
        vacc = accuracy_score(yv, pv)
        history.append({"epoch": ep, "train_loss": tot / n, "train_acc": correct / n,
                        "val_loss": vloss, "val_acc": vacc, "epoch_sec": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)
        if vacc > best_val:                       # keep best epoch on validation (early-stopping checkpoint)
            best_val, best_state = vacc, {k: v.clone() for k, v in model.state_dict().items()}
    train_sec = time.time() - t_start

    model.load_state_dict(best_state)
    t0 = time.time(); yt, pt, tloss = predict(model, te); infer_sec = time.time() - t0
    p, r, f, _ = precision_recall_fscore_support(yt, pt, average="macro")
    pc = precision_recall_fscore_support(yt, pt, average=None)
    res = {
        "model": args.model, "params": n_params, "epochs": args.epochs,
        "best_epoch": int(np.argmax([h["val_acc"] for h in history]) + 1),
        "best_val_acc": best_val, "test_acc": accuracy_score(yt, pt), "test_loss": tloss,
        "macro_precision": p, "macro_recall": r, "macro_f1": f,
        "per_class": {c: {"precision": pc[0][i], "recall": pc[1][i], "f1": pc[2][i]}
                      for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(yt, pt).tolist(),
        "train_sec": train_sec, "test_infer_sec": infer_sec, "history": history,
        "hyperparams": vars(args),
    }
    json.dump(res, open(os.path.join(args.out, f"{args.model}_results.json"), "w"), indent=2)
    np.save(os.path.join(args.out, f"{args.model}_test_preds.npy"), pt)
    torch.save(best_state, os.path.join(args.out, f"{args.model}_best.pt"))
    print(json.dumps({k: v for k, v in res.items() if k not in ("history",)}, indent=2))


if __name__ == "__main__":
    main()
