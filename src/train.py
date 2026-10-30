"""
Common training / evaluation script - identical protocol for every model.  (Owner: Nehal Rai)

Examples
  python src/train.py --model mlp
  python src/train.py --model bilstm --hp '{"hidden": 256}' --tag bilstm_h256 --no_test
  python src/train.py --model transformer --lr 5e-4 --warmup 500 --seed 7

Protocol (same for all models): Adam, batch 128, cross-entropy, gradient-norm clipping 1.0,
at most --epochs epochs with early stopping on validation accuracy (--patience),
best-validation checkpoint restored before the single test-set evaluation.
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


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def loader(path, bs, shuffle, seed=0):
    d = np.load(path)
    ds = TensorDataset(torch.from_numpy(d["X"]).long(), torch.from_numpy(d["L"]).long(),
                       torch.from_numpy(d["y"]).long())
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g)


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
    ap.add_argument("--hp", default="{}", help="JSON dict of model constructor arguments")
    ap.add_argument("--tag", default=None, help="output file prefix (default: model name)")
    ap.add_argument("--data", default="processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=6, help="maximum epochs")
    ap.add_argument("--patience", type=int, default=2, help="early-stopping patience (epochs)")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=0, help="linear learning-rate warm-up steps")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_test", action="store_true", help="tuning run: skip the test set")
    args = ap.parse_args()
    hp = json.loads(args.hp)
    tag = args.tag or args.model
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    vocab_size = len(json.load(open(os.path.join(args.data, "vocab.json"))))
    tr = loader(os.path.join(args.data, "train.npz"), args.bs, True, args.seed)
    va = loader(os.path.join(args.data, "val.npz"), 512, False)

    model = MODELS[args.model](vocab_size, **hp)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / args.warmup) if args.warmup else 1.0)
    crit = nn.CrossEntropyLoss()

    history, best_val, best_state, bad, t_start = [], -1, None, 0, time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot, correct, n = 0.0, 0, 0
        for X, L, y in tr:
            opt.zero_grad()
            out = model(X, L)
            loss = crit(out, y)
            loss.backward()
            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); sched.step()
            tot += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
        yv, pv, vloss = predict(model, va)
        vacc = accuracy_score(yv, pv)
        history.append({"epoch": ep, "train_loss": tot / n, "train_acc": correct / n,
                        "val_loss": vloss, "val_acc": vacc, "epoch_sec": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)
        if vacc > best_val:                       # best-validation checkpoint
            best_val, bad = vacc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:              # early stopping
                break
    train_sec = time.time() - t_start

    res = {"model": args.model, "tag": tag, "params": n_params, "model_hp": hp,
           "epochs_run": len(history),
           "best_epoch": int(np.argmax([h["val_acc"] for h in history]) + 1),
           "best_val_acc": best_val, "train_sec": train_sec, "history": history,
           "hyperparams": vars(args)}

    if not args.no_test:
        model.load_state_dict(best_state)
        te = loader(os.path.join(args.data, "test.npz"), 512, False)
        t0 = time.time(); yt, pt, tloss = predict(model, te); infer_sec = time.time() - t0
        p, r, f, _ = precision_recall_fscore_support(yt, pt, average="macro")
        pc = precision_recall_fscore_support(yt, pt, average=None)
        res.update({
            "test_acc": accuracy_score(yt, pt), "test_loss": tloss,
            "macro_precision": p, "macro_recall": r, "macro_f1": f,
            "per_class": {c: {"precision": pc[0][i], "recall": pc[1][i], "f1": pc[2][i]}
                          for i, c in enumerate(CLASSES)},
            "confusion_matrix": confusion_matrix(yt, pt).tolist(),
            "test_infer_sec": infer_sec})
        np.save(os.path.join(args.out, f"{tag}_test_preds.npy"), pt)
        torch.save(best_state, os.path.join(args.out, f"{tag}_best.pt"))

    json.dump(res, open(os.path.join(args.out, f"{tag}_results.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k not in ("history",)}, indent=2))


if __name__ == "__main__":
    main()
