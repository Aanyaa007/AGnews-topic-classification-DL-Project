"""
Shared preprocessing pipeline for AG News (used by all four models).

1. Load CSVs, join title + description, clean text        (text_utils.py)
2. Stratified 90/10 train/validation split, seed 42; official test set untouched
3. Tokenise, build vocabulary from the TRAIN split only   (vocab.py)
4. Encode, pad/truncate to 64 tokens, save .npz + vocab.json + stats.json

Run:  python src/preprocess.py
"""
import argparse
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

from text_utils import load_split, tokenize
from vocab import build_vocab, encode, split_stats

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="processed")
    ap.add_argument("--val_size", type=float, default=0.10)
    ap.add_argument("--max_vocab", type=int, default=30000)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    full_train = load_split(os.path.join(args.data_dir, "train.csv"))
    test = load_split(os.path.join(args.data_dir, "test.csv"))
    train, val = train_test_split(full_train, test_size=args.val_size,
                                  stratify=full_train["label"], random_state=SEED)
    splits = [("train", train), ("val", val), ("test", test)]

    tok = {name: [tokenize(t) for t in df["text"]] for name, df in splits}
    stoi, counts = build_vocab(tok["train"], args.max_vocab, args.min_freq)   # train only

    stats = {"vocab_size": len(stoi), "max_len": args.max_len,
             "unique_train_tokens": len(counts)}
    for name, df in splits:
        X, L = encode(tok[name], stoi, args.max_len)
        y = df["label"].to_numpy(dtype=np.int64)
        np.savez_compressed(os.path.join(args.out_dir, f"{name}.npz"), X=X, L=L, y=y)
        df.to_csv(os.path.join(args.out_dir, f"{name}_clean.csv"), index=False)
        stats[name] = split_stats(tok[name], X, y, stoi, args.max_len)

    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(stoi, f)
    os.makedirs("results", exist_ok=True)
    for path in (os.path.join(args.out_dir, "stats.json"), "results/data_stats.json"):
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
