"""Vocabulary construction and integer encoding.  (Owner: Urvi Kedar Mapsenkar)"""
from collections import Counter

import numpy as np

PAD, UNK = "<pad>", "<unk>"


def build_vocab(token_lists, max_size=30000, min_freq=2):
    """Build word->id map from TRAINING tokens only. id 0 = <pad>, id 1 = <unk>."""
    counts = Counter(t for toks in token_lists for t in toks)
    words = [w for w, c in counts.most_common() if c >= min_freq][: max_size - 2]
    itos = [PAD, UNK] + words
    return {w: i for i, w in enumerate(itos)}, counts


def encode(token_lists, stoi, max_len=64):
    """Tokens -> padded/truncated id matrix (N, max_len) plus true lengths."""
    ids = np.zeros((len(token_lists), max_len), dtype=np.int32)       # 0 = <pad>
    lengths = np.zeros(len(token_lists), dtype=np.int32)
    unk = stoi[UNK]
    for i, toks in enumerate(token_lists):
        seq = [stoi.get(t, unk) for t in toks[:max_len]]
        ids[i, : len(seq)] = seq
        lengths[i] = max(len(seq), 1)
    return ids, lengths


def split_stats(token_lists, X, y, stoi, max_len):
    """Length / truncation / out-of-vocabulary statistics for one split."""
    raw_len = np.array([len(t) for t in token_lists])
    n_unk = int((X == stoi[UNK]).sum())
    return {
        "n": int(len(y)),
        "per_class": np.bincount(y, minlength=4).tolist(),
        "len_mean": float(raw_len.mean()), "len_median": float(np.median(raw_len)),
        "len_p95": float(np.percentile(raw_len, 95)), "len_max": int(raw_len.max()),
        "truncated_pct": float((raw_len > max_len).mean() * 100),
        "oov_pct": float(n_unk / max(raw_len.sum(), 1) * 100),
    }
