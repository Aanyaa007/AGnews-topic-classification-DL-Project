"""Model 2 - 1D CNN in the style of Kim (2014).  (Owner: Urvi Kedar Mapsenkar)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DClassifier(nn.Module):
    """Parallel Conv1d layers (kernel widths 3/4/5) act as learned n-gram detectors;
    max-over-time pooling keeps the strongest response of each filter."""

    def __init__(self, vocab_size, num_classes=4, emb_dim=128, n_filters=100,
                 kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x).transpose(1, 2)                     # (B, E, T)
        pad = (x == 0).unsqueeze(1)                                 # (B, 1, T)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))[:, :, : x.size(1)]
            c = c.masked_fill(pad, -1e4)                            # ignore padding positions
            pooled.append(c.max(dim=2).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))
