"""Model 4 - Transformer encoder.  (Owner: Aanyaa Agarwwal)

STATUS: draft. Forward pass verified; not yet trained or tuned (planned for 12 Oct).
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))                 # (1, T, D)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    """Embedding + positions -> N self-attention encoder layers -> mean pooling -> linear."""

    def __init__(self, vocab_size, num_classes=4, d_model=128, nhead=4, num_layers=2,
                 dim_ff=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        pad = x == 0                                                # (B, T) True at padding
        h = self.pos(self.embedding(x) * math.sqrt(self.d_model))
        h = self.encoder(self.dropout(h), src_key_padding_mask=pad)
        h = h.masked_fill(pad.unsqueeze(-1), 0.0)
        doc = h.sum(1) / lengths.clamp(min=1).unsqueeze(1).float()  # mean over real tokens
        return self.fc(self.dropout(doc))
