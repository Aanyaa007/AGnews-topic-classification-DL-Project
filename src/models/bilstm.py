"""Model 3 - BiLSTM.  (Owner: Nehal Rai)

STATUS: draft. Forward pass verified; not yet trained or tuned (planned for 8 Oct).
"""
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMClassifier(nn.Module):
    """Embedding -> bidirectional LSTM -> max pooling over time -> linear."""

    def __init__(self, vocab_size, num_classes=4, emb_dim=128, hidden=128,
                 num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)        # skip padding steps
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        out = out.masked_fill((x == 0).unsqueeze(-1), -1e4)         # ignore padding in pooling
        pooled = out.max(dim=1).values                              # (B, 2H)
        return self.fc(self.dropout(pooled))
