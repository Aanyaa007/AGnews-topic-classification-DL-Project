"""Model 1 - MLP baseline.  (Owner: Siya Srivastava)"""
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """Mean-pooled word embeddings -> fully connected layers.

    Averaging the embeddings of the non-padding tokens gives a fixed-size,
    order-independent document vector (a dense bag of words).
    """

    def __init__(self, vocab_size, num_classes=4, emb_dim=128, hidden=256, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        mask = (x != 0).unsqueeze(-1).float()                         # (B, T, 1)
        emb = self.embedding(x) * mask
        doc = emb.sum(1) / lengths.clamp(min=1).unsqueeze(1).float()  # mean pooling
        h = self.dropout(F.relu(self.fc1(doc)))
        return self.fc2(h)
