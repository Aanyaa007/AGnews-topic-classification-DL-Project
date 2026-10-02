"""Model registry: one file per architecture family / team member."""
from .mlp import MLPClassifier
from .cnn import CNN1DClassifier
from .bilstm import BiLSTMClassifier
from .transformer import TransformerClassifier

MODELS = {
    "mlp": MLPClassifier,              # Model 1 - Siya Srivastava
    "cnn": CNN1DClassifier,            # Model 2 - Urvi Kedar Mapsenkar
    "bilstm": BiLSTMClassifier,        # Model 3 - Nehal Rai (draft)
    "transformer": TransformerClassifier,  # Model 4 - Aanyaa Agarwwal (draft)
}
