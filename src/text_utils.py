"""Text loading, cleaning and tokenisation for AG News.  (Owner: Siya Srivastava)"""
import html
import re

import pandas as pd

CLASSES = ["World", "Sports", "Business", "Sci/Tech"]

_URL = re.compile(r"https?://\S+|www\.\S+")
_ENTITY = re.compile(r"&?#\d+;|&?\b(quot|amp|lt|gt|nbsp|apos);", re.I)
_NONWORD = re.compile(r"[^a-z0-9' ]+")
_SPACES = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def clean_text(text: str) -> str:
    """Normalise one raw AG News string."""
    text = html.unescape(text)
    text = text.replace("\\", " ")          # AG News stores newlines as backslashes
    text = re.sub(r"&?#39;", "'", text)     # escaped apostrophe -> keep "women's" readable
    text = _ENTITY.sub(" ", text)           # leftover entities such as 'quot;'
    text = _URL.sub(" ", text)
    text = text.lower()
    text = _NONWORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokenize(text: str):
    """Regex word tokeniser (letters/digits, keeps contractions)."""
    return _TOKEN.findall(text)


def load_split(path: str) -> pd.DataFrame:
    """Read an AG News CSV -> DataFrame with cleaned `text` and 0-based `label`."""
    df = pd.read_csv(path, header=None, names=["label", "title", "description"])
    df["label"] = df["label"] - 1                       # 1..4 -> 0..3
    df["text"] = (df["title"].fillna("") + " . " + df["description"].fillna("")).map(clean_text)
    return df[["text", "label"]]
