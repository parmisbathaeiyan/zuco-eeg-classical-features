"""Match ZuCo sentences to their sentiment labels from the csv.

The .mat `content` strings and the csv `sentence` column are the same
sentences but not always byte-identical (stray whitespace, smart quotes), so we
match on a normalised form rather than relying on order.
"""

import re

import pandas as pd


def _normalise(text):
    text = str(text).lower().strip()
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)  # punctuation-insensitive
    return text.strip()


def load_label_lookup(csv_path):
    """Return `{normalised_sentence: (sentence_id, label)}` from the csv."""
    df = pd.read_csv(csv_path)
    df = df[["sentence_id", "sentence", "sentiment_label"]].dropna()
    lookup = {}
    for _, row in df.iterrows():
        key = _normalise(row["sentence"])
        if key:
            lookup[key] = (int(row["sentence_id"]), int(row["sentiment_label"]))
    return lookup


def match(content, lookup):
    """Look up one sentence; `(None, None)` if it isn't in the csv."""
    return lookup.get(_normalise(content), (None, None))
