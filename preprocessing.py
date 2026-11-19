"""
preprocessing.py
Everything that touches raw text or the raw CSV lives here, so training and
prediction use EXACTLY the same cleaning pipeline.
"""
import re
import pickle
from collections import Counter

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. Text cleaning
# --------------------------------------------------------------------------
_URL = re.compile(r"https?://\S+|www\.\S+")
_REDDIT_REF = re.compile(r"(?<![\w])/?[ur]/\w+")
_NON_ALPHA = re.compile(r"[^a-z' ]+")
_SPACES = re.compile(r"\s+")

_CONTRACTIONS = [
    ("won't", "will not"), ("can't", "can not"), ("n't", " not"),
    ("'m", " am"), ("'re", " are"), ("'ve", " have"),
    ("'ll", " will"), ("'d", " would"), ("'s", ""),
]


def clean_text(text) -> str:
    """Lower-case, strip URLs / subreddit refs / symbols, expand contractions.
    Stop-words are deliberately KEPT: words like 'not', 'no', 'never', 'I'
    carry a lot of meaning in mental-health text."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    t = str(text).replace("\u2019", "'").replace("\u2018", "'").lower()
    t = _URL.sub(" ", t)
    t = _REDDIT_REF.sub(" ", t)
    for a, b in _CONTRACTIONS:
        t = t.replace(a, b)
    t = _NON_ALPHA.sub(" ", t)
    t = t.replace("'", "")
    return _SPACES.sub(" ", t).strip()


# --------------------------------------------------------------------------
# 2. Light-weight tokenizer + padding (no dependency on Keras internals, so
#    saved files keep working across TensorFlow versions)
# --------------------------------------------------------------------------
PAD, OOV = 0, 1


class SimpleTokenizer:
    def __init__(self, num_words=10000, word_index=None):
        self.num_words = num_words
        self.word_index = word_index or {}

    def fit(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        self.word_index = {w: i + 2 for i, (w, _) in
                           enumerate(counter.most_common(self.num_words - 2))}
        return self

    def texts_to_sequences(self, texts):
        wi = self.word_index
        return [[wi.get(w, OOV) for w in t.split()] for t in texts]

    def to_dict(self):
        return {"num_words": self.num_words, "word_index": self.word_index}

    @classmethod
    def from_dict(cls, d):
        return cls(d["num_words"], d["word_index"])


def pad_sequences(seqs, max_len):
    out = np.zeros((len(seqs), max_len), dtype="int32")
    for i, s in enumerate(seqs):
        s = s[:max_len]
        out[i, :len(s)] = s
    return out


def save_tokenizer(tok: SimpleTokenizer, path):
    with open(path, "wb") as f:
        pickle.dump(tok.to_dict(), f)


def load_tokenizer(path) -> SimpleTokenizer:
    with open(path, "rb") as f:
        return SimpleTokenizer.from_dict(pickle.load(f))


# --------------------------------------------------------------------------
# 3. Dataset loading and inspection
# --------------------------------------------------------------------------
DERIVED_LABEL = "derived_community"


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def _is_index_like(name):
    return str(name).lower().startswith("unnamed") or str(name).lower() in ("index", "id")


def _derive_block_labels(df):
    """If the CSV is several exports stacked on top of each other (index column
    restarts from 0), label each block. This is only a PROXY label (which
    community a post came from) - it is NOT a risk label."""
    for c in df.columns:
        if _is_index_like(c):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().all():
                restarts = int((s.diff() < 0).sum())
                if 1 <= restarts <= 20:
                    blocks = (s.diff() < 0).cumsum() + 1
                    return blocks.map(lambda b: f"block_{b}")
    return None


def detect_columns(df):
    text_c, label_c = [], []
    for c in df.columns:
        if _is_index_like(c) or c == DERIVED_LABEL:
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        nunique = s.nunique()
        if pd.api.types.is_numeric_dtype(s):
            if 2 <= nunique <= 30:
                label_c.append(c)
            continue
        avg_len = s.astype(str).str.len().mean()
        if avg_len > 30:
            text_c.append((c, avg_len))
        elif 2 <= nunique <= 30:
            label_c.append(c)
    text_c = [c for c, _ in sorted(text_c, key=lambda x: -x[1])]
    return text_c, label_c


def load_dataset(path):
    """Returns (DataFrame, meta). Adds a derived proxy label only when the file
    has no label column of its own."""
    df = _read_csv(path)
    text_c, label_c = detect_columns(df)
    derived = False
    if not label_c:
        blocks = _derive_block_labels(df)
        if blocks is not None:
            df[DERIVED_LABEL] = blocks
            derived = True
    return df, {"derived": derived}


def dataset_info(df, filename, derived):
    text_c, label_c = detect_columns(df)
    if derived:
        label_c = [DERIVED_LABEL]
    cols = []
    for c in df.columns:
        cols.append({
            "name": str(c),
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "unique": int(df[c].nunique()),
        })
    main_text = text_c[0] if text_c else None
    dup = int(df[main_text].dropna().duplicated().sum()) if main_text else 0
    sample = df.head(5).copy()
    for c in sample.columns:
        sample[c] = sample[c].astype(str).str.slice(0, 120)
    title_col = "title" if "title" in df.columns and "title" != main_text else (
        text_c[1] if len(text_c) > 1 else None)
    return {
        "filename": filename,
        "rows": int(len(df)),
        "columns": cols,
        "text_candidates": text_c,
        "label_candidates": label_c,
        "suggested_text": main_text,
        "suggested_title": title_col,
        "suggested_label": label_c[0] if label_c else None,
        "derived_label": DERIVED_LABEL if derived else None,
        "duplicate_text_rows": dup,
        "sample": sample.fillna("").to_dict(orient="records"),
    }


def class_counts(df, column):
    vc = df[column].dropna().astype(str).value_counts()
    return [{"value": k, "count": int(v)} for k, v in vc.items()]


# --------------------------------------------------------------------------
# 4. Building the training frame (clean -> filter -> de-duplicate)
# --------------------------------------------------------------------------
# Posts written by moderators or researchers rather than people describing
# their own state. Heuristic list; edit it to suit your dataset.
NOISE_PATTERNS = re.compile(
    r"check-?in (?:post|thread)|megathread|welcome to /?r/|announcement|"
    r"\bama with\b|participants? (?:needed|wanted)|looking for participants|"
    r"recruiting participants|(?:conducting|doing) (?:a )?(?:research|study|survey)|\birb\b|"
    r"our rules",
    re.I,
)

RISKY_NAME, NORMAL_NAME = "Potentially Risky", "Normal "


def build_training_frame(df, text_col, label_col, title_col=None,
                         risky_classes=None, filter_noise=True, min_words=5):
    """Returns (frame[clean,label], report). The label column is only used as
    the target - it never becomes an input feature."""
    rep = {"rows_in": int(len(df))}
    title = df[title_col].fillna("").astype(str) if title_col else ""
    body = df[text_col].fillna("").astype(str)
    raw = (title + ". " + body) if title_col else body
    work = pd.DataFrame({"raw": raw, "label": df[label_col]})

    before = len(work)
    work = work[work["label"].notna()]
    rep["removed_missing_label"] = int(before - len(work))

    if filter_noise:
        before = len(work)
        work = work[~work["raw"].str.contains(NOISE_PATTERNS)]
        rep["removed_noise_posts"] = int(before - len(work))
    else:
        rep["removed_noise_posts"] = 0

    work["clean"] = work["raw"].map(clean_text)
    before = len(work)
    work = work[work["clean"].str.split().str.len() >= min_words]
    rep["removed_too_short"] = int(before - len(work))

    # labels -> strings (binary risk mapping if requested)
    work["label"] = work["label"].astype(str)
    if risky_classes:
        risky = set(map(str, risky_classes))
        work["label"] = np.where(work["label"].isin(risky), RISKY_NAME, NORMAL_NAME)

    # Duplicate-text leakage prevention: remove duplicates BEFORE splitting.
    before = len(work)
    conflict = work.groupby("clean")["label"].nunique()
    conflict = set(conflict[conflict > 1].index)
    rep["removed_conflicting_labels"] = int(work["clean"].isin(conflict).sum())
    work = work[~work["clean"].isin(conflict)]
    n1 = len(work)
    work = work.drop_duplicates("clean")
    rep["removed_duplicates"] = int(n1 - len(work))
    rep["rows_final"] = int(len(work))
    rep["class_counts"] = work["label"].value_counts().to_dict()
    return work[["clean", "label"]].reset_index(drop=True), rep
