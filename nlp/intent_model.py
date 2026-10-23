"""Trained intent classifier (TF-IDF word + character n-grams -> logistic regression).

Character n-grams make it tolerant to typos; day names, clock times and numbers are
masked so the model learns *sentence shape* instead of memorising "Monday".
It is trained on first use from data/intent_train.jsonl (about a second), so no
pickle files have to match your scikit-learn version.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

TRAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "intent_train.jsonl"

_DAY = re.compile(r"\b(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)(?:day|days|s)?\b", re.I)
_TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b|\bnoon\b|\bmidday\b", re.I)
_NUM = re.compile(r"\b\d+(?:st|nd|rd|th)?\b")


def mask(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = _TIME.sub(" timetok ", t)
    t = _DAY.sub(" daytok ", t)
    t = _NUM.sub(" numtok ", t)
    return re.sub(r"\s+", " ", t).strip()


class IntentModel:
    _shared = None

    def __init__(self, rows: list):
        self.pipe = Pipeline([
            ("feats", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
            ])),
            ("clf", LogisticRegression(C=20, max_iter=2000)),
        ])
        self.pipe.fit([mask(r["text"]) for r in rows], [r["intent"] for r in rows])
        self.classes = list(self.pipe.classes_)

    @classmethod
    def from_file(cls, path=TRAIN_PATH):
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        return cls(rows)

    @classmethod
    def shared(cls):
        if cls._shared is None:
            cls._shared = cls.from_file()
        return cls._shared

    def predict_proba(self, text: str) -> dict:
        p = self.pipe.predict_proba([mask(text)])[0]
        return {c: float(x) for c, x in zip(self.classes, p)}
