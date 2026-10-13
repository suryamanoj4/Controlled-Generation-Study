"""Attribute classifiers: TF-IDF stylometry and frozen-embedding logistic regression.

Never evaluate style with the steered model's own family; these are external instruments.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def encode_texts(
    encoder,
    tokenizer,
    texts: Sequence[str],
    device=None,
    batch_size: int = 16,
    max_length: int = 256,
) -> np.ndarray:
    """Mean-pooled last-hidden-state embeddings from a frozen encoder."""
    import torch

    device = device or next(encoder.parameters()).device
    encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)
            h = encoder(**enc).last_hidden_state.float()
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.append(pooled.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


class EmbeddingLRClassifier:
    """Frozen encoder embeddings + logistic regression (sentiment / toxicity)."""

    def __init__(self, encoder_name: str = "distilbert-base-uncased", max_length: int = 256,
                 device: str | None = None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = AutoModel.from_pretrained(encoder_name, torch_dtype=torch.float32).to(self.device)
        self.max_length = max_length
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, C=1.0)
        self.metadata: dict = {"encoder": encoder_name}

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "EmbeddingLRClassifier":
        X = encode_texts(self.encoder, self.tokenizer, texts, self.device, max_length=self.max_length)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, np.asarray(labels))
        return self

    def predict_proba_positive(self, texts: Sequence[str]) -> np.ndarray:
        X = encode_texts(self.encoder, self.tokenizer, texts, self.device, max_length=self.max_length)
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]


class TfidfStyleClassifier:
    """Character n-gram TF-IDF + logistic regression (stylometry-style instrument)."""

    def __init__(self, seed: int = 0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=200_000,
            lowercase=True, sublinear_tf=True,
        )
        self.clf = LogisticRegression(max_iter=2000, C=4.0, random_state=seed)
        self.metadata: dict = {"kind": "tfidf_char"}

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "TfidfStyleClassifier":
        X = self.vectorizer.fit_transform(list(texts))
        self.clf.fit(X, np.asarray(labels))
        return self

    def predict_proba_positive(self, texts: Sequence[str]) -> np.ndarray:
        X = self.vectorizer.transform(list(texts))
        if X.shape[1] == 0:
            return np.full(len(texts), 0.5)
        return self.clf.predict_proba(X)[:, 1]


def evaluate_classifier(clf, texts: Sequence[str], labels: Sequence[int]) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    labels = np.asarray(labels)
    scores = clf.predict_proba_positive(texts)
    preds = (scores >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "n": int(len(labels)),
    }
    if len(set(labels.tolist())) > 1:
        out["auroc"] = float(roc_auc_score(labels, scores))
    return out
