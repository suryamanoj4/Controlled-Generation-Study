"""Behavioral metrics: strength, coherence, content preservation, failure regions."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..utils import distinct_n, ngram_repetition


# ---------------------------------------------------------------- primitives

def strength(classifier, texts: Sequence[str]) -> np.ndarray:
    return np.asarray(classifier.predict_proba_positive(list(texts)), dtype=np.float64)


def perplexity(model, tokenizer, texts: Sequence[str], batch_size: int = 8,
               max_length: int = 256, device=None) -> float:
    """Token-level perplexity of `texts` under the (frozen) base model."""
    import torch
    import torch.nn.functional as F

    device = device or next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + batch_size]]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_length).to(device)
            logits = model(**enc).logits[:, :-1].float()
            targets = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].float()
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
            ).view(targets.shape)
            total_nll += float((nll * mask).sum().cpu())
            total_tokens += int(mask.sum().cpu())
    if total_tokens == 0:
        return float("nan")
    return float(np.exp(total_nll / total_tokens))


def distinct_2(texts: Sequence[str]) -> float:
    return distinct_n(texts, n=2)


def repetition_4(text: str) -> float:
    return ngram_repetition(text, n=4)


def content_similarity_pairs(a: Sequence[str], b: Sequence[str]) -> np.ndarray:
    """Cosine similarity between paired steered/unsteered outputs.

    Uses sentence-transformers when available, else a TF-IDF char n-gram fallback.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        ea = model.encode(list(a), normalize_embeddings=True)
        eb = model.encode(list(b), normalize_embeddings=True)
        return (ea * eb).sum(axis=1)
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit(
            list(a) + list(b)
        )
        xa, xb = vec.transform(list(a)), vec.transform(list(b))
        num = np.asarray(xa.multiply(xb).sum(axis=1)).ravel()
        da = np.sqrt(np.asarray(xa.multiply(xa).sum(axis=1)).ravel())
        db = np.sqrt(np.asarray(xb.multiply(xb).sum(axis=1)).ravel())
        return num / np.clip(da * db, 1e-9, None)


# ---------------------------------------------------------------- curve summaries

def compute_sfwc(points: list[dict], baseline_coherence: float, budget: float = 0.9) -> dict:
    """Strength at the largest scale satisfying the coherence budget (primary scalar).

    `points`: [{"m": float, "strength": float, "coherence": float}, ...]
    """
    pts = sorted(points, key=lambda p: p["m"])
    threshold = budget * baseline_coherence
    eligible = [p for p in pts if p["coherence"] >= threshold]
    if eligible:
        best = max(eligible, key=lambda p: p["strength"])
        return {"sfwc": best["strength"], "m_star": best["m"], "fell_back": False}
    if not pts:
        return {"sfwc": None, "m_star": None, "fell_back": True}
    best = max(pts, key=lambda p: p["coherence"])
    return {"sfwc": best["strength"], "m_star": best["m"], "fell_back": True}


def alpha_target(points: list[dict], target_strength: float) -> Optional[float]:
    """Smallest m whose strength reaches `target_strength` (or None)."""
    for p in sorted(points, key=lambda p: p["m"]):
        if p["strength"] >= target_strength:
            return float(p["m"])
    return None


def alpha_break(points: list[dict], baseline_coherence: float, budget: float = 0.9) -> Optional[float]:
    """First m where coherence falls below the budget (failure region entry point)."""
    threshold = budget * baseline_coherence
    for p in sorted(points, key=lambda p: p["m"]):
        if p["coherence"] < threshold:
            return float(p["m"])
    return None
