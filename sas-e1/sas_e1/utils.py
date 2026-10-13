"""Small shared utilities: seeding, IO, spectral entropies, text-diversity metrics."""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------- seeding / IO

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # torch not required for numpy-only utilities
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def mark_done(dir_path: str | Path, meta: dict | None = None) -> None:
    save_json(Path(dir_path) / "_done.json", meta or {"done": True})


def is_done(dir_path: str | Path) -> bool:
    return (Path(dir_path) / "_done.json").exists()


# ---------------------------------------------------------------- spectral metrics

def spectral_entropy(eigenvalues: Sequence[float] | np.ndarray) -> float:
    """Shannon entropy of the normalized eigenvalue distribution (nats)."""
    lam = np.asarray(eigenvalues, dtype=np.float64)
    lam = np.clip(lam, 0.0, None)
    total = lam.sum()
    if total <= 0:
        return 0.0
    p = lam / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def effective_rank(eigenvalues: Sequence[float] | np.ndarray) -> float:
    """exp(entropy) — how many directions carry the signal."""
    return float(math.exp(spectral_entropy(eigenvalues)))


def stable_rank(matrix: np.ndarray) -> float:
    m = np.asarray(matrix, dtype=np.float64)
    fro2 = float((m * m).sum())
    spectral = float(np.linalg.norm(m, ord=2))
    if spectral <= 0:
        return 0.0
    return fro2 / (spectral ** 2)


def topk_mass(eigenvalues: Sequence[float] | np.ndarray, k: int) -> float:
    lam = np.sort(np.clip(np.asarray(eigenvalues, dtype=np.float64), 0, None))[::-1]
    total = lam.sum()
    if total <= 0:
        return 0.0
    return float(lam[:k].sum() / total)


def safe_svd(matrix: np.ndarray, k: int | None = None) -> np.ndarray:
    """Singular values (descending), optionally truncated to k."""
    sv = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    return sv[:k] if k else sv


# ---------------------------------------------------------------- text diversity

def ttr(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def mtld(tokens: Sequence[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (simple bidirectional implementation)."""
    toks = list(tokens)
    if len(toks) < 10:
        return 0.0

    def _one(seq: Sequence[str]) -> float:
        types: set[str] = set()
        count = 0
        factors = 0.0
        for w in seq:
            count += 1
            types.add(w)
            if not types:
                continue
            if len(types) / count <= threshold:
                factors += 1.0
                types = set()
                count = 0
        if count > 0:
            t = len(types) / count
            if t < threshold:
                factors += (1.0 - t) / (1.0 - threshold)
        return factors

    f1 = _one(toks)
    f2 = _one(list(reversed(toks)))
    if f1 <= 0 or f2 <= 0:
        return 0.0
    return float((len(toks) / f1 + len(toks) / f2) / 2.0)


def ngram_repetition(text: str, n: int = 4) -> float:
    """Fraction of repeated n-grams (1 - unique/total) over whitespace tokens."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    grams: set[tuple] = set()
    total = 0
    for text in texts:
        words = text.split()
        for i in range(len(words) - n + 1):
            grams.add(tuple(words[i : i + n]))
            total += 1
    if total == 0:
        return 0.0
    return len(grams) / total


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)
