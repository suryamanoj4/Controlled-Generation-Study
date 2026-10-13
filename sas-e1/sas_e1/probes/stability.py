"""P4 — stability probes: subspace overlap, direction matching, split-half spectra."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..steering.operators import Steerer


def top_directions(W: np.ndarray, k: int = 8) -> np.ndarray:
    """Right singular vectors (input-space directions) of W, as columns."""
    if W is None:
        return np.zeros((0, 0))
    U, s, Vt = np.linalg.svd(np.asarray(W, dtype=np.float64), full_matrices=False)
    k = min(k, len(s))
    return Vt[:k].T


def principal_angles(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    if U1.size == 0 or U2.size == 0:
        return np.zeros(0)
    sv = np.linalg.svd(U1.T @ U2, compute_uv=False)
    return np.clip(sv, 0.0, 1.0)


def subspace_overlap(U1: np.ndarray, U2: np.ndarray) -> float:
    cos = principal_angles(U1, U2)
    return float((cos ** 2).mean()) if cos.size else 0.0


def hungarian_direction_overlap(W1: np.ndarray, W2: np.ndarray, k: int = 8) -> float:
    """PW-MCC-style: match top-k directions by |cosine| under optimal assignment."""
    from scipy.optimize import linear_sum_assignment  # local import

    U1, U2 = top_directions(W1, k), top_directions(W2, k)
    if U1.size == 0 or U2.size == 0:
        return 0.0
    cos = np.abs(U1.T @ U2)
    rows, cols = linear_sum_assignment(-cos)
    return float(cos[rows, cols].mean())


def sorted_spectrum_distance(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.sort(np.asarray(a, dtype=np.float64))[::-1]
    b = np.sort(np.asarray(b, dtype=np.float64))[::-1]
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    pa = np.pad(a, (0, n - len(a)))
    pb = np.pad(b, (0, n - len(b)))
    denom = max(np.linalg.norm(pa), np.linalg.norm(pb), 1e-12)
    return float(np.linalg.norm(pa - pb) / denom)


def vector_cosines(vectors: Sequence[np.ndarray]) -> list[float]:
    out = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, b = vectors[i], vectors[j]
            denom = max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
            out.append(float(np.dot(a, b) / denom))
    return out
