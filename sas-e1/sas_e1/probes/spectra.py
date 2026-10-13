"""Complexity probes: action spectrum (primary) and parameter spectrum (secondary)."""
from __future__ import annotations

import math

import numpy as np

from ..steering.fitting import canonical_singular_values
from ..steering.operators import Steerer
from ..utils import effective_rank, spectral_entropy, stable_rank, topk_mass


def _eig_metrics(eigenvalues: np.ndarray) -> dict:
    ev = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    total = float(ev.sum())
    p = ev / total if total > 0 else ev
    return {
        "H": spectral_entropy(ev),
        "erank": effective_rank(ev),
        "stable_rank": stable_rank(np.diag(np.sqrt(ev))) if total > 0 else 0.0,
        "top1_mass": topk_mass(ev, 1),
        "top2_mass": topk_mass(ev, 2),
        "top4_mass": topk_mass(ev, 4),
        "n_directions_99": int(np.searchsorted(np.cumsum(p), 0.99) + 1) if total > 0 else 0,
    }


def action_spectrum(steerer: Steerer, X: np.ndarray, alpha: float = 1.0, top_k: int = 64) -> dict:
    """P2 — entropy of the displacement second moment E[dh dh^T] on a neutral corpus.

    This probe is parametrization-free: a rank-1 vector shows effective rank ~1 by
    construction (the sanity check), while operators reveal how many directions they use.
    """
    D = steerer.displacement(X, alpha=alpha)
    if D.size == 0:
        return {"n": 0}
    C = (D.T @ D) / D.shape[0]
    ev = np.linalg.eigvalsh(C)[::-1]
    metrics = _eig_metrics(ev)
    metrics.update(
        {
            "n": int(D.shape[0]),
            "mean_displacement": float(np.linalg.norm(D, axis=1).mean()),
            "cv_displacement": float(
                np.linalg.norm(D, axis=1).std() / max(np.linalg.norm(D, axis=1).mean(), 1e-9)
            ),
            "eigenvalues": ev[:top_k],
        }
    )
    return metrics


def parameter_spectrum(steerer: Steerer) -> dict:
    """P3 — canonical SVD spectrum (W2T): meaningful only for the canonical operator W."""
    W = steerer.W
    out: dict = {
        "has_operator": W is not None,
        "bias_norm": float(np.linalg.norm(steerer.b)),
    }
    if W is None:
        out.update({"rank": 1, "H_param": None, "erank_param": None, "bias_energy_fraction": 1.0})
        return out
    sv = canonical_singular_values(W)
    ev = sv ** 2
    out.update(
        {
            "rank": int((sv > 1e-8).sum()),
            "H_param": spectral_entropy(ev),
            "erank_param": effective_rank(ev),
            "top1_mass": topk_mass(ev, 1),
            "bias_energy_fraction": float(
                out["bias_norm"] ** 2 / (out["bias_norm"] ** 2 + float((W.astype(np.float64) ** 2).sum()) + 1e-12)
            ),
            "singular_values": sv[:64],
        }
    )
    return out
