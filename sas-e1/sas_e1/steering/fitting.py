"""Fitting the steering family: closed-form vector/operators and cached-activation adapters."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .operators import Steerer


# ---------------------------------------------------------------- moments

def moments(X: np.ndarray, ridge: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(len(X) - 1, 1)
    if ridge:
        cov = cov + ridge * np.eye(cov.shape[0])
    return mu.astype(np.float32), cov.astype(np.float32)


def truncate_rank(W: np.ndarray, rank: int) -> np.ndarray:
    if rank is None or rank <= 0:
        return np.zeros_like(W)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    r = min(rank, len(s))
    return ((U[:, :r] * s[:r]) @ Vt[:r]).astype(np.float32)


def _sqrtm_psd(S: np.ndarray, eps: float) -> np.ndarray:
    w, V = np.linalg.eigh(np.asarray(S, dtype=np.float64))
    w = np.clip(w, eps, None)
    return (V * np.sqrt(w)) @ V.T


# ---------------------------------------------------------------- closed forms

def fit_vector(mu_pos: np.ndarray, mu_neg: np.ndarray, name: str = "A") -> Steerer:
    """Arm A: classic difference-in-means steering vector (W = 0)."""
    b = (mu_pos - mu_neg).astype(np.float32)
    return Steerer(name=name, b=b, W=None, meta={"kind": "vector"})


def fit_random(d: int, delta: float, seed: int = 0) -> Steerer:
    """Arm R: random direction with matched effect norm."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d).astype(np.float32)
    v /= max(float(np.linalg.norm(v)), 1e-9)
    return Steerer(name="R", b=(v * float(delta)).astype(np.float32), W=None,
                   meta={"kind": "random", "seed": seed})


def fit_moment_operators(
    mu_pos: np.ndarray,
    cov_pos: np.ndarray,
    mu_neg: np.ndarray,
    cov_neg: np.ndarray,
    ridge: float = 1.0e-3,
    sv_clip: float = 10.0,
    ranks: Optional[list[int]] = None,
    include_full: bool = True,
) -> dict[str, Steerer]:
    """Arm F: closed-form mean+covariance-matching affine operator, rank-truncated.

    M* = Sigma_P^{1/2} Sigma_N^{-1/2}; W* = M* - I; W_r = trunc_r(W*);
    b_r = mu_P - (I + W_r) mu_N.
    (Representation-Surgery / Conceptor lineage, design doc §3.3.)
    """
    d = mu_pos.shape[0]
    sp = _sqrtm_psd(cov_pos, ridge)
    sn = _sqrtm_psd(cov_neg, ridge)
    M = sp @ np.linalg.inv(sn)
    # clip extreme singular values for stability
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s = np.clip(s, 1.0 / sv_clip, sv_clip)
    M = (U * s) @ Vt
    W = M - np.eye(d)

    out: dict[str, Steerer] = {}
    if include_full:
        b_full = (mu_pos - M @ mu_neg).astype(np.float32)
        out["Ffull"] = Steerer(name="Ffull", b=b_full, W=W.astype(np.float32),
                               meta={"kind": "closed_form", "rank": d})
    for r in ranks or []:
        Wr = truncate_rank(W, r)
        br = (mu_pos - (np.eye(d, dtype=np.float64) + Wr) @ mu_neg).astype(np.float32)
        out[f"F{r}"] = Steerer(name=f"F{r}", b=br, W=Wr.astype(np.float32),
                               meta={"kind": "closed_form", "rank": int(r)})
    return out


# ---------------------------------------------------------------- trained adapter

def fit_trained_adapter(
    X_neg: np.ndarray,
    mu_pos: np.ndarray,
    cov_pos: np.ndarray,
    rank: int,
    steps: int = 300,
    batch_size: int = 128,
    lr: float = 1.0e-3,
    lambda_cov: float = 1.0,
    seed: int = 0,
    standardize: bool = False,
    device: Optional[str] = None,
) -> Steerer:
    """Arm C: trained low-rank + bias, fit on cached layer-L activations only.

    L = ||mean(f(X_N)) - mu_P||^2 + lambda_cov * ||cov(f(X_N)) - Sigma_P||_F^2,
    f(h) = h + (h A) B^T + b, with B and b zero-initialized (identity at init).
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X = np.asarray(X_neg, dtype=np.float32)
    mu_t = np.asarray(mu_pos, dtype=np.float32).copy()
    cov_t = np.asarray(cov_pos, dtype=np.float32).copy()
    if standardize:
        m = X.mean(axis=0, keepdims=True)
        s = X.std(axis=0, keepdims=True) + 1e-5
        X = (X - m) / s
        mu_t = (mu_t - m[0]) / s[0]
        cov_t = cov_t / np.outer(s[0], s[0])

    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    mu_pt = torch.as_tensor(mu_t, dtype=torch.float32, device=device)
    cov_pt = torch.as_tensor(cov_t, dtype=torch.float32, device=device)
    n, d = Xt.shape

    A = (torch.randn(d, rank, device=device) / math.sqrt(d)).requires_grad_(True)
    B = torch.zeros(d, rank, device=device, requires_grad=True)
    b = torch.zeros(d, device=device, requires_grad=True)
    opt = torch.optim.AdamW([A, B, b], lr=lr)
    history: list[float] = []

    for _step in range(steps):
        idx = torch.as_tensor(rng.integers(0, n, size=min(batch_size, n)), device=device)
        h = Xt[idx]
        f = h + (h @ A) @ B.T + b
        mean_term = ((f.mean(dim=0) - mu_pt) ** 2).mean()
        fc = f - f.mean(dim=0, keepdim=True)
        cov = (fc.T @ fc) / max(f.shape[0] - 1, 1)
        cov_term = ((cov - cov_pt) ** 2).mean()
        loss = mean_term + lambda_cov * cov_term
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))

    W = (B.detach() @ A.detach().T).cpu().numpy().astype(np.float32)  # (d, d)
    bias = b.detach().cpu().numpy().astype(np.float32)
    return Steerer(
        name=f"C{rank}",
        b=bias,
        W=W,
        meta={
            "kind": "trained",
            "rank": int(rank),
            "seed": int(seed),
            "steps": int(steps),
            "final_loss": history[-1] if history else None,
            "loss_first": history[0] if history else None,
        },
        extra={"A": A.detach().cpu().numpy().astype(np.float32),
               "B": B.detach().cpu().numpy().astype(np.float32)},
    )


# ---------------------------------------------------------------- spectra helpers

def canonical_singular_values(W: np.ndarray) -> np.ndarray:
    """SVD of the (canonical) operator — the only meaningful spectrum of BA."""
    if W is None:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.svd(np.asarray(W, dtype=np.float64), compute_uv=False)
