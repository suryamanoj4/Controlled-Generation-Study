"""P5 — contrast separation pre-screen (Braun et al.): predict rank needs before fitting."""
from __future__ import annotations

import numpy as np


def contrast_separation(
    X_pos: np.ndarray,
    X_neg: np.ndarray,
    seed: int = 0,
    max_points: int = 2000,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, silhouette_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    X_pos = np.asarray(X_pos, dtype=np.float64)
    X_neg = np.asarray(X_neg, dtype=np.float64)

    def _sub(X):
        if len(X) > max_points:
            idx = rng.choice(len(X), size=max_points, replace=False)
            return X[idx]
        return X

    Xp, Xn = _sub(X_pos), _sub(X_neg)
    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(Xtr), ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(scaler.transform(Xte))[:, 1]))

    zp, zn = scaler.transform(Xp), scaler.transform(Xn)
    centroid_rms = float(np.linalg.norm(zp.mean(0) - zn.mean(0)) / np.sqrt(X.shape[1]))

    sil = None
    if len(X) >= 200:
        n_sil = min(len(X), 1200)
        idx = rng.choice(len(X), size=n_sil, replace=False)
        sil = float(silhouette_score(scaler.transform(X[idx]), y[idx]))

    return {
        "auc": auc,
        "centroid_rms_distance": centroid_rms,
        "silhouette": sil,
        "n_pos": int(len(Xp)),
        "n_neg": int(len(Xn)),
    }
