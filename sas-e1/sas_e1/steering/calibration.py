"""Effect-norm alpha calibration (design doc §3.7).

Raw alpha is not comparable across steerer families; we calibrate each steerer so that
the mean displacement norm on the neutral corpus equals the vector arm's mean ||b||.
For linear (unclipped) operators the mapping is exact and closed-form.
"""
from __future__ import annotations

import numpy as np

from .operators import Steerer


def reference_delta(vector_steerer: Steerer) -> float:
    return float(np.linalg.norm(vector_steerer.b))


def calibrate_alpha(steerer: Steerer, X_ref: np.ndarray, delta_ref: float) -> float:
    D = steerer.displacement(X_ref, alpha=1.0)
    mean_norm = float(np.linalg.norm(D, axis=1).mean()) if D.size else 0.0
    if mean_norm <= 0:
        return 1.0
    return float(delta_ref / mean_norm)
