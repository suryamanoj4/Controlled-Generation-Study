import numpy as np

from sas_e1.evaluation import metrics as M
from sas_e1.probes import spectra, stability
from sas_e1.steering import fitting
from sas_e1.utils import effective_rank, mtld, ngram_repetition, spectral_entropy, ttr


def test_entropy_and_erank():
    assert abs(spectral_entropy([1.0])) < 1e-12
    assert abs(effective_rank([1.0, 1.0]) - 2.0) < 1e-9


def test_text_diversity():
    assert abs(ttr(list("aabb")) - 0.5) < 1e-9
    assert mtld(("a b " * 16).split()) > 0  # repetitive input must yield a positive MTLD
    assert ngram_repetition("a b a b a b a b", n=2) > 0.5


def test_sfwc_and_failure_points():
    pts = [
        {"m": 0.5, "strength": 0.2, "coherence": 1.0},
        {"m": 1.0, "strength": 0.6, "coherence": 0.95},
        {"m": 2.0, "strength": 0.9, "coherence": 0.7},
    ]
    out = M.compute_sfwc(pts, baseline_coherence=1.0, budget=0.9)
    assert out["m_star"] == 1.0 and abs(out["sfwc"] - 0.6) < 1e-12 and not out["fell_back"]
    assert M.alpha_target(pts, 0.5) == 1.0
    assert M.alpha_break(pts, 1.0, 0.9) == 2.0


def test_action_spectrum_vector_vs_operator():
    rng = np.random.default_rng(0)
    d = 16
    Xn = rng.normal(size=(200, d)).astype(np.float32)
    b = rng.normal(size=d).astype(np.float32)
    vec = fitting.fit_vector(b, np.zeros(d, dtype=np.float32))
    s_vec = spectra.action_spectrum(vec, Xn)
    assert s_vec["erank"] < 1.5

    W = rng.normal(size=(d, d)).astype(np.float32) * 0.1
    op = fitting.fit_vector(np.zeros(d, dtype=np.float32), np.zeros(d, dtype=np.float32))
    op.W = W
    s_op = spectra.action_spectrum(op, Xn)
    assert s_op["erank"] > s_vec["erank"]


def test_stability_helpers():
    rng = np.random.default_rng(1)
    W = rng.normal(size=(8, 8))
    U = stability.top_directions(W, k=4)
    assert abs(stability.subspace_overlap(U, U) - 1.0) < 1e-6
    assert stability.sorted_spectrum_distance([2, 1], [2, 1]) < 1e-12
