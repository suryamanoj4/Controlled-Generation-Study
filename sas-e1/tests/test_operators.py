import numpy as np
import pytest

from sas_e1.steering import calibration, fitting
from sas_e1.steering.operators import Steerer, no_op_steerer


def synthetic(seed: int = 0, n: int = 300, d: int = 16):
    rng = np.random.default_rng(seed)
    mu_n = np.zeros(d, dtype=np.float32)
    cov_n = np.eye(d, dtype=np.float32)
    mu_p = rng.normal(0, 0.4, size=d).astype(np.float32)
    cov_p = np.diag(np.linspace(1.0, 2.0, d)).astype(np.float32)
    return (
        rng.multivariate_normal(mu_n, cov_n, size=n).astype(np.float32),
        rng.multivariate_normal(mu_p, cov_p, size=n).astype(np.float32),
        mu_p, cov_p, mu_n, cov_n,
    )


def test_vector_fit():
    Xn, Xp, mu_p, _, mu_n, _ = synthetic()
    st = fitting.fit_vector(mu_p, mu_n)
    assert np.allclose(st.b, mu_p - mu_n, atol=1e-4)
    assert st.W is None


def test_moment_operator_covariance_matching():
    Xn, Xp, mu_p, cov_p, mu_n, cov_n = synthetic()
    ops = fitting.fit_moment_operators(mu_p, cov_p, mu_n, cov_n, ridge=1e-3, ranks=[1, 4])
    W = ops["Ffull"].W
    d = Xn.shape[1]
    map_cov = (np.eye(d) + W) @ (np.eye(d) + W).T
    rel = np.linalg.norm(map_cov - cov_p) / np.linalg.norm(cov_p)
    assert rel < 0.2


def test_rank_truncation():
    Xn, Xp, mu_p, cov_p, mu_n, cov_n = synthetic()
    ops = fitting.fit_moment_operators(mu_p, cov_p, mu_n, cov_n, ridge=1e-3, ranks=[2])
    sv = fitting.canonical_singular_values(ops["F2"].W)
    assert int((sv > 1e-6).sum()) == 2


def test_calibration_matches_effect_norm():
    Xn, Xp, mu_p, _, mu_n, _ = synthetic()
    A = fitting.fit_vector(mu_p, mu_n)
    delta = calibration.reference_delta(A)
    R = fitting.fit_random(Xn.shape[1], delta, seed=0)
    alpha = calibration.calibrate_alpha(R, Xn, delta)
    norm = np.linalg.norm(R.displacement(Xn, alpha), axis=1).mean()
    assert abs(norm - delta) < 1e-4


def test_no_op_and_roundtrip(tmp_path):
    X = np.random.default_rng(0).normal(size=(5, 8)).astype(np.float32)
    noop = no_op_steerer(8)
    assert np.allclose(noop.apply(X), X)
    st = Steerer(name="T", b=np.ones(8, dtype=np.float32), W=np.eye(8, dtype=np.float32) * 0.1)
    path = tmp_path / "t.npz"
    st.save(path)
    loaded = Steerer.load(path)
    assert np.allclose(loaded.apply(X), st.apply(X), atol=1e-6)


def test_trained_adapter_reduces_loss():
    torch = pytest.importorskip("torch")
    Xn, Xp, mu_p, cov_p, _, _ = synthetic()
    st = fitting.fit_trained_adapter(Xn, mu_p, cov_p, rank=2, steps=60, batch_size=64, seed=0)
    assert st.W.shape == (Xn.shape[1], Xn.shape[1])
    assert st.meta["final_loss"] < st.meta["loss_first"]
