#!/usr/bin/env python
"""Numpy-only numerical smoke test for the SAS-E1 core (no downloads required).

Checks: operator fitting/rank behavior, alpha calibration, action spectra,
separation probe, matching protocol, curve summaries, and serialization.
Torch-dependent parts (trained adapter) are skipped if torch is unavailable.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sas_e1.data.corpora import match_corpora  # noqa: E402
from sas_e1.evaluation import metrics as M  # noqa: E402
from sas_e1.probes import separation as sep  # noqa: E402
from sas_e1.probes import spectra, stability  # noqa: E402
from sas_e1.steering import calibration, fitting  # noqa: E402
from sas_e1.steering.operators import Steerer, no_op_steerer  # noqa: E402
from sas_e1.utils import effective_rank, mtld, spectral_entropy, ttr  # noqa: E402

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILED.append(name)


def synthetic(seed: int = 0, n: int = 400, d: int = 32):
    rng = np.random.default_rng(seed)
    mu_n = np.zeros(d, dtype=np.float32)
    cov_n = np.eye(d, dtype=np.float32)
    mu_p = rng.normal(0, 0.5, size=d).astype(np.float32)
    scale = np.linspace(1.0, 2.5, d).astype(np.float32)
    cov_p = np.diag(scale)
    Xn = rng.multivariate_normal(mu_n, cov_n, size=n).astype(np.float32)
    Xp = rng.multivariate_normal(mu_p, cov_p, size=n).astype(np.float32)
    return Xp, Xn, mu_p, cov_p, mu_n, cov_n


def main() -> int:
    Xp, Xn, mu_p, cov_p, mu_n, cov_n = synthetic()

    # ---- closed-form vector
    A = fitting.fit_vector(mu_p, mu_n)
    check("vector b = mu_p - mu_n", np.allclose(A.b, mu_p - mu_n, atol=1e-5))

    # ---- closed-form moment operators
    ops = fitting.fit_moment_operators(mu_p, cov_p, mu_n, cov_n, ridge=1e-3, sv_clip=10.0,
                                       ranks=[1, 4], include_full=True)
    check("operator family present", set(ops) == {"Ffull", "F1", "F4"})
    # the affine map should map the negative covariance to the positive covariance
    W = ops["Ffull"].W
    map_cov = (np.eye(d := Xn.shape[1]) + W) @ (np.eye(d) + W).T
    rel_cov = np.linalg.norm(map_cov - cov_p) / np.linalg.norm(cov_p)
    check("Ffull matches positive covariance", rel_cov < 0.2, f"rel_err={rel_cov:.3f}")
    sv1 = fitting.canonical_singular_values(ops["F1"].W)
    check("F1 is rank-1 (one nonzero sv)", int((sv1 > 1e-6).sum()) == 1)

    # ---- random control + calibration
    delta_ref = calibration.reference_delta(A)
    R = fitting.fit_random(Xn.shape[1], delta_ref, seed=0)
    alpha = calibration.calibrate_alpha(R, Xn, delta_ref)
    mean_disp = np.linalg.norm(R.displacement(Xn, alpha), axis=1).mean()
    check("calibration matches effect norm", abs(mean_disp - delta_ref) < 1e-4,
          f"{mean_disp:.4f} vs {delta_ref:.4f}")

    # ---- action spectra
    s_vec = spectra.action_spectrum(A, Xn, alpha=A.alpha_unit)
    s_op = spectra.action_spectrum(ops["Ffull"], Xn, alpha=1.0)
    check("vector action is rank-1 by construction", s_vec["erank"] < 1.5,
          f"erank={s_vec['erank']:.2f}")
    check("operator action is higher-rank", s_op["erank"] > s_vec["erank"] + 0.5,
          f"erank={s_op['erank']:.2f}")

    # ---- parameter spectrum
    p = spectra.parameter_spectrum(ops["F4"])
    check("parameter spectrum rank <= 4", p["rank"] <= 4, f"rank={p['rank']}")

    # ---- separation probe (sklearn optional)
    try:
        separation = sep.contrast_separation(Xp, Xn, seed=0)
        check("separation AUC high on synthetic", separation["auc"] > 0.9,
              f"AUC={separation['auc']:.3f}")
    except ImportError:
        print("[SKIP] separation probe (scikit-learn not installed)")

    # ---- matching protocol
    pos = [{"text": "a b c", "n_tokens": 50 + (i % 7)} for i in range(100)]
    neg = [{"text": "d e f", "n_tokens": 51 + (i % 9)} for i in range(120)]
    pos_m, neg_m, diag = match_corpora(pos, neg, seed=0)
    t_pos = sum(d["n_tokens"] for d in pos_m)
    t_neg = sum(d["n_tokens"] for d in neg_m)
    check("matched totals close", abs(t_pos - t_neg) <= max(d["n_tokens"] for d in neg_m),
          f"{t_pos} vs {t_neg}")
    check("diagnostics present", {"ks_pvalue", "ttr_delta"} <= set(diag))

    # ---- metric helpers
    check("entropy of rank-1 spectrum = 0", abs(spectral_entropy([1.0])) < 1e-12)
    check("erank of uniform 2-spectrum = 2", abs(effective_rank([1.0, 1.0]) - 2.0) < 1e-9)
    check("mtld positive", mtld(("the quick brown fox jumps over the lazy dog " * 3).split()) > 0)
    check("ttr sane", abs(ttr(list("aabb")) - 0.5) < 1e-9)
    pts = [{"m": 0.5, "strength": 0.2, "coherence": 1.0},
           {"m": 1.0, "strength": 0.6, "coherence": 0.95},
           {"m": 2.0, "strength": 0.9, "coherence": 0.7}]
    sfwc = M.compute_sfwc(pts, baseline_coherence=1.0, budget=0.9)
    check("SFWC picks coherent best point", sfwc["m_star"] == 1.0 and abs(sfwc["sfwc"] - 0.6) < 1e-9)
    check("alpha_target works", M.alpha_target(pts, 0.5) == 1.0)
    check("alpha_break works", M.alpha_break(pts, 1.0, 0.9) == 2.0)

    # ---- stability helpers
    check("subspace overlap self = 1", abs(stability.subspace_overlap(
        stability.top_directions(ops["F4"].W), stability.top_directions(ops["F4"].W)) - 1.0) < 1e-6)
    dist = stability.sorted_spectrum_distance([3, 2, 1], [3, 2, 1])
    check("spectrum distance self = 0", dist < 1e-12)

    # ---- serialization round trip
    tmp = pathlib.Path("/tmp/sas_e1_smoke_steerer.npz")
    ops["F4"].save(tmp)
    loaded = Steerer.load(tmp)
    check("steerer round-trip", np.allclose(loaded.apply(Xn[:3]), ops["F4"].apply(Xn[:3]), atol=1e-5))
    check("no-op is identity", np.allclose(no_op_steerer(8).apply(np.ones((3, 8))), 1.0))

    # ---- trained adapter (torch optional)
    try:
        import torch  # noqa: F401

        st = fitting.fit_trained_adapter(Xn, mu_p, cov_p, rank=2, steps=80, batch_size=64, seed=0)
        check("trained adapter exists", st.W is not None and st.W.shape == (32, 32))
        check("trained loss decreases", st.meta["final_loss"] < st.meta["loss_first"],
              f"{st.meta['loss_first']:.3f} -> {st.meta['final_loss']:.3f}")
    except ImportError:
        print("[SKIP] trained adapter (torch not installed)")

    print()
    if FAILED:
        print(f"SMOKE TEST FAILED: {len(FAILED)} check(s): {', '.join(FAILED)}")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
