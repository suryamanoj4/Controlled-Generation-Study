"""Analysis: SFWC/r*/complexity tables, gates G0–G4, figures, markdown report."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import get
from ..probes.spectra import parameter_spectrum  # noqa: F401  (kept for API symmetry)
from ..steering.operators import Steerer
from ..utils import bootstrap_ci, load_json, read_jsonl, save_json
from .pipeline import activations_path, attr_dir, run_root, _attr_keys


def _load_conditions(cfg, attr: str) -> list[dict]:
    path = attr_dir(cfg, attr, "eval") / "conditions.jsonl"
    return read_jsonl(path) if path.exists() else []


def _arm_curve(rows: list[dict], arm: str) -> list[dict]:
    pts = [r for r in rows if r["arm"] == arm]
    if not pts:
        return []
    out = []
    for m in sorted({r["m"] for r in pts}):
        group = [r for r in pts if r["m"] == m]
        out.append(
            {
                "m": m,
                "strength": float(np.mean([g["strength_mean"] for g in group])),
                "strength_std": float(np.std([g["strength_mean"] for g in group])),
                "coherence": float(np.nanmean([g["coherence"] for g in group])),
                "ppl_ratio": float(np.nanmean([g["ppl_ratio"] for g in group])),
                "content_sim": float(np.nanmean([g["content_sim_mean"] for g in group])),
                "distinct2": float(np.nanmean([g["distinct2"] for g in group])),
                "rep4": float(np.nanmean([g["rep4_mean"] for g in group])),
            }
        )
    return out


def _sfwc(points: list[dict], budget: float = 0.9) -> dict:
    pts = [p for p in points if np.isfinite(p.get("strength", np.nan)) and np.isfinite(p.get("coherence", np.nan))]
    if not pts:
        return {"sfwc": None, "m_star": None, "fell_back": True}
    eligible = [p for p in pts if p["coherence"] >= budget]
    if eligible:
        best = max(eligible, key=lambda p: p["strength"])
        return {"sfwc": best["strength"], "m_star": best["m"], "fell_back": False}
    best = max(pts, key=lambda p: p["coherence"])
    return {"sfwc": best["strength"], "m_star": best["m"], "fell_back": True}


def _r_star(sfwc_by_rank: dict[int, float]) -> int | None:
    vals = {k: v for k, v in sfwc_by_rank.items() if v is not None}
    if not vals:
        return None
    best = max(vals.values())
    for k in sorted(vals):
        if vals[k] >= 0.95 * best:
            return int(k)
    return max(vals)


def analyze(cfg) -> Path:
    out = run_root(cfg) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    attrs = _attr_keys(cfg)
    summary: dict = {"attributes": {}}

    for attr in attrs:
        rows = _load_conditions(cfg, attr)
        if not rows:
            print(f"[analyze] no conditions for {attr} — skipped")
            continue
        entry: dict = {"curves": {}, "sfwc": {}, "level": getattr(cfg.attributes, attr).level,
                       "kind": getattr(cfg.attributes, attr).kind}
        arms = sorted({r["arm"] for r in rows}, key=lambda a: (len(a), a))
        for arm in arms:
            curve = _arm_curve(rows, arm)
            entry["curves"][arm] = curve
            if arm not in ("E", "P"):
                entry["sfwc"][arm] = _sfwc(curve)
        # r* over closed-form ranks
        sfwc_by_rank_c = {}
        for rank in (1, 4, 16):
            key = f"F{rank}"
            if key in entry["sfwc"] and not entry["sfwc"][key].get("fell_back", False):
                sfwc_by_rank_c[rank] = entry["sfwc"][key]["sfwc"]
        entry["r_star_closed_form"] = _r_star(sfwc_by_rank_c)
        sfwc_by_rank_t = {}
        for rank in (1, 4):
            key = f"C{rank}"
            if key in entry["sfwc"] and not entry["sfwc"][key].get("fell_back", False):
                sfwc_by_rank_t[rank] = entry["sfwc"][key]["sfwc"]
        entry["r_star_trained"] = _r_star(sfwc_by_rank_t)

        actions = load_json(attr_dir(cfg, attr, "probes") / "action_spectra.json", {}) or {}
        entry["action_spectra"] = {k: {kk: v[kk] for kk in ("H", "erank", "top1_mass", "mean_displacement", "cv_displacement") if kk in v}
                                   for k, v in actions.items()}
        entry["separation"] = load_json(attr_dir(cfg, attr, "probes") / "separation.json", {}) or {}
        entry["stability"] = load_json(attr_dir(cfg, attr, "probes") / "stability.json", {}) or {}
        entry["diagnostics"] = load_json(attr_dir(cfg, attr, "extract") / "diagnostics.json", {}) or {}
        entry["identity_check"] = load_json(attr_dir(cfg, attr, "extract") / "identity_check.json", {}) or {}
        entry["classifier"] = load_json(attr_dir(cfg, attr, "eval") / "classifier.json", {}) or {}
        summary["attributes"][attr] = entry

    # ---------------- gates
    gates: dict = {}
    if summary["attributes"]:
        gates["G0_identity"] = {
            "max_logit_diff": float(np.nanmax([a["identity_check"].get("max_logit_diff", np.nan)
                                               for a in summary["attributes"].values()])),
        }
        levels = ["L1", "L2", "L3"]
        level_h = {}
        for attr, entry in summary["attributes"].items():
            ref = entry["action_spectra"].get("F16") or entry["action_spectra"].get("Ffull") or {}
            h = ref.get("H") or ref.get("erank")
            if h is not None:
                level_h.setdefault(entry["level"], []).append(float(h))
        level_means = {lvl: float(np.mean(v)) for lvl, v in level_h.items()}
        gates["G1_ladder"] = {"level_means": level_means,
                              "confirmatory_style_gt_sentiment": None}
        if "L3" in level_means and "L1" in level_means:
            gates["G1_ladder"]["confirmatory_style_gt_sentiment"] = bool(level_means["L3"] > level_means["L1"])

        gates["G2_necessity"] = {
            attr: {"r_star_closed_form": e["r_star_closed_form"], "r_star_trained": e["r_star_trained"]}
            for attr, e in summary["attributes"].items()
        }

        # difficulty vs complexity (exploratory, n small)
        from scipy import stats as sstats

        xs, ys = [], []
        for attr, entry in summary["attributes"].items():
            ref = entry["action_spectra"].get("F16") or {}
            if ref.get("H") is None:
                continue
            s_human = entry["classifier"].get("s_human") or 0.7
            target = 0.7 * s_human
            alpha_t = None
            for p in entry["curves"].get("F16", []):
                if p["strength"] >= target:
                    alpha_t = p["m"]
                    break
            if alpha_t is None:
                continue
            xs.append(float(ref["H"]))
            ys.append(float(alpha_t))
        if len(xs) >= 3:
            rho, p = sstats.spearmanr(xs, ys)
            gates["G3_validity"] = {"spearman_rho": float(rho), "p": float(p), "n": len(xs)}
        # TTR confound
        xs2, ys2 = [], []
        for attr, entry in summary["attributes"].items():
            ref = entry["action_spectra"].get("F16") or {}
            ttr_delta = entry["diagnostics"].get("ttr_delta")
            if ref.get("H") is not None and ttr_delta is not None:
                xs2.append(float(ttr_delta))
                ys2.append(float(ref["H"]))
        if len(xs2) >= 3:
            rho2, p2 = sstats.spearmanr(xs2, ys2)
            gates["G4_confound"] = {"spearman_H_vs_ttr_delta": float(rho2), "p": float(p2), "n": len(xs2)}

    summary["gates"] = gates
    save_json(out / "summary.json", summary)
    _write_report(cfg, summary, out)
    _make_figures(summary, out)
    print(f"[analyze] wrote {out / 'report.md'}")
    return out


def _write_report(cfg, summary: dict, out: Path) -> None:
    lines = ["# SAS-E1 — analysis report", "", f"Run: `{cfg.run_name}`", ""]
    lines.append("## Gates")
    for gate, value in summary.get("gates", {}).items():
        lines.append(f"- **{gate}**: `{value}`")
    lines.append("")
    lines.append("## Per-attribute summary")
    lines.append("")
    lines.append("| Attribute | Level | H_action (F16) | erank | r* closed | r* trained | SFWC F16 | SFWC C4 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for attr, e in summary["attributes"].items():
        ref = e["action_spectra"].get("F16", {})
        sfwc_f = (e["sfwc"].get("F16") or {}).get("sfwc")
        sfwc_c = (e["sfwc"].get("C4") or {}).get("sfwc")
        lines.append(
            f"| {attr} | {e['level']} | {ref.get('H')} | {ref.get('erank')} | "
            f"{e['r_star_closed_form']} | {e['r_star_trained']} | {sfwc_f} | {sfwc_c} |"
        )
    lines.append("")
    lines.append("## Curves (strength / coherence per scale m)")
    for attr, e in summary["attributes"].items():
        lines.append(f"### {attr}")
        lines.append("")
        lines.append("| arm | m | strength | coherence | ppl_ratio | content_sim |")
        lines.append("|---|---|---|---|---|---|")
        for arm, curve in e["curves"].items():
            for p in curve:
                lines.append(
                    f"| {arm} | {p['m']} | {p['strength']:.3f} | {p['coherence']:.3f} | "
                    f"{p['ppl_ratio']:.3f} | {p['content_sim']:.3f} |"
                )
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _make_figures(summary: dict, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[analyze] matplotlib unavailable: {exc}")
        return

    # F1: complexity per attribute
    attrs = list(summary["attributes"])
    if attrs:
        H = [summary["attributes"][a]["action_spectra"].get("F16", {}).get("H") or 0 for a in attrs]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.bar(attrs, H)
        ax.set_ylabel("H_action (F16)")
        ax.set_title("F1 — measured complexity per attribute")
        fig.tight_layout()
        fig.savefig(out / "F1_complexity.png", dpi=150)
        plt.close(fig)

    # F2: SFWC vs rank per attribute
    fig, ax = plt.subplots(figsize=(7, 4))
    for attr, e in summary["attributes"].items():
        ranks, vals = [], []
        for rank in (1, 4, 16):
            s = e["sfwc"].get(f"F{rank}")
            if s and s["sfwc"] is not None:
                ranks.append(rank)
                vals.append(s["sfwc"])
        if ranks:
            ax.plot(ranks, vals, marker="o", label=attr)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rank r")
    ax.set_ylabel("SFWC")
    ax.set_title("F2 — strength at fixed coherence vs rank")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "F2_saturation.png", dpi=150)
    plt.close(fig)

    # F4: strength vs m curves
    fig, ax = plt.subplots(figsize=(7, 4))
    for attr, e in summary["attributes"].items():
        for arm in ("A", "F16", "C4"):
            curve = e["curves"].get(arm)
            if curve:
                ax.plot([p["m"] for p in curve], [p["strength"] for p in curve], marker=".",
                        label=f"{attr}:{arm}")
    ax.set_xlabel("scale multiplier m")
    ax.set_ylabel("strength")
    ax.set_title("F4 — strength vs scale (all attributes)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "F4_strength.png", dpi=150)
    plt.close(fig)
