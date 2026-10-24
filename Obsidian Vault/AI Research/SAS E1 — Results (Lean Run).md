# SAS E1 — Results (Lean Run)

> **Status:** complete — run 2026-10-18; **fable rebuilt & re-run 2026-10-24**; analysis 2026-10-24 17:40 IST · 8 attributes · all stages passed
> **Parent:** [[SAS E1 — Complexity Study Design]] · [[Style Adapter Steering — Model-Agnostic Narrative Style Control]]
> **Artifacts:** `sas-e1/artifacts-lean/artifacts/lean/` (canonical merged copy) — full tables in `analysis/report.md`, machine-readable `analysis/summary.json`

## What ran

- **Model** BLOOMZ-1b7, layer 12 (50% depth), bf16 (L4-class Colab); hook verified bit-exact (G0).
- **Attributes (8)** — English ladder: `sentiment` (L1), `toxicity` (L2), `shakespeare` (L3), `fable` (L3); Indic: `sentiment_hi` (L1), `story_hi` (L3), `sentiment_te` (L1), `story_te` (L3).
- **Operators** — nested family: `A` (vector), closed-form `F1/F4/F16/Ffull`, trained `C1/C4` (2 seeds), controls `E` (identity), `R` (random), prompt baseline `P`; effect-norm α calibration, m ∈ {0.5, 1, 2, 4}; 40 prompts × 2 generation seeds; 2 closed-form resamples.
- **Data** — 1000 matched docs/side (EN legs; fable's Wikipedia contrast rebuilt from sentence-packed lead chunks of 20–55 words after the original long-paragraph contrast collapsed its matched pool), 400/side (Indic sentiment), uniform 200-doc author budget (Indic story legs); classifier instrument 800/400 (EN), 400/250, 45/25 (story).
- **Scale** — 67 evaluated conditions per attribute (2.7k generations), probes + gate analysis; ~9 h GPU wall across two Colab sessions plus a ~1 h fable re-run (generation-dominated; checkpoint/resume worked per-condition).

## Gates

| Gate | Result | Read |
|---|---|---|
| **G0 sanity** | `max_logit_diff = 0.0` | ✅ identity hook is an exact no-op |
| **G1 ladder** | `H`: L1 **0.054** < L2 **0.316** < L3 **0.849**; style > sentiment confirmatory `True` | ✅ **confirmed** |
| **G2 necessity** | `r*` = 1 for most legs (closed-form *and* trained), including corrected fable; `story_hi` closed-form needs **16**; `sentiment_te` trained 4; several `None` (no rank pair passed the SFWC filter) | ~ mixed — low rank usually saturates |
| **G3 validity** | gate **omitted** (n = 2; only `story_hi`/`story_te` reach 0.7·s_human at m ≤ 4 — both instrument-degenerate) | ⚠️ not reportable — α ceiling too low |
| **G4 confound** | H vs TTR-Δ: ρ = 0.45, p = 0.26 (n = 8) | ⚪ no significant lexical confound |

## Complexity (action spectra, F16 reference arm)

| Attribute | Level | H (F16) | erank | top-1 mass |
|---|---|---|---|---|
| sentiment | L1 | 0.023 | 1.02 | 0.998 |
| toxicity | L2 | 0.316 | 1.37 | 0.952 |
| shakespeare | L3 | **1.668** | **5.30** | 0.602 |
| fable | L3 | 0.547 | 1.73 | 0.909 |
| sentiment_hi | L1 | 0.042 | 1.04 | 0.995 |
| story_hi | L3 | 0.899 | 2.46 | 0.821 |
| sentiment_te | L1 | 0.098 | 1.10 | 0.988 |
| story_te | L3 | 0.281 | 1.33 | 0.957 |

Action entropy grows with rank inside every leg (`F1 < F4 < F16 < Ffull`), and trained arms stay near rank-1 (e.g., shakespeare C1 H=0.001, C4 H=0.106; fable C4 H=0.258) despite their rank-4 parameterization. **L3 is heterogeneous**: shakespeare 5.30 ≫ story_hi 2.46 > fable 1.73 > story_te 1.33 — the level mean is carried mainly by shakespeare and story_hi.

## Behavioral summary (coherence ≥ 0.9 budget)

| Attribute | SFWC F16 (m*) | SFWC C4 (m*) | classifier AUROC | s_human |
|---|---|---|---|---|
| sentiment | 0.339 (0.5) | 0.410 (0.5) | 0.880 | 0.79 |
| toxicity | 0.229 (0.5) | 0.231 (0.5) | 0.944 | 0.86 |
| shakespeare | 0.484 (0.5) | 0.478 (0.5) | 0.842 | 0.72 |
| fable | 0.427 (2.0) | 0.570 (4.0†) | 1.000 | 0.93 |
| sentiment_hi | 0.265 (0.5) | 0.276 (0.5) | 0.828 | 0.72 |
| story_hi | 0.488 (4.0) | 0.488 (0.5) | 0.981 | 0.69 |
| sentiment_te | 0.222 (0.5) | 0.231 (0.5) | 0.854 | 0.78 |
| story_te | 0.589 (0.5) | 0.601 (4.0) | 1.000 | 0.76 |

† best-coherence fallback: no fable condition met coherence ≥ 0.9.

Vector `A` and random `R` collapse coherence beyond m ≈ 1–2 (e.g., sentiment `R` @ m=4: ppl_ratio 28, coherence 0.04). Prompting `P` gives the strongest raw strength in most legs but usually destroys coherence. Best strength-under-coherence is attained at the **smallest tested α** in almost every leg. **Fable's behavioral axis is weak**: E baseline is already 0.489 and no arm reaches 0.7·s_human (best F16 0.609 @ m=0.5; P drops to 0.246); coherence rises above 1 at m ≥ 2 (ppl_ratio < 1). Treat fable behavior as instrument-limited; its spectra are the usable signal.

## Key findings

1. **The complexity ladder is confirmed** — L1 attributes act at essentially rank 1 (erank ≈ 1.0, top-1 ≈ 0.99), L2 slightly richer, and L3 style legs are higher-dimensional on average (mean 0.849), though heterogeneous: shakespeare erank 5.3, story_hi 2.46, fable 1.73, story_te 1.33.
2. **The nested family behaves as designed** — action entropy increases monotonically with rank in every leg.
3. **Training did not buy complexity**: trained adapters (C1/C4) act as near-rank-1 maps (H ≤ 0.4, erank ≤ 1.5) in every leg, and their cross-seed subspaces barely overlap (0.03–0.20) while biases stay stable (bias cosine 0.95–0.99). Any style-level complexity in this harness lives in the *closed-form* rank sweep, not in gradient training.
4. **Closed-form fits are highly reproducible** — resample spectrum distances ≤ 0.02 for the F-arms; vector cosines 0.94–0.999 (story_hi 0.86). The complexity measurements are stable, not noise.
5. **Necessity is mostly rank-1 under the SFWC criterion** — `r*` = 1 in 5 of the 6 measurable closed-form legs (including corrected fable; `story_hi` is the exception, needing rank 16; `sentiment_te` trained needs 4). High entropy does not translate into a need for higher intervention rank at matched coherence.
6. **No lexical confound** (G4 n.s.); **validity correlation is not assessable with this α range** — only 2 legs reach the 0.7·s_human target at m ≤ 4, so G3 is omitted. Extending the α grid (8/16) or lowering the target is required.
7. **Methodological caution — undersampling inflates measured complexity.** The first fable fit (77/69 docs, from the original long-paragraph contrast) gave H(F16) = 1.04, erank 2.84; the corrected 1000/1000 fit gives **H = 0.55, erank 1.73** — roughly 2× inflation from an underdetermined covariance. Low-resource legs need the length-matching safeguard (and enough matched docs) before their spectra are interpreted.

## Caveats and exclusions

- **Superseded fable run.** The earlier fable numbers (from the 77/69-doc corpus) are retained only in the archive; they are *not* used. The rebuilt corpus and re-run are the reference (fix commit `8f02c76`, 1000/1000 matched docs).
- **Fable's behavioral instrument is saturated** — classifier AUROC ≈ 1.0 on the corpora, but the unsteered baseline already scores 0.489 and no steered arm reaches 0.7·s_human; prompting scores *below* baseline (0.246). Use fable's action spectra and coherence, not its absolute strength.
- **Indic story strength is degenerate.** `story_hi`/`story_te` strength is constant (0.488 / 0.589) across *all* arms — the char-n-gram style classifier saturates on model-generated Indic text. Their strength/SFWC numbers are uninformative; **only action spectra, stability and coherence should be used** from these legs.
- **`story_te` coherence > 1** (ppl_ratio < 1) throughout — BLOOMZ's Telugu neutral generations are weak, so steer-vs-baseline coherence there is distorted.
- **G3 not reportable** (n = 2, both degenerate instruments); **figures not generated** on Colab (matplotlib backend) — all tables are complete in `report.md`.
- Cross-leg *strength* values are not comparable (different instruments and baselines, and fable's classifier changed when its contrast corpus was rebuilt); the ladder claim correctly rests on `H_action`, which is operator-side.

## Reproduce / resume

```bash
bash scripts/setup.sh configs/lean.yaml
uv run python scripts/run_experiment.py --config configs/lean.yaml --stage all
# targeted leg re-run (example: fable)
uv run python scripts/run_experiment.py --config configs/lean.yaml --stage all --attr fable
```

Every stage/attribute is resumable via `_done.json` markers, and eval resumes at per-condition granularity (`gens/*.jsonl`). Re-analyze any time with `--stage analyze`.

## Next steps

- Extend the α ceiling (e.g., multipliers 8 and 16) or lower the strength target so G3 (complexity vs difficulty) becomes assessable on healthy legs.
- Optional instrument work: a style classifier that is not OOD-degenerate on generated Indic text (and possibly a stronger fable instrument).
- Replicate on a second model family and a layer sweep (25/50/75%) before claiming model-agnosticism.
- Write-up can proceed on the full EN ladder + Indic sentiment; disclose the Indic story and fable behavioral limits, and the G3 α-ceiling gap.
