# SAS E1 — Results (Lean Run)

> **Status:** complete — run 2026-10-18; fable rebuilt 2026-10-24 · **behavioral validity audit 2026-10-26 (this note corrected)** · 8 attributes · all stages completed
> **Parent:** [[SAS E1 — Complexity Study Design]] · [[Style Adapter Steering — Model-Agnostic Narrative Style Control]]
> **Artifacts:** `sas-e1/artifacts-lean/artifacts/lean/` (canonical merged copy) — tables in `analysis/report.md`, machine-readable `analysis/summary.json`, human-audit samples in `sas-e1/artifacts-lean/steering_samples.md`

## What ran

- **Model** BLOOMZ-1b7, layer 12 (50% depth), bf16 (L4-class Colab); hook verified bit-exact (G0).
- **Attributes (8)** — English ladder: `sentiment` (L1), `toxicity` (L2), `shakespeare` (L3), `fable` (L3); Indic: `sentiment_hi` (L1), `story_hi` (L3), `sentiment_te` (L1), `story_te` (L3).
- **Operators** — nested family: `A` (vector), closed-form `F1/F4/F16/Ffull`, trained `C1/C4` (2 seeds), controls `E` (identity), `R` (random), prompt baseline `P`; effect-norm α calibration, m ∈ {0.5, 1, 2, 4}; 40 prompts × 2 generation seeds; 2 closed-form resamples.
- **Data** — 1000 matched docs/side (EN legs; fable's Wikipedia contrast rebuilt from sentence-packed lead chunks of 20–55 words), 400/side (Indic sentiment), uniform 200-doc author budget (Indic story legs); classifier instrument 800/400 (EN), 400/250, 45/25 (story).
- **Scale** — 67 evaluated conditions per attribute (2.7k generations), probes + gate analysis; ~9 h GPU wall plus a ~1 h fable re-run (generation-dominated; checkpoint/resume per condition).

## ⚠️ Validity audit (2026-10-26) — read before any behavioral claim

A human audit of the saved generations found that **the model does not generate the Indic target languages at all** for the in-language prompts:

| Leg | Generations in target script | Empty generations | Median words | Behavioral status |
|---|---|---|---|---|
| `sentiment_hi` | **0 / 2640 (0.0%)** | 34% | 6 | **invalid — withdrawn** |
| `story_hi` | **5 / 2640 (0.2%)** | 43% | 4 | **invalid — withdrawn** |
| `sentiment_te` | **0 / 2640 (0.0%)** | 56% | 0 | **invalid — withdrawn** |
| `story_te` | **0 / 2640 (0.0%)** | 35% | 5 | **invalid — withdrawn** |

Outputs are English fragments or empty (early EOS); the Hindi/Telugu prompts themselves are correct. Consequently **all four Indic behavioral results (strength/SFWC) are invalid** — this also explains the constant story strengths and Telugu coherence > 1. The Indic **representation** results (action spectra from the authentic Wikisource corpora) are unaffected and remain valid.

For the EN legs, generations are in-language but the human check at m = 2 (prompt #0) shows: sentiment = visible content shift with degraded quality; toxicity = barely perceptible change; **shakespeare = no perceptible Early-Modern register**; fable = degeneration (A: 23/40 empty at m=2; F16: repetition loops). EN empty-generation rates: sentiment 0.2%, toxicity 1.0%, shakespeare 1.3%, fable 9.3%; median output 15–24 words. **Classifier strength alone is therefore not sufficient evidence of meaningful steering** — a human/judge validity gate is required. Full samples: `sas-e1/artifacts-lean/steering_samples.md`.

## Gates

| Gate | Result | Read |
|---|---|---|
| **G0 sanity** | `max_logit_diff = 0.0` | ✅ identity hook is an exact no-op |
| **G1 ladder** | `H`: L1 **0.054** < L2 **0.316** < L3 **0.849**; style > sentiment confirmatory `True` | ✅ **confirmed** (operator-side; unaffected by the audit) |
| **G2 necessity** | valid evidence only where a non-fallback SFWC point exists: shakespeare `1 / 1`, toxicity `– / 1`, fable `1 / –` (weak instrument); sentiment `– / –`. The Indic `r*` values (`story_hi` 16, `sentiment_te` trained 4) are **withdrawn** — artifacts of invalid generations | ~ thin, EN-only |
| **G3 validity** | **no valid assessment**: the only legs that reached 0.7·s_human were the two invalid Indic story legs; EN legs never reached it at m ≤ 4 | ⚠️ not reportable — needs α extension + valid Indic generations |
| **G4 confound** | H vs TTR-Δ: ρ = 0.45, p = 0.26 (n = 8) | ⚪ no significant lexical confound (corpus-side; valid) |

## Complexity (action spectra, F16 reference arm) — valid, operator-side

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

## Behavioral summary — EN legs only (Indic rows withdrawn)

| Attribute | SFWC F16 (m*) | SFWC C4 (m*) | AUROC (corpus instrument) | Behavioral validity |
|---|---|---|---|---|
| sentiment | 0.339† (0.5) | 0.410† (0.5) | 0.880 | classifier-valid; no point met the 0.9 budget; human check = content shift + degradation |
| toxicity | 0.229† (0.5) | 0.231† (0.5) | 0.944 | weak visible effect; only A/C1 had valid budgeted points |
| shakespeare | 0.484 (0.5) | 0.478† (0.5) | 0.842 | classifier-valid; **no perceptible style shift** in the human check at m=2 |
| fable | 0.427 (2.0) | 0.570† (4.0) | 1.000 | weak: saturated baseline; degeneration at m=2 (23/40 empty for A; F16 loops) |

† best-coherence fallback: no condition met coherence ≥ 0.9 (the strict budget is rarely met — most SFWC values in **all** legs are fallbacks).

Vector `A` and random `R` collapse coherence beyond m ≈ 1–2; prompting `P` gives the strongest raw strength in most legs but usually destroys coherence. Best strength-under-coherence is attained at the **smallest tested α** in almost every leg.

## Key findings

1. **The complexity ladder is confirmed** — L1 attributes act at essentially rank 1 (erank ≈ 1.0, top-1 ≈ 0.99), L2 slightly richer, and L3 style legs are higher-dimensional on average (mean 0.849), though heterogeneous: shakespeare erank 5.3, story_hi 2.46, fable 1.73, story_te 1.33. *This is the primary result and does not depend on generation quality.*
2. **The nested family behaves as designed** — action entropy increases monotonically with rank in every leg.
3. **Training did not buy complexity** — trained adapters (C1/C4) act as near-rank-1 maps (H ≤ 0.4, erank ≤ 1.5) in every leg; cross-seed subspace overlap is low (0.03–0.20) while biases stay stable (cosine 0.95–0.99). Any style-level action complexity lives in the *closed-form* rank sweep, not in gradient training of this objective.
4. **Closed-form fits are highly reproducible** — resample spectrum distances ≤ 0.02 for the F-arms; vector cosines 0.94–0.999 (story_hi 0.86).
5. **Necessity evidence is EN-only and thin** — where a valid (non-fallback) SFWC point exists, rank-1 saturates: shakespeare r* = 1 (closed-form and trained), toxicity trained 1, fable closed-form 1 (instrument weak). Sentiment has no valid budgeted point. The Indic r* values are withdrawn with the invalid generations. "Rank ~1 is sufficient" is therefore supported **where measurable**, not universally.
6. **No lexical confound** (G4 n.s.); **G3 is not assessable**: EN legs never reach 0.7·s_human at m ≤ 4, and the only legs that did (Indic story) have invalid generations. Needs an extended α ceiling **and** a valid Indic generation setup.
7. **Methodological caution — undersampling inflates measured complexity.** The first fable fit (77/69 docs) gave H(F16) = 1.04, erank 2.84; the corrected 1000/1000 fit gives **H = 0.55, erank 1.73** (~2× inflation from an underdetermined covariance).
8. **Behavioral validity is the binding constraint.** The pipeline applies steering mechanically (identity exact; outputs change), but: (a) the model produced ~0% in-language Indic output, invalidating all Indic behavioral metrics; (b) at m=2 the EN style legs show little or no perceptible style change and fable degenerates. Classifier strength alone overstates behavioral success — a language-match gate and human/judge validation are mandatory before further behavioral claims.

## Caveats

- **Indic behavioral results are withdrawn** (0–5/2640 in-language generations; 34–56% empty). Indic *representation* results remain valid.
- **EN style behavior needs validation** — human inspection shows no perceptible Shakespearean register at m=2 and fable degeneration; classifier numbers are not sufficient evidence.
- **SFWC budget is strict** — most entries are best-coherence fallbacks; `r*` rests on 1–4 valid points per leg. Report the strength–coherence frontier and budget sensitivity (0.7/0.8/0.9) in the next analysis.
- **Fable's behavioral axis is weak** — classifier AUROC ≈ 1.0 on corpora but the unsteered baseline already scores 0.489, no arm reaches 0.7·s_human, and steering degenerates at m=2.
- **G3 not reportable**; **figures not generated** on Colab (matplotlib backend) — all tables are in `report.md`.
- Cross-leg *strength* values are not comparable (different instruments/baselines); the ladder claim correctly rests on `H_action`.
- **Superseded fable run** — the 77/69-doc numbers are kept only in the archive; see finding 7.

## Reproduce / resume

```bash
bash scripts/setup.sh configs/lean.yaml
uv run python scripts/run_experiment.py --config configs/lean.yaml --stage all
# targeted leg re-run (example: fable)
uv run python scripts/run_experiment.py --config configs/lean.yaml --stage all --attr fable
```

Every stage/attribute is resumable via `_done.json` markers; eval resumes per condition (`gens/*.jsonl`). Re-analyze any time with `--stage analyze`.

## Next steps

- **Fix Indic generation first**: explicit in-language instruction + few-shot in-language exemplars, `min_new_tokens` to prevent empty outputs, and an automated in-language check (e.g., script-ratio ≥ 0.8) as a hard gate before any Indic eval is scored. Smoke-test on 5 prompts before a full re-run; consider a stronger multilingual model for the behavioral arm.
- **Human/judge validity gate** for EN style legs (pairwise, position-swapped, m ∈ {0.5, 1}); re-check whether style steering is perceptible at all at 1.7B.
- **Extend the α ceiling** (8/16) on healthy legs and report `r*` across coherence budgets — only then can G3/G2 be stated.
- **Replicate representation results** across a second model family and layers (extract+fit+probes only — no generation needed).
- **Write-up scope**: full representation ladder (all 8 legs) + EN behavioral results *with* judge/human validation; Indic behavioral section deferred until the language fix.
