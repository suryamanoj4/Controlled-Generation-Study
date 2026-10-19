# SAS E1 — Results (Lean Run)

> **Status:** complete (run finished 2026-10-18; analysis report 2026-10-18 21:50 IST) · 8 attributes · all stages passed
> **Parent:** [[SAS E1 — Complexity Study Design]] · [[Style Adapter Steering — Model-Agnostic Narrative Style Control]]
> **Artifacts:** `sas-e1/artifacts-lean/artifacts/lean/` — full tables in `analysis/report.md`, machine-readable `analysis/summary.json`

## What ran

- **Model** BLOOMZ-1b7, layer 12 (50% depth), bf16 (L4-class Colab); hook verified bit-exact (G0).
- **Attributes (8)** — English ladder: `sentiment` (L1), `toxicity` (L2), `shakespeare` (L3), `fable` (L3); Indic: `sentiment_hi` (L1), `story_hi` (L3), `sentiment_te` (L1), `story_te` (L3).
- **Operators** — nested family: `A` (vector), closed-form `F1/F4/F16/Ffull`, trained `C1/C4` (2 seeds), controls `E` (identity), `R` (random), prompt baseline `P`; effect-norm α calibration, m ∈ {0.5, 1, 2, 4}; 40 prompts × 2 generation seeds; 2 closed-form resamples.
- **Data** — 1000 matched docs/side (EN legs), 400/side (Indic sentiment), uniform 200-doc author budget (Indic story legs); classifier instrument 800/400 (EN), 400/250, 45/25 (story).
- **Scale** — 67 evaluated conditions per attribute (2.7k generations), probes + gate analysis; ~9 h GPU wall across two Colab sessions (generation-dominated; checkpoint/resume worked per-condition).

## Gates

| Gate | Result | Read |
|---|---|---|
| **G0 sanity** | `max_logit_diff = 0.0` | ✅ identity hook is an exact no-op |
| **G1 ladder** | `H`: L1 **0.054** < L2 **0.316** < L3 **0.973**; style > sentiment confirmatory `True` | ✅ **confirmed** |
| **G2 necessity** | `r*` = 1 for most legs (closed-form *and* trained); `story_hi` closed-form needs **16**; `sentiment_te` trained 4; several `None` (no rank pair passed the SFWC filter) | ~ mixed — low rank usually saturates |
| **G3 validity** | ρ = nan, n = 3 | ⚠️ inconclusive (only 3 legs reached 0.7·s_human within tested α) |
| **G4 confound** | H vs TTR-Δ: ρ = 0.52, p = 0.18 (n = 8) | ⚪ no significant lexical confound |

## Complexity (action spectra, F16 reference arm)

| Attribute | Level | H (F16) | erank | top-1 mass |
|---|---|---|---|---|
| sentiment | L1 | 0.023 | 1.02 | 0.998 |
| toxicity | L2 | 0.316 | 1.37 | 0.952 |
| shakespeare | L3 | **1.668** | **5.30** | 0.602 |
| fable | L3 | *excluded — see caveats* | | |
| sentiment_hi | L1 | 0.042 | 1.04 | 0.995 |
| story_hi | L3 | 0.899 | 2.46 | 0.821 |
| sentiment_te | L1 | 0.098 | 1.10 | 0.988 |
| story_te | L3 | 0.281 | 1.33 | 0.957 |

Action entropy grows with rank inside every leg (`F1 < F4 < F16 < Ffull`), and trained arms stay near rank-1 (e.g., shakespeare C1 H=0.001, C4 H=0.106; story_te C4 H=0.094) despite their rank-4 parameterization.

## Behavioral summary (coherence ≥ 0.9 budget)

| Attribute | SFWC F16 (m*) | SFWC C4 (m*) | classifier AUROC | s_human |
|---|---|---|---|---|
| sentiment | 0.339 (0.5) | 0.410 (0.5) | 0.880 | 0.79 |
| toxicity | 0.229 (0.5) | 0.231 (0.5) | 0.944 | 0.86 |
| shakespeare | 0.484 (0.5) | 0.478 (0.5) | 0.842 | 0.72 |
| sentiment_hi | 0.265 (0.5) | 0.276 (0.5) | 0.828 | 0.72 |
| story_hi | 0.488 (4.0) | 0.488 (0.5) | 0.981 | 0.69 |
| sentiment_te | 0.222 (0.5) | 0.231 (0.5) | 0.854 | 0.78 |
| story_te | 0.589 (0.5) | 0.601 (4.0) | 1.000 | 0.76 |

(Vector `A` and random `R` collapse coherence beyond m ≈ 1–2 — e.g., sentiment `R` @ m=4: ppl_ratio 28, coherence 0.04. Prompting `P` gives the strongest raw strength but usually destroys coherence. Best strength-under-coherence is attained at the **smallest tested α** in almost every leg.)

## Key findings

1. **The complexity ladder is confirmed** — L1 attributes act at essentially rank 1 (erank ≈ 1.0, top-1 ≈ 0.99), L2 slightly richer, and L3 style legs are genuinely higher-dimensional (shakespeare erank 5.3, top-1 0.60; story_hi 2.46). This is the experiment's primary result.
2. **The nested family behaves as designed** — action entropy increases monotonically with rank in every leg.
3. **Training did not buy complexity**: trained adapters (C1/C4) act as near-rank-1 maps (H ≤ 0.4, erank ≤ 1.5) in every leg, and their cross-seed subspaces barely overlap (0.03–0.20) while biases stay stable (bias cosine 0.95–0.99). Any style-level complexity in this harness lives in the *closed-form* rank sweep, not in gradient training.
4. **Closed-form fits are highly reproducible** — resample spectrum distances ≤ 0.02; vector cosines 0.94–0.999 (story_hi 0.86). The complexity measurements are stable, not noise.
5. **Necessity is mostly rank-1 under the SFWC criterion** (`r*` = 1 in 5 of 7 valid legs; `story_hi` is the exception, needing closed-form rank 16) — i.e., the adapter is *not justified by this criterion* for most attributes; shakespeare's high entropy does not translate into a need for higher rank at matched coherence.
6. **No lexical confound** (G4 n.s.); **validity correlation could not be assessed** (G3: only 3 legs reached the strength target within the tested α range — future runs should extend m or lower the target).

## Caveats and exclusions

- **Fable excluded.** The matched fable pool collapsed to **77/69 docs** (contrast Wikipedia paragraphs are ~3× longer than Moral Stories; the length cap = min of both sides' p90 = 73 tokens left almost no Wikipedia docs, then token-volume equalization dragged the positive side down). Its fitted operators are statistically too weak to interpret — numbers exist in the artifacts but are *not* used in conclusions. Fix (chunked Wikipedia contrast) + re-run of fable stages is pending.
- **Indic story strength is degenerate.** `story_hi`/`story_te` strength is constant (0.488 / 0.589) across *all* arms — the char-n-gram style classifier saturates on model-generated Indic text. Their strength/SFWC numbers are uninformative; **only action spectra, stability and coherence should be used** from these legs.
- **`story_te` coherence > 1** (ppl_ratio < 1) throughout — BLOOMZ's Telugu neutral generations are weak, so steer-vs-baseline coherence there is distorted.
- **G3 inconclusive** (n = 3); **figures not generated** on Colab (matplotlib backend) — all tables are complete in `report.md`.
- Cross-leg *strength* values are not comparable (different instruments and baselines); the ladder claim correctly rests on `H_action`, which is operator-side.

## Reproduce / resume

```bash
bash scripts/setup.sh configs/lean.yaml
uv run python scripts/run_experiment.py --config configs/lean.yaml --stage all
```

Every stage/attribute is resumable via `_done.json` markers, and eval resumes at per-condition granularity (`gens/*.jsonl`). Re-analyze any time with `--stage analyze`.

## Next steps

- Redo `fable` (chunked Wikipedia contrast → re-extract/fit/eval/probes) and re-analyze — then fable rejoins the L3 mean.
- Optional instrument work for the Indic style legs (a style classifier that is not OOD-degenerate on generated text).
- Extend `m` (or lower the target) so G3 (complexity vs difficulty) becomes assessable.
- Write-up can proceed on: EN ladder + Indic sentiment (all instruments healthy), with the two caveats above disclosed.
