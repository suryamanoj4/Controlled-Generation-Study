# SAS E2 — Phase 1 Pilot Results (Sentiment Steering)

> **Status:** closed as a pilot / negative result (2026-11-02)
> **Decision:** do **not** run the complete 5-language Phase 1 grid; proceed to Phase 2
> **Scope note:** the frozen plan's original Phase 1 was *language control*. It was superseded by the sentiment-L1 pivot after the Phase 1 dataset was built. This note documents the actual Phase 1 that was run.
> **Related:** [[SAS E2 — Frozen Study Plan (Low-Resource Control & Style Steering)]] · [[SAS E2 — Low-Resource Style Steering Literature Review]]
> **Code:** `sas-e2/scripts/` · **Artifacts:** `sas-e2/artifacts/`

## 1. Why Phase 1 was run

Phase 1 was the cleanest L1 test of the training-free steering mechanism before moving to author style:

- attribute: **sentiment**
- languages: **en, hi, bn, te, ta**
- model: **Qwen3-4B**, 4-bit NF4, fp16 compute, T4
- vector: **DiffMean / CAA**, fitted from `stat.jsonl`
- behavioural arm: **vector-only**; closed-form operators deferred to Phase 2

The intended logic was: if a mean-difference direction cannot produce a controlled sentiment shift in a clean classification task, style steering in Phase 2 is unlikely to be defensible.

## 2. Setup

### Data

| Split | Per language |
|---|---:|
| `stat` (vector fit) | 500 (250 positive / 250 negative) |
| `clf_train` | 500 Indic / 4,000 English |
| `clf_eval` | 152 Indic / 1,000 English |
| prompts | 30 neutral topics |

### Vector fit

- Qwen3-4B @ `1cfa9a7208912126459214e8b04321603b3df60c`
- `max_length=512` (0 truncation for en/hi/bn, 11/500 te, 2/500 ta)
- last-token pooling, all 36 decoder blocks, alpha convention `h' = h + αv`
- 20 bootstrap resamples for stability

### Classifier

XLM-R base, trained on `clf_train` with language-balanced sampling.

| Lang | Held-out AUROC |
|---|---:|
| en | 0.953 |
| hi | 0.992 |
| bn | 0.982 |
| te | 0.977 |
| ta | 0.987 |
| **overall** | **0.962** |

The classifier is strong on human reviews. On generated text it saturates heavily (many p = 0.999 or 0.001), so the **logit margin** is the primary outcome in the revised pilot.

### Generation

- layer 18, alpha 1.0, 160 new tokens, batch size 8
- Qwen3 non-thinking sampling: `temperature=0.7`, `top_p=0.8`, `top_k=20`
- direct-answer system prompt
- direct in-language prompt templates; Hindi t01 grammar fixed for the revised pilot

## 3. Pilot 1 — strict 3×3 grid

Design: prompt sentiment `{positive, neutral, negative}` × vector `{none, +1, −1}`, plus one random vector on neutral. 10 prompts/language, 500 generations, **1489 s (~24.8 min)**.

### Mechanical gates

- empty: **0/500**
- script ratio: **0.99–1.00**
- thinking leakage: **0**
- vector changed output: all non-baseline conditions

### Neutral prompt, ±vector (probability of positive)

| Lang | +v | −v | Δp |
|---|---:|---:|---:|
| en | 0.999 | 0.769 | **+0.230** |
| hi | 0.991 | 0.700 | **+0.291** |
| bn | 0.999 | 0.398 | **+0.601** |
| te | 0.802 | 0.010 | **+0.792** |
| ta | 0.464 | 0.100 | **+0.364** |

This was the strongest signal: on neutral prompts the vector separated +v from −v in every language.

### Prompt control

| Lang | positive prompt | neutral | negative prompt | Ordered? |
|---|---:|---:|---:|---|
| en | 0.999 | 0.999 | 0.001 | ceiling |
| hi | 0.972 | 0.999 | 0.000 | **no** |
| bn | 0.998 | 0.577 | 0.001 | yes |
| te | 0.999 | 0.699 | 0.155 | yes |
| ta | 0.698 | 0.296 | 0.101 | yes |

### Headroom and random control

| Lang | negative + v | positive − v | random Δp | Random flat? |
|---|---:|---:|---:|---|
| en | +0.099 | +0.000 | −0.000 | yes |
| hi | +0.000 | −0.116 | −0.181 | **no** |
| bn | +0.446 | −0.597 | +0.023 | yes |
| te | −0.051 | −0.208 | −0.109 | **no** |
| ta | −0.000 | −0.298 | +0.156 | **no** |

Only Bengali passed both headroom tests and the random control. The strict pilot result was **FAIL** (`mechanical_ok=True`, `changed_ok=True`, `all_ok=False`).

## 4. Pilot 2 — revised mid-scale prompts

The failure was attributed partly to ceiling/floor effects and a single random control, so the revised pilot used:

- **balanced** prompt: “include both positives and negatives”
- **mildly positive**: “mostly positive, one minor complaint”
- **mildly negative**: “mostly negative, one minor compliment”
- no vector, +v, −v for each
- **5 norm-matched random vectors** on the balanced prompt
- logit margin as the primary outcome

700 generations, **2025 s (~33.8 min)**, batch size 8.

### Mechanical gates

- empty: **0/700**
- script ratio: **0.99–1.00**
- thinking leakage: **0**
- all vector conditions changed output

### Prompt control (margin)

| Lang | mild positive | balanced | mild negative | Ordered? |
|---|---:|---:|---:|---|
| en | +6.55 | +4.64 | −6.94 | **yes** |
| hi | +5.68 | +6.61 | −6.22 | no |
| bn | −1.66 | −2.32 | −0.97 | no |
| te | +5.11 | +6.27 | −0.31 | no |
| ta | −4.64 | +4.92 | −4.73 | no |

Only **English** followed the graded sentiment instructions. In Indic, the model often ignored the “balanced”/“mild” distinction or produced positive-biased/meta text.

### Random-null comparison (balanced prompt)

| Lang | +v | −v | random mean [min,max] | +v > max? | −v < min? |
|---|---:|---:|---|---:|---:|
| en | +1.40 | −2.58 | +0.48 [−0.28,+1.76] | no | yes |
| hi | −0.70 | −7.71 | −1.34 [−2.75,−0.00] | no | yes |
| bn | +5.12 | +1.13 | +4.35 [+3.48,+5.83] | no | yes |
| te | −0.76 | −1.85 | −0.66 [−1.16,−0.10] | no | yes |
| ta | −2.61 | −0.57 | −2.70 [−3.89,−1.43] | no | no |

The **negative** direction went below the random minimum in en/hi/bn/te. The **positive** direction never exceeded the random maximum in any language. The strict revised pilot also **FAILED**.

## 5. Primary reason for failure

**The vector is active, but the effect is not sentiment-specific or large enough to identify as steering.**

This is not a case of “the vector does nothing”:

- every vector condition changed the output;
- the neutral +v/−v separation was large in pilot 1;
- the negative direction beat the random null in 4/5 languages.

The failure is **effectiveness + specificity**, driven by five factors, in order:

1. **No controlled neutral baseline.** The model writes positive reviews even for balanced/neutral prompts. In the revised pilot the balanced margins were +4.6 en, +6.6 hi, +6.3 te, +4.9 ta, −2.3 bn. Positive steering has no headroom; negative steering does. The prompts do not create a mid-scale sentiment baseline.
2. **The vector carries non-sentiment variation.** It is a difference of means over **independent** positive/negative reviews, so topic, product, length, style and quality differences are included. A norm-matched random vector moves the classifier by several margin points; the positive vector never exceeds that null.
3. **Classifier saturation and domain shift.** XLM-R is strong on human reviews (AUROC 0.95–0.99) but overconfident on generated text. Probability is uninformative at 0/1; the logit margin is better but still a domain-shifted instrument.
4. **Indic instruction-following is weak.** Balanced/mildly positive/mildly negative instructions are often not followed; Hindi, Bengali, Telugu and Tamil produce meta or positive-biased text. The prompt control works only in English.
5. **The pilot is small.** n=10 prompts, one layer (18), one α (1.0). This adds noise, but the ceiling/random-null problems would remain with more prompts.

### What would have counted as success

- prompt control ordered: positive > balanced > negative;
- `+v` above the random-vector maximum and `−v` below the random minimum;
- no empty/script/thinking failures;
- consistent in at least 4/5 languages;
- with 20–30 prompts and a layer/α sweep.

The current data meet the mechanical gates but not the directional/control gates.

### What would be needed to fix it

- a **contrastive prompt-pair vector** (CAA-style) on held-out topics, not independent reviews;
- a genuinely **neutral/mixed content baseline**, or sentiment transfer from a neutral/negative prefix;
- a stronger Indic instruction-following model (e.g. 8B) and native prompt rewriting;
- a classifier validated/calibrated on generated text, plus human audit;
- multiple random vectors, 20–30 prompts, layer/α sweep.

That is a substantial detour, not a small parameter change.

## 6. Decision

**Phase 1 is closed after the pilot. No complete 5-language run.**

Rationale:

- the mechanism is not inert, but the effect is weak, one-sided and not separable from random perturbation;
- the Indic prompt compliance problem is a model/task issue, not a vector-only issue;
- a full run would consume compute without resolving the construct-validity problem;
- the style contribution (Phase 2/3) is the main study, and the Phase 1 harness is now reusable.

Phase 1 is retained as a **methods / negative result**:

- the dataset, vector fitter, classifier, batched generation, random-null and prompt-control harness are built and reusable;
- the negative result motivates moving to human-authored style corpora, where the signal does not depend on the model following a sentiment instruction.

## 7. What is reusable

- `sas-e2/datasets/phase1/` — sentiment data, prompts, FLORES diagnostic
- `sas-e2/scripts/download_models.py` — resumable Qwen3 downloader
- `sas-e2/scripts/fit_vectors.py` — 512-token DiffMean fit + cache
- `sas-e2/scripts/train_classifier.py` — XLM-R sentiment instrument
- `sas-e2/scripts/test_generation.py` — batched generation smoke test
- `sas-e2/scripts/pilot_generation.py` — strict 3×3 pilot
- `sas-e2/scripts/revised_pilot.py` — mid-scale + random-null pilot
- `sas-e2/artifacts/vectors/qwen3-4b/1cfa9a72/` — 512-token vectors
- `sas-e2/artifacts/classifier/xlmr_phase1/metrics.json` — classifier metrics
- `sas-e2/artifacts/pilot/` — both pilot JSONs and READMEs

Batching result: 500 generations in ~25 min, 700 in ~34 min, ~3 s/generation on the T4.

## 8. Next: Phase 2

Start Phase 2 on **English, Hindi, Telugu**, where author corpora are verified:

- EN: Shakespeare vs other early-modern drama
- HI: Premchand vs Prasad
- TE: Gurajada vs Veerasalingam (cap at 177 docs)
- BN/TA: pending public-domain verification

Phase 2 design should use **neutral content prompts + author-style vectors**, not “write in this style” instructions. Carry over:

- batched generation and paired seeds;
- logit-margin / style-classifier validation;
- random-vector null;
- prompt/format controls;
- human audit.

Phase 3 remains dependent on Phase 2 vectors and should not start before Phase 2 is validated.

## 9. Open issues

- Hindi/Bengali/Tamil prompt translations still need native verification.
- The sentiment classifier is not validated on generated text; it is a human-review instrument.
- The vector is fitted from independent reviews, not contrastive pairs.
- Bengali/Tamil author corpora are not yet available.
- Phase 1 language-control grid from the frozen plan was not run; the sentiment pivot is the actual Phase 1.

## 10. Bottom line

> **Phase 1 is a pipeline success and a steering negative result.** The vector is active, the harness works, and the negative direction is visible — but the positive effect is not separable from random perturbation and prompt/classifier bias. The complete Phase 1 run is cancelled; the programme moves to Phase 2 style steering with the evaluation lessons carried forward.
