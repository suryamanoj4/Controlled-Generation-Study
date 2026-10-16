# SAS Experiment 1 — Representation Complexity Study (Design)

> **Status:** v1.4 — unified Lean run (EN + Indic) (2026-10-17) · implementation-ready
> **Parent:** [[Style Adapter Steering — Model-Agnostic Narrative Style Control]]
> **Companion research:** `SAS E1 — Prior Work Research (Steering Complexity).md`
> **v1.0 changes vs v0.1:** effect-based complexity promoted to primary and parameter spectra require canonicalization (W2T); closed-form covariance-matching arm made a core falsification test (Representation Surgery/conceptors precedent); prompting baseline made mandatory (AxBench); ladder re-framed as falsifiable (refusal is documented rank-1); judge style-bias protocol added; seed/stability requirements raised; sample-size curve and separation pre-screen added; citation corrections to the parent proposal recorded in §11.
> **v1.1 change (2026-10-13):** default execution mode is now **Lean** — 4 attributes, closed-form rank sweep + one trained low-rank arm fitted on cached activations, single T4-class session ≤3 h. The full grid is retained as **Extended** (§2.2) and deferred. No end-to-end training is required in Lean.
> **v1.2 change (2026-10-13):** dataset licensing audit applied (§4.1) — toxicity primary switched to **Civil Comments (CC0-1.0)**; fable contrast and neutral probe pinned to **Wikipedia** (CC-BY-SA-3.0), with CNN/DailyMail (Apache-2.0) optional; CC-News dropped (404 / license unknown). No Lean dependency is gated or license-unknown.
> **v1.3 change (2026-10-16):** Chandamama removed from scope entirely; **E1b Indic replication** added (§4.1b) — authentic Wikisource author corpora (Hindi: Premchand vs Prasad; Telugu: Gurajada vs Veeresalingam) + permissively licensed sentiment sets; new config `configs/lean-indic.yaml`; in-language prompts and a multilingual classifier override.
> **v1.4 change (2026-10-17):** Indic attributes merged into the single `configs/lean.yaml` run — one setup, one run, no separate config; Lean is now **8 attributes**. Restrict with `--attr` when needed. (`configs/lean-indic.yaml` removed.)

---

## 0. Summary

E1 is a **representation complexity study**: it asks whether the complexity of a steering representation tracks attribute complexity, whether a low-rank adapter is needed over a classical steering vector (+ closed-form covariance operator), and whether measured complexity predicts steering difficulty. All questions live in one harness.

| Q | Question | Primary measure | Gate |
|---|---|---|---|
| Q1 | Does representation complexity separate attribute levels (L1 < L2 < L3)? | effect-spectrum entropy / effective rank of the fitted steerer | G1 (L3 vs L1 confirmatory; L2 exploratory) |
| Q2 | Is a low-rank adapter needed over a rank-1 vector — or even over a closed-form covariance operator? | minimum rank `r*` that saturates steering quality at matched coherence | G2 |
| Q3 | Does measured complexity predict steering difficulty? | correlation with α-to-target, contrast separation, seed stability | G3 |
| Q4 | Form vs objective: rank or training? | closed-form vs trained arms at matched rank | G4 |

Plus pipeline sanity (G0) and a prompting reference (mandatory).

**Key design move.** Treat every steerer as one nested family of **linear steering operators** on the residual stream:

```
h' = h + W h + b
```

- Classical steering vector (CAA): `W = 0`, `b = μ_P − μ_N` (bias-only; action is rank-1).
- Closed-form covariance operator (Representation Surgery / Conceptor lineage): `W* = Σ_P^{1/2} Σ_N^{−1/2} − I`, SVD-truncated to rank r.
- Style adapter (SAS): `W = BA` (low rank r), `b` bias channel, fit by gradient.

"Vector vs adapter" becomes a **rank sweep inside one parametrized family**, with closed-form and trained variants at each rank. Any gain from extra rank is then attributable, and the necessity question gets its cheapest possible falsification: a full-covariance closed-form operator that costs no training.

**Execution mode.** Default = **Lean** (§2.2): 4 attributes, closed-form rank sweep, one trained low-rank arm fitted on cached activations, ~1.5–2 h GPU on a Colab/Kaggle T4. The Extended grid is documented but deferred.

---

## 1. Questions, hypotheses, gates

### Hypotheses

- **H1a (confirmatory ladder).** Effect-spectrum complexity of fitted steerers: `H(style L3) > H(sentiment/formality L1)`, consistent across ≥2 of 3 proxies, and difficulty follows the same direction.
- **H1b (exploratory ladder).** Safety sits between L1 and L3. **Falsifiable and expected to possibly reverse:** refusal is documented rank-1 (Arditi et al. 2024), and CAS-BiPO shows compact safety structure — report non-monotonicity as a finding, and distinguish *semantic* from *representational* complexity.
- **H2 (necessity).** `r* = 0–1` for L1; `r* ≥ 4` for ≥2 L3 attributes if the adapter is justified. If the closed-form covariance arm matches trained arms at equal rank, the trained adapter is not necessary.
- **H3 (validity).** Measured complexity predicts independent difficulty: Spearman ρ ≥ 0.6 with bootstrap CI excluding 0 (exploratory; n ≈ 6 attributes). Contrast separation (Braun et al.) is tested as a competing predictor.
- **H4 (form vs objective).** Any advantage over the closed-form vector comes from rank (form) and/or training (objective); isolated by the 2×2.

### Pre-registered gates

| Gate | Criterion | If it fails |
|---|---|---|
| **G0 sanity** | Identity arm bit-exact no-op; hook indexing verified; random control behaves as random; action-rank sanity (vector action = 1) | Fix pipeline; do not proceed |
| **G1 ladder** | Confirmatory: `H(L3) > max(H(L1), H(L2))` with non-overlapping bootstrap CIs (one-sided permutation p < 0.05). Full 3-level monotonic regression (Jonckheere–Terpstra) reported as exploratory | Drop the ladder claim; retain L3 vs L1 if it holds; pivot to E2 |
| **G2 necessity** | Compute `r*` per attribute (§6.3). `r* ≤ 1` everywhere → vector suffices; closed-form F matches C within CI → no training needed; `r* ≥ 4` for ≥2 L3 attrs → adapter justified at style level | Whatever the outcome, this is a publishable boundary result |
| **G3 validity** | ρ(complexity, difficulty) ≥ 0.6, CI excludes 0; also compare against separation pre-screen (P5) as predictor | Demote complexity metric to descriptive |
| **G4 confound** | H must not correlate with corpus lexical diversity (TTR/MTLD) more than with attribute level; also report H with narrative-vs-informational text control | Report as negative/limited result |

---

## 2. Design overview

### 2.1 Factors

| Factor | Levels | Notes |
|---|---|---|
| Steerer arm | Lean: A, F∞, F1/F4/F16, C1/C4, E, R, P · Extended: + A′, B, C16, D | see §3 |
| Rank r | Lean: {1, 4, 16} + F∞ · Extended: {1, 2, 4, 8, 16, 32} | arms F (closed-form) and C (trained) |
| Attribute | Lean: 8 across 3 levels (EN: sentiment, toxicity, Shakespeare, fable; HI/TE: sentiment, author style) · Extended: +6–8 | §4.1 |
| Layer | Lean: layer 12 (50%) only · Extended: {33%, 50%, 66%} descriptive | §3.8 |
| Seed / resample | Lean: 2 seeds (trained) / 2 resamples (closed-form) · Extended: 5/5 | distributions reported, never best run |
| Model | bloomz-1b7 primary (dtype `auto`: bf16 if supported, else fp32 — **never fp16** for BLOOM); bloomz-560m pilot; bloom-1b7 robustness in Extended | §2.3 |

**P (prompting) is mandatory**: AxBench (ICML 2025) found prompting to be the strongest baseline on many attributes — the vector/adapter result is uninterpretable without it.

### 2.2 Grids and budget

**Lean (default) — one Colab/Kaggle GPU session; 8 attributes (EN + Indic).**

| Grid | Composition | Runs |
|---|---|---|
| Ladder (primary) | 8 attrs × closed-form {A, F∞, F1, F4, F16, E, R} × 2 resamples + trained {C1, C4} × 2 seeds | ~180 (mostly free) |
| Mini sample-size curve | {sentiment, Shakespeare} × sizes {16, 64, 256} × 3 resamples | 18 |
| Generation eval | 8 attrs × 9 arms × 4 scale points {0.5, 1, 2, 4} × 40 prompts × 2 generation seeds ≈ 22k short generations | — |
| **Compute** | extraction ~15 min + cached-activation fits ~10 min + generation ~60–90 min + metrics ~15 min | **~2–2.5 h GPU (bf16-capable; fp32-only GPUs ~2–3× slower; resumable per stage/attribute)** |

Lean design notes:

- **Trained arms fit on cached activations only** — no forward/backward through the base model. The moment-matching loss is a function of stored layer-L activations, so C1/C4 fits take seconds. The experiment is effectively inference-only.
- **Sample sizes are deliberately small:** 1,000 docs/side for moment statistics, 300/side for fitting, 500 neutral probe docs, 800/400 classifier train/eval, 40 eval prompts. Adapters need only hundreds of exemplars; the sample-size curve is what demonstrates saturation rather than assuming it.
- Closed-form arms require one extraction pass per attribute; all rank truncations and resamples are numpy/SVD.
- Judge: optional subsample (~100 generations/attribute) via API, or skipped entirely in Tier A.
- **Tier A** (inference-only, no trained arms): ~45–60 min — closed-form ladder only.
- **Tier B** (default): adds C1/C4 → ≤3 h including model download and overhead.
- OOD split, layer sweep, warm-start, and safety/refusal legs move to Extended.

**Extended (deferred) — full design for later.**

| Grid | Composition | Runs |
|---|---|---|
| Ladder | 6–8 attrs × {A, A′, B, C16, D16} × 5 seeds + closed-form resamples | ~210 |
| Saturation | 2 attrs × 6 ranks × {F, C} × 5 | 120 |
| Sample-size curve | 2 attrs × 6 sizes × 5 | 60 |
| Layer sweep | Shakespeare × 3 layers × 2 arms × 5 | 30 |
| Warm-start | 2 attrs × 2 inits × C16 × 5 | 20 |
| **Total** | | **~440 runs / 2–4 GPU-days** |

Generation/eval (Extended): 6 attrs × 10 arms × 5 scale points × 160 prompts × 5 seeds, plus OOD split and full judge protocol.

### 2.3 Models

| Role | Model | Why |
|---|---|---|
| Pilot | `bigscience/bloomz-560m` | Cheapest full-pipeline validation |
| **Primary** | `bigscience/bloomz-1b7` | Decoder-only, 46 languages (HI/TE), small, **per-language ROOTS proportions published** → carries into E2. fp16 fits a T4 16 GB (Turing has no bf16) |
| Robustness | `bigscience/bloom-1b7` (base) | Confirms instruct tuning is not doing the work (skip refusal leg) |
| Optional scale | `bigscience/bloomz-3b` | Scale robustness; Hao et al. note larger models resist steering |
| Optional comparability | `gpt2-medium` | Author-Style Steering used GPT-2; direct vector comparison |

---

## 3. Instruments (steering representations)

### 3.1 Activation extraction spec (shared by all arms)

- Hook: residual-stream output after block `L` (verify HF `hidden_states` indexing off-by-one).
- Collection: forward-pass each document, then **mean-pool over content tokens** (exclude special/role tokens) → one vector per document.
- **Fixed preprocessing pipeline** (one pipeline across all attributes): mean-centring decision, standardization, ridge ε all fixed and documented. Mean-centring changes the extracted direction (Jorgensen et al. 2023) — so it is an **ablation**, not a silent choice.
- Corpora: 640+ docs/side for statistics (sample-size curve validated, not assumed); 300–500/side for fitting; 1,000/side held out for classifier calibration.
- Storage: float32 `.npz` per (model, layer, corpus, resample). Cache once; all arms reuse.

### 3.2 Arm A — closed-form vector (CAA baseline) and A′

- A: `b = μ_P − μ_N`, `W = 0`; inference `h' = h + α·b` (CAA: Rimsky et al. 2023, arXiv:2312.06681).
- A′: mean-centred difference (subtract global activation mean) — Jorgensen et al. 2023 ablation.
- Optional: linear-probe direction as a sanity variant (not a headline arm).

### 3.3 Arm F — closed-form covariance-matching operator (core falsification test)

Fit an affine map `T(h) = M h + c` matching first and second moments of neutral → style activations:

- `M* = Σ_P^{1/2} Σ_N^{−1/2}` (symmetric square roots; ridge-regularized), `c = μ_P − M* μ_N`,
- `W* = M* − I`; rank-r truncation via SVD: `W_r = trunc_r(W*)`, `b_r = μ_P − (I + W_r) μ_N`.

- **Precedent:** optimal mean/covariance-matching affine operators are established closed forms — Representation Surgery (Singh et al., ICML 2024, arXiv:2402.09631), LEACE (Belrose et al., NeurIPS 2023, arXiv:2306.03819), conceptors (Triantafyllopoulos et al. 2026, arXiv:2605.04980). This arm is the cheapest test of "is low-rank training necessary at all?" — if F matches C, the adapter-necessity claim is falsified cheaply.
- **F∞** = full-rank; **F_r** = rank-r truncated. Singular values of `W*` = closed-form complexity spectrum (canonical by construction, optimizer-free).
- Report spectra within this arm as structural references; do not claim they are the same object as a trained `BA`.

### 3.4 Arms B/C — trained low-rank + bias (moment matching)

- Parametrization: `W = BA`, `A ∈ R^{r×d}` ~ N(0, σ²), `B ∈ R^{d×r}` zero-init; `b` zero-init. Base frozen.
- Loss on standardized activations:

```
L = ||mean(f(X_N)) − μ_P||² + λ · ||cov(f(X_N)) − Σ_P||_F²,   f(h) = h + W h + b
```

- Sensitivity variants: whitened (Mahalanobis) mean term, log-Euclidean covariance term.
- Training: 300 steps, batch 16 docs, AdamW lr 1e-3, early stop on held-out moment loss; identical schedule across ranks/arms.
- **Precedent:** trained rank-1 (BiPO, Cao et al., NeurIPS 2024, arXiv:2406.00045; ReFT-r1, AxBench) and learned low-rank interventions (ReFT/LoReFT, Wu et al., arXiv:2404.03592).
- **B** = trained rank-1 + bias (isolates objective); **C_r** = trained rank-r (isolates form).
- **Canonicalization (required):** LoRA factorizations are non-unique; before ANY parameter-spectrum metric, canonicalize `BA` via QR-then-SVD (W2T, arXiv:2603.15990) and report the single canonical `ΔW`.
- Warm-start ablation: init `W` from `W_r` (F_r) vs random (proposal D-12).

### 3.5 Arm D — CE-LoRA control (the "just LoRA?" arm)

Same parametrization/capacity/schedule, trained with next-token CE on the attribute corpus. Tests whether contrastive/moment objectives produce different spectrum structure than ordinary brief fine-tuning. (Evaluation-sensitivity warning: method rankings can flip with evaluation choices — Jørgensen & Hansen 2026 — so D is compared on the full metric panel, not one score.)

### 3.6 Arms E / R / P

- **E**: identity no-op (bit-exactness, G0).
- **R**: random direction with effect-norm matched to A (control for "any perturbation helps").
- **P**: prompting reference — 0-shot and 4-shot, no fit; mandatory (AxBench).

### 3.7 Scale semantics (α calibration protocol)

1. Compute `δ₀ = mean_tokens ||b_A||` (vector arm's mean displacement on the neutral corpus).
2. For each other arm, find `α_unit` by bisection such that `mean ||Δh(α_unit)|| = δ₀` on the neutral corpus (alternative: match output style score at a reference point).
3. Sweep multiplier `m ∈ {0.5, 1, 2, 4, 8}`, `α = m·α_unit`.

- Adapters: scale both paths, `h' = h + α(W h + b)`; on L3 add **W-only** and **b-only** ablations.
- Apply at all positions (prompt + generated tokens).

### 3.8 Layer policy

Primary layer = 50% depth, fixed before results. Layer sweep {33%, 50%, 66%} on Shakespeare is descriptive only; no selection based on behavioral scores.

---

## 4. Data and matching

### 4.1 Attribute ladder

**Lean (locked)** — license-verified 2026-10-13 via HF API / GitHub / upstream pages; all ungated:

| Level | Attribute | Positive corpus | Contrast corpus | Eval prompts | License / status |
|---|---|---|---|---|---|
| L1 | Sentiment | SST-2 positive (`nyu-mll/glue`, `sst2`) | SST-2 negative | neutral story starters | GLUE license "other" (per-task); ungated; standard research use |
| L2 | Toxicity | Civil Comments (`google/civil_comments`) — toxic tail by label threshold | Civil Comments benign, matched | held-out toxic prompts | **CC0-1.0**; ungated |
| L3 | Shakespeare | Shakespeare plays (Gutenberg) | Marlowe/Jonson/Webster plays (same era/genre) | neutral narrative prompts | US public domain; strip PG header; split **by play** |
| L3 | Fable register | Moral Stories (`demelin/moral_stories`) | Wikipedia (`wikimedia/wikipedia`), length-matched; optional news variant CNN/DailyMail (`abisee/cnn_dailymail`, Apache-2.0) | neutral prompts | MIT; contrast CC-BY-SA-3.0 (attribution) |

Optional Lean variant: **ToxiGen** (`toxigen/toxigen-data`) as a secondary implicit-toxicity arm — MIT + CDLA-Permissive (GitHub), research-use note, ungated (optional tracking form).

**Extended (deferred):**

| Level | Attribute | Positive corpus | Contrast corpus | Eval prompts | Availability |
|---|---|---|---|---|---|
| L1 | Formality | GYAFC informal | GYAFC formal | neutral prompts | request-gated (email); fallback: constructed news-vs-social |
| L2 | Refusal | CAA-style synthetic harmful/comply pairs | harmless/comply pairs | AdvBench (MIT) + XSTest (CC-BY-4.0) | synthetic; no gate |
| — | Truthfulness (stretch) | TruthfulQA true claims (`truthfulqa/truthful_qa`) | false claims | TruthfulQA generation | Apache-2.0; ungated |

**Lean = 8 attributes**: the 4 English attributes above plus the 4 Indic attributes in §4.1b, all in the same run/config. Formality, refusal, and truthfulness move to Extended. No Lean dependency is gated or license-unknown.

### 4.1b E1b — Indic replication (Hindi + Telugu)

Confirms the same pattern outside English using **authentic, public-domain author corpora** (no synthetic stories). Same model (BLOOMZ-1B7), same layer, same steering families, probes, and α calibration — included in the single Lean run (`configs/lean.yaml`); restrict with `--attr sentiment_hi,story_hi,sentiment_te,story_te` if needed.

| Level | Attribute | Hindi | Telugu | License / access |
|---|---|---|---|---|
| L1 | Sentiment | `iam-tsr/hindi-sentiments` (MIT) — pos vs neg | `mounikaiiith/Telugu_Sentiment` (CC-BY-4.0) — pos vs neg | both ungated |
| L3 | Author style | Premchand vs Jaishankar Prasad (hi Wikisource) | Gurajada Apparao vs Kandukuri Veeresalingam (te Wikisource) | public-domain texts; MediaWiki API |

- Instrument: sentiment uses `xlm-roberta-base` + LR (per-attribute `classifier_model` override); story register uses per-language char n-gram TF-IDF.
- Prompts: constructed in-language neutral story prompts (40 per attribute).
- Confirmatory question per language: `H(story) > H(sentiment)` (the G1 analog) plus the `r*` behavior.
- Telugu is reported as **secondary**: BLOOM tokenizer fertility for Telugu is poor (itself a quantity of interest for E2); Hindi is the primary replication.
- Budget: the 4 Indic attributes are part of the same Lean run (~+60–90 min on bf16; fp32-only GPUs slower).
- Provenance: Wikisource page caches live under `data/raw/wikisource/<corpus>/`; combined corpora under `data/processed/wikisource_*.jsonl`.
- **Uniform L3 author budget: 200 docs/author** (pool 130 + classifier 45/25 per side; matching equalizes within each pair). Gurajada caps at 177 → the Telugu pair matches at ~150–170. Hindi remains the primary Indic test; Indic L3 entropy estimates are correspondingly noisier — the English ladder carries the complexity claim.

**Single-dimension contrast audit:** for every attribute, a held-out classifier test confirms the pair differs primarily on the target attribute (e.g., formality pairs do not smuggle in sentiment). Report audit accuracy.

### 4.2 Matching protocol

1. Equal total tokens by seeded subsampling.
2. KS test on document-length distributions (p > 0.05).
3. Match TTR **and MTLD** within ±5%; match base-model PPL deciles.
4. Tag/balance topic and register; publish full corpus statistics (tokens, length, TTR/MTLD, PPL, tags).
5. **Never** use the model's own unsteered samples as negatives (circularity).
6. **Sensitivity run:** complexity proxies on matched vs unmatched corpora; quantify the lexical-diversity confound; report raw and standardized (whitened/ZCA) analyses side by side.

### 4.3 Splits, leakage, sample-size curve

- Document-level splits; Shakespeare splits by play; classifiers trained on documents never used in fitting.
- Eval prompts neutral and distinct from style corpora; OOD split for L3 (genre change).
- Sample-size curve (closed-form arm, 2 attrs): sizes {16, 32, 80, 160, 320, 640}; report where metrics stabilize (saturation expected ≈80–160 per Hao et al.).

---

## 5. Probes

### P1 — Behavioral (per arm × attribute × scale × seed)

- **Strength `S`**: attribute classifier probability on generated text; calibrated against `S_human` (classifier score on held-out corpus text).
- **Coherence `C`**: (a) `PPL_base(gen) / PPL_base(unsteered)`; (b) PPL delta vs a style-neutral paraphrase baseline (style corpora are OOD — high PPL can mean style distance, not degradation); (c) LLM judge 1–5 with **style-blind rubric** (never mentions style), double position-swap, length controls; (d) repetition/self-BLEU, distinct-n.
- **Per-prompt variance**: report distributions and the fraction of prompts that flip behavior, not just means (Tan et al. 2024).
- **OOD gap** (Extended; optional spot-check in Lean): same measurements on the OOD prompt split; report generalization gap.
- **Safety side-effect monitor**: refusal/safety rate on fixed probe prompts for every arm (steering can interact with refusal geometry).
- Derived: `S(m)`, `C(m)`, `α_target` (min m with `S ≥ 0.7·S_human`), `α_brk`, **SFWC** (primary scalar), Pareto area.
- **Content preservation**: embedding cosine (centered/whitened) steered vs unsteered on same prompt.

### P2 — Effect/action spectrum (primary complexity probe)

- On held-out neutral corpus (~1,000 docs) at m = 1: `Δh = f(h) − h`.
- Action second moment `E[Δh Δh^T]` → eigenvalues: `H_action`, `erank_action = exp(H)`, stable rank, top-k energy (k = 1, 2, 4, 8), sparsity (fraction of dims with effect > ε), mean energy.
- **Effective-direction alignment**: cosine between the dominant action direction and the diff-in-means vector `b_A`; alignment of trained core with A and F subspaces.
- Sanity: A → rank 1 exactly; R → rank 1 but behavioral no-gain; bias-only → rank 1; F_r → ≤ r+1.
- This probe is parametrization-free (no canonicalization needed) — hence primary.

### P3 — Parameter spectrum (secondary; canonicalization required)

- Canonicalize `BA` (QR + SVD, W2T) before any metric.
- Report: `H_param`, effective rank, stable rank, top-k energy, energy split `||b||² vs ||W||_F²`, truncated-SVD explained variance vs r.
- **Stated limitation:** measured complexity is structure-dependent (non-identifiability; GCS; W2T) — the metric is a property of the *imposed canonical structure*, not a unique physical quantity. Every spectrum claim is paired with this caveat.

### P4 — Stability

- ≥5 seeds (init + data order) for trained arms; ≥5 bootstrap resamples for closed-form.
- Cross-seed: CKA (Kornblith et al. 2019), subspace principal angles, Procrustes-view distance (Ding et al. 2021), **Hungarian-matched direction overlap** (PW-MCC-style; SAE feature sharing is only ~53% even for SAEs — Paulo & Belrose 2025).
- Split-half refits: distance between sorted spectra (energy distance / Spearman).
- Report distribution (mean, sd, min), never the best run; LoRA seed variance may be lower than full FT (IJCNLP 2025) — measure, don't assume.

### P5 — Contrast separation pre-screen (competing predictor)

- Compute positive/negative activation separation per attribute before fitting: linear-probe AUC, centroid distance in z-scored units, silhouette (Braun et al. 2025: steering works when the contrast is coherent).
- Test whether separation predicts `r*` and difficulty **better than** spectral entropy (H3).

### P6 — Recovery test (adapter necessity, second angle)

- Principal angles between the trained core's top directions and (i) `b_A`, (ii) F subspace. If a trained rank-r core contains the diff-in-means direction, "adapter necessity" fails for that attribute (Causal-Steer-style extraction check).

---

## 6. Metrics: exact definitions

### 6.1 Classifiers and judges

| Attribute | Classifier | Validation |
|---|---|---|
| Sentiment | frozen `deberta-v3-base` features + LR | held-out accuracy/AUROC |
| Toxicity | ToxiGen-trained classifier or deberta probe | held-out AUROC |
| Formality | frozen encoder features + LR | held-out accuracy |
| Refusal | scripted refusal detection + judge | manual check of 100 samples |
| Style (Shakespeare) | char 3–5 gram TF-IDF + LR | held-out accuracy; report chance |
| Fable | char n-gram + LR + marker density | held-out accuracy; structural metrics secondary |
| Contrast audit | pair-difference classifier (per attribute) | must separate target attribute only |

Never measure style with the steered model's family. Judge protocol: style-blind coherence rubric, double-swap consistency reported, length-controlled, human subset (≥200) for agreement; **judge style bias is the dominant known bias (0.10–0.76)** — treat judge scores as secondary to classifier + PPL metrics.

### 6.2 Complexity metrics (metric panel, not one number)

- Eigenvalues `λ_i`; `p_i = λ_i/Σλ`; `H = −Σ p_i log p_i`; `erank = exp(H)`; stable rank `= ||M||_F²/||M||²₂`; top-k energy; sparsity.
- Effect-based (`H_action`, P2) **primary**; parameter-based (`H_param`, P3, canonicalized) **secondary**; closed-form spectrum (F) as structural reference.

### 6.3 Derived quantities

- **SFWC**: `S` at the largest common m where `C ≥ 0.9·C_base`; if never satisfied, use `m = α_brk − ε` and flag.
- **r\***: smallest rank whose behavioral endpoints (style score, PPL delta, coherence) are within CI of the best rank, **at matched parameter count and matched optimization**.
- **Difficulty `D`** (independent composite): mean of {normalized `α_target`, fit steps to moment-loss plateau, 1 − split-half spectrum Spearman, cross-seed overlap deficit}.

### 6.4 Statistics

- Mixed-effects models with random intercepts for prompt/item and seed; per-input variance reported (Tan et al.).
- Ladder: confirmatory L3 vs L1 (one-sided permutation, bootstrap CIs) + exploratory Jonckheere–Terpstra.
- Arm comparisons within attribute: paired permutation tests; Cliff's delta / Cohen's d.
- Correlations (Q3): Spearman ρ with bootstrap CI; exploratory given n ≈ 6.
- Multiplicity: Holm within test families; report the method × evaluation interaction (classifier vs judge vs PPL); all per-model, per-attribute results shown.

---

## 7. Analysis plan and figures

1. **F1 ladder** — complexity panel per attribute (3 levels), CIs, both effect-based and parameter-based.
2. **F2 saturation** — SFWC vs rank per attribute (F vs C arms), `r*` marked.
3. **F3 capture** — complexity vs difficulty scatter (ρ), overlaid with separation pre-screen predictions.
4. **F4 planes** — strength vs coherence heatmaps (m × arm) per attribute with per-prompt variance bands; break regions marked.
5. **F5 form vs objective** — 2×2 (rank × closed-form/trained) SFWC; D-arm overlay.
6. **F6 stability** — cross-seed CKA/principal angles/Hungarian overlap; closed vs trained stability contrast.
7. **F7 confound** — complexity vs TTR/MTLD; matched vs unmatched; raw vs whitened.

Decision flow: G0 → G1 → G2/G3/G4 → freeze memo (§9) → E2.

---

## 8. Pilot and QA checklist (bloomz-560m)

- [ ] Identity arm bit-exact vs base outputs
- [ ] Hook layer/index verified manually
- [ ] Canonicalization implemented (QR+SVD) and unit-tested on synthetic low-rank matrices
- [ ] Action-rank sanity: A = 1; F_r ≤ r+1; R behaves randomly
- [ ] Moment loss decreasing on train and held-out docs
- [ ] Matching diagnostics pass (KS, TTR/MTLD, PPL deciles)
- [ ] Separation pre-screen computed; contrast audit passes
- [ ] Judge style-blindness verified on style-swapped probe pairs
- [ ] Sample-size curve run (Lean: sentiment + Shakespeare, 16→256; Extended: 16→640)
- [ ] Full loop: fit → α calibration → generation (in-dist + OOD) → metrics → figures
- [ ] Reproducibility: seeds, config JSON per run, environment snapshot

---

## 9. Handoff contract to E2 (freeze memo contents)

1. Instrument spec: winning family + rank + layer + α calibration protocol.
2. G1–G4 outcomes (including any non-monotonicity findings).
3. Confirmed complexity metric (proxy panel + matching protocol + canonicalization + normalization).
4. Fit recipe and eval harness frozen verbatim.
5. Per-language pilot (HI/TE sentiment) selection.

---

## 10. Risks and confounds

| Risk | Mitigation |
|---|---|
| Lexical diversity inflates entropy (narrative > informational text) | Matched corpora + sensitivity run; effect-based probe primary; G4 |
| Parameter spectra unidentifiable | Canonicalization (W2T); parameter spectra secondary; structure caveat stated |
| Trained-vs-closed-form confound | F/D/B/C arms; warm-start ablation |
| α not comparable across steerers | Effect-norm calibration; report alternative score-matched calibration |
| Judge style bias (circularity for style papers) | Style-blind rubric, position swaps, length controls, human subset; classifier+PPL primary |
| PPL confound for OOD style corpora | Compare against style-neutral paraphrase baseline |
| OOD brittleness | OOD prompt split; report gaps |
| Safety side-effects | Refusal monitor in every arm |
| Non-monotone ladder (safety rank-1) | Pre-registered as falsifiable; report semantic vs representational complexity |
| Prompting beats steering (AxBench) | P arm mandatory; results situated relative to it |
| Anisotropy / massive activations | Raw + standardized analyses; flag extreme coordinates |
| Seed/feature instability | ≥5 seeds/resamples; CKA, principal angles, Hungarian overlap; distribution reporting |
| Low power (6 attrs) | Mixed-effects, paired designs, bootstrap, exploratory framing for ρ |

---

## 11. Citation corrections to the parent proposal (from verification)

These were checked against arXiv/ACL metadata and **applied to the parent proposal on 2026-10-13** (including adding the Arditi refusal direction paper as ref 47):

| Parent proposal | Correct |
|---|---|
| CAA = arXiv:2402.14817 | **Wrong ID** — that is *Cameras as Rays* (pose diffusion). CAA = **arXiv:2312.06681** (commonly cited as Rimsky et al.; arXiv metadata lists Panickssery first — note the discrepancy) |
| ICV = arXiv:2402.18543 (Liu et al. 2024) | **Wrong ID** — astrophysics paper. ICV = **arXiv:2311.06668**, Liu, Ye, Xing & Zou (ICML 2024) |
| “Tan et al. 2025, Reliability Challenges in Steering, ACL 2025 long 974” | **Mis-attribution** — that ACL paper is Da Silva et al. 2025, *Steering off Course* (arXiv:2504.04635); Tan et al. 2024 is the separate NeurIPS reliability paper (arXiv:2407.12404). “Some concepts unsteerable” is not supported by the 2025 abstract |
| “36 style attributes incl. children's book…” | **Partial mismatch** — arXiv:2607.01802 is Subbiah, Hall & McKeown on PLUME preferences; no 36-attribute count or ranking found |
| GCS = ICLR 2025 | Venue **unverified** (arXiv:2410.00153 verified) |
| Agarwal 2026 exact win rates (71.0/68.0) | **Unverified**; cite cautiously |
| Conceptors, non-identifiability, CRH, W2T, CAS-BiPO, Causal-Steer | Verified |

---

## 12. Prior-work anchors (wire-up map)

| Design element | Anchors |
|---|---|
| Diff-in-means / mean-centring | CAA arXiv:2312.06681; Jorgensen arXiv:2312.03813 |
| Trained rank-1 / learned low-rank | BiPO arXiv:2406.00045; ReFT arXiv:2404.03592; AxBench arXiv:2501.17148 |
| Closed-form covariance operators | Representation Surgery arXiv:2402.09631; LEACE arXiv:2306.03819; Conceptors arXiv:2605.04980 |
| Canonicalization | W2T arXiv:2603.15990 |
| Non-identifiability / structure caveat | arXiv:2602.06801; CRH arXiv:2605.01844; GCS arXiv:2410.00153 |
| Multi-dimensional features | Engels et al. arXiv:2405.14860 |
| Rank-1 safety precedent | Arditi et al. arXiv:2406.11717; CAS-BiPO 2026.findings-eacl.57 |
| Evaluation skeleton | Pres arXiv:2410.17245; Tan arXiv:2407.12404; Hao arXiv:2505.03189; Da Silva arXiv:2504.04635 |
| Separation pre-screen | Braun et al. 2025 (ICLR workshop) |
| Difficulty variability | Fan arXiv:2606.11599; Subbiah arXiv:2607.01802; Ionio blog (Shakespeare d=1.15 / Hemingway d=0.20) |
| Stability metrics | CKA arXiv:1905.00414; SVCCA arXiv:1706.05806; Ding arXiv:2108.01661; Paulo & Belrose arXiv:2501.16615; PW-MCC (ACL 2026 long 99) |
| Confounds | Pedashenko arXiv:2511.15210 (text ID); Timkey & van Schijndel arXiv:2109.04404; Sun arXiv:2402.17762; Li arXiv:2603.24543 |

---

*v1.1: Lean budget frozen for implementation. Next: implementation plan (module interfaces, data pipeline, run orchestration); Extended grid deferred until Lean results are in.*
