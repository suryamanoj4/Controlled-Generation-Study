# SAS E2 — Frozen Study Plan (Low-Resource Control & Style Steering)

> **Status:** frozen 2026-10-26 · supersedes the exploratory notes in [[SAS E2 — Low-Resource Style Steering Literature Review]]
> **Programme:** follow-up to [[SAS E1 — Complexity Study Design]] / [[SAS E1 — Results (Lean Run)]]
> **Framing:** *training-free inference-time control under low-resource conditions*, with an Indic core and English as the high-resource anchor
> **Hardware:** Colab T4 (16 GB, no bf16) · both models at **4-bit NF4**, fp16 compute

## 1. Research programme (3 phases)

| Phase | Question | Languages | Models | Core comparison |
|---|---|---|---|---|
| **1. Language control** | Can a steering vector force the output language in low-resource conditions, and does success track model competence? | `en, hi, bn, te, ta` | Qwen3-4B first; second model decided after results | DiffMean vector vs prompting vs closed-form rank-4 |
| **2. Author-style steering** | Can the same operators induce author style in Indic languages, and how does it compare with prompting? | `en, hi, bn, te, ta` (style-capable subset) | same as Phase 1 | vector vs closed-form vs prompting, judge + human |
| **3. Cross-lingual style transfer** | Do style directions transfer across languages, and does transfer follow family, script, or representation similarity? | EN → HI/BN/TE/TA; HI↔BN; TE↔TA; HI↔TE; HI↔TA; BN↔TE; BN↔TA | primary model (4B) initially | monolingual vs cross-lingual vs mean/residual directions |

Phase 1 is a replication + extension of published language-steering work (CLaS-Bench / Cegin et al.) with two additions: the **competence-gating analysis** and the **closed-form operator** arm.

---

## 2. Phase 1 — Language control (frozen protocol)

### 2.1 Languages
`en` (anchor), `hi`, `bn`, `te`, `ta`. Frozen. No Swahili/Amharic.

### 2.2 Models — **one complete run per model**
The unit of execution is a **complete model run**: all languages × conditions × metrics × analysis for one model, finished and archived before starting the next model. Do not interleave models.

**Run order and decision:**
1. **Qwen3-4B-Instruct** (`Qwen/Qwen3-4B`) — run first, complete.
2. After Phase 1 results, choose the second complete run:
   - **Qwen3-1.7B-Instruct** (`Qwen/Qwen3-1.7B`) — the *lower-competence* point (planned default);
   - **or a 7B/8B-class model at 4-bit** (e.g., `Qwen/Qwen3-8B` or `Qwen/Qwen2.5-7B-Instruct`) — the *higher-competence* ceiling, if the 4B result is clear and 1.7B looks redundant/too weak.
   - Decision rule: if the 4B shows a **flat** language gradient → a 7B/8B second model is more informative (extends the ceiling). If the 4B shows a **clear gradient** → 1.7B is the cheaper confirmation (extends the floor).
3. Each run uses **identical prompts, conditions, metrics and analysis**; only the model changes. Report models separately (no pooling).

### 2.3 Data (minimal)
| Component | Size | Notes |
|---|---|---|
| Vector fitting | **FLORES-200 devtest, 1,012 sentences/language** (min 500) | parallel, human; 200 held out for layer/α selection |
| Eval prompts | **100/language** | 70 from CLaS-Bench for comparability (license check) + 30 in-house/translated, native-checked |
| Human audit | **25 outputs/language/condition** | 2 annotators; κ ≥ 0.6 |
| Judge | all generated outputs | pinned model/version, temperature 0, fixed rubric |

No CulturaX-scale data needed for the first pass; FLORES is sufficient for language vectors.

### 2.4 Conditions and cells
- **Cells:** EN→{hi,bn,te,ta} (4) + native→same (4) = **8 cells**; optional EN→EN sanity (9th).
- **Conditions:** English instruction; native-language instruction; DiffMean vector (α=0.5/1/2); closed-form rank-4 (same α); random-vector control (norm-matched).
- **Layers:** 25/50/75% depth (4B: 9/18/27; 1.7B: 7/14/21), select best on validation.
- **Seeds:** 1 full seed; second seed only at the best α.
- **Expected volume:** ~3k generations for the pilot grid, ~9k–13k per complete model run (with all conditions/α).

### 2.5 Metrics
- **Forcing rate** — % outputs in the target language (script ratio + language ID; human-verified subset).
- **Relevance** — judge 0–4; harmonic mean with forcing (CLaS protocol).
- **Fluency** — judge 0–4 (and model PPL as secondary).
- **Empty/degenerate rate**, output length.
- **Competence proxies** — tokenizer fertility (tokens/word), base-model PPL on held-out FLORES, script fidelity under neutral prompting.
- **Statistics** — Wilson CIs; McNemar (steering vs prompting, same prompts); BH correction; mixed-effects `forcing ~ competence × method + (1|language) + (1|model) + (1|prompt)`.

### 2.6 Gates (Phase 1)
- **G-A0:** script/langID detector ≥95% on human-written FLORES samples.
- **G-A1:** EN baseline forcing ≥90%.
- **G-A2:** random-vector forcing within ±5pp of baseline.
- **G-A3:** judge–human κ ≥ 0.6 on audited outputs.
- **G-A4:** no condition with >20% empty generations.
- **Smoke test (before the full run):** 20 prompts × 5 languages × model, `enable_thinking=False`; primary model must reach script ratio ≥0.8 and empty <10% in **≥4/5 languages**. A weaker second model is acceptable — that is the low-competence point.

### 2.7 Model settings (Qwen3)
- `enable_thinking=False` (thinking traces would contaminate the language metric and multiply cost).
- Non-thinking sampling per model card: `temperature=0.7, top_p=0.8, top_k=20`; `presence_penalty=1.5` only as a sensitivity (can cause language mixing).
- Chat template applied; max_new_tokens 128–160.
- 4-bit NF4 + double quantization, fp16 compute; record bitsandbytes version.

### 2.8 Compute (T4, per complete run)
| Model | Phase 1 run |
|---|---|
| Qwen3-4B 4-bit | **~4–5 h** |
| Qwen3-1.7B 4-bit | **~2.5–3.5 h** |
| Qwen3-8B / Qwen2.5-7B 4-bit | **~7–9 h** (or a reduced grid in ~3–4 h) |

Pilot (4B only): **~1–1.5 h**. All estimates assume minimal data and one full seed; add 30–50% buffer for retries.

---

## 3. Phase 2 — Author-style steering (frozen protocol)

### 3.1 Languages and corpora
| Language | Positive | Contrast | Available docs |
|---|---|---|---|
| EN | Shakespeare | other early-modern drama (Marlowe/Jonson/Webster) | 1,730 / 1,196 (verified) |
| HI | Premchand | Prasad | 1,638 / 1,002 (verified) |
| BN | Tagore | Sarat Chandra | **verify PD + Wikisource availability** |
| TE | Gurajada | Veerasalingam | 177 / 678 (thin; cap at 177) |
| TA | PD prose pair (e.g., Pudhumaipithan vs Kalki) | — | **verify PD + Wikisource** |

### 3.2 Data budget
- Target **300–500 matched docs/side** where available; TE is capped at 177 (small-n, vector-only claim).
- Classifier instrument: **≥300/side** where possible; below that, report with an explicit small-n flag (no 45/25 classifiers this time).
- Prompts: **100/language**.
- Human eval: **25 outputs/language/style**, 2 annotators.

### 3.3 Conditions
Baseline; DiffMean vector; closed-form rank-4; in-language few-shot prompting; optional vector+prompting combination. α ∈ {0.5,1,2,4}; 2–3 layers; second seed at best α.

### 3.4 Metrics and gates
- Authorship classifier **validated on generations**; AUROC on held-out corpora ≥0.85.
- Judge: style match, coherence, language fidelity (position-swapped).
- LaBSE content preservation; diversity (distinct-2/rep-4); empty rate.
- Gates: language fidelity ≥0.8; judge–human κ ≥0.6; random-vector control ≈ baseline; no >20% empty.

### 3.5 Compute (T4)
- Primary model: **~4–5 h**; second model (if run): **~3–5 h**.

---

## 4. Phase 3 — Cross-lingual style transfer (frozen protocol)

### 4.1 Transfer matrix
- EN → HI, BN, TE, TA (4)
- HI ↔ BN (2), TE ↔ TA (2)
- HI → TE, HI → TA, BN → TE, BN → TA (4)
- **Total: 10–12 ordered pairs.**

### 4.2 Conditions per pair
Baseline; monolingual (target-fit) direction; cross-lingual (source-fit) direction; mean-of-other-language direction; residual direction; prompting. α ∈ {0.5,1,2}; 100 prompts/pair; primary model only initially.

### 4.3 Analysis
- Transfer gain vs **family distance, script overlap, embedding similarity, competence** (covariates; not a full factorial).
- Source/target asymmetry (a language as source vs target).
- Shared-vs-residual decomposition (Liu et al. 2026 method).
- Random-vector controls; McNemar + BH; human/judge validation on a subset.

### 4.4 Compute (T4)
- Primary model: **~2–3 h** for the 10-pair matrix.

---

## 5. Data budget summary (all phases)

| Phase | Fit data | Eval data | Human |
|---|---|---|---|
| 1 | FLORES 1,012/language | 100 prompts/language (8–9 cells) | 25/language/condition |
| 2 | 300–500 docs/side (TE 177) | 100 prompts/language, 2 styles | 25/language/style |
| 3 | reuse Phase 2 vectors | 100 prompts/pair × 10–12 pairs | 25/pair (subset) |

## 6. Decision gates and contingencies

1. **Smoke test** (4B, 5 languages): ≥4/5 languages with script ratio ≥0.8, empty <10%. If it fails ≥2 languages → use the Indic-tuned Qwen3-4B community model, Gemma-3-4B-it, or treat those languages as competence-floor cases only.
2. **4B Phase 1 result** decides the second model: 1.7B (floor) vs 7B/8B 4-bit (ceiling).
3. **Telugu style thinness**: vector-only claim; do not fit covariance/adapters on 177 docs.
4. **Tamil/Bengali corpora**: PD + Wikisource verification is a precondition for Phase 2; if unavailable, drop the style leg for that language and keep it in Phases 1/3.
5. **Judge reliability**: κ ≥0.6 before trusting automatic style/quality scores; otherwise revise the rubric or use human-only for that leg.

## 7. Implementation checklist (what code must be added)

- Qwen3 chat template + `enable_thinking=False`; non-thinking sampling defaults.
- 4-bit NF4 loading path (bitsandbytes) + quantization metadata in the run config.
- FLORES-200 download/prepare; **language-vector fitting** (mean-diff vs other languages).
- **Language-forcing metric** (script ratio + language ID + empty rate).
- Judge script (pinned model/version, rubric, JSON output) and human-eval sheet.
- Run layout `artifacts/phase1/<model_slug>/...`; one complete model run per invocation.
- Smoke-test script (20 prompts × 5 languages × model).
- Reuse E1 infrastructure: hooks, generation, resumable stages, analysis.

## 8. Reproducibility checklist

- Model IDs + revisions; quantization config (bitsandbytes version, NF4, fp16 compute).
- Tokenizer fertility per language; script-detector validation.
- Prompt set version; sampling parameters; max_new_tokens.
- Judge model/version/temperature/rubric; human κ.
- Seeds; code commit; per-run config JSON.

## 9. Open decisions (not blocking the freeze)

- Prompt licensing: CLaS-Bench's 70 vs a fully in-house 100-prompt set.
- Tamil and Bengali author-pair PD verification.
- Whether the second model is 1.7B or 7B/8B (decided after the 4B run).
- Human-eval pool and compensation.

## 10. References

- [[SAS E2 — Low-Resource Style Steering Literature Review]] — framing, language-count norms, implementation audit, data budgets.
- [[SAS E1 — Results (Lean Run)]] — complexity/representation background and the failed-Indic-generation lesson.
