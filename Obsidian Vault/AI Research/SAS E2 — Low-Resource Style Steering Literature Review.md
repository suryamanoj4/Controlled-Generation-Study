# SAS E2 — Low-Resource Style Steering: Literature Review

> **Status:** review complete (2026-10-26) · scope: activation steering / style control in multilingual and low-resource settings · method: web search + arXiv/ACL Anthology abstracts & full texts (URLs below)
> **Frozen plan:** [[SAS E2 — Frozen Study Plan (Low-Resource Control & Style Steering)]]
> **Purpose:** establish the framing, language-count norms, and experiment-format conventions for the proposed Study 2 (low-resource style control, Indic core)

## 1. How papers frame this area — five recurring archetypes

| Archetype | Core claim | Representative papers |
|---|---|---|
| **A. Benchmark / evaluation-gap** | "Multilingual steering has no standard evaluation" → contribute a benchmark + protocol | CLaS-Bench (32 langs, language steering); TST detox 9-lang metric benchmark; Few-shot style transfer (evaluation gap in LRLs) |
| **B. Cheap control for low-resource languages** | Steering is a training-free alternative to prompting/fine-tuning where data is scarce | Cegin et al. (11 langs synthetic data); CLAS (INLG 2026); Leino & Karlgren (control vectors, multilingual QA) |
| **C. Cross-lingual transfer** | Do directions learned in one language steer another? What is shared vs language-specific? | Figurative-language steering (6 langs, 408 comparisons); CLAS; ReCoVeR (18 langs); Maraia et al. (EACL 2026) |
| **D. Steering vs prompting / fine-tuning** | Is representation steering worth it compared with simple baselines? | AxBench (prompting wins); Cegin et al. (zero- vs few-shot); ReCoVeR |
| **E. Author/style control from small samples** | Can we induce a specific author/register without training? | Style Vectors (EACL 2024); StyleMC; personalized contrastive steering (2025) — **all English** |

**Where the gap is:** archetypes B–D are multilingual but test *language* or *concept* control; archetype E is author/style control but essentially monolingual English. **Author-style activation steering in Indic/low-resource languages is essentially unstudied.** The closest Indic work is style *transfer by training* (Krishna et al. ACL 2022; Mukherjee et al. INLG 2024) and Bengali as one of six languages in cross-lingual figurative steering (Liu et al. 2026).

## 2. How many languages papers actually use

| Paper | #Langs | Languages (scope) | Task |
|---|---|---|---|
| CLaS-Bench (Findings ACL 2026) | **32** | typologically diverse, high + low resource | language forcing |
| ReCoVeR (Findings EMNLP 2025) | **18** | multilingual | language steering |
| Cegin et al. (EMNLP 2026) | **11** | am, cs, da, de, he, id, jv, mt, sk, sl, su | synthetic data (sentiment/topic) |
| TST detox benchmark (LREC 2026) | **9** | ar, am, zh, en, de, hi, ru, es, uk | detoxification + metrics |
| Mukherjee et al. (INLG 2024) | **8** | hi, magahi, ml, mr, pa, or, te, ur | sentiment transfer |
| Krishna et al. (ACL 2022) | **7** | **hi, bn, kn, gu, te** + es, sw | formality / code-mixing |
| Liu et al. (2026) | **6** | en, zh, **bn**, es, it, de | figurative steering |
| XFORMAL (2021) | **3–4** | fr, pt-BR, it (+ en) | formality transfer |
| Leino & Karlgren (NEJLT 2025) | **2** | en, fi | language + style control |
| Style Vectors / StyleMC / AxBench | **1** | English | style control / steering benchmark |

**Norm by sub-area**
- *Language steering benchmarks*: 18–32 languages (breadth is the point).
- *Multilingual style/figurative steering*: **6–9 languages** — the practical sweet spot.
- *Author-style steering*: **1 language** (English) — the gap.
- *Models*: **2–4** per paper; **7–32B** model sizes (8B is the modal choice). A 1.7B model is **below the norm** — consistent with our BLOOMZ behavioral failure.
- *Human evaluation*: the strong papers include it or a validated judge: Liu et al. manually audit 800 outputs (93.5% agreement); Krishna et al. crowd-source 200 generations/model; CLaS-Bench proofreads translations with native speakers and uses judge-based relevance.

## 3. Experiment-format conventions (what "proven" looks like)

| Paper | Models | Conditions | Metrics | Evidence standard |
|---|---|---|---|---|
| **CLaS-Bench** | Llama-3.1-8B-Instruct, Aya-Expanse-8B | DiffMean, probe directions, language neurons, PCA/LDA, SAE, prompting | forcing success + semantic relevance (judge) → harmonic mean | 32 langs × 70 parallel questions; layer-wise analysis |
| **Cegin et al.** | Gemma-2 9B/27B, Llama-3.1 8B/70B | language & quality steering × 3 depths × α; zero- vs few-shot | downstream XLM-R F1 + 4 diversity metrics + LLM judge | 11 langs; statistical tests (Mann–Whitney) |
| **Liu et al. (figurative)** | Qwen3-8B/32B, Llama-3.1-8B, Ministral-3-8B | CAA directions, mono- vs cross-lingual, random-vector control, α=1 | target-behavior rate + coherence; judge + manual audit | 6 langs × 4 categories; 408 comparisons; McNemar + BH |
| **Krishna et al.** | mT5-based UR/diffUR variants | few-shot exemplars (3–10), backtranslation, Indic variants | classifier transfer acc, LaBSE sim, langID, copy rate, calibration | 7 langs; 4 Indic human-annotated sets; human eval |
| **Mukherjee et al.** | parallel/non-parallel/cross-lingual/shared; Llama2, GPT-3.5 | sentiment transfer | BLEU-style + human judgment | 8 Indic langs; 1k pos + 1k neg per language |
| **AxBench** | Gemma-2 2B/9B | prompting, SFT, LoRA, ReFT, DiffMean, SAE, probes | fluency + instruction-following + concept presence | 500 concepts; large-scale; held-out prompts |
| **Style Vectors** | English LLM(s) | contrastive style vectors, layer/strength sweep | style strength + fluency + human eval | single language, several styles |

**Format template that recurs:** 2–4 models × 5–9 languages × 1–4 attributes × {baseline, steering (vector/closed-form), prompting, optional fine-tuning} × 3–5 α values × {automatic metrics + validated judge/human subset} + matched controls (random vector) + statistical tests.

## 4. What this implies for Study 2 (recommendations)

**Framing (combine B + C + D, with A as a secondary contribution):**
> *Cheap, training-free style control in low-resource languages: how far does activation steering get, and does base-model language competence gate controllability?*

**Language matrix (5–7, matching the style-steering norm):**

| Role | Languages |
|---|---|
| Reference / high competence | English |
| Core (Indic, deepest data + human eval) | **Hindi, Telugu** (optionally Bengali) |
| Generalization (non-Indic LRL) | **Swahili** (precedented in Krishna et al.) + optionally Amharic |

**Models:** 2–4 at **7–8B** (the norm), e.g. Qwen2.5-7B-Instruct, Gemma-2-9B-it, Aya-23-8B; include **BLOOMZ-1b7 as the weak-competence anchor**. Run a script/fluency smoke test before any experiment.

**Conditions:** baseline, vector (DiffMean/CAA), closed-form rank-4, few-shot in-language prompting, optional LoRA; α ∈ {0.5, 1, 2, 4}; 2–3 layers.

**Metrics/gates (adopt the proven standards):** attribute strength (classifier *validated on generations*), language fidelity (script/langID), content preservation (LaBSE), fluency, diversity; judge + human subset (2–4 languages); random-vector control; statistical tests with correction; **language-fidelity gate before scoring**.

**Claims this design can defend:**
1. Low-rank steering is competitive with in-language prompting at high competence (or not — either is a result).
2. Controllability is predicted by language competence (fluency/fertility), not by language identity.
3. Style directions partially transfer across languages (source/target asymmetry, as in Liu et al.).
4. A reusable evaluation protocol for LRL style control.

## 5. Note on "StyleAdapterLM"

No peer-reviewed paper or established repo with that exact name was found. Closest matches: (a) **StyleAdapter** — an *image* stylization model; (b) a Sakana AI-Scientist PDF "multi-style adapter" for LMs (not peer-reviewed — do not cite); (c) the style-vector/steering literature above. If you have a specific URL/repo for StyleAdapterLM, share it and it can be assessed separately.

## 6. Implementation audit — English author-style steering vs our harness

| Work | Mechanism | Model / site | Strength control | Evaluation | Data | Same family as ours? |
|---|---|---|---|---|---|---|
| **Style Vectors** (Konen et al., EACL 2024; code MIT) | add style vector to hidden states (activation-based; also a trained variant) | Alpaca-7B; multiple layers | layer sweep, scale sweep | style classifiers, probing study, human inspection, prompting baseline | Yelp, Shakespeare, GoEmotions | ✅ **yes — same as A** (vector addition); trained variant ≈ our C family |
| **StyleVector** (Zhang et al., 2025) | training-free style vector in activation space, content/style disentanglement | LLM (7B-class); layer injection | α | PEFT/RAG baselines; 8% relative gain, 1700× less storage | personalized writing samples | ✅ yes (A + disentanglement) |
| **Ionio author-style pipeline** (2026, *blog — not peer-reviewed*) | `v_author = mean(h_author) − mean(h_all)`, L2-normalize, `h' = h + α v` | GPT-2; 7 layers from output side | α ∈ {0.5, 1, 2, 4} | paired baseline vs steered; LLM judge + MiniLM embedding similarity | 1.5k **LLM-synthesized** style pairs (circularity caveat) | ✅ **essentially identical to our A arm** |
| **CAA** (Rimsky et al., 2024) + IBM `activation-steering`, Microsoft `llm-steer-instruct` | contrastive activation addition; residual-stream hooks | various | α, layers | task metrics | contrastive pairs | ✅ yes (reference implementations/libraries) |
| **StyleMC** (Khan et al., 2023/24) | contrastively trained stylometric representation + **sequence-level discriminative re-scoring at decoding** | author-adapted LM | energy weight | style/fluency/anonymization | small author samples | ❌ **no** — decoding-time control, not activation addition |
| **Authorship style transfer with policy optimization** (2024) | RL / policy optimization | fine-tuned LM | training | style + content metrics | author corpora | ❌ no — training-based |

**Conclusion:** for *training-free style steering*, the reference implementations (Style Vectors, StyleVector, Ionio, CAA) are the **same mechanism we already have** — mean-difference vector injected into the residual stream at a chosen layer, with layer and α sweeps. Our **closed-form F-r arms are an extension** beyond that literature (they only use mean vectors), and our evaluation harness (classifier + PPL + content-sim) is compatible. Two gaps to close before reuse: add a **judge/human validation** (their standard) and **language-fidelity/script gate** (our lesson). "StyleAdapter" has **no LM implementation**; the image model and the Sakana AI-Scientist PDF are not usable references.

## 7. Recommended phased Study 2 plan

**Phase A — LRL control + steering vs prompting (fast, 1–2 weeks).**
- Languages (4): **English (anchor), Hindi, Telugu, Swahili**.
- Models (2): Qwen2.5-7B-Instruct + one of Gemma-2-9B-it / Aya-23-8B; BLOOMZ-1b7 as weak-competence anchor.
- Conditions: baseline, DiffMean vector, closed-form rank-4, prompting (English instruction vs native instruction).
- Metrics: language-forcing rate, semantic relevance (judge), **script fidelity gate**, fluency; 70–100 prompts.
- Claim: controllability is predicted by base-model competence, not language identity.

**Phase B — author-style steering (Indic core).**
- Languages: English (Shakespeare vs other drama), **Hindi (Premchand vs Prasad)**, **Telugu (Gurajada vs Veerasalingam)**; optional Bengali (Tagore vs Sarat Chandra).
- Conditions: vector (CAA/DiffMean), closed-form rank-4, in-language prompting; α ∈ {0.5, 1, 2, 4}; 2–3 layers.
- Metrics: authorship classifier **validated on generations**, judge (style/coherence/language), human subset (native speakers), LaBSE content preservation, diversity.
- Advantage over the English literature: **authentic public-domain corpora** instead of LLM-synthesized style pairs.

**Phase C — cross-lingual style transfer (the novel core).**
- Source × target matrix: EN→HI, EN→TE, HI→TE (and reverse), monolingual vs cross-lingual directions.
- Decompose directions into the mean of other-language directions vs residual (Liu et al. method), with random-vector controls and McNemar/BH tests.
- Claim: style directions have a shared cross-lingual component even when the surface style is language-specific; transfer strength tracks model competence.

**Scope guards:** 4–6 languages total, 2–3 models, one attribute per language pair, 3–5 α values, 2–3 layers, 100 prompts. Do not combine all phases in one run; A is the go/no-go screen, B establishes the style effect, C is the paper's novelty.

## 8. References (verified abstracts/full texts)

- Gurgurov et al. **CLaS-Bench: A Cross-Lingual Alignment and Steering Benchmark.** Findings ACL 2026. https://aclanthology.org/2026.findings-acl.1086/
- Cegin, Gurgurov, Al Ghussin, Ostermann. **Want Better Synthetic Data? Steer It: Activation Steering for Low-Resource Language Generation.** EMNLP 2026. https://arxiv.org/abs/2606.18389
- Liu, Zhan, Yao, Ghosh, Jiang. **Cross-Lingual Steering for Figurative Language Generation.** arXiv 2605.30443 (2026). https://arxiv.org/abs/2605.30443
- Krishna et al. **Few-shot Controllable Style Transfer for Low-Resource Multilingual Settings.** ACL 2022. https://aclanthology.org/2022.acl-long.514/
- Mukherjee et al. **Multilingual Text Style Transfer: Datasets & Models for Indian Languages.** INLG 2024. https://aclanthology.org/2024.inlg-main.41.pdf
- Protasov et al. **Evaluating Text Style Transfer: A Nine-language Benchmark for Text Detoxification.** LREC 2026. https://aclanthology.org/2026.lrec-1.358/
- Leino & Karlgren. **Controlling Language and Style of Multi-lingual Generative Language Models with Control Vectors.** NEJLT 11 (2025). https://aclanthology.org/2025.nejlt-1.1/
- Pokharel, Agrawal, Nagar. **Cross-Lingual Activation Steering for Multilingual Language Models (CLAS).** INLG 2026. https://arxiv.org/abs/2601.16390
- Wu et al. **AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders.** ICML 2025. https://arxiv.org/abs/2501.17148
- Konen et al. **Style Vectors for Steering Generative Large Language Models.** Findings EACL 2024. https://aclanthology.org/2024.findings-eacl.52/
- Patel et al. **Learning to Generate Text in Arbitrary Writing Styles (StyleMC).** arXiv 2312.17242. https://arxiv.org/abs/2312.17242
- ReCoVeR (language steering, 18 languages). Findings EMNLP 2025. https://doi.org/10.18653/v1/2025.findings-emnlp.1056
- Briakou et al. **XFORMAL: A Benchmark for Multilingual Formality Style Transfer.** NAACL 2021. (3 languages + English)
- Bartoszcze et al. **Representation Engineering for Large-Language Models: Survey and Research Challenges.** arXiv 2502.17601. https://arxiv.org/abs/2502.17601
- Konen et al. code: https://github.com/DLR-SC/style-vectors-for-steering-llms
- Zhang et al. **Personalized Text Generation with Contrastive Activation Steering (StyleVector).** arXiv 2503.05213. https://arxiv.org/abs/2503.05213
- Rimsky et al. **Steering Llama 2 via Contrastive Activation Addition (CAA).** arXiv 2312.06681. https://arxiv.org/abs/2312.06681
- Patel. **Author-Style Steering via Contrastive Activation Vectors.** Ionio blog, Jan 2026 — *not peer-reviewed*. https://research.ionio.ai/posts/2025/author-style-steering-activation-vectors
- IBM. **activation-steering library.** https://github.com/IBM/activation-steering
- Stolfo et al. **Improving Instruction-Following in Language Models through Activation Steering.** ICLR 2025. https://github.com/microsoft/llm-steer-instruct
- **Authorship Style Transfer with Policy Optimization.** arXiv 2403.08043. https://arxiv.org/abs/2403.08043
