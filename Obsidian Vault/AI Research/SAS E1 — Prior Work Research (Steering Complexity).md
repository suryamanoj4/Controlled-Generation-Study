# SAS E1 — Prior Work Research (Steering Complexity)

> Prepared 2026-10-12 for the E1 design (representation-complexity study on a frozen small decoder-only LM).
> Method: targeted lookups against the arXiv API, ACL Anthology, ICLR/NeurIPS proceedings pages, official repos, and web search. Sources are tagged **[peer-reviewed]**, **[preprint]**, **[blog]**, **[repo]**. Non-verifiable claims and proposal-level citation errors are listed in §“Verified vs unverified”.

E1 compares rank-1 vector + bias (closed-form CAA), trained rank-1 + bias, and LoRA-style low-rank adapters (r=1..32) across attribute levels L1 (sentiment/formality), L2 (toxicity/safety/refusal), L3 (author style, narrative register), measuring representation complexity (spectral entropy, effective rank, sparsity, layer spread) and behavioral quality (strength–coherence sweeps).

---

## 1. Measures of concept/behavior complexity in LLM representations

| Work | Object measured | Complexity notion | Headline result |
|---|---|---|---|
| Engels et al. 2024, *Not All Language Model Features Are One-Dimensionally Linear*, arXiv:2405.14860 **[preprint]** | SAE features (GPT-2, Mistral-7B) | Irreducible dimension | Some features are irreducibly multi-dimensional (e.g., circular day/month features); not decomposable into independent or non-co-occurring 1-D features |
| Park, Choe & Veitch 2023, *The Linear Representation Hypothesis and the Geometry of LLMs*, arXiv:2311.03658 **[preprint]** | Concept directions | Identifiability under a chosen inner product | “Linear representation” is only defined relative to counterfactuals; a causal inner product is needed to make distances/projections meaningful; probing and steering are two sides of the same formalization |
| Zhao et al. 2024, *Beyond Single Concept Vector: Gaussian Concept Subspace (GCS)*, arXiv:2410.00153 (proposal claims ICLR 2025 — venue unverified) **[preprint]** | Concept subspace | Data/training variance of single vectors | Single concept vectors vary with data and training; a Gaussian subspace over concept vectors is more robust and still usable for intervention (sampling a vector) |
| Venkatesh & Kurapath 2026, *On the Non-Identifiability of Steering Vectors*, arXiv:2602.06801 **[preprint]** | Steering interventions | Behavioral equivalence class | Orthogonal perturbations of a steering vector are near-equivalent behaviorally; null-space dimensionality estimated by SVD of activation covariance; structure must be imposed for a representation claim |
| Gao, Zhang & Liu 2026, *The Cylindrical Representation Hypothesis (CRH)*, arXiv:2605.01844 **[preprint]** | Concept geometry | Axis + normal-plane dimensionality | Beyond LRH: a central axis drives generation, a normal plane controls steering sensitivity; overlapping concepts produce sample-specific axis-orthogonal structure, explaining instability/failure |
| Han, Neri & Jiang 2026, *W2T: LoRA Weights Already Know What They Can Do*, arXiv:2603.15990 **[preprint]** | LoRA ΔW | Canonical form | A LoRA update has infinitely many factorizations; QR-then-SVD gives a canonical form, needed before any weight-derived analysis |
| Pedashenko et al. 2025/26, *Unveiling Intrinsic Dimension of Texts*, arXiv:2511.15210 / EACL 2026 **[peer-reviewed]** | Text embeddings | Intrinsic dimension (ID), effective rank (Roy & Vetterli) | ID is domain- and style-sensitive: informational text is low-ID; opinionated/narrative/personalized writing is substantially higher-ID; ID is complementary to entropy metrics |
| Aghajanyan et al. 2021 (ACL) **[peer-reviewed]** | Fine-tuning parameter space | Intrinsic dimension | Task adaptation lives in a very low-dimensional subspace — the founding prior that rank-1/low-rank may be enough |
| Arditi et al. 2024, *Refusal ... Single Direction*, arXiv:2406.11717 (NeurIPS 2024) **[peer-reviewed]** | Refusal behavior | Direction count | Refusal is mediated by one direction across 13 chat models up to 72B |
| Biderman et al. 2024, *LoRA Learns Less and Forgets Less*, arXiv:2405.09673 **[preprint/TMLR]** | LoRA ΔW | Rank | Rank matters for capacity claims; behavior differs systematically from full FT |

**Why this matters for E1**
- There is a live, unresolved tension: some behaviors are exactly rank-1 (refusal), while features can be irreducibly multi-dimensional and steering vectors are non-identifiable. E1 must therefore *impose* a canonical structure (rank spectrum of a canonicalized operator) and report that the measured complexity is structure-dependent — a stated limitation, not a hidden one.
- GCS and the non-identifiability result jointly demand that E1 quantify extraction variance (data/bootstrap/seed) before attributing spectral entropy to attribute complexity.
- No retrieved work reports SVD-spectrum entropy / effective rank *of steering representations* as an attribute-complexity axis; the closest analogues are SAE-feature dimensionality and text-level ID. This gap is E1's measurement contribution, but it is unoccupied for a reason (metric fragility) — see §6–7.
- CRH predicts precisely the strength–coherence/failure-region phenomena E1 plans to sweep, and offers a testable alternative predictor (normal-plane geometry) to spectral entropy.

## 2. Steering vectors vs LoRA/adapters; rank-1 vs low-rank; trained and sparse steering

| Work | Intervention | Training | Key comparison |
|---|---|---|---|
| CAA (Rimsky et al. 2023; arXiv:2312.06681) **[preprint]** | rank-1 + α | none (difference-in-means) | Reference closed-form vector; note the proposal's arXiv ID (2402.14817) is wrong — see “Verified vs unverified” |
| BiPO (Cao et al. 2024, NeurIPS 2024; arXiv:2406.00045) **[peer-reviewed]** | trained rank-1 | DPO-style preference optimization | Trained vectors beat CAA-class extraction on TruthfulQA; supports “gradient-trained steering” arm |
| CAS-BiPO (Doan, Le & Nguyen, Findings EACL 2026; aclanthology 2026.findings-eacl.57) **[peer-reviewed]** | sparse learned mask over vector | end-to-end regularized training | 97–100% of dense effectiveness with 10–30% of dimensions; 94.3% of steering effect in 26.7% of dims |
| AxBench (Wu et al. 2025, ICML 2025; arXiv:2501.17148) **[peer-reviewed]** | benchmark | — | Prompting > finetuning > representation methods for steering topics; SAEs not competitive; DiffMean best concept detector; rank-1 learned ReFT-r1 is competitive |
| Jørgensen & Hansen 2026, arXiv:2605.31183 **[preprint]** | SAE steering | — | Partial rebuttal: SAE steering can approach LoRA reference when evaluation choices are matched — method ranking is evaluation-sensitive |
| Causal-Steer (Wang, Yao & Chen, ICLR 2026) **[peer-reviewed]** | style vector extracted from LoRA ΔW | per-model LoRA fine-tune | LoRA as causal intervention; continuous bidirectional α; inverse of E1's direction (LoRA → vector) |
| ReFT/LoReFT (Wu et al. 2024; arXiv:2404.03592) **[peer-reviewed, NeurIPS 2024]** | learned rank-r linear subspace intervention | supervised | The canonical “learned low-rank intervention” precedent for E1's adapter arm |
| “Shifting Perspectives” (Siddique et al., Findings EACL 2026; arXiv:2503.05371) **[peer-reviewed]** | 8 bias vectors vs LoRA-r16 vs prompting | steering: none; LoRA: 3 epochs | Steering wins on BBQ (8/9 axes), loses to FT on PPL-based StereoSet; *not* compute-matched |
| Weight arithmetic (Fierro & Roger 2025; arXiv:2511.05408) **[preprint]** | weight-space vectors | LoRA FT to get θ± | Weight vs activation steering vs one activation baseline (persona vectors) |
| Steer2Adapt (arXiv:2602.07276) **[preprint]** | reusable low-dim subspace | few-shot composition | A notably small shared subspace suffices for effective steering across tasks |
| “Why Steering Works” (Xu et al., ACL 2026 long 1463; arXiv:2602.02343) **[peer-reviewed]** | unified framework | — | LoRA and activation steering are both “dynamic weight updates”; preference–utility regularities as a function of control strength |
| “From Weights to Activations” survey (Ostermann et al. 2026; arXiv:2604.14090) **[preprint]** | survey | — | Steering is rarely evaluated in the same framework as fine-tuning/PEFT — the gap E1 addresses |
| Ionio Author-Style Steering (Patel 2026) **[blog]** | rank-1 per author, GPT-2 117M | none (LLM-synthesized pairs) | Layer 5 (from top) best; α 0.5–4.0; per-author effect sizes differ sharply: Shakespeare d=1.15, Austen d=0.96, Hemingway d=0.20 |

**Why this matters for E1**
- Every arm E1 plans has a precedent, but they have never been placed on a single cost-matched rank axis: closed-form rank-1 (CAA), trained rank-1 (BiPO/ReFT-r1), learned low-rank (LoReFT), sparse (CAS-BiPO), and full-covariance closed forms (Representation Surgery/conceptors, §5). E1's “is a low-rank adapter necessary over a vector” question is genuinely open.
- AxBench warns that prompting is the strongest steering baseline on many attributes; E1 must include a prompted/few-shot baseline or the LoRA-vs-vector result is uninterpretable.
- The Jørgensen & Hansen rebuttal implies E1's rank conclusions must be robust to evaluation choices (classifier, judge, α calibration); report method × evaluation interactions.
- Causal-Steer and Causal-Steer-like inverse pipelines show that style vectors can be *extracted from* adapters; E1 can use this as an alignment check (does a trained rank-r adapter contain a dominant style direction?).

## 3. Established evaluation protocols for steering

| Protocol / finding | Source | Detail |
|---|---|---|
| Four evaluation requirements | Pres, Ruis, Lubana & Krueger 2024 (NeurIPS 2024 FM-Workshop; arXiv:2410.17245) **[peer-reviewed workshop]** | Downstream-like contexts; account for model likelihoods; standardized cross-behavior comparison; baseline comparisons. Their pipeline exposes intervention brittleness |
| Variance and overestimation | Tan et al. 2024 (NeurIPS 2024; arXiv:2407.12404) **[peer-reviewed]** | Steerability highly variable across inputs; spurious biases contribute per input; OOD brittleness to prompt changes; current metrics overestimate effectiveness |
| CAE patterns/limits | Hao et al. 2025 (ICLR 2025; arXiv:2505.03189) **[peer-reviewed]** | Diminishing returns ≈80–160 samples; only in-distribution reliable; adversarial inputs can reverse behavior; steering harms perplexity; larger models more resistant |
| Model-scale reliability | Da Silva et al. 2025 (ACL 2025 long 974; arXiv:2504.04635) **[peer-reviewed]** | 36 models / 14 families; DoLa, function vectors, task vectors often show no improvement or degradation |
| Style-transfer metrics | NAACL-SRW 2025 “Evaluating TST Evaluation” (2025.naacl-srw.41) **[peer-reviewed]**; TSTBench 2025 **[preprint]**; Jin et al. 2022 survey **[peer-reviewed]** | Style strength via held-out classifier; content preservation via BLEU/SBERT (imperfect); fluency via PPL; automated metrics only weakly reliable → triangulate with humans |
| Authorship attribution / stylometry | Kumarage et al. 2023 (IEEE CyberC) **[peer-reviewed]**; code stylometry arXiv:2506.17323 **[preprint]**; Frontiers in AI 2026 stylometric framework **[peer-reviewed]** | Stylometric + transformer classifiers discriminate authorship at high accuracy; lexical diversity and function-word frequency are top features → useful style-strength instruments, but their features overlap with the proposed complexity confound |
| LLM-judge bias | Shi et al. 2024 (arXiv:2406.07791) **[preprint]**; Soumik 2026 (arXiv:2604.23178) **[preprint]**; plus established position/verbosity/self-preference results (Zheng et al. 2023; Wang et al. 2024; Panickssery et al. 2024) **[preprints, not re-fetched]** | Position-consistency 70–77% for many judges; in a 2026 systematic study **style bias is the dominant bias (0.10–0.76), far exceeding position bias (≤0.04)**; mitigations: swapping, ensembles, rubrics, length controls |

**Why this matters for E1**
- Adopt Pres et al.'s four criteria as the protocol skeleton, and report *per-input/per-prompt* variance (Tan et al.), not just mean improvement.
- For strength–coherence sweeps: schedule α to failure, define the failure region operationally (e.g., first α where PPL increases >τ or judge coherence falls below baseline CI), and calibrate arms to a common effect scale (see §Design implications).
- Style bias in judges is a circularity hazard for a style-transfer paper: a judge that rewards stylistic flourish will reward the treatment. Use style-blind coherence rubrics, position swaps with consistency reporting, length controls, and human evaluation.
- PPL-based coherence is necessary but confounded: style corpora are OOD for the base model, so higher PPL can reflect style distance rather than degradation (Hao et al. show steering harms PPL). Report PPL deltas against norm-matched controls and against a style-neutral paraphrase baseline.

## 4. Attribute difficulty / steerability

- **Fan, Cheng, Li & Feizi 2026, *When is Your LLM Steerable?*, arXiv:2606.11599 [preprint]:** 150 concepts at multiple abstraction levels; some are easy (consistently formal tone), some effectively impossible (exactly 3 bullet points); steerability is predictable from early internal states. *Why E1:* existing evidence for a difficulty ladder exists but is concept- and prompt-conditioned; a per-attribute difficulty measure correlated with E1's complexity metrics is a natural validation.
- **Subbiah, Hall & McKeown 2026, arXiv:2607.01802 [preprint]:** trait expressibility varies substantially across PLUME preferences (including children's-book style, third-person perspective, informal tone); positive/negative example transfer degrades; vector composition often breaks sign control. *Why E1:* L3 attributes are not uniformly steerable; per-attribute effect sizes must be treated as a distribution, and composition is off the table until single-attribute behavior is stable.
- **Braun et al. 2025 (ICLR 2025 workshop) [peer-reviewed workshop]:** steering is reliable only when the target behavior is represented by a coherent direction (good separation of positive/negative activations); prompt type matters. *Why E1:* “complexity” may be better operationalized as *coherence of the activation contrast* than as dictionary size; a cheap pre-screen (separation/coherence) can predict rank needs before fitting.
- **Refusal (Arditi et al. 2024) and safety sparsity (CAS-BiPO) [peer-reviewed]:** L2 safety behaviors are often low-rank and sparse, so an entropy ladder sentiment < safety < style is **not** guaranteed by semantics — “safety” is semantically complex but representationally simple in several documented cases. *Why E1:* state the ladder as falsifiable and expect non-monotonicity; distinguish *semantic* complexity from *representation* complexity in the paper.
- **Tales of Morality (Findings EMNLP 2025, 2025.findings-emnlp.1029) [peer-reviewed]:** LLMs over-emphasize Care and flatten narrative arcs even under moral conditioning. *Why E1:* supports structural (arc-shape, foundation-distribution) metrics for L3 register control beyond style classifiers.
- **Ionio blog [blog]:** within author style, difficulty varies by author (d=1.15 vs 0.20). *Why E1:* use ≥2–3 L3 attributes (Shakespeare, Austen, fable register) rather than one.

## 5. Fitting objectives for steering representations

| Objective | Form | Cost | Precedent |
|---|---|---|---|
| Difference-in-means (CAA) | `v = μ+ − μ−` | closed form | Rimsky et al. 2023 |
| Mean-centred diff | subtract global activation mean | closed form | Jorgensen et al. 2023, arXiv:2312.03813 (mean-centring; also function vectors) |
| Preference-optimized vector (BiPO) | DPO-style ranking of concept vs neutral responses | gradient, hours | Cao et al. NeurIPS 2024 |
| Activation scaling | learn per-dimension multiplicative scalars | gradient, sparse | aclanthology 2024.findings-emnlp.479 |
| SAE-Targeted Steering (SAE-TS) | approximate causal effect, construct targeted vector | closed form + fit | arXiv:2411.02193 [preprint] |
| LEACE | closed-form affine erasure minimizing embedding change | closed form | Belrose et al. NeurIPS 2023, arXiv:2306.03819 |
| Representation Surgery | optimal **mean-matching** and **covariance-matching** affine operators (closed form) | closed form | Singh et al. ICML 2024, arXiv:2402.09631 |
| Conceptors | soft projection from activation covariance; Boolean composition; “conceptor quota” layer diagnostic (r≤0.96) | closed form (SVD) | Triantafyllopoulos et al. 2026, arXiv:2605.04980; original conceptor theory Jaeger 2014 (not re-fetched) |
| DAS / causal alignment | gradient search over subspace alignments | gradient | Geiger et al. ICML 2024, arXiv:2303.02536 |
| RAVEL benchmark | disentanglement evaluation | — | Huang et al. ACL 2024, arXiv:2402.17700 |
| PCA / whitening | remove dominant/anisotropic directions or full-covariance ZCA | closed form | Mu & Viswanath 2018 (all-but-the-top); Su et al. 2021 whitening (not re-fetched) |
| Procrustes alignment | orthogonal map between representation spaces | closed form | Ding et al. NeurIPS 2021, arXiv:2108.01661; MUSE (Conneau et al. 2018, not re-fetched) |

**Why this matters for E1**
- The proposal's claim that no closed-form operator handles “complex” attributes is an empirical question, not a fact: Representation Surgery and conceptors give *closed-form covariance-based affine operators* that strictly generalize rank-1 addition and cost almost nothing. E1 should add at least one such arm — if full-covariance closed form matches the trained low-rank adapter on L3, the paper's adapter-necessity claim is falsified cheaply.
- Objectives differ in what spectrum they can reveal: diff-in-means is rank-1 by construction (H=0, trivially), trained spectra depend on initialization/regularization, and canonicalization (W2T) is required for LoRA. Conceptors/SVD spectra are canonical by construction — use them as sanity references.
- Mean-centring, whitening, and LEACE are legitimate preprocessing choices that change measured entropy; E1 must fix one pipeline across attributes and report the alternatives as an ablation.

## 6. Similarity / stability evaluation

- **CKA** (Kornblith et al. ICML 2019, arXiv:1905.00414) and **SVCCA** (Raghu et al. NeurIPS 2017, arXiv:1706.05806) **[peer-reviewed]**: standard representation-similarity tools; CKA measures beyond invertible-linear invariance and can profile layer/representation dimensionality. *Why E1:* use cross-seed CKA/subspace principal angles as the stability report for learned cores.
- **Ding et al. NeurIPS 2021 (arXiv:2108.01661) [peer-reviewed]:** existing similarity measures disagree; statistical testing + Procrustes-view distances are needed to say whether two representations are “the same”. *Why E1:* any claim that two seeds/attributes yield the same adapter must be tested, not eyeballed.
- **SAE seed instability:** Paulo & Belrose 2025 (arXiv:2501.16615) — only ~53% of TopK SAE features shared across seeds; “Unstable Features, Reproducible Subspaces” (arXiv:2606.12138, 2026) **[preprints]**; ACL 2026 long 99 proposes PW-MCC (assignment-based consistency) as a standard reporting axis **[peer-reviewed]**. *Why E1:* feature-level identifiability fails under seed changes even in the tool E1 might use to interpret adapters; report assignment-matched direction overlap (Hungarian matching on cosine), not just aggregate spectra.
- **Steering-specific instability:** GCS (data/training variance), non-identifiability (null-space), Tan et al. (input-level variance), Hao et al. (sample-size saturation ≈80). *Why E1:* stability is a first-class result in this literature; multi-seed (−init, −data order, −bootstrap resampling) reporting is expected.
- **LoRA seed variance:** IJCNLP 2025 short (2025.ijcnlp-short.3) **[peer-reviewed]:** LoRA fine-tuning shows lower random-seed variance than full fine-tuning. *Why E1:* do not assume trained arms are unstable; measure both extraction (closed form) and training (rank-r) stability.

## 7. Confounds and pitfalls

1. **The contrast corpus *is* the feature.** Mean-centring and corpus choice change the direction (Jorgensen et al. 2023); Ionio subtracts the global cross-author mean. Unsupervised bias-steering work (Findings EMNLP 2025, 2025.emnlp-main.1439) **[peer-reviewed]** shows training-set disparity thresholds materially change vector quality. *E1 action:* fixed, published contrast sets; one-dimension-difference pairs; report corpus statistics.
2. **Lexical-diversity → rank confound.** Narrative text has higher ID than informational text (Pedashenko et al. 2025/26); stylometry studies list lexical diversity as a top authorship feature (Frontiers 2026). *E1 action:* match corpora on token count, length distribution, TTR/MTLD, topic/register, and PPL; report all; prefer effect-based (Δh) complexity as primary and parameter-spectrum metrics as secondary; optionally control statistically.
3. **Data scale.** Diminishing returns ≈80–160 samples (Hao et al. 2025); one-shot optimization is highly variable but can generalize (LessWrong reports; **[blog]**). *E1 action:* fix extraction size above saturation and report a sample-size curve for the closed-form arm.
4. **Anisotropy / rogue dimensions / massive activations.** A single dimension can dominate cosine similarity (Timkey & van Schijndel, EMNLP 2021, arXiv:2109.04404; 2021.emnlp-main.372) **[peer-reviewed]**; massive activations act as fixed bias terms (Sun et al. 2024, arXiv:2402.17762) **[preprint]**. *E1 action:* standardize/whiten activations for similarity analyses (or report both raw and standardized), and exclude/report massive activation coordinates.
5. **OOD brittleness.** In-distribution gains do not transfer to changed prompts (Tan et al. 2024; Hao et al. 2025). *E1 action:* include an OOD prompt split; report generalization gaps.
6. **Judge circularity for style.** Style bias dominates judge bias in systematic measurement (Soumik 2026). *E1 action:* style-blind coherence prompts, double-swap, length controls, human subset.
7. **Safety side-effects.** Steering vectors geometrically interact with the refusal direction (Li et al. 2026, arXiv:2603.24543 **[preprint]**). *E1 action:* monitor refusal/safety rates as a control in every arm.
8. **Factorization non-identifiability.** LoRA ΔW spectra are ambiguous (W2T); α is not comparable across arms unless effect-normalized. *E1 action:* canonicalize, norm-match α, and pre-register the saturation criterion.

---

## Verified vs unverified

| Proposal claim / reference | Status after verification |
|---|---|
| CAA = arXiv:2402.14817 | **Wrong ID.** 2402.14817 = *Cameras as Rays: Pose Estimation via Ray Diffusion* (Zhang et al.). CAA is arXiv:2312.06681, *Steering Llama 2 via Contrastive Activation Addition* (arXiv metadata lists first author as N. Panickssery; commonly cited as Rimsky et al. — name discrepancy worth noting in the paper) |
| ICV = arXiv:2402.18543 (Liu et al. 2024) | **Wrong ID.** 2402.18543 = an astrophysics paper. ICV is Liu, Ye, Xing & Zou 2023, arXiv:2311.06668, *In-context Vectors* (ICML 2024) |
| Tan et al. 2025, *Reliability Challenges in Steering*, ACL 2025 long 974 | **Mis-attribution.** The ACL paper is Da Silva, Sethuraman, Rajagopal, Hajishirzi & Kumar (2025), *Steering off Course* (arXiv:2504.04635); it studies DoLa/function vectors/task vectors over 36 models/14 families. Tan et al. (2024) is the separate NeurIPS reliability paper (arXiv:2407.12404). “Some concepts unsteerable” is **not** supported by the 2025 abstract |
| “Steering-vectors limits study (2026), 36 style attributes incl. children's book/third person/stream of consciousness” | **Partial mismatch.** arXiv:2607.01802 = Subbiah, Hall & McKeown, *On the Limits of Steering Vectors for Preference-Aligned Generation*; attributes from PLUME include children's-book style, third-person perspective, informal tone, simile. No “36 attributes” count and no style-steerability ranking found |
| Agarwal 2026, *Cross-Architecture Steering Transfer*, arXiv:2608.05164 | **Verified** (single author; systematics over 5 models/20 directed pairs; “suggestive discontinuity near 1.7B”). Exact win-rate numbers (71.0/68.0) **not verified**; arXiv ID-month vs published-date metadata is internally inconsistent — cite cautiously |
| GCS = ICLR 2025 | **Not verified** (arXiv:2410.00153 verified; venue claim unconfirmed) |
| Conceptors (arXiv:2605.04980), non-identifiability (2602.06801), cylindrical hypothesis (2605.01844), W2T (2603.15990), CAS-BiPO (Findings EACL 2026), Causal-Steer (ICLR 2026) | **Verified** (arXiv/ACL/ICLR proceedings; CAS-BiPO authors Doan, Le & Nguyen; Causal-Steer authors Wang, Yao & Chen) |
| Ionio Author-Style Steering: static grid α, LLM-synthesized dataset, per-author results, layer×α sweep | **Verified from the blog itself** (1500+ LLM-transformed pairs; GPT-2 117M; layers 1–7; α 0.5–4.0; Layer 5 best; Shakespeare/Austen/Hemingway d=1.15/0.96/0.20). **Blog, not peer-reviewed**; “adaptive α listed as future work” and “dataset circularity acknowledged” consistent with the blog sections but not independently audited |
| Proposal's “dataset circularity” critique of Ionio | Supported: pairs are LLM-transformed, not authentic author text |
| Foundational cites not re-fetched this session (Roy & Vetterli 2007; Jaeger 2014; Vig et al. 2020; Mu & Viswanath 2018; Su et al. 2021; Conneau et al. 2018; Zheng et al. 2023; Wang et al. 2024; Panickssery et al. 2024) | **Unverified here**; treat details as needing citation checks before submission |
| “No prior work provides the attribute-complexity ladder as a measured axis” | **Supported** for SVD-spectrum-of-steering-representation specifically: no direct prior hit found. Adjacent evidence exists (SAE multi-dimensional features; refusal rank-1; text-ID studies) and should be cited as adjacent, not absent |

---

## Design implications for E1

**Metrics**
1. Canonicalize every trained adapter (W2T: QR + SVD → single ΔW) before reporting spectra; state that spectra of factorizations are otherwise meaningless.
2. Report a metric *panel*, not one number: spectral entropy H (σ²-normalized), effective rank exp(H), stable rank ‖M‖_F²/‖M‖²₂, top-k energy (k=1,2,4,8), sparsity (fraction of dims with effect > ε), and layer spread (layers whose causal effect exceeds noise).
3. Make the **effect-based** complexity measure primary (entropy/distribution of token-wise ‖Δh‖; alignment of the effective direction with the diff-in-means vector). Parameter-spectrum metrics are exactly where the lexical-diversity confound bites; report both and their correlation.
4. Add a coherence/separation pre-screen (Braun et al.): positive/negative activation separation per attribute, computed before fitting; test whether it predicts rank saturation better than spectra.

**Baselines (all matched on layer, data, and effect scale)**
5. Closed-form diff-in-means; mean-centred diff-in-means; linear-probe direction.
6. Trained rank-1 + bias (ReFT-r1/BiPO-style) and LoRA-style rank sweep r=1..32 with bias.
7. Closed-form covariance-matching affine operator (Representation Surgery) and/or conceptor — the cheapest tests of "is low-rank training necessary at all?"
8. Randomized controls: random direction with matched effect norm; no-op (identity-init) adapter.
9. Prompting/few-shot baseline — mandatory (AxBench finding).
10. Where feasible, a small LoRA fine-tune as the costly reference — the proposal plans this; keep it.

**Rank saturation**
11. Pre-register saturation: smallest r whose behavioral endpoints (style score, PPL delta, coherence) are within CI of the best rank at matched parameter count and matched optimization.
12. Add a recovery test: cosine/Procrustes alignment between the canonical r=1 component and the closed-form diff-in-means direction; if a trained rank-r core contains the vector direction, "adapter necessity" fails for that attribute.

**Strength sweep / failure regions**
13. Calibrate α across arms by matching measured Δh norm (or output style score at a reference point) before sweeping — raw α is not comparable across representations.
14. Define the failure region operationally and report it per attribute: first α where PPL rises beyond a pre-set τ or coherence falls outside baseline CI; plot strength–coherence curves with per-prompt variance bands (Tan et al.) and report the fraction of prompts that flip behavior.

**Corpus / confound controls**
15. Publish corpus statistics (tokens, length distribution, TTR/MTLD, PPL under base model, topic/register tags) and match L1/L2/L3 within the study; never use the model's own unsteered samples as negatives.
16. Enforce single-dimension contrast pairs; audit with a classifier that the pair differs only on the target attribute.
17. Run a sample-size curve (e.g., 16/32/80/160/320/640 examples) for at least one attribute; report where metrics stabilize.
18. Report raw and standardized (whitened/ZCA) activation analyses side by side; flag massive-activation coordinates.

**Stability and statistics**
19. ≥5 seeds for trained arms (init + data order) and ≥5 bootstrap resamples for closed-form extraction; report the distribution (mean, sd, min), not the best run.
20. Cross-seed similarity via CKA, subspace principal angles, and Hungarian-matched direction overlap (PW-MCC-style); canonicalize before comparing.
21. Mixed-effects models with random effects for prompt/item; bootstrap CIs; paired tests; multiplicity correction across the rank × attribute grid; report effect sizes.

**Pitfalls to avoid**
22. Do not interpret parameter entropy as attribute complexity without corpus controls and effect-based corroboration.
23. Do not assume the monotone ladder H(L1) < H(L2) < H(L3): refusal is documented rank-1, so L2 may come out *simpler* than L1 in some models; frame the ladder as falsifiable and expect reversal.
24. Do not validate a representation claim by behavior alone (non-identifiability); always pair spectrum claims with an identifiability/stability caveat.
25. Do not headline gains against an uncalibrated α or a missing prompting baseline; both are established reviewer traps (Tan et al.; AxBench).

*End of report.*
