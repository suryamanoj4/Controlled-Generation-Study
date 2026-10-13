# Style Adapter Steering (SAS): Model-Agnostic Author-Style Control for LLMs

> **Research Proposal** · Draft v3 (2026-09-12) · Status: design under discussion
> **Working title:** "IP-Adapter for narrative style: a model-agnostic style adapter for LLMs"
> **Companion artifact:** the earlier design-discussion draft of this document (containing the full debate history) is preserved in git/session history.

---

## Abstract

Author-style transfer — generating text that matches a defined narrative voice (a specific author, a storytelling tradition, a register) — currently requires either fine-tuning the base model or heavy prompt engineering, both of which are brittle or costly. We propose **Style Adapter Steering (SAS)**: an identity-preserving, model-agnostic adapter module that controls a frozen LLM's narrative style at inference time, without fine-tuning and without RAG.

The design synthesizes three proven lineages: the **form** of IP-Adapter (lightweight pluggable module, scale knob, no-op until activated), the **ideology** of contrastive activation steering (CAA / Author-Style Steering vectors as the style signal source), and the **mitigation** of sparse steering (CAS-BiPO-style masks against the α-excessiveness problem). The style object is the adapter — a low-rank operator plus bias channel — fitted per model by a brief adapter-only run (n→n, no dimension changes, no-op until activated). The *framework* (form, recipe, measurements) is model-agnostic; the *artifact* is per-model — we do not claim cross-model weight transfer.

Two empirical axes define the paper's contribution: **(1) attribute complexity** — measured from the adapter itself via an entropy proxy (we predict sentiment < safety < author style), and **(2) language representation** — steering gain per language correlated with the model's published training splits. Primary application: author-style transfer (public-domain corpora); a low-resource storytelling tradition (Chandamama) serves as an example setting, and a cross-lingual author test ("make Hindi sound like Shakespeare") probes how much of style is language-neutral.

---

## 1. Problem Statement & Motivation

### 1.1 The problem

**Author-style transfer:** generate text in a **defined narrative voice** — e.g., a specific author's style (Shakespeare, Austen), a story-telling tradition (Chandamama Kathalu / Panchatantra: frame-story structure, talking-animal characters, didactic moral codas, folk-tale register), or any well-defined style corpus — **without fine-tuning the base model and without RAG**. Chandamama is one candidate style corpus among several, chosen when we want an authentic low-resource-language test case.

Prior decoding-time control (FUDGE, DExperts, GeDi, PPLM) works reliably for *token-level* attributes (sentiment, formality, topic) but the literature repeatedly shows it is hard for *complex, discourse-level* attributes — precisely our target. Vector-based activation steering (CAA, Author-Style Steering) handles surface styles but (a) requires a per-model vector, (b) uses a static strength coefficient whose excess collapses coherence, and (c) is limited by LLM-synthesized style exemplars.

### 1.2 Core intuition

*The base model already knows the language; it lacks a direction in which the narration flows.* We want to supply that direction as an external, model-agnostic artifact.

### 1.3 Constraints (hard requirements)

1. **No fine-tuning** of the main generative model (frozen weights).
2. **No RAG** (no content injection / retrieval of story text).
3. **No shared-vocabulary requirement** between any external model and the generator (this killed the earlier logit-fusion design; activation-level control is tokenizer-agnostic).
4. **Minimal training cost** at most — never a full training cycle.
5. Feasible at **small model scale** — including low-resource languages as one example setting (e.g., Telugu for the Chandamama corpus), not as the primary claim.

---

## 2. Research Questions & Objectives

### 2.1 Research questions

- **RQ1:** Can author/narrative *style* (a discourse-level, multi-dimensional attribute) be controlled at inference time — unlike sentiment/topic/formality, which decoding-time control handles well but which the literature repeatedly shows is hard for complex structural attributes?
- **RQ2:** Can the steering representation be made **model-agnostic** — a style operator fitted by a shared recipe, rather than a per-model vector?
- **RQ3:** Can a brief, adapter-only fit (minutes, hundreds of exemplars) achieve effective steering at any model scale, without relying on cross-model transfer?
- **RQ4:** Can we **measure attribute complexity from the adapter itself** (entropy of the steering representation) — and does steering effectiveness correlate with (a) that measured complexity and (b) the model's per-language training representation?

> Note on the language axis: multilingual models do **not** represent languages equally (English-centric processing is documented — Schut et al., 2025; "curse of multilinguality" — Conneau et al., 2020). That inequality is precisely what makes the per-language ablation a *measurement of representation quality* rather than a confound.

### 2.2 Objectives

1. Design and implement the SAS adapter (identity-preserving, model-agnostic).
2. Demonstrate adapter advantages over static steering vectors (input-dependent modulation, compactness, model-agnostic recipe).
3. Validate an entropy-based measure of attribute complexity (the "ladder" as a measured axis).
4. Probe language representation: steering gain × training splits correlation.

---

## 3. Background: The Steering Pipeline

All inference-time control methods follow one Bayesian formula with different plugs:

> p_θ(y | attribute) ∝ p_θ(y) · r(y | attribute)

...where `r` is an external evidence source, and methods differ along three axes: **what r is**, **where it is injected**, **how it is optimized**.

### 3.1 Three injection points (the "spectrum")

| Injection point | Mechanism | Representative methods |
|---|---|---|
| **Logits** (outermost) | Add scorer signal to next-token distribution | FUDGE, DExperts, GeDi, CriticControl, ARGS |
| **Hidden states / activations** | Add steering direction to residual stream | ActAdd/CAA, ITI, ICV, Author-Style Steering, CAS-BiPO |
| **Attention** (innermost) | Modify attention weights/patterns | InstABoost, ICR, ITI (head-level) |

Key insight: logit-level fusion between two *different* models requires shared vocabulary; **activation-level and attention-level control do not** (they operate on hidden states, which are tokenizer-agnostic). Our design therefore lives at the activation level.

### 3.2 How a steering vector is computed (contrastive recipe)

1. Forward-pass exemplar texts (style corpus vs. generic corpus) through the frozen model.
2. Record hidden states at the chosen layer; mean-pool over tokens; average over exemplars.
3. `v = normalize(μ_style − μ_generic)` — per layer.
4. At inference: `h' = h + α·v` at that layer for every generated token.

The vector is a **constant offset** in activation space, computed offline and reused.

---

## 4. Proposed Method: Style Adapter Steering (SAS)

### 4.1 Core design

Replace the per-model steering *vector* with an **identity-preserving, model-agnostic adapter module**:

- A LoRA-style adapter attached to the residual stream at mid-layers of the base model: `h' = h + (B·A)h + b`
- **Identity by default:** zero-init `B` (and zero `b`) ⇒ `h' = h` exactly — the module is a no-op until activated, like a norm layer that doesn't change the model's behavior. This is LoRA's standard design contract (also used by ControlNet and IP-Adapter).
- **n→n dimensions:** the adapter maps hidden dimension to itself; no dimension projections needed. The adapter "sits between 2 layers" of *any* model of *any* width (each instance has its own n×n/low-rank matrices, fitted per model).
- **The style object is the adapter, not the vector.** A library of style adapters (Shakespeare-style, Austen-style, Chandamama-style, moral-fable-style, ...) is produced by one shared recipe: per-style corpora, one brief fit per model (§4.3).

### 4.2 Why an adapter is more expressive than a vector

| | Steering vector | Style adapter (LoRA + bias) |
|---|---|---|
| Form | constant offset `α·v` | input-dependent `(BA)h + b` |
| Content-dependence | style intensity fixed | style can modulate with content (e.g., frame narration but not dialogue) |
| Replication of vector | — | needs bias channel `b` (a constant offset is *not* expressible by the linear term `(BA)h` alone) |
| Cross-model claim | per-model extraction only | none — framework is agnostic, artifact is per-model (§4.6) |

Design note: to *recover* classic vector steering as a special case, the adapter must keep a **bias/offset path** in addition to the low-rank path.

### 4.3 The fitting step (the key mechanism)

Each (style, model) pair gets one brief adapter-only run:

1. Attach the adapter at the chosen mid-layers (see Appendix A, D-3), with identity-preserving zero-init (no-op).
2. Run a **brief adapter-only training** (minutes; a few hundred exemplars) on an alignment/contrastive objective that fits the operator to the attribute *and* to this model's activation statistics.
3. Freeze again; use as a pure inference-time module with a strength knob (α / adapter scale).

Rationale: we **never rely on zero-shot cross-model transfer** (which is scale-gated, ≥ ~1.7B, §6.2) — each model is fitted individually, so the method holds at any scale. This is the theoretical reason the design is robust where transfer-based steering is brittle.

### 4.4 Vision: style adapter library

- One **style adapter** per (style, model) pair, fitted briefly from authentic corpora (fixes the "dataset circularity" limitation of Author-Style Steering, which used LLM-generated style exemplars — authentic corpora such as public-domain author texts or the Chandamama corpus avoid this).
- Fitted adapters are tiny and disposable; the **recipe** (corpora, contrastive protocol, fitting, measurement) is the reusable asset.
- Comparable to the IP-Adapter ecosystem in image generation (pluggable module, scale knob) — with the difference that weights are per-model, not shared (§4.6).
- **Any well-defined style corpus works** — author styles (Shakespeare/Austen), storytelling traditions (Chandamama/Panchatantra), or a cross-lingual author test ("can we make Hindi sound like Shakespeare?" — a clean probe of how much of the style signal is language-neutral vs. lexical/register-bound).

### 4.5 Design position: a synthesis (honest framing)

We are **not claiming a new mechanism class**. The design deliberately synthesizes three proven lineages:

1. **Form: IP-Adapter** — lightweight pluggable adapter, frozen base, scale knob, no-op until activated (image domain; we keep the form and UX, not the trained-once-reused-across-models claim).
2. **Ideology: vector-based activation steering** — contrastive activation directions (CAA / Author-Style Steering) as the *source* of the style signal.
3. **Mitigation: sparse steering** (CAS-BiPO) — learned masks over the adapter dimensions to combat the **steering-degree problem** (α excessiveness → coherence collapse, documented in Author-Style Steering's failure modes).

The claimed novelty is therefore **demonstrated advantages of the adapter form** over static vectors:

- **Compactness:** one low-rank operator + bias per (style, model) — orders of magnitude below fine-tuning cost, and no permanent model modification.
- **Optimality:** input-dependent modulation `(BA)h + b` is strictly more expressive than a constant offset, and the entropy of the learned core should reflect true attribute complexity (see §6.3).
- **Model-agnosticism:** the framework (form, recipe, measurements) transfers to any model; each model pays only a brief fit (§4.3).

Plus two empirical contributions no prior work provides: the attribute-complexity ladder as a *measured* axis (§6.3) and steering-gain × training-split correlation (§6.4).

### 4.6 Attribute knowledge representation: design space & choice

The adapter must *represent* the attribute's knowledge. Four representation families exist in the literature; they differ in form, estimation cost, expressiveness, and what they allow us to *measure*:

| Representation | Form | Estimation cost | Expressiveness | Per-model work |
|---|---|---|---|---|
| **Single vector** (CAA, Author-Style Steering) | constant offset `α·v` | free (contrastive forward passes) | rank-1; content-independent; unstable across data/training (GCS's critique); behaviorally non-identifiable without imposed structure | extract per model |
| **Probed subspace (GCS)** (Zhao et al., 2024) | Gaussian distribution over concept vectors | linear probes per concept & layer | multidimensional, robust to single-vector instability — but still a *static vector add* at inference (sampled from the distribution) | probe per model |
| **Closed-form operator (Conceptor)** (Triantafyllopoulos et al., 2026) | soft projection matrix from activation correlations | **zero training** (closed-form SVD) | full subspace with continuous per-principal-direction attenuation; Boolean composition (AND/OR/NOT); layer-selection diagnostic | estimate per model |
| **Fitted low-rank operator + bias (ours)** | `h' = h + (BA)h + b`, identity-preserving init | brief adapter-only run (minutes, hundreds of exemplars) | input-dependent modulation; subsumes vector via bias channel; measurable structure (SVD spectrum, §6.3) | fit per model |

**Our choice and its justification:**

- **Reject rank-1 vectors:** on robustness grounds (GCS's central argument) and because unconstrained vectors are behaviorally interchangeable — the non-identifiability result means *structure must be imposed* for a representation claim to be meaningful.
- **Reject closed-form conceptor estimation as our mechanism:** cheapest, but it is per-model attribute-source machinery with no demonstrated handling of discourse-level attributes (sentiment/political-leaning only), and its insight is fully reusable as a baseline and as a warm-start option (D-12).
- **Choose the fitted low-rank operator:** (a) most expressive family with provably sub-fine-tuning cost; (b) identity-preserving zero-init keeps the no-op guarantee; (c) its SVD yields the entire measurement layer — canonical form (W2T), spectrum entropy = attribute complexity (§6.3), and spectral decomposition into language-neutral vs. language-entangled components (Chang et al. 2022; Xie et al. 2022) for the cross-lingual leg (§6.4).

**Model-agnostic, not portable (explicit framing):** the *framework* — adapter form, contrastive data recipe, fitting procedure, measurement protocol — transfers to any model; the *artifact* (adapter weights) is per-model, fitted by a brief run. We deliberately do **not** claim cross-model weight transfer: evidence shows it is scale-gated (Agarwal 2026) and brittle (Da Silva et al. 2025). This removes the weakest experimental burden and makes the method work at any model scale.

**Lightweight filter applied to the literature:** references whose machinery implies fine-tuning-scale training (SAE training pipelines, per-concept probe training at scale, fine-tune-then-merge pipelines) are used for insight only or excluded. Our cost thesis — orders of magnitude below fine-tuning (quantified in §7.9) — requires the method's own dependencies to stay in that regime.

---

## 5. Related Work

### 5.1 Decoding-time / logit-level control (simple attributes)

| Method | Year/Venue | Attribute tested | Gap for our case |
|---|---|---|---|
| PPLM | ICLR 2020 | topic, sentiment, detox | needs hidden-state classifier; gradient cost |
| FUDGE | NAACL 2021 | formality, topic, poetry | trained discriminator; simple attributes |
| GeDi | EMNLP 2021 Findings | formality, topic, sentiment | generative discriminator needs tuning |
| DExperts | ACL 2021 | detox, sentiment | expert/anti-expert LMs, shared-vocab assumption |
| CriticControl | Findings ACL 2023 | topic, sentiment, detox | critic trained from reward models |
| ARGS/RAD family | 2023–2024 | preference-aligned decoding | value/reward scoring of partial sequences |

**Key finding:** the literature consistently shows decoding-time control works reliably for *token-level* attributes (sentiment, formality, topic, toxicity) and is genuinely hard for *complex, discourse-level* attributes — precisely our target (frame structure, moral coda, narration register).

### 5.2 Activation steering (vector-based)

| Method | Year/Venue | Mechanism | Gap for our case |
|---|---|---|---|
| ActAdd / Activation Engineering | 2023 (Anthropic) | contrastive mean-diff vectors, residual stream | English, sentiment/toxicity; simple style transfer only |
| CAA | 2024 | contrastive pairs, 7 alignment tasks; refusal direction transferable within family | no style, no cross-architecture |
| ITI | NeurIPS 2023 | attention-head probes + activation shifts | probe training; truthfulness focus |
| ICV (In-Context Vectors) | ICML 2024 | vectors from few-shot demos | task ability, not style |
| **Author-Style Steering** | 2026 (Ionio) | author-mean − all-mean vectors on GPT-2; layer×α sweep; **α is static grid-searched, NOT adaptive**; adaptive scaling explicitly listed as future work; dataset circularity acknowledged | English authors only; dense vector; static α; LLM-synthesized exemplars |
| CAS-BiPO | Findings EACL 2026 | causal-mediation framing (front-door identifiability); sparse learned mask `m ⊙ v`; 10–30% of dims = 97–100% of dense effectiveness | safety behaviors only; needs preference data + gradient optimization |
| Causal-Steer | ICLR 2026 | LoRA-as-causal-intervention; extracts style vectors from LoRA deltas; continuous α interpolation | **inverse of our direction** (LoRA → vector; we want vector/core → adapter); still per-model LoRA training |
| Steering-vectors limits study (Subbiah et al.) | 2026 | PLUME preference attributes incl. children's-book style, third person, informal tone, simile | trait expressibility varies substantially; composition often breaks sign control; style steering limited beyond surface styles |
| What Drives Representation Steering? | 2026 | causal analysis of steering mechanisms | — |

### 5.3 Cross-model / cross-architecture transfer (the crux for us)

| Work | Year/Venue | Finding |
|---|---|---|
| **Cross-Architecture Steering Transfer in LMs** (Agarwal) | 2026 | Concept directions from one model steer *independently trained* models: a universal direction can steer without per-model supervision (effect sizes unverified); **transfer degrades below ~1.7B params**; SAE-based feature alignment across 20 directed pairs |
| **Activation Space Interventions Can Be Transferred Between LLMs** (Oozeer et al.) | ICML 2025 | Learned lightweight "translator" maps one model's activation space to another; transfers safety interventions (backdoor removal, refusal) small→large without retraining |
| Steering Vector Transfer via Orthonormal Transformations | (semi-verified) | orthogonal transform + scaling preserves geometry; semantic pairing +77% transfer cosine |
| Textual Steering Vectors → MLLMs (Gan et al.) | ACL 2026 | text-only steering vectors transfer to multimodal models (+7.3% spatial accuracy) |
| Reasoning steering (DeepSeek-R1-Distill) | 2025 | steering vectors generalize across model sizes within a family |
| **Steering off Course** (Da Silva et al.) | ACL 2025 | **counter-evidence:** steering performance high-variance across 36 models / 14 families; many interventions show no improvement or active degradation even after tuning |

### 5.4 Universal features (theoretical substrate)

| Work | Finding |
|---|---|
| SAE universality (Lan et al.) | feature spaces of independently trained LLMs similar under rotation-invariant measures; SVCCA evidence |
| SAE features: classification & transferability (Gallifant et al., EMNLP 2025) | SAE features transfer cross-model zero-shot (Gemma-2 2B → 9B) |
| Language-specific SAE features (Deng et al., ACL 2025) | SAE features per language; steering via feature ablation/addition — directly relevant to the Indic-language angle |
| Linear Representation Hypothesis / Platonic Representation Hypothesis | concepts as linear directions; converged geometry across models (Park et al. 2023; Huh et al. 2024) |

### 5.5 Adapter / module-based control (our design lineage)

| Work | Mechanism | Relevance |
|---|---|---|
| LoRA (Hu et al.) | low-rank adapters, zero-init identity | our design contract |
| Adapters (Houlsby et al.; Pfeiffer et al.) | pluggable residual adapters | lineage |
| **IP-Adapter** (Ye et al.) | **lightweight style adapter trained once, frozen base, pluggable into any model fine-tuned from same base, scale knob** | **closest existing instantiation of our philosophy — in the image domain** |
| Task arithmetic (Ilharco et al.) | task vectors in weight space; arithmetic composition | weight-space analogue of vector steering |
| Weight-space steering ("Steering LMs with Weight Arithmetic") | contrastive weight directions generalize across behaviors | same-architecture transplant only (tensor shape matching) |
| FiLM (Perez et al.) | feature-wise linear modulation, n→n | norm-layer-like modulation precedent |
| Guided LoRA initialization | LoRA init as direction-selection problem | relevant if we ever train style cores |

### 5.6 Data & evaluation groundwork

| Resource | Use |
|---|---|
| **Public-domain author corpora** (Project Gutenberg: Shakespeare, Austen, etc.) | Primary author-style leg; authentic (no LLM circularity); natural per-author contrast sets (verify per-text licensing) |
| **Chandamama Kathalu corpus (Swecha)** — *example style corpus* | ~40,000 proofread Telugu story pages; gated HF dataset (contact required) |
| Swecha datathon / AI Chandamama Kathalu SLM — *example prior work* | prior work = **fine-tuning only** (our training-free alternative is unoccupied) |
| Moral Stories (Emelin et al., EMNLP 2021) | 720K structured branching narratives, moral norms |
| TF1-EN-3M (2026) | 3M synthetic moral fables; evaluation methodology; explicitly Western (Aesop) — cultural-bias gap we can fill |
| Tales of Morality (Findings EMNLP 2025) | LLMs over-emphasize "Care" moral foundation and flatten narrative arcs **even under moral conditioning** — our evaluation target |
| TALES (Bhagat et al.) | 88% cultural misrepresentation in LLM Indic stories; 2,925 annotations, 14 languages |
| Multilingual TinyStories (Halder & Mukherjee) | 132,942 stories, 17 Indian languages |

### 5.7 Gap summary (novelty statement)

> **No published work combines: (a) a model-agnostic, identity-preserving style *adapter* (not vector); (b) author-style / narrative-style attributes (not sentiment/formality); (c) authentic style corpora — including, but not limited to, low-resource-language traditions; (d) no base-model fine-tuning and no RAG; (e) small-scale feasibility.** The components exist individually (IP-Adapter philosophy, cross-model translators, SAE universality, CAS-BiPO sparsity, Causal-Steer LoRA vectors); the synthesis is unoccupied. Honest positioning: we are **not introducing a new mechanism class** — the claimed novelty is (1) the adapter form with *proven* advantages over vectors, (2) the entropy-based attribute-complexity metric (§6.3), and (3) the steering-gain × language-representation probe (§6.4).

---

## 6. Hypotheses & Theoretical Feasibility

### 6.1 Evidence FOR

1. **Feature universality is real** (Lan et al. 2024; Agarwal 2026): independently trained models share aligned geometry — the substrate our per-model fits build on.
2. **Steering representations are effective within a model** (Agarwal 2026; Author-Style Steering): directions extracted from a model's own activations steer it reliably; cross-model transfer is a bonus we do not rely on.
3. **Learned cross-model mappings exist** (Oozeer et al., ICML 2025): a translator transfers interventions across models — evidence that activation-space geometry is mappable; our design avoids needing this by fitting per model.
4. **Identity-preserving adapters are standard** (LoRA zero-init contract) — the no-op-until-activated design is proven in production.
5. **Authentic corpus advantage:** our cores come from genuine style corpora (public-domain author texts; Chandamama for the low-resource example), avoiding the circularity limitation of Author-Style Steering (LLM-synthesized exemplars) — this is an *improvement* over the closest prior method, not a copy.
6. **Hidden-state level ⇒ no tokenizer constraint** (unlike logit-level fusion) — removes the earlier blocker.

### 6.2 Evidence AGAINST / Conditions

1. **Scale threshold:** cross-model feature alignment collapses below ~0.8B, "suggestive discontinuity near 1.7B" (Agarwal 2026). **Not a threat to our core claim:** we never rely on cross-model transfer — each model is fitted individually. Scale analysis is kept as a robustness check (§7.7).
2. **Steering brittleness:** Da Silva et al. (ACL 2025) — high variance across model families; some families degrade. Our paper must report per-model results, not cherry-pick.
3. **Entanglement risk:** fitted adapters may absorb model-specific idiosyncrasies; mitigation = measure core stability across seeds/models via SVD-spectrum comparison (Appendix A, D-4).
4. **Layer-semantics mismatch:** "mid-layer = style" is model-relative; architecture differences shift where style abstractions live (D-3).
5. **Excessiveness/coherence:** the Author-Style α problem persists in adapter form; candidates: CAS-BiPO-style sparse masks, per-layer learned gates, adaptive α (all open decisions).
6. **Evaluation subjectivity:** narrative style lacks standard metrics; LLM judges are biased (Tales of Morality findings) — need human eval + structural metrics (MFT distributions, narrative-arc shape, coda presence).

### 6.3 H1: The attribute-complexity / entropy hypothesis

**Claim:** the steering representation's *entropy* scales with attribute complexity — simple attributes (sentiment) yield low-entropy adapter cores; complex attributes (author/story style) yield high-entropy cores.

**Scattered supporting evidence (not yet a formal taxonomy — that's our contribution):**

- Some behaviors reduce to ~1 direction (refusal — Arditi et al., 2024); CAS-BiPO needed only 10–30% of dimensions for safety behaviors (Doan et al., 2026); ITI found truthfulness concentrated in few heads (Li et al., 2023).
- Feature complexity varies across concepts: multi-dimensional/circular features exist (Engels et al., 2024).
- Steerability varies widely across preference attributes (Subbiah et al., 2026, arXiv:2607.01802); prompt-based control captures only *surface* lexical cues vs. deep discourse features (Author-Style Steering related work).

**Proposed measurable proxies (entropy of the adapter):**

1. **SVD-spectrum entropy** of the low-rank core: `H = −Σ p_i log p_i`, `p_i = σ_i² / Σ σ_j²` — how spread the steering signal is across rank directions.
2. **Effective rank** (`exp(H)`).
3. **Mask sparsity** (CAS-BiPO-style) — fraction of active dimensions per attribute.
4. **Layer spread** — how many layers the attribute requires.
5. **Entropy of the induced hidden-state change** (distribution of `‖Δh‖` across tokens) — alternative, less corpus-dependent.

**Prediction:** `H(sentiment) < H(safety) < H(style)` for the same model, corpus-matched.

**Confound to control:** corpus lexical diversity inflates rank — sentiment corpora are narrow, style corpora broad. Mitigations: matched-size/matched-perplexity corpora; or measure entropy of *effect* (proxy 5) rather than *parameters*; or normalize by corpus statistics.

**Why it matters:** if the hypothesis holds, attribute complexity becomes *measurable from the artifact itself* — no external judges needed for the difficulty ladder, and it predicts where steering will be brittle.

### 6.4 H2: The language-representation probe

**Setup:** one multilingual base model (candidates: mT5-XL — per-language proportions published in the mT5/mC4 paper under α-sampling; BLOOM-1B7 — decoder-only, OSCAR/ROOTS per-language stats; see Appendix A, D-6). Fit the *same* attribute adapter (same recipe, same corpus); measure steering gain per language.

**Analysis:** correlate per-language steering gain with (log) training-data proportion and token-fertility covariates. Precedent for behavior↔representation correlations exists (Schut et al., 2025 — English-centric reasoning; Winata et al. — language norm correlates with data proportion), but steering-gain correlation is unstudied.

**Caveat — style-language entanglement:** a style core trained on a specific language's corpus (e.g., Telugu Chandamama) encodes that language's specific markers; cross-lingual transfer of *style* is contaminated by *language* signal. Mitigations: (a) language-neutral attributes (sentiment/formality/safety) for the cross-lingual leg; (b) per-language style cores where parallel corpora exist; (c) the "Hindi Shakespeare" test as an explicit entanglement probe (if an English author's style core transfers to Hindi output, the style signal is partly language-neutral).

---

## 7. Experimental Plan

### 7.1 Experiment 1: adapter vs. vector

Same corpus, same model: static CAA/Author-Style vector vs. SAS adapter (pre-fit identity init vs. post-fit). Metrics: style strength, coherence, α-excessiveness behavior (sweep), and the *failure region* (where does each break?). This is the core novelty demonstration.

### 7.2 Experiment 2: attribute-complexity ladder

≥2–3 attributes per level (L1: sentiment, formality; L2: safety/toxicity, truthfulness; L3: story style, narrative coherence). Same model, corpus-matched, *same* adapter capacity. Measure entropy proxies (§6.3) per attribute; test the monotonicity prediction `H(L1) < H(L2) < H(L3)` and correlate with steering gain.

### 7.3 Experiment 3: language-representation probe

Same attribute across 6–10 languages of one multilingual base; correlate steering gain with log training proportion + token fertility (mT5/mC4 or BLOOM/OSCAR splits). Language-neutral attributes for this leg; style leg language-matched.

### 7.4 Experiment 4: cross-lingual author-style test ("Hindi Shakespeare")

Does an English-author style core transfer to Hindi output? Quantifies style-signal language-neutrality; also an entanglement diagnostic.

### 7.5 Evaluation metrics

1. **Style strength:** held-out style-vs-generic classifier agreement (trained per style corpus); style-marker density; LLM-judge pairwise scores (baseline vs. steered), 1–5.
2. **Structural fidelity (novel):** moral-coda presence; Moral Foundations distribution vs. human baseline (Tales-of-Morality method); narrative-arc shape (staging → tension → resolution); frame-story structure detection.
3. **Quality:** perplexity under base model; GPT-4/target-language-capable-judge coherence; distinct-n / self-BLEU; human eval subset.
4. **Entropy proxies** (per §6.3) as the attribute-complexity measurement.

### 7.6 Datasets

Public-domain author corpora (Shakespeare/Austen via Project Gutenberg) for the primary author-style leg; Chandamama corpus (Swecha) as the low-resource style example; Moral Stories / TF1-EN-3M-style templates for controlled prompts; TALES-style cultural annotations for cultural-fidelity checks.

### 7.7 Baselines & ablations

- **Baselines:** prompting-only, few-shot prompting, CAA/Author-Style vectors (per-model), LoRA fine-tune (the "costly" reference point).
- **Ablations:** adapter vs. vector (both from same corpus); fit vs. no-fit (identity init); α/scale sweep; layer sweep; sparse mask on/off; scale analysis (small vs. ≥1.7B base) as a robustness check (we do not rely on cross-model transfer).
- **Rigor:** multi-seed runs, per-model results reported individually (no cherry-picking), statistical tests (paired, effect sizes).

### 7.8 Experiment 5: objective ablation — the "just LoRA?" test

Pre-empts the top reviewer objection ("isn't this just a briefly trained LoRA?"). Same adapter form, same capacity, same corpus — three training signals:

| Training signal | What it optimizes | Expected outcome (hypothesis) |
|---|---|---|
| Task fine-tune CE (style-corpus completion) | next-token prediction on style texts | strong style mimicry but model-wide shift; spectrum dominated by a few directions (overfit to lexical surface); no clean complexity signature |
| Style CE (short run, style corpus only) | same objective, fewer steps | weaker version of the above |
| Contrastive steering objective (ours) | operator that moves activations along the style-vs-generic contrast | measurable spectrum structure: entropy/rank matches attribute complexity (§6.3) and the language-neutral/entangled split (§6.4) |

Claim to test: **only the contrastive steering objective produces a core whose spectrum encodes the attribute's complexity** — CE objectives overfit lexical surface. This isolates the *representation* (and its objective) as the contribution, not the training step itself. If the claim fails, the paper must honestly report the boundary where CE-trained adapters become indistinguishable.

### 7.9 Cost analysis: SAS vs. fine-tuning (quantified)

| | Full fine-tuning | LoRA fine-tuning | SAS (ours) |
|---|---|---|---|
| Training budget | GPU-days–weeks, full epochs | GPU-hours–days | **minutes** — forward passes + adapter-only fit on ~hundreds of exemplars |
| Trainable params | 100% of model | ~0.1–1% of model | projection matrices only (same order as LoRA, far fewer steps) |
| Per-model cost | full retrain | full retrain | one brief fit |
| Per-new-style cost | full retrain | full retrain | one brief fit |
| Inference overhead | none (merged weights) | none (merged weights) | fixed small matmul `2·r·d` per token per attached layer; no KV-cache/context growth |
| Deployment | replaced weights | replaced weights | pluggable module, no-op until activated |

To be reported empirically in the paper: measured GPU-hours, FLOPs, and wall-clock per pipeline stage, plus the strength of the claim we target — **>10× cheaper than LoRA fine-tuning, >100× cheaper than full fine-tuning** (targets to be confirmed, not assumed).

### 7.10 Dataset curation plan (experiment × claim × corpus matrix)

Dataset choice is downstream of claims. The matrix below assigns each experiment its style/positive corpus, its contrast corpus, and the matching protocol that keeps the entropy measure (§6.3) and the style classifier honest.

| Exp | Claim it supports | Attribute | Langs | Style / positive corpus | Contrast / negative corpus | Matching protocol | Eval data |
|---|---|---|---|---|---|---|---|
| E1 | Adapter ≥ vector at matched coherence; failure regions differ | Author style | EN | Shakespeare (Project Gutenberg, public domain) | Austen + other Gutenberg fiction (matched era) | Equal token counts; matched sentence-length dist. | Style classifier; coherence; α sweep |
| E2·L1 | `H(L1)` low; gain high | Sentiment | EN (+HI, TE) | Yelp polarity / SST-2 positive | Same dataset negative | Equal sizes; matched type-token ratio (TTR) and perplexity under base model | Steering gain; 5 entropy proxies |
| E2·L1 | Formality | EN | GYAFC informal | GYAFC formal | same | |
| E2·L2 | `H(L2)` mid | Toxicity / safety | EN | ToxiGen toxic; CAA-style refusal pairs (synthetic, per CAA methodology) | ToxiGen benign; comply pairs | same | |
| E2·L2 | Truthfulness | EN | TruthfulQA false-claim generation | TruthfulQA true-claim generation | same | |
| E2·L3 | `H(L3)` high; gain low | Author style | EN | Gutenberg authors (E1 protocol) | Matched Gutenberg fiction | same as E1 | |
| E2·L3 | Narrative register | EN | Moral Stories (fable register) | News / encyclopedia prose (length-matched) | same | |
| E3 | Gain ∝ log(training tokens) + token fertility | Sentiment (language-neutral) | 6–10 langs spanning mC4 proportions (en, hi, te, ta, mr, bn, ur, es, fr, sw) | Per-language sentiment corpora *(availability to verify)* | Matched negatives | Equal sizes per language | Correlation with published mC4 per-language proportions (Xue et al. 2021, Fig. 1/appendix) |
| E4 | Style signal is partly language-neutral; spectrum predicts transfer | Author style | EN → HI | Shakespeare core (from E1) | Hindi narrative prose (generic — OSCAR/mC4 `hi` slice) | n/a (transfer leg) | Hindi-capable LLM judge + human eval |
| E5 | Only the contrastive objective yields the spectrum structure | Author style | EN | Same as E1 | Same as E1 | same as E1 | Spectrum structure; steering gain |

**Hard requirements the corpora must satisfy:**

1. **Contrast corpora are the feature definition** — the style-vs-generic contrast *is* the attribute. Matched same-genre contrast (not web-random) is mandatory: web-random inflates the entropy measure (§6.3 confound) and poisons the style classifier.
2. **The generic contrast must NOT be the base model's own unsteered samples** (circularity — the "generic" then inherits the model's distribution).
3. **Corpus-matched protocol for E2** — every attribute gets equal token counts, matched length distribution, matched TTR/perplexity; diversity statistics reported in the paper (transparency against the §6.3 confound).
4. **Authentic corpora everywhere feasible** — no LLM-synthesized style exemplars (the Author-Style Steering circularity fix); exceptions limited to the CAA-style synthetic refusal pairs (L2), where synthesis is the established methodology.
5. **E3 languages chosen from published mC4 proportions** — the same 6–10 languages must have (a) published training proportions, (b) available sentiment corpora, (c) sentiment judged language-neutral.

**Availability checks before model choice (D-6):**

- Hindi and Telugu sentiment corpora (English alternatives are trivial — Yelp/SST-2; per-language sentiment is the constraint for E3). Candidates: Hindi movie-review / SentiWordNet-derived sets; Telugu sentiment tweet sets; multilingual sets (e.g., XED-emotion, MT-Sentiment) — verify presence, license, and size.
- GYAFC availability (HF mirror exists) and its contrast structure.
- ToxiGen / TruthfulQA access; Moral Stories (Emelin et al., EMNLP 2021) on HF.
- Chandamama multilingual editions (13+ languages historically; digital availability per language — Telugu confirmed via Swecha; Hindi/Tamil/Marathi editions unverified).
- Hindi narrative prose slice (OSCAR/mC4 `hi`) usable as generic contrast without license issues.

---

## 8. Expected Contributions & Novelty

*"Style as a fitted, model-agnostic adapter: inference-time author-style control for LLMs."*

1. **Mechanism:** the adapter-as-steering-representation with identity-preserving zero-init and bias channel, with *demonstrated* advantages over static vectors (input-dependent modulation, compactness, model-agnostic recipe).
2. **Measurement:** an **entropy-based measure of attribute complexity** extracted from the learned adapter core, validated against an attribute-complexity ladder (§6.3).
3. **Probe:** a **language-representation probe** — steering gain per language correlated with the model's published training splits (§6.4).
4. **Application:** authentic-corpus style steering (fixes the dataset-circularity limitation), demonstrated on author styles and, as one example, a low-resource narrative tradition, with the cross-lingual author test ("Hindi Shakespeare") as a secondary leg.

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Zero-shot universality collapses below ~1.7B (Agarwal 2026) | Not applicable to the core claim — we never rely on cross-model transfer; scale analysis kept as a robustness check (§7.7) |
| Steering brittleness across model families (Da Silva et al. 2025) | Report per-model results; multi-seed; no cherry-picking |
| Fitted adapters absorb model idiosyncrasies | Measure core stability across seeds/models via SVD-spectrum comparison (D-4) |
| "Mid-layer = style" is not universal across architectures | Attach at several layers with learned gates; let the fit select (D-3) |
| α excessiveness collapses coherence | CAS-BiPO-style sparse masks, learned gates, adaptive α (D-5) |
| Style-language entanglement contaminates cross-lingual results | Language-neutral attributes for cross-lingual leg; per-language cores; "Hindi Shakespeare" as diagnostic |
| Corpus diversity inflates entropy measure | Corpus-matched sets; induced-Δh entropy proxy; normalization (D-11) |
| Narrative evaluation subjectivity / LLM-judge bias | Triangulate: human eval + structural metrics + classifier + LLM judge |

---

## Appendix A: Open Design Decisions (debate list — resolve as we go)

- **D-1. What is shared vs. per-model?** Resolved toward per-model fitting (§4.6). Remaining nuance: (a) per-model full adapter, random init (simplest); (b) per-model adapter warm-started from a previous model's fit (soft portability — an optional ablation, never a claim); (c) shared corpus/recipe only.
- **D-2. Bias channel:** include a constant-offset path to replicate classic vectors (yes/no; and if yes, is α the scale of the offset, the adapter, or both)?
- **D-3. Attachment location:** single mid-layer (which one, by what rule — fractional depth? learned gates over multiple layers?) vs. all layers with learned gating.
- **D-4. Fitting details:** how many exemplars (100? 1,000?), which loss (next-token CE on style corpus? feature-matching to target style activations? contrastive?), how long, and what is frozen.
- **D-5. Strength control:** static α (grid-searched, as prior work) vs. adaptive α (Author-Style lists it as future work; we could claim it) vs. CAS-BiPO-style sparse masking.
- **D-6. Base model choice:** target-language-capable small model (e.g., Telugu-capable for the Chandamama example) vs. English-only validation first vs. **multilingual model with documented per-language training splits** (mT5-XL — mC4 proportions published; BLOOM-1B7 — decoder-only, OSCAR/ROOTS stats). The 1.7B threshold (Agarwal 2026) applies to cross-model transfer, which we do not rely on; scale choice is therefore free — but measure fit quality across scales as a robustness check.
- **D-7. Scorer (optional second stage):** do we add a FUDGE-style style scorer / sequence-level review (Best-of-N gate) on top of the adapter, or keep the adapter pure?
- **D-8. Style core training data:** author-style cores (public-domain texts — Shakespeare, Austen, ...) as the primary leg; storytelling-tradition corpora (Chandamama, Panchatantra+Jataka family) as the secondary/low-resource example; per-style vs. style-family cores; a cross-lingual author test ("Hindi Shakespeare") as an entanglement probe. Also: what is the *generic* contrast corpus (per language — where from?).
- **D-9. Evaluation judges:** human (native speakers of the target language, cost), LLM judge (which), structural metrics (MFT via lexicon, arc shape via LIWC-style analysis), and how many outputs for statistical power (multi-seed, low-resource variance).
- **D-10. Style-language entanglement:** (a) language-neutral attributes for the cross-lingual ablation leg; (b) per-language style cores from parallel corpora (Chandamama had 13+ language editions — availability check needed); (c) the cross-lingual author-style test as a deliberate probe of language-neutrality.
- **D-11. Entropy-based complexity metric:** which proxy (SVD-spectrum entropy of core / effective rank / mask sparsity / layer spread / induced-Δh entropy), and the corpus-matching protocol to control diversity confounds (§6.3). Decide before the ladder experiments so all attributes share one metric.
- **D-12. Representation-choice sub-questions (fixed choice: fitted low-rank operator + bias, §4.6):** (a) initialize the core from a GCS-estimated subspace or a conceptor (warm start) vs. random? (b) regularize the core's spectrum during fitting (e.g., soft-rank penalty) to sharpen the complexity metric? (c) is the bias channel fitted per model too, or shared across attributes?

---

## Appendix B: Consolidated References (verified 2026-09-11; 36–46 added 2026-09-12 under the lightweight-cost filter; citation corrections + ref 47 applied 2026-10-13)

1. Dathathri et al. 2020, PPLM — [arXiv:1912.02164](https://arxiv.org/abs/1912.02164)
2. Yang & Klein 2021, FUDGE — [arXiv:2104.05218](https://arxiv.org/abs/2104.05218)
3. Krause et al. 2021, GeDi — [arXiv:2009.06367](https://arxiv.org/abs/2009.06367)
4. Liu et al. 2021, DExperts — [arXiv:2105.03023](https://arxiv.org/abs/2105.03023)
5. Kim et al. 2023, CriticControl — [arXiv:2212.10938](https://arxiv.org/abs/2212.10938)
6. Turner et al. 2023, Activation Engineering — [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
7. Panickssery et al. 2023, CAA (widely cited as "Rimsky et al.") — [arXiv:2312.06681](https://arxiv.org/abs/2312.06681)
8. Li et al. 2023, ITI — [arXiv:2306.03341](https://arxiv.org/abs/2306.03341)
9. Liu, Ye, Xing & Zou 2023, ICV (In-Context Vectors) — [arXiv:2311.06668](https://arxiv.org/abs/2311.06668) (ICML 2024)
10. Patel 2026, Author-Style Steering via Contrastive Activation Vectors — [Ionio](https://research.ionio.ai/posts/2025/author-style-steering-activation-vectors)
11. Doan et al. 2026, Causal Activation Steering via Sparse Mediation (CAS-BiPO) — [2026.findings-eacl.57](https://aclanthology.org/2026.findings-eacl.57/) (BiPO: Cao et al., 2024)
12. Causal-Steer 2026 — [OpenReview yfiOYXvsX5](https://openreview.net/forum?id=yfiOYXvsX5)
13. Da Silva et al. 2025, Steering off Course — [ACL 2025 long 974](https://aclanthology.org/2025.acl-long.974/) · [arXiv:2504.04635](https://arxiv.org/abs/2504.04635)
14. Agarwal 2026, Cross-Architecture Steering Transfer — [arXiv:2608.05164](https://arxiv.org/abs/2608.05164)
15. Oozeer et al. 2025, Activation Space Interventions Can Be Transferred — [ICML 2025](https://icml.cc/virtual/2025/poster/45778)
16. Gan et al. 2026, Textual Steering Vectors for MLLMs — [2026.acl-long.1861](https://aclanthology.org/2026.acl-long.1861/)
17. Lan et al. 2024, SAE Universality — [arXiv:2410.06981](https://arxiv.org/abs/2410.06981)
18. Gallifant et al. 2025, SAE Transferability — [2025.emnlp-main.1521](https://aclanthology.org/2025.emnlp-main.1521/)
19. Deng et al. 2025, Language-Specific SAE Features — [2025.acl-long.229](https://aclanthology.org/2025.acl-long.229/)
20. Ye et al. 2023, IP-Adapter — [arXiv:2308.06721](https://arxiv.org/abs/2308.06721)
21. Ilharco et al. 2022, Task Arithmetic — [arXiv:2212.04089](https://arxiv.org/abs/2212.04089)
22. Weight Arithmetic Steering — [arXiv:2511.05408](https://arxiv.org/abs/2511.05408)
23. Hu et al. 2022, LoRA — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
24. Houlsby et al. 2019, Adapters — [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)
25. Huh et al. 2024, Platonic Representation Hypothesis — [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)
26. Emelin et al. 2021, Moral Stories — [EMNLP 2021](https://aclanthology.org/2021.emnlp-main.54/)
27. TF1-EN-3M — [arXiv:2504.20605](https://arxiv.org/abs/2504.20605)
28. Tales of Morality — [2025.findings-emnlp.1029](https://aclanthology.org/2025.findings-emnlp.1029/)
29. TALES (Bhagat et al.) — [arXiv:2511.21322](https://arxiv.org/abs/2511.21322)
30. Multilingual TinyStories — [arXiv:2603.14563](https://arxiv.org/abs/2603.14563)
31. Chandamama Kathalu dataset — [HF swechatelangana](https://huggingface.co/datasets/swechatelangana/chandamama-kathalu)
32. Swecha coverage — [MediaNama](https://www.medianama.com/2023/12/223-folklore-ai-swecha-telugu-chatbot/), [Hindu BusinessLine](https://www.thehindubusinessline.com/info-tech/gen-ai-gives-a-new-lease-of-life-to-old-favourite-chandamama-stories/article67716201.ece)
33. Schut et al. 2025, Do Multilingual LLMs Think In English? — [arXiv:2502.15603](https://arxiv.org/abs/2502.15603)
34. Engels et al. 2024, Not All Language Model Features Are Linear — [arXiv:2405.14860](https://arxiv.org/abs/2405.14860)
35. Xue et al. 2021, mT5 (mC4 per-language proportions, Fig. 1 + appendix) — [arXiv:2010.11934](https://arxiv.org/abs/2010.11934)
36. Zhao et al. 2024, Beyond Single Concept Vector: Gaussian Concept Subspace (GCS) — [arXiv:2410.00153](https://arxiv.org/abs/2410.00153) (venue unverified)
37. Triantafyllopoulos et al. 2026, Conceptors for Semantic Steering — [arXiv:2605.04980](https://arxiv.org/abs/2605.04980)
38. On the Non-Identifiability of Steering Vectors in LLMs — [arXiv:2602.06801](https://arxiv.org/abs/2602.06801)
39. The Cylindrical Representation Hypothesis for Language Model Steering — [arXiv:2605.01844](https://arxiv.org/abs/2605.01844)
40. Chang et al. 2022, The Geometry of Multilingual Language Model Representations — [2022.emnlp-main.9](https://aclanthology.org/2022.emnlp-main.9/)
41. Xie et al. 2022, Discovering Low-rank Subspaces for Language-agnostic Multilingual Representations — [2022.emnlp-main.379](https://aclanthology.org/2022.emnlp-main.379/)
42. W2T: LoRA Weights Already Know What They Can Do (canonical LoRA form = QR+SVD) — [arXiv:2603.15990](https://arxiv.org/abs/2603.15990)
43. Chronopoulou et al. 2024, Language and Task Arithmetic with Parameter-Efficient Layers — [2024.mrl-1.7](https://aclanthology.org/2024.mrl-1.7/)
44. Shah et al. 2024, Correlations between Multilingual LM Geometry and Crosslingual Transfer Performance — [2024.lrec-main.361](https://aclanthology.org/2024.lrec-main.361/)
45. RepE Taxonomy, Opportunities, and Challenges (concept-operator / RI-RC vocabulary) — [arXiv:2502.19649](https://arxiv.org/abs/2502.19649)
46. Park et al. 2023, The Linear Representation Hypothesis and the Geometry of LLMs — [arXiv:2311.03658](https://arxiv.org/abs/2311.03658)
47. Arditi et al. 2024, Refusal in Language Models Is Mediated by a Single Direction — [arXiv:2406.11717](https://arxiv.org/abs/2406.11717) (NeurIPS 2024)

---

*Reformatted as research proposal 2026-09-11 · updated 2026-09-12 (§4.6 representation choice, refs 36–46, D-12, terminology pass: model-agnostic / not portable, §7.8–7.9 objective ablation & cost analysis) · updated 2026-10-13 (citation corrections: CAA/ICV IDs fixed, Da Silva et al. replaces mis-attributed Tan et al., Subbiah et al. limits study, GCS venue, Agarwal effect sizes; +ref 47) · Living document — edit as design decisions are resolved.*