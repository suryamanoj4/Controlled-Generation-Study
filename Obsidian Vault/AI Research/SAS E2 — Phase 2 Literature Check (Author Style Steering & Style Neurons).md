# SAS E2 — Phase 2 Literature Check (Author Style Steering & Style Neurons)

> **Status:** literature check complete (2026-11-02) · before Phase 2 dataset curation and pipeline design
> **Related:** [[SAS E2 — Frozen Study Plan (Low-Resource Control & Style Steering)]] · [[SAS E2 — Phase 1 Pilot Results (Sentiment Steering)]] · [[SAS E2 — Low-Resource Style Steering Literature Review]]
> **Purpose:** establish what the author-style steering and style-neuron literature actually does, what data it uses, how it evaluates, and which choices we should adopt before curating Indic author corpora.

## 1. Executive summary

The literature points to six design decisions for Phase 2:

1. **Use contrastive, same-content pairs**, not independent author vs non-author corpora. The strongest results come from contrasting an author's text against a style-neutral version of the *same content* (StyleVector; few-shot author-style transfer).
2. **Use human-authored public-domain corpora** as the primary data. The Ionio pipeline uses LLM-generated author-style pairs and explicitly notes the circularity; its Hemingway effect is small (d ≈ 0.20) while Shakespeare is large (d ≈ 1.15).
3. **Sweep mid-to-late layers and α.** Style Vectors finds layers 18–20 of 33 best; StyleVector finds ~layer 15+ of 32 best; Ionio finds the 5th layer from the output best. α too large produces nonsense/repetition.
4. **Do not assume one vector captures author style.** A3S (EMNLP 2026) shows a common authorship backbone plus aspect-specific residuals; naive aggregation fails. StyleVector's limitations note the same single-vector conflation.
5. **Evaluate with a validated style classifier + content preservation + fluency + human pairwise judgment.** Style classifiers are the standard style-strength metric, but TST evaluation work shows metrics are unreliable in isolation.
6. **Add neuron-level arms only after the mean-vector arm works.** sNeuron-TST and NAM-TST show style-specific FFN neurons and strong style accuracy, but deactivation hurts fluency and needs contrastive decoding.

## 2. Papers that matter

### 2.1 Style Vectors for Steering Generative LLMs
**Konen et al., Findings of EACL 2024** · [arXiv:2402.01618](https://arxiv.org/abs/2402.01618) · [code](https://github.com/DLR-SC/style-vectors-for-steering-llms)

- **Method:** activation-based style vectors
  \[
  v_s^{(i)} = \bar a_s^{(i)} - \bar a_{S\setminus s}^{(i)}
  \]
  where \(\bar a_s\) is the mean layer-\(i\) activation for style class \(s\) and \(\bar a_{S\setminus s}\) is the mean over all other styles. Add \(v_s\) at layer \(i\): \(a' = a + \lambda v_s\).
- **Data:** Yelp (sentiment), GoEmotions (emotion), Shakespeare (modern vs Shakespearean; 18,395 paired sentences per side).
- **Findings:** layers **18–20 of 33** best; activation-based vectors outperform trained steering vectors; effect is stronger on **subjective prompts** than factual prompts; too-large \(\lambda\) produces nonsense/repetition; style vectors carry **target bias** (e.g., Yelp vectors drift toward food topics).
- **Evaluation:** VADER / RoBERTa emotion classifier on generated text; no human evaluation (explicit limitation).
- **Lesson for us:** use labeled style corpora, mid-to-late layers, λ sweep, and **human evaluation** because classifier-only evaluation is not enough. Style vectors are not content-neutral.

### 2.2 StyleVector: Personalized Text Generation with Contrastive Activation Steering
**Zhang et al., ACL 2025** · [arXiv:2503.05213](https://arxiv.org/abs/2503.05213) · [ACL](https://aclanthology.org/2025.acl-long.353/)

- **Method:** for each user/author, generate a **style-agnostic response** for the same input, then contrast:
  \[
  s_u^\ell = \frac{1}{|P_u|}\sum_i \left(h_\ell(x_i \oplus y_i) - h_\ell(x_i \oplus \hat y_i)\right)
  \]
  Last-token hidden states; mean difference; add \(\alpha s_u^\ell\) at **generated token positions**.
- **Data:** LaMP and LongLaMP personalization benchmarks; Llama-2-7B-chat; style-neutral responses from GPT-3.5.
- **Findings:** +8% METEOR / +11% ROUGE-L relative; one vector per user vs 1700× storage reduction over PEFT; best layers are **middle-to-late (~15+ of 32)**; α < 0 moves away from the user style; α too large disrupts generation.
- **Evaluation:** ROUGE-L/METEOR content metrics; RAG and PEFT baselines.
- **Lesson for us:** the **same-content contrast** is the important part. For author style, the negative side should be a style-neutral version of the same passage, not an unrelated author. Evaluate content preservation, not only style.

### 2.3 Author-Style Steering via Contrastive Activation Vectors
**Ionio, 2026 (blog, not peer-reviewed)** · [post](https://research.ionio.ai/posts/2025/author-style-steering-activation-vectors)

- **Method:** \(v_{\text{author}} = \text{mean}(h_{\text{author}}) - \text{mean}(h_{\text{all}})\), mean-pooled token activations, L2-normalized; add \(h' = h + \alpha v\).
- **Data:** 1,500+ **LLM-generated** style-transformed pairs for Shakespeare, Austen, Hemingway; GPT-2; 100 narrative prompts.
- **Findings:** layers 1–7 from the output side; α 0.5–4; best at the **5th layer from the output**; style judge + MiniLM similarity. Shakespeare +0.57 (d≈1.15), Austen +0.48 (d≈0.96), Hemingway +0.10 (d≈0.20). No uniform length inflation.
- **Limitations:** **synthetic style supervision / dataset circularity**; vectors reflect LLM approximations of authors, not original prose; Hemingway's minimalist style is close to GPT-2's default, so the effect is weak; high α causes truncation/repetition/coherence loss.
- **Lesson for us:** do **not** build the primary Phase 2 dataset from LLM-generated author-style pairs. Use human-authored texts. Expect author-dependent effect sizes; some authors may be too close to the model's default style. Report length and fluency controls.

### 2.4 Style-Specific Neurons for Steering LLMs in Text Style Transfer (sNeuron-TST)
**Lai, Hangya & Fraser, EMNLP 2024** · [arXiv:2410.00593](https://arxiv.org/abs/2410.00593) · [code](https://github.com/wenlai-lavine/sNeuron-TST)

- **Method:** identify **FFN neurons** that activate for source and target styles; select top-\(k\) per style; remove overlapping neurons; **deactivate source-style-only neurons** during decoding; add a **contrastive decoding** step (DoLa-style) to recover fluency.
- **Findings:** style-specific neurons exist; style information concentrates in the **last few layers**; deactivating source-only neurons increases target-style vocabulary but hurts fluency; contrastive decoding restores fluency; stronger than APE/AVF/PNMA neuron-selection baselines.
- **Benchmarks:** formality, toxicity, politics, politeness, **authorship (Shakespeare/modern)**, sentiment; 12 directions.
- **Metrics:** style accuracy, **LaBSE** content preservation, **PPL** fluency; also copy-rate analysis.
- **Lesson for us:** neuron-level steering is a credible secondary arm; overlap removal is essential; deactivation alone is not enough; need contrastive decoding/quality control. Authorship is already a benchmark task.

### 2.5 Neuron Activation Modulation for Text Style Transfer (NAM-TST)
**Kong et al., Findings of ACL 2025** · [ACL](https://aclanthology.org/2025.findings-acl.403/)

- **Method:** gradient-based activation-difference analysis to identify style-related neurons; compute source-target activation differences; during generation align style-neuron activations to the target style.
- **Findings:** mitigates the **unidirectional issue** in LLM style transfer (models often transfer only in the safer direction); improves style transfer quality while preserving content consistency.
- **Lesson for us:** style transfer has a directional asymmetry; evaluate both directions (author→neutral and neutral→author). Neuron modulation is a potential operator, but it needs gradient-based attribution and careful content checks.

### 2.6 Predicting Steering Vectors and Adapter Weights for Few-Shot Author-Style Transfer
**Popp et al., W-NUT @ EMNLP 2026** · [arXiv:2610.03163](https://arxiv.org/abs/2610.03163)

- **Model:** **Qwen3-4B-Instruct-2507**, 36 layers, d=2,560 — the closest paper to our setting.
- **Method:** author-level contrastive activation steering; author abstracts vs **style-neutral generations for the same content**; also a network that predicts steering vectors and a hypernetwork that predicts LoRA adapters.
- **Evaluation:** kNN author classifier (DistilBERT), style vs quality trade-off; 1,024 max tokens; T=0.7, repetition penalty 1.05, top-p 0.8, top-k 20.
- **Findings:** fine-tuning buys the most style but loses fluency; hypernetwork gives the best style-quality trade-off; manually extracted and predicted vectors are near-orthogonal yet comparable — **there may be multiple valid style directions**, not one canonical axis.
- **Lesson for us:** exactly our model size; use topic/content-fixed author vs neutral contrast; expect a style-quality trade-off; do not assume one vector is the only solution; evaluate quality/fluency.

### 2.7 Can Activation Steering Capture Multidimensional Authorship Style? (A3S)
**Tran, Bao & Carpuat, EMNLP 2026** · [arXiv:2609.04792](https://arxiv.org/abs/2609.04792)

- **Method:** structured contrastive prompting along rhetorically motivated dimensions; per-aspect contrastive directions; a common authorship backbone plus aspect-specific residuals; **interference-aware aggregation**; per-instance steering strength; training-free.
- **Findings:** naive aggregation of aspect directions fails because they conflict on residuals; A3S improves multidimensional author style transfer and beats a trained baseline in preference evaluations; keeps target-exemplar overlap low.
- **Lesson for us:** author style is **multidimensional**. A single mean-difference vector may conflate lexical, syntactic, and discourse style. Start with a mean vector, but design the pilot to test aspect-specific vectors (lexical, syntactic, tone/discourse) and interference-aware aggregation.

### 2.8 Methodological and evaluation papers

- **Mean-centring** *(Improving Activation Steering in Language Models with Mean-Centring)*: subtracting the mean of all training activations improves steering vectors; relevant to our vector construction.
- **A Sober Look at Steering Vectors for LLMs**: steering is not universally effective; effect sizes and specificity need controls; supports our Phase 1 random-null lesson.
- **Evaluating Text Style Transfer Evaluation: Are There Any Reliable Metrics?** (NAACL 2025 SRW): style classifiers, content metrics and LLM judges are individually unreliable; human evaluation is necessary.
- **TSTBench** (2025): style strength (RoBERTa classifier), content preservation (sBLEU/sBERT), fluency (PPL/CoLA), joint score; notes that metrics designed for fine-tuned models may not fit LLM rewrites.
- **CAA / Contrastive Activation Addition** (Rimsky et al., 2024): contrastive prompt pairs, layer/α sweep, random control; the base recipe.
- **ITI / RepE**: probe-based directions and inference-time intervention; useful as baselines, less specific to author style.

### 2.9 Indic style work and the gap

- **Mukherjee et al., INLG 2024** — multilingual sentiment transfer for eight Indian languages (Hindi, Magahi, Marathi, Malayalam, Punjabi, Odia, Telugu, Urdu); 1,000 positive + 1,000 negative style-parallel sentences per language; Llama2/GPT-3.5 baselines. This is **sentiment transfer**, not author style.
- **Liu et al., 2026** — cross-lingual figurative language steering; includes Bengali; CAA directions, mono- vs cross-lingual, random control, judge/manual audit.
- **Cegin et al., 2026** — language/quality steering across 11 languages; 3 depths × α; downstream classifier, diversity, judge.
- **No paper found on author-style activation steering in Indic languages.** The closest work is sentiment/style transfer by training, not author-level activation steering. This is the gap and the opportunity.

## 3. What this means for Phase 2 dataset curation

### Primary data
- **Human-authored public-domain corpora**, not LLM-generated style pairs. The Ionio circularity caveat is decisive.
- **Author-level pairs with the same content:** each passage should have an author version and a **style-neutral paraphrase/continuation of the same content**. The neutral side can be model-generated, but must be validated and recorded; otherwise use human neutral paraphrases.
- **At least two authors per language** (target + contrast) plus neutral. Ideally 3+ authors to test held-out author generalization.
- **300–500 docs per side where available.** Telugu is capped at 177 (Gurajada vs Veerasalingam); report the small-n flag.
- **Match length, topic and period** where possible. Avoid topic confounds; use same-content pairs rather than independent corpora.
- **Held-out splits by work/year**, not random sentences, to test author generalization.

### Candidate corpora
- **EN:** Shakespeare vs other early-modern drama (Marlowe/Jonson/Webster).
- **HI:** Premchand vs Prasad.
- **TE:** Gurajada vs Veerasalingam (177 cap).
- **BN / TA:** public-domain verification still pending; do not block the EN/HI/TE pilot on them.

### Data to avoid as primary
- LLM-generated "write like author" pairs (Ionio) — circular.
- Independent author vs non-author corpora without content matching — topic/style confound (our Phase 1 lesson).
- Synthetic style transfer outputs evaluated against the same model family — circular.

### Prompt set
- **Neutral content prompts**, no author name at inference. The vector carries style; the prompt carries content.
- Separate **prompting baseline** with the author name ("Write in the style of X") only for comparison.
- 100 prompts/language for the full run; 10–20 for the pilot.
- Include **factual and subjective prompts** (Style Vectors: style effects are stronger on subjective prompts).
- Evaluate **both directions**: neutral→author and author→neutral; style transfer is directional (NAM-TST).

## 4. What this means for the pipeline

### Vector construction
- Contrastive **same-content** pairs: author text vs style-neutral text.
- Mean difference first; also try logistic regression/PCA (StyleVector compares them; mean difference was best).
- **L2-normalize**; optionally **mean-centre**.
- Last-token pooling; also test mean pooling.
- Fit per author, per layer; cache vectors.

### Injection
- Mid-to-late layers. Qwen3-4B has 36 layers; start with **18, 24, 27, 31** (and maybe 9 for control).
- α ∈ {0.5, 1, 2, 4}; α too high → repetition/truncation/coherence loss.
- Add at **generated token positions** (StyleVector) and compare with all positions (CAA).
- Consider **aspect-specific vectors** (lexical, syntactic, discourse) and interference-aware aggregation (A3S) after the mean-vector pilot.
- Add **neuron-level arms** (sNeuron-TST / NAM-TST) only after the vector arm is validated; they need contrastive decoding for fluency.

### Evaluation
- **Author/style classifier** (DistilBERT/XLM-R) trained on author vs contrast, validated on **generated text**, with held-out works.
- **Content preservation:** LaBSE cosine, BLEURT/ROUGE; **length** and **copy rate**.
- **Fluency:** external PPL (e.g., GPT-2 or a multilingual model) and human rating.
- **Style strength:** classifier probability/margin + pairwise LLM judge (1–5) + human pairwise judgment.
- **Human eval:** 25 outputs/language/author, 2 annotators, κ ≥ 0.6.
- **Random-vector null:** 3–5 norm-matched random vectors; fitted vector must beat the null, not just baseline.
- **Prompting baseline:** zero-shot author-name prompting; optionally few-shot exemplars.
- **Fine-tuning/LoRA baseline:** for the style-quality trade-off; not the main arm.
- **Statistical tests:** paired tests, BH correction; report effect sizes and CIs.

### Efficiency
- Batched generation, one model load per model; paired seeds; length buckets.
- Cache vectors, classifier, and judge responses.
- Pilot EN/HI/TE first; add BN/TA when corpora exist.
- Phase 3 only after Phase 2 vectors are validated.

## 5. Recommended Phase 2 pilot (before full curation)

**Goal:** check that author style is steerable with human-authored, same-content data before building the full corpus.

- Languages: **EN, HI, TE**
- Authors: 2 per language (target + contrast)
- Data: 30–50 passages per author for the pilot; same-content neutral paraphrase for each
- Layers: **18, 27** (plus 31 if time)
- α: **0.5, 1, 2**
- Conditions: baseline, author vector, random vector (3 seeds), prompting baseline
- Prompts: 10 neutral content prompts/language
- Metrics: author classifier (held-out), LaBSE content, PPL, LLM pairwise, 5 human outputs/author
- Pass: author vector > random null and > baseline on the style classifier, with content preservation and no fluency collapse, in ≥2/3 languages.

If the pilot works, scale to 300–500 docs/side; if not, the bottleneck is data/style separability, not the operator.

## 6. Risks and open questions

- **Author style is multidimensional**; one vector may be insufficient (A3S; StyleVector limitation).
- **Effect sizes are author-dependent**; Hemingway-like minimal styles may be weak (Ionio d≈0.20).
- **Synthetic data circularity**; do not use LLM-generated author pairs as the primary corpus.
- **Style classifier reliability**; validate on generated text, not only human corpora.
- **Content/style entanglement**; same-content pairs are essential.
- **Fluency trade-off**; high α and neuron deactivation degrade fluency.
- **Indic data scarcity**; Bengali/Tamil public-domain corpora still pending; Telugu cap 177.
- **No prior Indic author-style activation steering**; this is a genuine gap, but also less methodological support.

## 7. Sources

- Konen et al., *Style Vectors for Steering Generative Large Language Models*, Findings EACL 2024. https://arxiv.org/abs/2402.01618
- Zhang et al., *Personalized Text Generation with Contrastive Activation Steering*, ACL 2025. https://arxiv.org/abs/2503.05213
- Patel, *Author-Style Steering via Contrastive Activation Vectors*, Ionio 2026 (blog). https://research.ionio.ai/posts/2025/author-style-steering-activation-vectors
- Lai et al., *Style-Specific Neurons for Steering LLMs in Text Style Transfer*, EMNLP 2024. https://arxiv.org/abs/2410.00593
- Kong et al., *Neuron Activation Modulation for Text Style Transfer*, Findings ACL 2025. https://aclanthology.org/2025.findings-acl.403/
- Popp et al., *Predicting Steering Vectors and Adapter Weights for Few-Shot Author-Style Transfer*, W-NUT @ EMNLP 2026. https://arxiv.org/abs/2610.03163
- Tran et al., *Can Activation Steering Capture Multidimensional Authorship Style?*, EMNLP 2026. https://arxiv.org/abs/2609.04792
- Rimsky et al., *Steering Llama-2 with Contrastive Activation Additions*, 2024. https://turntrout.com/llama2-steering-vectors
- *Improving Activation Steering in Language Models with Mean-Centring*. https://kclpure.kcl.ac.uk/portal/en/publications/improving-activation-steering-in-language-models-with-mean-centri
- *Evaluating Text Style Transfer Evaluation: Are There Any Reliable Metrics?*, NAACL 2025 SRW. https://aclanthology.org/2025.naacl-srw.41.pdf
- TSTBench, *A Comprehensive Benchmark for Text Style Transfer*, 2025. https://www.mdpi.com/1099-4300/27/6/575
- Mukherjee et al., *Multilingual Text Style Transfer: Datasets & Models for Indian Languages*, INLG 2024. https://aclanthology.org/2024.inlg-main.41.pdf
- Liu et al., *Cross-Lingual Steering for Figurative Language Generation*, 2026. https://arxiv.org/abs/2605.30443
- Cegin et al., *Want Better Synthetic Data? Steer It*, EMNLP 2026. https://arxiv.org/abs/2606.18389
