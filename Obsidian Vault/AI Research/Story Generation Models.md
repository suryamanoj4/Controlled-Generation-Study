# Story Generation Models

## Overview

Neural story generation is the task of automatically producing coherent, creative, and engaging narrative text using language models. It spans architectures from early LSTM/RNN-based models to modern large-scale Transformers and LLMs.

---

## Key Papers

### Foundational & Hierarchical Models

| Paper | Year | Authors | Key Contribution |
|-------|------|---------|-----------------|
| **Hierarchical Neural Story Generation** | 2018 | Fan, Lewis, Dauphin | Introduced WritingPrompts dataset; premise→text hierarchical approach with model fusion. |
| **Strategies for Structuring Story Generation** | 2019 | Fan, Lewis, Dauphin | Coarse-to-fine: predicate-argument skeleton → surface text improves entity/event coherence. |
| **Plan-And-Write** | 2019 | Yao, Peng, et al. | First explicit plan→write framework; compares static vs dynamic planning. |
| **Plan, Write, and Revise** | 2019 | Goldfarb-Tarrant et al. | Interactive human-in-the-loop; 10–50% quality improvement with collaborative revision. |

### Plot & Event-Guided Models

| Paper | Year | Key Idea |
|-------|------|----------|
| **Event Representations for Story Generation** | 2016 | Martin et al. — ASER knowledge base; event graph planning. |
| **Cue Me In** | 2020 | Peng et al. — Content-inducing approaches for interactive story generation. |
| **Knowledge Base Story Generation** | 2021 | Ammanabrolu et al. — ConceptNet/Wiki grounding for narrative coherence. |

### Infilling & Revision

| Paper | Year | Key Idea |
|-------|------|----------|
| **HINT: Hierarchical INfillTing** | 2019 | Ippolito et al. — Generate middle given beginning+end; improves ending coherence. |
| **Fusion Models** | 2018 | Fan et al. — Concatenate pretrained LM + content model representations. |
| **Write, Execute, Assess** | 2020 | Prabhumoye et al. — Cyclic writing, event simulation, coherence checking. |

### Adversarial & Diversity Models

| Paper | Year | Key Idea |
|-------|------|----------|
| **DRAGON** | 2021 | Hu et al. — GAN training with discriminator for story-level coherence. |
| **T-CVAE** | 2021 | Wang & Wan — Transformer CVAE; latent story-plan space for diversity. |

### LLM-Based & Recent

| Paper | Year | Key Idea |
|-------|------|----------|
| **TinyStories** | 2023 | Eldan & Li — Synthetic simple stories; GPT-4-as-judge evaluation. |
| **LLaMA** | 2023 | Touvron et al. — Open foundation model family widely used for story gen. |
| **Generative Agents** | 2023 | Park et al. — Agent-based narrative simulation with memory+reflection. |
| **Story Generation with LLMs** | 2024 | Chen et al. — Benchmarks GPT-3/4, LLaMA on WritingPrompts. |
| **RecurrentGPT** | 2023 | Zhou et al. — Language-based LSTM-style recurrence for arbitrarily long text generation. |
| **DOC** | 2024 | Deep open-ended story generation with 3-4 level hierarchy + story state tracking. |
| **STORYGENERATOR** | 2025 | Multi-agent framework (4 agents: planner, writer, editor, consistency check) for 100K+ word narratives. |

### Controlled Generation

| Paper | Year | Key Idea |
|-------|------|----------|
| **PPLM** | 2020 | Dathathri et al. — Plug-and-play: gradient-based attribute control without LM retraining. |
| **GeDi** | 2020 | Krause et al. — Generative discriminators guide large LMs via Bayes rule; 30x faster than PPLM. |
| **FUDGE** | 2021 | Yang & Klein — Future discriminators on partial sequences for global attribute control. |
| **Contrastive Decoding** | 2023 | Li et al. — Contrast pretrained vs amateur LM scores to reduce repetition in open-ended gen. |

### Diffusion & State-Space Models

| Paper | Year | Key Idea |
|-------|------|----------|
| **Diffusion-LM** | 2022 | Li et al. — First continuous diffusion LM; enables fine-grained controllable generation. |
| **SSD-LM** | 2023 | Han et al. — Semi-autoregressive simplex-based diffusion; plug-and-play classifier guidance. |
| **Mamba Story Ending** | 2024 | Sharma & Santhanam — First SSM (Mamba) fine-tuned for story ending generation; competitive with GPT-3.5. |

---

## Architectural Patterns

```
Pattern                  Representative Models              Core Idea
────────────────────────────────────────────────────────────────────────
Left-to-Right LM         GPT-2, GPT-3, LLaMA               Autoregressive next-token prediction
Hierarchical             HNSG, Plan-and-Write, Strategies   Generate premise/plan first, then expand
Coarse-to-Fine           Strategies (pred-arg → surface)    Predicate-argument skeletons → text
Fusion / Modality Gating HNSG fusion                        Combine content & language model representations
Infilling / In-painting  HINT                               Generate middle given beginning + ending
VAE / Latent Variable    T-CVAE, PlotMachines               Latent plan space for diversity & control
GAN-based                DRAGON, RELGAN                     Discriminator enforces story-level coherence
LLM + Prompting          GPT-3/4, instruction-tuned         Few-shot / in-context story generation
Agent-based              Generative Agents                  Memory + reflection + planning loop
Diffusion LM             Diffusion-LM, SSD-LM               Iterative denoising for controllable generation
State-Space Model        Mamba (Sharma 2024)                Linear complexity SSM for long-form narrative
Multi-Level Hierarchical DOC, STORYGENERATOR (2025)         3-4 level (world→chapter→paragraph→sentence)
Recurrent / Memory       RecurrentGPT, DOME                 Language-based LSTM-style recurrence
Retrieval-Augmented      STORM, RECOMP                      External memory retrieval for long-form coherence
Symbolic Planning        HTN planning, GLAMOR               Causal coherence via structured planning
```

---

## Evaluation Methods

### Automated Metrics

| Metric | What It Measures | Limitation |
|--------|-----------------|------------|
| Perplexity | Fluency | Does not measure coherence |
| BLEU / ROUGE | N-gram overlap with reference | Poor for open-ended creative text |
| BERTScore | BERT-based semantic similarity | Still reference-dependent |
| Diversity (dist-1/2, TTR) | Lexical variety | Does not capture plot quality |
| Self-BLEU / Repetition | Intra-story repetition | Narrow focus |
| MAUVE | Distribution alignment with human text | Aggregate, not per-story |
| Sentence-BERT similarity | Coherence between sentence pairs | Correlation still imperfect |

### Human Evaluation Dimensions

- **Coherence** — Logical causal/temporal flow
- **Consistency** — Stable facts, events, character attributes
- **Fluency** — Well-formed surface text
- **Interestingness** — Compelling or surprising narrative
- **Relevance** — Adherence to prompt/title
- **Novelty** — Going beyond cliché or obvious plots

### Story-Specific Methods

- **Story Cloze Test** (Mostafazadeh et al., 2016) — Choose correct ending from two options
- **GPT-4-as-Judge** (Eldan & Li, 2023) — LLM grades stories on multiple dimensions
- **Entity Tracking** — Count entity drift/retention across sentences
- **Event Chain Coherence** — Compare to script knowledge
- **Temporal Conflict Analysis** (DOME, 2024) — Detect contradictions via temporal knowledge graphs
- **Graph-Based Hallucination Detection** — Extract entity graph; check for contradictions

---

## Extended Topics

### Structured Prompts & Story Grammar

| Approach | Key Work | Description |
|----------|----------|-------------|
| **Narrative Schemas** | Chambers & Jurafsky (2009) | Event chains (scripts/schemas) learned from text; instantiated for story gen. |
| **PlotMachines** | Goldbraikh & Elhadad (2020) | Outline-conditioned generation with dynamic plot state tracking. |
| **RecurrentGPT** | Zhou et al. (2023) | LSTM-style recurrence: short-term memory (recent paragraphs) + long-term (plot plan). |
| **CoT Story Planning** | Multiple (2024) | Chain-of-Thought prompting for plot decomposition before generation. |

### Plot Generation via Planning Algorithms

| Paradigm | Key Work | Strength | Weakness |
|----------|----------|----------|----------|
| **HTN Planning** | Ware & Young (2014) | Guaranteed causal coherence | Requires hand-authored action models |
| **LLM-Based Planning** | GLAMOR (2023+), CoT planners | Flexible, no manual modeling | Less reliable causal structure |
| **Causal Narrative Planning** | Multiple (2024) | Causal event chains; character actions | Computationally expensive |
| **Plan-and-Write** | Yao et al. (2019) | Keyword-sequence planning | Limited to short stories |

### Retrieval-Augmented Story Generation

| System | Year | Key Idea |
|--------|------|----------|
| **STORM** | 2024 | Retrieves stylistically similar passages for stylistic consistency. |
| **RECOMP** | 2024 | Compresses retrieved segments into compact representations; enables 10K+ word stories. |
| **Hierarchical RAG** | 2025 | Two-level: retrieve plot outlines at planning level, sentence exemplars at surface level. |

### Self-Play & Synthetic Data

| Technique | Key Work | Description |
|-----------|----------|-------------|
| **Self-Play Fine-Tuning** | 2024 | Generator + discriminator trained alternately; GAN-style at narrative level. |
| **Synthetic Outline-Text Pairs** | 2023 | GPT-4 generates (outline, story) pairs → fine-tune smaller models (1B-7B). |
| **Self-Improving Iterative** | 2024 | Generate → self-score → fine-tune on top drafts; RLHF-style with self-data. |
| **Author-Reader Self-Play** | 2024 | Author agent + Reader agent interaction produces branching narrative training data. |

### Hierarchical Generation Beyond 2022

| System | Year | Levels | Max Length | Key Innovation |
|--------|------|--------|------------|----------------|
| **DOC** | 2024 | 3-4 (world → chapter → paragraph → sentence) | 50K+ words | Story state tracking with entity/location/plot thread state |
| **RecurrentGPT** | 2023 | Recurrent paragraph-level | Unlimited | Language-based LSTM recurrence; natural language memory |
| **STORYGENERATOR** | 2025 | Multi-agent (4 agents) | 100K+ words | Specialized agents: planner, writer, editor, consistency |
| **Multi-Level Hierarchical** | 2025 | 3 (arc → chapter → scene) | 100K+ words | Shared "story graph" data structure |
| **Scene-Level Planning** | 2024 | Scene-by-scene | 10K+ words | Explicit scene: location, characters, narrative goal |

---

## Key Datasets

| Dataset | Size | Genre | Description |
|---------|------|-------|-------------|
| **WritingPrompts** | 300K pairs | Reddit prompt+story | Introduced by Fan et al. (2018) |
| **ROCStories** | 50K | 5-sentence everyday stories | Story Cloze Test split |
| **BookSum** | 405 books | Long-form fiction | Book-level summaries |
| **Story Commonsense** | 60K | Event chains + inferences | Event sequence knowledge graph |
| **TinyStories** | 2.1M | Synthetic simple stories | ~3-year-old vocabulary |
| **CBT** | 98 books | Children's stories | Cloze-style story understanding |
| **WikiPlots** | 112K | Plot summaries | Wikipedia plot sections |
| **Prose Antecedent** | ~100K | Novel-length fiction | NaNoGenMo corpus |
| **LitBank** | 100 novels | Literature | Annotated entities, events, narrative structure |

---

## Open Challenges

1. **Long-Form Coherence** — Topic drift beyond 2K tokens; quadratic attention scaling
2. **Plot Consistency & Endings** — Locally fluent but unresolved plot arcs
3. **Character Consistency** — Entity tracking brittleness; attribute/personality drift
4. **Diversity vs Quality Trade-off** — Sampling methods trade one for the other
5. **Evaluation** — No widely accepted automatic metric for open-ended narrative
6. **Rare Events & Creativity** — Models default to frequent/mundane patterns
7. **Grounding & Common Sense** — Stories violate physical/social common sense
8. **Length Scalability** — Novel-length (>50K tokens) remains unsolved
9. **Controllability** — Genre, sentiment, character traits, plot outcome control
10. **RLHF Biases** — Alignment training reduces creative risk-taking

---

## References

- Fan, Lewis, Dauphin (2018). [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833). ACL 2018.
- Yao et al. (2019). [Plan-And-Write](https://arxiv.org/abs/1811.05701). AAAI 2019.
- Goldfarb-Tarrant et al. (2019). [Plan, Write, and Revise](https://arxiv.org/abs/1904.02357). NAACL 2019 Demo.
- Eldan & Li (2023). [TinyStories](https://arxiv.org/abs/2305.07759). arXiv.
- Park et al. (2023). [Generative Agents](https://arxiv.org/abs/2304.03442). UIST 2023.
- Touvron et al. (2023). [LLaMA](https://arxiv.org/abs/2302.13971). arXiv.
- Li et al. (2022). [Diffusion-LM](https://arxiv.org/abs/2205.14217). NeurIPS 2022.
- Han et al. (2023). [SSD-LM](https://arxiv.org/abs/2210.17432). ACL 2023.
- Sharma & Santhanam (2024). [Mamba Story Ending Generation](https://arxiv.org/abs/2410.10848). arXiv.
- Yang & Klein (2021). [FUDGE](https://arxiv.org/abs/2104.05218). NAACL 2021.
- Dathathri et al. (2020). [PPLM](https://arxiv.org/abs/1912.02164). ICLR 2020.
- Zhou et al. (2023). [RecurrentGPT](https://arxiv.org/abs/2305.13304). arXiv.
- Chambers & Jurafsky (2009). [Narrative Schemas](https://aclanthology.org/N09-1025/). NAACL 2009.
- Xu et al. (2022). [SODA](https://arxiv.org/abs/2210.14638). EMNLP 2022.

---

*Related: [[Story Generation Index]], [[Low Resource Story Generation]], [[Agent Based Story Generation]]*
