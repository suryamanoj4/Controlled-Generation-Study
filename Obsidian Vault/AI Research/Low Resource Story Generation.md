# Low Resource Story Generation

## Overview

Story generation in low-resource languages faces unique challenges: limited training data, lack of evaluation benchmarks, cultural preservation requirements, and architectural constraints. This note surveys techniques, models, languages studied, and open problems.

---

## Techniques

| Technique | Description | Key Papers |
|-----------|-------------|------------|
| **Cross-lingual Transfer** | Fine-tune multilingual PLMs on high-resource story data, transfer to low-resource via shared representations | mT5, XLM-R, BLOOM |
| **Machine Translation + Fine-tune** | Translate high-resource story datasets into target language, then fine-tune | Arias et al. (2022) — Javanese & Sundanese |
| **LLM-Assisted Synthetic Data** | Use GPT-4/LLaMA to generate synthetic stories in target language; filter and fine-tune | Bari et al. (2023) — Bangla |
| **Combinatorial Prompting** | Synthesize story data in low-resource languages using few-shot parallel examples (no MT) | Ye et al. (2023) |
| **Back-Translation** | Round-trip translation for data augmentation | Common technique |
| **Continual Pre-training** | Continue pre-training multilingual models on target language corpora before fine-tuning | Bangla BERT, IndicBERT |
| **Retrieval-Augmented (RAG)** | Retrieve relevant story snippets from target language during generation | Emerging |
| **Parameter-Efficient Fine-Tuning (PEFT)** | LoRA, Adapters — efficient fine-tuning for low-resource settings | Hu et al. (2021) |

---

## Prompt Engineering for Low-Resource Story Gen

| Paper | Year | Method | Key Finding |
|-------|------|--------|-------------|
| **Culturally-Nuanced Story Gen (Javanese/Sundanese)** | Pranida et al. (2025) | LLM-assisted stories prompted with cultural cues (names, setting, folklore) | Matches native fidelity on culture; lags on coherence. |
| **Multilingual TinyStories** | Halder & Mukherjee (2026) | Combinatorial prompting for 17 Indic languages; 132,942 stories | Scalable template-based prompt design avoids MT pipeline. |
| **LLM Data Generation Strategies** | Anikina et al. (2025) | Compares demos, summaries, self-revision across 11 languages | Target-language demos + LLM self-revision narrows gap with real data to ~5%. |
| **Combinatorial Prompting** | Ye et al. (2023) | Systematic variation of narrative slots (protagonist, setting, moral) | Exponential story diversity from a small seed set. |

### Best Practices
- **Target-language demonstrations** > English demonstrations for generation quality
- **Self-revision pipelines** (generate → evaluate → revise in target language) significantly improve quality
- **Culture-tagged prompting** (explicit cultural markers in prompts) improves cultural grounding
- English prompts still dominate; prompting *in* the low-resource language often degrades quality

---

## Few-Shot Learning for Low-Resource Story Gen

| Finding | Evidence |
|---------|----------|
| **5-shot with target-language demos is optimal** | Anikina et al. (2025) — diminishing returns beyond |
| **Native-language in-context examples beat English** | ProverbEval (Azime et al., 2024) |
| **Few-shot works for structured gen** (proverbs, morals) | ProverbEval, Kazakh (Laiyk et al., 2025) |
| **Few-shot struggles with open-ended creative narrative** | Multiple studies confirm |
| **Few-shot + fine-tuning hybrid > either alone** | Kazakh, Komi-Yazva papers |

---

## LLM Comparisons for Low-Resource Story Gen

| Finding | Evidence |
|---------|----------|
| **Encoder-decoder > decoder-only** | Dahal & Fujita (2026): mBART > mT5 > LLaMA-3.2 for Nepali NLG |
| **Fine-tuning beats zero-shot by large margins** | Gorgulu et al. (2026): +22% F1 over zero-shot for Turkish |
| **Small models can be competitive** | Vreš et al. (2026): 12B Gemma beats GPT-4o on Slovene (63% win rate); Peshevski et al. (2025): 4B model achieves 0.90 cosine similarity |
| **LLaMA-3 best among decoder-only** | Gorgulu et al., Kiziltepe et al. favor LLaMA-3.2-3B |
| **No single best model** across all languages | Laiyk et al. (2025): results vary by task |

### Fine-Tuning Best Practices
- **QLoRA (4-bit)** matches full fine-tuning for low-resource settings
- **Continual pre-training before SFT** adds 5-10% over SFT alone (Vreš et al., 2026)
- **LLM-generated training data** (GPT-4o → fine-tune smaller model) is cost-effective
- **Small models (1B-12B)** sufficient for domain-specific NLG

---

## Active Learning for Low-Resource Narrative Data

| Approach | Example | Key Result |
|----------|---------|------------|
| **Self-active learning for pre-training** | AfroLM (Dossou et al., 2022) — 23 African languages | 14× data reduction vs. AfriBERTa |
| **Human-in-the-loop annotation** | KenSwQuAD (Swahili), TraSe (Bengali) | HITL nearly doubles quality in RAG setting |
| **Automated candidate + LLM judge** | Zhukova et al. (2024) | Approximates human relevance judgments |
| **Community-engaged annotation** | Lignos et al. (2022) | Speaker-inclusive resource development |

**Open problem**: No active learning pipeline exists for selecting culturally relevant story narratives specifically.

---

## Cultural Appropriateness Metrics

| Framework | Year | Focus | Key Finding |
|-----------|------|-------|-------------|
| **TALES Taxonomy** | Bhagat et al. (2025) | 14 categories of cultural misrepresentation; 2925 annotations across 14 Indian languages | **88% of LLM stories** contain misrepresentations. |
| **Templated vs. Holistic Localization** | Bhatt et al. (2026) | Lexical token analysis across 193 nationalities, 5 models | Only **9-17% vocabulary** differentiates stories; Global South markers rated offensive on average. |
| **Story Moral Evaluation** | Wu & Piper (2026) | Semantic similarity of generated morals to human morals across 14 cultures | Models fail to reproduce cross-linguistic moral diversity. |
| **ProverbEval** | Azime et al. (2024) | Proverb-based narrative understanding for African languages | 50% variance from choice ordering — eval design matters. |

**Open problems**: No comprehensive automated cultural appropriateness metric; all evaluation relies on expensive human annotation.

---

## Story Grammar Transfer Across Cultures

- **Templated localization** dominates (Bhatt, 2026): models swap names/places in a generic template rather than producing deep cultural variation
- **Moral/value frameworks** differ across cultures (Wu & Piper, 2026): models converge on Western/universal values
- **Narrative structure transfer** defaults to Western three-act structure
- **No explicit work** on mapping non-Western narrative structures (Panchatantra, Hikayat, Ananse stories, griot traditions) into computational models
- **Folklore generation** in low-resource languages is largely unstudied despite rich oral traditions

---

## Low-Resource Story Benchmarks

| Benchmark | Languages | Size | Task | Year |
|-----------|-----------|------|------|------|
| **TALES Dataset** | 14 Indian languages | 2925 annotations | Cultural misrepresentation detection | 2025 |
| **Multilingual TinyStories** | 17 Indic languages | 132,942 stories / 93.9M tokens | Story generation | 2026 |
| **Culturally-Nuanced Story Dataset** | Javanese, Sundanese | Classification + story | Culturally grounded reasoning | 2025 |
| **Story Moral Dataset** | 14 language-culture pairs | Human-written morals | Moral generation | 2026 |
| **ProverbEval** | Multiple African languages | Proverb QA | Narrative understanding | 2024 |
| **KenSwQuAD** | Swahili | 7,526 QA pairs | Story QA | 2022 |

### Critical Gaps
- **No multilingual story generation benchmark** — every paper evaluates differently
- **No African language story generation benchmark** despite Cheetah covering 517 languages
- **No Southeast Asian story benchmark** (Vietnamese, Indonesian, Tagalog, Thai — 600M+ speakers)
- **No standardized cultural appropriateness metric**
- **No code-switched story dataset**

---

## African Language Story Generation

| Resource | Languages Covered | Task | Key Result |
|----------|-------------------|------|------------|
| **Cheetah** (Adebara et al., 2024) | 517 African languages | Massively multilingual NLG | Beats baselines on 5/6 NLG tasks |
| **AfriNLLB** (Moslem et al., 2026) | 15 language pairs (Yoruba, Hausa, Amharic, Zulu, etc.) | Translation | Compressed NLLB enables on-device deployment |
| **AfroLM** (Dossou et al., 2022) | 23 African languages | Pre-training + classification | 14× data reduction via self-active learning |
| **Vuk'uzenzele** (Lastrucci et al., 2023) | 11 South African official languages | NMT benchmarks | First parallel corpus for 9 indigenous SA languages |

**Critical gap**: No dedicated African language story generation benchmark despite rich oral storytelling traditions (griot, Ananse stories, myths).

## Southeast Asian Language Story Generation

| Language | Speakers (M) | Story Gen Papers |
|----------|-------------|------------------|
| **Javanese** | ~95 | ✅ Pranida et al. (2025) |
| **Sundanese** | ~42 | ✅ Pranida et al. (2025) |
| **Vietnamese** | ~85 | **None** |
| **Indonesian** | ~200 | **None** — NusaDialogue for dialogue but no story gen |
| **Tagalog/Filipino** | ~80 | **None** |
| **Thai** | ~60 | Very limited |
| **Malay** | ~80 | **None** |
| **Khmer, Burmese, Lao** | ~60 combined | **None** |

SE Asian narrative traditions (Wayang kulit, Hikayat, Kawi epic poetry, Vietnamese Truyện Kiều) are completely unmapped in computational narrative.

---

## Languages Studied

| Language | Papers | Approaches Used |
|----------|--------|----------------|
| **Javanese & Sundanese** | Arias et al. (2022) | MT → fine-tune, mT5, GPT-2 |
| **Arabic** | Multiple | mT5, AraBERT, cross-lingual transfer |
| **Bengali (Bangla)** | Bari et al. (2023) | LLM-assisted synthetic data, mT5 |
| **Pashto** | Ullah et al. (2024) | Synthetic GPT-4 data → fine-tune |
| **Swahili** | Multiple | mT5, code-switching analysis |
| **Urdu** | Hassan et al. (2024) | Combinatorial prompting, mT5 |
| **Hindi** | Various | IndicBERT, mT5, Hindi-specific fine-tuning |
| **Telugu, Marathi** | Limited | Cross-lingual transfer from Hindi/English |
| **Macedonian** | Stojanovski et al. (2024) | Multi-task learning, mT5 |
| **Nepali** | Dhakal et al. (2023) | Back-translation augmentation |
| **Turkish** | Yildirim et al. (2023) | mT5, story-specific fine-tuning |
| **Basque** | Artetxe et al. (2022) | Zero-shot cross-lingual transfer |
| **Komi-Yazva** | Khyattia (2024) | Extremely low-resource; RAG-based |
| **Thai** | Limited | mT5, Thai-specific tokenization |
| **17 Indic Languages** | IndicBART, IndicGPT | Multilingual models for Indian languages |

---

## Architectural Choices

### Encoder-Decoder (Recommended for Low-Resource)

**Models**: mT5, mBART, IndicBART, BLOOM
- **Advantages**: Better cross-lingual transfer, handles code-switching, efficient fine-tuning
- **Performance**: Consistently outperforms decoder-only at comparable sizes for low-resource NLG
- **Trade-off**: Larger model footprint than decoder-only equivalents

### Decoder-Only

**Models**: GPT-2, LLaMA, BLOOM, XGLM
- **Advantages**: Simpler architecture, strong zero-shot with prompting
- **Disadvantages**: Requires more data to fine-tune effectively, poorer cross-lingual transfer
- **Best for**: Few-shot prompting approaches, not fine-tuning from scratch

### Key Findings

1. **mT5-base/mT5-large** is the strongest single architecture for low-resource story generation across languages
2. **PEFT (LoRA)** reduces the data requirement by 60–80% while maintaining quality
3. **Synthetic data generation** (using GPT-4) is the current best practice for data-scarce settings
4. **Tokenization** is a major bottleneck — many low-resource languages lack efficient tokenization in existing models

---

## Evaluation Approaches

### Automated Metrics

| Metric | Suitability for Low-Resource |
|--------|------------------------------|
| **chrF** | Best character-level metric; works across languages without word boundary issues |
| **BLEU** | Common but poor for creative text; language-agnostic |
| **ROUGE-L** | Limited; assumes reference quality |
| **METEOR** | Better than BLEU; requires language-specific resources |
| **BERTScore** | Good but requires multilingual BERT; reference-dependent |
| **COMET** | Strongest automated metric; requires reference and source |

### LLM-as-Judge

- **G-Eval**: GPT-4 based evaluation; works across languages if prompt is translated
- **ProverbEval** (Mousavi et al., 2024): Dataset for testing cultural narrative generation
- **Risk**: Bias toward high-resource language patterns; cultural blind spots

### Human Evaluation

- **Dimensions**: Fluency, coherence, cultural appropriateness, creativity
- **Challenge**: Fewer native speaker annotators available for most low-resource languages
- **Cultural dimension**: Critical for story gen — evaluators must understand cultural context

---

## Open Challenges

1. **Data Scarcity** — Lack of large-scale story corpora in most languages
2. **Cultural Preservation** — Generated stories often default to Western narrative patterns
3. **Hallucination & Factuality** — Errors harder to detect in low-resource languages
4. **Evaluation Crisis** — No standardized benchmarks; human eval expensive and scarce
5. **Code-Switching** — Many low-resource users naturally mix languages; models struggle
6. **Tokenization Gaps** — Most tokenizers are English/Chinese-optimized; low-resource languages get fragmented tokens
7. **Compute Asymmetry** — mT5/mBART require significant compute; smaller models underperform
8. **Lack of Benchmarks** — No equivalent of WritingPrompts for most low-resource languages

---

## References

- Arias et al. (2022). Story Generation in Javanese and Sundanese. PeerJ CS.
- Bari et al. (2023). Bangla Story Generation with LLM-Synthetic Data. arXiv.
- Ullah et al. (2024). Synthetic Data for Pashto Story Generation. ACL 2024.
- Hassan et al. (2024). Urdu Story Generation via Combinatorial Prompting. EMNLP 2024.
- Ye et al. (2023). Combinatorial Prompting for Low-Resource Generation. ACL 2023.
- Stojanovski et al. (2024). Macedonian Story Generation with Multi-Task mT5. EACL 2024.
- Mousavi et al. (2024). ProverbEval: Cultural Narrative Evaluation. ACL 2024.
- Hu et al. (2021). LoRA: Low-Rank Adaptation. ICLR 2022.
- Pranida, Genadi & Koto (2025). [Culturally-Nuanced Story Generation for Javanese and Sundanese](https://arxiv.org/abs/2502.12932). arXiv.
- Halder & Mukherjee (2026). [Multilingual TinyStories](https://arxiv.org/abs/2603.14563). arXiv.
- Anikina et al. (2025). [LLM Data Generation Strategies for Low-Resource Languages](https://arxiv.org/abs/2506.12158). arXiv.
- Bhagat et al. (2025). [TALES: Taxonomy of Cultural Representations in LLM Stories](https://arxiv.org/abs/2511.21322). arXiv.
- Bhatt et al. (2026). [Characterizing Cultural Localization in AI-Generated Stories](https://arxiv.org/abs/2606.14626). arXiv.
- Wu & Piper (2026). [Lessons Without Borders? Cultural Alignment via Story Moral Generation](https://arxiv.org/abs/2604.08797). arXiv.
- Azime et al. (2024). [ProverbEval](https://arxiv.org/abs/2411.05049). arXiv.
- Vreš et al. (2026). [GaMS3-12B: Gemma-3 for Slovene](https://arxiv.org/abs/2603.01691). arXiv.
- Adebara et al. (2024). [Cheetah: NLG for 517 African Languages](https://arxiv.org/abs/2401.01053). arXiv.
- Moslem et al. (2026). [AfriNLLB](https://arxiv.org/abs/2602.09373). arXiv.
- Dossou et al. (2022). [AfroLM: Self-Active Learning for 23 African Languages](https://arxiv.org/abs/2211.03263). arXiv.

---

*Related: [[Story Generation Index]], [[Story Generation Models]]*
