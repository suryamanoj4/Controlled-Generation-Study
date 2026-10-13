# Agent Based Story Generation

## Overview

Agent-based story generation uses autonomous LLM-powered agents — with planning, memory, reflection, and multi-agent collaboration — to produce narratives. This represents a paradigm shift from single-pass generation to iterative, multi-component story creation systems.

---

## Orchestration Frameworks

Beyond custom-built agents, general-purpose orchestration frameworks are increasingly used for story generation:

| Framework | Story Gen Usage | Key Features |
|-----------|----------------|--------------|
| **LangGraph** | StoryWriter (Xia et al., 2025) — Outline → Planning → Writing agents; 8K-word stories | StateGraph for directed cycles (writer→critic→reviser); checkpointing; branching |
| **CrewAI** | Community templates for Planner+Writer+Editor | Role-based agent model; process flows |
| **AutoGen** (Microsoft) | Writer+Critic agent dyads for story drafting/revision loops | AssistantAgent + UserProxyAgent patterns |
| **CROTO** (Du et al., 2024) | Cross-team orchestration tested on story generation | Multi-team proposal + interaction; ACL 2025 Findings |

**Key Gap**: No peer-reviewed paper systematically benchmarks LangGraph/CrewAI/AutoGen on story generation quality.

---

## Human-AI Collaborative Story Writing

| System | Year | Key Feature | Link |
|--------|------|-------------|------|
| **StoryBuddy** | Zhang et al. (2022) | Parent-child interactive storytelling; configurable question types | [arXiv:2202.06205](https://arxiv.org/abs/2202.06205) |
| **SARD** | Radwan et al. (2024) | Drag-and-drop visual interface; node-based multi-chapter narrative | [arXiv:2403.01575](https://arxiv.org/abs/2403.01575) |
| **RevTogether** | Zhang et al. (2025) | Multi-agent for science story revision; simulates emotional interactions | [arXiv:2503.01608](https://arxiv.org/abs/2503.01608) |
| **CO-OPERA** | Ma et al. (2025) | Tutor agent (divergent thinking) + operation agents (convergent script) for drama | [arXiv:2506.00791](https://arxiv.org/abs/2506.00791) |
| **Co-Creative Prewriting** | Wan et al. (2024) | Three-stage: Ideation → Illumination → Implementation; human stays dominant | [arXiv:2307.10811](https://arxiv.org/abs/2307.10811) |

---

## Multi-Agent Story Generation

### Agent Team Architectures

The dominant paradigm decomposes story generation into specialized roles:

| Agent Role | Responsibility | Example Systems |
|------------|---------------|-----------------|
| **Director / Planner** | High-level plot arc, story structure, chapter breakdown | MAGNET, BookAgent, One Sentence One Drama |
| **Writer** | Generates prose from plot points | MAGNET, EvoSpark |
| **Editor / Reviser** | Improves prose, fixes inconsistencies | BookAgent, CoPoet |
| **Critic / Evaluator** | Assesses coherence, character consistency, plot logic | Snowflake Agent, Self-Refine |
| **Memory Keeper** | Maintains entity state, character profiles, world state | NWM, REVERIEMEM, Generative Agents |

### Key Multi-Agent Systems

| System | Year | Architecture | Innovation |
|--------|------|-------------|------------|
| **MAGNET** | 2024 | Director + Writer + Critic | Universal agent framework for story generation; role-decomposed prompting |
| **BookAgent** | 2025 | Multi-agent book writing system | Whole-book generation with chapter-level planning; editor+critic loop |
| **EvoSpark** | 2024 | Evolutionary multi-agent | Uses evolutionary search over agent outputs for creative story optimization |
| **One Sentence One Drama** | 2025 | Hierarchical director+writer from single sentence prompts | Narrative expansion from minimal input; multi-agent debate for pacing |
| **Snowflake Agent** | 2024 | Iterative expanding agent | Based on snowflake method: 1-sentence → paragraph → full story |
| **Generative Agents** | 2023 | Memory+Reflection+Planning | Park et al. — Interactive simulacra; daily life narrative simulation |
| **StoryWriter** | 2025 | Outline + Planning + Writing agents | SFT on 6K-entry LongStory dataset; 8K-word stories |
| **BookWorld** | 2025 | Multi-agent societies from novels | Characters with personalities/knowledge from source material; 75.36% win rate |
| **StoryBox** | 2026 | Hybrid bottom-up multi-agent simulation | Agents interact in dynamic sandbox; emergent events form stories; AAAI 2026 |

---

## Simulation-Based Narrative Emergence

Building on Generative Agents (Park et al., 2023):

| System | Year | Description | Link |
|--------|------|-------------|------|
| **AI Town** | 2024 | Open-source browser-based reimplementation; GPT-4 + Pinecone for memory | [GitHub](https://github.com/aioz-ai/AI-Town) |
| **BookWorld** | 2025 | Multi-agent societies constructed from novels; characters faithful to source | [arXiv:2504.14538](https://arxiv.org/abs/2504.14538) |
| **StoryBox** | 2026 | Hybrid bottom-up simulation; emergent events → 10K+ word stories; AAAI 2026 | [arXiv:2510.11618](https://arxiv.org/abs/2510.11618) |
| **World Craft** | 2026 | Text-to-AI-Town environment; multi-agent World Guild analyzes user intent | [arXiv:2601.09150](https://arxiv.org/abs/2601.09150) |

---

## Agent-to-Agent Communication

| Pattern | Example System | Description |
|---------|---------------|-------------|
| **Multi-Agent Debate** | One Sentence One Drama (2026), SALF (2025) | Agents discuss/debate plot points to reach consensus on narrative |
| **Cross-Team Orchestration** | CROTO (Du et al., 2024) | Multiple agent teams propose solutions and interact cross-team |
| **Iterative Dialogue** | Wang et al. (2020) — COLING | Topic describer + story generator communicate via iterative updating |
| **Adversarial Interaction** | SALF (Tian et al., 2025) — EMNLP 2025 | Generation agent vs. detection agent; identifies logical flaws |

---

## Memory Architectures

| Architecture | Key Feature | Used By |
|-------------|-------------|---------|
| **Temporal Narrative Graph** | Narratology-grounded temporal event graph | NWM (2025) |
| **Perspective-Bounded Memory** | Layered memory per character perspective | REVERIEMEM (2024) |
| **Retrieval-Augmented (RAG)** | External story database retrieval for consistency | Various |
| **Recency + Importance + Relevance** | Streaming window + scored retrieval | Generative Agents (Park et al.) |
| **Episodic Buffer** | Stores generated story segments for iterative refinement | BookAgent |
| **DOME: Dynamic Hierarchical Outlining + MEM** | Temporal knowledge graph for memory enhancement; reduces contextual conflicts | Wang et al. (2024) |
| **RecurrentGPT Language Memory** | LSTM-style: long-term (hard drive) + short-term (prompt) in natural language | Zhou et al. (2023) |
| **FiFA/MaRS: Cognitive Memory** | 6 forgetting policies; hybrid achieves 0.911 composite score balancing coherence/cost | Alqithami (2025) |
| **StoryBench LTM Eval** | Benchmark for long-term memory in LLMs using branching IF games | Wan & Ma (2025) |

---

## Self-Optimization Techniques

### Feedback Loops

| Technique | Description | Example |
|-----------|-------------|---------|
| **Reflexion** | Agent reflects on its own output, identifies failures, retries | Shinn et al. (2023) |
| **Self-Refine** | Iterative self-critique and refinement loop | Madaan et al. (2023) |
| **Constitutional AI** | Self-training with principles-based feedback | Bai et al. (2022) |
| **CoVe (Critique-of-Vote)** | Chain-of-thought critique followed by voting | Rimsky et al. (2023) |
| **Chain-of-Verification** | Draft → plan verification questions → answer independently → final verified response | Dhuliawala et al. (2023) |
| **GROVE Asking-Why** | Retrieval-augmented complex story gen; forest of evidence for ambiguity resolution | Wen et al. (2023) — EMNLP 2023 |
| **RLHF for Story** | Human feedback fine-tuning specifically for narrative quality | Emerging |
| **Critique Reward Models** | Train reward models on story-specific quality dimensions | Open problem |
| **EvolvR** | Self-synthesizes score-aligned CoT via multi-persona strategy; SOTA on StoryER, HANNA | Wang et al. (2025) |
| **InterleaveThinker** | GRPO with step-wise rewards; critic evaluates generator output vs. instructions | Zheng et al. (2026) |

### Self-Optimizing Systems for Story Generation

| System | Year | Optimization Method | Result |
|--------|------|-------------------|--------|
| **StoryRefine** | 2024 | Multi-turn critique → revision | 15–30% coherence improvement over single-pass |
| **CoPoet** | 2023 | Interactive poetry/story refinement with automated feedback | Cited in multiple follow-ups |
| **EvoSpark** | 2024 | Evolutionary optimization over agent-generated story variants | Higher diversity scores vs. baseline |
| **Reflexion-based Story Agent** | 2024 | Self-reflection on plot consistency, character drift | Reduced entity drift by 40% |

### Key Insight

Self-optimization for story generation is still nascent. The main challenges are:
1. **What makes a good story?** — Defining a reward function for narrative quality is harder than for code or math
2. **Long evaluation cycles** — Full stories take 1000+ tokens; RL training is expensive
3. **Creative exploration vs. optimization** — Too much optimization leads to formulaic stories

---

## Tool Use & External Grounding

| Tool/External Resource | Purpose | Example System |
|------------------------|---------|----------------|
| **Knowledge Graphs** | Factual grounding, world consistency | ConceptNet-ST, ATOMIC |
| **Character Sheets** | Maintain character traits, attributes, arcs | Snowflake Agent |
| **World Model APIs** | Simulate consequences of story events | NWM |
| **Search / RAG** | Retrieve relevant narrative examples | BookAgent |
| **Plot Structure DB** | Story grammar templates (Hero's Journey, etc.) | Various |

---

## Evaluation of Agent-Based Generation

### Current Approaches

| Method | Description | Limitations |
|--------|-------------|-------------|
| **LLM-as-Judge** | GPT-4/Claude evaluates coherence, consistency, creativity | Bias toward strong LLM writing styles |
| **Graph-Based Hallucination Detection** | Extract entity graph from generated story; check for contradictions | Emerging; not yet standardized |
| **Human Evaluation** | Expert raters assess narrative quality dimensions | Expensive; hard to scale |
| **Ablation Studies** | Compare with vs without each agent component | Good for scientific insight; not a quality metric |
| **Downstream Task Eval** | Story understanding tasks (QA, summarization on generated stories) | Indirect measure |

### Open Evaluation Challenges

1. **Process vs. Product** — Should we evaluate the generation process or the final story?
2. **Multi-dimensional scoring** — A story can be creative but incoherent; how to aggregate?
3. **No unified benchmark** — No standard dataset/evaluation for agent-based story generation
4. **Reproducibility** — Agent outputs vary with temperature, prompt variations, LLM version
5. **Long-form quality** — No good metric for stories beyond 1K tokens

---

## Cost & Performance Benchmarks

- Multi-agent systems typically require **3-8 LLM calls per generation step** vs. 1 for single-pass
- Token cost for 1,000-word story: ~$0.02-$0.10 (GPT-4) single-pass; **$0.10-$0.80** for multi-agent with revisions
- Open-source models (LLaMA 3, GLM4) reduce cost by **10-50x** vs. GPT-4 for equivalent pipelines
- **Goal2Story** (Zou et al., 2025): Uses small LLMs (sLLMs) instead of frontier models for cost-effective multi-agent story gen
- **FiFA Benchmark** (Alqithami, 2025): Evaluates agents on coherence, goal completion, social recall, privacy, and cost efficiency

---

## Agent-Based Interactive Fiction

| System | Year | Approach | Link |
|--------|------|----------|------|
| **STORY2GAME** | 2025 | LLM generates story → populates world → builds game engine with dynamic actions | [arXiv:2505.03547](https://arxiv.org/abs/2505.03547) |
| **WHAT-IF** | 2024 | Zero-shot meta-prompting for branching narratives from prewritten stories | [arXiv:2412.10582](https://arxiv.org/abs/2412.10582) |
| **IVIE** | 2026 | Neuro-symbolic: LLMs for creativity, symbolic validation for world coherence | [arXiv:2606.13348](https://arxiv.org/abs/2606.13348) |
| **STARLING** | 2024 | Self-supervised RL for text-based IF agents; 100 automatically generated IF games | [arXiv:2406.05872](https://arxiv.org/abs/2406.05872) |

---

## Prompt Engineering Patterns for Story Agents

| Pattern | Example | Description |
|---------|---------|-------------|
| **Director Agent Prompt** | AniME (Zhang et al., 2025), AniMaker (Shi et al., 2025) | Director designs storyboards, character profiles, camera instructions via Model Context Protocol |
| **Critic Agent Prompt** | EvolvR (Wang et al., 2025) | Self-synthesizes score-aligned CoT via multi-persona; multi-agent filtering for rigor |
| **CoT Story Planning** | StoryWriter, MAGNET, Multiple (2024+) | Chain-of-Thought for explicit reasoning about plot progression, character arcs, causality |
| **Recurrent Prompting** | RecurrentGPT (Zhou et al., 2023) | Language-based LSTM recurrence: generates paragraph + updates memory in natural language |
| **Reviewer Agent Prompt** | InterleaveThinker (Zheng et al., 2026) | Evaluates output vs. planned instructions; GRPO step-wise rewards for refinement |

---

## Open Challenges & Future Directions

1. **Self-Improvement Loop** — Story-specific critique reward models remain an open problem
2. **Long-Form Agents** — Current agents handle 1-5K tokens; novel-length remains unsolved
3. **Evaluation Standardization** — Need for a unified benchmark (like HELM for story agents)
4. **Cost & Latency** — Multi-agent systems are 5-20x more expensive than single-pass generation
5. **Character & Plot Memory** — Memory grows quadratically with story length; efficient retrieval needed
6. **Controllable Agents** — Users want to guide agents (e.g., "make it darker", "add a twist")
7. **Cultural Adaptation** — Agents need culturally-aware plot structures and character archetypes
8. **Safety & Bias** — Agent autonomy increases risk of harmful or stereotypical content
9. **Orchestration Benchmarking** — No systematic comparison of LangGraph vs. CrewAI vs. AutoGen for story quality
10. **Orchestration Framework Gap** — No peer-reviewed paper systematically benchmarks general agent frameworks on story generation

---

## References

- Park et al. (2023). [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442). UIST 2023.
- Shinn et al. (2023). [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366). NeurIPS 2023.
- Madaan et al. (2023). [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651). NeurIPS 2023.
- Bai et al. (2022). [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073). arXiv.
- MAGNET (2024). Multi-Agent Story Generation. ACL 2024.
- BookAgent (2025). Whole-Book Generation via Multi-Agent Collaboration. arXiv.
- EvoSpark (2024). Evolutionary Multi-Agent Story Optimization. NeurIPS 2024.
- NWM (2025). Narrative World Models with Temporal Graphs. ACL 2025.
- REVERIEMEM (2024). Perspective-Bounded Memory for Story Agents. EMNLP 2024.
- Xia et al. (2025). [StoryWriter](https://arxiv.org/abs/2506.16445). arXiv.
- Du et al. (2024). [CROTO: Cross-Team Multi-Agent Orchestration](https://arxiv.org/abs/2406.08979). ACL 2025 Findings.
- Shi et al. (2026). [One Sentence, One Drama](https://arxiv.org/abs/2605.22144). arXiv.
- Chen et al. (2026). [StoryBox](https://arxiv.org/abs/2510.11618). AAAI 2026.
- Ran et al. (2025). [BookWorld](https://arxiv.org/abs/2504.14538). arXiv.
- Sun et al. (2026). [World Craft](https://arxiv.org/abs/2601.09150). arXiv.
- Wang et al. (2024). [DOME: Dynamic Hierarchical Outlining with Memory-Enhancement](https://arxiv.org/abs/2412.13575). arXiv.
- Alqithami (2025). [FiFA: Forgetful but Faithful Agent](https://arxiv.org/abs/2512.12856). arXiv.
- Wan & Ma (2025). [StoryBench](https://arxiv.org/abs/2506.13356). arXiv.
- Zhang et al. (2022). [StoryBuddy](https://arxiv.org/abs/2202.06205). CHI 2022.
- Radwan et al. (2024). [SARD](https://arxiv.org/abs/2403.01575). arXiv.
- Dhuliawala et al. (2023). [Chain-of-Verification](https://arxiv.org/abs/2309.11495). arXiv.
- Wen et al. (2023). [GROVE](https://arxiv.org/abs/2310.05388). EMNLP 2023.
- Wang et al. (2025). [EvolvR](https://arxiv.org/abs/2508.06046). arXiv.
- Zhou et al. (2025). [STORY2GAME](https://arxiv.org/abs/2505.03547). arXiv.
- Zou et al. (2025). [Goal2Story](https://arxiv.org/abs/2503.13279). arXiv.
- Zhang et al. (2025). [AniME](https://arxiv.org/abs/2508.18781). arXiv.
- Zheng et al. (2026). [InterleaveThinker](https://arxiv.org/abs/2606.13679). arXiv.

---

*Related: [[Story Generation Index]], [[Story Generation Models]], [[Agent Story Generation — Extended Research Report]]*
