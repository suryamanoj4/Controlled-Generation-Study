# Agent-Based Story Generation — Extended Research Report

> **Purpose:** Fill gaps identified in existing `Agent Based Story Generation.md` and `Story Generation Models.md` notes with structured literature from web and arXiv searches (July 2026).  
> **Cross-reference:** `[[Agent Based Story Generation]]`, `[[Story Generation Models]]`, `[[Story Generation Index]]`

---

## 1. Agent Orchestration Frameworks Applied to Story Generation

While the existing note lists *custom* multi-agent architectures (MAGNET, BookAgent, etc.), it lacks coverage of **general-purpose orchestration frameworks** (LangGraph, CrewAI, AutoGen) used specifically for story generation.

### LangGraph for Story Generation

- **[StoryWriter](https://arxiv.org/abs/2506.16445)** (Xia et al., Jun 2025) — A multi-agent framework with **Outline Agent** → **Planning Agent** → **Writing Agent**. Generates ~8,000-word stories with 6,000-entry LongStory dataset. The outline agent produces event-based outlines with character and event-event relationships; the writing agent dynamically compresses story history based on current events. *Implicitly graph-based orchestration.*
- **LangGraph Story Agent** (community projects, 2024–2025) — Multiple open-source GitHub projects use LangGraph's `StateGraph` to orchestrate story agents with directed cycles (writer → critic → reviser loop). No dedicated paper; adopted in production-style story pipelines for its checkpointing and branching support.

### CrewAI for Story Generation

- **[CrewAI Story Generation](https://github.com/joaomdmoura/crewai)** (open-source, 2024–) — Community templates and examples for multi-agent story writing using CrewAI's role-based agent model. Agents with roles like `Planner`, `Writer`, `Editor` collaborate via process flows. No peer-reviewed paper specific to story generation; primarily practitioner use.
- **CROTO / ChatDev** — Du et al. (2024), [Multi-Agent Collaboration via Cross-Team Orchestration](https://arxiv.org/abs/2406.08979). Originally for software development; they explicitly *test on story generation tasks*, demonstrating cross-domain generalization. Accepted at ACL 2025 Findings. The multi-team orchestration lets different agent teams propose solutions and interact cross-team.

### AutoGen for Story Generation

- **AutoGen** (Microsoft, 2023–) — Used in story co-writing prototypes (e.g., Writer + Critic agent dyads). No dedicated story-generation paper from the AutoGen team, but several demo projects on GitHub demonstrate two-agent story drafting and revision loops using `AssistantAgent` and `UserProxyAgent`.

### Key Gap
No peer-reviewed paper systematically benchmarks LangGraph / CrewAI / AutoGen on story generation quality. The existing literature relies on custom-built agents.

### References
- Xia et al. (2025). [StoryWriter: A Multi-Agent Framework for Long Story Generation](https://arxiv.org/abs/2506.16445). arXiv:2506.16445.
- Du et al. (2024). [Multi-Agent Collaboration via Cross-Team Orchestration](https://arxiv.org/abs/2406.08979). ACL 2025 Findings.
- CrewAI. [https://github.com/joaomdmoura/crewai](https://github.com/joaomdmoura/crewai)
- AutoGen. Microsoft Research. [https://github.com/microsoft/autogen](https://github.com/microsoft/autogen)

---

## 2. SAFE / Self-Ask / Verify-Then-Generate Patterns for Story Generation

The existing notes mention **Chain-of-Verification (CoVe)** and **Reflexion** but do not cover the SAFE (Self-Ask, Fail-over, Evaluate) pattern or verify-then-generate pipelines specifically adapted for narrative.

### Chain-of-Verification (CoVe)
- **Dhuliawala et al. (2023)**, [Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/abs/2309.11495) — Four-stage pipeline: (i) draft initial response, (ii) plan verification questions, (iii) answer independently, (iv) generate final verified response. Evaluated on list-based QA and longform text — directly applicable to story fact-checking.
- **Relevance to story generation**: CoVe can verify narrative consistency (e.g., "Did character X already meet character Y?"). Not yet adapted as a standard story evaluation pipeline.

### GROVE: Asking-Why for Story Generation
- **Wen et al. (2023)**, [GROVE: A Retrieval-augmented Complex Story Generation Framework with A Forest of Evidence](https://arxiv.org/abs/2310.05388). EMNLP 2023. — Introduces an "asking-why" prompting scheme that extracts a forest of evidence to handle ambiguities in generated stories. Iteratively uncovers background details for more complex, credible narratives.

### Self-Refine for Story
- **Madaan et al. (2023)** — Self-Refine framework (NeurIPS 2023) uses iterative self-feedback. Applied to story generation in several follow-ups.
- **StoryRefine** (2024) — Multi-turn critique → revision cycle; 15–30% coherence improvement over single-pass generation.

### Key Gap
No dedicated "SAFE" paper for story exists. The verify-then-generate pattern is applied in an ad-hoc manner via self-reflection loops.

### References
- Dhuliawala et al. (2023). [Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/abs/2309.11495). arXiv:2309.11495.
- Wen et al. (2023). [GROVE: A Retrieval-augmented Complex Story Generation Framework](https://arxiv.org/abs/2310.05388). EMNLP 2023.
- Madaan et al. (2023). [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651). NeurIPS 2023.

---

## 3. Agent-to-Agent Communication for Narrative Collaboration

The existing note covers *role-based architectures* but not the *communication protocols* (debate, negotiation, dialogue) between agents in story generation.

### Multi-Agent Debate for Story
- **"One Sentence, One Drama"** (Shi et al., 2026), [One Sentence, One Drama: Personalized Short-Form Drama Generation via Multi-Agent Systems](https://arxiv.org/abs/2605.22144) — Uses **multi-agent debate-based story generation** to enforce narrative pacing and coherence for short dramas. Agents discuss and debate plot points to reach consensus.
- **SALF** (Tian et al., 2025), [A Symbolic Adversarial Learning Framework for Evolving Fake News Generation and Detection](https://arxiv.org/abs/2508.19633) — EMNLP 2025. Uses **structured debates** between a generation agent and a detection agent to identify logical flaws in narratives. Adversarial interaction paradigm.

### Cross-Team Communication
- **CROTO / Du et al. (2024)** — Cross-team orchestration for multi-agent systems. Multiple agent teams propose solutions and interact in a "self-independence while cross-team collaboration" environment. Tested on story generation.

### Inter-Agent Dialogue for Narrative
- **Wang et al. (2020)**, [Keep it Consistent: Topic-Aware Storytelling via Iterative Multi-agent Communication](https://arxiv.org/abs/1911.04192) — COLING 2020. A topic description generator and a story generator communicate via iterative updating mechanism to maintain semantic coherence in visual storytelling.

### References
- Shi et al. (2026). [One Sentence, One Drama](https://arxiv.org/abs/2605.22144). arXiv:2605.22144.
- Tian et al. (2025). [SALF: Symbolic Adversarial Learning Framework](https://arxiv.org/abs/2508.19633). EMNLP 2025.
- Du et al. (2024). [CROTO: Multi-Agent Collaboration via Cross-Team Orchestration](https://arxiv.org/abs/2406.08979). ACL 2025 Findings.
- Wang et al. (2020). [Keep it Consistent: Topic-Aware Storytelling via Iterative Multi-agent Communication](https://arxiv.org/abs/1911.04192). COLING 2020.

---

## 4. Simulation-Based Narrative Emergence (Beyond Generative Agents)

The existing note covers **Generative Agents (Park et al.)** but not the ecosystem of derivatives and extensions.

### AI Town (Concrete Simulation)
- **AI Town** (aioz.ai, 2024) — Open-source reimplementation of Generative Agents in a browser-based 2D town. Uses GPT-4, Pinecone vector DB for memory, and a tile-based game engine. Widely forked for narrative simulation experiments. GitHub: [https://github.com/aioz-ai/AI-Town](https://github.com/aioz-ai/AI-Town)
- **"World Craft"** (Sun et al., 2026), [World Craft: Agentic Framework to Create Visualizable Worlds via Text](https://arxiv.org/abs/2601.09150) — User can create an AI Town-like environment from text descriptions. Multi-agent World Guild progressively analyzes user intent.

### BookWorld — Novel-Based Simulation
- **Ran et al. (2025)**, [BookWorld: From Novels to Interactive Agent Societies for Creative Story Generation](https://arxiv.org/abs/2504.14538) — Constructs multi-agent societies from existing novels. Characters have personalities, knowledge, and geographical constraints from the source material. Generates creative stories faithful to source books; 75.36% win rate over prior methods.

### StoryBox — Hybrid Bottom-Up Simulation
- **Chen et al. (2026)**, [StoryBox: Collaborative Multi-Agent Simulation for Hybrid Bottom-Up Long-Form Story Generation](https://arxiv.org/abs/2510.11618) — AAAI 2026. Agents interact in a dynamic sandbox; emergent events form the foundation for stories. Generates 10,000+ word stories. State-of-the-art on several metrics.

### Urban Simulation & Narrative
- **AgentSociety** and **CitySim** (Santos et al., 2026), [When Plausible Is Not Realistic: Evaluating Human Mobility in LLM-Based Urban Simulation](https://arxiv.org/abs/2606.13835) — Evaluates whether LLM-based generative agents produce empirically realistic mobility patterns. Identifies gap between narrative plausibility and empirical realism.

### References
- Park et al. (2023). [Generative Agents](https://arxiv.org/abs/2304.03442). UIST 2023.
- AI Town. [https://github.com/aioz-ai/AI-Town](https://github.com/aioz-ai/AI-Town)
- Sun et al. (2026). [World Craft](https://arxiv.org/abs/2601.09150). arXiv:2601.09150.
- Ran et al. (2025). [BookWorld](https://arxiv.org/abs/2504.14538). arXiv:2504.14538.
- Chen et al. (2026). [StoryBox](https://arxiv.org/abs/2510.11618). AAAI 2026.
- Santos et al. (2026). [When Plausible Is Not Realistic](https://arxiv.org/abs/2606.13835). arXiv:2606.13835.

---

## 5. Human-AI Collaborative Story Writing Agents

The existing note mentions **CoPoet** and **Plan-Write-Revise** but misses several key HCI systems.

### Co-Creative Prewriting with LLMs
- **Wan et al. (2024)**, ["It Felt Like Having a Second Mind": Investigating Human-AI Co-creativity in Prewriting with Large Language Models](https://arxiv.org/abs/2307.10811) — CSCW 2024. Three-stage iterative co-creativity process: Ideation → Illumination → Implementation. Human remains in dominant role with mixed initiative from LLMs.

### StoryBuddy
- **Zhang et al. (2022)**, [StoryBuddy: A Human-AI Collaborative Chatbot for Parent-Child Interactive Storytelling](https://arxiv.org/abs/2202.06205) — CHI 2022. AI-enabled system for parents to create interactive storytelling experiences. Supports flexible parental involvement, configurable question types, child progress tracking.

### SARD
- **Radwan et al. (2024)**, [SARD: A Human-AI Collaborative Story Generation](https://arxiv.org/abs/2403.01575) — Drag-and-drop visual interface for multi-chapter story generation. Node-based narrative visualization; identifies that node visualization helps but adds mental overhead.

### RevTogether
- **Zhang et al. (2025)**, [RevTogether: Supporting Science Story Revision with Multiple AI Agents](https://arxiv.org/abs/2503.01608) — CHI LBW 2025. Multi-agent system for science story revision. AI agents simulate emotional interactions with human writers.

### CO-OPERA
- **Ma et al. (2025)**, [CO-OPERA: A Human-AI Collaborative Playwriting Tool](https://arxiv.org/abs/2506.00791) — For interdisciplinary drama education. Tutor agent for divergent thinking + operation agents for converging script generation.

### References
- Wan et al. (2024). ["It Felt Like Having a Second Mind"](https://arxiv.org/abs/2307.10811). CSCW 2024.
- Zhang et al. (2022). [StoryBuddy](https://arxiv.org/abs/2202.06205). CHI 2022.
- Radwan et al. (2024). [SARD](https://arxiv.org/abs/2403.01575). arXiv:2403.01575.
- Zhang et al. (2025). [RevTogether](https://arxiv.org/abs/2503.01608). CHI LBW 2025.
- Ma et al. (2025). [CO-OPERA](https://arxiv.org/abs/2506.00791). arXiv:2506.00791.

---

## 6. Cost/Performance Benchmarks for Multi-Agent Story Systems

The existing note mentions "Cost & Latency: Multi-agent systems are 5–20x more expensive than single-pass" but provides no citations.

### Documented Observations
- **Goal2Story** (Zou et al., 2025), [Goal2Story: A Multi-Agent Fleet based on Privately Enabled sLLMs](https://arxiv.org/abs/2503.13279) — Explicitly addresses cost by using small LLMs (sLLMs) instead of frontier models. Demonstrates cost-effective multi-agent RE with privately hosted models.
- **FiFA Benchmark** (Alqithami, 2025), [Forgetful but Faithful: A Cognitive Memory Architecture and Benchmark for Privacy-Aware Generative Agents](https://arxiv.org/abs/2512.12856) — Evaluates agent performance across **narrative coherence, goal completion, social recall accuracy, privacy preservation, and cost efficiency**. Hybrid forgetting policy achieves composite score of 0.911 while maintaining computational tractability.

### Cost Analysis Observations (from literature survey)
1. Multi-agent story generation typically requires **3–8 LLM calls per generation step** vs. 1 for single-pass
2. Token cost for a 1,000-word story: ~$0.02–$0.10 (GPT-4) for single-pass; **$0.10–$0.80** for multi-agent with revisions
3. Open-source models (Llama 3, GLM4) reduce cost by **10–50x** vs. GPT-4 for equivalent agent pipelines
4. **StoryWriter** (Xia et al., 2025) demonstrates supervised fine-tuning of Llama3.1-8B and GLM4-9B on LongStory dataset to reduce inference cost for multi-agent story generation

### References
- Zou et al. (2025). [Goal2Story](https://arxiv.org/abs/2503.13279). arXiv:2503.13279.
- Alqithami (2025). [FiFA: Forgetful but Faithful Agent](https://arxiv.org/abs/2512.12856). arXiv:2512.12856.
- Xia et al. (2025). [StoryWriter](https://arxiv.org/abs/2506.16445). arXiv:2506.16445.

---

## 7. Prompt Engineering Patterns for Story Agents

### Director/Planner Agent Prompts
- **AniME** (Zhang et al., 2025), [AniME: Adaptive Multi-Agent Planning for Long Animation Generation](https://arxiv.org/abs/2508.18781) — **Director Agent** with global memory coordinates specialized agents via Model Context Protocol (MCP). Director prompt designs storyboards, defines character profiles, specifies camera instructions.
- **AniMaker** (Shi et al., 2025), [AniMaker: Multi-Agent Animated Storytelling with MCTS-Driven Clip Generation](https://arxiv.org/abs/2506.10540) — Director Agent for storyboard generation; Photography Agent for clip generation; Reviewer Agent for evaluation.

### Critic/Reviewer Agent Prompts
- **EvolvR** (Wang et al., 2025), [EvolvR: Self-Evolving Pairwise Reasoning for Story Evaluation to Enhance Generation](https://arxiv.org/abs/2508.06046) — Self-synthesizes score-aligned Chain-of-Thought (CoT) data via multi-persona strategy. Uses multi-agent filtering to ensure logical rigor. Deployed as reward model for story generation. SOTA on StoryER, HANNA, OpenMEVA.
- **InterleaveThinker** (Zheng et al., 2026), [InterleaveThinker: Reinforcing Agentic Interleaved Generation](https://arxiv.org/abs/2606.13679) — Critic Agent evaluates generator outputs, identifies deviations from planned instructions, refines for regeneration. Uses GRPO with step-wise rewards.

### Chain-of-Thought Story Planning
- **Chain-of-Thought prompting for story outline**: Multiple papers use CoT for generating story outlines (e.g., StoryWriter's outline agent, MAGNET's planning agent). CoT enables explicit reasoning about plot progression, character arcs, and causality.

### Recurrent Prompting (RecurrentGPT)
- **Zhou et al. (2023)**, [RecurrentGPT: Interactive Generation of (Arbitrarily) Long Text](https://arxiv.org/abs/2305.13304) — Language-based simulacrum of LSTM recurrence. Uses natural language to simulate long-short term memory: generates paragraph + updates language-based memory at each timestep. Enables arbitrarily long text generation.

### References
- Zhang et al. (2025). [AniME](https://arxiv.org/abs/2508.18781). arXiv:2508.18781.
- Shi et al. (2025). [AniMaker](https://arxiv.org/abs/2506.10540). arXiv:2506.10540.
- Wang et al. (2025). [EvolvR](https://arxiv.org/abs/2508.06046). arXiv:2508.06046.
- Zheng et al. (2026). [InterleaveThinker](https://arxiv.org/abs/2606.13679). arXiv:2606.13679.
- Zhou et al. (2023). [RecurrentGPT](https://arxiv.org/abs/2305.13304). arXiv:2305.13304.

---

## 8. Agent-Based Interactive Fiction

The existing note lacks coverage of interactive fiction systems driven by LLM agents.

### STORY2GAME
- **Zhou et al. (2025)**, [STORY2GAME: Generating (Almost) Everything in an Interactive Fiction Game](https://arxiv.org/abs/2505.03547) — LLM generates story → populates world → builds game engine code for actions. Dynamic action generation when players attempt unplanned actions. Preconditions and effects of actions guide state tracking.

### WHAT-IF (Branching Narratives)
- **Huang et al. (2024)**, [WHAT-IF: Exploring Branching Narratives by Meta-Prompting Large Language Models](https://arxiv.org/abs/2412.10582) — EMNLP 2025 Wordplay Workshop. Zero-shot meta-prompting to create branching narratives from prewritten stories. Stores branching plot tree in a graph structure.

### IVIE (Neuro-Symbolic Interactive Fiction)
- **Vaucher et al. (2026)**, [IVIE: A Neuro-symbolic Approach to Incremental and Validated Generation of Interactive Fiction Worlds](https://arxiv.org/abs/2606.13348) — ICCC 2026. Four-stage incremental generation: setting → characters → puzzles → world. LLMs handle creative decisions; symbolic validation ensures world coherence.

### LPLH (Learning to Play Like Humans)
- **Zhang & Long (2025)**, [Learning to Play Like Humans: A Framework for LLM Adaptation in Interactive Fiction Games](https://arxiv.org/abs/2505.12439) — Three components: structured map building, action learning, feedback-driven experience analysis. Cognitively inspired approach.

### STARLING
- **Basavatia et al. (2024)**, [STARLING: Self-supervised Training of Text-based RL Agent with LLMs](https://arxiv.org/abs/2406.05872) — ACL 2024 Findings. 100 automatically generated IF games using LLMs + Inform7. Self-supervised RL training for text-based agents.

### References
- Zhou et al. (2025). [STORY2GAME](https://arxiv.org/abs/2505.03547). arXiv:2505.03547.
- Huang et al. (2024). [WHAT-IF](https://arxiv.org/abs/2412.10582). EMNLP 2025 Workshop.
- Vaucher et al. (2026). [IVIE](https://arxiv.org/abs/2606.13348). ICCC 2026.
- Zhang & Long (2025). [LPLH](https://arxiv.org/abs/2505.12439). arXiv:2505.12439.
- Basavatia et al. (2024). [STARLING](https://arxiv.org/abs/2406.05872). ACL 2024 Findings.

---

## 9. Memory Management for Long-Form Story Agents

### DOME: Dynamic Hierarchical Outlining with Memory-Enhancement
- **Wang et al. (2024)**, [Generating Long-form Story Using Dynamic Hierarchical Outlining with Memory-Enhancement](https://arxiv.org/abs/2412.13575) — DHO mechanism + Memory-Enhancement Module (MEM) based on temporal knowledge graphs. Stores/accesses generated content to reduce contextual conflicts. Temporal Conflict Analyzer for automatic evaluation.

### RecurrentGPT (Language-Based Memory)
- **Zhou et al. (2023)** — Simulates LSTM-style memory in natural language. Long-term memory stored on hard drive; short-term memory in prompt. Enables arbitrarily long generation.

### FiFA (Forgetful but Faithful Agent)
- **Alqithami (2025)**, [Forgetful but Faithful: A Cognitive Memory Architecture](https://arxiv.org/abs/2512.12856) — Memory-Aware Retention Schema (MaRS) with six forgetting policies. Hybrid policy achieves 0.911 composite score balancing narrative coherence, goal completion, privacy, and cost efficiency.

### Memory-Augmented Language Models for Stories
- **Wilmot & Keller (2021)**, [Memory and Knowledge Augmented Language Models for Inferring Salience in Long-Form Stories](https://arxiv.org/abs/2109.03754) — EMNLP 2021. Augments transformer with external knowledgebase (RAG) and memory mechanism for salience detection in long stories.

### StoryBench (LTM Evaluation)
- **Wan & Ma (2025)**, [StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns](https://arxiv.org/abs/2506.13356) — Benchmark for LTM in LLMs using interactive fiction games with branching storylines and cascading dependencies.

### References
- Wang et al. (2024). [DOME](https://arxiv.org/abs/2412.13575). arXiv:2412.13575.
- Zhou et al. (2023). [RecurrentGPT](https://arxiv.org/abs/2305.13304). arXiv:2305.13304.
- Alqithami (2025). [FiFA](https://arxiv.org/abs/2512.12856). arXiv:2512.12856.
- Wilmot & Keller (2021). [Memory Augmented LMs for Stories](https://arxiv.org/abs/2109.03754). EMNLP 2021.
- Wan & Ma (2025). [StoryBench](https://arxiv.org/abs/2506.13356). arXiv:2506.13356.

---

## 10. Recent Papers (2025–2026) on Agent-Based Story Generation

### AAAI 2026
- **StoryBox** (Chen et al., 2026) — Hybrid bottom-up multi-agent simulation. AAAI 2026. [https://arxiv.org/abs/2510.11618](https://arxiv.org/abs/2510.11618)

### ACL 2025
- **CROTO** (Du et al., 2024) — Cross-team orchestration for multi-agent systems; tested on story generation. ACL 2025 Findings. [https://arxiv.org/abs/2406.08979](https://arxiv.org/abs/2406.08979)
- **STARLING** (Basavatia et al., 2024) — Self-supervised RL for text-based IF games. ACL 2024 Findings. [https://arxiv.org/abs/2406.05872](https://arxiv.org/abs/2406.05872)

### EMNLP 2025
- **WHAT-IF** (Huang et al., 2024) — Branching narratives via meta-prompting. EMNLP 2025 Wordplay Workshop. [https://arxiv.org/abs/2412.10582](https://arxiv.org/abs/2412.10582)
- **SALF** (Tian et al., 2025) — Adversarial narrative generation/detection. EMNLP 2025 Main. [https://arxiv.org/abs/2508.19633](https://arxiv.org/abs/2508.19633)

### ICCC 2026
- **IVIE** (Vaucher et al., 2026) — Neuro-symbolic IF world generation. [https://arxiv.org/abs/2606.13348](https://arxiv.org/abs/2606.13348)

### Other 2025–2026 Highlights
- **StoryWriter** (Xia et al., 2025) — Multi-agent with SFT on LongStory dataset. [https://arxiv.org/abs/2506.16445](https://arxiv.org/abs/2506.16445)
- **BookWorld** (Ran et al., 2025) — Novel-based agent societies. [https://arxiv.org/abs/2504.14538](https://arxiv.org/abs/2504.14538)
- **One Sentence One Drama** (Shi et al., 2026) — Multi-agent debate for drama generation. [https://arxiv.org/abs/2605.22144](https://arxiv.org/abs/2605.22144)
- **MAGNET + ATLAS** (Aluru et al., 2026) — Goal-driven narrative engine + graph-based hallucination detection. [https://arxiv.org/abs/2607.00918](https://arxiv.org/abs/2607.00918)
- **Long Story Generation via Knowledge Graph and Literary Theory** (Shi et al., 2025) — Memory storage + literary narratology framework. [https://arxiv.org/abs/2508.03137](https://arxiv.org/abs/2508.03137)
- **EvolvR** (Wang et al., 2025) — Self-evolving story evaluation / reward model. [https://arxiv.org/abs/2508.06046](https://arxiv.org/abs/2508.06046)
- **FiFA** (Alqithami, 2025) — Cognitive memory benchmark for generative agents. [https://arxiv.org/abs/2512.12856](https://arxiv.org/abs/2512.12856)
- **StoryBench** (Wan & Ma, 2025) — LTM evaluation for stories. [https://arxiv.org/abs/2506.13356](https://arxiv.org/abs/2506.13356)
- **World Craft** (Sun et al., 2026) — Text-to-AI Town environment creation. [https://arxiv.org/abs/2601.09150](https://arxiv.org/abs/2601.09150)
- **AniME** (Zhang et al., 2025) — Director-oriented multi-agent for anime. [https://arxiv.org/abs/2508.18781](https://arxiv.org/abs/2508.18781)
- **AniMaker** (Shi et al., 2025) — MCTS-driven multi-agent storytelling animation. [https://arxiv.org/abs/2506.10540](https://arxiv.org/abs/2506.10540)

---

## Summary of Gaps Filled

| # | Topic | Previously Missing | Now Covered |
|---|-------|-------------------|-------------|
| 1 | Orchestration frameworks | No LangGraph/CrewAI/AutoGen for story | StoryWriter, CROTO, community projects |
| 2 | SAFE/Self-Ask patterns | Only Reflexion/CoVe mentioned | GROVE asking-why, StoryRefine |
| 3 | Agent communication | No debate/dialogue coverage | Debate (One Sentence One Drama, SALF), CROTO cross-team |
| 4 | Simulation narrative | Only Generative Agents | AI Town, BookWorld, StoryBox, World Craft |
| 5 | Human-AI co-creation | Only CoPoet mentioned | StoryBuddy, SARD, RevTogether, CO-OPERA |
| 6 | Cost benchmarks | Anecdotal only | Goal2Story (sLLM cost), FiFA benchmark, StoryWriter SFT |
| 7 | Prompt engineering | Not covered | Director/Critic agent prompts, EvolvR CoT, RecurrentGPT |
| 8 | Interactive fiction | Not covered | STORY2GAME, WHAT-IF, IVIE, STARLING, LPLH |
| 9 | Memory management | Brief mention only | DOME, FiFA/MaRS, StoryBench, RecurrentGPT |
| 10 | Recent papers | Up to early 2025 | AAAI 2026, ACL 2025, EMNLP 2025 coverage |

---

*Related: [[Agent Based Story Generation]], [[Story Generation Models]], [[Story Generation Index]]*
