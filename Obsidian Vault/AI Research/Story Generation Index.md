# Story Generation Index

## Core Topics

- [[Story Generation Models]] — Neural architectures, key papers, datasets, evaluations, controlled generation, diffusion/SSM models, planning algorithms, RAG, self-play, and hierarchical approaches
- [[Low Resource Story Generation]] — Cross-lingual transfer, multilingual models, prompt engineering, few-shot learning, LLM comparisons, active learning, cultural metrics, story grammar transfer, benchmarks, African/SE Asian languages
- [[Agent Based Story Generation]] — Multi-agent systems, orchestration frameworks, human-AI co-creation, simulation-based narrative, agent communication, self-optimization, memory architectures, interactive fiction, cost benchmarks
- [[Agent Story Generation — Extended Research Report]] — Detailed supplementary findings on agent-based story generation
- [[Style Adapter Steering — Model-Agnostic Narrative Style Control]] — Research proposal: self-aligning, pluggable style adapters for inference-time author-style control (no fine-tuning, no RAG); attribute-complexity & language-representation axes; Hindi/Telugu author-style replication as the low-resource example
- [[SAS E1 — Complexity Study Design]] — Experiment 1 design (v1.7): complexity ladder vs. adapter necessity, probes/metrics/gates, unified EN + Indic Lean run, dataset licensing
- [[SAS E1 — Prior Work Research (Steering Complexity)]] — Verified prior art on steering-representation complexity, evaluation protocols, and design implications for E1
- [[SAS E1 — Results (Lean Run)]] — Lean run results (2026-10-18; fable rebuilt 2026-10-24; validity audit 2026-10-26): representation ladder confirmed (H: L1 0.05 < L2 0.32 < L3 0.85); Indic behavioral results withdrawn (no in-language generations); EN style needs judge validation

## Extended Topics Covered in This Vault

- **Story Generation Models**: Mamba/SSM, Diffusion-LM, FUDGE/PPLM/GeDi controlled generation, RecurrentGPT, DOC, STORYGENERATOR, narrative schemas, HTN/LLM planning, RAG story gen, self-play/synthetic data, hierarchical beyond 2022
- **Low-Resource**: Cultural prompting strategies, 5-shot target-language optimal, mBART > mT5 > LLaMA comparisons, active learning (AfroLM 14× reduction), TALES taxonomy (88% misrepresentation), templated localization, African (517 languages via Cheetah), SE Asian (zero story gen papers for 6/8 languages)
- **Agent-Based**: LangGraph/CrewAI/AutoGen story pipelines, StoryWriter/BookAgent/StoryBox, AI Town/BookWorld simulation, multi-agent debate, StoryBuddy/SARD/CO-OPERA co-creation, Goal2Story cost optimization, EvolvR reward models, STORY2GAME interactive fiction, DOME memory, FiFA benchmark

## Related Topics

- Large Language Models for Creative Writing
- Narrative Understanding and Coherence
- Evaluation Metrics for Open-Ended Generation
- State-Space Models (Mamba) for Long-Form Text
- Diffusion Models for Controllable Text Generation

---

## Quick Reference

| Topic | Key Architecture | Key Challenge |
|-------|-----------------|---------------|
| General Story Gen | Transformer (GPT, LLaMA), Hierarchical (Plan→Write), Diffusion/SSM | Long-form coherence, plot consistency |
| Low-Resource Story Gen | Multilingual Encoder-Decoder (mT5, mBART), PEFT (LoRA) | Data scarcity, cultural preservation |
| Agent-Based Story Gen | Multi-agent teams (Director/Writer/Critic), orchestration frameworks | Self-optimization loops, evaluation, cost |

---

*Created: 2026-07-14 | Extended: 2026-07-14*
