# Report on Controlled Text Generation

---

## Abstract

This report examines six foundational approaches to controlled text generation and story generation: Hierarchical Neural Story Generation (Fan et al., 2018), Plan-and-Write (Yao et al., 2019), the Plug and Play Language Model (Dathathri et al., 2020), FUDGE (Yang & Klein, 2021), GeDi (Krause et al., 2021), and text diffusion models (Li et al., 2022; Han et al., 2023). Each approach represents a distinct paradigm for imposing structure and control over the language generation process. This analysis compares their architectural designs, control mechanisms, strengths, limitations, and situates them within the broader landscape of controllable natural language generation.

---

## 1. Previous Works

### 1.1 Hierarchical Neural Story Generation

**Paper**: Fan, Lewis, Dauphin — *Hierarchical Neural Story Generation* (ACL 2018)
**Link**: [https://arxiv.org/abs/1805.04833](https://arxiv.org/abs/1805.04833)

**Architecture**: The system decomposes story generation into a two-stage hierarchical process. A convolutional sequence-to-sequence (ConvS2S) model first generates a premise or prompt — a high-level topic sentence describing the story. A second model then conditions on this premise to produce the full narrative text. This separation of concerns allows the model to establish global narrative intent before committing to surface-level word choices.

**Key Mechanisms**:

- **Gated Multi-Head Self-Attention**: An improved attention mechanism for capturing long-range dependencies across the generated text.
- **Model Fusion**: The base pre-trained seq2seq model is fused with representations from the premise encoder, enabling the generated text to remain faithful to the original premise while maintaining fluency.

**Strengths**: The hierarchical decomposition mitigates the problem of topic drift by fixing a high-level premise before generation begins. The fusion mechanism allows the model to balance premise-adherence with linguistic fluency.

**Limitations**: The model's ability to maintain coherence degrades over very long generations. The premise remains static throughout generation, limiting the model's ability to adapt or revise narrative direction dynamically.

---

### 1.2 Plan and Write

**Paper**: Yao et al. — *Plan-And-Write: Towards Better Automatic Storytelling* (AAAI 2019)
**Link**: [https://arxiv.org/abs/1811.05701](https://arxiv.org/abs/1811.05701)

**Architecture**: This approach extends the hierarchical paradigm by introducing an explicit planning stage where a sequence of keywords (the "plot") is generated before the story text. The system evaluates two distinct planning strategies:

- **Static Planning**: The complete plot (keyword sequence) is generated in its entirety before any story text is written. The story generation model then conditions on the full plot, ensuring every keyword appears in the final narrative.
- **Dynamic Planning**: Plot keywords and story text are generated interleaved — the model plans one or a few steps ahead before writing the corresponding section, then plans the next segment based on what has been written.

**Key Mechanisms**: The plot acts as a content scaffold. By constraining the story to include specific keywords, the model enforces topical coverage and narrative progression. Each keyword serves as a checkpoint that the generation must hit, providing a coarse-grained outline for the story.

**Strengths**: The static approach guarantees plot fidelity — every planned keyword appears in the generated story. The dynamic approach provides more flexibility, allowing the model to adapt the plot based on the text it has already written.

**Limitations**: The model exhibits significant constraints:

- **Long-range coherence**: Generated stories struggle to maintain coherent narrative structure beyond short-to-medium lengths. The keyword-level plan provides insufficient guidance for complex narrative arcs.
- **Plot coupling**: Stories are tightly bound to the keyword sequence, resulting in mechanical or unnatural narratives. The model inserts keywords rigidly rather than weaving them organically into the narrative.
- **Generality**: The tight plot-to-text coupling limits the approach's applicability to natural, open-ended story generation, where plot and prose ideally co-evolve.

---

### 1.3 Plug and Play Language Model (PPLM)

**Paper**: Dathathri et al. — *Plug and Play Language Models: A Simple Approach to Controlled Text Generation* (ICLR 2020)
**Link**: [https://arxiv.org/abs/1912.02164](https://arxiv.org/abs/1912.02164)

**Architecture**: PPLM departs fundamentally from the hierarchical and planning-based paradigms. Rather than training or modifying the source language model, it introduces a separate, lightweight *attribute model* that steers generation at inference time. The core insight is formalised as a Bayesian decomposition:

$$p(x \mid a) \propto p(x) \cdot p(a \mid x)$$

where $p(x)$ is the base language model and $p(a \mid x)$ is the attribute model representing the probability that the generated text exhibits the desired attribute. By updating the language model's latent representations to increase $p(a \mid x)$, the system biases generation toward the target attribute without retraining the base model.

**Attribute Model Types**:

- **Bag-of-Words (BOW)**: A simple attribute model that assigns probability based on the presence or absence of designated keywords. Lightweight and interpretable.
- **Discriminator Classifiers**: Small neural classifiers trained to detect specific attributes (e.g., topic, sentiment) in partial sequences. These models are orders of magnitude smaller than the base language model.

**Control Mechanism**: The attribute model steers generation by modifying the history matrix $H(t)$ — the key-value cache maintained by the transformer language model. Every autoregressive step reads from $H(t)$ to predict the next token. PPLM performs gradient-based updates on $H(t)$ to shift the probability distribution toward tokens that increase $p(a \mid x)$. The magnitude of this update can be controlled, allowing fine-grained adjustment of attribute strength — conceptually similar to a dial rather than an on-off switch.

**Strengths**:

- **Modularity**: Attribute models can be attached or detached without modifying the base LM. Multiple attribute models can operate simultaneously.
- **Efficiency**: Attribute models are small relative to the base LM, requiring minimal additional compute.
- **Flexibility**: The steering strength is continuously adjustable, enabling nuanced control.

**Limitations**:

- **Fluency degradation**: Aggressive steering can degrade the fluency of generated text, as the updates to $H(t)$ may push the model away from natural language distributions.
- **Attribute model quality**: The quality of control is bounded by the accuracy of the attribute model. A poor discriminator leads to poor steering.
- **Inference overhead**: Gradient updates at each generation step add computational cost compared to unconstrained generation.

---

### 1.4 FUDGE: Future Discriminator for Generation

**Paper**: Yang & Klein — *FUDGE: Controlled Text Generation With Future Discriminators* (NAACL 2021)
**Link**: [https://arxiv.org/abs/2104.05218](https://arxiv.org/abs/2104.05218)

**Architecture**: FUDGE follows the same core Bayesian formulation introduced in PPLM — the posterior probability of a sequence given a desired attribute factorizes as the product of the base language model probability and the attribute probability. The fundamental difference lies in *implementation*. Where PPLM uses gradient ascent to update the last hidden state of the base model at every decoding step, FUDGE works directly on the output logits of the base model. The base model's next-token probabilities are renormalized — multiplied with — the attribute probabilities produced by a trained *future discriminator*, completely eliminating the need for gradient computation through the language model.

**Key Mechanisms**: The central component is a future discriminator that, given a partial sequence (a prefix of the text generated so far), predicts whether the *future* complete sequence will exhibit the target attribute. Unlike conventional discriminators that evaluate the text produced so far, FUDGE's discriminator is trained on full sequences to judge the attribute of the entire output that is yet to be generated. At each decoding step, the base model's probability distribution is adjusted by Bayes rule: the probability of each candidate continuation is reweighted by the discriminator's estimate of the attribute given the extended prefix. This produces the same plug-and-play behaviour as PPLM, but through a purely probabilistic renormalization of the output distribution.

**Strengths**:

- **Modularity**: Inherits PPLM's advantage of controlling a frozen language model without any retraining.
- **Expressiveness**: Structured and complex attributes can be expressed far more easily than with PPLM. Attributes that are not readily discriminable in PPLM's hidden-state space — such as the sentiment of the story's *ending*, or constraints on the future text as a whole — can be captured naturally, because the discriminator is trained to reason about future sequences.
- **Efficiency**: Since no per-step gradient updates are required, control is computationally cheaper than PPLM; the discriminator is simply evaluated and multiplied into the distribution at each step.
- **Global reach**: Because the discriminator scores the future completion, control can target long-range properties of the output rather than only the immediate next-token distribution.

**Limitations**:

- **Per-attribute training**: A separate future discriminator must be trained (on labeled full sequences) for each attribute of interest.
- **Local application**: Like PPLM, control is applied step-by-step during decoding and does not by itself guarantee global narrative structure or long-form coherence.
- **Discriminator quality bound**: The effectiveness of control remains bounded by the accuracy of the future discriminator.

---

### 1.5 GeDi: Generative Discriminator Guided Sequence Generation

**Paper**: Krause, Gotmare, McCann, Keskar, Joty, Socher, Rajani — *GeDi: Generative Discriminator Guided Sequence Generation* (ICLR 2021)
**Link**: [https://arxiv.org/abs/2009.06367](https://arxiv.org/abs/2009.06367)

**Architecture**: GeDi belongs to the same Bayesian renormalization family as FUDGE, but replaces the separately trained discriminator with a *class-conditioned language model* (CC-LM) — such as CTRL — which serves as a generative discriminator. The CC-LM is a small language model trained to generate sequences conditioned on class tokens: both the *desired* class and the *undesired* (negative) class. During decoding, the CC-LM computes the probability of the partial sequence under each class condition, and Bayes rule is applied to derive the posterior probability that the sequence belongs to the desired class. This posterior is then used to reweight the output distribution of the large autoregressive base LM at every generation step.

**Key Mechanisms**:

- **Class-conditioned generation**: The CC-LM provides $p(x \mid c)$ for the desired class $c$ and $p(x \mid \neg c)$ for the negative class.
- **Bayesian posterior**: The probability that a sequence belongs to the desired class is computed as
  $$p(c \mid x) = \frac{p(x \mid c)\,p(c)}{p(x \mid c)\,p(c) + p(x \mid \neg c)\,p(\neg c)}$$
  — a contrastive formulation that only a generative model (not a classifier) can provide directly.
- **Renormalization**: The base LM's token probabilities are multiplied by this posterior (raised to a controllable guidance strength) and renormalized. This is the same Bayes law used by PPLM and FUDGE, but the attribute probability is computed by a generative model rather than a classifier. The approach is conceptually a form of classifier-free guidance, contrasting desired and undesired class generations.
- **Class reduction at generation time**: Because the CC-LM conditions on the class token, the effective sample space of the desired class is sharply narrowed during generation, concentrating probability mass on continuations that stay within the class distribution.

**Strengths**:

- **Speed**: Approximately 30x faster than PPLM, since no gradient updates are computed through the base LM.
- **Stronger control signal**: The CC-LM produces coherent, fluent conditional sequences, so the derived posterior is far more informative and less noisy than the scalar output of a small classifier — the control signal itself is generative.
- **Flexible attributes**: Supports topic, sentiment, detoxification, and gender control; guidance strength is continuously adjustable via a parameter.
- **Modularity**: Inherits the plug-and-play property — the base LM remains frozen and unmodified.

**Limitations**:

- **CC-LM training**: A class-conditioned LM must be trained or fine-tuned for the attribute classes of interest; control is limited to attributes expressible as class conditions.
- **Local application**: Like FUDGE and PPLM, control is applied step-by-step during autoregressive decoding and does not by itself guarantee global narrative structure or long-form coherence.
- **Base LM constraint**: The output remains autoregressive — the continuous, whole-sequence control offered by diffusion models is unavailable.

---

### 1.6 Diffusion Models for Text Generation

**Motivation**: All approaches discussed so far operate on autoregressive language models, where text is inherently discrete — each token is selected independently from its own probability distribution, and control must be injected locally during decoding. In image and audio domains, diffusion models predict a complete continuous stream of data in one process, which makes modular, fine-grained guidance natural. Researchers therefore hypothesised that diffusion models would provide the modular control that autoregressive models failed to deliver, giving rise to a family of text diffusion models.

**Diffusion-LM** (Li, Thickstun, Gulrajani, Liang, Hashimoto — NeurIPS 2022)
**Link**: [https://arxiv.org/abs/2205.14217](https://arxiv.org/abs/2205.14217)

Diffusion-LM is the first diffusion model applied to text. It embeds tokens into a continuous latent space and trains a continuous diffusion model over these embeddings. Generation starts from noise and iteratively denoises the latent representation, which is then rounded back into discrete tokens. Because the model operates on a continuous latent, control can be injected through *guidance* at multiple granularities — prefixes, span infilling, bag-of-words constraints, and even syntactic structure — in a manner analogous to classifier guidance in image diffusion. Its principal drawbacks are slow inference (many denoising steps) and the fluency loss introduced by the embedding-rounding gap between the continuous latent and discrete tokens.

**SSD-LM** (Han, Kumar, Tsvetkov — ACL 2023)
**Link**: [https://arxiv.org/abs/2210.17432](https://arxiv.org/abs/2210.17432)

SSD-LM (Semi-autoregressive Simplex-based Diffusion Language Model) addresses the key weaknesses of Diffusion-LM. It performs diffusion directly on the probability *simplex* rather than on continuous embeddings, so the output of the diffusion process is naturally a valid token probability distribution — eliminating the embedding-rounding gap and improving fluency. It generates semi-autoregressively, producing a block of tokens at a time, which speeds up inference relative to full-sequence diffusion while retaining the diffusion machinery. Most importantly, SSD-LM supports *plug-and-play classifier guidance*: off-the-shelf classifiers can steer generation during the denoising process without any modification to the generative model — a form of control far more modular than the training-time control of the hierarchical approaches.

**Strengths** (diffusion paradigm):

- **Continuous, modular control**: Because generation happens in continuous space, guidance can be injected at arbitrary granularities — prefixes, spans, bag-of-words, syntactic constraints, and whole-sequence properties.
- **Plug-and-play steering**: Classifier guidance (as in SSD-LM) enables attribute control without retraining the generative model — comparable in modularity to PPLM/FUDGE/GeDi, but applied over the entire sequence rather than step-by-step.
- **Whole-sequence planning**: The denoising process can be conditioned to satisfy global constraints that stepwise autoregressive methods struggle with.

**Limitations** (diffusion paradigm):

- **Slow inference**: Iterative denoising requires many forward passes, making diffusion LMs substantially slower than autoregressive decoding.
- **Fluency gap**: Diffusion-generated text has historically lagged autoregressive models in fluency and coherence — a critical weakness for narrative generation, which demands sustained linguistic quality.
- **Discretisation complexity**: Bridging continuous diffusion and discrete text requires embedding-rounding (Diffusion-LM) or simplex parameterisation (SSD-LM), each adding architectural complexity and failure modes.

## 3. Comparative Analysis

### 3.1 Architectural Paradigms

| Approach | Core Paradigm | LM Modification | Control Signal | Granularity |
|----------|--------------|-----------------|----------------|-------------|
| Hierarchical Story Gen | Two-stage premise→text | Trained fusion model | Generated premise (text) | Story-level premise |
| Plan and Write | Explicit plot→text planning | Trained with plot conditioning | Keyword sequence (plot) | Token-level keyword constraints |
| PPLM | Latent-space steering (gradient ascent on hidden state) | No LM modification | Attribute model (classifier/BOW) | Continuous latent updates |
| FUDGE | Logit-space renormalization via future discriminator | No LM modification | Future discriminator (trained on full sequences) | Token-level Bayesian renormalization |
| GeDi | Logit-space renormalization via generative discriminator (CC-LM) | No LM modification | Class-conditioned LM (e.g., CTRL) | Token-level Bayesian renormalization |
| Diffusion (Diffusion-LM, SSD-LM) | Continuous latent/simplex diffusion with iterative denoising | No LM modification (guidance only) | Classifier guidance / whole-sequence conditioning | Continuous, whole-sequence |

### 3.2 Control Mechanisms

The six approaches represent a progression in control philosophy:

1. **Top-down control** (Hierarchical, Plan-and-Write): Structure is imposed before generation begins. The premise or plot acts as a fixed blueprint. This ensures global coherence but limits adaptability.

2. **Bottom-up steering** (PPLM, FUDGE, GeDi): Control emerges from local updates to the generation process. PPLM nudges token probabilities through gradient ascent on the hidden state, while FUDGE and GeDi refine this by renormalizing the output distribution directly with discriminator probabilities. All provide fine-grained control but none guarantees global narrative structure.

3. **The FUDGE refinement**: FUDGE addresses the two principal weaknesses of PPLM — the computational cost of per-step gradient updates and the difficulty of expressing structured attributes in hidden-state space. By replacing gradient ascent with Bayesian renormalization, it achieves cheaper control and can enforce long-range, structured attributes (e.g., the property of the future ending) that PPLM cannot readily express.

4. **The GeDi refinement**: GeDi replaces the classifier with a class-conditioned *generative* LM, deriving the attribute posterior by contrasting desired and undesired class generations. This yields a stronger, more coherent control signal (approximately 30x faster than PPLM) and supports attributes that are difficult to define for a standalone classifier.

5. **The diffusion paradigm**: Diffusion models abandon autoregressive decoding entirely, operating in continuous space where control is injected through guidance during the denoising process. This enables the most modular form of control — arbitrary granularity and whole-sequence properties — but at significant inference cost and with an historical fluency gap.

6. **Hybrid potential**: Neither paradigm inherently excludes the other — hierarchical planning could, in principle, be combined with any of the steering approaches (PPLM/FUDGE/GeDi) or diffusion guidance to achieve both global structure and local attribute control.

### 3.3 Key Limitations Across Approaches

| Approach | Long-range coherence | Naturalness | Flexibility | Scalability to long text | Computational overhead |
|----------|---------------------|-------------|-------------|--------------------------|------------------------|
| Hierarchical | Degrades with length | Moderate | Fixed premise | Limited | Training required |
| Plan-and-Write | Limited | Low (rigid plot coupling) | Static or interleaved plans | Limited | Training required |
| PPLM | Not addressed | Degrades under strong steering | Continuously adjustable | Theoretically unlimited | Inference-time gradient updates |
| FUDGE | Partially addressed (future discriminators score the eventual completion) | Less distortion than PPLM (no hidden-state perturbation) | Continuously adjustable; supports structured attributes | Theoretically unlimited | Inference-time discriminator evaluations (lighter than PPLM) |
| GeDi | Partially addressed via full-sequence class contrast | Generally high (generative control signal) | Adjustable guidance strength; class-based attributes | Theoretically unlimited | Inference-time CC-LM evaluations (approximately 30x faster than PPLM) |
| Diffusion (Diffusion-LM, SSD-LM) | Addressed via whole-sequence denoising | Fluency gap vs autoregressive LMs | Highly modular, continuous guidance | Constrained by slow inference | Very high (many denoising steps) |

---

## 4. Key Findings

1. **Hierarchical decomposition improves coherence but at a cost**: Separating premise generation from text generation reduces topic drift but introduces a rigid pipeline that cannot revise its high-level plan based on emergent narrative.

2. **Plot coupling is a double-edged sword**: Explicit keyword planning ensures topical coverage but produces mechanical narratives. The tight constraint between plot and text in Plan-and-Write limits naturalness, suggesting that effective story generation requires more flexible, latent planning representations.

3. **Inference-time control is more modular than training-time control**: PPLM's ability to steer a frozen LM via separate attribute models represents a significant architectural advantage — it decouples control from training, enabling dynamic composition of attribute models.

4. **Continuous control is preferable to discrete constraint**: PPLM's strength-adjustable steering (via gradient magnitude on $H(t)$) provides finer-grained control than the binary keyword-inclusion constraint used in Plan-and-Write.

5. **No single approach addresses all dimensions of control**: Hierarchical approaches provide global structure, PPLM provides local attribute control, and Plan-and-Write provides keyword adherence. A unified framework combining hierarchical planning with latent-space or logit-space steering remains an open direction.

6. **Logit-space control is cheaper and more expressive than hidden-state control**: FUDGE's direct renormalization of output probabilities avoids the per-step gradient updates required by PPLM. Because its discriminator is trained on future sequences, FUDGE can enforce structured and long-range attributes (such as the property of the story's ending) that are not readily distinguishable in PPLM's hidden-state space. This suggests the control interface — where the attribute signal is injected into the generation process — is as important as the attribute model itself.

7. **Generative discriminators yield stronger control signals than classifiers**: GeDi replaces the classifier with a class-conditioned language model, deriving the attribute posterior by contrasting desired and undesired class generations. Because the CC-LM produces coherent conditional sequences, the control signal is more informative and less noisy than a classifier's scalar output — control is approximately 30x faster than PPLM and text remains more fluent. This establishes a clear hierarchy in the steering family: gradient ascent (PPLM) → future classifier (FUDGE) → generative discriminator (GeDi).

8. **Continuous generation enables the most modular control, at a price**: Diffusion models (Diffusion-LM, SSD-LM) operate in continuous latent or simplex space, allowing guidance at arbitrary granularities — prefixes, spans, bag-of-words, syntax, and whole-sequence properties — through plug-and-play classifier guidance without modifying the generative model. However, iterative denoising makes inference slow, and generated text has historically lagged autoregressive models in fluency — a critical trade-off for narrative generation, where sustained linguistic quality matters as much as controllability.

---

## 5. Conclusion

This report has analysed six foundational approaches to controlled story generation, each representing a distinct paradigm for imposing structure on the language generation process. Hierarchical Neural Story Generation introduced the premise→text decomposition that underpins many subsequent systems. Plan-and-Write demonstrated explicit plot-level planning but revealed the limitations of rigid keyword constraints. PPLM introduced a modular, inference-time control mechanism that decouples attribute control from model training. FUDGE refined this line of work by replacing gradient ascent with future-discriminator-based renormalization of the output distribution — cheaper, and capable of expressing structured, long-range attributes. GeDi further strengthened the steering family by using class-conditioned generative discriminators, and diffusion models (Diffusion-LM, SSD-LM) abandoned autoregressive decoding altogether to enable continuous, whole-sequence guidance — at the cost of inference speed and fluency.

The analysis suggests that effective story generation requires hybrid approaches — combining the global coherence of hierarchical planning with the flexible, fine-grained control of latent-, logit-, or continuous-space steering. Subsequent work in agent-based story generation, memory-augmented architectures, and multi-agent frameworks builds on the foundations laid by these approaches, each of which continues to inform current research directions.

---

## References

1. Fan, A., Lewis, M., & Dauphin, Y. (2018). *Hierarchical Neural Story Generation*. Proceedings of ACL 2018. [https://arxiv.org/abs/1805.04833](https://arxiv.org/abs/1805.04833)

2. Yao, L., Peng, N., Weischedel, R., Knight, K., Zhao, D., & Yan, R. (2019). *Plan-And-Write: Towards Better Automatic Storytelling*. Proceedings of AAAI 2019. [https://arxiv.org/abs/1811.05701](https://arxiv.org/abs/1811.05701)

3. Dathathri, S., Madotto, A., Lan, J., Hung, J., Frank, E., Molino, P., Yosinski, J., & Liu, R. (2020). *Plug and Play Language Models: A Simple Approach to Controlled Text Generation*. Proceedings of ICLR 2020. [https://arxiv.org/abs/1912.02164](https://arxiv.org/abs/1912.02164)

4. Yang, K., & Klein, D. (2021). *FUDGE: Controlled Text Generation With Future Discriminators*. Proceedings of NAACL 2021. [https://arxiv.org/abs/2104.05218](https://arxiv.org/abs/2104.05218)

5. Krause, B., Gotmare, A. D., McCann, B., Keskar, N. S., Joty, S., Socher, R., & Rajani, N. (2021). *GeDi: Generative Discriminator Guided Sequence Generation*. Proceedings of ICLR 2021. [https://arxiv.org/abs/2009.06367](https://arxiv.org/abs/2009.06367)

6. Li, X., Thickstun, J., Gulrajani, I., Liang, P., & Hashimoto, T. (2022). *Diffusion-LM Improves Controllable Text Generation*. Proceedings of NeurIPS 2022. [https://arxiv.org/abs/2205.14217](https://arxiv.org/abs/2205.14217)

7. Han, X., Kumar, S., & Tsvetkov, Y. (2023). *SSD-LM: Semi-autoregressive Simplex-based Diffusion Language Model for Text Generation and Modular Control*. Proceedings of ACL 2023. [https://arxiv.org/abs/2210.17432](https://arxiv.org/abs/2210.17432)
