# Awesome Large Reasoning Model (LRM) Safety 🔥

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Auto Update](https://github.com/wonderNefelibata/Awesome-LRM-Safety/actions/workflows/arxiv-update.yml/badge.svg)

A curated list of **security and safety research** for Large Reasoning Models (LRMs) like DeepSeek-R1, OpenAI o1, and other cutting-edge models. Focused on identifying risks, mitigation strategies, and ethical implications.

---

## 📜 Table of Contents
- [Awesome Large Reasoning Model (LRM) Safety 🔥](#awesome-large-reasoning-model-lrm-safety-)
  - [📜 Table of Contents](#-table-of-contents)
  - [🚀 Motivation](#-motivation)
  - [🤖 Large Reasoning Models](#-large-reasoning-models)
    - [Open Source Models](#open-source-models)
    - [Close Source Models](#close-source-models)
  - [📰 Latest arXiv Papers (Auto-Updated)](#-latest-arxiv-papers-auto-updated)
  - [🔑 Key Safety Domains(coming soon)](#-key-safety-domainscoming-soon)
  - [🔖 Dataset \& Benchmark](#-dataset--benchmark)
    - [For Traditional LLM](#for-traditional-llm)
    - [For Advanced LRM](#for-advanced-lrm)
  - [📚 Survey](#-survey)
    - [LRM Related](#lrm-related)
    - [LRM Safety Related](#lrm-safety-related)
  - [🛠️ Projects \& Tools(coming soon)](#️-projects--toolscoming-soon)
    - [Model-Specific Resources(example)](#model-specific-resourcesexample)
    - [General Tools(coming soon)(example)](#general-toolscoming-soonexample)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)
  - [❓ FAQ](#-faq)
  - [🔗 References](#-references)

---

## 🚀 Motivation

Large Reasoning Models (LRMs) are revolutionizing AI capabilities in complex decision-making scenarios. However, their deployment raises critical safety concerns.

This repository aims to catalog research addressing these challenges and promote safer LRM development.

## 🤖 Large Reasoning Models

### Open Source Models
  

| Name | Organization | Date | Technic | Cold-Start | Aha Moment | Modality |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1 | DeepSeek | 2025/01/22 | GRPO | ✅   | ✅   | text-only |
| QwQ-32B | Qwen | 2025/03/06 | -   | -   | -   | text-only |

### Close Source Models
  

| Name | Organization | Date | Technic | Cold-Start | Aha Moment | Modality |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI-o1 | OpenAI | 2024/09/12 | -   | -   | -   | text,image |
| Gemini-2.0-Flash-Thinking | Google | 2025/01/21 | -   | -   | -   | text,image |
| Kimi-k1.5 | Moonshot | 2025/01/22 | -   | -   | -   | text,image |
| OpenAI-o3-mini | OpenAI | 2025/01/31 | -   | -   | -   | text,image |
| Grok-3 | xAI | 2025/02/19 | -   | -   | -   | text,image |
| Claude-3.7-Sonnet | Anthropic | 2025/02/24 | -   | -   | -   | text,image |
| Gemini-2.5-Pro | Google | 2025/03/25 | -   | -   | -   | text,image |

---

## 📰 Latest arXiv Papers (Auto-Updated)
It is updated every 12 hours, presenting the latest 20 relevant papers.And Earlier Papers can be found [here](./articles/README.md).


<!-- LATEST_PAPERS_START -->


| Date       | Title                                      | Authors           | Abstract Summary          |
|------------|--------------------------------------------|-------------------|---------------------------|
| 2026-02-06 | [InftyThink+: Effective and Efficient Infinite-Horizon Reasoning via Reinforcement Learning](http://arxiv.org/abs/2602.06960v1) | Yuchen Yan, Liang Jiang et al. | Large reasoning models achieve strong performance by scaling inference-time chain-of-thought, but this paradigm suffers from quadratic cost, context length limits, and degraded reasoning due to lost-in-the-middle effects. Iterative reasoning mitigates these issues by periodically summarizing intermediate thoughts, yet existing methods rely on supervised learning or fixed heuristics and fail to optimize when to summarize, what to preserve, and how to resume reasoning. We propose InftyThink+, an end-to-end reinforcement learning framework that optimizes the entire iterative reasoning trajectory, building on model-controlled iteration boundaries and explicit summarization. InftyThink+ adopts a two-stage training scheme with supervised cold-start followed by trajectory-level reinforcement learning, enabling the model to learn strategic summarization and continuation decisions. Experiments on DeepSeek-R1-Distill-Qwen-1.5B show that InftyThink+ improves accuracy by 21% on AIME24 and outperforms conventional long chain-of-thought reinforcement learning by a clear margin, while also generalizing better to out-of-distribution benchmarks. Moreover, InftyThink+ significantly reduces inference latency and accelerates reinforcement learning training, demonstrating improved reasoning efficiency alongside stronger performance. |
| 2026-02-06 | [Endogenous Resistance to Activation Steering in Language Models](http://arxiv.org/abs/2602.06941v1) | Alex McKenzie, Keenan Pepper et al. | Large language models can resist task-misaligned activation steering during inference, sometimes recovering mid-generation to produce improved responses even when steering remains active. We term this Endogenous Steering Resistance (ESR). Using sparse autoencoder (SAE) latents to steer model activations, we find that Llama-3.3-70B shows substantial ESR, while smaller models from the Llama-3 and Gemma-2 families exhibit the phenomenon less frequently. We identify 26 SAE latents that activate differentially during off-topic content and are causally linked to ESR in Llama-3.3-70B. Zero-ablating these latents reduces the multi-attempt rate by 25%, providing causal evidence for dedicated internal consistency-checking circuits. We demonstrate that ESR can be deliberately enhanced through both prompting and training: meta-prompts instructing the model to self-monitor increase the multi-attempt rate by 4x for Llama-3.3-70B, and fine-tuning on self-correction examples successfully induces ESR-like behavior in smaller models. These findings have dual implications: ESR could protect against adversarial manipulation but might also interfere with beneficial safety interventions that rely on activation steering. Understanding and controlling these resistance mechanisms is important for developing transparent and controllable AI systems. Code is available at github.com/agencyenterprise/endogenous-steering-resistance. |
| 2026-02-06 | [TamperBench: Systematically Stress-Testing LLM Safety Under Fine-Tuning and Tampering](http://arxiv.org/abs/2602.06911v1) | Saad Hossain, Tom Tseng et al. | As increasingly capable open-weight large language models (LLMs) are deployed, improving their tamper resistance against unsafe modifications, whether accidental or intentional, becomes critical to minimize risks. However, there is no standard approach to evaluate tamper resistance. Varied data sets, metrics, and tampering configurations make it difficult to compare safety, utility, and robustness across different models and defenses. To this end, we introduce TamperBench, the first unified framework to systematically evaluate the tamper resistance of LLMs. TamperBench (i) curates a repository of state-of-the-art weight-space fine-tuning attacks and latent-space representation attacks; (ii) enables realistic adversarial evaluation through systematic hyperparameter sweeps per attack-model pair; and (iii) provides both safety and utility evaluations. TamperBench requires minimal additional code to specify any fine-tuning configuration, alignment-stage defense method, and metric suite while ensuring end-to-end reproducibility. We use TamperBench to evaluate 21 open-weight LLMs, including defense-augmented variants, across nine tampering threats using standardized safety and capability metrics with hyperparameter sweeps per model-attack pair. This yields novel insights, including effects of post-training on tamper resistance, that jailbreak-tuning is typically the most severe attack, and that Triplet emerges as a leading alignment-stage defense. Code is available at: https://github.com/criticalml-uw/TamperBench |
| 2026-02-06 | [SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks](http://arxiv.org/abs/2602.06854v1) | Mingqian Feng, Xiaodong Liu et al. | Multi-turn jailbreaks capture the real threat model for safety-aligned chatbots, where single-turn attacks are merely a special case. Yet existing approaches break under exploration complexity and intent drift. We propose SEMA, a simple yet effective framework that trains a multi-turn attacker without relying on any existing strategies or external data. SEMA comprises two stages. Prefilling self-tuning enables usable rollouts by fine-tuning on non-refusal, well-structured, multi-turn adversarial prompts that are self-generated with a minimal prefix, thereby stabilizing subsequent learning. Reinforcement learning with intent-drift-aware reward trains the attacker to elicit valid multi-turn adversarial prompts while maintaining the same harmful objective. We anchor harmful intent in multi-turn jailbreaks via an intent-drift-aware reward that combines intent alignment, compliance risk, and level of detail. Our open-loop attack regime avoids dependence on victim feedback, unifies single- and multi-turn settings, and reduces exploration complexity. Across multiple datasets, victim models, and jailbreak judges, our method achieves state-of-the-art (SOTA) attack success rates (ASR), outperforming all single-turn baselines, manually scripted and template-driven multi-turn baselines, as well as our SFT (Supervised Fine-Tuning) and DPO (Direct Preference Optimization) variants. For instance, SEMA performs an average $80.1\%$ ASR@1 across three closed-source and open-source victim models on AdvBench, 33.9% over SOTA. The approach is compact, reproducible, and transfers across targets, providing a stronger and more realistic stress test for large language model (LLM) safety and enabling automatic redteaming to expose and localize failure modes. Our code is available at: https://github.com/fmmarkmq/SEMA. |
| 2026-02-06 | [Statistical-Based Metric Threshold Setting Method for Software Fault Prediction in Firmware Projects: An Industrial Experience](http://arxiv.org/abs/2602.06831v1) | Marco De Luca, Domenico Amalfitano et al. | Ensuring software quality in embedded firmware is critical, especially in safety-critical domains where compliance with functional safety standards (ISO 26262) requires strong guarantees of software reliability. While machine learning-based fault prediction models have demonstrated high accuracy, their lack of interpretability limits their adoption in industrial settings. Developers need actionable insights that can be directly employed in software quality assurance processes and guide defect mitigation strategies. In this paper, we present a structured process for defining context-specific software metric thresholds suitable for integration into fault detection workflows in industrial settings. Our approach supports cross-project fault prediction by deriving thresholds from one set of projects and applying them to independently developed firmware, thereby enabling reuse across similar software systems without retraining or domain-specific tuning. We analyze three real-world C-embedded firmware projects provided by an industrial partner, using Coverity and Understand static analysis tools to extract software metrics. Through statistical analysis and hypothesis testing, we identify discriminative metrics and derived empirical threshold values capable of distinguishing faulty from non-faulty functions. The derived thresholds are validated through an experimental evaluation, demonstrating their effectiveness in identifying fault-prone functions with high precision. The results confirm that the derived thresholds can serve as an interpretable solution for fault prediction, aligning with industry standards and SQA practices. This approach provides a practical alternative to black-box AI models, allowing developers to systematically assess software quality, take preventive actions, and integrate metric-based fault prediction into industrial development workflows to mitigate software faults. |
| 2026-02-06 | [SuReNav: Superpixel Graph-based Constraint Relaxation for Navigation in Over-constrained Environments](http://arxiv.org/abs/2602.06807v1) | Keonyoung Koh, Moonkyeong Jung et al. | We address the over-constrained planning problem in semi-static environments. The planning objective is to find a best-effort solution that avoids all hard constraint regions while minimally traversing the least risky areas. Conventional methods often rely on pre-defined area costs, limiting generalizations. Further, the spatial continuity of navigation spaces makes it difficult to identify regions that are passable without overestimation. To overcome these challenges, we propose SuReNav, a superpixel graph-based constraint relaxation and navigation method that imitates human-like safe and efficient navigation. Our framework consists of three components: 1) superpixel graph map generation with regional constraints, 2) regional-constraint relaxation using graph neural network trained on human demonstrations for safe and efficient navigation, and 3) interleaving relaxation, planning, and execution for complete navigation. We evaluate our method against state-of-the-art baselines on 2D semantic maps and 3D maps from OpenStreetMap, achieving the highest human-likeness score of complete navigation while maintaining a balanced trade-off between efficiency and safety. We finally demonstrate its scalability and generalization performance in real-world urban navigation with a quadruped robot, Spot. |
| 2026-02-06 | [On the Identifiability of Steering Vectors in Large Language Models](http://arxiv.org/abs/2602.06801v1) | Sohan Venkatesh, Ashish Mahendran Kurapath | Activation steering methods, such as persona vectors, are widely used to control large language model behavior and increasingly interpreted as revealing meaningful internal representations. This interpretation implicitly assumes steering directions are identifiable and uniquely recoverable from input-output behavior. We formalize steering as an intervention on internal representations and prove that, under realistic modeling and data conditions, steering vectors are fundamentally non-identifiable due to large equivalence classes of behaviorally indistinguishable interventions. Empirically, we validate this across multiple models and semantic traits, showing orthogonal perturbations achieve near-equivalent efficacy with negligible effect sizes. However, identifiability is recoverable under structural assumptions including statistical independence, sparsity constraints, multi-environment validation or cross-layer consistency. These findings reveal fundamental interpretability limits and clarify structural assumptions required for reliable safety-critical control. |
| 2026-02-06 | [Crowd-FM: Learned Optimal Selection of Conditional Flow Matching-generated Trajectories for Crowd Navigation](http://arxiv.org/abs/2602.06698v1) | Antareep Singha, Laksh Nanwani et al. | Safe and computationally efficient local planning for mobile robots in dense, unstructured human crowds remains a fundamental challenge. Moreover, ensuring that robot trajectories are similar to how a human moves will increase the acceptance of the robot in human environments. In this paper, we present Crowd-FM, a learning-based approach to address both safety and human-likeness challenges. Our approach has two novel components. First, we train a Conditional Flow-Matching (CFM) policy over a dataset of optimally controlled trajectories to learn a set of collision-free primitives that a robot can choose at any given scenario. The chosen optimal control solver can generate multi-modal collision-free trajectories, allowing the CFM policy to learn a diverse set of maneuvers. Secondly, we learn a score function over a dataset of human demonstration trajectories that provides a human-likeness score for the flow primitives. At inference time, computing the optimal trajectory requires selecting the one with the highest score. Our approach improves the state-of-the-art by showing that our CFM policy alone can produce collision-free navigation with a higher success rate than existing learning-based baselines. Furthermore, when augmented with inference-time refinement, our approach can outperform even expensive optimisation-based planning approaches. Finally, we validate that our scoring network can select trajectories closer to the expert data than a manually designed cost function. |
| 2026-02-06 | [compar:IA: The French Government's LLM arena to collect French-language human prompts and preference data](http://arxiv.org/abs/2602.06669v1) | Lucie Termignon, Simonas Zilinskas et al. | Large Language Models (LLMs) often show reduced performance, cultural alignment, and safety robustness in non-English languages, partly because English dominates both pre-training data and human preference alignment datasets. Training methods like Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) require human preference data, which remains scarce and largely non-public for many languages beyond English. To address this gap, we introduce compar:IA, an open-source digital public service developed inside the French government and designed to collect large-scale human preference data from a predominantly French-speaking general audience. The platform uses a blind pairwise comparison interface to capture unconstrained, real-world prompts and user judgments across a diverse set of language models, while maintaining low participation friction and privacy-preserving automated filtering. As of 2026-02-07, compar:IA has collected over 600,000 free-form prompts and 250,000 preference votes, with approximately 89% of the data in French. We release three complementary datasets -- conversations, votes, and reactions -- under open licenses, and present initial analyses, including a French-language model leaderboard and user interaction patterns. Beyond the French context, compar:IA is evolving toward an international digital public good, offering reusable infrastructure for multilingual model training, evaluation, and the study of human-AI interaction. |
| 2026-02-06 | [Beyond Static Alignment: Hierarchical Policy Control for LLM Safety via Risk-Aware Chain-of-Thought](http://arxiv.org/abs/2602.06650v1) | Jianfeng Si, Lin Sun et al. | Large Language Models (LLMs) face a fundamental safety-helpfulness trade-off due to static, one-size-fits-all safety policies that lack runtime controllabilityxf, making it difficult to tailor responses to diverse application needs. %As a result, models may over-refuse benign requests or under-constrain harmful ones. We present \textbf{PACT} (Prompt-configured Action via Chain-of-Thought), a framework for dynamic safety control through explicit, risk-aware reasoning. PACT operates under a hierarchical policy architecture: a non-overridable global safety policy establishes immutable boundaries for critical risks (e.g., child safety, violent extremism), while user-defined policies can introduce domain-specific (non-global) risk categories and specify label-to-action behaviors to improve utility in real-world deployment settings. The framework decomposes safety decisions into structured Classify$\rightarrow$Act paths that route queries to the appropriate action (comply, guide, or reject) and render the decision-making process transparent.   Extensive experiments demonstrate that PACT achieves near state-of-the-art safety performance under global policy evaluation while attaining the best controllability under user-specific policy evaluation, effectively mitigating the safety-helpfulness trade-off. We will release the PACT model suite, training data, and evaluation protocols to facilitate reproducible research in controllable safety alignment. |
| 2026-02-06 | [Estimating Exam Item Difficulty with LLMs: A Benchmark on Brazil's ENEM Corpus](http://arxiv.org/abs/2602.06631v1) | Thiago Brant, Julien Kühn et al. | As Large Language Models (LLMs) are increasingly deployed to generate educational content, a critical safety question arises: can these models reliably estimate the difficulty of the questions they produce? Using Brazil's high-stakes ENEM exam as a testbed, we benchmark ten proprietary and open-weight LLMs against official Item Response Theory (IRT) parameters for 1,031 questions. We evaluate performance along three axes: absolute calibration, rank fidelity, and context sensitivity across learner backgrounds. Our results reveal a significant trade-off: while the best models achieve moderate rank correlation, they systematically underestimate difficulty and degrade significantly on multimodal items. Crucially, we find that models exhibit limited and inconsistent plasticity when prompted with student demographic cues, suggesting they are not yet ready for context-adaptive personalization. We conclude that LLMs function best as calibrated screeners rather than authoritative oracles, supporting an "evaluation-before-generation" pipeline for responsible assessment design. |
| 2026-02-06 | [TrapSuffix: Proactive Defense Against Adversarial Suffixes in Jailbreaking](http://arxiv.org/abs/2602.06630v1) | Mengyao Du, Han Fang et al. | Suffix-based jailbreak attacks append an adversarial suffix, i.e., a short token sequence, to steer aligned LLMs into unsafe outputs. Since suffixes are free-form text, they admit endlessly many surface forms, making jailbreak mitigation difficult. Most existing defenses depend on passive detection of suspicious suffixes, without leveraging the defender's inherent asymmetric ability to inject secrets and proactively conceal gaps. Motivated by this, we take a controllability-oriented perspective and develop a proactive defense that nudges attackers into a no-win dilemma: either they fall into defender-designed optimization traps and fail to produce an effective adversarial suffix, or they can succeed only by generating adversarial suffixes that carry distinctive, traceable fingerprints. We propose TrapSuffix, a lightweight fine-tuning approach that injects trap-aligned behaviors into the base model without changing the inference pipeline. TrapSuffix channels jailbreak attempts into these two outcomes by reshaping the model's response landscape to adversarial suffixes. Across diverse suffix-based jailbreak settings, TrapSuffix reduces the average attack success rate to below 0.01 percent and achieves an average tracing success rate of 87.9 percent, providing both strong defense and reliable traceability. It introduces no inference-time overhead and incurs negligible memory cost, requiring only 15.87 MB of additional memory on average, whereas state-of-the-art LLM-based detection defenses typically incur memory overheads at the 1e4 MB level, while composing naturally with existing filtering-based defenses for complementary protection. |
| 2026-02-06 | [Do Prompts Guarantee Safety? Mitigating Toxicity from LLM Generations through Subspace Intervention](http://arxiv.org/abs/2602.06623v1) | Himanshu Singh, Ziwei Xu et al. | Large Language Models (LLMs) are powerful text generators, yet they can produce toxic or harmful content even when given seemingly harmless prompts. This presents a serious safety challenge and can cause real-world harm. Toxicity is often subtle and context-dependent, making it difficult to detect at the token level or through coarse sentence-level signals. Moreover, efforts to mitigate toxicity often face a trade-off between safety and the coherence, or fluency of the generated text. In this work, we present a targeted subspace intervention strategy for identifying and suppressing hidden toxic patterns from underlying model representations, while preserving overall ability to generate safe fluent content. On the RealToxicityPrompts, our method achieves strong mitigation performance compared to existing baselines, with minimal impact on inference complexity. Across multiple LLMs, our approach reduces toxicity of state-of-the-art detoxification systems by 8-20%, while maintaining comparable fluency. Through extensive quantitative and qualitative analyses, we show that our approach achieves effective toxicity reduction without impairing generative performance, consistently outperforming existing baselines. |
| 2026-02-06 | [The hidden risks of temporal resampling in clinical reinforcement learning](http://arxiv.org/abs/2602.06603v1) | Thomas Frost, Hrisheekesh Vaidya et al. | Offline reinforcement learning (ORL) has shown potential for improving decision-making in healthcare. However, contemporary research typically aggregates patient data into fixed time intervals, simplifying their mapping to standard ORL frameworks. The impact of these temporal manipulations on model safety and efficacy remains poorly understood. In this work, using both a gridworld navigation task and the UVA/Padova clinical diabetes simulator, we demonstrate that temporal resampling significantly degrades the performance of offline reinforcement learning algorithms during live deployment. We propose three mechanisms that drive this failure: (i) the generation of counterfactual trajectories, (ii) the distortion of temporal expectations, and (iii) the compounding of generalisation errors. Crucially, we find that standard off-policy evaluation metrics can fail to detect these drops in performance. Our findings reveal a fundamental risk in current healthcare ORL pipelines and emphasise the need for methods that explicitly handle the irregular timing of clinical decision-making. |
| 2026-02-06 | [The Law of Task-Achieving Body Motion: Axiomatizing Success of Robot Manipulation Actions](http://arxiv.org/abs/2602.06572v1) | Malte Huerkamp, Jonas Dech et al. | Autonomous agents that perform everyday manipulation actions need to ensure that their body motions are semantically correct with respect to a task request, causally effective within their environment, and feasible for their embodiment. In order to enable robots to verify these properties, we introduce the Law of Task-Achieving Body Motion as an axiomatic correctness specification for body motions. To that end we introduce scoped Task-Environment-Embodiment (TEE) classes that represent world states as Semantic Digital Twins (SDTs) and define applicable physics models to decompose task achievement into three predicates: SatisfiesRequest for semantic request satisfaction over SDT state evolution; Causes for causal sufficiency under the scoped physics model; and CanPerform for safety and feasibility verification at the embodiment level. This decomposition yields a reusable, implementation-independent interface that supports motion synthesis and the verification of given body motions. It also supports typed failure diagnosis (semantic, causal, embodiment and out-of-scope), feasibility across robots and environments, and counterfactual reasoning about robot body motions. We demonstrate the usability of the law in practice by instantiating it for articulated container manipulation in kitchen environments on three contrasting mobile manipulation platforms |
| 2026-02-06 | [Baichuan-M3: Modeling Clinical Inquiry for Reliable Medical Decision-Making](http://arxiv.org/abs/2602.06570v1) | Baichuan-M3 Team, : et al. | We introduce Baichuan-M3, a medical-enhanced large language model engineered to shift the paradigm from passive question-answering to active, clinical-grade decision support. Addressing the limitations of existing systems in open-ended consultations, Baichuan-M3 utilizes a specialized training pipeline to model the systematic workflow of a physician. Key capabilities include: (i) proactive information acquisition to resolve ambiguity; (ii) long-horizon reasoning that unifies scattered evidence into coherent diagnoses; and (iii) adaptive hallucination suppression to ensure factual reliability. Empirical evaluations demonstrate that Baichuan-M3 achieves state-of-the-art results on HealthBench, the newly introduced HealthBench-Hallu and ScanBench, significantly outperforming GPT-5.2 in clinical inquiry, advisory and safety. The models are publicly available at https://huggingface.co/collections/baichuan-inc/baichuan-m3. |
| 2026-02-06 | [Safety Controller Synthesis for Stochastic Polynomial Time-Delayed Systems](http://arxiv.org/abs/2602.06569v1) | Omid Akbarzadeh, MohammadHossein Ashoori et al. | This work develops a theoretical framework for safety controller synthesis in discrete-time stochastic nonlinear polynomial systems subject to time-invariant delays (dt-SNPS-td). While safety analysis of stochastic systems using control barrier certificates (CBC) has been widely studied, developing safety controllers for stochastic systems with time delays remains largely unexplored. The main challenge arises from the need to account for the influence of delayed components when formulating and enforcing safety conditions. To address this, we employ Krasovskii control barrier certificates, which extend the conventional CBC framework by augmenting it with an additional summation term that captures the influence of delayed states. This formulation integrates both the current and delayed components into a unified barrier structure, enabling safety synthesis for stochastic systems with time delays. The proposed approach synthesizes safety controllers under input constraints, offering probabilistic safety guarantees robust to such delays: it ensures that all trajectories of the dt-SNPS-td remain within the prescribed safe region while fulfilling a quantified probabilistic bound. To achieve this, our method reformulates the safety constraints as a sum-of-squares optimization program, enabling the systematic construction of Krasovskii CBC together with their associated safety controllers. We validate the proposed framework through three case studies, including two physical systems, demonstrating its effectiveness and practical applicability. |
| 2026-02-06 | [AgentCPM-Report: Interleaving Drafting and Deepening for Open-Ended Deep Research](http://arxiv.org/abs/2602.06540v1) | Yishan Li, Wentong Chen et al. | Generating deep research reports requires large-scale information acquisition and the synthesis of insight-driven analysis, posing a significant challenge for current language models. Most existing approaches follow a plan-then-write paradigm, whose performance heavily depends on the quality of the initial outline. However, constructing a comprehensive outline itself demands strong reasoning ability, causing current deep research systems to rely almost exclusively on closed-source or online large models. This reliance raises practical barriers to deployment and introduces safety and privacy concerns for user-authored data. In this work, we present AgentCPM-Report, a lightweight yet high-performing local solution composed of a framework that mirrors the human writing process and an 8B-parameter deep research agent. Our framework uses a Writing As Reasoning Policy (WARP), which enables models to dynamically revise outlines during report generation. Under this policy, the agent alternates between Evidence-Based Drafting and Reasoning-Driven Deepening, jointly supporting information acquisition, knowledge refinement, and iterative outline evolution. To effectively equip small models with this capability, we introduce a Multi-Stage Agentic Training strategy, consisting of cold-start, atomic skill RL, and holistic pipeline RL. Experiments on DeepResearch Bench, DeepConsult, and DeepResearch Gym demonstrate that AgentCPM-Report outperforms leading closed-source systems, with substantial gains in Insight. |
| 2026-02-06 | [Subgraph Reconstruction Attacks on Graph RAG Deployments with Practical Defenses](http://arxiv.org/abs/2602.06495v1) | Minkyoo Song, Jaehan Kim et al. | Graph-based retrieval-augmented generation (Graph RAG) is increasingly deployed to support LLM applications by augmenting user queries with structured knowledge retrieved from a knowledge graph. While Graph RAG improves relational reasoning, it introduces a largely understudied threat: adversaries can reconstruct subgraphs from a target RAG system's knowledge graph, enabling privacy inference and replication of curated knowledge assets. We show that existing attacks are largely ineffective against Graph RAG even with simple prompt-based safeguards, because these attacks expose explicit exfiltration intent and are therefore easily suppressed by lightweight safe prompts. We identify three technical challenges for practical Graph RAG extraction under realistic safeguards and introduce GRASP, a closed-box, multi-turn subgraph reconstruction attack. GRASP (i) reframes extraction as a context-processing task, (ii) enforces format-compliant, instance-grounded outputs via per-record identifiers to reduce hallucinations and preserve relational details, and (iii) diversifies goal-driven attack queries using a momentum-aware scheduler to operate within strict query budgets. Across two real-world knowledge graphs, four safety-aligned LLMs, and multiple Graph RAG frameworks, GRASP attains the strongest type-faithful reconstruction where prior methods fail, reaching up to 82.9 F1. We further evaluate defenses and propose two lightweight mitigations that substantially reduce reconstruction fidelity without utility loss. |
| 2026-02-06 | [Auditing Rust Crates Effectively](http://arxiv.org/abs/2602.06466v1) | Lydia Zoghbi, David Thien et al. | We introduce Cargo Scan, the first interactive program analysis tool designed to help developers audit third-party Rust code. Real systems written in Rust rely on thousands of transitive dependencies. These dependencies are as dangerous in Rust as they are in other languages (e.g., C or JavaScript) -- and auditing these dependencies today means manually inspecting every line of code. Unlike for most industrial languages, though, we can take advantage of Rust's type and module system to minimize the amount of code that developers need to inspect to the code that is potentially dangerous. Cargo Scan models such potentially dangerous code as effects and performs a side-effects analysis, tailored to Rust, to identify effects and track them across crate and module boundaries. In most cases (69.2%) developers can inspect flagged effects and decide whether the code is potentially dangerous locally. In some cases, however, the safety of an effect depends on the calling context -- how a function is called, potentially by a crate the developer imports later. Hence, Cargo Scan tracks context-dependent information using a call-graph, and collects audit results into composable and reusable audit files. In this paper, we describe our experience auditing Rust crates with Cargo Scan. In particular, we audit the popular client and server HTTP crate, hyper, and all of its dependencies; our experience shows that Cargo Scan can reduce the auditing burden of potentially dangerous code to a median of 0.2% of lines of code when compared to auditing whole crates. Looking at the Rust ecosystem more broadly, we find that Cargo Scan can automatically classify ~3.5K of the top 10K crates on crates.io as safe; of the crates that do require manual inspection, we find that most of the potentially dangerous side-effects are concentrated in roughly 3% of these crates. |

<!-- LATEST_PAPERS_END --> 

---

## 🔑 Key Safety Domains(coming soon)
![LLM Safety Category](/assets/img/image1.png "LLM Safety Category")

**Fig.1**: LLM Safety [[Ma et al., 2025]([arXiv:2502.05206](https://arxiv.org/abs/2502.05206))]

Here we only list the security scenarios involved in the most popular research directions.

- Adversarial Attack
  - white box
  - black box
  - grey box
- Jailbreak Attacks
  - white box
    - gradient-based
  - black box
    - prompt injection
    - role play
    - encodind-based
    - multilingual-based
- Backdoor Attacks 
- DDos Attack
- Privacy Leakage
- System Data Leakage
- Deepfake

---

## 🔖 Dataset & Benchmark
### For Traditional LLM
Please refer to [dataset&benchmark for LLM](./collection/dataset/dataset_for_LLM.md)

### For Advanced LRM
Please refer to [dataset&benchmark for LRM](./collection/dataset/dataset_for_LRM.md)

---

## 📚 Survey
### LRM Related
- Efficient Inference for Large Reasoning Models: A Survey
- A Survey of Efficient Reasoning for Large Reasoning Models: Language, Multimodality, and Beyond
- Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models
- A Survey on Post-training of Large Language Models
- Reasoning Language Models: A Blueprint
- Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning Large Language Models
### LRM Safety Related
- Efficient Inference for Large Reasoning Models: A Survey
---

## 🛠️ Projects & Tools(coming soon)
### Model-Specific Resources(example)
- **DeepSeek-R1 Safety Kit**  
  Official safety evaluation toolkit for DeepSeek-R1 reasoning modules

- **OpenAI o1 Red Teaming Framework**  
  Adversarial testing framework for multi-turn reasoning tasks

### General Tools(coming soon)(example)
- [ReasonGuard](https://github.com/example/reasonguard )  
  Real-time monitoring for reasoning chain anomalies

- [Ethos](https://github.com/example/ethos )  
  Ethical alignment evaluation suite for LRMs

---

## 🤝 Contributing
We welcome contributions! Please:
1. Fork the repository
2. Add resources via pull request
3. Ensure entries follow the format:
   ```markdown
   - [Year] [Paper Title](URL)  
     *Brief description (5-15 words)*
   ```
4. Maintain topical categorization

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License
This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## ❓ FAQ
**Q: How do I stay updated?**  
A: Watch this repo and check the "Recent Updates" section (coming soon).

**Q: Can I suggest non-academic resources?**  
A: Yes! Industry reports and blog posts are welcome if they provide novel insights.

**Q: How are entries verified?**  
A: All submissions undergo community review for relevance and quality.

---
## 🔗 References

Ma, X., Gao, Y., Wang, Y., Wang, R., Wang, X., Sun, Y., Ding, Y., Xu, H., Chen, Y., Zhao, Y., Huang, H., Li, Y., Zhang, J., Zheng, X., Bai, Y., Wu, Z., Qiu, X., Zhang, J., Li, Y., Sun, J., Wang, C., Gu, J., Wu, B., Chen, S., Zhang, T., Liu, Y., Gong, M., Liu, T., Pan, S., Xie, C., Pang, T., Dong, Y., Jia, R., Zhang, Y., Ma, S., Zhang, X., Gong, N., Xiao, C., Erfani, S., Li, B., Sugiyama, M., Tao, D., Bailey, J., Jiang, Y.-G. (2025). *Safety at Scale: A Comprehensive Survey of Large Model Safety*. arXiv:2502.05206.

---

> *"With great reasoning power comes great responsibility."* - Adapted from [AI Ethics Manifesto]



