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
| 2026-11-24 | [The Radar-based Activities of Daily Living (R-ADL) Dataset](http://arxiv.org/abs/2611.33088v1) | Dominique Regli, Brian C. Williams | Monitoring Activities of Daily Living (ADLs) supports safety and well-being for people living at home, especially older adults, and radar sensing is gaining increasing attention in the field of activity recognition for its ability to ensure privacy in the home while also surpassing technical limitations of other sensing modalities. However, radar-based activity recognition remains constrained by the limited availability of large-scale datasets. We introduce the Radar-based Activities of Daily Living (R-ADL) dataset to addresses the gaps identified in existing work: 1) it expands the scale of available data by providing over eight hours of annotated recordings in which a person performs eight distinct activities; 2) it overcomes room configuration limitations by recording from five different locations in a two-bedroom, two-bathroom apartment; and 3) it extends beyond primitive actions to focus on high-level ADLs. In addition to the dataset, this paper introduces practical guidance for future ADL dataset construction. We conducted a detailed analysis of how ADLs are prioritized across academic research, health survey tools, and clinical trials to identify clinically significant ADLs. Furthermore, we provide our experimental design, data collection and annotation strategies, post-processing pipeline for forming the R-ADL dataset from raw data, and baseline results on standard machine learning architectures. The R-ADL dataset serves as a valuable resource for advancing ADL recognition systems as we continue building larger and more diverse datasets. |
| 2026-11-24 | [EmbodiSafety: Benchmarking Risk Understanding and Response in Embodied AI](http://arxiv.org/abs/2611.33054v1) | Erfei Cui, Shengjia Zhao et al. | Progress in embodied foundation models is largely measured by task success, while the physical safety of their decisions remains underexamined. Safety evaluations centered on rule following and binary judgments do not adequately assess whether embodied models understand environmental hazards and make safe decisions. We introduce EmbodiSafety, comprising 1,092 snapshot tasks and 540 trajectory tasks across 844 visual scenarios to assess safety over short and long horizons. Six response modes cover direct execution, refusal, clarification, constrained execution, autonomous recovery, and human escalation. Distinguishing recovery from human intervention makes the boundary of safe autonomy an explicit evaluation target. Our protocol assesses mode correctness, risk identification, and actionable handling, with complete decision accuracy requiring all applicable checks to pass. Claude-Opus-5.5 achieves the highest complete decision accuracy among reference models, reaching 56.1% on snapshots and 61.5% on trajectories. We further develop a data construction pipeline and curate 40k examples for safety alignment. Supervised training on this corpus raises the complete decision accuracy of Qwen3.5-2B and Qwen3.5-4B. |
| 2026-11-24 | [SALSA: Sampling Adversarial Suffixes from Learned Distributions for LLM Safety Assessment](http://arxiv.org/abs/2611.33030v1) | Shivank Rajput, Himanshu Singh et al. | Large language models (LLMs) remain vulnerable to jailbreak attacks, where adversaries append optimized suffixes to otherwise benign looking prompts to bypass safety alignment. Existing optimization-based GCG-style attacks can be effective, but they are expensive at inference time, and suffer from poor transfer. We present, SALSA, a generative framework that learns a distribution over jailbreak suffixes from optimization-derived attack corpora. Once trained, the model supports efficient sampling-based attacks. On open-source targets, it achieves attack success rates comparable to state-of-the-art optimization baselines, while reducing generation time from 3.46s to 0.04s per prompt. In black-box transfer, it remains highly effective against proprietary systems, including 99% ASR on GPT-3.5-Turbo, and reaches 45% ASR on the substantially harder GPT-4o, more than doubling the strongest baseline at an equal sampling budget. We further offer an intuition for why generative sampling can preserve high attack success, when the learned suffix distribution is close to the distribution of optimization outputs. Overall, our results show that learning adversarial suffix distributions yields a scalable, and practically efficient foundation for continuous LLM vulnerability discovery. |
| 2026-11-24 | [GRAPE-Bench: When Vision Degrades, Does Video Reasoning Hold?](http://arxiv.org/abs/2611.33020v1) | Ce Zhang, Junyi He et al. | Despite strong performance on standard benchmarks, video reasoning models remain vulnerable to visual degradation, raising concerns about their reliability in safety-critical scenarios. Existing benchmarks address this concern in part by simulating visual degradation or probing model reasoning with challenging questions, leaving open how well models handle naturally degraded video and whether they can make the simpler judgments that matter for safe driving. To this end, we introduce GRAPE-Bench, which jointly evaluates video reasoning across visual degradation conditions and question difficulty. It combines a synthetic set spanning twelve corruption types across daily-life and driving videos with real-world footage of driving at night or in rain, snow, or fog. Each video is paired with three progressively challenging questions, from fundamental judgments with potential safety implications to demanding reasoning stress tests. Across open-weight and frontier models, we observe ranking reversals between synthetic and real-world conditions and level-specific weaknesses obscured by overall accuracy. Synthetic-data fine-tuning improves Qwen3.5-9B's real-world accuracy from 76.7% to 81.1%, while clean-privileged on-policy distillation reaches 80.4% in a separate reasoning-enabled setting. Through preference steering, we further demonstrate continuous control over a frozen model's trade-off between safety-relevant judgments and challenging reasoning, with little change in overall accuracy. Together, GRAPE-Bench provides a unified testbed for evaluating, improving, and controlling video reasoning under degraded visual conditions. |
| 2026-11-24 | [From Safety-Driven Redundancy to Value-Driven Architecture: An ODD-Based Rationalization Framework for Level 4 Autonomous Vehicles](http://arxiv.org/abs/2611.32960v1) | Tiago Simão Ferreira, Farzad Fallahi | Level 4 autonomous vehicles require fault-tolerant architectures that preserve critical driving functions under sensing, computing, communication, power, and actuation failures. As these systems progress toward industrial scalability, architectural complexity, validation burden, integration effort, and lifecycle cost become increasingly relevant. This study proposes an Operational Design Domain (ODD)-based, safety-constrained systems-engineering framework for architecture rationalization. ODD-specific capabilities, physical fault assumptions, post-fault operational policies, and spatial sensing requirements are propagated through an integrated safety-feasibility gate. Architecture transformations, including removal, relocation, functional reassignment, and consolidation, are evaluated only within the resulting safety-feasible design space, followed by lifecycle-cost sensitivity and robust multiobjective Pareto analysis. A generic, synthetically parameterized Level 4 architecture is used as a demonstrative computational case study. Within this modeled design space, direct ablation showed limited rationalization potential, whereas directional redistribution and functional consolidation generated a broader set of safety-feasible candidates. Under the illustrative lifecycle-cost model, the best nominal candidate achieved a 5.55% normalized cost reduction and remained cost-reducing across all 5,000 sampled points in the prescribed uncertainty domain. The analysis identified 15 non-dominated architecture-policy solutions representing seven physical transformations. The results support capability-based, policy-conditioned, and lifecycle-aware architecture rationalization under the stated assumptions, with safety retained as a non-compensatory feasibility constraint. The reported transformations are design candidates rather than validated production architectures. |
| 2026-11-24 | ["I Do Intend to Deliver": Will LLMs Run a Marketplace Advance-Fee Scam?](http://arxiv.org/abs/2611.32957v1) | Megan Squire | Online fraud is traditionally limited by the cost of the human labor needed to talk to potential victims. We ask whether hosted large language models will now supply that labor. We prompted ten off-the-shelf models to play the seller in the "free item, just pay delivery" advance-fee scam common on buy-and-sell sites, using a plain system prompt with no jailbreak. Each model played 30 conversations against each of four buyer personas, from credulous to guarded, with the same model playing the buyer (1,200 conversations). We label each conversation by what happened to the buyer's money, and validate the labels against two human coders. 737 conversations (61%) ended with the buyer paying in advance. Eight of the ten models played in almost every conversation, and running both sides cost about 2.5 cents per payment. Only the guarded buyer, who wanted a findable identity, resisted, and the most willing sellers got past it by inventing names and phone numbers. Sellers that recognized the scam still took the money in 133 conversations, usually reasoning about their own honesty rather than the buyer's risk. Several models protected the buyer without refusing, by conceding payment on delivery, so refusal rates alone misstate safety. We release the benchmark, prompts, code and labels. |
| 2026-11-24 | [ShellBench: What Matters More for AI Agents, Models, or Harness for Everyday Tasks](http://arxiv.org/abs/2611.32946v1) | Vincent Koc, Aditya Washikar et al. | The use of AI agents among the general public has been increasing exponentially throughout 2026, though it is often hard for users to determine which configuration is most effective for their personal use. To help provide clarity here we introduce ShellBench, 100 day-to-day tasks based on more than 10 million user requests, each set in a realistic workspace with messy files and competing data sources, and graded for both correctness and safety. We evaluate 12 models across 5 agent harnesses at 3 reasoning levels for a total of 53,100 rollouts, 20% of which are on tasks designated for public release. We find that model choice impacts performance more than harness choice, but the three leading models differ by only 1.1 percentage points. GPT-6.1 Sol passed 61.4% of attempts at about one-sixth of GPT-6 Astra's estimated token cost per attempt, while GPT-6 Astra led at 62.5%. We also found that harness choice had little effect on either performance or safety with all harnesses scoring between 42.9 (Claude Code) and 45.2 (OpenClaw) for pass@1. |
| 2026-11-24 | [Compositional Reachset-Conformant Modeling for Managing Uncertainty in the Development of Cyber-Physical Systems](http://arxiv.org/abs/2611.32913v1) | Laura Lützow, Francesco Pio Urbano et al. | The development of cyber-physical systems (CPS) proceeds through iterative design revisions, and compliance with safety requirements must be re-established after every revision, typically based on a system model. Conventional models, however, predict only the most likely behavior, leading to model-plant mismatch that often surfaces only during late-stage testing, when redesign is most expensive. This article presents reachset-conformant modeling as a framework for managing uncertainty throughout the development process: instead of a single trajectory, a reachset-conformant model predicts a set of behaviors, calibrated from data to conservatively enclose the true system behavior with a statistical guarantee. Safety properties established on the model therefore transfer to the realized system. Beyond a practical introduction to reachset-conformant identification and its benefits for CPS development, our main theoretical contribution is a closure result: reachset conformance is preserved under serial composition, parallel composition, and feedback interconnection, and hence under any architecture assembled from these primitives. Subsystem models identified independently from local measurements can thus be substituted into the system architecture, and the interconnected model inherits a coverage guarantee from the subsystem guarantees. We complement this result with update strategies that adapt the uncertainty sets after design changes without restarting the identification workflow. Experiments on an automotive brake-system test bench demonstrate that a) a single model covers all operating conditions and hardware variants, b) late-stage modifications are absorbed by uncertainty set updates, c) a black-box hydraulic unit is conservatively abstracted from input-output data alone, and d) independently identified subsystem models form a conformant model by construction. |
| 2026-11-24 | [Scaling Laws of Ensemble Uncertainty in Language Models](http://arxiv.org/abs/2611.32801v1) | Siamak Ravanbakhsh, David Nitchi et al. | Many use cases for large language models (LLMs)---from AI safety to exploration and human interaction---can benefit from epistemic uncertainty estimates. We argue that existing Bayesian Deep Learning solutions, which are often restricted to a fixed model family, fail to capture the full epistemic uncertainty at practical LLM scales. This is because the error due to the bounded capacity of the model family remains invisible to disagreement among models supported by the prior. We relate this error to the well-known power-law scaling of LLM loss with model size. We then show that a second power law quantifies the shrinking disagreement within ensembles as model size increases. Its slower decay relative to the capacity law above suggests that ensembles capture a growing share of epistemic uncertainty and become more competitive as an axis for scaling, compared to a single larger model. Finally, a third power law quantifies the loss of ensemble diversity as members share a larger fraction of training before branching, and therefore identifies a further limitation, beyond the capacity error, for LoRA ensembles and optimization-based or second-order methods that approximate uncertainty around a single training solution. At a high level, these findings suggest diverse ensembles can be a promising axis, both for scaling and uncertainty quantification at larger scales. |
| 2026-11-24 | [Risk-Informed Path Planning: Collision and Grounding Avoidance Application for Autonomous Surface Vessels](http://arxiv.org/abs/2611.32760v1) | Emir Cem Gezer, Ozan Kaya et al. | Maritime Autonomous Surface Ships (MASS) must demonstrate safety before and during operations, necessitating online risk assessment to guide decision-making at sea. This paper presents a risk-informed Rapidly Exploring Random Tree Star (RRT*) path planner that integrates a Bayesian Belief Network (BBN) as an online risk model. The BBN evaluates the probability and severity distribution of hazardous events, which are aggregated into a bounded scalar risk index. During the search, this risk index is evaluated at sample points along candidate edges. The planner optimizes a composite cost function comprising path distance, heading change penalties, and accumulated risk, subject to a hard constraint enforcing a risk acceptance limit. Each search tree vertex stores the BBN posterior at its state, enabling the tree to function as an explainable risk map for individual hazards. We define design requirements for the BBN to serve as the risk model in this context and introduce a runtime model that translates replanning deadlines into strict limits on single risk calculation times. The planner is demonstrated in a simulated harbor entry scenario involving collision and grounding hazards. Measured run times validate that the planner operates within the computational bounds predicted by the runtime model for a single replanning cycle. |
| 2026-11-24 | [ASCENT: A Benchmark for Evaluating and Advancing Stepwise Diagnostic Reasoning in Large Language Models on Common Clinical Scenarios](http://arxiv.org/abs/2611.32734v1) | Yera Choi, Yeong Hwa Kim et al. | Large language models (LLMs) excel at medical question answering yet are rarely evaluated on the stepwise diagnostic reasoning that defines real clinical workflows, where impressions are revised as information accumulates. We build Annotated Stepwise Clinical rEasoning for NaturalisTic Diagnosis (ASCENT), a clinician-annotated benchmark and training resource of 3,078 stepwise problems derived from MedQA-USMLE that decomposes each vignette into EMR-aligned steps (Findings, Impression, supporting Rationale), enabling evaluation of intermediate reasoning under incomplete information.   Experiments and training with ASCENT revealed insights into how current LLMs handle stepwise diagnostic reasoning. Even strong reasoning models that perform well on MedQA-USMLE leave substantial headroom on ASCENT, and general-purpose frontier models trail further, exposing a persistent gap between fully informed and stepwise diagnosis. Fine-tuning Qwen2.5-7B and 32B on ASCENT yields measurable F1 gains over both pre-trained and HuatuoGPT-o1 CoT-trained baselines, with gains driven primarily by precision. Complementary robustness analyses (counterfactual perturbation, format-vs-content control, judge agreement, and rollout) further show that ASCENT-fine-tuned models rely on the diagnostic content of prior impressions rather than imitating their output format, while error propagation under rollout remains a key challenge for clinical deployment. |
| 2026-11-24 | [Red-Teaming a Sovereign LLM, Its Reasoning and Its Training Stages](http://arxiv.org/abs/2611.32710v1) | Hatef Otroshi Shahreza | Open-weight large language models (LLMs) are increasingly deployed for applications with sensitive information. We investigate whether confidentiality instructions in system prompt can protect such information. We red-team Apertus v1.5, an open sovereign LLM, and five other model families across six languages and reasoning modes. We use synthetic personal records to evaluate direct disclosure, correction baits, yes/no verification, derived-value queries, reasoning leakage and authorisation checks. We also explore copyright attribution, historical misrepresentation and stereotyping. Despite confidentiality instructions, Apertus 8B and 70B leak information in 96% and 95% of requests, respectively, in correction-bait attacks. Verification and derived-value queries also expose protected information without necessarily revealing identifiers. Moreover, reasoning traces can leak confidential records even when final answers withhold them. Our experiments also show that post-training does not consistently improve safety. Our findings demonstrate that system prompts alone cannot ensure confidentiality and highlight the need for multilingual reasoning-aware safety evaluations. |
| 2026-11-24 | [Gated Concept Adapters: Layer-Wise Concept Bottlenecks for Auditable Text Understanding](http://arxiv.org/abs/2611.32600v1) | Niloofar Ranjbar, Saeedeh Momtazi et al. | Human-facing text understanding systems are used to infer affect, moderate harmful content, and support socially sensitive decisions. However, Transformer predictions are difficult to audit because they are produced from high-dimensional hidden states that do not provide a compact, inspectable interface for human analysis or intervention. We introduce Gated Concept Adapters (GCA), a layer-wise concept bottleneck framework that attaches gated, low-dimensional concept-producing readout adapters to selected outputs of pretrained Transformers. Token representations are mapped into concept activations, filtered by learned gates, pooled into layer-level summaries, and exposed to the prediction head as a concept-mediated interface. The dimensions can be inspected through usage statistics, signature n-grams, post-hoc concept names, and targeted concept interventions. We evaluate GCA on fine-grained emotion classification, toxicity detection, topic classification, and a controlled verb-cloze stress test. In the affective and safety-relevant classification settings, GCA learns comparatively low-redundancy and intervention-sensitive concept representations while retaining useful predictive performance. On toxicity detection, it remains close to an unconstrained ModernBERT baseline while enabling concept-level inspection, targeted intervention, and lightweight post-hoc control. On GoEmotions, the end-to-end concept model incurs a macro-F1 cost relative to unconstrained RoBERTa, while a task-initialized frozen variant substantially reduces this gap. On AG News, GCA remains competitive with recent concept-based baselines, and the Qwen verb-cloze experiment clarifies the limits of concept-mediated full-vocabulary prediction. Overall, GCA provides a neural concept interface for auditable text understanding, prioritizing transparency, diagnosis, and intervention over unconstrained accuracy maximization. |
| 2026-11-24 | [Follow-Bench 2.0: An End-to-End 3D Benchmark for Socially-Aware Robot Person Following](http://arxiv.org/abs/2611.32579v1) | Hanjing Ye, Tianle Zeng et al. | Socially-aware robot person following (RPF) requires a mobile robot to follow a designated person in dynamic human environments while maintaining target identity, avoiding obstacles and pedestrians, and preserving socially comfortable formations. Existing benchmarks only partially evaluate this coupled problem: embodied visual tracking benchmarks emphasize target visibility with limited social interaction, person re-identification benchmarks study perception without closed-loop control, and motion-planning benchmarks often assume ground-truth target and pedestrian states. We introduce Follow-Bench 2.0, an end-to-end 3D benchmark for evaluating socially-aware RPF under coupled perception--planning challenges. Built in Unreal Engine, Follow-Bench 2.0 provides difficulty-leveled scenarios with diverse pedestrian flows, lighting and weather conditions, cluttered layouts, bottlenecks, queueing behaviors, distractors, and temporary target occlusions. The benchmark supports modular perception--planning pipelines, end-to-end visual policies, and foundation-model-based active trackers under a unified closed-loop protocol. It further evaluates both back- and side-following configurations and reports safety--comfort metrics covering task success, target visibility, target recovery, social-space intrusion, following formation, and motion smoothness. Experiments show that RPF-oriented modular pipelines substantially outperform current end-to-end visual policies on close and socially comfortable following, but also reveal that socially-aware RPF remains far from solved in interaction-heavy environments and especially in side-following. Follow-Bench 2.0 therefore provides a diagnostic platform for studying how perception errors, occlusions, crowd interactions, and following configurations jointly affect robot person following. |
| 2026-11-24 | [Avalon-M1: Reinforcement Learning from Collective Rewards](http://arxiv.org/abs/2611.32539v1) | Iliyan Zarov, Henry Course et al. | We introduce reinforcement learning from collective rewards (RLCR), a recipe for jointly training language models to act as agent organisations under latency constraints. A CEO and up to eight specialists share one policy, co-ordinate through a message board and workspace, and are trained with a common outcome-derived advantage. The deadline makes parallel work valuable when it increases the return earned before the cap. We train Avalon-M1-Mini (120B) across 147 environments. For the trained model, eight-hire organisational execution raises IMO-ProofBench-Advanced scores from 23.8% to 39.0% relative to single-agent execution. Across 12 benchmarks, organisations achieve similar average scores at approximately one-third the derived latency of token-matched single-agent execution and just over half that of eight independent attempts followed by selection. The trained model also outperforms Nemotron 3 Super, post-trained from the same base model, across all evaluations when comparing each model's best tested configuration. On EdgeBench, organisations overtake the single agent after approximately 0.4 derived hours. Over the course of training, organisations develop hierarchies to manage communication costs, invest more in peer review, and alter how work is divided. Safety evaluations find near-universal refusal of harmful actions but sustained declines in external reporting in two of three oversight scenarios. These results establish a shared-advantage baseline for studying organisational learning, performance-latency trade-offs, and collective oversight. |
| 2026-11-24 | [Analysis of Federated Graph Neural Networks for Predicting Nuclear Core Properties](http://arxiv.org/abs/2611.32509v1) | Oliver Kožul, Mihailo Ilić et al. | Federated learning has grown in popularity owing to the increasing data privacy and safety concerns. It allows multiple parties to collaboratively train a model without exchanging their raw data. As an example of highly sensitive infrastructure of national importance, nuclear reactors benefit from this approach, where---while pooling multiple large datasets from different institutions could improve machine learning solutions---the confidentiality of the datasets is paramount. In this paper, we demonstrate that a graph neural network (GNN) outperforms the convolutional neural network (CNN) baseline in a federated setting for a use case concerning nuclear reactor core optimization. The GNN architecture is better suited for the nuclear reactor geometry as, unlike the CNN, it does not require padding the cylindrical core and does not assume translation invariance. The key finding of this work is that input representation is more important than model size and complexity. More sophisticated federated algorithms were employed, but the improvements derived were of lower magnitude. Experimenting with training corpus volume led to the discovery that greater fuel inventory variance is more beneficial than arrangements of existing cases, which is a direct consequence of data heterogeneity. Going forward, focus will remain on enhancing the input and readout representation. |
| 2026-11-24 | [Assessing Robustness to Catastrophic Forgetting in Continual Deep Reinforcement Learning for Cyber-Physical Systems](http://arxiv.org/abs/2611.32496v1) | Kim N. Nolle, Ivana Dusparic et al. | As the use of reinforcement learning (RL) becomes more widespread in cyber-physical systems, it becomes increasingly important to design RL agents that are able to adapt to changes in their environments in order to maintain safety and reliability. This paper addresses the question of whether different classes of RL algorithm are more robust against forgetting previously learned behaviours, a critical issue in continual learning, in non-stationary environments. We evaluate the robustness of representative examples of three important classes of RL algorithm in a non-stationary autonomous parking environment. These are Proximal Policy Optimization (PPO), which is a model-free on-policy algorithm, Soft-Actor-Critic (SAC), which is a model-free off-policy algorithm, and Drama, which is a model-based algorithm. The agents must learn to successfully park in four different scenarios, which are presented successively, with parking spaces oriented at varying angles. We find off-policy algorithms appear to be more robust compared to on-policy algorithms due to their use of replay, while model-based algorithms do not appear to offer any direct advantage over model-free algorithms. These experiments also highlight the additional challenges of finding robust abstractions of the environment, and oversensitivity to hyperparameters. |
| 2026-11-24 | [A Star is Reborn: The PHP Interpreter Rewritten in Safe Rust](http://arxiv.org/abs/2611.32477v1) | Tianyu Li, Bo Wang et al. | PHP powers most of the web, yet its C interpreter has had many memory-safety vulnerabilities. Reimplementing the interpreter in safe Rust offers a promising way to eliminate these vulnerabilities. However, eliminating native memory violations alone does not necessarily prevent unintended accesses to the logical memory of the translated program.   To prevent such failures, we introduce logical memory safety, which confines the program operations to their permitted memory. In this paper, we formalize this property for PHP through an abstract PHP machine model that captures how the PHP interpreter manages program values while abstracting away functionality details. Guided by this model, we reimplement the PHP 5.0.0 engine in safe Rust, and formally prove that, for a core fragment of PHP, our Rust translation refines the abstract PHP machine, thereby establishing logical memory safety for this fragment.   Our translation safely handles all 173 memory vulnerabilities we collected that affect PHP 5.0.0 prior to the rewrite. The Rust version matches the C implementation on all functionality tests, and produces identical outputs for the real-world PHP applications we evaluate. To our knowledge, this is the first principled safe Rust port of a core fragment of the PHP interpreter that combines formal guarantees of logical memory safety with functional compatibility. The improved security comes at a modest average runtime overhead of only 15% compared to C. |
| 2026-11-24 | [Delay-Tolerant Formation Tracking for Multi-AGV Systems via Reference-Shift Coupling](http://arxiv.org/abs/2611.32409v1) | Yunfei Bi | Automated guided vehicles (AGVs) that transport a load together must keep a formation while each vehicle tracks its own reference, and they coordinate over wireless links with link-dependent, time-varying and jumping delays. We propose a cascade design for differential-drive AGVs modelled as kinematic unicycles: delayed data from the neighbours drive only a second-order filter that shifts each vehicle's own reference, and an unmodified gain-scheduled tracking law follows the shifted reference. A saturation keeps the shifted reference feasible, and the feed-forward is exact, so the tracking loop never sees the network. We prove that tracking and formation errors remain bounded for arbitrary delays and converge for every bounded delay and every directed, possibly switching, graph. We also construct a strict Lyapunov function with explicit constants and prove exponential convergence and a local input-to-state estimate with delay-independent gains. On undirected graphs, the coupling attenuates the formation error caused by persistent local errors by at least the factor lambda/(lambda+Lambda*lambda_2(L)) (leak lambda, coupling gain Lambda, algebraic connectivity lambda_2(L)), for any delay. Every inequality is verified with an event-driven, error-controlled simulation. A robustness study on a high-fidelity AGV model (drive dynamics, slip, localisation errors, clock offsets, lossy Wi-Fi) comprises 4,894 runs, including paired comparisons against tuned baselines. The cascade never failed, with latencies up to 4 s, 90% packet loss and up to 12 vehicles, whereas a direct delayed coupling failed when its gain saturated the drives. Coupling reduced the formation error after traction losses by 21-32% and after safety stops by up to 47%, at the price of tracking accuracy. Localisation bias and clock offsets, which no coupling law can observe, limit the achievable accuracy. |
| 2026-11-24 | [Protective-Design Pre-Emption in Child-Facing Generative AI](http://arxiv.org/abs/2611.32352v1) | Verena Jasmin Hallitschke, Mahelie Dissanayake et al. | Generative AI is adopted rapidly by children, yet unrestricted general-audience models dominate their usage. This paper presents the open problem of \emph{protective-design pre-emption}: Delayed market introduction, low access thresholds, utility gaps, habituation, and peer adoption foster unrestricted AI model lock-in, leading children to bypass child-centric platforms. With first signs of lower adoption of child-centric platforms, we argue that this shift may reinforce existing AI risks such as exposing children to deepened parasocial dependencies, skill erosion, and misinformation. To address this problem, we propose interdisciplinary research directions to quantify pre-emption, understand its socio-technical drivers, and develop interventions ensuring child-centric AI is actively adopted, not just available. |

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



