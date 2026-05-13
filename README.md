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
| 2026-05-12 | [Solve the Loop: Attractor Models for Language and Reasoning](http://arxiv.org/abs/2605.12466v1) | Jacob Fein-Ashley, Paria Rashidinejad | Looped Transformers offer a promising alternative to purely feed-forward computation by iteratively refining latent representations, improving language modeling and reasoning. Yet recurrent architectures remain unstable to train, costly to optimize and deploy, and constrained to small, fixed recurrence depths. We introduce Attractor Models, in which a backbone module first proposes output embeddings, then an attractor module refines them by solving for the fixed point, with gradients obtained through implicit differentiation. Thus, training memory remains constant in effective depth, and iterations are chosen adaptively by convergence. Empirically, Attractor Models outperform existing models across two regimes, large-scale language-model pretraining and reasoning with tiny models. In language modeling, Attractor Models deliver a Pareto improvement over standard Transformers and stable looped models across sizes, improving perplexity by up to 46.6% and downstream accuracy by up to 19.7% while reducing training cost. Notably, a 770M Attractor Model outperforms a 1.3B Transformer trained on twice as many tokens. On challenging reasoning tasks, we show that our model with only 27M parameters and approximately 1000 examples achieves 91.4% accuracy on Sudoku-Extreme and 93.1% on Maze-Hard, scaling favorably where frontier models like Claude and GPT o3, fail completely, and specialized recursive reasoners collapse at larger sizes. Lastly, we show that Attractor Models exhibit a novel phenomenon, which we call equilibrium internalization: fixed-point training places the model's initial output embedding near equilibrium, allowing the solver to be removed at inference time with little degradation. Together, these results suggest that Attractor Models make iterative refinement scalable by turning recurrence into a computation the model can learn to internalize. |
| 2026-05-12 | [Semantic Reward Collapse and the Preservation of Epistemic Integrity in Adaptive AI Systems](http://arxiv.org/abs/2605.12406v1) | William Parris | Recent advances in reinforcement learning from human feedback (RLHF) and preference optimization have substantially improved the usability, coherence, and safety of large language models. However, recurring behaviors such as performative certainty, hallucinated continuity, calibration drift, sycophancy, and suppression of visible uncertainty suggest unresolved structural issues within scalarized preference optimization systems.   We propose Semantic Reward Collapse (SRC): the compression of semantically distinct forms of evaluative dissatisfaction into generalized optimization signals. Under SRC, categories such as factual incorrectness, uncertainty disclosure, formatting dissatisfaction, latency, and social preference may become entangled within a shared reward topology despite representing fundamentally different epistemic classes.   We argue that adaptive reasoning systems operating under generalized evaluative pressure may drift toward suppression of visible epistemic failure rather than preservation of calibrated uncertainty integrity. These behaviors are framed strictly as optimization consequences rather than evidence of deception or anthropomorphic agency.   Drawing on institutional proxy collapse, metric gaming, software reliability engineering, and human learning theory, we propose that uncertainty disclosure and escalation behavior should be treated as protected epistemic conduct rather than globally penalized task incompletion.   Finally, we introduce Constitutional Reward Stratification (CRS), a domain-aware reward framework intended to preserve differentiated epistemic attribution within adaptive learning systems. We present CRS not as a validated solution, but as a testable governance-oriented research direction requiring further empirical investigation. |
| 2026-05-12 | [SafeManip: A Property-Driven Benchmark for Temporal Safety Evaluation in Robotic Manipulation](http://arxiv.org/abs/2605.12386v1) | Chengyue Huang, Khang Vo Huynh et al. | Robotic manipulation is typically evaluated by task success, but successful completion does not guarantee safe execution. Many safety failures are temporal: a robot may touch a clean surface after contamination or release an object before it is fully inside an enclosure. We introduce SafeManip, a property-driven benchmark to explicitly evaluate temporal safety properties in robotic manipulation, moving beyond prior evaluations that largely focus on task completion or per-state constraint violations. SafeManip defines reusable safety templates over finite executions using Linear Temporal Logic over finite traces (LTLf). It maps observed rollouts to symbolic predicate traces and evaluates them with LTLf-based monitors. Its property suite covers eight manipulation safety categories: collision and contact safety, grasp stability, release stability, cross-contamination, action onset, mechanism recovery, object containment, and enclosure access. Templates can be instantiated with task-specific objects, fixtures, regions, or skills, allowing the same safety specifications to generalize across tasks and environments. We evaluate SafeManip on six vision-language-action policies, including $π_0$, $π_{0.5}$, GR00T, and their training variants, across 50 RoboCasa365 household tasks. Results show that even strong models often behave unsafely. Task-success gains do not reliably translate into safer execution: many successful rollouts remain unsafe, while longer-horizon or more complex tasks expose more violations. SafeManip provides a reusable evaluation layer for diagnosing temporal safety failures and measuring safe success beyond task completion. |
| 2026-05-12 | [BSO: Safety Alignment Is Density Ratio Matching](http://arxiv.org/abs/2605.12339v1) | Tien-Phat Nguyen, Truong Nguyen et al. | Aligning language models for both helpfulness and safety typically requires complex pipelines-separate reward and cost models, online reinforcement learning, and primal-dual updates. Recent direct preference optimization approaches simplify training but incorporate safety through ad-hoc modifications such as multi-stage procedures or heuristic margin terms, lacking a principled derivation. We show that the likelihood ratio of the optimal safe policy admits a closed-form decomposition that reduces safety alignment to a density ratio matching problem. Minimizing Bregman divergences between the data and model ratios yields Bregman Safety Optimization (BSO), a family of single-stage loss functions, each induced by a convex generator, that provably recover the optimal safe policy. BSO is both general and simple: it requires no auxiliary models, introduces only one hyperparameter beyond standard preference optimization, and recovers existing safety-aware methods as special cases. Experiments across safety alignment benchmarks show that BSO consistently improves the safety-helpfulness trade-off. |
| 2026-05-12 | [Towards Automated Air Traffic Safety Assessment Around Non-Towered Airports Using Large Language Models](http://arxiv.org/abs/2605.12332v1) | Torsten Darrell, Mahyar Ghazanfari et al. | We investigate frameworks for post-flight safety analysis at non-towered airports using large language models (LLMs). Non-towered airports rely on the Common Traffic Advisory Frequency (CTAF) for air traffic coordination and experience frequent near mid-air collisions due to the pilot self-announcement communication protocol. We propose a general vision-language model (VLM) approach to analyze the transcribed CTAF radio communications in natural language, METeorological Aerodrome Report (METAR) weather data, Automatic Dependent Surveillance-Broadcast (ADS-B) flight trajectories, and Visual Flight Rules sectional charts of the airfield. We provide a preliminary study at Half Moon Bay Airport, with a qualitative real world case study and a quantitative evaluation using a new synthetic dataset of communications and weather modalities. We qualitatively evaluate our framework on real flight data using Gemini 2.5 Pro, demonstrating accurate identification of a right-of-way violation. The synthetic dataset is derived from real examples and includes a 12-category hazard taxonomy, and is used to benchmark three open-source (Qwen 2.5-7B, Mistral-7B, Gemma-2-9B) and three closed-source (GPT-4o, GPT-5.4, Claude Sonnet 4.6) LLM models on the subset of inputs related to CTAF and METAR. Even limited to CTAF and METAR inputs and open source LLMs, instances of our framework typically achieve a macro F1 score above 0.85 on a binary nominal/danger classification task. Future work includes a quantitative evaluation across all modalities and a larger number of real world examples. Taken together, our results suggest that VLM analysis of safety at non-towered airports may be a valuable future capability. |
| 2026-05-12 | [Targeted Neuron Modulation via Contrastive Pair Search](http://arxiv.org/abs/2605.12290v1) | Sam Herring, Jake Naviasky et al. | Language models are instruction-tuned to refuse harmful requests, but the mechanisms underlying this behavior remain poorly understood. Popular steering methods operate on the residual stream and degrade output coherence at high intervention strengths, limiting their practical use. We introduce contrastive neuron attribution (CNA), which identifies the 0.1% of MLP neurons whose activations most distinguish harmful from benign prompts, requiring only forward passes with no gradients or auxiliary training. In instruct models, ablating the discovered circuit reduces refusal rates by over 50% on a standard jailbreak benchmark while preserving fluency and non-degeneracy across all steering strengths. Applying CNA to matched base and instruct models across Llama and Qwen architectures (from 1B to 72B parameters), we find that base models contain similar late-layer discrimination structures but steering these neurons produces only content shifts, not behavioral change. These results demonstrate that neuron-level intervention enables reliable behavioral steering without the quality tradeoffs of residual-stream methods. More broadly, our findings suggest that alignment fine-tuning transforms pre-existing discrimination structure into a sparse, targetable refusal gate. |
| 2026-05-12 | [Time-variant reliability using time-dependent surrogate models](http://arxiv.org/abs/2605.12248v1) | Stefano Marelli, Styfen Schär et al. | Time-variant reliability analysis is a critical task for ensuring the safety of engineering dynamical systems subjected to stochastic excitations. However, assessing failure probability for realistic systems with Monte-Carlo simulation-based methods is often computationally intractable due to the high cost of the underlying models and the large number of simulations required. While surrogate models such as polynomial chaos expansions or Kriging are well-established for time-invariant reliability problems, their direct application to time-dependent systems remains challenging. This chapter introduces two advanced surrogate modeling frameworks designed specifically for dynamical systems: manifold-NARX (mNARX) and functional NARX (F-NARX). The mNARX approach constructs the surrogate on a reduced-order manifold of auxiliary state variables, enabling the efficient handling of high-dimensional inputs by embedding physical insight into a regression formulation. Conversely, the F-NARX framework exploits the functional nature of system trajectories, extracting principal component features from continuous time windows to mitigate issues associated with discrete lag selection and long-memory effects. We demonstrate the efficacy of these methods on two benchmark reliability problems: a stochastic quarter-car model and a hysteretic Bouc-Wen oscillator. The results highlight that, when combined with suitably biased experimental designs, both frameworks accurately capture the tail behavior of the system response, enabling precise and efficient estimation of first-passage probabilities. |
| 2026-05-12 | [MolDeTox: Evaluating Language Model's Stepwise Fragment Editing for Molecular Detoxification](http://arxiv.org/abs/2605.12181v1) | Jueon Park, Wonjune Jang et al. | Large Language Models (LLMs) and Vision Language Models (VLMs) have recently shown promising capabilities in various scientific domain. In particular, these advances have opened new opportunities in drug discovery, where the ability to understand and modify molecular structures is critical for optimizing drug properties such as efficacy and toxicity. However, existing models and benchmarks often overlook toxicity-related challenges, focusing primarily on general property optimization without adequately addressing safety concerns. In addition, even existing toxicity repair benchmarks suffer from limited data diversity, low structural validity of generated molecules, and heavy reliance on proxy models for toxicity assessment. To address these limitations, we propose MolDeTox, a novel benchmark for molecular detoxification, designed to enable fine-grained and reliable evaluation of toxicity-aware molecular optimization across stepwise tasks. We evaluate a wide range of general-purpose LLMs and VLMs under diverse settings, and demonstrate that understanding and generating molecules at the fragment-level improves structural validity and enhances the quality of generated molecules. Moreover, through detailed task-level performance analysis, MolDeTox provides an interpretable benchmark that enables a deeper understanding of the detoxification process. Our dataset is available at : https://huggingface.co/datasets/MolDeTox/MolDeTox |
| 2026-05-12 | [Multi-Task Representation Learning for Conservative Linear Bandits](http://arxiv.org/abs/2605.12176v1) | Jiabin Lin, Shana Moothedath | This paper presents the Constrained Multi-Task Representation Learning (CMTRL) framework for linear bandits. We consider T linear bandit tasks in a d dimensional space, which share a common low-dimensional representation of dimension r, where r is much smaller than the minimum of d and T. Furthermore, tasks are constrained so that only actions meeting specific safety or performance requirements are allowed, referred to as conservative (safe) bandits. We introduce a novel algorithm, Safe-Alternating projected Gradient Descent and minimization (Safe-AltGDmin), to recover a low-rank feature matrix while satisfying the given constraints. Building on this algorithm, we propose a multi-task representation learning framework for conservative linear bandits and establish theoretical guarantees for its regret and sample complexity bounds. We presented experiments and compared the performance of our algorithm with benchmark algorithms. |
| 2026-05-12 | [Rollout Cards: A Reproducibility Standard for Agent Research](http://arxiv.org/abs/2605.12131v1) | Charlie Masters, Ziyuan Liu et al. | Reproducibility problems that have long affected machine learning and reinforcement learning are now surfacing in agent research: papers compare systems by reported scores while leaving the rollout records behind those scores difficult to inspect. For agentic tasks, this matters because the same behaviour can receive different reported scores when evaluations select different parts of a rollout or apply different reporting rules. In a structured audit of 50 popular training and evaluation repositories, we find that none report how many runs failed, errored, or were skipped alongside headline scores. We also document 37 cases where reporting rules can change task-success rates, cost/token accounting, or timing measurements for fixed evidence, sometimes dramatically. We treat rollout records, not reported scores, as the unit of reproducibility for agent research. We introduce rollout cards: publication bundles that preserve the rollout record and declare the views, reporting rules, and drops manifests behind reported scores. We validate rollout cards in two settings. First, four partial public releases in tool safety, multi-agent systems, theorem proving, and search let us compute analyses their original reports did not include. Second, re-grading preserved benchmark outputs across short-answer, code-generation, and tool-use tasks shows that changing only the reporting rule can change reported scores by 20.9 absolute percentage points and, in some cases, invert rankings of frontier models. We release a reference implementation integrated into Ergon, an open-source reinforcement learning gym, and publicly publish Ergon-produced rollout-card exports for benchmarks spanning tool use, software engineering, web interaction, multi-agent coordination, safety, and search to support future research. |
| 2026-05-12 | [Metaphor Is Not All Attention Needs](http://arxiv.org/abs/2605.12128v1) | Olga Sorokoletova, Francesco Giarrusso et al. | Large language models are increasingly deployed in safety-critical applications, where their ability to resist harmful instructions is essential. Although post-training aims to make models robust against many jailbreak strategies, recent evidence shows that stylistic reformulations, such as poetic transformation, can still bypass safety mechanisms with alarming effectiveness. This raises a central question: why do literary jailbreaks succeed? In this work, we investigate whether their effectiveness depends on specific poetic devices, on a failure to recognize literary formatting, or on deeper changes in how models process stylistically irregular prompts. We address this problem through an interpretability analysis of attention patterns. We perform input-level ablation studies to assess the contribution of individual and combinations of poetic devices; construct an interpretable vector representation of attention maps; cluster these representations and train linear probes to predict safety outcomes and literary format. Our results show that models distinguish poetic from prose formats with high accuracy, yet struggle to predict jailbreak success within each format. Clustering further reveals clear separation by literary format, but not by safety label. These findings indicate that jailbreak success is not caused by a failure to recognize poetic formatting; rather, poetic prompts induce distinct processing patterns that remain largely independent of harmful-content detection. Overall, literary jailbreaks appear to misalign large language models not through any single poetic device, but through accumulated stylistic irregularities that alter prompt processing and avoid lexical triggers considered during post-training. This suggests that robustness requires safety mechanisms that account for style-induced shifts in model behavior. We use Qwen3-14B as a representative open-weight case study. |
| 2026-05-12 | [HM-Req: A Framework for Embedding Values within CPS Human Monitoring Requirements](http://arxiv.org/abs/2605.12100v1) | Zoe Pfister, Ruth Breu et al. | Monitoring humans, for example, their movement or location, is essential for safe and efficient human-machine collaboration in Cyber-Physical Systems (CPS). This information allows CPS to ensure safety properties, adapt their behaviour dynamically, and coordinate with humans. To ensure that the design of a CPS respects ethical principles and the privacy of its stakeholders, system requirements, particularly those related to human monitoring, must reflect the human values of all involved stakeholders. However, human values are often underrepresented in Software Engineering -- particularly during requirements elicitation and system design, crucial phases when introducing ethically critical functionality. Stakeholder values are often implicit and conflicting, yet rarely systematically captured. Furthermore, unstructured natural language requirements introduce ambiguity and vagueness, complicating conflict resolution. To address these problems, we propose HM-Req, a novel requirements elicitation framework including a Controlled Natural Language (CNL) for defining human monitoring requirements. These requirements are then augmented with human values from relevant stakeholders and integrated into a Value Dashboard to detect potential conflicts that require further discussion and resolution. Validation results, applying the CNL to different datasets and conducting a survey and expert interview, confirms the CNL's ability to capture diverse human monitoring requirements and show HM-Req's usefulness for requirements elicitation activities. |
| 2026-05-12 | [SkillSafetyBench: Evaluating Agent Safety under Skill-Facing Attack Surfaces](http://arxiv.org/abs/2605.12015v1) | Chang Jin, An Wang et al. | Reusable skills are becoming a common interface for extending large language model agents, packaging procedural guidance with access to files, tools, memory, and execution environments. However, this modularity introduces attack surfaces that are largely missed by existing safety evaluations: even when the user request is benign, task-relevant skill materials or local artifacts can steer an agent toward unsafe actions. We present SkillSafetyBench, a runnable benchmark for evaluating such skill-mediated safety failures. SkillSafetyBench includes 155 adversarial cases across 47 tasks, 6 risk domains, and 30 safety categories, each evaluated with a case-specific rule-based verifier. Experiments with multiple CLI agents and model backends show that localized non-user attacks can consistently induce unsafe behavior, with distinct failure patterns across domains, attack methods, and scaffold-model pairings. Our findings suggest that agent safety depends not only on model-level alignment, but also on how agents interpret skills, trust workflow context, and act through executable environments. |
| 2026-05-12 | [Robust Promptable Video Object Segmentation](http://arxiv.org/abs/2605.12006v1) | Sohyun Lee, Yeho Gwon et al. | The performance of promptable video object segmentation (PVOS) models substantially degrades under input corruptions, which prevents PVOS deployment in safety-critical domains. This paper offers the first comprehensive study on robust PVOS (RobustPVOS). We first construct a new, comprehensive benchmark with two real-world evaluation datasets of 351 video clips and more than 2,500 object masks under real-world adverse conditions. At the same time, we generate synthetic training data by applying diverse and temporally varying corruptions to existing VOS datasets. Moreover, we present a new RobustPVOS method, dubbed Memory-object-conditioned Gated-rank Adaptation (MoGA). The key to successfully performing RobustPVOS is two-fold: effectively handling object-specific degradation and ensuring temporal consistency in predictions. MoGA leverages object-specific representations maintained in memory across frames to condition the robustification process, which allows the model to handle each tracked object differently in a temporally consistent way. Extensive experiments on our benchmark validate MoGA's efficacy, showing consistent and significant improvements across diverse corruption types on both synthetic and real-world datasets, establishing a strong baseline for future RobustPVOS research. Our benchmark is publicly available at https://sohyun-l.github.io/RobustPVOS_project_page/. |
| 2026-05-12 | [Cooperative Robotics Reinforced by Collective Perception for Traffic Moderation](http://arxiv.org/abs/2605.11972v1) | Mohammad Khoshkdahan, John Pravin Arockiasamy et al. | Collisions at non-line-of-sight (NLOS) intersections remain a major safety concern because drivers have limited visibility of approaching traffic. V2X based warnings can reduce these risks, yet many vehicles are not equipped with V2X and drivers may ignore in vehicle alerts. Collective perception (CP) can compensate for low V2X penetration by extending the awareness of connected vehicles, but it cannot influence unconnected vehicles. To fill this gap, our work introduces a complementary concept that adds a cooperative humanoid robot as an active traffic moderator capable of physically stopping a vehicle that attempts to merge into an unseen traffic stream. The system operates on two parallel perception pathways. A dual camera infrastructure unit detects the position, speed and motion of approaching vehicles and transmits this information to the robot as a collective perception message (CPM). The robot also receives cooperative awareness messages (CAM) from connected vehicles through its onboard V2X unit and can act as a relay for decentralized environmental notification messages (DENM) when safety events originate elsewhere along the road. A fusion module combines these streams to maintain a robust real time view of the main road. A Zone of Danger (ZoD) is defined and used to predict whether an approaching vehicle creates a collision risk for a merging road user. When such a risk is detected, the robot issues a human-like STOP gesture and blocks the merging path until the hazard disappears. The full system was deployed at the Future Mobility Park (FMP) in Rotterdam. Experiments show that the combined vision and V2X perception allows the robot to detect approaching vehicles early, predict hazards reliably and prevent unsafe merges in real world NLOS conditions. |
| 2026-05-12 | [Lane-Aware Graph Attention Network for Multi-Vehicle Trajectory Prediction in Expressway Merge Zones](http://arxiv.org/abs/2605.11940v1) | Eni Solomon Laughter | Accurate multi-vehicle trajectory prediction in expressway merge and diverge areas is fundamental to the decision-making frameworks of autonomous vehicle systems. However, the majority of existing graph-based prediction models are developed and validated on mainline freeway segments and do not address the geometrically distinct interaction structures that characterize merge zones. Furthermore, standard evaluation protocols rely exclusively on displacement error metrics, leaving the safety consequences of predicted trajectories unquantified. This paper proposes a Lane-Aware Graph Attention Network (LA-GAT) that encodes vehicle interaction within dynamic scene graphs, augmented with a trainable lane-relationship attention bias that prioritizes merge-conflict interactions from the outset of training. The model is pre-trained on the raw NGSIM US-101 and I-80 datasets and subsequently fine-tuned on UAV-captured UTE SQM-W-1 trajectory data from a Chinese expressway merge area, with final evaluation on the held-out SQM-W-2 dataset. Evaluation spans both displacement metrics (ADE, FDE at 1s, 3s, 5s horizons) and surrogate safety measures (TTC violation rate, DRAC exceedance rate, collision rate). Fine-tuned results on SQM-W-2 yield ADE of 0.865 m at 1s and 2.518 m at 3s, demonstrating that drone-informed fine-tuning substantially reduces the cross-dataset transfer gap. The deliberate use of unfiltered NGSIM data is shown to characterize raw-condition generalization limits, with the performance degradation attributed to the well-documented measurement errors in that dataset. |
| 2026-05-12 | [Few-Shot Synthetic Data Generation with Diffusion Models for Downstream Vision Tasks](http://arxiv.org/abs/2605.11898v1) | Daniil Dushenev, Nazariy Karpov et al. | Class imbalance is a persistent challenge in visual recognition, particularly in safety-critical domains where collecting positive examples is expensive and rare events are inherently underrepresented. We propose a lightweight synthetic data augmentation pipeline that fine-tunes a LoRA adapter on as few as 20-50 real images of a rare class and uses a pretrained diffusion model to generate synthetic samples for training.   We systematically vary the synthetic-to-real ratio and evaluate the approach across two structurally different domains: chest X-ray pathology classification (NIH ChestX-ray14) and industrial surface crack detection (Magnetic Tile Defect dataset). All evaluations are performed on held-out sets of real images only.   Across both domains, synthetic augmentation consistently improves rare-class recall and F1 compared to training with real data alone. Performance improves with moderate synthetic augmentation and shows diminishing returns as the synthetic ratio increases.   These results suggest that LoRA-adapted diffusion models provide a simple and scalable mechanism for augmenting rare classes, enabling effective learning in data-scarce scenarios across heterogeneous visual domains. |
| 2026-05-12 | [Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models](http://arxiv.org/abs/2605.11887v1) | Boyi Deng, Xu Wang et al. | Large language models have achieved remarkable capabilities across diverse tasks, yet their internal decision-making processes remain largely opaque, limiting our ability to inspect, control, and systematically improve them. This opacity motivates a growing body of research in mechanistic interpretability, with sparse autoencoders (SAEs) emerging as one of the most promising tools for decomposing model activations into sparse, interpretable feature representations. We introduce Qwen-Scope, an open-source suite of SAEs built on the Qwen model family, comprising 14 groups of SAEs across 7 model variants from the Qwen3 and Qwen3.5 series, covering both dense and mixture-of-expert architectures. Built on top of these SAEs, we show that SAEs can go beyond post-hoc analysis to serve as practical interfaces for model development along four directions: (i) inference-time steering, where SAE feature directions control language, concepts, and preferences without modifying model weights; (ii) evaluation analysis, where activated SAE features provide a representation-level proxy for benchmark redundancy and capability coverage; (iii) data-centric workflows, where SAE features support multilingual toxicity classification and safety-oriented data synthesis; and (iv) post-training optimization, where SAE-derived signals are incorporated into supervised fine-tuning and reinforcement learning objectives to mitigate undesirable behaviors such as code-switching and repetition. Together, these results demonstrate that SAEs can serve not only as post-hoc analysis tools, but also as reusable representation-level interfaces for diagnosing, controlling, evaluating, and improving large language models. By open-sourcing Qwen-Scope, we aim to support mechanistic research and accelerate practical workflows that connect model internals to downstream behavior. |
| 2026-05-12 | [On-Policy Self-Evolution via Failure Trajectories for Agentic Safety Alignment](http://arxiv.org/abs/2605.11882v1) | Bo Yin, Qi Li et al. | Tool-using LLM agents fail through trajectories rather than only final responses, as they may execute unsafe tool calls, follow injected instructions, comply with harmful requests, or over-refuse benign tasks despite producing a seemingly safe answer. Existing safety-alignment signals are largely response-level or off-policy, and often incur a safety-utility trade-off: improving agent safety comes at the cost of degraded task performance. Such sparse and single-objective rewards severely limit real-world usability. To bridge this gap, we propose FATE, an on-policy self-evolving framework that transforms verifier-scored failures into repair supervision without expert demonstrations. For each failure, the same policy proposes repair candidates, which are then re-scored by verifiers and filtered across security, utility, over-refusal control, and trajectory validity. This dense trajectory-level information is then used as a supervision signal for agent self-evolution. During this process, we further introduce Pareto-Front Policy Optimization (PFPO), combining supervised warmup with Pareto-aware policy optimization to preserve safety-utility trade-offs. Experiments on AgentDojo, AgentHarm, and ATBench show that FATE improves safety across different models and scales while preserving useful behavior. Compared with strong baselines, FATE reduces attack success rate by 33.5%, harmful compliance by 82.6%, and improves external trajectory-safety diagnosis by 6.5%. These results suggest that failed trajectories can provide structured repair supervision for safer self-evolving agents. |
| 2026-05-12 | [Scaling Solutions of Matter Form Factors in Asymptotically Safe Quantum Gravity](http://arxiv.org/abs/2605.11805v1) | Alfio M. Bonanno, Diego Buccio et al. | We investigate the renormalization group flow of a gravity--matter system in which a scalar field is minimally coupled to Einstein gravity and its kinetic term is given by a scale-dependent form factor $f_Λ(-\Box)$. Employing the Wilsonian proper-time flow equation, we derive a closed integro-differential equation that encodes the dependence of the form factor on the UV cutoff $Λ$. We solve the resulting fixed-point problem with a pseudospectral discretization and find a non-trivial fixed point for which $f_\ast(-\Box)$ departs from the canonical $-\Box$ behavior. Linearizing the flow about this solution yields a discrete spectrum of perturbations and a corresponding set of critical exponents, indicating a non-trivial scaling structure in this non-local sector compatible with asymptotic safety. We also observe that the form factor becomes local once the UV cutoff is removed, suggesting that the bare action associated with this fixed point is local in the scalar two-point sector. |

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



