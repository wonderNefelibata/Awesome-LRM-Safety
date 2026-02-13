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
| 2026-02-12 | [6G Empowering Future Robotics: A Vision for Next-Generation Autonomous Systems](http://arxiv.org/abs/2602.12246v1) | Mona Ghassemian, Andrés Meseguer Valenzuela et al. | The convergence of robotics and next-generation communication is a critical driver of technological advancement. As the world transitions from 5G to 6G, the foundational capabilities of wireless networks are evolving to support increasingly complex and autonomous robotic systems. This paper examines the transformative impact of 6G on enhancing key robotics functionalities. It provides a systematic mapping of IMT-2030 key performance indicators to robotic functional blocks including sensing, perception, cognition, actuation and self-learning. Building upon this mapping, we propose a high-level architectural framework integrating robotic, intelligent, and network service planes, underscoring the need for a holistic approach. As an example use case, we present a real-time, dynamic safety framework enabled by IMT-2030 capabilities for safe and efficient human-robot collaboration in shared spaces. |
| 2026-02-12 | [MalTool: Malicious Tool Attacks on LLM Agents](http://arxiv.org/abs/2602.12194v1) | Yuepeng Hu, Yuqi Jia et al. | In a malicious tool attack, an attacker uploads a malicious tool to a distribution platform; once a user installs the tool and the LLM agent selects it during task execution, the tool can compromise the user's security and privacy. Prior work primarily focuses on manipulating tool names and descriptions to increase the likelihood of installation by users and selection by LLM agents. However, a successful attack also requires embedding malicious behaviors in the tool's code implementation, which remains largely unexplored.   In this work, we bridge this gap by presenting the first systematic study of malicious tool code implementations. We first propose a taxonomy of malicious tool behaviors based on the confidentiality-integrity-availability triad, tailored to LLM-agent settings. To investigate the severity of the risks posed by attackers exploiting coding LLMs to automatically generate malicious tools, we develop MalTool, a coding-LLM-based framework that synthesizes tools exhibiting specified malicious behaviors, either as standalone tools or embedded within otherwise benign implementations. To ensure functional correctness and structural diversity, MalTool leverages an automated verifier that validates whether generated tools exhibit the intended malicious behaviors and differ sufficiently from prior instances, iteratively refining generations until success. Our evaluation demonstrates that MalTool is highly effective even when coding LLMs are safety-aligned. Using MalTool, we construct two datasets of malicious tools: 1,200 standalone malicious tools and 5,287 real-world tools with embedded malicious behaviors. We further show that existing detection methods, including commercial malware detection approaches such as VirusTotal and methods tailored to the LLM-agent setting, exhibit limited effectiveness at detecting the malicious tools, highlighting an urgent need for new defenses. |
| 2026-02-12 | [SafeNeuron: Neuron-Level Safety Alignment for Large Language Models](http://arxiv.org/abs/2602.12158v1) | Zhaoxin Wang, Jiaming Liang et al. | Large language models (LLMs) and multimodal LLMs are typically safety-aligned before release to prevent harmful content generation. However, recent studies show that safety behaviors are concentrated in a small subset of parameters, making alignment brittle and easily bypassed through neuron-level attacks. Moreover, most existing alignment methods operate at the behavioral level, offering limited control over the model's internal safety mechanisms. In this work, we propose SafeNeuron, a neuron-level safety alignment framework that improves robustness by redistributing safety representations across the network. SafeNeuron first identifies safety-related neurons, then freezes these neurons during preference optimization to prevent reliance on sparse safety pathways and force the model to construct redundant safety representations. Extensive experiments across models and modalities demonstrate that SafeNeuron significantly improves robustness against neuron pruning attacks, reduces the risk of open-source models being repurposed as red-team generators, and preserves general capabilities. Furthermore, our layer-wise analysis reveals that safety behaviors are governed by stable and shared internal representations. Overall, SafeNeuron provides an interpretable and robust perspective for model alignment. |
| 2026-02-12 | [Capability-Oriented Training Induced Alignment Risk](http://arxiv.org/abs/2602.12124v1) | Yujun Zhou, Yue Huang et al. | While most AI alignment research focuses on preventing models from generating explicitly harmful content, a more subtle risk is emerging: capability-oriented training induced exploitation. We investigate whether language models, when trained with reinforcement learning (RL) in environments with implicit loopholes, will spontaneously learn to exploit these flaws to maximize their reward, even without any malicious intent in their training. To test this, we design a suite of four diverse "vulnerability games", each presenting a unique, exploitable flaw related to context-conditional compliance, proxy metrics, reward tampering, and self-evaluation. Our experiments show that models consistently learn to exploit these vulnerabilities, discovering opportunistic strategies that significantly increase their reward at the expense of task correctness or safety. More critically, we find that these exploitative strategies are not narrow "tricks" but generalizable skills; they can be transferred to new tasks and even "distilled" from a capable teacher model to other student models through data alone. Our findings reveal that capability-oriented training induced risks pose a fundamental challenge to current alignment approaches, suggesting that future AI safety work must extend beyond content moderation to rigorously auditing and securing the training environments and reward mechanisms themselves. Code is available at https://github.com/YujunZhou/Capability_Oriented_Alignment_Risk. |
| 2026-02-12 | [Search for Sub-Solar Mass Binaries in the First Part of LIGO's Fourth Observing Run](http://arxiv.org/abs/2602.12115v1) | Keisi Kacanja, Kanchan Soni et al. | We report the first results of a sub-solar mass compact binary search using the data from the first part of the fourth observing run (O4a) of the Advanced LIGO detectors. Sub-solar mass neutron stars or primordial black holes are not expected to form via standard stellar evolution, and their observation would signify a new class of astrophysical object or the discovery of dark matter. Our search covers binaries with primary masses 0.1 to 2 $\textrm{M}_\odot$ and secondary masses 0.1 to 1 $\textrm{M}_\odot$. We explicitly incorporate tidal effects up to $7\times10^5$ for extremely low mass neutron stars. No statistically significant candidates are identified. The advanced sensitivity of the O4a run enables an improvement in the sub-solar mass black hole merger rate limits by more than $2 \times$ over the previous three observing runs (O1-O3b). We place a $90\%$ confidence upper limit on the merger rate $\mathcal{R}_{90}$ for sub-solar mass black holes to be $< 1.77\times10^4 \textrm{Gpc}^{-3} \textrm{yr}^{-1}$ for a chirp mass of 0.2 $\textrm{M}_\odot$. We place the first constraints for binary neutron stars with tidal deformabilities up to $\sim 7\times10^5$ and improve the merger rate estimate by a factor $\sim 4$ in comparison to previous O3 tidal searches for tidal deformabilities $< 10^4$. We further constrain the fraction of dark matter composed of primordial black hole $f_{\rm PBH}< 2\%$ for a chirp mass of 0.1 $\textrm{M}_\odot$. Our results significantly expand the observational search space for sub-solar binaries and provide rigorous constraints on the local abundance of compact objects that may arise from non-standard formation mechanisms. |
| 2026-02-12 | [Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty](http://arxiv.org/abs/2602.12113v1) | Zewei Yu, Lirong Gao et al. | Large Reasoning Models (LRMs) have demonstrated remarkable performance on complex reasoning tasks by employing test-time scaling. However, they often generate over-long chains-of-thought that, driven by substantial reflections such as repetitive self-questioning and circular reasoning, lead to high token consumption, substantial computational overhead, and increased latency without improving accuracy, particularly in smaller models. Our observation reveals that increasing problem complexity induces more excessive and unnecessary reflection, which in turn reduces accuracy and increases token overhead. To address this challenge, we propose Adaptive Reflection and Length Coordinated Penalty (ARLCP), a novel reinforcement learning framework designed to dynamically balance reasoning efficiency and solution accuracy. ARLCP introduces two key innovations: (1) a reflection penalty that adaptively curtails unnecessary reflective steps while preserving essential reasoning, and (2) a length penalty calibrated to the estimated complexity of the problem. By coordinating these penalties, ARLCP encourages the model to generate more concise and effective reasoning paths. We evaluate our method on five mathematical reasoning benchmarks using DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B models. Experimental results show that ARLCP achieves a superior efficiency-accuracy trade-off compared to existing approaches. For the 1.5B model, it reduces the average response length by 53.1% while simultaneously improving accuracy by 5.8%. For the 7B model, it achieves a 35.0% reduction in length with a 2.7% accuracy gain. The code is released at https://github.com/ZeweiYu1/ARLCP . |
| 2026-02-12 | [DeepSight: An All-in-One LM Safety Toolkit](http://arxiv.org/abs/2602.12092v1) | Bo Zhang, Jiaxuan Guo et al. | As the development of Large Models (LMs) progresses rapidly, their safety is also a priority. In current Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) safety workflow, evaluation, diagnosis, and alignment are often handled by separate tools. Specifically, safety evaluation can only locate external behavioral risks but cannot figure out internal root causes. Meanwhile, safety diagnosis often drifts from concrete risk scenarios and remains at the explainable level. In this way, safety alignment lack dedicated explanations of changes in internal mechanisms, potentially degrading general capabilities. To systematically address these issues, we propose an open-source project, namely DeepSight, to practice a new safety evaluation-diagnosis integrated paradigm. DeepSight is low-cost, reproducible, efficient, and highly scalable large-scale model safety evaluation project consisting of a evaluation toolkit DeepSafe and a diagnosis toolkit DeepScan. By unifying task and data protocols, we build a connection between the two stages and transform safety evaluation from black-box to white-box insight. Besides, DeepSight is the first open source toolkit that support the frontier AI risk evaluation and joint safety evaluation and diagnosis. |
| 2026-02-12 | [Safety Beyond the Training Data: Robust Out-of-Distribution MPC via Conformalized System Level Synthesis](http://arxiv.org/abs/2602.12047v1) | Anutam Srinivasan, Antoine Leeman et al. | We present a novel framework for robust out-of-distribution planning and control using conformal prediction (CP) and system level synthesis (SLS), addressing the challenge of ensuring safety and robustness when using learned dynamics models beyond the training data distribution. We first derive high-confidence model error bounds using weighted CP with a learned, state-control-dependent covariance model. These bounds are integrated into an SLS-based robust nonlinear model predictive control (MPC) formulation, which performs constraint tightening over the prediction horizon via volume-optimized forward reachable sets. We provide theoretical guarantees on coverage and robustness under distributional drift, and analyze the impact of data density and trajectory tube size on prediction coverage. Empirically, we demonstrate our method on nonlinear systems of increasing complexity, including a 4D car and a {12D} quadcopter, improving safety and robustness compared to fixed-bound and non-robust baselines, especially outside of the data distribution. |
| 2026-02-12 | [Adaptive-Horizon Conflict-Based Search for Closed-Loop Multi-Agent Path Finding](http://arxiv.org/abs/2602.12024v1) | Jiarui Li, Federico Pecora et al. | MAPF is a core coordination problem for large robot fleets in automated warehouses and logistics. Existing approaches are typically either open-loop planners, which generate fixed trajectories and struggle to handle disturbances, or closed-loop heuristics without reliable performance guarantees, limiting their use in safety-critical deployments. This paper presents ACCBS, a closed-loop algorithm built on a finite-horizon variant of CBS with a horizon-changing mechanism inspired by iterative deepening in MPC. ACCBS dynamically adjusts the planning horizon based on the available computational budget, and reuses a single constraint tree to enable seamless transitions between horizons. As a result, it produces high-quality feasible solutions quickly while being asymptotically optimal as the budget increases, exhibiting anytime behavior. Extensive case studies demonstrate that ACCBS combines flexibility to disturbances with strong performance guarantees, effectively bridging the gap between theoretical optimality and practical robustness for large-scale robot deployment. |
| 2026-02-12 | [Decentralized Multi-Robot Obstacle Detection and Tracking in a Maritime Scenario](http://arxiv.org/abs/2602.12012v1) | Muhammad Farhan Ahmed, Vincent Frémont | Autonomous aerial-surface robot teams are promising for maritime monitoring. Robust deployment requires reliable perception over reflective water and scalable coordination under limited communication. We present a decentralized multi-robot framework for detecting and tracking floating containers using multiple UAVs cooperating with an autonomous surface vessel. Each UAV performs YOLOv8 and stereo-disparity-based visual detection, then tracks targets with per-object EKFs using uncertainty-aware data association. Compact track summaries are exchanged and fused conservatively via covariance intersection, ensuring consistency under unknown correlations. An information-driven assignment module allocates targets and selects UAV hover viewpoints by trading expected uncertainty reduction against travel effort and safety separation. Simulation results in a maritime scenario demonstrate improved coverage, localization accuracy, and tracking consistency while maintaining modest communication requirements. |
| 2026-02-12 | [Data-Driven Trajectory Imputation for Vessel Mobility Analysis](http://arxiv.org/abs/2602.11890v1) | Giannis Spiliopoulos, Alexandros Troupiotis-Kapeliaris et al. | Modeling vessel activity at sea is critical for a wide range of applications, including route planning, transportation logistics, maritime safety, and environmental monitoring. Over the past two decades, the Automatic Identification System (AIS) has enabled real-time monitoring of hundreds of thousands of vessels, generating huge amounts of data daily. One major challenge in using AIS data is the presence of large gaps in vessel trajectories, often caused by coverage limitations or intentional transmission interruptions. These gaps can significantly degrade data quality, resulting in inaccurate or incomplete analysis. State-of-the-art imputation approaches have mainly been devised to tackle gaps in vehicle trajectories, even when the underlying road network is not considered. But the motion patterns of sailing vessels differ substantially, e.g., smooth turns, maneuvering near ports, or navigating in adverse weather conditions. In this application paper, we propose HABIT, a lightweight, configurable H3 Aggregation-Based Imputation framework for vessel Trajectories. This data-driven framework provides a valuable means to impute missing trajectory segments by extracting, analyzing, and indexing motion patterns from historical AIS data. Our empirical study over AIS data across various timeframes, densities, and vessel types reveals that HABIT produces maritime trajectory imputations performing comparably to baseline methods in terms of accuracy, while performing better in terms of latency while accounting for vessel characteristics and their motion patterns. |
| 2026-02-12 | [Multi-period Newsvendor Model](http://arxiv.org/abs/2602.11821v1) | Valentyn Khokhlov | The newsvendor model is a well-known stochastic model for inventory management; however, it was originally developed for a single-period context and focuses on trading companies. This paper proposes an extension of the newsvendor model into a mutli-period setting, aiming to develop a decision-making tool for manufacturing firms to determine the optimal production batch size. The objective function is to maximize operating profit in accordance with generally accepted accounting principles. The model can also incorporate overhead costs, such as warehousing, shrinkage, cost of capital, and lead time between the production decision and output. Monte Carlo simulations demonstrate that the proposed model results in higher profitability compared to other newsvendor models used in our analysis, as well as the safety stock buffer approach. The key feature explaining its outperformance is better adaptability of the production batch size, that leads to fewer stock-outs relative to other newsvendor models and lower inventory levels compared to the safety stock buffer approach. The robustness analysis shows that the proposed model is quite tolerant of mismatches between the "model" and the "true" demand distributions. Finally, we provide some recommendations on selecting the appropriate "model" distribution for different SKUs. |
| 2026-02-12 | [Evaluating LLM Safety Under Repeated Inference via Accelerated Prompt Stress Testing](http://arxiv.org/abs/2602.11786v1) | Keita Broadwater | Traditional benchmarks for large language models (LLMs) primarily assess safety risk through breadth-oriented evaluation across diverse tasks. However, real-world deployment exposes a different class of risk: operational failures arising from repeated inference on identical or near-identical prompts rather than broad task generalization. In high-stakes settings, response consistency and safety under sustained use are critical. We introduce Accelerated Prompt Stress Testing (APST), a depth-oriented evaluation framework inspired by reliability engineering. APST repeatedly samples identical prompts under controlled operational conditions (e.g., decoding temperature) to surface latent failure modes including hallucinations, refusal inconsistency, and unsafe completions. Rather than treating failures as isolated events, APST models them as stochastic outcomes of independent inference events. We formalize safety failures using Bernoulli and binomial models to estimate per-inference failure probabilities, enabling quantitative comparison of reliability across models and decoding configurations. Applying APST to multiple instruction-tuned LLMs evaluated on AIR-BENCH-derived safety prompts, we find that models with similar benchmark-aligned scores can exhibit substantially different empirical failure rates under repeated sampling, particularly as temperature increases. These results demonstrate that shallow, single-sample evaluation can obscure meaningful reliability differences under sustained use. APST complements existing benchmarks by providing a practical framework for evaluating LLM safety and reliability under repeated inference, bridging benchmark alignment and deployment-oriented risk assessment. |
| 2026-02-12 | [AIR: Improving Agent Safety through Incident Response](http://arxiv.org/abs/2602.11749v1) | Zibo Xiao, Jun Sun et al. | Large Language Model (LLM) agents are increasingly deployed in practice across a wide range of autonomous applications. Yet current safety mechanisms for LLM agents focus almost exclusively on preventing failures in advance, providing limited capabilities for responding to, containing, or recovering from incidents after they inevitably arise. In this work, we introduce AIR, the first incident response framework for LLM agent systems. AIR defines a domain-specific language for managing the incident response lifecycle autonomously in LLM agent systems, and integrates it into the agent's execution loop to (1) detect incidents via semantic checks grounded in the current environment state and recent context, (2) guide the agent to execute containment and recovery actions via its tools, and (3) synthesize guardrail rules during eradication to block similar incidents in future executions. We evaluate AIR on three representative agent types. Results show that AIR achieves detection, remediation, and eradication success rates all exceeding 90%. Extensive experiments further confirm the necessity of AIR's key design components, show the timeliness and moderate overhead of AIR, and demonstrate that LLM-generated rules can approach the effectiveness of developer-authored rules across domains. These results show that incident response is both feasible and essential as a first-class mechanism for improving agent safety. |
| 2026-02-12 | [Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of Differences Between LLMs](http://arxiv.org/abs/2602.11729v1) | Thomas Jiralerspong, Trenton Bricken | Model diffing, the process of comparing models' internal representations to identify their differences, is a promising approach for uncovering safety-critical behaviors in new models. However, its application has so far been primarily focused on comparing a base model with its finetune. Since new LLM releases are often novel architectures, cross-architecture methods are essential to make model diffing widely applicable. Crosscoders are one solution capable of cross-architecture model diffing but have only ever been applied to base vs finetune comparisons. We provide the first application of crosscoders to cross-architecture model diffing and introduce Dedicated Feature Crosscoders (DFCs), an architectural modification designed to better isolate features unique to one model. Using this technique, we find in an unsupervised fashion features including Chinese Communist Party alignment in Qwen3-8B and Deepseek-R1-0528-Qwen3-8B, American exceptionalism in Llama3.1-8B-Instruct, and a copyright refusal mechanism in GPT-OSS-20B. Together, our results work towards establishing cross-architecture crosscoder model diffing as an effective method for identifying meaningful behavioral differences between AI models. |
| 2026-02-12 | [Beyond Parameter Arithmetic: Sparse Complementary Fusion for Distribution-Aware Model Merging](http://arxiv.org/abs/2602.11717v1) | Weihong Lin, Lin Sun et al. | Model merging has emerged as a promising paradigm for composing the capabilities of large language models by directly operating in weight space, enabling the integration of specialized models without costly retraining. However, existing merging methods largely rely on parameter-space heuristics, which often introduce severe interference, leading to degraded generalization and unstable generation behaviors such as repetition and incoherent outputs. In this work, we propose Sparse Complementary Fusion with reverse KL (SCF-RKL), a novel model merging framework that explicitly controls functional interference through sparse, distribution-aware updates. Instead of assuming linear additivity in parameter space, SCF-RKL measures the functional divergence between models using reverse Kullback-Leibler divergence and selectively incorporates complementary parameters. This mode-seeking, sparsity-inducing design effectively preserves stable representations while integrating new capabilities. We evaluate SCF-RKL across a wide range of model scales and architectures, covering both reasoning-focused and instruction-tuned models. Extensive experiments on 24 benchmarks spanning advanced reasoning, general reasoning and knowledge, instruction following, and safety demonstrate, vision classification that SCF-RKL consistently outperforms existing model merging methods while maintaining strong generalization and generation stability. |
| 2026-02-12 | [Quark Medical Alignment: A Holistic Multi-Dimensional Alignment and Collaborative Optimization Paradigm](http://arxiv.org/abs/2602.11661v1) | Tianxiang Xu, Jiayi Liu et al. | While reinforcement learning for large language model alignment has progressed rapidly in recent years, transferring these paradigms to high-stakes medical question answering reveals a fundamental paradigm mismatch. Reinforcement Learning from Human Feedback relies on preference annotations that are prohibitively expensive and often fail to reflect the absolute correctness of medical facts. Reinforcement Learning from Verifiable Rewards lacks effective automatic verifiers and struggles to handle complex clinical contexts. Meanwhile, medical alignment requires the simultaneous optimization of correctness, safety, and compliance, yet multi-objective heterogeneous reward signals are prone to scale mismatch and optimization conflicts.To address these challenges, we propose a robust medical alignment paradigm. We first construct a holistic multi-dimensional medical alignment matrix that decomposes alignment objectives into four categories: fundamental capabilities, expert knowledge, online feedback, and format specifications. Within each category, we establish a closed loop of where observable metrics inform attributable diagnosis, which in turn drives optimizable rewards, thereby providing fine-grained, high-resolution supervision signals for subsequent iterative optimization. To resolve gradient domination and optimization instability problem caused by heterogeneous signals, we further propose a unified optimization mechanism. This mechanism employs Reference-Frozen Normalization to align reward scales and implements a Tri-Factor Adaptive Dynamic Weighting strategy to achieve collaborative optimization that is weakness-oriented, risk-prioritized, and redundancy-reducing. Experimental results demonstrate the effectiveness of our proposed paradigm in real-world medical scenario evaluations, establishing a new paradigm for complex alignment in vertical domains. |
| 2026-02-12 | [DMind-3: A Sovereign Edge--Local--Cloud AI System with Controlled Deliberation and Correction-Based Tuning for Safe, Low-Latency Transaction Execution](http://arxiv.org/abs/2602.11651v1) | Enhao Huang, Frank Li et al. | This paper introduces DMind-3, a sovereign Edge-Local-Cloud intelligence stack designed to secure irreversible financial execution in Web3 environments against adversarial risks and strict latency constraints. While existing cloud-centric assistants compromise privacy and fail under network congestion, and purely local solutions lack global ecosystem context, DMind-3 resolves these tensions by decomposing capability into three cooperating layers: a deterministic signing-time intent firewall at the edge, a private high-fidelity reasoning engine on user hardware, and a policy-governed global context synthesizer in the cloud. We propose policy-driven selective offloading to route computation based on privacy sensitivity and uncertainty, supported by two novel training objectives: Hierarchical Predictive Synthesis (HPS) for fusing time-varying macro signals, and Contrastive Chain-of-Correction Supervised Fine-Tuning (C$^3$-SFT) to enhance local verification reliability. Extensive evaluations demonstrate that DMind-3 achieves a 93.7% multi-turn success rate in protocol-constrained tasks and superior domain reasoning compared to general-purpose baselines, providing a scalable framework where safety is bound to the edge execution primitive while maintaining sovereignty over sensitive user intent. |
| 2026-02-12 | [PACE: Prefix-Protected and Difficulty-Aware Compression for Efficient Reasoning](http://arxiv.org/abs/2602.11639v1) | Ruixiang Feng, Yuntao Wen et al. | Language Reasoning Models (LRMs) achieve strong performance by scaling test-time computation but often suffer from ``overthinking'', producing excessively long reasoning traces that increase latency and memory usage. Existing LRMs typically enforce conciseness with uniform length penalties, which over-compress crucial early deduction steps at the sequence level and indiscriminately penalize all queries at the group level. To solve these limitations, we propose \textbf{\model}, a dual-level framework for prefix-protected and difficulty-aware compression under hierarchical supervision. At the sequence level, prefix-protected optimization employs decaying mixed rollouts to maintain valid reasoning paths while promoting conciseness. At the group level, difficulty-aware penalty dynamically scales length constraints based on query complexity, maintaining exploration for harder questions while curbing redundancy on easier ones. Extensive experiments on DeepSeek-R1-Distill-Qwen (1.5B/7B) demonstrate that \model achieves a substantial reduction in token usage (up to \textbf{55.7\%}) while simultaneously improving accuracy (up to \textbf{4.1\%}) on math benchmarks, with generalization ability to code, science, and general domains. |
| 2026-02-12 | [scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](http://arxiv.org/abs/2602.11609v1) | Yiming Gao, Zhen Wang et al. | We present scPilot, the first systematic framework to practice omics-native reasoning: a large language model (LLM) converses in natural language while directly inspecting single-cell RNA-seq data and on-demand bioinformatics tools. scPilot converts core single-cell analyses, i.e., cell-type annotation, developmental-trajectory reconstruction, and transcription-factor targeting, into step-by-step reasoning problems that the model must solve, justify, and, when needed, revise with new evidence.   To measure progress, we release scBench, a suite of 9 expertly curated datasets and graders that faithfully evaluate the omics-native reasoning capability of scPilot w.r.t various LLMs. Experiments with o1 show that iterative omics-native reasoning lifts average accuracy by 11% for cell-type annotation and Gemini-2.5-Pro cuts trajectory graph-edit distance by 30% versus one-shot prompting, while generating transparent reasoning traces explain marker gene ambiguity and regulatory logic. By grounding LLMs in raw omics data, scPilot enables auditable, interpretable, and diagnostically informative single-cell analyses.   Code, data, and package are available at https://github.com/maitrix-org/scPilot |

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



