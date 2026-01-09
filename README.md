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
| 2026-01-08 | [An interpretable data-driven approach to optimizing clinical fall risk assessment](http://arxiv.org/abs/2601.05194v1) | Fardin Ganjkhanloo, Emmett Springer et al. | In this study, we aim to better align fall risk prediction from the Johns Hopkins Fall Risk Assessment Tool (JHFRAT) with additional clinically meaningful measures via a data-driven modelling approach. We conducted a retrospective cohort analysis of 54,209 inpatient admissions from three Johns Hopkins Health System hospitals between March 2022 and October 2023. A total of 20,208 admissions were included as high fall risk encounters, and 13,941 were included as low fall risk encounters. To incorporate clinical knowledge and maintain interpretability, we employed constrained score optimization (CSO) models to reweight the JHFRAT scoring weights, while preserving its additive structure and clinical thresholds. Recalibration refers to adjusting item weights so that the resulting score can order encounters more consistently by the study's risk labels, and without changing the tool's form factor or deployment workflow. The model demonstrated significant improvements in predictive performance over the current JHFRAT (CSO AUC-ROC=0.91, JHFRAT AUC-ROC=0.86). This performance improvement translates to protecting an additional 35 high-risk patients per week across the Johns Hopkins Health System. The constrained score optimization models performed similarly with and without the EHR variables. Although the benchmark black-box model (XGBoost), improves upon the performance metrics of the knowledge-based constrained logistic regression (AUC-ROC=0.94), the CSO demonstrates more robustness to variations in risk labeling. This evidence-based approach provides a robust foundation for health systems to systematically enhance inpatient fall prevention protocols and patient safety using data-driven optimization techniques, contributing to improved risk assessment and resource allocation in healthcare settings. |
| 2026-01-08 | [Inside Out: Evolving User-Centric Core Memory Trees for Long-Term Personalized Dialogue Systems](http://arxiv.org/abs/2601.05171v1) | Jihao Zhao, Ding Chen et al. | Existing long-term personalized dialogue systems struggle to reconcile unbounded interaction streams with finite context constraints, often succumbing to memory noise accumulation, reasoning degradation, and persona inconsistency. To address these challenges, this paper proposes Inside Out, a framework that utilizes a globally maintained PersonaTree as the carrier of long-term user profiling. By constraining the trunk with an initial schema and updating the branches and leaves, PersonaTree enables controllable growth, achieving memory compression while preserving consistency. Moreover, we train a lightweight MemListener via reinforcement learning with process-based rewards to produce structured, executable, and interpretable {ADD, UPDATE, DELETE, NO_OP} operations, thereby supporting the dynamic evolution of the personalized tree. During response generation, PersonaTree is directly leveraged to enhance outputs in latency-sensitive scenarios; when users require more details, the agentic mode is triggered to introduce details on-demand under the constraints of the PersonaTree. Experiments show that PersonaTree outperforms full-text concatenation and various personalized memory systems in suppressing contextual noise and maintaining persona consistency. Notably, the small MemListener model achieves memory-operation decision performance comparable to, or even surpassing, powerful reasoning models such as DeepSeek-R1-0528 and Gemini-3-Pro. |
| 2026-01-08 | [Safe Continual Reinforcement Learning Methods for Nonstationary Environments. Towards a Survey of the State of the Art](http://arxiv.org/abs/2601.05152v1) | Timofey Tomashevskiy | This work provides a state-of-the-art survey of continual safe online reinforcement learning (COSRL) methods. We discuss theoretical aspects, challenges, and open questions in building continual online safe reinforcement learning algorithms. We provide the taxonomy and the details of continual online safe reinforcement learning methods based on the type of safe learning mechanism that takes adaptation to nonstationarity into account. We categorize safety constraints formulation for online reinforcement learning algorithms, and finally, we discuss prospects for creating reliable, safe online learning algorithms.   Keywords: safe RL in nonstationary environments, safe continual reinforcement learning under nonstationarity, HM-MDP, NSMDP, POMDP, safe POMDP, constraints for continual learning, safe continual reinforcement learning review, safe continual reinforcement learning survey, safe continual reinforcement learning, safe online learning under distribution shift, safe continual online adaptation, safe reinforcement learning, safe exploration, safe adaptation, constrained Markov decision processes, safe reinforcement learning, partially observable Markov decision process, safe reinforcement learning and hidden Markov decision processes, Safe Online Reinforcement Learning, safe online reinforcement learning, safe online reinforcement learning, safe meta-learning, safe meta-reinforcement learning, safe context-based reinforcement learning, formulating safety constraints for continual learning |
| 2026-01-08 | [$PC^2$: Politically Controversial Content Generation via Jailbreaking Attacks on GPT-based Text-to-Image Models](http://arxiv.org/abs/2601.05150v1) | Wonwoo Choi, Minjae Seo et al. | The rapid evolution of text-to-image (T2I) models has enabled high-fidelity visual synthesis on a global scale. However, these advancements have introduced significant security risks, particularly regarding the generation of harmful content. Politically harmful content, such as fabricated depictions of public figures, poses severe threats when weaponized for fake news or propaganda. Despite its criticality, the robustness of current T2I safety filters against such politically motivated adversarial prompting remains underexplored. In response, we propose $PC^2$, the first black-box political jailbreaking framework for T2I models. It exploits a novel vulnerability where safety filters evaluate political sensitivity based on linguistic context. $PC^2$ operates through: (1) Identity-Preserving Descriptive Mapping to obfuscate sensitive keywords into neutral descriptions, and (2) Geopolitically Distal Translation to map these descriptions into fragmented, low-sensitivity languages. This strategy prevents filters from constructing toxic relationships between political entities within prompts, effectively bypassing detection. We construct a benchmark of 240 politically sensitive prompts involving 36 public figures. Evaluation on commercial T2I models, specifically GPT-series, shows that while all original prompts are blocked, $PC^2$ achieves attack success rates of up to 86%. |
| 2026-01-08 | [Driving on Registers](http://arxiv.org/abs/2601.05083v1) | Ellington Kirby, Alexandre Boulch et al. | We present DrivoR, a simple and efficient transformer-based architecture for end-to-end autonomous driving. Our approach builds on pretrained Vision Transformers (ViTs) and introduces camera-aware register tokens that compress multi-camera features into a compact scene representation, significantly reducing downstream computation without sacrificing accuracy. These tokens drive two lightweight transformer decoders that generate and then score candidate trajectories. The scoring decoder learns to mimic an oracle and predicts interpretable sub-scores representing aspects such as safety, comfort, and efficiency, enabling behavior-conditioned driving at inference. Despite its minimal design, DrivoR outperforms or matches strong contemporary baselines across NAVSIM-v1, NAVSIM-v2, and the photorealistic closed-loop HUGSIM benchmark. Our results show that a pure-transformer architecture, combined with targeted token compression, is sufficient for accurate, efficient, and adaptive end-to-end driving. Code and checkpoints will be made available via the project page. |
| 2026-01-08 | [DAVOS: An Autonomous Vehicle Operating System in the Vehicle Computing Era](http://arxiv.org/abs/2601.05072v1) | Yuxin Wang, Yuankai He et al. | Vehicle computing represents a fundamental shift in how autonomous vehicles are designed and deployed, transforming them from isolated transportation systems into mobile computing platforms that support both safety-critical, real-time driving and data-centric services. In this setting, vehicles simultaneously support real-time driving pipelines and a growing set of data-driven applications, placing increased responsibility on the vehicle operating system to coordinate computation, data movement, storage, and access. These demands highlight recurring system considerations related to predictable execution, data and execution protection, efficient handling of high-rate sensor data, and long-term system evolvability, commonly summarized as Safety, Security, Efficiency, and Extensibility (SSEE). Existing vehicle operating systems and runtimes address these concerns in isolation, resulting in fragmented software stacks that limit coordination between autonomy workloads and vehicle data services. This paper presents DAVOS, the Delaware Autonomous Vehicle Operating System, a unified vehicle operating system architecture designed for the vehicle computing context. DAVOS provides a cohesive operating system foundation that supports both real-time autonomy and extensible vehicle computing within a single system framework. |
| 2026-01-08 | [AlgBench: To What Extent Do Large Reasoning Models Understand Algorithms?](http://arxiv.org/abs/2601.04996v1) | Henan Sun, Kaichi Yu et al. | Reasoning ability has become a central focus in the advancement of Large Reasoning Models (LRMs). Although notable progress has been achieved on several reasoning benchmarks such as MATH500 and LiveCodeBench, existing benchmarks for algorithmic reasoning remain limited, failing to answer a critical question: Do LRMs truly master algorithmic reasoning? To answer this question, we propose AlgBench, an expert-curated benchmark that evaluates LRMs under an algorithm-centric paradigm.   AlgBench consists of over 3,000 original problems spanning 27 algorithms, constructed by ACM algorithmic experts and organized under a comprehensive taxonomy, including Euclidean-structured, non-Euclidean-structured, non-optimized, local-optimized, global-optimized, and heuristic-optimized categories. Empirical evaluations on leading LRMs (e.g., Gemini-3-Pro, DeepSeek-v3.2-Speciale and GPT-o3) reveal substantial performance heterogeneity: while models perform well on non-optimized tasks (up to 92%), accuracy drops sharply to around 49% on globally optimized algorithms such as dynamic programming. Further analysis uncovers \textbf{strategic over-shifts}, wherein models prematurely abandon correct algorithmic designs due to necessary low-entropy tokens. These findings expose fundamental limitations of problem-centric reinforcement learning and highlight the necessity of an algorithm-centric training paradigm for robust algorithmic reasoning. |
| 2026-01-08 | [When to Act: Calibrated Confidence for Reliable Human Intention Prediction in Assistive Robotics](http://arxiv.org/abs/2601.04982v1) | Johannes A. Gaus, Winfried Ilg et al. | Assistive devices must determine both what a user intends to do and how reliable that prediction is before providing support. We introduce a safety-critical triggering framework based on calibrated probabilities for multimodal next-action prediction in Activities of Daily Living. Raw model confidence often fails to reflect true correctness, posing a safety risk. Post-hoc calibration aligns predicted confidence with empirical reliability and reduces miscalibration by about an order of magnitude without affecting accuracy. The calibrated confidence drives a simple ACT/HOLD rule that acts only when reliability is high and withholds assistance otherwise. This turns the confidence threshold into a quantitative safety parameter for assisted actions and enables verifiable behavior in an assistive control loop. |
| 2026-01-08 | [Scale effects methodology applied to uprising jets in ASTRID](http://arxiv.org/abs/2601.04979v1) | B Jourdy, D Guenadou et al. | CEA is involved in the development of the 4 th generation of nuclear reactors, among which are the Sodium Cooled Fast Reactors (SFR). To support their design and safety, specific codes were developed and should be validated using experimental results from relevant mock-ups. Due to the complexity of building a fullsized prototype in the nuclear field, most of the experiments are performed on reduced-sized models but it may lead to scale effects. Such scale effects are under study in this paper, focusing on a critical issue in SFR reactor, which is the rising of the jet outgoing the core at low power. For this purpose, a scale effects methodology is detailed, using the SFR's reduced scale mock-up MICAS as the reference scale. To achieve that, dimensionless Navier-Stokes equations under Boussinesq's approximation are considered and the Vaschy-Buckingham theorem is applied to determine the most relevant dimensionless numbers, among them the densimetric Froude number. Experimental campaigns have been performed to measure velocity fields using the Particle Image Velocimetry (PIV). Their analysis clearly shown dependency of mean pathway of the jet on this dimensionless number. Particularly, plotting the jet angle as a function of the normalized densimetric Froude number, a change of behaviour for a threshold value of 0.45 has been observed. This result also shown negligible effects on the jet rise happening before jet impingement on the Upper Core Structure (UCS). |
| 2026-01-08 | [CFD modeling of hydrogen release and dispersion in a congested container](http://arxiv.org/abs/2601.04970v1) | Hector Amino, Lynda Porcheron et al. | Hydrogen plays an important role in driving decarbonization within the current global energy landscape. As hydrogen infrastructures rapidly expand beyond their traditional applications, there is a need for comprehensive safety practices, solutions, and regulations. Within this framework, the dispersion of hydrogen in enclosed facilities presents a significant safety concern due to its potential for explosive accidents. In this study, hydrogen dispersion in a confined and congested environment (37 m${}^3$ container) is studied using computational fluid dynamics simulations. The experimental setup mirrors previous INERIS investigations, featuring a centrally located hydrogen injection point on the floor with a 20 mm diameter injector and a release rate of 35 g/s, resulting in a Froude number of 650. This yields an inertial jet and a challenging dispersion scenario for numerical prediction. Concentration mapping is carried out by 3 oxygen analyzers distributed throughout the 37 m3 chamber. Comparisons are made between the measured and numerical data to validate the solver used under such conditions. Best practice guidelines are followed, and sensitivity studies involving grid refinement and boundary conditions are conducted to ensure robust simulation results. The findings highlight the model's ability to reproduce the hydrogen concentration distribution for both empty and congested containers and underline the role of accounting for leakages in such scenarios. |
| 2026-01-08 | [Proof of Commitment: A Human-Centric Resource for Permissionless Consensus](http://arxiv.org/abs/2601.04813v1) | Homayoun Maleki, Nekane Sainz et al. | Permissionless consensus protocols require a scarce resource to regulate leader election and provide Sybil resistance. Existing paradigms such as Proof of Work and Proof of Stake instantiate this scarcity through parallelizable resources like computation or capital. Once acquired, these resources can be subdivided across many identities at negligible marginal cost, making linear Sybil cost fundamentally unattainable.   We introduce Proof of Commitment (PoCmt), a consensus primitive grounded in a non-parallelizable resource: real-time human engagement. Validators maintain a commitment state capturing cumulative human effort, protocol participation, and online availability. Engagement is enforced through a Human Challenge Oracle that issues identity-bound, time-sensitive challenges, limiting the number of challenges solvable within each human window.   Under this model, sustaining multiple active identities requires proportional human-time effort. We establish a cost-theoretic separation showing that protocols based on parallelizable resources admit zero marginal Sybil cost, whereas PoCmt enforces a strictly linear cost profile. Using a weighted-backbone analysis, we show that PoCmt achieves safety, liveness, and commitment-proportional fairness under partial synchrony.   Simulations complement the analysis by isolating human-time capacity as the sole adversarial bottleneck and validating the predicted commitment drift and fairness properties. These results position PoCmt as a new point in the consensus design space, grounding permissionless security in sustained human effort rather than computation or capital. |
| 2026-01-08 | [Thinking-Based Non-Thinking: Solving the Reward Hacking Problem in Training Hybrid Reasoning Models via Reinforcement Learning](http://arxiv.org/abs/2601.04805v1) | Siyuan Gan, Jiaheng Liu et al. | Large reasoning models (LRMs) have attracted much attention due to their exceptional performance. However, their performance mainly stems from thinking, a long Chain of Thought (CoT), which significantly increase computational overhead. To address this overthinking problem, existing work focuses on using reinforcement learning (RL) to train hybrid reasoning models that automatically decide whether to engage in thinking or not based on the complexity of the query. Unfortunately, using RL will suffer the the reward hacking problem, e.g., the model engages in thinking but is judged as not doing so, resulting in incorrect rewards. To mitigate this problem, existing works either employ supervised fine-tuning (SFT), which incurs high computational costs, or enforce uniform token limits on non-thinking responses, which yields limited mitigation of the problem. In this paper, we propose Thinking-Based Non-Thinking (TNT). It does not employ SFT, and sets different maximum token usage for responses not using thinking across various queries by leveraging information from the solution component of the responses using thinking. Experiments on five mathematical benchmarks demonstrate that TNT reduces token usage by around 50% compared to DeepSeek-R1-Distill-Qwen-1.5B/7B and DeepScaleR-1.5B, while significantly improving accuracy. In fact, TNT achieves the optimal trade-off between accuracy and efficiency among all tested methods. Additionally, the probability of reward hacking problem in TNT's responses, which are classified as not using thinking, remains below 10% across all tested datasets. |
| 2026-01-08 | [Belief in Authority: Impact of Authority in Multi-Agent Evaluation Framework](http://arxiv.org/abs/2601.04790v1) | Junhyuk Choi, Jeongyoun Kwon et al. | Multi-agent systems utilizing large language models often assign authoritative roles to improve performance, yet the impact of authority bias on agent interactions remains underexplored. We present the first systematic analysis of role-based authority bias in free-form multi-agent evaluation using ChatEval. Applying French and Raven's power-based theory, we classify authoritative roles into legitimate, referent, and expert types and analyze their influence across 12-turn conversations. Experiments with GPT-4o and DeepSeek R1 reveal that Expert and Referent power roles exert stronger influence than Legitimate power roles. Crucially, authority bias emerges not through active conformity by general agents, but through authoritative roles consistently maintaining their positions while general agents demonstrate flexibility. Furthermore, authority influence requires clear position statements, as neutral responses fail to generate bias. These findings provide key insights for designing multi-agent frameworks with asymmetric interaction patterns. |
| 2026-01-08 | [RiskAtlas: Exposing Domain-Specific Risks in LLMs through Knowledge-Graph-Guided Harmful Prompt Generation](http://arxiv.org/abs/2601.04740v1) | Huawei Zheng, Xinqi Jiang et al. | Large language models (LLMs) are increasingly applied in specialized domains such as finance and healthcare, where they introduce unique safety risks. Domain-specific datasets of harmful prompts remain scarce and still largely rely on manual construction; public datasets mainly focus on explicit harmful prompts, which modern LLM defenses can often detect and refuse. In contrast, implicit harmful prompts-expressed through indirect domain knowledge-are harder to detect and better reflect real-world threats. We identify two challenges: transforming domain knowledge into actionable constraints and increasing the implicitness of generated harmful prompts. To address them, we propose an end-to-end framework that first performs knowledge-graph-guided harmful prompt generation to systematically produce domain-relevant prompts, and then applies dual-path obfuscation rewriting to convert explicit harmful prompts into implicit variants via direct and context-enhanced rewriting. This framework yields high-quality datasets combining strong domain relevance with implicitness, enabling more realistic red-teaming and advancing LLM safety research. We release our code and datasets at GitHub. |
| 2026-01-08 | [AM$^3$Safety: Towards Data Efficient Alignment of Multi-modal Multi-turn Safety for MLLMs](http://arxiv.org/abs/2601.04736v1) | Han Zhu, Jiale Chen et al. | Multi-modal Large Language Models (MLLMs) are increasingly deployed in interactive applications. However, their safety vulnerabilities become pronounced in multi-turn multi-modal scenarios, where harmful intent can be gradually reconstructed across turns, and security protocols fade into oblivion as the conversation progresses. Existing Reinforcement Learning from Human Feedback (RLHF) alignment methods are largely developed for single-turn visual question-answer (VQA) task and often require costly manual preference annotations, limiting their effectiveness and scalability in dialogues. To address this challenge, we present InterSafe-V, an open-source multi-modal dialogue dataset containing 11,270 dialogues and 500 specially designed refusal VQA samples. This dataset, constructed through interaction between several models, is designed to more accurately reflect real-world scenarios and includes specialized VQA pairs tailored for specific domains. Building on this dataset, we propose AM$^3$Safety, a framework that combines a cold-start refusal phase with Group Relative Policy Optimization (GRPO) fine-tuning using turn-aware dual-objective rewards across entire dialogues. Experiments on Qwen2.5-VL-7B-Instruct and LLaVA-NeXT-7B show more than 10\% decrease in Attack Success Rate (ASR) together with an increment of at least 8\% in harmless dimension and over 13\% in helpful dimension of MLLMs on multi-modal multi-turn safety benchmarks, while preserving their general abilities. |
| 2026-01-08 | [Excess Description Length of Learning Generalizable Predictors](http://arxiv.org/abs/2601.04728v1) | Elizabeth Donoway, Hailey Joren et al. | Understanding whether fine-tuning elicits latent capabilities or teaches new ones is a fundamental question for language model evaluation and safety. We develop a formal information-theoretic framework for quantifying how much predictive structure fine-tuning extracts from the train dataset and writes into a model's parameters. Our central quantity, Excess Description Length (EDL), is defined via prequential coding and measures the gap between the bits required to encode training labels sequentially using an evolving model (trained online) and the residual encoding cost under the final trained model. We establish that EDL is non-negative in expectation, converges to surplus description length in the infinite-data limit, and provides bounds on expected generalization gain. Through a series of toy models, we clarify common confusions about information in learning: why random labels yield EDL near zero, how a single example can eliminate many bits of uncertainty about the underlying rule(s) that describe the data distribution, why structure learned on rare inputs contributes proportionally little to expected generalization, and how format learning creates early transients distinct from capability acquisition. This framework provides rigorous foundations for the empirical observation that capability elicitation and teaching exhibit qualitatively distinct scaling signatures. |
| 2026-01-08 | [ToolGate: Contract-Grounded and Verified Tool Execution for LLMs](http://arxiv.org/abs/2601.04688v1) | Yanming Liu, Xinyue Peng et al. | Large Language Models (LLMs) augmented with external tools have demonstrated remarkable capabilities in complex reasoning tasks. However, existing frameworks rely heavily on natural language reasoning to determine when tools can be invoked and whether their results should be committed, lacking formal guarantees for logical safety and verifiability. We present \textbf{ToolGate}, a forward execution framework that provides logical safety guarantees and verifiable state evolution for LLM tool calling. ToolGate maintains an explicit symbolic state space as a typed key-value mapping representing trusted world information throughout the reasoning process. Each tool is formalized as a Hoare-style contract consisting of a precondition and a postcondition, where the precondition gates tool invocation by checking whether the current state satisfies the required conditions, and the postcondition determines whether the tool's result can be committed to update the state through runtime verification. Our approach guarantees that the symbolic state evolves only through verified tool executions, preventing invalid or hallucinated results from corrupting the world representation. Experimental validation demonstrates that ToolGate significantly improves the reliability and verifiability of tool-augmented LLM systems while maintaining competitive performance on complex multi-step reasoning tasks. This work establishes a foundation for building more trustworthy and debuggable AI systems that integrate language models with external tools. |
| 2026-01-08 | [Nightmare Dreamer: Dreaming About Unsafe States And Planning Ahead](http://arxiv.org/abs/2601.04686v1) | Oluwatosin Oseni, Shengjie Wang et al. | Reinforcement Learning (RL) has shown remarkable success in real-world applications, particularly in robotics control. However, RL adoption remains limited due to insufficient safety guarantees. We introduce Nightmare Dreamer, a model-based Safe RL algorithm that addresses safety concerns by leveraging a learned world model to predict potential safety violations and plan actions accordingly. Nightmare Dreamer achieves nearly zero safety violations while maximizing rewards. Nightmare Dreamer outperforms model-free baselines on Safety Gymnasium tasks using only image observations, achieving nearly a 20x improvement in efficiency. |
| 2026-01-08 | [Optimizing Path Planning using Deep Reinforcement Learning for UGVs in Precision Agriculture](http://arxiv.org/abs/2601.04668v1) | Laukik Patade, Rohan Rane et al. | This study focuses on optimizing path planning for unmanned ground vehicles (UGVs) in precision agriculture using deep reinforcement learning (DRL) techniques in continuous action spaces. The research begins with a review of traditional grid-based methods, such as A* and Dijkstra's algorithms, and discusses their limitations in dynamic agricultural environments, highlighting the need for adaptive learning strategies. The study then explores DRL approaches, including Deep Q-Networks (DQN), which demonstrate improved adaptability and performance in two-dimensional simulations. Enhancements such as Double Q-Networks and Dueling Networks are evaluated to further improve decision-making. Building on these results, the focus shifts to continuous action space models, specifically Deep Deterministic Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3), which are tested in increasingly complex environments. Experiments conducted in a three-dimensional environment using ROS and Gazebo demonstrate the effectiveness of continuous DRL algorithms in navigating dynamic agricultural scenarios. Notably, the pretrained TD3 agent achieves a 95 percent success rate in dynamic environments, demonstrating the robustness of the proposed approach in handling moving obstacles while ensuring safety for both crops and the robot. |
| 2026-01-08 | [UniBiDex: A Unified Teleoperation Framework for Robotic Bimanual Dexterous Manipulation](http://arxiv.org/abs/2601.04629v1) | Zhongxuan Li, Zeliang Guo et al. | We present UniBiDex a unified teleoperation framework for robotic bimanual dexterous manipulation that supports both VRbased and leaderfollower input modalities UniBiDex enables realtime contactrich dualarm teleoperation by integrating heterogeneous input devices into a shared control stack with consistent kinematic treatment and safety guarantees The framework employs nullspace control to optimize bimanual configurations ensuring smooth collisionfree and singularityaware motion across tasks We validate UniBiDex on a longhorizon kitchentidying task involving five sequential manipulation subtasks demonstrating higher task success rates smoother trajectories and improved robustness compared to strong baselines By releasing all hardware and software components as opensource we aim to lower the barrier to collecting largescale highquality human demonstration datasets and accelerate progress in robot learning. |

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



