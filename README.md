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
| 2026-02-02 | [Reward-free Alignment for Conflicting Objectives](http://arxiv.org/abs/2602.02495v1) | Peter Chen, Xiaopeng Li et al. | Direct alignment methods are increasingly used to align large language models (LLMs) with human preferences. However, many real-world alignment problems involve multiple conflicting objectives, where naive aggregation of preferences can lead to unstable training and poor trade-offs. In particular, weighted loss methods may fail to identify update directions that simultaneously improve all objectives, and existing multi-objective approaches often rely on explicit reward models, introducing additional complexity and distorting user-specified preferences. The contributions of this paper are two-fold. First, we propose a Reward-free Alignment framework for Conflicted Objectives (RACO) that directly leverages pairwise preference data and resolves gradient conflicts via a novel clipped variant of conflict-averse gradient descent. We provide convergence guarantees to Pareto-critical points that respect user-specified objective weights, and further show that clipping can strictly improve convergence rate in the two-objective setting. Second, we improve our method using some heuristics and conduct experiments to demonstrate the compatibility of the proposed framework for LLM alignment. Both qualitative and quantitative evaluations on multi-objective summarization and safety alignment tasks across multiple LLM families (Qwen 3, Llama 3, Gemma 3) show that our method consistently achieves better Pareto trade-offs compared to existing multi-objective alignment baselines. |
| 2026-02-02 | [Drift-Bench: Diagnosing Cooperative Breakdowns in LLM Agents under Input Faults via Multi-Turn Interaction](http://arxiv.org/abs/2602.02455v1) | Han Bao, Zheyuan Zhang et al. | As Large Language Models transition to autonomous agents, user inputs frequently violate cooperative assumptions (e.g., implicit intent, missing parameters, false presuppositions, or ambiguous expressions), creating execution risks that text-only evaluations do not capture. Existing benchmarks typically assume well-specified instructions or restrict evaluation to text-only, single-turn clarification, and thus do not measure multi-turn disambiguation under grounded execution risk. We introduce \textbf{Drift-Bench}, the first diagnostic benchmark that evaluates agentic pragmatics under input faults through multi-turn clarification across state-oriented and service-oriented execution environments. Grounded in classical theories of communication, \textbf{Drift-Bench} provides a unified taxonomy of cooperative breakdowns and employs a persona-driven user simulator with the \textbf{Rise} evaluation protocol. Experiments show substantial performance drops under these faults, with clarification effectiveness varying across user personas and fault types. \MethodName bridges clarification research and agent safety evaluation, enabling systematic diagnosis of failures that can lead to unsafe executions. |
| 2026-02-02 | [Robust Safety-Critical Control of Networked SIR Dynamics](http://arxiv.org/abs/2602.02452v1) | Saba Samadi, Brooks A. Butler et al. | We present a robust safety-critical control framework tailored for networked susceptible-infected-recovered (SIR) epidemic dynamics, leveraging control barrier functions (CBFs) and robust control barrier functions to address the challenges of epidemic spread and mitigation. In our networked SIR model, each node must keep its infection level below a critical threshold, despite dynamic interactions with neighboring nodes and inherent uncertainties in the epidemic parameters and measurement errors, to ensure public health safety. We first derive a CBF-based controller that guarantees infection thresholds are not exceeded in the nominal case. We enhance the framework to handle realistic epidemic scenarios under uncertainties by incorporating compensation terms that reinforce safety against uncertainties: an independent method with constant bounds for uniform uncertainty, and a novel approach that scales with the state to capture increased relative noise in early or suppressed outbreak stages. Simulation results on a networked SIR system illustrate that the nominal CBF controller maintains safety under low uncertainty, while the robust approaches provide formal safety guarantees under higher uncertainties; in particular, the novel method employs more conservative control efforts to provide larger safety margins, whereas the independent approach optimizes resource allocation by allowing infection levels to approach the boundaries in steady epidemic regimes. |
| 2026-02-02 | [sVIRGO: A Scalable Virtual Tree Hierarchical Framework for Distributed Systems](http://arxiv.org/abs/2602.02438v1) | Lican Huang | We propose sVIRGO, a scalable virtual tree hierarchical framework for large-scale distributed systems. sVIRGO constructs virtual hierarchical trees directly on physical nodes, allowing each node to assume multiple hierarchical roles without overlay networks. The hierarchy preserves locality and is organized into configurable layers within regions. Coordination across thousands of regions is achieved via virtual upper-layer roles dynamically mapped onto nodes up to the top layer.   Each region maintains multiple active coordinators that monitor local health and perform dynamic re-selection if failures occur. Temporary drops below the minimum threshold do not compromise coordination, ensuring near-zero recovery latency, bounded communication overhead, and exponentially reduced failure probability while maintaining safety, liveness, and robustness under mobile, interference-prone, or adversarial conditions.   Communication is decoupled from the hierarchy and may use multi-frequency wireless links. Two message hop strategies are supported: (i) with long-distance infrastructure-assisted channels, coordinators exploit the virtual tree to minimize hops; (ii) without such channels, messages propagate via adjacent regions.   sVIRGO also supports Layer-Scoped Command Execution. Commands and coordination actions are executed within the scope of each hierarchical layer, enabling efficient local and regional decision-making while limiting unnecessary global propagation. |
| 2026-02-02 | [David vs. Goliath: Verifiable Agent-to-Agent Jailbreaking via Reinforcement Learning](http://arxiv.org/abs/2602.02395v1) | Samuel Nellessen, Tal Kachman | The evolution of large language models into autonomous agents introduces adversarial failures that exploit legitimate tool privileges, transforming safety evaluation in tool-augmented environments from a subjective NLP task into an objective control problem. We formalize this threat model as Tag-Along Attacks: a scenario where a tool-less adversary "tags along" on the trusted privileges of a safety-aligned Operator to induce prohibited tool use through conversation alone. To validate this threat, we present Slingshot, a 'cold-start' reinforcement learning framework that autonomously discovers emergent attack vectors, revealing a critical insight: in our setting, learned attacks tend to converge to short, instruction-like syntactic patterns rather than multi-turn persuasion. On held-out extreme-difficulty tasks, Slingshot achieves a 67.0% success rate against a Qwen2.5-32B-Instruct-AWQ Operator (vs. 1.7% baseline), reducing the expected attempts to first success (on solved tasks) from 52.3 to 1.3. Crucially, Slingshot transfers zero-shot to several model families, including closed-source models like Gemini 2.5 Flash (56.0% attack success rate) and defensive-fine-tuned open-source models like Meta-SecAlign-8B (39.2% attack success rate). Our work establishes Tag-Along Attacks as a first-class, verifiable threat model and shows that effective agentic attacks can be elicited from off-the-shelf open-weight models through environment interaction alone. |
| 2026-02-02 | [Uncertainty-Aware Image Classification In Biomedical Imaging Using Spectral-normalized Neural Gaussian Processes](http://arxiv.org/abs/2602.02370v1) | Uma Meleti, Jeffrey J. Nirschl | Accurate histopathologic interpretation is key for clinical decision-making; however, current deep learning models for digital pathology are often overconfident and poorly calibrated in out-of-distribution (OOD) settings, which limit trust and clinical adoption. Safety-critical medical imaging workflows benefit from intrinsic uncertainty-aware properties that can accurately reject OOD input. We implement the Spectral-normalized Neural Gaussian Process (SNGP), a set of lightweight modifications that apply spectral normalization and replace the final dense layer with a Gaussian process layer to improve single-model uncertainty estimation and OOD detection. We evaluate SNGP vs. deterministic and MonteCarlo dropout on six datasets across three biomedical classification tasks: white blood cells, amyloid plaques, and colorectal histopathology. SNGP has comparable in-distribution performance while significantly improving uncertainty estimation and OOD detection. Thus, SNGP or related models offer a useful framework for uncertainty-aware classification in digital pathology, supporting safe deployment and building trust with pathologists. |
| 2026-02-02 | [The SPARSE-Relativization Framework and Applications to Optimal Proof Systems](http://arxiv.org/abs/2602.02294v1) | Fabian Egidy | We investigate the following longstanding open questions raised by Krajíček and Pudlák (J. Symb. L. 1989), Sadowski (FCT 1997), Köbler and Messner (CCC 1998) and Messner (PhD 2000).   Q1: Does TAUT have (p-)optimal proof systems?   Q2: Does QBF have (p-)optimal proof systems?   Q3: Are there arbitrarily complex sets with (p-)optimal proof systems?   Recently, Egidy and Glaßer (STOC 2025) contributed to these questions by constructing oracles that show that there are no relativizable proofs for positive answers of these questions, even when assuming well-established conjectures about the separation of complexity classes. We continue this line of research by providing the same proof barrier for negative answers of these questions. For this, we introduce the SPARSE-relativization framework, which is an application of the notion of bounded relativization by Hirahara, Lu, and Ren (CCC 2023). This framework allows the construction of sparse oracles for statements such that additional useful properties (like an infinite polynomial-time hierarchy) hold. By applying the SPARSE-relativization framework, we show that the oracle construction of Egidy and Glaßer also yields the following new oracle.   O1: No set in PSPACE\NP has optimal proof systems, $\mathrm{NP} \subsetneq \mathrm{PH} \subsetneq \mathrm{PSPACE}$, and PH collapses   We use techniques of Cook and Krajíček (J. Symb. L. 2007) and Beyersdorff, Köbler, and Müller (Inf. Comp. 2011) and apply our SPARSE-relativization framework to obtain the following new oracle.   O2: All sets in PSPACE have p-optimal proof systems, there are arbitrarily complex sets with p-optimal proof systems, and PH is infinite   Together with previous results, our oracles show that questions Q1 and Q2 are independent of an infinite or collapsing polynomial-time hierarchy. |
| 2026-02-02 | [Before Autonomy Takes Control: Software Testing in Robotics](http://arxiv.org/abs/2602.02293v1) | Nils Chur, Thiago Santos de Moura et al. | Robotic systems are complex and safety-critical software systems. As such, they need to be tested thoroughly. Unfortunately, robot software is intrinsically hard to test compared to traditional software, mainly since the software needs to closely interact with hardware, account for uncertainty in its operational environment, handle disturbances, and act highly autonomously. However, given the large space in which robots operate, anticipating possible failures when designing tests is challenging. This paper presents a mapping study by considering robotics testing papers and relating them to the software testing theory. We consider 247 robotics testing papers and map them to software testing, discussing the state-of-the-art software testing in robotics with an illustrated example, and discuss current challenges. Forming the basis to introduce both the robotics and software engineering communities to software testing challenges. Finally, we identify open questions and lessons learned. |
| 2026-02-02 | [RACA: Representation-Aware Coverage Criteria for LLM Safety Testing](http://arxiv.org/abs/2602.02280v1) | Zeming Wei, Zhixin Zhang et al. | Recent advancements in LLMs have led to significant breakthroughs in various AI applications. However, their sophisticated capabilities also introduce severe safety concerns, particularly the generation of harmful content through jailbreak attacks. Current safety testing for LLMs often relies on static datasets and lacks systematic criteria to evaluate the quality and adequacy of these tests. While coverage criteria have been effective for smaller neural networks, they are not directly applicable to LLMs due to scalability issues and differing objectives. To address these challenges, this paper introduces RACA, a novel set of coverage criteria specifically designed for LLM safety testing. RACA leverages representation engineering to focus on safety-critical concepts within LLMs, thereby reducing dimensionality and filtering out irrelevant information. The framework operates in three stages: first, it identifies safety-critical representations using a small, expert-curated calibration set of jailbreak prompts. Second, it calculates conceptual activation scores for a given test suite based on these representations. Finally, it computes coverage results using six sub-criteria that assess both individual and compositional safety concepts. We conduct comprehensive experiments to validate RACA's effectiveness, applicability, and generalization, where the results demonstrate that RACA successfully identifies high-quality jailbreak prompts and is superior to traditional neuron-level criteria. We also showcase its practical application in real-world scenarios, such as test set prioritization and attack prompt sampling. Furthermore, our findings confirm RACA's generalization to various scenarios and its robustness across various configurations. Overall, RACA provides a new framework for evaluating the safety of LLMs, contributing a valuable technique to the field of testing for AI. |
| 2026-02-02 | [Bridging the Sim-to-Real Gap with multipanda ros2: A Real-Time ROS2 Framework for Multimanual Systems](http://arxiv.org/abs/2602.02269v1) | Jon Škerlj, Seongjin Bien et al. | We present $multipanda\_ros2$, a novel open-source ROS2 architecture for multi-robot control of Franka Robotics robots. Leveraging ros2 control, this framework provides native ROS2 interfaces for controlling any number of robots from a single process. Our core contributions address key challenges in real-time torque control, including interaction control and robot-environment modeling. A central focus of this work is sustaining a 1kHz control frequency, a necessity for real-time control and a minimum frequency required by safety standards. Moreover, we introduce a controllet-feature design pattern that enables controller-switching delays of $\le 2$ ms, facilitating reproducible benchmarking and complex multi-robot interaction scenarios. To bridge the simulation-to-reality (sim2real) gap, we integrate a high-fidelity MuJoCo simulation with quantitative metrics for both kinematic accuracy and dynamic consistency (torques, forces, and control errors). Furthermore, we demonstrate that real-world inertial parameter identification can significantly improve force and torque accuracy, providing a methodology for iterative physics refinement. Our work extends approaches from soft robotics to rigid dual-arm, contact-rich tasks, showcasing a promising method to reduce the sim2real gap and providing a robust, reproducible platform for advanced robotics research. |
| 2026-02-02 | [Alignment-Aware Model Adaptation via Feedback-Guided Optimization](http://arxiv.org/abs/2602.02258v1) | Gaurav Bhatt, Aditya Chinchure et al. | Fine-tuning is the primary mechanism for adapting foundation models to downstream tasks; however, standard approaches largely optimize task objectives in isolation and do not account for secondary yet critical alignment objectives (e.g., safety and hallucination avoidance). As a result, downstream fine-tuning can degrade alignment and fail to correct pre-existing misaligned behavior. We propose an alignment-aware fine-tuning framework that integrates feedback from an external alignment signal through policy-gradient-based regularization. Our method introduces an adaptive gating mechanism that dynamically balances supervised and alignment-driven gradients on a per-sample basis, prioritizing uncertain or misaligned cases while allowing well-aligned examples to follow standard supervised updates. The framework further learns abstention behavior for fully misaligned inputs, incorporating conservative responses directly into the fine-tuned model. Experiments on general and domain-specific instruction-tuning benchmarks demonstrate consistent reductions in harmful and hallucinated outputs without sacrificing downstream task performance. Additional analyses show robustness to adversarial fine-tuning, prompt-based attacks, and unsafe initializations, establishing adaptively gated alignment optimization as an effective approach for alignment-preserving and alignment-recovering model adaptation. |
| 2026-02-02 | [Malware Detection Through Memory Analysis](http://arxiv.org/abs/2602.02184v1) | Sarah Nassar | This paper summarizes the research conducted for a malware detection project using the Canadian Institute for Cybersecurity's MalMemAnalysis-2022 dataset. The purpose of the project was to explore the effectiveness and efficiency of machine learning techniques for the task of binary classification (i.e., benign or malicious) as well as multi-class classification to further include three malware sub-types (i.e., benign, ransomware, spyware, or Trojan horse). The XGBoost model type was the final model selected for both tasks due to the trade-off between strong detection capability and fast inference speed. The binary classifier achieved a testing subset accuracy and F1 score of 99.98\%, while the multi-class version reached an accuracy of 87.54\% and an F1 score of 81.26\%, with an average F1 score over the malware sub-types of 75.03\%. In addition to the high modelling performance, XGBoost is also efficient in terms of classification speed. It takes about 37.3 milliseconds to classify 50 samples in sequential order in the binary setting and about 43.2 milliseconds in the multi-class setting. The results from this research project help advance the efforts made towards developing accurate and real-time obfuscated malware detectors for the goal of improving online privacy and safety. *This project was completed as part of ELEC 877 (AI for Cybersecurity) in the Winter 2024 term. |
| 2026-02-02 | [Self-Evolving Coordination Protocol in Multi-Agent AI Systems: An Exploratory Systems Feasibility Study](http://arxiv.org/abs/2602.02170v1) | Jose Manuel de la Chica Rodriguez, Juan Manuel Vera Díaz | Contemporary multi-agent systems increasingly rely on internal coordination mechanisms to combine, arbitrate, or constrain the outputs of heterogeneous components. In safety-critical and regulated domains such as finance, these mechanisms must satisfy strict formal requirements, remain auditable, and operate within explicitly bounded limits. Coordination logic therefore functions as a governance layer rather than an optimization heuristic.   This paper presents an exploratory systems feasibility study of Self-Evolving Coordination Protocols (SECP): coordination protocols that permit limited, externally validated self-modification while preserving fixed formal invariants. We study a controlled proof-of-concept setting in which six fixed Byzantine consensus protocol proposals are evaluated by six specialized decision modules. All coordination regimes operate under identical hard constraints, including Byzantine fault tolerance (f < n/3), O(n2) message complexity, complete non-statistical safety and liveness arguments, and bounded explainability.   Four coordination regimes are compared in a single-shot design: unanimous hard veto, weighted scalar aggregation, SECP v1.0 (an agent-designed non-scalar protocol), and SECP v2.0 (the result of one governed modification). Outcomes are evaluated using a single metric, proposal coverage, defined as the number of proposals accepted. A single recursive modification increased coverage from two to three accepted proposals while preserving all declared invariants.   The study makes no claims regarding statistical significance, optimality, convergence, or learning. Its contribution is architectural: it demonstrates that bounded self-modification of coordination protocols is technically implementable, auditable, and analyzable under explicit formal constraints, establishing a foundation for governed multi-agent systems. |
| 2026-02-02 | [Mitigating Safety Tax via Distribution-Grounded Refinement in Large Reasoning Models](http://arxiv.org/abs/2602.02136v1) | Yingsha Xie, Tiansheng Huang et al. | Safety alignment incurs safety tax that perturbs a large reasoning model's (LRM) general reasoning ability. Existing datasets used for safety alignment for an LRM are usually constructed by distilling safety reasoning traces and answers from an external LRM or human labeler. However, such reasoning traces and answers exhibit a distributional gap with the target LRM that needs alignment, and we conjecture such distributional gap is the culprit leading to significant degradation of reasoning ability of the target LRM. Driven by this hypothesis, we propose a safety alignment dataset construction method, dubbed DGR. DGR transforms and refines an existing out-of-distributional safety reasoning dataset to be aligned with the target's LLM inner distribution. Experimental results demonstrate that i) DGR effectively mitigates the safety tax while maintaining safety performance across all baselines, i.e., achieving \textbf{+30.2\%} on DirectRefusal and \textbf{+21.2\%} on R1-ACT improvement in average reasoning accuracy compared to Vanilla SFT; ii) the degree of reasoning degradation correlates with the extent of distribution shift, suggesting that bridging this gap is central to preserving capabilities. Furthermore, we find that safety alignment in LRMs may primarily function as a mechanism to activate latent knowledge, as a mere \textbf{10} samples are sufficient for activating effective refusal behaviors. These findings not only emphasize the importance of distributional consistency but also provide insights into the activation mechanism of safety in reasoning models. |
| 2026-02-02 | [There Is More to Refusal in Large Language Models than a Single Direction](http://arxiv.org/abs/2602.02132v1) | Faaiz Joad, Majd Hawasly et al. | Prior work argues that refusal in large language models is mediated by a single activation-space direction, enabling effective steering and ablation. We show that this account is incomplete. Across eleven categories of refusal and non-compliance, including safety, incomplete or unsupported requests, anthropomorphization, and over-refusal, we find that these refusal behaviors correspond to geometrically distinct directions in activation space. Yet despite this diversity, linear steering along any refusal-related direction produces nearly identical refusal to over-refusal trade-offs, acting as a shared one-dimensional control knob. The primary effect of different directions is not whether the model refuses, but how it refuses. |
| 2026-02-02 | [Probabilistic Performance Guarantees for Multi-Task Reinforcement Learning](http://arxiv.org/abs/2602.02098v1) | Yannik Schnitzer, Mathias Jackermeier et al. | Multi-task reinforcement learning trains generalist policies that can execute multiple tasks. While recent years have seen significant progress, existing approaches rarely provide formal performance guarantees, which are indispensable when deploying policies in safety-critical settings. We present an approach for computing high-confidence guarantees on the performance of a multi-task policy on tasks not seen during training. Concretely, we introduce a new generalisation bound that composes (i) per-task lower confidence bounds from finitely many rollouts with (ii) task-level generalisation from finitely many sampled tasks, yielding a high-confidence guarantee for new tasks drawn from the same arbitrary and unknown distribution. Across state-of-the-art multi-task RL methods, we show that the guarantees are theoretically sound and informative at realistic sample sizes. |
| 2026-02-02 | [Constrained Process Maps for Multi-Agent Generative AI Workflows](http://arxiv.org/abs/2602.02034v1) | Ananya Joshi, Michael Rudow | Large language model (LLM)-based agents are increasingly used to perform complex, multi-step workflows in regulated settings such as compliance and due diligence. However, many agentic architectures rely primarily on prompt engineering of a single agent, making it difficult to observe or compare how models handle uncertainty and coordination across interconnected decision stages and with human oversight. We introduce a multi-agent system formalized as a finite-horizon Markov Decision Process (MDP) with a directed acyclic structure. Each agent corresponds to a specific role or decision stage (e.g., content, business, or legal review in a compliance workflow), with predefined transitions representing task escalation or completion. Epistemic uncertainty is quantified at the agent level using Monte Carlo estimation, while system-level uncertainty is captured by the MDP's termination in either an automated labeled state or a human-review state. We illustrate the approach through a case study in AI safety evaluation for self-harm detection, implemented as a multi-agent compliance system. Results demonstrate improvements over a single-agent baseline, including up to a 19\% increase in accuracy, up to an 85x reduction in required human review, and, in some configurations, reduced processing time. |
| 2026-02-02 | [Light Alignment Improves LLM Safety via Model Self-Reflection with a Single Neuron](http://arxiv.org/abs/2602.02027v1) | Sicheng Shen, Mingyang Lv et al. | The safety of large language models (LLMs) has increasingly emerged as a fundamental aspect of their development. Existing safety alignment for LLMs is predominantly achieved through post-training methods, which are computationally expensive and often fail to generalize well across different models. A small number of lightweight alignment approaches either rely heavily on prior-computed safety injections or depend excessively on the model's own capabilities, resulting in limited generalization and degraded efficiency and usability during generation. In this work, we propose a safety-aware decoding method that requires only low-cost training of an expert model and employs a single neuron as a gating mechanism. By effectively balancing the model's intrinsic capabilities with external guidance, our approach simultaneously preserves utility and enhances output safety. It demonstrates clear advantages in training overhead and generalization across model scales, offering a new perspective on lightweight alignment for the safe and practical deployment of large language models. Code: https://github.com/Beijing-AISI/NGSD. |
| 2026-02-02 | [Obstacle Detection at Level Crossings under Adverse Weather Conditions -- A Survey](http://arxiv.org/abs/2602.01974v1) | Chenyang Yan, Mats Bengtsson | Level crossing accidents remain a significant safety concern in modern railway systems, particularly under adverse weather conditions that degrade sensor performance. This review surveys state-of-the-art sensor technologies and fusion strategies for obstacle detection at railway level crossings, with a focus on robustness, detection accuracy, and environmental resilience. Individual sensors such as inductive loops, cameras, radar, and LiDAR offer complementary strengths but involve trade-offs, including material dependence, reduced visibility, and limited resolution in harsh environments. We analyze each modality's working principles, weather-induced vulnerabilities, and mitigation strategies, including signal enhancement and machine-learning-based denoising. We further review multi-sensor fusion approaches, categorized as data-level, feature-level, and decision-level architectures, that integrate complementary information to improve reliability and fault tolerance. The survey concludes with future research directions, including adaptive fusion algorithms, real-time processing pipelines, and weather-resilient datasets to support the deployment of intelligent, fail-safe detection systems for railway safety. |
| 2026-02-02 | [Efficient Epistemic Uncertainty Estimation for Large Language Models via Knowledge Distillation](http://arxiv.org/abs/2602.01956v1) | Seonghyeon Park, Jewon Yeom et al. | Quantifying uncertainty in Large Language Models (LLMs) is essential for mitigating hallucinations and enabling risk-aware deployment in safety-critical tasks. However, estimating Epistemic Uncertainty(EU) via Deep Ensembles is computationally prohibitive at the scale of modern models. We propose a framework that leverages the small draft models to efficiently estimate token-level EU, bypassing the need for full-scale ensembling. Theoretically grounded in a Bias-Variance Decomposition, our approach approximates EU via Jensen-Shannon divergence among drafts (variance proxy) and KL divergence between the draft mixture and the target (bias proxy). To further ensure accuracy without significant overhead, we introduce Online Stochastic Distillation (OSD) to efficiently approximate target aggregation and the Data-Diverse Drafts (DDD) strategy to enhance draft diversity for better target approximation. Extensive experiments on GSM8K demonstrate that our method reduces the estimation error (RMSE) by up to 37% compared to baselines. Crucially, our approach achieves Hallucination Detection performance competitive with heavy perturbation-based methods like TokUR while incurring negligible inference costs, offering a practical solution for uncertainty-aware LLM deployment. |

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



