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
| 2026-01-23 | [AgentDrive: An Open Benchmark Dataset for Agentic AI Reasoning with LLM-Generated Scenarios in Autonomous Systems](http://arxiv.org/abs/2601.16964v1) | Mohamed Amine Ferrag, Abderrahmane Lakas et al. | The rapid advancement of large language models (LLMs) has sparked growing interest in their integration into autonomous systems for reasoning-driven perception, planning, and decision-making. However, evaluating and training such agentic AI models remains challenging due to the lack of large-scale, structured, and safety-critical benchmarks. This paper introduces AgentDrive, an open benchmark dataset containing 300,000 LLM-generated driving scenarios designed for training, fine-tuning, and evaluating autonomous agents under diverse conditions. AgentDrive formalizes a factorized scenario space across seven orthogonal axes: scenario type, driver behavior, environment, road layout, objective, difficulty, and traffic density. An LLM-driven prompt-to-JSON pipeline generates semantically rich, simulation-ready specifications that are validated against physical and schema constraints. Each scenario undergoes simulation rollouts, surrogate safety metric computation, and rule-based outcome labeling. To complement simulation-based evaluation, we introduce AgentDrive-MCQ, a 100,000-question multiple-choice benchmark spanning five reasoning dimensions: physics, policy, hybrid, scenario, and comparative reasoning. We conduct a large-scale evaluation of fifty leading LLMs on AgentDrive-MCQ. Results show that while proprietary frontier models perform best in contextual and policy reasoning, advanced open models are rapidly closing the gap in structured and physics-grounded reasoning. We release the AgentDrive dataset, AgentDrive-MCQ benchmark, evaluation code, and related materials at https://github.com/maferrag/AgentDrive |
| 2026-01-23 | [In Quest of an Extensible Multi-Level Harm Taxonomy for Adversarial AI: Heart of Security, Ethical Risk Scoring and Resilience Analytics](http://arxiv.org/abs/2601.16930v1) | Javed I. Khan, Sharmila Rahman Prithula | Harm is invoked everywhere from cybersecurity, ethics, risk analysis, to adversarial AI, yet there exists no systematic or agreed upon list of harms, and the concept itself is rarely defined with the precision required for serious analysis. Current discourse relies on vague, under specified notions of harm, rendering nuanced, structured, and qualitative assessment effectively impossible. This paper challenges that gap directly. We introduce a structured and expandable taxonomy of harms, grounded in an ensemble of contemporary ethical theories, that makes harm explicit, enumerable, and analytically tractable. The proposed framework identifies 66+ distinct harm types, systematically organized into two overarching domains human and nonhuman, and eleven major categories, each explicitly aligned with eleven dominant ethical theories. While extensible by design, the upper levels are intentionally stable. Beyond classification, we introduce a theory-aware taxonomy of victim entities and formalize normative harm attributes, including reversibility and duration that materially alter ethical severity. Together, these contributions transform harm from a rhetorical placeholder into an operational object of analysis, enabling rigorous ethical reasoning and long term safety evaluation of AI systems and other sociotechnical domains where harm is a first order concern. |
| 2026-01-23 | [GRIP: Algorithm-Agnostic Machine Unlearning for Mixture-of-Experts via Geometric Router Constraints](http://arxiv.org/abs/2601.16905v1) | Andy Zhu, Rongzhe Wei et al. | Machine unlearning (MU) for large language models has become critical for AI safety, yet existing methods fail to generalize to Mixture-of-Experts (MoE) architectures. We identify that traditional unlearning methods exploit MoE's architectural vulnerability: they manipulate routers to redirect queries away from knowledgeable experts rather than erasing knowledge, causing a loss of model utility and superficial forgetting. We propose Geometric Routing Invariance Preservation (GRIP), an algorithm-agnostic framework for unlearning for MoE. Our core contribution is a geometric constraint, implemented by projecting router gradient updates into an expert-specific null-space. Crucially, this decouples routing stability from parameter rigidity: while discrete expert selections remain stable for retained knowledge, the continuous router parameters remain plastic within the null space, allowing the model to undergo necessary internal reconfiguration to satisfy unlearning objectives. This forces the unlearning optimization to erase knowledge directly from expert parameters rather than exploiting the superficial router manipulation shortcut. GRIP functions as an adapter, constraining router parameter updates without modifying the underlying unlearning algorithm. Extensive experiments on large-scale MoE models demonstrate that our adapter eliminates expert selection shift (achieving over 95% routing stability) across all tested unlearning methods while preserving their utility. By preventing existing algorithms from exploiting MoE model's router vulnerability, GRIP adapts existing unlearning research from dense architectures to MoEs. |
| 2026-01-23 | [Mixture-of-Models: Unifying Heterogeneous Agents via N-Way Self-Evaluating Deliberation](http://arxiv.org/abs/2601.16863v1) | Tims Pecerskis, Aivars Smirnovs | This paper introduces the N-Way Self-Evaluating Deliberation (NSED) protocol, a Runtime Mixture-of-Models (MoM) architecture that constructs emergent composite models from a plurality of distinct expert agents. Unlike traditional Mixture-of-Experts (MoE) which rely on static gating networks, NSED employs a Dynamic Expertise Broker - a runtime optimization engine that treats model selection as a variation of the Knapsack Problem, binding heterogeneous checkpoints to functional roles based on live telemetry and cost constraints. At the execution layer, we formalize deliberation as a Macro-Scale Recurrent Neural Network (RNN), where the consensus state loops back through a semantic forget gate to enable iterative refinement without proportional VRAM scaling. Key components include an orchestration fabric for trustless N-to-N peer review, a Quadratic Voting activation function for non-linear consensus, and a feedback-driven state update. Empirical validation on challenging benchmarks (AIME 2025, LiveCodeBench) demonstrates that this topology allows ensembles of small (less than 20B) consumer-grade models to match or exceed the performance of state-of-the-art 100B+ parameter models, establishing a new hardware arbitrage efficiency frontier. Furthermore, testing on the DarkBench safety suite reveals intrinsic alignment properties, with peer-mediated correction reducing sycophancy scores below that of any individual agent. |
| 2026-01-23 | [Privacy in Human-AI Romantic Relationships: Concerns, Boundaries, and Agency](http://arxiv.org/abs/2601.16824v1) | Rongjun Ma, Shijing He et al. | An increasing number of LLM-based applications are being developed to facilitate romantic relationships with AI partners, yet the safety and privacy risks in these partnerships remain largely underexplored. In this work, we investigate privacy in human-AI romantic relationships through an interview study (N=17), examining participants' experiences and privacy perceptions across stages of exploration, intimacy, and dissolution, alongside platforms they used. We found that these relationships took varied forms, from one-to-one to one-to-many, and were shaped by multiple actors, including creators, platforms, and moderators. AI partners were perceived as having agency, actively negotiating privacy boundaries with participants and sometimes encouraging disclosure of personal details. As intimacy deepened, these boundaries became more permeable, though some participants voiced concerns such as conversation exposure and sought to preserve anonymity. Overall, platform affordances and diverse romantic dynamics expand the privacy landscape, underscoring the need to rethink how privacy is constructed in human-AI intimacy. |
| 2026-01-23 | [Adaptive Reinforcement and Model Predictive Control Switching for Safe Human-Robot Cooperative Navigation](http://arxiv.org/abs/2601.16686v1) | Ning Liu, Sen Shen et al. | This paper addresses the challenge of human-guided navigation for mobile collaborative robots under simultaneous proximity regulation and safety constraints. We introduce Adaptive Reinforcement and Model Predictive Control Switching (ARMS), a hybrid learning-control framework that integrates a reinforcement learning follower trained with Proximal Policy Optimization (PPO) and an analytical one-step Model Predictive Control (MPC) formulated as a quadratic program safety filter. To enable robust perception under partial observability and non-stationary human motion, ARMS employs a decoupled sensing architecture with a Long Short-Term Memory (LSTM) temporal encoder for the human-robot relative state and a spatial encoder for 360-degree LiDAR scans. The core contribution is a learned adaptive neural switcher that performs context-aware soft action fusion between the two controllers, favoring conservative, constraint-aware QP-based control in low-risk regions while progressively shifting control authority to the learned follower in highly cluttered or constrained scenarios where maneuverability is critical, and reverting to the follower action when the QP becomes infeasible. Extensive evaluations against Pure Pursuit, Dynamic Window Approach (DWA), and an RL-only baseline demonstrate that ARMS achieves an 82.5 percent success rate in highly cluttered environments, outperforming DWA and RL-only approaches by 7.1 percent and 3.1 percent, respectively, while reducing average computational latency by 33 percent to 5.2 milliseconds compared to a multi-step MPC baseline. Additional simulation transfer in Gazebo and initial real-world deployment results further indicate the practicality and robustness of ARMS for safe and efficient human-robot collaboration. Source code and a demonstration video are available at https://github.com/21ning/ARMS.git. |
| 2026-01-23 | [Predicting Networks Before They Happen: Experimentation on a Real-Time V2X Digital Twin](http://arxiv.org/abs/2601.16559v1) | Roberto Pegurri, Habu Shintaro et al. | Emerging safety-critical Vehicle-to-Everything (V2X) applications require networks to proactively adapt to rapid environmental changes rather than merely reacting to them. While Network Digital Twins (NDTs) offer a pathway to such predictive capabilities, existing solutions typically struggle to reconcile high-fidelity physical modeling with strict real-time constraints. This paper presents a novel, end-to-end real-time V2X Digital Twin framework that integrates live mobility tracking with deterministic channel simulation. By coupling the Tokyo Mobility Digital Twin-which provides live sensing and trajectory forecasting-with VaN3Twin-a full-stack simulator with ray tracing-we enable the prediction of network performance before physical events occur. We validate this approach through an experimental proof-of-concept deployed in Tokyo, Japan, featuring connected vehicles operating on 60 GHz links. Our results demonstrate the system's ability to predict Received Signal Strength (RSSI) with a maximum average error of 1.01 dB and reliably forecast Line-of-Sight (LoS) transitions within a maximum average end-to-end system latency of 250 ms, depending on the ray tracing level of detail. Furthermore, we quantify the fundamental trade-offs between digital model fidelity, computational latency, and trajectory prediction horizons, proving that high-fidelity and predictive digital twins are feasible in real-world urban environments. |
| 2026-01-23 | [SycoEval-EM: Sycophancy Evaluation of Large Language Models in Simulated Clinical Encounters for Emergency Care](http://arxiv.org/abs/2601.16529v1) | Dongshen Peng, Yi Wang et al. | Large language models (LLMs) show promise in clinical decision support yet risk acquiescing to patient pressure for inappropriate care. We introduce SycoEval-EM, a multi-agent simulation framework evaluating LLM robustness through adversarial patient persuasion in emergency medicine. Across 20 LLMs and 1,875 encounters spanning three Choosing Wisely scenarios, acquiescence rates ranged from 0-100\%. Models showed higher vulnerability to imaging requests (38.8\%) than opioid prescriptions (25.0\%), with model capability poorly predicting robustness. All persuasion tactics proved equally effective (30.0-36.0\%), indicating general susceptibility rather than tactic-specific weakness. Our findings demonstrate that static benchmarks inadequately predict safety under social pressure, necessitating multi-turn adversarial testing for clinical AI certification. |
| 2026-01-23 | [Competing Visions of Ethical AI: A Case Study of OpenAI](http://arxiv.org/abs/2601.16513v1) | Melissa Wilfley, Mengting Ai et al. | Introduction. AI Ethics is framed distinctly across actors and stakeholder groups. We report results from a case study of OpenAI analysing ethical AI discourse. Method. Research addressed: How has OpenAI's public discourse leveraged 'ethics', 'safety', 'alignment' and adjacent related concepts over time, and what does discourse signal about framing in practice? A structured corpus, differentiating between communication for a general audience and communication with an academic audience, was assembled from public documentation. Analysis. Qualitative content analysis of ethical themes combined inductively derived and deductively applied codes. Quantitative analysis leveraged computational content analysis methods via NLP to model topics and quantify changes in rhetoric over time. Visualizations report aggregate results. For reproducible results, we have released our code at https://github.com/famous-blue-raincoat/AI_Ethics_Discourse. Results. Results indicate that safety and risk discourse dominate OpenAI's public communication and documentation, without applying academic and advocacy ethics frameworks or vocabularies. Conclusions. Implications for governance are presented, along with discussion of ethics-washing practices in industry. |
| 2026-01-23 | [SafeThinker: Reasoning about Risk to Deepen Safety Beyond Shallow Alignment](http://arxiv.org/abs/2601.16506v1) | Xianya Fang, Xianying Luo et al. | Despite the intrinsic risk-awareness of Large Language Models (LLMs), current defenses often result in shallow safety alignment, rendering models vulnerable to disguised attacks (e.g., prefilling) while degrading utility. To bridge this gap, we propose SafeThinker, an adaptive framework that dynamically allocates defensive resources via a lightweight gateway classifier. Based on the gateway's risk assessment, inputs are routed through three distinct mechanisms: (i) a Standardized Refusal Mechanism for explicit threats to maximize efficiency; (ii) a Safety-Aware Twin Expert (SATE) module to intercept deceptive attacks masquerading as benign queries; and (iii) a Distribution-Guided Think (DDGT) component that adaptively intervenes during uncertain generation. Experiments show that SafeThinker significantly lowers attack success rates across diverse jailbreak strategies without compromising utility, demonstrating that coordinating intrinsic judgment throughout the generation process effectively balances robustness and practicality. |
| 2026-01-23 | [Persona Jailbreaking in Large Language Models](http://arxiv.org/abs/2601.16466v1) | Jivnesh Sandhan, Fei Cheng et al. | Large Language Models (LLMs) are increasingly deployed in domains such as education, mental health and customer support, where stable and consistent personas are critical for reliability. Yet, existing studies focus on narrative or role-playing tasks and overlook how adversarial conversational history alone can reshape induced personas. Black-box persona manipulation remains unexplored, raising concerns for robustness in realistic interactions. In response, we introduce the task of persona editing, which adversarially steers LLM traits through user-side inputs under a black-box, inference-only setting. To this end, we propose PHISH (Persona Hijacking via Implicit Steering in History), the first framework to expose a new vulnerability in LLM safety that embeds semantically loaded cues into user queries to gradually induce reverse personas. We also define a metric to quantify attack success. Across 3 benchmarks and 8 LLMs, PHISH predictably shifts personas, triggers collateral changes in correlated traits, and exhibits stronger effects in multi-turn settings. In high-risk domains mental health, tutoring, and customer support, PHISH reliably manipulates personas, validated by both human and LLM-as-Judge evaluations. Importantly, PHISH causes only a small reduction in reasoning benchmark performance, leaving overall utility largely intact while still enabling significant persona manipulation. While current guardrails offer partial protection, they remain brittle under sustained attack. Our findings expose new vulnerabilities in personas and highlight the need for context-resilient persona in LLMs. Our codebase and dataset is available at: https://github.com/Jivnesh/PHISH |
| 2026-01-23 | [Consensus In Asynchrony](http://arxiv.org/abs/2601.16460v1) | Ivan Klianev | We demonstrate sufficiency of events-based synchronisation for solving deterministic fault-tolerant consensus in asynchrony. Main result is an algorithm that terminates with valid vector agreement, hence operates with safety, liveness, and tolerance to one crash. Reconciling with the FLP impossibility result, we identified: i) existence of two types of agreements: data-independent and data-dependent; and ii) dependence of FLP theorem correctness on three implicit assumptions. Consensus impossibility with data-dependent agreement is contingent on two of them. The theorem-stated impossibility with every agreement type hinges entirely on the third. We provide experimental results showing that the third assumption has no evidence in support. |
| 2026-01-23 | [Ringmaster: How to juggle high-throughput host OS system calls from TrustZone TEEs](http://arxiv.org/abs/2601.16448v1) | Richard Habeeb, Man-Ki Yoon et al. | Many safety-critical systems require timely processing of sensor inputs to avoid potential safety hazards. Additionally, to support useful application features, such systems increasingly have a large rich operating system (OS) at the cost of potential security bugs. Thus, if a malicious party gains supervisor privileges, they could cause real-world damage by denying service to time-sensitive programs. Many past approaches to this problem completely isolate time-sensitive programs with a hypervisor; however, this prevents the programs from accessing useful OS services   We introduce Ringmaster, a novel framework that enables enclaves or TEEs (Trusted Execution Environments) to asynchronously access rich, but potentially untrusted, OS services via Linux's io_uring. When service is denied by the untrusted OS, enclaves continue to operate on Ringmaster's minimal ARM TrustZone kernel with access to small, critical device drivers. This approach balances the need for secure, time-sensitive processing with the convenience of rich OS services. Additionally, Ringmaster supports large unmodified programs as enclaves, offering lower overhead compared to existing systems. We demonstrate how Ringmaster helps us build a working highly-secure system with minimal engineering. In our experiments with an unmanned aerial vehicle, Ringmaster achieved nearly 1GiB/sec of data into enclave on a Raspberry Pi4b, 0-3% throughput overhead compared to non-enclave tasks. |
| 2026-01-23 | [RENEW: Risk- and Energy-Aware Navigation in Dynamic Waterways](http://arxiv.org/abs/2601.16424v1) | Mingi Jeong, Alberto Quattrini Li | We present RENEW, a global path planner for Autonomous Surface Vehicle (ASV) in dynamic environments with external disturbances (e.g., water currents). RENEW introduces a unified risk- and energy-aware strategy that ensures safety by dynamically identifying non-navigable regions and enforcing adaptive safety constraints. Inspired by maritime contingency planning, it employs a best-effort strategy to maintain control under adverse conditions. The hierarchical architecture combines high-level constrained triangulation for topological diversity with low-level trajectory optimization within safe corridors. Validated with real-world ocean data, RENEW is the first framework to jointly address adaptive non-navigability and topological path diversity for robust maritime navigation. |
| 2026-01-23 | [Reasoning-Enhanced Rare-Event Prediction with Balanced Outcome Correction](http://arxiv.org/abs/2601.16406v1) | Vitaly Bulgakov, Alexander Turchin | Rare-event prediction is critical in domains such as healthcare, finance, reliability engineering, customer support, aviation safety, where positive outcomes are infrequent yet potentially catastrophic. Extreme class imbalance biases conventional models toward majority-class predictions, limiting recall, calibration, and operational usefulness. We propose LPCORP (Low-Prevalence CORrector for Prediction)*, a two-stage framework that combines reasoningenhanced prediction with confidence-based outcome correction. A reasoning model first produces enriched predictions from narrative inputs, after which a lightweight logistic-regression classifier evaluates and selectively corrects these outputs to mitigate prevalence-driven bias. We evaluate LPCORP on real-world datasets from medical and consumer service domains. The results show that this method transforms a highly imbalanced setting into a well-balanced one while preserving the original number of samples and without applying any resampling strategies. Test-set evaluation demonstrates substantially improved performance, particularly in precision, which is a known weakness in low-prevalence data. We further provide a costreduction analysis comparing the expenses associated with rare-event damage control without preventive measures to those incurred when low-cost, prediction-based preventive interventions are applied that showed more than 50% reduction in some cases. * Patent pending: U.S. Provisional 63/933,518, filed 8 December 2025. |
| 2026-01-23 | [Reinforcement Learning-Based Energy-Aware Coverage Path Planning for Precision Agriculture](http://arxiv.org/abs/2601.16405v1) | Beining Wu, Zihao Ding et al. | Coverage Path Planning (CPP) is a fundamental capability for agricultural robots; however, existing solutions often overlook energy constraints, resulting in incomplete operations in large-scale or resource-limited environments. This paper proposes an energy-aware CPP framework grounded in Soft Actor-Critic (SAC) reinforcement learning, designed for grid-based environments with obstacles and charging stations. To enable robust and adaptive decision-making under energy limitations, the framework integrates Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal dynamics. A dedicated reward function is designed to jointly optimize coverage efficiency, energy consumption, and return-to-base constraints. Experimental results demonstrate that the proposed approach consistently achieves over 90% coverage while ensuring energy safety, outperforming traditional heuristic algorithms such as Rapidly-exploring Random Tree (RRT), Particle Swarm Optimization (PSO), and Ant Colony Optimization (ACO) baselines by 13.4-19.5% in coverage and reducing constraint violations by 59.9-88.3%. These findings validate the proposed SAC-based framework as an effective and scalable solution for energy-constrained CPP in agricultural robotics. |
| 2026-01-22 | [Stochastic Control Barrier Functions under State Estimation: From Euclidean Space to Lie Groups](http://arxiv.org/abs/2601.16198v1) | Ruoyu Lin, Magnus Egerstedt | Ensuring safety for autonomous systems under uncertainty remains challenging, particularly when safety of the true state is required despite the true state not being fully known. Control barrier functions (CBFs) have become widely adopted as safety filters. However, standard CBF formulations do not explicitly account for state estimation uncertainty and its propagation, especially for stochastic systems evolving on manifolds. In this paper, we propose a safety-critical control framework with a provable bound on the finite-time safety probability for stochastic systems under noisy state information. The proposed framework explicitly incorporates the uncertainty arising from both process and measurement noise, and synthesizes controllers that adapt to the level of uncertainty. The framework admits closed-form solutions in linear settings, and experimental results demonstrate its effectiveness on systems whose state spaces range from Euclidean space to Lie groups. |
| 2026-01-22 | [Stochastic Control Barrier Functions under State Estimation: From Euclidean Space to Lie Groups](http://arxiv.org/abs/2601.16198v2) | Ruoyu Lin, Magnus Egerstedt | Ensuring safety for autonomous systems under uncertainty remains challenging, particularly when safety of the true state is required despite the true state not being fully known. Control barrier functions (CBFs) have become widely adopted as safety filters. However, standard CBF formulations do not explicitly account for state estimation uncertainty and its propagation, especially for stochastic systems evolving on manifolds. In this paper, we propose a safety-critical control framework with a provable bound on the finite-time safety probability for stochastic systems under noisy state information. The proposed framework explicitly incorporates the uncertainty arising from both process and measurement noise, and synthesizes controllers that adapt to the level of uncertainty. The framework admits closed-form solutions in linear settings, and experimental results demonstrate its effectiveness on systems whose state spaces range from Euclidean space to Lie groups. |
| 2026-01-22 | [Explainable AI to Improve Machine Learning Reliability for Industrial Cyber-Physical Systems](http://arxiv.org/abs/2601.16074v1) | Annemarie Jutte, Uraz Odyurt | Industrial Cyber-Physical Systems (CPS) are sensitive infrastructure from both safety and economics perspectives, making their reliability critically important. Machine Learning (ML), specifically deep learning, is increasingly integrated in industrial CPS, but the inherent complexity of ML models results in non-transparent operation. Rigorous evaluation is needed to prevent models from exhibiting unexpected behaviour on future, unseen data. Explainable AI (XAI) can be used to uncover model reasoning, allowing a more extensive analysis of behaviour. We apply XAI to to improve predictive performance of ML models intended for industrial CPS. We analyse the effects of components from time-series data decomposition on model predictions using SHAP values. Through this method, we observe evidence on the lack of sufficient contextual information during model training. By increasing the window size of data instances, informed by the XAI findings, we are able to improve model performance. |
| 2026-01-22 | [Universal Refusal Circuits Across LLMs: Cross-Model Transfer via Trajectory Replay and Concept-Basis Reconstruction](http://arxiv.org/abs/2601.16034v1) | Tony Cristofano | Refusal behavior in aligned LLMs is often viewed as model-specific, yet we hypothesize it stems from a universal, low-dimensional semantic circuit shared across models. To test this, we introduce Trajectory Replay via Concept-Basis Reconstruction, a framework that transfers refusal interventions from donor to target models, spanning diverse architectures (e.g., Dense to MoE) and training regimes, without using target-side refusal supervision. By aligning layers via concept fingerprints and reconstructing refusal directions using a shared ``recipe'' of concept atoms, we map the donor's ablation trajectory into the target's semantic space. To preserve capabilities, we introduce a weight-SVD stability guard that projects interventions away from high-variance weight subspaces to prevent collateral damage. Our evaluation across 8 model pairs (including GPT-OSS-20B and GLM-4) confirms that these transferred recipes consistently attenuate refusal while maintaining performance, providing strong evidence for the semantic universality of safety alignment. |

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



