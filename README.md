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
| 2026-02-18 | [Towards a Science of AI Agent Reliability](http://arxiv.org/abs/2602.16666v1) | Stephan Rabanser, Sayash Kapoor et al. | AI agents are increasingly deployed to execute important tasks. While rising accuracy scores on standard benchmarks suggest rapid progress, many agents still continue to fail in practice. This discrepancy highlights a fundamental limitation of current evaluations: compressing agent behavior into a single success metric obscures critical operational flaws. Notably, it ignores whether agents behave consistently across runs, withstand perturbations, fail predictably, or have bounded error severity. Grounded in safety-critical engineering, we provide a holistic performance profile by proposing twelve concrete metrics that decompose agent reliability along four key dimensions: consistency, robustness, predictability, and safety. Evaluating 14 agentic models across two complementary benchmarks, we find that recent capability gains have only yielded small improvements in reliability. By exposing these persistent limitations, our metrics complement traditional evaluations while offering tools for reasoning about how agents perform, degrade, and fail. |
| 2026-02-18 | [Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment](http://arxiv.org/abs/2602.16660v1) | Yuyan Bu, Xiaohao Liu et al. | The widespread deployment of large language models (LLMs) across linguistic communities necessitates reliable multilingual safety alignment. However, recent efforts to extend alignment to other languages often require substantial resources, either through large-scale, high-quality supervision in the target language or through pairwise alignment with high-resource languages, which limits scalability. In this work, we propose a resource-efficient method for improving multilingual safety alignment. We introduce a plug-and-play Multi-Lingual Consistency (MLC) loss that can be integrated into existing monolingual alignment pipelines. By improving collinearity between multilingual representation vectors, our method encourages directional consistency at the multilingual semantic level in a single update. This allows simultaneous alignment across multiple languages using only multilingual prompt variants without requiring additional response-level supervision in low-resource languages. We validate the proposed method across different model architectures and alignment paradigms, and demonstrate its effectiveness in enhancing multilingual safety with limited impact on general model utility. Further evaluation across languages and tasks indicates improved cross-lingual generalization, suggesting the proposed approach as a practical solution for multilingual consistency alignment under limited supervision. |
| 2026-02-18 | [Agentic AI, Medical Morality, and the Transformation of the Patient-Physician Relationship](http://arxiv.org/abs/2602.16553v1) | Robert Ranisch, Sabine Salloch | The emergence of agentic AI marks a new phase in the digital transformation of healthcare. Distinct from conventional generative AI, agentic AI systems are capable of autonomous, goal-directed actions and complex task coordination. They promise to support or even collaborate with clinicians and patients in increasingly independent ways. While agentic AI raises familiar moral concerns regarding safety, accountability, and bias, this article focuses on a less explored dimension: its capacity to transform the moral fabric of healthcare itself. Drawing on the framework of techno-moral change and the three domains of decision, relation and perception, we investigate how agentic AI might reshape the patient-physician relationship and reconfigure core concepts of medical morality. We argue that these shifts, while not fully predictable, demand ethical attention before widespread deployment. Ultimately, the paper calls for integrating ethical foresight into the design and use of agentic AI. |
| 2026-02-18 | [Vulnerability Analysis of Safe Reinforcement Learning via Inverse Constrained Reinforcement Learning](http://arxiv.org/abs/2602.16543v1) | Jialiang Fan, Shixiong Jiang et al. | Safe reinforcement learning (Safe RL) aims to ensure policy performance while satisfying safety constraints. However, most existing Safe RL methods assume benign environments, making them vulnerable to adversarial perturbations commonly encountered in real-world settings. In addition, existing gradient-based adversarial attacks typically require access to the policy's gradient information, which is often impractical in real-world scenarios. To address these challenges, we propose an adversarial attack framework to reveal vulnerabilities of Safe RL policies. Using expert demonstrations and black-box environment interaction, our framework learns a constraint model and a surrogate (learner) policy, enabling gradient-based attack optimization without requiring the victim policy's internal gradients or the ground-truth safety constraints. We further provide theoretical analysis establishing feasibility and deriving perturbation bounds. Experiments on multiple Safe RL benchmarks demonstrate the effectiveness of our approach under limited privileged access. |
| 2026-02-18 | [Recursive language models for jailbreak detection: a procedural defense for tool-augmented agents](http://arxiv.org/abs/2602.16520v1) | Doron Shavit | Jailbreak prompts are a practical and evolving threat to large language models (LLMs), particularly in agentic systems that execute tools over untrusted content. Many attacks exploit long-context hiding, semantic camouflage, and lightweight obfuscations that can evade single-pass guardrails. We present RLM-JB, an end-to-end jailbreak detection framework built on Recursive Language Models (RLMs), in which a root model orchestrates a bounded analysis program that transforms the input, queries worker models over covered segments, and aggregates evidence into an auditable decision. RLM-JB treats detection as a procedure rather than a one-shot classification: it normalizes and de-obfuscates suspicious inputs, chunks text to reduce context dilution and guarantee coverage, performs parallel chunk screening, and composes cross-chunk signals to recover split-payload attacks. On AutoDAN-style adversarial inputs, RLM-JB achieves high detection effectiveness across three LLM backends (ASR/Recall 92.5-98.0%) while maintaining very high precision (98.99-100%) and low false positive rates (0.0-2.0%), highlighting a practical sensitivity-specificity trade-off as the screening backend changes. |
| 2026-02-18 | [VIGOR: Visual Goal-In-Context Inference for Unified Humanoid Fall Safety](http://arxiv.org/abs/2602.16511v1) | Osher Azulay, Zhengjie Xu et al. | Reliable fall recovery is critical for humanoids operating in cluttered environments. Unlike quadrupeds or wheeled robots, humanoids experience high-energy impacts, complex whole-body contact, and large viewpoint changes during a fall, making recovery essential for continued operation. Existing methods fragment fall safety into separate problems such as fall avoidance, impact mitigation, and stand-up recovery, or rely on end-to-end policies trained without vision through reinforcement learning or imitation learning, often on flat terrain. At a deeper level, fall safety is treated as monolithic data complexity, coupling pose, dynamics, and terrain and requiring exhaustive coverage, limiting scalability and generalization. We present a unified fall safety approach that spans all phases of fall recovery. It builds on two insights: 1) Natural human fall and recovery poses are highly constrained and transferable from flat to complex terrain through alignment, and 2) Fast whole-body reactions require integrated perceptual-motor representations. We train a privileged teacher using sparse human demonstrations on flat terrain and simulated complex terrains, and distill it into a deployable student that relies only on egocentric depth and proprioception. The student learns how to react by matching the teacher's goal-in-context latent representation, which combines the next target pose with the local terrain, rather than separately encoding what it must perceive and how it must act. Results in simulation and on a real Unitree G1 humanoid demonstrate robust, zero-shot fall safety across diverse non-flat environments without real-world fine-tuning. The project page is available at https://vigor2026.github.io/ |
| 2026-02-18 | [MMA: Multimodal Memory Agent](http://arxiv.org/abs/2602.16493v1) | Yihao Lu, Wanru Cheng et al. | Long-horizon multimodal agents depend on external memory; however, similarity-based retrieval often surfaces stale, low-credibility, or conflicting items, which can trigger overconfident errors. We propose Multimodal Memory Agent (MMA), which assigns each retrieved memory item a dynamic reliability score by combining source credibility, temporal decay, and conflict-aware network consensus, and uses this signal to reweight evidence and abstain when support is insufficient. We also introduce MMA-Bench, a programmatically generated benchmark for belief dynamics with controlled speaker reliability and structured text-vision contradictions. Using this framework, we uncover the "Visual Placebo Effect", revealing how RAG-based agents inherit latent visual biases from foundation models. On FEVER, MMA matches baseline accuracy while reducing variance by 35.2% and improving selective utility; on LoCoMo, a safety-oriented configuration improves actionable accuracy and reduces wrong answers; on MMA-Bench, MMA reaches 41.18% Type-B accuracy in Vision mode, while the baseline collapses to 0.0% under the same protocol. Code: https://github.com/AIGeeksGroup/MMA. |
| 2026-02-18 | [Certifying Hamilton-Jacobi Reachability Learned via Reinforcement Learning](http://arxiv.org/abs/2602.16475v1) | Prashant Solanki, Isabelle El-Hajj et al. | We present a framework to \emph{certify} Hamilton--Jacobi (HJ) reachability learned by reinforcement learning (RL). Building on a discounted initial time \emph{travel-cost} formulation that makes small-step RL value iteration provably equivalent to a forward Hamilton--Jacobi (HJ) equation with damping, we convert certified learning errors into calibrated inner/outer enclosures of strict backward reachable tube. The core device is an additive-offset identity: if $W_λ$ solves the discounted travel-cost Hamilton--Jacobi--Bellman (HJB) equation, then $W_\varepsilon:=W_λ+ \varepsilon$ solves the same PDE with a constant offset $λ\varepsilon$. This means that a uniform value error is \emph{exactly} equal to a constant HJB offset. We establish this uniform value error via two routes: (A) a Bellman operator-residual bound, and (B) a HJB PDE-slack bound. Our framework preserves HJ-level safety semantics and is compatible with deep RL. We demonstrate the approach on a double-integrator system by formally certifying, via satisfiability modulo theories (SMT), a value function learned through reinforcement learning to induce provably correct inner and outer backward-reachable set enclosures over a compact region of interest. |
| 2026-02-18 | [Reactive Motion Generation With Particle-Based Perception in Dynamic Environments](http://arxiv.org/abs/2602.16462v1) | Xiyuan Zhao, Huijun Li et al. | Reactive motion generation in dynamic and unstructured scenarios is typically subject to essentially static perception and system dynamics. Reliably modeling dynamic obstacles and optimizing collision-free trajectories under perceptive and control uncertainty are challenging. This article focuses on revealing tight connection between reactive planning and dynamic mapping for manipulators from a model-based perspective. To enable efficient particle-based perception with expressively dynamic property, we present a tensorized particle weight update scheme that explicitly maintains obstacle velocities and covariance meanwhile. Building upon this dynamic representation, we propose an obstacle-aware MPPI-based planning formulation that jointly propagates robot-obstacle dynamics, allowing future system motion to be predicted and evaluated under uncertainty. The model predictive method is shown to significantly improve safety and reactivity with dynamic surroundings. By applying our complete framework in simulated and noisy real-world environments, we demonstrate that explicit modeling of robot-obstacle dynamics consistently enhances performance over state-of-the-art MPPI-based perception-planning baselines avoiding multiple static and dynamic obstacles. |
| 2026-02-18 | [Parameter-Free Adaptive Multi-Scale Channel-Spatial Attention Aggregation framework for 3D Indoor Semantic Scene Completion Toward Assisting Visually Impaired](http://arxiv.org/abs/2602.16385v1) | Qi He, XiangXiang Wang et al. | In indoor assistive perception for visually impaired users, 3D Semantic Scene Completion (SSC) is expected to provide structurally coherent and semantically consistent occupancy under strictly monocular vision for safety-critical scene understanding. However, existing monocular SSC approaches often lack explicit modeling of voxel-feature reliability and regulated cross-scale information propagation during 2D-3D projection and multi-scale fusion, making them vulnerable to projection diffusion and feature entanglement and thus limiting structural stability.To address these challenges, this paper presents an Adaptive Multi-scale Attention Aggregation (AMAA) framework built upon the MonoScene pipeline. Rather than introducing a heavier backbone, AMAA focuses on reliability-oriented feature regulation within a monocular SSC framework. Specifically, lifted voxel features are jointly calibrated in semantic and spatial dimensions through parallel channel-spatial attention aggregation, while multi-scale encoder-decoder fusion is stabilized via a hierarchical adaptive feature-gating strategy that regulates information injection across scales.Experiments on the NYUv2 benchmark demonstrate consistent improvements over MonoScene without significantly increasing system complexity: AMAA achieves 27.25% SSC mIoU (+0.31) and 43.10% SC IoU (+0.59). In addition, system-level deployment on an NVIDIA Jetson platform verifies that the complete AMAA framework can be executed stably on embedded hardware. Overall, AMAA improves monocular SSC quality and provides a reliable and deployable perception framework for indoor assistive systems targeting visually impaired users. |
| 2026-02-18 | [Dual-Quadruped Collaborative Transportation in Narrow Environments via Safe Reinforcement Learning](http://arxiv.org/abs/2602.16353v1) | Zhezhi Lei, Zhihai Bi et al. | Collaborative transportation, where multiple robots collaboratively transport a payload, has garnered significant attention in recent years. While ensuring safe and high-performance inter-robot collaboration is critical for effective task execution, it is difficult to pursue in narrow environments where the feasible region is extremely limited. To address this challenge, we propose a novel approach for dual-quadruped collaborative transportation via safe reinforcement learning (RL). Specifically, we model the task as a fully cooperative constrained Markov game, where collision avoidance is formulated as constraints. We introduce a cost-advantage decomposition method that enforces the sum of team constraints to remain below an upper bound, thereby guaranteeing task safety within an RL framework. Furthermore, we propose a constraint allocation method that assigns shared constraints to individual robots to maximize the overall task reward, encouraging autonomous task-assignment among robots, thereby improving collaborative task performance. Simulation and real-time experimental results demonstrate that the proposed approach achieves superior performance and a higher success rate in dual-quadruped collaborative transportation compared to existing methods. |
| 2026-02-18 | [Helpful to a Fault: Measuring Illicit Assistance in Multi-Turn, Multilingual LLM Agents](http://arxiv.org/abs/2602.16346v1) | Nivya Talokar, Ayush K Tarun et al. | LLM-based agents execute real-world workflows via tools and memory. These affordances enable ill-intended adversaries to also use these agents to carry out complex misuse scenarios. Existing agent misuse benchmarks largely test single-prompt instructions, leaving a gap in measuring how agents end up helping with harmful or illegal tasks over multiple turns. We introduce STING (Sequential Testing of Illicit N-step Goal execution), an automated red-teaming framework that constructs a step-by-step illicit plan grounded in a benign persona and iteratively probes a target agent with adaptive follow-ups, using judge agents to track phase completion. We further introduce an analysis framework that models multi-turn red-teaming as a time-to-first-jailbreak random variable, enabling analysis tools like discovery curves, hazard-ratio attribution by attack language, and a new metric: Restricted Mean Jailbreak Discovery. Across AgentHarm scenarios, STING yields substantially higher illicit-task completion than single-turn prompting and chat-oriented multi-turn baselines adapted to tool-using agents. In multilingual evaluations across six non-English settings, we find that attack success and illicit-task completion do not consistently increase in lower-resource languages, diverging from common chatbot findings. Overall, STING provides a practical way to evaluate and stress-test agent misuse in realistic deployment settings, where interactions are inherently multi-turn and often multilingual. |
| 2026-02-18 | [Collaborative Safe Bayesian Optimization](http://arxiv.org/abs/2602.16235v1) | Alina Castell Blasco, Maxime Bouton | Mobile networks require safe optimization to adapt to changing conditions in traffic demand and signal transmission quality, in addition to improving service performance metrics. With the increasing complexity of emerging mobile networks, traditional parameter tuning methods become too conservative or complex to evaluate. For the first time, we apply safe Bayesian optimization to mobile networks. Moreover, we develop a new safe collaborative optimization algorithm called CoSBO, leveraging information from multiple optimization tasks in the network and considering multiple safety constraints. The resulting algorithm is capable of safely tuning the network parameter online with very few iterations. We demonstrate that the proposed method improves sample efficiency in the early stages of the optimization process by comparing it against the SafeOpt-MC algorithm in a mobile network scenario. |
| 2026-02-18 | [UCTECG-Net: Uncertainty-aware Convolution Transformer ECG Network for Arrhythmia Detection](http://arxiv.org/abs/2602.16216v1) | Hamzeh Asgharnezhad, Pegah Tabarisaadi et al. | Deep learning has improved automated electrocardiogram (ECG) classification, but limited insight into prediction reliability hinders its use in safety-critical settings. This paper proposes UCTECG-Net, an uncertainty-aware hybrid architecture that combines one-dimensional convolutions and Transformer encoders to process raw ECG signals and their spectrograms jointly. Evaluated on the MIT-BIH Arrhythmia and PTB Diagnostic datasets, UCTECG-Net outperforms LSTM, CNN1D, and Transformer baselines in terms of accuracy, precision, recall and F1 score, achieving up to 98.58% accuracy on MIT-BIH and 99.14% on PTB. To assess predictive reliability, we integrate three uncertainty quantification methods (Monte Carlo Dropout, Deep Ensembles, and Ensemble Monte Carlo Dropout) into all models and analyze their behavior using an uncertainty-aware confusion matrix and derived metrics. The results show that UCTECG-Net, particularly with Ensemble or EMCD, provides more reliable and better-aligned uncertainty estimates than competing architectures, offering a stronger basis for risk-aware ECG decision support. |
| 2026-02-18 | [SIT-LMPC: Safe Information-Theoretic Learning Model Predictive Control for Iterative Tasks](http://arxiv.org/abs/2602.16187v1) | Zirui Zang, Ahmad Amine et al. | Robots executing iterative tasks in complex, uncertain environments require control strategies that balance robustness, safety, and high performance. This paper introduces a safe information-theoretic learning model predictive control (SIT-LMPC) algorithm for iterative tasks. Specifically, we design an iterative control framework based on an information-theoretic model predictive control algorithm to address a constrained infinite-horizon optimal control problem for discrete-time nonlinear stochastic systems. An adaptive penalty method is developed to ensure safety while balancing optimality. Trajectories from previous iterations are utilized to learn a value function using normalizing flows, which enables richer uncertainty modeling compared to Gaussian priors. SIT-LMPC is designed for highly parallel execution on graphics processing units, allowing efficient real-time optimization. Benchmark simulations and hardware experiments demonstrate that SIT-LMPC iteratively improves system performance while robustly satisfying system constraints. |
| 2026-02-18 | [Axle Sensor Fusion for Online Continual Wheel Fault Detection in Wayside Railway Monitoring](http://arxiv.org/abs/2602.16101v1) | Afonso Lourenço, Francisca Osório et al. | Reliable and cost-effective maintenance is essential for railway safety, particularly at the wheel-rail interface, which is prone to wear and failure. Predictive maintenance frameworks increasingly leverage sensor-generated time-series data, yet traditional methods require manual feature engineering, and deep learning models often degrade in online settings with evolving operational patterns. This work presents a semantic-aware, label-efficient continual learning framework for railway fault diagnostics. Accelerometer signals are encoded via a Variational AutoEncoder into latent representations capturing the normal operational structure in a fully unsupervised manner. Importantly, semantic metadata, including axle counts, wheel indexes, and strain-based deformations, is extracted via AI-driven peak detection on fiber Bragg grating sensors (resistant to electromagnetic interference) and fused with the VAE embeddings, enhancing anomaly detection under unknown operational conditions. A lightweight gradient boosting supervised classifier stabilizes anomaly scoring with minimal labels, while a replay-based continual learning strategy enables adaptation to evolving domains without catastrophic forgetting. Experiments show the model detects minor imperfections due to flats and polygonization, while adapting to evolving operational conditions, such as changes in train type, speed, load, and track profiles, captured using a single accelerometer and strain gauge in wayside monitoring. |
| 2026-02-17 | [The Limits of Long-Context Reasoning in Automated Bug Fixing](http://arxiv.org/abs/2602.16069v1) | Ravi Raju, Mengmeng Ji et al. | Rapidly increasing context lengths have led to the assumption that large language models (LLMs) can directly reason over entire codebases. Concurrently, recent advances in LLMs have enabled strong performance on software engineering benchmarks, particularly when paired with agentic workflows. In this work, we systematically evaluate whether current LLMs can reliably perform long-context code debugging and patch generation. Using SWE-bench Verified as a controlled experimental setting, we first evaluate state-of-the-art models within an agentic harness (mini-SWE-agent), where performance improves substantially: GPT-5-nano achieves up to a 31\% resolve rate on 100 samples, and open-source models such as Deepseek-R1-0528 obtain competitive results. However, token-level analysis shows that successful agentic trajectories typically remain under 20k tokens, and that longer accumulated contexts correlate with lower success rates, indicating that agentic success primarily arises from task decomposition into short-context steps rather than effective long-context reasoning. To directly test long-context capability, we construct a data pipeline where we artificially inflate the context length of the input by placing the relevant files into the context (ensuring perfect retrieval recall); we then study single-shot patch generation under genuinely long contexts (64k-128k tokens). Despite this setup, performance degrades sharply: Qwen3-Coder-30B-A3B achieves only a 7\% resolve rate at 64k context, while GPT-5-nano solves none of the tasks. Qualitative analysis reveals systematic failure modes, including hallucinated diffs, incorrect file targets, and malformed patch headers. Overall, our findings highlight a significant gap between nominal context length and usable context capacity in current LLMs, and suggest that existing agentic coding benchmarks do not meaningfully evaluate long-context reasoning. |
| 2026-02-17 | [Extracting and Analyzing Rail Crossing Behavior Signatures from Videos using Tensor Methods](http://arxiv.org/abs/2602.16057v1) | Dawon Ahn, Het Patel et al. | Railway crossings present complex safety challenges where driver behavior varies by location, time, and conditions. Traditional approaches analyze crossings individually, limiting the ability to identify shared behavioral patterns across locations. We propose a multi-view tensor decomposition framework that captures behavioral similarities across three temporal phases: Approach (warning activation to gate lowering), Waiting (gates down to train passage), and Clearance (train passage to gate raising). We analyze railway crossing videos from multiple locations using TimeSformer embeddings to represent each phase. By constructing phase-specific similarity matrices and applying non-negative symmetric CP decomposition, we discover latent behavioral components with distinct temporal signatures. Our tensor analysis reveals that crossing location appears to be a stronger determinant of behavior patterns than time of day, and that approach-phase behavior provides particularly discriminative signatures. Visualization of the learned component space confirms location-based clustering, with certain crossings forming distinct behavioral clusters. This automated framework enables scalable pattern discovery across multiple crossings, providing a foundation for grouping locations by behavioral similarity to inform targeted safety interventions. |
| 2026-02-17 | [Multi-Objective Alignment of Language Models for Personalized Psychotherapy](http://arxiv.org/abs/2602.16053v1) | Mehrab Beikzadeh, Yasaman Asadollah Salmanpour et al. | Mental health disorders affect over 1 billion people worldwide, yet access to care remains limited by workforce shortages and cost constraints. While AI systems show therapeutic promise, current alignment approaches optimize objectives independently, failing to balance patient preferences with clinical safety. We survey 335 individuals with lived mental health experience to collect preference rankings across therapeutic dimensions, then develop a multi-objective alignment framework using direct preference optimization. We train reward models for six criteria -- empathy, safety, active listening, self-motivated change, trust/rapport, and patient autonomy -- and systematically compare multi-objective approaches against single-objective optimization, supervised fine-tuning, and parameter merging. Multi-objective DPO (MODPO) achieves superior balance (77.6% empathy, 62.6% safety) compared to single-objective optimization (93.6% empathy, 47.8% safety), and therapeutic criteria outperform general communication principles by 17.2%. Blinded clinician evaluation confirms MODPO is consistently preferred, with LLM-evaluator agreement comparable to inter-clinician reliability. |
| 2026-02-17 | [MedProbCLIP: Probabilistic Adaptation of Vision-Language Foundation Model for Reliable Radiograph-Report Retrieval](http://arxiv.org/abs/2602.16019v1) | Ahmad Elallaf, Yu Zhang et al. | Vision-language foundation models have emerged as powerful general-purpose representation learners with strong potential for multimodal understanding, but their deterministic embeddings often fail to provide the reliability required for high-stakes biomedical applications. This work introduces MedProbCLIP, a probabilistic vision-language learning framework for chest X-ray and radiology report representation learning and bidirectional retrieval. MedProbCLIP models image and text representations as Gaussian embeddings through a probabilistic contrastive objective that explicitly captures uncertainty and many-to-many correspondences between radiographs and clinical narratives. A variational information bottleneck mitigates overconfident predictions, while MedProbCLIP employs multi-view radiograph encoding and multi-section report encoding during training to provide fine-grained supervision for clinically aligned correspondence, yet requires only a single radiograph and a single report at inference. Evaluated on the MIMIC-CXR dataset, MedProbCLIP outperforms deterministic and probabilistic baselines, including CLIP, CXR-CLIP, and PCME++, in both retrieval and zero-shot classification. Beyond accuracy, MedProbCLIP demonstrates superior calibration, risk-coverage behavior, selective retrieval reliability, and robustness to clinically relevant corruptions, underscoring the value of probabilistic vision-language modeling for improving the trustworthiness and safety of radiology image-text retrieval systems. |

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



