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
| 2025-11-04 | [Optimizing AI Agent Attacks With Synthetic Data](http://arxiv.org/abs/2511.02823v1) | Chloe Loughridge, Paul Colognese et al. | As AI deployments become more complex and high-stakes, it becomes increasingly important to be able to estimate their risk. AI control is one framework for doing so. However, good control evaluations require eliciting strong attack policies. This can be challenging in complex agentic environments where compute constraints leave us data-poor. In this work, we show how to optimize attack policies in SHADE-Arena, a dataset of diverse realistic control environments. We do this by decomposing attack capability into five constituent skills -- suspicion modeling, attack selection, plan synthesis, execution and subtlety -- and optimizing each component individually. To get around the constraint of limited data, we develop a probabilistic model of attack dynamics, optimize our attack hyperparameters using this simulation, and then show that the results transfer to SHADE-Arena. This results in a substantial improvement in attack strength, reducing safety score from a baseline of 0.87 to 0.41 using our scaffold. |
| 2025-11-04 | [Curriculum Design for Trajectory-Constrained Agent: Compressing Chain-of-Thought Tokens in LLMs](http://arxiv.org/abs/2511.02690v1) | Georgios Tzannetos, Parameswaran Kamalaruban et al. | Training agents to operate under strict constraints during deployment, such as limited resource budgets or stringent safety requirements, presents significant challenges, especially when these constraints render the task complex. In this work, we propose a curriculum learning strategy that gradually tightens constraints during training, enabling the agent to incrementally master the deployment requirements. Inspired by self-paced learning techniques in unconstrained reinforcement learning (RL), our approach facilitates a smoother transition to challenging environments by initially training on simplified versions of the constraints and progressively introducing the full deployment conditions. We provide a theoretical analysis using an RL agent in a binary-tree Markov Decision Process (MDP) to demonstrate that our curriculum strategy can accelerate training relative to a baseline approach that imposes the trajectory constraints from the outset. Moreover, we empirically validate the effectiveness and generality of our method across both RL and large language model (LLM) agents in diverse settings, including a binary-tree MDP, a multi-task navigation domain, and a math reasoning task with two benchmarks. These results highlight the potential of curriculum design in enhancing the efficiency and performance of agents operating under complex trajectory constraints during deployment. Moreover, when applied to LLMs, our strategy enables compression of output chain-of-thought tokens, achieving a substantial inference speedup on consumer hardware, demonstrating its effectiveness for resource-constrained deployment. |
| 2025-11-04 | [Influence Diagrams for Robust Multi-Target Tracking](http://arxiv.org/abs/2511.02637v1) | Priyank Behera, C. Robert Kenley | Multi-Target Tracking (MTT) is foundational for radar, defense, and autonomous systems, where tracking accuracy directly affects decision-making and safety. For linear systems with Gaussian process and measurement noise, the Kalman filter remains the gold standard for state estimation. However, its performance can degrade in real-world scenarios where measurement noise is temporally correlated. This violates the white-noise assumptions that Kalman filters have. Various approaches include state augmentation of the Kalman filter, but this approach is susceptible to failure due to ill-conditioned problem formulations. This work investigates the limitations of classical Kalman filtering in colored noise environments and presents an influence diagram-based approach to the Joint Probabilistic Data Association Filter (JPDAF). Simulation results on benchmark scenarios demonstrate that the Influence Diagram JPDAF (ID-JPDAF) achieves lower root mean square error (RMSE) than classical methods. These findings highlight the potential of influence diagram models for advancing multi-target tracking performance in radar and related applications. |
| 2025-11-04 | [Adaptive GR(1) Specification Repair for Liveness-Preserving Shielding in Reinforcement Learning](http://arxiv.org/abs/2511.02605v1) | Tiberiu-Andrei Georgescu, Alexander W. Goodall et al. | Shielding is widely used to enforce safety in reinforcement learning (RL), ensuring that an agent's actions remain compliant with formal specifications. Classical shielding approaches, however, are often static, in the sense that they assume fixed logical specifications and hand-crafted abstractions. While these static shields provide safety under nominal assumptions, they fail to adapt when environment assumptions are violated. In this paper, we develop the first adaptive shielding framework - to the best of our knowledge - based on Generalized Reactivity of rank 1 (GR(1)) specifications, a tractable and expressive fragment of Linear Temporal Logic (LTL) that captures both safety and liveness properties. Our method detects environment assumption violations at runtime and employs Inductive Logic Programming (ILP) to automatically repair GR(1) specifications online, in a systematic and interpretable way. This ensures that the shield evolves gracefully, ensuring liveness is achievable and weakening goals only when necessary. We consider two case studies: Minepump and Atari Seaquest; showing that (i) static symbolic controllers are often severely suboptimal when optimizing for auxiliary rewards, and (ii) RL agents equipped with our adaptive shield maintain near-optimal reward and perfect logical compliance compared with static shields. |
| 2025-11-04 | [Trustworthy Quantum Machine Learning: A Roadmap for Reliability, Robustness, and Security in the NISQ Era](http://arxiv.org/abs/2511.02602v1) | Ferhat Ozgur Catak, Jungwon Seo et al. | Quantum machine learning (QML) is a promising paradigm for tackling computational problems that challenge classical AI. Yet, the inherent probabilistic behavior of quantum mechanics, device noise in NISQ hardware, and hybrid quantum-classical execution pipelines introduce new risks that prevent reliable deployment of QML in real-world, safety-critical settings. This research offers a broad roadmap for Trustworthy Quantum Machine Learning (TQML), integrating three foundational pillars of reliability: (i) uncertainty quantification for calibrated and risk-aware decision making, (ii) adversarial robustness against classical and quantum-native threat models, and (iii) privacy preservation in distributed and delegated quantum learning scenarios. We formalize quantum-specific trust metrics grounded in quantum information theory, including a variance-based decomposition of predictive uncertainty, trace-distance-bounded robustness, and differential privacy for hybrid learning channels. To demonstrate feasibility on current NISQ devices, we validate a unified trust assessment pipeline on parameterized quantum classifiers, uncovering correlations between uncertainty and prediction risk, an asymmetry in attack vulnerability between classical and quantum state perturbations, and privacy-utility trade-offs driven by shot noise and quantum channel noise. This roadmap seeks to define trustworthiness as a first-class design objective for quantum AI. |
| 2025-11-04 | [DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](http://arxiv.org/abs/2511.02495v1) | Zixuan Liu, Siavash H. Khajavi et al. | Recent advances in multi-modal models have demonstrated strong performance in tasks such as image generation and reasoning. However, applying these models to the fire domain remains challenging due to the lack of publicly available datasets with high-quality fire domain annotations. To address this gap, we introduce DetectiumFire, a large-scale, multi-modal dataset comprising of 22.5k high-resolution fire-related images and 2.5k real-world fire-related videos covering a wide range of fire types, environments, and risk levels. The data are annotated with both traditional computer vision labels (e.g., bounding boxes) and detailed textual prompts describing the scene, enabling applications such as synthetic data generation and fire risk reasoning. DetectiumFire offers clear advantages over existing benchmarks in scale, diversity, and data quality, significantly reducing redundancy and enhancing coverage of real-world scenarios. We validate the utility of DetectiumFire across multiple tasks, including object detection, diffusion-based image generation, and vision-language reasoning. Our results highlight the potential of this dataset to advance fire-related research and support the development of intelligent safety systems. We release DetectiumFire to promote broader exploration of fire understanding in the AI community. The dataset is available at https://kaggle.com/datasets/38b79c344bdfc55d1eed3d22fbaa9c31fad45e27edbbe9e3c529d6e5c4f93890 |
| 2025-11-04 | [AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models](http://arxiv.org/abs/2511.02376v1) | Aashray Reddy, Andrew Zagula et al. | Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs, yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves up to 95% attack success rate on Llama-3.1-8B within six turns a 24 percent improvement over single turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests then iteratively refines them. Extensive evaluation across commercial and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses. |
| 2025-11-04 | [Optimizing Multi-UAV 3D Deployment for Energy-Efficient Sensing over Uneven Terrains](http://arxiv.org/abs/2511.02368v1) | Rushi Moliya, Dhaval K. Patel et al. | In this work, we consider a multi-unmanned aerial vehicle (UAV) cooperative sensing system where UAVs are deployed to sense multiple targets in terrain-aware line of sight (LoS) conditions in uneven terrain equipped with directional antennas. To mitigate terrain-induced LoS blockages that degrade detection performance, we incorporate a binary LoS indicator and propose a bounding volume hierarchy (BHV)-based adaptive scheme for efficient LoS evaluation. We formulate a bi-objective problem that maximizes the probability of cooperative detection with minimal hover energy constraints governing spatial, orientational, and safety constraints. To address the problem, which is inherently non-convex, we propose a hierarchical heuristic framework that combines exploration through a genetic algorithm (GA) with per-UAV refinement via particle swarm optimization (PSO), where a penalty-based fitness evaluation guides solutions toward feasibility, bounded within constraints. The proposed methodology is an effective trade-off method of traversing through a complex search space and maintaining terrain-aware LoS connectivity and energy aware deployment. Monte Carlo simulations on real-world terrain data show that the proposed GA+PSO framework improves detection probability by 37.02% and 36.5% for 2 and 3 UAVs, respectively, while reducing average excess hover energy by 45.0% and 48.9% compared to the PSO-only baseline. Relative to the non-optimized scheme, it further achieves 59.5% and 54.2% higher detection probability with 59.8% and 65.9% lower excess hover energy, thereby showing its effectiveness with a small number of UAVs over uneven terrain. |
| 2025-11-04 | [LiveSecBench: A Dynamic and Culturally-Relevant AI Safety Benchmark for LLMs in Chinese Context](http://arxiv.org/abs/2511.02366v1) | Yudong Li, Zhongliang Yang et al. | In this work, we propose LiveSecBench, a dynamic and continuously updated safety benchmark specifically for Chinese-language LLM application scenarios. LiveSecBench evaluates models across six critical dimensions (Legality, Ethics, Factuality, Privacy, Adversarial Robustness, and Reasoning Safety) rooted in the Chinese legal and social frameworks. This benchmark maintains relevance through a dynamic update schedule that incorporates new threat vectors, such as the planned inclusion of Text-to-Image Generation Safety and Agentic Safety in the next update. For now, LiveSecBench (v251030) has evaluated 18 LLMs, providing a landscape of AI safety in the context of Chinese language. The leaderboard is publicly accessible at https://livesecbench.intokentech.cn/. |
| 2025-11-04 | [An Automated Framework for Strategy Discovery, Retrieval, and Evolution in LLM Jailbreak Attacks](http://arxiv.org/abs/2511.02356v1) | Xu Liu, Yan Chen et al. | The widespread deployment of Large Language Models (LLMs) as public-facing web services and APIs has made their security a core concern for the web ecosystem. Jailbreak attacks, as one of the significant threats to LLMs, have recently attracted extensive research. In this paper, we reveal a jailbreak strategy which can effectively evade current defense strategies. It can extract valuable information from failed or partially successful attack attempts and contains self-evolution from attack interactions, resulting in sufficient strategy diversity and adaptability. Inspired by continuous learning and modular design principles, we propose ASTRA, a jailbreak framework that autonomously discovers, retrieves, and evolves attack strategies to achieve more efficient and adaptive attacks. To enable this autonomous evolution, we design a closed-loop "attack-evaluate-distill-reuse" core mechanism that not only generates attack prompts but also automatically distills and generalizes reusable attack strategies from every interaction. To systematically accumulate and apply this attack knowledge, we introduce a three-tier strategy library that categorizes strategies into Effective, Promising, and Ineffective based on their performance scores. The strategy library not only provides precise guidance for attack generation but also possesses exceptional extensibility and transferability. We conduct extensive experiments under a black-box setting, and the results show that ASTRA achieves an average Attack Success Rate (ASR) of 82.7%, significantly outperforming baselines. |
| 2025-11-04 | [Whole-body motion planning and safety-critical control for aerial manipulation](http://arxiv.org/abs/2511.02342v1) | Lin Yang, Jinwoo Lee et al. | Aerial manipulation combines the maneuverability of multirotors with the dexterity of robotic arms to perform complex tasks in cluttered spaces. Yet planning safe, dynamically feasible trajectories remains difficult due to whole-body collision avoidance and the conservativeness of common geometric abstractions such as bounding boxes or ellipsoids. We present a whole-body motion planning and safety-critical control framework for aerial manipulators built on superquadrics (SQs). Using an SQ-plus-proxy representation, we model both the vehicle and obstacles with differentiable, geometry-accurate surfaces. Leveraging this representation, we introduce a maximum-clearance planner that fuses Voronoi diagrams with an equilibrium-manifold formulation to generate smooth, collision-aware trajectories. We further design a safety-critical controller that jointly enforces thrust limits and collision avoidance via high-order control barrier functions. In simulation, our approach outperforms sampling-based planners in cluttered environments, producing faster, safer, and smoother trajectories and exceeding ellipsoid-based baselines in geometric fidelity. Actual experiments on a physical aerial-manipulation platform confirm feasibility and robustness, demonstrating consistent performance across simulation and hardware settings. The video can be found at https://youtu.be/hQYKwrWf1Ak. |
| 2025-11-04 | [Learning A Universal Crime Predictor with Knowledge-guided Hypernetworks](http://arxiv.org/abs/2511.02336v1) | Fidan Karimova, Tong Chen et al. | Predicting crimes in urban environments is crucial for public safety, yet existing prediction methods often struggle to align the knowledge across diverse cities that vary dramatically in data availability of specific crime types. We propose HYpernetwork-enhanced Spatial Temporal Learning (HYSTL), a framework that can effectively train a unified, stronger crime predictor without assuming identical crime types in different cities' records. In HYSTL, instead of parameterising a dedicated predictor per crime type, a hypernetwork is designed to dynamically generate parameters for the prediction function conditioned on the crime type of interest. To bridge the semantic gap between different crime types, a structured crime knowledge graph is built, where the learned representations of crimes are used as the input to the hypernetwork to facilitate parameter generation. As such, when making predictions for each crime type, the predictor is additionally guided by its intricate association with other relevant crime types. Extensive experiments are performed on two cities with non-overlapping crime types, and the results demonstrate HYSTL outperforms state-of-the-art baselines. |
| 2025-11-04 | [Quantitative Risk Assessment in Radiation Oncology via LLM-Powered Root Cause Analysis of Incident Reports](http://arxiv.org/abs/2511.02223v1) | Yuntao Wang, Siamak P. Najad-Davarani et al. | Background: Modern large language models (LLMs) offer powerful reasoning that converts narratives into structured, taxonomy-aligned data, revealing patterns across planning, delivery, and verification. Embedded as agentic tools, LLMs can assist root-cause analysis and risk assessment (e.g., failure mode and effect analysis FMEA), produce auditable rationales, and draft targeted mitigation actions.   Methods: We developed a data-driven pipeline utilizing an LLM to perform automated root cause analysis on 254 institutional safety incidents. The LLM systematically classified each incident into structured taxonomies for radiotherapy pathway steps and contributory factors. Subsequent quantitative analyses included descriptive statistics, Analysis of Variance (ANOVA), multiple Ordinal Logistic Regression (OLR) analyses to identify predictors of event severity, and Association Rule Mining (ARM) to uncover systemic vulnerabilities.   Results: The high-level Ordinal Logistic Regression (OLR) models identified specific, significant drivers of severity. The Pathway model was statistically significant (Pseudo R2 = 0.033, LR p = 0.015), as was the Responsibility model (Pseudo R2 = 0.028, LR p < 0.001). Association Rule Mining (ARM) identified high-confidence systemic rules, such as "CF5 Teamwork, management and organisational" (n = 8, Conf = 1.0) and the high-frequency link between "(11) Pre-treatment planning process" and "CF2 Procedural" (n = 152, Conf = 0.916).   Conclusion: The LLM-powered, data-driven framework provides a more objective and powerful methodology for risk assessment than traditional approaches. Our findings empirically demonstrate that interventions focused on fortifying high-risk process steps and mitigating systemic failures are most effective for improving patient safety. |
| 2025-11-04 | [Optimizing Multi-Lane Intersection Performance in Mixed Autonomy Environments](http://arxiv.org/abs/2511.02217v1) | Manonmani Sekar, Nasim Nezamoddini | One of the main challenges in managing traffic at multilane intersections is ensuring smooth coordination between human-driven vehicles (HDVs) and connected autonomous vehicles (CAVs). This paper presents a novel traffic signal control framework that combines Graph Attention Networks (GAT) with Soft Actor-Critic (SAC) reinforcement learning to address this challenge. GATs are used to model the dynamic graph- structured nature of traffic flow to capture spatial and temporal dependencies between lanes and signal phases. The proposed SAC is a robust off-policy reinforcement learning algorithm that enables adaptive signal control through entropy-optimized decision making. This design allows the system to coordinate the signal timing and vehicle movement simultaneously with objectives focused on minimizing travel time, enhancing performance, ensuring safety, and improving fairness between HDVs and CAVs. The model is evaluated using a SUMO-based simulation of a four-way intersection and incorporating different traffic densities and CAV penetration rates. The experimental results demonstrate the effectiveness of the GAT-SAC approach by achieving a 24.1% reduction in average delay and up to 29.2% fewer traffic violations compared to traditional methods. Additionally, the fairness ratio between HDVs and CAVs improved to 1.59, indicating more equitable treatment across vehicle types. These findings suggest that the GAT-SAC framework holds significant promise for real-world deployment in mixed-autonomy traffic systems. |
| 2025-11-04 | [LLMs as Judges: Toward The Automatic Review of GSN-compliant Assurance Cases](http://arxiv.org/abs/2511.02203v1) | Gerhard Yu, Mithila Sivakumar et al. | Assurance cases allow verifying the correct implementation of certain non-functional requirements of mission-critical systems, including their safety, security, and reliability. They can be used in the specification of autonomous driving, avionics, air traffic control, and similar systems. They aim to reduce risks of harm of all kinds including human mortality, environmental damage, and financial loss. However, assurance cases often tend to be organized as extensive documents spanning hundreds of pages, making their creation, review, and maintenance error-prone, time-consuming, and tedious. Therefore, there is a growing need to leverage (semi-)automated techniques, such as those powered by generative AI and large language models (LLMs), to enhance efficiency, consistency, and accuracy across the entire assurance-case lifecycle. In this paper, we focus on assurance case review, a critical task that ensures the quality of assurance cases and therefore fosters their acceptance by regulatory authorities. We propose a novel approach that leverages the \textit{LLM-as-a-judge} paradigm to automate the review process. Specifically, we propose new predicate-based rules that formalize well-established assurance case review criteria, allowing us to craft LLM prompts tailored to the review task. Our experiments on several state-of-the-art LLMs (GPT-4o, GPT-4.1, DeepSeek-R1, and Gemini 2.0 Flash) show that, while most LLMs yield relatively good review capabilities, DeepSeek-R1 and GPT-4.1 demonstrate superior performance, with DeepSeek-R1 ultimately outperforming GPT-4.1. However, our experimental results also suggest that human reviewers are still needed to refine the reviews LLMs yield. |
| 2025-11-04 | [Permissioned Blockchain in Advanced Air Mobility: A Performance Analisys for UTM](http://arxiv.org/abs/2511.02171v1) | Rodrigo Nunes, André Melo et al. | The rapid adoption of Uncrewed Aerial Vehicles (UAVs) has driven aviation authorities to propose distributed Uncrewed Traffic Management (UTM) architectures. Several studies have advocated blockchain as a promising technology to meet these requirements. However, since UTM is a safety-critical and highly regulated domain, compliance with standards and regulatory frameworks is as crucial as performance and security. This work benchmarks two distributed architectures aligned with current regulatory frameworks: the Linux Foundation's InterUSS platform and a Hyperledger Fabric-based private ledger. Our findings reveal that blockchain-based systems require architectures specifically designed for aeronautical performance constraints. |
| 2025-11-03 | [Model Predictive Control with Multiple Constraint Horizons](http://arxiv.org/abs/2511.02114v1) | Allan Andre do Nascimento, Han Wang et al. | In this work we propose a Model Predictive Control (MPC) formulation that splits constraints in two different types. Motivated by safety considerations, the first type of constraint enforces a control-invariant set, while the second type could represent a less restrictive constraint on the system state. This distinction enables closed-loop sub- optimality results for nonlinear MPC with heterogeneous state constraints (distinct constraints across open loop predicted states), and no terminal elements. Removing the non-invariant constraint recovers the partially constrained case. Beyond its theoretical interest, heterogeneous constrained MPC shows how constraint choices shape the system's closed loop. In the partially constrained case, adjusting the constraint horizon (how many predicted- state constraints are enforced) trades estimation accuracy for computational cost. Our analysis yields first, a sub- optimality upper-bound accounting for distinct constraint sets, their horizons and decay rates, that is tighter for short horizons than prior work. Second, to our knowledge, we give the first lower bound (beyond open-loop cost) on closed-loop sub-optimality. Together these bounds provide a powerful analysis framework, allowing designers to evaluate the effect of horizons in MPC sub-optimality. We demonstrate our results via simulations on nonlinear and linear safety-critical systems. |
| 2025-11-03 | [Efficient Quantification of Time-Series Prediction Error: Optimal Selection Conformal Prediction](http://arxiv.org/abs/2511.02103v1) | Boyu Pang, Kostas Margellos | Uncertainty is almost ubiquitous in safety-critical autonomous systems due to dynamic environments and the integration of learning-based components. Quantifying this uncertainty--particularly for time-series predictions in multi-stage optimization--is essential for safe control and verification tasks. Conformal Prediction (CP) is a distribution-free uncertainty quantification tool with rigorous finite-sample guarantees, but its performance relies on the design of the nonconformity measure, which remains challenging for time-series data. Existing methods either overfit on small datasets, or are computationally intensive on long-time-horizon problems and/or large datasets. To overcome these issues, we propose a new parameterization of the score functions and formulate an optimization program to compute the associated parameters. The optimal parameters directly lead to norm-ball regions that constitute minimal-average-radius conformal sets. We then provide a reformulation of the underlying optimization program to enable faster computation. We provide theoretical proofs on both the validity and efficiency of predictors constructed based on the proposed approach. Numerical results on various case studies demonstrate that our method outperforms state-of-the-art methods in terms of efficiency, with much lower computational requirements. |
| 2025-11-03 | [A Comparison of Road Grade Preview Signals from Lidar and Maps](http://arxiv.org/abs/2511.02006v1) | Logan Schexnaydre, Aman Poovalappil et al. | Road grade can impact the energy efficiency, safety, and comfort associated with automated vehicle control systems. Currently, control systems that attempt to compensate for road grade are designed with one of two assumptions. Either the grade is only known once the vehicle is driving over the road segment through proprioception, or complete knowledge of the oncoming road grade is known from a pre-made map. Both assumptions limit the performance of a control system, as not having a preview signal prevents proactive grade compensation, whereas relying only on map data potentially subjects the control system to missing or outdated information. These limits can be avoided by measuring the oncoming grade in real-time using on-board lidar sensors. In this work, we use point returns accumulated during travel to estimate the grade at each waypoint along a path. The estimated grade is defined as the difference in height between the front and rear wheelbase at a given waypoint. Kalman filtering techniques are used to mitigate the effects of odometry and motion uncertainty on the grade estimates. This estimator's performance is compared to the measurements of a map created with a GNSS/INS system via a field experiment. When compared to the map-based system, the lidar-based estimator produces an unbiased error with a standard deviation of 0.6 degrees at an average range of 52.7 meters. By having similar precision to map-based systems, automotive lidar-based grade estimation systems are shown to be a valid approach for measuring road grade when a map is unavailable or inaccurate. In using lidar as an input signal for grade-based control system tasks, autonomous vehicles achieve higher redundancy and independence in contrast to existing methods. |
| 2025-11-03 | [LARK -- Linearizability Algorithms for Replicated Keys in Aerospike](http://arxiv.org/abs/2511.01843v1) | Andrew Goodng, Kevin Porter et al. | We present LARK (Linearizability Algorithms for Replicated Keys), a synchronous replication protocol that achieves linearizability while minimizing latency and infrastructure cost, at significantly higher availability than traditional quorum-log consensus. LARK introduces Partition Availability Conditions (PAC) that reason over the entire database cluster rather than fixed replica sets, improving partition availability under independent failures by roughly 3x when tolerating one failure and 10x when tolerating two. Unlike Raft, Paxos, and Viewstamped Replication, LARK eliminates ordered logs, enabling immediate partition readiness after leader changes -- with at most a per-key duplicate-resolution round trip when the new leader lacks the latest copy. Under equal storage budgets -- where both systems maintain only f+1 data copies to tolerate f failures -- LARK continues committing through data-node failures while log-based protocols must pause commits for replica rebuilding. These properties also enable zero-downtime rolling restarts even when maintaining only two copies. We provide formal safety arguments and a TLA+ specification, and we demonstrate through analysis and experiments that LARK achieves significant availability gains. |

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



