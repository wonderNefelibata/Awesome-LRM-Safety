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
| 2026-09-11 | [ASTRIL-MPC: Autonomous Traversal Framework of Articulated Tracked Robots with Language-Guided Neural-Kinematic MPC](http://arxiv.org/abs/2609.13083v1) | Zhenfeng Gan, Yanbo Chen et al. | In urban search and rescue, articulated tracked robots (ATRs) must traverse structured but contact-rich environments such as stairwells and cluttered building interiors. Reliable autonomy remains challenging because robot-terrain interaction (RTI) is hybrid and discontinuous, and effective flipper-track coordination is difficult to model analytically. We present ASTRIL-MPC, a language-guided neural kinematics model predictive control (MPC) framework for autonomous traversal. A learned kinematics model predicts short-horizon task-state increments from a height sequence and recent trajectories; NMPC plans with multi-objective costs and strict feasibility constraints; and a large language model (LLM) proposes bounded updates to selected weights and bounds through a safety-checked interface with range clipping, rate limiting, and consistency checks. The compiled predictor enables a full control cycle within 100 ms. Across three traversal tasks and a multi-height generalization setting, ASTRIL-MPC improves an aggregate traversal-quality score by up to 71% over a non-adaptive NMPC and by 67% over a PPO baseline, while eliminating measurable collision impacts during descent. These results indicate that combining learned kinematics, optimization-based planning, and language-guided retuning yields data-efficient and robust autonomy for articulated tracked robots. |
| 2026-09-11 | [TileNet: Tile-Based CNN-SVM Architecture for Autonomous Unmanned Aerial Systems Inspection of Flat Roofs](http://arxiv.org/abs/2609.13013v1) | Samuel Dunthorne, Hashim A. Hashim | Flat roofs are among the most influential components of the building envelope, governing both structural performance and thermal efficiency, and thereby contributing directly to household energy consumption, carbon emissions, and long-term environmental sustainability. Timely detection of roof defects is essential for reducing heating and cooling losses, preventing moisture-driven degradation such as mold growth, and supporting national climate-change mitigation goals. This paper presents a real-time, Unmanned Aerial System (UAS)-based deep learning framework that autonomously detects defects using live imagery captured during dual-altitude aerial passes. The multi-resolution flight strategy is designed to aid the identification of both small, fine-scale defects and larger structural issues, enabling more comprehensive assessments. To meet the strict computational and power constraints of embedded UAS hardware, the proposed framework integrates a tile-based architecture with a lightweight Convolution Neural Network-Support Vector Machine (CNN-SVM) classifier designed for low-latency onboard inference. The final model-comprising five convolutional layers and four dense layers, the last a linear SVM head, achieved a mean test accuracy of $94.4\%$ ($95\%$ confidence interval $\pm0.4\%$ over three seeds) on a photo-level split ($43,383$ training, $3,869$ validation, and $2,540$ test tiled and augmented images), outperforming GoogLeNet ($89.2\%$) and AlexNet ($79.8\%$). Experimental evaluations using real UAS imagery collected by onsite visits with DJI Matrice 350 RTK drone demonstrate that the system supports rapid, repeatable, and safe roof inspections while reducing human risk, lowering operational costs, and enabling more sustainable building maintenance. |
| 2026-09-11 | [Comfort by Construction: Adaptive, Comfort-Bounded Action Spaces for Learned Driving Policies](http://arxiv.org/abs/2609.13011v1) | Anna Rothenhäusler, Daniel Jost et al. | Data-driven driving simulators command accelerations and steering rates from a fixed grid without constraining the realized accelerations and jerks. As a result, reinforcement-learning policies inflate safety metrics through abrupt, last-second maneuvers that lie far outside the range of human driving and would be unacceptable to occupants of a real vehicle, so the metrics measure simulator permissiveness rather than policy quality. Enforcing comfort bounds naively is not enough: lateral limits shrink quadratically with speed, so clamping a static grid saturates it and destroys fine-grained control ("grid collapse"). We propose an adaptive action parameterization that rediscretizes the grid at every step to span exactly the per-step feasible control set, via closed-form inversion of the lateral-jerk constraint. We further present PufferDrive-Editor, a browser-based tool to audit realized kinematics and author kinematically challenging scenes. On the Waymo Open Motion Dataset and a hand-authored slalom, our adaptive model holds comfort violations below 1% while outperforming clipped-grid and direct-jerk baselines in navigability. |
| 2026-09-11 | [IntentFuzz: A Protocol-Aware Fuzzer for Automated Invariant Violation Detection in Intent-Based Cross-Chain Bridges](http://arxiv.org/abs/2609.13004v1) | André Augusto, Christof Ferreira Torres et al. | Cross-chain bridges move value between blockchains. Intent-based bridges are a variant where a solver fulfills a user's declared outcome and an off-chain settlement layer later reconciles the fill against the deposit. Existing smart-contract fuzzers and static analyzers only flag known-bad code patterns or require protocol-specific hand-written assertions. This work formalizes a taxonomy separating invariant violations, safety properties a contract must enforce locally, from settlement exposures legitimately delegated to the off-chain settlement layer, and proposes IntentFuzz: a protocol-aware fuzzer that recovers a bridge's intent structure and deposit/fill function roles directly from unannotated Solidity source, then synthesizes multi-step fuzz sequences using an LLM-based fallback to help build call arguments. IntentFuzz recovers the correct intent structure in 9/9 benchmark protocols and classifies deposit and fill functions with 100% recall and 82% combined precision; across a corpus of 77 manually labeled contracts, it reaches 79.5% bridge-classification precision and 97.2% recall, and among confirmed bridges, struct selection reaches 88.6% precision and recall while deposit and fill classification each reach 100% recall. On 23 planted-bug mutants, IntentFuzz attains 100% recall and 100% precision, executing 273 templates (507 transactions in a median of 14ms per template). Across 24 real-world deployments, it confirms 17 genuine invariant violations under heuristic-only input generation, rising to 22 with its LLM-assisted tier enabled, spanning eight vulnerable GitHub repositories, each finding reproducible against public, deployed bytecode. |
| 2026-09-11 | [Safe Stabilising Full-Order Affine Control Barrier Functions for Linear Systems (Extended)](http://arxiv.org/abs/2609.12990v1) | Faisal Lawan, Joaquin Carrasco et al. | Control barrier function safety filters enforce constraints by modifying a nominal input, but the resulting switching can destabilise the closed loop even when the nominal and filtered modes are individually stable. This paper presents a design framework for safe and globally exponentially stabilising controllers for linear systems with a single full-relative-degree affine constraint. We show that the filtered-mode spectrum is fixed by the barrier tuning and is independent of the plant, nominal controller, and quadratic-program weighting. This structure yields an explicit nominal controller for which the safety filter remains inactive everywhere. For a prescribed nominal controller, we prove that the nominal and filtered modes admit a strong common quadratic Lyapunov function if and only if the ratio of the nominal characteristic polynomial to the barrier polynomial is strongly strictly positive real. This equivalence characterises the existence of a common quadratic Lyapunov and provides a scalar frequency-domain test, along with an explicit interval of admissible gains. Building on these results, the extended analysis derives an explicit common storage function and reduces an existing LMI synthesis condition to a feasibility test in a single matrix variable. A flexible two-mass example explains a known instability mechanism and demonstrates how the proposed design restores safety and global exponential stability. |
| 2026-09-11 | [End-to-End Battery Dispatch with Exact Rainflow Degradation via Mixed-Integer Differentiable Predictive Control](http://arxiv.org/abs/2609.12968v1) | Eshagh Safarzadeh Ravajiri, Jan Drgona et al. | Optimal dispatch of battery energy storage systems requires balancing energy arbitrage against cycle-induced degradation, which is accurately quantified through rainflow cycle counting. However, rainflow's combinatorial, nondifferentiable algorithm is incompatible with both convex optimization and gradient-based neural network training. We present a self-supervised mixed-integer differentiable predictive control framework that trains neural policies directly on exact rainflow degradation through a novel differentiable rainflow layer combining exact gradients at state-of-charge extrema with dense proxy gradients on incremental changes, enabling stable end-to-end training while preserving true degradation physics. A mixed-integer differentiable architecture enforces power balance, dynamics, and mode exclusivity, with a safety filter guaranteeing constraint satisfaction. We evaluate the framework on 3,650 real battery-day scenarios (a 10-battery fleet over 365 days) spanning three utility regions (SDG&E California, Xcel Energy Colorado, APS Arizona) with diverse time-of-use pricing structures. A single-battery trained policy achieves a 0.35% performance gap on its training distribution with over 200x speedup, while fleet-wide training generalizes across all households and utility regions, achieving a 3.33% performance gap with 564x computational speedup (62 seconds vs. 9.7 hours) and 100% feasibility. The millisecond-scale inference reduces computation by over two orders of magnitude compared to mixed-integer solvers, enabling practical deployment at scale. |
| 2026-09-11 | [Distributed Stochastic Optimal Control for Pattern-Oriented Swarms](http://arxiv.org/abs/2609.12959v1) | Qingrui Zhang, Chenghao Yu et al. | While offering significant promise for diverse applications, pattern-oriented swarms encounter multifaceted challenges in geometric control, self-organization, and safe navigation through dynamic environments. In this paper, we present a GRF-based stochastic optimal control framework to address these challenges within a unified probabilistic architecture. By extending the GRF into the temporal domain, the proposed framework casts collective coordination as a Bayesian inference task, enabling swarms to accommodate environmental uncertainty, satisfy non-convex constraints, and reconcile heterogeneous dynamics across diverse platforms. We develop an uncertainty- and safety-aware collision avoidance module for navigation in the presence of stochastic obstacle motion. The unscented transform is employed to propagate state uncertainty for both dynamic obstacles and neighboring agents, yielding principled confidence bounds for collision avoidance. In addition, density-guided pattern control is introduced, which encodes geometric patterns as implicit density fields. This representation decouples pattern specification from explicit agent-to-target assignments, thereby facilitating intrinsic self-healing and elastic reconfiguration in a distributed manner. The proposed framework is extensively evaluated through Monte Carlo simulations across diverse scenarios. Its model-agnostic nature is demonstrated on both quadrotor and fixed-wing UAV swarms, highlighting its generalizability across platforms with heterogeneous dynamics. Finally, the efficacy and robustness of the proposed method are validated through indoor experiments with a 15-quadrotor swarm and outdoor deployments involving 4 custom-built autonomous quadrotors. These experiments substantiate the proposed framework's capacity to maintain reliable geometric pattern transitions and safety-aware navigation within real-world environments. |
| 2026-09-11 | [ARC: Autonomous Robotics Compliance A Three-Layer Governance Architecture for Deployed Autonomous Systems](http://arxiv.org/abs/2609.12932v1) | Tord Eide, Einar Holt | Proposed governance framework for autonomous robotic systems, introducing a three-layer compliance architecture (ARC) instantiated through model safety validation, cognitive certification benchmarks, and operational authorization standards. |
| 2026-09-11 | [NeuroClick: Preserving Surgeon Autonomy through Hands-Free Earable Tooth-Click Control in Neurosurgery](http://arxiv.org/abs/2609.12910v1) | Jonas Hummel, Maximilian Burzer et al. | Neurosurgeons frequently interact with operating room (OR) technologies while sterility and occupied hands constrain control. We introduce earables as a direct, hands-free control platform for neurosurgery using tooth-click input. Formative OR observations and interviews with 10 domain experts grounded the design. Using OpenEarable 2.0 data from 12 participants, we developed a real-time recognition pipeline whose classifier achieved a median macro F1-score of 98.6% under leave-one-subject-out cross-validation. We evaluated the technique with 20 neurosurgeons during a simulated resection task in a neurosurgical OR. Participants reported few focus shifts and rated Earable favorably for workflow integration and perceived safety. Autonomous microscope control was rated significantly higher with Earable than Delegation, whereas Delegation enabled faster task completion under continuous assistant availability. Workload, usability, and task errors showed no significant differences. Preferences depended on training, reliability, context, and assistant availability. Earables thus add a direct, hands-free option for controlling selected functions alongside established workflows. |
| 2026-09-11 | [Before the Tipping Point: Force-Guided Active Perception for Shape-Agnostic Estimation of 3D Centers of Mass](http://arxiv.org/abs/2609.12894v1) | Steven M. Hyland, Jing Xiao et al. | Estimating the 3D center of mass of unknown objects is challenging when grasping is infeasible, geometry is irregular, or mass distribution is uneven. We present a force-based method that estimates CoM height and mass from a single sub-critical tipping experiment by a robot manipulator. The robot applies a quasistatic elevated push and retract motion, using force-angle measurements recorded during tipping to identify parameters from the object trajectory. Our proposed push-retract cycle mitigates frictional bias, enabling generalized fitting. We experimentally validate our method using a robot manipulator with a six-axis force torque sensor on varying types of objects without prior shape information and without specific models. We also propose a method to prevent toppling, keeping the object in a sub-critical tipping regime by leveraging a safety margin. In experimental studies, our method recovers mass, CoM height, and toppling angle with relative errors below 5.0 percent across all unknown objects. This work demonstrates reliable 3D inertial parameter estimation under proper safety thresholds in tipping. Our proposed method informs and enables reliable non-prehensile manipulation and robotic grasping of challenging objects that were previously infeasible. |
| 2026-09-11 | [Probing Inflationary Origins of Primordial Black Holes with LIGO--Virgo--KAGRA O1--O4a data](http://arxiv.org/abs/2609.12867v1) | Haipeng An, Huai-Ke Guo et al. | Large primordial curvature perturbations not only produce primordial black holes (PBHs) but also inevitably source a scalar-induced stochastic gravitational-wave background upon horizon reentry. We analyze the combined LIGO--Virgo--KAGRA O1--O4a data to constrain two representative inflationary mechanisms for generating such perturbations: ultra-slow-roll inflation and an inflationary phase transition. Detecting no evidence for either scenario, we place 95% credible upper limits on the curvature-spectrum amplitude across the frequency range accessible to ground-based interferometers. Translated into the PBH context, these limits already exceed conventional constraints, probing abundance fractions far below unity. Our results remain robust even when the PBHs themselves are too rare to be directly detected or have evaporated. This work demonstrates that stochastic gravitational-wave observations offer a powerful and complementary probe of small-scale inflationary physics and PBH formation, with upcoming interferometers promising to extend sensitivity to a wider range of inflationary epochs and PBH masses. |
| 2026-09-11 | [VertexCBF: Improving Neural Control Barrier Functions via Vertex-Restricted Control Search](http://arxiv.org/abs/2609.12831v1) | Bojan Derajić, Sebastian Bernhard et al. | As the number of autonomous robots continues to grow, safety becomes increasingly important. Control barrier functions (CBFs) provide a theoretically grounded framework for ensuring safety, but existing design methods often face limitations in effectiveness, scalability, or interpretability, and may result in overly conservative safe sets. In this paper, we propose \emph{VertexCBF}, a framework for learning neural CBFs in a scalable, systematic, and explainable way. We approximate the stationary Hamilton--Jacobi value function using a neural network trained via a combination of physics-informed and sparsely supervised learning. By exploiting control-affine dynamics and a convex polytope control set, under which the Hamiltonian is maximized at the control vertices, we efficiently generate supervision points via GPU-parallel vertex-restricted tree search, while a residual architecture guarantees that the learned CBF is never larger than the specified constraint function. We evaluate the method on 15 systems and compare it against relevant baselines, showing that it reliably recovers large safe sets where the baselines are conservative or fail completely. In addition, we perform a hardware experiment in which a mobile robot safely avoids pedestrians using a neural CBF trained with our method. |
| 2026-09-11 | [Batten the Hatches: Cybersecurity with Military Mariners](http://arxiv.org/abs/2609.12810v1) | Ryan Von Brock, Anna Raymaker et al. | Cyberwarfare has become a key component of contemporary geopolitical conflict. However, there has been extremely limited systematic investigation into how cybersecurity is handled by military organizations and personnel. The military context is unique compared to other operational ones, with immense resource availability (U.S. military spending approached 1 trillion dollars in 2024), a rigid chain of command, and extraordinary consequences for its actions. Thus, military cybersecurity is a distinct yet understudied topic.   In this paper, we take an early step at understanding military cybersecurity by investigating how service members understand, recognize, and respond to cyber risk. We focus on maritime services and carefully consider organizational barriers to design an unclassified study and conduct semi-structured interviews with 20 military mariners from U.S. Navy and Coast Guard vessels. Through our investigation, we identify unique consequences of compromising military systems, including weapon takeover and purposeful geopolitical escalation. We find that cybersecurity is organizationally abstract on ships, so mariners build cyber risk models from informal experience rather than formal instruction. They nonetheless make cybersecurity actionable by recognizing operational impacts and responding with a safety-oriented incident-response model that creates resilience but may delay cyber attribution and containment. These findings inform actionable recommendations to help military operators frame cyber threats, merge longstanding nautical doctrine with modern systems, and apply military insights to the civilian sector, all to secure the broader maritime environment. |
| 2026-09-11 | [What is the Difference Between Me and You? Benchmarking the Quality Gap Between Human-Written and AI-Generated Code](http://arxiv.org/abs/2609.12708v1) | Cristina Improta, Pietro Liguori et al. | AI coding assistants are becoming co-authors of production software, yet their evaluation centers on functional correctness, leaving open whether their code differs from human code in the quality dimensions dominating lifecycle cost. We compare human-written and AI-generated code at scale: 787,562 function pairs across Python, Java, and C, each human function mined from open-source repositories paired with implementations generated from its docstring by three AI assistants (OpenAI GPT models, DeepSeek-Coder, Qwen2.5-Coder). We characterize structural complexity and statistical naturalness, and map static-analysis findings onto Orthogonal Defect Classification for defects and the Common Weakness Enumeration for vulnerabilities, making authors and languages directly comparable. AI-generated code is structurally compressed and stylistically templated: roughly half the size and branching of human code, clustering apart at the style level. Defect profiles differ in kind: human code concentrates issues of mature codebases, AI code repetitive boilerplate; security is language-dependent, with LLMs producing more, and more severe, findings in Python and Java but fewer high-severity memory-safety findings than humans in C. Once size is controlled for, complexity metrics carry little signal, while naturalness separates authors. Finally, we release CQBench, a benchmark of 27,346 issue-prone tasks with baselines and an evaluation pipeline for quality assurance and security testing. |
| 2026-09-11 | [Bridging the First-Hour Gap: Evaluating AI Reliability and Benchmarking Deficiencies in Cyber Incident Response for Law Enforcement](http://arxiv.org/abs/2609.12681v1) | Roshin Sleeba C, Hiran V Nath | The actions of frontline law enforcement officers in the initial hour of a cyber incident play a vital role in determining the ultimate success of an investigation. The minor mistakes they commit might result in irreversible critical impacts. The integrity of the investigation can be compromised, and the prosecution of cyber criminals can be hindered due to minor mistakes that happen in the initial hour. These are mainly because of the volatile nature of digital artifacts that might lead to procedural errors and evidence attrition. This paper provides a systematic survey of decision-support architectures designed to assist first responders of a cybercrime, categorizing them into playbooks, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG) frameworks, and Agentic AI systems. The survey critically considers the constraints of limited technical proficiency and inconsistent forensic infrastructure in a practical scenario. Our analysis identifies RAG-based systems as a relatively viable intermediate solution due to their natural language adaptability. However, significant risk factors like prompt sensitivity and the potential for confident hallucinations in legal contexts pose a major challenge. Furthermore, we review current benchmarks in cybersecurity and demonstrate that they are not sufficient to capture the specific safety and legal requirements of law enforcement, focusing on the initial hour of the cybercrime. We conclude by arguing for the necessity of a new evaluation benchmark focused on naive query robustness and evidence preservation, so as to ensure that AI-driven guidance aligns with the mandatory demands of judicial proceedings. |
| 2026-09-11 | [Personalized and Trust-Aware Health Recommendation Policies for a Construction Workplace](http://arxiv.org/abs/2609.12679v1) | Atefeh Mollabagher, Yogesh Gautam et al. | Construction workers face workplace risks such as fatigue, heat stress, and other physically demanding conditions that can negatively affect their health and safety. Although monitoring these risks is important, timely and personalized health interventions are also needed to help prevent negative impacts on workers' well-being and productivity. To this end, in this paper, we propose a model to capture the interactions between a trust-aware health recommender system and workers who differ in health and trust sensitivity. Specifically, in our proposed dynamic model, worker health evolves over time, worker trust is affected by both health and recommendation dynamics, and trust in turn affects compliance with future recommendations. Given this model, we characterize the recommender policy, including a health-based recommendation triggering threshold and the recommendation frequency. We do so using both model-based short-horizon control and model-free reinforcement learning. We then investigate how recommendation frequencies are adjusted for different workers to balance their health, productivity, and trust. Our findings provide insight into the design of personalized health recommendation policies in construction workplaces and beyond. |
| 2026-09-11 | [Access Control as Verified Parse Constraints](http://arxiv.org/abs/2609.12488v1) | Saranachon Iammongkol, Zhiyi Huang et al. | Commercial security gateways repeatedly ship implementation bugs in the code path between the network and the policy decision: hand-written enforcement logic that diverges from the policy author's intent, and ad-hoc request parsers at the network boundary that introduce memory-safety flaws of their own. In both cases the bug is in the deployed enforcement code, not in the policy. Existing approaches either leave the enforcement runtime unverified or connect a formal model to a hand-written engine only by differential testing.   Our contribution is a class result: a forward-only, backtrack-free EverParse validator is a verified recognizer for a bounded, finite-state class, and access-control decision functions with fixed-offset fields and bounded disjunction belong to it, so one machine-checked proof transfers to every policy in the class rather than being re-established per policy. Concretely, we encode a bounded policy language's decision function into a fixed-size byte buffer and verify the enforcement code once---covering all byte values---with an SMT solver, proving the validator accepts if and only if the decision function accepts, for every policy, request, and session. Editing rule content over a fixed endpoint set then needs no new proof; adding endpoints reruns the toolchain; extending the language needs new proofs. We establish faithful enforcement of a policy, not that a policy is itself secure.   The verified gate is platform-independent, requiring only EverParse/Z3 and a C compiler, whose correctness we assume. We demonstrate a deployment on the seL4 microkernel, which ensures every request passes through the gate and that unverified components cannot corrupt the verified enforcement chain. |
| 2026-09-11 | [First Experimental Evidence of Helicon Current Drive](http://arxiv.org/abs/2609.12474v1) | J. B. Lestz, R. I. Pinsker et al. | Helicon current drive is an attractive solution for driving current to sustain steady state tokamak operation in reactor conditions. Dedicated DIII-D experiments have been conducted with a MW-level helicon system and successfully demonstrated core power deposition and current drive with helicon waves launched via a traveling wave antenna. The profile of the measured electron temperature response to helicon power injection is in good agreement with time-dependent integrated modeling that incorporates ray tracing and the effects of thermal transport simultaneously. When the helicon power is injected continuously to drive co-$I_p$ current, the reconstructed safety factor profile flattens significantly faster and sawteeth are triggered earlier than in comparison shots where the helicon is replaced by a comparable amount of electron cyclotron heating. Calculation of the helicon-driven current profile yields a peaked profile in the core, consistent with the observed power deposition profile and in good agreement with ray tracing predictions. Taken together, these experimental results represent strong evidence for the first definitive observation of auxiliary current drive due to helicon waves on any device. |
| 2026-09-11 | [Hieronym: Leveraging Hierarchical Multi-Source Information for Function Renaming in Stripped Binary](http://arxiv.org/abs/2609.12457v1) | Xiaoling Zhang, Jian Sun et al. | Function renaming in stripped binaries can substantially assist reverse engineers by improving code readability, yet it is a challenging task. The difficulty stems from the need to accurately capture function semantics from low-level binary code across diverse instruction sets, architectures, and compiler optimizations, and to express these semantics in concise, human-readable names. Existing approaches either inadequately capture comprehensive function semantics or exhibit limited generalization to previously unseen binaries. In this paper, we present Hieronym, a generative large language model (LLM)-based framework for stripped binary function renaming. Hieronym adopts a hierarchical summarization-driven domain adaptation strategy and integrates multi-source information, including global binary context, local calling context, and intrinsic function semantics, to enhance the LLM's understanding of binary code. To enable systematic evaluation, we further propose a dual-layer evaluation framework that incorporates both token-level and whole-name-level metrics. We evaluate Hieronym on binary functions compiled with four compiler optimization levels (O0-O3) for four architectures (x64, x86, ARM, and MIPS). Experimental results demonstrate that Hieronym significantly outperforms state-of-the-art methods, achieving token-level improvements of 50.12% in precision, 41.75% in recall, and 45.10% in F1-score, as well as a 79.94% improvement in name-level accuracy, while also exhibiting strong generalization capability. Moreover, experiments on real-world malware samples further validate the practical effectiveness of Hieronym in security-critical scenarios. |
| 2026-09-11 | [HeatCache: Thermal-aware Energy-efficient LLM Inference Scheduling for Chassis-level Liquid Cooling in Sustainable Edge Server Rooms](http://arxiv.org/abs/2609.12449v1) | Rui Lu, Huanghuang Liang et al. | LLM inference is increasingly deployed at institution-scale edges to meet service requirements. However, multi-GPU inference consumes a large amount of electricity and produces substantial heat. To improve sustainability, operators and regulations often demand raising the ambient setpoint to reduce cooling electricity. This can increase thermal throttling and hardware aging, leading to Service-Level Objective violations. In this paper, we present HeatCache, a thermal-aware, energy-efficient LLM inference scheduler for commercial chassis-level AIO liquid-cooled GPUs at sustainable ambient temperatures. HeatCache treats AIO loops as a temporary heat buffer, measured by heat budget and schedules requests to minimize energy subject to thermal safety and SLO constraints, based on an electrical-informed heat-demand estimation from HeatiTS. We implement HeatCache atop vLLM and show that it reduces computing energy by up to 18.0%, decreases thermal-throttle exposure by 81.7%, and maintains SLO violation rates below 0.9% even up to $48~^{\circ}\mathrm{C}$. |

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



