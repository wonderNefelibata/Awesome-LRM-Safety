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
| 2026-11-13 | [Pairing AIS with RF: Analysis of a Markovian Approach for Maritime Data Fusion](http://arxiv.org/abs/2611.18563v1) | Daniel Grimm, Kyra Jongman et al. | Monitoring maritime activities is important for safety and security, particularly for detecting illegal activities, but remains challenging due to the size and complexity of the maritime domain. The Automatic Identification System (AIS) is commonly used for this purpose, despite its known vulnerabilities. In contrast, space-based radio frequency (RF) observations provide an independent source of vessel activity but lack ship identification and therefore cannot identify vessels on their own. Following the CRISP-DM process, this study applies a Pair Hidden Markov Model (PHMM) to align individual RF detections with AIS trajectories, using temporal and spatial information while modeling noise and timing discrepancies. We evaluate it on NOAA AIS data with synthetic RF detections against nearest-neighbor baselines and analyze its errors with bias tests on trajectory length, sampling density, stationarity and AIS--RF distance. The length-corrected Forward score reaches a top-1 precision of $0.7776$ on the validation split of multi-candidate detections, where errors are possible, and an AUC of $0.637$ for detecting dark vessels among the detections that pass the prefilter. With a single RF detection, the score mostly reflects the AIS trajectory itself, and its errors favor longer and more stationary trajectories. A log-odds score, which compares the full model with a null model without match transitions, raises precision to $0.8580$ and AUC to $0.852$. Its length-corrected variant no longer shows a significant bias towards stationary trajectories and short inter-message distances, but still favors candidates farther from the RF detection. A time-weighted nearest-neighbor baseline without fitting remains ahead (precision $0.9013$, AUC $0.904$), so this single-detection, hand-designed PHMM formulation approaches but does not beat it. |
| 2026-11-13 | [Box-conditional Hypothesis Testing for Out-of-Distribution Object Detection](http://arxiv.org/abs/2611.18519v1) | Donghoon Lee, Jusung Park et al. | Detecting out-of-distribution (OOD) objects is essential for ensuring the reliability and safety of object detectors in real-world deployment. We treat the decision on a detected box as a hypothesis test with the null hypothesis that the box covers an object of an in-distribution (ID) class. The null distribution changes with the size of the box and its contact with the image border, and the detector observes both. The test can therefore rank a box only against ID boxes of the same size and border contact, which gives a distribution-free bound on the false-alarm rate for each of them and uncovers the separation that one null distribution hides. The power of the test also rests on the layer it reads. Layers built only for the ID classes have no reason to keep what sets an unknown object apart, so the test reads the features before them. We propose \textbf{PLUM}, a training-free score that pools the last backbone stage inside the box and ranks its distance to the ID classes among ID boxes of the same size and border contact. On the standard benchmark, PLUM reaches the highest AUROC among the compared methods without external models, with 91.5 and 94.5 on PASCAL-VOC and 95.1 and 97.5 on BDD100K. Code: https://github.com/DDOZMI/PLUM |
| 2026-11-13 | [Accounting for Trust and Safety: Bounding Content-Moderation Quality from Records You Already Hold](http://arxiv.org/abs/2611.18437v1) | Aayush Agarwal, Daron Veer Singh Sohi et al. | Platforms use expert audits to measure content-moderation quality. This paper develops a formal method for checking those claims using records platforms already maintain, such as appeal reversals and confirmed reports. These records provide one-sided bounds: they can show that a stated quality target was missed, but cannot prove that it was met. The paper presents the theory, limitations, constructed examples, and a prospective protocol for future evaluation. It reports no analysis of Meta operational data, no experiment, and no production result. |
| 2026-11-13 | [CS-MPC: Contraction-Certified Sampling-Based MPC for Nonlinear Systems Under Uncertainty](http://arxiv.org/abs/2611.18390v1) | Yashwanth Kumar Nakka | We present contraction-certified sampling-based model predictive control (CS-MPC) for tracking and safety of nonlinear robotic systems under parametric uncertainty, disturbances, and safety constraints that are revealed online. CS-MPC uses a contraction-certified tracking feedback inside every sampled rollout, which concentrates the rollouts around the reference, and an execution layer that verifies the applied input using a quantile-tightened discrete-time control barrier function condition. We derive a verification procedure for nonlinear contraction of the feedback-closed dynamics, and we prove a finite-sample stochastic stability bound and a mission-level probabilistic safety guarantee. We validate CS-MPC in simulation on a rover, a quadrotor, and an asteroid landing. When compared with standard model predictive path integral (MPPI) control, CS-MPC attains the same tracking error with 7 to 20 times fewer rollouts for the rover, and a 2 to 4 times lower tracking error for the quadrotor, with no safety violations in 200 Monte Carlo missions. |
| 2026-11-13 | [VLAN: Vision-Language Accessible Navigation](http://arxiv.org/abs/2611.18346v1) | Jie Hu, Yu Zheng et al. | Accessible mobility is user-conditioned rather than scene-intrinsic, as walkable paths for one embodiment are inaccessible to another. Yet, existing benchmarks evaluate traversability under a fixed and limited embodiment. We present Vision-Language Accessible Navigation (VLAN), containing 1,000 scenes, 20K navigation trajectories, and 160K keyframes across diverse lighting, environmental conditions, scene semantics, and path geometries. For each trajectory, VLAN provides embodiment-specific actions and accessibility features for legged and wheeled agents, including humanoid and quadruped robots and people with visual or mobility impairments, while everyday obstacles and accessibility labels are categorized as Passable, Bypass, HighRisk, and Blocked. Unlike goal-only evaluation, our benchmark measures physical safety and alignment with user-specific constraints via goal-reaching, collision-aware, and accessibility-aligned metrics. We benchmark modern vision-language models and find they often fail to consistently adapt obstacle relevance, path selection, and safety criteria across user profiles, suggesting accessibility-aware navigation requires dedicated datasets, tasks, and metrics beyond conventional embodied navigation evaluation. All data and code open-source at https://github.com/InSAI-Lab/VLAN. |
| 2026-11-13 | [Control Invariance on Manifolds through Tangent Bundle Splitting](http://arxiv.org/abs/2611.18339v1) | Luke Georgiou, Ross Drummond et al. | Control invariance for interconnected non-linear systems on manifolds requires conditions under which subsystem safety constraints can be combined to form an invariant set. This paper develops local product type control invariant sets on manifolds using tangent bundle splitting, adapted product coordinates, and subsystem tangency conditions satisfied simultaneously by one admissible input at each state. The resulting conditions establish control invariance of product sets without requiring independent subsystem dynamics, with separate control choices possible when inputs are independently constrained and each acts only on its subsystem. This provides a way to verify subsystem safety without discarding dynamic coupling and is illustrated by a torus example in which adapted coordinates give a curved constraint region a product representation. |
| 2026-11-13 | [Safety Analysis of Parameterized Networks with Shared Counters](http://arxiv.org/abs/2611.18175v1) | June Roupin, Arnaud Sangnier | We study networks with an unbounded number of processes sharing some natural counters that can be decremented, incremented and tested to zero. We look at two safety problems: the first one asks whether there is a network with an execution bringing a process in a given final state and the other one asks the execution to bring all the processes in the final state. In their full generality, these two problems are undecidable, but we show that decidability can be regained by either using a single counter or by forbidding the use of zero-tests (as it is is the case in Vector Addition Systems with States). For these decidable cases, we obtain low complexity results, the problems being solvable either in non-deterministic logarithmic space or in polynomial time. |
| 2026-11-13 | [Safety Certificates for Systems under Attack via Stackelberg Set Games](http://arxiv.org/abs/2611.18173v1) | Nikolaos Athanasopoulos, David Laverty et al. | We study a Stackelberg set selection game for linear systems subject to bounded disturbances and additive attacks. The defender selects the vector parameter of a polytopic safe set template; the attacker, after observing this choice, selects the vector parameter of an admissible attack set. The payoff is a signed, facet-normalised invariance margin that tests invariance exactly and quantifies either the available contraction or the facet inflation needed when invariance fails. Both controlled and robust controlled invariance are treated, corresponding to different information patterns for the control input. Exploiting the structure of parameterised polyhedra, we show that within each validity region the margin is concave in the attack parameter, which reduces the follower minimisation to finitely many extreme parameters. The complete max--min game is then solved by a finite family of LPs, one per defender validity region, each enforcing all attacker regions under a common defender decision and guaranteed margin. The framework handles hard state constraints and additional costs related to the safe set size, and recovers optimal set parameters together with state and attack realisations that activate the certificate. A load frequency control example under false data injection illustrates the construction, the validity decomposition, and the two information patterns. |
| 2026-11-13 | [Large Language Model-Empowered Intelligent Agent for Adaptive Fast-Charging Control of Lithium-Ion Batteries](http://arxiv.org/abs/2611.18140v1) | Joey Chan, Haoyu Pan et al. | Fast charging of lithium-ion batteries is a multi-objective control problem that requires reducing charging time and sustaining high energy throughput while suppressing accelerated degradation and safety-critical electrochemical reactions. Conventional protocols are easy to implement but have limited ability to adapt to variations in battery chemistry, operating conditions, and aging state. Model-based optimization and conventional data-driven control often require accurate models, costly iterative search, or task-specific policies. We develop a large language model (LLM)-empowered intelligent agent for adaptive battery fast-charging control. In this closed-loop framework, the agent interprets structured battery states, generates executable charging actions, observes the resulting battery response, and continuously adjusts subsequent decisions. We construct a large-scale simulation corpus with PyBaMM across five battery chemistries and 11 parameter sets and convert the resulting state-action-response trajectories into structured process data for agent training. Our two-stage strategy combines supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO). SFT establishes battery-control semantics and feasible decision priors from simulation-derived demonstrations. GRPO aligns the agent with a multi-objective control objective covering energy throughput, degradation mitigation, and operational constraints. Closed-loop simulations show that, under matched energy-throughput requirements, the agent achieves longer cycling life than conventional charging baselines and optimization-based strategies across heterogeneous battery configurations. This study provides a simulation-grounded proof of concept for using LLM agents as sequential decision makers in battery management and takes a step toward language-model-enabled control of physical energy systems. |
| 2026-11-13 | [Agent-as-a-Judge](http://arxiv.org/abs/2611.18067v1) | Runyang You, Hongru Cai et al. | Artificial intelligence has progressed from task-specific systems toward general models spanning perception, generation, reasoning, and autonomous interaction, making reliable evaluation essential for comparison, deployment, and post-training feedback. At scale, modern evaluation pipelines increasingly rely on LLM-as-a-Judge to produce scores, preferences, and critiques; in its fixed-context setting, the judge relies on the supplied evaluand and evaluation materials. Yet a reliable verdict may require evidence beyond the supplied evaluand and context---either because critical environmental states remain unrecorded in the static execution log, or because controversial factual claims require external verification. Passive judges cannot obtain that missing evidence before deciding, especially when evaluating agent trajectories, executable artifacts, or domain-specific reports. Recent work therefore develops active evaluators that seek missing evidence and update their grounds for judgment before deciding. We bring these advances together through Agent-as-a-Judge, focusing on how evaluators construct and control checking processes within the broader LLM-based judging landscape. We define the scope through evaluation-directed checks and the use of their results, covering both structured workflows and observation-driven control. We organize the design space by multi-agent collaboration, tool integration, memory, and planning, and show how these mechanisms supply the checks and evidence on which a verdict rests. We consolidate published scores and reproduce missing entries for key agentic judges; the comparison reveals uneven agentic gains and motivates dedicated agentic evaluation. We conclude by characterizing application demands and open challenges in cost, reliability, safety, and governance that condition trusted deployment of agentic judges at scale. |
| 2026-11-13 | [DeCompCert: A Verified Decompiler for CompCert-Generated AArch64](http://arxiv.org/abs/2611.18055v1) | Swarn Priya, Basavesh Ammanaghatta Shivakumar et al. | DeCompCert is a Rocq-verified decompiler for AArch64 assembly generated by CompCert. It reconstructs CompCert C through nine reverse passes, combining deductively verified transformations with translation validation. Untrusted OCaml and Datalog analyses propose candidate reconstructions, while verified validators accept only candidates that satisfy the required semantic correspondence. The composed correctness theorem guarantees that every defined, safe behavior of an accepted C reconstruction is matched by the input CompCert assembly; under a safety assumption, the two programs are observationally equivalent. We evaluate DeCompCert on approximately 2.16 million lines of C. With exact function and call signatures, it reconstructs and recompiles all 3,173 eligible contexts under both unoptimized and optimized CompCert pipelines. Without retained signatures, its cumulative inference portfolio validates 2,606 of 3,164 contexts (82.4%), while the platform-plus-IDA configuration validates 2,407 contexts (76.1%). Native execution and output-quality experiments further evaluate the generated C and characterize the remaining limitations in signature recovery and source resemblance. |
| 2026-11-13 | [A Generalizable Approach to Safety-Aware Robotic Behavior Planning with an Explicit Constraint Space](http://arxiv.org/abs/2611.18026v1) | Xianxu Wu, Shang-Ching Liu et al. | Large language and vision-language models have substantially advanced robotic behavior planning from natural-language instructions and visual observations. However, safe manipulation in tabletop environments remains challenging because the safety constraints required for a task depend on the surrounding scene and may not be explicitly specified in the instruction. We propose a safety-aware behavior planning framework based on an explicit Constraint Space that stores reusable textual safety rules. Rather than retrieving rules solely by semantic similarity, we formulate constraint selection as task- and environment-conditioned applicability learning. A lightweight query-conditioned dual-tower model identifies the constraints applicable to the current context, which are then provided together with the original task and image to a multimodal behavior planner. To handle incorrect constraint selections and previously unseen hazards, we introduce a multimodal Safety Inspector that evaluates generated primitive plans, reconstructs the complete applicable constraint set, and provides corrective planning feedback. Validated inspection feedback is further used for test-time adaptation of constraint applicability and parameter-efficient adaptation of the Safety Inspector through LoRA, while newly identified constraints are incorporated into the Constraint Space. The resulting framework integrates explicit safety knowledge representation,context-aware constraint selection, corrective safety inspection, and test-time knowledge adaptation for safety-aware robotic behavior planning. |
| 2026-11-13 | [Soteria: Formally Verified Planning with Runtime Enforcement for Safe LLM Agents](http://arxiv.org/abs/2611.17880v1) | Mike He, Ankush Desai et al. | LLM agents that execute multi-step tool calls must satisfy two objectives simultaneously: completing the user's task (utility) and conforming to domain policies and correctness constraints (safety). These objectives are in tension -- blocking an unsafe action prevents a violation but can leave the agent stranded with no principled way to recover. Existing approaches sacrifice one objective for the other. We introduce Soteria, a framework that reconciles safety and utility through verified hierarchical planning and runtime enforcement of specifications. Before execution, the agent generates a structured plan that is formally verified against the specifications prior to any tool invocation. During execution, the verified plan serves dual roles: it ensures that the agent takes specification-conformant trajectories and provides guidance when unsafe actions are blocked. Across multiple benchmarks covering a diverse range of tool-use tasks, Soteria achieves 1.7 times the average task success rate of the best baselines, with no violations of policy specifications, demonstrating that safety and utility need not be traded off. |
| 2026-11-13 | [Per-Agent-Safe Navigation of Globally Driven Robot Collectives via Affine Field Design](http://arxiv.org/abs/2611.17859v1) | Seunghwan Jang, Steven Ceron et al. | Safe navigation of robot collectives requires coordinating motion and deformation so that the collective can traverse constrained environments while ensuring each robot's safety. Existing controllers for swarmalator collectives largely rely on local control, where each robot receives an individual command, which causes the control and communication requirements to grow with the number of agents. We introduce a broadcast control formulation in which a single spatially affine field governs both the motion and deformation of the collective. The scalar component of the field regulates collective size, and its spatial gradient generates collective translation through non-reciprocal interactions. Both components enter the collective dynamics directly, which allows per-robot wall, actuation, and cohesion constraints to be enforced using control barrier functions within a single quadratic program for the whole collective. The resulting controller continuously adapts the collective size to the local environment, and enables safe passage through corridors narrower than the unconstrained collective without pre-planned resizing. We further derive how the available actuation budget determines the reachable speed and minimum passable corridor width, and we demonstrate that two independently driven collectives can simultaneously traverse a shared junction using distinct fixed phase labels without inter-collective coordination. |
| 2026-11-13 | [Aligning AI Teams](http://arxiv.org/abs/2611.17820v1) | Siddarth Srinivasan, Morgan Jane Matthews et al. | Previous work has shown that alignment training on individual LLM agents can fail to transfer to multi-agent settings. We investigate the underlying mechanisms for this failure, and explore interventions to preserve alignment transfer to multi-agent LLM teams. We focus on two software-engineering tasks (sepsis triage, a news recommender) and ten consulting-proposal generation tasks. We present four findings: (1) The single-vs-team safety gap does not show a clear trend with more capable models. We find that Anthropic's Mythos Preview shows the largest gap on one of our tasks but the smallest on another; (2) Many natural interventions, such as having agents critique each other's work or coordinate in a group chat, do not reliably bring agent teams back to single-agent alignment behavior; (3) diffusion of responsibility substantially accounts for the safety gap: agents in teams are less likely to proactively check for system-level safety failures and more likely to ignore, rationalize, or deflect responsibility for issues when they arise; (4) having a designated safety lead agent ensure system-level safety, and having a coordinator agent publish a safety specifications document are effective at restoring single-agent alignment. We also release magelab, a multi-agent LLM orchestration and experimentation framework. Our findings can help practitioners design agent teams that successfully preserve the behaviors expected of aligned individual agents. |
| 2026-11-13 | [LLMs are Vulnerable to Malicious Requests with Indistinguishable Intent](http://arxiv.org/abs/2611.17818v1) | Wenzhuo Xu, Zonghao Ying et al. | Jailbreak attacks can circumvent the safety mechanisms of Large Language Models (LLMs) and lead to the generation of harmful or policy-violating content. We first examine eight jailbreak methods against an LLM-based filter and observe that requests resembling normal user queries tend to have higher filter pass rates than those containing obvious anomalous patterns. This observation motivates the hypothesis that resemblance to normal requests may contribute to jailbreak success. We introduce \textbf{Detail Guidance Attack (DGA)}, a framework that uses an auxiliary LLM to construct detailed requests and interacts with the victim model in a single turn. Across 11 victim models and three datasets, the complete DGA pipeline achieves high attack success rates under our automated judging criterion, including rates above 95\% in several evaluated settings. We also evaluate a DGA-aware filter and conduct a user study on everyday requests with ambiguous or dual-use intent. The survey documents differing expectations about whether models should respond; the usability costs of the filter remain to be measured. |
| 2026-11-13 | [Super-Analysis for Aggressive yet Scalable Compiler Optimization](http://arxiv.org/abs/2611.17737v1) | Amirali Ebrahimzadeh, Amir Shaikhha | Aggressive compiler optimization today occupies two distant extremes. Conventional optimizers run in milliseconds but reason through ad-hoc heuristics that miss opportunities; superoptimizers synthesize faster code but take hours, confining them to small kernels and offline use. We introduce solver-based super-analysis, a middle ground that uses expensive SMT-based analysis while transformations remain conventional, avoiding program synthesis entirely. The key argument for super-analysis is that conventional compilers have rich transformation rules, but proving them to be safe requires complex reasoning: proving a safety check dead often needs facts from three independent LLVM analyses simultaneously, a conjunctive entailment problem suited to an SMT solver, not to any single dataflow pass.   We instantiate super-analysis in ODeSSy, an LLVM pass that proves sanitizer traps and native-language safety checks unreachable and removes them. One mechanism serves four frontend languages and yields speedups in each, including a 3.78x speedup for a GEMM implemented in Julia. A further solver-based optimization, guard synthesis, handles checks that are not dead for every input: the solver synthesizes a cheap invariant safety guard under which a fast check-free loop is cloned, speeding up a Swift Base64 encoder by 1.44x and a Julia LZ77 compressor by 4.01x. Compared with Souper, the synthesizing superoptimizer for LLVM, ODeSSy removes more safety checks in significantly less time, making it viable as an online compiler pass. |
| 2026-11-13 | [Ego2Flow: Benchmarking the Traffic-System Impact of Autonomous Driving on Maps Distilled from Real-World Roads](http://arxiv.org/abs/2611.17720v1) | Yu Dong, Weiying Shen et al. | End-to-end autonomous driving is predominantly evaluated through ego-vehicle safety and task completion. Its deployment in urban traffic, however, creates sustained interactions among vehicles whose collective effects remain insufficiently evaluated. We introduce Ego2Flow, a closed-loop benchmark for assessing the traffic-system impact of autonomous driving on maps distilled from real-world cities. To construct realistic 3D urban environments for sustained closed-loop traffic simulation, we develop an agent-guided map distillation framework that integrates real-world maps and observations with complementary geographic information. Through an iterative feedback mechanism and a traffic information graph that jointly guide scene generation, the framework preserves lane topology, critical road geometry and decision-relevant visual conditions while simplifying nonessential details and completing unobserved regions. Using our proposed framework, we generate and publicly release six CARLA maps from the Argoverse 2 dataset, covering the corresponding real-world regions and enabling paired comparisons of driving behavior under real and simulated observations. On these maps, Ego2Flow evaluates how autonomous driving policies affect traffic efficiency, safety, stability and energy consumption across diverse road structures. Compared with existing construction methods, our framework achieves the lowest average displacement error between planning outputs from paired real and simulated observations. We further benchmark representative driving policies to characterize their system-level impact on urban traffic. Project Page: https://github.com/Fisher-007/Ego2Flow. |
| 2026-11-13 | [Safety Side Effects of Preventative Persona Steering: An Empirical Audit](http://arxiv.org/abs/2611.17620v1) | Sabrina Tang, Nikolai Vo et al. | High-level behaviors such as sycophancy and hallucination correspond to directions in a model's activation space (Chen et al., 2025). These directions, known as persona vectors, can suppress undesirable traits either at inference time, subtracting the vector during generation, or through preventative steering, adding the vector during fine-tuning, so the optimizer never needs to encode the trait in weights, then removing the vector at evaluation. Chen et al. (2025) recommend preventative steering because it preserves Massive Multitask Language Understanding (MMLU) accuracy, but validations of this recommendation have measured capability alone. We primarily evaluate results on preventative steering of sycophancy and hallucination vectors on three models, measuring three safety benchmarks alongside trait expression, coherence, and MMLU. Our findings show that preventative steering still changes refusal behavior, but by an amount that depends on the trait and does not establish that preventative persona steering is safety-neutral, and steering methods should be evaluated on several safety benchmarks. |
| 2026-11-12 | [Grounding Object-to-Object Hazard Reasoning into 3D Risk Fields for Task-Conditioned Safety](http://arxiv.org/abs/2611.17566v1) | Mariana Sosa Guzmán, Yide Song et al. | Safety in robotics does not solely depend on geometric constraints, but also on semantics: the given task, environment properties, and interactions between objects in the scene. We present a framework based on vision-language models (VLMs) to infer task- and scene-dependent safety constraints for robot manipulation. We provide the VLM a structured safety tool based on an RGB-D observation and a natural-language task instruction that first decides whether the task should be rejected since its execution may result in a non-recoverable hazardous outcome. Otherwise, it identifies preventable object-pair hazards and specifies the spatial relations that must be avoided. Open-vocabulary segmentation and depth information are then used to ground the detected objects in 3D and construct continuous risk fields that represent the corresponding hazardous regions. These fields are integrated with an obstacle avoidance module that corrects nominal robot motions during execution while accounting for the risk field. We evaluate the framework on preventable hazards, tasks requiring rejection, and safe variants in simulation and the real-world. Project website: https://vlm-hazard-riskfield.github.io |

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



