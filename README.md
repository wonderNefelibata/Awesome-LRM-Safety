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
| 2026-06-10 | [Semantically-Aware Diver Activity Recognition Framework for Effective Underwater Multi-Human-Robot Collaboration](http://arxiv.org/abs/2606.12374v1) | Sadman Sakib Enan, Junaed Sattar | Effective multi-human-robot collaboration is essential for expanding human-led operations in the challenging and high-risk underwater environment. For autonomous underwater vehicles (AUVs) to become true teammates, they must be able to comprehend their surroundings and recognize a diver's activities to offer assistance and ensure safety. Towards this goal, we introduce DAR-Net, a novel transformer-based framework that analyzes complex underwater scenes to classify diver activities. Our contribution lies in a semantically guided learning formulation that couples transformer-based temporal reasoning with pixel-level scene supervision. This multi-loss training strategy explicitly aligns global activity recognition with local human-robot interaction semantics, which is particularly critical in low-visibility underwater conditions. To address the significant challenge of data scarcity in this domain, we present the first-ever Underwater Diver Activity (UDA) dataset, a foundational resource containing over 2,600 annotated images with pixel-level masks. Through rigorous experimental evaluations in a controlled environment, we demonstrate that DAR-Net achieves promising accuracy in recognizing six distinct diver activities, outperforming state-of-the-art models. While this dataset provides a crucial baseline, our work serves as a pioneering step, laying the groundwork for future research and facilitating the development of more intelligent, collaborative underwater robotic systems. |
| 2026-06-10 | [Verifiable Environments Are LEGO Bricks: Recursive Composition for Reasoning Generalization](http://arxiv.org/abs/2606.12373v1) | Hao Xiang, Qiaoyu Tang et al. | Reinforcement Learning (RL) with verifiable environments has emerged as a powerful approach for enhancing the reasoning capabilities of Large Language Models (LLMs). While prior research demonstrates that scaling environment quantity improves RL performance, existing manual or individual construction methods suffer from linear scaling limits, thereby hindering scalable reasoning generalization. This paper introduces RACES (\textbf{R}ecursive \textbf{A}utomated \textbf{C}omposition for \textbf{E}nvironment \textbf{S}caling), a framework that conceptualizes verifiable environments as composable building blocks that can be recursively assembled. The key insight is that when the codomain (output type) of one environment matches the domain (input type) of another, they can be automatically fused into a new verifiable environment, enabling recursive composition. RACES is implemented with 300 individual environments and defines a set of composition operators (\textsc{SEQUENTIAL}, \textsc{PARALLEL}, \textsc{SORT}, and \textsc{SELECT}) that induce diverse reasoning patterns. Extensive experiments show that RL training on these composite environments consistently enhances reasoning generalization. Specifically, RACES improves DeepSeek-R1-Distill-Qwen-14B by an average of 3.1 points (from 48.2 to 51.3) and boosts Qwen3-14B performance from 58.8 to 61.1 on six benchmarks, which are unseen during the construction of training environments. Moreover, RACES achieves performance comparable to training on 300 individual environments using only 50 base environments, demonstrating significant efficiency in environment utilization. |
| 2026-06-10 | [Traceable Virtual Sea Trials in the Marine Robotics Unity Simulator for Manoeuvring Assessment of Unmanned Surface Vehicles](http://arxiv.org/abs/2606.12349v1) | Paria Rezayan | Accurate identification of hydrodynamic derivatives is essential for control and navigation of Unmanned Surface Vehicles (USVs), but high-fidelity manoeuvring data from physical sea trials are constrained by cost and safety. Turning Circle (TC) and Zig-Zag (ZZ) trials remain fundamental to IMO and ITTC assessment procedures. This paper extends the Marine Robotics Unity Simulator (MARUS) by introducing a standardised Virtual Sea Trial framework for automated execution and data generation of TC/ZZ manoeuvres, with traceable command-actuation logging, system-identification (SI)-focused data conditioning, and automated extraction of IMO/ITTC-aligned manoeuvring metrics. A key contribution is a dedicated TC/ZZ data acquisition and post-processing pipeline, improving the repeatability and auditability of simulator-based manoeuvres while producing SI-ready datasets for hydrodynamic-derivative identification and digital-twin workflows. Another feature is explicit command-execution separation for differential-thrust steering, where inputs are recorded as ordered rudder-equivalent commands and realised actuation is logged as an execution-level proxy derived from applied thrust. Case-study results demonstrate repeatable and compliant manoeuvre behaviour. For TC tests, the normalised advance differs by approximately 3.9 percent between port and starboard sides, while the tactical diameter differs by approximately 4.6 to 4.7 percent. For ZZ tests, first and second overshoot excesses remain below 1 degree for both +/- 10 degree and +/- 20 degree manoeuvres, satisfying IMO criteria, while peak yaw rates range from approximately 4.1 to 5.8 deg/s. Overall, the framework provides a repeatable and auditable virtual sea-trial workflow for generating IMO/ITTC-aligned datasets and supporting system identification, hydrodynamic-derivative estimation, and digital-twin calibration. |
| 2026-06-10 | [ALIGNBEAM : Inference-Time Alignment Transfer via Cross-Vocabulary Logit Mixing](http://arxiv.org/abs/2606.12342v1) | Chirag Chawla, Pratinav Seth et al. | Domain fine-tuning degrades the safety of large language models: fine-tuned specialists readily comply with harmful prompts framed in domain language. Existing inference-time defenses that mix logits from a safe anchor model require both models to share a vocabulary, which rules them out for the cross-family specialists where safety is most degraded. We present ALIGNBEAM, a training-free method that lifts this restriction by translating anchor logits into the target model's vocabulary token-by-token at each decoding step; a small LLM judge then selects the safest among K candidate continuations. No weights are changed, and the safety-utility trade-off can be tuned at deployment without retraining. Across both cross-vocabulary and same-vocabulary evaluation pairs, ALIGNBEAM substantially raises refusal on adversarial benchmarks while keeping task accuracy and inference overhead within practical bounds. The results show that safety alignment can be transferred between model families at inference time, without touching either model's weights. |
| 2026-06-10 | [OCELOT: Inference-Leakage Budgets for Privacy-Preserving LLM Agents](http://arxiv.org/abs/2606.12341v1) | Jin Xie, Songze Li | Large language model (LLM) agents increasingly act on a user's behalf -- reading personal files, calling tools, transacting with external services -- possibly leaking personally identifiable information (PII) across trust boundaries at every step. Privacy here is a property not of a single output but of an entire trajectory, and three properties make it hard: leakage is cumulative, as individually innocuous releases accumulate across honest-but-curious or colluding sinks into inferences about a protected secret; bidirectional, as a malicious observation can inject instructions that turn the agent's own reasoning model against the user; and task-dependent, as the same field is necessary for one recipient yet gratuitous for another. Per-release contextual-integrity filters, information-flow controls, and posterior-leakage monitors each address part of this but none controls cumulative, inference-based leakage at runtime. We recast agent privacy as \emph{posterior-risk control} and present OCELOT, a runtime mediator that budgets how much an adversary's belief about a secret may improve across a trajectory, rather than filtering outputs. Its mechanism, \emph{Witness-Verified Declassification}, separates judgment from trust: an untrusted, locally fine-tuned defender model inspects each candidate release and emits structured evidence -- labeled atoms and proposed declassification operators -- which a deterministic verifier audits, charging a certified min-entropy cost for the chosen variant and authorizing the least-disclosing useful release under a sink-trust-weighted budget recorded on a tamper-evident ledger. Across diverse agent benchmarks and recent defenses, OCELOT attains significantly lower leakage at higher task utility, resists adaptive injection, jailbreak, cumulative inference, and sink collusion, and adds only modest overhead. |
| 2026-06-10 | [UGV-Conditioned Multi-UAV Informative Planning on a Shared Exposure Belief](http://arxiv.org/abs/2606.12306v1) | Lars Oerlemans, Moji Shi et al. | Safe ground navigation in large, threat-augmented environments requires aerial support that actively reduces the risks that a ground vehicle faces along its route. Existing aerial reconnaissance systems focus on mapping or covering the environment, but do not direct sensing toward regions that are most relevant for ground vehicle safety. In this paper, we address the problem of coordinating a team of unmanned aerial vehicles (UAVs) to improve the safety of an unmanned ground vehicle (UGV) navigating through unknown threat zones. A key aspect of our approach is a shared exposure belief that is updated online from aerial observations and used jointly by the UAV team and the ground vehicle. This enables us to direct aerial sensing towards route-relevant regions while allowing the UGV to replan around newly revealed threats. We coordinate the UAV team through spatial region assignment to avoid redundant sensing. Simulation experiments show that our approach reduces cumulative UGV exposure by 38% compared to a system that does not account for hazard levels, and reduces redundant aerial coverage from 38.8% to 3.7% under our multi-UAV coordination scheme. |
| 2026-06-10 | [The data-driven extreme value distribution: non-parametric tail estimation with a derived stability criterion](http://arxiv.org/abs/2606.12174v1) | Michael Sandbichler, Tobias Hell | Quantifying the likelihood of extreme events underpins risk assessment, yet classical Extreme Value Theory relies on asymptotic assumptions that fail in the data-sparse, non-stationary regimes practitioners increasingly face. We introduce the Data-Driven Extreme Value Distribution (DDEVD), a non-parametric estimator that aggregates all observations metastatistically and reconstructs the base distribution with a kernel, removing parametric tail assumptions. We derive its optimal bandwidth and prove a stability law $m < C\,n^{1+γ/2}$ relating reliable extrapolation to the extreme value index $γ$. In sub-hourly Alpine precipitation, DDEVD recovers stable 100-year return levels from single decades (calibration ratio $0.96$), departing from the full-record reference by over $50\,\%$ in fewer than one window in fifty -- versus one in five for a GEV fit. In metallurgical micrographs, it matches a generalised extreme-value fit on the safety-relevant grain-size tail, where the standard log-normal over-predicts by $58\,\%$ at $1\,\mathrm{cm}^{2}$. |
| 2026-06-10 | [AerialClaw: An Open-Source Framework for LLM-Driven Autonomous Aerial Agents](http://arxiv.org/abs/2606.12142v1) | Ke Li, Jianfei Yang et al. | Unmanned aerial vehicles (UAVs) are increasingly used in inspection, search and rescue, environmental monitoring, and emergency response. However, most UAV applications still rely on pre-defined command sequences or task-specific pipelines, where developers manually connect perception, planning, flight control, simulation, logging, and safety modules. This limits the flexibility, reproducibility, and extensibility of autonomous aerial systems. This paper presents AerialClaw, an open-source software framework that enables UAVs to operate as decision-making aerial agents rather than merely command-following platforms. Given a natural-language mission, AerialClaw allows an LLM-based agent to understand the task, maintain context, invoke executable aerial skills, observe perception and runtime feedback, and iteratively update its decisions in a closed loop. The framework adopts a modular brain-skill-runtime architecture, combining hard skills for atomic UAV operations, Markdown-based soft skills for reusable task strategies, document-driven agent state and capability boundaries, memory-driven reflection, safety-oriented runtime validation, and platform-agnostic execution adapters. AerialClaw supports lightweight mock execution, PX4 SITL with Gazebo, and AirSim-based simulation, together with a web console, pluggable model backends, example missions, simulation assets, and staged deployment scripts. By combining standardized aerial skills, document-driven agent state, memory, and closed-loop LLM decision-making, AerialClaw provides a reproducible and extensible open-source framework for building UAV systems that can interpret missions, make decisions, execute skills, and adapt their behavior from feedback. |
| 2026-06-10 | [Performance Analysis of YOLOv11 and YOLOv8 for Mixed Traffic Object Detection under Adverse Weather Conditions in Developing Countries](http://arxiv.org/abs/2606.12066v1) | Quoc Thuan Nguyen, Ha Anh Vu et al. | In modern vehicular systems, robust performance under harsh conditions has become a critical problem of autonomous driving. Our study delivers a comprehensive evaluation of the newest iteration of the YOLO series, which is YOLOv11 Nano architecture benchmarked against the widely adopted YOLOv8 Nano as a baseline on a custom fused dataset that combines the Indian Driving Dataset (IDD) [1] and Berkeley Deep Drive Dataset (BDD100K) [2]. We have analyzed the trade-offs among detection accuracy, inference speed, and computational efficiency in high-entropy scenarios involving dense mixed traffic, rain, and low-light conditions. Specifically, YOLOv11n achieves a mean Average Precision (mAP@50) of 46.6%, with a notable 3.2% improvement in Precision over the baseline, effectively reducing false positives in cluttered scenes. Furthermore, the proposed model exhibits enhanced energy efficiency, requiring 22% fewer FLOPs (6.3G vs. 8.1G) while maintaining real-time inference speed of 70.9 FPS on a Tesla T4 GPU, offering an optimal trade-off for safety-critical edge deployment. |
| 2026-06-10 | [Automating Geometry-Intensive Compliance Checking in BIM: Graph-Based Semantic Reasoning Framework](http://arxiv.org/abs/2606.12065v1) | Zixuan Xiao, Pei Troh Koh et al. | Automating compliance check for geometry-intensive regulations remains a significant technical bottleneck in Building Information Modeling (BIM), primarily due to the semantic disparity between high-level regulatory logic and structured IFC data. Existing methods, often reliant on static rule templates, struggle to traverse multi-hop reasoning chains or resolve latent spatial dependencies across multiple building entities. To address these challenges, a Spatial-Geometric Reasoning System for Building Information Modeling (SGR-BIM) is proposed as an integrative graph-driven reasoning framework. SGR-BIM dynamically constructs a cross-modal knowledge graph that aligns user intent, regulatory semantics, and BIM geometry, enabling interpretable reasoning without rigid hard-coding. Validated on 679 expert-verified queries from fire safety codes, the framework achieves 84.3% accuracy, representing an 8.6% improvement over enhanced-tool single-agent baselines. This research provides a graph-based semantic reasoning paradigm, enhancing the transparency and flexibility of automated geometric compliance check workflows in the Architecture, Engineering, and Construction (AEC) industry. |
| 2026-06-10 | [ChargeBD: Character-Aware Heterogeneous Agent Reasoning for Guided Engineering in Battery Development](http://arxiv.org/abs/2606.12057v1) | Rui Huang, Zekun Jiang et al. | Redox-flow battery (RFB) research spans molecular design, electrolyte optimization, electrode and membrane materials, stack operation, system management, and safety analysis, making it a constrained, multi-scale, and multi-objective energy-storage R&D problem. Although large language models (LLMs) can support scientific knowledge integration and proposal generation, generic LLM reasoning remains insufficiently adaptive across innovation-oriented exploration, rule-based execution, mechanistic modeling, and system-level trade-offs. Here we introduce ChargeBD, a character-aware heterogeneous-agent reasoning framework for guided engineering in battery development. Starting from a 50-question RFB-specific task set, we construct a 500-question ESS-LLM Benchmark and define MBTI-inspired persona agents as structured cognitive-bias templates rather than psychometric instruments or representations of real personalities. DeepSeek-V3-Plus is selected as the shared base model, and 16 MBTI-inspired persona agents are evaluated to construct a persona capability matrix and a cognitive advantage matrix. |
| 2026-06-10 | [A Lightweight Multi-Agent Framework for Automated Concrete Barrier Design](http://arxiv.org/abs/2606.12040v1) | Wanting Wang, Xiye Ma et al. | The design of reinforced concrete highway barriers is a safety-critical process that requires strict compliance with regulatory provisions such as the AASHTO-LRFD bridge design guidelines. Current engineering practice relies heavily on manual, iterative, and heuristic calculations to satisfy complex nonlinear material and mechanics constraints. Although Large Language Models (LLMs) demonstrate strong generative capabilities, their direct application to structural engineering remains limited by hallucination risks and insufficient physical grounding. To address these challenges, this study proposes a novel "generation-evaluation-optimization" closed-loop framework for automated concrete barrier design using the multi-agent orchestration capabilities of AutoGen. Experimental results demonstrate that the proposed agentic framework achieves over 98% design accuracy, significantly outperforming standalone general-purpose LLMs. More importantly, the study reveals that design performance is not necessarily correlated with model scale, where an 8B-parameter lightweight model could outperform unconstrained 631B-parameter flagship models. This finding highlights the potential to substantially reduce computational costs while improving the accessibility of AI-assisted engineering tools for industry applications. The source code for the proposed multi-agent design framework is available at the project GitHub repository: https://github.com/MXY820/barrier-design. Keywords: Structural Engineering; Multi-Agent Systems; Large Language Models; Concrete Barrier Design; AutoGen; Design Automation. |
| 2026-06-10 | [Human-Enhanced Loop Modeling (HELM): Agent-Based Finite Element Modeling of Concrete Bridge Barriers](http://arxiv.org/abs/2606.12025v1) | Quankai Wang, Yulin Xie et al. | Finite element (FE) modeling of safety-critical infrastructure such as bridge barriers requires high-fidelity nonlinear dynamic analysis, yet the current FE modeling process remains labor-intensive and lacks automation. This paper presents the Human-Enhanced Loop Modeling (HELM) framework, a collaborative human-agent protocol that decomposes long-sequence finite element modeling into discrete, visually verifiable checkpoints across geometry generation, boundary condition definition, and material assignment. The framework is demonstrated through a 20-case matrix of reinforced concrete bridge barriers under MASH TL-4 and TL-5 lateral loading conditions, interfacing specialized agents with two widely used commercial FE softwares, i.e., ANSYS and LS-PrePost. Experimental results show that HELM improves the baseline autonomous modeling success rate from 20% to 75%, with agent-level pass rates for geometry and boundary condition tasks approximately doubling. Error analysis reveals that spatial reasoning and algebraic logic limitations constitute the primary failure modes, underscoring the value of structured human-in-the-loop intervention for modeling automation. The complete agent design code and prompts are open-sourced and can be accessed at: https://github.com/SimAgentDev/Ansys-LSPP-AgentKit. |
| 2026-06-10 | [Runtime Enforcement of Hybrid System Properties](http://arxiv.org/abs/2606.12022v1) | Mir Md Sajid Sarwar, Srinivas Pinisetty et al. | Runtime enforcement has emerged as a promising approach for ensuring the safety of autonomous and cyber-physical systems operating in uncertain and dynamic environments. Unlike traditional runtime verification, runtime enforcement actively intervenes during execution to prevent property violations by modifying unsafe system behaviors. Existing enforcement frameworks primarily focus on untimed or discrete-time specifications and are often limited to delaying or suppressing events, making them inadequate for reactive systems exhibiting complex continuous dynamics. In this paper, we propose a runtime enforcement framework where safety requirements are modeled using Hybrid Automata (HA). The framework combines discrete-event editing with continuous-time monitoring to support enforcement actions such as suppression, delay, and insertion of events at arbitrary time instants. Upon observing environmental inputs, the automaton is initialized, and runtime reachability analysis is used to synthesize safe corrective actions. We formally define the enforcement problem for safety hybrid automata, establish enforceability conditions, and present an online enforcement algorithm for reactive systems. A detailed case study on an Adaptive Cruise Control (ACC) system demonstrates the effectiveness of the proposed approach in maintaining safety properties under unsafe controller behaviors. Experimental results show that the framework introduces minimal computational overhead while ensuring continuous compliance with safety requirements in real time. |
| 2026-06-10 | [Online Shift Detection and Conformal Adaptation for Deployed Safety Classifiers](http://arxiv.org/abs/2606.11949v1) | Jun Wen Leong | We present an online monitoring system for distributional shift in deployed safety classifiers, using calibrated sequential statistics to detect when a classifier has moved out of distribution. Upon detection, a conformal abstention layer adapts decision thresholds to recover a target error rate epsilon=0.1. In a pre-registered factorial evaluation (4 classifiers x 5 shift conditions x 20 seeds x 2 window sizes, 800 cells), the system achieves 86.6% valid detection (693/800, 95% CI [84.1%, 88.8%]) with mean latency of 39.5 steps. Detection holds across three ground-truth regimes: synthetic onset (86.6%), real temporal jailbreaks (85%, 17/20), and GCG adversarial attacks. Weighted conformal prediction recovers up to 39 pp of lost coverage for DeBERTa (ESS=46/300) but collapses for all other classifiers (ESS~300): logistic density ratio estimation achieves perfect source/target separability in high-dimensional embedding spaces, clipping all importance weights to the floor. DeBERTa shows a gradient from effective correction (paraphrase, ESS=46) to near-total collapse (adversarial suffix, ESS=206). PCA to 32 dimensions breaks the collapse, recovering 33 pp for Llama Guard and 21 pp for ShieldGemma. Variance decomposition reveals classifier (eta^2=0.243), shift type (eta^2=0.237), and their interaction (eta^2=0.185) all contribute substantially to detection latency variance (all p<0.001), indicating per-classifier monitoring profiles are necessary. |
| 2026-06-10 | [AutoMine Solution for AV2 2026 Scenario Mining Challenge](http://arxiv.org/abs/2606.11874v1) | Songliang Cao, Jiele Zhao et al. | With the development of autonomous driving systems, mining high-value, safety-critical, and planning-relevant scenarios from large-scale driving logs has become essential for data-driven evaluation. In this paper, we propose AutoMine, a robust self-refining scenario mining method based on LLMs and VLMs. AutoMine uses semantics-preserving prompt augmentation to reduce LLM prompt sensitivity, combines robust trajectory atomic functions with VLM-based functions to handle perception noise and open-world visual cues, and refines generated code through execution feedback from real logs. In the Argoverse 2 Scenario Mining Competition at CVPR 2026, AutoMine achieves a HOTA-Temporal score of 36.38 and a Timestamp BA score of 77.21. |
| 2026-06-10 | [Systematic Cybersecurity Risk Analysis of European Rail Traffic Management System](http://arxiv.org/abs/2606.11839v1) | Kacper Darowski, Sebastian N. Peters et al. | European Rail Traffic Management System (ERTMS) is a widely adopted standard unifying train management in the EU. While the standard allows for use cases like fully autonomous driving, cybersecurity has been an afterthought. Risk analysis enables the systematic assessment and prioritization of threats and mitigations. To date, it remains unclear which threats are most significant in ERTMS. This study systematically models components of ERTMS and analyzes their security in light of threats identified in the underlying technologies. The results suggest a concerning state of ERTMS, despite its critical role in railway safety. The use of legacy standards like EuroBalises and GSM-Railway (GSM-R) introduces vulnerabilities that persist across minimal ERTMS implementations, deployments incorporating various optional safety measures, and prospective future evolutions of the system, e.g., adopting Future Railway Mobile Communication System (FRMCS). Fully transitioning to European Train Control System (ETCS) level 2 was identified as the most significant measure for advancing ERTMS cybersecurity. The results indicate that a shift of ERTMS toward security is required to ensure availability and safe operation. While the chosen methodology proved its feasibility and shows remaining weaknesses of ERTMS, future work is needed to develop railway-centric adaptations to improve the quantification and evaluation of the computed risks. |
| 2026-06-10 | [Designing AI-Supported Focus Groups: A Role x Modality Playbook](http://arxiv.org/abs/2606.11835v1) | Zhiqing Wang, Steven Dow | Collecting participants' lived experiences is central to design research. Focus groups are uniquely valuable because participants not only share individual accounts but also respond to one another, surfacing comparison, disagreement, and collective sensemaking. However, focus groups are resource-intensive and highly sensitive to facilitation: moderators must probe for specificity, balance participation, manage topic flow, and sustain psychological safety, and subtle facilitation choices can shape what becomes salient. Recent HCI work and commercial meeting tools show that generative AI can scaffold live conversation through prompting, turn regulation, thematic mapping, and real-time summarization. Yet UXR teams lack a clear map of what these capabilities mean in focus groups and what methodological risks they introduce. We synthesize AI supports for live conversation and translate them into a focus-group-specific playbook organized by AI role (tool, co-host, host) and modality (text, voice, embodied).We synthesize prior work on AI-supported live conversation and propose a focus-group-specific playbook of AI supports organized by role (tool, co-host, host) and modality (text, voice, embodied). We characterize interactional trade-offs and identify open questions for evaluating AI-supported focus groups as methodological configurations. |
| 2026-06-10 | [VEQ: a fast parametric Grad--Shafranov solver for fixed-boundary tokamak equilibria with flexible source profiles](http://arxiv.org/abs/2606.11821v1) | Ruohan Zhang, Huasheng Xie et al. | Veloce EQuilibrium (VEQ) is a compact parametric framework for tokamak modeling workflows that repeatedly query continuous fixed-boundary equilibria at low latency. The VEQPy implementation evaluated here is an axisymmetric fixed-boundary Grad-Shafranov solver whose main solve enforces a variationally induced projected residual. Its active unknowns are MXH-type flux-surface harmonics and shifted-Chebyshev coefficients for radial profile and source closures. Six input routes accept pressure-gradient, toroidal-field-function, poloidal-flux-gradient, enclosed toroidal current, current-density and safety-factor information through route-specific closures, while all routes map to the same finite-dimensional residual operator. Controlled tests show route consistency for smooth, mutually compatible inputs generated from a common reference equilibrium. For Pareto-selected reduced configurations in three G-EQDSK cases, the most accurate selected rows correspond to a D-shaped case (9 active parameters, minor-radius-normalized shape error 1.4e-3, solve-only median 1.6 ms), an H-mode case (65, 1.1e-3, 19 ms), and an X-point case treated as a smoothed fixed-boundary representation of a diverted boundary (94, 1.9e-3, 15 ms). Sampled pointwise strong-form Grad-Shafranov diagnostics show that enriching the active representation mainly improves interior force balance, whereas the global RMS and maximum values for the H-mode and X-point cases remain dominated by near-boundary contributions. In an isolated one-dimensional transport-geometry coupling test against the target geometry read from G-EQDSK, the temperature-profile response remains below about one percent. These results support using VEQ for repeated equilibrium-geometry queries, provided that pointwise diagnostics are retained to screen cases requiring boundary refinement, local correction or higher-fidelity equilibrium solves. |
| 2026-06-10 | [Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code](http://arxiv.org/abs/2606.11817v1) | Yitong Zhang, Shiteng Lu et al. | Large Language Models (LLMs) are increasingly used for code generation, raising concerns that they may be misused to produce malicious code. Meanwhile, Grammar-Constrained Decoding (GCD) has been widely adopted to improve the reliability of LLM-generated code by enforcing syntactic validity. In this paper, we reveal a counterintuitive risk: this reliability-oriented technique can itself become an attack surface. We uncover a new jailbreak attack, termed CodeSpear, that exploits GCD to induce LLMs into generating malicious code. Our experiments show that simply applying a benign code grammar constraint can effectively jailbreak LLMs.   To address this vulnerability, we propose CodeShield, a safety alignment approach that robustly preserves safe behavior even under attacker-controlled grammar constraints. CodeShield aligns the model in the code modality by teaching it to generate honeypot code under GCD. Such code is semantically harmless, so it does not implement the malicious request, and structurally diverse, so it is difficult to suppress through grammar tightening. At the same time, CodeShield still preserves natural-language refusals when natural language is available. Experiments on 10 popular LLMs across 4 benchmarks show that CodeSpear outperforms representative jailbreak baselines and increases the attack success rate by more than 30 percentage points on average. CodeShield also restores safety under CodeSpear while preserving benign utility. Our findings reveal a fundamental risk of GCD and call for greater attention to its potential security implications. |

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



