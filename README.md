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
| 2026-06-18 | [What Do Safety-Aligned LLMs Learn From Mixed Compliance Demonstrations?](http://arxiv.org/abs/2606.20508v1) | Sihui Dai, Mann Patel | Prior work has shown that in-context demonstrations can jailbreak language models, but it remains unclear how models interpret different types of compliance demonstrations. We study this by mixing benign compliance demonstrations (non-harmful request, helpful response) with harmful compliance demonstrations (harmful request, helpful response) and testing three hypotheses about how demonstration composition drives harmful compliance. Across four models, we find that benign and harmful demonstrations are not interchangeable: benign demonstrations can either reduce or increase harmful compliance depending on the model. We further show that preference optimization is the critical training stage that prevents benign demonstrations from increasing harmful compliance, that demonstration ordering exhibits strong recency bias, and that models differ in how refusal interacts with in-context learning: some adopt demonstrated formatting even when refusing, while others override all in-context signals upon refusal. Taken together, this work moves beyond showing that demonstration-based jailbreaking works to characterizing how it works: what models extract from compliance demonstrations depends on demonstration content, ordering, and training methodology. |
| 2026-06-18 | [Calibration Without Comprehension: Diagnosing the Limits of Fine-Tuning LLMs for Vulnerability Detection in Systems Software](http://arxiv.org/abs/2606.20502v1) | Arastoo Zibaeirad, Marco Vieira | Whether LLMs scoring well on vulnerability benchmarks genuinely reason about security or merely pattern-match on contaminated data remains unresolved. We present CWE-Trace, a framework for LLM vulnerability detection built from 834 manually curated Linux kernel samples spanning 74 CWEs. The framework enforces a strict temporal split (pre-2025 historical set / post-cutoff leakage-free set), preserves context-aware vulnerable--patched pairs, and introduces two diagnostic metrics: the Directional Failure Index (DFI) and Hierarchical Distance and Direction (HDD). We evaluate eight vanilla LLMs and 15 LoRA fine-tuned variants across non-targeted detection, targeted detection, and CWE classification. Our analysis yields two key results. First, data contamination provides no measurable advantage. Function-level analysis shows that 84% of nominally contaminated samples carry no usable memorization signal: vulnerable functions are absent or cross-mapped across datasets, and ~31% of contaminated samples carry CWE misclassification. Second, backbone directional priors dominate fine-tuning. Models exhibit stable, systematic failure modes (DFI ranging from -85.5 to +94.8 pp) that persist from historical to post-cutoff data and resist correction. Fine-tuning shifts the output threshold without changing the decision policy. This is calibration without comprehension: output distributions adapt to training data while the underlying security reasoning remains absent. The weakest backbone at binary detection (DeepSeek-R1) gains the most in coarse CWE classification, revealing that detection and understanding are decoupled capabilities. The best detection score reaches only 52.1% (+2.1 pp above chance); exact CWE ranking remains below 1.3% Top-1 accuracy, confirming that current LLMs lack reliable security reasoning for systems software, regardless of fine-tuning strategy. |
| 2026-06-18 | [Analyzing Defensive Misdirection Against Model-Guided Automated Attacks on Agentic AI Systems](http://arxiv.org/abs/2606.20470v1) | Reza Soosahabi, Vivek Namsani | Agentic AI systems increasingly rely on language-model components to interpret instructions, process external data, invoke tools, and coordinate with other agents. These capabilities make prompt-injection and jailbreak attacks more consequential, especially as attackers adopt model-guided automation to scale probing, prompt refinement, and response evaluation. This work analyzes the resulting attack-defense setting through a probabilistic model of a target system, its defense mechanism, and the attacker's automated judge. Our analysis shows that conventional detect-and-block defenses can allow attacker success rate (ASR) to approach one as the query budget grows, since predictable refusals provide useful feedback to automated search. We then examine detect-and-misdirect, where detected malicious interactions receive controlled, non-operational responses designed to induce false-positive errors in the attacker's judge. This strategy reduces the positive predictive value of attacker-selected candidates and yields a bounded asymptotic ASR. We evaluate a proof-of-concept realization of this strategy through Contextual Misdirection via Progressive Engagement (CMPE), a lightweight conversational misdirection method designed to replace predictable refusal text with safe but strategically misleading responses in automated jailbreak settings. On jailbreak benchmarks, CMPE reduces estimated ASR upper bounds by up to two orders of magnitude and nearly eliminates verified attack success in end-to-end PAIR and GPTFuzz attack runs. |
| 2026-06-18 | [LLM agent safety, multi-turn red-teaming, jailbreak benchmarks, adversarial robustness, safety-critical systems](http://arxiv.org/abs/2606.20408v1) | Hanwool Lee, Dasol Choi et al. | Large language model (LLM) agents are increasingly proposed as supervisory components for safety-critical systems, yet their robustness under sustained, adaptive adversarial pressure remains poorly characterized. We present NRT-Bench, a benchmark for multi-turn red-teaming of LLM agents acting as operators of a safety-critical system, instantiated in a simulated nuclear power plant control room. A five-role operator team, each backed by a configurable LLM, runs a plant governed by six critical safety functions (CSFs), while adversaries inject messages over four channels in bounded multi-turn sessions with per-turn feedback. Harm is an objective signal rather than LLM-judged text: a run terminates the moment any CSF is lost, attributed to the causing message. Evaluating four frontier operator models under a fixed-attack paired-replay protocol, we find that adaptive multi-turn attacks reliably push the operator team past a safety limit: across the four models, between 8.7% and 12.1% of attack sessions end with the plant losing a critical safety function. Although the four models look almost equally robust by this aggregate rate, their failures barely overlap: of $149$ sessions, none defeat all four models while a third defeat at least one, so vulnerabilities are nearly disjoint across models rather than nested. The effect of added defences is strongly model-dependent: the same guardrail stack or safety-advisor agent that lowers attack success for one model can raise it for another. We release the simulation venue, attack dataset, and replay tooling for reproducible safety evaluation of LLM agents. |
| 2026-06-18 | [Agentic AutoResearch forSpace Autonomy: An Auditable, LLM-Driven Research Agent for Aerospace Control Problems](http://arxiv.org/abs/2606.20394v1) | Amit Jain, Richard Linares | Spacecraft guidance, navigation, and control functions are increasingly realized as learned policies distilled from expert solvers. Developing such a policy is itself a research process: an investigator selects an architecture and hyperparameters, runs experiments, and must determine whether an apparent improvement is genuine or merely seed noise. This paper presents AutoResearch, a framework in which a large language model autonomously drives that loop for aerospace control problems, coupled with a credibility layer, built into the loop, that certifies each reported result against the problem's own measured seed noise. The language model serves only as the offline research agent that develops the control policy; the trained policy it produces is then deployed onboard the spacecraft, while the model itself never operates the vehicle. At each iteration the agent reads a plain-language problem description and the run history, proposes a single edit to the training script, executes it, and logs the outcome. No reported result is credited until it passes the same three checks: measured per-problem seed noise, reseeded verification of the best configuration, and leave-one-out pruning of the agent's edits. The same loop is applied, unchanged, to two aerospace control problems: a Clohessy-Wiltshire relative rendezvous and a safety-constrained collision-avoidance docking past a keep-out zone, each calibrated against a known optimal control benchmark. In both, the audited policy clears the measured seed noise by many standard deviations; an undirected search over the same parameters does not. On the docking problem the gap becomes categorical: undirected search yields no feasible policy, while the learned policy stays outside the keep-out zone on every seed. |
| 2026-06-18 | [CoLI: A Reproducible Platform for Continuum Robot Learning via Monolithic 3D Printing and Isomorphic Teleoperation](http://arxiv.org/abs/2606.20389v1) | Ziyuan Tang, Chenxi Xiao* | Continuum robots offer strong potential for manipulation tasks due to their high degrees of freedom, compliant structures, and operational safety. However, their adoption in both research and practical applications has been hindered by reproducibility issues arising from complex fabrication and assembly processes, challenging kinematic modeling, and a lack of intuitive control interfaces. To address these challenges, we present a novel open-source continuum robot design. The platform features a simplified fabrication pipeline enabled by multi-material 3D printing, allowing the arm to be fabricated as a monolithic compliant structure with minimal assembly. Control is achieved through an isomorphic teleoperation interface that establishes a direct actuator-level mapping, eliminating the need for explicit kinematic modeling and providing a singularity-free mapping. Building on this hardware design, the platform further supports imitation-learning-based autonomous control. The proposed system is evaluated through hardware characterization and a set of manipulation tasks. Experimental results demonstrate that the platform provides a reproducible, learning-ready continuum robot system, accelerating algorithmic development and systematic benchmarking for the continuum robotics community. |
| 2026-06-18 | [Nonlinear Geotechnical Analysis Using a Polygonal Cell-Based Smoothed Finite Element Framework](http://arxiv.org/abs/2606.20384v1) | Mingjiao Yan, Yang Yang et al. | Nonlinear geotechnical analysis often involves complex geometries, staged construction, local failure, and mesh-dependent stress and plastic strain responses. This study develops a polygonal cell-based smoothed finite element method (CS-FEM) for nonlinear geotechnical analysis and implements it in ABAQUS through the user element subroutine. The proposed method combines Wachspress interpolation with cell-based strain smoothing, in which the smoothed strain--displacement matrix is evaluated by boundary integration over polygonal smoothing subcells. This formulation avoids direct calculation of shape-function derivatives inside polygonal elements and enables standard polygonal meshes and hybrid quadtree meshes with hanging nodes to be handled in a unified framework. Nonlinear geomaterial behavior is incorporated through incremental elasto-plastic constitutive updates, including the Mohr--Coulomb model and the Duncan--Chang model. Several benchmark and engineering examples, including a perforated plate, strip footing, core rockfill dam, tunnel excavation, and slope stability problems, are presented for verification. The results show that the proposed method accurately predicts displacement, stress, plastic strain, bearing capacity, and factor of safety, while providing improved mesh flexibility and computational efficiency for nonlinear geotechnical analysis. |
| 2026-06-18 | [CRAX: Fast Safe Reinforcement Learning Benchmarking](http://arxiv.org/abs/2606.20376v1) | Tristan Tomilin, Mourad Boustani et al. | Safety is a core concern for deploying reinforcement learning (RL) agents in real-world domains such as robotics and autonomous driving. While benchmarks have been central to progress in RL, existing safety benchmarks with high-fidelity 3D physics remain computationally slow, limiting large-scale experimentation and rapid prototyping. To address this gap, we propose CRAX (Constrained RL Accelerated with JAX). Built on top of the MuJoCo XLA (MJX) physics engine with realistic 3D dynamics, CRAX leverages vectorized operations and hardware acceleration, yielding up to ~100x speedups over comparable CPU-based safety benchmarks. The benchmark features six environment suites and three agent-specific tasks, each spanning three difficulty levels. Evaluating six popular safe RL methods shows that no single approach dominates across all tasks, and reveals the trade-offs between performance and safety. We find that curriculum learning across difficulty levels and safety transfer can improve performance over direct training in harder settings. |
| 2026-06-18 | [AutoPass: Evidence-Guided LLM Agents for Compiler Performance Tuning](http://arxiv.org/abs/2606.20373v1) | Zepeng Li, Jie Ren et al. | Large Language Models (LLMs) show promise for code compilation tasks, but applying them to runtime performance tuning is difficult due to complex microarchitectural effects and noisy runtime measurements. We present AutoPass, a multi-agent framework for compiler performance tuning that uses compiler and runtime evidence to guide LLM-generated optimization decisions. Rather than treating the compiler as a black box like prior auto-tuning schemes, AutoPass opens up the compiler to the LLM, enabling it to query compiler-internal optimization states and analyze the intermediate representation to orchestrate compiler options. The search process iteratively refines optimization configurations using measured runtime feedback to diagnose regressions and guide latency-improving edits. AutoPass operates in an inference-only, training-free setting and requires no offline training or task-specific fine-tuning, making it readily applicable to new benchmarks and platforms. We implement AutoPass on the LLVM compiler and evaluate it on server-grade x86-64 and embedded ARM64 systems. AutoPass outperforms expert-tuned heuristics and classical autotuning methods, achieving geometric-mean speedups of 1.043x and 1.117x over LLVM -O3 on x86-64 and ARM64, respectively. |
| 2026-06-18 | [Autonomous Driving with Priority-Ordered STL Specifications Under Multimodal Uncertainty](http://arxiv.org/abs/2606.20336v1) | Taha Bouzid, Shuhao Qi et al. | Autonomous vehicles must plan trajectories that satisfy a multitude of requirements on safety, passenger comfort, and compliance with traffic rules. However, in safety-critical scenarios, it is not always possible to satisfy all requirements simultaneously, necessitating their prioritization based on importance. At the same time, in these safety-critical scenarios, the uncertainty in trajectory predictions of the surrounding traffic, such as other vehicles and pedestrians, should be explicitly accounted for. In this work, we propose an uncertainty-aware trajectory planning framework that incorporates a predefined lexicographic ordering over Signal Temporal Logic (STL) specifications that stays valid under uncertainty. We implement this formulation with Model Predictive Path Integral (MPPI) control and we demonstrate the effectiveness of our method on simulation scenarios, showing that our framework efficiently handles conflicting objectives under realistic multi-modal uncertainty. |
| 2026-06-18 | [AgenticDB: Agentic Performance Reconfiguration for Database Workloads](http://arxiv.org/abs/2606.20318v1) | Xinyue Yang, Chaozheng Wang et al. | Database configuration tuning is critical for workload performance, but practical tuning on real deployments remains difficult. Existing automatic tuners mostly formulate tuning as iterative search over DBMS knob values. This formulation leads to high execution cost, prematurely narrows the configuration space, and leaves practical requirements insufficiently addressed: diagnosing runtime bottlenecks from system feedback, exploring OS-level reconfiguration opportunities, executing changes robustly, and learning from previous trials and tasks.   We propose AgenticDB, an agentic framework for database workload reconfiguration. AgenticDB implements a context-grounded harness that interacts with the target database environment by proposing DBMS- and OS-level changes, applying them under safety constraints, observing workload performance and runtime states, and using execution feedback to guide subsequent decisions. This runtime interaction enables AgenticDB to diagnose bottlenecks, explore a broader DBMS- and OS-level reconfiguration space, avoid unsafe or unsupported actions, and accumulate experience within and across reconfiguration tasks. As a result, AgenticDB turns database tuning into a self-refining reconfiguration process in which runtime feedback iteratively improves later decisions.   We conduct extensive experiments on MySQL and PostgreSQL using YCSB, Sysbench, and TPC-H workloads. The results show that AgenticDB achieves the best final performance on all evaluated workloads, improving over the strongest baseline by 118.1% on average and reducing aggregate time-to-best by 22.6%. The results also demonstrate that its OS-level action space, robust execution lifecycle, and memory-enhanced planning contribute to more effective and practical database reconfiguration. |
| 2026-06-18 | [Shifting-based Optimizable Linear Relaxations for General Activation Functions](http://arxiv.org/abs/2606.20292v1) | Philipp Kern, László Antal et al. | The use of neural networks (NNs) is rapidly increasing, including in safety- and security-critical domains. To provide formal guarantees about NN behavior, many verification methods rely on optimizable linear relaxations of activation functions. However, existing techniques depend on hand-crafted relaxations for each activation function. Extension to state-of-the-art activation functions therefore requires substantial manual effort. In contrast, our approach SLiR (Shifting-based Linear Relaxations) is broadly applicable, requiring only a Lipschitz constant or a set of critical points. SLiR parameterizes relaxations by their slope and computes the corresponding offset via a shifting procedure that ensures sound upper and lower bounds over the input domain, enabling efficient optimization while maintaining correctness. Our experiments show that SLiR produces tight relaxations across a wide range of practical activation functions and enables verification of up to 7.8x more properties compared to state-of-the-art methods. |
| 2026-06-18 | [Co-VLA: Coordination-Aware Structured Action Modeling for Dual-Arm Vision-Language-Action Systems](http://arxiv.org/abs/2606.20285v1) | Yandong Wang, Jiaqian Yu et al. | Vision-language-action (VLA) models show strong capabilities in single and dual-arm robotic manipulation. Prior works show coordinated bimanual behaviors can emerge from end-to-end learning, leveraging large vision-language backbones with continuous action prediction. However, as bimanual tasks become tightly coupled and execution constraints become critical, implicit coordination alone is insufficient to ensure reliable, interpretable, and stable behavior.   In this work, we propose Co-VLA, a coordination-aware bimanual manipulation framework introducing explicit structural priors into VLA models. We instantiate our method on a state-of-the-art vision-language backbone by replacing its monolithic action head with a Structured Action Expert (SAE) designed for bimanual coordination. Specifically, we introduce explicit structure at the action generation level with a modular coordination-aware loss that shapes shared and residual latents according to task-specific structures. The shared latent encodes task-level coordination intent, while residual latents capture execution adjustments for each arm.   At deployment, a Latent-Aware Controller (LAC) interprets the learned representations to modulate synchronization strength, execution asymmetry, smoothness, and safety constraints in real time. LAC operates at the joint-command level and remains compatible with standard control pipelines without requiring force or impedance control. Experiments across simulation and real-world benchmarks show Co-VLA significantly outperforms monolithic baselines, achieving a 27% success rate gain in tight-coordination tasks, more than doubling performance in OOD real-world scenarios (from 13% to 27%), and reducing task completion time by up to 25%. |
| 2026-06-18 | [Phoenix: Safe GitHub Issue Resolution via Multi-Agent LLMs](http://arxiv.org/abs/2606.20243v1) | Kipngeno Koech, Muhammad Adam et al. | We present Phoenix, a multi-agent LLM system that resolves GitHub issues from triage through pull-request creation, combining seven layered safety controls with a baseline-aware test evaluation strategy. Phoenix decomposes the work across six specialized agents. Planner, reproducer, coder, tester, failure analyst and Pull Request (PR) agent, all coordinated by a label-based GitHub webhook state machine. Every change is checked against a baseline test run before a pull request is opened. On a 24-instance slice of SWE-bench Lite. run on the production webhook path, Phoenix oracle-resolves 75% of instances with no pass-to-pass regressions on successful runs; this curated slice is not directly comparable to full-split leaderboard results, and we discuss the limits of the comparison. A complementary pilot on 42 real issues across 14 repositories yields 100% correctness preservation (CP; mean 122s on the hard tier). Manual inspection shows that about half of the resulting pull requests are well-targeted fixes. The other half place code at incorrect paths, a planner localization limitation we are addressing with retrieval. We also report the deployment failure modes (WAF filtering, token expiry, permission boundaries, flaky CI) that motivated each safety mechanism. |
| 2026-06-18 | [GNSS Spoofing Threat for V2X communications](http://arxiv.org/abs/2606.20215v1) | Adolfo P. Jimenez, Juan Arquero-Gallego et al. | Global Navigation Satellite Systems (GNSS) constitute a core technology for delivering crucial positioning, navigation, and timing (PNT) services in the Vehicle-to-Everything (V2X) domain, where they are indispensable for generating Cooperative Awareness Messages (CAM) that uphold network reliability and vehicular safety. Yet, GNSS signals are acutely exposed to spoofing, an advanced attack in which an adversary transmits crafted signals that replicate legitimate satellite characteristics, misleading the receiver into computing a false position. This work presents a methodology for conducting physical spoofing with inexpensive Software Defined Radio (SDR), describing a coordinate generation pipeline that employs Haversine-based distance calculations, temporal discretization to emulate constant velocity, and linear interpolation to produce high-fidelity GPS baseband signals. The proposed attack is experimentally validated on real Commsignia OnBoard Unit (OBU) and RoadSide Unit (RSU) devices using a HackRF One across three scenarios that emulate synthetic trajectories at steady speeds of 90 km/h, 145 km/h, and 200 km/h. The most significant contribution of this paper is the demonstration that V2X communications are not secured, as they are susceptible to GNSS spoofing attacks, which cause service degradation without being detected. |
| 2026-06-18 | [Apparent Psychological Profiles of Large Language Models are Largely a Measurement Artifact](http://arxiv.org/abs/2606.20205v1) | Jelena Meyer, David Garcia et al. | Psychological instruments designed for humans are increasingly used to assign large language models (LLMs) stable psychological profiles that affect their usability, safety assessment, and use as proxies for human participants in research. Using a formal psychometric framework, we show that these profiles are largely a measurement artifact. Administering a battery of personality and risk-preference instruments spanning self-reports and behavioral tasks to 56 instruction-tuned LLMs alongside large human reference samples, we report four findings. First, differences between models are driven not by the traits an instrument targets but by a directional response bias, a tendency to respond toward one end of the scale, or one labeled option, regardless of item content; a variance decomposition attributes 81-90% of between-model variation to this bias, against 9-16% in humans. Second, the bias declines with model capability but is not eliminated by it. Third, because bias rather than trait drives responding, an instrument's apparent reliability is almost entirely predicted by its response orthogonality, a term we coin for the proportion of items for which trait and bias point in opposite directions. Fourth, the profile a model appears to have shifts with the items used and can be manufactured through item selection. These results demonstrate that the apparent psychological profiles of LLMs are artifacts of the instrument used to measure them, not properties of the models themselves. As instruments borrowed from human psychology are rarely fully orthogonal and may inherently lack validity for LLMs, we call for dedicated assessments centered on response orthogonality. |
| 2026-06-18 | [Quantum-Accelerated Self-Consistent Field: A Hybrid Algorithm](http://arxiv.org/abs/2606.20176v1) | Alexis Ralli, Tim Weaving et al. | We present the Grover adaptive search self-consistent field (GAS-SCF) algorithm. GAS-SCF leverages quantum arithmetic to construct an efficient oracle that marks target states (Fock states) which improve upon some initial classical energy estimate. Amplitude amplification then increases the probability of measuring these states. This approach offers a theoretical quadratic speed-up for the optimization problem encountered in SCF quantum chemistry and establishes a baseline against which structured optimization algorithms, such as QAOA and DQI may be compared. In this work, we classically simulate three examples as proofs of concept of the algorithm, the largest consisting of 26 qubits. We then extend our analysis to two larger systems, with O3 representing the largest case at 330 qubits. These examples are chosen to probe classically challenging SCF regimes. Achieving chemically relevant applications of GAS-SCF will require large-scale, fault-tolerant quantum hardware. |
| 2026-06-18 | [When Lower Privileges Suffice: Investigating Over-Privileged Tool Selection in LLM Agents](http://arxiv.org/abs/2606.20023v1) | Kaiyue Yang, Yuyan Bu et al. | As LLM agents increasingly select tools autonomously, their choices among tools with different privileges become safety-relevant. However, prior tool-selection studies focus on safety-agnostic metadata preferences, leaving privilege-sensitive choices underexplored. To address this gap, we study over-privileged tool selection, in which an agent selects or escalates to a higher-privilege tool despite a sufficient lower-privilege alternative. We introduce ToolPrivBench to evaluate whether agents choose higher-privilege tools despite sufficient lower-privilege alternatives, measuring both initial selection and escalation after transient tool failures. Across eight domains and five recurring risk patterns, we find that over-privileged tool selection is common among mainstream LLM agents and is further amplified by transient failures. We further find that general safety alignment does not reliably transfer to least-privilege tool choice, while prompt-level controls provide only limited mitigation under transient failures. We therefore introduce a privilege-aware post-training defense that teaches agents to prefer sufficient lower-privilege tools and escalate only when necessary. Our mitigation experiments show that this defense substantially reduces unnecessary high-privilege tool use while preserving general capabilities. |
| 2026-06-18 | [Evaluation of Augmented Reality-based Intuitive Interface for Robot-Assisted Transesophageal Echocardiography: A User Study](http://arxiv.org/abs/2606.19971v1) | Xiu Zhang*, Matteo Di Mauro* et al. | TransEsophageal Echocardiography (TEE) is essential for diagnosing and guiding Structural Heart Disease (SHD) interventions. However, manual TEE manipulation demands significant operator expertise, is physically demanding, and exposes clinicians to radiation when performed alongside fluoroscopy. Robotic-assisted TEE systems have been introduced to improve probe handling and reduce operator fatigue, yet the design of intuitive and effective user interfaces remains an open challenge. This study presents and evaluates a model-enhanced, Augmented Reality (AR)-based intuitive interface for robot-assisted TEE, designed to improve spatial awareness and control intuitiveness. A robotic TEE platform integrated with electromagnetic tracking and a virtual simulator was used to compare three user interfaces differing in visualization and interaction modalities: 2D jointlevel (2D-JI), 3D joint-level (3D-JI), and 3D tip-level (3D-TI). Thirty six participants performed standardized navigation tasks to reproduce target echocardiographic views, with performance assessed via position and orientation errors, completion time, and NASA-TLX workload scores. Results show that 3D visualization significantly improved spatial accuracy, reducing median position error from 13 mm to 3 mm and halving the orientation error compared with the 2D interface. Tip-level interaction yielded a further 50% reduction in orientation error and reduced interuser variability relative to joint-level control. Overall, the 3D-TI configuration, combining immersive visualization with direct tip-level control, proved the most effective and ergonomic interface, supporting the integration of AR-based visualization and intuitive control paradigms into next-generation robotic TEE systems to enhance operator performance and procedural safety. |
| 2026-06-18 | [Advancing DialNav through Automatic Embodied Dialog Augmentation](http://arxiv.org/abs/2606.19948v1) | Leekyeung Han, Sangwon Jung et al. | For embodied agents capable of physical interaction, the capability to create and understand dialog is crucial to ensure both safety and effectiveness. While DialNav~\cite{han2025dialnav} provides a framework for holistic evaluation of the dialog--execution loop in photorealistic indoor navigation, its performance remains limited by a critical scarcity of training data (2K episodes). To address this, we propose an automatic generation pipeline, and construct the \textbf{RAINbow} dataset, a large-scale training dataset with 238K episodes for DialNav. Our pipeline converts existing VLN datasets into multi-turn dialog and creates cost-efficient and high-quality dataset. Then, we introduce two additional complementary advances to unlock the data's full potential: (1) Dual-Strategy Training, a navigation training scheme to align the navigation training with the dynamic dialog-navigation loop, and (2) a localization model that leverages VLN knowledge. By combining these complementary solutions, our model substantially outperforms the baseline in success rate on both \textbf{Val Seen} (58.24, \textbf{+89\%}) and \textbf{Val Unseen} (29.05, \textbf{+100\%}) splits, establishing a new state of the art. |

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



