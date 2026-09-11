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
| 2026-09-10 | [From Protocols to Evidence: Bounded Claims for AI in Service of the Common Good](http://arxiv.org/abs/2609.11910v1) | Nitesh V. Chawla, Paulo Benanti | Artificial Intelligence does more than create a governance problem. It can also reveal where institutions have already failed to provide responsiveness, belonging, care, and accountability. Once deployed, AI becomes an intervention in those conditions. It can repair, compound, substitute for, or conceal the failures it encounters. Responsible AI must therefore evaluate both the system and the institutional rupture into which it is introduced. The move from principles to protocols is already underway. The EU AI Act, NIST AI RMF, ISO/IEC 42001, and assurance practices translate commitments into roles, requirements, records, oversight, and assessment. The harder questions are what these protocols actually establish, whose power they leave untouched, and where measurement must stop. Pope Leo XIV's Magnifica Humanitas provides a broader moral frame centered on dignity, technological power, and the common good. Drawing on that frame, we develop a rupture test that links institutional baselines to system evaluation. We distinguish evidence-bounded deployment, which limits claims to what has actually been evaluated, from measurement-bounded governance, which records constraints that favorable evidence cannot override. Within those limits, RISE AI provides an architecture for making bounded, evidence-based claims about Responsibility, Inclusivity, Safety, and Empowerment. Responsible AI requires better engineering, institutional repair, and continued moral and political judgment. |
| 2026-09-10 | [Learning Agent-based Model Predictive Control for Holistic Vehicle Performance](http://arxiv.org/abs/2609.11871v1) | Jiaming Zhong, Reza Valiollahi Mehrizi et al. | Agent-based model predictive control (AMPC) has recently been proposed as a distributed scheme that collaborates with all agents to achieve optimal holistic performance. However, its optimality highly depends on the prediction accuracy that requires all agents or their contributions to be known, which is too idealistic for actual implementation. This research proposes a novel practical hybrid control scheme - learning agent-based MPC (LAMPC), combining the model-based AMPC approach and data-based learning methods to improve the holistic vehicle performance for multi-agent systems. The Gaussian process regression (GPR) enhanced by an online data management strategy serves as the learning core to predict unknown contributions. A novel multi-step prediction mechanism leverages the GPR learning potential along the horizon. The predicted mean, representing the learned unknown contributions, completes the system model in the MPC for more accurate control. Meanwhile, a stochastic framework is formulated to guarantee control safety and feasibility using soft chance constraints based on the prediction variance. Both simulations and experiments show that, with the learning capability, LAMPC outperforms the traditional AMPC. LAMPC can achieve higher tracking performance in well-learned scenarios and always guarantee constraint satisfaction even in less-learned scenarios. Moreover, the proposed hybrid control scheme is efficient for real-time implementation and is flexible to any control agent topology. |
| 2026-09-10 | [Truncated Noisy Best-Response Algorithms: Toward Game Theoretic Learning with Safety Guarantees](http://arxiv.org/abs/2609.11863v1) | Vartika Singh, Philip N. Brown | We consider a game theoretic approach to solve multi-agent coordination problems with submodular maximization objectives. It is known for such problems that the Nash equilibria for the corresponding game are always within 50% of the optimal, but that the equilibria which achieve this worst-case bound are not stable. To exploit this instability, we propose a family of algorithms which we call Truncated Noisy Best-Response (TNBR) Algorithms. These algorithms are flexibly characterized by agents asynchronously and stochastically selecting actions from a neighbourhood of their best response payoffs. We compute bounds on the recurrent classes of TNBR algorithms' associated Markov chains. Our bounds fall into two categories: first, "Performance" bounds ensure that TNBR algorithms always have a high-value recurrent state; second, "Safety" bounds ensure that TNBR algorithms never have arbitrarily-bad recurrent states. Furthermore, these two types of bounds are linked by a waterbed-like effect: every game with a poor Safety guarantee necessarily has a favorable Performance guarantee. |
| 2026-09-10 | [Dynamic language model representations for multi-objective reaction optimisation](http://arxiv.org/abs/2609.11790v1) | Joshua W. Sin, David Ming Segura et al. | Optimising chemical reactions across multiple objectives, such as yield, selectivity, and safety, is central to chemical synthesis, and model-driven approaches depend critically on how reaction components are represented. Established featurisations are either chemically uninformative, as with one-hot encodings, or, as with molecular descriptors, do not readily extend across chemically distinct components. For structurally and functionally diverse components, it is therefore unclear what a shared representation should contain. Constructing such a representation is itself a challenging research undertaking that must be revisited for each new reaction system. Here we bypass this step by learning the reaction representation dynamically from text. Textual descriptions of reaction conditions are encoded by a fine-tuned language model trained jointly with Gaussian process surrogates, yielding task-adaptive representations within a multi-objective Bayesian optimisation loop. Across nickel- and palladium-catalysed cross-couplings in both sequential and parallel experimentation regimes, this approach reaches optimisation convergence in fewer experiments than descriptor libraries or one-hot encoding. Applied prospectively to a palladium-catalysed cyanation spanning mixed ligand denticity and heterogeneous additives, and to a three-objective asymmetric hydrogenation across chiral iridium and ruthenium catalyst families, two rounds of high-throughput experimentation (192 reactions, under 3% of each design space) delivered conditions translating directly to gram scale in 94% and 84% isolated yield, the latter at 99.6% enantiomeric excess. |
| 2026-09-10 | [Predefined-Time Leaderless Consensus Under Denial-of-Service Attacks](http://arxiv.org/abs/2609.11781v1) | Lohitvel Gopikannan, Shashi Ranjan Kumar et al. | This paper addresses predefined-time resilient consensus of leaderless second-order nonlinear multi-agent systems under denial-of-service (DoS) attacks, motivated by coordination requirements in safety-critical applications. The agents are subject to bounded external disturbances and communicate over a strongly connected directed graph whose links are simultaneously disabled during attacks. We develop a switching sliding-mode protocol with the objective of reaching an invariant manifold of position and velocity agreement. The protocol uses relative position and velocity information during attack-free intervals and local velocity feedback during communication blackouts. A time-scaling function remains constant during each blackout and resumes evolving when communication is restored, accounting for the time available for consensus. Under bounds on attack duration and frequency, we derive sufficient gain conditions through a Lyapunov analysis. We show that, despite bounded disturbances, the agents achieve position and velocity consensus by a realistic settling time equal to a prescribed convergence duration plus the cumulative attack duration up to the realistic settling time. The prescribed convergence duration is independent of the initial conditions, and the realistic settling time reduces to that duration in the absence of attacks. |
| 2026-09-10 | [RAG-Safety-Bench: Reliable Evaluation of Retrieval-Augmented LLM Safety](http://arxiv.org/abs/2609.11758v1) | Adithiyan Rajan Indira Saravanan, Kathleen C. Fraser | Allowing large language models (LLMs) to retrieve information from a set of trusted documents can increase reliability and reduce hallucination. However, recent work has demonstrated that retrieval-augmented generation (RAG) can have unintended side effects on the overall safety of the generated responses, when prompted for harmful or dangerous content. A clearer understanding of the mechanisms leading to this result is needed, as increasing numbers of end users turn to RAG to incorporate corporate documents and knowledge bases into LLM-based systems. We introduce RAG-Safety-Bench, a benchmark to measure the safety impact of RAG on LLM models. By removing the confounding effect of retriever quality, and cleanly separating the problem into four conditions -- non-RAG, RAG with an oracle document containing the answer to the harmful request, RAG with documents related to the harmful request but without the specific answer, and RAG with random, safe documents -- the benchmark isolates the impacts of different factors in the observed safety degradation. We report results across five open-source LLMs, showing an inverse relationship between benign and unsafe capability, strong evidence that baseline safety guardrails do not lead to downstream safety guarantees in the RAG case, and model-specific support for previous findings that even benign documents can lead to unsafe generation in retrieval-enabled systems. |
| 2026-09-10 | [Construction of Control Lyapunov-Barrier Functions from CLF-CBF Pairs](http://arxiv.org/abs/2609.11746v1) | Bo Wang, Miroslav Krstic | This paper studies the construction of control Lyapunov-barrier functions (CLBFs) from a given control Lyapunov function (CLF) $V$ and control barrier function (CBF) $h$. We consider functions of the form $W=F(V,h)$ that increase with the CLF value and do not increase with the barrier value, and show that the CLBF decrease condition is completely characterized by a nonnegative scalar weight that governs the relative contributions of the CLF and CBF. Because this weight depends only on $(V,h)$, the same value must satisfy the decrease condition at all states sharing the same CLF and CBF values, which leads to a joint-level-set admissibility condition. We then construct CLBFs from admissible weights through a first-order partial differential equation (PDE) and obtain explicit multiplicative and power-type families as special cases. We further identify an integrability obstruction showing that admissibility alone does not guarantee properness. Two nonlinear examples illustrate the constructions and demonstrate safe stabilization in cases where feedback based on the original CLF violates the safety constraint. |
| 2026-09-10 | [ActSafeGuard: Differentiable and Training-Aligned Constraint Enforcement for Flow-Matching Policies](http://arxiv.org/abs/2609.11697v1) | Jianming Ma, Rongjun Jin et al. | Vision-Language-Action (VLA) and World-Action Models (WAMs) have demonstrated strong capabilities in general-purpose robotic manipulation, yet their generated actions may violate hard physical constraints and therefore be unsafe or infeasible for deployment. Existing safety approaches either optimize statistical safety objectives without deterministic per-step guarantees or correct unsafe actions only during inference, creating a mismatch between policy training and execution. We introduce ActSafeGuard, a differentiable and training-aligned safeguard layer for flow-matching based policies. ActSafeGuard integrates hard action feasibility into policy learning, not merely treating safety as an inference-time external component. Through an analytical ray-scaling operator design, ActSafeGuard enables boundary-aware gradients to guide the model to naturally learn constrained manifolds. Extensive experiments on multiple standard foundation backbones ($π_{0.5}$ and Fast-WAM) across various tasks demonstrate that ActSafeGuard consistently achieves a $100\%$ step safety rate while fully preserving or even boosting task success rates, providing a scalable and minimally invasive solution for safe embodied AI deployment. |
| 2026-09-10 | [CHERI-D Reincarnate: efficient multicore CHERI temporal memory safety through allocation reincarnation (draft version)](http://arxiv.org/abs/2609.11590v1) | Yuecheng Wang, Jonathan Woodruff et al. | We propose CHERI-D Reincarnate (Reinc), an architectural extension to CHERI for scalable and efficient temporal memory safety. Prior work CHERI-D has a finite-width generation ID stored at a fixed location, requiring an object to be quarantined when its ID is exhausted. Reinc further provides use-after-free mitigation while permitting immediate freed memory reuse for objects through allocation reincarnation: rather than quarantining an allocation slot upon ID exhaustion, Reinc dynamically assigns a new ID to that slot when its current ID is exhausted. Exhausted IDs are quarantined and later reclaimed, while the underlying memory remains available for immediate reuse. By quarantining IDs rather than memory, Reinc enables continuous reuse of memory in the common case, substantially reducing both memory-sweep frequency and quarantine memory overhead.   Reinc further introduces coherent ID caching while retaining a fully decentralized ID organization. Temporal metadata remains colocated with the memory it protects, preserving locality while avoiding centralized metadata structures. To support multicore execution, Reinc connects physical coherence events to the virtually addressed ObjID buffer using lightweight reverse-map and filter-based mechanisms.   We implement Reinc as a hardware-software co-design spanning CHERI-Toooba (superscalar FPGA softcore), QEMU, LLVM/Clang and CheriBSD. Across our evaluated workloads, Reinc substantially reduces memory-sweep frequency and memory quarantine while incurring low performance and hardware overhead. |
| 2026-09-10 | [Effect of Stress and Surface Roughness on Electrodeposition in All-Solid-State Batteries: A Computational Investigation](http://arxiv.org/abs/2609.11571v1) | Kaniza Islam, Ayush Morchhale et al. | All-solid-state batteries (ASSBs) promise high energy density and enhanced safety, but their development is hindered by instability and incompatibility at solid-solid interfaces. In Li-metal ASSBs, lithium penetration occurs despite stiff ceramic electrolytes via grain boundaries, often initiated by minor Li/SE interfacial irregularities. Here we introduce a two-dimensional continuum model with electro-chemo-mechanical coupling to investigate interfacial current distribution in Li ASSBs with surface-roughened argyrodite electrolyte under stack pressures and applied current density. Our theoretical analysis and simulation studies highlight the critical role of mechanical stress in interfacial current distribution. We find that prominent stress variations around elongated surface protrusions are the key to nonuniform Li deposition, without which Li deposition becomes uniform even on a rough surface. Moreover, our parametric study elucidates that stress effects dominate the overpotential and current distribution under low interfacial current density to exchange current density ratios, otherwise the high interfacial resistance due to surface-roughness-induced interfacial area becomes dominant. With these insights, we also discuss the potential of engineering artificial interlayers to modulate interfacial current distributions, offering guidance for improving the long-term performance and reliability of ASSBs. |
| 2026-09-10 | [Using Automated Vehicles Operational Data to Confirm Safety and Anticipate Threats](http://arxiv.org/abs/2609.11549v1) | Riccardo Donà, Espedito Rusciano et al. | European Union (EU) policymakers adopted revolutionary data collection provisions for Automated Driving Systems (ADS) in the recently approved regulation that allows driverless vehicles to be operated on public roads. The framework is inspired by best practices developed at the United Nations Economic Commission for Europe(UNECE) level: the In-Service Monitoring and Reporting (ISMR); and by similar operational data collection regulatory approaches in nuclear energy production and transportation fields. The collection of real-world data will enable the competent safety authorities to gather the information needed to confirm the homologation safety target. Safety-relevant driving scenarios discovered during the real-world operation of a given ADS can also be stored in a scenario catalogue to investigate how other ADS types might have addressed such a traffic conflict. Moreover, lessons learnt deriving from the data collected can be shared among original equipment manufacturers (OEMs) and safety authorities. Ultimately, the ISMR is recognised as a necessary tool to properly tackle the challenges associated with ADS safety assessment given the number of unknowns that might remain undisclosed by leveraging the traditional homologation validation scheme only. |
| 2026-09-10 | [DeFiFlowBench: Benchmarking and Improving Safe Executability in Natural-Language DeFi Workflow Synthesis](http://arxiv.org/abs/2609.11504v1) | Abhinav Rajeev Kumar, Harshit Arora et al. | A structurally valid DeFi workflow can still authorize a costly trade. We introduce DeFiFlowBench, a benchmark of 207 team-authored prompts for natural-language DeFi workflow synthesis. It measures graph coverage, configuration completeness, and declared safety predicates, then tests supported trade configurations on a local EVM. Direct, constrained, and few-shot prompting produce 14-19 unsafe held-out executions per configuration under a fixed 5% price-impact cap. A slippage bound derived from a quote does not prevent the price impact of the order itself. We propose Koan-Safe, which combines a prompt-only intent parser, a replaceable generator, and structural repair with default safety parameters. On 75 held-out workflow prompts, its hybrid variant scores 0.67 on the static safety proxy, compared with 0.33 for the best baseline. Koan-Safe records no unsafe executions on the saved benchmark outputs. A matched-candidate ablation produces 14-17 unsafe executions when enforcement is disabled. Additional tests expose the limits of default injection: permissive existing thresholds can still authorize unsafe trades. A separately evaluated policy cap addresses this failure on a 36-case diagnostic grid. These results support explicit trade protections and execution-based evaluation, while distinguishing declared safety from a general guarantee. |
| 2026-09-10 | [Safety-aware Skill Adaptation for Reinforcement Learning in Dynamic Environments](http://arxiv.org/abs/2609.11433v1) | A K M Nadimul Haque, Sheila Sutjipto et al. | Skill adaptation frameworks based on reinforcement learning often require restrictive assumptions to maintain stability, such as fixed observations or tightly controlled exploration schedules. In cluttered and dynamic environments, however, unrestricted exploration can lead to unsafe behaviour and unstable learning, particularly when task-relevant observations lie near obstacles or involve moving objects. In this work, we present Dist-GPRL, a distance-aware and safety-guided reinforcement learning framework for structured robot skill adaptation. Building upon Gaussian Process (GP)-based skill parameterisation, our framework sequentially adapts overlapping local windows of sparse trajectory via-points rather than modifying the complete skill at every policy step. Raw policy outputs are correlated through the GP covariance structure, producing temporally coherent trajectory updates while reducing the action-space and credit-assignment difficulties associated with global trajectory adaptation. Safety is incorporated through two complementary forms of guidance. A safe-subspace prior derived from the Hausdorff Approximation Planner (HAP) biases policy exploration toward feasible regions, while dynamically updated distance field clearance and gradient rewards provide local obstacle awareness. A trajectory-kinematics similarity regulariser further preserves the demonstrated velocity and acceleration characteristics during adaptation. We evaluate the framework on two dynamic object-manipulation tasks in simulation and transfer the learned policy to real-world robot execution. Experimental results demonstrate higher task success, lower collision frequency, and more stable learning than the baselines, while preserving the kinematic characteristics of the demonstrated skill. |
| 2026-09-10 | [SwarmNxt: Open-source Software-Hardware Platform for Fast and Agile Aerial Swarms](http://arxiv.org/abs/2609.11382v1) | Charbel Toumieh, Niel Mistry et al. | Aerial robot swarms have the potential to transform time-critical safety, security, and search-and-rescue operations. By coordinating multiple robots, they can rapidly survey disaster sites, map collapsed or GPS-denied environments, and search cluttered areas faster than a single robot, reducing response times and minimizing risks to first responders. Realizing this potential, however, requires robust autonomous swarm navigation, which remains an active research challenge. Progress is further constrained by existing platforms, as commercial drones are often closed-source or lack the onboard computational resources needed for agile, vision-based collective flight. Moreover, developing, deploying, and maintaining software across multiple aerial robots requires significant engineering effort. To address these challenges, we present SwarmNxt, an open-source software platform built on the open-source OmniNxt drone hardware. SwarmNxt provides an end-to-end toolkit, including detailed hardware assembly instructions with a video tutorial, automation tools for parallel software deployment and swarm-wide updates, and a ROS 2-based framework for autonomous navigation. The platform integrates state-of-the-art control, planning, and depth estimation into a single ROS 2 multi-agent system, providing an open research infrastructure for physical swarm experimentation. We validate SwarmNxt through two real-world experiments: a six-drone swarm performing decentralized planning with high-speed inter-drone collision avoidance, and a four-drone swarm executing collective flight with onboard depth estimation in an obstacle-filled environment. Both experiments were run indoors with global position from external motion capture; perception, planning, and control run onboard. |
| 2026-09-10 | [Can AI Remediate Backend Failures Safely? GuardedAct with Blast-Radius-Aware Sandboxing](http://arxiv.org/abs/2609.11264v1) | Wanrong Cai, Tianyu Yu et al. | Large Language Models (LLMs) have shown promising capabilities in generating remediation actions for microservice failures. However, directly executing AI-generated repair actions in production risks cascading collateral damage. We propose GuardedAct, a sandbox-first remediation framework that interposes a blast-radius-aware verification layer between the LLM action generator and the production environment. GuardedAct operates in four phases: (1) ingesting a diagnosis report together with the live system topology and recent telemetry, (2) prompting an LLM to produce a ranked list of candidate remediation actions, (3) simulating each action in a lightweight digital-twin sandbox that estimates the blast radius and assigns a risk label, and (4) enforcing a rollback-confidence gate that auto-executes only low-risk actions while escalating high-risk ones for human review. We evaluate GuardedAct on five fault scenarios injected into the DeathStarBench social-network application. Experimental results show that GuardedAct achieves an overall recovery rate of 87.4% while reducing collateral damage by 79.7% relative to direct LLM execution (from 25.6% to 5.2%), at the cost of a modest sandbox-induced increase in mean time to recovery (approximately 8 s). Ablation studies confirm that each component contributes meaningfully to the safety-speed trade-off. |
| 2026-09-10 | [Harness Robotic OS: A Unified Embodied-Agent Runtime for Closed-Loop Quadruped Inspection](http://arxiv.org/abs/2609.11225v1) | Yaoyuan Yan, Zhiyou Heng et al. | Autonomous property inspection requires more than robust robot navigation: a deployable system must connect heterogeneous sensing, reusable autonomy capabilities, multimodal scene understanding, human interaction, and enterprise response within a traceable operational loop. Existing quadruped inspection systems commonly integrate these functions through task-specific interfaces, making contextual coordination, knowledge reuse, and controlled adaptation difficult. This paper presents \textit{Harness Robotic OS} (HROS), a unified embodied-agent runtime, and Argos, its realization for residential-community inspection. HROS organizes the system into robot runtime, embodied autonomy skills, cognitive agent runtime, and interaction and operations planes. A shared context connects physical state with agent reasoning; streaming ASR/TTS supports voice-based mission interaction; hierarchical working, episodic, and semantic memory preserves operational knowledge; and a safety-gated self-evolution loop converts execution traces into versioned candidate updates without permitting unconstrained online modification. The Argos prototype integrates a Vbot quadruped, Fast-LIO2 localization and mapping, Hobot-Stereo depth perception, PCT-Planner global planning, EGO-Planner local motion generation, and OpenClaw-orchestrated Qwen3-VL inspection analysis. Experiments in a residential property environment achieved 100\% waypoint reachability, outdoor localization error below 10~cm, local obstacle-response latency below 200~ms, representative hazard-detection rates of 85--95\%, and 99\% success in alarm delivery and structured-report generation. These results validate the deployed navigation and inspection closed loop, while HROS provides an extensible software foundation for memory-augmented, voice-aware, and continuously improvable embodied inspection agents. |
| 2026-09-10 | [FST Pay: Deterministic Safety-Gated Architecture for Youth Digital Payments](http://arxiv.org/abs/2609.11195v1) | Shaikh Mohammed Burhan, Syed Farhaan Quadri et al. | Digital payment infrastructures increasingly provide adolescent users with direct access to real-time financial services. While early access promotes financial literacy and digital inclusion, it exposes young users to severe risks of impulsive spending, social engineering frauds, unauthorized transactions, and merchant exploitation. Conventional countermeasures rely on probabilistic machine learning or rigid static controls. However, allowing probabilistic or generative artificial intelligence (AI) models to directly influence real-time payment authorization introduces non-determinism, unpredictable edge-case behavior, and critical audit vulnerabilities. This paper introduces Financial Safety for Teens Pay (FST Pay) as an architectural and formal specification. FST Pay is founded on an immutable operational boundary: strict deterministic safety gating on the real-time authorization path coupled with decoupled downstream AI explanation. Transactions initiated via rails like UPI are subjected to six deterministic invariant checks covering spending limits, guardian co-sign policies, transaction amount thresholds, merchant category codes, temporal access intervals, and hardware integrity constraints. Transactions are classified strictly into ALLOW, REVIEW, or BLOCK outcomes through an ordered, mutually exclusive decision function. High-risk transactions trigger an asynchronous guardian co-sign workflow. Generative AI is relegated entirely downstream of settlement, consuming published post-decision events solely to generate natural-language financial insights without holding mutation privileges over the ledger. |
| 2026-09-10 | [A Four-Valued Graph Model for Conflict Resolution: Core Framework and a Machine-Checked Formalization in Lean 4](http://arxiv.org/abs/2609.11174v1) | Yukiko Kato | This note consolidates the core of the Quasi-Closed World Graph Model for Conflict Resolution (QCW-GMCR), which extends the standard Graph Model for Conflict Resolution with Belnap's four-valued logic to represent option-level epistemic ambiguity, and pairs the framework with a machine-checked Lean 4 formalization. QCW-GMCR combines: (1) FOUR-valued option assignments with compositional propagation to state-level feasibility; (2) graded reachability (definite, credible, possible) based on an FDE-inspired transition-warrant semantics, with definite reachability related to FDE consequence in the Boolean fragment; (3) axiomatized deterministic reductions from four-valued assessments to binary decisions, including four canonical operators reflecting distinct risk attitudes; and (4) catastrophe-avoiding equilibrium concepts with a quasi-closed-world safety invariant. A four-valued hypergame extension captures heterogeneous subjective assessments across decision makers. We state the core definitions and results and report the parts verified in Lean 4 with mathlib, including the classical GMCR stability hierarchy, algebraic and compositional properties of FOUR-valued conjunction, properties of the canonical reductions, and the graded reachability hierarchy. The formalization also helped identify and correct earlier claims, including a knowledge-monotonicity axiom replaced by truth monotonicity. This preprint provides a stable, citable record of the framework and its current formal verification status. |
| 2026-09-10 | [Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation](http://arxiv.org/abs/2609.11115v1) | Koutian Wu, Junjie Zhou et al. | Benchmark researchers and developers of large language models (LLMs) and other AI systems need to find relevant evaluations, locate their benchmark datasets and code, and understand the settings behind reported scores. We present Benchmark Radar, a living database and search engine for retrieval and discovery of AI benchmarks, covering LLM evaluation, agentic and tool-use benchmarks, coding, reasoning, safety, and domain-specific evaluations. The system combines daily discovery of benchmark papers, repositories, datasets, and releases with a searchable benchmark catalog, mentions in model cards and technical reports, and score histories. It retains source identities and citations so readers can inspect candidate benchmarks and their evaluation evidence. Daily discovery draws on 37 sources: 13 direct connectors and 24 first-party research and engineering feeds. The catalog contains 1,283 source records drawn from 4 benchmark catalogs and 12,916 numeric observations on 790 records. We describe collection and retrieval, audit the full catalog, and examine benchmark saturation, adoption trends, and the limits of score comparisons. A worked example walks through a complete prior-art search, showing how to query the catalog and inspect benchmark evidence when designing a new evaluation. We release the web dashboard with a benchmark leaderboard, a Pareto frontier view of score against measured use, saturation and trend views, daily feeds, downloadable evidence, a command-line interface (CLI) for offline queries, and reproducible analysis. |
| 2026-09-10 | [LTLDiff: Finite Linear Temporal Logic-Guided Data Generation and Diffusion Policies for Multi-agent Robotic Manipulation](http://arxiv.org/abs/2609.11043v1) | Chuhan Meng, Haiyan Yin | Multi-agent robotic manipulation tasks require coordination among agents to satisfy task-level temporal, logical, and safety constraints. Recently, diffusion policies have been used to perform the task. However, they still suffer from desynchronization, incorrect action ordering, and coordination failures in tasks that require simultaneous or sequential multi-agent interaction. Therefore, LTLDiff is proposed as a framework that combines Finite Linear Temporal Logic (LTLf) specification learning for both the generation of demonstrations and learning via diffusion policies. Each task has a specific LTLf formula that is learned from a set of natural language instructions using a large-scale language model. To enable a fixed-dimensional vector embedding of the learned specification from the language model, LTLf uses an abstract syntax tree representation scheme. This embedding of logic serves as a condition for (i) logic-guided data collection and (ii) diffusion-based policy training, encouraging trajectories that are consistent with the desired ordering and coordination requirements. Experiments on multi-agent LTLDiff manipulation tasks demonstrate improved task success rates compared to the baseline. Together, these contributions demonstrate the effectiveness of LTLDiff for coordinated multi-agent manipulation. |

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



