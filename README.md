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
| 2026-03-10 | [Model Merging in the Era of Large Language Models: Methods, Applications, and Future Directions](http://arxiv.org/abs/2603.09938v1) | Mingyang Song, Mao Zheng | Model merging has emerged as a transformative paradigm for combining the capabilities of multiple neural networks into a single unified model without additional training. With the rapid proliferation of fine-tuned large language models~(LLMs), merging techniques offer a computationally efficient alternative to ensembles and full retraining, enabling practitioners to compose specialized capabilities at minimal cost. This survey presents a comprehensive and structured examination of model merging in the LLM era through the \textbf{FUSE} taxonomy, a four-dimensional framework organized along \textbf{F}oundations, \textbf{U}nification Strategies, \textbf{S}cenarios, and \textbf{E}cosystem. We first establish the theoretical underpinnings of merging, including loss landscape geometry, mode connectivity, and the linear mode connectivity hypothesis. We then systematically review the algorithmic landscape, spanning weight averaging, task vector arithmetic, sparsification-enhanced methods, mixture-of-experts architectures, and evolutionary optimization approaches. For each method family, we analyze the core formulation, highlight representative works, and discuss practical trade-offs. We further examine downstream applications across multi-task learning, safety alignment, domain specialization, multilingual transfer, and federated learning. Finally, we survey the supporting ecosystem of open-source tools, community platforms, and evaluation benchmarks, and identify key open challenges including theoretical gaps, scalability barriers, and standardization needs. This survey aims to equip researchers and practitioners with a structured foundation for advancing model merging. |
| 2026-03-10 | [Emergency Locator Transmitters in the Era of More Electric Aircraft: A Comprehensive Review of Energy, Integration and Safety Challenges](http://arxiv.org/abs/2603.09918v1) | Juana M. Martínez-Heredia, Adrián Portos et al. | The progressive electrification of aircraft systems under the more electric aircraft (MEA) paradigm is reshaping the design and qualification constraints of safety-critical avionics. Emergency locator transmitters (ELTs), which are essential for post-accident localization and search and rescue (SAR) operations, have evolved from legacy 121.5/243 MHz beacons to digitally encoded 406 MHz systems, typically retaining 121.5 MHz as a homing signal in combined units. In parallel, the modernization of the Cospas-Sarsat infrastructure, especially MEOSAR, together with multi-constellation global navigation satellite system (GNSS) integration and second-generation beacon capabilities, is reducing detection latency and enabling richer distress messaging. However, MEA platforms impose stricter constraints on available power, thermal management, wiring density, and electromagnetic compatibility (EMC). As a result, ELT performance increasingly depends not only on the device itself, but also on its installation conditions and on the aircraft's overall electrical environment. This review summarizes the ELT architectures and activation/operational cycles, outlines key technological milestones, and consolidates the main integration challenges for MEA, with emphasis on energy autonomy, battery qualification frameworks, EMC and installation practices, and survivability-driven failure modes (e.g., antenna/feedline damage, mounting, and post-impact shielding). Finally, emerging trends include ELT for distress tracking (DT), energy-based designs, advanced health monitoring, and certification-ready pathways for next-generation SAR services are discussed, highlighting research directions that can deliver demonstrable, certifiable gains in reliability, energy efficiency, and robust integration for future electrified aircraft. |
| 2026-03-10 | [Robust Cooperative Localization in Featureless Environments: A Comparative Study of DCL, StCL, CCL, CI, and Standard-CL](http://arxiv.org/abs/2603.09886v1) | Nivand Khosravi, Meysam Basiri et al. | Cooperative localization (CL) enables accurate position estimation in multi-robot systems operating in GPS-denied environments. This paper presents a comparative study of five CL approaches: Centralized Cooperative Localization (CCL), Decentralized Cooperative Localization (DCL), Sequential Cooperative Localization (StCL), Covariance Intersection (CI), and Standard Cooperative Localization (Standard-CL). All methods are implemented in ROS and evaluated through Monte Carlo simulations under two conditions: weak data association and robust detection. Our analysis reveals fundamental trade-offs among the methods. StCL and Standard-CL achieve the lowest position errors but exhibit severe filter inconsistency, making them unsuitable for safety-critical applications. DCL demonstrates remarkable stability under challenging conditions due to its measurement stride mechanism, which provides implicit regularization against outliers. CI emerges as the most balanced approach, achieving near-optimal consistency while maintaining competitive accuracy. CCL provides theoretically optimal estimation but shows sensitivity to measurement outliers. These findings offer practical guidance for selecting CL algorithms based on application requirements. |
| 2026-03-10 | [The Bureaucracy of Speed: Structural Equivalence Between Memory Consistency Models and Multi-Agent Authorization Revocation](http://arxiv.org/abs/2603.09875v1) | Vladyslav Parakhin | The temporal assumptions underpinning conventional Identity and Access Management collapse under agentic execution regimes. A sixty-second revocation window permits on the order of $6 \times 10^3$ unauthorized API calls at 100 ops/tick; at AWS Lambda scale, the figure approaches $6 \times 10^5$. This is a coherence problem, not merely a latency problem. We define a Capability Coherence System (CCS) and construct a state-mapping $\varphi : Σ_{\rm MESI} \to Σ_{\rm auth}$ preserving transition structure under bounded-staleness semantics. A safety theorem bounds unauthorized operations for the execution-count Release Consistency-directed Coherence (RCC) strategy at $D_{\rm rcc} \leq n$, independent of agent velocity $v$ -- a qualitative departure from the $O(v \cdot \mathrm{TTL})$ scaling of time-bounded strategies. Tick-based discrete event simulation across three business-contextualised scenarios (four strategies, ten deterministic seeds each) confirms: RCC achieves a $120\times$ reduction versus TTL-based lease in the high-velocity scenario (50 vs. 6,000 unauthorized operations), and $184\times$ under anomaly-triggered revocation. Zero bound violations across all 120 runs confirm the per-capability safety guarantee. Simulation code: https://github.com/hipvlady/prizm |
| 2026-03-10 | [Beyond Fine-Tuning: Robust Food Entity Linking under Ontology Drift with FoodOntoRAG](http://arxiv.org/abs/2603.09758v1) | Jan Drole, Ana Gjorgjevikj et al. | Standardizing food terms from product labels and menus into ontology concepts is a prerequisite for trustworthy dietary assessment and safety reporting. The dominant approach to Named Entity Linking (NEL) in the food and nutrition domains fine-tunes Large Language Models (LLMs) on task-specific corpora. Although effective, fine-tuning incurs substantial computational cost, ties models to a particular ontology snapshot (i.e., version), and degrades under ontology drift. This paper presents FoodOntoRAG, a model- and ontology-agnostic pipeline that performs few-shot NEL by retrieving candidate entities from domain ontologies and conditioning an LLM on structured evidence (food labels, synonyms, definitions, and relations). A hybrid lexical--semantic retriever enumerates candidates; a selector agent chooses a best match with rationale; a separate scorer agent calibrates confidence; and, when confidence falls below a threshold, a synonym generator agent proposes reformulations to re-enter the loop. The pipeline approaches state-of-the-art accuracy while revealing gaps and inconsistencies in existing annotations. The design avoids fine-tuning, improves robustness to ontology evolution, and yields interpretable decisions through grounded justifications. |
| 2026-03-10 | [ENIGMA-360: An Ego-Exo Dataset for Human Behavior Understanding in Industrial Scenarios](http://arxiv.org/abs/2603.09741v1) | Francesco Ragusa, Rosario Leonardi et al. | Understanding human behavior from complementary egocentric (ego) and exocentric (exo) points of view enables the development of systems that can support workers in industrial environments and enhance their safety. However, progress in this area is hindered by the lack of datasets capturing both views in realistic industrial scenarios. To address this gap, we propose ENIGMA-360, a new ego-exo dataset acquired in a real industrial scenario. The dataset is composed of 180 egocentric and 180 exocentric procedural videos temporally synchronized offering complementary information of the same scene. The 360 videos have been labeled with temporal and spatial annotations, enabling the study of different aspects of human behavior in industrial domain. We provide baseline experiments for 3 foundational tasks for human behavior understanding: 1) Temporal Action Segmentation, 2) Keystep Recognition and 3) Egocentric Human-Object Interaction Detection, showing the limits of state-of-the-art approaches on this challenging scenario. These results highlight the need for new models capable of robust ego-exo understanding in real-world environments. We publicly release the dataset and its annotations at https://iplab.dmi.unict.it/ENIGMA-360. |
| 2026-03-10 | [Ensuring Data Freshness in Multi-Rate Task Chains Scheduling](http://arxiv.org/abs/2603.09738v1) | José Luis Conradi Hoffmann, Antônio Augusto Fröhlich | In safety-critical autonomous systems, data freshness presents a fundamental design challenge. While the Logical Execution Time (LET) paradigm ensures compositional determinism, it often does so at the cost of injected latency, degrading the phase margin of high-frequency control loops. Furthermore, mapping heterogeneous, multi-rate sensor fusion requirements onto rigid task-centric schedules typically implies in resource-inefficient oversampling. This paper proposes a Task-based scheduling framework extended with data freshness constraints. Unlike traditional models, scheduling decisions are driven by the lifespan of data. We introduce task offset based on the data freshness constraint to order data production in a Just-in-Time (JIT) fashion: the completion of the production of data with strictest data freshness constraint is delayed to the instant its consumers will be ready to use it. This allows for flexible task release offsets. We introduce a formal methodology to decompose Data Dependency Graphs into Dominant Paths by tracing the strictest data freshness constraints backward from the actuators. Based on this decomposition, we propose a Consensus Offset Search algorithm that synchronizes shared producers and private predecessors. This approach enforces end-to-end data freshness without the artificial latency of LET buffering. We formally prove that this offset-based alignment preserves the 100\% schedulability capacity of Global EDF, ensuring data freshness while eliminating the computational overhead of redundant sampling. |
| 2026-03-10 | [$M^2$-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](http://arxiv.org/abs/2603.09737v1) | Kaixin Lin, Kunyu Peng et al. | Semantic occupancy prediction enables dense 3D geometric and semantic understanding for autonomous driving. However, existing camera-based approaches implicitly assume complete surround-view observations, an assumption that rarely holds in real-world deployment due to occlusion, hardware malfunction, or communication failures. We study semantic occupancy prediction under incomplete multi-camera inputs and introduce $M^2$-Occ, a framework designed to preserve geometric structure and semantic coherence when views are missing. $M^2$-Occ addresses two complementary challenges. First, a Multi-view Masked Reconstruction (MMR) module leverages the spatial overlap among neighboring cameras to recover missing-view representations directly in the feature space. Second, a Feature Memory Module (FMM) introduces a learnable memory bank that stores class-level semantic prototypes. By retrieving and integrating these global priors, the FMM refines ambiguous voxel features, ensuring semantic consistency even when observational evidence is incomplete. We introduce a systematic missing-view evaluation protocol on the nuScenes-based SurroundOcc benchmark, encompassing both deterministic single-view failures and stochastic multi-view dropout scenarios. Under the safety-critical missing back-view setting, $M^2$-Occ improves the IoU by 4.93%. As the number of missing cameras increases, the robustness gap further widens; for instance, under the setting with five missing views, our method boosts the IoU by 5.01%. These gains are achieved without compromising full-view performance. The source code will be publicly released at https://github.com/qixi7up/M2-Occ. |
| 2026-03-10 | [OOD-MMSafe: Advancing MLLM Safety from Harmful Intent to Hidden Consequences](http://arxiv.org/abs/2603.09706v1) | Ming Wen, Kun Yang et al. | While safety alignment for Multimodal Large Language Models (MLLMs) has gained significant attention, current paradigms primarily target malicious intent or situational violations. We propose shifting the safety frontier toward consequence-driven safety, a paradigm essential for the robust deployment of autonomous and embodied agents. To formalize this shift, we introduce OOD-MMSafe, a benchmark comprising 455 curated query-image pairs designed to evaluate a model's ability to identify latent hazards within context-dependent causal chains. Our analysis reveals a pervasive causal blindness among frontier models, with the highest 67.5% failure rate in high-capacity closed-source models, and identifies a preference ceiling where static alignment yields format-centric failures rather than improved safety reasoning as model capacity grows. To address these bottlenecks, we develop the Consequence-Aware Safety Policy Optimization (CASPO) framework, which integrates the model's intrinsic reasoning as a dynamic reference for token-level self-distillation rewards. Experimental results demonstrate that CASPO significantly enhances consequence projection, reducing the failure ratio of risk identification to 7.3% for Qwen2.5-VL-7B and 5.7% for Qwen3-VL-4B while maintaining overall effectiveness. |
| 2026-03-10 | [MM-tau-p$^2$: Persona-Adaptive Prompting for Robust Multi-Modal Agent Evaluation in Dual-Control Settings](http://arxiv.org/abs/2603.09643v1) | Anupam Purwar, Aditya Choudhary | Current evaluation frameworks and benchmarks for LLM powered agents focus on text chat driven agents, these frameworks do not expose the persona of user to the agent, thus operating in a user agnostic environment. Importantly, in customer experience management domain, the agent's behaviour evolves as the agent learns about user personality. With proliferation of real time TTS and multi-modal language models, LLM based agents are gradually going to become multi-modal. Towards this, we propose the MM-tau-p$^2$ benchmark with metrics for evaluating the robustness of multi-modal agents in dual control setting with and without persona adaption of user, while also taking user inputs in the planning process to resolve a user query. In particular, our work shows that even with state of-the-art frontier LLMs like GPT-5, GPT 4.1, there are additional considerations measured using metrics viz. multi-modal robustness, turn overhead while introducing multi-modality into LLM based agents. Overall, MM-tau-p$^2$ builds on our prior work FOCAL and provides a holistic way of evaluating multi-modal agents in an automated way by introducing 12 novel metrics. We also provide estimates of these metrics on the telecom and retail domains by using the LLM-as-judge approach using carefully crafted prompts with well defined rubrics for evaluating each conversation. |
| 2026-03-10 | [Towards Terrain-Aware Safe Locomotion for Quadrupedal Robots Using Proprioceptive Sensing](http://arxiv.org/abs/2603.09585v1) | Peiyu Yang, Jiatao Ding et al. | Achieving safe quadrupedal locomotion in real-world environments has attracted much attention in recent years. When walking over uneven terrain, achieving reliable estimation and realising safety-critical control based on the obtained information is still an open question. To address this challenge, especially for low-cost robots equipped solely with proprioceptive sensors (e.g., IMUs, joint encoders, and contact force sensors), this work first presents an estimation framework that generates a 2.5-D terrain map and extracts support plane parameters, which are then integrated into contact and state estimation. Then, we integrate this estimation framework into a safety-critical control pipeline by formulating control barrier functions that provide rigorous safety guarantees. Experiments demonstrate that the proposed terrain estimation method provides smooth terrain representations. Moreover, the coupled estimation framework of terrain, state, and contact reduces the mean absolute error of base position estimation by 64.8%, decreases the estimation variance by 47.2%, and improves the robustness of contact estimation compared to a decoupled framework. The terrain-informed CBFs integrate historical terrain information and current proprioceptive measurements to ensure global safety by keeping the robot out of hazardous areas and local safety by preventing body-terrain collision, relying solely on proprioceptive sensing. |
| 2026-03-10 | [Compartmentalization-Aware Automated Program Repair](http://arxiv.org/abs/2603.09544v1) | Jia Hu, Youcheng Sun et al. | Software compartmentalization breaks down an application into compartments isolated from each other: an attacker taking over a compartment will be confined to it, limiting the damage they can cause to the rest of the application. Despite the security promises of this approach, recent studies have shown that most existing compartmentalized software is plagued by vulnerabilities at cross-compartment interfaces, allowing an attacker taking over a compartment to escape its confinement and negate the security guarantees expected from compartmentalization. In that context, securing cross-compartment interfaces is notoriously difficult and engineering-intensive.   In light of recent advances in Automated Program Repair (APR), notably through the use of Large Language Models (LLMs), this paper presents a work in progress investigating the suitability of LLM-based APR at securing cross-compartment interfaces as automatically as possible. We observe that existing APR approaches and general purpose/code-centric LLMs used as is are unfit for this task, and present the design, implementation, and early results of a new APR framework dedicated to compartment interface safety. The framework integrates into a feedback loop 1) a specialized fuzzer uncovering cross-compartment interface vulnerabilities; 2) a patch generation component bridging the lack of compartmentalization awareness of existing LLMs with a series of analysis techniques; and 3) a patch validation component assessing the effectiveness of generated vulnerability fixes. We validate our framework over a sample interface vulnerability, comparing it to a naive use of general-purpose LLMs, and discuss future research avenues. |
| 2026-03-10 | [What Do We Care About in Bandits with Noncompliance? BRACE: Bandits with Recommendations, Abstention, and Certified Effects](http://arxiv.org/abs/2603.09532v1) | Nicolás Della Penna | Bandits with noncompliance separate the learner's recommendation from the treatment actually delivered, so the learning target itself must be chosen. A platform may care about recommendation welfare in the current mediated workflow, treatment learning for a future direct-control regime, or anytime-valid uncertainty for one of those targets. These objectives need not agree. We formalize this objective-choice problem, identify the direct-control regime in which recommendation and treatment objectives collapse, and show by example that recommendation welfare can strictly exceed every learner-measurable treatment policy when downstream actors use private information. For finite-context square-IV problems we propose BRACE, a parameter-free phase-doubling algorithm that performs IV inversion only after matrix certification and otherwise returns full-range but honest structural intervals. BRACE delivers simultaneous policy-value validity, fixed-gap identification of the operationally optimal recommendation policy, and fixed-gap identification of the structurally optimal treatment policy under contextual homogeneity and invertibility. We complement the theory with a finite-context empirical benchmark spanning direct control, mediated present-versus-future tradeoffs, weak identification, homogeneity failure, and rectangular overidentification. The experiments show that safety appears as regret on easy problems, as abstention and wide valid intervals under weak identification, as a reason to prefer recommendation welfare under homogeneity failure, and as tighter structural uncertainty when extra instruments are available. For rich contexts, we also derive an orthogonal score whose conditional bias factorizes into compliance-model and outcome-model errors, clarifying what must be stabilized for anytime-valid semiparametric IV inference. |
| 2026-03-10 | [RESBev: Making BEV Perception More Robust](http://arxiv.org/abs/2603.09529v1) | Lifeng Zhuo, Kefan Jin et al. | Bird's-eye-view (BEV) perception has emerged as a cornerstone of autonomous driving systems, providing a structured, ego-centric representation critical for downstream planning and control. However, real-world deployment faces challenges from sensor degradation and adversarial attacks, which can cause severe perceptual anomalies and ultimately compromise the safety of autonomous driving systems. To address this, we propose a resilient and plug-and-play BEV perception method, RESBev, which can be easily applied to existing BEV perception methods to enhance their robustness to diverse disturbances. Specifically, we reframe perception robustness as a latent semantic prediction problem. A latent world model is constructed to extract spatiotemporal correlations across sequential BEV observations, thereby learning the underlying BEV state transitions to predict clean BEV features for reconstructing corrupted observations. The proposed framework operates at the semantic feature level of the Lift-Splat-Shoot pipeline, enabling recovery that generalizes across both natural disturbances and adversarial attacks without modifying the underlying backbone. Extensive experiments on the nuScenes dataset demonstrate that, with few-shot fine-tuning, RESBev significantly improves the robustness of existing BEV perception models against various external disturbances and adversarial attacks. |
| 2026-03-10 | [TopoOR: A Unified Topological Scene Representation for the Operating Room](http://arxiv.org/abs/2603.09466v1) | Tony Danjun Wang, Ka Young Kim et al. | Surgical Scene Graphs abstract the complexity of surgical operating rooms (OR) into a structure of entities and their relations, but existing paradigms suffer from strictly dyadic structural limitations. Frameworks that predominantly rely on pairwise message passing or tokenized sequences flatten the manifold geometry inherent to relational structures and lose structure in the process. We introduce TopoOR, a new paradigm that models multimodal operating rooms as a higher-order structure, innately preserving pairwise and group relationships. By lifting interactions between entities into higher-order topological cells, TopoOR natively models complex dynamics and multimodality present in the OR. This topological representation subsumes traditional scene graphs, thereby offering strictly greater expressivity. We also propose a higher-order attention mechanism that explicitly preserves manifold structure and modality-specific features throughout hierarchical relational attention. In this way, we circumvent combining 3D geometry, audio, and robot kinematics into a single joint latent representation, preserving the precise multimodal structure required for safety-critical reasoning, unlike existing methods. Extensive experiments demonstrate that our approach outperforms traditional graph and LLM-based baselines across sterility breach detection, robot phase prediction, and next-action anticipation |
| 2026-03-10 | [SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments](http://arxiv.org/abs/2603.09460v1) | Shiyi Chen, Mingye Yang et al. | Efficiently training quadruped robot navigation in densely cluttered environments remains a significant challenge. Existing methods are either limited by a lack of safety and agility in simple obstacle distributions or suffer from slow locomotion in complex environments, often requiring excessively long training phases. To this end, we propose SEA-Nav (Safe, Efficient, and Agile Navigation), a reinforcement learning framework for quadruped navigation. Within diverse and dense obstacle environments, a differentiable control barrier function (CBF)-based shield constraints the navigation policy to output safe velocity commands. An adaptive collision replay mechanism and hazardous exploration rewards are introduced to increase the probability of learning from critical experiences, guiding efficient exploration and exploitation. Finally, kinematic action constraints are incorporated to ensure safe velocity commands, facilitating successful physical deployment. To the best of our knowledge, this is the first approach that achieves highly challenging quadruped navigation in the real world with minute-level training time. |
| 2026-03-10 | [Platooning as a Service (PlaaS): A Sustainable Transportation Framework for Connected and Autonomous Vehicles](http://arxiv.org/abs/2603.09256v1) | Bhosale Akshay Tanaji, Sayak Roychowdhury et al. | Vehicular platooning is a well-known transportation technique that helps reduce fuel consumption, carbon emissions, and road congestion. When integrated with Connected and Autonomous Vehicle (CAV) technologies, platooning enhances the overall safety and efficiency of the transportation system. This article presents Platooning as a Service (PlaaS) platform, a decision-support framework to promote sustainable transportation through platooning. We have formulated this problem as a Stackelberg game, with the platoon service provider (PSP) as the leader and the service users as the followers. The PSP sets the pricing policy, and the follower responds by choosing the distance to be travelled with the platoon. The optimal service contract between the PSP and the follower vehicle is established with the Stackelberg equilibrium, which is derived using Karush-Kuhn-Tucker optimality conditions. Additionally, we have examined the impact of government subsidies on the PlaaS platform in reducing carbon emissions. Our model has been applied to a specific example problem to illustrate the benefits for both players. We have also derived managerial insights through sensitivity analysis, exploring the effect of different velocity levels, vehicle dimensions, government subsidies, and operational urgency on the utilities of the players and CO2 emissions. Our analysis shows that PSP gains a higher profit from high delay-cost vehicles performing time-critical operations with higher platoon velocity. However, the benefits related to fuel consumption are only realized at moderate platoon velocities. |
| 2026-03-10 | [Safe or Slow? The Illusion of Thermal Stability Under Reduced-Velocity Nail Intrusion](http://arxiv.org/abs/2603.09254v1) | Eymen Ipek, Oliver Korak et al. | This study investigates the effects of nail penetration speed on the safety outcomes of large-format automotive lithium-ion pouch cells. Through six controlled tests varying the speed of nail insertion, we observed that lower penetration speeds did not induce thermal runaway; instead, the cells exhibited self-discharge while the nail remained embedded. These findings suggest that penetration speed is a critical factor in the onset of thermal runaway, providing valuable insights for the development of safer battery systems and more effective safety testing protocols. |
| 2026-03-10 | [Reasoning-Oriented Programming: Chaining Semantic Gadgets to Jailbreak Large Vision Language Models](http://arxiv.org/abs/2603.09246v1) | Quanchen Zou, Moyang Chen et al. | Large Vision-Language Models (LVLMs) undergo safety alignment to suppress harmful content. However, current defenses predominantly target explicit malicious patterns in the input representation, often overlooking the vulnerabilities inherent in compositional reasoning. In this paper, we identify a systemic flaw where LVLMs can be induced to synthesize harmful logic from benign premises. We formalize this attack paradigm as \textit{Reasoning-Oriented Programming}, drawing a structural analogy to Return-Oriented Programming in systems security. Just as ROP circumvents memory protections by chaining benign instruction sequences, our approach exploits the model's instruction-following capability to orchestrate a semantic collision of orthogonal benign inputs. We instantiate this paradigm via \tool{}, an automated framework that optimizes for \textit{semantic orthogonality} and \textit{spatial isolation}. By generating visual gadgets that are semantically decoupled from the harmful intent and arranging them to prevent premature feature fusion, \tool{} forces the malicious logic to emerge only during the late-stage reasoning process. This effectively bypasses perception-level alignment. We evaluate \tool{} on SafeBench and MM-SafetyBench across 7 state-of-the-art 0.LVLMs, including GPT-4o and Claude 3.7 Sonnet. Our results demonstrate that \tool{} consistently circumvents safety alignment, outperforming the strongest existing baseline by an average of 4.67\% on open-source models and 9.50\% on commercial models. |
| 2026-03-10 | [HelixTrack: Event-Based Tracking and RPM Estimation of Propeller-like Objects](http://arxiv.org/abs/2603.09235v1) | Radim Spetlik, Michal Pliska et al. | Safety-critical perception for unmanned aerial vehicles and rotating machinery requires microsecond-latency tracking of fast, periodic motion under egomotion and strong distractors. Frame-based and event-based trackers drift or break on propellers because periodic signatures violate their smooth-motion assumptions. We tackle this gap with HelixTrack, a fully event-driven method that jointly tracks propeller-like objects and estimates their rotations per minute (RPM). Incoming events are back-warped from the image plane into the rotor plane via a homography estimated on the fly. A Kalman Filter maintains instantaneous estimates of phase. Batched iterative updates refine the object pose by coupling phase residuals to geometry. To our knowledge, no public dataset targets joint tracking and RPM estimation of propeller-like objects. We therefore introduce the Timestamped Quadcopter with Egomotion (TQE) dataset with 13 high-resolution event sequences, containing 52 rotating objects in total, captured at distances of 2 m / 4 m, with increasing egomotion and microsecond RPM ground truth. On TQE, HelixTrack processes full-rate events (approx. 11.8x real time) faster than real time and microsecond latency. It consistently outperforms per-event and aggregation-based baselines adapted for RPM estimation. |

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



