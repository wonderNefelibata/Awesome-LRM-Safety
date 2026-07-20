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
| 2026-07-17 | [FVAttn: Adaptive Sparse Attention with Runtime Load Balancing for Video Generation](http://arxiv.org/abs/2607.16190v1) | Hao Liu, Chenghuan Huang et al. | Video Diffusion Transformers process long spatio-temporal sequences, making self-attention the main bottleneck in high-resolution video generation. Training-free sparse attention reduces this cost, but adaptive Top-$p$ routing creates uneven per-head workloads under multi-GPU sequence parallelism. The resulting workload heterogeneity turns sparse attention into a rank-level straggler problem. We present \method{}, a training-free sparse-attention system that improves the distributed execution efficiency of adaptive sparse attention under multi-GPU sequence parallelism. \method{} uses Top-$p$ routing, a Top-$k$ safety floor, and video-aware block organization as the sparse-routing frontend, then repairs the materialized mask at runtime. Runtime Load Balancing migrates a small number of heavy heads via P2P communication to shorten the current critical path. Slack-Aware Sparse Augmentation fills residual non-critical-rank slack with additional high-value blocks, while overlap hides scheduling and migration overhead behind existing computation. On step-distilled Wan2.2 I2V, \method{} reduces average load imbalance from 1.34 to 1.08 and delivers a $4.41\times$ attention speedup over FlashAttention, while achieving a $2.02$--$2.11\times$ DiT inference speedup with competitive video quality. |
| 2026-07-17 | [Vision-Language Assistant for Emotional Reactions to Risky Driving](http://arxiv.org/abs/2607.16181v1) | Harine Choi, Eun Hak Lee et al. | This study introduces a vision-language pipeline that detects risky driving behaviors and generates emotionally expressive responses to support driver awareness and comfort. Although vision-language models have advanced perception and reasoning in autonomous driving, existing systems rarely consider the emotional dimension or real-world user experience. Keep Yelling Assistant (KYA) detects high-risk driving maneuvers in real time, such as sudden cut-ins. It then produces emotional responses through a large language model tailored to driver preferences. The framework comprises two core modules. The vision module uses YOLOv8 variants to detect nearby vehicles and identify risky behaviors such as sudden cut-ins. Key driving metrics, including relative distance, speed, and projected reach time, are extracted and normalized to produce a structured behavior log. The language module processes this log with user-defined emotional tone settings, such as neutral, humorous, and analytical, and generates verbal reactions using state-of-the-art large language models, including ChatGPT-4o, Claude 3, Gemini 2.5, and Copilot. We evaluated the proposed system using dashcam videos containing risky driving behaviors and a user study involving 108 participants. Participants selected preferred response styles, and the large language models were evaluated based on emotional alignment. All models received favorable ratings, although preferences varied across personas. Notably, the combination of YOLOv8s and ChatGPT-4o achieved the highest score of 4.29 out of 5.00. By integrating real-world perception with emotionally adaptive dialogue, KYA introduces a new paradigm for emotionally intelligent in-vehicle artificial intelligence. It offers promising directions for improving safety, trust, and emotional well-being in both conventional and autonomous vehicles. |
| 2026-07-17 | [PRISA: Proactive Infrastructure LiDAR Framework for Intersection Safety Assessment](http://arxiv.org/abs/2607.16156v1) | Tam Bang, Hussam Abubakr et al. | Urban intersections are among the most hazardous locations in road networks, posing significant risks to vehicles and vulnerable road users (VRUs) such as pedestrians and cyclists. The complexity of multi-agent interactions demands continuous, real-time monitoring systems capable of anticipating conflicts before they escalate into crashes. We present PRISA, a modular infrastructure LiDAR framework leveraging privacy-preserving, low-light-robust roadside sensors for long-term traffic observation and real-time risk detection at the edge. The framework comprises two core components: a sensing and perception layer and a plug-and-play risk assessment module. The latter automatically curates site-specific training data from accumulated perception outputs to train a trajectory prediction model without manual annotation. It then deploys the trained model for continuous motion forecasting and dual surrogate safety evaluation, using Time-to-Collision (TTC) for longitudinal conflicts and Predicted Post-Encroachment Time (PPET) for crossing and VRU-involved interactions. PRISA is evaluated on the public R-LiViT dataset and deployed on an NVIDIA Jetson AGX Thor at a live signalized intersection in Chattanooga, Tennessee. PPET-based assessment operates at 194~ms end-to-end latency over a 2.4-second predictive horizon, with TTC-based detection and perception remaining within real-time constraints, demonstrating practical feasibility for proactive multi-agent intersection safety monitoring. |
| 2026-07-17 | [CLIFE: Camera-LiDAR Fusion Framework for Edge-Deployable Roadside VRU Perception](http://arxiv.org/abs/2607.16154v1) | Tam Bang, Hoang H. Nguyen et al. | Reliable roadside perception of vulnerable road users (VRUs) remains challenging under occlusions, variable lighting, and diverse weather conditions, particularly under strict edge-computing and latency constraints. Existing multi-sensor fusion systems rely on cloud or server-grade infrastructure, creating a deployment gap at real-world intersections. We present CLIFE, an edge-native camera-LiDAR fusion framework that integrates targetless online calibration and lightweight late-fusion tracking entirely on a single embedded device, without cloud offloading. CLIFE adaptively refines camera-LiDAR alignment on demand and performs multi-sensor fusion and track association with O(N log N) per-frame cost. We deploy CLIFE across 12 signalized intersections in Chattanooga and conduct an in-depth evaluation at a representative intersection using synchronized camera-LiDAR data that spans diverse daytime, nighttime, and weather conditions. Our experiments demonstrate that the fusion architecture substantially enhances the perceptual range and robustness of the individual sensors under varied environmental and traffic conditions. The late-fusion core operates at 53.2 FPS on the Jetson AGX Thor, ensuring high throughput for real-time intersection-scale applications. By centering perception at the edge, CLIFE provides a deployable foundation for downstream safety applications, while reducing bandwidth and calibration overhead for agencies operating multi-intersection corridors. |
| 2026-07-17 | [Harmonizing AI Safety Thresholds](http://arxiv.org/abs/2607.16112v1) | Wilber Sean Anterola, Matthew Ball et al. | Frontier AI companies have published capability thresholds that differ substantially, making it difficult for third parties to verify whether a threshold has been crossed or to compare requirements across companies. Moreover, without common minimum thresholds, risk mitigation may be inconsistent, creating a potential race to the bottom in safety standards. We develop a methodology for deriving harmonized thresholds across three risk domains. For misuse risks (cyber and biological), we take expected harm as the key primitive and use an explicit risk-modeling approach that accounts for risk channels and model release conditions. For automated AI R&D, we base our proposed threshold on the observed rate of AI progress rather than expected harm. Our analysis expands upon prior work and highlights existing empirical gaps and limitations. |
| 2026-07-17 | [The Honest Quorum Problem: Epistemic Byzantine Fault Tolerance for Agentic Infrastructure](http://arxiv.org/abs/2607.16109v1) | Jun He, Deying Yu | State machine replication (SMR) and Byzantine fault-tolerant (BFT) consensus guarantee agreement despite a bounded number of arbitrary, colluding faulty participants. However, these guarantees rely on participants outside this set correctly executing the protocol's transition semantics. Agentic validators expose a weaker boundary: an authenticated, responsive, non-equivocating, and protocol-compliant reasoning participant may still endorse a semantically invalid transition due to reasoning errors.   We call this failure mode an epistemic fault, and the collective phenomenon the Honest Quorum Problem (where "honest" means protocol-compliant, not semantically correct). Such a quorum can satisfy ordinary checks while forming a certificate for an invalid transition. Thus, agreement alone does not guarantee semantic validity or execution safety. Furthermore, because agentic validators often share model weights, training distributions, prompts, or toolchains, they are highly susceptible to correlated epistemic faults.   We define Epistemic Byzantine Fault Tolerance (EBFT), a fault-tolerance model for agentic infrastructure and post-deterministic distributed systems. EBFT augments the conventional Byzantine fault bound with two separate, confidence-indexed quantities: $e_δ$ bounds coherent invalid endorsements outside the Byzantine set, and $u_ε$ bounds unusable validator support that degrades liveness. These quantities characterize semantic safety risk and liveness degradation independently. We derive quorum-threshold conditions for semantic validity, consensus agreement, liveness, and feasible threshold selection, and outline a calibration methodology for estimating these budgets. We show that adding nominally distinct agents improves fault tolerance only when it measurably reduces the upper-tail concentration of invalid endorsements or unusable support. |
| 2026-07-17 | [Closing the AI Trust Gap: The Case for Independent Certification for Trustworthy AI](http://arxiv.org/abs/2607.15992v1) | Trisevgeni Papakonstantinou, Cansu Canca et al. | Over the past decade, responsible AI (RAI) has produced a substantial body of practice for identifying and mitigating the risks AI poses in high-stakes settings. Yet this work has not produced a market that rewards trustworthiness. Firms that invest seriously in safety, fairness, and oversight cannot consistently prove to consumers, regulators, and shareholders that their systems go beyond the bare minimum of compliance. What is missing is a way for society to recognize or compare the difference. The result is a trust gap: a structural condition in which responsible development efforts happen inside organizations but produce no external, independently recognized and verifiable signal of trustworthy outcomes. We argue this gap is sustained in part because of a focus on responsible AI (a matter of internal process) as opposed to trustworthy AI (a matter of independently verifiable real-world outcomes), and that it persists because of three compounding failures: (1) the market cannot distinguish trustworthy systems from their imitations; (2) evaluation targets models and outputs rather than deployed sociotechnical systems and their outcomes; (3) the measurement ecosystem is oriented toward avoiding harm rather than demonstrating benefit. Reviewing existing AI governance instruments and comparing them to certification regimes in healthcare, sustainability, and security, we show that none integrate a governance baseline, independently verified positive-outcome evidence, and market signaling in a single framework. We propose independent, outcome-oriented certification as the connective layer that can close the trust gap, complementing regulation and internal governance by making trustworthiness measurable, comparable, and commercially rewarded. |
| 2026-07-17 | [Refusal is Not Safety! Benchmarking Latent Safety Risks of LLM-Driven Content Humorization](http://arxiv.org/abs/2607.15977v1) | Yu Cui, Ruiqing Yue et al. | Safety defenses for large language models (LLMs) have been extensively studied, with existing approaches focusing on attack detection and refusal mechanisms. Such fixed-form direct refusal strategies may introduce the risk of prefix injection attacks. Recent work has explored a new direction that leverages humor as an indirect refusal mechanism to mitigate over-refusal in jailbreak scenarios and reduce prefix injection risks. However, this approach implicitly assumes that humorous responses are safe. Whether humorization itself introduces safety risks remains unexplored. To address this issue, we conduct an exploratory study involving over 30,000 real-world agent interaction records and 45 stand-up comedians, revealing practical safety concerns in LLM-based content humorization. Motivated by these findings, we propose \textsc{HumorSafe}, a novel framework for evaluating latent safety risk propagation during humorization. \textsc{HumorSafe} enables LLMs to learn harmful humorization patterns and use them to transform benign content into humorous content with safety risks. Across five frontier LLMs, we find that LLMs can introduce stereotypes and toxicity during humorization. We further propose \textsc{HumorPIA}, a prompt injection attack that exploits latent risks in humor-based defenses. \textsc{HumorPIA} preserves the appearance of safe humorous refusal while covertly injecting harmful content, allowing latent risks to evade existing detection mechanisms. Experiments show that it increases toxicity by 3.14$\times$ while maintaining an apparent safety rate of 97.8\% even under defense settings. Our findings highlight a gap in existing LLM safety evaluations under humorized settings. |
| 2026-07-17 | [Equilibrium analysis in a multi-agent reinsurance chain](http://arxiv.org/abs/2607.15962v1) | Kaizheng Wang, Wei Liu et al. | This paper investigates a multi-layer reinsurance chain within a stochastic differential game framework involving m competing insurers and n reinsurers. Specifically, Stackelberg differential games are employed to characterize the strategic interactions between reinsurance buyers and sellers at each layer of the chain. In addition, a non-zero-sum game model is established to capture the competitive behavior among insurers. Both insurers and reinsurers are allowed to invest in a risk-free asset and a risky asset. To examine the heterogeneity of reinsurance chains under different contract types, the analysis is conducted separately for proportional reinsurance and excess-of-loss reinsurance. By combining dynamic programming and game theory, closed-form equilibrium strategies for investment and reinsurance are derived by solving the extended Hamilton-Jacobi-Bellman (HJB) systems under the mean-variance (MV) criterion. Numerical analysis is conducted to explore the impact of key parameters on the equilibrium strategies. The results indicate that intensified competition in the insurance market leads to a reduction in the safety loadings of reinsurance contracts at each layer of the reinsurance chain. |
| 2026-07-17 | [Dynamic Constraint Reconstruction Based Control Barrier Functions for Safety-Critical Control of High-Dimensional Manipulators](http://arxiv.org/abs/2607.15961v1) | Bingsheng Zhang, Shen Wang et al. | Control barrier functions (CBFs) provide formal safety guarantees for constrained nonlinear systems, but their effectiveness relies on accurate system dynamics. In high-dimensional manipulators subject to unknown disturbances and model uncertainties, fixed safety constraints constructed from nominal dynamics may become inconsistent with the actual system behavior, leading to safety degradation or excessive conservatism. This paper proposes a dynamic constraint reconstruction based control barrier function (DCR-CBF) framework for safety-critical control of disturbed robotic manipulators. An extended state observer is employed to estimate lumped disturbances online, and the estimated disturbance is incorporated into high-order control barrier functions to reconstruct safety constraints according to the estimated true dynamics. To address estimation inaccuracies, a safety margin is introduced, and a sufficient condition is derived to guarantee forward invariance under bounded estimation errors. Simulation studies on a 4-DOF excavation manipulator demonstrate that the proposed DCR-CBF method achieves zero safety violation under strong unknown disturbances while significantly improving trajectory-tracking performance compared with standard and robust CBF methods. |
| 2026-07-17 | [Trans-Domain Digital Twin: Conceptual Foundations, Architecture, and Research Outlook](http://arxiv.org/abs/2607.15908v1) | Mansoorali Amiri | Complex systems comprise heterogeneous domains whose states, uncertainties, risks, and control consequences can cross domain boundaries. Existing cross-domain digital twin approaches broadly focus on comparison, reuse, semantic mapping, standardization, and interoperability, but do not inherently require operational connections among domain states, errors, objectives, constraints, decisions, and controls. This article proposes the trans-domain digital twin as an operational formulation along the continuum of Composite/Federated Digital Twin Systems. This approach connects heterogeneous domain twins through an aligned shared state, explicit coupling of data, models, states, errors, objectives, and controls, heterogeneous temporal coordination, joint decision-making, and feedback-based adaptation. The proposed framework presents a seven-layer conceptual architecture, a trans-domain orchestration core, minimum compliance conditions, a general operational formalism, progressive fast-meso-slow loops, and a single-episode offline training mechanism linked to bounded online adaptation. It also describes conceptual validation and evaluation criteria, a maturity model, a reference deployment architecture, and requirements for runtime safety, provenance, versioning, and model lifecycle management. The framework is conceptually mappable to standards for digital twins, model exchange, distributed simulation, and smart transducers; however, its formal compliance and operational effectiveness must be examined through independent benchmarks, uncertainty quantification, ablation testing, and field validation. |
| 2026-07-17 | [Red Light, Grey Zone: A Multi-Perspective Interactive Narrative for Autonomous Driving Ethics](http://arxiv.org/abs/2607.15888v1) | Mengyi Wei, Nianhua Liu et al. | Autonomous driving ethics is not only an expert concern, but also a public issue involving risk, responsibility, and governance. However, non-experts often struggle to interpret these issues in concrete incidents, especially when responsibility is distributed across multiple stakeholders. This paper investigates interactive narrative as a public-facing method for eliciting situated ethical reflection on autonomous driving. We present Red Light, Grey Zone, a web-based, multi-perspective interactive narrative prototype inspired by a real-world autonomous-driving incident. The prototype invites participants to compare stakeholder perspectives, examine scene materials, and make responsibility judgments in the face of ethical ambiguity. We report an exploratory user study (N=12) examining how differently non-experts responded to the prototype. Our analysis focuses on three dimensions of reflection: ethical cognition, responsibility-focused critical thinking, and multi-perspective reasoning. Exploratory pre-post results showed the strongest self-reported shift in responsibility-focused critical thinking among participants who completed the intended stakeholder-comparison process, while ethical cognition and multi-perspective reasoning showed positive directional trends. Qualitative findings further show how participants reflected on safety and market trade-offs, responsibility ambiguity, transparency and privacy, and governance gaps. Participants also used stakeholder comparison to corroborate evidence and, in many cases, broaden responsibility judgments from single-actor blame toward more distributed interpretations of accountability. Overall, the study suggests that multi-perspective interactive narratives may support non-expert reflection on accountability, evidence, and governance in AI-enabled systems. |
| 2026-07-17 | [PriEco-DRL: Joint Optimization of Electric-Bus Eco-Driving and Transit-Priority Adaptive Signals via Deep Reinforcement Learning](http://arxiv.org/abs/2607.15862v1) | Dingshan Sun, Ang Li et al. | Urban transit electrification requires balancing energy efficiency, schedule reliability, and ride comfort for electric buses (EBs), particularly when interacting with transit-priority adaptive signals in congested networks. This paper proposes PriEco-DRL, a joint optimization framework that integrates EB eco-driving with transit-priority adaptive signal control using deep reinforcement learning (DRL). The signal layer employs a priority-weighted max-pressure (Priority-MP) controller to allocate green time based on occupancy-aware pressures, while the vehicle layer adapts longitudinal control based on uncertain and dynamically evolving local signal cues. A structured reward combines guidance and event-based reinforcement to align EB arrivals with green opportunities while considering energy, time, comfort, and safety. The framework uses centralized training and decentralized execution (CTDE) with parameter sharing, allowing a single DRL agent to learn from multiple buses and routes using local observations. Experiments on a real-world corridor show that PriEco-DRL reduces EB energy consumption while maintaining network efficiency and transit priority compared with fixed-time, actuated, and rule-based signal-vehicle coordination baselines. Energy- and trajectory-based analyses reveal that the improvements stem from fewer unscheduled stop-start events and smoother speed regulation under adaptive signals. The results highlight a tunable energy-time trade-off, allowing flexible operational choices through reward weighting. |
| 2026-07-17 | [Converging Safety and Security: IO-Link Wireless and OPC UA over 5G under prEN 50742](http://arxiv.org/abs/2607.15840v1) | Henry Beuster, Thomas Robert Doebbert et al. | The integration of wireless communication technologies in industrial automation offers greater flexibility, but also exposes safety systems to a broader threat vector. Emerging regulations, such as the draft standard prEN 50742, mandate the convergence of functional safety and cybersecurity by requiring cryptographic security mechanisms directly in safety-critical communication. This paper presents an empirical evaluation of this safety-security convergence across a complete control chain, spanning from an IO-Link Wireless Safety device to a PLC via an OPC UA backbone. We measure the latencies and jitter of different Safety-Related Security Levels under prEN 50742 over Ethernet, Wi-Fi 6, and private 5G. Our results reveal that while cryptographic execution time is negligible, the resulting frame payload expansion severely restricts wireless fieldbus capacity, reducing the maximum number of devices per IO-Link Wireless track from 8 to 2. Furthermore, we demonstrate that, despite higher average latency, a private 5G provides sufficiently deterministic latency characteristics to preserve functional safety watchdog margins, unlike unlicensed Wi-Fi 6. |
| 2026-07-17 | [In the Driver's Seat: A Multi-Company Study on the Reality of Autonomous Driving System Testing](http://arxiv.org/abs/2607.15820v1) | Qunying Song, Yuan Gao et al. | Autonomous driving systems (ADS) are rapidly advancing and increasingly deployed in real-world applications. This creates growing demands for effective testing to ensure system functionality and safety. However, ADS testing remains complex and lacks well-established standards for scenario selection, performance evaluation, and acceptance criteria. To better understand current ADS testing practices and challenges, we conducted an interview study with experts working on ADS development and testing in nine companies from six different countries. Through thematic analysis, we synthesized industrial testing practices, challenges, potential solutions, future trends, and proposed an evidence-centered closed-loop testing framework for ADS testing. Our findings show that current practices primarily focus on scenario-based and X-in-the-loop testing approaches, supported by diverse tools, metrics, benchmarks, and testing strategies. The participants highlighted major challenges related to scenario realism, scenario coverage, simulation fidelity, and acceptance criteria, while also discussing potential solutions such as the use of AI, world models, and end-to-end approaches. Furthermore, participants envisioned future ADS testing to become more automated, data-driven, and transparent across the industry. Overall, this study provides a comprehensive industry-grounded overview of ADS testing, proposes an evidence-centered closed-loop testing framework to provide actionable guidance for ADS testing, and outlines important directions for future research and practice. |
| 2026-07-17 | [CoG-Guided Weight Correction for Fault-Tolerant Deep Neural Networks](http://arxiv.org/abs/2607.15753v1) | Bahram Parchekani, Samira Nazari et al. | Deep Neural Networks (DNNs) used in safety-critical applications are vulnerable to hardware and memory faults that corrupt network weights and degrade reliability. In this paper, we propose a Center of Gravity (CoG) guided weight correction method that restores faulty weights based on their spatial characteristics within each layer. The proposed approach detects and corrects weight faults using distance-aware correction rules, eliminating the need for retraining or architectural modification. The effectiveness of the proposed method in terms of the capability of tolerating hardware faults has been evaluated through performing fault injection at different Bit Error Rates (BERs).   Experiments on safety-critical LSTM-based Networks, including StageNet for disease progression tracking and MTFNet for cardiac anomaly detection, demonstrate fault tolerance improvements of up to 230x and 6.41x, respectively, at a BER of 10^{-3}, with negligible accuracy loss. When extended to Convolutional Neural Networks (CNNs), the method achieves up to 49.55x and 20.79x improvements under comparable fault conditions on ResNet-18 and VGG-16, respectively. To the best of our knowledge, this is the first work to apply the CoG concept to neural network weight tensors for enhancing model reliability. |
| 2026-07-17 | [Verified LLM-Driven Synthesis for Concept Design](http://arxiv.org/abs/2607.15718v1) | Alcino Cunha | Concept Design structures software systems around concepts: user-facing, self-contained units of functionality with a focused purpose. Concepts are composed into applications using synchronization rules called reactions, which specify how actions in one concept trigger actions in others. This paper first gives a formal semantics for concepts and reactions, enabling automatic verification of safety invariants in applications developed with this methodology. It then presents a CEGIS-style, LLM-driven synthesis procedure for generating reaction designs that satisfy such invariants. Because many different designs can satisfy the same invariant, we study two ways of steering synthesis toward the user's intended design: natural-language prompts and positive/negative scenarios. We also propose an LLM-driven scenario elicitation technique to support early design exploration. In an evaluation on three applications and twelve design variants using one LLM configuration, invariant-only synthesis reached verified designs quickly but often produced inconsistent designs across runs, some of which were implausible, showing that invariants alone underconstrain the design task. Scenario-guided synthesis recovered intended designs more consistently than natural-language prompting, although minimal scenarios can lead to overfitting. LLM-driven scenario elicitation, where the user classifies proposed scenarios rather than authoring them from scratch, recovered the intended designs in most variants when enough scenarios were elicited, but missed behaviors and non-determinism prevented reliable coverage in all cases. |
| 2026-07-17 | [Debris Evolution from Spacecraft Fragmentation in Earth-Moon Distant Retrograde Orbits](http://arxiv.org/abs/2607.15709v1) | Yuyan Wu, Peng Shu et al. | With the rapid surge in lunar exploration and the planned expansion of cislunar infrastructure, cislunar space has become a strategic focal point for global aerospace activities. This proliferation of spacecraft heightens the risk of fragmentation events, such as unintended explosions or orbital collisions which serve as the primary source of hazardous orbital debris. Given the potential threat these fragments pose to mission safety and long-term orbital sustainability, it is imperative to investigate their dynamical behavior within the Earth-Moon system. This study evaluates the dispersion of debris clouds following potential breakup events on Distant Retrograde Orbits (DROs) over a 30-day propagation period. The Circular Restricted Three-Body Problem (CR3BP) model is used to construct the reference orbits, while the NASA Standard Breakup Model is applied to simulate fragment generation at multiple locations along three DROs of varying sizes. These fragments are then propagated using the Bicircular Restricted Four-Body Problem (BCR4BP) for 30 days. To account for the variability of these events, multiple initial positions along each orbit are analyzed to capture a comprehensive range of post-explosion scenarios. Our analysis quantifies the fate of fragments within this window, specifically focusing on the escape mechanisms and the percentages of debris that either depart from the Earth-Moon gravitational sphere of influence or impact the lunar surface. Furthermore, we introduce an analytical approach to assess the potential collision risk to resident space objects operating within the vicinity of the parent orbit. The results provide insights into debris evolution and offer a foundation for developing safety guidelines for future cislunar activities. |
| 2026-07-17 | [RAVEN: Reinforcement-Adaptive Visibility-Graph Planning for Robust Humanoid Navigation with Collision-Free MPC](http://arxiv.org/abs/2607.15701v1) | Ruochen Hou, Shiqi Wang et al. | Humanoid navigation in dynamic environments requires long-horizon planning while respecting short-horizon dynamic and safety constraints. Classical visibility-graph planners combined with model predictive control (MPC) can efficiently generate collision-free trajectories, but their performance depends on manually tuned parameters and accurate system modeling. In real robotic systems, control delays, state-estimation noise, and locomotion uncertainties can cause overshoot and constraint violations even when the nominal path is geometrically optimal. We propose RAVEN, a hierarchical reinforcement learning (RL)-MPC framework for robust humanoid navigation. Unlike prior approaches that use learning to tune cost weights or replace planning entirely, RAVEN employs RL to adapt the geometric construction of a visibility-graph planner by modifying obstacle inflation and related graph parameters. By directly reshaping the free-space geometry, the learned planner alters the topology of the global path to compensate for delay and tracking imperfections. A collision-free MPC layer then tracks the planned trajectory while explicitly enforcing velocity bounds and obstacle-avoidance constraints. By training under realistic delays and observation noise, RAVEN learns planning adaptations that improve robustness while retaining explicit long-horizon geometric planning and constrained optimization, in contrast to end-to-end learning approaches. We evaluate RAVEN against a manually tuned visibility-graph MPC baseline and a pure RL navigation policy. Results demonstrate reduced overshoot near obstacles, improved robustness in narrow passages, and more reliable navigation under delay and noise. These findings indicate that reinforcement-adaptive graph construction combined with constrained MPC provides an effective and interpretable alternative to end-to-end learning for robust humanoid navigation. |
| 2026-07-17 | [MemoGuard: An Adaptive Runtime for Guarding Against Memory Traps in Communication-Limited Robot Navigation](http://arxiv.org/abs/2607.15589v1) | Rajat Bhattacharjya, Hyeonjong Ju et al. | Communication-limited robots in mission-critical scenarios such as disaster inspection and search-and-rescue must make reliable onboard decisions without access to remote operators or high-capacity reasoning services. Episodic memory reuse is an attractive low-cost fallback, but retrieval similarity does not guarantee execution validity, i.e., a retrieved action may match the current context yet be unsafe due to changed topology, insufficient battery margin, or unreliable prior outcomes. We call such high-similarity but execution-invalid episodes memory traps. This creates a safety-efficiency design space where similarity only reuse minimizes fallback cost but can be unsafe, while always invoking local reasoning improves safety at high computational and energy cost. This paper presents MemoGuard, a lightweight adaptive runtime that validates episodic memories against topology, resource, and outcome contracts before reuse, invoking fallback only when validation fails. In a graph-based corridor-inspection simulator, MemoGuard reduces battery safety violations by 76.6% over similarity-only top-1 reuse while reducing fallback calls by 21.4% over always reasoning. On an NVIDIA Jetson AGX Xavier with local llama3.2:3b fallback reasoning, this corresponds to 3.67 s and 36.97 J of avoided fallback-reasoning overhead per trial. We open-source MemoGuard at https://github.com/hetheiin/memoguard. |

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



