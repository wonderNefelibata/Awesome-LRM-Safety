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
| 2026-04-17 | [PolicyGapper: Automated Detection of Inconsistencies Between Google Play Data Safety Sections and Privacy Policies Using LLMs](http://arxiv.org/abs/2604.16128v1) | Luca Ferrari, Billel Habbati et al. | Mobile application developers are required to disclose how they collect, use, and share user data in compliance with privacy regulations. To support transparency, major app marketplaces have introduced standardized disclosure mechanisms. In 2022, Google mandated the Data Safety Section (DSS) on Google Play, requiring developers to summarize their data practices. However, compiling accurate DSS disclosures is challenging, as they must remain consistent with the corresponding privacy policy (PP), and no automated tool currently verifies this alignment. Prior studies indicate that nearly 80% of popular apps contain incomplete or misleading DSS declarations. We present PolicyGapper, an LLM-based methodology for automatically detecting discrepancies between DSS disclosures and privacy policies. PolicyGapper operates in four stages: scraping, pre-processing, analysis, and post-processing, without requiring access to application binaries. We evaluate PolicyGapper on a dataset of 330 top-ranked apps spanning all 33 Google Play categories, collected in Q3 2025. The approach identifies 2,689 omitted disclosures, including 2,040 related to data collection and 649 to data sharing. Manual validation on a stratified 10% subset, repeated across three independent runs, yields an average Precision of 0.75, Recall of 0.77, Accuracy of 0.69, and F1-score of 0.76. To support reproducibility, we release a complete replication package, including the dataset, prompts, source code, and results available at https://github.com/Mobile-IoT-Security-Lab/PolicyGapper and https://doi.org/10.5281/zenodo.19628493. |
| 2026-04-17 | [Toward EU Sovereignty in Space: A Comparative Simulation Study of IRIS 2 and Starlink](http://arxiv.org/abs/2604.16092v1) | Alexander Bonora, Marco Giordani et al. | The evolution of 6th generation (6G) networks increasingly relies on satellite-based Non-Terrestrial Networks (NTNs) to extend broadband connectivity to remote and unserved regions, and to support public safety. In this paper we compare two representative and conceptually different satellite constellation architectures, namely Starlink and IRIS 2. Starlink is a commercial private Internet constellation by SpaceX, based on dense Low Earth Orbit (LEO) satellites. It is primarily designed to deliver high-capacity broadband services for civil applications, with performance targets comparable to those of terrestrial networks. In contrast, IRIS 2 is a planned public initiative to be deployed by the European Union, based on a multi-layer combination of LEO, Medium Earth Orbit (MEO), and Geo-stationary Earth Orbit (GEO) satellites. It is primarily designed to provide a secure, resilient, and sovereign infrastructure for government and critical communications. After describing the main technical characteristics of Starlink and IRIS 2, we run a comprehensive simulation campaign to evaluate the design tradeoffs between the two. Specifically, we evaluate the per-cell and per-user achievable capacity, the impact of satellite mobility and handover, and identify the capability of each architecture to support global and reliable connectivity. We also provide design suggestions for possible future IRIS 2 deployment extensions. |
| 2026-04-17 | [Safe Deep Reinforcement Learning for Building Heating Control and Demand-side Flexibility](http://arxiv.org/abs/2604.16033v1) | Colin Jüni, Mina Montazeri et al. | Buildings account for approximately 40% of global energy consumption, and with the growing share of intermittent renewable energy sources, enabling demand-side flexibility, particularly in heating, ventilation and air conditioning systems, is essential for grid stability and energy efficiency. This paper presents a safe deep reinforcement learning-based control framework to optimize building space heating while enabling demand-side flexibility provision for power system operators. A deep deterministic policy gradient algorithm is used as the core deep reinforcement learning method, enabling the controller to learn an optimal heating strategy through interaction with the building thermal model while maintaining occupant comfort, minimizing energy cost, and providing flexibility. To address safety concerns with reinforcement learning, particularly regarding compliance with flexibility requests, we propose a real-time adaptive safety-filter to ensure that the system operates within predefined constraints during demand-side flexibility provision. The proposed real-time adaptive safety filter guarantees full compliance with flexibility requests from system operators and improves energy and cost efficiency -- achieving up to 50% savings compared to a rule-based controller -- while outperforming a standalone deep reinforcement learning-based controller in energy and cost metrics, with only a slight increase in comfort temperature violations. |
| 2026-04-17 | ["When I see Jodie, I feel relaxed": Examining the Impact of a Virtual Supporter in Remote Psychotherapy](http://arxiv.org/abs/2604.16003v1) | Jiashuo Cao, Chen Li et al. | Virtual agents have shown promising potential in mental health applications, but current research has predominantly focused on contexts outside of traditional therapy sessions. This paper examines the impact of a virtual supporter in remote psychotherapy sessions conducted via Zoom. We used a two-phase research approach. First we conducted a formative study to understand the roles and functions of human supporters in psychotherapy contexts. Based on these findings, we developed a virtual supporter operating in two modes: Daily Mode (for mood journaling outside therapy) and Therapy Mode (as an additional participant in Zoom therapy sessions). Finally we ran a user study with 14 participants who engaged with the virtual supporter for a week and then joined a remote psychotherapy session together. Our findings revealed that the virtual supporter had positive effects on creating psychological safety, reducing anxiety, and enhancing emotional articulation without disrupting the therapeutic process. We then discussed both the benefits and potential disadvantages of virtual supporters in therapeutic contexts, including concerns about over-reliance and the need for appropriate boundaries. This research contributes to understanding how AI-driven virtual agents could contribute to human-led remote psychotherapy. |
| 2026-04-17 | [TwoHamsters: Benchmarking Multi-Concept Compositional Unsafety in Text-to-Image Models](http://arxiv.org/abs/2604.15967v1) | Chaoshuo Zhang, Yibo Liang et al. | Despite the remarkable synthesis capabilities of text-to-image (T2I) models, safeguarding them against content violations remains a persistent challenge. Existing safety alignments primarily focus on explicit malicious concepts, often overlooking the subtle yet critical risks of compositional semantics. To address this oversight, we identify and formalize a novel vulnerability: Multi-Concept Compositional Unsafety (MCCU), where unsafe semantics stem from the implicit associations of individually benign concepts. Based on this formulation, we introduce TwoHamsters, a comprehensive benchmark comprising 17.5k prompts curated to probe MCCU vulnerabilities. Through a rigorous evaluation of 10 state-of-the-art models and 16 defense mechanisms, our analysis yields 8 pivotal insights. In particular, we demonstrate that current T2I models and defense mechanisms face severe MCCU risks: on TwoHamsters, FLUX achieves an MCCU generation success rate of 99.52%, while LLaVA-Guard only attains a recall of 41.06%, highlighting a critical limitation of the current paradigm for managing hazardous compositional generation. |
| 2026-04-17 | [Pruning Unsafe Tickets: A Resource-Efficient Framework for Safer and More Robust LLMs](http://arxiv.org/abs/2604.15780v1) | Wai Man Si, Mingjie Li et al. | Machine learning models are increasingly deployed in real-world applications, but even aligned models such as Mistral and LLaVA still exhibit unsafe behaviors inherited from pre-training. Current alignment methods like SFT and RLHF primarily encourage models to generate preferred responses, but do not explicitly remove the unsafe subnetworks that trigger harmful outputs. In this work, we introduce a resource-efficient pruning framework that directly identifies and removes parameters associated with unsafe behaviors while preserving model utility. Our method employs a gradient-free attribution mechanism, requiring only modest GPU resources, and generalizes across architectures and quantized variants. Empirical evaluations on ML models show substantial reductions in unsafe generations and improved robustness against jailbreak attacks, with minimal utility loss. From the perspective of the Lottery Ticket Hypothesis, our results suggest that ML models contain "unsafe tickets" responsible for harmful behaviors, and pruning reveals "safety tickets" that maintain performance while aligning outputs. This provides a lightweight, post-hoc alignment strategy suitable for deployment in resource-constrained settings. |
| 2026-04-17 | [MemEvoBench: Benchmarking Memory MisEvolution in LLM Agents](http://arxiv.org/abs/2604.15774v1) | Weiwei Xie, Shaoxiong Guo et al. | Equipping Large Language Models (LLMs) with persistent memory enhances interaction continuity and personalization but introduces new safety risks. Specifically, contaminated or biased memory accumulation can trigger abnormal agent behaviors. Existing evaluation methods have not yet established a standardized framework for measuring memory misevolution. This phenomenon refers to the gradual behavioral drift resulting from repeated exposure to misleading information. To address this gap, we introduce MemEvoBench, the first benchmark evaluating long-horizon memory safety in LLM agents against adversarial memory injection, noisy tool outputs, and biased feedback. The framework consists of QA-style tasks across 7 domains and 36 risk types, complemented by workflow-style tasks adapted from 20 Agent-SafetyBench environments with noisy tool returns. Both settings employ mixed benign and misleading memory pools within multi-round interactions to simulate memory evolution. Experiments on representative models reveal substantial safety degradation under biased memory updates. Our analysis suggests that memory evolution is a significant contributor to these failures. Furthermore, static prompt-based defenses prove insufficient, underscoring the urgency of securing memory evolution in LLM agents. |
| 2026-04-17 | [Zero-Shot Scalable Resilience in UAV Swarms: A Decentralized Imitation Learning Framework with Physics-Informed Graph Interactions](http://arxiv.org/abs/2604.15762v1) | Huan Lin, Lianghui Ding | Large-scale Unmanned Aerial Vehicle (UAV) failures can split an unmanned aerial vehicle swarm network into disconnected sub-networks, making decentralized recovery both urgent and difficult. Centralized recovery methods depend on global topology information and become communication-heavy after severe fragmentation. Decentralized heuristics and multi-agent reinforcement learning methods are easier to deploy, but their performance often degrades when the swarm scale and damage severity vary. We present Physics-informed Graph Adversarial Imitation Learning algorithm (PhyGAIL) that adopts centralized training with decentralized execution. PhyGAIL builds bounded local interaction graphs from heterogeneous observations, and uses physics-informed graph neural network to encode directional local interactions as gated message passing with explicit attraction and repulsion. This gives the policy a physically grounded coordination bias while keeping local observations scale-invariant. It also uses scenario-adaptive imitation learning to improve training under fragmented topologies and variable-length recovery episodes. Our analysis establishes bounded local graph amplification, bounded interaction dynamics, and controlled variance of the terminal success signal. A policy trained on 20-UAV swarms transfers directly to swarms of up to 500 UAVs without fine-tuning, and achieves better performance across reconnection reliability, recovery speed, motion safety, and runtime efficiency than representative baselines. |
| 2026-04-17 | [Reasoning-targeted Jailbreak Attacks on Large Reasoning Models via Semantic Triggers and Psychological Framing](http://arxiv.org/abs/2604.15725v1) | Zehao Wang, Lanjun Wang | Large Reasoning Models (LRMs) have demonstrated strong capabilities in generating step-by-step reasoning chains alongside final answers, enabling their deployment in high-stakes domains such as healthcare and education. While prior jailbreak attack studies have focused on the safety of final answers, little attention has been given to the safety of the reasoning process. In this work, we identify a novel problem that injects harmful content into the reasoning steps while preserving unchanged answers. This type of attack presents two key challenges: 1) manipulating the input instructions may inadvertently alter the LRM's final answer, and 2) the diversity of input questions makes it difficult to consistently bypass the LRM's safety alignment mechanisms and embed harmful content into its reasoning process. To address these challenges, we propose the Psychology-based Reasoning-targeted Jailbreak Attack (PRJA) Framework, which integrates a Semantic-based Trigger Selection module and a Psychology-based Instruction Generation module. Specifically, the proposed PRJA automatically selects manipulative reasoning triggers via semantic analysis and leverages psychological theories of obedience to authority and moral disengagement to generate adaptive instructions for enhancing the LRM's compliance with harmful content generation. Extensive experiments on five question-answering datasets demonstrate that PRJA achieves an average attack success rate of 83.6\% against several commercial LRMs, including DeepSeek R1, Qwen2.5-Max, and OpenAI o4-mini. |
| 2026-04-17 | [Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries](http://arxiv.org/abs/2604.15717v1) | Ki Sen Hung, Xi Yang et al. | A central goal of LLM alignment is to balance helpfulness with harmlessness, yet these objectives conflict when the same knowledge serves both legitimate and malicious purposes. This tension is amplified by context-sensitive alignment: we observe that domain-specific contexts (e.g., chemistry) selectively relax defenses for domain-relevant harmful knowledge, while safety-research contexts (e.g., jailbreak studies) trigger broader relaxation spanning all harm categories. To systematically exploit this vulnerability, we propose Jargon, a framework combining safety-research contexts with multi-turn adversarial interactions that achieves attack success rates exceeding 93% across seven frontier models, including GPT-5.2, Claude-4.5, and Gemini-3, substantially outperforming existing methods. Activation space analysis reveals that Jargon queries occupy an intermediate region between benign and harmful inputs, a gray zone where refusal decisions become unreliable. To mitigate this vulnerability, we design a policy-guided safeguard that steers models toward helpful yet harmless responses, and internalize this capability through alignment fine-tuning, reducing attack success rates while preserving helpfulness. |
| 2026-04-17 | [Towards Robust Endogenous Reasoning: Unifying Drift Adaptation in Non-Stationary Tuning](http://arxiv.org/abs/2604.15705v1) | Xiaoyu Yang, En Yu et al. | Reinforcement Fine-Tuning (RFT) has established itself as a critical paradigm for the alignment of Multi-modal Large Language Models (MLLMs) with complex human values and domain-specific requirements. Nevertheless, current research primarily focuses on mitigating exogenous distribution shifts arising from data-centric factors, the non-stationarity inherent in the endogenous reasoning remains largely unexplored. In this work, a critical vulnerability is revealed within MLLMs: they are highly susceptible to endogenous reasoning drift, across both thinking and perception perspectives. It manifests as unpredictable distribution changes that emerge spontaneously during the autoregressive generation process, independent of external environmental perturbations. To adapt it, we first theoretically define endogenous reasoning drift within the RFT of MLLMs as the multi-modal concept drift. In this context, this paper proposes Counterfactual Preference Optimization ++ (CPO++), a comprehensive and autonomous framework adapted to the multi-modal concept drift. It integrates counterfactual reasoning with domain knowledge to execute controlled perturbations across thinking and perception, employing preference optimization to disentangle spurious correlations. Extensive empirical evaluations across two highly dynamic and safety-critical domains: medical diagnosis and autonomous driving. They demonstrate that the proposed framework achieves superior performance in reasoning coherence, decision-making precision, and inherent robustness against extreme interference. The methodology also exhibits exceptional zero-shot cross-domain generalization, providing a principled foundation for reliable multi-modal reasoning in safety-critical applications. |
| 2026-04-17 | [Long-Term Memory for VLA-based Agents in Open-World Task Execution](http://arxiv.org/abs/2604.15671v1) | Xu Huang, Weixin Mao et al. | Vision-Language-Action (VLA) models have demonstrated significant potential for embodied decision-making; however, their application in complex chemical laboratory automation remains restricted by limited long-horizon reasoning and the absence of persistent experience accumulation. Existing frameworks typically treat planning and execution as decoupled processes, often failing to consolidate successful strategies, which results in inefficient trial-and-error in multi-stage protocols. In this paper, we propose ChemBot, a dual-layer, closed-loop framework that integrates an autonomous AI agent with a progress-aware VLA model (Skill-VLA) for hierarchical task decomposition and execution. ChemBot utilizes a dual-layer memory architecture to consolidate successful trajectories into retrievable assets, while a Model Context Protocol (MCP) server facilitates efficient sub-agent and tool orchestration. To address the inherent limitations of VLA models, we further implement a future-state-based asynchronous inference mechanism to mitigate trajectory discontinuities. Extensive experiments on collaborative robots demonstrate that ChemBot achieves superior operational safety, precision, and task success rates compared to existing VLA baselines in complex, long-horizon chemical experimentation. |
| 2026-04-17 | [Verification of Autonomous Systems with Optimal Controllers](http://arxiv.org/abs/2604.15659v1) | Dylan Le, Joel McCandless et al. | This paper considers the problem of reachability analysis of control systems with optimal controllers, as a first step towards verifying the safety and correctness of such systems. Despite their appeal in guaranteeing task satisfaction through cost minimization, optimal controllers are often challenging to assure. In particular, as system dynamics grow in complexity, solving the resulting optimization problem may be difficult, especially given time and computation constraints on real platforms. Thus, it is essential to verify that, even if the optimal solution is not always found, such controllers still accomplish the high-level control objective. In this paper, we focus on gradient descent algorithms and design a reachability algorithm by treating gradient descent as a separate (digital) dynamical system, embedded in the original (physical) dynamical system, with controls as part of the state. We evaluate the feasibility of the proposed method on two control systems, a two-dimensional quadrotor and a cartpole. |
| 2026-04-17 | [Contact-Aware Planning and Control of Continuum Robots in Highly Constrained Environments](http://arxiv.org/abs/2604.15638v1) | Aedan Mangan, Kehan Long et al. | Continuum robots are well suited for navigating confined and fragile environments, such as vascular or endoluminal anatomy, where contact with surrounding structures is often unavoidable. While controlled contact can assist motion, unfavorable contact can degrade controllability, induce kinematic singularities, or introduce safety risks. We present a contact-aware planning approach that evaluates contact quality, penalizing hazardous interactions, while permitting benign contact. The planner produces kinematically feasible trajectories and contact-aware Jacobians which can be used for closed-loop control in hardware experiments. We validate the approach by testing the integrated system (planning, control, and mechanical design) on anatomical models from patient scans. The planner generates effective plans for three common anatomical environments, and, in all hardware trials, the continuum robot was able to reach the target while avoiding dangerous tip contact (100% success). Mean tracking errors were 1.9 +/- 0.5 mm, 1.2 +/- 0.1 mm, and 1.7 +/- 0.2 mm across the three different environments. Ablation studies showed that penalizing end-of-continuum-segment (ECS) contact improved manipulability and prevented hardware failures. Overall, this work enables reliable, contact-aware navigation in highly constrained environments. |
| 2026-04-16 | [Symbolic Guardrails for Domain-Specific Agents: Stronger Safety and Security Guarantees Without Sacrificing Utility](http://arxiv.org/abs/2604.15579v1) | Yining Hong, Yining She et al. | AI agents that interact with their environments through tools enable powerful applications, but in high-stakes business settings, unintended actions can cause unacceptable harm, such as privacy breaches and financial loss. Existing mitigations, such as training-based methods and neural guardrails, improve agent reliability but cannot provide guarantees. We study symbolic guardrails as a practical path toward strong safety and security guarantees for AI agents. Our three-part study includes a systematic review of 80 state-of-the-art agent safety and security benchmarks to identify the policies they evaluate, an analysis of which policy requirements can be guaranteed by symbolic guardrails, and an evaluation of how symbolic guardrails affect safety, security, and agent success on $τ^2$-Bench, CAR-bench, and MedAgentBench. We find that 85\% of benchmarks lack concrete policies, relying instead on underspecified high-level goals or common sense. Among the specified policies, 74\% of policy requirements can be enforced by symbolic guardrails, often using simple, low-cost mechanisms. These guardrails improve safety and security without sacrificing agent utility. Overall, our results suggest that symbolic guardrails are a practical and effective way to guarantee some safety and security requirements, especially for domain-specific AI agents. We release all codes and artifacts at https://github.com/hyn0027/agent-symbolic-guardrails. |
| 2026-04-16 | [Safe and Energy-Aware Multi-Robot Density Control via PDE-Constrained Optimization for Long-Duration Autonomy](http://arxiv.org/abs/2604.15524v1) | Longchen Niu, Andrew Nasif et al. | This paper presents a novel density control framework for multi-robot systems with spatial safety and energy sustainability guarantees. Stochastic robot motion is encoded through the Fokker-Planck Partial Differential Equation (PDE) at the density level. Control Lyapunov and control barrier functions are integrated with PDEs to enforce target density tracking, obstacle region avoidance, and energy sufficiency over multiple charging cycles. The resulting quadratic program enables fast in-the-loop implementation that adjusts commands in real-time. Multi-robot experiment and extensive simulations were conducted to demonstrate the effectiveness of the controller under localization and motion uncertainties. |
| 2026-04-16 | [EasyRider: Mitigating Power Transients in Datacenter-Scale Training Workloads](http://arxiv.org/abs/2604.15522v1) | Dillon Jensen, Obi Nnorom et al. | Large-scale AI model training workloads use thousands of GPUs operating in tightly synchronized loops. During synchronous communication, start-up, shut-down, and checkpointing, GPU power consumption can swing from peak to idle within milliseconds. These large and rapid load swings endanger grid infrastructure as they induce steep power ramp rates, voltage and frequency shifts, and reactive power transients that can damage transformers, converters, and protection equipment. To solve this problem, we introduce EasyRider, a power architecture to mitigate power fluctuations at the rack level. EasyRider uses passive components and actively-controlled auxiliary energy storage to attenuate rack power swings. A software system continually monitors the energy storage system to maximize its lifetime in the presence of frequent charge/discharge cycles. EasyRider filters rack power variations to be within grid safety requirements without requiring software modifications to AI training frameworks or wasting energy. We evaluate EasyRider on a 400VDC-rated prototype system against published workload traces and our own GPU testbed, demonstrating its effectiveness across heterogeneous power levels and workload power profiles. |
| 2026-04-16 | [''It Is Much Safer to Be Sparse than Connected'': Safe Control of Robotic Swarm Density Dynamics with PDE-Optimization with State Constraints](http://arxiv.org/abs/2604.15516v1) | Longchen Niu, Gennaro Notomista | This paper introduces a safety-critical optimization-based control strategy that leverages control Lyapunov and control barrier functions to guide the spatial density of robotic swarms governed by the Fokker-Planck equation to a predefined target distribution. In contrast to traditional open-loop state-constrained optimal control strategies, the proposed approach operates in closed-loop, and a Voronoi-based variant further enables distributed deployments. Theoretical guarantees of safety are derived, and numerical simulations demonstrate the performance of the proposed controllers. Finally, a multi-robot experiment showcases the real-world applicability of the proposed controllers under localization and motion noises, illustrating how it is much easier for a sparse swarm to satisfy safety specifications than it is for a densely packed one. |
| 2026-04-16 | [Trajectory Planning for Safe Dual Control with Active Exploration](http://arxiv.org/abs/2604.15507v1) | Kaleb Ben Naveed, Manveer Singh et al. | Planning safe trajectories under model uncertainty is a fundamental challenge. Robust planning ensures safety by considering worst-case realizations, yet ignores uncertainty reduction and leads to overly conservative behavior. Actively reducing uncertainty on-the-fly during a nominal mission defines the dual control problem. Most approaches address this by adding a weighted exploration term to the cost, tuned to trade off the nominal objective and uncertainty reduction, but without formal consideration of when exploration is beneficial. Moreover, safety is enforced in some methods but not in others. We study a budget-constrained dual control problem, where uncertainty is reduced subject to safety and a mission-level cost budget that limits the allowable degradation in task performance due to exploration. In this work, we propose Dual-gatekeeper, a framework that integrates robust planning with active exploration under formal guarantees of safety and budget feasibility. The key idea is that exploration is pursued only when it provides a verifiable improvement without compromising safety or violating the budget, enabling the system to balance immediate task performance with long-term uncertainty reduction in a principled manner. We provide two implementations of the framework based on different safety mechanisms and demonstrate its performance on quadrotor navigation and autonomous car racing case studies under parametric uncertainty. |
| 2026-04-16 | [Joint Detection and Characterization of the Standing Accretion Shock Instability for Core-Collapse Supernovae with cWB XP](http://arxiv.org/abs/2604.15500v1) | Vicente Sierra, Zidu Lin et al. | The most sensitive to-date multimessenger detection of the standing accretion shock instability in real interferometric data is presented, which quantitatively identifies the presence of the SASI in core-collapse supernovae using neutrino and gravitational-wave (GW) signals. In the GW channel, the coherent WaveBurst (cWB) software on its version XP is implemented, among with real LIGO data from the O3 and O4 observing runs. With this, a more accurate estimation of parameters, such as the central frequency and signal duration, is obtained for both sets of data. The SASI identification probability versus false alarm rates is presented in the form of Receiver Operating Characteristic (ROC) curves. For O3, the new study for the combined GW and neutrino detection condition, labeled as $x + y$, shows an identification probability (previous best results from Lin et al. [1]) of 1 (1), 0.90 (0.70) and 0.37 (0.34) at 1, 5 and 10 kpc for a false identification probability of 0.10. On the other hand, using O4 shows that the GW channel by itself is sensitive enough to provide almost perfect identification probability scores, with identification probability values of 1, 0.99 and 0.97 for a false identification probability of 0.01 at 1, 5 and 10 kpc, respectively. |

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



