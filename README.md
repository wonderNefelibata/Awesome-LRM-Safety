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
| 2026-05-29 | [Stateful Online Monitoring Catches Distributed Agent Attacks](http://arxiv.org/abs/2605.31593v1) | Davis Brown, Samarth Bhargav et al. | Language models can find thousands of severe software vulnerabilities, and agents are increasingly being misused for cyberattacks. To avoid detection, attackers frequently distribute their misuse, splitting a harmful task across many user accounts so each individual transcript looks benign. Because safety monitors score only one agent context at a time, they are structurally blind to misuse that is only visible in aggregate, across many accounts. We show this gap is real by building, to our knowledge, the first distributed agent attack, a multi-agent scaffold that completes hard cybersecurity tasks while hiding the harmful objective across subagents with limited contexts, evading a standard monitor that catches it only a fifth as often as prior agent attacks. Towards a defense, we develop an online stateful monitor that uses real-time clustering to collect weak suspiciousness signals across many agent transcripts, and escalates only rarely to a language model that flags misuse across user accounts. In evaluations with large-scale simulated datacenter traffic, our monitor Pareto dominates standard monitors, catching distributed attacks 30% earlier and flagging cyber misuse before it reaches the most harmful stages. Crucially, this comes at negligible additional latency for ~99% of user traffic. This detection advantage persists but narrows as the benign background traffic grows very large. After an extensive red-teaming exercise, we improve the defense and surprisingly also find that it catches standard jailbreaks, since adaptive attackers reuse attack variants across accounts. Our results point toward a new class of safety monitors which reason over groups of users rather than isolated transcripts. |
| 2026-05-29 | [Used Car Salesbots? Honesty and Credulity of LLMs as Bargaining Agents under Partial Information](http://arxiv.org/abs/2605.31445v1) | Antonio Valerio Miceli-Barone, Vaishak Belle et al. | In this work we study agents in simulated bargaining scenarios, where a buyer and a seller communicate through a text channel and attempt to negotiate mutually beneficial trades, under different information regimes (complete information, information asymmetry or mutual uncertainty). We evaluate their performance w.r.t. game-theoretical solutions and further investigate their honesty (their tendency to disclose or withhold information or to mislead and deceive) as well as their credulity (their tendency to trust or distrust information provided by the other agent). We study zero-shot LLM agents with simple prompting scaffolding as well as fine-tuned agents, in order to investigate whether optimising the agents to maximise financial profits makes them stronger negotiators but also more dishonest and less trusting.   We find that off-the-shelf LLMs all substantially deviate from game-theoretical equilibria, they attempt to lie about their private information but cannot efficiently exploit information asymmetries. Fine-tuning on financial utility makes the agents stronger at achieving better deals but also more dishonest, highlighting the risks that optimising agents for a task can have on their safety. We release our code and a dataset of bargaining scenarios. |
| 2026-05-29 | [Adaptive Artificial Time-Delay Control with Barrier Lyapunov Constraints for Euler-Lagrange Robots](http://arxiv.org/abs/2605.31405v1) | Saksham Gupta, Rishabh Dev Yadav et al. | This paper addresses the challenge of simultaneously compensating for state-dependent uncertainties and enforcing time-varying state constraints in Euler-Lagrange systems, a common requirement in robotics that remains underserved by existing control designs. A novel adaptive control framework is developed that combines an artificial time-delay-based uncertainty estimation strategy, also known as time-delay estimation, with a barrier Lyapunov function to enforce constraint-aware control design. Specifically, a state-dependent upper bound on the time-delay estimation approximation error is analytically formulated, and an adaptive law is constructed to estimate its parameters online, enabling real-time state-dependent uncertainty compensation without relying on prior model knowledge. To ensure constraint compliance, the barrier Lyapunov function-based controller enforces time-varying bounds on both position and velocity. The resulting architecture is provably stable via Lyapunov analysis. Experimental results on a five-degree-of-freedom robotic manipulator validate the framework's capability, compared with the state of the art, in maintaining strict adherence to safety-critical constraints under dynamic uncertainties. |
| 2026-05-29 | [LLM Judges Inconsistently Disagree Across Safety Criteria and Harm Categories](http://arxiv.org/abs/2605.31381v1) | Krishnapriya Vishnubhotla, Soumya Vajjala et al. | We evaluate the consistency of automated judges in conducting a multi-dimensional safety evaluation in a reference-free setup. Our results indicate that Large Language Models are unreliable judges in identifying safety issues related to machine-generated advice in regulated domains such as finance, although they are more reliable at identifying more overt forms of unsafe/harmful content such as violence. The degree of inconsistency in a model's judgments can vary significantly by the chosen safety criteria and can be impacted by the language of the content and its linguistic style as well. Finally, there is high disagreement among different judges for the same output, across domains, safety criteria, and languages. These findings provide new insights on the practice of using LLMs as evaluators and offer several recommendations for practitioners on how to use automated judges in practical scenarios. |
| 2026-05-29 | [LiftNav: Path Planning via Semantic Lifting in TSDF-Guided Gaussian Splatting](http://arxiv.org/abs/2605.31376v1) | Hannah Schieber, Dominik Frischmann et al. | Autonomous robots in unknown indoor environments require both reliable collision avoidance and object-level understanding. Classical representations such as TSDF support safe planning but lack semantics, while photorealistic methods like Gaussian Splatting (GS) provide rich appearance yet suffer from soft geometry, limiting precise obstacle avoidance. We present LiftNav, a hybrid navigation framework built on GSFusion's TSDF+GS dual map, augmented with a real-time pipeline of YOLO-based detection, TSDF-based 3D lifting, and B-spline trajectory optimization. This design enables flexible semantic navigation without dense 3D embeddings. We further introduce a hinge-loss-based collision penalty that improves trajectory smoothness and safety. We evaluate our approach in a simulation using the Replica dataset. Compared against a state-of-the-art radiance field baseline we show a 100% feasibility rate and shorter trajectories. |
| 2026-05-29 | [dashi: A Python library for Dataset Shift Characterization to Support Trustworthy AI Development and Deployment](http://arxiv.org/abs/2605.31360v1) | David Fernández-Narro, Pablo Ferri et al. | The Artificial Intelligence (AI) life cycle requires a thorough understanding of the underlying data dynamics for robust, safe and cost-effective AI development and use. Dataset shifts are defined as changes between train and test data distributions. Whether occurring over time (temporal) or across different sites (multi-source), they can severely degrade model performance and compromise data quality. This is particularly important in health AI, where the safety and fundamental rights of patients can be severely affected by uncontrolled shifts both at training and operational stages. While the theoretical foundations of covariate, prior, and concept shifts are well established, there is a lack of accessible and comprehensive software tools to perform their analysis. We introduce dashi, an open-source Python library designed for the exploration, quantification, and characterization of dataset shifts. dashi provides a dual approach: an unsupervised approach that leverages information geometry and non-parametric statistical manifolds to data variability characterization and analysis (e.g., Information Geometric Temporal plots and Multi-Source Variability metrics like Global Probabilistic Deviation and Source Probabilistic Outlyingness), and a supervised approach that quantifies and characterizes model performance degradation. Both unsupervised and supervised approaches work across user-defined temporal and domain/source batches. We demonstrate the utility of dashi on three simulated and real-world health AI case studies on gestational diabetes mellitus, COVID-19 and emergency medical dispatch. By providing interactive visual analytics and variability metrics, dashi supports trustworthiness of AI life cycle stages enabling robust and safe machine learning pipelines through the assessment of data coherence and AI performance. |
| 2026-05-29 | [SQEEZ: Energy-efficient Location Sharing for Mobile Ad Hoc Networks](http://arxiv.org/abs/2605.31339v1) | Ram Ramanathan, Dmitrii Dugaev et al. | Periodic network-wide dissemination of node location data is crucial for shared situational awareness and collaborative mapping in mobile ad hoc and mesh networks for public safety, disaster relief, and military. A key challenge is to provide maximally accurate location information with minimal energy expenditure on part of the nodes. We present SQEEZ: a mechanism for reducing the Position Location Information (PLI) load that combines two orthogonal techniques: (1) adaptive suppression of location updates; and (2) temporal and inline compression of update packets. We describe the SQEEZ suppression and compression algorithms, analyze the tradeoff between location error and energy consumption, and introduce a new metric called \textit{Error-Penalized-Energy (EPE)} that normalizes the energy metric using the error incurred. Our simulation results show that, in the range of parameters studied, SQEEZ improves the EPE-efficiency and scalability in a 30-node random waypoint scenario by up to 4.4x and 2.3x respectively; and increases the EPE-efficiency by 7.5x in a 9-node real-world network trace. Compression provides larger improvements than suppression at high mobilities and vice-versa at low mobilities. |
| 2026-05-29 | [Reinforcement Learning Amplifies Emergent Misalignment from Harmless Rewards](http://arxiv.org/abs/2605.31328v1) | Magnus Jørgenvåg, David Kaczér et al. | Emergent misalignment (EM) is the surprising tendency of language models to become broadly misaligned after fine-tuning on narrowly misaligned examples. While EM has been extensively studied in the supervised fine-tuning (SFT) setting, evidence that it also arises from reinforcement learning (RL) is limited to large, closed-source models, leaving the phenomenon expensive to study and difficult to reproduce. We characterize EM from RL in small, off-the-shelf open-weight models along three axes. First, we show that rewarding narrow, overtly misaligned behavior produces substantially higher general-domain misalignment than sample-matched SFT. Second, we show that EM from RL can be induced by reward signals that could plausibly arise naturally, such as unpopular aesthetic preferences or poor rhetorical appeals. Third, we evaluate in-training mitigations developed for SFT-induced EM and find that they broadly transfer, with interleaving on-policy safety data performing best. |
| 2026-05-29 | [Gyrokinetic global simulation of Alfvenic ion temperature gradient mode in reversed magnetic shear](http://arxiv.org/abs/2605.31313v1) | Gengxian Li, Zhixin Lu et al. | In this work, a systematic study of electromagnetic instabilities driven by the temperature gradient in magnetically confined fusion plasmas with reversed magnetic shear is conducted using gyrokinetic particle-in-cell simulations. An electromagnetic instability arising in the low-beta regime is investigated, where beta=8*pi*nT/B^2 denotes the ratio of plasma pressure to magnetic pressure. Within a reversed shear safety factor (q) profile, when a mode rational surface coincides with the position of zero shear, an instability dominated by only one poloidal harmonic emerges, rather than the conventional ion-temperature-gradient (ITG) mode. Simulation results demonstrate that the instability exhibits pronounced electromagnetic polarization even in the low-beta regime, with a real frequency significantly higher than that of ITG modes, and show that it is destabilized by the temperature gradient and not by the density gradient. This instability can be observed even for a monotonic q profile with weak magnetic shear. Based on a systematic comparison with other typical electrostatic and electromagnetic instabilities, this instability is identified as a weak shear Alfvenic-ion-temperature-gradient (WSAITG) mode, which may provide an explanation for the low-frequency Alfven modes (LFAM) observed in experiments. Wave-particle resonance analysis in phase space reveals that, in contrast to the ITG mode, well-passing particles provide an additional resonant population that drives the WSAITG mode. |
| 2026-05-29 | [Mellum2 Technical Report](http://arxiv.org/abs/2605.31268v1) | Marko Kojic, Ivan Bondyrev et al. | We present Mellum 2, an open-weight 12B-parameter Mixture-of-Experts (MoE) language model with 2.5B active parameters per token. Mellum 2 is a general-purpose language model specialized in software engineering, spanning code generation and editing, debugging, multi-step reasoning, tool use and function calling, agentic coding, and conversational programming assistance, and it is the successor to the completion-focused 4B dense Mellum model. The architecture builds on the Mixture-of-Experts (64 experts, 8 active) and combines Grouped-Query Attention with 4 KV heads, Sliding Window Attention on three of every four layers, and a single Multi-Token Prediction head that doubles as both an auxiliary pre-training objective and a built-in draft model for speculative decoding; each choice was validated by ablation with inference efficiency on commodity GPUs as a design constraint. Pre-training spans approximately 10.6 trillion tokens through a three-phase curriculum that progressively shifts the mixture from diverse web data toward curated code and mathematical content, optimized with Muon under FP8 hybrid precision and a Warmup-Hold-Decay schedule with linear decay to zero. The pre-trained base is extended to a 128K context window via a layer-selective YaRN and then post-trained in two stages (supervised fine-tuning followed by RLVR), yielding two released variants: an Instruct model that answers directly and a Thinking model that emits an explicit reasoning trace before its final answer. Across code generation, math and reasoning, tool use, knowledge, and safety benchmarks, Mellum 2 is competitive with open-weight baselines in the 4B-14B range while running at the per-token compute of a 2.5B dense model. We release the base, instruct, and thinking checkpoints, together with this report on the architecture decisions, data pipeline, and training recipe behind them, under the Apache 2.0 license. |
| 2026-05-29 | [Safe Arrival Scheduling at Constraint Waypoints in UAM Corridors](http://arxiv.org/abs/2605.31243v1) | Sasinee Pruekprasert, Shinji Nakadai | This study introduces a novel Air Traffic Control (ATC) concept to support self-separation between vehicles in Urban Air Mobility (UAM) corridors. Our proposed scheme involves sharing intended arrival schedules at Constrained Waypoints (CWPs) among UAM operators. We propose two approaches to assist the arrival scheduling at CWPs by computing the minimum arrival time gap necessary for each pair of vehicles to ensure their safety throughout the flights within the corridor. The first approach considers the minimum separation distance required by the Near Mid-Air-Collision (NMAC) avoidance rules, while the second one is based on the Responsibility-Sensitive Safety (RSS) rules. We demonstrate that the NMAC-rule-based approach can effectively prevent collisions in normal circumstances, where the vehicles adhere to the speed limits of the corridor. However, this approach does not guarantee safety if vehicles exceed the speed limits. Conversely, while the RSS-rule-based approach ensures collision prevention during emergencies when vehicles exceed speed limits, it may require larger arrival time gaps under normal circumstances, which may lead to reduced traffic flow. Our results are confirmed through numerical simulations. |
| 2026-05-29 | [Simulation of collision avoidance behavior in crowd movement by data-driven approach](http://arxiv.org/abs/2605.31210v1) | Xuanwen Liang, Eric Wai Ming Lee | Crowd movement simulation is essential for pedestrian safety management and facility layout optimization. Data-driven models enhance trajectory prediction accuracy under Euclidean metrics, yet they suffer from excessively high collision rates, especially in bidirectional and multidirectional flows. In this paper, we establish a novel data-driven crowd simulation model that incorporates the pedestrian collision mechanism into the loss function to reduce collisions. A new lateral-acceleration-based collision loss function and a Voronoi-based motion feature extraction approach are proposed. The model is based on a Generative Adversarial Network (GAN) architecture and is termed CPGAN (Collision-Penalized GAN). We evaluate CPGAN in bidirectional flow scenarios, which involve frequent collision avoidance behaviors. Results show that the proposed lateral-acceleration-based collision loss significantly reduces opposite-direction pedestrian collision rates to levels comparable with controlled experiments. CPGAN effectively simulates bidirectional flow, reproducing lane formation and N-t curves. The research outcomes can provide inspiration for integrating pedestrian dynamics mechanisms into loss functions in data-driven crowd simulation. |
| 2026-05-29 | [Probabilistic Precipitation Nowcasting with Rectified Flow Transformers](http://arxiv.org/abs/2605.31204v1) | Johannes Schusterbauer, Jannik Wiese et al. | Accurate weather forecasts are essential across various domains and are safety-critical in extreme weather conditions. Compared to simulation-based forecasting, data-driven approaches show greater efficiency, enabling short-term, high-resolution nowcasting. In particular, diffusion models proved effective in weather nowcasting due to their strong probabilistic foundation. However, existing methods rely on deterministic compression to reduce the complexity of high-dimensional weather data, limiting their ability to capture uncertainty in the decoding process. In this work, we introduce $\textbf{FREUD}$, a $\textbf{Fr}$ame-wise $\textbf{E}$ncoder and $\textbf{U}$nited $\textbf{D}$ecoder model based on rectified flow transformers for efficient compression of spatio-temporal weather data. Frame-wise encoding enables continuous forecast updates, while the unified video decoder ensures temporal consistency. Our uncertainty-preserving first stage allows us to capture aleatoric uncertainty via ensembling, which is particularly beneficial for extreme weather events with high decoding variability. We achieve state-of-the-art performance in precipitation nowcasting with a compact latent-space rectified flow transformer on the SEVIR benchmark and show further performance gains by model and test-time scaling. Code available here: https://github.com/CompVis/weather-rf |
| 2026-05-29 | [Probing Collision Grounding in Vision-Language Models for Safe Human-Robot Collaboration](http://arxiv.org/abs/2605.31196v1) | Jun Wang, Xiaohao Xu et al. | Safe human--robot collaboration requires more than visual description: a monitor must determine whether the robot body is safely separated, already colliding with the scene or a person, or about to collide. We call this capability collision grounding: binding visual observations to robot body geometry, camera viewpoint, scene layout, human proximity, and temporal motion in order to infer present and imminent contact. We introduce TouchSafeBench, a physics-grounded benchmark for evaluating collision grounding in vision-language models (VLMs). Built in Habitat~3.0, TouchSafeBench contains 2,940 simulated indoor co-presence episodes across social navigation and social rearrangement, with synchronized multi-view RGB-D observations, top-down trajectory maps, calibrated camera metadata, and simulator-derived contact labels. We study two deployment-facing tasks: classifying the current safety state and warning about imminent collision before contact. Across three frontier or robotics-oriented VLMs and nine visual representations, current models remain far from reliable: the best average Macro-F1 stays below 50\%, explicit depth is not automatically transformed into robot-body collision evidence, and robot--scene contact is consistently harder than human-contact risk. TouchSafeBench reveals a central limitation of embodied VLMs: visual fluency does not imply physical accountability. Reliable robot safety monitors will need representations that explicitly bind viewpoint, robot morphology, metric geometry, and future collision. We will release the benchmark upon acceptance. |
| 2026-05-29 | [From Evidence to Design: Developing an AI-Augmented UX Research Point of View for Digital Wellbeing in Emergency and Public Safety Contexts](http://arxiv.org/abs/2605.31146v1) | Olumuyiwa Ayorinde, Huseyin Dogan et al. | This paper investigates how User Experience Research (UXR) methods can be combined with AI-supported analysis to develop clearer design direction for digital wellbeing interventions targeting Emergency and Public Safety Personnel (EPSP). EPSP work in high-stress, shift-based environments where cognitive fatigue and unpredictable schedules reduce engagement with conventional wellbeing tools. Using the UXR Point-of-View (PoV) framework, this study applied an AI-supported literature analysis process to identify recurring psychological, behavioural, and design patterns. Behaviour Change Techniques and Persuasive Technology principles were integrated throughout interpretation to connect evidence with practical design reasoning. The process resulted in a UXR PoV Pyramid, nine UXR Play Cards, and stakeholder focused PoV narratives. Findings show that effective wellbeing systems for EPSP must minimise cognitive effort, adapt to operational context, and prioritise psychological safety. The work demonstrates how AI can assist large-scale evidence interpretation while human researchers maintain responsibility for contextual judgement and design direction. |
| 2026-05-29 | [TARIC: Memory-Augmented Traversability-Aware Outdoor VLN under Interrupted Semantic Cues](http://arxiv.org/abs/2605.31121v1) | Tianle Zeng, Hanjing Ye et al. | Outdoor vision-language navigation (VLN) in long-range, open-world environments is frequently disrupted by semantic-cue interruptions, where informative goal cues become sparse, occluded, or leave the field of view. Once such cues disappear, agents enter a cue-free phase and often degrade into backtracking, oscillatory headings, or aimless exploration. While memory-based methods attempt to bridge these gaps, they often fail under traversability-driven detours: the remembered cue direction may be infeasible, forcing detours that prolong cue-free phases and gradually render robot-centric cues stale and implicit histories blurred. This makes traversability a stability condition for maintaining goal-directed guidance, rather than merely a local safety concern.   We propose a unified outdoor VLN framework that survives semantic-cue interruptions by maintaining traversability-consistent executable guidance throughout prolonged cue-free phases. Specifically, our method extracts semantic bearings from visibility-gated goal or exploration cues and grounds them into executable headings using a real-time near-field traversability profile, providing goal-consistent feasible guidance beyond reject-only safety filtering. To prevent guidance degradation during detours, we lift intermittent 2D evidence into a world-aligned 3D cue memory with an uncertainty-aware readout mechanism, ensuring guidance remains continuously reachable and stable as the robot moves.   We evaluate the framework on quadrupedal and wheeled platforms over 600--1000 m routes. Our method improves simulation success rate by over 10 percentage points over the strongest baseline and achieves a real-world success rate of 40%, compared to 17.5% for the strongest baseline, with substantially higher robustness during prolonged cue-free intervals. |
| 2026-05-29 | [ConsisGuard: Aligning Safety Deliberation with Policy Enforcement in LLM Guardrails](http://arxiv.org/abs/2605.31073v1) | Yan Wang, Zhixuan Chu et al. | Reasoning-based LLM guardrails improve safety moderation by generating explicit rationales before issuing final decisions. However, their rationales do not always lead to faithful enforcement: a model may recognize a harmful intent in its reasoning but still predict a safe label, or issue an unsafe decision without policy-grounded justification. We identify this safety-critical failure mode as the deliberation-to-enforcement gap. Unlike general chain-of-thought faithfulness, guardrail reliability requires policy execution consistency: the generated reasoning should be grounded in the safety policy, and the final decision should be entailed by that reasoning. We propose ConsisGuard, a consistency-aware framework for reasoning-based LLM guardrails. ConsisGuard performs Policy-to-Decision Trajectory Distillation and Functional Coupling Alignment, aligning the internal coupling between safety deliberation and decision enforcement. Experiments on prompt and response harmfulness detection benchmarks show that ConsisGuard improves detection performance while reducing policy execution failures. These results suggest that reliable reasoning-based guardrails require accurate faithful execution of safety policies. |
| 2026-05-29 | [Does Visual Information Play a Decisive Role in Vision-Language-Action Model Driving Behavior?](http://arxiv.org/abs/2605.31041v1) | Jingtao He, Hongliang Lu et al. | Vision-Language-Action (VLA) models have demonstrated promising capability in autonomous driving, highlighting the potential of unified multimodal architectures for jointly modeling perception and planning. However, how current VLA-based driving behavior is grounded in visual information remains poorly understood. Existing evaluation protocols mainly focus on aggregate performance metrics, lacking structured and practical diagnostics to quantify visual-behavior dependency. In this work, we introduce a structured multi-level visual perturbation framework to analyze visual-behavior dependency in VLA-based driving models systematically. The framework organizes controlled visual perturbations along three complementary dimensions: channellevel degradation, information-level disruption, and structurelevel modification. We apply it to VLA-based driving systems and evaluate behavioral responses under both open-loop trajectory prediction and interactive closed-loop safety evaluation. Experimental results reveal evaluation-dependent dependency patterns and uneven visual grounding across abstraction levels. These findings call for more structured analyses and principled design of VLA driving models to better understand how visual information shapes behavior and develop safer, more robust systems. |
| 2026-05-29 | [A study on a Real-Time VR-Based Teleoperation Framework for Manipulator in Dynamic Environment](http://arxiv.org/abs/2605.30989v1) | InGyu Choi, GeonYeong Go et al. | Robot teleoperation enables safe, non-contact task execution in hazardous environments where direct human access is difficult, and its application has expanded with recent VR technologies. Many VR teleoperation studies, however, have primarily served as data-collection tools for robot imitation learning, so they often do not explicitly address dynamic obstacles, workspace changes, or collision risks during operation. For real deployment aimed at operator safety, teleoperation must react to dynamic situations with low latency and remain robust to mistakes made by inexperienced operators. This paper presents a VR teleoperation framework that supports real-time manipulation while handling collisions with both static and moving obstacles. The framework integrates GPU-accelerated inverse kinematics and trajectory optimization within a VR interface to generate feasible joint commands at each control cycle under robot constraints. Experiments with a 7-DoF manipulator demonstrate stable online behavior and collision-aware motion generation across three scenarios: obstacle-free, static-obstacle, and moving-obstacle environments. The results indicate that the proposed approach generates motion consistent with the operator's command while producing safe detours when obstacles interfere with the commanded path. |
| 2026-05-29 | [EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents](http://arxiv.org/abs/2605.30924v1) | Dongwook Choi, Taeyoon Kwon et al. | MLLM-powered embodied agents deployed in real-world environments encounter physical hazards. However, existing approaches lack explicit mechanisms for identifying hazards and reasoning about action-conditioned risks, leading agents to either miss risky interactions or over-identify risks. To address this, we propose EMBGuard, the first MLLM-based safety guardrail for embodied agents designed to decouple physical risk reasoning from agent policy. By evaluating a (visual observation, action) pair, EMBGuard identifies hazardous configurations and provides natural language explanations of potential risks. Alongside EMBGuard, we contribute EMBHazard, a training dataset of 15.1K action-conditioned pairs, and EMBGuardTest, a benchmark of 329 manually curated real-world scenarios spanning seven physical risk categories. Through compositional variation of hazards and actions, we generate diverse risky and benign scenarios that agents may encounter during planning. Despite its compact size (2B, 4B), EMBGuard achieves performance competitive with proprietary MLLMs (e.g., GPT-5.1, Gemini-2.5-Pro) while significantly reducing the false-positive rates that hinder real-time deployment. We make the code, data, and models publicly available at https://github.com/dongwxxkchoi/EMBGuard |

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



