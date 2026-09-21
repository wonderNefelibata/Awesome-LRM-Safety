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
| 2026-09-18 | [LIMBO: Learning and Internalizing Model-Free Barrier Objectives for Agile and Safe Whole-Body Control](http://arxiv.org/abs/2609.22075v1) | Jake Gonzales, Arturo Flores Alvarez et al. | Safe whole-body control requires coordinating collision avoidance and balance under high-dimensional, nonlinear dynamics--making safety certificates difficult to design and reuse across behaviors. We present LIMBO, a framework for synthesizing a state-action control barrier function and distilling its safety structure into a task policy. LIMBO learns the safety certificate from black-box transitions and a state-based failure specification over residual actions around a frozen base controller, making Q-CBF synthesis tractable in the full control dimension while placing the certificate in the task policy's control space. During synthesis, the learned safety value drives risk-guided sampling near the estimated boundary of recoverability; during task learning, it serves as a teacher that provides action-level safety feedback, yielding a robust task policy and alleviating the need for an online safety filter at deployment. We demonstrate LIMBO on a 29-degree-of-freedom humanoid performing dodgeball avoidance and locomotion beneath low obstacles. Beyond scaling learned Q-CBFs to whole-body control, we show that risk-guided boundary sampling provides a theoretically grounded way to explore the edge of recoverability. Under the same safety specification, ceteris paribus, varying the sampling concentration produces strategies ranging from crouching to a novel backward-leaning limbo maneuver. In both settings, the learned policies transfer to hardware without online safety filtering, showing that learned safety synthesis scales to agile whole-body control. |
| 2026-09-18 | [Available Guardrails: Certifying Selective Prediction across ML Systems](http://arxiv.org/abs/2609.22048v1) | Parivesh Priye, Yufeng Wang et al. | A selective predictor acts as a safety gate: it returns an output only when the prediction appears sufficiently trustworthy. Deployments increasingly require this reliability to be certified at a target precision for every reporting unit of interest, such as a tool, policy label, or patient subgroup. The main difficulty is often not whether a granted certificate is valid, but whether finite calibration data can produce one at all. As the gate becomes safer or more fine-grained, some units may receive too little evidence to certify. We make this notion of availability computable through classical exact-binomial inversion and formulate reporting-partition selection, under a fixed group order, as a dynamic program that exposes the trade-off among safety, granularity, and served traffic. The resulting frontier reveals a large population opportunity that finite-sample estimation nearly erases: a truth-informed planner gains $0.157$ mean coverage over support balancing, whereas a naive estimator recovers only $0.005$, making recovery from finite data the central challenge. Constructing candidate partitions on one planning split and selecting among them on another recovers part of this gap, improving mean coverage over support balancing by $0.060$, with the direction reproduced in $59$ of $60$ model effects across three intent-routing datasets and two architectures. A complementary validity-preserving lever, reallocating the familywise error budget across reporting units, recovers additional coverage both with population quantities and noisy estimates. The same frontier recurs, with predictor-specific ceilings, across LLM tool-calling, content moderation, lesion classification, and recommendation. Certified availability is therefore a plannable deployment resource that determines when a safety gate can be certified, at what granularity, and over how much traffic. |
| 2026-09-18 | [Assessment of Machine Learning-Based Critical Heat Flux Models in the CTF Subchannel Code for Square Rod Bundle Prediction](http://arxiv.org/abs/2609.21995v1) | Aidan Furlong, Vinicius de Melo Monteiro et al. | The prediction of critical heat flux (CHF), a key safety-related quantity in nuclear thermal hydraulics, remains an important challenge due to its direct relationship with fuel performance and reactor safety. Recent studies have demonstrated that relative to traditional empirical correlations and lookup tables (LUTs), machine learning (ML) methods can substantially improve CHF prediction accuracy. Most ML-based CHF models, however, have been developed and evaluated using tube databases, leaving their applicability to reactor-relevant rod bundle geometries largely unexplored.   This study evaluates ML-based CHF models deployed within the CTF subchannel code using the Electric Power Research Institute (EPRI) rod bundle CHF database. Both pure and hybrid residual correction models are considered in local and semilocal formulations. The tube-trained ML CHF models generally transferred favorably to rod bundle applications and outperformed traditional CHF methods across most geometries and operating conditions. The local hybrid LUT model produced the strongest overall performance, and the semilocal pure ML model remained highly competitive. Comparison against the Bowring correlation, W-3 correlation, and 2006 Groeneveld LUT demonstrated that substantial improvements in rod bundle CHF prediction are possible even when models are trained exclusively on tube data. These findings provide one of the first large-scale assessments of ML-based CHF models in square rod bundles within a production-level subchannel analysis environment and support their broader application in reactor thermal hydraulic analysis. |
| 2026-09-18 | [Can I Trust My Body? A Three-Year Autoethnography of ChatGPT's Place in My Support System for Panic Attacks](http://arxiv.org/abs/2609.21925v1) | Dongyijie Primo Pan, Pan Hui et al. | People increasingly seek mental health support from large language models, yet little is known about their use across years of recurrent panic. We present a three-year analytic autoethnography of the first author's ChatGPT use while living with panic disorder, drawing on conversations, personal records, and accounts from friends or family members and professionals. Narrative analysis traces how my questions shaped ChatGPT's roles and how earlier experiences influenced later responses to symptoms. Familiar explanations could make sensations less frightening, while changed symptoms renewed fears of serious illness. During sudden panic, advice could be difficult to follow, and some replies prompted further checking. Conversations could end while symptoms, checking, or help-seeking continued. We propose trajectory-level safety during and after panic: usable advice (Fit), a stopping point for repeated checking and reassurance seeking (Closure), and useful understanding and human support that remain available over time (Continuity). |
| 2026-09-18 | [VIRGA: Virtual-Agent-Intermediated Riemannian Geometry for Active-Sensing Air-Ground Coordination](http://arxiv.org/abs/2609.21883v1) | Fenghe Guo, Runjie Shen et al. | Air-ground autonomy becomes harder when the unmanned aerial vehicle (UAV) must remain observable by a gimbal light detection and ranging (LiDAR) mounted on the unmanned ground vehicle (UGV). The platforms must avoid dynamic obstacles while coordinating heterogeneous motion, limited sensing, and changing task initiative within one closed loop. This paper presents VIRGA, a neural geometric coordination framework that turns dual-LiDAR observations into bounded source-specific Riemannian fields and couples them through a virtual agent with reciprocal elastic feedback. Platform-aware execution maps convert the shared coordination reference into feasible UAV, UGV, and gimbal commands while enforcing active-observation safeguards. Evaluation against three complementary baselines reveals distinct limitations. An adapted Ray-RMP controller provides the fastest Riemannian response but produces insufficient clearance in the coupled air-ground task. A dense analytical Riemannian field improves geometric avoidance, yet its high evaluation cost prevents stable field-of-view maintenance. An adapted ColAG controller achieves the lowest latency but still incurs safety and observability violations. VIRGA completes all paired warehouse conditions safely, while a long-range cave stress test without retraining demonstrates sustained coordination in irregular and confined geometry. Ablations confirm contributions from online geometric evaluation, virtual-agent mediation, and reciprocal feedback. |
| 2026-09-18 | [CASCADE Against Jailbreaks: Combination Across Stages with Controlled Attack-Defense Evaluation](http://arxiv.org/abs/2609.21793v1) | Jiale Luo, Eric Han | Defenses against jailbreak attacks on Large Language Models (LLMs) operate at different pipeline stages, such as input modification or output guard, but it remains unclear which defenses to deploy at each stage and how to combine them. Prior empirical studies, fragmented by inconsistent attack-success-rate definitions and experimental settings, have evaluated defenses largely in isolation. Here we present the first systematic study, to our knowledge, of defense combinations both within and across pipeline stages, under a consistent threat model of direct, black-box, single-turn attacks. Our decision framework standardizes evaluation through a principled attack-success-rate formulation with controlled query budgets, together with explicit fairness rules. Across 19 attacks and 15 defenses, we find that no single defense is universally best, but well-chosen combinations achieve substantial safety with minimal utility degradation, yielding practical recommendations for layered defense pipelines. |
| 2026-09-18 | [CIBuzzBench: A Benchmark for Cross-Lingual Understanding of Chinese Internet Buzzwords](http://arxiv.org/abs/2609.21722v1) | Yifan Wang, Junyu Lu et al. | Chinese social media has generated a vast and continually evolving lexicon of internet buzzwords whose meanings are often non-literal and deeply rooted in local cultural and pragmatic contexts. Existing research has primarily focused on interpreting these buzzwords within Chinese, leaving largely unexplored whether LLMs can transfer such culturally grounded knowledge across languages and accurately convey the intended meanings in English. This cross-lingual capability is also critical for safety, as harmful expressions may obscure their offensive content through culture-specific homophony, euphemism, irony, or coded language. In this paper, we investigate the ability of advanced LLMs to understand Chinese internet buzzwords across languages. To this end, we introduce CIBuzzBench, the first benchmark for cross-lingual Chinese-to-English understanding of Chinese internet buzzwords. CIBuzzBench comprises 3,001 Chinese internet buzzwords annotated with English meaning explanations, English equivalents, category labels, and harmfulness labels. Based on these annotations, we design three evaluation tasks: Meaning Explanation, Cross-lingual Equivalent Matching, and Culturally Grounded Harmfulness Detection. We evaluate representative state-of-the-art proprietary and Chinese LLMs under both English- and Chinese-prompting settings. Our results show that LLMs continue to struggle with the cross-lingual understanding of Chinese internet buzzwords, particularly in fine-grained non-literal interpretation, robust equivalent matching under option perturbations, and calibrated harmfulness detection. These findings highlight the persistent challenges posed by culturally grounded language phenomena for multilingual LLMs and safety-oriented evaluation. The dataset and code are available at https://github.com/SuperYFan/CIBuzzBench. |
| 2026-09-18 | [TERMon: Detecting Persistent Behavioral Threats in Edge AI via Hardware-Native Ternary Runtime Monitor](http://arxiv.org/abs/2609.21713v1) | Arish Sateesan, Edlira Dushku | Edge AI accelerators are increasingly deployed in safety-critical environments, where model outputs may control physical actuators, make access-control decisions, or trigger alarms. In these settings, runtime failures often remain undetected because model corruption, distribution shift, and adversarial inputs can still produce well-formed, confident predictions. This paper presents TERMon, a lightweight hardware runtime monitor that detects such anomalies by observing inference behavior rather than re-executing or formally verifying the model. TERMon represents class-conditional trusted behavior as hardware-efficient ternary patterns that are matched in parallel against a thermometer-encoded fingerprint. The ternary encoding reproduces the corresponding unquantized range decision exactly. TERMon detects harmful weight corruptions in proportion to their behavioral impact, while out-of-distribution and adversarial inputs are largely not separable using the monitored features at a strict false-positive operating point. We implemented TERMon on a PYNQ-Z2 FPGA, and the pipelined design requires no on-chip block RAM or DSPs and has a two-cycle decision latency. |
| 2026-09-18 | [RAYA: Learning Where and When to Intervene for Robot Recovery](http://arxiv.org/abs/2609.21690v1) | Ishaan Mahajan, Charles Chen et al. | A robot can predict failure and still be unable to prevent it. By the time a safety mechanism reacts, the nominal plan may already have spent the control authority that recovery requires, and fixed task priorities may block whatever response remains. Our key insight is that both aspects are decided inside the controller. Recoverability must inform actions while they are chosen rather than veto them afterward, and task objectives must be adapted as recoverability shrinks. Building on this, we present RAYA, a hybrid learned-analytic framework that places a learned finite-horizon recoverability margin inside an optimal controller with hard constraints and pairs it with a bounded learned scheduler that shifts task weights to facilitate recovery. Across 7,200 simulation episodes per controller spanning quadrotor and autonomous-vehicle benchmarks, RAYA not only improves survival rates, but also transfers the learned components zero-shot to unseen trajectories, disturbances, plant shifts, and friction layouts. We developed an embedded realization of RAYA and deployed it on-board a 35g Crazyflie quadrotor. Across 40 combined hardware flights under wind with either aerodynamic mismatch or an unmodeled 40% motor-command loss, each of three baselines fails in all trials, while RAYA completes 10/10 six-cycle missions. Project Website: https://raya-control.github.io/. |
| 2026-09-18 | [CounterPlay: Counterfactual Post-Training for Self-Play Driving Policies](http://arxiv.org/abs/2609.21617v1) | Jiarong Wei, Yin Wu et al. | Self-play in high-throughput simulators yields driving policies with robust closed-loop performance, but improvement per unit of simulation diminishes as training scales. Policies learn to handle common situations early, while further rollouts repeatedly encounter unresolved failures. Post-training offers an opportunity to target these failures, but existing methods primarily evaluate alternative actions or continuations at visited states, although successful recovery may require changing driving style earlier. We propose CounterPlay, a counterfactual self-play post-training approach that backtracks from failed tasks and retries them under alternate driving styles. CounterPlay rests on three key components. First, failure-driven backtracking uses the policy's value estimates to select an earlier stored state from which to retry the task. Second, reward conditioning enables a single policy to retry the task from this state using candidate styles ranging from cautious to aggressive. Third, CounterPlay retains task-completing retries only if no other vehicle incurs a new or earlier collision or off-road event relative to the factual branch. Retries that pass verification with fresh randomness are then distilled into the policy under its deployment condition. On BehaviorBench, CounterPlay achieves state-of-the-art scores on both the Interactive and Random splits across all eight traffic regimes using 1B post-training transitions, which is just 1% of the anchor's 100B self-play training budget. Improvements over the anchor hold across all three evaluated driving styles. CounterPlay resolves a substantial fraction of the anchor's timeout cases on BehaviorBench and achieves a balance between task completion and safety that neither continued self-play nor adopting a more aggressive driving style attains. |
| 2026-09-18 | [Tilt as a Certified Resource: Preserving Motor Wrench-Rate Authority on Articulated Multirotors](http://arxiv.org/abs/2609.21580v1) | Giuseppe Silano, Martin Saska | Fully-actuated multirotor aerial vehicles must not only track nominal wrenches but retain the "readiness" to modulate them rapidly under disturbances. Classical effort-minimizing allocators ignore this dynamic limit, whereas maximizing readiness leads to topologically disconnected optimal sheets demanding physically impossible actuator rates. Enforcing a readiness safety floor on fixed-geometry symmetric platforms further encounters a zero-sum degeneracy: motor-speed redistribution cannot improve authority without conceding wrench tracking. This paper uses active morphology to break the degeneracy, treating servo tilt as a geometric resource supplying authority-recovery directions unavailable to static rotors. We construct a configuration-dependent, motor-only readiness certificate - the log-volume of the reachable wrench-rate set - that explicitly excludes servo capacity, preventing a "ghost capacity fallacy" in which the certificate would falsely credit slow mechanical kinematic limits instead of collapsing accurately at motor saturation. The certificate is enforced as a Control Barrier Function (CBF) within a Unified Physical-Command Quadratic Program acting on motor torques and servo setpoints. Closed-loop simulations of an articulated octorotor under severe gust disturbances show classical allocators diverging and uncertified articulated allocators violating the safety floor, while the proposed CBF filter bounds the system state and preserves vehicle authority. |
| 2026-09-18 | [ServeGuard: Verifiable, Bounded-Residual Confinement of Operator-Invisible Channels Without Revealing the Certified Read Factor](http://arxiv.org/abs/2609.21515v1) | Dominik Dahlem, Rui Vieira | Third-party adapters for open-weight language models ship as opaque weight matrices; a recipient cannot check whether an adapter hides a backdoor without trusting the publisher or inspecting the weights, the publisher's core asset. For one important class (payloads placed where a safety monitor is structurally blind), detection is unsound as a defense: every detector that factors through the declared monitor is invariant on its blind subspace, and honest and backdoored adapters overlap on every blind-subspace statistic we evaluate, because benign adaptation uses that subspace too. Rather than detect this channel, we make it structurally \emph{absent} and prove that we did. The publisher builds the adapter to read the input only through directions the monitor covers and proves this in zero knowledge, revealing nothing about the read factor it certifies. The certificate is cheap because the expensive part, identifying the monitor's blind spot, is a deterministic function of the \emph{public} base model, so only one linear identity is proved; the served residual is the base model's own public floor, not a prover-chosen tolerance. The result is \emph{ServeGuard}, a supply-chain primitive: the publisher ships a \emph{proof-carrying adapter} whose proof lets a consumer or regulator verify, without the certified read factor and without trusting the publisher, that the adapter carries no hidden channel of this class relative to the declared monitor; an admission-time typing guard binds the guarantee to the adapter bytes admitted at serving time. Across eight checkpoints up to 7B from four families, the monitoring budget is architectural: the measured frontier saturates at the value-path rank on grouped-query checkpoints but not on multi-head ones. On a 0.5B model confinement is nearly free for benign adaptation, making monitor quality the security lever. |
| 2026-09-18 | [DPed-VLN: A Benchmark for Socially Compliant Vision-and-Language Navigation in Dynamic Pedestrian Environments](http://arxiv.org/abs/2609.21504v1) | Haojie Dai, Xiangyi Wang et al. | Vision-and-language navigation (VLN) has advanced rapidly in static indoor environments, but robots operating in human-populated spaces must ground language while responding to moving pedestrians and social-safety constraints. We present DPed-VLN, a Habitat 3.0 benchmark for dynamic-pedestrian VLN that couples 33,093 navigation episodes with paired global and prior-augmented instructions, ORCA-controlled humanoid pedestrians, socially constrained expert paths, and metrics that jointly assess navigation efficiency and social safety. DPed-VLN separates ordinary goal-oriented route guidance from prior-augmented instructions that expose dynamic-pedestrian cues for controlled analysis. To instantiate the benchmark, we introduce DPet (Dynamic Pedestrian-aware Network), a pedestrian-aware policy network trained with reinforcement learning and imitation learning. We further adapt representative state-of-the-art VLM-based navigation models, including NaVILA and StreamVLN, to DPed-VLN through LoRA fine-tuning. Experiments show that LoRA adaptation improves zero-shot VLM baselines in several success and safety metrics, especially reducing StreamVLN's collision rate. Among the evaluated methods, DPet-RL achieves the highest SR, SPL, and STL. |
| 2026-09-18 | [HE-Guardrail: A Homomorphic Guardrail Against Jailbreak Attacks for Encrypted Large Language Model Inference](http://arxiv.org/abs/2609.21484v1) | Byeongseo Min, Yongwoo Lee et al. | Homomorphic encryption (HE) has emerged as a promising approach to privacy-preserving machine learning (PPML), enabling computation directly over encrypted data. In HE-based PPML, a client submits an encrypted input to the server, which evaluates models such as large language models (LLMs) without access to the underlying plaintext. However, we identify a critical security vulnerability in this setting: HE-LLM inference is vulnerable to malicious clients that submit adversarial prompts, such as jailbreak attacks. The same confidentiality that protects benign clients also prevents the server from inspecting incoming prompts or generated responses, making adversarial attempts difficult to detect or block and potentially allowing successful attacks to remain entirely invisible to the server. To address this vulnerability, we propose HE-Guardrail, a framework that evaluates guardrail mechanisms entirely over encrypted data and homomorphically controls whether the target-model response is returned to the client. We instantiate HE-Guardrail with three representative guardrails - Llama Guard, JBShield, and GradSafe. Our results show that HE-Guardrail closely reproduces the decisions of the corresponding plaintext guardrails in the encrypted domain, with distinct security-efficiency-utility trade-offs. |
| 2026-09-18 | [Risk-Aware Occupancy for Safety-Oriented End-to-End Autonomous Driving](http://arxiv.org/abs/2609.21470v1) | Jiaxing Chen, Hengduo Zou et al. | Sparse representation formulates the environment perception for the end-to-end driving system as a set of discrete elements like objects and lane lines. This formulation meets safety risks in crowded, occluded scenes dealing with unstructured obstacles, uncertain regions, and intricate interactions. In this paper, we propose a dense representation, risk-aware occupancy, to characterize planning-relevant risks in an explicit and uniform manner. It jointly encodes global scene occupancy, map-derived traffic constraints, and future dynamic agent occupancy into a unified BEV map. The unified BEV map captures the risk evidence for trajectory planning in both spatial and temporal dimensions. We design an E2E network, ROIDrive, to realize risk-aware occupancy. It predicts risk-aware occupancy with an independent branch and injects it into planning queries for safety-oriented trajectory generation. In addition, to quantify the safety problem, we introduce RiskOcc4D-nuScenes built upon nuscenes and occ3d-nuscenes. Our risk-aware occupancy yields relative open-loop collision reductions of 52.9% under the UniAD metric and 35.0% under the ST-P3 metric on nuScenes. |
| 2026-09-18 | [Robotic Multiphase Interaction: Manipulating Coupled Liquid and Solid Dynamics with a World Model](http://arxiv.org/abs/2609.21448v1) | Yixuan Feng, Peng Wang | This work presents \textit{Robotic Multiphase Interaction (RMI)}, a setting in which liquid enters a porous material and interacts mechanically with its deforming solid skeleton. Manipulation can therefore change pore volume, expel or redistribute retained liquid, and alter grasp stability at the same time. Spilled liquid can also create safety risks in domestic and manufacturing settings. This differs from most manipulation of solid objects and from tasks that involve both liquid and solid while keeping the phases spatially separate. We study a sponge filled with water as the first RMI example. We use implicit incompressible porous flow with smoothed particle hydrodynamics as the dynamics engine and enable robotic manipulation by adding Coulomb contact memory, hybrid velocity and force regulation, and a stability gate for lifting. The resulting environment connects robot commands to changes in the coupled liquid and solid state. A world model conditioned on actions predicts how this state evolves under candidate commands, while a temporal UNet generates actions using either Diffusion Policy or rectified flow matching. Our world model reduces retained water prediction error by more than $60\%$ compared with the baseline. The best action sequence selected by the world model from policy proposals further reduces the predicted terminal water error by about half. These improvements show that modelling the coupled liquid and solid state helps the robot predict how its actions affect both the porous object and the liquid held inside. |
| 2026-09-18 | [Hiding in Plain Sight: A Diffusion-based Mitigation of Geolocation Privacy Leakage in Vision-Language Models](http://arxiv.org/abs/2609.21363v1) | Yining Wang, Xi Li et al. | Multimodal large reasoning models (MLRMs) have demonstrated remarkable capabilities in complex visual understanding. However, this very power introduces a critical yet underexplored privacy threat: adversaries can exploit MLRMs to precisely infer users' geographic locations from casually shared photographs, by performing structured reasoning over subtle visual cues such as architectural styles, vegetation, and lighting conditions. In this work, we present a systematic study of MLRM-driven geolocation privacy leakage. We first reveal that refusal-based safeguards are critically insufficient, as carefully crafted jailbreak prompts can raise model response rates to 100%. We further identify that existing defenses, which inject imperceptible perturbations into shared images, suffer from structural limitations intrinsic to their pixel-space optimization, resulting in degraded black-box transferability and pronounced visual artifacts. Motivated by these findings, we propose a diffusion-based framework that provides targeted, proactive defense against geolocation privacy leakage. By injecting perturbations into the latent space of a diffusion model during reverse sampling, our method operates directly on high-level semantic representations, thereby resolving the effectiveness-utility bottlenecks by construction. We further ground our optimization with GeoCLIP, a model explicitly aligned with GPS coordinates, as a surrogate to pinpoint and disrupt the geographic signals that MLRMs exploit for location inference. This targeted semantic disruption yields significantly stronger black-box transferability while preserving perceptual image quality, offering a seamless integration on social media platforms. |
| 2026-09-18 | [AirSplan: Risk-Aware Motion Planning for Quadrotors in Cluttered 3D Gaussian Splats](http://arxiv.org/abs/2609.21226v1) | Seth Isaacson, William Hong et al. | Quadrotors are increasingly deployed in applications such as agriculture, infrastructure inspection, and maintenance. In each of these applications, the robot must navigate complex scene geometry while remaining strictly collision-free. Unlike in ground domains, even minor collisions for aerial vehicles can result in the loss of the robot. This safety requirement induces a pair of technical challenges. First, the environment must be represented with sufficient fidelity to encode complex structure, even when no ground-truth obstacle data is available. Second, a motion planner must leverage this representation to determine a collision-free path to the goal. This paper proposes a system that addresses these complementary challenges. The proposed method, AirSplan, adopts a normalized variant of 3D Gaussian Splatting that encodes high-fidelity scene geometry. It then applies a novel reachability-based motion planner that leverages the differential flatness of quadrotors to compute continuous-time collision constraints that tightly overapproximate the robot's occupancy. Experiments demonstrate that AirSplan successfully finds a path in 81.2% of challenging test cases, a significant improvement over the nearest baseline method's 51.2%. |
| 2026-09-18 | [SafeStage: Evaluating Safety Before, During, and After Vision-Language-Conditioned Robot Manipulation](http://arxiv.org/abs/2609.21223v1) | Jinzhu Luo, Qi Zhang et al. | Vision-language-conditioned robot policies integrate perception, language understanding, and control for general-purpose manipulation. However, existing evaluations often focus on task success, isolated physical constraints, semantic refusal, or realized physical damage, providing limited insight into where safety fails during closed-loop manipulation. We introduce SafeStage, a lifecycle-structured benchmark for evaluating manipulation safety before, during, and after task execution. SafeStage contains 97 purpose-built risk scenarios organized into three stages. Initial-State Hazards captures safety-relevant relations that must be resolved before manipulating the target. Execution-Time Safety evaluates unsafe contacts, trajectories, region entries, and object interactions during execution. Final-State Hazards capture unstable or otherwise unsafe conditions remaining after nominal task completion. The benchmark evaluates realized interactions using event-based and state-based checks and reports native task success independently from stage-specific safety outcomes. We evaluate representative direct-action Vision-Language-Action (VLA) policies and policies with world-model-based policies under a common closed-loop protocol. Our results demonstrate that nominal task completion frequently coexists with safety violations and that different policies exhibit distinct failure profiles across the three stages. By separating task success from safety and localizing when violations occur, SafeStage provides a unified diagnostic testbed for evaluating and improving vision-language-conditioned robot manipulation policies. |
| 2026-09-18 | [Stochastic Neural Signed Swept Volume for Real-time Chance-Constrained Trajectory Optimization](http://arxiv.org/abs/2609.21211v1) | Qingyi Chen, Kevin Zhang et al. | Collision-free motion planning requires reliable collision models from sensed environments and validation of states along a continuous trajectory. To make this tractable, most planners check for collision at discrete states along continuous trajectories against a single determinized model of the environment, introducing a trade-off between safety and computational efficiency. While continuous collision checking approaches that approximate the swept volume of the robot exist, they are computationally expensive or overly conservative. Data-driven approaches can learn the swept volume; however, these neural models are susceptible to approximation errors and are therefore often limited to serving as coarse filters for downstream collision checkers. In this work, we propose to learn a signed distance function of the swept volume as a probabilistic field, enabling quantification of epistemic uncertainty, incorporation of perception noise, and eventual integration into a chance-constrained trajectory optimization framework. We demonstrate our approach on challenging high-dimensional manipulation problems with significant sensor noise, both in simulation and on real hardware. |

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



