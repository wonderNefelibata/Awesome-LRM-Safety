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
| 2026-08-14 | [Spatiotemporal Tube-Based Safety-Certificate for Autonomous Navigation of Articulated Vehicles](http://arxiv.org/abs/2608.14531v1) | Mohd. Faizuddin Faruqui, Ratnangshu Das et al. | Articulated vehicles are the workhorses of freight transportation, and their autonomous navigation is challenging. Their physical characteristics and motion constraints pose significant challenges in manoeuvring these vehicles on narrow routes. This paper presents a spatiotemporal tube-based approach to plan autonomous navigation of vehicles like tractor semi-trailers, truck/ tractor trailers, towing Automated Guided Vehicles (AGVs), and road trains. This planning approach provides a certified path plan for the truck or tractor, ensuring that the towed series of trailers always remains within the road corridor, limited by permissible corrections. The planning leverages the kinematics of the linked elements along with sway constraints to arrive at a safe tube for the actuated prime mover. We modify the spatiotemporal tube using permissible corrections to provide a route safety certificate to the vehicle for the given route. The proposed planning method is verified on a truck-trailer navigation simulation for a complex route. |
| 2026-08-14 | [Ensuring Safe Physical AI in Urban Mobility via Hazard-Informed Synthesized Envelopes](http://arxiv.org/abs/2608.14481v1) | Alexei Odinokov, Rostislav Yavorskiy | As heterogeneous robotic systems deploy across diverse urban zones, maintaining safety amid complex human-robot interactions remains a critical challenge. We present a unified framework that bridges systematic hazard analysis and runtime enforcement using hazard-informed safety envelopes. Rather than treating safety as a static constraint isolated within individual software modules, we introduce a cross-layer safety transformation process spanning symbolic, spatial, and dynamic world models. We show how this representation naturally interfaces with physical AI runtime harnesses to guarantee safe urban mobility. |
| 2026-08-14 | [Control-Informed Constraint Adaptation in Minimum-Time Trajectory Planning for Autonomous Racing](http://arxiv.org/abs/2608.14448v1) | Ann-Kathrin Schwehn, Alexander Langmann et al. | Autonomous racecars operate at the limits of vehicle dynamics, where small control errors translate into safety-critical behavior and lost performance. Trajectory planners assume perfect tracking and remain blind to execution errors. To guarantee safety, trajectory planners therefore restrict themselves to conservative spatial margins, leaving usable track space untapped. To overcome these issues, we introduce a control-informed online trajectory planning framework that learns from its own execution errors. By measuring systematic tracking deviations during runtime, we dynamically adapt spatial track constraints and iteratively expand the free-space planning area. The planner remains time-optimal while compensating for accumulated execution errors. This method was analyzed in a high-fidelity closed-loop simulation environment with autonomous racecars. The results demonstrate that our approach reduces lap time by 1.8\,s without increasing computational burden, maintaining a median runtime of 25 ms. Our finding indicates that feeding control-induced deviations back into the planning layer unlocks performance previously inaccessible to modular architectures and enables autonomous vehicles to exploit track limits systematically. |
| 2026-08-14 | [Tripwire: Triggering Aligned Refusal via Statistically Certified Safety Neurons](http://arxiv.org/abs/2608.14392v1) | Wei Zhao, Zhe Li et al. | Neuron- and path-level interventions offer the finest-grained route to defending large language models (LLMs) against jailbreak attacks, yet existing methods fall short of this promise, i.e., they often compromise model utility significantly. Specifically, one line of work suppresses toxic neurons to erase harmful semantics, but since such semantics are distributed across the network, blocking every pathway forces a large intervention footprint. An alternative line of research focus on identify safety neurons using external classifiers. While promising, the existing approaches suffer from compromising neurons that are important for the model utility as well. Moreover, both approaches remain always on and thus perturb every benign request even when no attack is present. To address these limitations, we present \ours{}, a training-free defense that first identifies safety-specific neurons through per-neuron hypothesis tests under false-discovery-rate control together with a utility-specificity filter. Based on this identification, a trigger-style clamp holds the selected neurons at their harmful-conditional mean activations, injecting an internal harmful-input signal that triggers the refusal behavior learned during alignment. The clamp is then realized by two provably equivalent deployment modes, namely a detector-gated inference-time intervention and an offline bias-patch weight edit. Extensive experiments across four safety-aligned LLMs and four representative attacks demonstrate that \ours{} reduces the average attack success rate to at most 2.0\% while incurring a utility drop of only 0.5\% to 5.3\% on MT-Bench, the smallest among all defenses. Code is available at https://anonymous.4open.science/r/Tripwire-65C4. |
| 2026-08-14 | [AgentRewind: Recoverable Execution for Long-Horizon LLM Agents](http://arxiv.org/abs/2608.14380v1) | Yu Zhuang, Kefei Chen et al. | Many real-world tasks require LLM agents to interact with their environments over long execution horizons. Errors that occur early in execution may propagate through both the agent context and environment state, and their effects may be difficult to reverse through subsequent actions. Existing methods mainly seek to reduce such errors through plan refinement and safety checks but provide little support after errors occur. To enable recovery during long-horizon execution, we present AgentRewind, a runtime recovery framework that records aligned checkpoints of the agent context and controlled environment, allowing agents to return to an earlier state and resume execution with information from previous attempts. We also construct MettleBench, a benchmark for evaluating task completion and partial progress on long-horizon engineering assignments containing a series of related requirements. Experiments across tasks, multiple models, execution strategies, and agent harnesses show that AgentRewind improves task success rate and average checklist progress over the compared baselines. |
| 2026-08-14 | [CORAL: Curriculum-Optimized Reward Adaptation for LiDAR-Based Goal-Directed Urban Driving](http://arxiv.org/abs/2608.14332v1) | Anisa Saleem, Duksu Kim | Reinforcement learning is promising for autonomous urban driving, but long-horizon goal-directed navigation asks a policy to acquire several competing behaviors at once--reaching a distant goal, tracking a route, avoiding obstacles, obeying signals--and a fixed objective gives no order in which to learn them. This paper presents CORAL, which advances two schedules together: a five-stage curriculum that progressively lengthens routes and tightens behavioral constraints, and a stage-aware reward whose component weights shift emphasis from mission progress toward route following, safety, smoothness, and rule compliance as the task hardens. The policy is a multi-stream actor-critic network trained with Proximal Policy Optimization (PPO) in CARLA on a compact 99-dimensional state pairing a polar LiDAR histogram with vehicle telemetry, ego-frame route geometry, and traffic-rule indicators--no point-cloud encoder, no bird's-eye-view rasterization. Against two PPO baselines under an identical protocol, CORAL reaches the goal in all twenty evaluation episodes on the longest routes under the full set of behavioral constraints, where the baselines reach 5% and 10%; a factorial ablation shows that neither schedule alone matches their combination: removing either lowers both success and route completion, and disabling both drops success to 55%. Trained in one town, the policy transfers zero-shot to seven unseen towns, succeeding in 68-98% of episodes on routes of the same 100-150 m length, with mean lateral deviation below 0.35 m. |
| 2026-08-14 | [Sensor-Driven Mission Synthesis for UAV/UGV Swarms: A TB-CSPN Coordination Architecture with Hardware-Enforced Safety](http://arxiv.org/abs/2608.14306v1) | Uwe M. Borghoff, Paolo Bottoni et al. | This paper presents a coordination architecture for heterogeneous UAV/UGV swarms that synthesises mission actions from uncertain, multi-modal sensor evidence while preserving hardware-enforced safety at the actuation boundary. The approach combines radar, RF, acoustic, and visual observations with Topic-Based Communication Space Petri Net (TB-CSPN) orchestration to support incremental mission formation under partial and evolving information. Consultant agents transform sensor outputs into temporally bounded semantic tokens, while supervisor agents provide authorisation and policy-governed release of mission transitions. This separation between interpretation, coordination, and execution yields auditable decision paths, constrains non-determinism within the coordination layer through guards and synchronisation, and enables bounded-time integration of heterogeneous evidence. To improve resilience in contested environments, including cyber compromise, spoofing, jamming, and communication loss, the digital coordination layer is complemented by independent analogue safety envelopes that clamp or veto unsafe actuator commands issued to individual vehicles. A coastal-surveillance case study illustrates how the proposed architecture enables dependable, governed, and physically safe swarm coordination under operational uncertainty. |
| 2026-08-14 | [Designing Mobile and Wearable Sensor-Fused Conversational Agents for Health and Wellbeing](http://arxiv.org/abs/2608.14273v1) | Hansoo Lee, Pablo Fonseca et al. | Mobile and wearable devices increasingly collect continuous wellbeing data, including sleep, activity, heart rate, stress, blood glucose, and blood pressure. Yet access to such data does not automatically help people interpret their condition or change behavior. Many health applications remain dashboard-first, presenting charts, thresholds, goals, and alerts while leaving users to decide what a change means and what action should follow. Conversely, generic LLM-based conversational agents (CAs) can provide fluent advice, but without personal sensor grounding, they cannot detect individualized patterns or provide contextual guidance. This three-hour tutorial teaches participants how to move from passive monitoring to actionable wellbeing dialogue. Participants examine a dashboard that combines wearable health-data visualization with conversational-agent feedback, then use Wearable Sensor-Dialogue Wellbeing Agent Studio (WSDWAS) to simulate wearables, generate sensor snapshots, configure agent personas and prompt blocks, and compare dialogue styles. Grounded in Positive Computing, the tutorial emphasizes autonomy, competence, privacy, safety, and boundaries between wellbeing support and medical advice. |
| 2026-08-14 | [A Temporal Barrier Framework for Collision Avoidance in Multi-Agent Autonomous Aerial Vehicles](http://arxiv.org/abs/2608.14239v1) | Benedikt Barthel Sorensen, Mitchell Black et al. | Operating teams of autonomous aircraft in dynamic, uncertain, and potentially adversarial environments requires safety protocols that are reliable yet selective, and allow agents to fly in close proximity while making progress toward mission objectives. We introduce adversarial time-to-collision (aTTC), a risk metric that quantifies, for a given agent, how quickly any surrounding agent could reach it assuming adversarial intent. We embed aTTC into the control barrier function (CBF) framework, defining the barrier directly in time rather than distance or velocity. The resulting aTTC-CBF is inherently anticipatory: agents modulate their own velocity based not on whether a peer is on a collision course, but on how quickly one could reach collision given its dynamical constraints. A differentiable neural-network surrogate makes the aTTC computable in real time within a standard CBF quadratic program. Across long time-horizon simulations of 3D independent-pursuit and formation-flight scenarios, the aTTC-CBF achieves up to twice the waypoint progress at half the collision rate of a higher-order distance-based CBF baseline. |
| 2026-08-14 | [Structure-Guided Spatiotemporal Attention Graph Neural Network for Traffic Flow Prediction](http://arxiv.org/abs/2608.14177v1) | Xuanmian He, Can Li et al. | Deep spatiotemporal models integrating graph convolutions and attention mechanisms have demonstrated excellent performance in network-level traffic flow prediction, owing to their exceptional ability to capture complex spatiotemporal dependencies. Despite their predictive success, deployment of such models in safety-critical urban systems remains constrained by their inherent lack of transparency. Existing post-hoc diagnostic methods often struggle with spurious correlations and fail to unveil the intrinsic decision-making mechanisms governing traffic dynamics, resulting in suboptimal interpretability and limited operational trustworthiness. To address these challenges, this paper proposes the Structure-Guided Spatiotemporal Attention Graph Neural Network (SGSAN). Departing from traditional architectures that rely on unconstrained adaptive graphs, SGSAN explicitly learns a static Directed Dependency Graph (DDG) to identify the invariant macroscopic propagation paths of traffic states. We further introduce an InfoNCE-based soft-coupling mechanism that anchors the model's dynamic spatiotemporal attention to this structural prior, offering a mechanistic account of the model's decision-making process while ensuring robust forecasting by aligning attention-based reasoning with identified macroscopic dependencies and preventing over-reliance on ephemeral local noise. Furthermore, a decoupled two-stage optimization framework is developed to resolve the fundamental conflict between structural discovery and predictive error minimization. Extensive experiments on multiple real-world datasets demonstrate that SGSAN achieves state-of-the-art predictive accuracy while providing built-in interpretability that organically aligns with the physical logic of traffic networks. |
| 2026-08-14 | [Birth of the Coil: another Milestone towards a fully reproducible low-field MRI scanner for head-imaging](http://arxiv.org/abs/2608.14139v1) | Umberto Zanovello, Julia Pfitzer et al. | Low-field magnetic resonance imaging (MRI) provides an accessible, portable, and low-cost alternative to high-field scanners, expanding diagnostic imaging to point-of-care settings. However, widespread adoption is fundamentally hindered by a severely reduced signal-to-noise ratio (SNR). At low frequencies, radiofrequency (RF) coil conductor losses - rather than tissue sample losses - predominantly govern the system's total noise, making meticulous RF coil optimization critical to recovering image quality. This work presents an open-source, optimized solenoid head coil tailored for the 50 mT open-source scanner (OSII ONE v2.1). The paper validates production reproducibility across three independent international institutions and introduce an open-source connector with integrated digital circuitry for coil identification and DC or logic signals. Comprehensive benchtop measurements, Electromagnetic Interference (EMI) coupling analysis, Specific Absorption Rate (SAR) safety simulations, and phantom and human volunteer imaging confirm the design's efficacy, safety, and reproducibility. The results of the paper, when combined with the material provided in the open-source dedicated repositories, set the basis for a fully reliable and reproducible component for the open-source OSII ONE MRI scanner. In addition, the same optimization strategy and design material can be exploited for designing other RF coils for imaging of other body parts. |
| 2026-08-14 | [Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety Classifiers](http://arxiv.org/abs/2608.14089v1) | Thiago Sandoval, Ufuk Topcu | Safety classifiers deployed with large language models often fail for two reasons: their decisions reflect the policy learned during training rather than the deployer's desired policy, and their performance degrades as deployment traffic evolves. We present Regime-Conditional Verification (RCV), a lightweight wrapper that adapts an off-the-shelf safety classifier without retraining it. RCV estimates, from the classifier's internal representations, the probability that each prediction disagrees with the deployer's policy, and selectively corrects predictions likely to be wrong. The same correctness estimates also provide a label-free signal for detecting distribution shift, enabling a maintenance loop that updates the correctness estimation layer and resorts to classifier fine-tuning only when necessary. Across three off-the-shelf safety classifiers and two benchmark datasets, RCV improves adherence to the deployer's policy in every classifier-dataset combination, catching up to 0.81 of previously missed unsafe content without modifying the underlying classifier. In a deployment study with ten attack campaigns, each a harm category held out of RCV's training, RCV detects every campaign in a dedicated injection panel; in the maintenance census most drift episodes are repaired without updating the classifier, and the fine-tune is reserved for the residual episodes that repair does not restore. |
| 2026-08-14 | [PILOT: Privileged Imitation Learning for End-to-End Motion Planning of Autonomous UAVs under Partial Observability](http://arxiv.org/abs/2608.14082v1) | Qingrui Zhang, Feng Xue et al. | Autonomous navigation in cluttered environments is hampered by partial observability and dynamic constraints. This paper presents PILOT, a constraint-aware privileged imitation learning framework for vision-based end-to-end UAV motion planning under partial observability. The framework distills planning strategies from a computationally intensive optimal control expert into a student policy regularized toward safety and dynamic requirements via a dual-objective loss function. To mitigate partial observability, a spatiotemporal perception fusion module using a Temporal Convolutional Network (TCN) is developed to integrate historical depth images and odometry. This module infers task-relevant latent context from historical observations, enhancing spatial awareness beyond the instantaneous FOV without maintaining persistent map memory. A trajectory parameterization layer mapping network outputs to a structured trajectory, while enabling explicit continuity, dynamic-consistency, and obstacle soft penalties during training, encouraging constraint satisfaction for unseen observations without formal guarantees. Simulations on quadrotor and fixed-wing aircraft demonstrate that PILOT achieves performance comparable to the privileged expert while reducing computational overhead by over 80\%. Successful indoor and outdoor zero-shot deployment confirms the practical feasibility and cross-domain generalization of the planner. |
| 2026-08-14 | [MACS: A Hybrid Multi-Agent Framework for Reliable Conversational E-Commerce Recommendation](http://arxiv.org/abs/2608.14068v1) | Juli Huang, Hannah Clay et al. | Conversational recommendation for e-commerce is increasingly mediated by large language models (LLMs), yet many real-world deployments operate under a stricter requirement: recommendations must be drawn only from a merchant's fixed catalog, without web search or unsupported product claims. In this setting, the main challenge is reliability under hard constraints: the system must satisfy user requirements, remain grounded in available inventory, and preserve preferences across multiple conversational turns. We present MACS (Multi-Agent Commerce System), a hybrid multi-agent framework for reliable conversational recommendation in fixed-catalog settings. MACS uses LLMs for language-facing tasks such as interpreting user requests, eliciting preferences, and generating responses, while correctness-critical operations, including product retrieval, hard-constraint filtering, brand exclusion, and progressive relaxation, are executed deterministically by the merchant agent. A session-persistent preference layer tracks constraints across turns, enabling consistent handling of budget overwrites and exclusion reversals. On a 140-query single-turn benchmark, MACS achieves the highest pass rate (87.1%) and perfect brand compliance (1.000). On a 10-scenario multi-turn benchmark, MACS achieves the strongest macro Pass@5 (72% vs. 56% GPT+Catalog / 52% Gemini+Catalog) with zero constraint drift. The advantage is sharpest on exclusion reversal (100% vs. 20% / 0%) and constraint accumulation (100% vs. 60% / 40%). Mean judged response quality is similar across systems (0.751 vs. 0.736). These results suggest that hybrid architectures combining deterministic constraint enforcement with session-persistent preference tracking provide stronger reliability-oriented performance than catalog-bound prompt-only baselines in the fixed-catalog merchant setting. |
| 2026-08-14 | [SSP: An Event-Matched Syn2Sim2Phy Cross-Domain Evaluation Framework for Autonomous Driving VLA Models](http://arxiv.org/abs/2608.14024v1) | Haojie Feng, Peizhi Zhang et al. | Vision-language-action (VLA) models for autonomous driving jointly produce scene interpretation, language-based reasoning, and driving trajectories. Existing evaluations often use independently selected synthetic, simulated, and physical data, so measured performance gaps can be confounded by changes in scenario content rather than genuine domain sensitivity. We propose SSP (Synthetic-Simulation-Physical), an event-matched Syn2Sim2Phy evaluation framework that anchors cross-domain comparison to the same safety-critical interaction. Starting from a synthetic long-tail video, SSP builds a validated event specification that preserves road topology, participant roles, relative motion, conflict evolution, passing order, response constraints, and event phases. Platform-specific realizations are then constructed in CARLA and on a closed proving ground and are evaluated only after transfer audits confirm preservation of mandatory event properties. SSP maps heterogeneous outputs from OpenEMMA, LLaViDA, and Alpamayo-R1 into common semantic slots and a 1 s trajectory window to assess output validity, semantic accuracy, critical-interaction recognition, trajectory quality, and risk response. Across Cut-in and vulnerable-road-user crossing cases, the macro-averaged Integrated VLA Capability Scores are 0.259, 0.291, and 0.325 in the Synthetic, Simulation, and Physical domains, respectively, while the best domain varies by scenario. Alpamayo-R1, OpenEMMA, and LLaViDA obtain scores of 0.405, 0.338, and 0.131. SSP provides a reproducible scene-transfer chain and an evidence-qualified evaluation of VLA behavior without assuming that the Physical domain is universally superior. |
| 2026-08-14 | [Batch-wise Adaptive Pruning: Periodic Neuron Activation-Aware Weight Pruning for Language Reasoning Model](http://arxiv.org/abs/2608.14003v1) | Yongmin Kim, Shota Takashiro et al. | Large Reasoning Models (LRMs) achieve strong performance on complex tasks through extended chain-of-thought generation, but incur substantial computational costs during inference. In production settings, batched inference is essential for high throughput, yet the existing training-free adaptive pruning methods we evaluate severely degrade in this regime. Because a batch must share a single pruning mask, these methods aggregate activations across samples and then apply threshold-based selection; the threshold, calibrated offline on unaggregated activations, no longer matches the aggregated distribution, so the realized sparsity ratio drifts and accuracy on reasoning tasks collapses under batched inference.   In this work, we propose a training-free adaptive pruning method designed specifically for batched inference in LRMs, built on two components. First, we replace threshold-based selection with periodic top-k selection over the aggregated importance scores, which is unaffected by the shift that aggregation induces in the activation distribution, and which runs selection once per update period rather than at every token, preserving the speedup. Second, based on the observation that important neurons re-fire periodically during long reasoning generation, we introduce an activation memory that accumulates importance across update phases so that recurring neurons are retained.   Experiments on diverse reasoning benchmarks demonstrate that our method outperforms the previous state-of-the-art adaptive pruning method by 39.7 percentage points in average accuracy at batch size 4 with 50% target sparsity on DeepSeek-R1-Distill-Qwen-7B, and reaches 1.40x speedup over dense inference at 50% actual sparsity. |
| 2026-08-14 | [Knowledge-Data-Dual-Driven Reinforcement Learning for Autonomous Vehicle Control in Mixed Traffic](http://arxiv.org/abs/2608.13878v1) | Jie Fang, Wei Zheng et al. | In mixed traffic, decision-making for autonomous vehicles (AVs) confronts three interrelated challenges. First, physics-based priors incorporated into reinforcement learning (RL) models fail to capture latent interactive vehicle intentions and diverse driver behaviors, limiting the proactive reasoning capabilities. Second, abrupt maneuvers by surrounding vehicles cause non-stationarity, leaving long-tail safety events under-explored. Third, hybrid action spaces destabilize unified RL training due to the different temporal scales of continuous car-following and discrete lane-changing maneuvers. To address these issues, we propose Knowledge-Data Dual-driven Reinforcement Learning (KDDRL). First, a conditional deep generative model synthesizes intention-aware future trajectories, converting passive perception into proactive predictive states. Second, a knowledge-data dual-driven paradigm operates on these predictive states, fusing probabilistic data-driven insights with physical constraints to guide safe exploration through safety-critical scenarios. Third, a coupling module compresses both intention-aware trajectories and physical constraints into compact shared embeddings. This unified representation enables asynchronous multi-timescale optimization of continuous car-following and discrete lane-changing while preserving mutual information. Evaluations on dataset-calibrated simulations demonstrate that KDDRL effectively handles intention uncertainty, accelerates training convergence, and outperforms conventional baseline methods in terms of safety, efficiency, and comfort. |
| 2026-08-14 | [Real-Time In-Domain Congestion Control for the LWR Traffic Model via Control Barrier Functions](http://arxiv.org/abs/2608.13841v1) | Zehang Zhu, Brian Block et al. | This paper presents a control barrier function-based method for real-time in-domain congestion control of the Lighthill-Whitham-Richards traffic model. Traffic congestion is formulated as a distributed safety control problem, leading to a infinite-dimensional optimization problem. Through the discretization and Karush-Kuhn-Tucker (KKT) analysis, the problem is converted into a high-dimensional quadratic program. A structure-exploiting primal-dual active set algorithm is then developed to compute the safe control input in real time, with convergence guarantees. Numerical simulations with different nominal controllers demonstrate the effectiveness and real-time feasibility of the proposed approach. |
| 2026-08-13 | [Control Barrier--Value Functions under Partial Observability: Safety Guarantees via Conformal Prediction](http://arxiv.org/abs/2608.13819v1) | Niloofar Jahanshahi, Mo Chen | This paper studies safety analysis and controller synthesis for partially observable nonlinear control systems. We extend the control barrier--value function (CBVF) framework, which combines Hamilton--Jacobi reachability and control barrier functions, to settings where full state information is not available and control is based on an estimated state. Given an estimator, we apply conformal prediction to the estimation error and obtain an error bound at a user-chosen miscoverage level. We incorporate this bound into the estimator-space safety analysis and define a CBVF-based safety certificate for partially observable systems. We then derive a finite-horizon probabilistic safety guarantee for the true system state. Finally, we propose a QP-based online safety filter for systems affine in the control and disturbance, whose solution enforces the CBVF safety condition in real time against bounded disturbance. The proposed framework is illustrated on a partially observable obstacle-avoidance case study. |
| 2026-08-13 | [GPU Offload in Rust: Portable, Safe, and Fast](http://arxiv.org/abs/2608.13759v1) | Manuel S. Drehwald, Marcelo Domínguez et al. | High-performance GPU programming has traditionally forced a compromise between execution efficiency and memory safety. While Rust guarantees compile-time memory safety for host CPUs via its strict ownership model, applying these constraints to massively parallel GPU execution environments has previously mandated either vendor-locked Domain-Specific Languages (DSLs) or escaping to explicit unsafe raw pointers. This paper presents a zero-overhead, multi-vendor GPU compilation framework built natively into the Rust compiler (rustc) and LLVM backends.   We leverage Rust's rich type system, ownership system, and strict aliasing guarantees (noalias) to efficiently manage and optimize data transfers through LLVM's Offload infrastructure. We expose the technical challenges of cross-vendor ABI lowering mismatches between Host and Device targets and introduce a two-pass compilation pipeline capable of safely handling both manual and compiler-generated memory movements. Evaluating our framework on RAJAPerf demonstrates that our rustc-based solution can generate competitive LLVM IR for GPU kernels, achieving a solid kernel performance against native, hand-optimized CUDA and HIP C++ baselines. |

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



