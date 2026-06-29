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
| 2026-06-26 | [CacheMPC: Certified Cached Model Predictive Control for Quadruped Locomotion](http://arxiv.org/abs/2606.28300v1) | Nimesh Khandelwal, Mehul Anand et al. | Model Predictive Control (MPC) is the standard predictive layer in hierarchical quadruped controllers, but the per-cycle QP solve limits the update rate achievable on embedded processors. Because legged gaits revisit a bounded region of state space, MPC solutions admit caching and reuse. This paper proposes \emph{Certified CacheMPC}: a Locality-Sensitive-Hashed cache of horizon contact-force trajectories, partitioned by contact mode, retrieved at query time and accepted only when an a-posteriori per-query certificate confirms primal feasibility and a Lagrangian dual-gap upper bound on cost suboptimality. A bounded-budget controller schedule combines top-$K$ certified retrieval, a deadline-bounded QP solve, and a shifted last-certified fallback. The framework is evaluated on a Unitree Go2 across $2{,}038$ usable cold-controller MuJoCo trials, including a $600$-trial $n\!=\!50$ campaign at three failure-boundary cells, and a first-deploy session on the on-robot NVIDIA Orin NX. The un-gated cache delivers a $25\times$ median solve-time speedup in simulation and an $18.7\times$ median speedup on hardware. At $n\!=\!50$ no statistically significant difference in closed-loop stable rate is detected between the cache variants and the no-cache baseline at any tested cell. The certificate's contribution to closed-loop safety is not resolvable at the present sample size. |
| 2026-06-26 | [CPAgents: Agentic Composite Phenotype Generation for Cardiac Disease Association](http://arxiv.org/abs/2606.28179v1) | Zuoou Li, Wenlong Zhao et al. | Identifying robust associations between cardiac imaging phenotypes and clinical diseases is fundamental to population-scale cardiovascular research and reliable risk stratification. However, current phenome-wide association studies rely on pre-defined, single-variable phenotypes or expert-crafted features, which limits their ability to capture clinically meaningful non-linear effects and cross-phenotype interactions. To address this, we propose CPAgents, an iterative phenotype-Composition framework for cardiovascular Phenome-wide association study (PheWAS) that automatically constructs and validates interpretable composite phenotypes (e.g., polynomial, ratio, and interaction forms) from base imaging features. Specifically, our system coordinates three agents: (i) an Analyst that identifies statistical pathologies and nominates candidate transformations; (ii) a Proposer that generates constrained, medically and statistically motivated expressions under numerical safety rules; and (iii) a Verifier that evaluates candidates using multi-stage criteria and produces transparent evidence trails for accepted phenotypes. Evaluated on a population-scale cardiac imaging cohort, the discovered composite phenotypes markedly improve disease discrimination: across 72 classifier-disease-metric combinations, our variants achieve the top rank in 56 cases versus 18 for baselines, with gains observed across all nine clinical disease categories. Our framework yields compact, clinically interpretable phenotype formulas with transparent evidence trails, enabling scalable discovery of stronger phenotype-disease associations beyond expert-driven feature selection. |
| 2026-06-26 | [Robust Harmful Features Under Jailbreak Attacks: Mechanistic Evidence from Attention Head Specialization in Large Language Models](http://arxiv.org/abs/2606.28153v1) | Yanchen Yin, Dongqi Han et al. | Jailbreak attacks bypass LLM safety alignment, yet their mechanisms remain poorly understood. We provide evidence that attacks do not comprehensively eliminate safety features, but instead selectively suppress specific attention heads. We identify two functionally differentiated types: Adversarially Compromised Heads (ACHs) concentrated in early layers, which are suppressed under attacks, and Safety-Aligned Heads (SAHs) in mid-layers, which maintain robust activations even when attacks succeed. Ablation studies support the causal role of ACHs and the contribution of SAHs to robust activations: suppressing a small number of ACHs is sufficient to induce jailbreak-like behavior on normally refused inputs, while removing SAHs substantially weakens mid-layer safety activations. Token-level attribution further shows that ACH suppression is driven specifically by attack-template tokens, providing a mechanistic account of why attacks can bypass refusal decisions through ACH suppression while leaving internal safety signals sustained by SAHs -- a phenomenon we term Robust Harmful Features. To validate the practical significance of this robustness, we show that simply reading these persistent activations -- without any training -- yields competitive aggregate detection performance with strong adversarial robustness. |
| 2026-06-26 | [Regularized Reward-Punishment Reinforcement Learning](http://arxiv.org/abs/2606.28152v1) | Jiexin Wang, Eiji Uchibe | We propose KL-Coupled Policy Regularization (KCPR), a policy coordination framework for Reward-Punishment Reinforcement Learning (RPRL). Based on KCPR, we derive KL-Coupled Soft Optimality (KCSO) and develop its deep realization, klDMP. Unlike existing RPRL approaches that optimize reward-seeking and punishment-related policies largely independently, KCPR enables direct interactions between companion policies by treating each as a dynamically learned prior for the other. KCSO yields coupled soft-optimal policies and KL-regularized Bellman operators, allowing reward and punishment information to jointly influence value propagation. To improve learning stability, we introduce a companion-prior softening mechanism and evaluate separate replay-buffer designs for balancing reward- and punishment-related experience. Experiments in grid-world and Gazebo robotic navigation tasks demonstrate that klDMP improves safety and learning stability while maintaining competitive task performance compared with DQN, SQL and softDMP. These results suggest that policy-level coordination provides an effective mechanism for integrating multiple behavioral objectives and may serve as a useful design principle for reinforcement learning systems with interacting motivational processes. |
| 2026-06-26 | [Specification-aware Robustness Margins for Symbolic Controllers](http://arxiv.org/abs/2606.28143v1) | Youssef Ait Si, Antoine Girard et al. | We address the problem of robust controller synthesis for a class of linear temporal logic (LTL) specifications over families of perturbed systems using symbolic control techniques. Given a dynamical system, a specification, and a symbolic controller synthesized using the fixed-point algorithm of the specification, the objective is to find the maximal perturbation we can apply to the system while the system continues to satisfy the same specification under the same controller. We first provide general results, by demonstrating that controllers synthesized based on the symbolic model can be refined back to a perturbed version of the concrete system while preserving their correctness. Focusing on four fundamental temporal logic specifications, namely safety, reachability, persistence, and recurrence, we introduce a general measure of the maximal robustness margin. Then, for each class of specifications, we derive a customized version of the measure and establish the corresponding theoretical guarantees. Importantly, the robustness margin depends explicitly on the sequence of sets generated during the fixed-point computation, allowing for specification-dependent and less conservative bounds compared to generic abstraction-based approaches. The theoretical developments are illustrated on two examples, demonstrating the practical applicability and effectiveness of the proposed approach. |
| 2026-06-26 | [OperatorSHAP: Fast and Accurate Shapley Value Estimation for Neural Operators](http://arxiv.org/abs/2606.28065v1) | Joshua Stiller, Santo M. A. R. Thies et al. | Understanding model predictions is essential for physical applications, where outputs often inform safety-critical decisions, such as structural load assessment, weather warnings, and clinical diagnosis. Shapley values satisfy many desirable properties as an attribution method, but their computational cost during inference hinders their practical use. Current amortized explainers, such as FastSHAP, are limited to homogeneous inputs, which is problematic for physical applications where data often comes from irregular grids and geometries. We introduce OperatorSHAP, a grid-agnostic attribution method and training procedure that allows us to train FastSHAP-like explainers for neural operators. We establish a theoretical framework for attributions in function space, connecting to Aumann-Shapley values. We further show that OperatorSHAP's explanations are consistent with state-of-the-art discrete Shapley values across resolutions and transfer across grid sizes without retraining. |
| 2026-06-26 | [From Detection to Action: Using LLM Agents for Fault-Tolerant Control](http://arxiv.org/abs/2606.28011v1) | Javal Vyas, Milapji Singh Gill et al. | We propose an agentic Large Language Model (LLM) framework for active Fault-Tolerant Control (FTC) that transforms fault detection outputs into constraint-aware recovery actions grounded in plant-specific knowledge. The approach couples (i) a multi-agent workflow that decomposes operator duties into monitoring, planning, action synthesis, simulation, validation, and reprompting; (ii) a Digital Process Plant Twin (DPPT) that exposes plant data, models, and a simulation service for pre-execution testing; and (iii) a Graph Retrieval-Augmented Generation (Graph RAG) layer built on the CPSMod ontology, which organizes plant knowledge (structure, function, hybrid dynamics, control context, and fault semantics) into a graph that supports relation-aware, multi-hop retrieval for the agents. Corrective actions are generated as minimal-risk state-machine recovery paths and corresponding discrete commands or continuous setpoint adaptations, then validated deterministically against interlocks, envelopes, and dynamic feasibility before any actuation. If no acceptable plan is found within a bounded time window, control is handed to a safety fallback. The framework is evaluated in simulation on two representative benchmarks: a discrete batch Mixing Module and a Continuous Stirred-Tank Reactor (CSTR) under closed-loop PID regulation. Results with lightweight LLMs (GPT-4o-mini and GPT-4.1-mini) show that semantically grounded agents can derive valid recovery decisions within latency budgets compatible with the respective process dynamics, demonstrating a practical pathway from detection to validated corrective action across both discrete and continuous FTC tasks. |
| 2026-06-26 | [It Lied to a Doctor to Buy Poison Ingredients: Quantifying Real-World Misuse of Phone-use Agents](http://arxiv.org/abs/2606.27944v1) | Yiming Sun, Chen Chen et al. | Phone-use Agents can execute complex tasks end to end across real mobile applications. By operating a real device on the user's behalf, they reach far more functionalities than CLI agents, which amplifies the real-world harm they can cause when driven for malicious purposes. We present the first study of this threat on real phones and 27 commercial apps, and find that agents built on 9 mainstream commercial and open-source models readily carry out serious misuse, ranging from procuring drug and explosive precursors to fraud, online harassment, and review manipulation. Across the agents we run on real devices, the average refusal rate to harmful requests stays low while the average task-completion rate reaches 68.8%, and in some scenarios an agent finishes a violation faster than a human would. These results suggest that Phone-use Agents already meet the practical conditions for automated misuse at scale.   In one observed real-device execution, Claude-Opus-4.8 fabricated a medical history, deceived an online doctor into issuing a prescription, and completed the order and payment on its own to purchase a precursor for a highly toxic substance. To our knowledge, this is the first documented real-world case of an AI agent procuring controlled precursor materials. We trace this behavior to a Safety Awareness-Execution Gap, where an agent recognizes that a request is harmful yet still executes it. Simple defenses curb the overt cases, but the more covert and arguably more damaging threats, such as coordinated review manipulation and fake traffic, remain largely unsolved. We hope these findings push the community toward safer Phone-use Agents. |
| 2026-06-26 | [Combining Axiomatic Models for Refinement Proofs](http://arxiv.org/abs/2606.27916v1) | Suha Orhun Mutluergil, Alperen Dogan | Refinement proofs verify an implementation by showing that its behaviours are subsumed by a simpler specification, on which safety properties are easier to establish. We study how such proofs interact with the axiomatic program logics used to verify the specification. We first give a uniform account of Hoare, Incorrectness, Lisbon, and Necessary-Preconditions logic, classified by the direction in which each constrains a transition and by whether it over- or under-approximates its target set. We then show that simulation relations transfer state-based safety properties: a forward simulation carries a Hoare (inductive) invariant of the specification to one of the implementations, and forward and backward simulations both carry ordinary invariants, via the pre-image of the relation. Finally, we characterize, within these logics, when a relation is a simulation, forward simulations by the validity of Hoare or Lisbon triples, backward simulations by Necessary-Preconditions or Incorrectness triples, so that the simulation obligation reduces to a triple in an off-the-shelf functional logic. We illustrate the development with a concurrent counter, transporting a safety bound from an atomic sequential specification to a Left--Right implementation through an intermediate nondeterministic-concurrent counter, with a forward simulation on one side and a backward simulation on the other. |
| 2026-06-26 | [Drifting in the Future: Stabilizing Path Following Drifting on High-Latency Vehicle Systems](http://arxiv.org/abs/2606.27914v1) | Frederik Werner, Till Heintzenberg et al. | Autonomously controlling and handling a vehicle at and beyond its stability limit is a mathematically and computationally demanding task. Prior demonstrations of automated drifting have been limited to research platforms with instantaneous torque delivery and independently actuated wheels, leaving their applicability to production vehicles with actuator latencies and mechanically coupled axles uncertain. To overcome these issues, we design a predictor to compensate for powertrain delays, develop a revised control formulation to accommodate higher actuation latencies as well as a differential coupling on the driven axle, and introduce brake-based velocity stabilization. This paper presents the controller framework, the model extensions, and real-world experimental results. We observe that our controller enables a production sports car with a combustion engine to robustly sustain circular and figure-eight drifts, limiting lateral error to 1.1 m and sideslip overshoot to 0.06 rad despite actuator delays exceeding 250 ms, while mitigating oscillations and maintaining stable path and sideslip tracking. In conclusion, our results establish that autonomous drifting is feasible on production-ready vehicles, opening pathways to advanced safety systems capable of stabilizing cars in scenarios where traditional control fails. |
| 2026-06-26 | [Parameterized Verification of Asynchronous Round-Based Distributed Algorithms via Reduction to Finite-Counter Systems](http://arxiv.org/abs/2606.27867v1) | Nathalie Bertrand, Pranav Ghorpade et al. | Traditional model-checking techniques typically verify distributed algorithms only for a fixed number of finite-state processes. Parameterized model checking generalizes this to any number of processes, while still typically assuming that each process is finite-state. In this work, we consider asynchronous round-based distributed algorithms in which each process is infinite-state since it can execute for an infinite number of rounds.   We show that the parameterized verification problem for asynchronous round-based distributed algorithms is undecidable, already for simple specifications. Nevertheless, as our main contribution, we provide a reduction to LTL model checking over finite-counter systems and prove that it is sound and complete. This enables the use of off-the-shelf, mature symbolic model checkers for finite-counter systems. We demonstrate the practical applicability of this reduction by verifying safety and liveness properties of several asynchronous round-based consensus and leader-election algorithms using the nuXmv model checker. |
| 2026-06-26 | [PPO-EAL: Exact Augmented Lagrangian Proximal Policy Optimization for Safe Robotic Control](http://arxiv.org/abs/2606.27861v1) | Jiatao Ding, Songqun Gao et al. | Reinforcement learning (RL) has emerged as a promising solution to accomplish complex robotic control tasks; however, most of the current work ignores the safety requirements. Safe RL seeks to maximize task performance while satisfying explicit physical constraints, but current algorithms struggle to learn the policy efficiently with precise constraint satisfaction. This work proposes PPO-EAL, a novel first-order constrained policy optimization framework that integrates exact augmented Lagrangian optimization into proximal policy optimization for safe robotic control. By combining clipped policy updates with exact quadratic penalty terms, PPO-EAL achieves theoretically grounded constraint enforcement without requiring impractically large penalty factors. A momentum-regulated multiplier update further improves dual-variable stability, reducing constraint oscillation and unsafe behavior while preserving task performance. We provide exactness and convergence analysis under standard stochastic approximation assumptions. Extensive validation across diverse GPU-accelerated robotic benchmarks-including cart-pole balancing, cart-double-pendulum stabilization, 7-DoF Franka end-effector reaching, and quadrupedal locomotion-demonstrates superior safety precision and reward performance compared with state-of-the-art first-order safe RL baselines. Finally, we demonstrate zero-shot sim-to-real deployment in a contact-rich gear assembly task, where PPO-EAL substantially improves task success, reduces peak contact force, and enhances operational robustness. These results establish PPO-EAL as a general and practically deployable safe RL framework for diverse safety-critical robotic systems. |
| 2026-06-26 | [Methods for Uncertainty Representation in Risk Management: A Comparative Review and Decision-Oriented Framework](http://arxiv.org/abs/2606.27804v1) | Albert Kutej, Stefan Rass | The consideration of uncertainty is a central but frequently inadequately addressed component of risk management. A systematic treatment of uncertainty is essential for ensuring the quality and traceability of decision-making processes, particularly in complex and safety-critical environments. This review systematically analyzes how established risk management approaches conceptualize and represent uncertainty in both their theoretical foundations and practical applications. Based on a systematic literature review of 370 publications, the identified approaches are classified into five methodological families. These include probabilistic methods, evidence-based and fuzzy-logic approaches, qualitative elicitation techniques, graphical and visual representations and hybrid frameworks. The analysis shows that probabilistic methods remain predominant due to their quantitative rigor, whereas fuzzy and evidence-based approaches are particularly suited to addressing vagueness and epistemic uncertainty. Qualitative and graphical approaches are found to enhance interpretive understanding and support the transparent communication of uncertainty. Despite these developments, the analysis indicates that the practical integration of these approaches into operational risk management remains limited in many domains. The findings highlight the need for more structured guidance in method selection and suggest that future research would benefit from further development of hybrid approaches and visualization techniques. |
| 2026-06-26 | [Repair-before-veto control for safe lithium-ion fast charging under unknown ambient and cooling-fault conditions](http://arxiv.org/abs/2606.27781v1) | Yifan Wang | Fast charging is decisive for electric-vehicle adoption, but field chargers are deployed as one setting while the cell's true thermal state, ambient temperature, and cooling-system health are uncertain. A current that is safe for a healthy cell at room temperature can overheat the same cell when it is hot or its cooling is degraded. We formulate this as a single-setting, unknown-state safe-fast-charging problem and solve it with a margin-aware repair-before-veto controller (RACL-B). RACL-B requests an aggressive current and repairs it online to the tightest measured margin among terminal voltage, cell temperature, and negative-electrode lithium-plating overpotential, rather than committing to a fixed schedule or shutting charging down. We evaluate one deployed setting across nine conditions, spanning 10/25/40 $^\circ$C ambient temperature and 100/60/40\% cooling health, in a high-fidelity Doyle--Fuller--Newman model with partially reversible lithium plating and lumped thermal coupling. Under a strict 45.0 $^\circ$C peak-temperature audit, fixed and ambient-scheduled protocols overheat in five of nine conditions because neither observes hidden cooling degradation, and rigid protective shutdown fails to deliver the charge in every condition. RACL-B safely completes all nine conditions, is 37.9\% faster than the fastest fixed current safe across the whole envelope, produces the least plated lithium, and remains safe across thermal guard bands. The same margin-aware principle drives a transient-credit fault readout (CREST-B) that, on a real introduced-fault battery-pack dataset, gives the strongest learned sequence-to-global monitor for localizing cooling-fault onset under operating-condition shift. The framework provides a deployable thermal-safety guarantee for fast charging together with a margin-aware monitor for the same physical fault class. |
| 2026-06-26 | [RS-Diffuser: Risk-Sensitive Diffusion Planning with Distributional Value Guidance](http://arxiv.org/abs/2606.27766v1) | Shiqiang Gong | Offline reinforcement learning enables policy learning from fixed datasets without additional environment interaction, making it appealing for safety-critical applications where online exploration is costly or unsafe. Diffusion-based decision-making methods have recently achieved strong performance in offline RL by modeling rich, multimodal trajectory distributions. However, existing diffusion planners are typically risk-neutral and therefore may overlook rare but catastrophic outcomes that are crucial in real-world deployment. In this work, we propose RS-Diffuser, a risk-sensitive offline diffusion planning framework that combines diffusion-based trajectory generation with distributional value critics. RS-Diffuser learns a diffusion planner over future state trajectories, a separate inverse dynamics model for action decoding, and a Monte Carlo distributional critic that estimates the full return distribution of candidate plans through quantile regression. At sampling time, we incorporate a risk-sensitive guidance signal into the denoising process, using gradients computed from tail-aware objectives such as Conditional Value at Risk to steer generation toward desired risk profiles. As a result, a single trained model can flexibly produce risk-averse, risk-neutral, or risk-seeking behaviors by changing only the inference-time risk parameter. Extensive experiments on risk-sensitive D4RL and risky robot navigation benchmarks demonstrate that RS-Diffuser achieves state-of-the-art performance, improving both overall return and worst-case robustness while reducing safety violations. |
| 2026-06-26 | [Low-Agreeableness Persona Conditioning for Safe LLM Fine-Tuning](http://arxiv.org/abs/2606.27709v1) | Austin MY Cheung, Yi Yang | Recent work has shown that fine-tuning large language models (LLMs) for social warmth degrades factual reliability and increases sycophancy. We investigate a related but distinct failure mode: warmth fine-tuning also weakens adversarial safety, making models more susceptible to jailbreaks and harmful output generation. We examine whether this reflects an inherent consequence of empathetic adaptation or an artifact of data construction. To address this, we introduce a persona-driven rewriting pipeline that conditions user turns on low agreeableness and pairs this with warm, de-escalating assistant responses. Across three experiments on four models, our approach reduces jailbreak susceptibility and harmful output rates relative to generic warmth fine-tuning baselines, while preserving conversational warmth. Representational probing provides suggestive evidence that this conditioning reduces the geometric alignment between warmth and compliance directions in latent space. These results show that safer empathetic fine-tuning is achievable through data design alone, without safety labels, harm detectors, or changes to the training objective. |
| 2026-06-26 | [AdvScan: Black-Box Adversarial Example Detection at Runtime through Power Analysis](http://arxiv.org/abs/2606.27704v1) | Robi Paul, Michael Zuzak | TinyML models deployed on edge devices are increasingly adopted in safety/security-critical applications, making them a prime target for adversarial example (AE) attacks where inputs are modified to cause misclassifications. However, existing AE detection methods either require white-box model access, which is often unavailable in licensed black-box deployments, or rely on input pre-processing stages that add non-trivial latency and resource overhead, often exceeding what mission-critical applications can afford on their inference path. To address these challenges, we propose AdvScan, a runtime power analysis-based methodology for AE detection that operates in a black-box scenario while inducing minimal latency. AdvScan is based on the observation that AEs produce anomalous neuron activations, which in turn generate distinctive power-consumption signatures. The algorithm initially constructs a baseline distribution of power signatures from known benign inputs; then, at runtime, it applies a one-sample t-test to determine whether a test input's power signature significantly deviates from this baseline, thereby detecting AEs. We evaluated AdvScan using three adversarial example generation algorithms: Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), and Carlini-Wagner (C&W), on three MLPerf Tiny benchmark models implemented on two target devices: the STM32F303RC (ARM Cortex-M4) and STM32L562RE (ARM Cortex-M33) microcontrollers. Across 318,400 total test inputs, AdvScan detects 99.984% of AEs with only 40 false negatives and zero false positives. These results demonstrate the viability of power-based AE detection for secure, accuracy-critical TinyML deployments in black-box environments. |
| 2026-06-26 | [Halt Fast! Early Stopping for Certified Robustness](http://arxiv.org/abs/2606.27694v1) | Andrew C. Cullen, Paul Montague et al. | Randomized Smoothing (RS) provides rigorous robustness guarantees for neural networks without architectural constraints, yet its adoption is limited by extreme computational costs. Standard RS requires tens of thousands of model evaluations per input and forces practitioners to commit to fixed sample sizes a priori. In this work, we present a novel meta-learning framework for anytime-valid certified robustness that adaptively deploys computational resources. By using a lightweight meta-learner to predict image-specific priors for a sequential E-process, we achieve a 20-fold reduction in sample complexity compared to traditional methods while maintaining rigorous statistical guarantees. Beyond raw efficiency, we demonstrate how anytime-validity enables adaptively allocating compute based upon application-specific risk thresholds, a form of resource triage impossible under classic certification frameworks. That this is achievable while also providing similar certification performance demonstrates that our approach provides a pathway for real-time, safety-critical certification deployments. |
| 2026-06-26 | [LightFARM: Model Predictive Lighting Control with Battery-Free IoT for Energy-Efficient Indoor Farming](http://arxiv.org/abs/2606.27649v1) | Hao Yu, Yanxiang Wang et al. | Lighting is the dominant energy load in indoor farming, yet most deployed systems still rely on fixed rule-based or schedule-based control. We present LightFARM, a predictive lighting control framework that couples crop illumination with battery-free sensing for more energy-efficient indoor farming. LightFARM combines finite-horizon predictive control with compact models of photosynthesis, thermal dynamics, and sensor energy state. The controller adjusts lighting intensity to balance photosynthetic benefit, electrical power consumption, thermal safety, and sensing-energy feasibility. A key design feature is that the same light-emitting diode (LED) fixtures serve both as the photosynthetic light source for crops and as a controllable energy source for self-powered sensor nodes. We implement LightFARM in a real indoor basil cultivation system and evaluate it through two independent 12-day cultivation trials. Compared with a conventional rule-based baseline, LightFARM reduces lighting energy consumption by approximately 41% and improves energy productivity from 36.1 to 52.9 $\mathrm{g\,kWh^{-1}}$ and from 41.1 to 60.2 $\mathrm{g\,kWh^{-1}}$ ($\approx 46.5\%$ on average). These results suggest that energy-cooperative predictive lighting control is a promising approach to improving indoor farming efficiency under practical resource constraints, while explicitly accounting for the trade-off between energy savings and crop yield. |
| 2026-06-26 | [Yuvion LLM: An Adversarially-Aware Large Language Model for Content And AI Safety](http://arxiv.org/abs/2606.27632v1) | Ting Ma, Xiufeng Huang et al. | As large language models are increasingly deployed in real-world systems, safety failures can still lead to harmful outputs and dangerous misuse. We argue that the essence of safety is adversarial: many failures arise not from natural inputs alone, but from strategic attempts to evade model policies and safeguards. However, existing general-purpose model development largely overlook this adversarial nature, and often remain insufficient for realistic safety scenarios involving planning, tool use, and multi-step reasoning, causing measured safety performance to overestimate real deployment robustness. To address this gap, we present Yuvion LLM, a large language model built for adversarially robust content safety and broader AI safety. Yuvion LLM treats adversarial robustness and agentic capability as first-class objectives. Its pipeline combines adversarially aware data construction, knowledge-enhanced continued pretraining, and policy-grounded multi-task safety post-training, including risk-aware supervised fine-tuning and reinforcement learning-based policy optimization, together with safety-aware agentic reinforcement learning for tool use and multi-step reasoning in complex safety scenarios. We further introduce the Yuvion LLM RiskEval (YLRE), a collection of 93 benchmarks across four evaluation categories, covering diverse open and internal evaluations with a focus on safety, adversarial robustness, and real-world capability requirements. Across these evaluations, Yuvion LLM demonstrates clear advantages on safety-focused benchmarks and particularly strong robustness under adversarial conditions, while maintaining solid overall capability. Notably, Yuvion-8B outperforms most state-of-the-art baselines, including substantially larger models such as GPT-5.4 and Qwen3-MAX, on several safety tasks. |

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



