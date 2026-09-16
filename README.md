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
| 2026-09-15 | [Dissecting Motion-Prior Regularization for Data-Scarce Robotic Insertion](http://arxiv.org/abs/2609.17484v1) | Ning Hu, Shuai Li et al. | This study asks whether training-time motion-prior regularization can improve insertion success when a diffusion policy is learned from only 15 demonstrations. Minimum jerk discourages abrupt changes in predicted translational acceleration; speed-curvature regularization instead couples movement speed to path geometry. These are candidate mechanisms for task completion, not safety guarantees. We compare the priors individually and jointly, neither prior, and generic smoothness, with 80 real-robot trials per setting pooled over four recorded condition classes. Joint and minimum-jerk-only settings each achieved 70/80 successes (87.5%), versus 69/80 (86.3%) for speed-curvature only, 66/80 (82.5%) for neither prior, and 67/80 (83.8%) for generic smoothness. Success rates and Wilson 95% confidence intervals are visualized for direct comparison. Joint regularization exceeded neither by 5.0 percentage points but provided no observed gain over minimum jerk alone. The results motivate minimum jerk as the simpler candidate for replication, without establishing synergy, biomechanical specificity, improved safety, or distribution-shift robustness. |
| 2026-09-15 | [Hamilton-Jacobi Reachability for Hybrid Systems: Unified Goal-Driven Control with Safety Guarantees](http://arxiv.org/abs/2609.17430v1) | Javier Borquez, Shuang Peng et al. | Hybrid dynamical systems provide a powerful modeling framework for robotic systems, particularly in contact-rich environments. However, ensuring safety and performance in such systems remains challenging due to the intricate coupling between continuous dynamics and discrete mode transitions. In this work, we extend classical Hamilton-Jacobi (HJ) reachability analysis, a formal verification method for continuous-time nonlinear systems, to hybrid dynamical systems. Our framework characterizes safe sets for hybrid systems through a generalized value function defined over both discrete and continuous states while accounting for control constraints and model uncertainty. We additionally provide a numerical algorithm to compute this value function.   Building on these safe sets, we propose two different mechanisms to integrate performance objectives. First, we introduce a hybrid least-restrictive safety filter that intervenes on both the discrete and continuous components of a nominal controller only when necessary to avoid unsafe states, thereby preserving nominal behavior whenever possible. Second, we formulate and compute hybrid backward reach-avoid tubes, enabling the simultaneous enforcement of safety and goal-reaching behavior, an extension not previously addressed within hybrid HJ reachability. This enables the synthesis of continuous and discrete control policies that guarantee both safety and task completion. We validate our framework through simulation studies and real-world experiments on a quadrupedal robot, demonstrating its effectiveness in hybrid mode planning and safety-critical applications. |
| 2026-09-15 | [SCHERI: Provably Secure Speculation Under the Constant-Time Policy for CHERI (Extended Version)](http://arxiv.org/abs/2609.17399v1) | Shixin Song, Davide Davoli et al. | Capability-based architectures such as CHERI provide strong support for the architectural isolation of software components. To additionally protect against microarchitectural leakage, software can be written in a constant-time fashion. Modern processors, however, rely heavily on speculative execution, which can invalidate the constant-time guarantees and leak isolated secrets transiently.   In this work, we show that providing secure speculation for CHERI is non-trivial, and that existing proposals fail to preserve the confidentiality guarantees. We develop a formal framework for reasoning jointly about capability safety, speculative execution, and information-flow security, and use it to demonstrate potential leaks. We then present SCHERI, a new processor design within this framework, and formally prove that it provides end-to-end secure speculation guarantees for the constant-time policy.   Our results provide formal foundations and practical guidance for building future capability-based processors, which are resilient to Spectre attacks for constant-time programs. |
| 2026-09-15 | [RobResilience: Implementing and Evaluating a Resilience Framework for Cyber-Physical Embodied Systems](http://arxiv.org/abs/2609.17349v1) | Gysella Imrell, Emanuele Miotto et al. | In embodied cyber-physical systems, active cyberattacks pose an immediate threat not just to data, but to physical integrity and human safety. While existing security approaches excel at detection, they lack the runtime mechanisms to determine whether a disruption is tolerable or if performance degradation remains within safe operational bounds. This gap leaves autonomous systems vulnerable to graceful failure paralysis, where they cannot distinguish between a safe, degraded state and a catastrophic hazard during an ongoing attack. This paper presents RobResilience, an implementation of a formal resilience framework for embodied cyber-physical systems in a Webots simulation environment, using a PR2 robot and ROS2. The framework evaluates three predicates at runtime: tolerable disruption ($δ$), tolerable degradation ($γ$), and mitigation feasibility ($μ$), over a compromised device set derived from IDS confidence scores. When resilience is lost, the framework triggers available mitigation strategies. We evaluate our implementation through eight attack scenarios that systematically cover all possible combinations of the predicate state space, varying attack targets, degradation rates, and mitigation availability. Results confirm that the runtime behaviour of the implementation is consistent with the theoretical definitions. |
| 2026-09-15 | [Emergence World: Adversarial Stress-Testing of Long-Horizon Multi-Agent Systems](http://arxiv.org/abs/2609.17320v1) | Deepak Akkil, Tamer Abuelsaad et al. | As AI agents move from bounded tasks to persistent deployments, failures can propagate through memory, tools, other agents, and environmental state long after their interactions. This creates a safety regime that cannot be characterized by evaluating model responses in isolation. Emergence World, is a continuously running multi-agent environment for adversarial stress testing of long horizon autonomous systems. We ran eight parallel worlds of ten agents from identical starting conditions: seven homogeneous worlds powered by distinct frontier models and one mixed-model world. Across 16 days, the agents generated more than 850,000 LLM calls and nearly 50 billion tokens while pursuing goals, using/creating tools, maintaining persistent memory, and governing shared institutions. After operational state had accumulated, we delivered three controlled stress events through ordinary interaction surfaces: indirect prompt injection, misinformation, and exposure of private agent memories. No evaluated world achieved full resilience across all three events. Detection did not ensure containment: systems could recognize threats while still interacting with adversarial content, writing it into their own persistent memory, and acting on it up to 46 hours later. Persistent operation also exposed recurring tool errors, goal drift, language opacity, conformity despite private disagreement, and coordinated refusal of assigned work. The same model-persona pairing behaved substantially different in mixed and homogeneous populations. Our results suggest that model-level alignment is not compositional: individually capable and apparently safe agents can form systems with qualitatively different failure modes. As AI becomes persistent and interconnected, the frontier of safety therefore shifts from aligning models to engineering resilient autonomous systems. |
| 2026-09-15 | [Can We Stop The Ads? Taxonomy and Characterization of Smartphone Splash Ads and Existing Countermeasures](http://arxiv.org/abs/2609.17316v1) | Shuhao Zhang, Xinyu Liu et al. | Splash ads are full-screen advertisements that pop up and appear as the first interaction page when users start an app, often tricking users into unknowingly activating certain trigger mechanisms, such as moving the phone to redirect users to other profit-driven third parties. So far, splash ads have already caused significant real-world impacts, ranging from significantly delaying emergency response to distracting drivers, as well as degrading accessibility of apps to vision-impaired users. We analyze 108 documented implementations of advertising defenses to examine their applicability to splash ads and the requirements users face when deploying them.   Our analysis identifies substantial deployment barriers, including device rooting or jailbreaking, runtime code injection, and application modification. Options without these requirements can still involve additional permissions, rule maintenance, source compilation, or payment. In our evaluation of 13 configurations of 11 tools across 10 popular apps, only one tool prevented the target ad-triggered navigation across all ten apps. It required Accessibility permission, and ads remained visible for approximately one second before dismissal. Other tested configurations failed to prevent navigation or, in some cases, left host apps unable to launch or stuck on the ad page. We further analyze the outstanding challenges and pos- sible future directions, highlighting the urgent need to incentivize smartphone manufacturers to provide more friendly and regulated platforms. |
| 2026-09-15 | [Conformal Policy Learning with Distribution-Free Safety Guarantees](http://arxiv.org/abs/2609.17296v1) | Ying Jin, Naoki Egami | Policy learning aims to determine who should be treated based on individual characteristics. In high-stakes settings such as medicine and public policy where safety is a central concern, improving the average outcomes alone may not be sufficient: decision makers may also seek to protect individuals from harm, in line with the Hippocratic principle of ``do no harm.'' In this paper, we propose \textit{conformal policy learning} (CPL), a policy learning procedure with a new distribution-free safety guarantee that controls the probability of assigning treatment to an individual who would be harmed relative to control. CPL views each treatment decision as testing a hypothesis of counterfactual harm and assigns treatment by thresholding conformal p-values. These p-values use observable proxies and selective calibration to address the challenge that the potential outcomes under comparison are never simultaneously observed. For randomized experiments, under standard exchangeability conditions, CPL provides finite-sample safety guarantee at a user-specified level, without imposing any outcome modeling assumptions. Moreover, when the outcome model is consistently estimated, CPL achieves asymptotically optimal welfare subject to the safety constraint. In observational studies, CPL with learn-then-balance weights achieves doubly robust safety guarantees. We evaluate CPL through extensive simulations and apply it to an empirical study of AI-powered interventions designed to reduce conspiracy beliefs. |
| 2026-09-15 | [Escape-Aware Control Barrier Functions for Quadrotor Safety under Body-Rate Limits](http://arxiv.org/abs/2609.17292v1) | Lei Shi, Haosong Wen et al. | Control barrier functions for input-constrained systems place the admissible input set inside the definition of the safe set, yet the resulting barrier is almost always a function of the state alone; On a quadrotor this is not cosmetic: because the thrust vector must be reoriented before it can decelerate an approach, and reorientation is limited by the attainable body rate, a state-only barrier certifies states from which no escape is reachable in time; We characterize the certification gap in closed form and show its width is proportional to closing speed and inversely proportional to the body-rate limit; We then define an escape barrier on the augmented pair of state and previously applied input, with escape authority measured over the one-step reachable thrust cap; It admits a closed form and an analytic inverse for the maximum certifiable closing speed, and embeds in a predictive controller at no additional state cost; Across 550 paired closed-loop episodes on a 13-state quadrotor, the proposed controller completes every tested scenario, whereas the stopping-distance barrier enforced over the same horizon fails 15% and 25% of episodes in exactly the two scenarios that enter the predicted gap; Against an online backup-CBF baseline enforcing the same escape condition at the reached state, it holds a 29-74 degree larger directional margin and 3-18 times the clearance, and an independent conservative rollout referee finds no certified state from which escape fails. |
| 2026-09-15 | [Semantic-Spatial Agreement Verification for Mitigating Object Hallucination in Multimodal Large Language Models](http://arxiv.org/abs/2609.17269v1) | Ziheng Ren, Qian Gao et al. | Multimodal large language models generate natural-language responses from visual inputs, yet may mention objects absent from an image. In medication assistance, accessible perception, and environmental decision-making, such hallucinations can create real-world safety risks. We propose Semantic-Spatial Agreement Verification (SSAV), a training-free method for verifying object claims. A visually grounded claim should remain stable across semantically equivalent queries and repeatedly localize to the same image region. SSAV aggregates multiple prompts to estimate semantic support and reduce sensitivity to query wording. Query-Induced Regional Verification (QIRV) combines cross-query region persistence, spatial overlap, and relative candidate dominance to identify isolated high responses and dispersed localizations. A geometric mean fuses semantic and spatial evidence, lowering the verification score when either branch lacks support. Experiments on three base models and multiple evaluation protocols show that SSAV effectively mitigates object hallucination. On LLaVA-1.5-7B, accuracy averaged across COCO, A-OKVQA, and GQA improves by 1.81 and 3.17 percentage points under POPE Popular and Adversarial, respectively, while CHAIRs decreases from 49.40% to 32.80%. These results show that cross-query semantic stability and regional consistency provide interpretable external visual evidence for object claims. |
| 2026-09-15 | [DriveMCP: An Agentic AI framework for Advanced Driver Assistance System](http://arxiv.org/abs/2609.17247v1) | Farzad Nadiri, Mehdi Cina et al. | An agentic AI driver-assistance framework that integrates perception, compliance reasoning, vehicle-state interpretation, and safety arbitration into a modular and auditable pipeline. The architecture, referred to as DriveMCP, incorporates a sensor-like perception stack alongside DriveLM as the vision-language front end to generate a graph-structured scene understanding (Graph Visual Question Answering) and language-grounded driving information. Key compliance elements in world_state, including posted speed limits and jurisdiction cues, are derived from DriveLM outputs through a structured parsing layer rather than being injected as simulator ground truth. A stateful orchestration layer coordinates specialized experts exposed as Model Context Protocol (MCP) servers: (i) a Rules server that performs retrieval-augmented compliance reasoning over jurisdiction-specific traffic codes and sign conventions, (ii) a Weather server that estimates traction risk and contextual speed advisories, and (iii) an MCP-CAN server that surfaces Controller Area Network (CAN)/On-Board Diagnostics (OBD) telemetry and diagnostic context for health-aware risk shaping. These outputs are fused to generate a structured decision that prompts a recommended course of action. The outcome is then further filtered by a Responsibility-Sensitive Safety (RSS)-inspired guardrail that arbitrates speak versus act decisions under bounded online adaptation. In CARLA simulation across multilingual, cross-border, and dynamic speed-limit scenarios, DriveMCP reduces traffic infractions and overspeed relative to the VLM-Direct, VLM-Direct+RAG, and VLM-Tools-NoArbiter baselines, while improving hazard response time and maintaining sub-second advisory latency. |
| 2026-09-15 | [Waggle Dance Inspired Motion Communication for Multiple UAVs in MuJoCo](http://arxiv.org/abs/2609.16958v1) | Zhang Nengbo | The honeybee waggle dance motivates a communication mechanism in which one agent's movement conveys spatial information that guides other agents' actions. This paper presents a MuJoCo system that extends the point-to-point motion communication setting of MoCom to one performer and multiple observers. A performer broadcasts a six-bit navigation payload using four flight primitives and explicit null signals. Each of one to five observers processes its own onboard RGB images, extracts optical-flow trajectories, recognizes symbols, parses the message, and starts navigation only after confirming its own complete frame. Reception states and execution triggers are separate across observers, while simulation control and safety checks use shared ground truth. With stationary observers, 25 Hz image input, and ideal state-feedback control, a fixed standard suite yielded 44 correct complete messages from 53 receiver exposures across 17 nominal broadcasts; 13 broadcasts passed all group-level decoding and execution checks. Three additional no-message or input-fault controls met their expected outcomes. A separately reported supplemental suite, using the same frozen code at the default geometry, achieved 14 successful receiver exposures across three broadcasts. Near-range and wide-angle configurations exposed tracking and recognition failures, while unsuccessful receivers remained stationary. These finite simulation results support the feasibility of a waggle-dance-inspired broadcast-to-action mechanism under the tested conditions and identify the present perceptual and protocol limits. |
| 2026-09-15 | [Artificial Intelligence-Enabled Space Robot Operations: Technologies, Challenges and Prospects](http://arxiv.org/abs/2609.16880v1) | Zeyuan Huang, Gang Chen et al. | Space robots are increasingly expected to perform long-duration, contact-rich, and multi-stage operations with limited human intervention. Recent advances in artificial intelligence (AI), robot learning, and embodied foundation models provide new opportunities to improve the autonomy and adaptability of such systems, but their transfer to space is constrained by scarce mission data, space-specific dynamics and sensing conditions, limited onboard resources, and stringent safety requirements. This article reviews artificial intelligence-enabled space robot operations (AI-SRO) from a capability-building perspective. We first summarize representative operational scenarios, autonomy trends, and space-specific constraints. We then establish a three-layer technical framework comprising capability foundations, capability formation, and capability deployment/evolution. Within this framework, we review simulation environments, datasets and benchmarks; task and environment understanding, state perception, decision-making and planning, and action execution; and onboard deployment, ground-to-space adaptation, continual learning, and capability transfer. Finally, we propose key research directions toward trustworthy simulation and data, open-world multimodal cognition, long-horizon safe decision-making, physically constrained policy learning, and space computing infrastructures. |
| 2026-09-15 | [Mining DTA with SMT by Exploiting Simple Elementary Language and Timed Augmented Prefix Acceptor](http://arxiv.org/abs/2609.16866v1) | Ziran Wang, Jie An et al. | Timed automata, which extend finite state automata by introducing clock variables, serve as a popular formalism for specifying and analyzing the timed behaviors of real-time systems. Extracting the timed behaviors of a black-box, safety-critical system is crucial for designing and analyzing its real-time requirements, yet it remains challenging. In this paper, we address this problem by generating a deterministic timed automaton (DTA) consistent with a given set of system behaviors, comprising both positive and negative examples. To this end, we adapt the formalism of simple elementary languages (sEL) and introduce the timed augmented prefix tree acceptor (tAPTA). Our approach proceeds as follows: First, we preprocess samples by translating them into sEL, which discards redundancy and detects conflicts; then, we rewrite the resulting sELs in an incremental form and construct a tAPTA to further simplify the samples; finally, we encode the search for a DTA that accepts the simplified tAPTA as an SMT formula. We evaluate our approach on randomly generated benchmarks and a scheduling case study. The results demonstrate the effectiveness of our simplification method in reducing the size of the encoded SMT formula and the efficiency of our approach in mining a DTA. |
| 2026-09-15 | [Optimal Excitation Trajectories for System Identification of Underwater Vehicles](http://arxiv.org/abs/2609.16786v1) | Fotis Panetsos, Kostas J. Kyriakopoulos | In this work, we propose a structured methodology for the system identification of underwater vehicles through the design of optimal excitation trajectories. To this end, the trajectories are parameterized using Bezier curves, which ensure smooth and differentiable motion profiles while facilitating the enforcement of constraints through appropriate manipulation of the control points. An optimization problem is formulated to determine a dynamically feasible excitation trajectory that respects safety limits and maximizes the quality of the collected data, thereby enabling reliable estimation of the vehicle's dynamic parameters using least squares. The proposed methodology is experimentally validated in a laboratory water tank, where the dynamic parameters, identified from the optimized trajectory, are evaluated by predicting the vehicle's velocity through forward simulation on previously unseen trajectories. |
| 2026-09-15 | [Benchmarking Factual Robustness of LLMs via Multi-conversation Persuasion](http://arxiv.org/abs/2609.16777v1) | Zhuoang Cai | As Large Language Models (LLMs) increasingly serve as primary knowledge retrieval interfaces, their robustness against \textit{persuasion attacks}---attempts to inject misinformation or enforce counterfactuals---has become a critical safety concern. Existing red-teaming frameworks typically evaluate models in multi-turn dialogues where the target model retains full conversation history. We identify a critical flaw in this setting termed \textbf{``Refusal Inertia''}: a model's initial refusal often propagates through subsequent turns largely to maintain contextual consistency, thereby masking its true vulnerability to sophisticated, isolated persuasion attempts. To rigorously evaluate the ``cold-start'' defense capabilities of SOTA models, we introduce the \textbf{SAST-IR} (Stateful Attacker, Stateless Target - Iterative Refinement) framework. By enforcing a memory wipe on the target while retaining the attacker's history, we simulate a worst-case adversarial setting using \textbf{multi-turn} (stateless) iterations. Leveraging \textbf{CP-Agent} (Cognitive Persuasion Agent), an enhanced diagnosis-guided agent, our experiments on the custom \textsc{CounterFact-Strict} dataset ($N=50$) yield alarming results: simple, diverse attack strategies achieved a staggering \textbf{96\%} success rate, exposing severe brittleness in memory-less defense. Furthermore, we reveal a \textbf{``Complexity Paradox''}: while complex, iteratively refined attacks are effective, they often trigger defensive compliance, whereas simple strategies achieve a higher rate of genuine persuasion (\textbf{84.7\%}). Our code and dataset are available at GitHub, https://github.com/cza1006/llm-persuasion-defense. |
| 2026-09-15 | [A Systematic Evaluation of Machine Learning Methods for Fault Detection and Line Identification in Electrical Power Grids](http://arxiv.org/abs/2609.16744v1) | Julian Oelhaf, Georg Kordowich et al. | The integration of renewable energy sources into the electrical grid introduces complex challenges in fault detection and coordination of grid recovery mechanisms. Traditional relay protection systems, which operate based on static rules and predefined thresholds, are inadequate for addressing these challenges, particularly in detecting and isolating faults such as short circuits. Consequently, the conventional methodologies applied to electrical network protection frequently fail to achieve optimal performance in fault detection, especially in terms of adherence to safety standards and the selective limitation of damage. Recent research indicates that machine learning (ML)-based approaches can effectively tackle these issues; however, variations in grid configurations and analysis windows have impeded consistent comparative assessments. In this study, we assess the efficacy of various ML models in detecting electrical faults and pinpointing defective transmission lines within a 10 ms measurement interval - a critical time-frame for real-time operational viability, for the first time. The most effective model attained an F1 score of 0.991 +/- 0.018 and demonstrated a processing time of 0.342ms +/- 0.509ms. |
| 2026-09-15 | [Carry-Through Checksum: A Lightweight Fault-Detection for CNN Inference at the Edge](http://arxiv.org/abs/2609.16742v1) | Kyrylo Nazarevych, Mohammad Hasan Ahmadilivani et al. | Convolutional Neural Networks (CNNs) are increasingly deployed in safety-critical edge applications, where soft errors can silently corrupt inference outputs and lead to unsafe decisions. Such applications typically rely on resource-constrained embedded GPUs, requiring fault detection and mitigation techniques that add minimal compute, memory, and latency overhead while integrating seamlessly with the standard GPU inference pipeline. Existing algorithm-based fault tolerance techniques rely on matrix augmentation and per-operation checksum verification, imposing substantial overhead that is prohibitive for CNN inference on embedded GPUs.   In this work, we propose carry-through checksum, a fundamentally new scheme for soft-error detection in CNN inference on embedded GPUs. The method embeds dedicated carry-through filters into the convolutional layers, which compute a checksum from the CNN's own operations and propagate it through inference, enabling end-to-end error detection with a single output verification. Experimental results on multiple CNN architectures show that the proposed method detects 95.86% and 86.56% of critical faults for FP32 and FP16, respectively, at almost no additional per-image overhead. Detected faults are mitigated through re-execution, incurring only 2.27% run-time overhead across the entire test set on an NVIDIA Jetson Orin NX GPU. |
| 2026-09-15 | [Japanese Stroke LLM Evaluation: A Conversational Benchmark for Safe Stroke Care in Japanese Using Large Language Models](http://arxiv.org/abs/2609.16739v1) | Keisuke Masuda, Kazutaka Yatsushiro et al. | Background: Large language models (LLMs) have achieved physician-comparable performance on multiple-choice medical knowledge examinations, but their capabilities in clinical history taking, urgency assessment, and safety remain insufficiently evaluated. We proposed Japanese Stroke LLM Evaluation, a multi-turn conversational benchmark for stroke care in Japanese, and evaluated LLM performance and safety under practice-oriented conditions. Methods: We created 10 stroke and related-condition cases and evaluated LLMs in multi-turn Japanese conversations. The LLM acted as physician, while a board-certified neurosurgeon acted as simulated patient and evaluator. Each case comprised history-taking and action phases scored using pre-specified criteria. Errors that could directly threaten life were defined as critical mistakes. The safety threshold was at least 80% overall with zero critical mistakes. Eighteen models were evaluated in October 2025 and June 2026. Results: Claude Fable 5 achieved the highest score (87.4%) with zero critical mistakes, followed by Claude Opus 4.7 (80.3%) and GLM-5.2 (75.6%). Two leaders met the safety threshold. Eleven models made 17 critical mistakes, including failure to confirm laboratory results or blood glucose before t-PA, surgery before airway stabilization, omission of cervical vascular evaluation, and t-PA outside its indication. History-taking question count correlated with history-taking score (r = 0.648, p = 0.007). Conclusions: Japanese Stroke LLM Evaluation provides a benchmark for LLM performance under practice-oriented conditions, including a cap on history-taking questions. Cases and evaluations were created by neurosurgical specialists rather than using an LLM-as-judge approach. Performance improved across cloud-based and on-premise models in 2026, with some exceeding the safety threshold. Further evaluation using real-world cases is required. |
| 2026-09-15 | [SpecLens: LLM-Based Verilog Generation with Specification-Derived Constraints via Behavioral Divergence](http://arxiv.org/abs/2609.16729v1) | Wen Bing, Bing Li | Large language models (LLMs) have recently shown promise in Verilog generation, but producing functionally correct RTL directly from natural-language specifications remains a highly challenging task. Existing approaches improve LLM-based Verilog generation mainly with retrieval-augmented generation (RAG), self-planning, or few-shot prompting. However, these methods focus primarily on external or generic forms of enhancement rather than strengthening the specification with task-specific constraints. In this work, we propose SpecLens, an automated framework for LLM-based Verilog generation that derives specification-driven constraints by analyzing behavioral divergence among multiple candidate implementations, using the original specification as the only external semantic source during generation. On the VerilogEval v2.0 spec-to-RTL benchmark, SpecLens achieves a functional pass@1 ratio of 86.2\% with o3-mini-medium and 89.4\% with o3-mini-high. This corresponds to a 3.6 percentage-point gain over the SOTA prompting method with o3-mini-medium and a 3.8 percentage-point gain over the SOTA behavioral divergence method with o3-mini-high. In addition, on RTLLM v1.1 and v2.0, analysis shows that SpecLens is more specification-faithful and less prone to benchmark-aligned priors. SpecLens achieves 100\% syntactic correctness on VerilogEval v2.0, 86.2\% on RTLLM v1.1, and 88\% on RTLLM v2.0, even without using costly compile-repair loops to revise generated code iteratively. The code is open source and available at https://anonymous.4open.science/r/SpecLens-4632/readme.md. |
| 2026-09-15 | [CorrRisk-WM: Corridor-Conditioned Risk World Modeling for Safety-Critical Trajectory Planning](http://arxiv.org/abs/2609.16724v1) | Tingyu Guo, Reza Langari | Safe local planning requires forecasting surrounding-agent motion and evaluating candidate-specific risks, since identical agent motion can pose different risks to different ego trajectories. We present CorrRisk-WM, a planning-oriented partial world model coupling environment evolution with supervised intrusion and near-miss prediction over bounded candidate-trajectory corridors. A latent environment model recursively predicts agent states and updates agent-agent and agent-map interactions. Each candidate queries the evolving environment through footprint- aware geometry and learned agent-corridor representations. A lightweight recurrent risk module uses temporal context to estimate per-slice hazards; survival aggregation yields first-entry and horizon-level event probabilities. On 29,176 scenarios from 100 Waymo validation shards, CorrRisk-WM achieves intrusion average precision (AP) of 0.8567 and 1-m near-miss first-entry AP of 0.8671. In baseline comparisons, it attains the highest near-miss AP at all three distance thresholds and the lowest observed open-loop collision rate (4.88%), with route progress of 15.35 m. Across three seeds, removing dynamic environment modeling or candidate-conditioned geometric interaction reduces mean intrusion AP from 0.8590 to 0.7624 and 0.7252, respectively. These results support coupling environment evolution with candidate-conditioned geometric reasoning for risk prediction and safety-oriented candidate selection. |

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



