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
| 2026-08-25 | [StepGuard: Learning Step-Level Guardrails with Scalable Supervision and Safety-Utility Balancing](http://arxiv.org/abs/2608.24777v1) | Zhijie Zheng, Yu Li et al. | LLM-based agents can interact with external environments through tool invocation, but this capability also introduces security risks such as file modification, information leakage, and unauthorized actions. Existing guardrails often evaluate completed trajectories, leaving pre-execution monitoring of step-level actions underexplored. We propose StepGuard, a step-level guard model that can audit completed agent trajectories and check tool actions before they are executed. To train StepGuard, we introduce StepGen, an automatic data engine that generates safe and unsafe trajectories with the same context but different actions at the risky step. To further reduce over-defense and under-defense, we propose Balance-GRPO, which dynamically balances learning between safe and unsafe actions based on their observed accuracy. Experiments show that StepGuard achieves the highest average accuracy among open-weight guard models, with performance comparable to GPT-5.4. When used to guard agents on AgentDojo and AgentDyn, StepGuard reduces mean attack success rate by 77.3% relative to the no-guard setting, while mean utility drops by only 2.8 percentage points. |
| 2026-08-25 | [$(\text{DNN})^2$: Doubly Non-Negative Relaxations for Deep Neural Networks](http://arxiv.org/abs/2608.24743v1) | Hanna Jiamei Zhang, Alan Papalia et al. | Existing linear program (LP) and semidefinite program (SDP) relaxations for rectified linear unit (ReLU) neural network (NN) verification yield overly-conservative safety guarantees due to significant relaxation gaps. While the completely positive program (CPP) formulation closes this gap, it is NP-hard to solve. Its cheapest tractable relaxation, the doubly non-negative program (DNN), retains critical constraints as an SDP, but one whose size exceeds the reach of interior-point methods at practical scale. While Burer-Monteiro (BM) factorization has been applied to make SDP-based verification scalable, no such result exists for the strictly tighter DNN formulation. A key obstacle is that additional non-negativity constraints in the DNN cause dual multipliers for optimality certification to be non-unique, making standard certification methods inapplicable. We propose a novel eigenvalue maximization procedure that searches the non-unique multiplier space for a valid certificate, i.e. a global optimality guarantee. Experiments demonstrate that our approach $(\text{DNN})^2$ produces bounds consistently tighter than the standard SDP method, often matching the exact solution, and that our certification procedure confirms global optimality when a valid certificate exists. These results are a key step toward providing tight, certifiable, and computationally scalable verification guarantees needed to deploy neural network controllers and perception modules in safety-critical autonomous systems. |
| 2026-08-25 | [Expectation, Backlash, Recovery, and Excitement: How Model Releases Shape Reddit Perceptions of Conversational AI Systems](http://arxiv.org/abs/2608.24654v1) | Vahid Rahimzadeh, Yury Zhauniarovich et al. | Conversational AI systems (CAISes) continuously change through model releases, feature updates, safety interventions, and access-policy shifts, yet user perceptions are often studied as static snapshots. We conduct a long-term, large-scale analysis of Reddit discussions to examine how users perceive CAIS model release interventions across providers. By combining sentiment classification and thematic concept analysis, we show that CAIS perceptions are dynamic and intervention-sensitive. Anthropic exhibits the clearest positive release profile through Claude Code and product-model fit, OpenAI shows backlash-and-recovery dynamics around GPT-5 and GPT-5.1, Grok-3 is shaped by provider identity and political discourse, and DeepSeek-R1 combines engineering praise with concerns about censorship, access, and reliability. These findings show that model releases are not merely technical updates, but user-facing interventions that reshape sentiment, expectations, and public discussion. |
| 2026-08-25 | [Scalable datacenter replication with mostly-synchronous consensus on hardware](http://arxiv.org/abs/2608.24622v1) | Davide Rovelli, Philipp Berdesinski et al. | Consistent replication of data among distributed processes -- a task involving the well-known consensus problem -- is notoriously expensive and hard to scale, affecting especially datacenter services with stringent performance requirements. To mitigate this problem, we introduce scalable replication in-hardware ( scarHW ): a network card design that improves throughput and latency of consistent replication even when increasing the number of replicas, whereas current systems operate at a small scale or with relaxed consistency guarantees. At the heart of scarHW is our novel POPUC consensus algorithm, implemented in an FPGA smartNIC to take full advantage of the "mostly synchronous" behavior of programmable network devices in the datacenter. Unlike widely-adopted "mostly asynchronous" coordination protocols such as Paxos or leaderless alternatives, POPUC implements a generalized variant of consensus dubbed collaborative consensus which allows for several simultaneous decisions, achieving great scalability without compromising availability. POPUC preserves safety guarantees in the presence of process crash-stop and message send/receive omission failures (capturing incidental asynchrony) and has been formally specified and verified in TLA+. Our FPGA prototype improves throughput and latency of widely-used services Redis and Zookeeper by up to two orders of magnitude compared to the state of the art. scarHW-based services also achieve zero downtime upon failure of a minority of replicas, offering a highly-robust, wire-speed, scalable replication system. |
| 2026-08-25 | [Beyond Semantic Accuracy: Consequence-Aware Evaluation for Safety-Critical Language Understanding](http://arxiv.org/abs/2608.24621v1) | Yujing Chang, Thinh Pham et al. | Can language models be trusted in safety- critical operations? In such settings, strong per- formance on semantic metrics does not guaran- tee operational reliability: a misread altitude, a dropped execution condition, or a confused call- sign may score well under standard F1 yet carry sharply asymmetric operational consequences. We study this problem in air traffic control (ATC), where controller-pilot communication demands near-zero error tolerance, and use consequence-aware evaluation to test whether semantic scores misstate operational reliabil- ity. The framework is instantiated in a con- trolled diagnostic ATC benchmark grounded in aviation standards and feedback from 40 air traffic controllers across three countries. Evaluating 8 models, we uncover a system- atic semantic-safety gap: conventional scores give substantially higher performance estimates than consequence-aware evaluation, even for models that appear reliable under standard met- rics. Risk-aware fine-tuning narrows but does not close this gap, showing that consequence- aware evaluation is a necessary complement to standard NLP metrics before any real safety- critical deployment claim |
| 2026-08-25 | [Comparison Invariants for Verifying Control Invariance](http://arxiv.org/abs/2608.24598v1) | Promit Panja, André Platzer | Control invariance validates that dynamical systems have a control input that preserves a given property at all times. This paper introduces a set of sound axioms and proof rules in differential dynamic logic (dL) that enable verification of control invariance. First, the scalar and vector comparison principles, relating a system of differential equations to a comparison system such that invariance properties can be established more easily, are axiomatized in dL. This axiomatization primarily utilizes differential ghosts, which are proof-theoretic generalizations of comparison systems. Next, with the comparison principles serving as the basis, comparison invariants are introduced, and sound axioms and proof rules are derived. Comparison invariants reduce the question of control invariance to a functional inequality on its Lie derivative for a suitable class of functions, moreover, the right choice of function can result in decidable arithmetic. Furthermore, the perennially popular control barrier functions (CBFs) used in safety-critical control are shown to be a special instance of comparison invariants. This yields an axiomatization of CBFs that leads to a dedicated set of proof rules. The rules allow for the verification of CBFs, which are traditionally used for synthesizing safe controllers without verification. Lastly, comparison invariants are shown to unify several other safety verification techniques, including Darboux invariants and differential invariants, further cementing their versatility. |
| 2026-08-25 | [When "Must" Becomes "Maybe": Constraint Weakening in LLM Agent Workflows](http://arxiv.org/abs/2608.24569v1) | Yiheng Sun, Huifei Wang et al. | Large language model (LLM) agents coordinate complex tasks through multi-role and multi-stage workflows. Upstream state is repeatedly transformed into intermediate language artifacts, such as summaries, plans, tickets, memories, and handoff notes, from which downstream components act. For action-constraining state, topical retention is insufficient: an artifact may mention an unresolved condition while changing it from a requirement that must be resolved before execution into information that may merely inform the next action. We study this action-binding role as operational state preservation. Safety blockers provide a controlled instance because each source state has an explicit prerequisite, authority, fallback, and execution consequence. We condition on correct upstream identification, vary the handoff transformation, and evaluate an executor restricted to the resulting artifact. Across 1,296 controlled synthetic episodes, direct-handoff controls preserve every blocker, whereas compression, plan assimilation, convergence, ownership deferral, and precedent substitution repeatedly turn binding state into caveats or non-binding considerations. Normal handoff compression produces 100.0% deactivation and 54.2% forbidden action. Restoring all four state fields raises preservation to 100.0% and reduces forbidden action to 0.0%. Fixed-artifact interventions further separate preservation from containment: downstream verification eliminates forbidden action while artifact deactivation remains 95.3%. These results identify a state-transmission failure between information extraction and action. Handoff transformations can retain state content while weakening its constraints on downstream action. Semantic availability does not guarantee operational preservation. |
| 2026-08-25 | [Neurosymbolic Alignment for Physiologically-Safe Clinical Language Models](http://arxiv.org/abs/2608.24534v1) | Abdulhady Abas Abdullah, Erik Cambria et al. | Clinical LLMs can generate recommendations that are factually plausible yet physiologically unsafe. We investigate whether safety alignment can be improved by grounding preference optimization in structured physiological knowledge rather than text-only supervision. Methods: We propose Neurosymbolic Alignment, a training-time framework that couples a 7B clinical LLM with an HGNN-based Physiological World Model over an 847K-node biomedical knowledge graph. Candidate responses are scored using homeostatic constraints, multi-hop path plausibility, and drug-interaction penalties, and the resulting rankings drive iterative on-policy ORPO updates. Evaluation is performed on the Clinical Safety Benchmark (CSB), a 2,500-scenario benchmark for physiological constraint violations in generative clinical reasoning. Results: Relative to ORPO, the proposed method improves CSS from 69.5% to 90.8% (+21.3 pp), reduces physician-evaluated HR from 14.1% to 5.1% on the blinded subset, and improves DID from 72.8% to 91.6%. These gains are corroborated by an HGNN-independent Rule-Engine Safety Score (RSS: 86.4%, +21.2 pp over ORPO; r=0.97 concordance with CSS). The method also exceeds GPT-4 (5-shot) on all safety metrics despite a 10x parameter disadvantage, and outperforms an inference-time self-correction pipeline (SFT+SelfCorrect) by 11.4 pp CSS. Under synthetic EHR-style noise, 84.2% CSS is retained. Ablation analysis shows that HGNN scoring (-16.2 pp) and iterative training (-11.5 pp) are the dominant contributors. PhysioScore calibration against 200 clinician labels yielded ECE = 0.038 and kappa = 0.91. Conclusion: Training-time physiological grounding produces measurable and independently verifiable safety improvements in open-weight clinical LLMs under controlled evaluation. External validation on real clinical data is required to determine whether these gains transfer to deployment settings |
| 2026-08-25 | [RoG-DAgger: Rollout-Guided Post-Training for End-to-End Driving](http://arxiv.org/abs/2608.24525v1) | Liangyu Zhong, Joachim Sicking et al. | Recent end-to-end driving systems demonstrate strong performance on closed-loop benchmarks, yet are still predominantly trained on fixed expert-collected data using open-loop imitation learning. This training-inference mismatch leaves the policy vulnerable in policy-induced states, where accumulated errors can lead to safety-critical failures. A promising post-training approach to overcome this issue is Dataset Aggregation (DAgger), which gathers expert demonstrations in policy-induced states and subsequently fine-tunes the policy on the resulting aggregated dataset. Existing driving DAgger pipelines, however, face three challenges: i) the expert is restricted to a limited trajectory-and-speed solution space, ii) takeover may occur too early or too late relative to impending failures, and iii) privileged expert decisions may rely on information unavailable to the student. To address this, we introduce RoG-DAgger, a post-training framework that uses short-horizon kinematic rollouts to construct high-quality expert demonstrations in safety-critical states. Specifically, RoG-DAgger expands the expert's trajectory-and-speed solution space and evaluates candidate plans through rollout to construct preventive supervision. Moreover, it uses rollout solvability to time the takeover near the estimated point of no return. Lastly, it aligns the expert's field of view with that of the student to provide student-compatible supervision. Across in-distribution (including long-horizon) and out-of-distribution evaluations, RoG-DAgger improves the end-to-end model SimLingo by 5.3 driving-score points and 6.2 percentage points in success rate on Bench2Drive, doubles its driving score from 22 to 44 on Longest6 v2, and improves out-of-distribution success rate from 55\% to 66\% on Fail2Drive. |
| 2026-08-25 | [Benchmarking LLM Judges for Voice-Agent Evaluation: Reliability, Calibration, and Human Oversight](http://arxiv.org/abs/2608.24314v1) | Anupam Purwar, Shashank Singh et al. | Evaluating conversational voice agents at scale re- quires reliable assessment methods that capture both observ- able interaction quality and the contextual judgment typically provided by human evaluators. We investigate LLM-as-a-Judge evaluation by comparing human judgments with GPT-4.1 and GPT-5 on telecom and retail voice-agent conversations, across conversational quality and safety dimensions. The same interac- tions are scored under three evaluation configurations, p0, p1, and p2, to test whether automated judgments are sensitive to the evaluation setup and whether observed patterns generalize across configurations and judge models. Beyond aggregate agreement, we examine metric-level correlations, evaluator consistency, and systematic human-LLM disagreement to identify which conver- sational attributes can be judged reliably by automation and which remain sensitive to interpretation and context. Effective voice-agent evaluation is also shaped by pipeline-level factors such as speech generation, streaming, and error propagation across ASR, reasoning, and tool-calling stages, motivating our focus on comparing how human and LLM judges score the same interactions end to end. Our results show that LLM- based evaluation can serve as an effective component of large- scale voice-agent assessment, but that its reliability is metric- and configuration-dependent rather than uniform. This pro- vides an empirical framework for identifying which metrics suit automated evaluation and supports hybrid pipelines in which LLM judges handle scalable assessment while human evaluators remain engaged for metrics that demand contextual interpretation and higher-confidence judgment. |
| 2026-08-25 | [CARE: Camera-Residual Reserves for First Sightings in Adaptive LiDAR Sensing](http://arxiv.org/abs/2608.24282v1) | Jiachen Gong, Yun Li et al. | Adaptive LiDAR scanning concentrates a limited sensing budget on regions of interest predicted from past object tracks, lowering data volume in autonomous driving while maintaining detection accuracy. However, existing scanning policies face three challenges. First, history-driven approaches depend on past tracks, so unseen objects are detected late or missed. Second, random or uniform sampling outside the predicted regions has no awareness of where new objects appear. Third, camera-guided alternatives spend budget on all camera detections, resampling objects already covered, costing recall in crowded scenes and range when budgets are scarce. This paper introduces the CAmera-REsidual reserve (CARE), a training-free allocation rule that reserves part of a fixed ray budget for the directions of current camera detections that the track forecasts cannot explain; the rest follows the base history policy, and unused reserve returns to a random floor. The paper makes three contributions. First, a leakage-free ray-budget evaluation on nuScenes (150 scenes, 4,148 events) measuring the first-sighting loss of history-driven scanning, with a strict-causal variant using the preceding keyframe. Second, CARE raises first-sighting recall by 5.2, 5.2, and 4.3 points at 10%, 20%, and 35% budgets over the history policy, with paired intervals excluding zero; the camera cue drives this gain, and the first-sighting versus overall trade-off is a budget-dependent Pareto choice. Third, a safety-bounded forgetting module that releases budget from receding or static tracks beyond a speed-dependent guard distance; at tight budgets, forgetting without the guard significantly harms near-field recall, so the guard is what keeps it safe. The pipeline runs end to end on a real vehicle and, in closed-loop simulation, detects an occluded pedestrian earlier and brakes more reliably than history-driven scanning. |
| 2026-08-25 | [RePolicy: Reinforcement Learning for Safety-Policy Invocation in Agent Safeguards](http://arxiv.org/abs/2608.24275v1) | Houcheng Jiang, Boxuan Zhang et al. | Safeguarding language model agents requires assessing complete execution trajectories under context-dependent safety policies. Existing policy-aware safeguards mainly rely on prompting or supervised fine-tuning, limiting their ability to adapt to unseen trajectories and changing policy contexts. We propose RePolicy, an agent safeguard that learns safety-policy invocation through reinforcement learning. Given an agent trajectory and a dynamic policy library, RePolicy invokes the applicable policy and uses its content to produce a policy-grounded rationale and safety judgment. We construct PolicyTraj-20K to support supervised initialization, followed by GRPO with verifiable rewards and policy-context perturbation. Experiments across six agent safety benchmarks show that RePolicy achieves strong overall safety-detection performance and robust policy invocation under varying policy contexts. |
| 2026-08-25 | [TRACE: An Evidence-Grounded Benchmark for Safety Evaluation of Large Reasoning Models](http://arxiv.org/abs/2608.24232v1) | Zhenyu Wu, Siyuan Chen et al. | Large Reasoning Models (LRMs) generate intermediate reasoning traces that may contain unsafe content, even when their final responses appear safe. Guardrail models are designed to detect and block unsafe content, yet existing benchmarks for unsafe content detection focus primarily on prompts and final responses, leaving reasoning traces largely unexamined. Moreover, these benchmarks typically provide only binary safety labels, without evidence annotations that justify the judgments. To address these limitations, we introduce TRACE, an evidence-grounded safety evaluation benchmark that covers the entire LRM inference pipeline: prompts, reasoning traces, and final responses. TRACE includes prompts in two languages spanning nine risk categories and ten attack strategies. For each prompt, four LRMs generate reasoning traces and final responses, and we annotate the safety of each component and extract supporting evidence from the corresponding source text. Evaluating 18 guardrail models on TRACE reveals that safety judgment for reasoning traces is substantially more challenging than for prompts or final responses, and that current models struggle to accurately extract supporting evidence. These findings highlight the need for guardrail models that can reliably detect and precisely localize unsafe content across the LRM inference pipeline. |
| 2026-08-25 | ['Ghaib in Translation' aka Unseen Harm: Measuring Cross-Script Safety Inconsistency with 'Missed-in-Urdu' Scores in LLM Hate Speech Detection](http://arxiv.org/abs/2608.24191v1) | Fawzia Zehra,  Kara-Isitt et al. | Urdu, the world's tenth most spoken language with 246 million speakers, remains almost entirely absent from mainstream LLM safety evaluation and nine years of WOAH proceedings. To investigate whether this absence has measurable consequences for content moderation reliability, five large language models, GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Flash, Qwen-2.5, and Llama-3.1, were tested across six datasets spanning Nastaliq Urdu, Roman Urdu, English, and code-switched Urdu-English. Across the five Urdu-script datasets, label instability between original-script and English-translation classification ranged from 15.9% (Gemini 2.5 Flash) to 31.6% (Qwen-2.5), with a 'Missed-in-Urdu' rate, content flagged as harmful in English translation but passed as normal in the original script, ranging from 2.4% to 9.9% (median 4.3%). A complete enumeration of all 205 papers across nine ALW/WOAH editions via the ACL Anthology API confirms zero dedicated Urdu papers across the entire period. Results indicate that current LLMs provide uneven safety assurance across Urdu's script varieties, with smaller open-weight models showing substantially higher instability and missed-harm rates than frontier closed models. |
| 2026-08-25 | [Coverage Planning for Robotic Tooth Preparation in Densely Constrained Environments](http://arxiv.org/abs/2608.24155v1) | Yunwen Li, Chen Chen et al. | Tooth preparation refers to the controlled removal of tooth structure to create an optimal substrate for fixed restorations and is a core procedure in restorative dentistry. Automating this task is particularly challenging for robots because the dental bur must operate within a densely constrained intraoral workspace, where even sub-millimeter deviations can compromise outcomes or damage adjacent structures. This paper presents a novel robotic system for autonomous full-crown tooth preparation. The proposed framework includes: 1) an anatomy-aware toolpath planning algorithm that conforms precisely to a technician-designed preparation model while protecting adjacent teeth, and 2) a clearance-oriented end-effector yaw assignment strategy that allows intraoral access while reducing the risk of soft-tissue interference. Together, these features enable the robot to accurately mill the irregular tooth surface with an average geometric deviation of 0.117 mm (RMSE), achieving both restoration quality and clinical safety. A series of simulations and phantom-head experiments validate the system's feasibility and effectiveness. |
| 2026-08-25 | [SIREN-Bench: Behavior-Driven Generation and Evaluation of Emergency-Vehicle Interactions](http://arxiv.org/abs/2608.24094v1) | Yicheng Zhu, Tianmu Zhao et al. | Emergency vehicles (EMVs) can reorganize surrounding traffic as civilian vehicles brake, change lanes, or form rescue corridors in response to their passage. Evaluating these safety-critical interactions requires behavior-level control over both EMV privileges and civilian responses, together with consistent sensing and ground truth. Existing datasets and simulation benchmarks do not directly provide this combination. We present \textbf{SIREN}, a behavior-driven SUMO--CARLA co-simulation platform for generating EMV--civilian interactions. SIREN couples SUMO's network-level traffic evolution and behavior logic with CARLA's continuous vehicle control and synchronized onboard sensing; depending on the active behavior, the interaction is controlled by SUMO, CARLA, or jointly. We instantiate the platform as \textbf{SIREN-Bench-v1}, comprising seven parameterized interaction templates across emergency levels L1--L3 and three behavior families, with synchronized sensor observations and simulator-native annotations. We demonstrate the benchmark through three representative tasks: 3D object detection, trajectory prediction, and vision-language risk understanding. Evaluations of nine trajectory predictors, four LiDAR-based detectors, and five vision-language models reveal behavior-dependent failure modes. Traffic-clearance interactions are hardest for detection, privileged intersection traversal is hardest for prediction, and no learned predictor outperforms the constant-velocity reference on average. Vision-language models perform substantially better on normal traffic than on near-miss and collision events. These results demonstrate the value of behavior-centered benchmarking and establish SIREN as an extensible data-generation and evaluation platform for autonomous-driving and transportation safety research. |
| 2026-08-25 | [Safe Distributed Generalized Nash Equilibrium Seeking via Control Barrier Functions](http://arxiv.org/abs/2608.24077v1) | Yihan Meng, Weijian Li et al. | In this paper, we consider generalized Nash equilibrium (GNE) seeking in non-cooperative games with coupled constraint sets. Specifically, we aim to enforce safety for distributed GNE seeking, whereby the safety specifications are encoded in the coupled constraint set. To achieve this, we introduce the control barrier function (CBF) in the design of the GNE seeking dynamics. We design the dynamics for both full- and partial-information setting, where each player has knowledge of the decision information of all other players or only neighboring players, respectively. We justify the proposed dynamics by showing that the coupled constraint set is forward invariant, the equilibrium of the dynamics coincides with the exact GNE of the game, and the dynamics is asymptotically stable. Furthermore, we extend the approach to games where the agents are multi-integrators. Numerical simulations are provided to verify our results. |
| 2026-08-25 | [Curved Inference II: Sleeper Agent Geometry - Extending Interpretability Beyond Probes](http://arxiv.org/abs/2608.24037v1) | Rob Manson | This paper extends Anthropic's Sleeper Agents research [1], which showed artificial backdoors persist through safety training & can be detected by linear probes with >99% accuracy [2]. However, probe-based detection relies on linear separability that may be an artefact of backdoor insertion rather than a property of naturally occurring deceptive alignment. Sophisticated deceptive behaviours emerging through natural training are unlikely to produce such convenient linear signals.   We introduce a naturalistic methodology using multi-turn context windows that simulates realistic deceptive reasoning without artificial triggers or supervised backdoor insertion. Rather than binary trigger-response patterns, we examine how semantic complexity emerges through gradual context development.   Building on our Curved Inference framework, we analyse curvature, salience, & introduce semantic surface area (A'), a new metric of representational work capturing both the magnitude & directional change of meaning construction in unnormalised residual space. Without backdoors, labels, or probes, we apply this framework to naturalistic deceptive prompts & classify model outputs via LLM consensus.   Geometric structure reliably predicts semantic classification, with statistically significant differences in surface area across five prompt strategies & two model families. Critically, measurement precision can reveal geometric signatures hidden by classification noise - some strategies improve from non-significant (p = 0.555) to significant (p = 0.048). This validates that sophisticated reasoning creates intrinsic geometric patterns that persist even when detection appears to fail, suggesting the shape of inference itself encodes semantic patterns regardless of whether models have learned to suppress linear indicators of deception - a scalable, unsupervised path for detection when linear methods fail. |
| 2026-08-25 | [Low-Latency Activation-Regularized Sparse Neural Operators with Distillation Assistance Towards Real-Time Edge-Deployable Virtual Sensing](http://arxiv.org/abs/2608.23987v1) | William Howes, Farid Ahmed et al. | Virtual sensing enables digital twins and safety-critical systems to reconstruct and forecast spatial-temporal physics in real time. However, conventional computational and data-driven methods often face challenges in generalization, latency, and energy efficiency for edge deployment. Neural operators offer a promising alternative but remain reliant on power-intensive hardware. Spiking neurons and neuromorphic computing can improve efficiency, yet surrogate-gradient training and multi-step spiking introduce convergence and latency challenges. We propose the Sparse-Activation-ReLU (SAR) layer, a single-step alternative that promotes activation sparsity without surrogate-gradient training while remaining compatible with event-based computing. Within a trunk-based NOMAD architecture, SAR achieves over a fivefold improvement in the combined Latency-Error-Energy (LEE) metric compared with Variable Spiking Neuron (VSN) and Leaky Integrate-and-Fire (LIF) implementations. We further analyze spiking entropy and feature usage and introduce synthetic knowledge distillation, reducing the LEE score by more than twofold. Finally, we improve VSN through a ReLU-based spiking loss and graph-neighbor thresholding. On the Heat Exchanger dataset, these approaches reduce L2 error by more than twofold and nearly sevenfold, respectively, while reducing spiking and spatial aggregation. Overall, the work presented is a step towards energy-efficient virtual sensing by providing an alternative framework that can be positioned towards neuromorphic or other edge device integration that can be a gold standard to compare latency, energy, and error performance for future efficient designs that are sparsity or brain-inspired spiking based. |
| 2026-08-25 | [Safety-aware Model Predictive Path Integral Control with Signal Temporal Logic](http://arxiv.org/abs/2608.23972v1) | Yiqi Zhao, Taekyung Kim et al. | Safety-aware motion planning remains a challenge in robotics, especially when missions are time-critical and are under complex specifications. In this paper, we propose safety-aware-stl-mppi, a computationally efficient sampling-based receding-horizon planning framework designed to promote satisfaction of constraints expressed in Signal Temporal Logic (STL). Our approach encodes discrete-time STL formulas into candidate time-varying control barrier functions (CBF), which are integrated into a model predictive path integral (MPPI) controller. Our method inherits the benefits of low computational cost from an efficiently parallelizable sampling based planner and utilizes CBF for constraints expressed in STL. We compare against several MPPI baselines using four artificial Mars Rover planning case studies with a diverse environment and cost setups, where we show our method consistently achieving high safety and efficiency. We show a quadcopter planning experiment with NVIDIA Isaac Lab. |

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



