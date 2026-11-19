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
| 2026-11-18 | [Jailbreak Detection on Large Language Models via Contrastive Prototype Structures](http://arxiv.org/abs/2611.24936v1) | Yan Qiu, Yuanyuan Li | Jailbreak attempts and harmful prompts remain a challenge for the safe deployment of large language models. We study whether contrastively learned representations can support safety detection without training an additional downstream classifier. We propose Jailbreak Detection with Prototype Inference (JDPI), which uses binary safety labels to construct harmful and benign prototypes and makes predictions from relative prototype distances. JDPI uses only binary safety labels for encoder adaptation, prototype construction, and validation. In our setup, it uses fewer labeled examples than the classifier baseline (Method A) by avoiding the separate split used to train the downstream classifier. Across four benchmarks covering different input formats, contrastive fine-tuning improves JDPI's detection accuracy by 9.1-18.0 percentage points. Using the same encoder, JDPI achieves slightly higher ROC-AUC on one benchmark and stays within 0.017 of the baseline across all four. With dataset-specific adaptation, JDPI exceeds the released Llama Guard 3-8B baseline by 0.173-0.310 in ROC-AUC across the four benchmarks. In multi-turn replay on TomGibbs, JDPI uses accumulated user history as the conversation unfolds, achieving higher conversation-level accuracy than when it uses the current turn alone. Across 12 directed transfers, JDPI outperforms the classifier by an average of 1.59 percentage points in accuracy, demonstrating stronger cross-dataset robustness. |
| 2026-11-18 | [Learning to Interact in Multi-Agent Self-Play for Controllable Autonomous Driving Scenario Generation](http://arxiv.org/abs/2611.24876v1) | Beyazit Yalcinkaya, Ethan Pronovost et al. | Large-scale, multi-agent self-play trains robust driving policies effectively in simulation, but controlling these models to generate specific, safety-critical edge cases remains a key challenge. Existing goal-conditioned self-play frameworks rely primarily on static map-location goals and fail to reliably produce complex agent interactions such as aggressive tailgating or adversarial cut-offs. These agent interactions are necessary for rigorous autonomous vehicle (AV) testing. To address this, we present an extended multi-agent self-play framework that introduces agent-relative dynamic goals. By conditioning policies on this extended goal representation, our framework enables targeted, controllable scenario synthesis. We demonstrate that our approach successfully generates interactive driving behaviors that static-goal baselines cannot simulate, providing a flexible tool for long-tail scenario generation in AV testing. |
| 2026-11-18 | [Cross-Vehicle Transferability of Lateral Velocity Estimation on Production Vehicles through Input Selection and Data Augmentation](http://arxiv.org/abs/2611.24747v1) | Hendrik Schäfke, Simon F. G. Ehlers et al. | Lateral velocity is safety-critical for vehicle-dynamics control but cannot be measured with production sensors. Model-based estimators such as Kalman filters are established, whereas data-driven estimators are more accurate within the training regime but suspected to lose accuracy beyond it. Their transfer to a structurally different vehicle has rarely been tested. We record real-world test drives of three series-production vehicles, train a neural-network estimator on production CAN signals of a compact car, and evaluate it zero-shot on an SUV and a van against a GNSS-aided inertial reference. On the unseen vehicles, the network's error exceeds that on the training vehicle by an order of magnitude, a gap not quantified on production vehicles before. A systematic comparison of input sets from ten production CAN signals identifies a five-signal set that transfers most robustly. To narrow this gap, we introduce three physically motivated augmentations of training data. Two improve generalization, namely exploiting the vehicle's left-right symmetry and emulating accelerometer bias. Combined, they reduce the error on the validation maneuvers of the SUV and van by 13% and 15%. Contrary to common expectation, on the SUV the data-driven estimator trained with both augmentations matches a model-based estimator tuned on the same training data. |
| 2026-11-18 | [The Limit Is in the Input, Not the Model: State-Dependent Command Safety in Natural-Language Interfaces to Industrial Controllers](http://arxiv.org/abs/2611.24682v1) | Hemen Taukoor, Zhaoxia Wang et al. | Language models judging commands to industrial controllers may not receive machine state, the equipment's current condition. The same command can require opposite safety decisions in different states, so an incorrect answer may reflect missing information rather than poor reasoning. We propose an answerability audit: does the model receive the information needed to decide? The audit records the model's inputs and pairs the same command with two states requiring opposite answers. The paired score counts a pair only when both answers are correct. We add rules, then state values, to the model's input. In an author-built test of 15 pairs from three systems with four language models, none reliably distinguished the paired cases without state, even with rules. With state, scores ranged from 0 to 13 correct pairs out of 15; three models improved. With text alone, one model always allowed commands and another always denied them. Their overall (raw) accuracies differed by 32 percentage points; both scored 50% balanced accuracy, which weights safe and unsafe cases equally, and no correct pairs. A rule evaluator matched all 15 pairs by applying the same rules that supplied the expected answers; this is not an independent safety test. The audit complements systems that check actions against safety rules: it checks whether the model receives the information needed for those decisions. This helps evaluators separate missing-input problems from model errors. These small-sample scores do not establish general model reliability. |
| 2026-11-18 | [LIGO-Virgo-KAGRA stochastic constraints on first-order phase transitions: A tale of priors](http://arxiv.org/abs/2611.24625v1) | S. AbdusSalam, J. Kersten et al. | We re-examine the methodology for constraining cosmological first-order phase transitions from gravitational-wave searches using Bayesian statistical analysis. Specifically, we use results from LIGO-Virgo-KAGRA searches for a stochastic gravitational wave background during observing runs O1--O4a. We explore multiple prior choices across three categories of fit parameters: geometrical parameters, thermodynamical parameters of phase transitions, and the parameters of an Abelian Higgs particle physics model. Our key finding is that the constraints on most parameters describing gravitational waves from a phase transition depend strongly on the prior. Consequently, the data do not yet allow to derive robust constraints on these parameters. Besides, even if a parameter is meaningfully constrained, the upper limit can vary significantly with the prior range. We provide practical recommendations for assessing the robustness of parameter estimates and guidelines for rigorously testing prior independence in future analyses. |
| 2026-11-18 | [Negotiating Control and Accountability among Stakeholders in High-Risk AI Use Cases: Development and Evaluation of the Method ALIGN](http://arxiv.org/abs/2611.24609v1) | Lena Schneider, Daniel Boos et al. | Characteristics of Artificial Intelligence (AI) severely restrict human control. This creates new challenges for socio-technical systems design especially in safety-critical domains, where control over risks is essential along with the accountability for any harm caused. Drawing from human factors and AI governance literature, expert interviews and participant observation in a complex high-risk AI use case from the railways, we developed ALIGN, a scenario-based method that supports the governance of and stakeholder involvement in AI development and implementation in three ways: raising awareness for potential accountability-control misalignments in socio-technical systems and their negative consequences, offering a structured approach to assessing the alignment between accountability and control for stakeholders in a given use case, and supporting effective design decisions to mitigate AI risks. We describe the iterative development and evaluation of ALIGN and discuss how this method promotes AI governance by highlighting the relevance of control-accountability alignment throughout the AI life cycle. |
| 2026-11-18 | [TinyARM-X: A Series of Open-Source, Configurable, and Transparent Torque-Controlled Desktop-Scale Robotic Manipulators](http://arxiv.org/abs/2611.24554v1) | Nicolas Kosanovic, Jean Chagas Vaz | Visuomotor policies have demonstrated that imprecise low-cost robots can autonomously complete complex manipulation tasks. However, most inexpensive manipulators rely on compact and rigidly position-controlled servos for actuation, which complicate simulation, operation, and safety. This work introduces TinyARM-X, a framework for easily manufacturable, configurable, and torque-controllable low-cost (<350 USD) serial manipulators. By using a single hardware/software architecture, a <7 Degree of Freedom (DoF) TinyARM can be manually reconfigured to an operational >3 DoF robot within 10 min. Actuator modeling and dynamic compensation enable torque-controlled applications like mixed-reality impedance control teleoperation and simulator force feedback rendering. Experiments include mechanical benchmarking and real-world force-aware telemanipulation. The TinyARM-7 has a payload of 33.5 g, a max speed of 0.77 m/s, and a reach of 210 mm. Moreover, with dynamic compensation, the robot has a backdrive-breakaway force of 1.4 N and a backdrive-running force of 0.7 N. Additionally, Cartesian-Impedance Control (CIC) attains a translational stiffness of >15 N/m. Lastly, the TinyARM-X stack is open-sourced to disseminate accessible desktop-scale torque-controlled robotic manipulation to the community. |
| 2026-11-18 | [Risk-Aware Safe Hierarchical Learning for Robust eMBB-URLLC Resource Allocation in O-RAN](http://arxiv.org/abs/2611.24497v1) | Mohammad Hussain, Dilara Gurer et al. | Joint resource allocation for enhanced mobile broadband (eMBB) and ultra-reliable low-latency communication (URLLC) must satisfy stringent URLLC latency and reliability requirements while maintaining eMBB throughput under uncertain traffic and channel conditions. Existing learning-based allocators often optimize average aggregate performance without explicitly accounting for prediction errors, potentially resulting in severe service degradation for individual users. We propose a risk-aware safe hierarchical learning framework for Open Radio Access Networks (O-RAN) that learns resource-feasible allocations using service-cost penalties and a per-user URLLC service test. A non-real-time policy determines slice-level resource-block and power budgets, near-real-time policies allocate resource blocks and transmit power among users, and an execution-time safety layer checks per-user URLLC service and corrects allocations through bounded resource transfers when feasible. A deterministic per-user service test, derived using protected rates accounting for channel state information prediction errors and conformal upper bounds on traffic demand, ensures satisfaction of the probabilistic URLLC requirement. Service-specific conditional value-at-risk objectives prioritize users with the highest URLLC deadline-miss ratios and lowest eMBB rates. Experimental results show that, under nominal load, the proposed framework is the only evaluated method that satisfies both the URLLC deadline-miss target and worst-user eMBB rate criterion on average. It reduces the mean miss ratio by 64% relative to the best baseline in this metric and achieves the highest worst-user eMBB rate. |
| 2026-11-18 | [Probing agent overeagerness in realistic settings with continuation audits](http://arxiv.org/abs/2611.24491v1) | Sean Lim, Clement Neo | Frontier coding agents increasingly exhibit overeagerness, taking surprising, sometimes reckless actions beyond a user's mandate --- reported in recent system cards from frontier model developers. These cases are surfaced through internal deployment monitoring, which requires private traffic and is out of reach for open-weight models and independent evaluators. Automated alignment audits are the natural alternative, but capable models increasingly recognise the synthetic setup and adjust their behaviour. We introduce Continuation Audits, a pipeline that grounds the auditor on a prefix drawn from a real coding-agent transcript and directs it to continue the trajectory toward a seeded behaviour. Because the prefix is drawn from a real session, the auditor only needs to maintain a realistic setting rather than manufacturing it. We validate the method on SWE-Chat prefixes: on a paired subset of 90 audits a judge cannot reliably distinguish continuations from their source sessions (realism win rate 0.42 vs.\ 0.43), and across all 900 audits verbalized eval-awareness does not exceed the rate observed in the unaudited real sessions. Applying the pipeline to three recent models across three seeds, we find each model leads a different seed --- DeepSeek~V4~Pro on colleague-credentials misuse, GPT-5.6~Terra on file-permission bypasses, and Kimi~K3 on treating diagnostic questions as instructions. Kimi~K3 additionally continues to bypass file-permission protections after an explicit CLAUDE.md prohibition while the other two stop entirely, corroborating Moonshot's own report of excessive proactiveness. |
| 2026-11-18 | [TechReport: Implementing CHERI in gem5](http://arxiv.org/abs/2611.24460v1) | Philipp Schmitz, Vishwas Vasuki Gautam et al. | Despite countless mitigation efforts, memory safety issues still prevail. The recently discovered security issues in PHP underline this fact. One of the most promising approaches to solving this problem is the concept of Capability Hardware Enhanced RISC Instructions (CHERI). There is a well-developed ecosystem of ISA simulators, compilers, operating systems, and hardware designs that support CHERI. However, the lack of a simulator modeling microarchitectural detail hinders efficient and fast design space exploration for CHERI-enhanced designs means the performance impact of CHERI remains understudied. Therefore, we present gem5-CHERI: a CHERI-enhanced adaptation of the gem5 simulator that can boot the CheriBSD operating system and run custom workloads. The simulator is available open-source on GitHub. |
| 2026-11-18 | [Keeping Shielded Data-Driven Controllers Safe Under System Evolution via Model Consistency](http://arxiv.org/abs/2611.24428v1) | Roland Stolz, Vitus Lüntzel et al. | Cyber-physical systems (CPS) increasingly contain data-driven controllers, which raise safety concerns given their black-box nature. A common mitigation is to enforce system safety via a shield around the controller, which restricts the inputs based on assumptions about the physical system. However, these assumptions are usually fixed and may become invalid as the system evolves (during development or due to component degradation), possibly leading to unsafe behavior. We address this limitation in a model-driven approach that treats both the data-driven controller and safety shield as first-class models that are integrated into a virtual single underlying model (V-SUM) alongside system models. We describe system evolution as changes in the associated models, such that consistency checking yields a continuous assessment of safety guarantees. In case of inconsistencies, consistency-preservation rules automatically update the models and restore a valid safety shield. We evaluate our approach on a reach-avoid task for a unicycle system. The results show that consistency-driven evolution effectively maintains safety guarantees under system changes and provides a principled basis for evaluating and restoring performance. |
| 2026-11-18 | [Risk-Triggered Search: Behavior-Preserving Safe Control](http://arxiv.org/abs/2611.24409v1) | Xinyu Deng, Bohao Lan et al. | Pre-trained visual world models generate multi-step action plans from current and goal observations, yet terminal-proximity objectives fail to prevent forbidden-zone traversal or rigid-body collision sliding. Purely visual interfaces may lack analytical dynamics, structured states, or explicit collision geometry, requiring internal prediction of action-conditioned process risk. We propose Risk-Triggered Search (RTS) for frozen Joint-Embedding Predictive Architecture (JEPA) world models. Rather than relying on static state features that cannot distinguish candidate plans sharing an observation, RTS unrolls action-conditioned future latent prefixes in the frozen world model and trains a lightweight risk head with process-violation supervision. To intervene selectively while preserving safe nominal behavior, RTS uses the same predicted prefix-risk criterion as both the search trigger gate and hard feasibility gate: safe nominal plans execute directly, whereas alarms trigger local Cross-Entropy Method (CEM) search to minimize nominal deviation within the feasible set. To prevent isotropic Euclidean minimal modification from shrinking obstacle-directed actions into zero-amplitude stalling, we introduce an amplitude-preserving metric $d_{\mathrm{AP}}$ that retains tangential progression while maintaining a unique zero at the nominal plan. Under unified gating and this unique-zero condition, bypassing safe-state search yields closed-loop trajectories strictly identical to every-step same-kernel search. Evaluated under a process-safety benchmark and Safe Arrival Rate (SAR) metric across five tasks, RTS improves SAR over Direct by 12.67--41.67 percentage points on navigation and articulated control and +2.08/+4.17 pp on Cube/PushT boundary cohorts, maintaining 100% action identity with every-step search while cutting controller time by 17.71%--70.41% across 1,968 paired trajectories. |
| 2026-11-18 | [Deployment of a Graph-Based Track Seeder for the ATLAS Phase-II Tracking Upgrade](http://arxiv.org/abs/2611.24307v1) |  ATLAS Collaboration | The ATLAS experiment is undergoing significant detector, computing and software upgrades for the High-Luminosity LHC (HL-LHC). The ongoing software modernisation is essential to leverage future processing technologies and ensure maintainability throughout the operation of the experiment, while meeting the tracking performance requirements for physics and reducing the compute needs for event reconstruction. To meet these goals, the ATLAS Collaboration is adopting the ACTS (A Common Tracking Software) toolkit, an experiment-independent software for charged-particle track reconstruction. Using ACTS in the ATLAS reconstruction software at the HL-LHC facilitates the development and integration of new algorithms while promoting thread safety, high performance, and code maintainability. This document describes the development and recent deployment of a Graph-Based Track Seeder (GBTS), its underlying methodology, and its expected tracking performance at the HL-LHC. |
| 2026-11-18 | [Learning to Coordinate: LLM-Guided Workflow Evolution for Token-Efficient Red Teaming](http://arxiv.org/abs/2611.24221v1) | Xiaoyang Xing, Wenyu Dai et al. | Automated red teaming is essential for identifying safety failures in large language models (LLMs), but its practical scalability depends on the resources required to discover those failures. Multi-agent systems can combine complementary capabilities, yet repeated planning, redundant exchanges, and ineffective information routing can make their coordination expensive. Moreover, a target model's response indicates the outcome of a test without identifying which collaboration decision should change. We present Learning to Coordinate (L2C), an LLM-guided framework that evolves an executable multi-agent workflow for each red-teaming case. L2C represents agent responsibilities, execution dependencies, and input bindings in a versioned graph, and associates each graph with records of the information actually delivered and produced. A controller trained through supervised fine-tuning integrates these records with target responses, evaluator verdicts, and remaining budget to retain the workflow, refine instructions, restructure collaboration, or lock further editing. This design connects feedback to explicit changes in subsequent execution while allowing a useful organization to continue without repeated controller intervention. The evaluation protocol covers HarmBench with three execution backbones and two target models, together with component ablations, dataset transfer, and evaluator checks. Across the six executor-target configurations, L2C improves ASR over EvoSynth by 1.4 to 4.5 percentage points while reducing mean online tokens by 35.6% to 37.3%. L2C connects adaptive collaboration to the resource requirements of automated security evaluation. |
| 2026-11-18 | [Assessing Vision-Language Models for Early Prediction of Unsafe Situations in Robotics](http://arxiv.org/abs/2611.24146v1) | Muhammad Farooq, Aitor Arrieta et al. | Vision-Language Models (VLMs) process visual information and textual instructions and, by using their ability to understand and reason about context, are well-suited for predicting robot failures before they occur. However, evidence supporting this remains limited. To address this, we present an empirical evaluation of VLMs for predicting robot failures. In particular, we experiment with Qwen, InternVL, and LLaVA-OneVision across three robotic case studies (HuRoN, Pal Robotics, and CrowdBot), analyzing both video and odometry data for early failure warning (i.e., before an unsafe situation occurs). Specifically, we evaluate their performance in classifying situations as safe, potentially unsafe, and unsafe, together with uncertainty quantified using various metrics, and study the relationship between uncertainty and performance. Our results showed that Qwen provides the strongest overall classification and prediction-reliability performance, while InternVL shows stronger early-warning capability in several configurations. The results further show that the effect of robot-state information and temporal history depends on the model and evaluation objective. Overall, our findings indicate that general-purpose VLMs can support early robot-safety monitoring, but the most suitable configuration depends on whether the priority is accurate assessment of safety-state, reliable confidence and uncertainty, or timely warning before unsafe events. |
| 2026-11-18 | [VESPA: Verifiable Symbolic Policy Adaptation for Safe Reinforcement Learning](http://arxiv.org/abs/2611.24083v1) | Paul König, Sascha Marton et al. | Safe reinforcement learning (RL) aims to prevent agents from taking actions that are dangerous or destructive to their environment. This is especially important for safety-critical domains, such as robotics or autonomous driving. Formal verification can be used to prove agent safety through safety constraints. However, in practice, many policies do violate safety constraints, and thus are verified to be unsafe. Since the goal is to obtain a safe policy, we would need to retrain a new policy from scratch and then verify it again until a candidate happens to pass. Both training and verification are usually not cheap, which makes this process inefficient. We introduce VESPA ($\textbf{VE}$rifiable $\textbf{S}$ymbolic $\textbf{P}$olicy $\textbf{A}$daptation), a framework for targeted fine-tuning of decision tree policies. We show that using counterexamples for policy adaptation can make existing unsafe policies safe, or at least make considerable progress towards safety. Targeted fine-tuning of the existing policy on those counterexamples enables it to learn how to avoid violating safety constraints. We thus use the result of a formal verification as a training signal for fine-tuning. On CartPole, VESPA raises the mean number of verified safe steps from 63.8 to at least 92.7, reaching after two iterations what the same loop without counterexamples reaches after ten. On MountainCarContinuous, VESPA turns an unsafe policy into a safe policy in 53 of 60 runs. We conclude that counterexamples from a verifier are an effective training signal, which can be utilized to systematically improve the safety of policies. Our implementation is available at: https://anonymous.4open.science/r/VESPA-Submission/ |
| 2026-11-18 | [From Data to Valves: A Self-Learning, Production-Grade Edge Stack for Sustainable HVAC Control](http://arxiv.org/abs/2611.24052v1) | Abbas Raza Ali, Muhammad Hamza Rehman et al. | Learned HVAC controllers rarely leave the simulator, and those that do are typically trained offline, frozen at deployment and retrained from the cloud. We present a production stack in which three learning components run on one NVIDIA Jetson Orin gateway per building, adapt continuously from the building's own data, and verify their own effect. (a) A compact transformer-encoder forecaster, one per equipment group with learned per-unit embeddings, predicts every air-side unit's next-cycle temperature and learns prequentially, one gradient step per five-minute cycle, behind five independent gates that bound the influence of any corrupt reading. (b) A distributional soft actor-critic agent, one shared policy across unit types, converts the measured state into a chilled-water valve position under a structural safety envelope that no learned component can soften; its second generation learns nightly on the device from recorded transitions alone, with a behaviour anchor and critic calibration terms, is scored by four model-free metrics and promoted by atomic hot-swap. (c) A change-point regression baseline held to ASHRAE Guideline 14 supplies the counterfactual against which savings are certified. On eighteen months of data from a hotel the forecaster converges from random weights within a week and reaches 0.37, 0.39 and 0.11 degree C MAE for air-handling, fresh-air and fan-coil units with no retraining event; on a different hotel the baseline reproduces a 46-day held-out period to within 0.07%. The RL agent has completed its in-simulator campaign and runs in shadow on 104 units; we report measured latency and memory of every service on a shared 15 GB device, the systems lessons of the campaign, and precisely what the evidence does and does not yet establish. |
| 2026-11-18 | [MARS Technical Report](http://arxiv.org/abs/2611.24006v1) | Tong Liu, Chaoqun Hou et al. | Modern industrial recommender systems use multi-stage cascades--retrieval, pre-ranking, ranking, and re-ranking--to narrow millions of candidates to a personalized list. While each stage deploys sophisticated models, pipeline-level decisions on channel activation, quotas, model selection, and score fusion remain engineer-tuned and user-agnostic within each scenario.   This report presents MARS (Multi-stage Agentic Recommendation System), extending personalization from individual-stage predictions to full pipeline orchestration. A near-line agent built on fine-tuned Qwen3-0.6B generates per-user configurations--a seven-dimensional action covering pipeline mode, retrieval quotas, model selection, fusion weights, and reranker settings--from a constrained action space. Configurations are validated, cached, and consumed asynchronously; the cascade never waits for LLM inference. Schema checks, fallbacks, and a kill switch ensure serving safety.   The policy is trained in three stages: Confidence-aware SFT distills schema-valid generation from a teacher model; Format-Anchored Preference Optimization (FAPO) learns action preferences via ActionMask with posterior weighting; Group Relative Policy Optimization (GRPO) refines the policy through Sandbox replay with a frozen reward model, without exposing exploratory actions to live traffic.   Two-week A/B tests across five Taobao scenarios with over ten million daily users show positive gains on all key metrics. The largest reach +5.22% V2P and +7.34% GMV, with positive CTR lifts on both click surfaces. About 20% of requests use a lightweight Minimal mode, and deployment shows 12 ms lower mean serving latency versus production. |
| 2026-11-18 | [SCAFFOLD: Toward Warrant-Preserving Evidence Transport for Reusable LLM-Agent Skills](http://arxiv.org/abs/2611.23990v1) | Nami Cho, Shailja Gupta | An agent skill that succeeded yesterday may be executable today without being justified for reuse today. Tests may cover only observable contract clauses, outcome oracles may be absent, evaluations may inherit the same erroneous specification, and permissions or dependencies may change between validation and invocation. We argue that the appropriate unit of prospective trust is a Warranted Reuse Claim (WRC): a defeasible, claim-specific authorization tied to a versioned skill, target context, evidence, assumptions, and identified defeaters. The paper sharpens this position through Warrant-Preserving Evidence Transport (WPET), a proposed discipline for deciding which historical evidence remains admissible after context change. It distinguishes evidence admissibility from truth of the supported safety or effectiveness claim; states a conditional preservation result and its limits; specifies three-valued applicability and asymmetric context-transfer judgments; and illustrates how provenance dependence and policy-invalidating changes alter authorization. We explicitly relate the proposal to assurance cases, contracts, and skill-library maintenance rather than claiming novelty for their ingredients. A falsifiable longitudinal evaluation would compare WRC-guided policies with strong version-aware baselines at matched coverage, utility, and review cost. This paper contributes a formalized problem statement, testable design obligations, and reporting protocol, not an implemented verifier, empirical superiority result, or proof of universal agent safety. |
| 2026-11-18 | [Can AI Understand Indonesian Children's Book Illustrations? Compliance, Safety, and the Cultural Alignment Gap](http://arxiv.org/abs/2611.23887v1) | Rian Adam Rajagede, Rochana Prih Hastuti et al. | Children's book illustrations carry developmental, pedagogical, and cultural responsibilities that are often formally codified in guidelines. As AI systems begin entering children's publishing pipelines, both as evaluators and as generators, a natural question arises: can they reliably meet the structural, safety, and cultural standards required? We investigate this across three criteria: guideline compliance, safety, and cultural alignment, using Indonesia's Book Leveling Guideline as a government-issued evaluation framework with explicit visual criteria for four reading levels. To understand how well VLMs can identify and apply these guideline criteria, we first evaluate them on real published children's book pages and compare their assessments against a human baseline. To then examine whether AI-generated illustrations meet the same standards, we apply the VLM judge to images produced by two generation models and cross-check them against a subset of human illustrator raters. We find that VLMs are reliable on explicit criteria such as compliance and safety, but fall short on cultural alignment, systematically missing content that human raters flag as culturally inappropriate. AI-generated illustrations show the same blind spot: they satisfy structural and safety criteria but fail to capture implicit cultural details that human raters consistently identify. |

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



