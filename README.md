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
| 2026-10-16 | [SparseMax-dSTL: Sound Robustness with Horizon-Independent Gradient Allocation for Temporal Logic Trajectory Optimization](http://arxiv.org/abs/2610.23837v1) | Sai Sandeep Damera, Ryan Matheu et al. | A gradient-based solver that enforces a Signal Temporal Logic (STL) specification uses a smooth approximation of its quantitative semantics, the robustness. In safety-critical trajectory optimization, this approximation must have two properties. First, it must be sound, that is, nonnegative only on trajectories that satisfy the specification. Second, at each temporal operator, the weight of its gradient on the violating samples must stay bounded away from zero, however long the horizon. The approximations in use, including the log-sum-exp, its sound variant, and the generalized-mean robustness, do not have both. We introduce lower approximations of the minimum and the maximum whose gradients are sparsemax projections, and we prove that the robustness built from them has both. It lies below the exact robustness and is therefore sound, and its error has a bound that is independent of the number of samples and known before optimization. The gradient of each approximation is exactly zero on every sample beyond a data-dependent threshold, so the weight on violating samples a gap beyond the others has a positive lower bound at every horizon. On a planar unicycle and on a seven-joint manipulator beside a person in simulation, the log-sum-exp approximation is positive on a violating trajectory. As the horizon of the manipulator's task grows from 10 to 80 s, the gradient weight of the sound variant on the violating samples falls from 0.22 to 0.04, while that of sparsemax stays at 1. In a constrained trajectory optimization of the unicycle, the constraint on the log-sum-exp robustness returns a violating trajectory, whereas the constraint on the sparsemax robustness returns a trajectory that satisfies the specification with the required margin. |
| 2026-10-16 | [OwlSight: A Panoramic Monitor for Chain-of-Thought Reasoning](http://arxiv.org/abs/2610.23670v1) | Lewen Yan, Yanxu Zhu et al. | Chain-of-Thought (CoT) monitoring offers actionable evidence for oversight AI systems, with the potential to enable early detection and intervention. However, existing approaches are focused on specific tasks, whereas the regulation of AI, particularly reasoning models, requires monitoring a broader range of tasks. We introduce Factor for CoT monitoring, a strategy that supports customizing monitoring requirements in the prompt, which enables a single monitor to cover different tasks by relying on different factors. Furthermore, we introduce two types of factors, fine-grained and coarse-grained (e.g., monitoring whether a specific email address has been leaked versus monitoring privacy leakage), to accommodate diverse user monitoring needs. Based on this, we develop a CoT monitor named OwlSight that relies solely on factor to monitor behaviors related to multiple dimensions, including capability, safety, and trustworthiness. Experiments across 15,971 reasoning trajectories show that OwlSight surpasses general-purpose LLMs by 1.32-30.57 and 11.71-27.81 points at fine and coarse granularity, respectively, and achieves state-of-the-art results on OpenAI's CoT monitorability benchmark. It also generalizes to unseen models' CoTs and remains robust to Factor rephrasings, CoT compression, and obfuscation, offering a promising paradigm for CoT monitoring that can contribute to the oversight of frontier AI. |
| 2026-10-16 | [Learning Robot Metareasoning: Adaptive Computation Budgets for Real-Time Control](http://arxiv.org/abs/2610.23662v1) | Lorenzo Magnino, Antonio Marino et al. | Planning while moving faces an historical trade-off between deliberation and reactivity. A planner running concurrently with execution needs time to compute, so the robot must commit to an open-loop action chunk for that duration: a longer commitment buys a better plan at the cost of feedback frequency. Existing controllers fix this action commitment, and consequently the computation budget, before deployment. However, the optimal computation time is highly state-dependent (due to factors such as unmodeled dynamics, safety constraints, and marginal optimisation gains), i.e. dynamic, therefore offline static tuning risks leaving systems blind during critical maneuvers or unable to find global effective solutions. We cast the identification of the optimal computation budget as a semi-Markov decision process (SMDP) in which a single decision defines both the robot action commitments and the time allocated to the solver. To solve this, we train a reinforcement-learning policy conditioned on the current state, the executing plan, and inexpensive planner statistics. Across ground-robot navigation, agile quadrotor flight, and contact-rich manipulation, the learned policy changes its budget with the demands of the scenario, discovering efficient reasoning patterns that outperform fixed-budget baselines. https://lorenzomagnino.github.io/learning-robot-metareasoning-review/ |
| 2026-10-16 | [Scalable long-range machine learning interatomic potentials with electrical response for atomistic material modeling](http://arxiv.org/abs/2610.23643v1) | Yicheng Chen, Xiaoyun Yu et al. | Accurate atomistic simulations of materials under electric fields require long-range electrostatics and electrical-response modeling at length scales beyond the reach of conventional first-principles methods. Yet these capabilities remain difficult to reconcile with the scalability of machine-learning interatomic potentials (MLIPs). Strictly local models enable efficient spatial parallelization but neglect long-range interactions, whereas existing electrostatic extensions introduce global dependencies that hinder distributed computation. Here, we overcome this limitation by analytically separating the long-range electrostatic forces and Born effective charges (BECs) into a global electrostatics calculation and strictly local derivatives of learned latent charges. This formulation preserves the spatial decomposability of equivariant MLIPs while enabling distributed evaluation of energies, forces, stresses, and BECs across different boundary conditions. Implemented as DPA4C-LR on the \texttt{nvalchemi} platform, the framework scales to 5.2 million atoms on four GPUs at approximately twice the computational cost of its short-range backbone. Trained on energies and forces, DPA4C-LR accurately infers BECs and reproduces the field-dependent infrared spectra of water. We further demonstrate its capabilities by recovering the characteristic slim polarization hysteresis of relaxor \ce{Pb(Mg_{1/3}Nb_{2/3})O3}, which is obscured by finite-size effects in smaller simulations; predicting a room-temperature field-driven ionic conductivity of $6.8 \pm 0.7$~mS/cm in amorphous \ce{Li_{0.9}NbO_{0.9}Cl_{4.1}}; and modeling interfacial thermal transport across Si/diamond interfaces with slab-corrected electrostatics. By combining long-range physics, electrical-response inference, and distributed computation, this framework extends predictive atomistic modeling toward device-relevant length scales. |
| 2026-10-16 | [PresenceBench: An Exploratory Evaluation of Large Language Model Adaptation to User State](http://arxiv.org/abs/2610.23625v1) | Peña Arroyo Gallego, Mauro Ruiz Moreno et al. | Presence-aware intelligence is the capacity of an AI system's intelligence layer to adapt appropriately to the functional state of the person it serves (here readiness, flow, and stability) while preserving task quality and safety. We introduce PresenceBench, an instrument for measuring it. PresenceBench evaluates how the intelligence layer uses a supplied state, not how that state is sensed, and breaks the capability into measurable properties. Two are tested here. Receptivity: does the model take up the state and give a better response than with no signal? Specificity: does any improvement depend on the correct state, rather than on extra structured context? A third, adaptation across states, is a required endpoint not measured in this release. A five-arm causal design answers both questions, against each model's own baseline and an equally structured mismatched-state control, scored by a correctness-gated human paired comparison. The exploratory evaluation covered four models across eighteen single-turn, six fixed three-turn and six simulated natural-conversation scenarios. Supplying a synthetic state, raw or translated into a response policy, gave human-preference point estimates above 0.5 for all four models (Claude Sonnet 5, DeepSeek V4 Pro, Gemini 3.6 Flash, GPT-5.6 Terra), with three of eight 95% intervals excluding chance. Specificity was not resolved: no per-model interval excluded chance, and the design cannot yet separate the state values from a generic instruction to adapt. Policy delivery reduced priced cost and output tokens against each model's baseline; these are resource-use changes, not efficiency gains, because correctness non-inferiority was not established. Safety-floor failures were rare (13 of 360 cases), with a non-significant excess under policy delivery. We report the instrument, the frozen results, and the controls a confirmatory study needs. |
| 2026-10-16 | [Stereo LWIR-RGB Automatic Labeling for Low-Light Camera-Based VRU Detection](http://arxiv.org/abs/2610.23621v1) | Dimitrios Bouzoulas, Pyry Jäppinen et al. | VRU detection in low-light conditions is a key component in ensuring VRU safety during the most collision prone time of the day. Currently, advances in low-light detection are limited due to the lack of large scale datasets and automated labeling methods. Most autolabeling methods use RGB detections as their labels source, which cannot be replicated in low-light conditions. To fill this gap in this paper we present an automated pipeline for low-light VRU labeling. We use LWIR-RGB image pairs and source our labels from LWIR as VRUs differentiate from the background due to their body temperature. Unlike our previous work that relied on a complex hardware setup to align the two modalities, we perform this experiment using a LWIR-RGB stereo setup, lowering hardware complexity and cost, making the pipeline easier to reproduce. We evaluate method end-to-end on a dataset collected in the capital region of Finland. Models trained on our generated labels beat both off-the-shelf baselines and models fine tuned on a hand labeled subset on F1, mAP@50 and LAMR metrics when tested on a manually labeled dataset, and the result holds for two different LWIR detectors. This shows that low-light VRU autolabeling can be made both accessible and effective without specialized alignment hardware. |
| 2026-10-16 | [SIT-Eval: Situation-Based Interactive Turn-by-Turn Evaluation of Conversational Agents](http://arxiv.org/abs/2610.23612v1) | Ravil Mussabayev, Saveliy Zavertyaev et al. | The rapid proliferation of conversational AI agents necessitates evaluation methods that assess performance across diverse dialogue scenarios, utilizing multiple situational criteria, as appropriate behavior depends on context and encompasses various aspects. Existing dialogue benchmarks typically use static natural conversations, which underrepresent rare situations---a critical safety gap, as low-probability events become frequent failures at production scale. However, limited sample sizes force a choice: either maintain natural dialogues for realism or turn to adaptive situation sampling for comprehensive yet statistically reliable coverage. We introduce SIT-Eval, an adjustable framework for evaluating situation-based conversational agents. We formalize an agent's desired behavior as situational rules: if situation is P, follow rules Q. At each conversation turn, an LLM-interlocutor enacts a uniformly sampled situation. A context-aware LLM-judge leverages all conditionally applicable rules as criteria to evaluate the agent's reply for that situation. The judge is validated using human-verified synthetic golden dialogues. Benchmarking 8 LLMs on 1292 conversation turns reveals failure modes hidden by aggregate metrics: models with similar overall scores vary widely in their adherence to individual rules. All models struggle most with ambiguity clarification and Socratic questioning. |
| 2026-10-16 | [Integrated sensing, communications, and identification for low-altitude economy](http://arxiv.org/abs/2610.23602v1) | Shanxing Zeng, Jun Wang et al. | With the rapid development of low-altitude economy (LAE), low-altitude information networks (LAINs) must provide reliable communication, environmental sensing, and target identification to support efficient and safe airspace operation. Although integrated sensing and communications (ISAC) provides a foundation for the joint design of communication and sensing, identification must also be incorporated to meet airspace safety and regulatory requirements. This paper reviews integrated sensing, communications, and identification (ISCAI) for LAE. We first review low-altitude communication, sensing, and identification approaches and explain how representative applications motivate their integration. We then summarize low-altitude ISAC fundamentals, including system architectures, channel modeling, unified signal models, and performance metrics, followed by communication-centric and sensing-centric designs, resource allocation, and aircraft trajectory optimization. Building on these, we introduce ISCAI, focusing on a backscatter communication (BackCom)-based identification mechanism in which backscatter devices serve as electronic license plates for aircraft. We present system and signal models for integrated sensing and identification (ISAI) and ISCAI, together with multifunctional receiver designs and mutual integration gains. Finally, we discuss open problems and future directions, including an information-theoretic framework, artificial intelligence-empowered system design, and reconfigurable intelligent surface-enhanced ISCAI. This review provides a structured perspective on the evolution of LAINs toward integrated communication, sensing, and identification. |
| 2026-10-16 | [Reason-guided Human Reward Modeling for Preference-based Reinforcement Learning](http://arxiv.org/abs/2610.23550v1) | Anshul Nayak, Shahil Shaik et al. | Reinforcement learning (RL) policies optimized for task rewards may produce behaviors that are efficient but misaligned with human expectations, safety preferences, or intent. This paper presents a reason-guided preference-based reinforcement learning (PbRL) framework that learns a human reward model from trajectory-level feedback. In addition to preference labels, the framework incorporates reason tags explaining why a behavior is preferred, enabling the learned reward to capture both human choices and their underlying rationale. Compared with preference-only reward learning, the reason-guided formulation provides additional supervision for learning human-aligned rewards and improves the interpretability of the resulting policy. The learned human reward is combined with the environment reward during policy optimization, allowing the agent to improve task performance while producing more interpretable, human- aligned, and preference-sensitive behavior |
| 2026-10-16 | [Uncertainty-aware AI-enabled framework for predicting hydrogen dispersion in confined spaces to enhance operator situational awareness](http://arxiv.org/abs/2610.23515v1) | Niclas Flehmig, Mary Ann Lundteigen et al. | We present a two-stage AI-enabled framework for enhancing operator situational awareness during compressed hydrogen bunkering in maritime confined spaces. In the event of a leak, the framework first estimates the unknown leakage rate from early sensor readings and then forecasts how the hydrogen cloud will disperse over the next time steps, providing operators with a predicted concentration map together with confidence bounds. The framework is based on two independent AI models building one AI system. The AI system is designed as a decision-support tool. It augments sensor readings, and thus enhances operator situational awareness and supports decision-making. This enables preventive safety barriers to be deployed before hazardous concentrations are reached, while an independent emergency shutdown system remains the ultimate safety layer. The system design incorporates safeguards to ensure the AI system operates within its competence envelope. Trained and tested on an open dataset of confined-space hydrogen releases, the framework achieves a mean absolute error of $0.052$~g/s in leakage-rate estimation and $\approx0.018$ in hydrogen-concentration prediction, approaching the sensor precision bound of $\pm0.01$. The hydrogen-concentration prediction provides well-calibrated uncertainty estimates that allow operators to judge the reliability of each forecast. These results suggest that the framework may provide early information about hazardous concentration build-up, potentially supporting preventive actions by operators. |
| 2026-10-16 | [RECREG: Recovery-based Shielding for Regular Safety Properties in Reinforcement Learning](http://arxiv.org/abs/2610.23445v1) | Alexander W. Goodall, Francesco Belardinelli | In many real-world scenarios we need to consider formal requirements, such as safety and constraint compliance, to deploy Reinforcement Learning (RL) agents. In this paper we introduce RecReg, a safe RL framework capable of enforcing non-Markovian regular safety properties via recovery-based shielding. RecReg is a unified approach that supports different levels of a priori knowledge and safety guarantees. The paper primarily focuses on building the theoretical foundations for RecReg in the tabular setting, providing rigorous safety and convergence guarantees that are not present in related work. We provide an extensive set of empirical results using the MASA-Safe-RL library, and we outline an extension of RecReg to deep RL domains using SOTA model-based RL. Our empirical results demonstrate that RecReg achieves efficient learning and matches the theory in this paper. |
| 2026-10-16 | [Monitor-Validity Failures: LLM Safety Evidence Paths at Evolving Framework Boundaries](http://arxiv.org/abs/2610.23362v1) | Junchi Liao | Modern LLM applications rely on runtime monitors that inspect tool calls, traces, streams, detector outputs, and verifier inputs. This creates an interface-reliability problem: a framework path may remain operational while policy-relevant evidence never reaches its decision point. In one representative case, a regex detector returned a positive match, but the framework stranded the result before the block decision. This differs from jailbreaks and detector errors because the evidence is already recognized yet disconnected from enforcement. We study this repair-boundary signature with canary contracts: parent/fixed confirmation units that place a known payload only on the changed evidence surface and record monitor visibility together with the downstream decision for action-boundary cases. Applying the protocol to public framework repairs, we confirm 18 evidence-path failures across 11 LLM or agent safety project groups and four implementation ecosystems. Nine are established-boundary repairs, eight are new-surface policy-coverage gaps, and one is a boundary-only evidence case. Fourteen reach E1 enforcement boundaries: seven native action outcomes, six configured enforcement boundaries, and one synthetic-consumer consequence. The remaining cases affect verifier, safety-prompt, or observability channels. The cases form carrier, adapter, authority, and surface families and show how ordinary application behavior can remain intact while a safety-specific evidence path misses its decision. Exact revisions, deterministic outcomes, negative controls, and a frozen prospective audit document the confirmation rule. Monitor assumptions therefore need regression tests at the schemas, streams, tools, and adapters that frameworks evolve. |
| 2026-10-16 | [STC-MGNet: Spatial-Temporal-Channel Attention-Enhanced Multimodal Graph Network for Pedestrian Intention Prediction](http://arxiv.org/abs/2610.23356v1) | Yuanzhe Li, Meng Wang | Accurate prediction of pedestrian crossing intentions plays a critical role in enhancing the safety of pedestrian-vehicle interactions in autonomous driving. However, effectively capturing subtle spatiotemporal patterns in pedestrian pose sequences and integrating complementary visual and ego-vehicle motion information remain challenging. This paper proposes a spatial-temporal-channel attention-enhanced multimodal graph network (STC-MGNet) for pedestrian crossing intention prediction. STC-MGNet introduces a spatial-temporal-channel (STC) attention-enhanced pose graph encoder to learn discriminative representations from pedestrian pose sequences. Three parallel attention branches selectively emphasize informative skeletal joints, observation frames, and feature channels, with their outputs integrated through learnable fusion weights. To leverage complementary visual and ego-vehicle motion information, a multi-stage cross-modal channel modulation mechanism is introduced to adaptively refine pose representations through modality-specific channel gating at successive graph encoding stages. Experimental results demonstrate that STC-MGNet achieves accuracy rates of 70%, 87%, and 88% on the JAADbeh, JAADall, and PIE datasets, respectively, outperforming several representative baseline methods. Ablation experiments further highlight the complementary contributions of visual and ego-vehicle motion information to prediction performance. |
| 2026-10-16 | [Self-Reflective Chain-of-Thought (SRCoT): Evidence-Based Text Generation via a Closed-Loop Retrieve--Reflect--Revise Pipeline on Edge LLMs](http://arxiv.org/abs/2610.23311v1) | Fabian Broszio, Michael Hoffmann et al. | Large language models (LLMs) can generate fluent outputs that lack source support or conflict with local operational constraints. We introduce Self-Reflective Chain-of-Thought (SRCoT), a modular architecture that organizes generation as a closed Retrieve--Reflect--Revise loop. SRCoT separates environment state, persistent knowledge, accumulated experience, and short-term memory into provenance-labeled stores. BM25 retrieval supplies local context; subsequent passes expose intermediate drafts, missing-information markers, follow-up queries, and optional claim--evidence mappings. These artifacts inform retrieval, revision, clarification, or abstention decisions. We examine the architecture through qualitative, trace-based case studies in two fully synthetic settings: emergency medical services and Munich-Sim, an emergency-operations simulation. Local prototype runs on an Apple M4-family laptop with 128 GB memory illustrate source-labeled documentation, explicit gap marking, and structured dispatch-style outputs. The traces also reveal provenance-assignment errors, incomplete evidence coverage, and unsafe assumptions that remain possible despite prompting constraints. These cases demonstrate report-level inspectability and local execution, not comparative superiority, hallucination reduction, clinical validity, or operational safety. The architecture separates retrieval and control from the model interface, but behavior across different models remains to be evaluated. SRCoT is a research proof of concept requiring human review, with its principal contribution lying in an inspectable control procedure rather than a new model. |
| 2026-10-16 | [MobileJailBench: Red Queen Jailbreak Benchmarking for Mobile Agents](http://arxiv.org/abs/2610.23303v1) | Songping Wang, Borui Li et al. | MLLM-based mobile agents are rapidly gaining autonomy in perception, planning, and closed-loop execution in dynamic mobile environments. However, this autonomy also broadens the attack surface and amplifies the risks of malicious misuse, including privacy leakage, device compromise, and large-scale phishing. Existing jailbreak evaluations leave three critical gaps for mobile agents: they focus on conversational attacks instead of mobile workflows, use static task sets with fixed difficulty, and rely narrowly on attack success rate rather than comprehensive security assessment. To close these gaps, we present MobileJailBench, the first evolving jailbreak benchmark for mobile agents, comprising three key components: (1) a benchmark comprising 520 tasks spanning three jailbreak paradigms and 14 risk categories, built on a high-fidelity mobile sandbox that we construct, enabling realistic workflows across over 20 widely used apps; (2) Dual-Dynamic Evolution, which combines Failure-Distilled Jailbreak Evolution with Weakness-Driven Benchmark Expansion; and (3) Stratified Adjudication Framework for assessing security awareness, execution behavior, and risk consequences. Extensive experiments show that current mobile agents are particularly vulnerable to implicit jailbreaks, as their safety alignment remains largely superficial and fails to generalize from conversational refusal to dynamic mobile interactions. Moreover, our dynamic evolution mechanism sharpens practical threat boundaries by continually uncovering new vulnerabilities, reaching a 100% ASR on Qwen3.5-122B. MobileJailBench provides a reproducible, continuously evolving framework for tracking the true security boundaries of mobile agents. |
| 2026-10-16 | [FLARE: Physics-Grounded Evaluation of Language-Model Agents for Feeder Restoration with Degraded Observations](http://arxiv.org/abs/2610.23251v1) | BHVSP Subrahmanyam, Neeraj Choudhary | Restoring service after distribution-grid faults requires decisions from incomplete and conflicting information while respecting electrical constraints. Addressing this need, we introduce FLARE (Feeder-Level Agent Restoration Evaluation), a physics-grounded benchmark for post-isolation feeder restoration. FLARE con- tains 800 AC-feasible cases across five observation settings and allows agents to gather evidence, evaluate switches, and perform restoration actions under fixed operational limits. We evaluate five instruction-tuned models over 78,750 episodes, including topology, distributed-generation, and loading shifts. Mean normalized regret ranges from 0.082 to 0.148. Although every model has zero median regret, P95 regret ranges from 0.814 to 1.000, revealing an important failure tail. Shadow replay shows that 96.4% of 15,706 guard-rejected actions would have caused a physical violation. FLARE enables reproducible evaluation of language-model based restoration support while emphasizing the need for external safety checks. |
| 2026-10-16 | [AMEND: Constraint-Guided Multi-Agent Text Detoxification with Isolated Critics](http://arxiv.org/abs/2610.23183v1) | Tianyu Wang | Text detoxification requires balancing toxicity removal, semantic preservation, and fluency. Existing methods typically optimize an unconstrained scalar objective---enabling safety scores to mask semantic loss---or rely on open-ended iterative debate prone to semantic drift and high computational cost. We propose AMEND, an inference-time framework that decouples detoxification into role-conditioned proposal generation, isolated multidimensional criticism, and constraint-first selection. AMEND evaluates diverse candidates with role-blind critics and enforces strict semantic and fluency feasibility before ranking by utility. To maximize efficiency, AMEND Lite restricts subsequent search to at most one targeted repair round only when residual toxic spans remain. On ParaDetox, one-round selection over diverse roles improves the joint evaluation metric by $+0.0508$ over the best single role. On a frozen confirmation test set, AMEND Lite achieves a joint score of $0.6597 \pm 0.0062$, outperforming unconstrained multi-round loops by $+0.0489$ ($p < 0.05$) while reducing inference requests by 28.4%. |
| 2026-10-16 | [Disentangling Language Effects on Tool Admission and Post-Retrieval Safety](http://arxiv.org/abs/2610.23093v1) | Aditya Nawal, Manit Baser et al. | LLM agents increasingly use retrieval tools across languages, yet their safety evaluation remains English-centric. It remains unclear whether language affects tool admission, post-retrieval generation, or both. We independently vary request, retrieved-page, and required response languages in a controlled URL-retrieval setting across ten models and seven languages. We identify Safety Loss through Interlingual Processing (SLIP). Non-English harmful requests increase mean tool admission from 46.9% to 67.4%, with a positive mean change in every model. This pattern also persists in an autonomous retrieval setup across all four models tested in this setting. Requiring answers in the page language increases pooled harmfulness on matched tool calls from 50.3% to 57.4% across four closed-weight models, mainly in Chinese and Hindi. Translating pages alone produces no consistent harm increase in eight models with comparable inference settings. Automated audits show that a non-harmful response may be a genuine safety refusal, a deflection citing language difficulties, or an unusable answer. Harmfulness labels alone cannot distinguish these outcomes. Multilingual safety evaluations should therefore measure tool admission and response harmfulness separately and distinguish refusal from capability failure. |
| 2026-10-16 | [Rethinking GRPO for Large Reasoning Model Safety Alignment](http://arxiv.org/abs/2610.23085v1) | Jiawei Chen, Yi Zhang et al. | Large reasoning models (LRMs) exhibit strong reasoning capabilities but remain vulnerable to unsafe outputs. While Group Relative Policy Optimization (GRPO) is widely used for safety alignment, its group-relative optimization may provide insufficient safety-directed learning signals. In this paper, we identify two failure modes: advantage collapse, where homogeneous rollout groups yield vanishing advantages, and unsafe diversification, where unsafe behaviors persist and become increasingly diverse. Our theoretical analysis reveals that behavioral consistency weakens within-group safety contrasts, while entropy regularization can diversify unsafe responses when relative signals vanish. To address these challenges, we propose Beacon, comprising Safety Signal Recovery (SSR) and Safety-Utility Decoupled Alignment (SUDA). SSR combines advantage-guided trajectory filtering and safe reference supervision to recover effective safety learning signals, while SUDA employs category-aware rewards and benign-only KL anchoring to mitigate over-refusal without compromising safety. Experiments across four Qwen3 and DeepSeek-R1-Distill variants and eight benchmarks show that our framework achieves the best average safety and reasoning performance, reducing average safety error by up to 32.83 percentage points while maintaining low over-refusal. |
| 2026-10-16 | [Before Retrieval Begins: Safety and Tool-Use Decisions in LLM Agents](http://arxiv.org/abs/2610.23071v1) | Aditya Nawal, Manit Baser et al. | Large language models increasingly act as agents that call external tools, yet most tool-safety research examines only what follows a tool call. The prior decision of whether to call the tool remains largely unexamined and can influence downstream safety. Across the seven open-weight models, the unweighted model-average harm rate after admitted harmful requests is 74.5%. To identify what governs this decision, we apply a factorial intervention to the input request. Within our generated construction, neutralizing the requested-operation component produces the largest admission contrast (19 pp). A linear direction estimated from this contrast influences held-out admission in four of seven models; effects on operation-neutralized requests in some models show that it is not an operation-specific mechanism. We then test whether written refusal is behaviorally and representationally coupled to the same decision. Written refusal accompanies 33.6% of admitted harmful calls. Plain-text refusal training reduces unsafe admission in models that retain benign tool use, showing that response-level safety training can transfer to the call decision. Together, these results establish a partial, model-dependent relation between written refusal and tool admission. They motivate direct evaluation and supervision of tool admission rather than treating text refusal as its proxy. |

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



