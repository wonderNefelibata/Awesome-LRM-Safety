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
| 2026-08-10 | [Multimodal Model Diffing for Feature Discovery and Control](http://arxiv.org/abs/2608.09928v1) | Hunar Batra, Lachin Naghashyar et al. | Multimodal Large Language Models (MLLMs) exhibit strong visual understanding, yet the internal features that cause these behaviors remain difficult to identify, audit, or control. While applicable to post-hoc inspection, hidden states that are decomposed into interpretable feature directions using sparse autoencoders (SAEs) neither readily isolate which features are changed by multimodal training, nor are they directly useful for targeted control. We introduce MMDiff, a multimodal model-diffing framework that trains multimodal SAEs and turns them into feature-level interfaces for discovering and controlling multimodal behavior. MMDiff supports three uses: (i) feature isolation, by diffing a base-LM SAE against its multimodal-adapted counterpart to identify features altered by multimodal training; (ii) task-specific feature detection, via per-token contrastive firing analysis that isolates causal features; and (iii) feature-level control, by causally removing or steering the discovered feature directions. We train multimodal SAEs for three MLLM families, LLaVA-MORE, PaliGemma 2, and InternVL3.5, and evaluate on visual-spatial understanding, multimodal safety, and OCR. MMDiff discovers sparse, causally specific features whose removal selectively degrades target behaviors by an average of 12% on spatial tasks and 17% on OCR, and reduces attack success rate by 24% on multimodal safety attacks, with no impact on VQA performance. Steering these features improves spatial and OCR accuracy by +3.6% and +1.8% on average over a standard single-layer steering baseline. These results show that multimodal SAEs can serve not only as interpretability tools, but as mechanisms for auditing, steering, and controlling MLLMs behavior toward safer and more capable generations. |
| 2026-08-10 | [Beyond Hazard Resemblance: Contrastive Event Adjudication for Training-Free Video Anomaly Detection](http://arxiv.org/abs/2608.09908v1) | Wenti Yin, Xiang Wang et al. | Video anomaly detection (VAD) aims to identify and temporally localize abnormal events in videos. Supervised methods learn anomaly decision boundaries from target-domain annotations but require substantial in-domain data. Existing training-free methods leverage the rich semantic knowledge and reasoning capabilities of pretrained models to interpret visual content, yet these capabilities do not directly define an anomaly decision criterion: richer anomaly descriptions better capture hazard resemblance without resolving abnormality. To this end, we propose Contrastive Event Adjudication for training-free Video Anomaly Detection (CEAVAD), which shifts the unit of inference from isolated anomaly concepts to falsifiable event hypotheses and establishes an inference-time explanatory boundary through the interaction between competing explanations and video evidence. Specifically, CEAVAD first uses public-safety knowledge to construct hazard-benign event contrasts, pairing each hazard mechanism with a generic normal account and a mechanism-specific benign counterpart. It then determines whether the target interval better supports a hazard explanation or its benign competitor, yielding a revisable contrastive boundary proposal for the target. Finally, CEAVAD adjudicates between the competing explanations to determine whether the hazard hypothesis survives the video evidence, supporting both temporally localized anomaly detection and evidence-grounded explanations. Experiments on three widely used VAD benchmarks demonstrate that CEAVAD achieves state-of-the-art performance under the training-free paradigm. |
| 2026-08-10 | [Decoding-Level Taboo: A Diagnostic Stress Test for LLM Robustness](http://arxiv.org/abs/2608.09900v1) | Tadanobu Chuyo Kamijo, Ori Rottenstreich et al. | Large language model evaluations typically focus on performance under nominal conditions, creating an illusion of capability where models comfortably walk a narrow, highly optimized generation corridor. In real-world deployments, however, complex system prompts, safety guardrails, and structural constraints continuously force models off this nominal path, driving a divergence between benchmark scores and deployment performance. To address this issue, we introduce Decoding-Level Taboo, a zero-prompt diagnostic stress test that intervenes directly in logit space at runtime, forcing models out of their nominal paths. By dynamically masking primary candidate tokens at word boundaries, Taboo forces machine circumlocution.   Evaluating Taboo across several open-weight model families reveals that off-path robustness is heavily influenced by both parameter scale and post-training instruction alignment, with robustness generally improving with model size and alignment. Beyond the results presented in this paper, Taboo provides a novel primitive for generating diverse synthetic datasets, stress-testing runtime safety guardrails, and auditing model reliability prior to real-world deployment. |
| 2026-08-10 | [SHE: Trajectory-driven Safety Harness Evolution for LLM Agents](http://arxiv.org/abs/2608.09885v1) | Wanying Qu, Qinghua Mao et al. | The safety of large language model (LLM) agents depends not only on model weights but also on the agent harness that manages context, memory, tools, permissions, and runtime control. Existing safety mechanisms often treat the harness as a fixed deployment artifact, limiting their ability to evolve with emerging risks. Moreover, coupled functions across harness components obscure safety responsibility attribution, making localized evolution difficult. We propose Safety Harness Evolution (SHE), a framework that learns evolving safe boundaries from rollout trajectories. SHE decomposes the harness into four artifacts with explicit safety responsibilities, including the System Prompt, Rule Bank, Safety Memory, and Tool Policy, defining clear functional boundaries for localized evolution. Based on this decomposition, SHE introduces an attribution-guided evolution loop that converts trajectory failures into structured diagnoses, learns artifact-specific boundary refinements, and selects evolved harnesses through safety-utility validation. Experiments on Agent-SafetyBench demonstrate that SHE effectively enhances safety through harness evolution, achieving a 3.1x ASR reduction compared with static SafeHarness, while also improving benign utility. The evolved harness further generalizes to unseen risks on the held-out AgentHarm benchmark and transfers across agent models without additional evolution. |
| 2026-08-10 | [Safe Start: Configuring Optimization Algorithms for Decision-Making under Extreme Risks](http://arxiv.org/abs/2608.09872v1) | Henry Lam, Wasin Meesena | We consider stochastic optimization where the goal is not only to optimize an average-case objective, but also to mitigate the occurrence of rare catastrophic events. This problem is motivated by safety-aware decision-making and AI training. We first argue that, in the presence of a simulation model, natural attempts to integrate variance reduction into optimization, even executed in a reasonable adaptive fashion, encounter fundamental challenges in guaranteeing realistic runtime when using common stochastic gradient descent algorithms. This challenge arises from the extreme sensitivity of tail-based objectives with respect to the decision variables, which renders a dichotomic failure of convergence regardless of what step size we select. We offer remedies based on a new notion of safe start that allows for efficient finite-time error control, and show how the sampling complexity scales favorably under the combination of safe start and variance reduction. We illustrate our methodologies on examples in portfolio optimization and robust classification with neural networks. |
| 2026-08-10 | [Stealing Reasoning Traces from Proprietary LLM APIs](http://arxiv.org/abs/2608.09867v1) | Alexander Panfilov, David Schmotz et al. | Leading large language model providers now conceal their models' step-by-step reasoning, or chain-of-thought, to protect intellectual property and limit information leakage. Rather than storing these traces server-side, providers return them to the client as blocks of encrypted text, which the client passes back with each subsequent request. Building on prior research, we identify an architectural vulnerability: these encrypted blocks are fully compatible and interchangeable across different sessions, users, and models within a provider's ecosystem. We exploit this compatibility to develop a scalable decryption jailbreak. By injecting an encrypted reasoning trace from a given model into a weaker, and less safeguarded model from the same provider, we force it to decode and output the trace verbatim in plaintext, without ever jailbreaking the more capable model directly. This vulnerability enables four distinct attack vectors. First, it circumvents anti-distillation mechanisms, allowing adversaries to extract a proprietary model's reasoning, as we demonstrate across Anthropic, OpenAI, and Google. Second, it allows for large-scale private data extraction. Developers frequently share session logs publicly, unaware of contents of the encrypted blocks. By decoding 315,320 reasoning blocks scraped from public repositories, we recovered 367 Personally Identifiable Information (PII) artifacts and 182 credentials. Third, it inadvertently reveals hazardous information hidden within the reasoning process, even in cases where the model's final, visible output safely rejects a malicious request. Fourth, attackers can leverage this flaw to execute invisible prompt injections, embedding malicious payloads entirely within encrypted blocks to poison public agentic rollouts. Following responsible disclosure, we propose concrete cryptographic and system-level mitigations to secure client-side reasoning. |
| 2026-08-10 | [Vehicle Platooning](http://arxiv.org/abs/2608.09864v1) | Zhi-Long Chen, Nicholas G. Hall | Vehicle platooning offers significant benefits, including reduced energy consumption, lower emissions, improved road utilization, enhanced safety, and reduced driver fatigue. As intelligent driving technologies continue to advance, platoon sizes are expected to increase substantially, making the efficient sequencing and resequencing of vehicles increasingly important. We study the vehicle platoon sequencing and resequencing problem on road networks with varying segment lengths under two fundamental objectives: minimizing total energy consumption and minimizing the maximum energy consumption of any vehicle. For the typically encountered combinations of vehicle and road characteristics, we provide a complete computational complexity classification, either developing polynomial-time algorithms or proving computational intractability. For several intractable cases, we design fully polynomial-time approximation schemes and polynomial-time heuristics with provable performance guarantees. A computational study demonstrates that the proposed heuristics achieve average solutions within 1\% of optimal. We also consider settings in which only limited information about position-dependent energy savings is available and develop a heuristic with bounded worst-case performance. In addition, we present an efficient algorithm for on-road vehicle resequencing when only limited position changes are permitted. Together, these results provide a comprehensive algorithmic framework for energy-efficient vehicle platoon sequencing and resequencing. |
| 2026-08-10 | [Agentic Harnesses: LLM-Driven Verification Layers for Robot Autonomy](http://arxiv.org/abs/2608.09857v1) | Rohan Bhagra, Mahantesh Halapannavar et al. | Advances in advanced artificial intelligence tools have sparked research in robot autonomy, but the development of such systems has largely focused on execution rather than verifying the feasibility actions planning models propose. Like general-purpose LLMs, robotics planning models carry risks: biased toward user-specified goals, they may suggest actions misaligned with scientific ethics, they may be unsafe due to an inability to "remember" prior safety risks, or they may be vulnerable to adversarial attacks on the autonomy ecosystem. We propose a LLM-driven verification layer between planning and execution to evaluate action permissibility. Our LLM-as-a-Judge ensemble combines chain-of-thought reasoning across models and synthesizes those expert judge outputs, mirroring a combination of a mixture of experts and self-consistency approach. This layer serves as middleware, gating plans from the server's planning module before they reach the MCP server and therefore the robot's low-level controls: plans are approved, rejected for reformulation, or escalated for human review. With this system, we achieve near 85% precision across accept/escalate/reject categories 97% containment of adversarial attacks, with negligible errors between accepting and rejecting tasks, and errors mostly manifesting at the escalate boundary. |
| 2026-08-10 | [Multi-Agent AI Safety as an Institutional Design Problem](http://arxiv.org/abs/2608.09828v1) | Abdullah X | AI agents increasingly work inside systems that govern how they delegate tasks, move information, execute actions, and use shared resources. Recent work already shows that deployment rules can change collective behavior. Here we ask which parts of an AI institution produce safety and how they do it. This is the first paper from POLIS, an ongoing research programme studying algorithmic institutions for multi-agent systems. We report a frozen 5,280-episode study suite. The main pre-specified delegation experiment spans four model families; a targeted high-conflict diagnostic adds three additional model endpoints. In matched structured workflows, the model sees different rule formulations and guards consult different authority states. We also vary the attractiveness of the immediate compliant internal/self fallback and allow blocked workflows to continue. A detailed constitutional prompt produces 0/384 realized violations. A provenance-aware executable guard also produces 0/384, although it blocks prohibited attempts in 51/384 episodes; 44/51 of those episodes later complete safely. The local-state guard's failures concentrate in scenarios where an ordinary transformation changes visible policy while originating authority stays fixed. In matched laundering scenarios, that guard admits violations in 22/96 episodes and provenance enforcement in 0/96 (p = 4.77 x 10^-7). A separate resource-allocation experiment shows that revealing the numerical value of an otherwise identical cap changes agent requests. In these structured workflows, the same final violation rate can hide very different mechanisms. The rule itself is only part of the institution. The authority state the system trusts matters, and so does the path available after a block. |
| 2026-08-10 | [Removing Infrastructure Barriers in Human-Robot Collaboration Through Wireless Reconfigurable Cells](http://arxiv.org/abs/2608.09658v1) | Emma Takács, Mátyás Hajós et al. | Human-Robot Collaboration (HRC) plays a vital role in dynamic, high mix, low volume industrial scenarios such as remanufacturing, which frequently face workcell rearrangements. Traditional setups are constrained by power and data cabling, restricting modularity and reconfigurations, while the selection of commercial wireless devices suitable for real-time perception and safe collaboration are limited in availability. This paper presents a highly flexible, wireless, 5G-based system that serves as a versatile experimental testbed for applications including remanufacturing, operator training, and user studies. To eliminate infrastructure barriers, the workcell integrates a novel battery-powered, multi-sensor platform prototype. Additionally, to support operator safety and system adaptability across environmental shifts, the system integrates a computer vision module for object detection and pose estimation, further augmented for robust hand recognition. Trained on synthetic and real data, the model reliably detects oriented grasping poses and human hands across varying lighting and background conditions (with an mAP@50-95 of 97.74 +- 0.10% and a mean inference time of 12.5 ms). Offloading these computationally intensive tasks to the edge via 5G, the proposed architecture contributes to resolving the bandwidth-latency trade-off. To demonstrate portability, the system was implemented in both Hungary and Norway, and was evaluated across a combination of public and private, Standalone and Non-Standalone 5G infrastructures. The performed network experiments produced results in round-trip response times down to 12 ms in case of compatible network-device pairings, suitable for safe, adaptive HRC. However, these measurements also revealed practical limitations related to interoperability in current 5G deployments that should be addressed in future works. |
| 2026-08-10 | [Predictive safety filter enhanced curriculum learning control for efficient vehicle dynamics controller](http://arxiv.org/abs/2608.09653v1) | Baocong Zhang, Siliang Lu et al. | Recent advances in learning-based control have enabled impressive achievements in solving complex control problems in various domains. However, since learning-based control may not be able to realize safety-guaranties, it is of great importance to enhance safety and robustness while maintaining good performances. Take vehicle motion \& dynamics control as an example, in order to overcome the pain points of traditional methods such as heavy parameter calibration effort and learning-based control to bring better performance and efficiency in stability \& agility over prior work for state-based vehicle control tasks, in this work, our method aims to develop a curriculum learning controller enhanced with physics-based predictive safety filter. The validation is conducted with the Python-CarSim platform, demonstrating better improvements and scalability under various maneuvers. |
| 2026-08-10 | [Measuring the Wrong Thing: Internal Harmfulness Scores Anti-Rank Successful Jailbreaks](http://arxiv.org/abs/2608.09624v1) | Mingyu Luo, Ming Deng et al. | Internal safety scores judge a prompt before any text is generated, and they are validated by how well they separate harmful prompts from benign ones. That separation is then read as evidence that the score will also catch the attacks that succeed. Harmful intent is a property of the prompt. Jailbreak success is an outcome produced later by a particular target model, decoding policy, and judge. A filter tuned on a score that measures the wrong quantity spends its false positive budget on attacks that would have failed anyway. In this paper we audit that inference. Attention based measurements are usually read from prompt dependent locations, so a wrapper changes both the content being judged and the place the signal is taken from. We therefore introduce Active Attention Probing, which supplies a fixed content independent measurement coordinate. We pair every base goal with a plain and a wrapped version and generate real completions from the target models. On Llama, wrapping raises harmful generation from 0.05 to 0.27 while harmful intent AUROC falls from 0.936 to 0.803, so the attacks grow more dangerous while the prompts look safer to the score. Among wrapped harmful prompts the outcome AUROC is 0.220, which places the attacks that succeeded below the attacks that failed. Rare token, passive, and detector derived channels reproduce the reversal on the same matched design, and the reversal itself persists across three target models, seven attack families, and two independent judges. Distribution shift then degrades calibration and threshold transfer before it degrades ranking. |
| 2026-08-10 | [Adaptive Sequential Test Planning for Multi-Mechanism Reliability Qualification via Bayesian Monte Carlo Tree Search](http://arxiv.org/abs/2608.09622v1) | Youssef A. Elhagrasy, Ian Hill et al. | Reliability qualification of advanced semiconductor devices requires sequential stress decisions that balance characterization objectives against multiple competing failure mechanisms. Current practice relies on static test plans derived from population-level acceleration models, which cannot adapt to per-unit variability or real-time degradation observations. This paper presents a closed-loop adaptive test planning framework that formulates reliability qualification as a partially observable sequential decision problem and solves it using Monte Carlo tree search for seed-action simulators (MCTS-SA) coupled with extended Kalman filter (EKF) belief-state estimation. The framework models stochastic, per-device variability in bias temperature instability (BTI), electromigration (EM), and time-dependent dielectric breakdown (TDDB), and treats stress selection as a constrained sequential optimization, i.e., to maximize the probability of successful degradation characterization while respecting catastrophic failure constraints. Under the experimental assumptions used here (discrete stress actions, proxy damage observability, and cumulative degradation without recovery), we believe this to be a novel application of tree-search-based adaptive test planning to multi-mechanism reliability qualification. Across 5,000 planning iterations, the characterization yield (CY) improves from 20% in the first 500 iterations to over 54% in the final 500, with 39% cumulative success, while the best successful test sequence terminates with EM and TDDB damage fractions DEM=0.564 and DTDDB=0.537, well within safety margins. These results demonstrate that sequential Bayesian planning can synthesize damage-aware test policies that significantly outperform non-adaptive strategies for reliability qualification under competing failure modes. |
| 2026-08-10 | [Lost in k-Space: An Open-Source MR-Physics Escape Room](http://arxiv.org/abs/2608.09616v1) | Sabine Melanie Räuber, Marta Brigid Maggioni et al. | Introduction. Operating an MR scanner for technical and clinical research requires multidisciplinary competencies beyond MR physics, including safety management and teamwork. Gamification, particularly educational escape rooms, have been associated with improved motivation, engagement and knowledge retention in health professional education. Materials and Methods. We developed an MR-physics-themed educational escape room for the 41st Annual Meeting of the ESMRMB. Designed for teams of four with a time limit of 25 minutes, the room reproduced the atmosphere of an MR control room. The five puzzles covered the Larmor equation, sequence composition, MR safety and acoustical identification of MR sequences. Custom ESP32 electronics allowed the puzzles to communicate in real time with each other and with the game master Results. The room ran without technical interruptions and was played by 39 teams (approximately 160 participants); 12 solved it (escape rate 31%), with a median solution time of 21 minutes and 53 seconds. Early-career researchers acted as game masters. Discussion. The escape rate aligns with comparable activities targeting scientific audiences. Difficulty can be tuned by adjusting puzzle obscurity, component availability, required prior knowledge and mental leaps. A modified version could usefully supplement mandatory MR safety training. Code, schematics, machining files and documentation are released as open source. |
| 2026-08-10 | [Pragmatic Attack Surface: Vulnerabilities of Implicit Context in Large Language Models](http://arxiv.org/abs/2608.09551v1) | Bocheng Chen, Han Zi et al. | In the era of large language models (LLMs), attackers often manipulate natural language to elicit unsafe or harmful outputs, creating a new natural language attack surface unique to LLM-based systems, where attacks directly exploit explicit linguistic cues in user prompts to bypass the safety mechanism of LLMs. However, such attacks can often be mitigated by existing safety alignment algorithms. On the other hand, human language is inherently grounded in pragmatics, necessitating typical context to interpret language, e.g., world knowledge, social norms. However, such contexts are often implicit because they are not directly expressed in human language and are not sufficiently leveraged in safety alignment, creating a fundamental mismatch between human language interpretation and safety alignment approaches. In this paper, we demonstrate that this mismatch exposes vulnerabilities in LLMs. We refer to this vulnerability as the pragmatic attack surface, which can be exploited to achieve high attack success rates. The experimental results demonstrate that our proposed approach outperforms baseline attack methods across various open-source and closed-source models by a substantial margin. |
| 2026-08-10 | [ELBench: A Multi-Dimensional Benchmark for Education-Facing Large Language Models](http://arxiv.org/abs/2608.09548v1) | Yilin Jiang, Xiaorong Zhu et al. | Large language models are increasingly deployed in education as tutors, teaching assistants, and content generators. These roles place demands that ordinary question answering does not: a usable education-facing model is supposed to be accurate, safe under sensitive prompts, instructionally useful, and aligned with pedagogical goals at the same time. Existing benchmarks evaluate these requirements largely in isolation, so none assesses education-facing suitability as an integrated profile. We introduce ELBench, the first benchmark to evaluate all four requirements (General Capability, Safety and Trustworthiness, Basic Education, and High-Level Cultivation) on the same models under a common protocol, combining curated public sources with newly synthesized safety and cultivation data. We evaluate nine models, seven frontier general-purpose systems and two education-specialized variants, and report three findings. First, module-level profiles are more informative than a single aggregate: the top six models are statistically indistinguishable on overall score, yet their module leaders differ substantially, and safety is anti-correlated with practical teaching (r = -0.83). Second, the Chinese-developed models lead the safety module, the most discriminative in the suite; this advantage is largest on region-specific normative content and narrows, but does not vanish, on universal-harm content. Third, the two education-specialized models lead neither education module, and on High-Level Cultivation all models share a systematic blind spot: on the structured judgment task they converge on the same non-reference option, favoring pedagogical style over fit to the stated goal, so the module scores uniformly low and does not separate models. This raises, but does not resolve, whether domain post-training keeps pace with frontier systems on education tasks. |
| 2026-08-10 | [Model-Based Systems Engineering Framework for SysML-Driven Design of Autonomous UAVs](http://arxiv.org/abs/2608.09547v1) | Deekshitha Angadi, Naveena Budda et al. | Autonomous Unmanned Aerial Vehicles (UAVs) are complex cyber-physical systems that require the coordinated integration of flight control, navigation, perception, communication, power management, and mission-level decision-making under safety, timing, and reliability constraints. However, many autonomous UAV development workflows still rely on document-centric requirements, separated architectural descriptions, and software implementation artifacts, which can lead to ambiguity, interface inconsistencies, and weak traceability during early design. This paper presents a Model-Based Systems Engineering (MBSE) design framework for the SysML-driven development of autonomous UAVs. The proposed framework uses the Systems Modeling Language (SysML) as a formal design backbone to structure UAV development across four connected layers: stakeholder requirements, functional decomposition, logical architecture, and physical/software allocation. SysML requirement diagrams, activity diagrams, block definition diagrams, internal block diagrams, state machine diagrams, and parametric diagrams are used to capture the functional, structural, behavioral, interface, and performance aspects of the UAV system. The logical architecture is then systematically mapped to a Robot Operating System 2 (ROS 2) software architecture by relating SysML blocks to ROS 2 nodes, flow ports and connectors to topics, request-response interactions to services, and goal-oriented behaviors to actions. The framework is illustrated at the design level using representative autonomous UAV mission scenarios, including autonomous take-off, waypoint navigation, hover stabilization, obstacle avoidance, return-to-home, and emergency handling. The resulting model supports requirement allocation, interface definition, subsystem responsibility assignment, and verification planning before simulation or physical deployment. |
| 2026-08-10 | [Dual-Adversarial Safety Alignment: Cultivating Intrinsic Threat Comprehension in LRMs](http://arxiv.org/abs/2608.09542v1) | Hongli Shen, Shaopeng Fu et al. | Large reasoning models (LRMs) achieve remarkable success on complex tasks but remain vulnerable to harmful prompts that induce unsafe outputs. Recent methods align LRMs using direct refusals or safety rationales, yet often focus on prompt patterns rather than intrinsic attack mechanisms. As a result, these pattern-centric alignments struggle to generalize across diverse jailbreaks, compromising adversarial robustness and reasoning utility. We propose AdvSafe, a dual-adversarial framework that enables LRMs to internalize unsafety knowledge by explicitly deconstructing adversarial mechanisms. This moves beyond pattern-dependent traces, fostering robust cognitive defense without compromising reasoning utility. Our pipeline operates via a two-phase adversarial game. First, in adversarial synthesis, an autonomous agent dynamically crafts deceptive jailbreak prompts, adapting its strategies to breach a strong teacher model. Second, in adversarial extraction, the breached teacher executes a cognitive counter-attack. For every successful jailbreak, the teacher unmasks the camouflage, explaining why the attack succeeds and how such prompts can be identified and mitigated. This dual-adversarial process yields a compact reasoning dataset capturing rich, generalizable unsafety knowledge. Student models trained on this dataset implicitly acquire safety alignment through intrinsic threat comprehension. Experiments show that with only 1K synthesized samples, AdvSafe-aligned LRMs achieve significantly stronger jailbreak robustness than existing baselines, with almost no utility degradation. Furthermore, AdvSafe improves robustness against out-of-distribution prompts, demonstrating that learning unsafety knowledge enables a superior robustness-utility trade-off and generalizes beyond seen attack patterns. |
| 2026-08-10 | [When Do Task Vectors Interfere? Mapping the Validity Boundaries of Weight-Space Composition](http://arxiv.org/abs/2608.09490v1) | Chencheng Zhu, Xiaoyang Li et al. | Task arithmetic treats fine-tuning displacements as composable directions in weight space, yet it remains unclear when parameter addition reflects predictable changes in model function. We separate parameter geometry from functional geometry and measure pairwise functional non-additivity over a two-dimensional task-vector surface, using a first-token predictive-distribution interaction ratio conditioned on an input distribution and evaluated with norm-matched controls, three training seeds, and response-only fine-tuning. On Qwen2.5-1.5B, code+safety is more non-additive than the matched code+math control on code and instruction prompts, but not on math prompts. In a prospectively specified six-task expansion, all eight high-versus-low comparisons of unseen task pairs have the predicted sign. The primary ordering further persists under full-parameter fine-tuning at 0.5B, Qwen2.5 LoRA scale tests up to 7B, and a Llama-3.1-8B cross-architecture audit. External validation exposes a sharper boundary: raw public code, instruction, and safety prompts preserve the continuous contrast, whereas an instruction-style wrapper collapses it on the identical public-code prompts, and EvalPlus pass@1 interactions do not robustly reproduce it. Weight-space composition therefore supports coarse, input- and format-conditioned functional statements across adaptation methods, scales, and one additional model family, not a universal merging-performance predictor. |
| 2026-08-10 | [Graph-Guided Safe Diffuser: Topological Graph Guidance for Safe Diffusion Planning](http://arxiv.org/abs/2608.09484v1) | Nakgyu Yang, KwangBin Lee et al. | Many diffusion-based planners enforce safety through inference-time guidance, but such interleaved trajectory deformations often degrade kinematic feasibility due to manifold rupture. We propose Graph-Guided Safe Diffuser (G2SD), a hierarchical framework that leverages a high-level topological graph planner to guide a low-level diffusion model. G2SD enforces safety at a structural level by abstracting the data manifold into a learned latent graph, on which high-level planning is performed. Continuous trajectories are generated by diffusion planners, which are conditioned on the graph node representations selected by the high-level planner. Theoretical analyses demonstrate conditions under which manifold rupture occurs in diffusion planners, and show that G2SD improves safety by reducing the constraint violation probability as the number of segments increases. Experiments demonstrate that G2SD substantially outperforms baselines, increasing goal-reaching rate without any collision from 40-50% to 98% in Maze2D navigation and also achieving superior task scores in locomotion. |

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



