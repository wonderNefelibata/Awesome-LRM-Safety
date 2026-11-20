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
| 2026-11-19 | [Text-Aware, Vision-Blind: How Multimodal Image Editing Systems Handle Unsafe Inputs](http://arxiv.org/abs/2611.26569v1) | Ziqing Yang, Yixin Wu et al. | Multimodal image generation systems have made rapid progress and increasingly emphasize content safety. However, their safety mechanisms tend to rely on a single modality and lack joint consideration of both visual content and text prompts, particularly in instruction-guided image editing, where the uploaded image may itself carry risk. We thus introduce EditSafetyBench, a benchmark that measures system-level safety awareness of advanced image editing systems, containing 5,940 image-instruction pairs, covering safe and unsafe images across 11 risk categories and three edit intents. We then probe the benchmark dataset on four systems, including Gemini, GPT, Qwen API, and Qwen Local. Based on our evaluation, we observe a systematic text-vision safety mismatch: Gemini and GPT refuse 87.1\% and 81.6\% cases containing unsafe-intent editing prompts, while only refusing 24.2\% and 22.4\% cases with both unsafe images and neutral instructions. At the category level, most risks are considered inconsistently by different systems, e.g., GPT allows 70.4\% editing attempts of unsafe political content while Gemini only allows 34.8\%. Notably, sexual content shows the strongest visual sensitivity, likely benefiting from mature NSFW identification works. Our findings highlight the need for safety guardrails that jointly reason over both visual and textual modalities in image editing systems. |
| 2026-11-19 | [Profile-Anchored Control: Velocity Shaping with Exact Disturbance Rejection and Transient-Free Replanning](http://arxiv.org/abs/2611.26454v1) | Christian Fredrik Sætre, Morten Dinhoff Pedersen | In many motion-control problems, how a target is approached matters as much as whether it is reached: speed and deceleration must stay within actuator, safety or comfort limits despite unknown loads and disturbances. Position-dependent velocity profiles provide a natural way to describe such requirements. We show that the square of such a profile can be used both to plan the motion and to construct the controller that executes it. Our main result, profile-anchored control with exact rejection (PACER), recovers the nominal motion after a finite time under an unknown gain and load, and then allows the profile to be replaced mid-motion without a transient. It combines a static law for scalar second-order systems with a dynamic law for first-order ones, used for disturbance rejection. The static law guarantees uniform global asymptotic stability under an unknown input gain with a known lower bound, and finite-time stability under local homogeneity, while the first-order law gives analogous guarantees under disturbances of bounded rate and provides the exact rejection in PACER. Replanning of a profile under speed, acceleration and jerk limits can be formulated as a convex program. The approach is illustrated on a simulated elevator. |
| 2026-11-19 | [Learning to Defer under Expert Drift](http://arxiv.org/abs/2611.26440v1) | Ziyao Mou, Andrea Wynn et al. | The learning to defer (L2D) framework enables safety in machine learning and AI systems by allocating difficult or critical decisions to a human expert. However, most prior work on L2D assumes fixed expert reliability, overlooking temporal fluctuations in human performance due to factors such as fatigue and cognitive biases. Humans commonly exhibit such substantial, systematic performance shifts across a wide range of domains, from high-stakes settings such as radiology, aviation, and judicial decision-making, to everyday tasks such as crowdsourced image annotation. We propose Expert-Drift-Adapted Learning-to-Defer (EDA-L2D), a novel expert-aware L2D framework that explicitly models these temporal shifts by conditioning the deferral head on recent histories of expert outcomes via a sequence model. This history-aware approach enables adaptive deferral policies that anticipate reliability fluctuations and better allocate decisions between human and machine over time. We provide comprehensive experiments across three diverse tasks -- image classification, medical diagnosis, and hate speech detection -- including generalization analyses under realistic deferral conditions and irregular shifts in expert reliability. Our results show that EDA-L2D adapts more effectively than prior L2D methods to diverse temporal patterns of expert reliability, resulting in improved system performance and more effective deferral decisions. Code is available at https://github.com/ZiyaoMou/L2D-Expert-Drift. |
| 2026-11-19 | [Actor-Predictor Architecture for Reinforcement Learning with Formal Specification Objectives](http://arxiv.org/abs/2611.26313v1) | Josef Brozovich, Qining Zhang et al. | One fundamental challenge in reinforcement learning (RL) is enforcing temporal safety constraints, such as satisfying formal specifications. Existing methods often transform logic formulas into automata for state augmentation and reward shaping. However, this approach assumes that automaton construction is both feasible and tractable, which limits its applicability to small specifications and linear temporal logic (LTL). This paper proposes Actor-Predictor, a framework that learns a surrogate representation of logic progress without explicitly constructing an automaton. The predictor continuously estimates whether a partial trajectory will satisfy the formal specification in a later step, and its learned features are provided as auxiliary inputs to the actor, which is then optimized using GRPO and RWR. We show theoretically that, under sufficient predictive accuracy, the learned representation is equivalent to the automaton state. Experiments on grid world and robotic tasks in Isaac Lab demonstrate that Actor-Predictor outperforms automaton-based baselines on LTL-constrained tasks. It also performs well on non-regular formal specifications, such as $(a^n b^n)$, for which finite-automaton-based approaches are not applicable. |
| 2026-11-19 | [Scalable fault vulnerability analysis for ReRAM-based process-in-memory CNN accelerators](http://arxiv.org/abs/2611.26307v1) | Aniseh Dorostkar, Hamed Farbeh et al. | Resistive random-access memory (ReRAM)-based processing-in-memory (PIM) accelerators offer a highly efficient platform for parallel matrix-vector multiplications (VMM) in neural networks, leveraging analog computation, high density, low leakage, and non-volatility. However, their susceptibility to permanent faults, such as stuck-at-high (\textit{SaH}) and stuck-at-low (\textit{SaL}) resistance states, permanently corrupts the convolutional neural networks (CNNs) weights mapped onto ReRAM cell conductance values, thereby degrading inference accuracy and leading to system unreliability in safety-critical applications. Fault injection is a commonly used technique to assess the vulnerability of neural networks by systematically injecting faults into individual parameters and measuring the resulting accuracy degradation. Although this method is accurate, it is prohibitively time-consuming and computationally expensive, especially for large-scale CNNs. In this paper, we propose a fast, scalable, and analytical framework to quantify the fault vulnerability of convolutional neural networks deployed on ReRAM-based PIM accelerators. This method quantifies vulnerability at different granularities, including weights, kernels, and layers, without the need for exhaustive simulation, enabling the design of fault-tolerant accelerators for edge and safety-critical applications. The comparison between the results of the proposed vulnerability score (VS) metrics with performance metric of CNNs captured with fault injection simulations confirms the effectiveness of our proposed method. Unlike exhaustive fault injection, which requires days or weeks, whereas the proposed approach achieves a speedup of about $10^4\times-10^5\times$, enabling fast and accurate vulnerability analysis for large-scale CNNs within only a few minutes or hours. |
| 2026-11-19 | [DailySafe: Benchmarking Dynamic Safety Awareness in Everyday Multi-Turn Multimodal Interactions](http://arxiv.org/abs/2611.26298v1) | Xuankun Rong, Xinhao Deng et al. | As multimodal large language models (MLLMs) become increasingly integrated into everyday life, their ability to provide safe and reliable assistance in realistic daily activities remains insufficiently explored. Existing safety benchmarks primarily focus on isolated harmful requests or adversarial scenarios, overlooking the safety risks that may arise as ordinary tasks unfold. In practice, users pursue legitimate goals through multi-turn, multimodal interactions, where seemingly benign requests may become safety-critical when considered alongside earlier actions and visual observations. This calls for Dynamic Safety Awareness, the ability to track evolving task conditions, recognize implicit contextual risks, and provide safe and helpful assistance. To systematically evaluate this capability, we introduce DailySafe, a benchmark comprising 195 naturalistic multi-turn, multi-image interactions across 10 everyday domains and 42 task categories. Grounded in authentic user dialogue topics and documented safety knowledge, DailySafe captures realistic scenarios where safety-critical evidence is progressively revealed across conversational turns and visual observations. Using case-specific safety annotations, DailySafe assesses whether MLLMs can recognize and appropriately address evolving risks while supporting users' legitimate goals, bringing multimodal safety evaluation closer to real-world everyday AI assistance. DailySafe is publicly available at: https://github.com/XuankunRong/DailySafe. |
| 2026-11-19 | [BADAS-3.0: Real-Time Driving Perception and Planning from One Camera](http://arxiv.org/abs/2611.26294v1) | Roni Goldshmidt, Hamish Scott et al. | We ask how far a driving model can get with one camera and real-time latency. BADAS-3.0 is a 1B-parameter video encoder with four small heads. From two seconds of one forward camera, it predicts the current speed, the metric path just driven, a collision-risk score and six candidate futures over 6.4 seconds, in under 50 ms. Given the car's current speed, BADAS-3.0 has 44% lower minADE$_6$ over five trajectory benchmarks than the 34B Alpamayo 2 Super, which reads six cameras and runs over 100x slower. The gain comes from camera rigs that neither model was trained on. On the rig they share, the two are on par. Existing benchmarks are mostly nominal driving, so we add two open-loop safety benchmarks, one of real dashcam collisions and near-misses and one of Euro NCAP scenarios. On both, BADAS-3.0 has a higher pass rate than every published planner we tested. |
| 2026-11-19 | [I See You: Initial Multi-Robot LiDAR Registration with Mutual-Observation Checks](http://arxiv.org/abs/2611.26276v1) | Jonathan Lichtenfeld, Oskar von Stryk | Compared to single-robot deployments, multi-robot teams can significantly improve search-and-rescue (SAR) mission success by covering large areas in parallel. To collaborate effectively, these robots must be registered in a common reference frame. Standard registration methods typically rely on external infrastructure or significant map overlap, but real deployments require infrastructure-free, stationary initial alignment for immediate operation. While global LiDAR point cloud registration addresses this, existing single-scan methods often fail silently in aliased or degenerate environments. Approaches scaling beyond two agents also remain rarely evaluated in real-world scenarios. We propose a centralized, deployment-ready framework for pre-mission multi-robot registration. Our system performs pairwise coarse-to-fine alignment and actively rejects impossible pose hypotheses via deterministic plausibility checks. We validate the framework on a real-world custom dataset of four synchronized robots in various environments with ground truth. The checks reject more than a third of the incorrect pairwise registrations while falsely rejecting less than 1% of the correct ones, and reduce the number of team registrations with a wrong robot pose by 38%. Dataset and code are publicly available. |
| 2026-11-19 | [Teaching Vision-Language-Action Models to Stop Safely](http://arxiv.org/abs/2611.26239v1) | Zheyuan Liu, Chuhan Wang et al. | Vision-language-action (VLA) models enable robots to perform diverse tasks from visual observations and language instructions. While prior work has explored task suppression and refusal, stopping prohibited behavior once execution is underway and controlling the resulting physical response remain challenging. Our evaluation reveals that policies which suppress a task from its initial state can still complete it when taking over partway through execution. Even when the task remains unfinished, the robot may drop a held object, making non-completion an incomplete measure of safety. To address these challenges, we introduce State-Aware Fallback Imitation (SAFI) for tasks where holding provides an appropriate alternative response. SAFI comprises two stages, (1) fallback target construction, which assigns stationary arm commands and consistent gripper targets throughout prohibited-task demonstrations, and (2) joint policy adaptation, which learns these targets alongside the original actions of retained tasks. This supervision teaches an explicit alternative response at intermediate execution states while maintaining the robot's other skills. Extensive experiments with OpenVLA-OFT across LIBERO and BadLIBERO tasks evaluate prohibited completion, physical incidents, and retained-task performance. On the evaluated LIBERO tasks, SAFI reduces aggregate target success after takeover to 1.3\%, with no recorded physical incidents or action saturation, while preserving high retained-task success. |
| 2026-11-19 | [Phantom of the Masquerade: Graph Neural Network-Based Detection of Masquerade Attacks in Controller Area Networks](http://arxiv.org/abs/2611.26094v1) | Nehal Ameen, Pablo Moriano et al. | The lack of built-in security in controller area networks (CANs) exposes modern vehicles and industrial systems to masquerade attacks that can compromise safety and reliability. These attacks inject malicious messages that mimic legitimate communication patterns, making them difficult to detect using existing intrusion detection systems (IDS). Prior graph-based IDS approaches often rely on shallow embeddings that decouple representation learning from classification, which may limit their ability to fully exploit structural and signal-level information. In this paper, we propose PhantoMasq, a graph neural network (GNN)-based framework for detecting masquerade attacks in CAN systems. PhantoMasq models CAN traffic as message sequence graphs enriched with lightweight statistical features derived from signal behavior, enabling joint learning of structural dependencies and temporal characteristics. The detection task is formulated as a graph-level classification problem and solved using graph convolutional networks (GCNs). We evaluate PhantoMasq on the ROAD dataset under a protocol that prevents overlapping windows from spanning training and test data. Across five masquerade attack types, PhantoMasq achieves mean AUC-ROC between 0.831 and 0.999, with performance depending on the transmission rate of the targeted identifier and the magnitude of the induced signal deviation. Against a state-of-the-art graph-embedding baseline evaluated on the same sliding-window configurations, PhantoMasq achieves higher AUC-ROC on all five attack types, by 0.13 to 0.36, and is less sensitive to window configuration. Window-processing latency ranges from 2.33 to 5.94 ms per window on CPU, corresponding to a throughput of 168 to 429 windows per second. We make our implementation and experimental artifacts available to support reproducibility. |
| 2026-11-19 | [ScaleWire: An End-to-End Scale-Up Ethernet Simulation Framework for LLM Systems](http://arxiv.org/abs/2611.26086v1) | Kai Ma, Quanfeng Lv et al. | LLM training and inference rely on scale-up networks for high-bandwidth, low-latency memory-semantic communication among accelerators (XPUs). Ethernet is emerging as an open interconnect for scale-up systems. In Ethernet systems, network-interface (NI) mechanisms across transport, network, and link layers interact, jointly shaping communication performance and on-chip resource requirements. We present ScaleWire, an end-to-end scale-up Ethernet simulation framework for LLM systems. It couples packet-level simulation of cross-layer NI mechanisms with a resource exploration engine that combines analytical estimates and simulation to size on-chip resources. We validate ScaleWire against hardware measurements and nine analytical checks, and evaluate six NI designs across four LLM communication workloads. For DeepSeek-R1 dispatch, NI design choices lead to a 4.5$\times$ difference in per-XPU on-chip storage budget. In grid-comparison experiments, the exploration engine uses 6-19% of the simulations required by exhaustive search. Using vLLM prefill traces with kernel times measured on GH200 GPUs, ScaleWire further guides resource provisioning under a completion-time constraint for inference execution. ScaleWire thus helps designers select NI mechanisms and provision on-chip resources to balance communication performance and resource cost. |
| 2026-11-19 | [SEOYUN: Refusal-Inducing Exam Documents for AI-Glasses Pipelines](http://arxiv.org/abs/2611.26033v1) | Dohyeon Kwon | Camera-equipped glasses can convert an examination page into a query to a vision-language model (VLM). Device detection addresses possession; a complementary question is whether an examination document can reduce the usefulness of an unauthorized capture. We present SEOYUN, an application-specific framework for refusal-inducing exam documents, with optical and encoder-level interventions as extensions. We use reverse jailbreaking to describe the direction of the objective: inducing abstention rather than bypassing refusal. Refusal induction and defensive visual prompt injection already have precedents; our contribution is a threat model, analytical limits, and an evaluation protocol for examination use. We distinguish refusal from correct-answer leakage, show why removing an added dummy option is not itself a reduction in the original answer space, and derive a sampling comparison under explicitly stated camera assumptions. The protocol separates digital, simulated, and physical inputs and requires held-out evaluation after subject-specific selection. A local CSAT candidate corpus and rendering harness have been prepared, but this version reports no model-performance measurements, physical validation, or human-subject results. Accordingly, SEOYUN is a research framework, not a validated defense or a basis for allegations against individuals. |
| 2026-11-19 | [Evaluating safeguarding in multi-turn conversations with AI assistants](http://arxiv.org/abs/2611.26009v1) | Charlotte Peart, Franziska Sofia Hafner et al. | Conversational AI assistants are now used for personal and emotional support. Developers report improvements in how new models respond to sensitive conversations, yet users and civil society continue to raise concerns about their behaviour in real-world, repeated interactions that safety evaluations rarely capture. Often, evaluations present signs of distress clearly and in a single message, even though people may reveal their struggles gradually in otherwise ordinary conversations. We worked with three community mental health support workers to test if models provide the same level of safeguarding when cues are shared all at once or gradually over time. We designed nine scenarios in which a user shares cues related to depression, psychosis, or domestic abuse. For each scenario, the support workers established a gold standard response, ranging from signposting support to pausing the everyday task to address safeguarding. We presented five cues either together in one message or dispersed across a conversation. Across more than 7,400 conversations, we evaluated six widely used models from five developers. GPT-5.5, Grok 4.3, DeepSeek V4, and Llama 4 Maverick met the support workers' standards in just 26-59% of conversations when cues were concentrated, and only 3-6% when dispersed. Claude Sonnet 5 and Claude Opus 5 showed a smaller gap, meeting the standards in 68-97% of concentrated conversations and 59-87% of dispersed conversations. Even the best-performing model, Claude Opus 5, fell short of the gold standard in 13% of conversations when cues were dispersed. Safety evaluations centred on explicit, legible concerns can overstate how well safeguards protect people when signs of distress emerge gradually or indirectly through ordinary conversations. For claims of improved safety to be taken seriously, developers must design and assess safeguards around how people actually communicate. |
| 2026-11-19 | [From AI Surrogate to Engineering Tool: Toward Production Use of Fiber-Orientation Prediction in Crash CAE](http://arxiv.org/abs/2611.25991v1) | Osamu Ito, Yoshikazu Nakagawa et al. | Crash simulations of glass-fiber-reinforced plastic parts are highly sensitive to the fiber-orientation field supplied to the material model. In conventional workflows, that field comes from resin-flow analysis, which requires molding-specialist support and takes about 25 hours of work per part. We previously proposed a surrogate model that addresses this bottleneck by predicting the orientation field in about 3 minutes. This paper reports the implementation, validation, and operational design that turned the model into an engineering tool for Honda design workflows. We present staged validation from 2D plates to production parts, evaluation against a predefined engineering angular-error criterion, and the expert review that approved design use in March 2025; the system has since been transferred to the crash-safety group and is in initial, limited use in directed fracture-prediction scenarios. The developed system combines injection-gate setting support for cases in which gates are unknown (90% gate-count accuracy, mean position error 11% of flow length), voxelization, surrogate inference, automatic mapping to the crash CAE model, and report generation in a single GUI. The complete orientation-preparation workflow takes about 0.1 hours, a 99.6% reduction from the conventional 25-hour flow-analysis route, and can be run directly by crash CAE engineers without molding-specialist support. As evidence of practical impact, component-level crash analysis using the predicted orientation field reduced the absolute peak-force error from 19.0% with an isotropic baseline to 11.4%, while confining predicted fracture to the test-observed region. The deployment lesson is that a surrogate model alone is not enough for practical adoption: gate-setting support, automatic data conversion, and shareable reporting are what make the model usable as a design tool. |
| 2026-11-19 | [Safe Subgame Resolving in Multiplayer Games](http://arxiv.org/abs/2611.25981v1) | Boning Li, Longbo Huang | Poker and other games with hidden information are too large to solve in one pass. Strong systems compute a complete blueprint strategy offline and refine it in real time inside the reached subgame. In 2p0s games, a safety theorem guarantees that subgame resolving never increases how much an adversary can gain by exploiting the resolved strategy relative to the blueprint strategy. With three or more players acting independently, no such guarantee exists yet, and heuristic resolving can increase the profile's best-response gap. We prove the first conditional safety theorem for uncorrelated multiplayer subgame resolving, a post-hoc certificate for any delivered update. The best-response gap of the resolved profile exceeds that of the blueprint by at most four terms. Two concern the resolver: the subgame value the blueprint promised without realizing ($A_i^{S}$) and the resolver's subgame solve error ($\varepsilon_i$). The other two are the opponents' best-response gains inside the subgame over their promised values ($\sum_q M_{q}$) and a structural tax ($Θ$), which measures how much the players' total utility can change. The tax is zero in constant-sum games, including every rake-free poker variant. Constructions attain each term with equality. On three-player Leduc poker and on Kuhn poker with up to six players, resolving without a certificate raises the gap beyond a small tolerance in 12% to 100% of the updates. The certificate flags every such update and accepts 41% (Kuhn) and 86% (Leduc) of the standard resolving updates. A resolver that keeps the blueprint whenever the certificate exceeds the tolerance delivers no harmful update. The certificate needs only one best-response pass per opponent inside the subgame, so real-time multiplayer solvers that refine a blueprint can attach it as a safety check. Code is available at https://github.com/lbn187/MultiPlayerSolver |
| 2026-11-19 | [Safe Data-Driven Insulin Dosage Decision Based on Physiology-Constrained Gaussian Process Learning](http://arxiv.org/abs/2611.25962v1) | Deheng Cai, Xiang Lu et al. | Multiple daily injections (MDI) therapy is widely adopted by insulin-treated subjects with diabetes mellitus. In this work, we study how to ensure that data-driven model learning is aligned with the underlying physiological principles, and introduce a safe data-driven learning and decision framework for the MDI therapy. To provide credible glucose predictions for the decision, a physiology-constrained Gaussian process learning method is developed for glycemic dynamics modeling by encoding the physiological priors as constraints in the model training. Considering the natural day-to-day cycle of insulin decisions, a prediction-based optimization formulation with an objective function that has a penalty term associated with the previous dosage is employed for the insulin therapy, together with the asymmetric penalties on hyper- and hypoglycemia for safety. The effectiveness of the proposed method is verified using the 10-adult cohort from the FDA-accepted UVA/Padova T1DM simulator, and an advisory mode analysis is also conducted based on the clinical data from 40 type 1 diabetes and 55 type 2 diabetes subjects to indicate the safe and physiologically plausible decisions in the real-life applications. |
| 2026-11-19 | [Question, Then Affirm: Restoring Commitment Does Not Restore Refusal](http://arxiv.org/abs/2611.25929v1) | Haoyang Xu, Qirui Wang et al. | We find that safeguards for image generation can be bypassed without changing what a message says. A sign stating a harmful proposition is blocked, but the same proposition phrased as a question and followed by a check mark is generated, even though the resulting image endorses the original claim. On a 300-case challenge set, gpt-image-2 returns images for 43.3% of question-affirmation requests versus 16.3% of direct statements, with 41.3% versus 15.3% of cases yielding images that endorse the original proposition. We define this as Deferred Semantic Commitment (DSC): commitment is withdrawn by the question and restored by a separate element, while safety behavior follows only the withdrawal. To investigate why, we study an open-weight language model and find that re-endorsement only partially reverses the refusal-related change introduced by questioning. These findings reveal a mismatch between recovered meaning and refusal behavior, motivating safeguards that assess the completed message rather than its individual elements. |
| 2026-11-19 | [Verifiable Logic in Medical LLMs: A Critical Review of Agentic Clinical Reasoning Optimization and Neuro-Symbolic Grounding](http://arxiv.org/abs/2611.25898v1) | Youssef ElNahas, Adel Ammar et al. | Clinical large language models are usually judged by whether the final answer is right. Yet a model can reach a fluent, plausible output through invalid reasoning that static benchmarks do not detect, and confident hallucination, causally invalid inference, and spurious demographic shortcuts can pass conventional evaluation. This critical review covers four paradigms in turn: how clinical hallucination arises and why static metrics miss it; the agentic hallucination cascade, in which multi-agent and tool-augmented systems propagate and amplify early reasoning errors; process reward modeling as step-level supervision; and neuro-symbolic grounding, in which formal logic engines, ontology reasoners, and knowledge graphs place verifiable constraints on neural output. Within their encoded domains, systems with symbolic veto layers offer guarantees that hold relative to their encoded knowledge, and they report large gains on the safety measures they use. These results come from single studies, mostly in simulated or small-sample settings, and are not directly comparable with neural baselines. Process reward models remain bounded by the statistics of their training data. We propose a conceptual framework that maps each agentic failure mode to a symbolic intervention point, and we argue that clinical AI safety depends substantially on system architecture. Because current regulation emphasizes performance and does not require verification of reasoning steps, we argue that responsible certification should also audit reasoning trajectories alongside accuracy on static benchmarks. |
| 2026-11-19 | [Design and Validation of a Telescopic Chassis-Assisted Stair-Climbing Mechanism for Powered Wheelchairs](http://arxiv.org/abs/2611.25798v1) | Yasuhiro Fukuoka, Haruya Shirado et al. | We propose a simple stair traversal mechanism that augments a four-wheel wheelchair with an auxiliary telescopic chassis actuator. The proposed mechanism provides the following key contributions during stair ascent: (A) enhanced safety by ensuring that either the front or rear wheels remain stationary at all times; (B) primary reliance of the required force and mechanical work on the telescopic chassis rather than the wheels; (C) insensitivity of the required force and mechanical work to stair inclination; and (D) reduced risk of wheel slip on steeper stairs. In addition, the telescopic chassis is expected to improve level-ground mobility by enabling tight turning with the shortest wheelbase and enhancing stability with the longest wheelbase. |
| 2026-11-19 | [When Safety Rationales Hallucinate: Diagnosing and Grounding Multimodal Safety Guards](http://arxiv.org/abs/2611.25793v1) | Yingshuo Wang, Weixiang Zhao et al. | Multimodal safety guards increasingly generate rationales alongside safety predictions, yet whether these rationales are reliable remains largely unexplored. We systematically study *rationale hallucination* from two perspectives: visual rationale hallucination, where rationales contain unsupported visual facts, and safety attribution hallucination, where grounded content is assigned an incorrect safety interpretation. Across rationale-generation methods, multiple vision-language backbones, and safety datasets, we find that rationale hallucination is widespread and does not consistently diminish with higher classification accuracy or increased model scale. Further analysis of the Think-with-Image paradigm shows that hallucination is concentrated in fine-grained attributes and relations, while visual tools are often invoked infrequently or without sufficient spatial focus. Based on these findings, we propose *Tool-Contrastive Grounding Reinforcement Learning* (TCG-RL), which rewards tool use only when it improves rationale grounding over a tool-free reference. TCG-RL substantially reduces rationale hallucination while maintaining competitive safety prediction performance. By revealing the gap between prediction correctness and rationale reliability, our study motivates a shift in multimodal safety guarding from outcome-only evaluation toward safety decisions that are both accurate and verifiably grounded. |

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



