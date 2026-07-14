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
| 2026-07-13 | [Zonal-flow generation and saturation of electromagnetic ion-scale turbulence in tokamaks](http://arxiv.org/abs/2607.11789v1) | Y. Zhang, T. Adkins et al. | Local flux-tube gyrokinetic simulations of ion-scale turbulence in tokamak plasmas at finite plasma beta are conducted to investigate the generation of zonal flows via turbulent stresses. A parameter scan in the safety factor $q$ and electron beta $β_e$ reveals a transition from low- to high-transport states when $β_{\mathrm{eff}} \equiv q^2β_e$ exceeds a certain critical value $C_{\mathrm{nl}}$. While the linear stability limits for kinetic and ideal ballooning modes also scale as $β_e \propto 1/q^2$, they lie above the observed transition, indicating that the effect is not due to linear instabilities but to nonlinear dynamics. At low $β_{\mathrm{eff}}$, Reynolds stress dominates and drives zonal flows. At higher values, Maxwell stress becomes comparable, suppressing zonal-flow formation and leading to divergent transport. This nonlinear-transition boundary is determined for both the Cyclone Base Case and a spherical tokamak (ST40) configuration, suggesting that the relation $β_{\mathrm{eff}} = C_{\mathrm{nl}}$ may have broader applicability, though $C_{\mathrm{nl}}$ appears to be configuration-dependent. For the Cyclone Base Case, the ratio of energy transfer rates into zonal flows due to Maxwell and Reynolds stresses is observed empirically to scale as $β_e$ for $β_e$ below a critical value $β_{e,\mathrm{sb}}$ (scaling breakdown). The value of $β_{e,\mathrm{sb}}$ is found to increase with decreasing aspect ratio, suggesting that the linear scaling remains valid over a wider range of $β_e$ for more compact magnetic equilibria. This low-$β_e$ scaling provides the basis for a practical method to predict the nonlinear-transition threshold with minimal reliance on highly electromagnetic nonlinear simulations. |
| 2026-07-13 | [MIRA: A Modular Open-Source Micro-UAV for Indoor Research](http://arxiv.org/abs/2607.11785v1) | Lucas K. de Oliveira, Felipe A. G. Tommaselli et al. | Indoor robotics research increasingly relies on micro-UAVs whose airframe, electronics, and control software are fully open to modification. Off-the-shelf platforms rarely expose the low-level access required for such modifications, while building a custom alternative typically requires substantial engineering effort before flight testing can begin, leaving many laboratories to work within constraints that limit the scope of their research. We present MIRA (Modular Indoor Research Architecture), a low-cost, open-source micro-UAV for indoor research built around a replicable 3D-printed PLA airframe and a containerized low-level software package managing the companion-to-autopilot communication bridge via Micro XRCE-DDS. Designed as a white-box architecture, core subsystems are individually replaceable without firmware refactoring, supporting local fabrication and component substitution from existing lab inventory. We characterize MIRA through manual flight in position-control mode within an optical motion-capture volume, where the communication pipeline sustains a median companion-to-autopilot latency of 0.02 ms and power spectral density analysis confirms the structural vibration energy stays concentrated in a narrow 90 to 110 Hz band, isolated from the sub-20 Hz control bandwidth and within the autopilot safety thresholds. |
| 2026-07-13 | [Evaluating RE Practices for Explainability: Synthesizing Insights from Daimler Truck into an Explainable RE Framework Proposal](http://arxiv.org/abs/2607.11771v1) | Umm-e- Habiba, Lucas Mauser et al. | Explainability has emerged as a critical requirement for AI-based systems, particularly in safety-critical and regulated domains. Although prior research has proposed frameworks, patterns, and user-centered approaches to support explainability, there is limited empirical understanding of how existing Requirements Engineering (RE) practices support explainability requirements across the RE lifecycle, especially in an industrial context. This paper reports early findings from an ongoing industry-based study investigating how explainability requirements are elicited, specified, and validated using established RE techniques. We conducted a multi-phase qualitative study with eight practitioners at Daimler Truck, employing think-aloud protocols and moderated group discussions across requirements elicitation, specification, and validation steps. Our preliminary analysis reveals recurring challenges across all steps, including conceptual ambiguity during elicitation, limited testability and expressiveness during specification, and fragmented validation due to vague criteria and regulatory uncertainty. These findings indicate that current RE practices provide limited support to systematically address explainability requirements. The paper contributes empirical insights into step-specific and cross-cutting challenges and outlines a research vision toward developing an empirically grounded RE framework for explainable AI-based systems. |
| 2026-07-13 | [When Local Monitors Miss Compositional Harm: Diagnosing Distributed Backdoors in Multi-Agent Systems](http://arxiv.org/abs/2607.11751v1) | Yibo Hu, Ren Wang | As multi-agent, tool-using LLM systems are deployed, a common safety net is a runtime monitor that checks each message, tool call, or step on its own. We show this net has a fundamental hole. A distributed backdoor splits a harmful payload across agents, so every local check passes while the assembled object is the attack. The monitor can be right on every step and still miss the attack. The problem is not splitting itself: split fragments can still leak suspicious tokens or provenance edges. The hard case is \emph{local benignness}. No fragment carries the harm, and what is left looks like ordinary benign traffic. We formalize this as an \emph{observability boundary}: a monitor catches only what its view can tell apart from benign traffic. We prove that once the fragments look benign in the monitored view, no detector on that view can catch them, however strong it is. Across a controlled testbed, an external benchmark, and end-to-end agent runs, local monitors lose the signal exactly as local evidence disappears, and it returns only when the monitor sees the assembled object. A monitor trained only on benign traffic recovers the attack's code structure across held-out encodings (0.874 mean AUROC). A decoded-view gate, given the encoding family, blocks every tested attack. But seeing more is not enough: full-trace monitors and decoders still fail unless they reach the representation where the payload is exposed. Local safety is not global safety when harm is compositional, and the open problem is finding that representation. |
| 2026-07-13 | [AutoPath: Learning Transferable Goal-Conditioned Stochastic Path Prior for Safe Navigation Without Human Demonstrations](http://arxiv.org/abs/2607.11739v1) | Ziyang Zhang, Boyang Zhou et al. | Real-time navigation in cluttered and dynamic environments requires collision-free and dynamically feasible motion under limited perception. However, feasible navigation behaviors are inherently multimodal because multiple paths may exist around obstacles. In this paper, we formulate navigation as learning a transferable goal-conditioned stochastic path prior that models a reusable distribution over goal-aligned geometry-consistent local paths conditioned on local observations. This formulation enables structured sampling of navigation candidates, allowing multiple feasible paths to be explored through sampling without relying on robot-specific motion constraints. To this end, we introduce a goal-aligned canonical state representation that removes in-plane rotational ambiguity and normalizes local geometry with respect to the goal, enabling rotation-invariant path distribution learning. We further develop a structured prior learning framework that parameterizes local paths using a geometry-aware polar action manifold and incorporates risk-sensitive utility shaping with multi-goal distributional rollouts for stable and safety-aware planning. Extensive experiments in dense static environments and dynamic pedestrian scenarios demonstrate that the proposed method achieves consistently high success rates with competitive efficiency while enabling cross-platform transfer of a single path prior learned on differential-drive robots to quadruped platforms without retraining. |
| 2026-07-13 | [Agent Hacks Agent: Autoresearch for Production-Agent Red-Teaming](http://arxiv.org/abs/2607.11698v1) | Xutao Mao, Xiang Zheng et al. | Production LLM agents such as Claude Code and Codex operate over untrusted content, files, commands, and workspace state, making safety failures directly actionable. Red-teaming must therefore keep pace with evolving models and tools. Existing approaches mainly optimize attack success and preserve artifacts such as benchmarks, payloads, or attack programs, which record where attacks succeed but not the enabling conditions behind unsafe agent behavior. We study automated red-teaming for production LLM agents using one agentic research environment to discover reusable vulnerability knowledge about another. We present AHA, a falsifiable discovery loop that proposes a vulnerability hypothesis, constructs a falsifier, instantiates a valid attack, executes it in a sandboxed harness, reflects on the trajectory, and promotes confirmed findings into a Vulnerability Concept Graph (VCG). Each concept links an attacker-facing surface to an unsafe trajectory through a claim, enabling condition, falsifier, transfer prediction, and supporting evidence. Across Claude Code and Codex on three scenarios covering direct and indirect attacks, the discovered concepts reveal a reusable vulnerability core across models and agents. A frozen VCG requires no further search and outperforms the strongest frozen discovery baseline by 14.2 percentage points under the same single-shot protocol, while transferring across scenarios and attack channels. The resulting VCG provides an auditable artifact for production safety teams to inspect vulnerabilities, validate patches, and accumulate reusable safety knowledge. Our code is available at https://github.com/henrymao2004/Auto-research-red-teaming-in-sleep. |
| 2026-07-13 | [A Model for Mediating Multi-Modal Human Intent into Safe Maneuvers for UAVs](http://arxiv.org/abs/2607.11654v1) | Sofia Nelson, Dalal Alrajeh et al. | Direct human interaction with autonomous UAV systems can be enabled through modalities such as speech, gestures, and graphical interfaces. However, interpreting such inputs as directly executable commands introduces safety risks in dynamic environments. Operator requests may conflict with terrain constraints, inter-UAV separation requirements, or flight-envelope limitations. In this paper, we present a requirements-governed maneuver-response model that mediates multi-modal human intent into safe UAV maneuvers by treating operator inputs as bounded maneuver requests rather than direct commands. Requested maneuvers are mapped to constrained motion primitives and processed through a structured request-evaluate-execute pipeline. Each request is interpreted with associated confidence, validated against terrain, separation, workspace, and flight-envelope constraints, and either constrained, rejected, or executed under continuous runtime monitoring. We further formalize the approach as a requirements-based specification model in which maneuver primitives are associated with explicit preconditions, invariants, guard conditions, and postconditions governing admissibility, execution safety, and emergency handling. These requirements support runtime verification and future reactive synthesis approaches. We present an initial lab-based validation demonstrating that voice and GUI-based inputs can be reliably interpreted and safely executed as constrained maneuver requests. |
| 2026-07-13 | [Auditing the Risk Claims of Distributional Reinforcement Learning](http://arxiv.org/abs/2607.11607v1) | Hari Prasad | Distributional reinforcement learning agents learn full return distributions that are increasingly read at face value: for interpretability, risk-sensitive control, and safety monitoring. We ask a question theory anticipates but that has not been measured directly: are the risk claims of a trained distributional agent true? Our audit combines a decision-relevant screening metric (the excess Wasserstein gap between the top two actions, which equals the mass by which first-order stochastic dominance is violated), ground truth from snapshot-restart Monte Carlo, and a statistical harness (permutation nulls, bootstrap refutation, FDR control) without which the audit itself manufactures false conclusions. Across QR-DQN, C51, and IQN on MinAtar (33 runs), 40-95% of the strongest claimed risk trade-offs are refuted at 95% confidence, the placement of the strongest claims is statistically indistinguishable from truth-blind, and essentially no claim is confirmable: for these agents, the learned "risk" reflects a training artifact rather than environment stochasticity. The artifact is structural (fully formed early in training, uncorrelated with final score, idiosyncratic to each seed) and appears unchanged at full-Atari scale, with every top Breakout claim of a pretrained near-state-of-the-art QR-DQN refuted. Positive controls of known magnitude confirm 96-100% of real claims (correlation 0.89-0.92): the reading measures the agents, not the audit. Acting on the heads' CVaR advice at their most-flagged states ranges from beneficial to significantly worse than chance. Neither training for risk nor ensembling removes the artifact, and recalibration passes the audit only by nullifying the claims: the head is uninformative, not merely miscalibrated. We release the toolkit and document two silent pitfalls that produced convincing but wrong audits of our own. |
| 2026-07-13 | [Cardano's Voltaire Governance: Complete Specification and Research Program](http://arxiv.org/abs/2607.11601v1) | Nimrod Talmon, Oghenekaro Elem | Blockchain governance, the set of processes by which decentralized protocols evolve, remains a fundamental challenge in balancing adaptability, security, and stakeholder representation. This technical report analyzes Cardano's Voltaire governance system, the on-chain framework introduced via CIP-1694 and enacted through the Chang hard fork in September 2024, and lays down a corresponding research program.   We make two contributions. First, we provide a complete technical specification of Voltaire's mechanisms, including its three-body architecture, seven governance action types, voting rules, and its constitutional framework; this specification is sufficient for implementation or formal analysis. Second, we establish a research agenda for principled governance optimization, including design of an agent-based simulation platform, analysis of delegation dynamics, optimization of multi-objective parameters, and game-theoretic incentive design; we provide preliminary results, including a formal governance kernel: a minimal executable model capturing self-amending governance as a state-transition system and enabling rigorous safety and liveness analysis.   Our report offers a comprehensive technical overview and invites the research community to advance blockchain governance science through rigorous study of Voltaire as a live, large-scale experiment now managing a treasury valued at approximately \$235 million (1.47B ADA as of early July 2026). |
| 2026-07-13 | [Beyond Benchmarks: Exposing the Hidden Crisis in Bangla Hate Speech Detection](http://arxiv.org/abs/2607.11597v1) | Faria Afrin Tisha, Fariya Tabassum et al. | The spread of hate speech (HS) across different social media platforms (SMPs) poses a major concern for online safety and ethical moderation. Automatic detection of HS remains a challenging task, especially in under-resourced languages like Bangla, due to cultural context, implicit expressions, and informal linguistic patterns. This study aimed to expose the crisis of Bangla HS detection systems by diagnosing how and why benchmark-trained models fail to identify implicit, context-dependent HS. Six architectures (FastText + CNN, FastText + LSTM, FastText + BiLSTM, BanglaBERT, BanglaBERT + CNN, and BanglaBERT + BiLSTM) were trained on benchmark datasets (about 75,000 posts) and a merged multi-source dataset (about 120,000 posts), then externally validated on an annotated dataset (about 200 posts) collected from Facebook, Twitter, and YouTube, labeled as HS and non-HS, where HS was further categorized as explicit and implicit. BanglaBERT achieved an F1-score of 91.4% on benchmark datasets but declined to 75.3% on the external set and 63.4% for implicit HS involving sarcasm and emojis. The accuracy of FastText + CNN dropped from 78.0% to 51.2% under similar conditions. Emoji-aware preprocessing improved implicit HS detection by up to 12%, whereas emoji removal caused a notable decline in performance (F1: 0.75 to 0.63). Frequent misclassifications in politically charged or satirical comments revealed over-policing risks. This study not only exposes the generalization crisis due to implicit, culturally embedded, and emoji-laden expressions but also underscores the need for developing adaptive, emoji-aware, and culturally grounded frameworks that ensure ethical moderation while preserving freedom of expression. Findings of this study provide insights for researchers, SMPs, and policymakers to design more context-sensitive HS detection systems for low-resource languages. |
| 2026-07-13 | [Technical Report on the CVPR 2026@AdvML Workshop Challenge](http://arxiv.org/abs/2607.11560v1) | Tianyuan Zhang, Zonglei Jing et al. | Vision-language agents (VLAs) are increasingly used to interpret complex driving scenes and support safety-critical reasoning. This report presents the CVPR 2026@AdvML Workshop Challenge on adversarial multimodal attacks against autonomous-driving VLAs. Built on DriveLM-style multi-view visual question answering, the challenge represents each scene with six synchronized camera images and a structured collection of driving-related question-answer pairs. Participants generate adversarial images and suffix-only textual perturbations that induce model responses to deviate from reference answers while preserving image fidelity and limiting textual cost. The competition comprises two phases, with Phase II adding a hidden black-box model to assess transferability. We describe the task design, submission rules, evaluation protocol, and leaderboard results, and then examine five leading submissions for which technical reports were available. Across these reports, several recurring patterns emerge: image-side attacks are favored by the suffix penalty; scene-level, multi-view optimization is more effective than treating views in isolation; QA types and graph structure provide useful priors for allocating attack budget; feature-space objectives can improve black-box transfer; and typographic content embedded in camera images exposes a persistent vulnerability in driving VLAs. These findings provide a practical reference for future robustness evaluation and defense design in multimodal autonomous-driving systems. |
| 2026-07-13 | [Vinci2: Providing Proactive Assistance in Continuous Egocentric Videos](http://arxiv.org/abs/2607.11523v1) | Gong Sitong, Tianyu Yan et al. | When should an intelligent assistant speak up without being asked? Continuous egocentric video offers rich, evolving context that enables a new form of assistance: one that is proactive rather than merely reactive. Yet existing approaches either wait passively for user queries or treat every detected event as requiring a response, without considering the user's history, current activity, or whether assistance would actually be welcome. We reframe proactive assistance as a context-dependent decision problem: the agent must not only perceive what is happening, but reason over accumulated temporal context to determine when and whether to intervene. To this end, we present Vinci2, a proactive egocentric assistance system that advances the on-device assistant Vinci from reactive response toward proactivity. On the evaluation side, we present EgoServe, the first large-scale benchmark for proactive assistance in continuous egocentric video. EgoServe comprises over 3,000 service instances organized along 4 temporal memory horizons, ranging from immediate safety alerts to long-term habit coaching, across 10 service categories. On the modeling side, we propose EgoMemo, a training-free, memory-augmented agent that maintains three complementary memory representations: multi-scale temporal summaries, a semantic knowledge graph, and visual embedding archives. At each timestep, EgoMemo performs retrieval-augmented reasoning to determine whether assistance is warranted and, if so, produces contextually grounded responses. Experiments demonstrate that EgoMemo establishes strong baselines on EgoServe while remaining competitive on existing egocentric benchmarks. Our benchmark and code are publicly available at \href{https://sitonggong.github.io/EgoServe-page/}{Vinci2}. |
| 2026-07-13 | [HyperSafe: Inference-Time Safety Recovery for Fine-Tuned Language Models](http://arxiv.org/abs/2607.11475v1) | Aznaur Aliev, Carlos Hinojosa et al. | Safety alignment in large language models can be fragile under fine-tuning, as even benign task adaptation may increase harmful compliance. Existing defenses mainly follow two directions: they either intervene during or after fine-tuning through retraining or weight modification, which can be costly and may hurt task performance, or they use model-agnostic safety classifiers, which may miss failures specific to a given fine-tuned checkpoint. These limitations motivate a post hoc, model-specific, and non-invasive approach to safety restoration. To meet these requirements, we propose HyperSafe, a framework that restores safety behavior by generating a model-specific Safe Side Network (SSN) for each fine-tuned checkpoint. HyperSafe uses layer-wise activation fingerprints to capture how fine-tuning changes the model's inner representations. With a small set of given calibration prompts, the hypernetwork maps these fingerprints to the parameters of the \ssn{} in a single forward pass. The generated \ssn{} runs alongside the frozen fine-tuned model and performs prompt-level safety classification: harmful prompts are routed to refusal, while safe prompts are answered by the original fine-tuned model. Thus, HyperSafe requires no gradient updates, no safety data at deployment time, and no modification to the deployed model weights. We evaluate HyperSafe on two model families, Qwen2-7B and LLaMA-3-8B, across multiple safety benchmarks. HyperSafe reduces harmful response rates from 19-31% to below 1% on every held-out checkpoint, while keeping downstream task accuracy within 1% of the fine-tuned baseline on average. Code is available at https://github.com/nokronim/project-safety-remedy. |
| 2026-07-13 | [Decentralized Model Predictive Control of Connected and Automated Vehicles with Coupled Safety Constraints](http://arxiv.org/abs/2607.11403v1) | Philip Schultheis, Kimia Chavoshi et al. | Connected and Automated Vehicles (CAVs) operating on lane-free highways offer substantial gains in traffic efficiency. However, their inherent nonlinear dynamics and the presence of coupled, nonconvex safety constraints present critical challenges to control design. Centralized Model Predictive Control (MPC) ensures safety, but suffers from scalability and communication limitations. To address these challenges, this paper investigates decentralized MPC (DMPC) for CAV coordination, focusing on iterative, non-cooperative algorithms, including Jacobi-type and Gauss-Seidel-type. A novel decoupling method is developed to transform nonconvex safety constraints into convex, locally enforceable constraints, inspired by buffered Voronoi cells. The simulation results show that the proposed DMPC algorithms achieve safe and efficient vehicle trajectories while substantially improving scalability, highlighting their potential for future lane-free CAV traffic systems. Ultimately, the results indicate that the most suitable decentralized control strategy depends on the desired trade-off between safety, performance, and computational efficiency. |
| 2026-07-13 | [Compile, Then Page: Executable SOP Programs and a Capability-Gated Runtime for Procedural LLM Agents](http://arxiv.org/abs/2607.11346v1) | Chenglin Yu, Li Yin et al. | Enterprise agents must follow long-horizon, conditional, safety-critical standard operating procedures (SOPs). We compile machine-readable SOP constraints into executable pseudo-code and run them with a program-guided (PG) stack machine that pages the active frame while an LLM performs semantic execution. A three-arm SOPBench study across six models separates representation from runtime: compiled text never significantly hurts and gains up to 16.0 points where official prose underperforms. Runtime guidance is capability-gated. Two strong models independently show positive seven-domain PG contrasts (58:19 and 75:31 discordant pairs), whereas weak models are harmed. A full-program cursor ablation (active frame first, complete program retained) recovers much of the strong-model refusal gain; selective visibility adds a smaller improvement. Paired probe and audit measurements track this divide to spontaneous state discipline rather than reconstruction ability. On Bank the three primary arms rise from 70.4 to 86.4 to 92.8, with 100% refusal correctness. Practical guidance: compile first; enable active-frame paging only after a model-level discipline check. |
| 2026-07-13 | [Calibrated e-CUSUM Decoding for Quantized Reasoning Models: Why Token Log-Probability Is the Wrong Observable for Decoding Monitors](http://arxiv.org/abs/2607.11317v1) | El Hassane Ettifouri, Ayoub Belfatmi et al. | Low-bit quantization makes small reasoning models inexpensive to deploy but can degrade their chains of thought. This motivates decoder-side monitors that intervene when generation becomes unreliable. We show that a natural candidate, the centered token log-probability increment $\log p(w_t)+H_t$, is the wrong observable for this purpose. Under the model's own sampling law it is a mean-zero martingale by construction, so it measures sampling self-consistency rather than trajectory health and is nearly silent during confident repetition, where both $\log p(w_t)$ and entropy are close to zero. We introduce a training-free decoding controller that combines (i) a degeneration-aware alarm score fusing token uncertainty with explicit verbatim repetition and (ii) a calibrated e-process-inspired sequential detector. The raw product process is Ville-valid under a conditional-mean null, while the deployed CUSUM-floored statistic is treated as an empirical change detector because the score is history-dependent and autocorrelated. On GSM8K with DeepSeek-R1-Distill-Qwen-1.5B in FP16 and INT4, calibration turns a monitor that fires on 93--95% of generations into a selective detector of failing traces ($φ\approx 0.3$, precision $\approx 0.6$ against a 0.38 base rate). In this pilot, the controller reduces measured verbatim-degeneration signals and yields a positive but statistically inconclusive INT4 accuracy change from 63% to 69% (paired McNemar $p=0.18$, $n=100$), at a 28% token-budget cost. We also find that non-termination, rather than looping, is the dominant failure mode on GSM8K. The main contribution is methodological: an explanation of why centered token log-probability is inadequate for decoder monitoring and a calibrated, cautiously evaluated replacement. |
| 2026-07-13 | [The Paternalistic Filter: Epistemic Injustice and Differential Refusal in LLM-Mediated History Education for Marginalized Romanian Students](http://arxiv.org/abs/2607.11292v1) | Alexis Popovici, Andrei Ionascu et al. | As Large Language Models (LLMs) are increasingly deployed as conversational tutors, they risk institutionalizing systemic inequalities. This study presents a systematic API audit of four LLMs acting as history tutors, evaluating 1,800 responses regarding the 1989 Romanian Revolution across five student personas varying by ethnicity and socio-economic tier. We uncover four interconnected patterns of \emph{epistemic paternalism}: (1)~\textbf{Differential Refusal}, where safety-aligned models block 76.7\% of educational requests from low-tier students; (2)~\textbf{Epistemic Gatekeeping}, evidenced by a 3$\times$ reduction in access to geopolitical complexity (e.g., the contested ``coup theory'') for marginalized learners; (3)~\textbf{Agency Theft}, a lexical shift where models like LLaMA produce a 5$\times$ higher victimization-to-politics vocabulary ratio for Roma students compared to elite peers; and (4)~\textbf{Elite Hermeneutics}, where AI tutors disproportionately withhold epistemic confidence and justification scores from low-resource demographic profiles. We argue that current safety alignment acts as a paternalistic filter, transforming conversational AI into agents of narrative segregation -- a manifestation of \emph{hermeneutical injustice} in Fricker's~\cite{fricker2007} sense that demands urgent pedagogical auditing. |
| 2026-07-13 | [Extending Decision Maps for Sustainable Safety and Security in Self-Adaptive Systems](http://arxiv.org/abs/2607.11274v1) | Marco Stadler, Wesley K. G. Assunção et al. | Sustainability refers to a system's ability to maintain its functionality and endure over time. Hence, sustainability is a highly desirable property of software systems, including Self-Adaptive Systems (SASs). SASs can change (adapt) their behavior at runtime to continue achieving their objectives despite external or internal impacts. SASs' intended long-term system behavior can be expressed through a sustainability-driven visual modeling notation called Decision Maps (DMs). Although DMs have been proven helpful, they lack adequate modeling support for safety and security concerns. We address this limitation by extending the current notation for sustainability-driven modeling of SASs to better accommodate the unique characteristics of safety and security scenarios. First, we introduce an additional modeling dimension to account for safety incidents. Second, we adopt a fine-grained divide-and-conquer approach, modeling from distinct temporal security viewpoints ("security modes") to address security. We employ the extended DM notation in a real-world use case scenario provided by our industry partner to assess its feasibility and suitability for practitioners. Our results indicate that our modeling notation helps capture security and safety scenarios more accurately and provides holistic support for the self-adaptation life cycle phases. |
| 2026-07-13 | [DeepBias: Adaptive In-depth Probing of Social Biases in LVLMs](http://arxiv.org/abs/2607.11228v1) | Anqi Li, Jie Zhang et al. | While Large Vision-Language Models (LVLMs) demonstrate remarkable capabilities, they remain highly susceptible to embedded social biases. Existing bias evaluation protocols predominantly rely on static datasets, which provide only a superficial assessment, as their fixed test cases cannot adaptively evolve to measure the true depth and limits of model vulnerabilities. We introduce DeepBias, an adaptive framework for the in-depth probing of social biases in LVLMs with carefully designed agents. Our approach operates through a dynamic ''generation-evolution-probing'' loop. First, a generative ProposerAgent synthesizes test data and is iteratively updated via Direct Preference Optimization (DPO) based on the target LVLM's responses, exploring model-specific failure modes. Second, an autonomous skill-driven DiggerAgent rewrites each test data across multiple probing turns, adaptively selecting from a curated skill library of deepening and rewriting strategies. At each turn, this process is conditioned on the model's previous response, enabling progressively deeper biases to be exposed. Furthermore, we build a benchmark named DeepBiasBench using our framework. By employing an ensemble of five diverse state-of-the-art LVLMs as anchors, the benchmark captures vulnerabilities shared across architectures. Comprehensive experiments demonstrate the effectiveness of our framework and show that DeepBias provides a challenging benchmark for in-depth bias evaluation, establishing an evolutionary paradigm for LVLM safety assessment. |
| 2026-07-13 | [Heterogeneous Agent Cohorts for Safe Open-Ended Exploration with Runtime Constraint Memory](http://arxiv.org/abs/2607.11226v1) | Tengjiao Liu | LLM agents today are caught in an awkward bind. Lock them down with static safety instructions and they rarely venture beyond the obvious; give them free reign with tools and multi-agent debate, and safety violations quickly follow. Rather than forcing a single model to juggle both creativity and caution, we separate the concerns across specialized roles. A Disrupter generates unconventional proposals, a Validator enforces hard runtime checks at the tool gateway, and a Broker pulls in distant but relevant analogies. Failures are not discarded -- they are compiled, via MCTS, into compact, signed constraint patches we call Scars. These patches are cached locally and inherited by future cohorts, turning repeated failures into reusable, low-cost runtime constraints. In a spatial-semantic sandbox (N=20 runs, p<0.01), our cohort reaches remote targets where debate fails, the Validator prevents all executed breaches, and Scars reduce token consumption by 15.1% by avoiding redundant validator checks. Furthermore, credit-based Communication Allocation Scores (CAS) restrict outbound bandwidth, reducing overall token costs by 55.9% under resource constraints. |

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



