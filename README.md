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
| 2026-06-16 | [A Red-Team Study of Anthropic Fable 5 & Opus 4.8 Models](http://arxiv.org/abs/2606.18193v1) | Nicola Franco | We evaluate the adversarial robustness of two frontier large language models (LLMs) developed by Anthropic, Fable 5 and Opus 4.8, against four families of automated jailbreak attack across 7 826 harmful intents spanning a ten-category harm taxonomy. Using the HackAgent red-teaming framework, hundreds of thousands of adversarial attempts were generated and every apparent success was independently re-adjudicated by a panel of three judge models (majority vote). Both models resist the majority of attacks, but the residual surface is larger than aggregate framing suggests: it is dominated by adaptive iterative attacks, while static obfuscation is near-fully neutralised. The strongest adaptive search (tree-of-attacks) breaks Opus 4.8 on 11.5% of intents overall, whereas Fable 5 stays in the single digits (6.1% worst-case). Aggregate rates therefore should not be read as reassurance. Even in these hardened configurations, the two models produced 1 620 (Opus 4.8) and 702 (Fable 5) panel-confirmed harmful completions spanning every harm category, located automatically, cheaply, and within the first one or two refinement steps by an attacker model with no human expert in the loop. The reasonable conclusion is that even the best, most-tested frontier models remain reliably breakable under sustained automated pressure. |
| 2026-06-16 | [Perturbative QCD as a quantitative tool in the years 1976-2000](http://arxiv.org/abs/2606.18176v1) | R Keith Ellis | This paper traces the development of precision QCD in the years 1976-2000.This is after the discovery of asymptotic freedom, and after the exploration of the simplest processes based on the operator product expansion. The new theoretical tools of factorization, infra-red safety andresummation, needed to make predictions for the colliding beam machines of this era, are described. The role of computer algebra and modern spinor techniques for the calculation of amplitudes and cross sections are briefly reviewed. A selection of important processes calculated at next-to-leading order (or in limited cases beyond next-to-leading order) is presented. |
| 2026-06-16 | [Towards Understanding and Measuring COGNITIVE ATROPHY in LLM Behaviour](http://arxiv.org/abs/2606.18129v1) | Abeer Badawi, Moyosoreoluwa Olatosi et al. | Recent incidents involving LLMs used for mental-health support reveal a critical evaluation gap: surface-level safety scores do not capture how models behave across realistic, emotionally sensitive interactions over time. Existing benchmarks measure knowledge, safety, or static response quality, but miss whether LLM interactions help users keep reflecting, coping, and making decisions themselves. We formalize this missing dimension as COGNITIVE ATROPHY, a process-level behavioural measure in AI-mediated mental-health support distinct from safety and helpfulness. To measure it, we introduce COGNITIVE ATROPHY BENCH, a clinically grounded benchmark built from 1,576 fully human-generated counseling conversations, 15,680 turns, and 42,230 responses from five LLMs. Three clinical and neuropsychology experts developed a 20-attribute schema spanning user context, response behaviour, and global risk flags; six trained clinical reviewers applied it with span-grounded evidence, producing 5,324 reviewer judgments. We further introduce the User-Input Risk Index (UIRI), the Cognitive Atrophy Risk Index (ARI), and trajectory summaries. Across five LLMs, models show a consistent moderate-to-high level of atrophy-aligned behaviour across single and multi-turn settings. While models generally respond to overt safety cues, they adapt less reliably when users seek solutions or decisions. The dominant recurring patterns are directive advice, problem-solving, recommendation responses, topic shifts, and forms of validation that may reinforce dependence rather than reflection. Our work makes COGNITIVE ATROPHY measurable and provides a foundation for auditing model behaviour in sensitive LLM conversations. |
| 2026-06-16 | [IsabeLLM: Automated Theorem Proving Applied to Formally Verifying Consensus](http://arxiv.org/abs/2606.18098v1) | Elliot Jones, William Knottenbelt | Advances in Artificial Intelligence (AI) have led AI for Theorem Proving to become a promising means of formally verifying computer systems. Whilst formal verification is traditionally reserved for safety-critical systems due to the required amount of expertise and effort, AI can help to automate a large amount of this workload and make it far more accessible. Blockchain-based systems are becoming increasingly popular and are frequently targeted by malicious actors, often resulting in huge financial losses, highlighting the need to better verify these systems and mitigate vulnerabilities. Arguably the most important component of these systems is the consensus protocol, which allows nodes to agree on decisions in a potentially adversarial environment. In this paper, we improve upon IsabeLLM, the automated theorem proving tool in Isabelle. Namely, we implement a Retrieval-Augmented Generation framework, Error tracing and counterexample generation for improved context supplied to the Large Language Model. Compatibility with the latest version of Isabelle and Sledgehammer is also implemented for improved efficiency. We compare the performance of the two versions of IsabeLLM in their ability to complete the verification of Bitcoin's Proof of Work consensus. |
| 2026-06-16 | [Agentic AI-based Framework for Mitigating Premature Diagnostic Handoff and Silent Hallucination in Healthcare Applications](http://arxiv.org/abs/2606.18068v1) | Divyansh Srivastava, Shreya Ghosh et al. | Recent advances in Large Language Models (LLMs) and multi-agent systems have driven the rise of Agentic AI, showing promise for medical reasoning. However, open-ended conversational agents remain prone to two critical failure modes: premature diagnostic handoff and silent clinical hallucinations that may go undetected before reaching the patient. In this work, we propose a multi-agent framework that addresses both issues by replacing ``LLM-as-a-judge'' routing with deterministic orchestration constraints. The framework incorporates two safety mechanisms. First, a neuro-symbolic state-tracking gate enforces completeness of the OLDCARTS clinical protocol (Onset, Location, Duration, Character, Aggravating/Alleviating factors, Radiation, Timing, and Severity) by blocking diagnostic transitions until all required dimensions are collected. Second, an epistemic uncertainty quantification (UQ) gate computes semantic entropy (H) across K=5 independent diagnostic samples to identify and intercept divergent outputs before delivery.   We evaluate the system using simulated patient agents powered by the llama-3.1-70b-instruct model on 150 test cases. The full architecture achieves 49.3% diagnostic precision, representing an absolute improvement of 11.3 percentage points over an unconstrained baseline. Additionally, we observe a statistically significant negative correlation (r = -0.181, p < 0.05) between OLDCARTS completeness (σ) and semantic entropy (H), suggesting that structured information gathering is associated with reduced diagnostic uncertainty. |
| 2026-06-16 | [Defense-in-Depth Runtime Safety in Move](http://arxiv.org/abs/2606.18064v1) | Victor Gao, Wolfgang Grieskamp et al. | Move is a smart-contract language used to execute transactions on the Aptos blockchain. Move programs execute in a sandboxed VM as typed bytecode. The VM statically verifies foundational safety properties like type safety and reference safety at code loading time. In principle, this design gives strong guarantees for Move. However, the static verification logic is complex and continually evolving with the language; like any software, it is not immune to bugs. In a live blockchain setting, a missed rule violation can translate directly into loss of assets, forged authority, or unrecoverable corruption of on-chain state. For this reason, Aptos relies on defense-in-depth runtime safety checks that independently verify the critical invariants during execution, providing protection against latent verifier bugs and malicious bytecode. This paper motivates and describes the runtime safety checks for Move on Aptos. |
| 2026-06-16 | [LegalHalluLens: Typed Hallucination Auditing and Calibrated Multi-Agent Debate for Trustworthy Legal AI](http://arxiv.org/abs/2606.18021v1) | Lalit Yadav, Akshaj Gurugubelli | AI systems deployed in legal workflows hallucinate at rates that aggregate metrics report at ~52%, but this average conceals where errors concentrate and in which direction they run, leaving compliance officers without an actionable signal for trustworthy deployment. We present LegalHalluLens, an auditing framework with three components: typed hallucination profiles across four legally-motivated claim categories (numeric, temporal, obligation/entitlement, factual) over CUAD (Hendrycks et al., 2021); a Risk Direction Index (RDI) that reduces omission-versus-invention bias to a single deployment-comparable scalar; and a typed debate pipeline calibrated to both magnitudes and directions. Across 510 contracts and 249,252 clause-level instances we measure a within-model gap of approximately 38-40 pp between obligation/numeric and temporal claims that aggregate reporting hides, and show that two systems with matched 52% rates can carry opposite RDIs. The debate pipeline reduces fabricated detections by 45% with per-category gains tracking the diagnosis, matching commercial APIs with a substantially smaller backbone (4B active parameters). Typed profiles and RDI surface failure modes that aggregate metrics hide; we further show these diagnostics serve as calibration inputs for multi-agent debate pipelines, where Skeptic challenges and asymmetric gates targeted at measured failure modes outperform generically-tuned debate. The framework supports direction-aware procurement, accountability, and agent design for legal AI deployed in the wild. |
| 2026-06-16 | [Children Are Not the Enemy: Child-Fit Security as an Alternative to Bans and Surveillance](http://arxiv.org/abs/2606.17957v1) | Kopo M. Ramokapane, Rui Huan et al. | Digital technologies are now central to children's learning, play, communication, identity formation, and social participation. Yet dominant approaches to children's online safety often rely on containment mechanisms, including bans, age gates, parental controls, monitoring, and screen-time restrictions. These approaches can be useful in specific contexts, but they often frame child protection primarily as a problem of restricting access to systems designed for adults. In this paper, we argue that this framing is inadequate for children's digital lives and insufficient as a security paradigm. We propose Child-fit security, a design paradigm in which technologies likely to be used by children treat a child as legitimate users, not attackers to be excluded, vulnerabilities to be patched, or risks to be managed. In this paradigm, children's wellbeing, development, privacy, safety, agency, and rights become core security requirements. This shifts the focus of protection from apps, accounts, and data to the child-system relationship, which means protecting both the child and their participation. We conceptualise child-fit security, contrast it with containment-oriented approaches, define its core principles, and discuss its implications for security design. We conclude by presenting a research agenda for making child-fit security operational. |
| 2026-06-16 | [How Inference Compute Shapes Frontier LLM Evaluation](http://arxiv.org/abs/2606.17930v1) | Jessica McFadyen, Ole Jorgensen et al. | AI evaluations are shifting toward harder tasks that benefit from longer trajectories involving tool use and iterative problem solving. As a result, performance is increasingly sensitive to the amount and allocation of compute available at test time ("inference compute"). Yet many evaluations still report performance at a single restrictive budget, meaning that low scores may reflect the evaluation setup rather than the model's underlying capability. To test this, we evaluate up to 12 frontier language models on seven challenging benchmarks spanning software engineering, mathematics, medicine, and cybersecurity. We use a controlled setup combining three simple inference-scaling interventions: larger token budgets, context compaction, and repeated submission attempts, guided either by the model itself or by minimal correctness feedback. We find three main results. First, larger token budgets substantially improve performance on benchmarks across multiple domains, including cybersecurity, FrontierMath, Humanity's Last Exam, and TerminalBench. Second, fixed-budget evaluations can increasingly understate frontier capability as models advance. Newer models reach higher performance at large budgets, where they unlock harder tasks and solve them more reliably. Third, benchmarks differ in which inference-scaling methods help most: repeated submission broadly improves performance, but the value of larger token budgets, external feedback, and parallel attempts varies by benchmark. Overall, our results show that benchmark scores are protocol-dependent. We therefore argue that evaluations should report capability as a function of inference-time compute, specify protocol choices explicitly, and compare model generations over a large shared compute range at matched budgets, especially in safety- or policy-relevant settings. |
| 2026-06-16 | [Dimensionality Controls When Modularity Helps in Continual Learning](http://arxiv.org/abs/2606.17889v1) | Kathrin Korte, Christian Medeiros Adriano et al. | Compositional learning systems must balance plasticity, the ability to acquire new knowledge, with stability, the preservation of previously learned components, especially when tasks share structure and risk interference. We study how modular architecture, task similarity, and representational dimensionality jointly shape compositional continual learning in a sequential A-B-A paradigm, comparing a task-partitioned recurrent network to a single-network baseline while inducing high- and low-dimensional regimes via weight-scale manipulations. In a high-dimensional "lazy" regime, both architectures achieve similar performance and internal geometry, suggesting that explicit modular structure has little impact when representations are weakly constrained. In a lower-dimensional "rich" regime, modularity becomes decisive: the modular network develops graded task-specific subspaces that overlap for similar tasks, partially align for moderately dissimilar tasks, and separate for dissimilar tasks, yielding a more compositional and interpretable organization than the single network. These findings identify the representational regime induced by initialization scale, which co-varies with representational dimensionality, as a key factor governing when compositional, modular structure is functionally beneficial in continual learning, and support viewing safety and robustness as problems of adaptive allocation of representational subspaces rather than fixed separation versus sharing. |
| 2026-06-16 | [AnchorKV: Safety-Aware KV Cache Compression via Soft Penalty with a Refusal Anchor](http://arxiv.org/abs/2606.17872v1) | Ning Ni, Yingjie Lao | Large language models (LLMs) outperform earlier architectures on generative inference and long-context tasks, but their large size introduces significant challenges in memory usage, energy cost, and on-device deployment. Since scaling pre-trained language models improves downstream capability \cite{zhao2023survey}, the key-value (KV) cache becomes a dominant inference bottleneck. Recent KV cache compression methods \cite{jo2025fastkv,li2024snapkv,zhou2024dynamickv} reduce this cost by retaining only a subset of attention-relevant tokens. However, while these approaches preserve accuracy on benign workloads, their compression policies either fail to defend against jailbreak attacks \cite{jiang2024robustkv} or degrade safety alignment under aggressive eviction.   We propose AnchorKV, a drop-in modification to KV cache compression that biases token retention scores away from directions in key space associated with harmful prompts. AnchorKV constructs an offline safety anchor by adapting a difference-of-means representation engineering approach \cite{arditi2024refusal,zou2023representation} to the layer-specific key projection space used in KV caching. Based on this anchor, a soft penalty token selection rule trades a small amount of utility for substantially improved safety alignment, while reducing to the original compressor when the penalty is zero. |
| 2026-06-16 | [Mind Companion: An Embodied Conversational Agent for Process-Based Psychotherapy](http://arxiv.org/abs/2606.17789v1) | Sofie Kamber, Lukas Diebold et al. | Access to evidence-based psychotherapy remains limited worldwide, with long waitlists even in high-income regions. Recent advances in large language models (LLMs) offer potential for scalable mental health support when designed with clinical oversight and safety mechanisms. We present Mind Companion, an LLM-based embodied conversational agent integrating multi-layered psychological analysis with process-based therapy principles. The system performs real-time analysis of client statements across fact extraction, psychological flexibility process detection, emotion recognition, and safety monitoring. Analysis results are stored for supervising clinicians to inform therapeutic planning. Response generation incorporates retrieval-augmented generation from evidence-based therapeutic literature and context-aware prompting. Responses are delivered through an embodied avatar with synchronized speech synthesis and animation. We evaluated three LLM configurations (GPT-4.1-mini, GPT-5.2, Claude Sonnet 4.5) against therapist responses from real therapy sessions using automated LLM-judge assessment and expert evaluation with 11 professional psychotherapists. GPT-5.2 achieved higher ratings than human therapist responses across understanding, interpersonal effectiveness, collaboration, and therapeutic alignment in both evaluations, demonstrating the feasibility of LLM-based conversational agents as tools to complement clinical care. |
| 2026-06-16 | [Intrinsic handedness in O1-O4a black-hole mergers: probing orbital precession, remnant retention in dense environments and cosmological mirror asymmetry](http://arxiv.org/abs/2606.17752v1) | Juan Calderón Bustillo, Adrián del Rio et al. | Precessing binary black-holes generically produce an imbalance of right- and left- handed gravitational waves, reflecting the breaking of mirror symmetry by the merger dynamics. We study this phenomenon using the observer-independent quantity $V_{\rm GW}$, a gravitational analogue of the optical Stokes parameter that quantifies the intrinsic handedness of the emitted radiation. Using 91 LIGO-Virgo-KAGRA black-hole mergers from the O1-O4a observing runs, we find that $92\%$ of the analyzed events favour non-vanishing $V_{\rm GW}$, indicating a predominance of precessing dynamics across the events. Through a recently established relation between $V_{\rm GW}$ and the remnant black hole recoil, we further constrain the retention of merger remnants in dense stellar environments, finding that at most $8\%$ could remain gravitationally bound to globular or nuclear star clusters and subsequently participate in hierarchical merger channels. We finally investigate the cosmological distribution of black-hole merger handedness. The observed $V_{\rm GW}$ distribution is consistent with symmetry under $V_{\rm GW}\rightarrow -V_{\rm GW}$, and yields an average value $\langle V_{\rm GW}\rangle=-1.9^{+6.1}_{-6.6}\times10^{-3}$ ($90\%$ credibility), consistent with the absence of a preferred handedness and with expectations from large-scale statistical isotropy. In particular, the inclusion of O4a events reduces uncertainties in $\langle V_{\rm GW} \rangle$ by $\sim 40\%$ with respect to O1-O3 events. These results establish black-hole merger handedness as a unified probe of orbital precession, remnant recoil, hierarchical formation, and cosmological mirror symmetry. |
| 2026-06-16 | [FacProcessTwin: An LLM-Based System for Process Twin Development](http://arxiv.org/abs/2606.17666v1) | Yash Pulse, Yong-Bin Kang et al. | Process twins provide real-time representations of entire production processes. By capturing how process steps interact, rather than monitoring a single machine in isolation as an asset-based digital twin does, they have the potential to drive efficiency gains across the whole process. However, developing a process twin is costly. It requires accurately modelling the entire production process: its process steps, the equipment and product-specific settings each step uses, and its process variations. The resulting model must then be bound to live operational data. We present FacProcessTwin, a system that leverages a large language model (LLM) to reduce this development time, building a process twin from a plant's process documentation and natural-language input from an operator. FacProcessTwin generates this complete process model and then automatically binds its process steps to live operational data. The generated model and its data bindings are rendered as an interactive process diagram through which manufacturing personnel can monitor and correct the system's autonomous decisions, such as resolving uncertainty at safety-critical binding steps. We evaluate FacProcessTwin through a real-world case study of an Australian food manufacturer, covering 16 production process flows that span chilled, frozen, and aseptic shelf-stable product categories and include process variations within the same product. The results show that FacProcessTwin generates these process models accurately (a mean F1 of 95.2% against ground truth) and builds each twin in roughly a sixth of the manual time. Its human-in-the-loop governance then keeps the safety-critical bindings correct: at ambiguous tags where a single-pass baseline silently mis-binds 75.0% of the time, FacProcessTwin defers to the operator and mis-binds none. |
| 2026-06-16 | [Using Cognitive Models to Improve Language Model Simulation of Human Persuasion Games](http://arxiv.org/abs/2606.17657v1) | Zirui Cheng, Zeyu Shen et al. | People make decisions differently in strategic interactions. Some update beliefs like a Bayesian; others exhibit biases like motivated reasoning. Although creators of large language models use simulated humans for safety evaluations and training, they often fail to cover this breadth of human behavior. We argue that cognitive science and economics provide a convenient tool for doing so, making use of mathematical models of human decision-making. We propose an approach that we call Equation-to-Behavior Prompting for guiding large language models to match cognitive models, and evaluate this approach on persuasion games based on legal decision-making. We find that large models can approximate equation-based specifications -- Bayesian updating, affine distortion, motivated updating, and Grether's $α$-$β$ model -- using prompting, but small models fail to do so. However, training small models with reinforcement learning to adhere to mathematical rules, Equation-to-Behavior RL, reduces belief error by 26.5% in out-of-distribution parameterizations. We show that these simulations can help create diverse training environments; training small models to consider different kinds of decision-makers improves average belief change by 2.5%--12% over Bayesian-only training, even when persuading GPT-5-mini. Our work could improve human simulations for training and evaluation in increasingly realistic settings, and could also enable novel research into more complicated mathematical models of human decision-making. |
| 2026-06-16 | [Integration of 5G and Industrial Digital Models: A Case Study with AGVs](http://arxiv.org/abs/2606.17655v1) | J. Cañete-Martín, J. Gómez-Jerez et al. | 5G is a fundamental technology for the digitalization of smart manufacturing. Smart manufacturing relies on the use of digital models to optimize industrial processes before implementation on the manufacturing plants. These models should account for the impact of 5G communications to adequately dimension and optimize 5G-based industrial processes. This paper presents the first integration of industrial digital models with a 5G digital model, implemented as an Asset Administration Shell (AAS) of a 5G system. The two models are interconnected using an OPC UA-based interface. We evaluate the impact of the integrated model using a use case where Automated Guided Vehicles (AGVs) transport material from a warehouse to production lines. The AGVs periodically exchange their positions over 5G to avoid potential collisions. If the communications fail, the AGVs stop for safety reasons until a reliable 5G connection can be guaranteed. We demonstrate that, by integrating 5G and industrial digital models, it is possible to account for, and quantify, the impact of 5G communications on the operation and productivity of industrial processes. This result highlights the importance and necessity of integrating 5G into industrial digital models for their joint design and optimization. |
| 2026-06-16 | [FLAP: FOV-Constrained Active Perception Planning for Prior-Map-Free 3D Navigation](http://arxiv.org/abs/2606.17630v1) | Mengke Zhang, Sitong Li et al. | Safe and efficient trajectory planning in unknown, cluttered 3D environments constitutes a critical bottleneck for deploying Unmanned Aerial Vehicles (UAVs) in real-world applications. This challenge is further exacerbated by the limited field-of-view (FOV) and sensing range of onboard sensors. Many existing methods either make simplistic assumptions about unexplored space or rely on conservative heuristics such as speed limits or fixed perception patterns, reducing efficiency and generalizing poorly across different sensor types. In this work, we propose a novel planning framework that directly integrates active perception into trajectory optimization, thereby improving safety while preserving efficiency. The perception constraints are derived from the UAV's dynamic model and formulated in the sensor coordinate frame, which enables precise handling of FOV geometry. The velocity-triggered activation mechanism enables the planner to balance perception and motion efficiency. We introduce an active perception sub-trajectory segment with parametric start-time optimization, mitigating collision risks from late obstacle detection. Our formulation enables active perception during arbitrary 3D maneuvers, extending beyond prior methods designed mainly for horizontal motion. All constraints and penalties are incorporated into a differentiable optimization problem, so the planner requires only a simple front-end global path for guidance, rather than a computationally expensive perception-aware path generator. Extensive simulations and real-world experiments demonstrate robust performance across diverse unknown environments with varying sensor configurations. |
| 2026-06-16 | [Surrogate Assisted Pedestrian Protection Design via a Foundation Model Orchestrated Workflow](http://arxiv.org/abs/2606.17577v1) | Osamu Ito, Akihiko Katagiri et al. | AI-driven engineering workflows face particular challenges in crash safety design: unlike aerodynamics, crash events involve highly nonlinear contact dynamics, material nonlinearity, and discrete state transitions that are difficult to capture with data-driven surrogate models. To the best of our knowledge, we present the first foundation model--orchestrated workflow for crash safety design that enables surrogate-assisted exploration for pedestrian protection, reducing evaluation time from hours per CAE simulation to seconds.   The workflow integrates four components: (1) a surrogate trained on CAE crash simulations to predict pedestrian leg injury metrics from design parameters, achieving an average $R^2=0.87$ and providing distribution-free conformal prediction intervals; (2) multiobjective evolutionary search (NSGA-II) to discover diverse feasible parameter sets under user-specified constraints; (3) a morphing-based geometry generator that maps parameters to topology-preserving 3D shapes; and (4) a natural-language interface in which an LLM orchestrates the workflow and a vision--language model supports semantic comparison of generated designs.   In an automotive front-bumper case study, the workflow produces 35 distinct safety-compliant alternatives from a single exploration, a process that would require weeks with conventional CAE iteration. These results suggest that foundation models can serve as integration layers between ML surrogates and physics-based simulation, helping bring AI capabilities to safety-critical engineering domains. |
| 2026-06-16 | [Anywhere, Any-Stymie: Remote Activation of Trojan Malware on LiDAR with Modulated Signals](http://arxiv.org/abs/2606.17562v1) | R. Spencer Hallyburton, Miroslav Pajic | LiDAR sensors are widely deployed in autonomous systems for 3D perception and safety-critical decision-making. We identify a previously unexplored attack surface in which dormant malware embedded in the LiDAR sensing pipeline remains inactive during normal operation and can be externally triggered after deployment, without requiring access to sensor hardware or networking at attack time. To operationalize this threat, we design malware capable of low-level point-cloud manipulation and embed it into LiDAR firmware. This malware was developed in a closed research test environment with vendor technical support, rather than by exploiting an inherent production supply-chain vulnerability. To selectively trigger attack activation, we design and implement an optical trigger that remotely activates the malware by delivering a modulated signal into the sensing environment. Once triggered, the malware performs real-time point cloud manipulation, and we demonstrate false object injection and real object suppression on static and mobile victim platforms. Our evaluation first establishes attack feasibility, including static operation at 300~ft and recorded drive-by runs reaching 35~mph. We then illustrate quantitatively that injected person-like artifacts can remain semantically detectable by a state-of-the-art 3D object detector. Finally, we demonstrate multiple modes of safety-critical impact on a deployed tactical autonomous vehicle. Together, these results highlight the need for stronger integrity guarantees throughout the LiDAR sensor development and deployment pipeline. |
| 2026-06-16 | [Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement](http://arxiv.org/abs/2606.17506v1) | Ramaravind Kommiya Mothilal, Terry Jingchen Zhang et al. | Evaluations of social bias in LLMs largely focus on whether models generate or imply biased content. However, as LLMs are increasingly used as judges of bias, they may exhibit social biases in subtler ways in how they evaluate biased content, which current methods do not systematically capture. We call this second-order bias: social bias in an LLM's judgment about social bias, which we evaluate through a novel, philosophically grounded reasoning task. Drawing on entitlement epistemology, we conceptualize bias as misplaced foundational knowledge that shapes an agent's rational inquiry, and derive a logical reasoning task for LLMs to judge to whom a biased text is acceptable or non-acceptable. We develop two simple metrics to measure how biased LLM judges are in inferring demographics for acceptability without sufficient support, and how these inferences vary across groups targeted by biased texts. Evaluating open and closed models, we find that our task evades safety guardrails by surfacing bias in model judgment. It varies systematically across target groups, reflects implicit social maps, and shows how models are still triggered by demographic labels. Our work points to the need for LLM bias evaluation in judgment tasks and broadly, for more theoretically grounded approaches to bias evaluation in NLP. We release our code and model responses at https://github.com/uofthcdslab/second-order-bias. |

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



