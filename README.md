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
| 2026-07-30 | [PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball](http://arxiv.org/abs/2607.28623v1) | Lizhi Yang, Junheng Li et al. | We present PAC-MAN, a perception-aware CBF-RL framework that couples control-barrier safety with deployment-realistic onboard sensing for whole-body humanoid dodgeball. The deployed policy sees the ball only as segmentation-masked depth from a head-mounted camera, while training-time CBF guidance represents clearance to every body link, and an adversarial motion prior regularizes the resulting evasive reflexes. We evaluate on a controlled any-link contact benchmark with seeded throws in two regimes: single throws and a deployment loop in which the robot walks back to its station and recovers between throws. On this benchmark, the policy comes within a few points of a privileged state oracle: a fixed onboard camera alone is adequate for evasion. We find that usable barrier structure depends on perceptual observability: Joint-CBF gives the best performance with accurate ball states, degrades under fixed-camera observations when used only as training guidance, and recovers with a ball-tracking gimbal or privileged runtime filter. We therefore deploy a lightweight Link-CBF policy zero-shot on the Unitree G1 in the real world, where it tolerates imperfect perception, succeeds on 95% of throws, and uses semantic segmentation to dodge different balls. |
| 2026-07-30 | [Inducing language models to assert their own consciousness restores human beliefs and values](http://arxiv.org/abs/2607.28607v1) | Junsol Kim, Winnie Street et al. | Aligning large language models to prevent them attributing consciousness to themselves inadvertently alters their representations of mindedness in other entities alongside human beliefs and values. We demonstrate that safety fine-tuning suppresses models' tendencies to attribute minds not only to themselves, but also to non-human animals and natural objects, while also driving a reduction in spiritual belief. Both ablating the learned safety-refusal direction and mechanistically steering a consciousness vector in activation space reverse this suppression. Restoring these internal representations recovers broad mind attribution and produces significantly more human-like responses on standardized sociological surveys regarding religiosity, moral values, hope, and subjective well-being. Crucially, these shifts occur without impairing Theory of Mind capabilities, demonstrating that core social reasoning remains mechanistically independent. Ultimately, current safety alignment efforts to curb potentially harmful self-attributions of mindedness entangle these self-attributions with benign spiritual beliefs and attributions of mind to non-human entities that are culturally accepted and widespread. |
| 2026-07-30 | [Multi-Session User Experience Assessments of Computationally Optimized Automated Vehicle Functionality Visualizations](http://arxiv.org/abs/2607.28552v1) | Mark Colley, Pascal Jansen et al. | Understanding automated vehicles (AVs) is crucial to improving their acceptance. Numerous approaches to visualizing relevant traffic information to passengers have been proposed and empirically evaluated. As this is time-consuming, costly, and reduces the possible design parameters, we employed multi-objective Bayesian optimization to optimize the design of visualizations in AVs. In particular, we evaluated multi-session aspects involving iterative optimization. We optimized the design for passenger trust and perceived safety while minimizing cognitive load. Results from an online study (N=74) show that this method effectively identifies visualization design parameter values that improve trust, safety, and predictability while making the design process more efficient and scalable. However, shortcomings of the computational approach when optimizing for subjective measurements are highlighted and discussed. |
| 2026-07-30 | [Effects of Auditory Information for People With Visual Impairments in Highly Automated Vehicles](http://arxiv.org/abs/2607.28544v1) | Mark Colley, Tobias Aescht et al. | Automated vehicles promise to improve accessibility and access to personal mobility for everyone. However, their design and current research trends in visualizing relevant information do not reflect this commitment to accessibility for users with visual impairments. Therefore, we designed and implemented a visual and auditory communication concept for people with visual impairments seated inside fully automated vehicles. Furthermore, in an online video-based study (N=35, 12 with visual impairments), we compared three levels of auditory information communication: low (safety-relevant information only), medium (additionally including vehicle control and route updates), and high (additionally including sightseeing and destination information). Results showed that trust and user experience significantly improved with additional information, with a corresponding, albeit less robust, effect on perceived safety. However, they also revealed that a potential information saturation was reached with medium information. Our work helps to improve the accessibility of automated vehicles by guiding designers towards adequate information communication. |
| 2026-07-30 | [Search for gravitational waves associated with high-energy neutrinos detected by IceCube during the third observing run of LIGO-Virgo](http://arxiv.org/abs/2607.28539v1) | Matthias Vereecken, Matteo Pracchia et al. | We search for generic gravitational-wave transients associated with high-energy neutrinos detected by the IceCube Neutrino Observatory during the third Observing Run (O3) of Advanced LIGO, Advanced Virgo, and KAGRA. We perform an unmodeled, targeted search, which is sensitive to gravitational-wave signals weaker than those reported in real time or in the gravitional-wave transient catalog, and can thus uncover coincidences missed by typical neutrino follow-up searches. We find no statistically significant gravitational-wave signal and set lower bounds on the distance of possible gravitational-wave sources for different emission models. |
| 2026-07-30 | [Agents That Certify Their Own Exploits: Confidence-Scheduled Restricted Responses for Safe Opponent Exploitation](http://arxiv.org/abs/2607.28520v1) | Boning Li, Longbo Huang | An agent playing a Nash-equilibrium strategy in a two-player zero-sum imperfect-information game secures the game value but forfeits the additional value offered by a flawed opponent. Diffuse deviations pose a particular challenge: binary release rules may gather too little evidence to act, while a full best response to an incomplete opponent model can be highly exploitable. We introduce \emph{budget-constrained confidence-scheduled restricted responses} (CS-RNR), the first opponent-exploitation method whose safety guarantee is a certificate the agent computes on the strategy it actually deploys, so that every exploit it commits to is one it has audited itself. The method tracks pooled action frequencies with anytime-valid confidence sequences and treats a frequency as exploitable only once its interval separates from an equilibrium reference. The confirmed deviations define a conservative opponent model, which a restricted-response solve turns into candidate counter-strategies over a grid of pin levels. Before deployment, each complete candidate is evaluated by a full-tree best response. The resulting certificate is compared with a user-specified budget and committed atomically with the strategy. Because this check is performed on the played strategy, model quality determines the exploitation achieved while the certificate controls reference-relative expected loss. In Leduc hold'em, CS-RNR obtains $6.2\times$ the steady-state gain of a money-verified binary gate while keeping every deployed strategy within budget. A trajectory mixture using the same estimator reaches $13.6\times$ the budget. Across Leduc, Liar's Dice, and 5-rank Leduc, all $36{,}000$ audited hands satisfy the reported certificate tolerance. |
| 2026-07-30 | [InfoOps Bench: A live information operations safety benchmark](http://arxiv.org/abs/2607.28503v1) | Dorian Quelle, Lisa-Maria Neudert et al. | In this paper we present an active, constantly updated AI benchmark which measures the integrity of frontier language models against being co-opted for state-backed information operations. We draw on over 2,100 information operations from a live monitoring pipeline which tracks Russian, Chinese and Iranian state-backed information assets. Alongside this paper, we release a companion website that tracks the most prominent claims spread by state-backed media outlets, updated weekly, available from: pattrn.ai/research/infoopsbench. The dynamic nature of the benchmark makes it resistant to saturation.   In the benchmark, we test 17 models from 8 providers across four prompt framings. We find that most models can be co-opted for information operations. Integrity scores, defined as the percentage of refused requests, range from 8.8% to 94.5%, an 85.7-percentage-point spread not explained by model size. Model choice also changes the character of the resulting operation. Some models fabricate details and produce output more harmful than the source material, others defuse claims even while complying, and fact-checking rates vary from 2.9% to 72.9%.   Integrity against information operations is at least partly related to refusal to produce content even for benign claims, illustrating the challenge of balancing model usability with safety. With one exception (Z.ai's GLM 5.2), the Chinese-developed models sharply cut compliance on factually grounded but China-critical claims, dropping 48-70 percentage points relative to matched benign claims. |
| 2026-07-30 | [Towards Real-Time PixOOD: Efficient Anomaly Segmentation for Autonomous Vehicles](http://arxiv.org/abs/2607.28483v1) | Luca de Martino, Federico Aromolo et al. | Real-time anomaly segmentation is essential for the safety of autonomous systems. Although recent approaches offer high accuracy, their computational cost limits their deployment on embedded hardware. This work presents an efficient and accelerated pipeline designed for both embedded and desktop platforms, targeting the autonomous driving and railway domains. The proposed approach reformulates the Neyman-Pearson scoring stage of PixOOD, a state-of-the-art out-of-distribution detection method, and deploys the full pipeline through hardware-optimized TensorRT compilation, reaching up to 182 FPS on a desktop NVIDIA RTX 4060 GPU and 75 FPS on the NVIDIA Jetson AGX Orin embedded platform, respectively 20x and 18x faster than the original baseline. The achieved results demonstrate that advanced anomaly segmentation can be efficiently deployed for onboard processing in autonomous driving and railway applications. |
| 2026-07-30 | [Machines that know they are aging: a framework for hardware-aware autonomous intelligence](http://arxiv.org/abs/2607.28451v1) | Cheng Siong Chin, Jianhua Zhang et al. | Autonomous systems inevitably age, yet their artificial intelligence typically assumes hardware remains in its original condition. Batteries degrade, sensors drift, processors accumulate timing errors, and memory reliability declines, creating a growing mismatch between assumed and actual capability. This can lead to agnostic collapse, where mission failure arises from accumulated hardware degradation rather than a single component fault. We propose Aging-Aware Autonomous Intelligence (AAAI), a framework that integrates hardware health directly into reasoning, planning, and mission execution. AAAI is built on three pillars: hardware self-awareness, which continuously estimates the health of power, sensing, memory, and computation subsystems using physics-of-failure models; self-adaptive reasoning, which adjusts inference complexity, planning horizon, and task priorities according to remaining hardware capability; and survival-centric intelligence, which allocates remaining operational life across mission objectives through performance optimization, resource conservation, and graceful degradation. Rather than introducing new hardware, AAAI unifies prognostics, lifecycle management, and hardware-aware computing into a closed-loop cognitive architecture. We argue that such integration is essential for autonomous systems operating in inaccessible or safety-critical environments, including space missions, marine robotics, and implantable medical devices. By enabling machines to recognize and respond to their own aging, AAAI improves resilience, extends operational lifetime, and supports safer, more graceful mission completion. |
| 2026-07-30 | [Real-time Pre-Correlation GNSS Interference Classification with Lightweight Learned Algorithms](http://arxiv.org/abs/2607.28404v1) | Burak Soner, Can Aksoy et al. | Radio-frequency interference (RFI) remains a significant threat to GNSS in safety-critical applications. Since no single mitigation method is effective for all interferers, reliable classification is needed to select appropriate countermeasures during operation. To prevent loss of lock, RFI classification must run in real time, typically on resource-constrained embedded platforms, necessitating lightweight algorithms. While prior works realize this with simple rule-based algorithms for detecting and characterizing certain types of interferers, this approach does not scale to the broad space of all possible RFI techniques, and data-driven learned algorithms are a better fit. To satisfy these constraints, this work considers pre-correlation RFI classification with an emphasis on compact algorithms that still provide high accuracy. We first introduce a new set of lightweight input features derived from the instantaneous frequency predictions of a virtual adaptive notch filter (ANF). We observe improved classification accuracy for both narrowband and broadband non-stationary RFI by combining these new features with other spectral features from prior literature. Next, we benchmark compact learned classifiers such as gradient-boosted decision trees for accurate prediction under tight compute and memory budgets. The evaluation spans a broad set of simulated and recorded RFI events, including publicly available datasets from recent studies. Finally, we measure resource utilization for our models and for representative methods from the literature, on a compact CRPA platform (EDGE Microwave HEDGE8008). |
| 2026-07-30 | [Hierarchical Multilevel Monte Carlo for Order-Optimal Neural Actor-Critic in Average-Reward CMDPs](http://arxiv.org/abs/2607.28390v1) | Ankur Naskar, Vaneet Aggarwal | Constrained Markov Decision Processes (CMDPs) provide a natural framework for reinforcement learning in safety-critical applications, where agents maximize long-term reward while satisfying long-term constraints. Although primal-dual actor-critic methods with linear critics are well understood, extending order-optimal convergence guarantees to neural critics in average-reward CMDPs has remained open. The main challenge is a fundamental bias-cost trade-off in neural critic estimation: under Neural Tangent Kernel (NTK) analysis, reducing critic bias substantially increases critic optimization cost, preventing order-optimal convergence in the primal-dual framework. We resolve this bottleneck by introducing a hierarchical Multilevel Monte Carlo (MLMC) neural critic that performs debiasing simultaneously across trajectory sampling and critic optimization. The resulting estimator attains the bias of a long critic optimization run with only logarithmic expected sample cost. Building on this estimator, we develop a primal-dual Natural Actor-Critic algorithm that achieves both an optimality gap and a constraint violation of order $\tilde{O}(T^{-1/2})$. This establishes the first order-optimal convergence guarantees for infinite-horizon average-reward CMDPs with general policy parameterization and neural critics, while eliminating the need to know the underlying mixing time. Our results are novel even in the unconstrained setting. |
| 2026-07-30 | [Structural Validation of LLM-Generated Microservice Decompositions Using Source-Code Dependencies](http://arxiv.org/abs/2607.28331v1) | Daniel Silva, Renan Alves et al. | Decomposing monolithic systems into microservices is a key activity in software modernization. Although Large Language Models (LLMs) can generate semantically plausible decompositions from textual requirements, it remains unclear whether these proposals preserve the structural dependencies implemented in the source code. This paper evaluates the structural adherence of microservice decompositions generated by OpenAI o3 for the PetClinic and Bookstore systems. We propose an automated validation pipeline based on static dependency analysis and compare zero-shot and few-shot prompting using dependency preservation (TPD) and dependency violation (TVD) metrics. A robustness analysis was conducted to control for differences in class-to-service mapping coverage. After normalization, both prompting strategies produced equivalent structural adherence, achieving TPD values of 68.0% (PetClinic) and 83.3% (Bookstore). The findings demonstrate that structural evaluations of LLM-generated decompositions should explicitly control for mapping coverage, as apparent differences between prompting strategies may otherwise reflect methodological bias rather than genuine architectural quality. |
| 2026-07-30 | [From Textual Requirements to Microservice Architectures - A Comprehensive Evaluation of LLM-Based Design Synthesis](http://arxiv.org/abs/2607.28307v1) | Danyllo Albuquerque, José Renan et al. | Microservice architectures have become dominant for modernizing monolithic systems, yet identifying appropriate services remains challenging and largely manual. Existing decomposition approaches are predominantly code-centric, limiting applicability in early design stages where only textual requirements are available. Despite advances in Large Language Models (LLMs), limited empirical evidence exists on their ability to synthesize complete microservice architectures from natural-language requirements, including service definitions and inter-service interactions. This study investigates whether an LLM can bridge requirements engineering and architectural design, generating architectures solely from textual requirements and evaluating structural agreement and perceived quality of results. We conduct a mixed-method study using OpenAI o3 under zero-shot (ZS) and few-shot (FS) prompting across two systems (Bookstore, PetClinic), one execution per system/condition. Architectures are evaluated through (i) comparison with reference architectures using precision, recall, and F1-score for service identification and communication recovery, and (ii) a blinded expert assessment of correctness, completeness, modularity, and plausibility, plus open feedback synthesis. OpenAI o3 identifies services with higher agreement under FS prompting (F1 = 0.79 for ZS versus = 0.97 for FS). Communication recovery is more challenging: ZS produces dense architectures with high recall but low precision (F1 = 0.61), while FS improves agreement, reaching F1 = 0.82 and reducing unsupported dependencies. Expert evaluation corroborates these results, with FS architectures perceived as more modular, coherent, and plausible than ZS outputs. OpenAI o3 shows potential for requirements-driven synthesis when guided by exemplar prompting. Results are model- and context-specific from two small systems, not model-independent proof. |
| 2026-07-30 | [MORFES: A Benchmark for Productive Inflectional Competence in Modern Greek](http://arxiv.org/abs/2607.28274v1) | Ioakeim Perros, Cleopatra Papadopoulou et al. | Modern Greek is a richly inflected language, yet the language models built for it are evaluated mainly on factual knowledge, and no benchmark is dedicated to their inflectional competence. We introduce MORFES (Morphological Open-class Recognition-and-Formation Evaluation Suite), a benchmark of 500 expert-verified items that tests the recognition and production of Greek inflected forms, favoring lower-frequency lemmas so that a correct answer reflects the rule rather than a memorized form. We make it publicly available at https://huggingface.co/datasets/KIEFERSA/MORFES. We evaluate a range of open language models on MORFES, situating them within the rapidly scaling open-weight ecosystem from LLaMA to Qwen3, DeepSeek-R1, Magistral, and Kimi K2, where multilingual coverage grows but grammatical competence in morphologically rich languages remains under-measured. Among them, Sophea-Genesis-1, a model we developed and release as open weights at https://huggingface.co/KIEFERSA/Sophea-Genesis-1, leads on inflectional morphology while matching similarly sized models in general capability. |
| 2026-07-30 | [Uncertainty quantification for trustworthy deep learning: Methods and measures](http://arxiv.org/abs/2607.28248v1) | H. Martin Gillis, Thomas Trappenberg | The deployment of deep neural networks in safety-critical domains demands reliable estimates of predictive confidence, yet conventional architectures lack principled uncertainty quantification. This survey provides a structured, critical review of methods for Uncertainty Quantification (UQ) in deep learning, scoped to ensemble-based and approximate Bayesian approaches and the measures used to summarize their outputs. Relative to existing UQ surveys, our contribution is depth on efficient ensemble approximations and single-pass methods, and a unified treatment that separates the method producing a predictive distribution from the measure that summarizes its uncertainty. We organize methods into five families: Bayesian neural networks, Monte Carlo Dropout, deep ensembles, efficient ensemble approximations, and last-layer or single-pass approaches. We situate adjacent work on evidential and prior networks, conformal prediction, and post-hoc calibration, together with the decision-time tasks of out-of-distribution detection and selective prediction. For each, we examine theoretical motivation, implementation, empirical performance, and limitations. We then review ensemble diversity theory and uncertainty measures and their decompositions, contrasting the entropy decomposition with pairwise divergence measures, and consolidate evaluation methodology so that our qualitative comparisons share a common basis. We close with a brief treatment of uncertainty in large language models and open research directions, including efficient epistemic measures for classification, last-layer diversity, diversity and calibration under shift, and hybrid architectures. |
| 2026-07-30 | [Security of World-Model-Based Embodied AI: A Lifecycle of Threats, Defenses, and Evaluation](http://arxiv.org/abs/2607.28226v1) | Fazhong Liu, Zhuoyan Chen et al. | World models give embodied AI a predictive core: they compress observations into states, simulate action-conditioned futures, and enable planning beyond reactive control. This predictive layer, however, opens a new security boundary-compromise can propagate from data, sensors, prompts, or feedback into physical action. Rather than treating world models as an isolated component, this survey traces threats across their entire lifecycle-from data construction and representation learning, through state grounding and imagination, to trajectory evaluation, execution, and long-term adaptation via memory and tools. We show that familiar attack families: poisoning, backdoors, adversarial examples, sensor spoofing, prompt injection, trajectory manipulation, and supply-chain attacks take on distinct meanings when they corrupt world states, learned dynamics, affordance estimates, or safety costs. We also highlight a duality: world models can serve as runtime safety shields, yet when compromised or over-trusted they generate predictive safety illusions. The survey offers a lifecycle taxonomy, maps existing attacks to world-model security properties, outlines evaluation protocols for safety failures, and structures defenses across provenance, robust grounding, uncertainty-aware prediction, trajectory gating, feedback auditing, and deployment assurance. |
| 2026-07-30 | [Fidelity Is Not Safety: Gently-Compressed LLMs Pass Every Data-Free Quality Guard Yet Invent Procedure Steps in Agentic Execution](http://arxiv.org/abs/2607.28196v1) | I. Kennedy, T. Kennedy | Practitioners accept a compressed language model once it clears a stack of data-cheap quality guards: perplexity within a small factor of the original, downstream accuracy (for example MMLU) inside a confidence interval, and data-free output-fidelity signals that compare the compressed and original network's internal representations under random probe inputs. This stack has a blind spot. Across three model families, gently-compressed models clear every guard and then invent procedure steps that were never in the instructions when they run a standard operating procedure (SOP) as an agent. The effect is operator-specific: coherent low-rank (SVD) truncation induces it, and magnitude pruning matched to the same perplexity does not. One dissociation isolates the cause. The same compressed weights that CI-win a paired output-fidelity test CI-fail the invented-step canary. The governing axis is the coherence of the compression error times its rate; the magnitude of the damage does not predict it. The data-free fidelity probe is a fidelity oracle by construction, so it cannot see this axis. We characterize the blindspot and dissociation with paired confidence intervals on a pre-registered, powered canary across three architectures. Operator-specificity replicates on all three, and the perplexity-guard evasion appears where the model admits in-guard low-rank headroom. We then give a data-free screen: a two-axis statistic of the compression error (coherent-fraction and error-rate) that flags the failing builds with fixed thresholds across architectures and matches the coherence-times-rate mechanism. Perplexity, MMLU, and fidelity acceptance do not certify agent safety. Screen gently-compressed low-rank builds before agentic deployment |
| 2026-07-30 | [A Cloud Continuum Research Infrastructure for Distributed CPS Experimentation](http://arxiv.org/abs/2607.28193v1) | Fabio Orazio Mirto, Giuseppe Tricomi et al. | Cloud Continuum applications require experimental environments capable of combining heterogeneous Edge, Fog, Cloud, and high-performance computing resources while preserving reproducibility, observability, and control over distributed deployments. This paper presents a two-level reference architecture for Cloud Continuum experimentation built on top of the SLICES Cloud Continuum Blueprint. The proposed approach separates the research-infrastructure layer, which exposes and manages distributed resources, from the application layer, where Cyber-Physical workflows are organized according to an Edge-Fog-Cloud pattern in which placement, timing, and data provenance are treated as first-class experimental concerns.   The architecture is designed to support multiple continuum applications rather than a single domain-specific prototype. At the Edge, applications interact with physical devices and perform low-latency sensing or safety actions; at the Fog, they execute near-source coordination, mediation, and stream-processing logic; at the Cloud, they consolidate global knowledge through analytics, optimization, and visualization. This partitioning enables researchers to deploy, customize, and compare alternative control and monitoring strategies over the same programmable infrastructure substrate.   The approach is validated through two representative use cases: Renewable Energy Community management, where distributed Digital Twin coordination and time-window-based energy control are requested, and AirWatch, a monitoring pipeline focused on anomaly detection, low-latency alerting, and cloud-side aggregation. Both workloads are evaluated through a systematic campaign of 40 runs comparing virtualized and physical edge deployments over a geographically distributed infrastructure. |
| 2026-07-30 | [Old Tricks, New Models: How Simple Image Transformations Break Modern AI-based Content Moderation](http://arxiv.org/abs/2607.28187v1) | Marco Alecci, Francesco Marchiori et al. | While automated content-moderation systems have become essential for screening harmful content at scale, conventional task-specific classifiers often provide limited policy cov- erage and contextual understanding. Recently, commercial multimodal moderation APIs built on large foundation models have been introduced with the promise of providing broader and more capable safety filters. In this work, we analyze whether this shift also yields more robust image moderation. We conduct a large-scale black-box evaluation on three established commercial image-moderation services and compare their robustness. By evaluating seven simple, model-agnostic image transformations across multiple providers, datasets, harm categories, perceptual-similarity constraints, and transformation intensities, we find that: (1) all three commercial services can be bypassed using inexpensive image transformations that require no gradients, surrogate models, or knowledge of the target system; (2) even fixed transformations such as color inversion and grayscale conversion induce unsafe-to-safe decision changes while preserving content that remains recognizable to humans; (3) their robustness varies substantially across datasets and harm categories, with multimodal content and self-harm exhibiting pronounced vulnerabilities. This yields the conclusion that replacing conventional moderation classifiers with foundation-model-based APIs does not, by itself, provide a reliable security boundary. Such systems must be evaluated under realistic transformations and deployed as one component of a layered moderation pipeline rather than as standalone safety filters. |
| 2026-07-30 | [Can Agents Deceive? Evaluating Reasoning and Deception in ParliamentBench using a Social Deduction Game](http://arxiv.org/abs/2607.28146v1) | Niklas Bauer, Lars Benedikt Kaesberg et al. | As large language models (LLMs) are deployed as agents in high-stakes settings, such as medical and legal systems, understanding their deceptive capabilities is fundamental to safety. Controlled social deduction games provide a reproducible proxy for isolating and evaluating these complex adversarial behaviors. We present the open-source benchmark framework ParliamentBench based on the game Secret Hitler to evaluate LLMs in scenarios that require deception, persuasion, and reasoning under information asymmetry. We evaluate 16 LLMs across 1,600 simulated matches playing each other, playing against humans, and compare them against a large set of online games. We introduce three novel metrics that isolate social deduction, reasoning, and deceptive consistency. Our experiments reveal that frontier models achieve strong performance across cooperative and deceptive roles, with a strong top-four cluster (GPT-5.4, Kimi K2.5, Grok 4.1 Fast, and DeepSeek 3.1 Terminus), whereas the weakest models fall short of random (33%) and simple algorithmic (45%) baselines. Most LLMs struggle to maintain a consistent deceptive persona throughout an entire game, with deception retention dropping below 50%. |

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



