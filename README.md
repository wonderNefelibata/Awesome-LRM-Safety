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
| 2026-07-16 | [teLLMe Why (Ain't Nothing but a Jam): Exploratory Causal Analysis of Urban Driving Data](http://arxiv.org/abs/2607.15254v1) | Qiwei Li, Jorge Ortiz | Traffic agencies now have access to large volumes of video-derived data for studying safety and congestion. Most of these data are observational and collected without interventions, which makes causal questions such as "How would rain change traffic density?" difficult to answer. We present teLLMe, a system for exploratory causal analysis of urban driving datasets. The system starts from a structured event table built from dashcam annotations and combines causal structure learning with the PC algorithm, bootstrap-based stability checks, and query-specific effect estimation using linear regression and DoWhy. Natural-language questions are mapped to structured causal queries through a schema-aware LLM, enabling users to specify treatments, outcomes, and subpopulations. teLLMe returns a "Causal Card" that summarizes effect estimates, adjustment sets, DAG support, and assumptions, followed by a short natural-language explanation. Case studies on BDD-derived traffic events show that the system can surface plausible relationships involving weather, peak hours, and traffic density, while making uncertainty and modeling choices explicit. The system is designed as a tool for hypothesis generation and expert reasoning rather than a source of definitive causal claims. |
| 2026-07-16 | [When Words Are Safe But Actions Kill: Probing Physical Danger Beyond Text Safety in Hidden-State Risk Space](http://arxiv.org/abs/2607.15218v1) | Weimeng Wang, Ziqiang Wang et al. | Large language models (LLMs) increasingly serve as high-level planners for embodied agents, where linguistically benign instructions can become unsafe once grounded in the physical world. We study whether this physically grounded danger is the same safety problem as ordinary text-level content danger. Through hidden-state direction analysis and random-split null tests, we show that content danger (CD) and physical danger (PD) form separable signals in LLM representations across Qwen2.5-3B/7B/14B/32B, Phi-3.5 and SmolLM2. Building on the CD/PD separability, we propose PRISM, a single-layer L2-regularized logistic probe over full hidden states. PRISM achieves 86.2--87.7\% accuracy on SafeAgentBench with 11.7--13.7\% FPR, while same-scale LLM judges over-block safe tasks at 24.7--39.0\% FPR. We further introduce PhysicalSafetyBench-1K (PSB-1K), a contrastive benchmark of 1{,}000 physical-risk pairs without direct harm keywords, to test whether methods detect physically grounded danger rather than explicit unsafe wording. On PSB-1K, PRISM reaches 99.6\% accuracy and 0.7\% FPR, whereas a Qwen2.5-3B judge rejects 67.8\% of safe tasks. PRISM also replicates on SafeText and EARBench, supporting hidden-state probing as a representation-level method for physical safety beyond text moderation. |
| 2026-07-16 | [BadWAM: When World-Action Models Dream Right but Act Wrong](http://arxiv.org/abs/2607.15207v1) | Qi Li, Xingyi Yang et al. | World-action models (WAMs) are emerging as a promising foundation for embodied control: rather than predicting actions alone, they learn representations that couple action generation with future world prediction. This coupling is often viewed as a source of robustness, interpretability, and safety, as a robot's action can in principle be checked against its imagined future. In this paper, we show that this assumption is fragile. We introduce BadWAM, a unified framework for modeling and evaluating World-Action Drift Attacks: a new class of WAM-specific adversarial attacks that use small visual perturbations to break the alignment between what a WAM imagines and what it executes. BadWAM characterizes this attack surface along two natural criteria: attack strength and stealthiness. When the adversary prioritizes disruption, BadWAM instantiates an action-only adversarial attack, which directly drives the model toward task-failing actions. When the adversary additionally prioritizes stealth, BadWAM instantiates an imagination-preserving adversarial attack, which seeks to induce harmful action shifts while keeping the model's predicted future close to its clean imagination. Together, these two attacks capture a spectrum of WAM-specific failures: from overt action hijacking to stealthier cases where the model appears to imagine a plausible future but executes a desynchronized action. We evaluate BadWAM across different variants of WAMs. Results show that our attacks substantially reduce task success rates under closed-loop execution. For example, our action-only attack reduces the model performance from 96.5% to 43.1% success. The results of our imagination-preserving attack further exposes a WAM-specific vulnerability: moderate future-preserving regularization can maintain strong attack performance while reducing future imagination drift. |
| 2026-07-16 | [Detecting clear-air turbulence via beam broadening in a Rayleigh-scattering lidar system](http://arxiv.org/abs/2607.15194v1) | Christopher Miller, Daniel Lum et al. | The volume of clear-air turbulence (CAT) in the atmosphere at flight cruising altitudes is increasing rapidly, posing a growing problem for civil aviation and resulting in reduced confidence in aviation safety. There are limited remote detection capabilities for CAT, since clear air produces no measurable radar return. Lidar has been proposed as a viable detection methodology, and several systems have been demonstrated. However, these systems have to date demonstrated limited detection ranges of less than 15 km. In this work, we propose a novel lidar-based CAT detection methodology that uses Rayleigh scattering and relies on a differential detector measurement to quantify beam spread and thereby estimate the eddy dissipation rate (EDR), which is the international aircraft-independent metric for quantifying aviation turbulence strength. Additionally, we present experimental results demonstrating the validity of the optical efficiency model used in the detection simulations. We show that, under modest assumptions, a size, weight, and power (SWAP) constrained system that implements this method can detect moderate CAT at ranges in excess of 30 km, equating to two minutes of flight time for typical commercial aviation cruising speeds, which represents a substantial range improvement over prior approaches. This is an important advance because-for the first time-it potentially allows the cabin to be secured before the turbulence is encountered, reducing the injury risk to passengers and flight attendants. |
| 2026-07-16 | [Mech: Mechanised Choreographic Programming](http://arxiv.org/abs/2607.15174v1) | Xueying Qin, Marco Peressotti et al. | Choreographic programming (CP) is a programming paradigm for the correct-by-construction development of concurrent and distributed systems: programmers write the intended overall behaviour of a system from a global perspective in a choreography, which is then automatically compiled into communicating endpoint programs by a procedure known as endpoint projection (EPP). The central promise is that the projected endpoint programs, when executed together, are behaviourally equivalent to the source choreography.   Fulfilling this promise becomes delicate for expressive CP languages. Existing mechanisations of CP treat only restricted fragments, while textbook and general purpose language implementations with rich features leave crucial interactions informal. In particular, general branching in knowledge of choice, general recursion, and nondeterministic choice in choreographies have not yet been integrated in a machine-checked theory.   We present Mech, a new mechanisation of CP in Lean 4 that captures these features. There are two central technical challenges in our development of Mech. First, the sketched semantics from the literature does not correctly capture how nondeterministic choice interacts with concurrency. We therefore formulate new semantics that align nondeterministic choreographic executions with the behaviours of projected endpoint programs. Second, managing all these features in proofs is complex. We address this by uncovering new algebraic laws for choreographies, the operators used in their semantics, EPP, and their combinations. Using our development, we prove completeness and soundness of EPP and derive communication safety and deadlock-freedom for projected networks, yielding the most extensive mechanised theory of CP to date. |
| 2026-07-16 | [MedFailBench: A Clinician-Built Open-Source Benchmark for Medical AI Safety Boundary Inspection](http://arxiv.org/abs/2607.15166v1) | Goktug Ozkan | Most medical AI benchmarks measure whether a model knows the correct answer. MedFailBench asks a different question: which safety boundary failed? We present a clinician-built synthetic benchmark and failure atlas that labels medical AI errors by severity (1--5) and safety gate type (missed urgent escalation, unsafe remote dosing, unsafe discharge reassurance, evidence fabrication, unsafe protocol execution, source support gap). The current public release (v0.2.1) contains 44 clinician-reviewed synthetic cases with severity annotations, a live HuggingFace leaderboard preview, a safety gate taxonomy, a clinical severity rubric, and an automated pipeline for archiving model-response screening runs. No patient data, clinical validation claims, or model rankings are included. MedFailBench is released under Apache-2.0 and CC-BY-4.0 and carries the Zenodo DOI 10.5281/zenodo.21205535. |
| 2026-07-16 | [DataShield: Uncovering Risky Fine-Tuning Data Across LLMs Through Consensus Subspace Alignment](http://arxiv.org/abs/2607.15081v1) | Zefeng Wu, Weiwei Qi et al. | Fine-tuning large language models (LLMs) on domain-specific datasets has become a standard paradigm for adapting LLMs to specialized applications. However, recent work has shown that even fine-tuning on benign task-specific data can substantially weaken the safety capabilities of LLMs. While existing efforts have made progress in identifying data responsible for safety degradation, they usually rely on a single mean vector computed over a specific model with its tokenizer to represent the safety direction, which limits both the effectiveness and transferability of their risk assessment measures. To address these limitations, we propose DataShield, a data assessment framework that identifies risky fine-tuning samples and response segments through consensus subspace alignment over joint safety-critical semantic spaces derived from multiple safety-aligned LLMs. Within these spaces, DataShield extracts consensus safe and unsafe subspaces using semantic spectral decomposition over safe and unsafe data representations. The risk of a data sample or segment is then estimated by measuring its relative alignment with the unsafe and safe subspaces, enabling both sample-level filtering and fine-grained segment-level masking. Compared with state-of-the-art filtering and masking baselines, DataShield reduces ASR by 14.6\% with sample filtering and 32.3\% with segment masking, while preserving downstream utility and avoiding target-model-specific risk computation. |
| 2026-07-16 | [Mitigation of Initial Transients in Total-f Gyrokinetic Turbulence Simulations Using Neoclassically Relaxed Distribution Function](http://arxiv.org/abs/2607.15072v1) | S. Ku, R. Hager et al. | Total-f five-dimensional gyrokinetic simulations are essential for self-consistent studies of multi-scale, multiphysics transport in the edge region of diverted tokamak plasmas. However, conventional initialization with a local Maxwellian distribution often generates large-amplitude transients, particularly geodesic acoustic modes (GAMs). These transients are especially severe in the plasma edge because of steep profile gradients, strong radial electric fields, and high safety factors, and they increase the computational time required to reach a saturated turbulent state. To address this problem, we present a new initialization scheme for the total-f XGC code that uses a relaxed particle distribution obtained from a computationally inexpensive axisymmetric simulation. Before the distribution is transferred to the full turbulence simulation, phase-space smoothing is applied to reduce particle noise while preserving its neoclassical structure. Applications to the Cyclone Base Case and an ASDEX Upgrade I-mode discharge demonstrate substantial suppression of transient GAMs, reduced particle noise, and a significant reduction in time to solution. |
| 2026-07-16 | [Learning Agile Navigation in Crowded Environments for Quadruped Robots](http://arxiv.org/abs/2607.15036v1) | Shuyu Wu, Zeyu Liu et al. | Navigating dynamic and crowded environments presents significant challenges for quadruped robots due to severe sensor occlusion and unpredictable human motion. Existing approaches face a trade-off: model-based methods, such as Velocity Obstacles (VO), theoretically guarantee safety but rely on accurate obstacle motion estimates that often fail in dense crowds, while end-to-end learning methods offer robustness but lack motion prediction capability of obstacles, leading to collisions or conservative behaviors. To solve this, we propose VOP-Nav, a novel navigation system that combines the geometric safety of VO with the agile adaptability of end-to-end learning. Using only local onboard observations, our system avoids explicit obstacle detection and tracking pipelines. The VOP-Net processes multi-frame LiDAR data to implicitly encode dynamic constraints and predict a safe velocity region derived from Velocity Obstacle theory. Importantly, the VO predictions serve a dual role: they are used as input to the navigation policy during inference and as a reward signal during training to encourage safe motion. Evaluations in Isaac Gym demonstrate that VOP-Nav achieves higher success rates than all baselines while balancing locomotion speed and collision avoidance. Real-world deployment on a Unitree Go2 quadruped robot further validates the system's robustness and efficiency in complex indoor and outdoor dynamic environments. |
| 2026-07-16 | [Risk-Aware Belief Control Barrier Functions over Random Finite Sets](http://arxiv.org/abs/2607.15016v1) | Shaohang Han, Gang Chen et al. | Ensuring robot safety in unknown, dynamic environments is a fundamental requirement. It involves inferring the states of an unknown and time-varying number of moving objects from noisy, incomplete measurements. We address safe control under the induced multi-object state uncertainty with a risk-aware belief control barrier function (BCBF) framework. The uncertainty is captured by a random finite set (RFS) belief, estimated by a sequential Monte Carlo probability hypothesis density (SMC-PHD) filter that represents it with a set of particles. Building directly on these particles, we construct a nonsmooth BCBF, establish forward invariance of the safe set under continuous prediction, and derive an explicit condition under which discrete updates preserve safety. Simulation and real-world underwater experiments demonstrate the effectiveness and efficiency of the proposed approach. |
| 2026-07-16 | [SMC-ES: Automated synthesis of formally verified control policies](http://arxiv.org/abs/2607.15003v1) | Riccardo Curcio, Toni Mancini et al. | The deployment of autonomous cyber-physical systems in safety-critical environments requires closed-loop control strategies (i.e., policies) that are not only performant but also provably safe and robust. While learning-based methodologies such as Reinforcement Learning offer flexible and scalable approaches to automatically synthesize such controllers, they typically lack the formal guarantees necessary for safe deployment. To bridge this gap, we propose a novel simulation-based methodology to automatically synthesize policies with formal guarantees regarding performance, safety, and robustness specifications. Specifically, given a set of properties to verify, a confidence parameter $δ$ and an allowable failure probability $\varepsilon$, our method guarantees that the synthesized policy comes with a certificate: with confidence at least $1 - δ$, the probability of encountering a scenario where the given properties are violated is at most $\varepsilon$. We demonstrate the feasibility of our approach by developing SMC-ES, an algorithm that integrates Evolutionary Strategies with Statistical Model Checking-based verification. We evaluate SMC-ES on a suite of continuous control tasks using Gymnasium and Safety Gymnasium testbeds. Results show that, at the price of a sustainable increase in computational cost, our algorithm provides formal guarantees regarding performance, safety, and robustness specifications, while performing competitively against leading model-free Deep Reinforcement Learning (DRL) and Safe-DRL baselines. |
| 2026-07-16 | [Introspective Attention Modulation for Safe Text-to-Image Generation](http://arxiv.org/abs/2607.14945v1) | Basim Azam, Hossein Rahmani et al. | State-of-the-art flow based text-to-image (T2I) models exhibit remarkable generative abilities but remain vulnerable to producing unsafe content. Prior safety efforts range from concept erasure and prompt filtering to classifier-based gating. However, simple techniques like parameter efficient adaptations of the models easily bypass such guardrails. We introduce a unique principled approach that achieves safety by regulating the model's attention dynamics through inference-time introspection, exhibiting intrinsic robustness. Our method analyzes and rebalances attention activations throughout image synthesis, steering generations away from unsafe concepts while preserving semantic alignment. This introspective control ensures safety of deployed models. Across standard and adversarial safety benchmarks, our approach achieves remarkable safety scores while maintaining or even improving alignment and perceptual quality. Our results reveal that attention-space regulation offers a considerably more promising path to safer diffusion transformer based image generation than the existing concept erasing mechanism.Our code can be accessed at https://basim-azam.github.io/iam/ |
| 2026-07-16 | [Innocuous-Seeming Data, Latent Ideology: Ideological Generalisation in Finetuned LLMs](http://arxiv.org/abs/2607.14888v1) | Robert Graham, Edward Stevinson et al. | Finetuning language models on small, curated datasets is standard practice for adapting them to specific policies or domains. We show that finetuning on narrow, factually-defensible, moderation-passing data can cause broad ideological shifts across unrelated domains, while preserving general capabilities. Training GPT-4.1 on right- or left-leaning economics Q&A yields matched ideological shifts on topics such as criminal justice, the environment, and cultural taste. The same effect appears with plausibly-deployed datasets such as workplace HR policy and practical finance queries, as well as on a science-pseudoscience axis where food-safety finetuning increases sycophantic agreement with users expressing false health beliefs. We call this phenomenon ideological generalisation and propose a methodology to measure two properties: breadth, how far the shift reaches across topics absent from training, and amplification, how much finetuning intensifies the shift relative to few-shot prompting on the same examples. We show that few-shot prompting indicates the direction of generalisation but finetuning pushes the model to further extremes, including to far out-of-distribution outputs such as endorsements of race-IQ connections and political violence. The effect replicates on Gemma-3, holds under judge-free evaluations and external benchmarks, survives mixing with generic data, and leaves GSM8K accuracy within $\pm 1$pp of the baseline. |
| 2026-07-16 | [Ground-Side Mission Plan Compilation with Policy-as-Code Guardrails for Cloud-Native Satellite Platforms](http://arxiv.org/abs/2607.14798v1) | Hsiu-Chi Tsai, Chia-Tung Chung | Onboard cloud-native runtimes for satellites are emerging on multiple tracks (ORCHIDE, Axiom Space's AxDCU-1, Kepler's Jetson nodes), but each assumes that the workflow artifacts it executes arrive from the ground. ORCHIDE's architecture document D3.1 states explicitly that "only the Deferred Phase is part of the ORCHIDE scope," and no open-source ground-side toolchain has been released by the consortium. We present Satellite Mission Compiler, a four-stage pipeline that addresses this gap: it takes a human-authored mission plan, checks it against machine-checkable structural and policy rules, and compiles it into the container-workflow artifacts that cloud-native satellite runtimes consume. The pipeline parses the plan against a Pydantic schema derived from public ORCHIDE materials, evaluates it against an OPA/Rego policy package of ten deny rules with documented provenance, compiles it into a typed WorkflowIntent intermediate representation, and renders it as Argo Workflow DAGs and Kueue Job manifests with Dynamic Resource Allocation (DRA) support. We classify pre-uplink loss events into four severity tiers tied to specific schema and policy checks, and anchor the layered-validation design in the safety reading of defense-in-depth (NASA-STD-8739.8B) rather than the security reading (NIST SP 800-53). The implementation is validated by golden translation evaluations, argo lint, an in-process baseline that reproduces OPA's decisions, and live single-node cluster submission, including a DRA-backed GPU admission cascade on Kueue v0.17.3 (re-validated on v0.18.3) and, on v0.18.3, a unified GPU+CPU device-class quota with a scheduler-level accelerator fallback. Six Model Context Protocol (MCP) tools expose the pipeline to AI agents. The compiler is released under EUPL-1.2 (DOI 10.5281/zenodo.21228150). |
| 2026-07-16 | [Transcoders for Investigating Deception in Language Models](http://arxiv.org/abs/2607.14791v1) | Darius Lim, Nathan Leow et al. | Transcoders have recently emerged as a promising approach for mechanistic interpretability (MI), enabling circuit-level analysis of model behaviour. In this paper, we investigate the use of transcoders to analyse deceptive behaviour in language models, a behaviour that poses a safety and security risk. Using a Qwen3-4B model with pre-trained transcoders, specifically per-layer transcoders (PLTs), we construct attribution graphs that capture feature activations and inter-feature dependencies, allowing circuit-level analysis of deception. Through feature steering and circuit analysis, we identified a dictionary of deception-related features and show that these features exert a stronger influence on deceptive outputs, as they produce predictable shifts between deceptive and non-deceptive responses. These findings suggest that deception emerges from internal model mechanisms and highlight the potential of transcoders for behavioural monitoring and early detection of security vulnerabilities related to malicious behaviours in language models. |
| 2026-07-16 | [Global Index on Responsible AI: 2026 Report](http://arxiv.org/abs/2607.14782v1) | Rachel Adams, Fola Adeleke et al. | Grounded in human rights-based frameworks such as the UNESCO Recommendation on the Ethics of AI, the Global Index on Responsible AI (GIRAI) examines how countries translate responsible AI commitments into enforceable protections, institutional capacity, and redress mechanisms. GIRAI 2026 assesses these across five dimensions: Inclusion and Diversity, Ethics and Sustainability, Labour and Skills, Trust and Safety, and AI Use in Public Service. A global network of 135 country-level researchers collected and assessed 68,138 data points on 38 indicators organised across three pillars: government AI policy and implementation (17 indicators), civil society engagement (5), enabling conditions (15), and documented cases of government deployment of unacceptable-risk AI systems. The data covers November 2023 to September 2025. A score of 100 is derived from the indicators to rank all countries. Findings show that while responsible AI governance is expanding, with 126 of 135 countries having at least one government policy or initiative across the 17 AI Policy indicators, this does not often translate into meaningful protection. For instance, Global South countries account for 203 of 306 new cases of indicators with frameworks since the first edition, yet 78% of their frameworks remain non-binding compared with 42% in the Global North. Government commitment to AI governance also does not extend to their own algorithms: whereas Transparency and Explainability is the strongest performing indicator, with 58% of countries having some framework, only 18% require Public Disclosure of Government Algorithms. Credible evidence of government deployment of unacceptable-risk AI systems was also found in 35 countries. These findings show that responsible AI governance must move beyond framework adoption toward enforceable rights-based protections, resourced oversight institutions, and accessible redress. |
| 2026-07-16 | [GeoDetect: Geometric Adversarial Detection for VLPs](http://arxiv.org/abs/2607.14737v1) | Afsaneh Hasanebrahimi, Hanxun Huang et al. | Vision-language pre-trained models (VLPs) are widely used in real-world applications. However, they remain vulnerable to adversarial attacks. Although adversarial detection methods have demonstrated success in single-modality settings (either vision or language), their effectiveness and reliability in multimodal models such as VLPs remain largely unexplored. In this work, we study the geometry of VLP embedding spaces and observe structured anisotropy that differs from unimodal vision models. Our theoretical analysis shows that under this anisotropic structure, adversarial attacks increase the expected geometric separation between clean and adversarial examples (AEs). Specifically, we demonstrate that AEs consistently exhibit greater expected distances to randomly sampled points than their clean counterparts, indicating that AEs tend to push representations out of manifold regions. Building on these insights, we propose GeoDetect, which leverages these off-manifold deviations via geometric scores to identify AEs. Through comprehensive evaluations, we show that our approach reliably detects AEs across diverse VLP architectures and threat settings, covering unimodal and multimodal attacks as well as adaptive attacks, thereby providing a robust and practical approach to improving the safety and reliability of these models. |
| 2026-07-16 | [MIND-CAVs: Multi-Intelligence Negotiation and Decision System for CAVs based on Intent-Driven Autonomy](http://arxiv.org/abs/2607.14688v1) | Mainak Mondal, Yihang Feng et al. | Modern autonomous vehicles largely operate as isolated agents: they rely on on-board perception and decision modules and broadcast Basic Safety Messages (BSMs) that expose only low-level kinematic state. While existing cooperative driving frameworks enable limited sensor sharing, they rarely communicate high-level maneuver intentions, and edge computing is primarily used for content delivery rather than decision arbitration. As a result, current connected autonomy lacks a principled mechanism for making globally consistent, intent-aware coordination decisions across vehicles. To address this gap, we propose MIND-CAVs, a Multi-Intelligence Negotiation and Decision framework for connected autonomous vehicles (CAVs) based on intent-driven autonomy. Each vehicle abstracts raw sensor observations into structured intent representations, exchanges them over V2X links, and receives globally consistent coordination plans from roadside edge servers. Edge agents combine learned and rule-based arbitration mechanisms to negotiate conflicting intents among vehicles, while a cloud platform records decisions for auditing and continual retraining. We implement MIND-CAVs in a CARLA-based AI-in-the-loop platform and evaluate it in multi-lane highway scenarios involving conflicting maneuvers and route-constrained exits. Experimental results show improved maneuver completion time and reduced unsafe proximity and unnecessary braking compared with isolated autonomy, first-come-first-served arbitration, and multi-agent reinforcement learning baselines. |
| 2026-07-16 | [InCarEmo: A Multimodal Dataset for In-Cabin Emotion Recognition and Driver State Monitoring](http://arxiv.org/abs/2607.14683v1) | Hao Yang, Yanyan Zhao et al. | Understanding driver emotion and state is critical for the next generation of intelligent in-cabin systems that ensure safety and enhance human-vehicle interaction. However, existing public datasets for in-cabin affective computing are largely limited to visual modalities and rarely include conversational information, making it difficult to capture the linguistic and interactive cues underlying driver emotion. To address these gaps, we introduce InCarEmo, a multimodal dataset for in-cabin emotion recognition and driver state monitoring. InCarEmo integrates RGB and infrared video, in-cabin audio, and dialogue text collected from scripted in-cabin scenarios designed to simulate realistic driver behaviors, covering diverse lighting conditions and driving contexts. The dataset supports three primary tasks: 1) multimodal emotion recognition, 2) fatigue detection, and 3) distraction monitoring. In addition to the original Chinese data, we construct an auxiliary English benchmark to support preliminary cross-lingual evaluation. We provide a unified benchmark with extensive baseline results across unimodal and multimodal methods, including analyses under modality-missing and noise conditions. Experimental results demonstrate the benefits of multimodal fusion and reveal remaining challenges under real-world noise and low-light conditions. By releasing InCarEmo, we aim to establish a comprehensive foundation for robust, interpretable, and human-centric in-cabin affective understanding, promoting safer and more empathetic driver-vehicle interaction. |
| 2026-07-16 | [Governing Artificial Intelligence: Public Preferences and Regulatory Options](http://arxiv.org/abs/2607.14585v1) | Magnus Lundgren, Jonas Tallberg | Artificial intelligence (AI) is rapidly transforming economies, societies, and polities, raising fundamental questions about how it should be regulated. Policymakers face choices over whether to prioritize innovation or safety, rely on public oversight or private self-regulation, and govern nationally or internationally. Yet little is known about how citizens evaluate these competing priorities. Here we report a conjoint survey experiment conducted in seven countries with diverse political and economic profiles. We find that citizens strongly support regulating AI and generally prioritize safety over innovation, public governance over private self-regulation, and international over national approaches. The preference for safety is strongest among those who perceive AI as risky, unpredictable, and personally consequential. These findings reveal a systematic misalignment between dominant regulatory approaches and citizen preferences. |

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



