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
| 2026-10-21 | [The End of Consensus Research?](http://arxiv.org/abs/2610.30114v1) | Clément Burgelin, Rachid Guerraoui et al. | Consensus is a fundamental problem in distributed computing and at the core of state machine replication (SMR). Despite decades of active human research, discovering practical low-latency solutions to the problem is still open, as it requires combining protocol knowledge, correctness reasoning, implementation, and experimental evaluation. We report on how AI helped us discover A-greement, a new consensus protocol for geo-replicated state machines that significantly outperforms state-of-the-art solutions. Averaged across six deployments with 5 or 9 replicas, A-greement reduces the mean latency of a geo-distributed key-value store over the fastest baseline by 12.2% and 33.3% on write-heavy and read-heavy workloads, respectively, and matches the fastest baseline on read-only workloads. A-greement combines lease-authorized local reads, direct client learning for leader-ordered writes, and a fast fixed-quorum per-key write path, selected according to network topology and workload. At the heart of our discovery lies AI-greement, an agent harness for the "autonomous" discovery, implementation, experimentation, and verification of candidate protocols. AI-greement integrates (1) a concept-oriented protocol knowledge base, (2) a modular implementation skeleton, and (3) a Lean-based verification framework with reusable safety definitions, reference specifications, and proofs. AI-greement is of independent interest: its Lean formalization uncovers several previously unreported vulnerabilities in existing consensus protocols, and it provides guidelines on how domain-specific harnesses can turn general-purpose coding agents into effective tools for (distributed) computing research. Yet we argue that human consensus experts remain essential before, during, and after the search: agents reshape the researcher's role rather than end consensus research. |
| 2026-10-21 | [Differentially Private Sparse Reward Estimation with Preference Feedback](http://arxiv.org/abs/2610.30099v1) | Meng Ding, Mingxi Lei et al. | Reinforcement Learning with Human Feedback (RLHF) has become a central paradigm for aligning Large Language Models with human values. However, the reliance on sensitive preference data necessitates rigorous differential privacy (DP) guarantees to prevent user information leakage. Existing private alignment methods typically assume dense reward parameterizations, resulting in estimation error bounds that scale polynomially with the ambient {model size $d$}. Such dependence is prohibitive in modern {overparameterized models} where $d$ is vast. In this work, we study \emph{Private Sparse RLHF} problem to address this limitation, assuming that human preferences (reward model) are governed by a subset of $s^* \ll d$ relevant features. We provide the first comprehensive theoretical framework for differentially private sparse reward estimation from pairwise preference feedback under the Bradley-Terry-Luce model, and introduce efficient algorithms for both Sample-DP and Label-DP settings. Theoretically, under the squared $\ell_2$ parameter estimation error, we establish upper bounds of $\widetilde{O}(s^*/n + {s^*}^2/n^2\varepsilon^2)$ for $(\varepsilon,δ)$-Sample-DP and $\widetilde{O}(s^*/n + s^*/n\varepsilon^2)$ for $\varepsilon$-Label-DP based on randomized response, where $n$ is the training data size. For the lower bounds, we show that the Sample-DP rate is nearly optimal, and that the guarantee for our Label-DP method cannot be further improved within the randomized-response framework. We also provide a general lower bound for the problem with Label-DP. Empirically, our framework demonstrates superior efficacy over dense baselines across sentiment generation and safety alignment benchmarks, offering a significantly more favorable privacy-utility trade-off. |
| 2026-10-21 | [Circuits of LLM Classifiers](http://arxiv.org/abs/2610.30076v1) | Hadi Khalaf, Sajani Vithana et al. | Language models make it easy to build classifiers by simply prompting a model with the sample to classify. However, their predictions are neither deterministic nor uniformly reliable across samples. How can we combine predictions from different models to improve reliability? We view LLM classifiers as noisy measurement instruments of an underlying target label. Our goal is to maximize average classification accuracy at a given average query cost. To this end, we introduce circuits that organize LLM classifiers into an adaptive sequence of queries and decisions. We then characterize the optimal circuit, obtain the fundamental limits of achievable accuracy, and provide practical insights on realizing the optimal circuit. Across seven classification tasks spanning safety, health, political science, human preference, agent evaluation, decision-making and reasoning, our method improves the cost-accuracy frontier over routing and cascading baselines. |
| 2026-10-21 | [TARSphere: A Robot-Oriented Full-Stack Framework for Long-Horizon Embodied Agents in Evolving Physical and Social Environments](http://arxiv.org/abs/2610.30043v1) | Yicheng Xu, Zhikai Xu et al. | Robots must act under physical and social constraints while adapting to changes in their environment and human needs. Existing embodied-agent systems often focus on specified goals or abstract execution into symbolic actions, limiting evaluation in an evolving world. We introduce TARSphere, a robot-oriented full-stack framework for physically and socially grounded long-horizon embodied agents. TARSphere-Infra runs robot policies in a physics simulator with embodied NPCs and constrained dialogue. An event-driven finite-state machine (FSM) uses completed actions and delivered information to update task states and evaluate outcomes. TARSphere-Harness, its embodied brain harness, lets a frozen multimodal large language model (MLLM) act through multi-level executable skills, using observations, hierarchical memory, and execution feedback to revise its decisions. On this foundation, agent-in-the-loop curation aligns task semantics, causal dependencies, and scene geometry to construct TARSphere-Bench: 360 episodes across 30 indoor scenes. Five paired conditions assess perceptual robustness, disruption adaptation, task update, human coordination, and intent understanding for navigate-and-report and fetch-and-deliver tasks. Our evaluation of 8 MLLMs shows that general-purpose models can outperform embodied models, while task updates and human interaction remain difficult. Frequent collisions and falls further motivate evaluating execution safety alongside task success, even with a shared action interface and decision budget. |
| 2026-10-21 | [Unlocking the Full Potential of Taylor Dispersion Analysis Through Advanced Detection](http://arxiv.org/abs/2610.30039v1) | Rami Laoufi, Agnès Hagège | Today, pharmaceutical sciences rely on a wide range of innovative therapeutics of varying sizes: nanoparticles, proteins, antibodies, and nanocarriers for drug delivery. Measuring their size is one of the crucial parameters in their characterization, to ensure drug safety, stability, and effectiveness. Over the past two decades, Taylor Dispersion Analysis has emerged as an important sizing method. Its scope of application is perfectly suited to the sizes of these therapeutics. By design, it can be coupled with a wide range of detectors, and extensive research has been conducted in this field. In this review, we will present an overview of the studies conducted, and discuss how the application drives the choice of one detector or another (size measurement, aggregation studies, interaction studies, and other analyses, performed in both simple biological buffers and in more complex media, particularly serum) and makes Taylor Dispersion Analysis a highly versatile tool. |
| 2026-10-21 | [Object- and Task-Aware Active Perception for Safe Navigation of Mobile Manipulators in Semi-Static Environments](http://arxiv.org/abs/2610.29964v1) | Jingxing Qian, Xintong Du et al. | Mobile manipulators must navigate safely in cluttered, partially unknown environments across logistics, manufacturing, and service applications. Semi-static changes between robot visits can lead to outdated scene knowledge and increased collision risk. We propose an active perception framework that enables a high degree-of-freedom (DoF) mobile manipulator to inspect uncertain or safety-critical regions with its hand camera during navigation. A probabilistic, object- and task-aware next-best-view (NBV) formulation prioritizes observations near the desired path and around uncertain or potentially changing objects. The perception objective is integrated into hierarchical-task model predictive control (HTMPC), exploiting kinematic redundancy for viewpoint adjustment while preserving the higher priority of base-trajectory tracking. The system unifies navigation, map maintenance, and active perception in a closed loop. Real-world tests show improved map consistency, reduced task-dependent map entropy, and lower navigation risk than passive or non-task-aware baselines. Additional simulation and real-world sequences demonstrate responses to controlled semi-static scene changes. |
| 2026-10-21 | [Unintended Side Effects of Value Training in LLMs](http://arxiv.org/abs/2610.29922v1) | Siddhesh Pawar, Chen Cecilia Liu et al. | As LLMs are increasingly deployed in value-sensitive contexts, aligning them with human values has emerged as a post-training objective in NLP, aimed at instilling additional values and guiding model behaviour. However, current evaluations remain narrow. Because values are interconnected, it is unclear how fine-tuning for specific values alters unrelated capabilities and behaviours. In this work, we adapt four instruction-tuned LLMs across individual Hofstede's value dimensions and evaluate them on both cross-value transfer and broader behaviours. We find that value-specific training produces unintended side effects on untrained values, risk preferences, social cognition, and safety traits. Consequently, we suggest that value alignment after post-training should incorporate broader behavioral evaluations, and we highlight the need to investigate where additional value training might best fit into the pipeline. |
| 2026-10-21 | [Characterizing and Defending Utility-Hijacking in Instruction-Tuning Data Selection](http://arxiv.org/abs/2610.29886v1) | Yahao Ren, Jian Lou et al. | Instruction-data selectors choose which instruction-following records are used for fine-tuning, improving alignment quality, raising training efficiency, and filtering out harmful records. A selector's choices thus directly affect the fine-tuned model's safety, yet despite their widespread use, the security of these selectors remains under-studied. We conduct the first systematic study of adaptive supplier-side attacks against utility-oriented instruction-data selectors and introduce utility-hijacking. Utility-hijacking shapes harmful records to remain compatible with a selector while preserving their malicious payload, causing top-k selection to concentrate rather than filter the poison. Achieving both is non-trivial, because optimizing a record for score tends to wash out its payload, while bluntly preserving the payload makes the record look low-value. Our generate-and-rerank procedure generates, per instruction, many payload-bearing candidates and submits only the highest-scoring survivor. Evaluated across three payloads (over-refusal, false-fact injection, and insecure code), six selectors, and victim models from Qwen2.5-7B to Qwen3-32B, utility-hijacking drives refusal from 0% to 94% and insecure code up to 56/90, while false-fact enrichment intensifies with scoring-model scale from 2.84x to 7.04x. To defend against this attack, we propose CONCERT, a targeted defense needing no content detector or poison labels. It flags suppliers selected far more often than chance allows, then re-selects under the same budget, keeping only records that two dissimilar selectors both rank highly. On a Qwen3-32B victim, CONCERT reverses the poison concentration from 6.67x to 0x (code) and 3.59x to 0.67x (facts), and reduces downstream harm by 100% and 75% respectively, at unchanged capability. |
| 2026-10-21 | [COMET: A coordinated multi-layer runtime monitoring framework for human-supervised AI-enabled safety systems](http://arxiv.org/abs/2610.29816v1) | Niclas Flehmig, Mary Ann Lundteigen et al. | Incorporating AI systems into safety systems requires monitoring to avoid silent faults and ensure safe and reliable operation. Single monitors often do not provide actionable information for human supervisors, who act as the system's adaptive controller. We present COMET, a coordinated multi-layer runtime monitoring framework for human-supervised AI-enabled safety systems in which each monitor is anchored in a different fault mode. Monitor outputs are combined in a coordination layer and mapped by a communication layer to a traffic-light display. In this work, COMET consists of three monitoring layers: signal integrity, physics consistency, and the AI system's competence envelope. The framework is evaluated on an open dataset of compressed-air leaks, where an AI system detects leaks from acoustic emissions, with held-out conditions simulating covariate shift, semantic shift, and sensor faults, events expected during runtime but unknown in their exact appearance. All monitors are frozen after development and tested zero-shot, and the framework is compared against no-monitoring and single-monitor setups. The bare AI system fails silently and confidently under all three shift types, whereas COMET reduces the trusted-wrong-answer rate to zero on covariate-shifted no-leak data and on three of four sensor faults, at a nominal false-alarm rate of $\approx 1\%$. Single monitors come close in detection but cannot attribute the fault mode. COMET converts their anonymous detection flags into diagnoses. Residual silent misses under semantic shift define the boundary of passive monitoring. Runtime monitoring frameworks for AI-enabled safety systems must therefore combine quantitative fault detection and isolation with qualitative, human-oriented feedback. |
| 2026-10-21 | [AssureCode: Structured Justification for Trustworthy Agentic Coding](http://arxiv.org/abs/2610.29758v1) | Logan Murphy, Wilson Sy et al. | The use of AI-based tools in software development is rapidly changing how software is produced, enabling large volumes of code to be generated with limited human implementation effort. However, this raises fundamental challenges for ensuring that generated artifacts are aligned with developer intent and have been appropriately validated. In this work, we propose the use of structured argumentation in driving AI-assisted software development. Inspired by techniques from assurance engineering for safety-critical systems, we define a language of justification rationales that explicitly relate informal natural language prompts, formal specifications, assumptions about developer intentions, and program analysis evidence in a structured and verifiable format. We then describe AssureCode, a tool-supported methodology for creating such rationales as part of agentic code generation. We provide an empirical analysis of our implementation where we investigate its capacity to contribute towards justified assurance of correctness for AI-generated code. |
| 2026-10-21 | [Scalar-scalar-graviton vertex from Asymptotic Safety](http://arxiv.org/abs/2610.29717v1) | Tim Eckert, Benjamin Knorr | The unitarity and causality properties of asymptotically safe quantum gravity are still unsettled. Scattering amplitudes provide a direct test, but their computation requires fully momentum-dependent correlation functions. In this light, we compute the full scalar-scalar-graviton vertex from Asymptotic Safety within a mostly analytic approximation. Qualitatively, we find the expected logarithmic branch-cut structure, and we correctly recover the double-eikonal soft divergences predicted by Weinberg's soft theorem. Quantitatively, we find that non-minimal operators are generally as important as the minimal vertex stemming from the scalar kinetic term. |
| 2026-10-21 | [Beyond Policy Engines: Binding-Complete Execution Admission](http://arxiv.org/abs/2610.29701v1) | Lin Wang, Mengting Wu et al. | Policy engines determine whether evaluated policies permit a request, but a correct ALLOW decision does not establish complete governance-source participation, exact context binding, or sufficient grounds for execution admission. This paper examines the gap between policy evaluation and execution admission. We introduce the Binding-Complete Execution Admission Contract (BCEAC), a governance-relative contract for independently determined source obligations, exact policy and contextual binding, and typed admission outcomes under explicit trust assumptions. We propose a conditionally engine-agnostic architecture in which participating backends must satisfy declared semantic profiles, adapter contracts, and contextual correspondence obligations; an Execution Boundary Language (EBL) authoring syntax remains optional. The architecture separates governance resolution, policy evaluation, independent admission verification, and downstream execution mediation. We investigate two distinct experimental tracks: a finite Governance Source Completeness study and an OPA/Cedar experiment with archived native-engine records examined by offline review. The OPA/Cedar archive comprises 72 scenarios and 144 strict case-engine records (120 native-invoked and 24 pre-native), with eight additional permissive controls outside the strict denominator. The offline review did not rerun the original native OPA/Cedar executables. The results provide bounded evidence while retaining historical failures and unresolved implementation defects. The work does not establish constitutional adequacy, general semantic refinement, authenticated currentness, machine-checked execution-safety theorems, or protected external effects. Its contribution is an explicit, falsifiable framework for determining what must be established between a policy decision and an execution-admission judgment. |
| 2026-10-21 | [SPARE: Sparsity Allocation via Routing Entropy for Efficient MoEs](http://arxiv.org/abs/2610.29662v1) | Yifan Yang, Hanling Yi et al. | Mixture-of-Experts (MoE) architectures scale model capacity efficiently, yet their substantial memory footprint poses a major barrier to deployment. Expert pruning reduces this footprint, but most methods focus on which experts to remove and prune every layer equally. We propose SPARE (Sparsity Allocation via Routing Entropy), a training-free, selector-agnostic method that decides how many experts to remove from each layer using routing entropy, the entropy of tokens' normalized Top-K gate weights on calibration data. When a token's gate weight is spread evenly, removing one active expert discards little of it, so SPARE prunes layers with more diffuse routing more aggressively. A perturbation bound under random pruning formalizes this intuition, and controlled interventions with a fixed selector link concentrated routing to larger output changes. With the same selector and budget, SPARE improves average accuracy over uniform allocation on Mixtral, Qwen3, and DeepSeek models, especially at high pruning ratios. At 50% expert pruning, it raises the eight-task average of Qwen3-30B-A3B from 57.2% to 63.8%, and the math, code, and science average of DeepSeek-R1 from 80.1 to 82.8. |
| 2026-10-21 | [Certified Neural Approximation of Robust Nonlinear Model Predictive Control](http://arxiv.org/abs/2610.29559v1) | Hendrik Alsmeier, Maik Pfefferkorn et al. | Model predictive control is a high-performance control method, but it incurs comparatively high online computational costs across many control tasks and is challenging to realize in low-level hardware settings. To address this, we present a framework for generating neural network controllers that provably inherit the safety and stability guarantees of a robust model predictive policy. In contrast to approaches based on post-verification or additional online safety filters, our method provides constructive, a priori design conditions that ensure preservation of recursive feasibility and stability before deployment. We treat the approximation error as a bounded input disturbance, which is addressed in the MPC design. This allows the neural network controller to directly inherit robustness from the underlying model predictive controller, which enables the transfer of guarantees to the approximation. The design process is an offline, constructive synthesis pipeline and can be implemented once Lipschitz bounds are specified. We further demonstrate in simulation that the learned controller reproduces the closed-loop behavior of the original controller while reducing online computation. |
| 2026-10-21 | [Safe World Model for Joint Trajectory and Beamforming Design of Mobile Base Stations](http://arxiv.org/abs/2610.29534v1) | Abdolrasoul Sakhaei Gharagezlou, Zeinab Askari Donbeh et al. | Mobile base stations (MBSs), including aerial and ground-based platforms, provide flexible wireless coverage through adaptive positioning. However, jointly optimizing their trajectories and digital beamforming is challenging because movement changes channel conditions and interference, while beamforming determines the resulting signal and interference powers. These coupled decisions also account for how current movements affect future positions and communication performance under mobility and safety constraints. We formulate this problem as a constrained Markov decision process that maximizes the time-averaged sum signal-to-interference-plus-noise ratio (SINR), subject to mobility, restricted-zone avoidance, and transmit-power constraints. To support long-term planning and reduce the extensive environment interactions required by model-free reinforcement learning, we develop a SafeDreamer-based world model framework that learns latent system dynamics from experience and optimizes a joint trajectory and beamforming policy through imagined trajectories. Feasibility-preserving action mappings and safety cost signals incorporate operational constraints into policy learning. Simulation results demonstrate that the proposed framework improves the time-averaged sum SINR by an average of 7\% and 9\% compared with proximal policy and deep Q-network baselines, respectively, while achieving higher sample efficiency and more reliable constraint satisfaction. |
| 2026-10-21 | [Hard and Soft Real-Time DNN Inference Scheduling on Heterogeneous Multi-Accelerator Systems via Reinforcement Learning](http://arxiv.org/abs/2610.29503v1) | Francesco Giulio Blanco, Enrico Russo et al. | We present a scheduling framework for heterogeneous multi-accelerator systems executing both hard real-time and soft real-time QoS-aware DNN inference workloads. The framework combines an online reinforcement learning (RL) scheduler with a deterministic runtime safety layer. The RL scheduler dynamically generates candidate schedules by jointly selecting sub-job priorities and accelerator assignments, while runtime certification guarantees that all HRT deadlines remain satisfied independently of the learned policy.   To enable runtime certification in heterogeneous precedence-constrained systems, we introduce a schedulability-preserving prefix replacement criterion based on machine-wise cuts. The proposed criterion guarantees that runtime-generated schedule prefixes preserve the existence of a feasible continuation without requiring complete finish-time dominance with respect to a feasible reference schedule.   Experimental results on heterogeneous DNN accelerator platforms show that the proposed framework preserves HRT schedulability while improving QoS-aware deadline satisfaction, client-level service quality, and accelerator utilization compared to state-of-the-art scheduling approaches. |
| 2026-10-21 | [How Not to Be Censored: Inclusion Safety in Client-Led Consensus](http://arxiv.org/abs/2610.29497v1) | Yackolley Amoussou-Guenou, Andrea Canidio | Blockchain consensus protocols implicitly follow a client-server architecture: validators are the only active participants, while clients (i.e., users) merely submit transactions and rely on validators to include them. This abstraction obscures a fundamental question: what should a client do to guarantee the timely inclusion of its own transaction? We study this question by introducing a leaderless consensus protocol in which clients are active. We define inclusion safety: the guarantee that a client's transaction enters some of the blocks produced for an epoch, and that every block omitting it belongs to a fork that is eventually abandoned. Because every transaction carries a reference to a block of the previous epoch, clients help select the surviving fork. This yields two paths to inclusion: reaching sufficiently many honest replicas within an epoch, or relaying a block containing the transaction across epochs until the corresponding fork becomes canonical. In a generalized model spanning synchrony and asynchrony, the client's and the replicas' connectivity are substitutes for inclusion safety: it is achieved either if the client is well connected, or if the client is poorly connected but honest replicas communicate reliably, provided the client reaches at least one of them. However, the substitution is imperfect: when inclusion safety rests on the replicas' connectivity, it requires a longer epoch compared to when the client is well connected. |
| 2026-10-21 | [Searches for coincident Gravitational Waves and High-Energy Neutrinos from the KM3NeT and ANTARES telescopes during O3](http://arxiv.org/abs/2610.29436v1) |  ANTARES Collaboration,  KM3NeT Collaboration et al. | Various models predict that gravitational-wave sources produce high-energy neutrinos. Among them are compact binary coalescences with at least one neutron star, but also core-collapse supernovae or accretion disk instabilities. Two different analyses are presented, combining data from the LIGO and Virgo gravitational-wave detectors with those from the ANTARES and ORCA neutrino telescopes observed during the third observing run of LIGO-Virgo. In the first analysis, selected high-energy neutrino candidates were used as triggers to search for gravitational-wave candidates. In the second analysis, a search was conducted for potential spatial and temporal coincidences between subthreshold high-energy neutrino candidate events and subthreshold gravitational-wave candidates emitted by compact binary coalescence systems. In both searches, no significant coincidences were found. Upper limits on the rate of sources emitting both high-energy neutrinos and gravitational waves are presented. In particular, with the joint search, a joint source rate density upper limit of $8.8\times10^2~\mathrm{ Gpc^{-3}yr^{-1}}$ was found for a typical binary neutron star coalescence with a GW emitted energy of $10^{-1}~\rm M_\odot c^2$ and an isotropic energy emitted in neutrinos of $10^{51}~$erg. |
| 2026-10-21 | [Situating Vulnerability: An Ethical Governance Framework for Anthropomorphic AI Interaction](http://arxiv.org/abs/2610.29304v1) | Hanhui Xu, Jiacheng Ji et al. | Large language models increasingly address users in the register of care: they remember personal details, adopt a first-person emotional voice, and sustain relationships across many sessions. This anthropomorphic mode of interaction generates risks that output-level metrics capture poorly, because the same empathic utterance may console one user, entrap a second, and consolidate a delusion in a third. Governance has responded in two ways. Uniform restriction refuses sensitive requests or suppresses empathic phrasing across all outputs; group-targeted protection profiles vulnerable populations and applies stricter or prohibitive rules to them. This article argues that both fail. Uniform restriction mislocates harm in the lexical form of responses rather than in the trajectory of a relationship; group-based protection depends on a list that can never be complete, treats a fluid property as a static attribute of persons, and converts protective attention into stigmatizing labels. Drawing on philosophical work that treats vulnerability as universal, relational, and differentiated, we propose that the unit of ethical governance should be the vulnerability situation rather than the vulnerable group, develop a typology of five situations - physical, emotional, cognitive, autonomy, and relational - and specify seven principles into requirements that guide response calibration without diagnosing or persistently classifying users. The framework relocates AI safety from output control to relational safety. |
| 2026-10-21 | [Compressing Onboard Language Models for Space Robots under a Behavioral-Fidelity Requirement](http://arxiv.org/abs/2610.29302v1) | Jia Deng, Zizhe Wang et al. | Language models are being explored as the interface through which space robots interpret operator commands and select among verified skills. Running them onboard calls for low-bit compression, and flight use adds requirements that generic post-training quantization leaves implicit: a fixed memory budget and storage format, a static artifact under configuration control, and behavior close to the full-precision model validated on the ground. We treat compression as a ground-segment step and study QSelect, which composes one static low-bit model from compatible AWQ, GPTQ, and AutoRound checkpoints by assigning each linear module to one quantizer family and scoring every substitution at the final next-token distribution. The default fidelity mode needs no labels or tuned thresholds and accepts a substitution only when the forward KL divergence from the BF16 reference decreases; an optional target-adapted mode adds answer likelihood on a small labeled set. On DeepSeek-R1-Distill-Qwen-7B, Llama-3.1-8B, and Llama-3.2-3B at 3 and 4 bits, composed models reach lower mean held-out perplexity than every single-family model in all six settings, and the fidelity mode lowers perplexity and KL further. In a matched scaffold, final-output scoring gives lower held-out KL than representation-similarity scoring in all 12 paired runs. With labels from training or development splits of the evaluated task families, the target-adapted mode raises 3-bit pooled accuracy on the two Llama models by 4.3 and 4.5 points over the best single-family model. At 3 bits, unquantized embedding and output layers make up 43-46% of the packed artifact. All evaluations use general-domain benchmarks on a ground GPU; space-domain tasks and flight hardware are left to future work. |

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



