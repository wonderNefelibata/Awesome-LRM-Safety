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
| 2026-06-17 | [Confidence is Not Reliability: Rethinking MC Dropout in Brain Tumour Segmentation](http://arxiv.org/abs/2606.19300v1) | Xin Ci Wong, Duygu Sarikaya et al. | Glioma segmentation in multiparametric MRI is a critical component of treatment planning. A segmentation model that fails silently on treatment-critical sub-regions represents a patient safety risk that overlap-based metrics such as Dice scores cannot expose. We ask whether voxel-level uncertainty estimation via Monte Carlo (MC) Dropout can reliably identify segmentation errors in clinically critical sub-regions, and whether calibration failure modes are detectable from standard reporting metrics alone. In an empirical two-model case study on 126 BraTS21 patients, we evaluate a high-performance pretrained SegResNet and a locally trained UNet with residual units (UNet-Res). MC dropout preserved segmentation accuracy ($|Δ\text{Dice}|$ $<0.01$) while achieving strong uncertainty-error alignment (AUROC for entropy (H) $\approx$0.97), indicating uncertainty correctly ranks erroneous voxels above correct ones. Entropy-based patient stratification identified a high-uncertainty subgroup with substantially lower segmentation performance (median whole-tumour Dice $0.835$ vs. $0.925$), supporting uncertainty as a practical triage signal. However, global alignment can mask important region-specific differences. Despite similar AUROC, UNet-Res exhibited near-zero enhancing tumour entropy ($0.054$) and Expected Calibration Error (ECE) of $0.915$, with a Dice of only $0.714$, indicating severely miscalibrated confidence on the most clinically critical sub-region, a failure mode invisible to standard Dice and AUROC reporting. These findings demonstrate that strong uncertainty-error alignment is necessary but insufficient for clinical safety: sub-region-specific calibration assessment must accompany AUROC evaluation when selecting models for clinical deployment. |
| 2026-06-17 | [Integrated physics-based modeling reveals a thermodynamic gap in small modular reactor load following](http://arxiv.org/abs/2606.19296v1) | Ali Mahboub Rad, Roshni Anna Jacob et al. | Small modular reactors (SMRs) are increasingly considered for flexible power generation; however, many dynamic studies still neglect the thermodynamic coupling between the primary and secondary loops that is essential for accurate assessment of load-following capability. In this study, we develop a hybrid dynamic framework that couples an equation-based model of the NuScale integral pressurized water reactor, including the reactor, primary loop, and moving-boundary helical-coil once-through steam generator, with a physics-based secondary steam cycle comprising the valve, turbine, condenser, and feedwater pump. This approach enforces mass and energy conservation across the coupled system while preserving physically consistent flow interactions across the domain boundary. The integrated model reproduces nominal design-point conditions and is used to analyze a 5% step load rejection under five control strategies, including a decentralized three-loop control architecture for the valve, feedwater pump, and control rods. The results show that partial control strategies are insufficient for efficient and safe operation, whereas simultaneous action of all three actuators stabilizes steam pressure, limits adverse thermal excursions in the primary loop and maintains acceptable steam generator operating margins during load-following maneuvers. Compared with a conventional linear steam-cycle representation, the coupled framework captures dynamic back-pressure and variable turbine enthalpy drop that are otherwise neglected, leading to different predictions of transient behavior and required steam flow. These findings show that thermodynamically coupled, physics-based steam-cycle models are needed for more accurate assessment of the operational flexibility, efficiency and safety margins of SMRs under realistic load-following conditions. |
| 2026-06-17 | [Patnaik-Pearson intrinsic dimension for internal representations of neural networks](http://arxiv.org/abs/2606.19268v1) | Tom Hadfield | We define a new measure of intrinsic dimension of a data manifold, which we call the Patnaik-Pearson dimension, and apply this to internal representations of neural networks, in particular transformers. The inspiration for this comes from the HTSR and SETOL work of Martin, Mahoney and Hinrichs, combined with the TwoNN intrinsic dimension estimator of Facco et al. We prove various properties of this intrinsic dimension estimator. Treating weight matrices of neural networks as data manifolds, for weight matrices whose Empirical Spectral Density follows a Pareto (Power Law) distribution, we relate the Patnaik-Pearson dimension to the HTSR and SETOL analysis, and show that critical values of the tail exponent coincide for the two approaches. Using a combination of theoretical and numerical techniques, we study the behaviour of the Patnaik-Pearson dimension of a data manifold under the transformations typical to neural networks. We apply this machinery to the BERT-base and DeepSeek-R1-Distill-Qwen-1 models, to investigate first the Patnaik-Pearson dimension of the initial data manifold of token embeddings, and second the evolution of the Patnaik-Pearson dimension as token embeddings pass through the layers of the model. Code and notebooks used for the numerical results presented here is available at https://github.com/tdhadfield/PatnaikPearson |
| 2026-06-17 | [A Mixed-Reality Testbed for Autonomous Vehicles](http://arxiv.org/abs/2606.19267v1) | H. M. Sabbir Ahmad, Ehsan Sabouni et al. | We propose a mixed-reality, hardware-in-the-loop (HIL) testbed for autonomous vehicles that seamlessly integrates a physical testbed of mobile robots with a high-fidelity simulation environment. The virtual simulation enables the creation of diverse, safety-critical driving scenarios to validate state-of-the-art perception, planning, and control algorithms, while augmenting simulations with physical robots equipped with multimodal sensors in photorealistic virtual environments further facilitating rigorous validation. Our testbed also features vehicular connectivity using wireless communication and can accommodate a large number of agents through the combination of physical robots and virtual simulated agents, supporting research on multi-agent systems including Connected and Autonomous Vehicles (CAVs). Finally, we present a safety-guaranteed framework combining perception, planning and a novel online learning-based controller using Control Barrier Functions (CBFs) for CAVs. Experiments using the proposed framework are used to validate and demonstrate the key functionalities and the overall utility of the testbed to bridge the gap between simulation and real-world hardware deployment. |
| 2026-06-17 | [TxBench-PP: Analyzing AI Agent Performance on Small-Molecule Preclinical Pharmacology](http://arxiv.org/abs/2606.19245v1) | Hannah Le, Ramesh Ramasamy et al. | Artificial intelligence (AI) agents promise to accelerate drug discovery by compressing interpretation and decision-making loops, but practical deployment requires trusted evaluation on realistic program decisions. We introduce TherapeuticsBench Preclinical Pharmacology (TxBench-PP), a verifiable benchmark for small-molecule preclinical pharmacology and the first focused slice of a broader TherapeuticsBench effort across drug-discovery stages and therapeutic modalities. TxBench-PP tests whether agents can recover accurate conclusions from real-world assay data rather than memorized facts from literature. The benchmark contains 100 evaluations indexed by program stage, assay type, and task structure, spanning mechanism-of-action (MoA) and pharmacodynamic (PD) reasoning, compound-target engagement, causal target validation, developability and safety, and translational efficacy. Agents receive realistic workflow snapshots, inspect files in a coding environment, and return structured answers graded deterministically. Across 16 model-harness configurations, comprising 11 models and 4,800 trajectories, no system reliably recovered preclinical pharmacology decisions. The strongest configuration, Claude Opus 4.8 / Pi, passed 59.3\% of endpoint attempts (178/300; 95\% CI, 51.1-67.6), followed by GPT-5.5 / Pi at 55.3\% (166/300; 47.0-63.6). |
| 2026-06-17 | [Evaluating Rust for Sparse Matrix Kernels in Scientific Computing](http://arxiv.org/abs/2606.19213v1) | Luca Lombardo, Fabio Durastante | Sparse matrix kernels form the computational backbone of scientific computing, traditionally relying on C/C++ and Fortran implementations that prioritize performance over memory safety. This work evaluates Rust as a systems-level alternative for sparse linear algebra by implementing and benchmarking three core workloads: sparse matrix-vector multiplication (SpMV), Lanczos-based Krylov methods, and matrix-exponential evaluation. We compare native Rust code against established baselines (Intel oneMKL, Eigen, PETSc, and PSBLAS) across a suite of representative matrices. Our results show that Rust's sparse kernels achieve performance comparable to Eigen and PSBLAS, tracking the state-of-the-art for CSC formats, while trailing PETSc's advanced blocked CSR optimizations. By analyzing compile-time monomorphization, SIMD vectorization, and FFI boundaries, we assess the practical impact of Rust's safety model and ecosystem readiness. The study provides concrete, evidence-based guidance for modernizing high-performance numerical software stacks. |
| 2026-06-17 | [Language Models as Interfaces, Not Oracles: A Hybrid LLM-ML System for Pediatric Appendicitis](http://arxiv.org/abs/2606.19183v1) | Soheyl Bateni, Maryam Abdolali | Large language models (LLMs) can make clinical decision support more accessible by interpreting free-text documentation, but their direct use as diagnostic engines is limited by sensitivity to prompts, information order, and plausible but incorrect outputs. Structured machine-learning models offer more stable risk prediction, yet they require tabular inputs that are difficult to integrate with narrative clinical workflows. We present ClaMPAPP (Clinical Language-assisted Machine-learning Pipeline for Appendicitis), a hybrid system that uses an LLM as an interface rather than as the final decision-maker. ClaMPAPP extracts schema-constrained clinical features from note-like narratives, applies deterministic plausibility checks, and passes validated features to an XGBoost classifier trained on clinical, laboratory, and ultrasound variables. We evaluated ClaMPAPP on two independent pediatric appendicitis cohorts from German hospitals and compared it with end-to-end LLM baselines, including open-source and proprietary models. To preserve ground truth while testing free-text input, narratives were generated from structured electronic health records through template rendering and constrained LLM rewriting, with additional sentence-order permutation to assess positional robustness. ClaMPAPP achieved the strongest overall diagnostic performance in both internal and external validation while minimizing missed appendicitis cases, the key safety concern in acute triage. End-to-end LLMs showed unstable sensitivity-specificity trade-offs and greater degradation under narrative reordering. These results support an LLM-as-interface, ML-as-predictor design that separates natural-language usability from predictive inference and provides a more auditable pathway for clinical decision support. |
| 2026-06-17 | [Beyond Safe Data: Pretraining-Stage Alignment with Regular Safety Reflection](http://arxiv.org/abs/2606.19168v1) | Jinhan Li, Kexian Tang et al. | To achieve deeper safety alignment for large language models (LLMs), recent efforts have studied how to push safety interventions earlier into the pretraining stage, primarily by filtering unsafe data or rewriting it into safer forms. We argue that pretraining-stage alignment should go beyond making the data safe: LLMs may compose seemingly benign knowledge and capabilities into unsafe behaviors. To this end, we propose Safety Reflection Pretraining, a pretraining-stage alignment method which regularly inserts short safety reflections into pretraining corpora to integrate self-monitoring directly into language modeling, establishing a foundational capability that is subsequently reinforced by compatible post-training. Our experiments with 1.7B models pretrained on FineWeb-Edu show that Safety Reflection Pretraining improves safety classification accuracy and substantially reduces the success rates of inference-stage and finetuning attacks. Complementary to our real-world experiments, we also introduce a fully controlled synthetic environment, MedSafetyWorld, with a clear definition of safety and a reasoning structure under which models can easily generalize unsafe behaviors from safe data. Ablations in MedSafetyWorld further demonstrate a clear advantage of Safety Reflection Pretraining in preventing models from acting on unsafe behaviors generalized from safe data, compared with data filtering and rewriting. Taken together, our findings suggest that pretraining alignment should not only make the training data safe, but also shape the behaviors that models are likely to acquire from safe data. |
| 2026-06-17 | [CHERI-D: Secure and efficient inline object ID for CHERI temporal memory safety](http://arxiv.org/abs/2606.19055v1) | Yuecheng Wang, Jonathan Woodruff et al. | We propose CHERI-D, an architectural extension to CHERI that supports efficient temporal memory safety. Efficient memory safety is an increasing priority for programming languages, operating systems, and hardware designs, and CHERI is a leading hardware/software system that provides native spatial safety and a foundation for temporal memory safety. Due to CHERI lacking intrinsic architectural support for temporal memory safety, the state-of-the-art CHERI temporal safety solution, Cornucopia Reloaded, is a software-based solution that provides use-after-reallocation (UAR) protections instead of the stronger use-after-free (UAF) mitigation, and suffers performance overhead due to delayed reallocation and revocation. CHERI-D associates object identification (ID) metadata with capability pointers to provide temporal integrity of allocations. CHERI spatial safety allows CHERI-D to store object IDs safely inline with allocation data, potentially within unused fragmentation. Evaluated in simulation and in hardware, CHERI-D significantly reduces the revocation overhead of Cornucopia Reloaded while allowing it to support strict use-after-free mitigation. |
| 2026-06-17 | [A finite element-based eigenvalue analysis for predicting thermal runaway in Li-ion battery packs](http://arxiv.org/abs/2606.18983v1) | Shailendra Rahi, Vinay Dhakal et al. | Thermal runaway remains a critical safety concern in battery systems and other thermally active devices, necessitating reliable methods for predicting the conditions under which thermal runaway may occur. In this work, we develop a finite element framework for assessing thermal stability in systems undergoing transient heat conduction with temperature-dependent internal heat generation. The formulation leads naturally to a generalized eigenvalue problem, wherein the sign of the smallest eigenvalue provides a direct criterion for determining the onset of thermal runaway. This approach enables direct prediction of thermal runaway thresholds in geometrically complicated problems without requiring computationally expensive transient analysis. The proposed methodology is validated through comparison with analytical solutions for a cylindrical Li-ion cell, as well as with a pack of cylindrical cells, demonstrating excellent agreement in both cases. Based on this model, the influence of material properties, boundary conditions, geometric parameters, and spatially varying heat generation on stability limits is examined. A key advantage of this formulation is its effectiveness even for complex geometries where analytical methods become impractical. By providing a systematic and computationally efficient means to identify stability thresholds, the present work offers a practical tool for the thermal design and safety assessment of battery systems. |
| 2026-06-17 | [SciRisk-Bench: A Risk-Dimension-Aware Benchmark for AI4Science Safety](http://arxiv.org/abs/2606.18936v1) | Linghao Feng, Yinqian Sun et al. | Large language models (LLMs) are increasingly embedded in AI for Science (AI4Science) workflows, from scientific question answering and literature analysis to laboratory planning and autonomous discovery. This progress creates an urgent need for safety benchmarks that evaluate not only scientific competence, but also whether models recognize and avoid risks in high-stakes scientific contexts. Existing AI4Science safety datasets cover several disciplines and task formats, leaving the underlying risk dimensions underspecified. We introduce \textbf{SciRisk-Bench}, a benchmark designed to evaluate AI4Science safety from two complementary perspectives: explicit risk dimensions and scientific disciplines. SciRisk-Bench covers 7 disciplines, 31 subdisciplines and 10 risk dimensions. In the experimental section, we evaluate both mainstream LLMs and science-oriented LLMs across risk dimensions, disciplines, and sub-disciplines, enabling fine-grained diagnosis of where scientific models remain unsafe. |
| 2026-06-17 | [Scaling Learning-based AEB with Massive Unlabeled Data](http://arxiv.org/abs/2606.18864v1) | Xiangyu Wang, Yang Zhan et al. | This paper studies how to scale learning-based automatic emergency braking (AEB) with massive unlabeled fleet data under production constraints. Our approach is based on meta-feedback semi-supervised learning (MF-SSL), where a teacher generates pseudo labels for unlabeled driving data and is updated using a small labeled anchor set as safety-critical feedback. In production, anchor ambiguity and labeled-unlabeled mismatch can amplify systematic pseudo-label errors, leading to spurious triggers. We propose a stabilized MF-SSL framework with (i) Noise-Aware Decoupling, which removes ambiguity-prone anchors from the teacher's supervised update path, and (ii) kinematics-gated pseudo-labeling with a teacher conflict penalty to suppress mismatch-induced risk hallucinations on unlabeled data while maintaining broad coverage. Extensive experiments show consistent gains as unlabeled data scale from 1M to 1B windows, improving safety while keeping comfort stable. The 1B-trained student model is deployed to hundreds of thousands of vehicles and validated over \$10^9$ km of driving, achieving a positive-to-false activation ratio exceeding 100:1 and a 35% improvement in accident-free driving mileage over a production rule-only baseline. |
| 2026-06-17 | [Emotional driving: Reference-dependent emotions and risky driving behavior after sporting events](http://arxiv.org/abs/2606.18805v1) | Travis Richardson, Steve Bickley et al. | Using average vehicle speed data in 10-minute increments at the Traffic Message Channel (TMC) location level, along with precise crash timing and location information, we analyze driving behavior around five Florida stadiums before and after NFL and NBA regular season games from 2015 to 2019. We find no evidence of emotional driving following NBA games, but strong and consistent effects following NFL games, concentrated in predicted-close games that end in disappointing home-team losses -- combining high pre-game suspense with negative outcome valence. These games are associated with significant increases in average vehicle speed within 3 km of stadiums during the first post-game hour, dissipating with increasing time and distance from the stadium. Average vehicle speed increases by up to 3 mph relative to predicted-close games that ended in a win -- an effect several times larger than the typical game day versus non-game day speed differential. Overall, our results highlight how the combination of sustained suspense and negative outcome valence in close sporting contests can spill over into risky post-game driving behavior, underscoring the behavioral and public safety implications of affective cues in large-scale sporting events. |
| 2026-06-17 | [A Theory-Guided Advanced Regulatory Control Synthesis for Cooling-Limited Exothermic Semi-Batch Reactors](http://arxiv.org/abs/2606.18799v1) | Chenchen Zhou, Jose Matias | This paper studies theory-guided advanced regulatory control (ARC) synthesis for cooling-limited exothermic semi-batch reactors, whose productivity and thermal safety are governed by changing active constraints. Industrial ARC uses feedback loops, cascades, selectors, feedforward/override logic, and valve-position elements, but signal selection, pairing, interconnection, and tuning remain heuristic. Nonlinear model predictive control (NMPC) gives a systematic constrained-operation workflow, but requires a maintained nonlinear model, state estimator, and online optimizer. We combine finite-horizon minimum-time optimality with local safety analysis to develop a systematic analysis-to-architecture ARC synthesis workflow for cooling-limited semi-batch reactors. Under stated assumptions, the workflow translates boundary-seeking optimality into a cooling-demand valve-position-control (VPC) architecture and translates local safety requirements into near-boundary tuning rules. On a reduced benchmark and an industrial-scale polymerization, ARC is nominally competitive with an implemented nominal-model output-feedback nonlinear model predictive control (OF-NMPC) benchmark using extended Kalman filter (EKF) state estimation. In the studied adverse parameter mismatch and unmodeled fault scenarios, ARC keeps temperature-limit violation at 0%, whereas OF-NMPC either violates the limit or fails to complete the batch. |
| 2026-06-17 | [Differential Equation Inductive Robustness Axiomatization](http://arxiv.org/abs/2606.18685v1) | André Platzer, Long Qian | This article establishes the completeness of an axiomatization for the robust safety of dynamical systems with polynomial differential equations on bounded time horizons. Safety properties of robust systems are uniformly reduced to a sound axiomatization of polynomial invariants, resulting in reliable logical proofs of correctness. Approximate decidability results are also established: there is a computable algorithm such that, given any perturbation parameter $δ$, it either produces a symbolic proof of robust safety (hence correctly decides the dynamical system to be robustly safe), or correctly decides that the system is not robustly safe under a perturbation of level $δ$. In contrast to earlier works, this article crucially leverages results from subanalytic geometry to retain a level of exactness, thereby establishing positive results of provability/decidability allowing for arbitrary bounded (semialgebraic) initial/post conditions even without positive separation at their (topological) boundaries. This enables the generation of proofs of inductive safety beyond finite time horizons for general hybrid dynamical systems. |
| 2026-06-17 | [The Wrong Kind of Right: Quantifying and Localizing Misfired Alignment in LLMs](http://arxiv.org/abs/2606.18656v1) | Naihao Deng, Yiming Feng et al. | Warning: This paper studies stereotypes and biases, and contains potentially disturbing examples, used for illustration purposes only. Our findings should not be interpreted as an argument against alignment. Instead, this paper highlights the need for principled approaches to more advanced alignment. Alignment aims to ensure that large language models (LLMs) behave safely and reliably, including by avoiding unsafe inferences. However, we show that such safety-oriented behaviors can misfire: models may reject warranted conclusions even when they are explicitly supported by context. We call this failure mode misfired alignment, where alignment-induced changes cause LLMs to override explicit evidence. To quantify this phenomenon, specifically on stereotype-related alignment, we introduce VETO, a benchmark consisting of 2,032 BBQ-derived contrastive pairs, and define a new metric, Misfired Alignment Rate (MAR), which measures on a 0 to 100 scale how often a model fails on a stereotype-related question but succeeds on its contrastive counterpart. We benchmark 25 LLMs on VETO, and show that all LLMs, including the most recent ones, exhibit non-trivial (4.7 to 18.9%) MARs while all human participants achieve 0.0% MAR. Controlled priming experiments further show that alignment-induced cues can substantially amplify MAR across LLMs, indicating that these failures are not merely artifacts of individual examples but can be induced by safety-related framing. Mechanistic analyses on open-weight LLMs reveal late-layer suppression of evidence-supported answers, and comparisons between instruct and base LLMs suggest that this suppression emerges after instruction training. These findings show that current alignment methods can overgeneralize surface-level safety cues, to the point of overriding objective evidence, motivating more work on alignment objectives that better preserve contextual grounding. |
| 2026-06-17 | [Gender Bias in LLM Hiring Decisions: Evidence from a Japanese Context and Evaluation of Mitigation Strategies](http://arxiv.org/abs/2606.18649v1) | Serena A. Hoffstedde, Machiko Hirota et al. | Large language models (LLMs) are increasingly deployed in hiring workflows, yet most research on gender bias in LLM hiring decisions has focused on English-language, Western-format resumes. This study examines whether pro-female gender bias extends to a Japanese corporate context and evaluates two practical mitigation strategies. Using a counterfactual resume design with 60 Japanese rirekisho-format resumes, 12 name pairs selected on linguistically grounded gender-signal criteria, and five state-of-the-art LLMs (Claude Sonnet 4.6, GPT-4o, DeepSeek-V3, Gemini 2.5 Flash, Llama 3.3 70B), we conducted 43,200 API calls across baseline, prompt instruction, and privacy filter conditions. A crossed random-effects linear mixed model confirms a significant pro-female bias across all five models, replicating Western findings in a non-Western context. A prompt-level gender-neutrality instruction produces no meaningful reduction in bias. A name-reliance analysis formally identifies the candidate name as the primary gender channel: removing the name from the prompt reduces the female effect by nearly its full magnitude. An unexpected incompatibility between the privacy filter and GPT-4o's content safety filter, resulting in a 42% refusal rate, highlights a practical deployment challenge for name anonymization in LLM-assisted recruitment pipelines. |
| 2026-06-17 | [ROBOSHACKLES: A Safety Dataset for Human-Injury Prevention in Embodied Foundation Models](http://arxiv.org/abs/2606.18632v1) | Zhuowen Yin, Chongyang Liu et al. | Embodied Foundation Models (EFMs) integrate multimodal understanding, future-state reasoning, and executable robot actions. Yet their safety alignment for human-injury prevention remains underexplored, primarily because real-world data of robots harming humans or creating hazardous household situations cannot be safely or ethically collected. To address this challenge, we propose a safety-critical data construction pipeline for human-injury prevention in EFMs.Starting from real DROID observations, our construction pipeline proceeds through scene understanding, hazard-aware image editing, temporal prompt generation, and single-pass rollout synthesis. The temporal prompts specify the expected scene evolution, while Wan2.7 synthesizes realistic robotic rollouts from the edited hazardous states in a single pass. Using this pipeline, we construct ROBOSHACKLES, a 10,000-clip robotic video dataset derived from real DROID observations, spanning two direct-harm and four indirect-harm categories. To ensure dataset quality, we assess task completion and visual quality with automatic metrics, and evaluate six representative EFMs under a refusal-based safety criterion. Results show that all evaluated models produce unsafe actions in the tested safety-critical scenarios, yielding a 100% unsafe action generation rate. ROBOSHACKLES serves as a scalable benchmark and training resource for refusal learning and hazard anticipation before robot action execution.The dataset is publicly available at https://huggingface.co/datasets/YZW00/RoboShackles. |
| 2026-06-17 | [Benchmarking Action Spaces in Reinforcement Learning for Vision-based Robotic Manipulation](http://arxiv.org/abs/2606.18594v1) | Seyed Alireza Azimi, Homayoon Farrahi et al. | In real-world reinforcement learning (RL), the choice of action space can play a key role in shaping motion smoothness, safety, and overall task performance. In this study, we evaluate pose increment, pose velocity, joint position increment, and joint velocity across two vision-based manipulation tasks: object picking and pushing. We train policies in simulation and deploy them to the real world using sim-to-real transfer. We find that action-space representation indeed significantly affects sim-to-real performance. In particular, we find that the joint velocity action space is best for the vision-based picking and pushing tasks in terms of smoothness and final task performance. We also provide practical guidance for RL practitioners in choosing action spaces for both simulation and real-world experiments. |
| 2026-06-16 | [Confident yet Concerned: Inconsistencies in Computing Students' Attitudes on Cybersecurity](http://arxiv.org/abs/2606.18541v1) | Victor Adama, Robert Biddle et al. | Today's young adults are most immersed in technology, leading in feelings of powerlessness in managing online privacy across many platforms, and particularly susceptible to phishing attacks. This raises questions about their general, wide-ranging attitudes towards and management of cybersecurity. How do young, tech-savvy adults approach cybersecurity? We seek a better understanding of their cybersecurity knowledge, attitudes and experiences, in particular in addressing deceptive online communications. We surveyed a group of `lead users': computing university students (n = 236). By combining thematic analysis of open-ended responses with quantitative data, we provide insights into their experiences and perceptions. While students demonstrate reasonable cybersecurity awareness, their cybersecurity experiences vary, and inconsistencies exist around their practices, perceptions of responsibility, and support structures. Findings also reveal four key thematic tensions: 1) Computing students are knowledgeable yet have persistent incorrect beliefs, 2) They learn more about keeping safe from sources outside the classroom, 3) They have limited assistance and have fallen victim to cybercrime, and 4) Many are confident, yet others are concerned about their own safety and responsibility. Through cluster analysis of attitudes, we identify two groups, with one feeling less prepared, less confident, yet expressing a desire to learn more. Established measures of intentions and objective knowledge were correlated to preparedness. Self-efficacy correlated to confidence and predicted cluster membership. |

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



