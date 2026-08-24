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
| 2026-08-21 | [CLEAR: Continuous Latent Adapter Routing for Utility-Preserving LLM Safety Alignment](http://arxiv.org/abs/2608.21278v1) | Chengxiao Wang, Enyi Jiang et al. | Improving the safety of large language models (LLMs) often comes at the expense of utility, as globally applied safety tuning may affect model responses to both harmful and benign inputs. We propose \textbf{C}ontinuous \textbf{L}at\textbf{E}nt \textbf{A}dapter \textbf{R}outing (CLEAR), a conditional safety adaptation framework that uses a lightweight hidden-state gate to continuously control the activation strength of a safety low-rank adapter. CLEAR aims to reduce harmful completions while avoiding unnecessary changes to the frozen backbone that could degrade performance on benign prompts. Experiments on widely used safety and utility benchmarks show that CLEAR improves robustness on HarmBench while reducing the utility degradation observed with globally applied safety tuning such as SFT or standard low-rank adaptation (LoRA). On Llama-3-8B-Instruct, CLEAR reduces HarmBench ASR from 32.3\% to 0.5\%, while retaining most of the base model's utility and achieving up to 7.1 percentage points higher GSM8K accuracy than globally applied SFT or LoRA. These results suggest that CLEAR is a promising mechanism for improving the safety--utility trade-off in LLM alignment. |
| 2026-08-21 | [The Exceedance Design Effect: Effective Sample Size for Thresholds under Clustering](http://arxiv.org/abs/2608.21262v1) | Adam Noonan | Many machine-learning systems set a threshold at a quantile of a calibration set: conformal predictors that promise 90% coverage by drawing their cutoff at the calibration set's 90th percentile, abstention gates that decline to answer when a model's score falls below the calibration set's tenth percentile, safety filters that block any output scoring above the 99th percentile of a reference set. All of them promise that the threshold will hold at the stated rate on new data. The promise assumes the calibration examples are independent, and in modern pipelines they usually are not: they share a prompt, a document, a reasoning trace. Survey statistics has known how to discount correlated data since 1965, by counting how many independent observations a sample is worth, but only for averages. We show that a threshold needs a different count. The count depends on how often clustered scores land on the same side of the threshold, and that changes with where the threshold is set. How similar the scores are as numbers does not enter. We prove a closed-form law for the resulting effective sample size and for the spread of the coverage a deployed system actually sees.   Three consequences follow. The correction now used in the conformal literature is the wrong quantity, and can miss in either direction. A dataset has no single effective sample size. It has one for each level the threshold is set at. And the damage is invisible in coverage averaged over many runs, and fully felt by whoever deploys once. On a released calibration set of 25,028 examples, we measure the reliability of about 1,300. |
| 2026-08-21 | [Toward Vision Language Model-based Assessment of Clinical Quality and Usability of LGE-MR Images for Cardiac Ablation Planning](http://arxiv.org/abs/2608.21180v1) | Bipasha Kundu, Abhishek Chaturvedi et al. | LGE cardiac MRI is widely used for left atrial fibrosis assessment and ablation planning in atrial fibrillation patients as knowledge of fibrotic tissue regions identified from LGE-MRI is critical for catheter ablation. Often, poor quality images used during ablation planning can cause mis-localization of ablation targets, directly impacting procedure safety and outcome. The decision of whether a scan meets the minimum quality threshold for ablation planning is currently made informally by the reviewing radiologist and is not captured by any automated system, yet it is arguably the most safety-critical output of the image quality assessment (IQA) process. However, variations in image quality caused by noise, motion artifacts, and poor boundary definition significantly compromise the reliability of downstream segmentation and clinical decision-making tasks. Manual quality assessment by expert radiologists is subjective and difficult to scale, while existing automated methods produce scalar scores without interpretable clinical reasoning. In this work, we propose a two-stage vision language model (VLM) framework for clinically grounded image quality assessment of left atrial LGE-MRI. In the first stage, a fine-tuned VLM generates structured radiology-style quality reports predicting five radiologist-defined criteria: Noise, Motion Artifact, LA Boundary Accuracy, PV Region Accuracy, and Under-segmentation Severity. In the second stage, a GPT-based reasoning module maps the predicted quality and reports to a structured quality scores and binary clinical usability decision for ablation planning. We curate a dataset of 60 annotated image slice-text pairs from 20 patients and benchmark four state-of-the-art VLM architectures. InternVL2 achieves the highest criterion-level accuracy (Avg ACC=0.65, PLCC=0.79), while DeepSeek achieves perfect clinical usability agreement (Acc=1.00, kappa=1.00). |
| 2026-08-21 | [SRL-MPC: Shape-Aware Reinforcement Learned Model Predictive Control](http://arxiv.org/abs/2608.21175v1) | Ruihua Han, Rui Gao et al. | Safe and efficient shape-aware navigation in heterogeneous crowds and robot fleets remains challenging. Traditional approaches often assume homogeneous robots, sparse workspaces, simplified geometry, offline computation, or handcrafted parameters to make the problem tractable, which limits their deployment in dense crowd scenarios. Toward this end, we propose Shape-Aware Reinforcement Learned Model Predictive Control (SRL-MPC), a method for safe, efficient, and adaptive navigation in crowds with heterogeneous shapes without geometry simplification. To encode shape-aware safety, we formulate high-order control barrier function (HOCBF) constraints from geometric separation features (GSFs) based on support function transformation. A reinforcement learning (RL) framework then learns a neural policy that reads GSFs and outputs real-time MPC parameter updates, enabling the MPC solver to adapt to neighboring crowd geometries. The key advantage of SRL-MPC is that it preserves the safety structure and generalizability of MPC while integrating the adaptability and intelligence of RL. Experiments in randomized crowd scenarios with arbitrary shaped robot fleets demonstrate the effectiveness, scalability, and robustness of SRL-MPC. The results show that SRL-MPC substantially outperforms representative baselines in safety and adaptability. Project website: https://hanruihua.github.io/srl_mpc_project/ |
| 2026-08-21 | [ReFrame: Evidence-Guided Test-Time Safety Alignment in Multimodal Large Language Models](http://arxiv.org/abs/2608.21100v1) | Wenzheng Jiang, Xuankun Rong et al. | While multimodal large language models (MLLMs) extend model capabilities beyond text, they also make safety alignment increasingly challenging. Multimodal safety alignment methods must address cross-modal jailbreaks, safety-awareness failures, and over-sensitive refusals. However, existing methods often rely on retraining or internal-state inspection, limiting their applicability to deployed closed-source MLLMs and motivating test-time safety alignment. We analyze this setting and identify two key obstacles, utility dominance and reasoning inertia, which cause models to overlook latent risks or follow malicious reasoning trajectories. Guided by these insights, we propose ReFrame, a training-free multimodal input reframing framework where two agents share a lightweight locally deployed MLLM: the evidence-generation agent constructs complementary risk and utility evidence, and the rewrite-and-routing agent converts it into a safe proxy prompt and image-routing decision before calling the downstream MLLM, without modifying it or accessing its internal information. Experiments across multiple MLLMs and benchmarks show that ReFrame improves jailbreak defense, safety awareness, and oversensitivity reduction while preserving multimodal utility. |
| 2026-08-21 | [Spike-Killer: Evidence-Gated LLM Assistance for Safe Performance Diagnosis on a Real Windows Workstation](http://arxiv.org/abs/2608.21069v1) | Baocheng Zeng, Jinhao Yang | LLM-assisted agents can synthesize system evidence, propose configuration changes, and automate diagnostic tasks, but their flexibility makes an imprecise action or an intrusive collector an operational risk. We present Spike-Killer, a human-approved workflow for diagnosing frame-time complaints on one real Windows workstation. The workflow treats each action as an evidence-gated transaction: it records the exact target state, classifies risk, preserves a snapshot, verifies a postcondition, and retains failed measurements as first-class evidence.   This experience paper reports a completed same-day study with Counter-Strike 2 as a demanding target application. The evidence bundle contains preserved state snapshots, exploratory microbenchmarks, a ten-run same-state repeatability probe, live telemetry, a repaired over-broad registry action, incompatible presentation-capture attempts, an invalid local replay, and a system-level tracing replacement. Windows Performance Recorder produced two CS2 local-Bot GPU traces of 90.69 and 85.85 seconds; both were attributed to cs2.exe, exposed DxgKrnl Present metadata, and had zero lost ETW buffers or events. These results qualify trace integrity, not performance: the study reports no frame intervals, P99 estimate, or intervention effect. The contribution is an auditable, human-in-the-loop pattern for trustworthy agent assistance on a real workstation, including explicit stop conditions when evidence is insufficient. |
| 2026-08-21 | [Robust Validation to Geometric Perturbations for Autonomous Pose Estimation](http://arxiv.org/abs/2608.21066v1) | Gregoire Theau, Melanie Ducoffe | Deploying autonomous systems in safety-critical domains demands guaranteed robustness against physically plausible geometric perturbations rather than abstract pixel-wise noise. In vision-based navigation and autonomous landing, machine learning components require rigorous validation under dynamic operational conditions such as camera rotations and lighting shifts. Extending findings on the failure of first-order spatial attacks in classification, we show that standard gradient-based heuristics (e.g. APGD) similarly fail on for pose estimation, often performing worse than a simple random sampling baseline.   To overcome these optimization bottlenecks, we reformulate pose estimation robustness within the framework of Global Lipschitzian Optimization (GLO). We argue that GLO offers a principled approach to robust validation, effectively localizing global optima with strong theoretical convergence guarantees. We evaluate this framework on a YOLOv8-Pose keypoint detector with a Perspective-n-Point (PnP) solver against rotation and contrast. In our evaluations, GLO successfully isolates critical failure modes where position deviations exceed safe operational limits, while rapidly pruning the search space by over 80%. To the best of our knowledge, this is the first study to extend geometric robustness validation to continuous keypoint regression and deep object detection, establishing a practical step toward certifying robust autonomous perception. |
| 2026-08-21 | [$Z^2$-ACT: End-to-End Verifiable Agentic Intent Control for Open 6G RAN](http://arxiv.org/abs/2608.21049v1) | Sunder Ali Khowaja, Kapal Dev et al. | With the progression in open and disaggregated 6G radio access networks, it is expected that the system will be able to host multi-vendors. In order to host multi-vendors, it is essential that AI-assisted control loops remain safe, verifiable, and auditable under concurrent operator intents and untrusted model inputs. The existing studies address the agentic coordination, formal intent constraints, zero-trust prompt verification and cryptographic accountability in isolation, which leaves pre-realization safety, continuous semantic verification and cross-domain audit incomplete when used individually. In this regard, we propose zero-knowledge auditable control and zero-trust verifiable agentic intent architecture ($Z^2$-ACT), which integrates the aforementioned four primitives across the non-real-time and near-real-time RICs. We encode the typed Intent Contracts as operator goals while the large language model inputs are only admitted after a practical adversarial intent check. The skill sequences in the proposed study are released only when a self-management gate is satisfied while every successful commit is recorded as a binding commitment with a zero-knowledge proof. Our experimental evaluation on public ColO-RAN measurements compares the full architecture against targeted ablations and a conventional reinforcement-learning baseline. A live large language model is used in the non-real-time path to translate operator intents into Intent Contracts; we report translation accuracy, the rate of invalid or hallucinated contracts, non-real-time latency, and behavior under adversarial or misleading intents. Near-real-time control remains trace-driven on the public KPM sequences. Results indicate improved actuation filtering and attack resilience at modest latency and signaling cost inside the near-real-time envelope. |
| 2026-08-21 | [Evaluating Large Language Model Performance on International Maritime Dangerous Goods Code Compliance](http://arxiv.org/abs/2608.21036v1) | Alexander Thomas, Hubert P. H. Shum et al. | The transport of dangerous goods by sea is a high-consequence activity governed by the International Maritime Dangerous Goods (IMDG) Code, a complex regulatory framework where errors in classification, packaging, stowage, or segregation can result in fire, explosion, toxic release, or loss of life or vessel. Correct compliance requires accurately interpreting hundreds of pages of interacting provisions, updated on a two-year amendment cycle. Practitioners increasingly use Large Language Models (LLMs) as decision-support tools, yet no systematic evaluation exists of whether they can reliably interpret IMDG requirements for safety-critical use.   This paper introduces DGEval, the first benchmark for evaluating LLM knowledge of IMDG Amendment 42-24. Built from expert-written questions on the NCB Hazcheck e-learning platform and structured lookups from the Dangerous Goods List (DGL), it comprises 1,678 questions across multiple-choice, open-ended, DGL lookup, and regulatory identification tasks. We evaluate 13 models from six providers across multiple thinking configurations, including one maritime domain-specific fine-tuned model, and test the effect of web search.   Although the best-performing model exceeds the human practitioner baseline on multiple-choice questions, all models are weakest in the operationally safety-critical areas of stowage, segregation, and regulatory recall. These results indicate that LLMs may support compliance tasks, particularly structured DGL lookups with web search, but unreliability in operational areas and regulatory-text recall means human oversight and authoritative source verification remain necessary before deployment in any safety-critical context. DGEval is designed as a safety assurance instrument to be applied continuously as models evolve, not as a settled characterisation of current capability. |
| 2026-08-21 | [Dorsal Hand Images for Immersive (XR) and Privacy-preserving Age Assurance and Child Safety](http://arxiv.org/abs/2608.21009v1) | Riccardo Bovo, George Loukas et al. | Ensuring that Extended Reality (XR) environments are age-appropriate is an important regulatory and safety challenge. However, current age assurance operates only at registration and cannot verify the age of the active user during a session. Face-based approaches, the dominant solution in social media and adult platforms, are impractical in XR, because they require removing the headset and taking a self-captured image, often on a mobile app. This both breaks immersion and introduces the privacy risk of sharing face pictures with third parties, which leaves XR platforms without a viable path to continuous, in-session and privacy-preserving age assurance. We propose the dorsal part of the hand as an alternative to the face, by exploiting the egocentric cameras that XR headsets inherently and naturally use to capture gesture interactions. To evaluate this, we collect an age- and sex-stratified, ethnodiverse dataset of 436 participants spanning the minor--adult boundary, captured under unconstrained lighting and orientation conditions. To characterise what is achievable with off-the-shelf methods at the minor--adult boundary, we evaluate standard neural network architectures for age assurance at the legally critical 18-year threshold. Analysis confirms performance is robust to skin-tone variation. On this dataset, the challenge-31 operating point achieves zero minor admission, making the system a viable first-stage filter for age assurance. These findings position dorsal hand morphometrics as an effective and more privacy-preserving biometric modality for in-session age assurance in XR. |
| 2026-08-21 | [A Safety-Driven Architectural Framework for Fail-Operational Drone Swarms in Critical Missions](http://arxiv.org/abs/2608.20906v1) | Luiz Giacomossi, Zafer Yigit et al. | The certification of Unmanned Aerial Vehicle (UAV) swarms for safety-critical operations requires verifiable design assurance. Airworthiness standards demand deterministic reliability, whereas multi-agent coordination algorithms execute non-deterministic models. This paper proposes a mixed-criticality architectural framework that applies SAE ARP4754B methods to swarm reconfiguration. First, a hardware-isolated Safety Monitor functions as a Run-Time Assurance (RTA) gateway, decoupling the flight-critical core from the non-deterministic Swarm Manager. Second, the monitor enforces formal safety contracts based on agent Health Vectors derived systematically from a Functional Hazard Assessment (FHA). Third, the framework propagates these Health Vectors to the collective planner to trigger fail-operational task reallocation, enabling intelligent swarm behaviors without compromising flight-critical isolation. Markov reliability modeling demonstrates that the $10^{-7}$ failures per flight hour Hazardous target is theoretically achievable for our SAIL IV scenario, provided the Safety Monitor meets $C_{monitor}>0.9991$, consistent with DAL B CMD/MON implementations. |
| 2026-08-21 | [Scalable Distributed Simulation-Based Testing for Automated Driving Systems](http://arxiv.org/abs/2608.20904v1) | Christian Geller, Benedikt Haas et al. | Virtual scenario-based testing is a key enabler for validating automated driving systems (ADS) and intelligent transport systems (ITS). However, executing large-scale test suites involving possibly thousands of scenarios remains labor-intensive and difficult to scale. This paper presents an end-to-end, DevOps-driven framework that automates build, deployment, and distributed execution of CARLA-based scenario tests of an ADS on a lightweight Kubernetes cluster. ROS 2 applications are packaged as standardized Kubernetes Helm charts generated from repository specifications, while entire simulation environments are composed declaratively via dynamic Helmfile manifests. The paper describes how a distributed testing workflow can be implemented in Argo Workflows to provision environments, aggregate and batch OpenSCENARIO test cases from configurable sources, execute scenarios in parallel across cluster nodes, and collect logs and resource metrics. In an evaluation on a multi-node K3s cluster running 200 scenarios, the best configuration speeds up end-to-end workflow time by more than a factor of eight compared to a sequential baseline. The results demonstrate significant gains in end-to-end execution time and quantify trade-offs between parallelism, orchestration overhead, and cluster stability. The framework is further demonstrated in a real-world ADS test application with connections to scenario sources and downstream evaluation modules. This demonstrates that the approach provides a strong foundation not only for scalable simulation testing, but also for generating traceable evidence that can support safety arguments. |
| 2026-08-21 | [A Collaborative Multi-Modality Interaction for VLA-based End-to-End Autonomous Driving](http://arxiv.org/abs/2608.20890v1) | Jingtao Sun, Xiaohai He et al. | Vision-Language-Action (VLA) models have emerged as a powerful paradigm for end-to-end autonomous driving by jointly integrating perception, reasoning, and decision making within a unified multimodal framework. However, most existing VLA models formulate end-to-end autonomous driving as a visual question answering task, leading to unreliable and less interpretable decision reasoning. In addition, they fail to establish effective multi-modal interaction across heterogeneous sensors, thereby limiting robust scene perception and reliable driving reasoning in long-tail driving scenarios. To this end, we propose a robust VLA-based end-to-end autonomous driving system that combines multi-modality interaction with multi-trajectory planning and optimization, enabling more reliable, interpretable, and safer driving decisions. Our method comprises three core components: (1) Affinity-Guided Optimal Transport for main-auxiliary modality two-way interaction; (2) Distribution-Consistent Modality Transfer for heterogeneous modality distribution transfer and cross-modal interaction; (3) Multi-modal Multi-Trajectory Planning along with Perception-Oriented Trajectory Refinement for better driving decisions to long-tail driving scenarios. Experimental results in open-loop and closed-loop datasets demonstrate improvements in safety long-horizon driving reasoning and road scene perception over existing driving systems, highlighting the ability of our mutli-modality interaction and multi-trajectory planning and optimization for scalable VLA-based systems. |
| 2026-08-21 | [Multi-Modal Traffic Sign Detection with Semantic Attributes for Autonomous Driving](http://arxiv.org/abs/2608.20874v1) | Meda Lazar, Sourab Sridhar et al. | Reliable traffic sign detection is a prerequisite for the global deployment of autonomous driving systems, where regulatory compliance and road safety depend on perceiving signs correctly across regions, ranges, and weather conditions. Despite recent progress, vision-based methods continue to face three fundamental limitations: poor cross-regional generalization due to high diversity across countries, degraded performance on small-object detection at long ranges (traffic signs occupy as little as $10{\times}10$ pixels at 200m), and fragile temporal tracking under the strongly non-linear perspective distortion that occurs as a vehicle approaches a sign. In this paper, we address the problem of robust, long-range, region-agnostic traffic sign perception by combining camera and Light Detection and Ranging (LiDAR) sensing. We present a multi-modal detection framework whose Intensity-Aware Deformable Fusion module aligns retro-reflective LiDAR cues with camera features, anchoring detection on geometric invariants rather than region-specific visual appearance. We further introduce a dual motion-model tracker that explicitly accounts for non-linear perspective transformations during vehicle approach, substantially improving temporal consistency over linear motion assumptions. Additionally, we develop a semantic attribute classification pipeline that estimates occlusion level, readability, sign embeddedness, and road relevance, providing actionable context to downstream planning. Extensive evaluation on our dataset, spanning 60+ countries and 2,500+ hours of driving data, shows that the proposed pipeline achieves an Object Miss Ratio (OMR) of 0.49% across 221,068 evaluation sequences, demonstrating globally generalizable traffic sign perception in commercial-grade autonomous driving systems. |
| 2026-08-21 | [Coverage-Driven Verification for Safety-by-Design in AI-Based Collision Avoidance Systems](http://arxiv.org/abs/2608.20864v1) | Thomas Stefani, Johann Maximilian Christensen et al. | Artificial Intelligence (AI) offers significant potential for future aviation systems; however, its integration into safety-critical applications requires compliance with the aviation sector's stringent safety standards. For AI and Machine Learning (ML)-based systems, the European Union Aviation Safety Agency (EASA) emphasizes the need to demonstrate the representativeness and completeness of the Operational Design Domain (ODD) and the associated data distributions used during development and verification. Despite this requirement, a structured engineering process for defining target distributions and evaluating representativeness within ODDs remains largely unexplored. This work presents a method for representativeness assessment of AI/ML constituent ODDs in the context of aviation safety assurance. Starting from the methodical identification of suitable target distributions, a process flow is proposed that guides developers from ODD definition and parameter distribution modeling to the quantitative assessment and interpretation of coverage results with respect to EASA's learning assurance objectives. As quantitative measures, the chi-squared goodness-of-fit test is examined and found unsuitable for the large data sets arising in this setting, leading to the adoption of the Kullback--Leibler divergence and Cramér's $V$ for the representativeness assessment. The method is demonstrated using the example of AI-based airborne collision avoidance, employing experimental data from previous Horizontal Collision Avoidance System (HCAS) and Vertical Collision Avoidance System (VCAS) simulations. The results illustrate how statistical distribution comparison methods can support the assessment of representativeness for safety-critical AI applications and contribute toward a systematic Safety-by-Design AI engineering process aligned with emerging EASA guidance. |
| 2026-08-21 | [Certified Multi-Turn Robustness for LLM Safety via Compositional Bounds and Safety Persistence](http://arxiv.org/abs/2608.20820v1) | Yang Liu, Bin Chong et al. | Large language models (LLMs) are vulnerable to multi-turn jailbreak attacks that progressively manipulate conversation context. Existing certified robustness methods are limited to single-turn inputs; naive multi-turn composition yields bounds that degrade exponentially in the number of turns. We introduce Multi-Turn Certified Robustness (MTCR), a framework that models conversational safety via State-Adversarial MDPs and defines $k$-turn certified robustness as the worst-case safety probability across $k$ adversarial turns. MTCR comprises: (i) compositional certification via embedding-space mode decomposition, yielding tighter certified lower bounds than naive multiplication; (ii) $(α,β)$-safety persistence, improving the degradation rate from $\underline{p}^{k}$ to $β^k$ (with $β> \underline{p}$) and yielding interpretable horizon estimates; (iii) matching information-theoretic upper bounds establishing tightness; and (iv) a unified algorithm combining these results. Experiments on six LLMs under $ε$-bounded and Crescendo-style attacks confirm that empirical safety consistently exceeds the certified bounds. |
| 2026-08-21 | [Automated Trajectory Evaluation for Mobile Agents via Step-Level Consequence Reasoning and Aggregation](http://arxiv.org/abs/2608.20797v1) | Pengshuai Yang, Zijing Gao et al. | Evaluating language-guided mobile agents has recently shifted from rule-based to model-based approaches to achieve scalable and automated assessments. However, existing holistic evaluation paradigms process entire trajectories at once, leading to substantial context overload. Moreover, they primarily focus on task completion while overlooking operational safety. To address these limitations, we introduce CRATE, a novel two-stage VLM-as-judge framework for automated mobile agent evaluation that is compatible with both open- and closed-source models. Leveraging a step-level consequence reasoning mechanism, CRATE independently extracts task-relevant visual clues and infers action-conditioned state changes at each step. The resulting step-level textual evidence is then synthesized through trajectory-level aggregation to deliver an evidence-grounded evaluation of task completion. Building upon this evaluation scheme, we further extend CRATE to CRATE-S for operational safety assessment. Extensive experiments validate the effectiveness and robustness of both CRATE and CRATE-S. Powered by Qwen2.5-VL-72B-Instruct, CRATE achieves an F1-score of 0.833 on AndroidWorld (outperforming SPA-Bench by 20%), while CRATE-S reaches an F1-score of 0.697 on MobileRisk, demonstrating strong alignment with benchmark ground truths. Code is available at https://anonymous.4open.science/r/CRATE-D580. |
| 2026-08-21 | [Beyond Endpoint Gains: A Weight-Delta Audit of Medical Specialization](http://arxiv.org/abs/2608.20768v1) | Praphul Singh, Shanu Kumar et al. | Specialist language models are usually understood through endpoint gains: the generalist scores lower, the specialist scores higher, and the difference is treated as evidence of specialization. This leaves the released update itself largely unexamined. We propose a paired weight-delta path audit and apply it to two public, aligned generalist-to-medical-specialist checkpoint pairs: Gemma-3-4B-IT to MedGemma-4B-IT and Qwen2.5-7B-Instruct to HuatuoGPT-o1-7B. In both pairs, the full decoder-side update strongly reconstructs measured medical benchmark movement (0.974 and 1.183 endpoint-normalized retention), making each decoder delta an appropriate substrate for the audit. Yet the movement is not cleanly localized. MLP is the strongest broad component family in both pairs, but mixed off-domain movements, 10-seed matched controls, and endpoint-anchored rollbacks prevent a unique coarse-family explanation. The audit therefore separates update-level reconstruction from component-level explanation. Its claims concern text-only multiple-choice benchmark movement, not clinical validation, repair, or circuit-level mechanism. |
| 2026-08-21 | [Thermal scaling laws for open-water swimming](http://arxiv.org/abs/2608.20719v1) | Henry van den Bedem, Ellen Kuhl | Open-water swimming defines a thermal phase-boundary problem in which metabolic heat production competes with environmental heat loss. We derive a scaling law that predicts the critical water temperature and shows how body size, swim pace, and insulation shift this boundary. Longitudinal warm- and cold-water data reveal transient dynamics that exceed single-compartment predictions, but emerge naturally from core--peripheral physiology. Together, our results suggest that thermal safety depends on swimmer-specific characteristics and swimming conditions, not on water temperature alone. |
| 2026-08-20 | [Logic-VLA: A Temporal Logic Conditioned Vision-Language-Action Model](http://arxiv.org/abs/2608.20556v1) | Celina Shiyu Wang, Yiqi Zhao et al. | Vision-language-action (VLA) models can follow natural-language (NL) task instructions, but such instructions may not precisely specify safety-critical or spatiotemporal requirements on the resulting behavior. We introduce Logic-VLA, a formal-requirement-aware VLA that conditions on Signal Temporal Logic (STL) specifications supplied at inference time. Logic-VLA uses a syntax-graph-based STL encoder pre-trained to capture temporal logic semantics. Policy adaptation proceeds in two stages: STL-conditioned supervised fine-tuning on satisfying demonstrations is followed by trajectory-level preference optimization over matched satisfying-violating rollout pairs using a flow-matching surrogate for Identity Preference Optimization. This formulation improves formal requirement satisfaction while preserving the nominal NL task. We evaluate Logic-VLA in closed-loop quadcopter navigation simulation across randomized photorealistic environments and test generalization to STL formulas unseen during training. Across the evaluation benchmarks, Logic-VLA improves STL satisfaction rate over an STL-blind base policy by 24.8 to 40.7 percentage points (pp) while reducing nominal NL task success by at most 1.8 pp, showing that a single VLA can adapt its behavior to varying formal requirements without requiring a separate policy for each specification. |

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



