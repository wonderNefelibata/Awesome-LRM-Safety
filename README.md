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
| 2025-12-05 | [A Residual Variance Matching Recursive Least Squares Filter for Real-time UAV Terrain Following](http://arxiv.org/abs/2512.05918v1) | Xiaobo Wu, Youmin Zhang | Accurate real-time waypoints estimation for the UAV-based online Terrain Following during wildfire patrol missions is critical to ensuring flight safety and enabling wildfire detection. However, existing real-time filtering algorithms struggle to maintain accurate waypoints under measurement noise in nonlinear and time-varying systems, posing risks of flight instability and missed wildfire detections during UAV-based terrain following. To address this issue, a Residual Variance Matching Recursive Least Squares (RVM-RLS) filter, guided by a Residual Variance Matching Estimation (RVME) criterion, is proposed to adaptively estimate the real-time waypoints of nonlinear, time-varying UAV-based terrain following systems. The proposed method is validated using a UAV-based online terrain following system within a simulated terrain environment. Experimental results show that the RVM-RLS filter improves waypoints estimation accuracy by approximately 88$\%$ compared with benchmark algorithms across multiple evaluation metrics. These findings demonstrate both the methodological advances in real-time filtering and the practical potential of the RVM-RLS filter for UAV-based online wildfire patrol. |
| 2025-12-05 | [Log-linear Dynamic Inversion for Thrusting Spacecraft on SE2(3)](http://arxiv.org/abs/2512.05888v1) | Micah K. Condie, Abigaile E. Woodbury et al. | We show that the dynamics of a thrusting spacecraft can be embedded in the Lie group SE2(3) in a form that is group-affine with application of a feed-forward control law. This structure implies that the configuration-tracking error evolves exactly linearly in the associated Lie algebra coordinates (log-linear dynamics), rather than arising from a local linearization of the nonlinear system. As a result, a broad class of linear analysis and synthesis tools becomes directly applicable to powered spacecraft motion on SE2(3). A simple numerical example confirms that the error predicted by the linear Lie-algebra dynamics matches the error computed from the full nonlinear system, illustrating the exact log-linear behavior. This foundational property opens a path toward rigorous tools for satellite docking, autonomous rendezvous and proximity operations, robust controller design, and convex safety certification-capabilities that are difficult to achieve with classical local linearizations such as Tschauner-Hempel/Yamanaka-Ankersen (TH/YA). |
| 2025-12-05 | [VRSA: Jailbreaking Multimodal Large Language Models through Visual Reasoning Sequential Attack](http://arxiv.org/abs/2512.05853v1) | Shiji Zhao, Shukun Xiong et al. | Multimodal Large Language Models (MLLMs) are widely used in various fields due to their powerful cross-modal comprehension and generation capabilities. However, more modalities bring more vulnerabilities to being utilized for jailbreak attacks, which induces MLLMs to output harmful content. Due to the strong reasoning ability of MLLMs, previous jailbreak attacks try to explore reasoning safety risk in text modal, while similar threats have been largely overlooked in the visual modal. To fully evaluate potential safety risks in the visual reasoning task, we propose Visual Reasoning Sequential Attack (VRSA), which induces MLLMs to gradually externalize and aggregate complete harmful intent by decomposing the original harmful text into several sequentially related sub-images. In particular, to enhance the rationality of the scene in the image sequence, we propose Adaptive Scene Refinement to optimize the scene most relevant to the original harmful query. To ensure the semantic continuity of the generated image, we propose Semantic Coherent Completion to iteratively rewrite each sub-text combined with contextual information in this scene. In addition, we propose Text-Image Consistency Alignment to keep the semantical consistency. A series of experiments demonstrates that the VRSA can achieve a higher attack success rate compared with the state-of-the-art jailbreak attack methods on both the open-source and closed-source MLLMs such as GPT-4o and Claude-4.5-Sonnet. |
| 2025-12-05 | [Safe Output Regulation of Coupled Hyperbolic PDE-ODE Systems](http://arxiv.org/abs/2512.05822v1) | Ji Wang, Miroslav Krstic | This paper presents a safe output regulation control strategy for a class of systems modeled by a coupled $2\times 2$ hyperbolic PDE-ODE structure, subject to fully distributed disturbances throughout the system. A state-feedback controller is developed by the {nonovershooting backstepping} method to simultaneously achieve exponential output regulation and enforce safety constraints on the system output, which is the state furthest from the control input. To handle unmeasured PDE states and external disturbances, a state observer and a disturbance estimator are designed. Explicit bounds on the estimation errors are derived and used to construct a robust safe regulator that accounts for the uncertainties. The proposed control scheme guarantees that: 1) If the system output is initially within the safe region, it remains there; otherwise, it will be rescued to the safety within a prescribed time; 2) The output tracking error converges to zero exponentially; 3) The observer accurately estimates both the distributed states and external disturbances, with estimation errors converging to zero exponentially; 4) All signals in the closed-loop system remain bounded. The effectiveness of the proposed method is demonstrated through a UAV delivery scenario with a cable-suspended payload, where the payload is regulated to track a desired reference while avoiding collisions with barriers. |
| 2025-12-05 | [Optimal Safety-Aware Scheduling for Multi-Agent Aerial 3D Printing with Utility Maximization under Dependency Constraints](http://arxiv.org/abs/2512.05815v1) | Marios-Nektarios Stamatopoulos, Shridhar Velhal et al. | This article presents a novel coordination and task-planning framework to enable the simultaneous conflict-free collaboration of multiple unmanned aerial vehicles (UAVs) for aerial 3D printing. The proposed framework formulates an optimization problem that takes a construction mission divided into sub-tasks and a team of autonomous UAVs, along with limited volume and battery. It generates an optimal mission plan comprising task assignments and scheduling while accounting for task dependencies arising from the geometric and structural requirements of the 3D design, inter-UAV safety constraints, material usage, and total flight time of each UAV. The potential conflicts occurring during the simultaneous operation of the UAVs are addressed at a segment level by dynamically selecting the starting time and location of each task to guarantee collision-free parallel execution. An importance prioritization is proposed to accelerate the computation by guiding the solution toward more important tasks. Additionally, a utility maximization formulation is proposed to dynamically determine the optimal number of UAVs required for a given mission, balancing the trade-off between minimizing makespan and the deployment of excess agents. The proposed framework's effectiveness is evaluated through a Gazebo-based simulation setup, where agents are coordinated by a mission control module allocating the printing tasks based on the generated optimal scheduling plan while remaining within the material and battery constraints of each UAV. |
| 2025-12-05 | [Global stability of vehicle-with-driver dynamics via Sum-of-Squares programming](http://arxiv.org/abs/2512.05806v1) | Martino Gulisano, Marco Gabiccini | This work estimates safe invariant subsets of the Region of Attraction (ROA) for a seven-state vehicle-with-driver system, capturing both asymptotic stability and the influence of state-safety bounds along the system trajectory. Safe sets are computed by optimizing Lyapunov functions through an original iterative Sum-of-Squares (SOS) procedure. The method is first demonstrated on a two-state benchmark, where it accurately recovers a prescribed safe region as the 1-level set of a polynomial Lyapunov function. We then describe the distinguishing characteristics of the studied vehicle-with-driver system: the control dynamics mimic human driver behavior through a delayed preview-tracking model that, with suitable parameter choices, can also emulate digital controllers. To enable SOS optimization, a polynomial approximation of the nonlinear vehicle model is derived, together with its operating-envelope constraints. The framework is then applied to understeering and oversteering scenarios, and the estimated safe sets are compared with reference boundaries obtained from exhaustive simulations. The results show that SOS techniques can efficiently deliver Lyapunov-defined safe regions, supporting their potential use for real-time safety assessment, for example as a supervisory layer for active vehicle control. |
| 2025-12-05 | [ARGUS: Defending Against Multimodal Indirect Prompt Injection via Steering Instruction-Following Behavior](http://arxiv.org/abs/2512.05745v1) | Weikai Lu, Ziqian Zeng et al. | Multimodal Large Language Models (MLLMs) are increasingly vulnerable to multimodal Indirect Prompt Injection (IPI) attacks, which embed malicious instructions in images, videos, or audio to hijack model behavior. Existing defenses, designed primarily for text-only LLMs, are unsuitable for countering these multimodal threats, as they are easily bypassed, modality-dependent, or generalize poorly. Inspired by activation steering researches, we hypothesize that a robust, general defense independent of modality can be achieved by steering the model's behavior in the representation space. Through extensive experiments, we discover that the instruction-following behavior of MLLMs is encoded in a subspace. Steering along directions within this subspace can enforce adherence to user instructions, forming the basis of a defense. However, we also found that a naive defense direction could be coupled with a utility-degrading direction, and excessive intervention strength harms model performance. To address this, we propose ARGUS, which searches for an optimal defense direction within the safety subspace that decouples from the utility degradation direction, further combining adaptive strength steering to achieve a better safety-utility trade-off. ARGUS also introduces lightweight injection detection stage to activate the defense on-demand, and a post-filtering stage to verify defense success. Experimental results show that ARGUS can achieve robust defense against multimodal IPI while maximally preserving the MLLM's utility. |
| 2025-12-05 | [Evaluating Concept Filtering Defenses against Child Sexual Abuse Material Generation by Text-to-Image Models](http://arxiv.org/abs/2512.05707v1) | Ana-Maria Cretu, Klim Kireev et al. | We evaluate the effectiveness of child filtering to prevent the misuse of text-to-image (T2I) models to create child sexual abuse material (CSAM). First, we capture the complexity of preventing CSAM generation using a game-based security definition. Second, we show that current detection methods cannot remove all children from a dataset. Third, using an ethical proxy for CSAM (a child wearing glasses, hereafter, CWG), we show that even when only a small percentage of child images are left in the training dataset, there exist prompting strategies that generate CWG from a child-filtered T2I model using only a few more queries than when the model is trained on the unfiltered data. Fine-tuning the filtered model on child images further reduces the additional query overhead. We also show that reintroducing a concept is possible via fine-tuning even if filtering is perfect. Our results demonstrate that current filtering methods offer limited protection to closed-weight models and no protection to open-weight models, while reducing the generality of the model by hindering the generation of child-related concepts or changing their representation. We conclude by outlining challenges in conducting evaluations that establish robust evidence on the impact of AI safety mitigations for CSAM. |
| 2025-12-05 | [LA-RL: Language Action-guided Reinforcement Learning with Safety Guarantees for Autonomous Highway Driving](http://arxiv.org/abs/2512.05686v1) | Yiming Shu, Jiahui Xu et al. | Autonomous highway driving demands a critical balance between proactive, efficiency-seeking behavior and robust safety guarantees. This paper proposes Language Action-guided Reinforcement Learning (LA-RL) with Safety Guarantees, a novel framework that integrates the semantic reasoning of large language models (LLMs) into the actor-critic architecture with an improved safety layer. Within this framework, task-specific reward shaping harmonizes the dual objectives of maximizing driving efficiency and ensuring safety, guiding decision-making based on both environmental insights and clearly defined goals. To enhance safety, LA-RL incorporates a safety-critical planner that combines model predictive control (MPC) with discrete control barrier functions (DCBFs). This layer formally constrains the LLM-informed policy to a safe action set, employs a slack mechanism that enhances solution feasibility, prevents overly conservative behavior and allows for greater policy exploration without compromising safety. Extensive experiments demonstrate that it significantly outperforms several current state-of-the-art methods, offering a more adaptive, reliable, and robust solution for autonomous highway driving. Compared to existing SOTA, it achieves approximately 20$\%$ higher success rate than the knowledge graph (KG) based baseline and about 30$\%$ higher than the retrieval augmented generation (RAG) based baseline. In low-density environments, LA-RL achieves a 100$\%$ success rate. These results confirm its enhanced exploration of the state-action space and its ability to autonomously adopt more efficient, proactive strategies in complex, mixed-traffic highway environments. |
| 2025-12-05 | [Scenario-aware Uncertainty Quantification for Trajectory Prediction with Statistical Guarantees](http://arxiv.org/abs/2512.05682v1) | Yiming Shu, Jiahui Xu et al. | Reliable uncertainty quantification in trajectory prediction is crucial for safety-critical autonomous driving systems, yet existing deep learning predictors lack uncertainty-aware frameworks adaptable to heterogeneous real-world scenarios. To bridge this gap, we propose a novel scenario-aware uncertainty quantification framework to provide the predicted trajectories with prediction intervals and reliability assessment. To begin with, predicted trajectories from the trained predictor and their ground truth are projected onto the map-derived reference routes within the Frenet coordinate system. We then employ CopulaCPTS as the conformal calibration method to generate temporal prediction intervals for distinct scenarios as the uncertainty measure. Building upon this, within the proposed trajectory reliability discriminator (TRD), mean error and calibrated confidence intervals are synergistically analyzed to establish reliability models for different scenarios. Subsequently, the risk-aware discriminator leverages a joint risk model that integrates longitudinal and lateral prediction intervals within the Frenet coordinate to identify critical points. This enables segmentation of trajectories into reliable and unreliable segments, holding the advantage of informing downstream planning modules with actionable reliability results. We evaluated our framework using the real-world nuPlan dataset, demonstrating its effectiveness in scenario-aware uncertainty quantification and reliability assessment across diverse driving contexts. |
| 2025-12-05 | [MedTutor-R1: Socratic Personalized Medical Teaching with Multi-Agent Simulation](http://arxiv.org/abs/2512.05671v1) | Zhitao He, Haolin Yang et al. | The significant gap between rising demands for clinical training and the scarcity of expert instruction poses a major challenge to medical education. With powerful capabilities in personalized guidance, Large Language Models (LLMs) offer a promising solution to bridge this gap. However, current research focuses mainly on one-on-one knowledge instruction, overlooking collaborative reasoning, a key skill for students developed in teamwork like ward rounds. To this end, we develop ClinEdu, a multi-agent pedagogical simulator with personality-driven patients and diverse student cohorts, enabling controlled testing of complex pedagogical processes and scalable generation of teaching data. Based on ClinEdu, we construct ClinTeach, a large Socratic teaching dialogue dataset that captures the complexities of group instruction. We then train MedTutor-R1, the first multimodal Socratic tutor designed for one-to-many instruction in clinical medical education. MedTutor-R1 is first instruction-tuned on our ClinTeach dataset and then optimized with reinforcement learning, using rewards derived from a three-axis rubric, covering structural fidelity, analytical quality, and clinical safety, to refine its adaptive Socratic strategies. For authentic in-situ assessment, we use simulation-based interactive evaluation that redeploys the tutor back into ClinEdu. Experimental results demonstrate that our MedTutor-R1 outperforms the base model by over 20% in average pedagogical score and is comparable to o3, while also exhibiting high adaptability in handling a varying number of students. This promising performance underscores the effectiveness of our pedagogical simulator, ClinEdu. |
| 2025-12-05 | [Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs](http://arxiv.org/abs/2512.05648v1) | Igor Shilov, Alex Cloud et al. | Large Language Models increasingly possess capabilities that carry dual-use risks. While data filtering has emerged as a pretraining-time mitigation, it faces significant challenges: labeling whether data is harmful is expensive at scale, and given improving sample efficiency with larger models, even small amounts of mislabeled content could give rise to dangerous capabilities. To address risks associated with mislabeled harmful content, prior work proposed Gradient Routing (Cloud et al., 2024) -- a technique that localizes target knowledge into a dedicated subset of model parameters so they can later be removed. We explore an improved variant of Gradient Routing, which we call Selective GradienT Masking (SGTM), with particular focus on evaluating its robustness to label noise. SGTM zero-masks selected gradients such that target domain examples only update their dedicated parameters. We test SGTM's effectiveness in two applications: removing knowledge of one language from a model trained on a bilingual synthetic dataset, and removing biology knowledge from a model trained on English Wikipedia. In both cases SGTM provides better retain/forget trade-off in the presence of labeling errors compared to both data filtering and a previously proposed instantiation of Gradient Routing. Unlike shallow unlearning approaches that can be quickly undone through fine-tuning, SGTM exhibits strong robustness to adversarial fine-tuning, requiring seven times more fine-tuning steps to reach baseline performance on the forget set compared to a finetuning-based unlearning method (RMU). Our results suggest SGTM provides a promising pretraining-time complement to existing safety mitigations, particularly in settings where label noise is unavoidable. |
| 2025-12-05 | [An Integrated System for WEEE Sorting Employing X-ray Imaging, AI-based Object Detection and Segmentation, and Delta Robot Manipulation](http://arxiv.org/abs/2512.05599v1) | Panagiotis Giannikos, Lampis Papakostas et al. | Battery recycling is becoming increasingly critical due to the rapid growth in battery usage and the limited availability of natural resources. Moreover, as battery energy densities continue to rise, improper handling during recycling poses significant safety hazards, including potential fires at recycling facilities. Numerous systems have been proposed for battery detection and removal from WEEE recycling lines, including X-ray and RGB-based visual inspection methods, typically driven by AI-powered object detection models (e.g., Mask R-CNN, YOLO, ResNets). Despite advances in optimizing detection techniques and model modifications, a fully autonomous solution capable of accurately identifying and sorting batteries across diverse WEEEs types has yet to be realized. In response to these challenges, we present our novel approach which integrates a specialized X-ray transmission dual energy imaging subsystem with advanced pre-processing algorithms, enabling high-contrast image reconstruction for effective differentiation of dense and thin materials in WEEE. Devices move along a conveyor belt through a high-resolution X-ray imaging system, where YOLO and U-Net models precisely detect and segment battery-containing items. An intelligent tracking and position estimation algorithm then guides a Delta robot equipped with a suction gripper to selectively extract and properly discard the targeted devices. The approach is validated in a photorealistic simulation environment developed in NVIDIA Isaac Sim and on the real setup. |
| 2025-12-05 | [PAC One-Step Safety Certification for Black-Box Discrete-Time Stochastic Systems](http://arxiv.org/abs/2512.05549v1) | Taoran Wu, Dominik Wagner et al. | This paper investigates the problem of safety certification for black-box discrete-time stochastic systems, where both the system dynamics and disturbance distributions are unknown, and only sampled data are available. Under such limited information, ensuring robust or classical quantitative safety over finite or infinite horizons is generally infeasible. To address this challenge, we propose a data-driven framework that provides theoretical one-step safety guarantees in the Probably Approximately Correct (PAC) sense. This one-step guarantee can be applied recursively at each time step, thereby yielding step-by-step safety assurances over extended horizons. Our approach formulates barrier certificate conditions based solely on sampled data and establishes PAC safety guarantees by leveraging the VC dimension, scenario approaches, Markov's inequality, and Hoeffding's inequality. Two sampling procedures are proposed, and three methods are proposed to derive PAC safety guarantees. The properties and comparative advantages of these three methods are thoroughly discussed. Finally, the effectiveness of the proposed methods are demonstrated through several numerical examples. |
| 2025-12-05 | [Matching Ranks Over Probability Yields Truly Deep Safety Alignment](http://arxiv.org/abs/2512.05518v1) | Jason Vega, Gagandeep Singh | A frustratingly easy technique known as the prefilling attack has been shown to effectively circumvent the safety alignment of frontier LLMs by simply prefilling the assistant response with an affirmative prefix before decoding. In response, recent work proposed a supervised fine-tuning (SFT) defense using data augmentation to achieve a \enquote{deep} safety alignment, allowing the model to generate natural language refusals immediately following harmful prefills. Unfortunately, we show in this work that the "deep" safety alignment produced by such an approach is in fact not very deep. A generalization of the prefilling attack, which we refer to as the Rank-Assisted Prefilling (RAP) attack, can effectively extract harmful content from models fine-tuned with the data augmentation defense by selecting low-probability "harmful" tokens from the top 20 predicted next tokens at each step (thus ignoring high-probability "refusal" tokens). We argue that this vulnerability is enabled due to the "gaming" of the SFT objective when the target distribution entropies are low, where low fine-tuning loss is achieved by shifting large probability mass to a small number of refusal tokens while neglecting the high ranks of harmful tokens. We then propose a new perspective on achieving deep safety alignment by matching the token ranks of the target distribution, rather than their probabilities. This perspective yields a surprisingly simple fix to the data augmentation defense based on regularizing the attention placed on harmful prefill tokens, an approach we call PRefill attEntion STOpping (PRESTO). Adding PRESTO yields up to a 4.7x improvement in the mean StrongREJECT score under RAP attacks across three popular open-source LLMs, with low impact to model utility. |
| 2025-12-05 | [GRASP: Graph Reasoning Agents for Systems Pharmacology with Human-in-the-Loop](http://arxiv.org/abs/2512.05502v1) | Omid Bazgir, Vineeth Manthapuri et al. | Quantitative Systems Pharmacology (QSP) modeling is essential for drug development but it requires significant time investment that limits the throughput of domain experts. We present \textbf{GRASP} -- a multi-agent, graph-reasoning framework with a human-in-the-loop conversational interface -- that encodes QSP models as typed biological knowledge graphs and compiles them to executable MATLAB/SimBiology code while preserving units, mass balance, and physiological constraints. A two-phase workflow -- \textsc{Understanding} (graph reconstruction of legacy code) and \textsc{Action} (constraint-checked, language-driven modification) -- is orchestrated by a state machine with iterative validation. GRASP performs breadth-first parameter-alignment around new entities to surface dependent quantities and propose biologically plausible defaults, and it runs automatic execution/diagnostics until convergence. In head-to-head evaluations using LLM-as-judge, GRASP outperforms SME-guided CoT and ToT baselines across biological plausibility, mathematical correctness, structural fidelity, and code quality (\(\approx\)9--10/10 vs.\ 5--7/10). BFS alignment achieves F1 = 0.95 for dependency discovery, units, and range. These results demonstrate that graph-structured, agentic workflows can make QSP model development both accessible and rigorous, enabling domain experts to specify mechanisms in natural language without sacrificing biomedical fidelity. |
| 2025-12-05 | [SEA-SafeguardBench: Evaluating AI Safety in SEA Languages and Cultures](http://arxiv.org/abs/2512.05501v1) | Panuthep Tasawong, Jian Gang Ngui et al. | Safeguard models help large language models (LLMs) detect and block harmful content, but most evaluations remain English-centric and overlook linguistic and cultural diversity. Existing multilingual safety benchmarks often rely on machine-translated English data, which fails to capture nuances in low-resource languages. Southeast Asian (SEA) languages are underrepresented despite the region's linguistic diversity and unique safety concerns, from culturally sensitive political speech to region-specific misinformation. Addressing these gaps requires benchmarks that are natively authored to reflect local norms and harm scenarios. We introduce SEA-SafeguardBench, the first human-verified safety benchmark for SEA, covering eight languages, 21,640 samples, across three subsets: general, in-the-wild, and content generation. The experimental results from our benchmark demonstrate that even state-of-the-art LLMs and guardrails are challenged by SEA cultural and harm scenarios and underperform when compared to English texts. |
| 2025-12-05 | [Spatiotemporal Tubes for Differential Drive Robots with Model Uncertainty](http://arxiv.org/abs/2512.05495v1) | Ratnangshu Das, Ahan Basu et al. | This paper presents a Spatiotemporal Tube (STT)-based control framework for differential-drive mobile robots with dynamic uncertainties and external disturbances, guaranteeing the satisfaction of Temporal Reach-Avoid-Stay (T-RAS) specifications. The approach employs circular STT, characterized by smoothly time-varying center and radius, to define dynamic safe corridors that guide the robot from the start region to the goal while avoiding obstacles. In particular, we first develop a sampling-based synthesis algorithm to construct a feasible STT that satisfies the prescribed timing and safety constraints with formal guarantees. To ensure that the robot remains confined within this tube, we then design analytically a closed-form, approximation-free control law. The resulting controller is computationally efficient, robust to disturbances and {model uncertainties}, and requires no model approximations or online optimization. The proposed framework is validated through simulation studies on a differential-drive robot and benchmarked against state-of-the-art methods, demonstrating superior robustness, accuracy, and computational efficiency. |
| 2025-12-05 | [TeleAI-Safety: A comprehensive LLM jailbreaking benchmark towards attacks, defenses, and evaluations](http://arxiv.org/abs/2512.05485v1) | Xiuyuan Chen, Jian Zhao et al. | While the deployment of large language models (LLMs) in high-value industries continues to expand, the systematic assessment of their safety against jailbreak and prompt-based attacks remains insufficient. Existing safety evaluation benchmarks and frameworks are often limited by an imbalanced integration of core components (attack, defense, and evaluation methods) and an isolation between flexible evaluation frameworks and standardized benchmarking capabilities. These limitations hinder reliable cross-study comparisons and create unnecessary overhead for comprehensive risk assessment. To address these gaps, we present TeleAI-Safety, a modular and reproducible framework coupled with a systematic benchmark for rigorous LLM safety evaluation. Our framework integrates a broad collection of 19 attack methods (including one self-developed method), 29 defense methods, and 19 evaluation methods (including one self-developed method). With a curated attack corpus of 342 samples spanning 12 distinct risk categories, the TeleAI-Safety benchmark conducts extensive evaluations across 14 target models. The results reveal systematic vulnerabilities and model-specific failure cases, highlighting critical trade-offs between safety and utility, and identifying potential defense patterns for future optimization. In practical scenarios, TeleAI-Safety can be flexibly adjusted with customized attack, defense, and evaluation combinations to meet specific demands. We release our complete code and evaluation results to facilitate reproducible research and establish unified safety baselines. |
| 2025-12-05 | [Concept-based Explainable Data Mining with VLM for 3D Detection](http://arxiv.org/abs/2512.05482v1) | Mai Tsujimoto | Rare-object detection remains a challenging task in autonomous driving systems, particularly when relying solely on point cloud data. Although Vision-Language Models (VLMs) exhibit strong capabilities in image understanding, their potential to enhance 3D object detection through intelligent data mining has not been fully explored. This paper proposes a novel cross-modal framework that leverages 2D VLMs to identify and mine rare objects from driving scenes, thereby improving 3D object detection performance. Our approach synthesizes complementary techniques such as object detection, semantic feature extraction, dimensionality reduction, and multi-faceted outlier detection into a cohesive, explainable pipeline that systematically identifies rare but critical objects in driving scenes. By combining Isolation Forest and t-SNE-based outlier detection methods with concept-based filtering, the framework effectively identifies semantically meaningful rare objects. A key strength of this approach lies in its ability to extract and annotate targeted rare object concepts such as construction vehicles, motorcycles, and barriers. This substantially reduces the annotation burden and focuses only on the most valuable training samples. Experiments on the nuScenes dataset demonstrate that this concept-guided data mining strategy enhances the performance of 3D object detection models while utilizing only a fraction of the training data, with particularly notable improvements for challenging object categories such as trailers and bicycles compared with the same amount of random data. This finding has substantial implications for the efficient curation of datasets in safety-critical autonomous systems. |

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



