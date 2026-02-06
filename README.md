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
| 2026-02-05 | [GUARDIAN: Safety Filtering for Systems with Perception Models Subject to Adversarial Attacks](http://arxiv.org/abs/2602.06026v1) | Nicholas Rober, Alex Rose et al. | Safety filtering is an effective method for enforcing constraints in safety-critical systems, but existing methods typically assume perfect state information. This limitation is especially problematic for systems that rely on neural network (NN)-based state estimators, which can be highly sensitive to noise and adversarial input perturbations. We address these problems by introducing GUARDIAN: Guaranteed Uncertainty-Aware Reachability Defense against Adversarial INterference, a safety filtering framework that provides formal safety guarantees for systems with NN-based state estimators. At runtime, GUARDIAN uses neural network verification tools to provide guaranteed bounds on the system's state estimate given possible perturbations to its observation. It then uses a modified Hamilton-Jacobi reachability formulation to construct a safety filter that adjusts the nominal control input based on the verified state bounds and safety constraints. The result is an uncertainty-aware filter that ensures safety despite the system's reliance on an NN estimator with noisy, possibly adversarial, input observations. Theoretical analysis and numerical experiments demonstrate that GUARDIAN effectively defends systems against adversarial attacks that would otherwise lead to a violation of safety constraints. |
| 2026-02-05 | [A Systematic Evaluation of Large Language Models for PTSD Severity Estimation: The Role of Contextual Knowledge and Modeling Strategies](http://arxiv.org/abs/2602.06015v1) | Panagiotis Kaliosis, Adithya V Ganesan et al. | Large language models (LLMs) are increasingly being used in a zero-shot fashion to assess mental health conditions, yet we have limited knowledge on what factors affect their accuracy. In this study, we utilize a clinical dataset of natural language narratives and self-reported PTSD severity scores from 1,437 individuals to comprehensively evaluate the performance of 11 state-of-the-art LLMs. To understand the factors affecting accuracy, we systematically varied (i) contextual knowledge like subscale definitions, distribution summary, and interview questions, and (ii) modeling strategies including zero-shot vs few shot, amount of reasoning effort, model sizes, structured subscales vs direct scalar prediction, output rescaling and nine ensemble methods. Our findings indicate that (a) LLMs are most accurate when provided with detailed construct definitions and context of the narrative; (b) increased reasoning effort leads to better estimation accuracy; (c) performance of open-weight models (Llama, Deepseek), plateau beyond 70B parameters while closed-weight (o3-mini, gpt-5) models improve with newer generations; and (d) best performance is achieved when ensembling a supervised model with the zero-shot LLMs. Taken together, the results suggest choice of contextual knowledge and modeling strategies is important for deploying LLMs to accurately assess mental health. |
| 2026-02-05 | [$f$-GRPO and Beyond: Divergence-Based Reinforcement Learning Algorithms for General LLM Alignment](http://arxiv.org/abs/2602.05946v1) | Rajdeep Haldar, Lantao Mei et al. | Recent research shows that Preference Alignment (PA) objectives act as divergence estimators between aligned (chosen) and unaligned (rejected) response distributions. In this work, we extend this divergence-based perspective to general alignment settings, such as reinforcement learning with verifiable rewards (RLVR), where only environmental rewards are available. Within this unified framework, we propose $f$-Group Relative Policy Optimization ($f$-GRPO), a class of on-policy reinforcement learning, and $f$-Hybrid Alignment Loss ($f$-HAL), a hybrid on/off policy objectives, for general LLM alignment based on variational representation of $f$-divergences. We provide theoretical guarantees that these classes of objectives improve the average reward after alignment. Empirically, we validate our framework on both RLVR (Math Reasoning) and PA tasks (Safety Alignment), demonstrating superior performance and flexibility compared to current methods. |
| 2026-02-05 | [From Bench to Flight: Translating Drone Impact Tests into Operational Safety Limits](http://arxiv.org/abs/2602.05922v1) | Aziz Mohamed Mili, Louis Catar et al. | Indoor micro-aerial vehicles (MAVs) are increasingly used for tasks that require close proximity to people, yet practitioners lack practical methods to tune motion limits based on measured impact risk. We present an end-to-end, open toolchain that converts benchtop impact tests into deployable safety governors for drones. First, we describe a compact and replicable impact rig and protocol for capturing force-time profiles across drone classes and contact surfaces. Second, we provide data-driven models that map pre-impact speed to impulse and contact duration, enabling direct computation of speed bounds for a target force limit. Third, we release scripts and a ROS2 node that enforce these bounds online and log compliance, with support for facility-specific policies. We validate the workflow on multiple commercial off-the-shelf quadrotors and representative indoor assets, demonstrating that the derived governors preserve task throughput while meeting force constraints specified by safety stakeholders. Our contribution is a practical bridge from measured impacts to runtime limits, with shareable datasets, code, and a repeatable process that teams can adopt to certify indoor MAV operations near humans. |
| 2026-02-05 | [Agent2Agent Threats in Safety-Critical LLM Assistants: A Human-Centric Taxonomy](http://arxiv.org/abs/2602.05877v1) | Lukas Stappen, Ahmet Erkan Turan et al. | The integration of Large Language Model (LLM)-based conversational agents into vehicles creates novel security challenges at the intersection of agentic AI, automotive safety, and inter-agent communication. As these intelligent assistants coordinate with external services via protocols such as Google's Agent-to-Agent (A2A), they establish attack surfaces where manipulations can propagate through natural language payloads, potentially causing severe consequences ranging from driver distraction to unauthorized vehicle control. Existing AI security frameworks, while foundational, lack the rigorous "separation of concerns" standard in safety-critical systems engineering by co-mingling the concepts of what is being protected (assets) with how it is attacked (attack paths). This paper addresses this methodological gap by proposing a threat modeling framework called AgentHeLLM (Agent Hazard Exploration for LLM Assistants) that formally separates asset identification from attack path analysis. We introduce a human-centric asset taxonomy derived from harm-oriented "victim modeling" and inspired by the Universal Declaration of Human Rights, and a formal graph-based model that distinguishes poison paths (malicious data propagation) from trigger paths (activation actions). We demonstrate the framework's practical applicability through an open-source attack path suggestion tool AgentHeLLM Attack Path Generator that automates multi-stage threat discovery using a bi-level search strategy. |
| 2026-02-05 | [IDSOR: Intensity- and Distance-Aware Statistical Outlier Removal for Weather-Robust LiDAR Point Clouds](http://arxiv.org/abs/2602.05876v1) | Chenyang Yan, Mats Bengtsson | LiDAR point clouds captured in rain or snow are often corrupted by weather-induced returns, which can degrade perception and safety-critical scene understanding. This paper proposes Intensity- and Distance-Aware Statistical Outlier Removal (IDSOR), a range-adaptive filtering method that jointly exploits intensity cues and neighborhood sparsity. By incorporating an empirical, range-dependent distribution of weather returns into the threshold design, IDSOR suppresses weather-induced points while preserving fine structural details without cumbersome manual parameter tuning. We also propose a variant that uses a previously proposed method to estimate the weather return distribution from data, and integrates it into IDSOR. Experiments on simulation-augmented level-crossing measurements and on the Winter Adverse Driving dataset (WADS) demonstrate that IDSOR achieves a favorable precision-recall trade-off, maintaining both precision and recall above 90% on WADS. |
| 2026-02-05 | [Whispers of the Butterfly: A Research-through-Design Exploration of In-Situ Conversational AI Guidance in Large-Scale Outdoor MR Exhibitions](http://arxiv.org/abs/2602.05826v1) | Dongyijie Primo Pan, Shuyue Li et al. | Large-scale outdoor mixed reality (MR) art exhibitions distribute curated virtual works across open public spaces, but interpretation rarely scales without turning exploration into a scripted tour. Through Research-through-Design, we created Dream-Butterfly, an in-situ conversational AI docent embodied as a small non-human companion that visitors summon for multilingual, exhibition-grounded explanations. We deployed Dream-Butterfly in a large-scale outdoor MR exhibition at a public university campus in southern China, and conducted an in-the-wild between-subject study (N=24) comparing a primarily human-led tour with an AI-led tour while keeping staff for safety in both conditions. Combining questionnaires and semi-structured interviews, we characterize how shifting the primary explanation channel reshapes explanation access, perceived responsiveness, immersion, and workload, and how visitors negotiate responsibility handoffs among staff, the AI guide, and themselves. We distill transferable design implications for configuring mixed human-AI guiding roles and embodying conversational agents in mobile, safety-constrained outdoor MR exhibitions. |
| 2026-02-05 | [Depth as Prior Knowledge for Object Detection](http://arxiv.org/abs/2602.05730v1) | Moussa Kassem Sbeyti, Nadja Klein | Detecting small and distant objects remains challenging for object detectors due to scale variation, low resolution, and background clutter. Safety-critical applications require reliable detection of these objects for safe planning. Depth information can improve detection, but existing approaches require complex, model-specific architectural modifications. We provide a theoretical analysis followed by an empirical investigation of the depth-detection relationship. Together, they explain how depth causes systematic performance degradation and why depth-informed supervision mitigates it. We introduce DepthPrior, a framework that uses depth as prior knowledge rather than as a fused feature, providing comparable benefits without modifying detector architectures. DepthPrior consists of Depth-Based Loss Weighting (DLW) and Depth-Based Loss Stratification (DLS) during training, and Depth-Aware Confidence Thresholding (DCT) during inference. The only overhead is the initial cost of depth estimation. Experiments across four benchmarks (KITTI, MS COCO, VisDrone, SUN RGB-D) and two detectors (YOLOv11, EfficientDet) demonstrate the effectiveness of DepthPrior, achieving up to +9% mAP$_S$ and +7% mAR$_S$ for small objects, with inference recovery rates as high as 95:1 (true vs. false detections). DepthPrior offers these benefits without additional sensors, architectural changes, or performance costs. Code is available at https://github.com/mos-ks/DepthPrior. |
| 2026-02-05 | [CASTLE: A Comprehensive Benchmark for Evaluating Student-Tailored Personalized Safety in Large Language Models](http://arxiv.org/abs/2602.05633v1) | Rui Jia, Ruiyi Lan et al. | Large language models (LLMs) have advanced the development of personalized learning in education. However, their inherent generation mechanisms often produce homogeneous responses to identical prompts. This one-size-fits-all mechanism overlooks the substantial heterogeneity in students cognitive and psychological, thereby posing potential safety risks to vulnerable groups. Existing safety evaluations primarily rely on context-independent metrics such as factual accuracy, bias, or toxicity, which fail to capture the divergent harms that the same response might cause across different student attributes. To address this gap, we propose the concept of Student-Tailored Personalized Safety and construct CASTLE based on educational theories. This benchmark covers 15 educational safety risks and 14 student attributes, comprising 92,908 bilingual scenarios. We further design three evaluation metrics: Risk Sensitivity, measuring the model ability to detect risks; Emotional Empathy, evaluating the model capacity to recognize student states; and Student Alignment, assessing the match between model responses and student attributes. Experiments on 18 SOTA LLMs demonstrate that CASTLE poses a significant challenge: all models scored below an average safety rating of 2.3 out of 5, indicating substantial deficiencies in personalized safety assurance. |
| 2026-02-05 | [ROMAN: Reward-Orchestrated Multi-Head Attention Network for Autonomous Driving System Testing](http://arxiv.org/abs/2602.05629v1) | Jianlei Chi, Yuzhen Wu et al. | Automated Driving System (ADS) acts as the brain of autonomous vehicles, responsible for their safety and efficiency. Safe deployment requires thorough testing in diverse real-world scenarios and compliance with traffic laws like speed limits, signal obedience, and right-of-way rules. Violations like running red lights or speeding pose severe safety risks. However, current testing approaches face significant challenges: limited ability to generate complex and high-risk law-breaking scenarios, and failing to account for complex interactions involving multiple vehicles and critical situations. To address these challenges, we propose ROMAN, a novel scenario generation approach for ADS testing that combines a multi-head attention network with a traffic law weighting mechanism. ROMAN is designed to generate high-risk violation scenarios to enable more thorough and targeted ADS evaluation. The multi-head attention mechanism models interactions among vehicles, traffic signals, and other factors. The traffic law weighting mechanism implements a workflow that leverages an LLM-based risk weighting module to evaluate violations based on the two dimensions of severity and occurrence. We have evaluated ROMAN by testing the Baidu Apollo ADS within the CARLA simulation platform and conducting extensive experiments to measure its performance. Experimental results demonstrate that ROMAN surpassed state-of-the-art tools ABLE and LawBreaker by achieving 7.91% higher average violation count than ABLE and 55.96% higher than LawBreaker, while also maintaining greater scenario diversity. In addition, only ROMAN successfully generated violation scenarios for every clause of the input traffic laws, enabling it to identify more high-risk violations than existing approaches. |
| 2026-02-05 | [HiCrowd: Hierarchical Crowd Flow Alignment for Dense Human Environments](http://arxiv.org/abs/2602.05608v1) | Yufei Zhu, Shih-Min Yang et al. | Navigating through dense human crowds remains a significant challenge for mobile robots. A key issue is the freezing robot problem, where the robot struggles to find safe motions and becomes stuck within the crowd. To address this, we propose HiCrowd, a hierarchical framework that integrates reinforcement learning (RL) with model predictive control (MPC). HiCrowd leverages surrounding pedestrian motion as guidance, enabling the robot to align with compatible crowd flows. A high-level RL policy generates a follow point to align the robot with a suitable pedestrian group, while a low-level MPC safely tracks this guidance with short horizon planning. The method combines long-term crowd aware decision making with safe short-term execution. We evaluate HiCrowd against reactive and learning-based baselines in offline setting (replaying recorded human trajectories) and online setting (human trajectories are updated to react to the robot in simulation). Experiments on a real-world dataset and a synthetic crowd dataset show that our method outperforms in navigation efficiency and safety, while reducing freezing behaviors. Our results suggest that leveraging human motion as guidance, rather than treating humans solely as dynamic obstacles, provides a powerful principle for safe and efficient robot navigation in crowds. |
| 2026-02-05 | [VLN-Pilot: Large Vision-Language Model as an Autonomous Indoor Drone Operator](http://arxiv.org/abs/2602.05552v1) | Bessie Dominguez-Dager, Sergio Suescun-Ferrandiz et al. | This paper introduces VLN-Pilot, a novel framework in which a large Vision-and-Language Model (VLLM) assumes the role of a human pilot for indoor drone navigation. By leveraging the multimodal reasoning abilities of VLLMs, VLN-Pilot interprets free-form natural language instructions and grounds them in visual observations to plan and execute drone trajectories in GPS-denied indoor environments. Unlike traditional rule-based or geometric path-planning approaches, our framework integrates language-driven semantic understanding with visual perception, enabling context-aware, high-level flight behaviors with minimal task-specific engineering. VLN-Pilot supports fully autonomous instruction-following for drones by reasoning about spatial relationships, obstacle avoidance, and dynamic reactivity to unforeseen events. We validate our framework on a custom photorealistic indoor simulation benchmark and demonstrate the ability of the VLLM-driven agent to achieve high success rates on complex instruction-following tasks, including long-horizon navigation with multiple semantic targets. Experimental results highlight the promise of replacing remote drone pilots with a language-guided autonomous agent, opening avenues for scalable, human-friendly control of indoor UAVs in tasks such as inspection, search-and-rescue, and facility monitoring. Our results suggest that VLLM-based pilots may dramatically reduce operator workload while improving safety and mission flexibility in constrained indoor environments. |
| 2026-02-05 | [A Comparative Study of 3D Person Detection: Sensor Modalities and Robustness in Diverse Indoor and Outdoor Environments](http://arxiv.org/abs/2602.05538v1) | Malaz Tamim, Andrea Matic-Flierl et al. | Accurate 3D person detection is critical for safety in applications such as robotics, industrial monitoring, and surveillance. This work presents a systematic evaluation of 3D person detection using camera-only, LiDAR-only, and camera-LiDAR fusion. While most existing research focuses on autonomous driving, we explore detection performance and robustness in diverse indoor and outdoor scenes using the JRDB dataset. We compare three representative models - BEVDepth (camera), PointPillars (LiDAR), and DAL (camera-LiDAR fusion) - and analyze their behavior under varying occlusion and distance levels. Our results show that the fusion-based approach consistently outperforms single-modality models, particularly in challenging scenarios. We further investigate robustness against sensor corruptions and misalignments, revealing that while DAL offers improved resilience, it remains sensitive to sensor misalignment and certain LiDAR-based corruptions. In contrast, the camera-based BEVDepth model showed the lowest performance and was most affected by occlusion, distance, and noise. Our findings highlight the importance of utilizing sensor fusion for enhanced 3D person detection, while also underscoring the need for ongoing research to address the vulnerabilities inherent in these systems. |
| 2026-02-05 | [Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification](http://arxiv.org/abs/2602.05535v1) | Tao Huang, Rui Wang et al. | Large vision-language models (LVLMs) have shown substantial advances in multimodal understanding and generation. However, when presented with incompetent or adversarial inputs, they frequently produce unreliable or even harmful content, such as fact hallucinations or dangerous instructions. This misalignment with human expectations, referred to as \emph{misbehaviors} of LVLMs, raises serious concerns for deployment in critical applications. These misbehaviors are found to stem from epistemic uncertainty, specifically either conflicting internal knowledge or the absence of supporting information. However, existing uncertainty quantification methods, which typically capture only overall epistemic uncertainty, have shown limited effectiveness in identifying such issues. To address this gap, we propose Evidential Uncertainty Quantification (EUQ), a fine-grained method that captures both information conflict and ignorance for effective detection of LVLM misbehaviors. In particular, we interpret features from the model output head as either supporting (positive) or opposing (negative) evidence. Leveraging Evidence Theory, we model and aggregate this evidence to quantify internal conflict and knowledge gaps within a single forward pass. We extensively evaluate our method across four categories of misbehavior, including hallucinations, jailbreaks, adversarial vulnerabilities, and out-of-distribution (OOD) failures, using state-of-the-art LVLMs, and find that EUQ consistently outperforms strong baselines, showing that hallucinations correspond to high internal conflict and OOD failures to high ignorance. Furthermore, layer-wise evidential uncertainty dynamics analysis helps interpret the evolution of internal representations from a new perspective. The source code is available at https://github.com/HT86159/EUQ. |
| 2026-02-05 | [Conditional Diffusion Guidance under Hard Constraint: A Stochastic Analysis Approach](http://arxiv.org/abs/2602.05533v1) | Zhengyi Guo, Wenpin Tang et al. | We study conditional generation in diffusion models under hard constraints, where generated samples must satisfy prescribed events with probability one. Such constraints arise naturally in safety-critical applications and in rare-event simulation, where soft or reward-based guidance methods offer no guarantee of constraint satisfaction. Building on a probabilistic interpretation of diffusion models, we develop a principled conditional diffusion guidance framework based on Doob's h-transform, martingale representation and quadratic variation process. Specifically, the resulting guided dynamics augment a pretrained diffusion with an explicit drift correction involving the logarithmic gradient of a conditioning function, without modifying the pretrained score network. Leveraging martingale and quadratic-variation identities, we propose two novel off-policy learning algorithms based on a martingale loss and a martingale-covariation loss to estimate h and its gradient using only trajectories from the pretrained model. We provide non-asymptotic guarantees for the resulting conditional sampler in both total variation and Wasserstein distances, explicitly characterizing the impact of score approximation and guidance estimation errors. Numerical experiments demonstrate the effectiveness of the proposed methods in enforcing hard constraints and generating rare-event samples. |
| 2026-02-05 | [Toward Operationalizing Rasmussen: Drift Observability on the Simplex for Evolving Systems](http://arxiv.org/abs/2602.05483v1) | Anatoly A. Krasnovsky | Monitoring drift into failure is hindered by Euclidean anomaly detection that can conflate safe operational trade-offs with risk accumulation in signals expressed as shares, and by architectural churn that makes fixed schemas (and learned models) stale before rare boundary events occur. Rasmussen's dynamic safety model motivates drift under competing pressures, but operationalizing it for software is difficult because many high-value operational signals (effort, remaining margin, incident impact) are compositional and their parts evolve. We propose a vision for drift observability on the simplex: model drift and boundary proximity in Aitchison geometry to obtain coordinate-invariant direction and distance-to-safety in interpretable balance coordinates. To remain comparable under churn, a monitor would continuously refresh its part inventory and policy-defined boundaries from engineering artifacts and apply lineage-aware aggregation. We outline early-warning diagnostics and falsifiable hypotheses for future evaluation. |
| 2026-02-05 | [Ontology-Driven Robotic Specification Synthesis](http://arxiv.org/abs/2602.05456v1) | Maksym Figat, Ryan M. Mackey et al. | This paper addresses robotic system engineering for safety- and mission-critical applications by bridging the gap between high-level objectives and formal, executable specifications. The proposed method, Robotic System Task to Model Transformation Methodology (RSTM2) is an ontology-driven, hierarchical approach using stochastic timed Petri nets with resources, enabling Monte Carlo simulations at mission, system, and subsystem levels. A hypothetical case study demonstrates how the RSTM2 method supports architectural trades, resource allocation, and performance analysis under uncertainty. Ontological concepts further enable explainable AI-based assistants, facilitating fully autonomous specification synthesis. The methodology offers particular benefits to complex multi-robot systems, such as the NASA CADRE mission, representing decentralized, resource-aware, and adaptive autonomous systems of the future. |
| 2026-02-05 | [Causal Front-Door Adjustment for Robust Jailbreak Attacks on LLMs](http://arxiv.org/abs/2602.05444v1) | Yao Zhou, Zeen Song et al. | Safety alignment mechanisms in Large Language Models (LLMs) often operate as latent internal states, obscuring the model's inherent capabilities. Building on this observation, we model the safety mechanism as an unobserved confounder from a causal perspective. Then, we propose the \textbf{C}ausal \textbf{F}ront-Door \textbf{A}djustment \textbf{A}ttack ({\textbf{CFA}}$^2$) to jailbreak LLM, which is a framework that leverages Pearl's Front-Door Criterion to sever the confounding associations for robust jailbreaking. Specifically, we employ Sparse Autoencoders (SAEs) to physically strip defense-related features, isolating the core task intent. We further reduce computationally expensive marginalization to a deterministic intervention with low inference complexity. Experiments demonstrate that {CFA}$^2$ achieves state-of-the-art attack success rates while offering a mechanistic interpretation of the jailbreaking process. |
| 2026-02-05 | [Clinical Validation of Medical-based Large Language Model Chatbots on Ophthalmic Patient Queries with LLM-based Evaluation](http://arxiv.org/abs/2602.05381v1) | Ting Fang Tan, Kabilan Elangovan et al. | Domain specific large language models are increasingly used to support patient education, triage, and clinical decision making in ophthalmology, making rigorous evaluation essential to ensure safety and accuracy. This study evaluated four small medical LLMs Meerkat-7B, BioMistral-7B, OpenBioLLM-8B, and MedLLaMA3-v20 in answering ophthalmology related patient queries and assessed the feasibility of LLM based evaluation against clinician grading. In this cross sectional study, 180 ophthalmology patient queries were answered by each model, generating 2160 responses. Models were selected for parameter sizes under 10 billion to enable resource efficient deployment. Responses were evaluated by three ophthalmologists of differing seniority and by GPT-4-Turbo using the S.C.O.R.E. framework assessing safety, consensus and context, objectivity, reproducibility, and explainability, with ratings assigned on a five point Likert scale. Agreement between LLM and clinician grading was assessed using Spearman rank correlation, Kendall tau statistics, and kernel density estimate analyses. Meerkat-7B achieved the highest performance with mean scores of 3.44 from Senior Consultants, 4.08 from Consultants, and 4.18 from Residents. MedLLaMA3-v20 performed poorest, with 25.5 percent of responses containing hallucinations or clinically misleading content, including fabricated terminology. GPT-4-Turbo grading showed strong alignment with clinician assessments overall, with Spearman rho of 0.80 and Kendall tau of 0.67, though Senior Consultants graded more conservatively. Overall, medical LLMs demonstrated potential for safe ophthalmic question answering, but gaps remained in clinical depth and consensus, supporting the feasibility of LLM based evaluation for large scale benchmarking and the need for hybrid automated and clinician review frameworks to guide safe clinical deployment. |
| 2026-02-05 | [Formal Synthesis of Certifiably Robust Neural Lyapunov-Barrier Certificates](http://arxiv.org/abs/2602.05311v1) | Chengxiao Wang, Haoze Wu et al. | Neural Lyapunov and barrier certificates have recently been used as powerful tools for verifying the safety and stability properties of deep reinforcement learning (RL) controllers. However, existing methods offer guarantees only under fixed ideal unperturbed dynamics, limiting their reliability in real-world applications where dynamics may deviate due to uncertainties. In this work, we study the problem of synthesizing \emph{robust neural Lyapunov barrier certificates} that maintain their guarantees under perturbations in system dynamics. We formally define a robust Lyapunov barrier function and specify sufficient conditions based on Lipschitz continuity that ensure robustness against bounded perturbations. We propose practical training objectives that enforce these conditions via adversarial training, Lipschitz neighborhood bound, and global Lipschitz regularization. We validate our approach in two practically relevant environments, Inverted Pendulum and 2D Docking. The former is a widely studied benchmark, while the latter is a safety-critical task in autonomous systems. We show that our methods significantly improve both certified robustness bounds (up to $4.6$ times) and empirical success rates under strong perturbations (up to $2.4$ times) compared to the baseline. Our results demonstrate effectiveness of training robust neural certificates for safe RL under perturbations in dynamics. |

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



