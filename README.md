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
| 2026-03-18 | [Specification-Aware Distribution Shaping for Robotics Foundation Models](http://arxiv.org/abs/2603.17969v1) | Sadık Bera Yüksel, Derya Aksaray | Robotics foundation models have demonstrated strong capabilities in executing natural language instructions across diverse tasks and environments. However, they remain largely data-driven and lack formal guarantees on safety and satisfaction of time-dependent specifications during deployment. In practice, robots often need to comply with operational constraints involving rich spatio-temporal requirements such as time-bounded goal visits, sequential objectives, and persistent safety conditions. In this work, we propose a specification-aware action distribution optimization framework that enforces a broad class of Signal Temporal Logic (STL) constraints during execution of a pretrained robotics foundation model without modifying its parameters. At each decision step, the method computes a minimally modified action distribution that satisfies a hard STL feasibility constraint by reasoning over the remaining horizon using forward dynamics propagation. We validate the proposed framework in simulation using a state-of-the-art robotics foundation model across multiple environments and complex specifications. |
| 2026-03-18 | [IndicSafe: A Benchmark for Evaluating Multilingual LLM Safety in South Asia](http://arxiv.org/abs/2603.17915v1) | Priyaranjan Pattnayak, Sanchari Chowdhuri | As large language models (LLMs) are deployed in multilingual settings, their safety behavior in culturally diverse, low-resource languages remains poorly understood. We present the first systematic evaluation of LLM safety across 12 Indic languages, spoken by over 1.2 billion people but underrepresented in LLM training data. Using a dataset of 6,000 culturally grounded prompts spanning caste, religion, gender, health, and politics, we assess 10 leading LLMs on translated variants of the prompt.   Our analysis reveals significant safety drift: cross-language agreement is just 12.8\%, and \texttt{SAFE} rate variance exceeds 17\% across languages. Some models over-refuse benign prompts in low-resource scripts, overflag politically sensitive topics, while others fail to flag unsafe generations. We quantify these failures using prompt-level entropy, category bias scores, and multilingual consistency indices.   Our findings highlight critical safety generalization gaps in multilingual LLMs and show that safety alignment does not transfer evenly across languages. We release \textsc{IndicSafe}, the first benchmark to enable culturally informed safety evaluation for Indic deployments, and advocate for language-aware alignment strategies grounded in regional harms. |
| 2026-03-18 | [In Perfect Harmony: Orchestrating Causality in Actor-Based Systems](http://arxiv.org/abs/2603.17909v1) | Vladyslav Mikytiv, Bernardo Toninho et al. | Runtime verification has gained popularity as a lightweight approach for increasing assurance in systems under scrutiny. Performing runtime checks enables dynamic monitoring and alerts for unexpected behavior, thereby improving reliability and correctness. Actor-based systems present significant challenges for runtime verification. Properties frequently span multiple actors with complex causal dependencies, while nondeterministic message interleavings can obscure execution semantics. Moreover, most existing monitoring tools are designed for single-process behavior. This paper presents ACTORCHESTRA, a runtime verification framework for Erlang that automatically tracks causality across multi-actor interactions. The framework instruments Erlang systems that comply with OTP guidelines via targeted code injection. This method establishes the orchestration infrastructure required to track causal relationships between actors without requiring manual modifications to the target system. To ease the specification of multi-actor properties, the framework provides WALTZ, a specification language that automatically compiles properties into executable Erlang monitors that integrate with the instrumented system. Three case studies demonstrate ACTORCHESTRA's effectiveness in detecting complex behavioral violations in real-world actor systems. A performance evaluation quantifies the runtime overhead of the monitoring infrastructure and analyzes the trade-offs between added safety guarantees and execution costs. |
| 2026-03-18 | [Generative Control as Optimization: Time Unconditional Flow Matching for Adaptive and Robust Robotic Control](http://arxiv.org/abs/2603.17834v1) | Zunzhe Zhang, Runhan Huang et al. | Diffusion models and flow matching have become a cornerstone of robotic imitation learning, yet they suffer from a structural inefficiency where inference is often bound to a fixed integration schedule that is agnostic to state complexity. This paradigm forces the policy to expend the same computational budget on trivial motions as it does on complex tasks. We introduce Generative Control as Optimization (GeCO), a time-unconditional framework that transforms action synthesis from trajectory integration into iterative optimization. GeCO learns a stationary velocity field in the action-sequence space where expert behaviors form stable attractors. Consequently, test-time inference becomes an adaptive process that allocates computation based on convergence--exiting early for simple states while refining longer for difficult ones. Furthermore, this stationary geometry yields an intrinsic, training-free safety signal, as the field norm at the optimized action serves as a robust out-of-distribution (OOD) detector, remaining low for in-distribution states while significantly increasing for anomalies. We validate GeCO on standard simulation benchmarks and demonstrate seamless scaling to pi0-series Vision-Language-Action (VLA) models. As a plug-and-play replacement for standard flow-matching heads, GeCO improves success rates and efficiency with an optimization-native mechanism for safe deployment. Video and code can be found at https://hrh6666.github.io/GeCO/ |
| 2026-03-18 | [Federated Distributional Reinforcement Learning with Distributional Critic Regularization](http://arxiv.org/abs/2603.17820v1) | David Millard, Cecilia Alm et al. | Federated reinforcement learning typically aggregates value functions or policies by parameter averaging, which emphasizes expected return and can obscure statistical multimodality and tail behavior that matter in safety-critical settings. We formalize federated distributional reinforcement learning (FedDistRL), where clients parametrize quantile value function critics and federate these networks only. We also propose TR-FedDistRL, which builds a per client, risk-aware Wasserstein barycenter over a temporal buffer. This local barycenter provides a reference region to constrain the parameter averaged critic, ensuring necessary distributional information is not averaged out during the federation process. The distributional trust region is implemented as a shrink-squash step around this reference. Under fixed-policy evaluation, the feasibility map is nonexpansive and the update is contractive in a probe-set Wasserstein metric under evaluation. Experiments on a bandit, multi-agent gridworld, and continuous highway environment show reduced mean-smearing, improved safety proxies (catastrophe/accident rate), and lower critic/policy drift versus mean-oriented and non-federated baselines. |
| 2026-03-18 | [An HMDP-MPC Decision-making Framework with Adaptive Safety Margins and Hysteresis for Autonomous Driving](http://arxiv.org/abs/2603.17802v1) | Siyuan Li, Chengyuan Liu et al. | This paper presents a unified decision-making framework that integrates Hybrid Markov Decision Processes (HMDPs) with Model Predictive Control (MPC), augmented by velocity-dependent safety margins and a prediction-aware hysteresis mechanism. Both the ego and surrounding vehicles are modeled as HMDPs, allowing discrete maneuver transition and kinematic evolution to be jointly considered within the MPC optimization. Safety margins derived from the Intelligent Driver Model (IDM) adapt to traffic context but vary with speed, which can cause oscillatory decisions and velocity fluctuations. To mitigate this, we propose a frozen-release hysteresis mechanism with distinct trigger and release thresholds, effectively enlarging the reaction buffer and suppressing oscillations. Decision continuity is further safeguarded by a two-layer recovery scheme: a global bounded relaxation tied to IDM margins and a deterministic fallback policy. The framework is evaluated through a case study, an ablation against a no-hysteresis baseline, and largescale randomized experiments across 18 traffic settings. Across 8,050 trials, it achieves a collision rate of only 0.05%, with 98.77% of decisions resolved by nominal MPC and minimal reliance on relaxation or fallback. These results demonstrate the robustness and adaptability of the proposed decision-making framework in heterogeneous traffic conditions. |
| 2026-03-18 | [Facial Movement Dynamics Reveal Workload During Complex Multitasking](http://arxiv.org/abs/2603.17767v1) | Carter Sale, Melissa N. Stolar et al. | Real-time cognitive workload monitoring is crucial in safety-critical environments, yet established measures are intrusive, expensive, or lack temporal resolution. We tested whether facial movement dynamics from a standard webcam could provide a low-cost alternative. Seventy-two participants completed a multitasking simulation (OpenMATB) under varied load while facial keypoints were tracked via OpenPose. Linear kinematics (velocity, acceleration, displacement) and recurrence quantification features were extracted. Increasing load altered dynamics across timescales: movement magnitudes rose, temporal organisation fragmented then reorganised into complex patterns, and eye-head coordination weakened. Random forest classifiers trained on pose kinematics outperformed task performance metrics (85% vs. 55% accuracy) but generalised poorly across participants (43% vs. 33% chance). Participant-specific models reached 50% accuracy with minimal calibration (2 minutes per condition), improving continuously to 73% without plateau. Facial movement dynamics sensitively track workload with brief calibration, enabling adaptive interfaces using commodity cameras, though individual differences limit cross-participant generalisation. |
| 2026-03-18 | [Grounded Multimodal Retrieval-Augmented Drafting of Radiology Impressions Using Case-Based Similarity Search](http://arxiv.org/abs/2603.17765v1) | Himadri Samanta | Automated radiology report generation has gained increasing attention with the rise of deep learning and large language models. However, fully generative approaches often suffer from hallucinations and lack clinical grounding, limiting their reliability in real-world workflows. In this study, we propose a multimodal retrieval-augmented generation (RAG) system for grounded drafting of chest radiograph impressions. The system combines contrastive image-text embeddings, case-based similarity retrieval, and citation-constrained draft generation to ensure factual alignment with historical radiology reports. A curated subset of the MIMIC-CXR dataset was used to construct a multimodal retrieval database. Image embeddings were generated using CLIP encoders, while textual embeddings were derived from structured impression sections. A fusion similarity framework was implemented using FAISS indexing for scalable nearest-neighbor retrieval. Retrieved cases were used to construct grounded prompts for draft impression generation, with safety mechanisms enforcing citation coverage and confidence-based refusal. Experimental results demonstrate that multimodal fusion significantly improves retrieval performance compared to image-only retrieval, achieving Recall@5 above 0.95 on clinically relevant findings. The grounded drafting pipeline produces interpretable outputs with explicit citation traceability, enabling improved trustworthiness compared to conventional generative approaches. This work highlights the potential of retrieval-augmented multimodal systems for reliable clinical decision support and radiology workflow augmentation |
| 2026-03-18 | [Robust Dynamic Pricing and Admission Control with Fairness Guarantees](http://arxiv.org/abs/2603.17764v1) | Yingqing Chen, Anni Li et al. | Dynamic pricing is commonly used to regulate congestion in shared service systems. This paper is motivated by the fact that when heterogeneaous user groups (in terms of price responsiveness) are present, conventional monotonic pricing can lead to unfair outcomes by disproportionately excluding price-elastic users, particularly under high or uncertain demand. The paper's contributions are twofold. First, we show that when fairness is imposed as a hard state constraint, the optimal (revenue maximizing) pricing policy is generally non-monotonic in demand. This structural result departs fundamentally from standard surge pricing rules and reveals that price reduction under heavy load may be necessary to maintain equitable access. Second, we address the problem that price elasticity among heterogeneous users is unobservable. To solve it, we develop a robust dynamic pricing and admission control framework that enforces resource capacity and fairness constraints for all user type distributions consistent with aggregate measurements. By integrating integral High Order Control Barrier Functions (iHOCBFs) with a worst case robust optimization framework, we obtain a controller that guarantees forward invariance of safety and fairness constraints while optimizing revenue. Numerical experiments demonstrate improved fairness and revenue performance relative to monotonic surge pricing policies. |
| 2026-03-18 | [Harm or Humor: A Multimodal, Multilingual Benchmark for Overt and Covert Harmful Humor](http://arxiv.org/abs/2603.17759v1) | Ahmed Sharshar, Hosam Elgendy et al. | Dark humor often relies on subtle cultural nuances and implicit cues that require contextual reasoning to interpret, posing safety challenges that current static benchmarks fail to capture. To address this, we introduce a novel multimodal, multilingual benchmark for detecting and understanding harmful and offensive humor. Our manually curated dataset comprises 3,000 texts and 6,000 images in English and Arabic, alongside 1,200 videos that span English, Arabic, and language-independent (universal) contexts. Unlike standard toxicity datasets, we enforce a strict annotation guideline: distinguishing \emph{Safe} jokes from \emph{Harmful} ones, with the latter further classified into \emph{Explicit} (overt) and \emph{Implicit} (Covert) categories to probe deep reasoning. We systematically evaluate state-of-the-art (SOTA) open and closed-source models across all modalities. Our findings reveal that closed-source models significantly outperform open-source ones, with a notable difference in performance between the English and Arabic languages in both, underscoring the critical need for culturally grounded, reasoning-aware safety alignment. \textcolor{red}{Warning: this paper contains example data that may be offensive, harmful, or biased.} |
| 2026-03-18 | [From Virtual Environments to Real-World Trials: Emerging Trends in Autonomous Driving](http://arxiv.org/abs/2603.17714v1) | A. Humnabadkar, A. Sikdar et al. | Autonomous driving technologies have achieved significant advances in recent years, yet their real-world deployment remains constrained by data scarcity, safety requirements, and the need for generalization across diverse environments. In response, synthetic data and virtual environments have emerged as powerful enablers, offering scalable, controllable, and richly annotated scenarios for training and evaluation. This survey presents a comprehensive review of recent developments at the intersection of autonomous driving, simulation technologies, and synthetic datasets. We organize the landscape across three core dimensions: (i) the use of synthetic data for perception and planning, (ii) digital twin-based simulation for system validation, and (iii) domain adaptation strategies bridging synthetic and real-world data. We also highlight the role of vision-language models and simulation realism in enhancing scene understanding and generalization. A detailed taxonomy of datasets, tools, and simulation platforms is provided, alongside an analysis of trends in benchmark design. Finally, we discuss critical challenges and open research directions, including Sim2Real transfer, scalable safety validation, cooperative autonomy, and simulation-driven policy learning, that must be addressed to accelerate the path toward safe, generalizable, and globally deployable autonomous driving systems. |
| 2026-03-18 | [Hierarchical Decision-Making under Uncertainty: A Hybrid MDP and Chance-Constrained MPC Approach](http://arxiv.org/abs/2603.17634v1) | Siyuan Li, Chengyuan Liu et al. | This paper presents a hierarchical decision-making framework for autonomous systems operating under uncertainty, demonstrated through autonomous driving as a representative application. Surrounding agents are modeled using Hybrid Markov Decision Processes (HMDPs) that jointly capture maneuver-level and dynamic-level uncertainties, enabling the multi-modal environmental prediction. The ego agent is modeled using a separate HMDP and integrated into a Model Predictive Control (MPC) framework that unifies maneuver selection with dynamic feasibility within a single optimization. A set of joint chance constraints serves as the bridge between environmental prediction and optimization, incorporating multi-modal environment predictions into the MPC formulation and ensuring safety across all plausible interaction scenarios. The proposed framework provides theoretical guarantees on recursive feasibility and asymptotic stability, and its benefits in terms of safety and efficiency are validated through comprehensive evaluations in highway and urban environments, together with comparisons against a rule-based baseline. |
| 2026-03-18 | [FoMo X: Modular Explainability Signals for Outlier Detection Foundation Models](http://arxiv.org/abs/2603.17570v1) | Simon Klüttermann, Tim Katzke et al. | Tabular foundation models, specifically Prior-Data Fitted Networks (PFNs), have revolutionized outlier detection (OD) by enabling unsupervised zero-shot adaptation to new datasets without training. However, despite their predictive power, these models typically function as opaque black boxes, outputting scalar outlier scores that lack the operational context required for safety-critical decision-making. Existing post-hoc explanation methods are often computationally prohibitive for real-time deployment or fail to capture the epistemic uncertainty inherent in zero-shot inference. In this work, we introduce FoMo-X, a modular framework that equips OD foundation models with intrinsic, lightweight diagnostic capabilities. We leverage the insight that the frozen embeddings of a pretrained PFN backbone already encode rich, context-conditioned relational information. FoMo-X attaches auxiliary diagnostic heads to these embeddings, trained offline using the same generative simulator prior as the backbone. This allows us to distill computationally expensive properties, such as Monte Carlo dropout based epistemic uncertainty, into a deterministic, single-pass inference. We instantiate FoMo-X with two novel heads: a Severity Head that discretizes deviations into interpretable risk tiers, and an Uncertainty Head that provides calibrated confidence measures. Extensive evaluation on synthetic and real-world benchmarks (ADBench) demonstrates that FoMo-X recovers ground-truth diagnostic signals with high fidelity and negligible inference overhead. By bridging the gap between foundation model performance and operational explainability, FoMo-X offers a scalable path toward trustworthy, zero-shot outlier detection. |
| 2026-03-18 | [Interpreting Context-Aware Human Preferences for Multi-Objective Robot Navigation](http://arxiv.org/abs/2603.17510v1) | Tharun Sethuraman, Subham Agrawal et al. | Robots operating in human-shared environments must not only achieve task-level navigation objectives such as safety and efficiency, but also adapt their behavior to human preferences. However, as human preferences are typically expressed in natural language and depend on environmental context, it is difficult to directly integrate them into low-level robot control policies. In this work, we present a pipeline that enables robots to understand and apply context-dependent navigation preferences by combining foundational models with a Multi-Objective Reinforcement Learning (MORL) navigation policy. Thus, our approach integrates high-level semantic reasoning with low-level motion control. A Vision-Language Model (VLM) extracts structured environmental context from onboard visual observations, while Large Language Models (LLM) convert natural language user feedback into interpretable, context-dependent behavioral rules stored in a persistent but updatable rule memory. A preference translation module then maps contextual information and stored rules into numerical preference vectors that parameterize a pretrained MORL policy for real-time navigation adaptation. We evaluate the proposed framework through quantitative component-level evaluations, a user study, and real-world robot deployments in various indoor environments. Our results demonstrate that the system reliably captures user intent, generates consistent preference vectors, and enables controllable behavior adaptation across diverse contexts. Overall, the proposed pipeline improves the adaptability, transparency, and usability of robots operating in shared human environments, while maintaining safe and responsive real-time control. |
| 2026-03-18 | [UniSAFE: A Comprehensive Benchmark for Safety Evaluation of Unified Multimodal Models](http://arxiv.org/abs/2603.17476v1) | Segyu Lee, Boryeong Cho et al. | Unified Multimodal Models (UMMs) offer powerful cross-modality capabilities but introduce new safety risks not observed in single-task models. Despite their emergence, existing safety benchmarks remain fragmented across tasks and modalities, limiting the comprehensive evaluation of complex system-level vulnerabilities. To address this gap, we introduce UniSAFE, the first comprehensive benchmark for system-level safety evaluation of UMMs across 7 I/O modality combinations, spanning conventional tasks and novel multimodal-context image generation settings. UniSAFE is built with a shared-target design that projects common risk scenarios across task-specific I/O configurations, enabling controlled cross-task comparisons of safety failures. Comprising 6,802 curated instances, we use UniSAFE to evaluate 15 state-of-the-art UMMs, both proprietary and open-source. Our results reveal critical vulnerabilities across current UMMs, including elevated safety violations in multi-image composition and multi-turn settings, with image-output tasks consistently more vulnerable than text-output tasks. These findings highlight the need for stronger system-level safety alignment for UMMs. Our code and data are publicly available at https://github.com/segyulee/UniSAFE |
| 2026-03-18 | [Bringing Network Coding into Multi-Robot Systems: Interplay Study for Autonomous Systems over Wireless Communications](http://arxiv.org/abs/2603.17472v1) | Anil Zaher, Kiril Solovey et al. | Communication is a core enabler for multi-robot systems (MRS), providing the mechanism through which robots exchange state information, coordinate actions, and satisfy safety constraints. While many MRS autonomy algorithms assume reliable and timely message delivery, realistic wireless channels introduce delay, erasures, and ordering stalls that can degrade performance and compromise safety-critical decisions of the robot task. In this paper, we investigate how transport-layer reliability mechanisms that mitigate communication losses and delays shape the autonomy-communication loop. We show that conventional non-coded retransmission-based protocols introduce long delays that are misaligned with the timeliness requirements of MRS applications, and may render the received data irrelevant. As an alternative, we advocate for adaptive and causal network coding, which proactively injects coded redundancy to achieve the desired delay and throughput that enable relevant data delivery to the robotic task. Specifically, this method adapts to channel conditions between robots and causally tunes the communication rates via efficient algorithms.   We present two case studies: cooperative localization under delayed and lossy inter-robot communication, and a safety-critical overtaking maneuver where timely vehicle-to-vehicle message availability determines whether an ego vehicle can abort to avoid a crash. Our results demonstrate that coding-based communication significantly reduces in-order delivery stalls, preserves estimation consistency under delay, and improves deadline reliability relative to retransmission-based transport. Overall, the study highlights the need to jointly design autonomy algorithms and communication mechanisms, and positions network coding as a principled tool for dependable multi-robot operation over wireless networks. |
| 2026-03-18 | [SafeLand: Safe Autonomous Landing in Unknown Environments with Bayesian Semantic Mapping](http://arxiv.org/abs/2603.17430v1) | Markus Gross, Andreas Greiner et al. | Autonomous landing of uncrewed aerial vehicles (UAVs) in unknown, dynamic environments poses significant safety challenges, particularly near people and infrastructure, as UAVs transition to routine urban and rural operations. Existing methods often rely on prior maps, heavy sensors like LiDAR, static markers, or fail to handle non-cooperative dynamic obstacles like humans, limiting generalization and real-time performance. To address these challenges, we introduce SafeLand, a lean, vision-based system for safe autonomous landing (SAL) that requires no prior information and operates only with a camera and a lightweight height sensor. Our approach constructs an online semantic ground map via deep learning-based semantic segmentation, optimized for embedded deployment and trained on a consolidation of seven curated public aerial datasets (achieving 70.22% mIoU across 20 classes), which is further refined through Bayesian probabilistic filtering with temporal semantic decay to robustly identify metric-scale landing spots. A behavior tree then governs adaptive landing, iteratively validates the spot, and reacts in real time to dynamic obstacles by pausing, climbing, or rerouting to alternative spots, maximizing human safety. We extensively evaluate our method in 200 simulations and 60 end-to-end field tests across industrial, urban, and rural environments at altitudes up to 100m, demonstrating zero false negatives for human detection. Compared to the state of the art, SafeLand achieves sub-second response latency, substantially lower than previous methods, while maintaining a superior success rate of 95%. To facilitate further research in aerial robotics, we release SafeLand's segmentation model as a plug-and-play ROS package, available at https://github.com/markus-42/SafeLand. |
| 2026-03-18 | [Proactive Knowledge Inquiry in Doctor-Patient Dialogue: Stateful Extraction, Belief Updating, and Path-Aware Action Planning](http://arxiv.org/abs/2603.17425v1) | Zhenhai Pan, Yan Liu et al. | Most automated electronic medical record (EMR) pipelines remain output-oriented: they transcribe, extract, and summarize after the consultation, but they do not explicitly model what is already known, what is still missing, which uncertainty matters most, or what question or recommendation should come next. We formulate doctor-patient dialogue as a proactive knowledge-inquiry problem under partial observability. The proposed framework combines stateful extraction, sequential belief updating, gap-aware state modeling, hybrid retrieval over objectified medical knowledge, and a POMDP-lite action planner. Instead of treating the EMR as the only target artifact, the framework treats documentation as the structured projection of an ongoing inquiry loop. To make the formulation concrete, we report a controlled pilot evaluation on ten standardized multi-turn dialogues together with a 300-query retrieval benchmark aggregated across dialogues. On this pilot protocol, the full framework reaches 83.3% coverage, 80.0% risk recall, 81.4% structural completeness, and lower redundancy than the chunk-only and template-heavy interactive baselines. These pilot results do not establish clinical generalization; rather, they suggest that proactive inquiry may be methodologically interesting under tightly controlled conditions and can be viewed as a conceptually appealing formulation worth further investigation for dialogue-based EMR generation. This work should be read as a pilot concept demonstration under a controlled simulated setting rather than as evidence of clinical deployment readiness. No implication of clinical deployment readiness, clinical safety, or real-world clinical utility should be inferred from this pilot protocol. |
| 2026-03-18 | [Is Your LLM-as-a-Recommender Agent Trustable? LLMs' Recommendation is Easily Hacked by Biases (Preferences)](http://arxiv.org/abs/2603.17417v1) | Zichen Tang, Zirui Zhang et al. | Current Large Language Models (LLMs) are gradually exploited in practically valuable agentic workflows such as Deep Research, E-commerce recommendation, and job recruitment. In these applications, LLMs need to select some optimal solutions from massive candidates, which we term as \textit{LLM-as-a-Recommender} paradigm. However, the reliability of using LLM agents for recommendations is underexplored. In this work, we introduce a \textbf{Bias} \textbf{Rec}ommendation \textbf{Bench}mark (\textbf{BiasRecBench}) to highlight the critical vulnerability of such agents to biases in high-value real-world tasks. The benchmark includes three practical domains: paper review, e-commerce, and job recruitment. We construct a \textsc{Bias Synthesis Pipeline with Calibrated Quality Margins} that 1) synthesizes evaluation data by controlling the quality gap between optimal and sub-optimal options to provide a calibrated testbed to elicit the vulnerability to biases; 2) injects contextual biases that are logical and suitable for option contexts. Extensive experiments on both SOTA (Gemini-{2.5,3}-pro, GPT-4o, DeepSeek-R1) and small-scale LLMs reveal that agents frequently succumb to injected biases despite having sufficient reasoning capabilities to identify the ground truth. These findings expose a significant reliability bottleneck in current agentic workflows, calling for specialized alignment strategies for LLM-as-a-Recommender. The complete code and evaluation datasets will be made publicly available shortly. |
| 2026-03-18 | [Dynamical Properties of Safety Filters for Linear Systems and Affine Control Barrier Functions](http://arxiv.org/abs/2603.17401v1) | Pol Mestres, Shima Sadat Mousavi et al. | This letter studies the dynamical properties of safety filters designed based on Control Barrier Functions (CBF). This mechanism, which is popular in safety-critical applications, takes a nominal controller and minimally modifies it to render it safe. Although CBF-based safety filters make the closed-loop system safe, characterizing their additional dynamical properties, such as stability, boundedness, or existence of spurious equilibria, remains a challenging problem. Here, we address this problem for the case of linear systems and an affine CBF constraint. We provide conditions under which the closed-loop system presents undesired equilibria, unbounded trajectories, or the origin is globally exponentially stable. |

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



