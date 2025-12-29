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
| 2025-12-26 | [Perceive and Calibrate: Analyzing and Enhancing Robustness of Medical Multi-Modal Large Language Models](http://arxiv.org/abs/2512.21964v1) | Dunyuan XU, Xikai Yang et al. | Medical Multi-modal Large Language Models (MLLMs) have shown promising clinical performance. However, their sensitivity to real-world input perturbations, such as imaging artifacts and textual errors, critically undermines their clinical applicability. Systematic analysis of such noise impact on medical MLLMs remains largely unexplored. Furthermore, while several works have investigated the MLLMs' robustness in general domains, they primarily focus on text modality and rely on costly fine-tuning. They are inadequate to address the complex noise patterns and fulfill the strict safety standards in medicine. To bridge this gap, this work systematically analyzes the impact of various perturbations on medical MLLMs across both visual and textual modalities. Building on our findings, we introduce a training-free Inherent-enhanced Multi-modal Calibration (IMC) framework that leverages MLLMs' inherent denoising capabilities following the perceive-and-calibrate principle for cross-modal robustness enhancement. For the visual modality, we propose a Perturbation-aware Denoising Calibration (PDC) which leverages MLLMs' own vision encoder to identify noise patterns and perform prototype-guided feature calibration. For text denoising, we design a Self-instantiated Multi-agent System (SMS) that exploits the MLLMs' self-assessment capabilities to refine noisy text through a cooperative hierarchy of agents. We construct a benchmark containing 11 types of noise across both image and text modalities on 2 datasets. Experimental results demonstrate our method achieves the state-of-the-art performance across multiple modalities, showing potential to enhance MLLMs' robustness in real clinical scenarios. |
| 2025-12-26 | [Drug discovery guided by maximum drug likeness](http://arxiv.org/abs/2512.21895v1) | Hao-Yu Zhu, Shi-Jie Du et al. | To overcome the high attrition rate and limited clinical translatability in drug discovery, we introduce the concept of Maximum Drug-Likeness (MDL) and develop an applicable Fivefold MDL strategy (5F-MDL) to reshape the screening paradigm. The 5F-MDL strategy integrates an ensemble of 33 deep learning sub-models to construct a 33-dimensional property spectrum that quantifies the global phenotypic alignment of candidate molecules with clinically approved drugs along five axes: physicochemical properties, pharmacokinetics, efficacy, safety, and stability. Using drug-likeness scores derived from this 33-dimensional profile, we prioritized 15 high-potential molecules from a 16-million-molecule library. Experimental validation demonstrated that the lead compound M2 not only exhibits potent antibacterial activity, with a minimum inhibitory concentration (MIC) of 25.6 ug/mL, but also achieves binding stability superior to cefuroxime, as indicated by Molecular Mechanics Poisson-Boltzmann surface area (MM-PBSA) calculations of -38.54 kcal/mol and a root-mean-square deviation (RMSD) of 2.8 A. This strategy could overcome scaffold constraints and offers an efficient route for discovering lead compounds with favorable prospects against drug-resistant bacteria. |
| 2025-12-26 | [CricBench: A Multilingual Benchmark for Evaluating LLMs in Cricket Analytics](http://arxiv.org/abs/2512.21877v1) | Vaibhav Devraj, Dhruv Kumar et al. | Cricket is the second most popular sport globally, commanding a massive following of over 2.5 billion fans globally. Enthusiasts and analysts frequently seek advanced statistical insights, such as long-term historical performance trends or complex player comparisons, that are often unavailable through standard web searches. While Large Language Models (LLMs) have advanced significantly in Text-to-SQL tasks, their capability to handle the domain-specific nuances, complex schema variations, and multilingual requirements inherent to sports analytics remains under-explored. To investigate this potential capability gap, we present CricBench, a comprehensive benchmark suite for evaluating LLMs on specialized cricket data. To curate a "Gold Standard" dataset, we collaborate with domain experts in cricket and SQL to manually author complex queries, ensuring logical correctness. Recognizing linguistic diversity, we construct the benchmark in both English and Hindi, establishing a framework that is open for further extension to other regional languages. We evaluate six state-of-the-art models, including GPT-4o, Claude 3.7 Sonnet, and open-source models, using a strict evaluation protocol. Our results reveal that high performance on general benchmarks does not guarantee success in specialized domains. While the open-weights reasoning model DeepSeek R1 achieves state-of-the-art performance (50.6%), surpassing proprietary giants like Claude 3.7 Sonnet (47.7%) and GPT-4o (33.7%), it still exhibits a significant accuracy drop when moving from general benchmarks (BIRD) to CricBench. Furthermore, we observe that code-mixed Hindi queries frequently yield parity or higher accuracy compared to English, challenging the assumption that English is the optimal prompt language for specialized SQL tasks. |
| 2025-12-26 | [TimeBill: Time-Budgeted Inference for Large Language Models](http://arxiv.org/abs/2512.21859v1) | Qi Fan, An Zou et al. | Large Language Models (LLMs) are increasingly deployed in time-critical systems, such as robotics, autonomous driving, embodied intelligence, and industrial automation, where generating accurate responses within a given time budget is crucial for decision-making, control, or safety-critical tasks. However, the auto-regressive generation process of LLMs makes it challenging to model and estimate the end-to-end execution time. Furthermore, existing efficient inference methods based on a fixed key-value (KV) cache eviction ratio struggle to adapt to varying tasks with diverse time budgets, where an improper eviction ratio may lead to incomplete inference or a drop in response performance. In this paper, we propose TimeBill, a novel time-budgeted inference framework for LLMs that balances the inference efficiency and response performance. To be more specific, we propose a fine-grained response length predictor (RLP) and an execution time estimator (ETE) to accurately predict the end-to-end execution time of LLMs. Following this, we develop a time-budgeted efficient inference approach that adaptively adjusts the KV cache eviction ratio based on execution time prediction and the given time budget. Finally, through extensive experiments, we demonstrate the advantages of TimeBill in improving task completion rate and maintaining response performance under various overrun strategies. |
| 2025-12-26 | [Method Decoration (DeMe): A Framework for LLM-Driven Adaptive Method Generation in Dynamic IoT Environments](http://arxiv.org/abs/2512.21817v1) | Hong Su | Intelligent IoT systems increasingly rely on large language models (LLMs) to generate task-execution methods for dynamic environments. However, existing approaches lack the ability to systematically produce new methods when facing previously unseen situations, and they often depend on fixed, device-specific logic that cannot adapt to changing environmental conditions.In this paper, we propose Method Decoration (DeMe), a general framework that modifies the method-generation path of an LLM using explicit decorations derived from hidden goals, accumulated learned methods, and environmental feedback. Unlike traditional rule augmentation, decorations in DeMe are not hardcoded; instead, they are extracted from universal behavioral principles, experience, and observed environmental differences. DeMe enables the agent to reshuffle the structure of its method path-through pre-decoration, post-decoration, intermediate-step modification, and step insertion-thereby producing context-aware, safety-aligned, and environment-adaptive methods. Experimental results show that method decoration allows IoT devices to derive ore appropriate methods when confronting unknown or faulty operating conditions. |
| 2025-12-26 | [Few Tokens Matter: Entropy Guided Attacks on Vision-Language Models](http://arxiv.org/abs/2512.21815v1) | Mengqi He, Xinyu Tian et al. | Vision-language models (VLMs) achieve remarkable performance but remain vulnerable to adversarial attacks. Entropy, a measure of model uncertainty, is strongly correlated with the reliability of VLM. Prior entropy-based attacks maximize uncertainty at all decoding steps, implicitly assuming that every token contributes equally to generation instability. We show instead that a small fraction (about 20%) of high-entropy tokens, i.e., critical decision points in autoregressive generation, disproportionately governs output trajectories. By concentrating adversarial perturbations on these positions, we achieve semantic degradation comparable to global methods while using substantially smaller budgets. More importantly, across multiple representative VLMs, such selective attacks convert 35-49% of benign outputs into harmful ones, exposing a more critical safety risk. Remarkably, these vulnerable high-entropy forks recur across architecturally diverse VLMs, enabling feasible transferability (17-26% harmful rates on unseen targets). Motivated by these findings, we propose Entropy-bank Guided Adversarial attacks (EGA), which achieves competitive attack success rates (93-95%) alongside high harmful conversion, thereby revealing new weaknesses in current VLM safety mechanisms. |
| 2025-12-25 | [Sharing with Frictions: Limited Transfers and Costly Inspections](http://arxiv.org/abs/2512.21793v1) | Federico Bobbio, Randall A. Berry et al. | The radio spectrum suitable for commercial wireless services is limited. A portion of the radio spectrum has been reserved for institutions using it for non-commercial purposes such as federal agencies, defense, public safety bodies and scientific institutions. In order to operate efficiently, these incumbents need clean spectrum access. However, commercial users also want access, and granting them access may materially interfere with the existing activity of the incumbents. Conventional market based mechanisms for allocating scarce resources in this context are problematic. Allowing direct monetary transfers to and from public or scientific institutions risks distorting their non-commercial mission. Moreover, often only the incumbent knows the exact value of the interference it experiences, and, likewise, only commercial users can predict accurately the expected monetary outcome from sharing the resource. Thus, our problem is to determine the efficient allocation of resources in the presence of private information without the use of direct monetary transfers. The problem is not unique to spectrum. Other resources that governments hold in trust share the same feature. We propose a novel mechanism design formulation of the problem, characterize the optimal mechanism and describe some of its qualitative properties. |
| 2025-12-25 | [Compliance Rating Scheme: A Data Provenance Framework for Generative AI Datasets](http://arxiv.org/abs/2512.21775v1) | Matyas Bohacek, Ignacio Vilanova Echavarri | Generative Artificial Intelligence (GAI) has experienced exponential growth in recent years, partly facilitated by the abundance of large-scale open-source datasets. These datasets are often built using unrestricted and opaque data collection practices. While most literature focuses on the development and applications of GAI models, the ethical and legal considerations surrounding the creation of these datasets are often neglected. In addition, as datasets are shared, edited, and further reproduced online, information about their origin, legitimacy, and safety often gets lost. To address this gap, we introduce the Compliance Rating Scheme (CRS), a framework designed to evaluate dataset compliance with critical transparency, accountability, and security principles. We also release an open-source Python library built around data provenance technology to implement this framework, allowing for seamless integration into existing dataset-processing and AI training pipelines. The library is simultaneously reactive and proactive, as in addition to evaluating the CRS of existing datasets, it equally informs responsible scraping and construction of new datasets. |
| 2025-12-25 | [Modified TSception for Analyzing Driver Drowsiness and Mental Workload from EEG](http://arxiv.org/abs/2512.21747v1) | Gourav Siddhad, Anurag Singh et al. | Driver drowsiness remains a primary cause of traffic accidents, necessitating the development of real-time, reliable detection systems to ensure road safety. This study presents a Modified TSception architecture designed for the robust assessment of driver fatigue using Electroencephalography (EEG). The model introduces a novel hierarchical architecture that surpasses the original TSception by implementing a five-layer temporal refinement strategy to capture multi-scale brain dynamics. A key innovation is the use of Adaptive Average Pooling, which provides the structural flexibility to handle varying EEG input dimensions, and a two - stage fusion mechanism that optimizes the integration of spatiotemporal features for improved stability. When evaluated on the SEED-VIG dataset and compared against established methods - including SVM, Transformer, EEGNet, ConvNeXt, LMDA-Net, and the original TSception - the Modified TSception achieves a comparable accuracy of 83.46% (vs. 83.15% for the original). Critically, the proposed model exhibits a substantially reduced confidence interval (0.24 vs. 0.36), signifying a marked improvement in performance stability. Furthermore, the architecture's generalizability is validated on the STEW mental workload dataset, where it achieves state-of-the-art results with 95.93% and 95.35% accuracy for 2-class and 3-class classification, respectively. These improvements in consistency and cross-task generalizability underscore the effectiveness of the proposed modifications for reliable EEG-based monitoring of drowsiness and mental workload. |
| 2025-12-25 | [MAction-SocialNav: Multi-Action Socially Compliant Navigation via Reasoning-enhanced Prompt Tuning](http://arxiv.org/abs/2512.21722v1) | Zishuo Wang, Xinyu Zhang et al. | Socially compliant navigation requires robots to move safely and appropriately in human-centered environments by respecting social norms. However, social norms are often ambiguous, and in a single scenario, multiple actions may be equally acceptable. Most existing methods simplify this problem by assuming a single correct action, which limits their ability to handle real-world social uncertainty. In this work, we propose MAction-SocialNav, an efficient vision language model for socially compliant navigation that explicitly addresses action ambiguity, enabling generating multiple plausible actions within one scenario. To enhance the model's reasoning capability, we introduce a novel meta-cognitive prompt (MCP) method. Furthermore, to evaluate the proposed method, we curate a multi-action socially compliant navigation dataset that accounts for diverse conditions, including crowd density, indoor and outdoor environments, and dual human annotations. The dataset contains 789 samples, each with three-turn conversation, split into 710 training samples and 79 test samples through random selection. We also design five evaluation metrics to assess high-level decision precision, safety, and diversity. Extensive experiments demonstrate that the proposed MAction-SocialNav achieves strong social reasoning performance while maintaining high efficiency, highlighting its potential for real-world human robot navigation. Compared with zero-shot GPT-4o and Claude, our model achieves substantially higher decision quality (APG: 0.595 vs. 0.000/0.025) and safety alignment (ER: 0.264 vs. 0.642/0.668), while maintaining real-time efficiency (1.524 FPS, over 3x faster). |
| 2025-12-25 | [RAPTOR: Real-Time High-Resolution UAV Video Prediction with Efficient Video Attention](http://arxiv.org/abs/2512.21710v1) | Zhan Chen, Zile Guo et al. | Video prediction is plagued by a fundamental trilemma: achieving high-resolution and perceptual quality typically comes at the cost of real-time speed, hindering its use in latency-critical applications. This challenge is most acute for autonomous UAVs in dense urban environments, where foreseeing events from high-resolution imagery is non-negotiable for safety. Existing methods, reliant on iterative generation (diffusion, autoregressive models) or quadratic-complexity attention, fail to meet these stringent demands on edge hardware. To break this long-standing trade-off, we introduce RAPTOR, a video prediction architecture that achieves real-time, high-resolution performance. RAPTOR's single-pass design avoids the error accumulation and latency of iterative approaches. Its core innovation is Efficient Video Attention (EVA), a novel translator module that factorizes spatiotemporal modeling. Instead of processing flattened spacetime tokens with $O((ST)^2)$ or $O(ST)$ complexity, EVA alternates operations along the spatial (S) and temporal (T) axes. This factorization reduces the time complexity to $O(S + T)$ and memory complexity to $O(max(S, T))$, enabling global context modeling at $512^2$ resolution and beyond, operating directly on dense feature maps with a patch-free design. Complementing this architecture is a 3-stage training curriculum that progressively refines predictions from coarse structure to sharp, temporally coherent details. Experiments show RAPTOR is the first predictor to exceed 30 FPS on a Jetson AGX Orin for $512^2$ video, setting a new state-of-the-art on UAVid, KTH, and a custom high-resolution dataset in PSNR, SSIM, and LPIPS. Critically, RAPTOR boosts the mission success rate in a real-world UAV navigation task by 18/%, paving the way for safer and more anticipatory embodied agents. |
| 2025-12-25 | [Towards Responsible and Explainable AI Agents with Consensus-Driven Reasoning](http://arxiv.org/abs/2512.21699v1) | Eranga Bandara, Tharaka Hewa et al. | Agentic AI represents a major shift in how autonomous systems reason, plan, and execute multi-step tasks through the coordination of Large Language Models (LLMs), Vision Language Models (VLMs), tools, and external services. While these systems enable powerful new capabilities, increasing autonomy introduces critical challenges related to explainability, accountability, robustness, and governance, especially when agent outputs influence downstream actions or decisions. Existing agentic AI implementations often emphasize functionality and scalability, yet provide limited mechanisms for understanding decision rationale or enforcing responsibility across agent interactions. This paper presents a Responsible(RAI) and Explainable(XAI) AI Agent Architecture for production-grade agentic workflows based on multi-model consensus and reasoning-layer governance. In the proposed design, a consortium of heterogeneous LLM and VLM agents independently generates candidate outputs from a shared input context, explicitly exposing uncertainty, disagreement, and alternative interpretations. A dedicated reasoning agent then performs structured consolidation across these outputs, enforcing safety and policy constraints, mitigating hallucinations and bias, and producing auditable, evidence-backed decisions. Explainability is achieved through explicit cross-model comparison and preserved intermediate outputs, while responsibility is enforced through centralized reasoning-layer control and agent-level constraints. We evaluate the architecture across multiple real-world agentic AI workflows, demonstrating that consensus-driven reasoning improves robustness, transparency, and operational trust across diverse application domains. This work provides practical guidance for designing agentic AI systems that are autonomous and scalable, yet responsible and explainable by construction. |
| 2025-12-25 | [Comparative Analysis of Deep Learning Models for Perception in Autonomous Vehicles](http://arxiv.org/abs/2512.21673v1) | Jalal Khan | Recently, a plethora of machine learning (ML) and deep learning (DL) algorithms have been proposed to achieve the efficiency, safety, and reliability of autonomous vehicles (AVs). The AVs use a perception system to detect, localize, and identify other vehicles, pedestrians, and road signs to perform safe navigation and decision-making. In this paper, we compare the performance of DL models, including YOLO-NAS and YOLOv8, for a detection-based perception task. We capture a custom dataset and experiment with both DL models using our custom dataset. Our analysis reveals that the YOLOv8s model saves 75% of training time compared to the YOLO-NAS model. In addition, the YOLOv8s model (83%) outperforms the YOLO-NAS model (81%) when the target is to achieve the highest object detection accuracy. These comparative analyses of these new emerging DL models will allow the relevant research community to understand the models' performance under real-world use case scenarios. |
| 2025-12-25 | [Emotion-Aware Smart Home Automation Based on the eBICA Model](http://arxiv.org/abs/2512.21589v1) | Masaaki Yamauchi, Yiyuan Liang et al. | Smart home automation that adapts to a user's emotional state can enhance psychological safety in daily living environments. This study proposes an emotion-aware automation framework guided by the emotional Biologically Inspired Cognitive Architecture (eBICA), which integrates appraisal, somatic responses, and behavior selection. We conducted a proof-of-concept experiment in a pseudo-smart-home environment, where participants were exposed to an anxiety-inducing event followed by a comfort-inducing automation. State anxiety (STAI-S) was measured throughout the task sequence. The results showed a significant reduction in STAI-S immediately after introducing the avoidance automation, demonstrating that emotion-based control can effectively promote psychological safety. Furthermore, an analysis of individual characteristics suggested that personality and anxiety-related traits modulate the degree of relief, indicating the potential for personalized emotion-adaptive automation. Overall, this study provides empirical evidence that eBICA-based emotional control can function effectively in smart home environments and offers a foundation for next-generation affective home automation systems. |
| 2025-12-25 | [Broadband tunable microwave photonic radar for simultaneous detection of human respiration, heartbeat, and speech with deep learning-based speech recognition](http://arxiv.org/abs/2512.21566v1) | Lei Gao, Dingding Liang et al. | Multimodal vital sign monitoring and speech detection hold significant importance in medical health, public safety, and other fields. This study proposes a broadband tunable microwave photonic radar system that can simultaneously monitor respiration, heartbeat, and speech. The system works by generating broadband radar signals to detect subtle skin displacements caused by these physiological activities. It then utilizes phase variations in radar echo signals to extract and reconstruct the corresponding physiological signals. In order to enhance the processing capability for speech signals, a convolutional neural network with a dual-channel feature fusion model is incorporated, enabling high-precision speech recognition. In addition, the system's frequency-tunable characteristic allows it to flexibly switch frequency bands to adapt to different working environments, greatly improving its practicality and environmental adaptability. In concept-verification experiments, speech signals were reconstructed and recognized in the Ku, K, and Ka bands, achieving recognition accuracies of 97.20%, 98.07%, and 97.43%, respectively. The system's capability to detect multimodal vital signs was also thoroughly validated using a respiratory and heartbeat simulator. During a 20-second monitoring period, while accurately reconstructing speech, the maximum average error counts for respiratory and heartbeat monitoring were 0.39 and 0.87, respectively, proving its reliability and effectiveness in multimodal vital sign monitoring. |
| 2025-12-25 | [Leash: Adaptive Length Penalty and Reward Shaping for Efficient Large Reasoning Model](http://arxiv.org/abs/2512.21540v1) | Yanhao Li, Lu Ma et al. | Existing approaches typically rely on fixed length penalties, but such penalties are hard to tune and fail to adapt to the evolving reasoning abilities of LLMs, leading to suboptimal trade-offs between accuracy and conciseness. To address this challenge, we propose Leash (adaptive LEngth penAlty and reward SHaping), a reinforcement learning framework for efficient reasoning in LLMs. We formulate length control as a constrained optimization problem and employ a Lagrangian primal-dual method to dynamically adjust the penalty coefficient. When generations exceed the target length, the penalty is intensified; when they are shorter, it is relaxed. This adaptive mechanism guides models toward producing concise reasoning without sacrificing task performance. Experiments on Deepseek-R1-Distill-Qwen-1.5B and Qwen3-4B-Thinking-2507 show that Leash reduces the average reasoning length by 60% across diverse tasks - including in-distribution mathematical reasoning and out-of-distribution domains such as coding and instruction following - while maintaining competitive performance. Our work thus presents a practical and effective paradigm for developing controllable and efficient LLMs that balance reasoning capabilities with computational budgets. |
| 2025-12-25 | [Spatiotemporal Tubes for Probabilistic Temporal Reach-Avoid-Stay Task in Uncertain Dynamic Environment](http://arxiv.org/abs/2512.21497v1) | Siddhartha Upadhyay, Ratnangshu Das et al. | In this work, we extend the Spatiotemporal Tube (STT) framework to address Probabilistic Temporal Reach-Avoid-Stay (PrT-RAS) tasks in dynamic environments with uncertain obstacles. We develop a real-time tube synthesis procedure that explicitly accounts for time-varying uncertain obstacles and provides formal probabilistic safety guarantees. The STT is formulated as a time-varying ball in the state space whose center and radius evolve online based on uncertain sensory information. We derive a closed-form, approximation-free control law that confines the system trajectory within the tube, ensuring both probabilistic safety and task satisfaction. Our method offers a formal guarantee for probabilistic avoidance and finite-time task completion. The resulting controller is model-free, approximation-free, and optimization-free, enabling efficient real-time execution while guaranteeing convergence to the target. The effectiveness and scalability of the framework are demonstrated through simulation studies and hardware experiments on mobile robots, a UAV, and a 7-DOF manipulator navigating in cluttered and uncertain environments. |
| 2025-12-24 | [Fast Navigation Through Occluded Spaces via Language-Conditioned Map Prediction](http://arxiv.org/abs/2512.21398v1) | Rahul Moorthy Mahesh, Oguzhan Goktug Poyrazoglu et al. | In cluttered environments, motion planners often face a trade-off between safety and speed due to uncertainty caused by occlusions and limited sensor range. In this work, we investigate whether co-pilot instructions can help robots plan more decisively while remaining safe. We introduce PaceForecaster, as an approach that incorporates such co-pilot instructions into local planners. PaceForecaster takes the robot's local sensor footprint (Level-1) and the provided co-pilot instructions as input and predicts (i) a forecasted map with all regions visible from Level-1 (Level-2) and (ii) an instruction-conditioned subgoal within Level-2. The subgoal provides the planner with explicit guidance to exploit the forecasted environment in a goal-directed manner. We integrate PaceForecaster with a Log-MPPI controller and demonstrate that using language-conditioned forecasts and goals improves navigation performance by 36% over a local-map-only baseline while in polygonal environments. |
| 2025-12-24 | [Surgical Scene Segmentation using a Spike-Driven Video Transformer with Real-Time Potential](http://arxiv.org/abs/2512.21284v1) | Shihao Zou, Jingjing Li et al. | Modern surgical systems increasingly rely on intelligent scene understanding to provide timely situational awareness for enhanced intra-operative safety. Within this pipeline, surgical scene segmentation plays a central role in accurately perceiving operative events. Although recent deep learning models, particularly large-scale foundation models, achieve remarkable segmentation accuracy, their substantial computational demands and power consumption hinder real-time deployment in resource-constrained surgical environments. To address this limitation, we explore the emerging SNN as a promising paradigm for highly efficient surgical intelligence. However, their performance is still constrained by the scarcity of labeled surgical data and the inherently sparse nature of surgical video representations. To this end, we propose \textit{SpikeSurgSeg}, the first spike-driven video Transformer framework tailored for surgical scene segmentation with real-time potential on non-GPU platforms. To address the limited availability of surgical annotations, we introduce a surgical-scene masked autoencoding pretraining strategy for SNNs that enables robust spatiotemporal representation learning via layer-wise tube masking. Building on this pretrained backbone, we further adopt a lightweight spike-driven segmentation head that produces temporally consistent predictions while preserving the low-latency characteristics of SNNs. Extensive experiments on EndoVis18 and our in-house SurgBleed dataset demonstrate that SpikeSurgSeg achieves mIoU comparable to SOTA ANN-based models while reducing inference latency by at least $8\times$. Notably, it delivers over $20\times$ acceleration relative to most foundation-model baselines, underscoring its potential for time-critical surgical scene segmentation. |
| 2025-12-24 | [Casting a SPELL: Sentence Pairing Exploration for LLM Limitation-breaking](http://arxiv.org/abs/2512.21236v1) | Yifan Huang, Xiaojun Jia et al. | Large language models (LLMs) have revolutionized software development through AI-assisted coding tools, enabling developers with limited programming expertise to create sophisticated applications. However, this accessibility extends to malicious actors who may exploit these powerful tools to generate harmful software. Existing jailbreaking research primarily focuses on general attack scenarios against LLMs, with limited exploration of malicious code generation as a jailbreak target. To address this gap, we propose SPELL, a comprehensive testing framework specifically designed to evaluate the weakness of security alignment in malicious code generation. Our framework employs a time-division selection strategy that systematically constructs jailbreaking prompts by intelligently combining sentences from a prior knowledge dataset, balancing exploration of novel attack patterns with exploitation of successful techniques. Extensive evaluation across three advanced code models (GPT-4.1, Claude-3.5, and Qwen2.5-Coder) demonstrates SPELL's effectiveness, achieving attack success rates of 83.75%, 19.38%, and 68.12% respectively across eight malicious code categories. The generated prompts successfully produce malicious code in real-world AI development tools such as Cursor, with outputs confirmed as malicious by state-of-the-art detection systems at rates exceeding 73%. These findings reveal significant security gaps in current LLM implementations and provide valuable insights for improving AI safety alignment in code generation applications. |

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



