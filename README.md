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
| 2025-08-04 | [Dam Management in the Era of Climate Change](http://arxiv.org/abs/2508.02636v1) | Cristina Di Girolami, M'hamed Mrad et al. | Climate change has a dramatic impact, particularly by concentrating rainfall into a few short periods, interspersed by long dry spells. In this context, the role of dams is crucial. We consider the optimal control of a dam, where the water level must not exceed a designated safety threshold, nor fall below a minimum level to ensure functionality and sustainability for for the outgoing river. To model dry spells and intense rainfall events, commonly referred to as water bombs, we introduce a Hawkes process, a well-known example of a self-exciting process characterised by time-correlated intensity, which endogenously reproduces the concentration of events. The problem is formulated as an optimal switching problem with constraints. We establish existence results and propose numerical methods for approximating the solution. Finally, we illustrate the main achievements of this approach through numerical examples. The main and counterintuitive result of our numerical analysis is that the optimal water level inside the dam increases with the self-exciting parameter. This result shows that, when facing the dilemma of managing the opposing risks of dam overtopping and dry spells, the former ultimately dominates the latter. In conclusion, dams will increasingly lose their role as water reserves and take on a greater role in flood protection. |
| 2025-08-04 | [Precision-Aware Video Compression for Reducing Bandwidth Requirements in Video Communication for Vehicle Detection-Based Applications](http://arxiv.org/abs/2508.02533v1) | Abyad Enan, Jon C Calhoun et al. | Computer vision has become a popular tool in intelligent transportation systems (ITS), enabling various applications through roadside traffic cameras that capture video and transmit it in real time to computing devices within the same network. The efficiency of this video transmission largely depends on the available bandwidth of the communication system. However, limited bandwidth can lead to communication bottlenecks, hindering the real-time performance of ITS applications. To mitigate this issue, lossy video compression techniques can be used to reduce bandwidth requirements, at the cost of degrading video quality. This degradation can negatively impact the accuracy of applications that rely on real-time vehicle detection. Additionally, vehicle detection accuracy is influenced by environmental factors such as weather and lighting conditions, suggesting that compression levels should be dynamically adjusted in response to these variations. In this work, we utilize a framework called Precision-Aware Video Compression (PAVC), where a roadside video camera captures footage of vehicles on roadways, compresses videos, and then transmits them to a processing unit, running a vehicle detection algorithm for safety-critical applications, such as real-time collision risk assessment. The system dynamically adjusts the video compression level based on current weather and lighting conditions to maintain vehicle detection accuracy while minimizing bandwidth usage. Our results demonstrate that PAVC improves vehicle detection accuracy by up to 13% and reduces communication bandwidth requirements by up to 8.23x in areas with moderate bandwidth availability. Moreover, in locations with severely limited bandwidth, PAVC reduces bandwidth requirements by up to 72x while preserving vehicle detection performance. |
| 2025-08-04 | [Understanding the Risks of Asphalt Art on the Reliability of Surveillance Perception Systems](http://arxiv.org/abs/2508.02530v1) | Jin Ma, Abyad Enan et al. | Artistic crosswalks featuring asphalt art, introduced by different organizations in recent years, aim to enhance the visibility and safety of pedestrians. However, their visual complexity may interfere with surveillance systems that rely on vision-based object detection models. In this study, we investigate the impact of asphalt art on pedestrian detection performance of a pretrained vision-based object detection model. We construct realistic crosswalk scenarios by compositing various street art patterns into a fixed surveillance scene and evaluate the model's performance in detecting pedestrians on asphalt-arted crosswalks under both benign and adversarial conditions. A benign case refers to pedestrian crosswalks painted with existing normal asphalt art, whereas an adversarial case involves digitally crafted or altered asphalt art perpetrated by an attacker. Our results show that while simple, color-based designs have minimal effect, complex artistic patterns, particularly those with high visual salience, can significantly degrade pedestrian detection performance. Furthermore, we demonstrate that adversarially crafted asphalt art can be exploited to deliberately obscure real pedestrians or generate non-existent pedestrian detections. These findings highlight a potential vulnerability in urban vision-based pedestrian surveillance systems and underscore the importance of accounting for environmental visual variations when designing robust pedestrian perception models. |
| 2025-08-04 | [Transportation Cyber Incident Awareness through Generative AI-Based Incident Analysis and Retrieval-Augmented Question-Answering Systems](http://arxiv.org/abs/2508.02523v1) | Ostonya Thomas, Muhaimin Bin Munir et al. | Technological advancements have revolutionized numerous industries, including transportation. While digitalization, automation, and connectivity have enhanced safety and efficiency, they have also introduced new vulnerabilities. With 95% of data breaches attributed to human error, promoting cybersecurity awareness in transportation is increasingly critical. Despite numerous cyberattacks on transportation systems worldwide, comprehensive and centralized records of these incidents remain scarce. To address this gap and enhance cyber awareness, this paper presents a large language model (LLM) based approach to extract and organize transportation related cyber incidents from publicly available datasets. A key contribution of this work is the use of generative AI to transform unstructured, heterogeneous cyber incident data into structured formats. Incidents were sourced from the Center for Strategic & International Studies (CSIS) List of Significant Cyber Incidents, the University of Maryland Cyber Events Database (UMCED), the European Repository of Cyber Incidents (EuRepoC), the Maritime Cyber Attack Database (MCAD), and the U.S. DOT Transportation Cybersecurity and Resiliency (TraCR) Examples of Cyber Attacks in Transportation (2018 to 2022). These were classified by a fine tuned LLM into five transportation modes: aviation, maritime, rail, road, and multimodal, forming a transportation specific cyber incident database. Another key contribution of this work is the development of a Retrieval Augmented Generation question answering system, designed to enhance accessibility and practical use by enabling users to query the curated database for specific details on transportation related cyber incidents. By leveraging LLMs for both data extraction and user interaction, this study contributes a novel, accessible tool for improving cybersecurity awareness in the transportation sector. |
| 2025-08-04 | [PoseGuard: Pose-Guided Generation with Safety Guardrails](http://arxiv.org/abs/2508.02476v1) | Kongxin Wang, Jie Zhang et al. | Pose-guided video generation has become a powerful tool in creative industries, exemplified by frameworks like Animate Anyone. However, conditioning generation on specific poses introduces serious risks, such as impersonation, privacy violations, and NSFW content creation. To address these challenges, we propose $\textbf{PoseGuard}$, a safety alignment framework for pose-guided generation. PoseGuard is designed to suppress unsafe generations by degrading output quality when encountering malicious poses, while maintaining high-fidelity outputs for benign inputs. We categorize unsafe poses into three representative types: discriminatory gestures such as kneeling or offensive salutes, sexually suggestive poses that lead to NSFW content, and poses imitating copyrighted celebrity movements. PoseGuard employs a dual-objective training strategy combining generation fidelity with safety alignment, and uses LoRA-based fine-tuning for efficient, parameter-light updates. To ensure adaptability to evolving threats, PoseGuard supports pose-specific LoRA fusion, enabling flexible and modular updates when new unsafe poses are identified. We further demonstrate the generalizability of PoseGuard to facial landmark-guided generation. Extensive experiments validate that PoseGuard effectively blocks unsafe generations, maintains generation quality for benign inputs, and remains robust against slight pose variations. |
| 2025-08-04 | [Multi-Class Human/Object Detection on Robot Manipulators using Proprioceptive Sensing](http://arxiv.org/abs/2508.02425v1) | Justin Hehli, Marco Heiniger et al. | In physical human-robot collaboration (pHRC) settings, humans and robots collaborate directly in shared environments. Robots must analyze interactions with objects to ensure safety and facilitate meaningful workflows. One critical aspect is human/object detection, where the contacted object is identified. Past research introduced binary machine learning classifiers to distinguish between soft and hard objects. This study improves upon those results by evaluating three-class human/object detection models, offering more detailed contact analysis. A dataset was collected using the Franka Emika Panda robot manipulator, exploring preprocessing strategies for time-series analysis. Models including LSTM, GRU, and Transformers were trained on these datasets. The best-performing model achieved 91.11\% accuracy during real-time testing, demonstrating the feasibility of multi-class detection models. Additionally, a comparison of preprocessing strategies suggests a sliding window approach is optimal for this task. |
| 2025-08-04 | [Vision Language Model-based Testing of Industrial Autonomous Mobile Robots](http://arxiv.org/abs/2508.02338v1) | Jiahui Wu, Chengjie Lu et al. | Autonomous Mobile Robots (AMRs) are deployed in diverse environments (e.g., warehouses, retail spaces, and offices), where they work alongside humans. Given that human behavior can be unpredictable and that AMRs may not have been trained to handle all possible unknown and uncertain behaviors, it is important to test AMRs under a wide range of human interactions to ensure their safe behavior. Moreover, testing in real environments with actual AMRs and humans is often costly, impractical, and potentially hazardous (e.g., it could result in human injury). To this end, we propose a Vision Language Model (VLM)-based testing approach (RVSG) for industrial AMRs developed by PAL Robotics in Spain. Based on the functional and safety requirements, RVSG uses the VLM to generate diverse human behaviors that violate these requirements. We evaluated RVSG with several requirements and navigation routes in a simulator using the latest AMR from PAL Robotics. Our results show that, compared with the baseline, RVSG can effectively generate requirement-violating scenarios. Moreover, RVSG-generated scenarios increase variability in robot behavior, thereby helping reveal their uncertain behaviors. |
| 2025-08-04 | [Is Uncertainty Quantification a Viable Alternative to Learned Deferral?](http://arxiv.org/abs/2508.02319v1) | Anna M. Wundram, Christian F. Baumgartner | Artificial Intelligence (AI) holds the potential to dramatically improve patient care. However, it is not infallible, necessitating human-AI-collaboration to ensure safe implementation. One aspect of AI safety is the models' ability to defer decisions to a human expert when they are likely to misclassify autonomously. Recent research has focused on methods that learn to defer by optimising a surrogate loss function that finds the optimal trade-off between predicting a class label or deferring. However, during clinical translation, models often face challenges such as data shift. Uncertainty quantification methods aim to estimate a model's confidence in its predictions. However, they may also be used as a deferral strategy which does not rely on learning from specific training distribution. We hypothesise that models developed to quantify uncertainty are more robust to out-of-distribution (OOD) input than learned deferral models that have been trained in a supervised fashion. To investigate this hypothesis, we constructed an extensive evaluation study on a large ophthalmology dataset, examining both learned deferral models and established uncertainty quantification methods, assessing their performance in- and out-of-distribution. Specifically, we evaluate their ability to accurately classify glaucoma from fundus images while deferring cases with a high likelihood of error. We find that uncertainty quantification methods may be a promising choice for AI deferral. |
| 2025-08-04 | [Simple Methods Defend RAG Systems Well Against Real-World Attacks](http://arxiv.org/abs/2508.02296v1) | Ilias Triantafyllopoulos, Renyi Qu et al. | Ensuring safety and in-domain responses for Retrieval-Augmented Generation (RAG) systems is paramount in safety-critical applications, yet remains a significant challenge. To address this, we evaluate four methodologies for Out-Of-Domain (OOD) query detection: GPT-4o, regression-based, Principal Component Analysis (PCA)-based, and Neural Collapse (NC), to ensure the RAG system only responds to queries confined to the system's knowledge base. Specifically, our evaluation explores two novel dimensionality reduction and feature separation strategies: \textit{PCA}, where top components are selected using explained variance or OOD separability, and an adaptation of \textit{Neural Collapse Feature Separation}. We validate our approach on standard datasets (StackExchange and MSMARCO) and real-world applications (Substance Use and COVID-19), including tests against LLM-simulated and actual attacks on a COVID-19 vaccine chatbot. Through human and LLM-based evaluations of response correctness and relevance, we confirm that an external OOD detector is crucial for maintaining response relevance. |
| 2025-08-04 | [Framework for Robust Motion Planning of Tethered Multi-Robot Systems in Marine Environments](http://arxiv.org/abs/2508.02287v1) | Markus Buchholz, Ignacio Carlucho et al. | This paper introduces CoralGuide, a novel framework designed for path planning and trajectory optimization for tethered multi-robot systems. We focus on marine robotics, which commonly have tethered configurations of an Autonomous Surface Vehicle (ASV) and an Autonomous Underwater Vehicle (AUV). CoralGuide provides safe navigation in marine environments by enhancing the A* algorithm with specialized heuristics tailored for tethered ASV-AUV systems. Our method integrates catenary curve modelling for tether management and employs Bezier curve interpolation for smoother trajectory planning, ensuring efficient and synchronized operations without compromising safety. Through simulations and real-world experiments, we have validated CoralGuides effectiveness in improving path planning and trajectory optimization, demonstrating its potential to significantly enhance operational capabilities in marine research and infrastructure inspection. |
| 2025-08-04 | [AirTrafficGen: Configurable Air Traffic Scenario Generation with Large Language Models](http://arxiv.org/abs/2508.02269v1) | Dewi Sid William Gould, George De Ath et al. | The manual design of scenarios for Air Traffic Control (ATC) training is a demanding and time-consuming bottleneck that limits the diversity of simulations available to controllers. To address this, we introduce a novel, end-to-end approach, AirTrafficGen, that leverages large language models (LLMs) to automate and control the generation of complex ATC scenarios. Our method uses a purpose-built, graph-based representation to encode sector topology (including airspace geometry, routes, and fixes) into a format LLMs can process. Through rigorous benchmarking, we show that state-of-the-art models like Gemini 2.5 Pro and OpenAI o3 can generate high-traffic scenarios whilst maintaining operational realism. Our engineered prompting enables fine-grained control over interaction presence, type, and location. Initial findings suggest these models are also capable of iterative refinement, correcting flawed scenarios based on simple textual feedback. This approach provides a scalable alternative to manual scenario design, addressing the need for a greater volume and variety of ATC training and validation simulations. More broadly, this work showcases the potential of LLMs for complex planning in safety-critical domains. |
| 2025-08-04 | [Hidden in the Noise: Unveiling Backdoors in Audio LLMs Alignment through Latent Acoustic Pattern Triggers](http://arxiv.org/abs/2508.02175v1) | Liang Lin, Miao Yu et al. | As Audio Large Language Models (ALLMs) emerge as powerful tools for speech processing, their safety implications demand urgent attention. While considerable research has explored textual and vision safety, audio's distinct characteristics present significant challenges. This paper first investigates: Is ALLM vulnerable to backdoor attacks exploiting acoustic triggers? In response to this issue, we introduce Hidden in the Noise (HIN), a novel backdoor attack framework designed to exploit subtle, audio-specific features. HIN applies acoustic modifications to raw audio waveforms, such as alterations to temporal dynamics and strategic injection of spectrally tailored noise. These changes introduce consistent patterns that an ALLM's acoustic feature encoder captures, embedding robust triggers within the audio stream. To evaluate ALLM robustness against audio-feature-based triggers, we develop the AudioSafe benchmark, assessing nine distinct risk types. Extensive experiments on AudioSafe and three established safety datasets reveal critical vulnerabilities in existing ALLMs: (I) audio features like environment noise and speech rate variations achieve over 90% average attack success rate. (II) ALLMs exhibit significant sensitivity differences across acoustic features, particularly showing minimal response to volume as a trigger, and (III) poisoned sample inclusion causes only marginal loss curve fluctuations, highlighting the attack's stealth. |
| 2025-08-04 | [PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable Reinforcement Learning](http://arxiv.org/abs/2508.02159v1) | Dongchi Huang, Jiaqi Wang et al. | Partial observability presents a significant challenge for safe reinforcement learning, as it impedes the identification of potential risks and rewards. Leveraging specific types of privileged information during training to mitigate the effects of partial observability has yielded notable empirical successes. In this paper, we propose Asymmetric Constrained Partially Observable Markov Decision Processes (ACPOMDPs) to theoretically examine the advantages of incorporating privileged information. Building upon ACPOMDPs, we propose the Privileged Information Guided Dreamer, a model-based safe reinforcement learning approach that leverages privileged information to enhance the agent's safety and performance through privileged representation alignment and an asymmetric actor-critic structure. Our empirical results demonstrate that our approach significantly outperforms existing methods in terms of safety and task-centric performance. Meanwhile, compared to alternative privileged model-based reinforcement learning methods, our approach exhibits superior performance and ease of training. |
| 2025-08-04 | [Don't Overthink It: A Survey of Efficient R1-style Large Reasoning Models](http://arxiv.org/abs/2508.02120v1) | Linan Yue, Yichao Du et al. | Recently, Large Reasoning Models (LRMs) have gradually become a research hotspot due to their outstanding performance in handling complex tasks. Among them, DeepSeek R1 has garnered significant attention for its exceptional performance and open-source nature, driving advancements in the research of R1-style LRMs. Unlike traditional Large Language Models (LLMs), these models enhance logical deduction and decision-making capabilities during reasoning by incorporating mechanisms such as long chain-of-thought and self-reflection through reinforcement learning. However, with the widespread application of these models, the problem of overthinking has gradually emerged. Specifically, when generating answers, these models often construct excessively long reasoning chains with redundant or repetitive steps, which leads to reduced reasoning efficiency and may affect the accuracy of the final answer. To this end, various efficient reasoning methods have been proposed, aiming to reduce the length of reasoning paths without compromising model performance and reasoning capability. By reviewing the current research advancements in the field of efficient reasoning methods systematically, we categorize existing works into two main directions based on the lens of single-model optimization versus model collaboration: (1) Efficient Reasoning with Single Model, which focuses on improving the reasoning efficiency of individual models; and (2) Efficient Reasoning with Model Collaboration, which explores optimizing reasoning paths through collaboration among multiple models. Besides, we maintain a public GitHub repository that tracks the latest progress in efficient reasoning methods. |
| 2025-08-04 | [Real-Time Conflict Prediction for Large Truck Merging in Mixed Traffic at Work Zone Lane Closures](http://arxiv.org/abs/2508.02109v1) | Abyad Enan, Abdullah Al Mamun et al. | Large trucks substantially contribute to work zone-related crashes, primarily due to their large size and blind spots. When approaching a work zone, large trucks often need to merge into an adjacent lane because of lane closures caused by construction activities. This study aims to enhance the safety of large truck merging maneuvers in work zones by evaluating the risk associated with merging conflicts and establishing a decision-making strategy for merging based on this risk assessment. To predict the risk of large trucks merging into a mixed traffic stream within a work zone, a Long Short-Term Memory (LSTM) neural network is employed. For a large truck intending to merge, it is critical that the immediate downstream vehicle in the target lane maintains a minimum safe gap to facilitate a safe merging process. Once a conflict-free merging opportunity is predicted, large trucks are instructed to merge in response to the lane closure. Our LSTM-based conflict prediction method is compared against baseline approaches, which include probabilistic risk-based merging, 50th percentile gap-based merging, and 85th percentile gap-based merging strategies. The results demonstrate that our method yields a lower conflict risk, as indicated by reduced Time Exposed Time-to-Collision (TET) and Time Integrated Time-to-Collision (TIT) values relative to the baseline models. Furthermore, the findings indicate that large trucks that use our method can perform early merging while still in motion, as opposed to coming to a complete stop at the end of the current lane prior to closure, which is commonly observed with the baseline approaches. |
| 2025-08-04 | [Towards Immersive Human-X Interaction: A Real-Time Framework for Physically Plausible Motion Synthesis](http://arxiv.org/abs/2508.02106v1) | Kaiyang Ji, Ye Shi et al. | Real-time synthesis of physically plausible human interactions remains a critical challenge for immersive VR/AR systems and humanoid robotics. While existing methods demonstrate progress in kinematic motion generation, they often fail to address the fundamental tension between real-time responsiveness, physical feasibility, and safety requirements in dynamic human-machine interactions. We introduce Human-X, a novel framework designed to enable immersive and physically plausible human interactions across diverse entities, including human-avatar, human-humanoid, and human-robot systems. Unlike existing approaches that focus on post-hoc alignment or simplified physics, our method jointly predicts actions and reactions in real-time using an auto-regressive reaction diffusion planner, ensuring seamless synchronization and context-aware responses. To enhance physical realism and safety, we integrate an actor-aware motion tracking policy trained with reinforcement learning, which dynamically adapts to interaction partners' movements while avoiding artifacts like foot sliding and penetration. Extensive experiments on the Inter-X and InterHuman datasets demonstrate significant improvements in motion quality, interaction continuity, and physical plausibility over state-of-the-art methods. Our framework is validated in real-world applications, including virtual reality interface for human-robot interaction, showcasing its potential for advancing human-robot collaboration. |
| 2025-08-04 | [AlignGuard-LoRA: Alignment-Preserving Fine-Tuning via Fisher-Guided Decomposition and Riemannian-Geodesic Collision Regularization](http://arxiv.org/abs/2508.02079v1) | Amitava Das, Abhilekh Borah et al. | Low-rank adaptation (LoRA) has become a standard tool for efficiently fine-tuning large language models (LLMs). Yet, even minor LoRA updates can induce alignment drift, weakening safety and behavioral constraints through entangled parameter changes. To address this, we propose AlignGuard-LoRA (AGL), a principled framework for preserving alignment during finetuning. AGL introduces several key components: a primary task loss for supervision, Fisher Information Matrix-based regularization to restrict updates in alignment-sensitive subspaces, and task-specific regularization to stabilize the integration of new knowledge. We further introduce collision-aware regularization, blending Riemannian overlap -- which penalizes coordinate-wise interference -- and geodesic separation -- which encourages disjoint update geometry. We curate DriftCaps, a targeted diagnostic benchmark of safe and unsafe prompts designed to quantify alignment drift and safety degradation. Empirical evaluations show that AGL mitigates alignment drift by up to 50% on safety-critical benchmarks without degrading downstream task performance. Comprehensive ablation confirms that each component contributes distinctly to preserving latent safety behaviors. Finally, we derive and validate a scaling law for catastrophic forgetting, revealing that AGL flattens post-finetuning loss escalation while preserving adaptation dynamics. AGL is a structurally grounded refinement of LoRA, ensuring alignment preservation with minimal trade-offs. To encourage further exploration and development, we open-source our implementation. |
| 2025-08-04 | [Risk identification based on similar case retrieval enhancement,](http://arxiv.org/abs/2508.02073v1) | Jiawei Li, Chengye Yang et al. | The goal of construction site risk and hazard identification is to enhance safety management through automation. Existing research based on large language models falls into two categories: image-text matching for collaborative reasoning, which struggles with complex hazard features, and instruction fine-tuning or dialogue guidance using professional datasets, which suffers from high training costs and poor generalization.To address this, we propose a hazard identification method using similar case retrieval enhancement. By integrating external knowledge and retrieved case contexts via prompt fine-tuning, we mitigate misjudgments caused by limited domain knowledge and weak feature associations. Our method includes three modules: retrieval library, image similarity retrieval, and large model retrieval enhancement, enabling efficient recognition without training. Experiments on real construction data show significant improvements. For instance, GLM-4V's recognition accuracy increased to 50\%, a 35.49\% boost. The method enhances accuracy, context understanding, and stability, offering new theoretical and technical support for hazard detection. |
| 2025-08-04 | [TRACEALIGN -- Tracing the Drift: Attributing Alignment Failures to Training-Time Belief Sources in LLMs](http://arxiv.org/abs/2508.02063v1) | Amitava Das, Vinija Jain et al. | Large Language Models (LLMs) fine-tuned to align with human values often exhibit alignment drift, producing unsafe or policy-violating completions when exposed to adversarial prompts, decoding perturbations, or paraphrased jailbreaks. While prior work has behaviorally characterized alignment failure, little is known about the training-time belief sources underlying these failures. We introduce TraceAlign, a unified framework for tracing unsafe completions back to their root causes in the model's training corpus. Central to our approach is the Belief Conflict Index (BCI), which quantifies semantic inconsistency between generated spans and aligned policies, based on retrieved training documents using suffix-array matching. We propose three complementary interventions: (i) TraceShield, an inference-time safety filter that refuses completions with high-BCI spans, (ii) Contrastive Belief Deconfliction Loss, a contrastive fine-tuning objective penalizing high-BCI continuations during DPO, and (iii) Prov-Decode, a provenance-aware decoding strategy that vetoes beam expansions predicted to yield high-BCI spans. Together, these defenses reduce alignment drift by up to 85% on our curated Alignment Drift Benchmark (ADB) while preserving utility on standard tasks, with delta less than 0.2 and improved refusal quality. We further derive a theoretical upper bound on drift likelihood via suffix-array span statistics, linking memorization frequency and length to adversarial reactivation risk. TraceAlign thus provides the first scalable, traceable, and grounded toolkit for understanding and mitigating alignment failures at source. To encourage further exploration and development, we open-source our implementation at: https://anonymous.4open.science/r/tracealign-2DA7 |
| 2025-08-04 | [Bench2ADVLM: A Closed-Loop Benchmark for Vision-language Models in Autonomous Driving](http://arxiv.org/abs/2508.02028v1) | Tianyuan Zhang, Ting Jin et al. | Vision-Language Models (VLMs) have recently emerged as a promising paradigm in autonomous driving (AD). However, current performance evaluation protocols for VLM-based AD systems (ADVLMs) are predominantly confined to open-loop settings with static inputs, neglecting the more realistic and informative closed-loop setting that captures interactive behavior, feedback resilience, and real-world safety. To address this, we introduce Bench2ADVLM, a unified hierarchical closed-loop evaluation framework for real-time, interactive assessment of ADVLMs across both simulation and physical platforms. Inspired by dual-process theories of cognition, we first adapt diverse ADVLMs to simulation environments via a dual-system adaptation architecture. In this design, heterogeneous high-level driving commands generated by target ADVLMs (fast system) are interpreted by a general-purpose VLM (slow system) into standardized mid-level control actions suitable for execution in simulation. To bridge the gap between simulation and reality, we design a physical control abstraction layer that translates these mid-level actions into low-level actuation signals, enabling, for the first time, closed-loop testing of ADVLMs on physical vehicles. To enable more comprehensive evaluation, Bench2ADVLM introduces a self-reflective scenario generation module that automatically explores model behavior and uncovers potential failure modes for safety-critical scenario generation. Overall, Bench2ADVLM establishes a hierarchical evaluation pipeline that seamlessly integrates high-level abstract reasoning, mid-level simulation actions, and low-level real-world execution. Experiments on diverse scenarios across multiple state-of-the-art ADVLMs and physical platforms validate the diagnostic strength of our framework, revealing that existing ADVLMs still exhibit limited performance under closed-loop conditions. |

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



