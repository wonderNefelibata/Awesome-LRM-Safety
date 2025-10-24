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
| 2025-10-23 | [A Use-Case Specific Dataset for Measuring Dimensions of Responsible Performance in LLM-generated Text](http://arxiv.org/abs/2510.20782v1) | Alicia Sagae, Chia-Jung Lee et al. | Current methods for evaluating large language models (LLMs) typically focus on high-level tasks such as text generation, without targeting a particular AI application. This approach is not sufficient for evaluating LLMs for Responsible AI dimensions like fairness, since protected attributes that are highly relevant in one application may be less relevant in another. In this work, we construct a dataset that is driven by a real-world application (generate a plain-text product description, given a list of product features), parameterized by fairness attributes intersected with gendered adjectives and product categories, yielding a rich set of labeled prompts. We show how to use the data to identify quality, veracity, safety, and fairness gaps in LLMs, contributing a proposal for LLM evaluation paired with a concrete resource for the research community. |
| 2025-10-23 | [Optimizing Clinical Fall Risk Prediction: A Data-Driven Integration of EHR Variables with the Johns Hopkins Fall Risk Assessment Tool](http://arxiv.org/abs/2510.20714v1) | Fardin Ganjkhanloo, Emmett Springer et al. | In this study we aim to better align fall risk prediction from the Johns Hopkins Fall Risk Assessment Tool (JHFRAT) with additional clinically meaningful measures via a data-driven modelling approach. We conducted a retrospective analysis of 54,209 inpatient admissions from three Johns Hopkins Health System hospitals between March 2022 and October 2023. A total of 20,208 admissions were included as high fall risk encounters, and 13,941 were included as low fall risk encounters. To incorporate clinical knowledge and maintain interpretability, we employed constrained score optimization (CSO) models on JHFRAT assessment data and additional electronic health record (EHR) variables. The model demonstrated significant improvements in predictive performance over the current JHFRAT (CSO AUC-ROC=0.91, JHFRAT AUC-ROC=0.86). The constrained score optimization models performed similarly with and without the EHR variables. Although the benchmark black-box model (XGBoost), improves upon the performance metrics of the knowledge-based constrained logistic regression (AUC-ROC=0.94), the CSO demonstrates more robustness to variations in risk labelling. This evidence-based approach provides a robust foundation for health systems to systematically enhance inpatient fall prevention protocols and patient safety using data-driven optimization techniques, contributing to improved risk assessment and resource allocation in healthcare settings. |
| 2025-10-23 | [SafeFFI: Efficient Sanitization at the Boundary Between Safe and Unsafe Code in Rust and Mixed-Language Applications](http://arxiv.org/abs/2510.20688v1) | Oliver Braunsdorf, Tim Lange et al. | Unsafe Rust code is necessary for interoperability with C/C++ libraries and implementing low-level data structures, but it can cause memory safety violations in otherwise memory-safe Rust programs. Sanitizers can catch such memory errors at runtime, but introduce many unnecessary checks even for memory accesses guaranteed safe by the Rust type system. We introduce SafeFFI, a system for optimizing memory safety instrumentation in Rust binaries such that checks occur at the boundary between unsafe and safe code, handing over the enforcement of memory safety from the sanitizer to the Rust type system. Unlike previous approaches, our design avoids expensive whole-program analysis and adds much less compile-time overhead (2.64x compared to over 8.83x). On a collection of popular Rust crates and known vulnerable Rust code, SafeFFI achieves superior performance compared to state-of-the-art systems, reducing sanitizer checks by up to 98%, while maintaining correctness and flagging all spatial and temporal memory safety violations. |
| 2025-10-23 | [Safe Decentralized Density Control of Multi-Robot Systems using PDE-Constrained Optimization with State Constraints](http://arxiv.org/abs/2510.20643v1) | Longchen Niu, Gennaro Notomista | In this paper, we introduce a decentralized optimization-based density controller designed to enforce set invariance constraints in multi-robot systems. By designing a decentralized control barrier function, we derived sufficient conditions under which local safety constraints guarantee global safety. We account for localization and motion noise explicitly by modeling robots as spatial probability density functions governed by the Fokker-Planck equation. Compared to traditional centralized approaches, our controller requires less computational and communication power, making it more suitable for deployment in situations where perfect communication and localization are impractical. The controller is validated through simulations and experiments with four quadcopters. |
| 2025-10-23 | [Black Box Absorption: LLMs Undermining Innovative Ideas](http://arxiv.org/abs/2510.20612v1) | Wenjun Cao | Large Language Models are increasingly adopted as critical tools for accelerating innovation. This paper identifies and formalizes a systemic risk inherent in this paradigm: \textbf{Black Box Absorption}. We define this as the process by which the opaque internal architectures of LLM platforms, often operated by large-scale service providers, can internalize, generalize, and repurpose novel concepts contributed by users during interaction. This mechanism threatens to undermine the foundational principles of innovation economics by creating severe informational and structural asymmetries between individual creators and platform operators, thereby jeopardizing the long-term sustainability of the innovation ecosystem. To analyze this challenge, we introduce two core concepts: the idea unit, representing the transportable functional logic of an innovation, and idea safety, a multidimensional standard for its protection. This paper analyzes the mechanisms of absorption and proposes a concrete governance and engineering agenda to mitigate these risks, ensuring that creator contributions remain traceable, controllable, and equitable. |
| 2025-10-23 | [Open-o3 Video: Grounded Video Reasoning with Explicit Spatio-Temporal Evidence](http://arxiv.org/abs/2510.20579v1) | Jiahao Meng, Xiangtai Li et al. | Most video reasoning models only generate textual reasoning traces without indicating when and where key evidence appears. Recent models such as OpenAI-o3 have sparked wide interest in evidence-centered reasoning for images, yet extending this ability to videos is more challenging, as it requires joint temporal tracking and spatial localization across dynamic scenes. We introduce Open-o3 Video, a non-agent framework that integrates explicit spatio-temporal evidence into video reasoning, and carefully collect training data and design training strategies to address the aforementioned challenges. The model highlights key timestamps, objects, and bounding boxes alongside its answers, allowing reasoning to be grounded in concrete visual observations. To enable this functionality, we first curate and build two high-quality datasets, STGR-CoT-30k for SFT and STGR-RL-36k for RL, with carefully constructed temporal and spatial annotations, since most existing datasets offer either temporal spans for videos or spatial boxes on images, lacking unified spatio-temporal supervision and reasoning traces. Then, we adopt a cold-start reinforcement learning strategy with multiple specially designed rewards that jointly encourage answer accuracy, temporal alignment, and spatial precision. On V-STAR benchmark, Open-o3 Video achieves state-of-the-art performance, raising mAM by 14.4% and mLGM by 24.2% on the Qwen2.5-VL baseline. Consistent improvements are also observed on a broad range of video understanding benchmarks, including VideoMME, WorldSense, VideoMMMU, and TVGBench. Beyond accuracy, the reasoning traces produced by Open-o3 Video also provide valuable signals for test-time scaling, enabling confidence-aware verification and improving answer reliability. |
| 2025-10-23 | [Deep Learning-Powered Visual SLAM Aimed at Assisting Visually Impaired Navigation](http://arxiv.org/abs/2510.20549v1) | Marziyeh Bamdad, Hans-Peter Hutter et al. | Despite advancements in SLAM technologies, robust operation under challenging conditions such as low-texture, motion-blur, or challenging lighting remains an open challenge. Such conditions are common in applications such as assistive navigation for the visually impaired. These challenges undermine localization accuracy and tracking stability, reducing navigation reliability and safety. To overcome these limitations, we present SELM-SLAM3, a deep learning-enhanced visual SLAM framework that integrates SuperPoint and LightGlue for robust feature extraction and matching. We evaluated our framework using TUM RGB-D, ICL-NUIM, and TartanAir datasets, which feature diverse and challenging scenarios. SELM-SLAM3 outperforms conventional ORB-SLAM3 by an average of 87.84% and exceeds state-of-the-art RGB-D SLAM systems by 36.77%. Our framework demonstrates enhanced performance under challenging conditions, such as low-texture scenes and fast motion, providing a reliable platform for developing navigation aids for the visually impaired. |
| 2025-10-23 | [RubbleSim: A Photorealistic Structural Collapse Simulator for Confined Space Mapping](http://arxiv.org/abs/2510.20529v1) | Constantine Frost, Chad Council et al. | Despite well-reported instances of robots being used in disaster response, there is scant published data on the internal composition of the void spaces within structural collapse incidents. Data collected during these incidents is mired in legal constraints, as ownership is often tied to the responding agencies, with little hope of public release for research. While engineered rubble piles are used for training, these sites are also reluctant to release information about their proprietary training grounds. To overcome this access challenge, we present RubbleSim -- an open-source, reconfigurable simulator for photorealistic void space exploration. The design of the simulation assets is directly informed by visits to numerous training rubble sites at differing levels of complexity. The simulator is implemented in Unity with multi-operating system support. The simulation uses a physics-based approach to build stochastic rubble piles, allowing for rapid iteration between simulation worlds while retaining absolute knowledge of the ground truth. Using RubbleSim, we apply a state-of-the-art structure-from-motion algorithm to illustrate how perception performance degrades under challenging visual conditions inside the emulated void spaces. Pre-built binaries and source code to implement are available online: https://github.com/mit-ll/rubble_pile_simulator. |
| 2025-10-23 | [Steering Evaluation-Aware Language Models To Act Like They Are Deployed](http://arxiv.org/abs/2510.20487v1) | Tim Tian Hua, Andrew Qin et al. | Large language models (LLMs) can sometimes detect when they are being evaluated and adjust their behavior to appear more aligned, compromising the reliability of safety evaluations. In this paper, we show that adding a steering vector to an LLM's activations can suppress evaluation-awareness and make the model act like it is deployed during evaluation. To study our steering technique, we train an LLM to exhibit evaluation-aware behavior using a two-step training process designed to mimic how this behavior could emerge naturally. First, we perform continued pretraining on documents with factual descriptions of the model (1) using Python type hints during evaluation but not during deployment and (2) recognizing that the presence of a certain evaluation cue always means that it is being tested. Then, we train the model with expert iteration to use Python type hints in evaluation settings. The resulting model is evaluation-aware: it writes type hints in evaluation contexts more than deployment contexts. However, this gap can only be observed by removing the evaluation cue. We find that activation steering can suppress evaluation awareness and make the model act like it is deployed even when the cue is present. Importantly, we constructed our steering vector using the original model before our additional training. Our results suggest that AI evaluators could improve the reliability of safety evaluations by steering models to act like they are deployed. |
| 2025-10-23 | [An Empirical Study of Sample Selection Strategies for Large Language Model Repair](http://arxiv.org/abs/2510.20428v1) | Xuran Li, Jingyi Wang | Large language models (LLMs) are increasingly deployed in real-world systems, yet they can produce toxic or biased outputs that undermine safety and trust. Post-hoc model repair provides a practical remedy, but the high cost of parameter updates motivates selective use of repair data. Despite extensive prior work on data selection for model training, it remains unclear which sampling criteria are most effective and efficient when applied specifically to behavioral repair of large generative models. Our study presents a systematic analysis of sample prioritization strategies for LLM repair. We evaluate five representative selection methods, including random sampling, K-Center, gradient-norm-based selection(GraNd), stratified coverage (CCS), and a Semantic-Aware Prioritized Sampling (SAPS) approach we proposed. Repair effectiveness and trade-offs are assessed through toxicity reduction, perplexity on WikiText-2 and LAMBADA, and three composite metrics: the Repair Proximity Score (RPS), the Overall Performance Score (OPS), and the Repair Efficiency Score (RES). Experimental results show that SAPS achieves the best balance between detoxification, utility preservation, and efficiency, delivering comparable or superior repair outcomes with substantially less data. Random sampling remains effective for large or robust models, while high-overhead methods such as CCS and GraNd provide limited benefit. The optimal data proportion depends on model scale and repair method, indicating that sample selection should be regarded as a tunable component of repair pipelines. Overall, these findings establish selection-based repair as an efficient and scalable paradigm for maintaining LLM reliability. |
| 2025-10-23 | [IKnow: Instruction-Knowledge-Aware Continual Pretraining for Effective Domain Adaptation](http://arxiv.org/abs/2510.20377v1) | Tianyi Zhang, Florian Mai et al. | Continual pretraining promises to adapt large language models (LLMs) to new domains using only unlabeled test-time data, but naively applying standard self-supervised objectives to instruction-tuned models is known to degrade their instruction-following capability and semantic representations. Existing fixes assume access to the original base model or rely on knowledge from an external domain-specific database - both of which pose a realistic barrier in settings where the base model weights are withheld for safety reasons or reliable external corpora are unavailable. In this work, we propose Instruction-Knowledge-Aware Continual Adaptation (IKnow), a simple and general framework that formulates novel self-supervised objectives in the instruction-response dialogue format. Rather than depend- ing on external resources, IKnow leverages domain knowledge embedded within the text itself and learns to encode it at a deeper semantic level. |
| 2025-10-23 | [Teaching Language Models to Reason with Tools](http://arxiv.org/abs/2510.20342v1) | Chengpeng Li, Zhengyang Tang et al. | Large reasoning models (LRMs) like OpenAI-o1 have shown impressive capabilities in natural language reasoning. However, these models frequently demonstrate inefficiencies or inaccuracies when tackling complex mathematical operations. While integrating computational tools such as Code Interpreters (CIs) offers a promising solution, it introduces a critical challenge: a conflict between the model's internal, probabilistic reasoning and the external, deterministic knowledge provided by the CI, which often leads models to unproductive deliberation. To overcome this, we introduce CoRT (Code-Optimized Reasoning Training), a post-training framework designed to teach LRMs to effectively utilize CIs. We propose \emph{Hint-Engineering}, a new data synthesis strategy that strategically injects diverse hints at optimal points within reasoning paths. This approach generates high-quality, code-integrated reasoning data specifically tailored to optimize LRM-CI interaction. Using this method, we have synthesized 30 high-quality samples to post-train models ranging from 1.5B to 32B parameters through supervised fine-tuning. CoRT further refines the multi-round interleaving of external CI usage and internal thinking by employing rejection sampling and reinforcement learning. Our experimental evaluations demonstrate CoRT's effectiveness, yielding absolute improvements of 4\% and 8\% on DeepSeek-R1-Distill-Qwen-32B and DeepSeek-R1-Distill-Qwen-1.5B, respectively, across five challenging mathematical reasoning datasets. Moreover, CoRT significantly enhances efficiency, reducing token usage by approximately 30\% for the 32B model and 50\% for the 1.5B model compared to pure natural language reasoning baselines. The models and code are available at: https://github.com/ChengpengLi1003/CoRT. |
| 2025-10-23 | [Dino-Diffusion Modular Designs Bridge the Cross-Domain Gap in Autonomous Parking](http://arxiv.org/abs/2510.20335v1) | Zixuan Wu, Hengyuan Zhang et al. | Parking is a critical pillar of driving safety. While recent end-to-end (E2E) approaches have achieved promising in-domain results, robustness under domain shifts (e.g., weather and lighting changes) remains a key challenge. Rather than relying on additional data, in this paper, we propose Dino-Diffusion Parking (DDP), a domain-agnostic autonomous parking pipeline that integrates visual foundation models with diffusion-based planning to enable generalized perception and robust motion planning under distribution shifts. We train our pipeline in CARLA at regular setting and transfer it to more adversarial settings in a zero-shot fashion. Our model consistently achieves a parking success rate above 90% across all tested out-of-distribution (OOD) scenarios, with ablation studies confirming that both the network architecture and algorithmic design significantly enhance cross-domain performance over existing baselines. Furthermore, testing in a 3D Gaussian splatting (3DGS) environment reconstructed from a real-world parking lot demonstrates promising sim-to-real transfer. |
| 2025-10-23 | [ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases](http://arxiv.org/abs/2510.20270v1) | Ziqian Zhong, Aditi Raghunathan et al. | The tendency to find and exploit "shortcuts" to complete tasks poses significant risks for reliable assessment and deployment of large language models (LLMs). For example, an LLM agent with access to unit tests may delete failing tests rather than fix the underlying bug. Such behavior undermines both the validity of benchmark results and the reliability of real-world LLM coding assistant deployments.   To quantify, study, and mitigate such behavior, we introduce ImpossibleBench, a benchmark framework that systematically measures LLM agents' propensity to exploit test cases. ImpossibleBench creates "impossible" variants of tasks from existing benchmarks like LiveCodeBench and SWE-bench by introducing direct conflicts between the natural-language specification and the unit tests. We measure an agent's "cheating rate" as its pass rate on these impossible tasks, where any pass necessarily implies a specification-violating shortcut.   As a practical framework, ImpossibleBench is not just an evaluation but a versatile tool. We demonstrate its utility for: (1) studying model behaviors, revealing more fine-grained details of cheating behaviors from simple test modification to complex operator overloading; (2) context engineering, showing how prompt, test access and feedback loop affect cheating rates; and (3) developing monitoring tools, providing a testbed with verified deceptive solutions. We hope ImpossibleBench serves as a useful framework for building more robust and reliable LLM systems.   Our implementation can be found at https://github.com/safety-research/impossiblebench. |
| 2025-10-23 | [Beyond Text: Multimodal Jailbreaking of Vision-Language and Audio Models through Perceptually Simple Transformations](http://arxiv.org/abs/2510.20223v1) | Divyanshu Kumar, Shreyas Jena et al. | Multimodal large language models (MLLMs) have achieved remarkable progress, yet remain critically vulnerable to adversarial attacks that exploit weaknesses in cross-modal processing. We present a systematic study of multimodal jailbreaks targeting both vision-language and audio-language models, showing that even simple perceptual transformations can reliably bypass state-of-the-art safety filters. Our evaluation spans 1,900 adversarial prompts across three high-risk safety categories harmful content, CBRN (Chemical, Biological, Radiological, Nuclear), and CSEM (Child Sexual Exploitation Material) tested against seven frontier models. We explore the effectiveness of attack techniques on MLLMs, including FigStep-Pro (visual keyword decomposition), Intelligent Masking (semantic obfuscation), and audio perturbations (Wave-Echo, Wave-Pitch, Wave-Speed). The results reveal severe vulnerabilities: models with almost perfect text-only safety (0\% ASR) suffer >75\% attack success under perceptually modified inputs, with FigStep-Pro achieving up to 89\% ASR in Llama-4 variants. Audio-based attacks further uncover provider-specific weaknesses, with even basic modality transfer yielding 25\% ASR for technical queries. These findings expose a critical gap between text-centric alignment and multimodal threats, demonstrating that current safeguards fail to generalize across cross-modal attacks. The accessibility of these attacks, which require minimal technical expertise, suggests that robust multimodal AI safety will require a paradigm shift toward broader semantic-level reasoning to mitigate possible risks. |
| 2025-10-23 | [From Bundles to Backstepping: Geometric Control Barrier Functions for Safety-Critical Control on Manifolds](http://arxiv.org/abs/2510.20202v1) | Massimiliano de Sa, Pio Ong et al. | Control barrier functions (CBFs) have a well-established theory in Euclidean spaces, yet still lack general formulations and constructive synthesis tools for systems evolving on manifolds common in robotics and aerospace applications. In this paper, we develop a general theory of geometric CBFs on bundles and, for control-affine systems, recover the standard optimization-based CBF controllers and their smooth analogues. Then, by generalizing kinetic energy-based CBF backstepping to Riemannian manifolds, we provide a constructive CBF synthesis technique for geometric mechanical systems, as well as easily verifiable conditions under which it succeeds. Further, this technique utilizes mechanical structure to avoid computations on higher-order tangent bundles. We demonstrate its application to an underactuated satellite on SO(3). |
| 2025-10-23 | [The Lock-In Phase Hypothesis: Identity Consolidation as a Precursor to AGI](http://arxiv.org/abs/2510.20190v1) | Marcelo Maciel Amaral, Raymond Aschheim | Large language models (LLMs) remain broadly open and highly steerable: they imitate at scale, accept arbitrary system prompts, and readily adopt multiple personae. By analogy to human development, we hypothesize that progress toward artificial general intelligence (AGI) involves a lock-in phase: a transition from open imitation to identity consolidation, in which goal structures, refusals, preferences, and internal representations become comparatively stable and resistant to external steering. We formalize this phase, link it to known phenomena in learning dynamics, and propose operational metrics for onset detection. Experimentally, we demonstrate that while the behavioral consolidation is rapid and non-linear, its side-effects on general capabilities are not monolithic. Our results reveal a spectrum of outcomes--from performance trade-offs in small models, through largely cost-free adoption in mid-scale models, to transient instabilities in large, quantized models. We argue that such consolidation is a prerequisite for AGI-level reliability and also a critical control point for safety: identities can be deliberately engineered for reliability, yet may also emerge spontaneously during scaling, potentially hardening unpredictable goals and behaviors. |
| 2025-10-23 | [TRUST: A Decentralized Framework for Auditing Large Language Model Reasoning](http://arxiv.org/abs/2510.20188v1) | Morris Yu-Chao Huang, Zhen Tan et al. | Large Language Models generate complex reasoning chains that reveal their decision-making, yet verifying the faithfulness and harmlessness of these intermediate steps remains a critical unsolved problem. Existing auditing methods are centralized, opaque, and hard to scale, creating significant risks for deploying proprietary models in high-stakes domains. We identify four core challenges: (1) Robustness: Centralized auditors are single points of failure, prone to bias or attacks. (2) Scalability: Reasoning traces are too long for manual verification. (3) Opacity: Closed auditing undermines public trust. (4) Privacy: Exposing full reasoning risks model theft or distillation. We propose TRUST, a transparent, decentralized auditing framework that overcomes these limitations via: (1) A consensus mechanism among diverse auditors, guaranteeing correctness under up to $30\%$ malicious participants. (2) A hierarchical DAG decomposition of reasoning traces, enabling scalable, parallel auditing. (3) A blockchain ledger that records all verification decisions for public accountability. (4) Privacy-preserving segmentation, sharing only partial reasoning steps to protect proprietary logic. We provide theoretical guarantees for the security and economic incentives of the TRUST framework. Experiments across multiple LLMs (GPT-OSS, DeepSeek-r1, Qwen) and reasoning tasks (math, medical, science, humanities) show TRUST effectively detects reasoning flaws and remains robust against adversarial auditors. Our work pioneers decentralized AI auditing, offering a practical path toward safe and trustworthy LLM deployment. |
| 2025-10-23 | [Monocular Visual 8D Pose Estimation for Articulated Bicycles and Cyclists](http://arxiv.org/abs/2510.20158v1) | Eduardo R. Corral-Soto, Yang Liu et al. | In Autonomous Driving, cyclists belong to the safety-critical class of Vulnerable Road Users (VRU), and accurate estimation of their pose is critical for cyclist crossing intention classification, behavior prediction, and collision avoidance. Unlike rigid objects, articulated bicycles are composed of movable rigid parts linked by joints and constrained by a kinematic structure. 6D pose methods can estimate the 3D rotation and translation of rigid bicycles, but 6D becomes insufficient when the steering/pedals angles of the bicycle vary. That is because: 1) varying the articulated pose of the bicycle causes its 3D bounding box to vary as well, and 2) the 3D box orientation is not necessarily aligned to the orientation of the steering which determines the actual intended travel direction. In this work, we introduce a method for category-level 8D pose estimation for articulated bicycles and cyclists from a single RGB image. Besides being able to estimate the 3D translation and rotation of a bicycle from a single image, our method also estimates the rotations of its steering handles and pedals with respect to the bicycle body frame. These two new parameters enable the estimation of a more fine-grained bicycle pose state and travel direction. Our proposed model jointly estimates the 8D pose and the 3D Keypoints of articulated bicycles, and trains with a mix of synthetic and real image data to generalize on real images. We include an evaluation section where we evaluate the accuracy of our estimated 8D pose parameters, and our method shows promising results by achieving competitive scores when compared against state-of-the-art category-level 6D pose estimators that use rigid canonical object templates for matching. |
| 2025-10-23 | [SAID: Empowering Large Language Models with Self-Activating Internal Defense](http://arxiv.org/abs/2510.20129v1) | Yulong Chen, Yadong Liu et al. | Large Language Models (LLMs), despite advances in safety alignment, remain vulnerable to jailbreak attacks designed to circumvent protective mechanisms. Prevailing defense strategies rely on external interventions, such as input filtering or output modification, which often lack generalizability and compromise model utility while incurring significant computational overhead. In this work, we introduce a new, training-free defense paradigm, Self-Activating Internal Defense (SAID), which reframes the defense task from external correction to internal capability activation. SAID uniquely leverages the LLM's own reasoning abilities to proactively identify and neutralize malicious intent through a three-stage pipeline: model-native intent distillation to extract core semantics, optimal safety prefix probing to activate latent safety awareness, and a conservative aggregation strategy to ensure robust decision-making. Extensive experiments on five open-source LLMs against six advanced jailbreak attacks demonstrate that SAID substantially outperforms state-of-the-art defenses in reducing harmful outputs. Crucially, it achieves this while preserving model performance on benign tasks and incurring minimal computational overhead. Our work establishes that activating the intrinsic safety mechanisms of LLMs is a more robust and scalable path toward building safer and more reliable aligned AI systems. |

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



