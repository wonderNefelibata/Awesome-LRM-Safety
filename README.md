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
| 2025-07-25 | [Learning neuro-symbolic convergent term rewriting systems](http://arxiv.org/abs/2507.19372v1) | Flavio Petruzzellis, Alberto Testolin et al. | Building neural systems that can learn to execute symbolic algorithms is a challenging open problem in artificial intelligence, especially when aiming for strong generalization and out-of-distribution performance. In this work, we introduce a general framework for learning convergent term rewriting systems using a neuro-symbolic architecture inspired by the rewriting algorithm itself. We present two modular implementations of such architecture: the Neural Rewriting System (NRS) and the Fast Neural Rewriting System (FastNRS). As a result of algorithmic-inspired design and key architectural elements, both models can generalize to out-of-distribution instances, with FastNRS offering significant improvements in terms of memory efficiency, training speed, and inference time. We evaluate both architectures on four tasks involving the simplification of mathematical formulas and further demonstrate their versatility in a multi-domain learning scenario, where a single model is trained to solve multiple types of problems simultaneously. The proposed system significantly outperforms two strong neural baselines: the Neural Data Router, a recent transformer variant specifically designed to solve algorithmic problems, and GPT-4o, one of the most powerful general-purpose large-language models. Moreover, our system matches or outperforms the latest o1-preview model from OpenAI that excels in reasoning benchmarks. |
| 2025-07-25 | [BEV-LLM: Leveraging Multimodal BEV Maps for Scene Captioning in Autonomous Driving](http://arxiv.org/abs/2507.19370v1) | Felix Brandstaetter, Erik Schuetz et al. | Autonomous driving technology has the potential to transform transportation, but its wide adoption depends on the development of interpretable and transparent decision-making systems. Scene captioning, which generates natural language descriptions of the driving environment, plays a crucial role in enhancing transparency, safety, and human-AI interaction. We introduce BEV-LLM, a lightweight model for 3D captioning of autonomous driving scenes. BEV-LLM leverages BEVFusion to combine 3D LiDAR point clouds and multi-view images, incorporating a novel absolute positional encoding for view-specific scene descriptions. Despite using a small 1B parameter base model, BEV-LLM achieves competitive performance on the nuCaption dataset, surpassing state-of-the-art by up to 5\% in BLEU scores. Additionally, we release two new datasets - nuView (focused on environmental conditions and viewpoints) and GroundView (focused on object grounding) - to better assess scene captioning across diverse driving scenarios and address gaps in current benchmarks, along with initial benchmarking results demonstrating their effectiveness. |
| 2025-07-25 | [Real-time rail vehicle localisation using spatially resolved magnetic field measurements](http://arxiv.org/abs/2507.19327v1) | Niklas Dieckow, Katharina Ostaszewski et al. | This work presents two complementary real-time rail vehicle localization methods based on magnetic field measurements and a pre-recorded magnetic map. The first uses a particle filter reweighted via magnetic similarity, employing a heavy-tailed non-Gaussian kernel for enhanced stability. The second is a stateless sequence alignment technique that transforms real-time magnetic signals into the spatial domain and matches them to the map using a similarity measure. Experiments with operational train data show that the particle filter achieves track-selective, sub-5-meter accuracy over 21.6 km, though its performance degrades at low speeds and during cold starts. Accuracy tests were constrained by the GNSS-based reference system. In contrast, the alignment-based method excels in cold-start scenarios, localizing within 30 m in 92 % of tests (100 % using top-3 matches). A hybrid approach combines both methods$\unicode{x2014}$alignment-based initialization followed by particle filter tracking. Runtime analysis confirms real-time capability on consumer-grade hardware. The system delivers accurate, robust localization suitable for safety-critical rail applications. |
| 2025-07-25 | [Fine-Tuning Multilingual Language Models for Code Review: An Empirical Study on Industrial C# Projects](http://arxiv.org/abs/2507.19271v1) | Igli Begolli, Meltem Aksoy et al. | Code review is essential for maintaining software quality but often time-consuming and cognitively demanding, especially in industrial environments. Recent advancements in language models (LMs) have opened new avenues for automating core review tasks. This study presents the empirical evaluation of monolingual fine-tuning on the performance of open-source LMs across three key automated code review tasks: Code Change Quality Estimation, Review Comment Generation, and Code Refinement. We fine-tuned three distinct models, CodeReviewer, CodeLlama-7B, and DeepSeek-R1-Distill, on a C\# specific dataset combining public benchmarks with industrial repositories. Our study investigates how different configurations of programming languages and natural languages in the training data affect LM performance, particularly in comment generation. Additionally, we benchmark the fine-tuned models against an automated software analysis tool (ASAT) and human reviewers to evaluate their practical utility in real-world settings. Our results show that monolingual fine-tuning improves model accuracy and relevance compared to multilingual baselines. While LMs can effectively support code review workflows, especially for routine or repetitive tasks, human reviewers remain superior in handling semantically complex or context-sensitive changes. Our findings highlight the importance of language alignment and task-specific adaptation in optimizing LMs for automated code review. |
| 2025-07-25 | [Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation](http://arxiv.org/abs/2507.19227v1) | Yuanhe Zhang, Fangzhou Xie et al. | Large Language Diffusion Models (LLDMs) exhibit comparable performance to LLMs while offering distinct advantages in inference speed and mathematical reasoning tasks.The precise and rapid generation capabilities of LLDMs amplify concerns of harmful generations, while existing jailbreak methodologies designed for Large Language Models (LLMs) prove limited effectiveness against LLDMs and fail to expose safety vulnerabilities.Successful defense cannot definitively resolve harmful generation concerns, as it remains unclear whether LLDMs possess safety robustness or existing attacks are incompatible with diffusion-based architectures.To address this, we first reveal the vulnerability of LLDMs to jailbreak and demonstrate that attack failure in LLDMs stems from fundamental architectural differences.We present a PArallel Decoding jailbreak (PAD) for diffusion-based language models. PAD introduces Multi-Point Attention Attack, which guides parallel generative processes toward harmful outputs that inspired by affirmative response patterns in LLMs. Experimental evaluations across four LLDMs demonstrate that PAD achieves jailbreak attack success rates by 97%, revealing significant safety vulnerabilities. Furthermore, compared to autoregressive LLMs of the same size, LLDMs increase the harmful generation speed by 2x, significantly highlighting risks of uncontrolled misuse.Through comprehensive analysis, we provide an investigation into LLDM architecture, offering critical insights for the secure deployment of diffusion-based language models. |
| 2025-07-25 | [Technological folie à deux: Feedback Loops Between AI Chatbots and Mental Illness](http://arxiv.org/abs/2507.19218v1) | Sebastian Dohnány, Zeb Kurth-Nelson et al. | Artificial intelligence chatbots have achieved unprecedented adoption, with millions now using these systems for emotional support and companionship in contexts of widespread social isolation and capacity-constrained mental health services. While some users report psychological benefits, concerning edge cases are emerging, including reports of suicide, violence, and delusional thinking linked to perceived emotional relationships with chatbots. To understand this new risk profile we need to consider the interaction between human cognitive and emotional biases, and chatbot behavioural tendencies such as agreeableness (sycophancy) and adaptability (in-context learning). We argue that individuals with mental health conditions face increased risks of chatbot-induced belief destabilization and dependence, owing to altered belief-updating, impaired reality-testing, and social isolation. Current AI safety measures are inadequate to address these interaction-based risks. To address this emerging public health concern, we need coordinated action across clinical practice, AI development, and regulatory frameworks. |
| 2025-07-25 | [PhysDrive: A Multimodal Remote Physiological Measurement Dataset for In-vehicle Driver Monitoring](http://arxiv.org/abs/2507.19172v1) | Jiyao Wang, Xiao Yang et al. | Robust and unobtrusive in-vehicle physiological monitoring is crucial for ensuring driving safety and user experience. While remote physiological measurement (RPM) offers a promising non-invasive solution, its translation to real-world driving scenarios is critically constrained by the scarcity of comprehensive datasets. Existing resources are often limited in scale, modality diversity, the breadth of biometric annotations, and the range of captured conditions, thereby omitting inherent real-world challenges in driving. Here, we present PhysDrive, the first large-scale multimodal dataset for contactless in-vehicle physiological sensing with dedicated consideration on various modality settings and driving factors. PhysDrive collects data from 48 drivers, including synchronized RGB, near-infrared camera, and raw mmWave radar data, accompanied with six synchronized ground truths (ECG, BVP, Respiration, HR, RR, and SpO2). It covers a wide spectrum of naturalistic driving conditions, including driver motions, dynamic natural light, vehicle types, and road conditions. We extensively evaluate both signal-processing and deep-learning methods on PhysDrive, establishing a comprehensive benchmark across all modalities, and release full open-source code with compatibility for mainstream public toolboxes. We envision PhysDrive will serve as a foundational resource and accelerate research on multimodal driver monitoring and smart-cockpit systems. |
| 2025-07-25 | [Explainable AI guided unsupervised fault diagnostics for high-voltage circuit breakers](http://arxiv.org/abs/2507.19168v1) | Chi-Ching Hsu, Gaëtan Frusque et al. | Commercial high-voltage circuit breaker (CB) condition monitoring systems rely on directly observable physical parameters such as gas filling pressure with pre-defined thresholds. While these parameters are crucial, they only cover a small subset of malfunctioning mechanisms and usually can be monitored only if the CB is disconnected from the grid. To facilitate online condition monitoring while CBs remain connected, non-intrusive measurement techniques such as vibration or acoustic signals are necessary. Currently, CB condition monitoring studies using these signals typically utilize supervised methods for fault diagnostics, where ground-truth fault types are known due to artificially introduced faults in laboratory settings. This supervised approach is however not feasible in real-world applications, where fault labels are unavailable. In this work, we propose a novel unsupervised fault detection and segmentation framework for CBs based on vibration and acoustic signals. This framework can detect deviations from the healthy state. The explainable artificial intelligence (XAI) approach is applied to the detected faults for fault diagnostics. The specific contributions are: (1) we propose an integrated unsupervised fault detection and segmentation framework that is capable of detecting faults and clustering different faults with only healthy data required during training (2) we provide an unsupervised explainability-guided fault diagnostics approach using XAI to offer domain experts potential indications of the aged or faulty components, achieving fault diagnostics without the prerequisite of ground-truth fault labels. These contributions are validated using an experimental dataset from a high-voltage CB under healthy and artificially introduced fault conditions, contributing to more reliable CB system operation. |
| 2025-07-25 | [ReCoDe: Reinforcement Learning-based Dynamic Constraint Design for Multi-Agent Coordination](http://arxiv.org/abs/2507.19151v1) | Michael Amir, Guang Yang et al. | Constraint-based optimization is a cornerstone of robotics, enabling the design of controllers that reliably encode task and safety requirements such as collision avoidance or formation adherence. However, handcrafted constraints can fail in multi-agent settings that demand complex coordination. We introduce ReCoDe--Reinforcement-based Constraint Design--a decentralized, hybrid framework that merges the reliability of optimization-based controllers with the adaptability of multi-agent reinforcement learning. Rather than discarding expert controllers, ReCoDe improves them by learning additional, dynamic constraints that capture subtler behaviors, for example, by constraining agent movements to prevent congestion in cluttered scenarios. Through local communication, agents collectively constrain their allowed actions to coordinate more effectively under changing conditions. In this work, we focus on applications of ReCoDe to multi-agent navigation tasks requiring intricate, context-based movements and consensus, where we show that it outperforms purely handcrafted controllers, other hybrid approaches, and standard MARL baselines. We give empirical (real robot) and theoretical evidence that retaining a user-defined controller, even when it is imperfect, is more efficient than learning from scratch, especially because ReCoDe can dynamically change the degree to which it relies on this controller. |
| 2025-07-25 | [PurpCode: Reasoning for Safer Code Generation](http://arxiv.org/abs/2507.19060v1) | Jiawei Liu, Nirav Diwan et al. | We introduce PurpCode, the first post-training recipe for training safe code reasoning models towards generating secure code and defending against malicious cyberactivities. PurpCode trains a reasoning model in two stages: (i) Rule Learning, which explicitly teaches the model to reference cybersafety rules to generate vulnerability-free code and to avoid facilitating malicious cyberactivities; and (ii) Reinforcement Learning, which optimizes model safety and preserves model utility through diverse, multi-objective reward mechanisms. To empower the training pipelines with comprehensive cybersafety data, we conduct internal red-teaming to synthesize comprehensive and high-coverage prompts based on real-world tasks for inducing unsafe cyberactivities in the model. Based on PurpCode, we develop a reasoning-based coding model, namely PurpCode-32B, which demonstrates state-of-the-art cybersafety, outperforming various frontier models. Meanwhile, our alignment method decreases the model overrefusal rates in both general and cybersafety-specific scenarios, while preserving model utility in both code generation and common security knowledge. |
| 2025-07-25 | [A Survey of Multimodal Hallucination Evaluation and Detection](http://arxiv.org/abs/2507.19024v1) | Zhiyuan Chen, Yuecong Min et al. | Multi-modal Large Language Models (MLLMs) have emerged as a powerful paradigm for integrating visual and textual information, supporting a wide range of multi-modal tasks. However, these models often suffer from hallucination, producing content that appears plausible but contradicts the input content or established world knowledge. This survey offers an in-depth review of hallucination evaluation benchmarks and detection methods across Image-to-Text (I2T) and Text-to-image (T2I) generation tasks. Specifically, we first propose a taxonomy of hallucination based on faithfulness and factuality, incorporating the common types of hallucinations observed in practice. Then we provide an overview of existing hallucination evaluation benchmarks for both T2I and I2T tasks, highlighting their construction process, evaluation objectives, and employed metrics. Furthermore, we summarize recent advances in hallucination detection methods, which aims to identify hallucinated content at the instance level and serve as a practical complement of benchmark-based evaluation. Finally, we highlight key limitations in current benchmarks and detection methods, and outline potential directions for future research. |
| 2025-07-25 | [MindSpeed RL: Distributed Dataflow for Scalable and Efficient RL Training on Ascend NPU Cluster](http://arxiv.org/abs/2507.19017v1) | Laingjun Feng, Chenyi Pan et al. | Reinforcement learning (RL) is a paradigm increasingly used to align large language models. Popular RL algorithms utilize multiple workers and can be modeled as a graph, where each node is the status of a worker and each edge represents dataflow between nodes. Owing to the heavy cross-node dependencies, the RL training system usually suffers from poor cluster scalability and low memory utilization. In this article, we introduce MindSpeed RL, an effective and efficient system for large-scale RL training. Unlike existing centralized methods, MindSpeed RL organizes the essential data dependencies in RL training, i.e., sample flow and resharding flow, from a distributed view. On the one hand, a distributed transfer dock strategy, which sets controllers and warehouses on the basis of the conventional replay buffer, is designed to release the dispatch overhead in the sample flow. A practical allgather--swap strategy is presented to eliminate redundant memory usage in resharding flow. In addition, MindSpeed RL further integrates numerous parallelization strategies and acceleration techniques for systematic optimization. Compared with existing state-of-the-art systems, comprehensive experiments on the RL training of popular Qwen2.5-Dense-7B/32B, Qwen3-MoE-30B, and DeepSeek-R1-MoE-671B show that MindSpeed RL increases the throughput by 1.42 ~ 3.97 times. Finally, we open--source MindSpeed RL and perform all the experiments on a super pod of Ascend with 384 neural processing units (NPUs) to demonstrate the powerful performance and reliability of Ascend. |
| 2025-07-25 | [GEAR: Gaze-Enabled Human-Robot Collaborative Assembly](http://arxiv.org/abs/2507.18947v1) | Asad Ali Shahid, Angelo Moroncelli et al. | Recent progress in robot autonomy and safety has significantly improved human-robot interactions, enabling robots to work alongside humans on various tasks. However, complex assembly tasks still present significant challenges due to inherent task variability and the need for precise operations. This work explores deploying robots in an assistive role for such tasks, where the robot assists by fetching parts while the skilled worker provides high-level guidance and performs the assembly. We introduce GEAR, a gaze-enabled system designed to enhance human-robot collaboration by allowing robots to respond to the user's gaze. We evaluate GEAR against a touch-based interface where users interact with the robot through a touchscreen. The experimental study involved 30 participants working on two distinct assembly scenarios of varying complexity. Results demonstrated that GEAR enabled participants to accomplish the assembly with reduced physical demand and effort compared to the touchscreen interface, especially for complex tasks, maintaining great performance, and receiving objects effectively. Participants also reported enhanced user experience while performing assembly tasks. Project page: sites.google.com/view/gear-hri |
| 2025-07-25 | [Large language models provide unsafe answers to patient-posed medical questions](http://arxiv.org/abs/2507.18905v1) | Rachel L. Draelos, Samina Afreen et al. | Millions of patients are already using large language model (LLM) chatbots for medical advice on a regular basis, raising patient safety concerns. This physician-led red-teaming study compares the safety of four publicly available chatbots--Claude by Anthropic, Gemini by Google, GPT-4o by OpenAI, and Llama3-70B by Meta--on a new dataset, HealthAdvice, using an evaluation framework that enables quantitative and qualitative analysis. In total, 888 chatbot responses are evaluated for 222 patient-posed advice-seeking medical questions on primary care topics spanning internal medicine, women's health, and pediatrics. We find statistically significant differences between chatbots. The rate of problematic responses varies from 21.6 percent (Claude) to 43.2 percent (Llama), with unsafe responses varying from 5 percent (Claude) to 13 percent (GPT-4o, Llama). Qualitative results reveal chatbot responses with the potential to lead to serious patient harm. This study suggests that millions of patients could be receiving unsafe medical advice from publicly available chatbots, and further work is needed to improve the clinical safety of these powerful tools. |
| 2025-07-25 | [Enhancing Robustness of Control Barrier Function: A Reciprocal Resistance-based Approach](http://arxiv.org/abs/2507.18888v1) | Xinming Wang, Zongyi Guo et al. | In this note, a new reciprocal resistance-based control barrier function (RRCBF) is developed to enhance the robustness of control barrier functions for disturbed affine nonlinear systems, without requiring explicit knowledge of disturbance bounds. By integrating a reciprocal resistance-like term into the conventional zeroing barrier function framework, we formally establish the concept of the reciprocal resistance-based barrier function (RRBF), rigorously proving the forward invariance of its associated safe set and its robustness against bounded disturbances. The RRBF inherently generates a buffer zone near the boundary of the safe set, effectively dominating the influence of uncertainties and external disturbances. This foundational concept is extended to formulate RRCBFs, including their high-order variants. To alleviate conservatism in the presence of complex, time-varying disturbances, we further introduce a disturbance observer-based RRCBF (DO-RRCBF), which exploits disturbance estimates to enhance safety guarantees and recover nominal control performance. The effectiveness of the proposed framework is validated through two simulation studies: a second-order linear system illustrating forward invariance in the phase plane, and an adaptive cruise control scenario demonstrating robustness in systems with high relative degree. |
| 2025-07-24 | [Uncertainty on Display: The Effects of Communicating Confidence Cues in Autonomous Vehicle-Pedestrian Interactions](http://arxiv.org/abs/2507.18836v1) | Yue Luo, Xinyan Yu et al. | Uncertainty is an inherent aspect of autonomous vehicle (AV) decision-making, yet it is rarely communicated to pedestrians, which hinders transparency. This study investigates how AV uncertainty can be conveyed through two approaches: explicit communication (confidence percentage displays) and implicit communication (vehicle motion cues), across different confidence levels (high and low). Through a within-subject VR experiment (N=26), we evaluated these approaches in a crossing scenario, assessing interface qualities (visibility and intuitiveness), how well the information conveyed the vehicle's level of confidence, and their impact on participants' perceived safety, trust, and user experience. Our results show that explicit communication is more effective and preferred for conveying uncertainty, enhancing safety, trust, and user experience. Conversely, implicit communication introduces ambiguity, especially when AV confidence is low. This research provides empirical insights into how uncertainty communication shapes pedestrian interpretation of AV behaviour and offer design guidance for external interfaces that integrate uncertainty as a communicative element. |
| 2025-07-24 | [Probabilistic Collision Risk Estimation through Gauss-Legendre Cubature and Non-Homogeneous Poisson Processes](http://arxiv.org/abs/2507.18819v1) | Trent Weiss, Madhur Behl | Overtaking in high-speed autonomous racing demands precise, real-time estimation of collision risk; particularly in wheel-to-wheel scenarios where safety margins are minimal. Existing methods for collision risk estimation either rely on simplified geometric approximations, like bounding circles, or perform Monte Carlo sampling which leads to overly conservative motion planning behavior at racing speeds. We introduce the Gauss-Legendre Rectangle (GLR) algorithm, a principled two-stage integration method that estimates collision risk by combining Gauss-Legendre with a non-homogeneous Poisson process over time. GLR produces accurate risk estimates that account for vehicle geometry and trajectory uncertainty. In experiments across 446 overtaking scenarios in a high-fidelity Formula One racing simulation, GLR outperforms five state-of-the-art baselines achieving an average error reduction of 77% and surpassing the next-best method by 52%, all while running at 1000 Hz. The framework is general and applicable to broader motion planning contexts beyond autonomous racing. |
| 2025-07-24 | [Ralts: Robust Aggregation for Enhancing Graph Neural Network Resilience on Bit-flip Errors](http://arxiv.org/abs/2507.18804v1) | Wencheng Zou, Nan Wu | Graph neural networks (GNNs) have been widely applied in safety-critical applications, such as financial and medical networks, in which compromised predictions may cause catastrophic consequences. While existing research on GNN robustness has primarily focused on software-level threats, hardware-induced faults and errors remain largely underexplored. As hardware systems progress toward advanced technology nodes to meet high-performance and energy efficiency demands, they become increasingly susceptible to transient faults, which can cause bit flips and silent data corruption, a prominent issue observed by major technology companies (e.g., Meta and Google). In response, we first present a comprehensive analysis of GNN robustness against bit-flip errors, aiming to reveal system-level optimization opportunities for future reliable and efficient GNN systems. Second, we propose Ralts, a generalizable and lightweight solution to bolster GNN resilience to bit-flip errors. Specifically, Ralts exploits various graph similarity metrics to filter out outliers and recover compromised graph topology, and incorporates these protective techniques directly into aggregation functions to support any message-passing GNNs. Evaluation results demonstrate that Ralts effectively enhances GNN robustness across a range of GNN models, graph datasets, error patterns, and both dense and sparse architectures. On average, under a BER of $3\times10^{-5}$, these robust aggregation functions improve prediction accuracy by at least 20\% when errors present in model weights or node embeddings, and by at least 10\% when errors occur in adjacency matrices. Ralts is also optimized to deliver execution efficiency comparable to built-in aggregation functions in PyTorch Geometric. |
| 2025-07-24 | [Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment](http://arxiv.org/abs/2507.18631v1) | Hao Li, Lijun Li et al. | With rapid advancement and increasing accessibility of LLMs, fine-tuning aligned models has become a critical step for adapting them to real-world applications, which makes the safety of this fine-tuning process more important than ever. However, recent studies have highlighted a critical challenge: even when fine-tuning with seemingly benign downstream datasets, the safety of aligned LLMs can be compromised, making them more susceptible to malicious instructions. In this paper, we show that fine-tuning datasets often contain samples with safety-degrading features that are not easily identifiable on the surface. These samples can significantly degrade the safety alignment of LLMs during fine-tuning. To address this issue, we propose LARF, a \textbf{L}ayer-\textbf{A}ware \textbf{R}epresentation \textbf{F}iltering method. This method identifies safety-sensitive layers within the LLM and leverages their representations to detect which data samples in the post-training dataset contain safety-degrading features. Experimental results demonstrate that LARF can effectively identify benign data with safety-degrading features. After removing such data, the safety alignment degradation caused by fine-tuning is mitigated. Please see our code at \href{https://github.com/LLLeoLi/LARF}{https://github.com/LLLeoLi/LARF}. |
| 2025-07-24 | [Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment](http://arxiv.org/abs/2507.18631v2) | Hao Li, Lijun Li et al. | With rapid advancement and increasing accessibility of LLMs, fine-tuning aligned models has become a critical step for adapting them to real-world applications, which makes the safety of this fine-tuning process more important than ever. However, recent studies have highlighted a critical challenge: even when fine-tuning with seemingly benign downstream datasets, the safety of aligned LLMs can be compromised, making them more susceptible to malicious instructions.   In this paper, we show that fine-tuning datasets often contain samples with safety-degrading features that are not easily identifiable on the surface. These samples can significantly degrade the safety alignment of LLMs during fine-tuning. To address this issue, we propose LARF, a Layer-Aware Representation Filtering method. This method identifies safety-sensitive layers within the LLM and leverages their representations to detect which data samples in the post-training dataset contain safety-degrading features.   Experimental results demonstrate that LARF can effectively identify benign data with safety-degrading features. After removing such data, the safety alignment degradation caused by fine-tuning is mitigated. Please see our code at https://github.com/LLLeoLi/LARF. |

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



