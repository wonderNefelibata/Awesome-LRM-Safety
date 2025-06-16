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
| 2025-06-13 | [Tracing LLM Reasoning Processes with Strategic Games: A Framework for Planning, Revision, and Resource-Constrained Decision Making](http://arxiv.org/abs/2506.12012v1) | Xiaopeng Yuan, Xingjian Zhang et al. | Large language models (LLMs) are increasingly used for tasks that require complex reasoning. Most benchmarks focus on final outcomes but overlook the intermediate reasoning steps - such as planning, revision, and decision making under resource constraints. We argue that measuring these internal processes is essential for understanding model behavior and improving reliability. We propose using strategic games as a natural evaluation environment: closed, rule-based systems with clear states, limited resources, and automatic feedback. We introduce a framework that evaluates LLMs along three core dimensions: planning, revision, and resource-constrained decision making. To operationalize this, we define metrics beyond win rate, including overcorrection risk rate, correction success rate, improvement slope, and over-budget ratio. In 4320 adversarial rounds across 12 leading models, ChatGPT-o3-mini achieves the top composite score, with a win rate of 74.7 percent, a correction success rate of 78.6 percent, and an improvement slope of 0.041. By contrast, Qwen-Plus, despite an overcorrection risk rate of 81.6 percent, wins only 25.6 percent of its matches - primarily due to excessive resource use. We also observe a negative correlation between overcorrection risk rate and correction success rate (Pearson r = -0.51, p = 0.093), suggesting that more frequent edits do not always improve outcomes. Our findings highlight the value of assessing not only what LLMs decide but how they arrive at those decisions |
| 2025-06-13 | [Compression Aware Certified Training](http://arxiv.org/abs/2506.11992v1) | Changming Xu, Gagandeep Singh | Deep neural networks deployed in safety-critical, resource-constrained environments must balance efficiency and robustness. Existing methods treat compression and certified robustness as separate goals, compromising either efficiency or safety. We propose CACTUS (Compression Aware Certified Training Using network Sets), a general framework for unifying these objectives during training. CACTUS models maintain high certified accuracy even when compressed. We apply CACTUS for both pruning and quantization and show that it effectively trains models which can be efficiently compressed while maintaining high accuracy and certifiable robustness. CACTUS achieves state-of-the-art accuracy and certified performance for both pruning and quantization on a variety of datasets and input specifications. |
| 2025-06-13 | [Automated Treatment Planning for Interstitial HDR Brachytherapy for Locally Advanced Cervical Cancer using Deep Reinforcement Learning](http://arxiv.org/abs/2506.11957v1) | Mohammadamin Moradi, Runyu Jiang et al. | High-dose-rate (HDR) brachytherapy plays a critical role in the treatment of locally advanced cervical cancer but remains highly dependent on manual treatment planning expertise. The objective of this study is to develop a fully automated HDR brachytherapy planning framework that integrates reinforcement learning (RL) and dose-based optimization to generate clinically acceptable treatment plans with improved consistency and efficiency. We propose a hierarchical two-stage autoplanning framework. In the first stage, a deep Q-network (DQN)-based RL agent iteratively selects treatment planning parameters (TPPs), which control the trade-offs between target coverage and organ-at-risk (OAR) sparing. The agent's state representation includes both dose-volume histogram (DVH) metrics and current TPP values, while its reward function incorporates clinical dose objectives and safety constraints, including D90, V150, V200 for targets, and D2cc for all relevant OARs (bladder, rectum, sigmoid, small bowel, and large bowel). In the second stage, a customized Adam-based optimizer computes the corresponding dwell time distribution for the selected TPPs using a clinically informed loss function. The framework was evaluated on a cohort of patients with complex applicator geometries. The proposed framework successfully learned clinically meaningful TPP adjustments across diverse patient anatomies. For the unseen test patients, the RL-based automated planning method achieved an average score of 93.89%, outperforming the clinical plans which averaged 91.86%. These findings are notable given that score improvements were achieved while maintaining full target coverage and reducing CTV hot spots in most cases. |
| 2025-06-13 | [Improving Large Language Model Safety with Contrastive Representation Learning](http://arxiv.org/abs/2506.11938v1) | Samuel Simko, Mrinmaya Sachan et al. | Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense |
| 2025-06-13 | [Real-World Deployment of a Lane Change Prediction Architecture Based on Knowledge Graph Embeddings and Bayesian Inference](http://arxiv.org/abs/2506.11925v1) | M. Manzour, Catherine M. Elias et al. | Research on lane change prediction has gained a lot of momentum in the last couple of years. However, most research is confined to simulation or results obtained from datasets, leaving a gap between algorithmic advances and on-road deployment. This work closes that gap by demonstrating, on real hardware, a lane-change prediction system based on Knowledge Graph Embeddings (KGEs) and Bayesian inference. Moreover, the ego-vehicle employs a longitudinal braking action to ensure the safety of both itself and the surrounding vehicles. Our architecture consists of two modules: (i) a perception module that senses the environment, derives input numerical features, and converts them into linguistic categories; and communicates them to the prediction module; (ii) a pretrained prediction module that executes a KGE and Bayesian inference model to anticipate the target vehicle's maneuver and transforms the prediction into longitudinal braking action. Real-world hardware experimental validation demonstrates that our prediction system anticipates the target vehicle's lane change three to four seconds in advance, providing the ego vehicle sufficient time to react and allowing the target vehicle to make the lane change safely. |
| 2025-06-13 | [The Throughput Gain of Hypercycle-level Resource Reservation for Time-Triggered Ethernet](http://arxiv.org/abs/2506.11745v1) | Peng Wang, Suman Sourav et al. | Time-Triggered Communication is a key technology for many safety-critical systems, with applications spanning the areas of aerospace and industrial control. Such communication relies on time-triggered flows, with each flow consisting of periodic packets originating from a source and destined for a destination node. Each packet needs to reach its destination before its deadline. Different flows can have different cycle lengths. To achieve assured transmission of time-triggered flows, existing efforts constrain the packets of a flow to be cyclically transmitted along the same path. Under such Fixed Cyclic Scheduling (FCS), reservation for flows with different cycle lengths can become incompatible over a shared link, limiting the total number of admissible flows. Considering the cycle lengths of different flows, a hyper-cycle has length equal to their least common multiple (LCM). It determines the time duration over which the scheduling compatibility of the different flows can be checked. In this work, we propose a more flexible schedule scheme called the Hypercycle-level Flexible Schedule (HFS) scheme, where a flow's resource reservation can change across cycles within a hypercycle. HFS can significantly increase the number of admitted flows by providing more scheduling options while remaining perfectly compatible with existing Time-Triggered Ethernet system. We show that, theoretically the possible capacity gain provided by HFS over FCS can be unbounded. We formulate the joint pathfinding and scheduling problem under HFS as an ILP problem which we prove to be NP-Hard. To solve HFS efficiently, we further propose a least-load-first heuristic (HFS-LLF), solving HFS as a sequence of shortest path problems. Extensive study shows that HFS admits up to 6 times the number of flows achieved by FCS. Moreover, our proposed HFS-LLF can run 104 times faster than solving HFS using a generic solver. |
| 2025-06-13 | [Model Organisms for Emergent Misalignment](http://arxiv.org/abs/2506.11613v1) | Edward Turner, Anna Soligo et al. | Recent work discovered Emergent Misalignment (EM): fine-tuning large language models on narrowly harmful datasets can lead them to become broadly misaligned. A survey of experts prior to publication revealed this was highly unexpected, demonstrating critical gaps in our understanding of model alignment. In this work, we both advance understanding and provide tools for future research. Using new narrowly misaligned datasets, we create a set of improved model organisms that achieve 99% coherence (vs. 67% prior), work with smaller 0.5B parameter models (vs. 32B), and that induce misalignment using a single rank-1 LoRA adapter. We demonstrate that EM occurs robustly across diverse model sizes, three model families, and numerous training protocols including full supervised fine-tuning. Leveraging these cleaner model organisms, we isolate a mechanistic phase transition and demonstrate that it corresponds to a robust behavioural phase transition in all studied organisms. Aligning large language models is critical for frontier AI safety, yet EM exposes how far we are from achieving this robustly. By distilling clean model organisms that isolate a minimal alignment-compromising change, and where this is learnt, we establish a foundation for future research into understanding and mitigating alignment risks in LLMs. |
| 2025-06-13 | [Linearly Solving Robust Rotation Estimation](http://arxiv.org/abs/2506.11547v1) | Yinlong Liu, Tianyu Huang et al. | Rotation estimation plays a fundamental role in computer vision and robot tasks, and extremely robust rotation estimation is significantly useful for safety-critical applications. Typically, estimating a rotation is considered a non-linear and non-convex optimization problem that requires careful design. However, in this paper, we provide some new perspectives that solving a rotation estimation problem can be reformulated as solving a linear model fitting problem without dropping any constraints and without introducing any singularities. In addition, we explore the dual structure of a rotation motion, revealing that it can be represented as a great circle on a quaternion sphere surface. Accordingly, we propose an easily understandable voting-based method to solve rotation estimation. The proposed method exhibits exceptional robustness to noise and outliers and can be computed in parallel with graphics processing units (GPUs) effortlessly. Particularly, leveraging the power of GPUs, the proposed method can obtain a satisfactory rotation solution for large-scale($10^6$) and severely corrupted (99$\%$ outlier ratio) rotation estimation problems under 0.5 seconds. Furthermore, to validate our theoretical framework and demonstrate the superiority of our proposed method, we conduct controlled experiments and real-world dataset experiments. These experiments provide compelling evidence supporting the effectiveness and robustness of our approach in solving rotation estimation problems. |
| 2025-06-13 | [Do Not Immerse and Drive? Prolonged Effects of Cybersickness on Physiological Stress Markers And Cognitive Performance](http://arxiv.org/abs/2506.11536v1) | Daniel Zielasko, Ben Rehling et al. | Extended exposure to virtual reality environments can induce motion sickness, often referred to as cybersickness, which may lead to physiological stress responses and impaired cognitive performance. This study investigates the aftereffects of VR-induced motion sickness with a focus on physiological stress markers and working memory performance. Using a carousel simulation to elicit cybersickness, we assessed subjective discomfort (SSQ, FMS), physiological stress (salivary cortisol, alpha-amylase, electrodermal activity, heart rate), and cognitive performance (n-Back task) over a 90-minute post-exposure period. Our findings demonstrate a significant increase in both subjective and physiological stress indicators following VR exposure, accompanied by a decline in working memory performance. Notably, delayed symptom progression was observed in a substantial proportion of participants, with some reporting peak symptoms up to 90 minutes post-stimulation. Salivary cortisol levels remained elevated throughout the observation period, indicating prolonged stress recovery. These results highlight the need for longer washout phases in XR research and raise safety concerns for professional applications involving post-exposure task performance. |
| 2025-06-13 | [Foundation Models in Autonomous Driving: A Survey on Scenario Generation and Scenario Analysis](http://arxiv.org/abs/2506.11526v1) | Yuan Gao, Mattia Piccinini et al. | For autonomous vehicles, safe navigation in complex environments depends on handling a broad range of diverse and rare driving scenarios. Simulation- and scenario-based testing have emerged as key approaches to development and validation of autonomous driving systems. Traditional scenario generation relies on rule-based systems, knowledge-driven models, and data-driven synthesis, often producing limited diversity and unrealistic safety-critical cases. With the emergence of foundation models, which represent a new generation of pre-trained, general-purpose AI models, developers can process heterogeneous inputs (e.g., natural language, sensor data, HD maps, and control actions), enabling the synthesis and interpretation of complex driving scenarios. In this paper, we conduct a survey about the application of foundation models for scenario generation and scenario analysis in autonomous driving (as of May 2025). Our survey presents a unified taxonomy that includes large language models, vision-language models, multimodal large language models, diffusion models, and world models for the generation and analysis of autonomous driving scenarios. In addition, we review the methodologies, open-source datasets, simulation platforms, and benchmark challenges, and we examine the evaluation metrics tailored explicitly to scenario generation and analysis. Finally, the survey concludes by highlighting the open challenges and research questions, and outlining promising future research directions. All reviewed papers are listed in a continuously maintained repository, which contains supplementary materials and is available at https://github.com/TUM-AVS/FM-for-Scenario-Generation-Analysis. |
| 2025-06-13 | [Investigating Vulnerabilities and Defenses Against Audio-Visual Attacks: A Comprehensive Survey Emphasizing Multimodal Models](http://arxiv.org/abs/2506.11521v1) | Jinming Wen, Xinyi Wu et al. | Multimodal large language models (MLLMs), which bridge the gap between audio-visual and natural language processing, achieve state-of-the-art performance on several audio-visual tasks. Despite the superior performance of MLLMs, the scarcity of high-quality audio-visual training data and computational resources necessitates the utilization of third-party data and open-source MLLMs, a trend that is increasingly observed in contemporary research. This prosperity masks significant security risks. Empirical studies demonstrate that the latest MLLMs can be manipulated to produce malicious or harmful content. This manipulation is facilitated exclusively through instructions or inputs, including adversarial perturbations and malevolent queries, effectively bypassing the internal security mechanisms embedded within the models. To gain a deeper comprehension of the inherent security vulnerabilities associated with audio-visual-based multimodal models, a series of surveys investigates various types of attacks, including adversarial and backdoor attacks. While existing surveys on audio-visual attacks provide a comprehensive overview, they are limited to specific types of attacks, which lack a unified review of various types of attacks. To address this issue and gain insights into the latest trends in the field, this paper presents a comprehensive and systematic review of audio-visual attacks, which include adversarial attacks, backdoor attacks, and jailbreak attacks. Furthermore, this paper also reviews various types of attacks in the latest audio-visual-based MLLMs, a dimension notably absent in existing surveys. Drawing upon comprehensive insights from a substantial review, this paper delineates both challenges and emergent trends for future research on audio-visual attacks and defense. |
| 2025-06-13 | [Taming Stable Diffusion for Computed Tomography Blind Super-Resolution](http://arxiv.org/abs/2506.11496v1) | Chunlei Li, Yilei Shi et al. | High-resolution computed tomography (CT) imaging is essential for medical diagnosis but requires increased radiation exposure, creating a critical trade-off between image quality and patient safety. While deep learning methods have shown promise in CT super-resolution, they face challenges with complex degradations and limited medical training data. Meanwhile, large-scale pre-trained diffusion models, particularly Stable Diffusion, have demonstrated remarkable capabilities in synthesizing fine details across various vision tasks. Motivated by this, we propose a novel framework that adapts Stable Diffusion for CT blind super-resolution. We employ a practical degradation model to synthesize realistic low-quality images and leverage a pre-trained vision-language model to generate corresponding descriptions. Subsequently, we perform super-resolution using Stable Diffusion with a specialized controlling strategy, conditioned on both low-resolution inputs and the generated text descriptions. Extensive experiments show that our method outperforms existing approaches, demonstrating its potential for achieving high-quality CT imaging at reduced radiation doses. Our code will be made publicly available. |
| 2025-06-13 | [On the Natural Robustness of Vision-Language Models Against Visual Perception Attacks in Autonomous Driving](http://arxiv.org/abs/2506.11472v1) | Pedram MohajerAnsari, Amir Salarpour et al. | Autonomous vehicles (AVs) rely on deep neural networks (DNNs) for critical tasks such as traffic sign recognition (TSR), automated lane centering (ALC), and vehicle detection (VD). However, these models are vulnerable to attacks that can cause misclassifications and compromise safety. Traditional defense mechanisms, including adversarial training, often degrade benign accuracy and fail to generalize against unseen attacks. In this work, we introduce Vehicle Vision Language Models (V2LMs), fine-tuned vision-language models specialized for AV perception. Our findings demonstrate that V2LMs inherently exhibit superior robustness against unseen attacks without requiring adversarial training, maintaining significantly higher accuracy than conventional DNNs under adversarial conditions. We evaluate two deployment strategies: Solo Mode, where individual V2LMs handle specific perception tasks, and Tandem Mode, where a single unified V2LM is fine-tuned for multiple tasks simultaneously. Experimental results reveal that DNNs suffer performance drops of 33% to 46% under attacks, whereas V2LMs maintain adversarial accuracy with reductions of less than 8% on average. The Tandem Mode further offers a memory-efficient alternative while achieving comparable robustness to Solo Mode. We also explore integrating V2LMs as parallel components to AV perception to enhance resilience against adversarial threats. Our results suggest that V2LMs offer a promising path toward more secure and resilient AV perception systems. |
| 2025-06-13 | [ReVeal: Self-Evolving Code Agents via Iterative Generation-Verification](http://arxiv.org/abs/2506.11442v1) | Yiyang Jin, Kunzhao Xu et al. | Recent advances in reinforcement learning (RL) with verifiable outcome rewards have significantly improved the reasoning capabilities of large language models (LLMs), especially when combined with multi-turn tool interactions. However, existing methods lack both meaningful verification signals from realistic environments and explicit optimization for verification, leading to unreliable self-verification. To address these limitations, we propose ReVeal, a multi-turn reinforcement learning framework that interleaves code generation with explicit self-verification and tool-based evaluation. ReVeal enables LLMs to autonomously generate test cases, invoke external tools for precise feedback, and improves performance via a customized RL algorithm with dense, per-turn rewards. As a result, ReVeal fosters the co-evolution of a model's generation and verification capabilities through RL training, expanding the reasoning boundaries of the base model, demonstrated by significant gains in Pass@k on LiveCodeBench. It also enables test-time scaling into deeper inference regimes, with code consistently evolving as the number of turns increases during inference, ultimately surpassing DeepSeek-R1-Zero-Qwen-32B. These findings highlight the promise of ReVeal as a scalable and effective paradigm for building more robust and autonomous AI agents. |
| 2025-06-13 | [A Step-by-Step Guide to Creating a Robust Autonomous Drone Testing Pipeline](http://arxiv.org/abs/2506.11400v1) | Yupeng Jiang, Yao Deng et al. | Autonomous drones are rapidly reshaping industries ranging from aerial delivery and infrastructure inspection to environmental monitoring and disaster response. Ensuring the safety, reliability, and efficiency of these systems is paramount as they transition from research prototypes to mission-critical platforms. This paper presents a step-by-step guide to establishing a robust autonomous drone testing pipeline, covering each critical stage: Software-in-the-Loop (SIL) Simulation Testing, Hardware-in-the-Loop (HIL) Testing, Controlled Real-World Testing, and In-Field Testing. Using practical examples, including the marker-based autonomous landing system, we demonstrate how to systematically verify drone system behaviors, identify integration issues, and optimize performance. Furthermore, we highlight emerging trends shaping the future of drone testing, including the integration of Neurosymbolic and LLMs, creating co-simulation environments, and Digital Twin-enabled simulation-based testing techniques. By following this pipeline, developers and researchers can achieve comprehensive validation, minimize deployment risks, and prepare autonomous drones for safe and reliable real-world operations. |
| 2025-06-12 | [Learning a Continue-Thinking Token for Enhanced Test-Time Scaling](http://arxiv.org/abs/2506.11274v1) | Liran Ringel, Elad Tolochinsky et al. | Test-time scaling has emerged as an effective approach for improving language model performance by utilizing additional compute at inference time. Recent studies have shown that overriding end-of-thinking tokens (e.g., replacing "</think>" with "Wait") can extend reasoning steps and improve accuracy. In this work, we explore whether a dedicated continue-thinking token can be learned to trigger extended reasoning. We augment a distilled version of DeepSeek-R1 with a single learned "<|continue-thinking|>" token, training only its embedding via reinforcement learning while keeping the model weights frozen. Our experiments show that this learned token achieves improved accuracy on standard math benchmarks compared to both the baseline model and a test-time scaling approach that uses a fixed token (e.g., "Wait") for budget forcing. In particular, we observe that in cases where the fixed-token approach enhances the base model's accuracy, our method achieves a markedly greater improvement. For example, on the GSM8K benchmark, the fixed-token approach yields a 1.3% absolute improvement in accuracy, whereas our learned-token method achieves a 4.2% improvement over the base model that does not use budget forcing. |
| 2025-06-12 | [How Well Can Reasoning Models Identify and Recover from Unhelpful Thoughts?](http://arxiv.org/abs/2506.10979v1) | Sohee Yang, Sang-Woo Lee et al. | Recent reasoning models show the ability to reflect, backtrack, and self-validate their reasoning, which is crucial in spotting mistakes and arriving at accurate solutions. A natural question that arises is how effectively models can perform such self-reevaluation. We tackle this question by investigating how well reasoning models identify and recover from four types of unhelpful thoughts: uninformative rambling thoughts, thoughts irrelevant to the question, thoughts misdirecting the question as a slightly different question, and thoughts that lead to incorrect answers. We show that models are effective at identifying most unhelpful thoughts but struggle to recover from the same thoughts when these are injected into their thinking process, causing significant performance drops. Models tend to naively continue the line of reasoning of the injected irrelevant thoughts, which showcases that their self-reevaluation abilities are far from a general "meta-cognitive" awareness. Moreover, we observe non/inverse-scaling trends, where larger models struggle more than smaller ones to recover from short irrelevant thoughts, even when instructed to reevaluate their reasoning. We demonstrate the implications of these findings with a jailbreak experiment using irrelevant thought injection, showing that the smallest models are the least distracted by harmful-response-triggering thoughts. Overall, our findings call for improvement in self-reevaluation of reasoning models to develop better reasoning and safer systems. |
| 2025-06-12 | [Build the web for agents, not agents for the web](http://arxiv.org/abs/2506.10953v1) | Xing Han Lù, Gaurav Kamath et al. | Recent advancements in Large Language Models (LLMs) and multimodal counterparts have spurred significant interest in developing web agents -- AI systems capable of autonomously navigating and completing tasks within web environments. While holding tremendous promise for automating complex web interactions, current approaches face substantial challenges due to the fundamental mismatch between human-designed interfaces and LLM capabilities. Current methods struggle with the inherent complexity of web inputs, whether processing massive DOM trees, relying on screenshots augmented with additional information, or bypassing the user interface entirely through API interactions. This position paper advocates for a paradigm shift in web agent research: rather than forcing web agents to adapt to interfaces designed for humans, we should develop a new interaction paradigm specifically optimized for agentic capabilities. To this end, we introduce the concept of an Agentic Web Interface (AWI), an interface specifically designed for agents to navigate a website. We establish six guiding principles for AWI design, emphasizing safety, efficiency, and standardization, to account for the interests of all primary stakeholders. This reframing aims to overcome fundamental limitations of existing interfaces, paving the way for more efficient, reliable, and transparent web agent design, which will be a collaborative effort involving the broader ML community. |
| 2025-06-12 | [Monitoring Decomposition Attacks in LLMs with Lightweight Sequential Monitors](http://arxiv.org/abs/2506.10949v1) | Chen Yueh-Han, Nitish Joshi et al. | Current LLM safety defenses fail under decomposition attacks, where a malicious goal is decomposed into benign subtasks that circumvent refusals. The challenge lies in the existing shallow safety alignment techniques: they only detect harm in the immediate prompt and do not reason about long-range intent, leaving them blind to malicious intent that emerges over a sequence of seemingly benign instructions. We therefore propose adding an external monitor that observes the conversation at a higher granularity. To facilitate our study of monitoring decomposition attacks, we curate the largest and most diverse dataset to date, including question-answering, text-to-image, and agentic tasks. We verify our datasets by testing them on frontier LLMs and show an 87% attack success rate on average on GPT-4o. This confirms that decomposition attack is broadly effective. Additionally, we find that random tasks can be injected into the decomposed subtasks to further obfuscate malicious intents. To defend in real time, we propose a lightweight sequential monitoring framework that cumulatively evaluates each subtask. We show that a carefully prompt engineered lightweight monitor achieves a 93% defense success rate, beating reasoning models like o3 mini as a monitor. Moreover, it remains robust against random task injection and cuts cost by 90% and latency by 50%. Our findings suggest that lightweight sequential monitors are highly effective in mitigating decomposition attacks and are viable in deployment. |
| 2025-06-12 | [Viability of Future Actions: Robust Safety in Reinforcement Learning via Entropy Regularization](http://arxiv.org/abs/2506.10871v1) | Pierre-François Massiani, Alexander von Rohr et al. | Despite the many recent advances in reinforcement learning (RL), the question of learning policies that robustly satisfy state constraints under unknown disturbances remains open. In this paper, we offer a new perspective on achieving robust safety by analyzing the interplay between two well-established techniques in model-free RL: entropy regularization, and constraints penalization. We reveal empirically that entropy regularization in constrained RL inherently biases learning toward maximizing the number of future viable actions, thereby promoting constraints satisfaction robust to action noise. Furthermore, we show that by relaxing strict safety constraints through penalties, the constrained RL problem can be approximated arbitrarily closely by an unconstrained one and thus solved using standard model-free RL. This reformulation preserves both safety and optimality while empirically improving resilience to disturbances. Our results indicate that the connection between entropy regularization and robustness is a promising avenue for further empirical and theoretical investigation, as it enables robust safety in RL through simple reward shaping. |

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



