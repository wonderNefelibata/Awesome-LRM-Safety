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
| 2025-10-20 | [Executable Knowledge Graphs for Replicating AI Research](http://arxiv.org/abs/2510.17795v1) | Yujie Luo, Zhuoyun Yu et al. | Replicating AI research is a crucial yet challenging task for large language model (LLM) agents. Existing approaches often struggle to generate executable code, primarily due to insufficient background knowledge and the limitations of retrieval-augmented generation (RAG) methods, which fail to capture latent technical details hidden in referenced papers. Furthermore, previous approaches tend to overlook valuable implementation-level code signals and lack structured knowledge representations that support multi-granular retrieval and reuse. To overcome these challenges, we propose Executable Knowledge Graphs (xKG), a modular and pluggable knowledge base that automatically integrates technical insights, code snippets, and domain-specific knowledge extracted from scientific literature. When integrated into three agent frameworks with two different LLMs, xKG shows substantial performance gains (10.9% with o3-mini) on PaperBench, demonstrating its effectiveness as a general and extensible solution for automated AI research replication. Code will released at https://github.com/zjunlp/xKG. |
| 2025-10-20 | [VERA-V: Variational Inference Framework for Jailbreaking Vision-Language Models](http://arxiv.org/abs/2510.17759v1) | Qilin Liao, Anamika Lochab et al. | Vision-Language Models (VLMs) extend large language models with visual reasoning, but their multimodal design also introduces new, underexplored vulnerabilities. Existing multimodal red-teaming methods largely rely on brittle templates, focus on single-attack settings, and expose only a narrow subset of vulnerabilities. To address these limitations, we introduce VERA-V, a variational inference framework that recasts multimodal jailbreak discovery as learning a joint posterior distribution over paired text-image prompts. This probabilistic view enables the generation of stealthy, coupled adversarial inputs that bypass model guardrails. We train a lightweight attacker to approximate the posterior, allowing efficient sampling of diverse jailbreaks and providing distributional insights into vulnerabilities. VERA-V further integrates three complementary strategies: (i) typography-based text prompts that embed harmful cues, (ii) diffusion-based image synthesis that introduces adversarial signals, and (iii) structured distractors to fragment VLM attention. Experiments on HarmBench and HADES benchmarks show that VERA-V consistently outperforms state-of-the-art baselines on both open-source and frontier VLMs, achieving up to 53.75% higher attack success rate (ASR) over the best baseline on GPT-4o. |
| 2025-10-20 | [QueST: Incentivizing LLMs to Generate Difficult Problems](http://arxiv.org/abs/2510.17715v1) | Hanxu Hu, Xingxing Zhang et al. | Large Language Models have achieved strong performance on reasoning tasks, solving competition-level coding and math problems. However, their scalability is limited by human-labeled datasets and the lack of large-scale, challenging coding problem training data. Existing competitive coding datasets contain only thousands to tens of thousands of problems. Previous synthetic data generation methods rely on either augmenting existing instruction datasets or selecting challenging problems from human-labeled data. In this paper, we propose QueST, a novel framework which combines difficulty-aware graph sampling and difficulty-aware rejection fine-tuning that directly optimizes specialized generators to create challenging coding problems. Our trained generators demonstrate superior capability compared to even GPT-4o at creating challenging problems that benefit downstream performance. We leverage QueST to generate large-scale synthetic coding problems, which we then use to distill from strong teacher models with long chain-of-thought or to conduct reinforcement learning for smaller models, proving effective in both scenarios. Our distillation experiments demonstrate significant performance gains. Specifically, after fine-tuning Qwen3-8B-base on 100K difficult problems generated by QueST, we surpass the performance of the original Qwen3-8B on LiveCodeBench. With an additional 112K examples (i.e., 28K human-written problems paired with multiple synthetic solutions), our 8B model matches the performance of the much larger DeepSeek-R1-671B. These findings indicate that generating complex problems via QueST offers an effective and scalable approach to advancing the frontiers of competitive coding and reasoning for large language models. |
| 2025-10-20 | [CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks](http://arxiv.org/abs/2510.17687v1) | Xu Zhang, Hao Li et al. | Multimodal Large Language Models (MLLMs) achieve strong reasoning and perception capabilities but are increasingly vulnerable to jailbreak attacks. While existing work focuses on explicit attacks, where malicious content resides in a single modality, recent studies reveal implicit attacks, in which benign text and image inputs jointly express unsafe intent. Such joint-modal threats are difficult to detect and remain underexplored, largely due to the scarcity of high-quality implicit data. We propose ImpForge, an automated red-teaming pipeline that leverages reinforcement learning with tailored reward modules to generate diverse implicit samples across 14 domains. Building on this dataset, we further develop CrossGuard, an intent-aware safeguard providing robust and comprehensive defense against both explicit and implicit threats. Extensive experiments across safe and unsafe benchmarks, implicit and explicit attacks, and multiple out-of-domain settings demonstrate that CrossGuard significantly outperforms existing defenses, including advanced MLLMs and guardrails, achieving stronger security while maintaining high utility. This offers a balanced and practical solution for enhancing MLLM robustness against real-world multimodal threats. |
| 2025-10-20 | [SARSteer: Safeguarding Large Audio Language Models via Safe-Ablated Refusal Steering](http://arxiv.org/abs/2510.17633v1) | Weilin Lin, Jianze Li et al. | Large Audio-Language Models (LALMs) are becoming essential as a powerful multimodal backbone for real-world applications. However, recent studies show that audio inputs can more easily elicit harmful responses than text, exposing new risks toward deployment. While safety alignment has made initial advances in LLMs and Large Vision-Language Models (LVLMs), we find that vanilla adaptation of these approaches to LALMs faces two key limitations: 1) LLM-based steering fails under audio input due to the large distributional gap between activations, and 2) prompt-based defenses induce over-refusals on benign-speech queries. To address these challenges, we propose Safe-Ablated Refusal Steering (SARSteer), the first inference-time defense framework for LALMs. Specifically, SARSteer leverages text-derived refusal steering to enforce rejection without manipulating audio inputs and introduces decomposed safe-space ablation to mitigate over-refusal. Extensive experiments demonstrate that SARSteer significantly improves harmful-query refusal while preserving benign responses, establishing a principled step toward safety alignment in LALMs. |
| 2025-10-20 | [An Empirical Study of Lagrangian Methods in Safe Reinforcement Learning](http://arxiv.org/abs/2510.17564v1) | Lindsay Spoor, Álvaro Serra-Gómez et al. | In safety-critical domains such as robotics, navigation and power systems, constrained optimization problems arise where maximizing performance must be carefully balanced with associated constraints. Safe reinforcement learning provides a framework to address these challenges, with Lagrangian methods being a popular choice. However, the effectiveness of Lagrangian methods crucially depends on the choice of the Lagrange multiplier $\lambda$, which governs the trade-off between return and constraint cost. A common approach is to update the multiplier automatically during training. Although this is standard in practice, there remains limited empirical evidence on the robustness of an automated update and its influence on overall performance. Therefore, we analyze (i) optimality and (ii) stability of Lagrange multipliers in safe reinforcement learning across a range of tasks. We provide $\lambda$-profiles that give a complete visualization of the trade-off between return and constraint cost of the optimization problem. These profiles show the highly sensitive nature of $\lambda$ and moreover confirm the lack of general intuition for choosing the optimal value $\lambda^*$. Our findings additionally show that automated multiplier updates are able to recover and sometimes even exceed the optimal performance found at $\lambda^*$ due to the vast difference in their learning trajectories. Furthermore, we show that automated multiplier updates exhibit oscillatory behavior during training, which can be mitigated through PID-controlled updates. However, this method requires careful tuning to achieve consistently better performance across tasks. This highlights the need for further research on stabilizing Lagrangian methods in safe reinforcement learning. The code used to reproduce our results can be found at https://github.com/lindsayspoor/Lagrangian_SafeRL. |
| 2025-10-20 | [Formally Exploring Time-Series Anomaly Detection Evaluation Metrics](http://arxiv.org/abs/2510.17562v1) | Dennis Wagner, Arjun Nair et al. | Undetected anomalies in time series can trigger catastrophic failures in safety-critical systems, such as chemical plant explosions or power grid outages. Although many detection methods have been proposed, their performance remains unclear because current metrics capture only narrow aspects of the task and often yield misleading results. We address this issue by introducing verifiable properties that formalize essential requirements for evaluating time-series anomaly detection. These properties enable a theoretical framework that supports principled evaluations and reliable comparisons. Analyzing 37 widely used metrics, we show that most satisfy only a few properties, and none satisfy all, explaining persistent inconsistencies in prior results. To close this gap, we propose LARM, a flexible metric that provably satisfies all properties, and extend it to ALARM, an advanced variant meeting stricter requirements. |
| 2025-10-20 | [HumanMPC - Safe and Efficient MAV Navigation among Humans](http://arxiv.org/abs/2510.17525v1) | Simon Schaefer, Helen Oleynikova et al. | Safe and efficient robotic navigation among humans is essential for integrating robots into everyday environments. Most existing approaches focus on simplified 2D crowd navigation and fail to account for the full complexity of human body dynamics beyond root motion. We present HumanMPC, a Model Predictive Control (MPC) framework for 3D Micro Air Vehicle (MAV) navigation among humans that combines theoretical safety guarantees with data-driven models for realistic human motion forecasting. Our approach introduces a novel twist to reachability-based safety formulation that constrains only the initial control input for safety while modeling its effects over the entire planning horizon, enabling safe yet efficient navigation. We validate HumanMPC in both simulated experiments using real human trajectories and in the real-world, demonstrating its effectiveness across tasks ranging from goal-directed navigation to visual servoing for human tracking. While we apply our method to MAVs in this work, it is generic and can be adapted by other platforms. Our results show that the method ensures safety without excessive conservatism and outperforms baseline approaches in both efficiency and reliability. |
| 2025-10-20 | [SAFE-D: A Spatiotemporal Detection Framework for Abnormal Driving Among Parkinson's Disease-like Drivers](http://arxiv.org/abs/2510.17517v1) | Hangcheng Cao, Baixiang Huang et al. | A driver's health state serves as a determinant factor in driving behavioral regulation. Subtle deviations from normalcy can lead to operational anomalies, posing risks to public transportation safety. While prior efforts have developed detection mechanisms for functionally-driven temporary anomalies such as drowsiness and distraction, limited research has addressed pathologically-triggered deviations, especially those stemming from chronic medical conditions. To bridge this gap, we investigate the driving behavior of Parkinson's disease patients and propose SAFE-D, a novel framework for detecting Parkinson-related behavioral anomalies to enhance driving safety. Our methodology starts by performing analysis of Parkinson's disease symptomatology, focusing on primary motor impairments, and establishes causal links to degraded driving performance. To represent the subclinical behavioral variations of early-stage Parkinson's disease, our framework integrates data from multiple vehicle control components to build a behavioral profile. We then design an attention-based network that adaptively prioritizes spatiotemporal features, enabling robust anomaly detection under physiological variability. Finally, we validate SAFE-D on the Logitech G29 platform and CARLA simulator, using data from three road maps to emulate real-world driving. Our results show SAFE-D achieves 96.8% average accuracy in distinguishing normal and Parkinson-affected driving patterns. |
| 2025-10-20 | [Bridging the gap between experimental burden and statistical power for quantiles equivalence testing](http://arxiv.org/abs/2510.17514v1) | Jun Wu, Stéphane Guerrier et al. | Testing the equivalence of multiple quantiles between two populations is important in many scientific applications, such as clinical trials, where conventional mean-based methods may be inadequate. This is particularly relevant in bridging studies that compare drug responses across different experimental conditions or patient populations. These studies often aim to assess whether a proposed dose for a target population achieves pharmacokinetic levels comparable to those of a reference population where efficacy and safety have been established. The focus is on extreme quantiles which directly inform both efficacy and safety assessments. When analyzing heterogeneous Gaussian samples, where a single quantile of interest is estimated, the existing Two One-Sided Tests method for quantile equivalence testing (qTOST) tends to be overly conservative. To mitigate this behavior, we introduce $\alpha$-qTOST, a finite-sample adjustment that achieves uniformly higher power compared to qTOST while maintaining the test size at the nominal level. Moreover, we extend the quantile equivalence framework to simultaneously assess equivalence across multiple quantiles. Through theoretical guarantees and an extensive simulation study, we demonstrate that $\alpha$-qTOST offers substantial improvements, especially when testing extreme quantiles under heteroskedasticity and with small, unbalanced sample sizes. We illustrate these advantages through two case studies, one in HIV drug development, where a bridging clinical trial examines exposure distributions between male and female populations with unbalanced sample sizes, and another in assessing the reproducibility of an identical experimental protocol performed by different operators for generating biodistribution profiles of topically administered and locally acting products. |
| 2025-10-20 | [Deep Self-Evolving Reasoning](http://arxiv.org/abs/2510.17498v1) | Zihan Liu, Shun Zheng et al. | Long-form chain-of-thought reasoning has become a cornerstone of advanced reasoning in large language models. While recent verification-refinement frameworks have enabled proprietary models to solve Olympiad-level problems, their effectiveness hinges on strong, reliable verification and correction capabilities, which remain fragile in open-weight, smaller-scale models. This work demonstrates that even with weak verification and refinement capabilities on hard tasks, the reasoning limits of such models can be substantially extended through a probabilistic paradigm we call Deep Self-Evolving Reasoning (DSER). We conceptualize iterative reasoning as a Markov chain, where each step represents a stochastic transition in the solution space. The key insight is that convergence to a correct solution is guaranteed as long as the probability of improvement marginally exceeds that of degradation. By running multiple long-horizon, self-evolving processes in parallel, DSER amplifies these small positive tendencies, enabling the model to asymptotically approach correct answers. Empirically, we apply DSER to the DeepSeek-R1-0528-Qwen3-8B model. On the challenging AIME 2024-2025 benchmark, DSER solves 5 out of 9 previously unsolvable problems and boosts overall performance, enabling this compact model to surpass the single-turn accuracy of its 600B-parameter teacher through majority voting. Beyond its immediate utility for test-time scaling, the DSER framework serves to diagnose the fundamental limitations of current open-weight reasoners. By clearly delineating their shortcomings in self-verification, refinement, and stability, our findings establish a clear research agenda for developing next-generation models with powerful, intrinsic self-evolving capabilities. |
| 2025-10-20 | [Empowering Real-World: A Survey on the Technology, Practice, and Evaluation of LLM-driven Industry Agents](http://arxiv.org/abs/2510.17491v1) | Yihong Tang, Kehai Chen et al. | With the rise of large language models (LLMs), LLM agents capable of autonomous reasoning, planning, and executing complex tasks have become a frontier in artificial intelligence. However, how to translate the research on general agents into productivity that drives industry transformations remains a significant challenge. To address this, this paper systematically reviews the technologies, applications, and evaluation methods of industry agents based on LLMs. Using an industry agent capability maturity framework, it outlines the evolution of agents in industry applications, from "process execution systems" to "adaptive social systems." First, we examine the three key technological pillars that support the advancement of agent capabilities: Memory, Planning, and Tool Use. We discuss how these technologies evolve from supporting simple tasks in their early forms to enabling complex autonomous systems and collective intelligence in more advanced forms. Then, we provide an overview of the application of industry agents in real-world domains such as digital engineering, scientific discovery, embodied intelligence, collaborative business execution, and complex system simulation. Additionally, this paper reviews the evaluation benchmarks and methods for both fundamental and specialized capabilities, identifying the challenges existing evaluation systems face regarding authenticity, safety, and industry specificity. Finally, we focus on the practical challenges faced by industry agents, exploring their capability boundaries, developmental potential, and governance issues in various scenarios, while providing insights into future directions. By combining technological evolution with industry practices, this review aims to clarify the current state and offer a clear roadmap and theoretical foundation for understanding and building the next generation of industry agents. |
| 2025-10-20 | [Agentic Reinforcement Learning for Search is Unsafe](http://arxiv.org/abs/2510.17431v1) | Yushi Yang, Shreyansh Padarha et al. | Agentic reinforcement learning (RL) trains large language models to autonomously call tools during reasoning, with search as the most common application. These models excel at multi-step reasoning tasks, but their safety properties are not well understood. In this study, we show that RL-trained search models inherit refusal from instruction tuning and often deflect harmful requests by turning them into safe queries. However, this safety is fragile. Two simple attacks, one that forces the model to begin response with search (Search attack), another that encourages models to repeatedly search (Multi-search attack), trigger cascades of harmful searches and answers. Across two model families (Qwen, Llama) with both local and web search, these attacks lower refusal rates by up to 60.0%, answer safety by 82.5%, and search-query safety by 82.4%. The attacks succeed by triggering models to generate harmful, request-mirroring search queries before they can generate the inherited refusal tokens. This exposes a core weakness of current RL training: it rewards continued generation of effective queries without accounting for their harmfulness. As a result, RL search models have vulnerabilities that users can easily exploit, making it urgent to develop safety-aware agentic RL pipelines optimising for safe search. |
| 2025-10-20 | [Introducing Linear Implication Types to $λ_{GT}$ for Computing With Incomplete Graphs](http://arxiv.org/abs/2510.17429v1) | Jin Sano, Naoki Yamamoto et al. | Designing programming languages that enable intuitive and safe manipulation of data structures is a critical research challenge. Conventional destructive memory operations using pointers are complex and prone to errors. Existing type systems, such as affine types and shape types, address this problem towards safe manipulation of heaps and pointers, but design of high-level declarative languages that allow us to manipulate complex pointer data structures at a higher level of abstraction is largely an open problem. The $\lambda_{GT}$ language, a purely functional programming language that treats hypergraphs (hereafter referred to as graphs) as primary data structures, addresses some of these challenges. By abstracting data with shared references and cycles as graphs, it enables declarative operations through pattern matching and leverages its type system to guarantee safety of these operations. Nevertheless, the previously proposed type system of $\lambda_{GT}$ leaves two significant open challenges. First, the type system does not support \emph{incomplete graphs}, that is, graphs in which some elements are missing from the graphs of user-defined types. Second, the type system relies on dynamic type checking during pattern matching. This study addresses these two challenges by incorporating linear implication into the $\lambda_{GT}$ type system, while introducing new constraints to ensure its soundness. |
| 2025-10-20 | [Integrating Trustworthy Artificial Intelligence with Energy-Efficient Robotic Arms for Waste Sorting](http://arxiv.org/abs/2510.17408v1) | Halima I. Kure, Jishna Retnakumari et al. | This paper presents a novel methodology that integrates trustworthy artificial intelligence (AI) with an energy-efficient robotic arm for intelligent waste classification and sorting. By utilizing a convolutional neural network (CNN) enhanced through transfer learning with MobileNetV2, the system accurately classifies waste into six categories: plastic, glass, metal, paper, cardboard, and trash. The model achieved a high training accuracy of 99.8% and a validation accuracy of 80.5%, demonstrating strong learning and generalization. A robotic arm simulator is implemented to perform virtual sorting, calculating the energy cost for each action using Euclidean distance to ensure optimal and efficient movement. The framework incorporates key elements of trustworthy AI, such as transparency, robustness, fairness, and safety, making it a reliable and scalable solution for smart waste management systems in urban settings. |
| 2025-10-20 | [Enhancing 5G V2X Mode 2 for Sporadic Traffic](http://arxiv.org/abs/2510.17395v1) | Dmitry Bankov, Artem Krasilov et al. | The emerging road safety and autonomous vehicle applications require timely and reliable data delivery between vehicles and between vehicles and infrastructure. To satisfy this demand, 3GPP develops a 5G Vehicle-to-Everything (V2X) technology. Depending on the served traffic type, 5G V2X specifications propose two channel access methods: (i) Mode 1, according to which a base station allocates resources to users, and (ii) Mode 2, according to which users autonomously select resources for their transmissions. In the paper, we consider a scenario with sporadic traffic, e.g., a vehicle generates a packet at a random time moment when it detects a dangerous situation, which imposes strict requirements on delay and reliability. To satisfy strict delay requirements, vehicles use Mode 2. We analyze the performance of Mode 2 for sporadic traffic and propose several approaches to improve it. Simulation results show that the proposed approaches can increase the system capacity by up to 40% with a low impact on complexity. |
| 2025-10-20 | [TabR1: Taming GRPO for tabular reasoning LLMs](http://arxiv.org/abs/2510.17385v1) | Pengxiang Cai, Zihao Gao et al. | Tabular prediction has traditionally relied on gradient-boosted decision trees and specialized deep learning models, which excel within tasks but provide limited interpretability and weak transfer across tables. Reasoning large language models (LLMs) promise cross-task adaptability with trans- parent reasoning traces, yet their potential has not been fully realized for tabular data. This paper presents TabR1, the first reasoning LLM for tabular prediction with multi-step reasoning. At its core is Permutation Relative Policy Optimization (PRPO), a simple yet efficient reinforcement learning method that encodes column-permutation invariance as a structural prior. By construct- ing multiple label-preserving permutations per sample and estimating advantages both within and across permutations, PRPO transforms sparse rewards into dense learning signals and improves generalization. With limited supervision, PRPO activates the reasoning ability of LLMs for tabular prediction, enhancing few-shot and zero-shot performance as well as interpretability. Comprehensive experiments demonstrate that TabR1 achieves performance comparable to strong baselines under full-supervision fine-tuning. In the zero-shot setting, TabR1 approaches the performance of strong baselines under the 32-shot setting. Moreover, TabR1 (8B) substantially outperforms much larger LLMs across various tasks, achieving up to 53.17% improvement over DeepSeek-R1 (685B). |
| 2025-10-20 | [Beyond Binary Out-of-Distribution Detection: Characterizing Distributional Shifts with Multi-Statistic Diffusion Trajectories](http://arxiv.org/abs/2510.17381v1) | Achref Jaziri, Martin Rogmann et al. | Detecting out-of-distribution (OOD) data is critical for machine learning, be it for safety reasons or to enable open-ended learning. However, beyond mere detection, choosing an appropriate course of action typically hinges on the type of OOD data encountered. Unfortunately, the latter is generally not distinguished in practice, as modern OOD detection methods collapse distributional shifts into single scalar outlier scores. This work argues that scalar-based methods are thus insufficient for OOD data to be properly contextualized and prospectively exploited, a limitation we overcome with the introduction of DISC: Diffusion-based Statistical Characterization. DISC leverages the iterative denoising process of diffusion models to extract a rich, multi-dimensional feature vector that captures statistical discrepancies across multiple noise levels. Extensive experiments on image and tabular benchmarks show that DISC matches or surpasses state-of-the-art detectors for OOD detection and, crucially, also classifies OOD type, a capability largely absent from prior work. As such, our work enables a shift from simple binary OOD detection to a more granular detection. |
| 2025-10-20 | [Bridging Embodiment Gaps: Deploying Vision-Language-Action Models on Soft Robots](http://arxiv.org/abs/2510.17369v1) | Haochen Su, Cristian Meo et al. | Robotic systems are increasingly expected to operate in human-centered, unstructured environments where safety, adaptability, and generalization are essential. Vision-Language-Action (VLA) models have been proposed as a language guided generalized control framework for real robots. However, their deployment has been limited to conventional serial link manipulators. Coupled by their rigidity and unpredictability of learning based control, the ability to safely interact with the environment is missing yet critical. In this work, we present the deployment of a VLA model on a soft continuum manipulator to demonstrate autonomous safe human-robot interaction. We present a structured finetuning and deployment pipeline evaluating two state-of-the-art VLA models (OpenVLA-OFT and $\pi_0$) across representative manipulation tasks, and show while out-of-the-box policies fail due to embodiment mismatch, through targeted finetuning the soft robot performs equally to the rigid counterpart. Our findings highlight the necessity of finetuning for bridging embodiment gaps, and demonstrate that coupling VLA models with soft robots enables safe and flexible embodied AI in human-shared environments. |
| 2025-10-20 | [Modelling complexity in system safety: generalizing the D2T2 methodology](http://arxiv.org/abs/2510.17351v1) | Silvia Tolo, John Andrews | Although Fault Tree and Event Tree analysis are still today the standard approach to system safety analysis for many engineering sectors, these techniques lack the capabilities of fully capturing the realistic, dynamic behaviour of complex systems, which results in a dense network of dependencies at any level, i.e. between components, trains of components or subsystems. While these limitations are well recognised across both industry and academia, the shortage of alternative tools able to tackle such challenges while retaining the computational feasibility of the analysis keeps fuelling the long-lived success of Fault Tree and Event Tree modelling. Analysts and regulators often rely on the use of conservative assumptions to mitigate the effect of oversimplifications associated with the use of such techniques. However, this results in the analysis output to be characterised by an unknown level of conservatism, with potential consequences on market competitiveness (i.e., over-conservatism) or safety (i.e., under-conservatism). This study proposes a generalization of the Dynamic and Dependent Tree Theory, which offers theoretical tools for the systematic integration of dependency modelling within the traditional Fault and Event Tree analysis framework. This is achieved by marrying the traditional combinatorial nature of failure analysis, formalised by the Fault and Event Tree language, with more flexible modelling solutions, which provide the flexibility required to capture complex system features. The main advantage of the proposed approach in comparison to existent solutions is the ability to take into account, under the same modelling framework, any type of dependency regardless of its nature and location, while retaining the familiarity and effectiveness of traditional safety modelling. |

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



