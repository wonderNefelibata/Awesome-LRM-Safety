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
| 2026-02-27 | [SafeGen-LLM: Enhancing Safety Generalization in Task Planning for Robotic Systems](http://arxiv.org/abs/2602.24235v1) | Jialiang Fan, Weizhe Xu et al. | Safety-critical task planning in robotic systems remains challenging: classical planners suffer from poor scalability, Reinforcement Learning (RL)-based methods generalize poorly, and base Large Language Models (LLMs) cannot guarantee safety. To address this gap, we propose safety-generalizable large language models, named SafeGen-LLM. SafeGen-LLM can not only enhance the safety satisfaction of task plans but also generalize well to novel safety properties in various domains. We first construct a multi-domain Planning Domain Definition Language 3 (PDDL3) benchmark with explicit safety constraints. Then, we introduce a two-stage post-training framework: Supervised Fine-Tuning (SFT) on a constraint-compliant planning dataset to learn planning syntax and semantics, and Group Relative Policy Optimization (GRPO) guided by fine-grained reward machines derived from formal verification to enforce safety alignment and by curriculum learning to better handle complex tasks. Extensive experiments show that SafeGen-LLM achieves strong safety generalization and outperforms frontier proprietary baselines across multi-domain planning tasks and multiple input formats (e.g., PDDLs and natural language). |
| 2026-02-27 | [Resilient Strategies for Stochastic Systems: How Much Does It Take to Break a Winning Strategy?](http://arxiv.org/abs/2602.24191v1) | Kush Grover, Markel Zubia et al. | We study the problem of resilient strategies in the presence of uncertainty. Resilient strategies enable an agent to make decisions that are robust against disturbances. In particular, we are interested in those disturbances that are able to flip a decision made by the agent. Such a disturbance may, for instance, occur when the intended action of the agent cannot be executed due to a malfunction of an actuator in the environment. In this work, we introduce the concept of resilience in the stochastic setting and present a comprehensive set of fundamental problems. Specifically, we discuss such problems for Markov decision processes with reachability and safety objectives, which also smoothly extend to stochastic games. To account for the stochastic setting, we provide various ways of aggregating the amounts of disturbances that may have occurred, for instance, in expectation or in the worst case. Moreover, to reason about infinite disturbances, we use quantitative measures, like their frequency of occurrence. |
| 2026-02-27 | [A multimodal slice discovery framework for systematic failure detection and explanation in medical image classification](http://arxiv.org/abs/2602.24183v1) | Yixuan Liu, Kanwal K. Bhatia et al. | Despite advances in machine learning-based medical image classifiers, the safety and reliability of these systems remain major concerns in practical settings. Existing auditing approaches mainly rely on unimodal features or metadata-based subgroup analyses, which are limited in interpretability and often fail to capture hidden systematic failures. To address these limitations, we introduce the first automated auditing framework that extends slice discovery methods to multimodal representations specifically for medical applications. Comprehensive experiments were conducted under common failure scenarios using the MIMIC-CXR-JPG dataset, demonstrating the framework's strong capability in both failure discovery and explanation generation. Our results also show that multimodal information generally allows more comprehensive and effective auditing of classifiers, while unimodal variants beyond image-only inputs exhibit strong potential in scenarios where resources are constrained. |
| 2026-02-27 | [Bi-level RL-Heuristic Optimization for Real-world Winter Road Maintenance](http://arxiv.org/abs/2602.24097v1) | Yue Xie, Zizhen Xu et al. | Winter road maintenance is critical for ensuring public safety and reducing environmental impacts, yet existing methods struggle to manage large-scale routing problems effectively and mostly reply on human decision. This study presents a novel, scalable bi-level optimization framework, validated on real operational data on UK strategic road networks (M25, M6, A1), including interconnected local road networks in surrounding areas for vehicle traversing, as part of the highway operator's efforts to solve existing planning challenges. At the upper level, a reinforcement learning (RL) agent strategically partitions the road network into manageable clusters and optimally allocates resources from multiple depots. At the lower level, a multi-objective vehicle routing problem (VRP) is solved within each cluster, minimizing the maximum vehicle travel time and total carbon emissions. Unlike existing approaches, our method handles large-scale, real-world networks efficiently, explicitly incorporating vehicle-specific constraints, depot capacities, and road segment requirements. Results demonstrate significant improvements, including balanced workloads, reduced maximum travel times below the targeted two-hour threshold, lower emissions, and substantial cost savings. This study illustrates how advanced AI-driven bi-level optimization can directly enhance operational decision-making in real-world transportation and logistics. |
| 2026-02-27 | [Speak Now: Safe Actor Programming with Multiparty Session Types](http://arxiv.org/abs/2602.24054v1) | Simon Fowler, Raymond Hu | Actor languages such as Erlang and Elixir are widely used for implementing scalable and reliable distributed applications, but the informally-specified nature of actor communication patterns leaves systems vulnerable to costly errors such as communication mismatches and deadlocks. Multiparty session types (MPSTs) rule out communication errors early in the development process, but until now, the many-sender, single-receiver nature of actor communication has made it difficult for actor languages to benefit from session types.   This paper introduces Maty, the first actor language design supporting both static multiparty session typing and the full power of actors taking part in multiple sessions. Maty therefore combines the error prevention mechanism of session types with the scalability and fault tolerance of actor languages. Our main insight is to enforce session typing through a flow-sensitive effect system, combined with an event-driven programming style and first-class message handlers. Using MPSTs allows us to guarantee communication safety: a process will never send or receive an unexpected message, nor will a session get stuck because an actor is waiting for a message that will never be sent. We extend Maty to support Erlang-style supervision and cascading failure, and show that this preserves Maty's strong metatheory. We implement Maty in Scala using an API generation approach, and demonstrate the expressiveness of our model by implementing a representative sample of the widely-used Savina actor benchmark suite; an industry-supplied factory scenario; and a chat server. |
| 2026-02-27 | [GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](http://arxiv.org/abs/2602.24027v1) | Xingyu Zhu, Beier Zhu et al. | Large vision-language models (LVLMs) have achieved remarkable progress in vision-language reasoning tasks, yet ensuring their safety remains a critical challenge. Recent input-side defenses detect unsafe images with CLIP and prepend safety prefixes to prompts, but they still suffer from inaccurate detection in complex scenes and unstable safety signals during decoding. To address these issues, we propose GuardAlign, a training-free defense framework that integrates two strategies. First, OT-enhanced safety detection leverages optimal transport to measure distribution distances between image patches and unsafe semantics, enabling accurate identification of malicious regions without additional computational cost. Second, cross-modal attentive calibration strengthens the influence of safety prefixes by adaptively reallocating attention across layers, ensuring that safety signals remain consistently activated throughout generation. Extensive evaluations on six representative MLLMs demonstrate that GuardAlign reduces unsafe response rates by up to 39% on SPA-VL, while preserving utility, achieving an improvement on VQAv2 from 78.51% to 79.21%. |
| 2026-02-27 | [Jailbreak Foundry: From Papers to Runnable Attacks for Reproducible Benchmarking](http://arxiv.org/abs/2602.24009v1) | Zhicheng Fang, Jingjie Zheng et al. | Jailbreak techniques for large language models (LLMs) evolve faster than benchmarks, making robustness estimates stale and difficult to compare across papers due to drift in datasets, harnesses, and judging protocols. We introduce JAILBREAK FOUNDRY (JBF), a system that addresses this gap via a multi-agent workflow to translate jailbreak papers into executable modules for immediate evaluation within a unified harness. JBF features three core components: (i) JBF-LIB for shared contracts and reusable utilities; (ii) JBF-FORGE for the multi-agent paper-to-module translation; and (iii) JBF-EVAL for standardizing evaluations. Across 30 reproduced attacks, JBF achieves high fidelity with a mean (reproduced-reported) attack success rate (ASR) deviation of +0.26 percentage points. By leveraging shared infrastructure, JBF reduces attack-specific implementation code by nearly half relative to original repositories and achieves an 82.5% mean reused-code ratio. This system enables a standardized AdvBench evaluation of all 30 attacks across 10 victim models using a consistent GPT-4o judge. By automating both attack integration and standardized evaluation, JBF offers a scalable solution for creating living benchmarks that keep pace with the rapidly shifting security landscape. |
| 2026-02-27 | [TSC: Topology-Conditioned Stackelberg Coordination for Multi-Agent Reinforcement Learning in Interactive Driving](http://arxiv.org/abs/2602.23896v1) | Xiaotong Zhang, Gang Xiong et al. | Safe and efficient autonomous driving in dense traffic is fundamentally a decentralized multi-agent coordination problem, where interactions at conflict points such as merging and weaving must be resolved reliably under partial observability. With only local and incomplete cues, interaction patterns can change rapidly, often causing unstable behaviors such as oscillatory yielding or unsafe commitments. Existing multi-agent reinforcement learning (MARL) approaches either adopt synchronous decision-making, which exacerbate non-stationarity, or depend on centralized sequencing mechanisms that scale poorly as traffic density increases. To address these limitations, we propose Topology-conditioned Stackelberg Coordination (TSC), a learning framework for decentralized interactive driving under communication-free execution, which extracts a time-varying directed priority graph from braid-inspired weaving relations between trajectories, thereby defining local leader-follower dependencies without constructing a global order of play. Conditioned on this graph, TSC endogenously factorizes dense interactions into graph-local Stackelberg subgames and, under centralized training and decentralized execution (CTDE), learns a sequential coordination policy that anticipates leaders via action prediction and trains followers through action-conditioned value learning to approximate local best responses, improving training stability and safety in dense traffic. Experiments across four dense traffic scenarios show that TSC achieves superior performance over representative MARL baselines across key metrics, most notably reducing collisions while maintaining competitive traffic efficiency and control smoothness. |
| 2026-02-27 | [APPO: Attention-guided Perception Policy Optimization for Video Reasoning](http://arxiv.org/abs/2602.23823v1) | Henghui Du, Chang Zhou et al. | Complex video reasoning, actually, relies excessively on fine-grained perception rather than on expert (e.g., Ph.D, Science)-level reasoning. Through extensive empirical observation, we have recognized the critical impact of perception. In particular, when perception ability is almost fixed, enhancing reasoning from Qwen3-8B to OpenAI-o3 yields only 0.7% performance improvement. Conversely, even minimal change in perception model scale (from 7B to 32B) boosts performance by 1.4%, indicating enhancing perception, rather than reasoning, is more critical to improve performance. Therefore, exploring how to enhance perception ability through reasoning without the need for expensive fine-grained annotation information is worthwhile. To achieve this goal, we specially propose APPO, the Attention-guided Perception Policy Optimization algorithm that leverages token-level dense rewards to improve model's fine-grained perception. The core idea behind APPO is to optimize those tokens from different responses that primarily focus on the same crucial video frame (called intra-group perception tokens). Experimental results on diverse video benchmarks and models with different scales (3/7B) demonstrate APPO consistently outperforms GRPO and DAPO (0.5%~4%). We hope our work provides a promising approach to effectively enhance model's perception abilities through reasoning in a low-cost manner, serving diverse scenarios and demands. |
| 2026-02-27 | [Learning to maintain safety through expert demonstrations in settings with unknown constraints: A Q-learning perspective](http://arxiv.org/abs/2602.23816v1) | George Papadopoulos, George A. Vouros | Given a set of trajectories demonstrating the execution of a task safely in a constrained MDP with observable rewards but with unknown constraints and non-observable costs, we aim to find a policy that maximizes the likelihood of demonstrated trajectories trading the balance between being conservative and increasing significantly the likelihood of high-rewarding trajectories but with potentially unsafe steps. Having these objectives, we aim towards learning a policy that maximizes the probability of the most $promising$ trajectories with respect to the demonstrations. In so doing, we formulate the ``promise" of individual state-action pairs in terms of $Q$ values, which depend on task-specific rewards as well as on the assessment of states' safety, mixing expectations in terms of rewards and safety. This entails a safe Q-learning perspective of the inverse learning problem under constraints: The devised Safe $Q$ Inverse Constrained Reinforcement Learning (SafeQIL) algorithm is compared to state-of-the art inverse constraint reinforcement learning algorithms to a set of challenging benchmark tasks, showing its merits. |
| 2026-02-27 | [The Auton Agentic AI Framework](http://arxiv.org/abs/2602.23720v1) | Sheng Cao, Zhao Chang et al. | The field of Artificial Intelligence is undergoing a transition from Generative AI -- probabilistic generation of text and images -- to Agentic AI, in which autonomous systems execute actions within external environments on behalf of users. This transition exposes a fundamental architectural mismatch: Large Language Models (LLMs) produce stochastic, unstructured outputs, whereas the backend infrastructure they must control -- databases, APIs, cloud services -- requires deterministic, schema-conformant inputs. The present paper describes the Auton Agentic AI Framework, a principled architecture for standardizing the creation, execution, and governance of autonomous agent systems. The framework is organized around a strict separation between the Cognitive Blueprint, a declarative, language-agnostic specification of agent identity and capabilities, and the Runtime Engine, the platform-specific execution substrate that instantiates and runs the agent. This separation enables cross-language portability, formal auditability, and modular tool integration via the Model Context Protocol (MCP). The paper formalizes the agent execution model as an augmented Partially Observable Markov Decision Process (POMDP) with a latent reasoning space, introduces a hierarchical memory consolidation architecture inspired by biological episodic memory systems, defines a constraint manifold formalism for safety enforcement via policy projection rather than post-hoc filtering, presents a three-level self-evolution framework spanning in-context adaptation through reinforcement learning, and describes runtime optimizations -- including parallel graph execution, speculative inference, and dynamic context pruning -- that reduce end-to-end latency for multi-step agent workflows. |
| 2026-02-27 | [SAGE-LLM: Towards Safe and Generalizable LLM Controller with Fuzzy-CBF Verification and Graph-Structured Knowledge Retrieval for UAV Decision](http://arxiv.org/abs/2602.23719v1) | Wenzhe Zhao, Yang Zhao et al. | In UAV dynamic decision, complex and variable hazardous factors pose severe challenges to the generalization capability of algorithms. Despite offering semantic understanding and scene generalization, Large Language Models (LLM) lack domain-specific UAV control knowledge and formal safety assurances, restricting their direct applicability. To bridge this gap, this paper proposes a train-free two-layer decision architecture based on LLMs, integrating high-level safety planning with low-level precise control. The framework introduces three key contributions: 1) A fuzzy Control Barrier Function verification mechanism for semantically-augmented actions, providing provable safety certification for LLM outputs. 2) A star-hierarchical graph-based retrieval-augmented generation system, enabling efficient, elastic, and interpretable scene adaptation. 3) Systematic experimental validation in pursuit-evasion scenarios with unknown obstacles and emergent threats, demonstrating that our SAGE-LLM maintains performance while significantly enhancing safety and generalization without online training. The proposed framework demonstrates strong extensibility, suggesting its potential for generalization to broader embodied intelligence systems and safety-critical control domains. |
| 2026-02-27 | [Interpretable Multimodal Gesture Recognition for Drone and Mobile Robot Teleoperation via Log-Likelihood Ratio Fusion](http://arxiv.org/abs/2602.23694v1) | Seungyeol Baek, Jaspreet Singh et al. | Human operators are still frequently exposed to hazardous environments such as disaster zones and industrial facilities, where intuitive and reliable teleoperation of mobile robots and Unmanned Aerial Vehicles (UAVs) is essential. In this context, hands-free teleoperation enhances operator mobility and situational awareness, thereby improving safety in hazardous environments. While vision-based gesture recognition has been explored as one method for hands-free teleoperation, its performance often deteriorates under occlusions, lighting variations, and cluttered backgrounds, limiting its applicability in real-world operations. To overcome these limitations, we propose a multimodal gesture recognition framework that integrates inertial data (accelerometer, gyroscope, and orientation) from Apple Watches on both wrists with capacitive sensing signals from custom gloves. We design a late fusion strategy based on the log-likelihood ratio (LLR), which not only enhances recognition performance but also provides interpretability by quantifying modality-specific contributions. To support this research, we introduce a new dataset of 20 distinct gestures inspired by aircraft marshalling signals, comprising synchronized RGB video, IMU, and capacitive sensor data. Experimental results demonstrate that our framework achieves performance comparable to a state-of-the-art vision-based baseline while significantly reducing computational cost, model size, and training time, making it well suited for real-time robot control. We therefore underscore the potential of sensor-based multimodal fusion as a robust and interpretable solution for gesture-driven mobile robot and drone teleoperation. |
| 2026-02-27 | [FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation](http://arxiv.org/abs/2602.23636v1) | Zhihao Ding, Jinming Li et al. | Ensuring the safety of LLM-generated content is essential for real-world deployment. Most existing guardrail models formulate moderation as a fixed binary classification task, implicitly assuming a fixed definition of harmfulness. In practice, enforcement strictness - how conservatively harmfulness is defined and enforced - varies across platforms and evolves over time, making binary moderators brittle under shifting requirements. We first introduce FlexBench, a strictness-adaptive LLM moderation benchmark that enables controlled evaluation under multiple strictness regimes. Experiments on FlexBench reveal substantial cross-strictness inconsistency in existing moderators: models that perform well under one regime can degrade substantially under others, limiting their practical usability. To address this, we propose FlexGuard, an LLM-based moderator that outputs a calibrated continuous risk score reflecting risk severity and supports strictness-specific decisions via thresholding. We train FlexGuard via risk-alignment optimization to improve score-severity consistency and provide practical threshold selection strategies to adapt to target strictness at deployment. Experiments on FlexBench and public benchmarks demonstrate that FlexGuard achieves higher moderation accuracy and substantially improved robustness under varying strictness. We release the source code and data to support reproducibility. |
| 2026-02-27 | [Evidential Neural Radiance Fields](http://arxiv.org/abs/2602.23574v1) | Ruxiao Duan, Alex Wong | Understanding sources of uncertainty is fundamental to trustworthy three-dimensional scene modeling. While recent advances in neural radiance fields (NeRFs) achieve impressive accuracy in scene reconstruction and novel view synthesis, the lack of uncertainty estimation significantly limits their deployment in safety-critical settings. Existing uncertainty quantification methods for NeRFs fail to capture both aleatoric and epistemic uncertainty. Among those that do quantify one or the other, many of them either compromise rendering quality or incur significant computational overhead to obtain uncertainty estimates. To address these issues, we introduce Evidential Neural Radiance Fields, a probabilistic approach that seamlessly integrates with the NeRF rendering process and enables direct quantification of both aleatoric and epistemic uncertainty from a single forward pass. We compare multiple uncertainty quantification methods on three standardized benchmarks, where our approach demonstrates state-of-the-art scene reconstruction fidelity and uncertainty estimation quality. |
| 2026-02-26 | [V-MORALS: Visual Morse Graph-Aided Estimation of Regions of Attraction in a Learned Latent Space](http://arxiv.org/abs/2602.23524v1) | Faiz Aladin, Ashwin Balasubramanian et al. | Reachability analysis has become increasingly important in robotics to distinguish safe from unsafe states. Unfortunately, existing reachability and safety analysis methods often fall short, as they typically require known system dynamics or large datasets to estimate accurate system models, are computationally expensive, and assume full state information. A recent method, called MORALS, aims to address these shortcomings by using topological tools to estimate3DR-eEgnciodnesr of Attraction (ROA) in a low-dimensional latent space. However, MORALS still relies on full state knowledge and has not been studied when only sensor measurements are available. This paper presents Visual Morse Graph-Aided Estimation of Regions of Attraction in a Learned Latent Space (V- MORALS). V-MORALS takes in a dataset of image-based trajectories of a system under a given controller, and learns a latent space for reachability analysis. Using this learned latent space, our method is able to generate well-defined Morse Graphs, from which we can compute ROAs for various systems and controllers. V-MORALS provides capabilities similar to the original MORALS architecture without relying on state knowledge, and using only high-level sensor data. Our project website is at: https://v-morals.onrender.com. |
| 2026-02-26 | [Too Immersive for the Field? Addressing Safety Risks in Extended Reality User Studies](http://arxiv.org/abs/2602.23497v1) | Tanja Kojić, Sara Srebot et al. | Extended Reality (XR) technologies are increasingly tested outside the lab, in homes, schools, and public spaces. While this shift enables more realistic user insights, it also introduces safety challenges that are often overlooked. Physical risks, psychological distress, and accessibility issues can be increased in field studies and unsupervised testing, such as at home or crowdsourced trials. Without clear instructions, safety decisions are left to individual researchers, raising questions of responsibility and consistency. This position paper outlines key safety risks in XR user testing beyond the lab and calls for practical strategies that are needed to help researchers run XR studies in a safe and inclusive way across different environments. |
| 2026-02-26 | [Refining Almost-Safe Value Functions on the Fly](http://arxiv.org/abs/2602.23478v1) | Sander Tonkens, Sosuke Kojima et al. | Control Barrier Functions (CBFs) are a powerful tool for ensuring robotic safety, but designing or learning valid CBFs for complex systems is a significant challenge. While Hamilton-Jacobi Reachability provides a formal method for synthesizing safe value functions, it scales poorly and is typically performed offline, limiting its applicability in dynamic environments. This paper bridges the gap between offline synthesis and online adaptation. We introduce refineCBF for refining an approximate CBF - whether analytically derived, learned, or even unsafe - via warm-started HJ reachability. We then present its computationally efficient successor, HJ-Patch, which accelerates this process through localized updates. Both methods guarantee the recovery of a safe value function and can ensure monotonic safety improvements during adaptation. Our experiments validate our framework's primary contribution: in-the-loop, real-time adaptation, in simulation (with detailed value function analysis) and on physical hardware. Our experiments on ground vehicles and quadcopters show that our framework can successfully adapt to sudden environmental changes, such as new obstacles and unmodeled wind disturbances, providing a practical path toward deploying formally guaranteed safety in real-world settings. |
| 2026-02-26 | [CACTUSDB: Unlock Co-Optimization Opportunities for SQL and AI/ML Inferences](http://arxiv.org/abs/2602.23469v1) | Lixi Zhou, Kanchan Chowdhury et al. | There is a growing demand for supporting inference queries that combine Structured Query Language (SQL) and Artificial Intelligence / Machine Learning (AI/ML) model inferences in database systems, to avoid data denormalization and transfer, facilitate management, and alleviate privacy concerns. Co-optimization techniques for executing inference queries in database systems without accuracy loss fall into four categories: (O1) Relational algebra optimization treating AI/ML models as black-box user-defined functions (UDFs); (O2) Factorized AI/ML inferences; (O3) Tensor-relational transformation; and (O4) General cross-optimization techniques. However, we found none of the existing database systems support all these techniques simultaneously, resulting in suboptimal performance. In this work, we identify two key challenges to address the above problem: (1) the difficulty of unifying all co-optimization techniques that involve disparate data and computation abstractions in one system; and (2) the lack of an optimizer that can effectively explore the exponential search space. To address these challenges, we present CactusDB, a novel system built atop Velox - a high-performance, UDF-centric database engine, open-sourced by Meta. CactusDB features a three-level Intermediate Representations (IR) that supports relational operators, expression operators, and ML functions to enable flexible optimization of arbitrary sub-computations. Additionally, we propose a novel Monte-Carlo Tree Search (MCTS)-based optimizer with query embedding, co-designed with our unique three-level IR, enabling shared and reusable optimization knowledge across different queries. Evaluation of 12 representative inference workloads and 2,000 randomly generated inference queries on well-known datasets, such as MovieLens and TPCx-AI, shows that CactusDB achieves up to 441 times speedup compared to alternative systems. |
| 2026-02-26 | [Safety First: Psychological Safety as the Key to AI Transformation](http://arxiv.org/abs/2602.23279v1) | Aaron Reich, Diana Wolfe et al. | Organizations continue to invest in artificial intelligence, yet many struggle to ensure that employees adopt and engage with these tools. Drawing on research highlighting the interpersonal and learning demands of technology use, this study examines whether psychological safety is associated with AI adoption and usage in the workplace. Using survey data from 2,257 employees in a global consulting firm, we test whether psychological safety is associated with adoption, usage frequency, and usage duration; and whether these relationships vary by organizational level, professional experience, or geographic region. Logistic and linear regression analyses show that psychological safety reliably predicts whether employees adopt AI tools but does not predict how often or how long they use AI once adoption has occurred. Moreover, the relationship between psychological safety and AI adoption is consistent across experience levels, role levels, and regions, and no moderation effects emerge. These findings suggest that psychological safety functions as a key antecedent of initial AI engagement but not of subsequent usage intensity. The study underscores the need to distinguish between adoption and sustained use and highlights opportunities for targeted organizational interventions in early-stage AI implementation. |

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



