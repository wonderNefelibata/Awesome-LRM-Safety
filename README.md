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
| 2026-02-20 | [AI-Wrapped: Participatory, Privacy-Preserving Measurement of Longitudinal LLM Use In-the-Wild](http://arxiv.org/abs/2602.18415v1) | Cathy Mengying Fang, Sheer Karny et al. | Alignment research on large language models (LLMs) increasingly depends on understanding how these systems are used in everyday contexts. yet naturalistic interaction data is difficult to access due to privacy constraints and platform control. We present AI-Wrapped, a prototype workflow for collecting naturalistic LLM usage data while providing participants with an immediate ``wrapped''-style report on their usage statistics, top topics, and safety-relevant behavioral patterns. We report findings from an initial deployment with 82 U.S.-based adults across 48,495 conversations from their 2025 histories. Participants used LLMs for both instrumental and reflective purposes, including creative work, professional tasks, and emotional or existential themes. Some usage patterns were consistent with potential over-reliance or perfectionistic refinement, while heavier users showed comparatively more reflective exchanges than primarily transactional ones. Methodologically, even with zero data retention and PII removal, participants may remain hesitant to share chat data due to perceived privacy and judgment risks, underscoring the importance of trust, agency, and transparent design when building measurement infrastructure for alignment research. |
| 2026-02-20 | [Modeling UAV-aided Roadside Cell-Free Networks with Matérn Hard-Core Point Processes](http://arxiv.org/abs/2602.18408v1) | Chenrui Qiu, Yongxu Zhu et al. | This paper investigates a uncrewed aerial vehicles (UAV)-assisted cell-free architecture for vehicular networks in road-constrained environments. Roads are modeled using a Poisson Line Process (PLP), with multi-layer roadside access points (APs) deployed via 1-D Poisson Point Process (PPP). Each user forms a localized cell-free cluster by associating with the nearest AP in each layer along its corresponding road. This forms a road-constrained cell-free architecture. To enhance coverage, UAV act as an aerial tier, extending access from 1-D road-constrained layouts (embedded in 2-D) to 3-D. We employ a Matérn Hard-Core (MHC) point process to model the spatial distribution of UAV base stations, ensuring a minimum safety distance between them. In order to enable tractable analysis of the aggregate signal from multiple APs, a distance-based power control scheme is introduced. Leveraging tools from stochastic geometry, we have studied the coverage probability. Furthermore, we analyze the impact of key system parameters on coverage performance, providing useful insights into the deployment and optimization of UAV-assisted cell-free vehicular networks. |
| 2026-02-20 | [Self-Aware Object Detection via Degradation Manifolds](http://arxiv.org/abs/2602.18394v1) | Stefan Becker, Simon Weiss et al. | Object detectors achieve strong performance under nominal imaging conditions but can fail silently when exposed to blur, noise, compression, adverse weather, or resolution changes. In safety-critical settings, it is therefore insufficient to produce predictions without assessing whether the input remains within the detector's nominal operating regime. We refer to this capability as self-aware object detection.   We introduce a degradation-aware self-awareness framework based on degradation manifolds, which explicitly structure a detector's feature space according to image degradation rather than semantic content. Our method augments a standard detection backbone with a lightweight embedding head trained via multi-layer contrastive learning. Images sharing the same degradation composition are pulled together, while differing degradation configurations are pushed apart, yielding a geometrically organized representation that captures degradation type and severity without requiring degradation labels or explicit density modeling.   To anchor the learned geometry, we estimate a pristine prototype from clean training embeddings, defining a nominal operating point in representation space. Self-awareness emerges as geometric deviation from this reference, providing an intrinsic, image-level signal of degradation-induced shift that is independent of detection confidence.   Extensive experiments on synthetic corruption benchmarks, cross-dataset zero-shot transfer, and natural weather-induced distribution shifts demonstrate strong pristine-degraded separability, consistent behavior across multiple detector architectures, and robust generalization under semantic shift. These results suggest that degradation-aware representation geometry provides a practical and detector-agnostic foundation. |
| 2026-02-20 | [Role-Adaptive Collaborative Formation Planning for Team of Quadruped Robots in Cluttered Environments](http://arxiv.org/abs/2602.18260v1) | Magnus Norén, Marios-Nektarios Stamatopoulos et al. | This paper presents a role-adaptive Leader-Follower-based formation planning and control framework for teams of quadruped robots operating in cluttered environments. Unlike conventional methods with fixed leaders or rigid formation roles, the proposed approach integrates dynamic role assignment and partial goal planning, enabling flexible, collision-free navigation in complex scenarios. Formation stability and inter-robot safety are ensured through a virtual spring-damper system coupled with a novel obstacle avoidance layer that adaptively adjusts each agent's velocity. A dynamic look-ahead reference generator further enhances flexibility, allowing temporary formation deformation to maneuver around obstacles while maintaining goal-directed motion. The Fast Marching Square (FM2) algorithm provides the global path for the leader and local paths for the followers as the planning backbone. The framework is validated through extensive simulations and real-world experiments with teams of quadruped robots. Results demonstrate smooth coordination, adaptive role switching, and robust formation maintenance in complex, unstructured environments. A video featuring the simulation and physical experiments along with their associated visualizations can be found at https://youtu.be/scq37Tua9W4. |
| 2026-02-20 | [BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards](http://arxiv.org/abs/2602.18193v1) | Yiran Yang, Zhaowei Liu et al. | Short-video platforms now host vast multimodal ads whose deceptive visuals, speech and subtitles demand finer-grained, policy-driven moderation than community safety filters. We present BLM-Guard, a content-audit framework for commercial ads that fuses Chain-of-Thought reasoning with rule-based policy principles and a critic-guided reward. A rule-driven ICoT data-synthesis pipeline jump-starts training by generating structured scene descriptions, reasoning chains and labels, cutting annotation costs. Reinforcement learning then refines the model using a composite reward balancing causal coherence with policy adherence. A multitask architecture models intra-modal manipulations (e.g., exaggerated imagery) and cross-modal mismatches (e.g., subtitle-speech drift), boosting robustness. Experiments on real short-video ads show BLM-Guard surpasses strong baselines in accuracy, consistency and generalization. |
| 2026-02-20 | [Capabilities Ain't All You Need: Measuring Propensities in AI](http://arxiv.org/abs/2602.18182v1) | Daniel Romero-Alvarado, Fernando Martínez-Plumed et al. | AI evaluation has primarily focused on measuring capabilities, with formal approaches inspired from Item Response Theory (IRT) being increasingly applied. Yet propensities - the tendencies of models to exhibit particular behaviours - play a central role in determining both performance and safety outcomes. However, traditional IRT describes a model's success on a task as a monotonic function of model capabilities and task demands, an approach unsuited to propensities, where both excess and deficiency can be problematic. Here, we introduce the first formal framework for measuring AI propensities by using a bilogistic formulation for model success, which attributes high success probability when the model's propensity is within an "ideal band". Further, we estimate the limits of the ideal band using LLMs equipped with newly developed task-agnostic rubrics. Applying our framework to six families of LLM models whose propensities are incited in either direction, we find that we can measure how much the propensity is shifted and what effect this has on the tasks. Critically, propensities estimated using one benchmark successfully predict behaviour on held-out tasks. Moreover, we obtain stronger predictive power when combining propensities and capabilities than either separately. More broadly, our framework showcases how rigorous propensity measurements can be conducted and how it yields gains over solely using capability evaluations to predict AI behaviour. |
| 2026-02-20 | [FENCE: A Financial and Multimodal Jailbreak Detection Dataset](http://arxiv.org/abs/2602.18154v1) | Mirae Kim, Seonghun Jeong et al. | Jailbreaking poses a significant risk to the deployment of Large Language Models (LLMs) and Vision Language Models (VLMs). VLMs are particularly vulnerable because they process both text and images, creating broader attack surfaces. However, available resources for jailbreak detection are scarce, particularly in finance. To address this gap, we present FENCE, a bilingual (Korean-English) multimodal dataset for training and evaluating jailbreak detectors in financial applications. FENCE emphasizes domain realism through finance-relevant queries paired with image-grounded threats. Experiments with commercial and open-source VLMs reveal consistent vulnerabilities, with GPT-4o showing measurable attack success rates and open-source models displaying greater exposure. A baseline detector trained on FENCE achieves 99 percent in-distribution accuracy and maintains strong performance on external benchmarks, underscoring the dataset's robustness for training reliable detection models. FENCE provides a focused resource for advancing multimodal jailbreak detection in finance and for supporting safer, more reliable AI systems in sensitive domains. Warning: This paper includes example data that may be offensive. |
| 2026-02-20 | [History-Constrained Systems](http://arxiv.org/abs/2602.18143v1) | Louwe B. Kuijer, David Purser et al. | We study verification problems for history-constrained systems (HCS), a model of guarded computation that uses nested systems. An outer system describes the process architecture in which a sequence of actions represents the communication between sub-systems through a global bus. Actions are either permitted or blocked locally by guards; these guards read and decide based on the sequence of actions so far in the global bus.   When HCS have both the outer systems and the local guard controllers modelled by finite automata, we show they have the same expressive power as regular languages and finite automata, but they are exponentially more succinct. We also analyse games on this model, representing the interaction between environment and controller, and show that solving such games is EXPTIME-complete, where the lower bound already holds for reachability/safety games and the upper bound holds for any $ω$-regular winning condition. Finally, we consider HCS with guards of greater expressive power, Vector Addition Systems with States (VASS). We show that with deterministic coverability-VASS guards the reachability problem is EXPSPACE-complete, while with reachability-VASS the problem is undecidable. |
| 2026-02-20 | [Toward Automated Virtual Electronic Control Unit (ECU) Twins for Shift-Left Automotive Software Testing](http://arxiv.org/abs/2602.18142v1) | Sebastian Dingler, Frederik Boenke | Automotive software increasingly outpaces hardware availability, forcing late integration and expensive hardware-in-the-loop (HiL) bottlenecks. The InnoRegioChallenge project investigated whether a virtual test and integration environment can reproduce electronic control unit (ECU) behavior early enough to run real software binaries before physical hardware exists. We report a prototype that generates instruction-accurate processor models in SystemC/TLM~2.0 using an agentic, feedback-driven workflow coupled to a reference simulator via the GNU Debugger (GDB). The results indicate that the most critical technical risk -- CPU behavioral fidelity -- can be reduced through automated differential testing and iterative model correction. We summarize the architecture, the agentic modeling loop, and project outcomes, and we extrapolate plausible technical details consistent with the reported qualitative findings. While cloud-scale deployment and full toolchain integration remain future work, the prototype demonstrates a viable shift-left path for virtual ECU twins, enabling reproducible tests, non-intrusive tracing, and fault-injection campaigns aligned with safety standards. |
| 2026-02-20 | [Interacting safely with cyclists using Hamilton-Jacobi reachability and reinforcement learning](http://arxiv.org/abs/2602.18097v1) | Aarati Andrea Noronha, Jean Oh | In this paper, we present a framework for enabling autonomous vehicles to interact with cyclists in a manner that balances safety and optimality. The approach integrates Hamilton-Jacobi reachability analysis with deep Q-learning to jointly address safety guarantees and time-efficient navigation. A value function is computed as the solution to a time-dependent Hamilton-Jacobi-Bellman inequality, providing a quantitative measure of safety for each system state. This safety metric is incorporated as a structured reward signal within a reinforcement learning framework. The method further models the cyclist's latent response to the vehicle, allowing disturbance inputs to reflect human comfort and behavioral adaptation. The proposed framework is evaluated through simulation and comparison with human driving behavior and an existing state-of-the-art method. |
| 2026-02-20 | [OODBench: Out-of-Distribution Benchmark for Large Vision-Language Models](http://arxiv.org/abs/2602.18094v1) | Ling Lin, Yang Bai et al. | Existing Visual-Language Models (VLMs) have achieved significant progress by being trained on massive-scale datasets, typically under the assumption that data are independent and identically distributed (IID). However, in real-world scenarios, it is often impractical to expect that all data processed by an AI system satisfy this assumption. Furthermore, failure to appropriately handle out-of-distribution (OOD) objects may introduce safety risks in real-world applications (e.g., autonomous driving or medical assistance). Unfortunately, current research has not yet provided valid benchmarks that can comprehensively assess the performance of VLMs in response to OOD data. Therefore, we propose OODBench, a predominantly automated method with minimal human verification, for constructing new benchmarks and evaluating the ability of VLMs to process OOD data. OODBench contains 40K instance-level OOD instance-category pairs, and we show that current VLMs still exhibit notable performance degradation on OODBench, even when the underlying image categories are common. In addition, we propose a reliable automated assessment metric that employs a Basic-to-Advanced Progression of prompted questions to assess the impact of OOD data on questions of varying difficulty more fully. Lastly, we summarize substantial findings and insights to facilitate future research in the acquisition and evaluation of OOD data. |
| 2026-02-20 | [Dynamic Deception: When Pedestrians Team Up to Fool Autonomous Cars](http://arxiv.org/abs/2602.18079v1) | Masoud Jamshidiyan Tehrani, Marco Gabriel et al. | Many adversarial attacks on autonomous-driving perception models fail to cause system-level failures once deployed in a full driving stack. The main reason for such ineffectiveness is that once deployed in a system (e.g., within a simulator), attacks tend to be spatially or temporally short-lived, due to the vehicle's dynamics, hence rarely influencing the vehicle behaviour. In this paper, we address both limitations by introducing a system-level attack in which multiple dynamic elements (e.g., two pedestrians) carry adversarial patches (e.g., on cloths) and jointly amplify their effect through coordination and motion. We evaluate our attacks in the CARLA simulator using a state-of-the-art autonomous driving agent. At the system level, single-pedestrian attacks fail in all runs (out of 10), while dynamic collusion by two pedestrians induces full vehicle stops in up to 50\% of runs, with static collusion yielding no successful attack at all. These results show that system-level failures arise only when adversarial signals persist over time and are amplified through coordinated actors, exposing a gap between model-level robustness and end-to-end safety. |
| 2026-02-20 | [Hybrid Non-informative and Informative Prior Model-assisted Designs for Mid-trial Dose Insertion](http://arxiv.org/abs/2602.17995v1) | Kana Yamada, Hisato Sunami et al. | In oncology phase I trials, model-assisted designs have been increasingly adopted because they enable adaptive yet operationally simple dose adjustment based on accumulating safety data, leading to a paradigm shift in dose-escalation methodology. In practice, a single mid-trial dose insertion may be considered to examine safer doses and/or to collect more informative efficacy data. In this study, we investigate methods to improve dose assignment and the selection of the maximum tolerated dose (MTD) or the optimal biological dose (OBD) when a new dose level is added during an ongoing trial under a model-assisted framework, by assigning informative prior information to the inserted dose. We propose a hybrid design that uses a non-informative model-assisted design at trial initiation and, upon dose insertion, applies an informative-prior extension only to the newly added dose. In addition, to address potential skeleton misspecification, we propose two adaptive extensions: (i) an online-weighting approach that updates the skeleton over time, and (ii) a Bayesian-mixture approach that robustly combines multiple candidate skeletons. We evaluate the proposed methods through simulation studies. |
| 2026-02-20 | [Mining Type Constructs Using Patterns in AI-Generated Code](http://arxiv.org/abs/2602.17955v1) | Imgyeong Lee, Tayyib Ul Hassan et al. | Artificial Intelligence (AI) increasingly automates various parts of the software development tasks. Although AI has enhanced the productivity of development tasks, it remains unstudied whether AI essentially outperforms humans in type-related programming tasks, such as employing type constructs properly for type safety, during its tasks. Moreover, there is no systematic study that evaluates whether AI agents overuse or misuse the type constructs under the complicated type systems to the same extent as humans. In this study, we present the first empirical analysis to answer these questions in the domain of TypeScript projects. Our findings show that, in contrast to humans, AI agents are 9x more prone to use the 'any' keyword. In addition, we observed that AI agents use advanced type constructs, including those that ignore type checks, more often compared to humans. Surprisingly, even with all these issues, Agentic pull requests (PRs) have 1.8x higher acceptance rates compared to humans for TypeScript. We encourage software developers to carefully confirm the type safety of their codebases whenever they coordinate with AI agents in the development process. |
| 2026-02-20 | [Operational Agency: A Permeable Legal Fiction for Tracing Culpability in AI Systems](http://arxiv.org/abs/2602.17932v1) | Anirban Mukherjee, Hannah Hanwen Chang | Modern artificial intelligence (AI) systems act with a high degree of independence yet lack legal personhood-a paradox that fractures doctrines grounded in human-centric notions of mens rea and actus reus. This Article introduces Operational Agency (OA)-a permeable legal fiction structured as an ex post evidentiary framework-and Operational Agency Graph (OAG), a tool for mapping causal interactions among human actors, organizations, and AI systems. OA evaluates an AI's observable operational characteristics: its goal-directedness (as a proxy for intent), predictive processing (as a proxy for foresight), and safety architecture (as a proxy for a standard of care). OAG operationalizes that analysis by embedding these characteristics in a causal graph to trace and apportion culpability among developers, fine-tuners, deployers, and users. Drawing on corporate criminal liability, the innocent-agent doctrine, and secondary and vicarious liability frameworks, the Article shows how OA and OAG strengthen existing doctrines. Across five real-world case studies spanning tort, civil rights, constitutional law, and antitrust, it demonstrates how the framework addresses challenges ranging from autonomous vehicle collisions to algorithmic price-fixing, offering courts a principled evidentiary method-and legislatures and industry a conceptual foundation-to ensure human accountability keeps pace with technological autonomy, without conferring personhood on AI. |
| 2026-02-19 | [El Agente Gráfico: Structured Execution Graphs for Scientific Agents](http://arxiv.org/abs/2602.17902v1) | Jiaru Bai, Abdulrahman Aldossary et al. | Large language models (LLMs) are increasingly used to automate scientific workflows, yet their integration with heterogeneous computational tools remains ad hoc and fragile. Current agentic approaches often rely on unstructured text to manage context and coordinate execution, generating often overwhelming volumes of information that may obscure decision provenance and hinder auditability. In this work, we present El Agente Gráfico, a single-agent framework that embeds LLM-driven decision-making within a type-safe execution environment and dynamic knowledge graphs for external persistence. Central to our approach is a structured abstraction of scientific concepts and an object-graph mapper that represents computational state as typed Python objects, stored either in memory or persisted in an external knowledge graph. This design enables context management through typed symbolic identifiers rather than raw text, thereby ensuring consistency, supporting provenance tracking, and enabling efficient tool orchestration. We evaluate the system by developing an automated benchmarking framework across a suite of university-level quantum chemistry tasks previously evaluated on a multi-agent system, demonstrating that a single agent, when coupled to a reliable execution engine, can robustly perform complex, multi-step, and parallel computations. We further extend this paradigm to two other large classes of applications: conformer ensemble generation and metal-organic framework design, where knowledge graphs serve as both memory and reasoning substrates. Together, these results illustrate how abstraction and type safety can provide a scalable foundation for agentic scientific automation beyond prompt-centric designs. |
| 2026-02-19 | [TFL: Targeted Bit-Flip Attack on Large Language Model](http://arxiv.org/abs/2602.17837v1) | Jingkai Guo, Chaitali Chakrabarti et al. | Large language models (LLMs) are increasingly deployed in safety and security critical applications, raising concerns about their robustness to model parameter fault injection attacks. Recent studies have shown that bit-flip attacks (BFAs), which exploit computer main memory (i.e., DRAM) vulnerabilities to flip a small number of bits in model weights, can severely disrupt LLM behavior. However, existing BFA on LLM largely induce un-targeted failure or general performance degradation, offering limited control over manipulating specific or targeted outputs. In this paper, we present TFL, a novel targeted bit-flip attack framework that enables precise manipulation of LLM outputs for selected prompts while maintaining almost no or minor degradation on unrelated inputs. Within our TFL framework, we propose a novel keyword-focused attack loss to promote attacker-specified target tokens in generative outputs, together with an auxiliary utility score that balances attack effectiveness against collateral performance impact on benign data. We evaluate TFL on multiple LLMs (Qwen, DeepSeek, Llama) and benchmarks (DROP, GSM8K, and TriviaQA). The experiments show that TFL achieves successful targeted LLM output manipulations with less than 50 bit flips and significantly reduced effect on unrelated queries compared to prior BFA approaches. This demonstrates the effectiveness of TFL and positions it as a new class of stealthy and targeted LLM model attack. |
| 2026-02-19 | [Evolution of Safety Requirements in Industrial Robotics: Comparative Analysis of ISO 10218-1/2 (2011 vs. 2025) and Integration of ISO/TS 15066](http://arxiv.org/abs/2602.17822v1) | Daniel Hartmann, Kristýna Hamříková et al. | Industrial robotics has established itself as an integral component of large-scale manufacturing enterprises. Simultaneously, collaborative robotics is gaining prominence, introducing novel paradigms of human-machine interaction. These advancements have necessitated a comprehensive revision of safety standards, specifically incorporating requirements for cybersecurity and protection against unauthorized access in networked robotic systems. This article presents a comparative analysis of the ISO 10218:2011 and ISO 10218:2025 standards, examining the evolution of their structure, terminology, technical requirements, and annexes. The analysis reveals significant expansions in functional safety and cybersecurity, the introduction of new classifications for robots and collaborative applications, and the normative integration of the technical specification ISO/TS 15066. Consequently, the new edition synthesizes mechanical, functional, and digital safety requirements, establishing a comprehensive framework for the design and operation of modern robotic systems. |
| 2026-02-19 | [Collaborative Processing for Multi-Tenant Inference on Memory-Constrained Edge TPUs](http://arxiv.org/abs/2602.17808v1) | Nathan Ng, Walid A. Hanafy et al. | IoT applications are increasingly relying on on-device AI accelerators to ensure high performance, especially in limited connectivity and safety-critical scenarios. However, the limited on-chip memory of these accelerators forces inference runtimes to swap model segments between host and accelerator memory, substantially inflating latency. While collaborative processing by partitioning the model processing between CPU and accelerator resources can reduce accelerator memory pressure and latency, naive partitioning may worsen end-to-end latency by either shifting excessive computation to the CPU or failing to sufficiently curb swapping, a problem that is further amplified in multi-tenant and dynamic environments.   To address these issues, we present SwapLess, a system for adaptive, multi-tenant TPU-CPU collaborative inference for memory-constrained Edge TPUs. SwapLess utilizes an analytic queueing model that captures partition-dependent CPU/TPU service times as well as inter- and intra-model swapping overheads across different workload mixes and request rates. Using this model, SwapLess continuously adjusts both the partition point and CPU core allocation online to minimize end-to-end response time with low decision overhead. An implementation on Edge TPU-equipped platforms demonstrates that SwapLess reduces mean latency by up to 63.8% for single-tenant workloads and up to 77.4% for multi-tenant workloads relative to the default Edge TPU compiler. |
| 2026-02-19 | [The 2025 AI Agent Index: Documenting Technical and Safety Features of Deployed Agentic AI Systems](http://arxiv.org/abs/2602.17753v1) | Leon Staufer, Kevin Feng et al. | Agentic AI systems are increasingly capable of performing professional and personal tasks with limited human involvement. However, tracking these developments is difficult because the AI agent ecosystem is complex, rapidly evolving, and inconsistently documented, posing obstacles to both researchers and policymakers. To address these challenges, this paper presents the 2025 AI Agent Index. The Index documents information regarding the origins, design, capabilities, ecosystem, and safety features of 30 state-of-the-art AI agents based on publicly available information and email correspondence with developers. In addition to documenting information about individual agents, the Index illuminates broader trends in the development of agents, their capabilities, and the level of transparency of developers. Notably, we find different transparency levels among agent developers and observe that most developers share little information about safety, evaluations, and societal impacts. The 2025 AI Agent Index is available online at https://aiagentindex.mit.edu |

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



