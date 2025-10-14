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
| 2025-10-13 | [MATH-Beyond: A Benchmark for RL to Expand Beyond the Base Model](http://arxiv.org/abs/2510.11653v1) | Prasanna Mayilvahanan, Ricardo Dominguez-Olmedo et al. | With the advent of DeepSeek-R1, a new wave of reinforcement learning (RL) methods has emerged that seem to unlock stronger mathematical reasoning. However, a closer look at the open-source ecosystem reveals a critical limitation: with sufficiently many draws (e.g., $\texttt{pass@1024}$), many existing base models already solve nearly all questions on widely used math benchmarks such as MATH-500 and AIME 2024. This suggests that the RL fine-tuning methods prevalent in the LLM reasoning literature largely sharpen existing solution modes rather than discovering entirely new ones. Such sharpening stands in contrast to the broader promise of RL: to foster exploration and to acquire new skills. To move beyond this plateau, we introduce MATH-Beyond (MATH-B), a benchmark deliberately constructed to defeat common open-source models of up to 8B parameters even under large sampling budgets. Improving performance on our benchmark via RL requires methods that learn to reason in ways that go beyond base model capabilities in repeated sampling. Since the problems are drawn from subsets of DAPO-Math-17K and DeepScaleR datasets, they remain topically equivalent to standard high-school math. Validating our premise, RL fine-tuned models such as Nemotron-Research-Reasoning-Qwen-1.5B and DeepScaleR-1.5B-Preview perform poorly on MATH-B at $\texttt{pass@1024}$, showing how existing approaches fall short on tackling harder instances. We hope MATH-B will catalyze exploration-driven RL approaches that elicit deeper reasoning capabilities. We release MATH-B at https://huggingface.co/datasets/brendel-group/MATH-Beyond. |
| 2025-10-13 | [Enhancing Long Chain-of-Thought Reasoning through Multi-Path Plan Aggregation](http://arxiv.org/abs/2510.11620v1) | Siheng Xiong, Ali Payani et al. | Inference-time scaling enhances the reasoning ability of a language model (LM) by extending its chain-of-thought (CoT). However, existing approaches typically generate the entire reasoning chain in a single forward pass, which often leads to CoT derailment, i.e., the reasoning trajectory drifting off course due to compounding errors. This problem is particularly severe for smaller LMs with long CoTs due to their limited capacity. To address this, we analyze raw long CoTs and uncover a reasoning hierarchy consisting of planning and execution steps. Our analysis reveals that most reasoning errors stem from incorrect planning. Motivated by this observation, we propose Multi-Path Plan Aggregation (MPPA), a framework that augments single-pass reasoning with plan exploration and aggregation. Following a variable interval schedule based on the token position, MPPA generates multiple candidate plans and aggregates them into a refined planning step. To maintain efficiency, we adopt a minimal design in which the base LM serves as the primary policy, while a lightweight LoRA module implements the plan aggregation policy. We further observe that outcome-reward RL is inefficient for long trajectories (e.g., exceeding 4K tokens). To overcome this, we introduce online Step-DPO, a process-level preference optimization scheme that leverages Twisted Sequential Monte Carlo (TSMC) to provide scalable stepwise supervision using small LMs. This yields more efficient training, improved stability, and higher accuracy. Extensive experiments on challenging math, science, and logical reasoning benchmarks demonstrate that, with only 10% SFT data and 5% of preference pairs, our method outperforms both the DeepSeek-R1 distillation baseline and the outcome-reward RL baseline across multiple base models and tasks. |
| 2025-10-13 | [(Dis)Proving Spectre Security with Speculation-Passing Style](http://arxiv.org/abs/2510.11573v1) | Santiago Arranz-Olmos, Gilles Barthe et al. | Constant-time (CT) verification tools are commonly used for detecting potential side-channel vulnerabilities in cryptographic libraries. Recently, a new class of tools, called speculative constant-time (SCT) tools, has also been used for detecting potential Spectre vulnerabilities. In many cases, these SCT tools have emerged as liftings of CT tools. However, these liftings are seldom defined precisely and are almost never analyzed formally. The goal of this paper is to address this gap, by developing formal foundations for these liftings, and to demonstrate that these foundations can yield practical benefits.   Concretely, we introduce a program transformation, coined Speculation-Passing Style (SPS), for reducing SCT verification to CT verification. Essentially, the transformation instruments the program with a new input that corresponds to attacker-controlled predictions and modifies the program to follow them. This approach is sound and complete, in the sense that a program is SCT if and only if its SPS transform is CT. Thus, we can leverage existing CT verification tools to prove SCT; we illustrate this by combining SPS with three standard methodologies for CT verification, namely reducing it to non-interference, assertion safety and dynamic taint analysis. We realize these combinations with three existing tools, EasyCrypt, BINSEC, and ctgrind, and we evaluate them on Kocher's benchmarks for Spectre-v1. Our results focus on Spectre-v1 in the standard CT leakage model; however, we also discuss applications of our method to other variants of Spectre and other leakage models. |
| 2025-10-13 | [Bag of Tricks for Subverting Reasoning-based Safety Guardrails](http://arxiv.org/abs/2510.11570v1) | Shuo Chen, Zhen Han et al. | Recent reasoning-based safety guardrails for Large Reasoning Models (LRMs), such as deliberative alignment, have shown strong defense against jailbreak attacks. By leveraging LRMs' reasoning ability, these guardrails help the models to assess the safety of user inputs before generating final responses. The powerful reasoning ability can analyze the intention of the input query and will refuse to assist once it detects the harmful intent hidden by the jailbreak methods. Such guardrails have shown a significant boost in defense, such as the near-perfect refusal rates on the open-source gpt-oss series. Unfortunately, we find that these powerful reasoning-based guardrails can be extremely vulnerable to subtle manipulation of the input prompts, and once hijacked, can lead to even more harmful results. Specifically, we first uncover a surprisingly fragile aspect of these guardrails: simply adding a few template tokens to the input prompt can successfully bypass the seemingly powerful guardrails and lead to explicit and harmful responses. To explore further, we introduce a bag of jailbreak methods that subvert the reasoning-based guardrails. Our attacks span white-, gray-, and black-box settings and range from effortless template manipulations to fully automated optimization. Along with the potential for scalable implementation, these methods also achieve alarmingly high attack success rates (e.g., exceeding 90% across 5 different benchmarks on gpt-oss series on both local host models and online API services). Evaluations across various leading open-source LRMs confirm that these vulnerabilities are systemic, underscoring the urgent need for stronger alignment techniques for open-sourced LRMs to prevent malicious misuse. Code is open-sourced at https://chenxshuo.github.io/bag-of-tricks. |
| 2025-10-13 | [IntersectioNDE: Learning Complex Urban Traffic Dynamics based on Interaction Decoupling Strategy](http://arxiv.org/abs/2510.11534v1) | Enli Lin, Ziyuan Yang et al. | Realistic traffic simulation is critical for ensuring the safety and reliability of autonomous vehicles (AVs), especially in complex and diverse urban traffic environments. However, existing data-driven simulators face two key challenges: a limited focus on modeling dense, heterogeneous interactions at urban intersections - which are prevalent, crucial, and practically significant in countries like China, featuring diverse agents including motorized vehicles (MVs), non-motorized vehicles (NMVs), and pedestrians - and the inherent difficulty in robustly learning high-dimensional joint distributions for such high-density scenes, often leading to mode collapse and long-term simulation instability. We introduce City Crossings Dataset (CiCross), a large-scale dataset collected from a real-world urban intersection, uniquely capturing dense, heterogeneous multi-agent interactions, particularly with a substantial proportion of MVs, NMVs and pedestrians. Based on this dataset, we propose IntersectioNDE (Intersection Naturalistic Driving Environment), a data-driven simulator tailored for complex urban intersection scenarios. Its core component is the Interaction Decoupling Strategy (IDS), a training paradigm that learns compositional dynamics from agent subsets, enabling the marginal-to-joint simulation. Integrated into a scene-aware Transformer network with specialized training techniques, IDS significantly enhances simulation robustness and long-term stability for modeling heterogeneous interactions. Experiments on CiCross show that IntersectioNDE outperforms baseline methods in simulation fidelity, stability, and its ability to replicate complex, distribution-level urban traffic dynamics. |
| 2025-10-13 | [Algorithmic analysis of a complex reliability system subject to multiple events with a preventive maintenance strategy and a Bernoulli vacation policy through MMAPs](http://arxiv.org/abs/2510.11506v1) | Juan Eloy Ruiz-Castro, Hugo Alaín Zapata-Ceballos | In this work, a single-unit multi-state system is considered. The system is subject to internal failures, as well as external shocks with multiple consequences. It also incorporates a preventive maintenance strategy and a Bernoulli vacation policy for the repairperson. It is algorithmically modeled in both continuous and discrete time using Marked Markovian Arrival Processes (MMAP). The system's operation/degradation level is divided into an indeterminate number of levels. Upon returning from a vacation period, the repair technician may initiate corrective repair, perform preventive maintenance, replace the unit, remain idle at the workplace, or begin a new vacation period. The decision in the latter two cases is made probabilistically based on the system's operational level. This methodology allows the model and its associated measures to be algorithmically derived in both transient and stationary regimes, presented in a matrix-algorithmic form. Analytical-matrix methods are used to obtain the system's steady-state behaviour as well as various performance measures. Costs and rewards are introduced to analyze when the system becomes profitable. Measures associated with costs over time and in the stationary regime are defined and considered for optimization studies. A numerical example demonstrates the versatility of the model by solving a probabilistic optimization problem using a multi-objective Pareto analysis approach and performing a comparative evaluation of multiple models. Genetic algorithm is applied to find the optimization results in the reduced solution space. All modeling and associated measures have been computationally implemented in Matlab. |
| 2025-10-13 | [Context-Aware Model-Based Reinforcement Learning for Autonomous Racing](http://arxiv.org/abs/2510.11501v1) | Emran Yasser Moustafa, Ivana Dusparic | Autonomous vehicles have shown promising potential to be a groundbreaking technology for improving the safety of road users. For these vehicles, as well as many other safety-critical robotic technologies, to be deployed in real-world applications, we require algorithms that can generalize well to unseen scenarios and data. Model-based reinforcement learning algorithms (MBRL) have demonstrated state-of-the-art performance and data efficiency across a diverse set of domains. However, these algorithms have also shown susceptibility to changes in the environment and its transition dynamics.   In this work, we explore the performance and generalization capabilities of MBRL algorithms for autonomous driving, specifically in the simulated autonomous racing environment, Roboracer (formerly F1Tenth). We frame the head-to-head racing task as a learning problem using contextual Markov decision processes and parameterize the driving behavior of the adversaries using the context of the episode, thereby also parameterizing the transition and reward dynamics. We benchmark the behavior of MBRL algorithms in this environment and propose a novel context-aware extension of the existing literature, cMask. We demonstrate that context-aware MBRL algorithms generalize better to out-of-distribution adversary behaviors relative to context-free approaches. We also demonstrate that cMask displays strong generalization capabilities, as well as further performance improvement relative to other context-aware MBRL approaches when racing against adversaries with in-distribution behaviors. |
| 2025-10-13 | [Constraint-Aware Reinforcement Learning via Adaptive Action Scaling](http://arxiv.org/abs/2510.11491v1) | Murad Dawood, Usama Ahmed Siddiquie et al. | Safe reinforcement learning (RL) seeks to mitigate unsafe behaviors that arise from exploration during training by reducing constraint violations while maintaining task performance. Existing approaches typically rely on a single policy to jointly optimize reward and safety, which can cause instability due to conflicting objectives, or they use external safety filters that override actions and require prior system knowledge. In this paper, we propose a modular cost-aware regulator that scales the agent's actions based on predicted constraint violations, preserving exploration through smooth action modulation rather than overriding the policy. The regulator is trained to minimize constraint violations while avoiding degenerate suppression of actions. Our approach integrates seamlessly with off-policy RL methods such as SAC and TD3, and achieves state-of-the-art return-to-cost ratios on Safety Gym locomotion tasks with sparse costs, reducing constraint violations by up to 126 times while increasing returns by over an order of magnitude compared to prior methods. |
| 2025-10-13 | [A Faster and More Reliable Middleware for Autonomous Driving Systems](http://arxiv.org/abs/2510.11448v1) | Yuankai He, Hanlin Chen et al. | Ensuring safety in high-speed autonomous vehicles requires rapid control loops and tightly bounded delays from perception to actuation. Many open-source autonomy systems rely on ROS 2 middleware; when multiple sensor and control nodes share one compute unit, ROS 2 and its DDS transports add significant (de)serialization, copying, and discovery overheads, shrinking the available time budget. We present Sensor-in-Memory (SIM), a shared-memory transport designed for intra-host pipelines in autonomous vehicles. SIM keeps sensor data in native memory layouts (e.g., cv::Mat, PCL), uses lock-free bounded double buffers that overwrite old data to prioritize freshness, and integrates into ROS 2 nodes with four lines of code. Unlike traditional middleware, SIM operates beside ROS 2 and is optimized for applications where data freshness and minimal latency outweigh guaranteed completeness. SIM provides sequence numbers, a writer heartbeat, and optional checksums to ensure ordering, liveness, and basic integrity. On an NVIDIA Jetson Orin Nano, SIM reduces data-transport latency by up to 98% compared to ROS 2 zero-copy transports such as FastRTPS and Zenoh, lowers mean latency by about 95%, and narrows 95th/99th-percentile tail latencies by around 96%. In tests on a production-ready Level 4 vehicle running Autoware.Universe, SIM increased localization frequency from 7.5 Hz to 9.5 Hz. Applied across all latency-critical modules, SIM cut average perception-to-decision latency from 521.91 ms to 290.26 ms, reducing emergency braking distance at 40 mph (64 km/h) on dry concrete by 13.6 ft (4.14 m). |
| 2025-10-13 | [HUGR: A Quantum-Classical Intermediate Representation](http://arxiv.org/abs/2510.11420v1) | Mark Koch, Agustín Borgna et al. | We introduce the Hierarchical Unified Graph Representation (HUGR): a novel graph based intermediate representation for mixed quantum-classical programs. HUGR's design features high expressivity and extensibility to capture the capabilities of near-term and forthcoming quantum computing devices, as well as new and evolving abstractions from novel quantum programming paradigms. The graph based structure is machine-friendly and supports powerful pattern matching based compilation techniques. Inspired by MLIR, HUGR's extensibility further allows compilation tooling to reason about programs at multiple levels of abstraction, lowering smoothly between them. Safety guarantees in the structure including strict, static typing and linear quantum types allow rapid development of compilation tooling without fear of program invalidation. A full specification of HUGR and reference implementation are open-source and available online. |
| 2025-10-13 | [Robust Closed-Form Control for MIMO Nonlinear Systems under Conflicting Time-Varying Hard and Soft Constraints](http://arxiv.org/abs/2510.11393v1) | Farhad Mehdifar, Charalampos P. Bechlioulis et al. | This paper introduces a novel robust closed-form control law to handle time-varying hard and soft constraints in uncertain high-relative-degree nonlinear MIMO systems. These constraints represent spatiotemporal specifications in mechanical systems' operational space, with hard constraints ensuring safety-critical requirements and soft constraints encoding performance or task objectives. Initially, all constraints are consolidated into two separate scalar time-varying hard and soft constraint functions, whose positive level sets define feasible regions. A closed-form control law is developed to enforce these constraints using appropriately designed reciprocal barriers and nonlinear transformation functions. When conflicts between hard and soft constraints arise, the control law prioritizes hard constraints by virtually relaxing soft constraints via a dynamic relaxation law. Notably, the proposed control law maintains low complexity by avoiding approximation schemes for coping with system uncertainties. Simulation results confirm the effectiveness of the proposed method. |
| 2025-10-13 | [A Denotational Product Construction for Temporal Verification of Effectful Higher-Order Programs](http://arxiv.org/abs/2510.11320v1) | Kazuki Watanabe, Mayuko Kori et al. | We propose a categorical framework for linear-time temporal verification of effectful higher-order programs, including probabilistic higher-order programs. Our framework provides a generic denotational reduction -- namely, a denotational product construction -- from linear-time safety verification of effectful higher-order programs to computation of weakest pre-conditions of product programs. This reduction enables us to apply existing algorithms for such well-studied computations of weakest pre-conditions, some of which are available as off-the-shelf solvers. We show the correctness of our denotational product construction by proving a preservation theorem under strong monad morphisms and an existence of suitable liftings along a fibration. We instantiate our framework with both probabilistic and angelic nondeterministic higher-order programs, and implement an automated solver for the probabilistic case based on the existing solver developed by Kura and Unno. To the best of our knowledge, this is the first automated verifier for linear-time temporal verification of probabilistic higher-order programs with recursion. |
| 2025-10-13 | [pyspect: An Extensible Toolbox for Automatic Construction of Temporal Logic Trees via Reachability Analysis](http://arxiv.org/abs/2510.11316v1) | Kaj Munhoz Arfvidsson, Loizos Hadjiloizou et al. | In this paper, we present pyspect, a Python toolbox that simplifies the use of reachability analysis for temporal logic problems. Currently, satisfying complex requirements in cyber-physical systems requires significant manual effort and domain expertise to develop the underlying reachability programs. This high development effort limits the broader adoption of reachability analysis for complex verification problems. To address this, pyspect provides a method-agnostic approach to performing reachability analysis for verifying a temporal logic specification via temporal logic trees (TLTs). It enables the specification of complex safety and liveness requirements using high-level logic formulations that are independent of any particular reachability technique or set representation. As a result, pyspect allows for the comparison of different reachability implementations, such as Hamilton-Jacobi and Hybrid Zonotope-based reachability analysis, for the same temporal logic specification. This design separates the concerns of implementation developers (who develop numerical procedures for reachability) and end-users (who write specifications). Through a simple vehicle example, we demonstrate how pyspect simplifies the synthesis of reachability programs, promotes specification reusability, and facilitates side-by-side comparisons of reachability techniques for complex tasks. |
| 2025-10-13 | [Adap-RPF: Adaptive Trajectory Sampling for Robot Person Following in Dynamic Crowded Environments](http://arxiv.org/abs/2510.11308v1) | Weixi Situ, Hanjing Ye et al. | Robot person following (RPF) is a core capability in human-robot interaction, enabling robots to assist users in daily activities, collaborative work, and other service scenarios. However, achieving practical RPF remains challenging due to frequent occlusions, particularly in dynamic and crowded environments. Existing approaches often rely on fixed-point following or sparse candidate-point selection with oversimplified heuristics, which cannot adequately handle complex occlusions caused by moving obstacles such as pedestrians. To address these limitations, we propose an adaptive trajectory sampling method that generates dense candidate points within socially aware zones and evaluates them using a multi-objective cost function. Based on the optimal point, a person-following trajectory is estimated relative to the predicted motion of the target. We further design a prediction-aware model predictive path integral (MPPI) controller that simultaneously tracks this trajectory and proactively avoids collisions using predicted pedestrian motions. Extensive experiments show that our method outperforms state-of-the-art baselines in smoothness, safety, robustness, and human comfort, with its effectiveness further demonstrated on a mobile robot in real-world scenarios. |
| 2025-10-13 | [What to expect from microscopic nuclear modelling for k$_{\rm eff}$ calculations ?](http://arxiv.org/abs/2510.11256v1) | D. Rochman, A. Koning et al. | Comparisons between predicted and benchmark k$_{\rm eff}$ values from criticality-safety systems are often used as metrics to estimate the quality of evaluated nuclear data libraries. Relevant nuclear data for these critical systems generally come from a mixture of expert knowledge and phenomenological predictions. In the present work, we use solely microscopic nuclear modelling from TALYS to estimate actinides cross sections and angular distributions, and we compare the calculated MCNP k$_{\rm eff}$ values for fast systems between the JEFF-3.3 evaluated library, phenomenological and microscopic modelling. The conclusion is that even if the evaluated library leads to the most adequate results, the microscopic nuclear modelling can reach very similar results for these integral quantities. It demonstrates the remarkable advances in the recent decades of microscopic nuclear reaction ingredients for applied integral observables. |
| 2025-10-13 | [Collaborative Shadows: Distributed Backdoor Attacks in LLM-Based Multi-Agent Systems](http://arxiv.org/abs/2510.11246v1) | Pengyu Zhu, Lijun Li et al. | LLM-based multi-agent systems (MAS) demonstrate increasing integration into next-generation applications, but their safety in backdoor attacks remains largely underexplored. However, existing research has focused exclusively on single-agent backdoor attacks, overlooking the novel attack surfaces introduced by agent collaboration in MAS. To bridge this gap, we present the first Distributed Backdoor Attack tailored to MAS. We decompose the backdoor into multiple distributed attack primitives that are embedded within MAS tools. These primitives remain dormant individually but collectively activate only when agents collaborate in a specific sequence, thereby assembling the full backdoor to execute targeted attacks such as data exfiltration. To fully assess this threat, we introduce a benchmark for multi-role collaborative tasks and a sandboxed framework to evaluate. Extensive experiments demonstrate that our attack achieves an attack success rate exceeding 95% without degrading performance on benign tasks. This work exposes novel backdoor attack surfaces that exploit agent collaboration, underscoring the need to move beyond single-agent protection. Code and benchmark are available at https://github.com/whfeLingYu/Distributed-Backdoor-Attacks-in-MAS. |
| 2025-10-13 | [Analyzing Data Quality and Decay in Mega-Constellations: A Physics-Informed Machine Learning Approach](http://arxiv.org/abs/2510.11242v1) | Katarina Dyreby, Francisco Caldas et al. | In the era of mega-constellations, the need for accurate and publicly available information has become fundamental for satellite operators to guarantee the safety of spacecrafts and the Low Earth Orbit (LEO) space environment. This study critically evaluates the accuracy and reliability of publicly available ephemeris data for a LEO mega-constellation - Starlink. The goal of this work is twofold: (i) compare and analyze the quality of the data against high-precision numerical propagation. (ii) Leverage Physics-Informed Machine Learning to extract relevant satellite quantities, such as non-conservative forces, during the decay process. By analyzing two months of real orbital data for approximately 1500 Starlink satellites, we identify discrepancies between high precision numerical algorithms and the published ephemerides, recognizing the use of simplified dynamics at fixed thresholds, planned maneuvers, and limitations in uncertainty propagations. Furthermore, we compare data obtained from multiple sources to track and analyze deorbiting satellites over the same period. Empirically, we extract the acceleration profile of satellites during deorbiting and provide insights relating to the effects of non-conservative forces during reentry. For non-deorbiting satellites, the position Root Mean Square Error (RMSE) was approximately 300 m, while for deorbiting satellites it increased to about 600 m. Through this in-depth analysis, we highlight potential limitations in publicly available data for accurate and robust Space Situational Awareness (SSA), and importantly, we propose a data-driven model of satellite decay in mega-constellations. |
| 2025-10-13 | [AI Alignment Strategies from a Risk Perspective: Independent Safety Mechanisms or Shared Failures?](http://arxiv.org/abs/2510.11235v1) | Leonard Dung, Florian Mai | AI alignment research aims to develop techniques to ensure that AI systems do not cause harm. However, every alignment technique has failure modes, which are conditions in which there is a non-negligible chance that the technique fails to provide safety. As a strategy for risk mitigation, the AI safety community has increasingly adopted a defense-in-depth framework: Conceding that there is no single technique which guarantees safety, defense-in-depth consists in having multiple redundant protections against safety failure, such that safety can be maintained even if some protections fail. However, the success of defense-in-depth depends on how (un)correlated failure modes are across alignment techniques. For example, if all techniques had the exact same failure modes, the defense-in-depth approach would provide no additional protection at all. In this paper, we analyze 7 representative alignment techniques and 7 failure modes to understand the extent to which they overlap. We then discuss our results' implications for understanding the current level of risk and how to prioritize AI alignment research in the future. |
| 2025-10-13 | [TraceAegis: Securing LLM-Based Agents via Hierarchical and Behavioral Anomaly Detection](http://arxiv.org/abs/2510.11203v1) | Jiahao Liu, Bonan Ruan et al. | LLM-based agents have demonstrated promising adaptability in real-world applications. However, these agents remain vulnerable to a wide range of attacks, such as tool poisoning and malicious instructions, that compromise their execution flow and can lead to serious consequences like data breaches and financial loss. Existing studies typically attempt to mitigate such anomalies by predefining specific rules and enforcing them at runtime to enhance safety. Yet, designing comprehensive rules is difficult, requiring extensive manual effort and still leaving gaps that result in false negatives. As agent systems evolve into complex software systems, we take inspiration from software system security and propose TraceAegis, a provenance-based analysis framework that leverages agent execution traces to detect potential anomalies. In particular, TraceAegis constructs a hierarchical structure to abstract stable execution units that characterize normal agent behaviors. These units are then summarized into constrained behavioral rules that specify the conditions necessary to complete a task. By validating execution traces against both hierarchical and behavioral constraints, TraceAegis is able to effectively detect abnormal behaviors. To evaluate the effectiveness of TraceAegis, we introduce TraceAegis-Bench, a dataset covering two representative scenarios: healthcare and corporate procurement. Each scenario includes 1,300 benign behaviors and 300 abnormal behaviors, where the anomalies either violate the agent's execution order or break the semantic consistency of its execution sequence. Experimental results demonstrate that TraceAegis achieves strong performance on TraceAegis-Bench, successfully identifying the majority of abnormal behaviors. |
| 2025-10-13 | [RAG-Pull: Imperceptible Attacks on RAG Systems for Code Generation](http://arxiv.org/abs/2510.11195v1) | Vasilije Stambolic, Aritra Dhar et al. | Retrieval-Augmented Generation (RAG) increases the reliability and trustworthiness of the LLM response and reduces hallucination by eliminating the need for model retraining. It does so by adding external data into the LLM's context. We develop a new class of black-box attack, RAG-Pull, that inserts hidden UTF characters into queries or external code repositories, redirecting retrieval toward malicious code, thereby breaking the models' safety alignment. We observe that query and code perturbations alone can shift retrieval toward attacker-controlled snippets, while combined query-and-target perturbations achieve near-perfect success. Once retrieved, these snippets introduce exploitable vulnerabilities such as remote code execution and SQL injection. RAG-Pull's minimal perturbations can alter the model's safety alignment and increase preference towards unsafe code, therefore opening up a new class of attacks on LLMs. |

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



