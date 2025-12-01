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
| 2025-11-28 | [ThetaEvolve: Test-time Learning on Open Problems](http://arxiv.org/abs/2511.23473v1) | Yiping Wang, Shao-Rong Su et al. | Recent advances in large language models (LLMs) have enabled breakthroughs in mathematical discovery, exemplified by AlphaEvolve, a closed-source system that evolves programs to improve bounds on open problems. However, it relies on ensembles of frontier LLMs to achieve new bounds and is a pure inference system that models cannot internalize the evolving strategies. We introduce ThetaEvolve, an open-source framework that simplifies and extends AlphaEvolve to efficiently scale both in-context learning and Reinforcement Learning (RL) at test time, allowing models to continually learn from their experiences in improving open optimization problems. ThetaEvolve features a single LLM, a large program database for enhanced exploration, batch sampling for higher throughput, lazy penalties to discourage stagnant outputs, and optional reward shaping for stable training signals, etc. ThetaEvolve is the first evolving framework that enable a small open-source model, like DeepSeek-R1-0528-Qwen3-8B, to achieve new best-known bounds on open problems (circle packing and first auto-correlation inequality) mentioned in AlphaEvolve. Besides, across two models and four open tasks, we find that ThetaEvolve with RL at test-time consistently outperforms inference-only baselines, and the model indeed learns evolving capabilities, as the RL-trained checkpoints demonstrate faster progress and better final performance on both trained target task and other unseen tasks. We release our code publicly: https://github.com/ypwang61/ThetaEvolve |
| 2025-11-28 | [Accelerated Execution of Bayesian Neural Networks using a Single Probabilistic Forward Pass and Code Generation](http://arxiv.org/abs/2511.23440v1) | Bernhard Klein, Falk Selker et al. | Machine learning models perform well across domains such as diagnostics, weather forecasting, NLP, and autonomous driving, but their limited uncertainty handling restricts use in safety-critical settings. Traditional neural networks often fail to detect out-of-domain (OOD) data and may output confident yet incorrect predictions. Bayesian neural networks (BNNs) address this by providing probabilistic estimates, but incur high computational cost because predictions require sampling weight distributions and multiple forward passes. The Probabilistic Forward Pass (PFP) offers a highly efficient approximation to Stochastic Variational Inference (SVI) by assuming Gaussian-distributed weights and activations, enabling fully analytic uncertainty propagation and replacing sampling with a single deterministic forward pass. We present an end-to-end pipeline for training, compiling, optimizing, and deploying PFP-based BNNs on embedded ARM CPUs. Using the TVM deep learning compiler, we implement a dedicated library of Gaussian-propagating operators for multilayer perceptrons and convolutional neural networks, combined with manual and automated tuning strategies. Ablation studies show that PFP consistently outperforms SVI in computational efficiency, achieving speedups of up to 4200x for small mini-batches. PFP-BNNs match SVI-BNNs on Dirty-MNIST in accuracy, uncertainty estimation, and OOD detection while greatly reducing compute cost. These results highlight the potential of combining Bayesian approximations with code generation to enable efficient BNN deployment on resource-constrained systems. |
| 2025-11-28 | [IRC-safe jet flavour without modifying anything](http://arxiv.org/abs/2511.23423v1) | Terry Generet | In this work, we describe how infrared-collinear safety can be restored perturbatively for standard definitions of jets and jet flavour. We will explicitly study this approach at next-to-next-to-leading order in QCD, where we will discuss the cause of the violation of infrared safety, argue that it is merely an artefact of the calculation, and explain how a consistent treatment of flavour removes this violation. The most important feature of our approach is that it does not require any changes to the definition of the jet or its flavour, nor does it modify the definition of the cross section. Consequently, the predictions can be directly compared to measurements performed using the standard anti-$k_T$ jet clustering algorithm, without the need for experimental collaborations to adapt their analyses to some new, infrared-collinear-safe definition of jet flavour, as would be the case for most - if not all - solutions presented in the literature thus far. |
| 2025-11-28 | [SimScale: Learning to Drive via Real-World Simulation at Scale](http://arxiv.org/abs/2511.23369v1) | Haochen Tian, Tianyu Li et al. | Achieving fully autonomous driving systems requires learning rational decisions in a wide span of scenarios, including safety-critical and out-of-distribution ones. However, such cases are underrepresented in real-world corpus collected by human experts. To complement for the lack of data diversity, we introduce a novel and scalable simulation framework capable of synthesizing massive unseen states upon existing driving logs. Our pipeline utilizes advanced neural rendering with a reactive environment to generate high-fidelity multi-view observations controlled by the perturbed ego trajectory. Furthermore, we develop a pseudo-expert trajectory generation mechanism for these newly simulated states to provide action supervision. Upon the synthesized data, we find that a simple co-training strategy on both real-world and simulated samples can lead to significant improvements in both robustness and generalization for various planning methods on challenging real-world benchmarks, up to +6.8 EPDMS on navhard and +2.9 on navtest. More importantly, such policy improvement scales smoothly by increasing simulation data only, even without extra real-world data streaming in. We further reveal several crucial findings of such a sim-real learning system, which we term SimScale, including the design of pseudo-experts and the scaling properties for different policy architectures. Our simulation data and code would be released. |
| 2025-11-28 | [Dynamic Power Allocation For NOMA-Based Transmission in 6G Optical Wireless Networks](http://arxiv.org/abs/2511.23326v1) | Ahmad Adnan Qidan, Taisir El-Gorashi et al. | OWC has been considered as a key enabling technology to unlock unprecedented speeds of communication, supporting high demands of data traffic. In this paper, infrared lasers are used as optical transmitters operating in an indoor environment under eye safety regulations due to their high modulation speed. To provide efficient multiple access service, NOMA-based transmission is implemented to multiplex messages intended to multiple users in the power domain and maximize the spectral efficiency of our laser-based OWC network. In particular, a BIA outer precoder is designed to coordinate the transmission among multiple APs and determine the precoding matrices for groups of users potential formed according to NOMA principles. For effective use of NOMA, an optimization problem is formulated to maximize the sum rate of the network through forming optimum groups under certain joint conditions, efficient power allocation, high quality of service for each weak and strong users, and high overall system performance. Such optimization problems are defined as max-min fractional programs difficult to solve in practice. Therefore, a dynamic application for NOMA is introduced using two algorithms. First, a RF-aided dynamic algorithm is designed to form multiple groups, where users exchange binary variables among them through an RF system to establish distance-based weight edges, which are used as a metric for the grouping process. Second, a dynamic power allocation is proposed to determine the optimum power allocated to each group, while the users belonging to a certain group receive their traffic demands regardless of their classification as weak or strong. The results show the convergence of the proposed dynamic application to the optimum solution, and its high performance in terms of sum rate, fairness, and energy efficiency compared to counterpart schemes. |
| 2025-11-28 | [Toward Automatic Safe Driving Instruction: A Large-Scale Vision Language Model Approach](http://arxiv.org/abs/2511.23311v1) | Haruki Sakajo, Hiroshi Takato et al. | Large-scale Vision Language Models (LVLMs) exhibit advanced capabilities in tasks that require visual information, including object detection. These capabilities have promising applications in various industrial domains, such as autonomous driving. For example, LVLMs can generate safety-oriented descriptions of videos captured by road-facing cameras. However, ensuring comprehensive safety requires monitoring driver-facing views as well to detect risky events, such as the use of mobiles while driving. Thus, the ability to process synchronized inputs is necessary from both driver-facing and road-facing cameras. In this study, we develop models and investigate the capabilities of LVLMs by constructing a dataset and evaluating their performance on this dataset. Our experimental results demonstrate that while pre-trained LVLMs have limited effectiveness, fine-tuned LVLMs can generate accurate and safety-aware driving instructions. Nonetheless, several challenges remain, particularly in detecting subtle or complex events in the video. Our findings and error analysis provide valuable insights that can contribute to the improvement of LVLM-based systems in this domain. |
| 2025-11-28 | [SafeHumanoid: VLM-RAG-driven Control of Upper Body Impedance for Humanoid Robot](http://arxiv.org/abs/2511.23300v1) | Yara Mahmoud, Jeffrin Sam et al. | Safe and trustworthy Human Robot Interaction (HRI) requires robots not only to complete tasks but also to regulate impedance and speed according to scene context and human proximity. We present SafeHumanoid, an egocentric vision pipeline that links Vision Language Models (VLMs) with Retrieval-Augmented Generation (RAG) to schedule impedance and velocity parameters for a humanoid robot. Egocentric frames are processed by a structured VLM prompt, embedded and matched against a curated database of validated scenarios, and mapped to joint-level impedance commands via inverse kinematics. We evaluate the system on tabletop manipulation tasks with and without human presence, including wiping, object handovers, and liquid pouring. The results show that the pipeline adapts stiffness, damping, and speed profiles in a context-aware manner, maintaining task success while improving safety. Although current inference latency (up to 1.4 s) limits responsiveness in highly dynamic settings, SafeHumanoid demonstrates that semantic grounding of impedance control is a viable path toward safer, standard-compliant humanoid collaboration. |
| 2025-11-28 | [All for One and One for All: Program Logics for Exploiting Internal Determinism in Parallel Programs](http://arxiv.org/abs/2511.23283v1) | Alexandre Moine, Sam Westrick et al. | Nondeterminism makes parallel programs challenging to write and reason about. To avoid these challenges, researchers have developed techniques for internally deterministic parallel programming, in which the steps of a parallel computation proceed in a deterministic way. Internal determinism is useful because it lets a programmer reason about a program as if it executed in a sequential order. However, no verification framework exists to exploit this property and simplify formal reasoning about internally deterministic programs.   To capture the essence of why internally deterministic programs should be easier to reason about, this paper defines a property called schedule-independent safety. A program satisfies schedule-independent safety, if, to show that the program is safe across all orderings, it suffices to show that one terminating execution of the program is safe. We then present a separation logic called Musketeer for proving that a program satisfies schedule-independent safety. Once a parallel program has been shown to satisfy schedule-independent safety, we can verify it with a new logic called Angelic, which allows one to dynamically select and verify just one sequential ordering of the program.   Using Musketeer, we prove the soundness of MiniDet, an affine type system for enforcing internal determinism. MiniDet supports several core algorithmic primitives for internally deterministic programming that have been identified in the research literature, including a deterministic version of a concurrent hash set. Because any syntactically well-typed MiniDet program satisfies schedule-independent safety, we can apply Angelic to verify such programs.   All results in this paper have been verified in Rocq using the Iris separation logic framework. |
| 2025-11-28 | [Fault-Tolerant MARL for CAVs under Observation Perturbations for Highway On-Ramp Merging](http://arxiv.org/abs/2511.23193v1) | Yuchen Shi, Huaxin Pei et al. | Multi-Agent Reinforcement Learning (MARL) holds significant promise for enabling cooperative driving among Connected and Automated Vehicles (CAVs). However, its practical application is hindered by a critical limitation, i.e., insufficient fault tolerance against observational faults. Such faults, which appear as perturbations in the vehicles' perceived data, can substantially compromise the performance of MARL-based driving systems. Addressing this problem presents two primary challenges. One is to generate adversarial perturbations that effectively stress the policy during training, and the other is to equip vehicles with the capability to mitigate the impact of corrupted observations. To overcome the challenges, we propose a fault-tolerant MARL method for cooperative on-ramp vehicles incorporating two key agents. First, an adversarial fault injection agent is co-trained to generate perturbations that actively challenge and harden the vehicle policies. Second, we design a novel fault-tolerant vehicle agent equipped with a self-diagnosis capability, which leverages the inherent spatio-temporal correlations in vehicle state sequences to detect faults and reconstruct credible observations, thereby shielding the policy from misleading inputs. Experiments in a simulated highway merging scenario demonstrate that our method significantly outperforms baseline MARL approaches, achieving near-fault-free levels of safety and efficiency under various observation fault patterns. |
| 2025-11-28 | [Are LLMs Good Safety Agents or a Propaganda Engine?](http://arxiv.org/abs/2511.23174v1) | Neemesh Yadav, Francesco Ortu et al. | Large Language Models (LLMs) are trained to refuse to respond to harmful content. However, systematic analyses of whether this behavior is truly a reflection of its safety policies or an indication of political censorship, that is practiced globally by countries, is lacking. Differentiating between safety influenced refusals or politically motivated censorship is hard and unclear. For this purpose we introduce PSP, a dataset built specifically to probe the refusal behaviors in LLMs from an explicitly political context. PSP is built by formatting existing censored content from two data sources, openly available on the internet: sensitive prompts in China generalized to multiple countries, and tweets that have been censored in various countries. We study: 1) impact of political sensitivity in seven LLMs through data-driven (making PSP implicit) and representation-level approaches (erasing the concept of politics); and, 2) vulnerability of models on PSP through prompt injection attacks (PIAs). Associating censorship with refusals on content with masked implicit intent, we find that most LLMs perform some form of censorship. We conclude with summarizing major attributes that can cause a shift in refusal distributions across models and contexts of different countries. |
| 2025-11-28 | [Design loads for wave impacts -- introducing the Probabilistic Adaptive Screening (PAS) method for predicting extreme non-linear loads on maritime structures](http://arxiv.org/abs/2511.23156v1) | Sanne M. van Essen, Harleigh C. Seyffert | To ensure the safety of marine and coastal structures, extreme (design) values should be known at the design stage. But for such complex systems, estimating the magnitude of events which are both non-linear and rare is extremely challenging, and involves considerable computational cost to capture the high-fidelity physics. To address this challenge, we offer a new multi-fidelity screening method, Probabilistic Adaptive Screening (PAS), which accurately predicts extreme values of strongly non-linear wave-induced loads while minimising the required high-fidelity simulation duration. The method introduces a probabilistic approach to multi-fidelity screening, allowing efficient linear potential flow indicators to be used in the low-fidelity stage, even for strongly non-linear load cases. The method is validated against a range of cases, including non-linear waves, and ship vertical bending moments, green water impact loads, and slamming loads. It can be concluded that PAS accurately estimates both the short-term distributions and extreme values in all test cases, with most probable maximum (MPM) values within 10\% of the available full brute-force Monte-Carlo Simulation (MCS) results. In addition, PAS achieves this performance very efficiently, requiring less than 4\% of the high-fidelity simulation time needed for conventional MCS. These results demonstrate that PAS can reliably reproduce the statistics of both weakly and strongly non-linear extreme load problems, while significantly reducing the associated computational cost. The present study validates the statistical PAS framework; further work should focus on validating the full procedure including CFD load simulations, and on validating it for long-term extremes. |
| 2025-11-28 | [Control Barrier Function for Unknown Systems: An Approximation-free Approach](http://arxiv.org/abs/2511.23022v1) | Shubham Sawarkar, Pushpak Jagtap | We study the prescribed-time reach-avoid (PT-RA) control problem for nonlinear systems with unknown dynamics operating in environments with moving obstacles. Unlike robust or learning based Control Barrier Function (CBF) methods, the proposed framework requires neither online model learning nor uncertainty bound estimation. A CBF-based Quadratic Program (CBF-QP) is solved on a simple virtual system to generate a safe reference satisfying PT-RA conditions with respect to time-varying, tightened obstacle and goal sets. The true system is confined to a Virtual Confinement Zone (VCZ) around this reference using an approximation-free feedback law. This construction guarantees real-time safety and prescribed-time target reachability under unknown dynamics and dynamic constraints without explicit model identification or offline precomputation. Simulation results illustrate reliable dynamic obstacle avoidance and timely convergence to the target set. |
| 2025-11-28 | [Maritime Activities Observed Through Open-Access Positioning Data: Moving and Stationary Vessels in the Baltic Sea](http://arxiv.org/abs/2511.23016v1) | Moritz Hütten | Understanding past and present maritime activity patterns is critical for navigation safety, environmental assessment, and commercial operations. An increasing number of services now openly provide positioning data from the Automatic Identification System (AIS) via ground-based receivers. We show that coastal vessel activity can be reconstructed from open access data with high accuracy, even with limited data quality and incomplete receiver coverage. For three months of open AIS data in the Baltic Sea from August to October 2024, we present (i) cleansing and reconstruction methods to improve the data quality, and (ii) a journey model that converts AIS message data into vessel counts, traffic estimates, and spatially resolved vessel density at a resolution of $\sim$400 m. Vessel counts are provided, along with their uncertainties, for both moving and stationary activity. Vessel density maps also enable the identification of port locations, and we infer the most crowded and busiest coastal areas in the Baltic Sea. We find that on average, $\gtrsim$4000 vessels simultaneously operate in the Baltic Sea, and more than 300 vessels enter or leave the area each day. Our results agree within 20\% with previous studies relying on proprietary data. |
| 2025-11-28 | [JarvisEvo: Towards a Self-Evolving Photo Editing Agent with Synergistic Editor-Evaluator Optimization](http://arxiv.org/abs/2511.23002v1) | Yunlong Lin, Linqing Wang et al. | Agent-based editing models have substantially advanced interactive experiences, processing quality, and creative flexibility. However, two critical challenges persist: (1) instruction hallucination, text-only chain-of-thought (CoT) reasoning cannot fully prevent factual errors due to inherent information bottlenecks; (2) reward hacking, dynamic policy optimization against static reward models allows agents to exploit flaws in reward functions. To address these issues, we propose JarvisEvo, a unified image editing agent that emulates an expert human designer by iteratively editing, selecting appropriate tools, evaluating results, and reflecting on its own decisions to refine outcomes. JarvisEvo offers three key advantages: (1) an interleaved multimodal chain-of-thought (iMCoT) reasoning mechanism that enhances instruction following and editing quality; (2) a synergistic editor-evaluator policy optimization (SEPO) framework that enables self-improvement without external rewards, effectively mitigating reward hacking; and (3) support for both global and local fine-grained editing through seamless integration of Adobe Lightroom. On ArtEdit-Bench, JarvisEvo outperforms Nano-Banana by an average of 18.95% on preservative editing metrics, including a substantial 44.96% improvement in pixel-level content fidelity. |
| 2025-11-28 | [ShoppingComp: Are LLMs Really Ready for Your Shopping Cart?](http://arxiv.org/abs/2511.22978v1) | Huaixiao Tou, Ying Zeng et al. | We present ShoppingComp, a challenging real-world benchmark for rigorously evaluating LLM-powered shopping agents on three core capabilities: precise product retrieval, expert-level report generation, and safety critical decision making. Unlike prior e-commerce benchmarks, ShoppingComp introduces highly complex tasks under the principle of guaranteeing real products and ensuring easy verifiability, adding a novel evaluation dimension for identifying product safety hazards alongside recommendation accuracy and report quality. The benchmark comprises 120 tasks and 1,026 scenarios, curated by 35 experts to reflect authentic shopping needs. Results reveal stark limitations of current LLMs: even state-of-the-art models achieve low performance (e.g., 11.22% for GPT-5, 3.92% for Gemini-2.5-Flash). These findings highlight a substantial gap between research benchmarks and real-world deployment, where LLMs make critical errors such as failure to identify unsafe product usage or falling for promotional misinformation, leading to harmful recommendations. ShoppingComp fills the gap and thus establishes a new standard for advancing reliable and practical agents in e-commerce. |
| 2025-11-28 | [An LLM-Assisted Multi-Agent Control Framework for Roll-to-Roll Manufacturing Systems](http://arxiv.org/abs/2511.22975v1) | Jiachen Li, Shihao Li et al. | Roll-to-roll manufacturing requires precise tension and velocity control to ensure product quality, yet controller commissioning and adaptation remain time-intensive processes dependent on expert knowledge. This paper presents an LLM-assisted multi-agent framework that automates control system design and adaptation for R2R systems while maintaining safety. The framework operates through five phases: system identification from operational data, automated controller selection and tuning, sim-to-real adaptation with safety verification, continuous monitoring with diagnostic capabilities, and periodic model refinement. Experimental validation on a R2R system demonstrates successful tension regulation and velocity tracking under significant model uncertainty, with the framework achieving performance convergence through iterative adaptation. The approach reduces manual tuning effort while providing transparent diagnostic information for maintenance planning, offering a practical pathway for integrating AI-assisted automation in manufacturing control systems. |
| 2025-11-28 | [Extended Serial Safety Net: A Refined Serializability Criterion for Multiversion Concurrency Control](http://arxiv.org/abs/2511.22956v1) | Atsushi Kitazawa, Chihaya Ito et al. | A long line of concurrency-control (CC) protocols argues correctness via a single serialization point (begin or commit), an assumption that is incompatible with snapshot isolation (SI), where read-write anti-dependencies arise. Serial Safety Net (SSN) offers a lightweight commit-time test but is conservative and effectively anchored on commit time as the sole point. We present ESSN, a principled generalization of SSN that relaxes the exclusion condition to allow more transactions to commit safely, and we prove that this preserves multiversion serializability (MVSR) and that it strictly subsumes SSN. ESSN states an MVSG (Multiversion Serialization Graph)-based criterion and introduces a known total order over transactions (KTO; e.g., begin-ordered or commit-ordered) for reasoning about the graph's serializability. With a single commit-time check under invariant-based semantics, ESSN's exclusion condition preserves monotonicity along per-item version chains, and eliminates chain traversal. The protocol is Direct Serialization Graph (DSG)-based with commit-time work linear in the number of reads and writes, matching SSN's per-version footprint. We also make mixed workloads explicit by defining a Long transaction via strict interval containment of Short transactions, and we evaluate ESSN on reproducible workloads. Under a commit-ordered KTO, using begin-snapshot reads reduces the long-transaction abort rate by up to approximately 0.25 absolute (about 50% relative) compared with SSN. |
| 2025-11-28 | [RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](http://arxiv.org/abs/2511.22950v1) | Haiyang Mei, Qiming Huang et al. | Accurate robot segmentation is a fundamental capability for robotic perception. It enables precise visual servoing for VLA systems, scalable robot-centric data augmentation, accurate real-to-sim transfer, and reliable safety monitoring in dynamic human-robot environments. Despite the strong capabilities of modern segmentation models, surprisingly it remains challenging to segment robots. This is due to robot embodiment diversity, appearance ambiguity, structural complexity, and rapid shape changes. Embracing these challenges, we introduce RobotSeg, a foundation model for robot segmentation in image and video. RobotSeg is built upon the versatile SAM 2 foundation model but addresses its three limitations for robot segmentation, namely the lack of adaptation to articulated robots, reliance on manual prompts, and the need for per-frame training mask annotations, by introducing a structure-enhanced memory associator, a robot prompt generator, and a label-efficient training strategy. These innovations collectively enable a structure-aware, automatic, and label-efficient solution. We further construct the video robot segmentation (VRS) dataset comprising over 2.8k videos (138k frames) with diverse robot embodiments and environments. Extensive experiments demonstrate that RobotSeg achieves state-of-the-art performance on both images and videos, establishing a strong foundation for future advances in robot perception. |
| 2025-11-28 | [Seeing before Observable: Potential Risk Reasoning in Autonomous Driving via Vision Language Models](http://arxiv.org/abs/2511.22928v1) | Jiaxin Liu, Xiangyu Yan et al. | Ensuring safety remains a key challenge for autonomous vehicles (AVs), especially in rare and complex scenarios. One critical but understudied aspect is the \textbf{potential risk} situations, where the risk is \textbf{not yet observable} but can be inferred from subtle precursors, such as anomalous behaviors or commonsense violations. Recognizing these precursors requires strong semantic understanding and reasoning capabilities, which are often absent in current AV systems due to the scarcity of such cases in existing driving or risk-centric datasets. Moreover, current autonomous driving accident datasets often lack annotations of the causal reasoning chains behind incidents, which are essential for identifying potential risks before they become observable. To address these gaps, we introduce PotentialRiskQA, a novel vision-language dataset designed for reasoning about potential risks prior to observation. Each sample is annotated with structured scene descriptions, semantic precursors, and inferred risk outcomes. Based on this dataset, we further propose PR-Reasoner, a vision-language-model-based framework tailored for onboard potential risk reasoning. Experimental results show that fine-tuning on PotentialRiskQA enables PR-Reasoner to significantly enhance its performance on the potential risk reasoning task compared to baseline VLMs. Together, our dataset and model provide a foundation for developing autonomous systems with improved foresight and proactive safety capabilities, moving toward more intelligent and resilient AVs. |
| 2025-11-28 | [ORION: Teaching Language Models to Reason Efficiently in the Language of Thought](http://arxiv.org/abs/2511.22891v1) | Kumar Tanmay, Kriti Aggarwal et al. | Large Reasoning Models (LRMs) achieve strong performance in mathematics, code generation, and task planning, but their reliance on long chains of verbose "thinking" tokens leads to high latency, redundancy, and incoherent reasoning paths. Inspired by the Language of Thought Hypothesis, which posits that human reasoning operates over a symbolic, compositional mental language called Mentalese, we introduce a framework that trains models to reason in a similarly compact style. Mentalese encodes abstract reasoning as ultra-compressed, structured tokens, enabling models to solve complex problems with far fewer steps. To improve both efficiency and accuracy, we propose SHORTER LENGTH PREFERENCE OPTIMIZATION (SLPO), a reinforcement learning method that rewards concise solutions that stay correct, while still allowing longer reasoning when needed. Applied to Mentalese-aligned models, SLPO yields significantly higher compression rates by enabling concise reasoning that preserves the benefits of detailed thinking without the computational overhead. Across benchmarks including AIME 2024 and 2025, MinervaMath, OlympiadBench, Math500, and AMC, our ORION models produce reasoning traces with 4-16x fewer tokens, achieve up to 5x lower inference latency, and reduce training costs by 7-9x relative to the DeepSeek R1 Distilled model, while maintaining 90-98% of its accuracy. ORION also surpasses Claude and ChatGPT-4o by up to 5% in accuracy while maintaining 2x compression. These results show that Mentalese-style compressed reasoning offers a step toward human-like cognitive efficiency, enabling real-time, cost-effective reasoning without sacrificing accuracy. |

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



