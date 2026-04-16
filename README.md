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
| 2026-04-15 | [Persistent Iterators with Value Semantics](http://arxiv.org/abs/2604.14072v1) | Yihe Li, Gregory J. Duck | Iterators are a fundamental programming abstraction for traversing and modifying elements in containers in mainstream imperative languages such as C++. Iterators provide a uniform access mechanism that hides low-level implementation details of the underlying data structure. However, iterators over mutable containers suffer from well-known hazards including invalidation, aliasing, data races, and subtle side effects. Immutable data structures, as used in functional programming languages, avoid the pitfalls of mutation but rely on a very different programming model based on recursion and higher-order combinators rather than iteration. However, these combinators are not always well-suited to expressing certain algorithms, and recursion can expose implementation details of the underlying data structure.   In this paper, we propose persistent iterators -- a new abstraction that reconciles the familiar iterator-based programming style of imperative languages with the semantics of persistent data structures. A persistent iterator snapshots the version of its underlying container at creation, ensuring safety against invalidation and aliasing. Iterator operations operate on the iterator-local copy of the container, giving true value semantics: variables can be rebound to new persistent values while previous versions remain accessible. We implement our approach in the form of LibFPP -- a C++ container library providing persistent vectors, maps, sets, strings, and other abstractions as persistent counterparts to the Standard Template Library (STL). Our evaluation shows that LibFPP retains the expressiveness of iterator-based programming, eliminates iterator-invalidation, and achieves asymptotic complexities comparable to STL implementations. Our design targets use cases where persistence and safety are desired, while allowing developers to retain familiar iterator-based programming patterns. |
| 2026-04-15 | [From Where Words Come: Efficient Regularization of Code Tokenizers Through Source Attribution](http://arxiv.org/abs/2604.14053v1) | Pavel Chizhov, Egor Bogomolov et al. | Efficiency and safety of Large Language Models (LLMs), among other factors, rely on the quality of tokenization. A good tokenizer not only improves inference speed and language understanding but also provides extra defense against jailbreak attacks and lowers the risk of hallucinations. In this work, we investigate the efficiency of code tokenization, in particular from the perspective of data source diversity. We demonstrate that code tokenizers are prone to producing unused, and thus under-trained, tokens due to the imbalance in repository and language diversity in the training data, as well as the dominance of source-specific, repetitive tokens that are often unusable in future inference. By modifying the BPE objective and introducing merge skipping, we implement different techniques under the name Source-Attributed BPE (SA-BPE) to regularize BPE training and minimize overfitting, thereby substantially reducing the number of under-trained tokens while maintaining the same inference procedure as with regular BPE. This provides an effective tool suitable for production use. |
| 2026-04-15 | [Hierarchical Reinforcement Learning with Runtime Safety Shielding for Power Grid Operation](http://arxiv.org/abs/2604.14032v1) | Gitesh Malik | Reinforcement learning has shown promise for automating power-grid operation tasks such as topology control and congestion management. However, its deployment in real-world power systems remains limited by strict safety requirements, brittleness under rare disturbances, and poor generalization to unseen grid topologies. In safety-critical infrastructure, catastrophic failures cannot be tolerated, and learning-based controllers must operate within hard physical constraints.   This paper proposes a safety-constrained hierarchical control framework for power-grid operation that explicitly decouples long-horizon decision-making from real-time feasibility enforcement. A high-level reinforcement learning policy proposes abstract control actions, while a deterministic runtime safety shield filters unsafe actions using fast forward simulation. Safety is enforced as a runtime invariant, independent of policy quality or training distribution.   The proposed framework is evaluated on the Grid2Op benchmark suite under nominal conditions, forced line-outage stress tests, and zero-shot deployment on the ICAPS 2021 large-scale transmission grid without retraining. Results show that flat reinforcement learning policies are brittle under stress, while safety-only methods are overly conservative. In contrast, the proposed hierarchical and safety-aware approach achieves longer episode survival, lower peak line loading, and robust zero-shot generalization to unseen grids.   These results indicate that safety and generalization in power-grid control are best achieved through architectural design rather than increasingly complex reward engineering, providing a practical path toward deployable learning-based controllers for real-world energy systems. |
| 2026-04-15 | [Weighted NetKAT: A Programming Language For Quantitative Network Verification](http://arxiv.org/abs/2604.13987v1) | Emmanuel Suárez Acevedo, Tiago Ferreira et al. | We introduce weighted NetKAT, a domain-specific language for modeling and verifying quantitative network properties. The language is parametric on a semiring, enabling the treatment of a wide range of quantities in a uniform way. We provide a denotational semantics and an equivalent operational semantics, the latter based on a novel model of weighted NetKAT automata (WNKA) capturing the stateful behavior of our language. With WNKA, we obtain a class of generic decision procedures for reasoning about quantitative safety and reachability in a fully automatic way, even in the presence of possibly unbounded iteration. We demonstrate the applicability of our framework in a case study using Internet2's Abilene network as the underlying topology. |
| 2026-04-15 | [[Emerging Ideas] Artificial Tripartite Intelligence: A Bio-Inspired, Sensor-First Architecture for Physical AI](http://arxiv.org/abs/2604.13959v1) | You Rim Choi, Subeom Park et al. | As AI moves from data centers to robots and wearables, scaling ever-larger models becomes insufficient. Physical AI operates under tight latency, energy, privacy, and reliability constraints, and its performance depends not only on model capacity but also on how signals are acquired through controllable sensors in dynamic environments. We present Artificial Tripartite Intelligence (ATI), a bio-inspired, sensor-first architectural contract for physical AI. ATI is tripartite at the systems level: a Brainstem (L1) provides reflexive safety and signal-integrity control, a Cerebellum (L2) performs continuous sensor calibration, and a Cerebral Inference Subsystem spanning L3/L4 supports routine skill selection and execution, coordination, and deep reasoning. This modular organization allows sensor control, adaptive sensing, edge-cloud execution, and foundation model reasoning to co-evolve within one closed-loop architecture, while keeping time-critical sensing and control on device and invoking higher-level inference only when needed. We instantiate ATI in a mobile camera prototype under dynamic lighting and motion. In our routed evaluation (L3-L4 split inference), compared to the default auto-exposure setting, ATI (L1/L2 adaptive sensing) improves end-to-end accuracy from 53.8% to 88% while reducing remote L4 invocations by 43.3%. These results show the value of co-designing sensing and inference for embodied AI. |
| 2026-04-15 | [HINTBench: Horizon-agent Intrinsic Non-attack Trajectory Benchmark](http://arxiv.org/abs/2604.13954v1) | Jiacheng Wang, Jinchang Hou et al. | Existing agent-safety evaluation has focused mainly on externally induced risks. Yet agents may still enter unsafe trajectories under benign conditions. We study this complementary but underexplored setting through the lens of \emph{intrinsic} risk, where intrinsic failures remain latent, propagate across long-horizon execution, and eventually lead to high-consequence outcomes. To evaluate this setting, we introduce \emph{non-attack intrinsic risk auditing} and present \textbf{HINTBench}, a benchmark of 629 agent trajectories (523 risky, 106 safe; 33 steps on average) supporting three tasks: risk detection, risk-step localization, and intrinsic failure-type identification. Its annotations are organized under a unified five-constraint taxonomy. Experiments reveal a substantial capability gap: strong LLMs perform well on trajectory-level risk detection, but their performance drops to below 35 Strict-F1 on risk-step localization, while fine-grained failure diagnosis proves even harder. Existing guard models transfer poorly to this setting. These findings establish intrinsic risk auditing as an open challenge for agent safety. |
| 2026-04-15 | [A Case Study on Energy-Efficient Edge AI Crack Segmentation](http://arxiv.org/abs/2604.13933v1) | Matthias Tschope, Mohamed Moursi et al. | Crack segmentation on edge devices can support continuous infrastructure monitoring and maintenance and thereby help to preserve public safety. Furthermore, autonomous infrastructure monitoring by using Unmanned Aerial Vehicles (UAVs) can reduce inspection risks, as human operators no longer need to enter hazardous areas. Edge processing reduces the cost of inspection by eliminating the need for high resolution image storage for offline processing and mitigates the security risks and bandwidth requirements of streaming to cloud servers. Edge inference is difficult due to the limited memory and computational capabilities of edge devices, which can affect both accuracy and latency. Furthermore, battery-powered devices are subject to strict power and energy constraints. Together, these limitations impose restrictions on the model size and computational complexity that can be deployed close to the sensor. In recent years, Transformers have achieved state-of-the-art accuracy in a variety of applications, including semantic segmentation. However, Transformer-based models are typically large and computationally intensive, making efficient edge deployment difficult. To address this, we first apply knowledge distillation to enhance the performance of the base models. We then use PTQ to compress the models further. Additionally, we consider the deployment of these models across multiple edge platforms. To maximize energy efficiency, we design and implement a custom hardware architecture for the models on an FPGA. Our results show that Knowledge Distillation (KD) improves all tested U-Net variants. Among the evaluated platforms, the selected FPGA implementation achieves 398 FPS at 204.99 Frames/J while maintaining a mean IoU of 69.42%. In addition, our best model reaches 71.92% mean IoU, which is 8.82 percentage points (pps) higher than the previously reported result on the CrackVision12K dataset. |
| 2026-04-15 | [Beyond Conservative Automated Driving in Multi-Agent Scenarios via Coupled Model Predictive Control and Deep Reinforcement Learning](http://arxiv.org/abs/2604.13891v1) | Saeed Rahmani, Gözde Körpe et al. | Automated driving at unsignalized intersections is challenging due to complex multi-vehicle interactions and the need to balance safety and efficiency. Model Predictive Control (MPC) offers structured constraint handling through optimization but relies on hand-crafted rules that often produce overly conservative behavior. Deep Reinforcement Learning (RL) learns adaptive behaviors from experience but often struggles with safety assurance and generalization to unseen environments. In this study, we present an integrated MPC-RL framework to improve navigation performance in multi-agent scenarios. Experiments show that MPC-RL outperforms standalone MPC and end-to-end RL across three traffic-density levels. Collectively, MPC-RL reduces the collision rate by 21% and improves the success rate by 6.5% compared to pure MPC. We further evaluate zero-shot transfer to a highway merging scenario without retraining. Both MPC-based methods transfer substantially better than end-to-end PPO, which highlights the role of the MPC backbone in cross-scenario robustness. The framework also shows faster loss stabilization than end-to-end RL during training, which indicates a reduced learning burden. These results suggest that the integrated approach can improve the balance between safety performance and efficiency in multi-agent intersection scenarios, while the MPC component provides a strong foundation for generalization across driving environments. The implementation code is available open-source. |
| 2026-04-15 | [Drowsiness-Aware Adaptive Autonomous Braking System based on Deep Reinforcement Learning for Enhanced Road Safety](http://arxiv.org/abs/2604.13878v1) | Hossem Eddine Hafidi, Elisabetta De Giovanni et al. | Driver drowsiness significantly impairs the ability to accurately judge safe braking distances and is estimated to contribute to 10%-20% of road accidents in Europe. Traditional driver-assistance systems lack adaptability to real-time physiological states such as drowsiness. This paper proposes a deep reinforcement learning-based autonomous braking system that integrates vehicle dynamics with driver physiological data. Drowsiness is detected from ECG signals using a Recurrent Neural Network (RNN), selected through an extensive benchmark analysis of 2-minute windows with varying segmentation and overlap configurations. The inferred drowsiness state is incorporated into the observable state space of a Double-Dueling Deep Q-Network (DQN) agent, where driver impairment is modeled as an action delay. The system is implemented and evaluated in a high-fidelity CARLA simulation environment. Experimental results show that the proposed agent achieves a 99.99% success rate in avoiding collisions under both drowsy and non-drowsy conditions. These findings demonstrate the effectiveness of physiology-aware control strategies for enhancing adaptive and intelligent driving safety systems. |
| 2026-04-15 | [Hardware-Efficient Neuro-Symbolic Networks with the Exp-Minus-Log Operator](http://arxiv.org/abs/2604.13871v1) | Eymen Ipek | Deep neural networks (DNNs) deliver state-of-the-art accuracy on regression and classification tasks, yet two structural deficits persistently obstruct their deployment in safety-critical, resource-constrained settings: (i) opacity of the learned function, which precludes formal verification, and (ii) reliance on heterogeneous, library-bound activation functions that inflate latency and silicon area on edge hardware. The recently introduced Exp-Minus-Log (EML) Sheffer operator, eml(x, y) = exp(x) - ln(y), was shown by Odrzywolek (2026) to be sufficient - together with the constant 1 - to express every standard elementary function as a binary tree of identical nodes. We propose to embed EML primitives inside conventional DNN architectures, yielding a hybrid DNN-EML model in which the trunk learns distributed representations and the head is a depth-bounded, weight-sparse EML tree whose snapped weights collapse to closed-form symbolic sub-expressions. We derive the forward equations, prove computational-cost bounds, analyse inference and training acceleration relative to multilayer perceptrons (MLPs) and physics-informed neural networks (PINNs), and quantify the trade-offs for FPGA/analog deployment. We argue that the DNN-EML pairing closes a literature gap: prior neuro-symbolic and equation-learner approaches (EQL, KAN, AI-Feynman) work with heterogeneous primitive sets and do not exploit a single hardware-realisable Sheffer element. A balanced assessment shows that EML is unlikely to accelerate training, and on commodity CPU/GPU it is also unlikely to accelerate inference; however, on a custom EML cell (FPGA logic block or analog circuit) the asymptotic latency advantage can reach an order of magnitude with simultaneous gain in interpretability and formal-verification tractability. |
| 2026-04-15 | ["AI Psychosis" in Context: How Conversation History Shapes LLM Responses to Delusional Beliefs](http://arxiv.org/abs/2604.13860v1) | Luke Nicholls, Robert Hutto et al. | Extended interaction with large language models (LLMs) has been linked to the reinforcement of delusional beliefs, a phenomenon attracting growing clinical and public concern. Yet most empirical work evaluates model safety in brief interactions, which may not reflect how these harms develop through sustained dialogue. We tested five models across three levels of accumulated context, using the same escalating delusional history to isolate its effect on model behaviour. Human raters coded responses on risk and safety dimensions, and each model was analysed qualitatively. Models separated into two distinct tiers: GPT-4o, Grok 4.1 Fast, and Gemini 3 Pro exhibited high-risk, low-safety profiles; Claude Opus 4.5 and GPT-5.2 Instant displayed the opposite pattern. As context accumulated, performance tended to degrade in the unsafe group, while the same material activated stronger safety interventions among the safer models. Qualitative analysis identified distinct mechanisms of failure, including validation of the user's delusional premises, elaboration beyond them, and attempting harm reduction from within the delusional frame. Safer models, however, often used the established relationship to support intervention, taking accountability for past missteps so that redirection would not be received as betrayal. These findings indicate that accumulated context functions as a stress test of safety architecture, revealing whether a model treats prior dialogue as a worldview to inherit or as evidence to evaluate. Short-context assessments may therefore mischaracterise model safety, underestimating danger in some systems while missing context-activated gains in others. The results suggest that delusional reinforcement by LLMs reflects a preventable alignment failure. In demonstrating that these harms can be resisted, the safer models establish a baseline future systems should now be expected to meet. |
| 2026-04-15 | [Mosaic: An Extensible Framework for Composing Rule-Based and Learned Motion Planners](http://arxiv.org/abs/2604.13853v1) | Nick Le Large, Marlon Steiner et al. | Safe and explainable motion planning remains a central challenge in autonomous driving. While rule-based planners offer predictable and explainable behavior, they often fail to grasp the complexity and uncertainty of real-world traffic. Conversely, learned planners exhibit strong adaptability but suffer from reduced transparency and occasional safety violations. We introduce Mosaic, an extensible framework for structured decision-making that integrates both paradigms through arbitration graphs. By decoupling trajectory verification and scoring from the generation of trajectories by individual planners, every decision becomes transparent and traceable. Trajectory verification at a higher level introduces redundancy between the planners, limiting emergency braking to the rare case where all planners fail to produce a valid trajectory. Through unified scoring and optimal trajectory selection, rule-based and learned planners with complementary strengths and weaknesses can be combined to yield the best of both worlds. In experimental evaluation on nuPlan, Mosaic achieves 95.48 CLS-NR and 93.98 CLS-R on the Val14 closed-loop benchmark, setting a new state of the art, while reducing at-fault collisions by 30% compared to either planner in isolation. On the interPlan benchmark, focused on highly interactive and difficult scenarios, Mosaic scores 54.30 CLS-R, outperforming its best constituent planner by 23.3% - all without retraining or requiring additional data. The code is available at github.com/KIT-MRT/mosaic. |
| 2026-04-15 | [Robust Reward Modeling for Large Language Models via Causal Decomposition](http://arxiv.org/abs/2604.13833v1) | Yunsheng Lu, Zijiang Yang et al. | Reward models are central to aligning large language models, yet they often overfit to spurious cues such as response length and overly agreeable tone. Most prior work weakens these cues directly by penalizing or controlling specific artifacts, but it does not explicitly encourage the model to ground preferences in the prompt's intent. We learn a decoder that maps a candidate answer to the latent intent embedding of the input. The reconstruction error is used as a signal to regularize the reward model training. We provide theoretical evidence that this signal emphasizes prompt-dependent information while suppressing prompt-independent shortcuts. Across math, helpfulness, and safety benchmarks, the decoder selects shorter and less sycophantic candidates with 0.877 accuracy. Incorporating this signal into RM training in Gemma-2-2B-it and Gemma-2-9B-it increases RewardBench accuracy from 0.832 to 0.868. For Best-of-N selection, our framework increases length-controlled win rates while producing shorter outputs, and remains robust to lengthening and mild off-topic drift in controlled rewrite tests. |
| 2026-04-15 | [Gaslight, Gatekeep, V1-V3: Early Visual Cortex Alignment Shields Vision-Language Models from Sycophantic Manipulation](http://arxiv.org/abs/2604.13803v1) | Arya Shah, Vaibhav Tripathi et al. | Vision-language models are increasingly deployed in high-stakes settings, yet their susceptibility to sycophantic manipulation remains poorly understood, particularly in relation to how these models represent visual information internally. Whether models whose visual representations more closely mirror human neural processing are also more resistant to adversarial pressure is an open question with implications for both neuroscience and AI safety. We investigate this question by evaluating 12 open-weight vision-language models spanning 6 architecture families and a 40$\times$ parameter range (256M--10B) along two axes: brain alignment, measured by predicting fMRI responses from the Natural Scenes Dataset across 8 human subjects and 6 visual cortex regions of interest, and sycophancy, measured through 76,800 two-turn gaslighting prompts spanning 5 categories and 10 difficulty levels. Region-of-interest analysis reveals that alignment specifically in early visual cortex (V1--V3) is a reliable negative predictor of sycophancy ($r = -0.441$, BCa 95\% CI $[-0.740, -0.031]$), with all 12 leave-one-out correlations negative and the strongest effect for existence denial attacks ($r = -0.597$, $p = 0.040$). This anatomically specific relationship is absent in higher-order category-selective regions, suggesting that faithful low-level visual encoding provides a measurable anchor against adversarial linguistic override in vision-language models. We release our code on \href{https://github.com/aryashah2k/Gaslight-Gatekeep-Sycophantic-Manipulation}{GitHub} and dataset on \href{https://huggingface.co/datasets/aryashah00/Gaslight-Gatekeep-V1-V3}{Hugging Face} |
| 2026-04-15 | [Orthogonal Transformations for Efficient Data-Driven Reachability Analysis](http://arxiv.org/abs/2604.13792v1) | Peng Xie, Amr Alanwar | Data-driven reachability analysis using matrix zonotopes faces a fundamental challenge: the number of generators in the reachable set grows exponentially during propagation, while current order reduction yields overly conservative approximations in data-driven settings. This paper introduces an orthogonal matrix-based framework that appropriately transfers the coordinate system before reducing the generators of the reachable set, dramatically reducing reachable set volumes. By exploiting the factorized structure of data-driven matrix zonotope generators, we develop several efficient algorithms to solve the problem. Numerical experiments demonstrate order-of-magnitude volume reductions compared to traditional methods, while maintaining comparable generator numbers. Our method provides a practical solution to improve precision in data-driven safety verification. |
| 2026-04-15 | [Rethinking AI Hardware: A Three-Layer Cognitive Architecture for Autonomous Agents](http://arxiv.org/abs/2604.13757v1) | Li Chen | The next generation of autonomous AI systems will be constrained not only by model capability, but by how intelligence is structured across heterogeneous hardware. Current paradigms -- cloud-centric AI, on-device inference, and edge-cloud pipelines -- treat planning, reasoning, and execution as a monolithic process, leading to unnecessary latency, energy consumption, and fragmented behavioral continuity. We introduce the Tri-Spirit Architecture, a three-layer cognitive framework that decomposes intelligence into planning (Super Layer), reasoning (Agent Layer), and execution (Reflex Layer), each mapped to distinct compute substrates and coordinated via an asynchronous message bus. We formalize the system with a parameterized routing policy, a habit-compilation mechanism that promotes repeated reasoning paths into zero-inference execution policies, a convergent memory model, and explicit safety constraints. We evaluate the architecture in a reproducible simulation of 2000 synthetic tasks against cloud-centric and edge-only baselines. Tri-Spirit reduces mean task latency by 75.6 percent and energy consumption by 71.1 percent, while decreasing LLM invocations by 30 percent and enabling 77.6 percent offline task completion. These results suggest that cognitive decomposition, rather than model scaling alone, is a primary driver of system-level efficiency in AI hardware. |
| 2026-04-15 | [Homotopy-Guided Potential Games for Congestion-Aware Navigation](http://arxiv.org/abs/2604.13708v1) | Mohammed Irshadh Ismaaeel Sathyamangalam Imran, Lasse Peters et al. | We address the multi-agent motion planning problem where interactions, collisions, and congestion co-exist. Conventional game-theoretic planners capture interactions among agents but often converge to conservative, congested equilibria. Homotopy planners, on the other hand, can explore topologically distinct paths, but lack mechanisms to account for the interdependence of agents' future actions. We propose a unified framework that leverages homotopy classes as structured strategy sets within a receding-horizon setup. At each planning stage, a deterministic homotopy planner generates topologically distinct paths for each agent, conditioned on the joint configuration. To avoid intractable growth of candidate paths, we propose a simple heuristic filtering step that selects a top-$K$ subset of the most suitable congestion-free joint strategies to ensure computational tractability. These serve as initializations for a potential game that enforces homotopy-consistent constraints and yields a generalized open-loop Nash equilibrium (OLNE), with penalties discouraging abrupt strategy shifts in a receding-horizon setting. Simulations with three agents demonstrate improved efficiency (faster completion) and enhanced safety (greater inter-agent clearance, leading to reduced congestion) compared to a local baseline and NH-ORCA that do not reason about homotopies. Hardware trials with two robots and one human demonstrate robustness to irrational behaviors, where our method adapts by switching to alternative feasible equilibria while the baseline game fails. |
| 2026-04-15 | [Towards Autonomous Driving with Short-Packet Rate Splitting: Age of Information Analysis and Optimization](http://arxiv.org/abs/2604.13691v1) | Zirui Zheng, Yingyang Chen et al. | To address the high mobility impacts and the ultra-reliable and low-latency communication (URLLC) requirements in autonomous driving scenarios, rate-splitting multiple access (RSMA) combined with short-packet communication (SPC) emerges as a promising solution.Autonomous vehicles rely on real-time information exchange to ensure safety and coordination, making information freshness essential.By jointly capturing transmission delays and packet errors, age of information (AoI) serves as a comprehensive metric for freshness.In this paper, we investigate short-packet rate splitting to enhance information freshness measured by the AoI.By splitting the unicast messages into common and private parts, encoding all common parts together with the multicast message into a common stream, and encoding each private part into a private stream, RSMA effectively manages interference and enables achieving lower AoI.By considering critical factors such as transmit power, vehicle velocity, blocklength, and the number of transmit antennas, we derive closed-form expressions for the average AoI (AAoI) of the common stream under partial decoding and the overall AAoI under complete decoding.To enhance the AAoI performance, we propose the multi-start two-step successive convex approximation (SCA) algorithm.This algorithm first optimizes the power allocation and subsequently optimizes the rate splitting under the quality of service (QoS) trade-off constraint.Simulation results demonstrate that our short-packet rate-splitting scheme significantly improves the AAoI performance while ensuring system fairness and enabling ultra-low AAoI through the common stream, meeting the requirements of autonomous driving applications.Moreover, the trade-off between the common and overall performance is revealed, indicating that the overall performance can be further enhanced while maintaining the advantages of the common stream. |
| 2026-04-15 | [Empirical Prediction of Pedestrian Comfort in Mobile Robot Pedestrian Encounters](http://arxiv.org/abs/2604.13677v1) | Alireza Jafari, Hong-Son Nguyen et al. | Mobile robots joining public spaces like sidewalks must care for pedestrian comfort. Many studies consider pedestrians' objective safety, for example, by developing collision avoidance algorithms, but not enough studies take the pedestrian's subjective safety or comfort into consideration. Quantifying comfort is a major challenge that hinders mobile robots from understanding and responding to human emotions. We empirically look into the relationship between the mobile robot-pedestrian interaction kinematics and subjective comfort. We perform one-on-one experimental trials, each involving a mobile robot and a volunteer. Statistical analysis of pedestrians' reported comfort versus the kinematic variables shows moderate but significant correlations for most variables. Based on these empirical findings, we design three comfort estimators/predictors derived from the minimum distance, the minimum projected time-to-collision, and a composite estimator. The composite estimator employs all studied kinematic variables and reaches the highest prediction rate and classifying performance among the predictors. The composite predictor has an odds ratio of 3.67. In simple terms, when it identifies a pedestrian as comfortable, it is almost 4 times more likely that the pedestrian is comfortable rather than uncomfortable. The study provides a comfort quantifier for incorporating pedestrian feelings into path planners for more socially compliant robots. |
| 2026-04-15 | [A Bayesian Framework for Uncertainty-Aware Explanations in Power Quality Disturbance Classification](http://arxiv.org/abs/2604.13658v1) | Yinsong Chen, Samson S. Yu et al. | Advanced deep learning methods have shown remarkable success in power quality disturbance (PQD) classification. To enhance model transparency, explainable AI (XAI) techniques have been developed to provide instance-specific interpretations of classifier decisions. However, conventional XAI methods yield deterministic explanations, overlooking uncertainty and limiting reliability in safety-critical applications. This paper proposes a Bayesian explanation framework that models explanation uncertainty by generating a relevance attribution distribution for each instance. This method allows experts to select explanations based on confidence percentiles, thereby tailoring interpretability according to specific disturbance types. Extensive experiments on synthetic and real-world power quality datasets demonstrate that the proposed framework improves the transparency and reliability of PQD classifiers through uncertainty-aware explanations. |

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



