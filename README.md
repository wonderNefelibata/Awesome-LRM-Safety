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
| 2025-05-09 | [LLMs Outperform Experts on Challenging Biology Benchmarks](http://arxiv.org/abs/2505.06108v1) | Lennart Justen | This study systematically evaluates 27 frontier Large Language Models on eight diverse biology benchmarks spanning molecular biology, genetics, cloning, virology, and biosecurity. Models from major AI developers released between November 2022 and April 2025 were assessed through ten independent runs per benchmark. The findings reveal dramatic improvements in biological capabilities. Top model performance increased more than 4-fold on the challenging text-only subset of the Virology Capabilities Test over the study period, with the top model now performing twice as well as expert virologists. Several models now match or exceed expert-level performance on other challenging benchmarks, including LAB-Bench CloningScenarios and the biology subsets of GPQA and WMDP. Contrary to expectations, chain-of-thought did not substantially improve performance over zero-shot evaluation, while extended reasoning features in o3-mini and Claude 3.7 Sonnet typically improved performance as predicted by inference scaling. Benchmarks such as PubMedQA and the MMLU and WMDP biology subsets exhibited performance plateaus well below 100%, suggesting benchmark saturation and errors in the underlying benchmark data. The analysis highlights the need for more sophisticated evaluation methodologies as AI systems continue to advance. |
| 2025-05-09 | [Safe-EF: Error Feedback for Nonsmooth Constrained Optimization](http://arxiv.org/abs/2505.06053v1) | Rustem Islamov, Yarden As et al. | Federated learning faces severe communication bottlenecks due to the high dimensionality of model updates. Communication compression with contractive compressors (e.g., Top-K) is often preferable in practice but can degrade performance without proper handling. Error feedback (EF) mitigates such issues but has been largely restricted for smooth, unconstrained problems, limiting its real-world applicability where non-smooth objectives and safety constraints are critical. We advance our understanding of EF in the canonical non-smooth convex setting by establishing new lower complexity bounds for first-order algorithms with contractive compression. Next, we propose Safe-EF, a novel algorithm that matches our lower bound (up to a constant) while enforcing safety constraints essential for practical applications. Extending our approach to the stochastic setting, we bridge the gap between theory and practical implementation. Extensive experiments in a reinforcement learning setup, simulating distributed humanoid robot training, validate the effectiveness of Safe-EF in ensuring safety and reducing communication complexity. |
| 2025-05-09 | [Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information](http://arxiv.org/abs/2505.06046v1) | Joshua Harris, Fan Grayson et al. | As Large Language Models (LLMs) become widely accessible, a detailed understanding of their knowledge within specific domains becomes necessary for successful real world use. This is particularly critical in public health, where failure to retrieve relevant, accurate, and current information could significantly impact UK residents. However, currently little is known about LLM knowledge of UK Government public health information. To address this issue, this paper introduces a new benchmark, PubHealthBench, with over 8000 questions for evaluating LLMs' Multiple Choice Question Answering (MCQA) and free form responses to public health queries, created via an automated pipeline. We also release a new dataset of the extracted UK Government public health guidance documents used as source text for PubHealthBench. Assessing 24 LLMs on PubHealthBench we find the latest private LLMs (GPT-4.5, GPT-4.1 and o1) have a high degree of knowledge, achieving >90% in the MCQA setup, and outperform humans with cursory search engine use. However, in the free form setup we see lower performance with no model scoring >75%. Therefore, whilst there are promising signs that state of the art (SOTA) LLMs are an increasingly accurate source of public health information, additional safeguards or tools may still be needed when providing free form responses on public health topics. |
| 2025-05-09 | [Priority-Driven Safe Model Predictive Control Approach to Autonomous Driving Applications](http://arxiv.org/abs/2505.05933v1) | Francesco Prignoli, Ying Shuai Quan et al. | This paper demonstrates the applicability of the safe model predictive control (SMPC) framework to autonomous driving scenarios, focusing on the design of adaptive cruise control (ACC) and automated lane-change systems. Building on the SMPC approach with priority-driven constraint softening -- which ensures the satisfaction of \emph{hard} constraints under external disturbances by selectively softening a predefined subset of adjustable constraints -- we show how the algorithm dynamically relaxes lower-priority, comfort-related constraints in response to unexpected disturbances while preserving critical safety requirements such as collision avoidance and lane-keeping. A learning-based algorithm approximating the time consuming SMPC is introduced to enable real-time execution. Simulations in real-world driving scenarios subject to unpredicted disturbances confirm that this prioritized softening mechanism consistently upholds stringent safety constraints, underscoring the effectiveness of the proposed method. |
| 2025-05-09 | [Human causal perception in a cube-stacking task](http://arxiv.org/abs/2505.05923v1) | Nikolai Bahr, Christoph Zetzsche et al. | In intuitive physics the process of stacking cubes has become a paradigmatic, canonical task. Even though it gets employed in various shades and complexities, the very fundamental setting with two cubes has not been thoroughly investigated. Furthermore, the majority of settings feature only a reduced, one dimensional (1D) decision space. In this paper an experiment is conducted in which participants judge the stability of two cubes stacked on top of each other. It is performed in the full 3D setting which features a 2D decision surface. The analysis yield a shape of a rotated square for the perceived stability area instead of the commonly reported safety margin in 1D. This implies a more complex decision behavior in human than previously assumed. |
| 2025-05-09 | [AgentXploit: End-to-End Redteaming of Black-Box AI Agents](http://arxiv.org/abs/2505.05849v1) | Zhun Wang, Vincent Siu et al. | The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentXploit, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentXploit on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentXploit exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites. |
| 2025-05-09 | [Exploring Dense Crowd Dynamics: State of the Art and Emerging Paradigms](http://arxiv.org/abs/2505.05826v1) | Thomas Chatagnon, Antoine Tordeux et al. | Dense pedestrian crowds may pose significant safety risks, yet their underlying dynamics remain insufficiently understood to reliably prevent accidents. In these environments, physical interactions and contact forces fundamentally shape the dynamics of the crowd. However, accurately describing these interindividual interactions requires specific modeling and analytical approaches. This chapter reviews paradigms and models used to represent pedestrian dynamics in various contexts, highlighting the transition from classical approaches to models tailored for dense crowd conditions. We argue that further investigation is needed, featuring new experimental studies and new modeling paradigms, to better capture the complex dynamics that emerge in high-density situations. |
| 2025-05-09 | [Unsupervised Anomaly Detection for Autonomous Robots via Mahalanobis SVDD with Audio-IMU Fusion](http://arxiv.org/abs/2505.05811v1) | Yizhuo Yang, Jiulin Zhao et al. | Reliable anomaly detection is essential for ensuring the safety of autonomous robots, particularly when conventional detection systems based on vision or LiDAR become unreliable in adverse or unpredictable conditions. In such scenarios, alternative sensing modalities are needed to provide timely and robust feedback. To this end, we explore the use of audio and inertial measurement unit (IMU) sensors to detect underlying anomalies in autonomous mobile robots, such as collisions and internal mechanical faults. Furthermore, to address the challenge of limited labeled anomaly data, we propose an unsupervised anomaly detection framework based on Mahalanobis Support Vector Data Description (M-SVDD). In contrast to conventional SVDD methods that rely on Euclidean distance and assume isotropic feature distributions, our approach employs the Mahalanobis distance to adaptively scale feature dimensions and capture inter-feature correlations, enabling more expressive decision boundaries. In addition, a reconstruction-based auxiliary branch is introduced to preserve feature diversity and prevent representation collapse, further enhancing the robustness of anomaly detection. Extensive experiments on a collected mobile robot dataset and four public datasets demonstrate the effectiveness of the proposed method, as shown in the video https://youtu.be/yh1tn6DDD4A. Code and dataset are available at https://github.com/jamesyang7/M-SVDD. |
| 2025-05-09 | [APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning](http://arxiv.org/abs/2505.05758v1) | Azim Ospanov, Roozbeh Yousefzadeh | Formal reasoning and automated theorem proving constitute a challenging subfield of machine learning, in which machines are tasked with proving mathematical theorems using formal languages like Lean. A formal verification system can check whether a formal proof is correct or not almost instantaneously, but generating a completely correct formal proof with large language models (LLMs) remains a formidable task. The usual approach in the literature is to prompt the LLM many times (up to several thousands) until one of the generated proofs passes the verification system. In this work, we present APOLLO (Automated PrOof repair via LLM and Lean cOllaboration), a modular, model-agnostic pipeline that combines the strengths of the Lean compiler with an LLM's reasoning abilities to achieve better proof-generation results at a low sampling budget. Apollo directs a fully automated process in which the LLM generates proofs for theorems, a set of agents analyze the proofs, fix the syntax errors, identify the mistakes in the proofs using Lean, isolate failing sub-lemmas, utilize automated solvers, and invoke an LLM on each remaining goal with a low top-K budget. The repaired sub-proofs are recombined and reverified, iterating up to a user-controlled maximum number of attempts. On the miniF2F benchmark, we establish a new state-of-the-art accuracy of 75.0% among 7B-parameter models while keeping the sampling budget below one thousand. Moreover, Apollo raises the state-of-the-art accuracy for Goedel-Prover-SFT to 65.6% while cutting sample complexity from 25,600 to a few hundred. General-purpose models (o3-mini, o4-mini) jump from 3-7% to over 40% accuracy. Our results demonstrate that targeted, compiler-guided repair of LLM outputs yields dramatic gains in both efficiency and correctness, suggesting a general paradigm for scalable automated theorem proving. |
| 2025-05-09 | [Efficient Full-Stack Private Federated Deep Learning with Post-Quantum Security](http://arxiv.org/abs/2505.05751v1) | Yiwei Zhang, Rouzbeh Behnia et al. | Federated learning (FL) enables collaborative model training while preserving user data privacy by keeping data local. Despite these advantages, FL remains vulnerable to privacy attacks on user updates and model parameters during training and deployment. Secure aggregation protocols have been proposed to protect user updates by encrypting them, but these methods often incur high computational costs and are not resistant to quantum computers. Additionally, differential privacy (DP) has been used to mitigate privacy leakages, but existing methods focus on secure aggregation or DP, neglecting their potential synergies. To address these gaps, we introduce Beskar, a novel framework that provides post-quantum secure aggregation, optimizes computational overhead for FL settings, and defines a comprehensive threat model that accounts for a wide spectrum of adversaries. We also integrate DP into different stages of FL training to enhance privacy protection in diverse scenarios. Our framework provides a detailed analysis of the trade-offs between security, performance, and model accuracy, representing the first thorough examination of secure aggregation protocols combined with various DP approaches for post-quantum secure FL. Beskar aims to address the pressing privacy and security issues FL while ensuring quantum-safety and robust performance. |
| 2025-05-08 | [Adaptive Stress Testing Black-Box LLM Planners](http://arxiv.org/abs/2505.05665v1) | Neeloy Chakraborty, John Pohovey et al. | Large language models (LLMs) have recently demonstrated success in generalizing across decision-making tasks including planning, control and prediction, but their tendency to hallucinate unsafe and undesired outputs poses risks. We argue that detecting such failures is necessary, especially in safety-critical scenarios. Existing black-box methods often detect hallucinations by identifying inconsistencies across multiple samples. Many of these approaches typically introduce prompt perturbations like randomizing detail order or generating adversarial inputs, with the intuition that a confident model should produce stable outputs. We first perform a manual case study showing that other forms of perturbations (e.g., adding noise, removing sensor details) cause LLMs to hallucinate in a driving environment. We then propose a novel method for efficiently searching the space of prompt perturbations using Adaptive Stress Testing (AST) with Monte-Carlo Tree Search (MCTS). Our AST formulation enables discovery of scenarios and prompts that cause language models to act with high uncertainty. By generating MCTS prompt perturbation trees across diverse scenarios, we show that offline analyses can be used at runtime to automatically generate prompts that influence model uncertainty, and to inform real-time trust assessments of an LLM. |
| 2025-05-08 | [UltraGauss: Ultrafast Gaussian Reconstruction of 3D Ultrasound Volumes](http://arxiv.org/abs/2505.05643v1) | Mark C. Eid, Ana I. L. Namburete et al. | Ultrasound imaging is widely used due to its safety, affordability, and real-time capabilities, but its 2D interpretation is highly operator-dependent, leading to variability and increased cognitive demand. 2D-to-3D reconstruction mitigates these challenges by providing standardized volumetric views, yet existing methods are often computationally expensive, memory-intensive, or incompatible with ultrasound physics. We introduce UltraGauss: the first ultrasound-specific Gaussian Splatting framework, extending view synthesis techniques to ultrasound wave propagation. Unlike conventional perspective-based splatting, UltraGauss models probe-plane intersections in 3D, aligning with acoustic image formation. We derive an efficient rasterization boundary formulation for GPU parallelization and introduce a numerically stable covariance parametrization, improving computational efficiency and reconstruction accuracy. On real clinical ultrasound data, UltraGauss achieves state-of-the-art reconstructions in 5 minutes, and reaching 0.99 SSIM within 20 minutes on a single GPU. A survey of expert clinicians confirms UltraGauss' reconstructions are the most realistic among competing methods. Our CUDA implementation will be released upon publication. |
| 2025-05-08 | [LiteLMGuard: Seamless and Lightweight On-Device Prompt Filtering for Safeguarding Small Language Models against Quantization-induced Risks and Vulnerabilities](http://arxiv.org/abs/2505.05619v1) | Kalyan Nakka, Jimmy Dani et al. | The growing adoption of Large Language Models (LLMs) has influenced the development of their lighter counterparts-Small Language Models (SLMs)-to enable on-device deployment across smartphones and edge devices. These SLMs offer enhanced privacy, reduced latency, server-free functionality, and improved user experience. However, due to resource constraints of on-device environment, SLMs undergo size optimization through compression techniques like quantization, which can inadvertently introduce fairness, ethical and privacy risks. Critically, quantized SLMs may respond to harmful queries directly, without requiring adversarial manipulation, raising significant safety and trust concerns.   To address this, we propose LiteLMGuard (LLMG), an on-device prompt guard that provides real-time, prompt-level defense for quantized SLMs. Additionally, our prompt guard is designed to be model-agnostic such that it can be seamlessly integrated with any SLM, operating independently of underlying architectures. Our LLMG formalizes prompt filtering as a deep learning (DL)-based prompt answerability classification task, leveraging semantic understanding to determine whether a query should be answered by any SLM. Using our curated dataset, Answerable-or-Not, we trained and fine-tuned several DL models and selected ELECTRA as the candidate, with 97.75% answerability classification accuracy.   Our safety effectiveness evaluations demonstrate that LLMG defends against over 87% of harmful prompts, including both direct instruction and jailbreak attack strategies. We further showcase its ability to mitigate the Open Knowledge Attacks, where compromised SLMs provide unsafe responses without adversarial prompting. In terms of prompt filtering effectiveness, LLMG achieves near state-of-the-art filtering accuracy of 94%, with an average latency of 135 ms, incurring negligible overhead for users. |
| 2025-05-08 | [Barrier Function Overrides For Non-Convex Fixed Wing Flight Control and Self-Driving Cars](http://arxiv.org/abs/2505.05548v1) | Eric Squires, Phillip Odom et al. | Reinforcement Learning (RL) has enabled vast performance improvements for robotics systems. To achieve these results though, the agent often must randomly explore the environment, which for safety critical systems presents a significant challenge. Barrier functions can solve this challenge by enabling an override that approximates the RL control input as closely as possible without violating a safety constraint. Unfortunately, this override can be computationally intractable in cases where the dynamics are not convex in the control input or when time is discrete, as is often the case when training RL systems. We therefore consider these cases, developing novel barrier functions for two non-convex systems (fixed wing aircraft and self-driving cars performing lane merging with adaptive cruise control) in discrete time. Although solving for an online and optimal override is in general intractable when the dynamics are nonconvex in the control input, we investigate approximate solutions, finding that these approximations enable performance commensurate with baseline RL methods with zero safety violations. In particular, even without attempting to solve for the optimal override at all, performance is still competitive with baseline RL performance. We discuss the tradeoffs of the approximate override solutions including performance and computational tractability. |
| 2025-05-08 | [Safety by Measurement: A Systematic Literature Review of AI Safety Evaluation Methods](http://arxiv.org/abs/2505.05541v1) | Markov Grey, Charbel-Raphaël Segerie | As frontier AI systems advance toward transformative capabilities, we need a parallel transformation in how we measure and evaluate these systems to ensure safety and inform governance. While benchmarks have been the primary method for estimating model capabilities, they often fail to establish true upper bounds or predict deployment behavior. This literature review consolidates the rapidly evolving field of AI safety evaluations, proposing a systematic taxonomy around three dimensions: what properties we measure, how we measure them, and how these measurements integrate into frameworks. We show how evaluations go beyond benchmarks by measuring what models can do when pushed to the limit (capabilities), the behavioral tendencies exhibited by default (propensities), and whether our safety measures remain effective even when faced with subversive adversarial AI (control). These properties are measured through behavioral techniques like scaffolding, red teaming and supervised fine-tuning, alongside internal techniques such as representation analysis and mechanistic interpretability. We provide deeper explanations of some safety-critical capabilities like cybersecurity exploitation, deception, autonomous replication, and situational awareness, alongside concerning propensities like power-seeking and scheming. The review explores how these evaluation methods integrate into governance frameworks to translate results into concrete development decisions. We also highlight challenges to safety evaluations - proving absence of capabilities, potential model sandbagging, and incentives for "safetywashing" - while identifying promising research directions. By synthesizing scattered resources, this literature review aims to provide a central reference point for understanding AI safety evaluations. |
| 2025-05-08 | [Reasoning Models Don't Always Say What They Think](http://arxiv.org/abs/2505.05410v1) | Yanda Chen, Joe Benton et al. | Chain-of-thought (CoT) offers a potential boon for AI safety as it allows monitoring a model's CoT to try to understand its intentions and reasoning processes. However, the effectiveness of such monitoring hinges on CoTs faithfully representing models' actual reasoning processes. We evaluate CoT faithfulness of state-of-the-art reasoning models across 6 reasoning hints presented in the prompts and find: (1) for most settings and models tested, CoTs reveal their usage of hints in at least 1% of examples where they use the hint, but the reveal rate is often below 20%, (2) outcome-based reinforcement learning initially improves faithfulness but plateaus without saturating, and (3) when reinforcement learning increases how frequently hints are used (reward hacking), the propensity to verbalize them does not increase, even without training against a CoT monitor. These results suggest that CoT monitoring is a promising way of noticing undesired behaviors during training and evaluations, but that it is not sufficient to rule them out. They also suggest that in settings like ours where CoT reasoning is not necessary, test-time monitoring of CoTs is unlikely to reliably catch rare and catastrophic unexpected behaviors. |
| 2025-05-08 | [PillarMamba: Learning Local-Global Context for Roadside Point Cloud via Hybrid State Space Model](http://arxiv.org/abs/2505.05397v1) | Zhang Zhang, Chao Sun et al. | Serving the Intelligent Transport System (ITS) and Vehicle-to-Everything (V2X) tasks, roadside perception has received increasing attention in recent years, as it can extend the perception range of connected vehicles and improve traffic safety. However, roadside point cloud oriented 3D object detection has not been effectively explored. To some extent, the key to the performance of a point cloud detector lies in the receptive field of the network and the ability to effectively utilize the scene context. The recent emergence of Mamba, based on State Space Model (SSM), has shaken up the traditional convolution and transformers that have long been the foundational building blocks, due to its efficient global receptive field. In this work, we introduce Mamba to pillar-based roadside point cloud perception and propose a framework based on Cross-stage State-space Group (CSG), called PillarMamba. It enhances the expressiveness of the network and achieves efficient computation through cross-stage feature fusion. However, due to the limitations of scan directions, state space model faces local connection disrupted and historical relationship forgotten. To address this, we propose the Hybrid State-space Block (HSB) to obtain the local-global context of roadside point cloud. Specifically, it enhances neighborhood connections through local convolution and preserves historical memory through residual attention. The proposed method outperforms the state-of-the-art methods on the popular large scale roadside benchmark: DAIR-V2X-I. The code will be released soon. |
| 2025-05-08 | [Robust Online Learning with Private Information](http://arxiv.org/abs/2505.05341v1) | Kyohei Okumura | This paper investigates the robustness of online learning algorithms when learners possess private information. No-external-regret algorithms, prevalent in machine learning, are vulnerable to strategic manipulation, allowing an adaptive opponent to extract full surplus. Even standard no-weak-external-regret algorithms, designed for optimal learning in stationary environments, exhibit similar vulnerabilities. This raises a fundamental question: can a learner simultaneously prevent full surplus extraction by adaptive opponents while maintaining optimal performance in well-behaved environments? To address this, we model the problem as a two-player repeated game, where the learner with private information plays against the environment, facing ambiguity about the environment's types: stationary or adaptive. We introduce \emph{partial safety} as a key design criterion for online learning algorithms to prevent full surplus extraction. We then propose the \emph{Explore-Exploit-Punish} (\textsf{EEP}) algorithm and prove that it satisfies partial safety while achieving optimal learning in stationary environments, and has a variant that delivers improved welfare performance. Our findings highlight the risks of applying standard online learning algorithms in strategic settings with adverse selection. We advocate for a shift toward online learning algorithms that explicitly incorporate safeguards against strategic manipulation while ensuring strong learning performance. |
| 2025-05-08 | [Advancing Neural Network Verification through Hierarchical Safety Abstract Interpretation](http://arxiv.org/abs/2505.05235v1) | Luca Marzari, Isabella Mastroeni et al. | Traditional methods for formal verification (FV) of deep neural networks (DNNs) are constrained by a binary encoding of safety properties, where a model is classified as either safe or unsafe (robust or not robust). This binary encoding fails to capture the nuanced safety levels within a model, often resulting in either overly restrictive or too permissive requirements. In this paper, we introduce a novel problem formulation called Abstract DNN-Verification, which verifies a hierarchical structure of unsafe outputs, providing a more granular analysis of the safety aspect for a given DNN. Crucially, by leveraging abstract interpretation and reasoning about output reachable sets, our approach enables assessing multiple safety levels during the FV process, requiring the same (in the worst case) or even potentially less computational effort than the traditional binary verification approach. Specifically, we demonstrate how this formulation allows rank adversarial inputs according to their abstract safety level violation, offering a more detailed evaluation of the model's safety and robustness. Our contributions include a theoretical exploration of the relationship between our novel abstract safety formulation and existing approaches that employ abstract interpretation for robustness verification, complexity analysis of the novel problem introduced, and an empirical evaluation considering both a complex deep reinforcement learning task (based on Habitat 3.0) and standard DNN-Verification benchmarks. |
| 2025-05-08 | [PaniCar: Securing the Perception of Advanced Driving Assistance Systems Against Emergency Vehicle Lighting](http://arxiv.org/abs/2505.05183v1) | Elad Feldman, Jacob Shams et al. | The safety of autonomous cars has come under scrutiny in recent years, especially after 16 documented incidents involving Teslas (with autopilot engaged) crashing into parked emergency vehicles (police cars, ambulances, and firetrucks). While previous studies have revealed that strong light sources often introduce flare artifacts in the captured image, which degrade the image quality, the impact of flare on object detection performance remains unclear. In this research, we unveil PaniCar, a digital phenomenon that causes an object detector's confidence score to fluctuate below detection thresholds when exposed to activated emergency vehicle lighting. This vulnerability poses a significant safety risk, and can cause autonomous vehicles to fail to detect objects near emergency vehicles. In addition, this vulnerability could be exploited by adversaries to compromise the security of advanced driving assistance systems (ADASs). We assess seven commercial ADASs (Tesla Model 3, "manufacturer C", HP, Pelsee, AZDOME, Imagebon, Rexing), four object detectors (YOLO, SSD, RetinaNet, Faster R-CNN), and 14 patterns of emergency vehicle lighting to understand the influence of various technical and environmental factors. We also evaluate four SOTA flare removal methods and show that their performance and latency are insufficient for real-time driving constraints. To mitigate this risk, we propose Caracetamol, a robust framework designed to enhance the resilience of object detectors against the effects of activated emergency vehicle lighting. Our evaluation shows that on YOLOv3 and Faster RCNN, Caracetamol improves the models' average confidence of car detection by 0.20, the lower confidence bound by 0.33, and reduces the fluctuation range by 0.33. In addition, Caracetamol is capable of processing frames at a rate of between 30-50 FPS, enabling real-time ADAS car detection. |

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



