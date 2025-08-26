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
| 2025-08-25 | [SafeBimanual: Diffusion-based Trajectory Optimization for Safe Bimanual Manipulation](http://arxiv.org/abs/2508.18268v1) | Haoyuan Deng, Wenkai Guo et al. | Bimanual manipulation has been widely applied in household services and manufacturing, which enables the complex task completion with coordination requirements. Recent diffusion-based policy learning approaches have achieved promising performance in modeling action distributions for bimanual manipulation. However, they ignored the physical safety constraints of bimanual manipulation, which leads to the dangerous behaviors with damage to robots and objects. To this end, we propose a test-time trajectory optimization framework named SafeBimanual for any pre-trained diffusion-based bimanual manipulation policies, which imposes the safety constraints on bimanual actions to avoid dangerous robot behaviors with improved success rate. Specifically, we design diverse cost functions for safety constraints in different dual-arm cooperation patterns including avoidance of tearing objects and collision between arms and objects, which optimizes the manipulator trajectories with guided sampling of diffusion denoising process. Moreover, we employ a vision-language model (VLM) to schedule the cost functions by specifying keypoints and corresponding pairwise relationship, so that the optimal safety constraint is dynamically generated in the entire bimanual manipulation process. SafeBimanual demonstrates superiority on 8 simulated tasks in RoboTwin with a 13.7% increase in success rate and a 18.8% reduction in unsafe interactions over state-of-the-art diffusion-based methods. Extensive experiments on 4 real-world tasks further verify its practical value by improving the success rate by 32.5%. |
| 2025-08-25 | [Caregiver-in-the-Loop AI: A Simulation-Based Feasibility Study for Dementia Task Verification](http://arxiv.org/abs/2508.18267v1) | Joy Lai, David Black et al. | Caregivers of people living with dementia (PLwD) experience stress when verifying whether tasks are truly completed, even with digital reminder systems. Generative AI, such as GPT-4, may help by automating task verification through follow-up questioning and decision support.   This feasibility study evaluates an AI-powered task verification system integrated with digital reminders for PLwD. It examines (1) GPT-4's ability to generate effective follow-up questions, (2) the accuracy of an AI-driven response flagging mechanism, and (3) the role of caregiver feedback in refining system adaptability. A simulated pipeline was tested on 64 anonymized reminders. GPT-4 generated follow-up questions with and without contextual information about PLwD routines. Responses were classified into High, Medium, or Low concern, and simulated caregiver feedback was used to refine outputs.   Results show that contextual information and caregiver input improved the clarity and relevance of AI-generated questions. The flagging system accurately identified concerns, particularly for safety-critical tasks, though subjective or non-urgent tasks remained challenging. Findings demonstrate the feasibility of AI-assisted task verification in dementia care. Context-aware AI prompts and caregiver feedback can enhance task monitoring, reduce caregiver stress, and strengthen PLwD support. Future work should focus on real-world validation and scalability. |
| 2025-08-25 | [MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](http://arxiv.org/abs/2508.18260v1) | Kaiwen Wei, Rui Shan et al. | Large reasoning models (LRMs) have shown significant progress in test-time scaling through chain-of-thought prompting. Current approaches like search-o1 integrate retrieval augmented generation (RAG) into multi-step reasoning processes but rely on a single, linear reasoning chain while incorporating unstructured textual information in a flat, context-agnostic manner. As a result, these approaches can lead to error accumulation throughout the reasoning chain, which significantly limits its effectiveness in medical question-answering (QA) tasks where both accuracy and traceability are critical requirements. To address these challenges, we propose MIRAGE (Multi-chain Inference with Retrieval-Augmented Graph Exploration), a novel test-time scalable reasoning framework that performs dynamic multi-chain inference over structured medical knowledge graphs. Specifically, MIRAGE 1) decomposes complex queries into entity-grounded sub-questions, 2) executes parallel inference chains, 3) retrieves evidence adaptively via neighbor expansion and multi-hop traversal, and 4) integrates answers using cross-chain verification to resolve contradictions. Experiments on three medical QA benchmarks (GenMedGPT-5k, CMCQA, and ExplainCPE) show that MIRAGE consistently outperforms GPT-4o, Tree-of-Thought variants, and other retrieval-augmented baselines in both automatic and human evaluations. Additionally, MIRAGE improves interpretability by generating explicit reasoning chains that trace each factual claim to concrete chains within the knowledge graph, making it well-suited for complex medical reasoning scenarios. The code will be available for further research. |
| 2025-08-25 | [Tractable Stochastic Hybrid Model Predictive Control using Gaussian Processes for Repetitive Tasks in Unseen Environments](http://arxiv.org/abs/2508.18203v1) | Leroy D'Souza, Yash Vardhan Pant et al. | Improving the predictive accuracy of a dynamics model is crucial to obtaining good control performance and safety from Model Predictive Controllers (MPC). One approach involves learning unmodelled (residual) dynamics, in addition to nominal models derived from first principles. Varying residual models across an environment manifest as modes of a piecewise residual (PWR) model that requires a) identifying how modes are distributed across the environment and b) solving a computationally intensive Mixed Integer Nonlinear Program (MINLP) problem for control. We develop an iterative mapping algorithm capable of predicting time-varying mode distributions. We then develop and solve two tractable approximations of the MINLP to combine with the predictor in closed-loop to solve the overall control problem. In simulation, we first demonstrate how the approximations improve performance by 4-18% in comparison to the MINLP while achieving significantly lower computation times (upto 250x faster). We then demonstrate how the proposed mapping algorithm incrementally improves controller performance (upto 3x) over multiple iterations of a trajectory tracking control task even when the mode distributions change over time. |
| 2025-08-25 | [Uncertain data assimilation for urban wind flow simulations with OpenLB-UQ](http://arxiv.org/abs/2508.18202v1) | Mingliang Zhong, Dennis Teutscher et al. | Accurate prediction of urban wind flow is essential for urban planning, pedestrian safety, and environmental management. Yet, it remains challenging due to uncertain boundary conditions and the high cost of conventional CFD simulations. This paper presents the use of the modular and efficient uncertainty quantification (UQ) framework OpenLB-UQ for urban wind flow simulations. We specifically use the lattice Boltzmann method (LBM) coupled with a stochastic collocation (SC) approach based on generalized polynomial chaos (gPC). The framework introduces a relative-error noise model for inflow wind speeds based on real measurements. The model is propagated through a non-intrusive SC LBM pipeline using sparse-grid quadrature. Key quantities of interest, including mean flow fields, standard deviations, and vertical profiles with confidence intervals, are efficiently computed without altering the underlying deterministic solver. We demonstrate this on a real urban scenario, highlighting how uncertainty localizes in complex flow regions such as wakes and shear layers. The results show that the SC LBM approach provides accurate, uncertainty-aware predictions with significant computational efficiency, making OpenLB-UQ a practical tool for real-time urban wind analysis. |
| 2025-08-25 | [Analysis and Detection of RIS-based Spoofing in Integrated Sensing and Communication (ISAC)](http://arxiv.org/abs/2508.18100v1) | Tingyu Shui, Po-Heng Chou et al. | Integrated sensing and communication (ISAC) is a key feature of next-generation 6G wireless systems, allowing them to achieve high data rates and sensing accuracy. While prior research has primarily focused on addressing communication safety in ISAC systems, the equally critical issue of sensing safety remains largely under-explored. In this paper, the possibility of spoofing the sensing function of ISAC in vehicle networks is examined, whereby a malicious reconfigurable intelligent surface (RIS) is deployed to compromise the sensing functionality of a roadside unit (RSU). For this scenario, the requirements on the malicious RIS' phase shifts design and number of reflecting elements are analyzed. Under such spoofing, the practical estimation bias of the vehicular user (VU)'s Doppler shift and angle-of-departure (AoD) for an arbitrary time slot is analytically derived. Moreover, from the attacker's view, a Markov decision process (MDP) is formulated to optimize the RIS' phase shifts design. The goal of this MDP is to generate complete and plausible fake trajectories by incorporating the concept of spatial-temporal consistency. To defend against this sensing spoofing attack, a signal temporal logic (STL)-based neuro-symbolic attack detection framework is proposed and shown to learn interoperable formulas for identifying spoofed trajectories. |
| 2025-08-25 | [Neither Valid nor Reliable? Investigating the Use of LLMs as Judges](http://arxiv.org/abs/2508.18076v1) | Khaoula Chehbouni, Mohammed Haddou et al. | Evaluating natural language generation (NLG) systems remains a core challenge of natural language processing (NLP), further complicated by the rise of large language models (LLMs) that aims to be general-purpose. Recently, large language models as judges (LLJs) have emerged as a promising alternative to traditional metrics, but their validity remains underexplored. This position paper argues that the current enthusiasm around LLJs may be premature, as their adoption has outpaced rigorous scrutiny of their reliability and validity as evaluators. Drawing on measurement theory from the social sciences, we identify and critically assess four core assumptions underlying the use of LLJs: their ability to act as proxies for human judgment, their capabilities as evaluators, their scalability, and their cost-effectiveness. We examine how each of these assumptions may be challenged by the inherent limitations of LLMs, LLJs, or current practices in NLG evaluation. To ground our analysis, we explore three applications of LLJs: text summarization, data annotation, and safety alignment. Finally, we highlight the need for more responsible evaluation practices in LLJs evaluation, to ensure that their growing role in the field supports, rather than undermines, progress in NLG. |
| 2025-08-25 | [Anatomy of the gem5 Simulator: AtomicSimpleCPU, TimingSimpleCPU, O3CPU, and Their Interaction with the Ruby Memory System](http://arxiv.org/abs/2508.18043v1) | Johan Söderström, Yuan Yao | gem5 is a popular modular-based computer system simulator, widely used in computer architecture research and known for its long simulation time and steep learning curve. This report examines its three major CPU models: the AtomicSimpleCPU (AS CPU), the TimingSimpleCPU (TS CPU), the Out-of-order (O3) CPU, and their interactions with the memory subsystem. We provide a detailed anatomical overview of each CPU's function call-chains and present how gem5 partitions its execution time for each simulated hardware layer.   We perform our analysis using a lightweight profiler built on Linux's perf_event interface, with user-configurable options to target specific functions and examine their interactions in detail. By profiling each CPU across a wide selection of benchmarks, we identify their software bottlenecks. Our results show that the Ruby memory subsystem consistently accounts for the largest share of execution time in the sequential AS and TS CPUs, primarily during the instruction fetch stage. In contrast, the O3 CPU spends a relatively smaller fraction of time in Ruby, with most of its time devoted to constructing instruction instances and the various pipeline stages of the CPU.   We believe that the anatomical view of each CPU's execution flow is valuable for educational purposes, as it clearly illustrates the interactions among simulated components. These insights form a foundation for optimizing gem5's performance, particularly for the AS, TS, and O3 CPUs. Moreover, our framework can be readily applied to analyze other gem5 components or to develop and evaluate new models. |
| 2025-08-25 | [Integration of Computer Vision with Adaptive Control for Autonomous Driving Using ADORE](http://arxiv.org/abs/2508.17985v1) | Abu Shad Ahammed, Md Shahi Amran Hossain et al. | Ensuring safety in autonomous driving requires a seamless integration of perception and decision making under uncertain conditions. Although computer vision (CV) models such as YOLO achieve high accuracy in detecting traffic signs and obstacles, their performance degrades in drift scenarios caused by weather variations or unseen objects. This work presents a simulated autonomous driving system that combines a context aware CV model with adaptive control using the ADORE framework. The CARLA simulator was integrated with ADORE via the ROS bridge, allowing real-time communication between perception, decision, and control modules. A simulated test case was designed in both clear and drift weather conditions to demonstrate the robust detection performance of the perception model while ADORE successfully adapted vehicle behavior to speed limits and obstacles with low response latency. The findings highlight the potential of coupling deep learning-based perception with rule-based adaptive decision making to improve automotive safety critical system. |
| 2025-08-25 | [Enhanced Drift-Aware Computer Vision Architecture for Autonomous Driving](http://arxiv.org/abs/2508.17975v1) | Md Shahi Amran Hossain, Abu Shad Ahammed et al. | The use of computer vision in automotive is a trending research in which safety and security are a primary concern. In particular, for autonomous driving, preventing road accidents requires highly accurate object detection under diverse conditions. To address this issue, recently the International Organization for Standardization (ISO) released the 8800 norm, providing structured frameworks for managing associated AI relevant risks. However, challenging scenarios such as adverse weather or low lighting often introduce data drift, leading to degraded model performance and potential safety violations. In this work, we present a novel hybrid computer vision architecture trained with thousands of synthetic image data from the road environment to improve robustness in unseen drifted environments. Our dual mode framework utilized YOLO version 8 for swift detection and incorporated a five-layer CNN for verification. The system functioned in sequence and improved the detection accuracy by more than 90\% when tested with drift-augmented road images. The focus was to demonstrate how such a hybrid model can provide better road safety when working together in a hybrid structure. |
| 2025-08-25 | [EndoUFM: Utilizing Foundation Models for Monocular depth estimation of endoscopic images](http://arxiv.org/abs/2508.17916v1) | Xinning Yao, Bo Liu et al. | Depth estimation is a foundational component for 3D reconstruction in minimally invasive endoscopic surgeries. However, existing monocular depth estimation techniques often exhibit limited performance to the varying illumination and complex textures of the surgical environment. While powerful visual foundation models offer a promising solution, their training on natural images leads to significant domain adaptability limitations and semantic perception deficiencies when applied to endoscopy. In this study, we introduce EndoUFM, an unsupervised monocular depth estimation framework that innovatively integrating dual foundation models for surgical scenes, which enhance the depth estimation performance by leveraging the powerful pre-learned priors. The framework features a novel adaptive fine-tuning strategy that incorporates Random Vector Low-Rank Adaptation (RVLoRA) to enhance model adaptability, and a Residual block based on Depthwise Separable Convolution (Res-DSC) to improve the capture of fine-grained local features. Furthermore, we design a mask-guided smoothness loss to enforce depth consistency within anatomical tissue structures. Extensive experiments on the SCARED, Hamlyn, SERV-CT, and EndoNeRF datasets confirm that our method achieves state-of-the-art performance while maintaining an efficient model size. This work contributes to augmenting surgeons' spatial perception during minimally invasive procedures, thereby enhancing surgical precision and safety, with crucial implications for augmented reality and navigation systems. |
| 2025-08-25 | [TRUCE-AV: A Multimodal dataset for Trust and Comfort Estimation in Autonomous Vehicles](http://arxiv.org/abs/2508.17880v1) | Aditi Bhalla, Christian Hellert et al. | Understanding and estimating driver trust and comfort are essential for the safety and widespread acceptance of autonomous vehicles. Existing works analyze user trust and comfort separately, with limited real-time assessment and insufficient multimodal data. This paper introduces a novel multimodal dataset called TRUCE-AV, focusing on trust and comfort estimation in autonomous vehicles. The dataset collects real-time trust votes and continuous comfort ratings of 31 participants during a simulator-based fully autonomous driving. Simultaneously, physiological signals, such as heart rate, gaze, and emotions, along with environmental data (e.g., vehicle speed, nearby vehicle positions, and velocity), are recorded throughout the drives. Standard pre- and post-drive questionnaires were also administered to assess participants' trust in automation and overall well-being, enabling the correlation of subjective assessments with real-time responses. To demonstrate the utility of our dataset, we evaluated various machine learning models for trust and comfort estimation using physiological data. Our analysis showed that tree-based models like Random Forest and XGBoost and non-linear models such as KNN and MLP regressor achieved the best performance for trust classification and comfort regression. Additionally, we identified key features that contribute to these estimations by using SHAP analysis on the top-performing models. Our dataset enables the development of adaptive AV systems capable of dynamically responding to user trust and comfort levels non-invasively, ultimately enhancing safety, user experience, and human-centered vehicle design. |
| 2025-08-25 | [CubeDN: Real-time Drone Detection in 3D Space from Dual mmWave Radar Cubes](http://arxiv.org/abs/2508.17831v1) | Yuan Fang, Fangzhan Shi et al. | As drone use has become more widespread, there is a critical need to ensure safety and security. A key element of this is robust and accurate drone detection and localization. While cameras and other optical sensors like LiDAR are commonly used for object detection, their performance degrades under adverse lighting and environmental conditions. Therefore, this has generated interest in finding more reliable alternatives, such as millimeter-wave (mmWave) radar. Recent research on mmWave radar object detection has predominantly focused on 2D detection of road users. Although these systems demonstrate excellent performance for 2D problems, they lack the sensing capability to measure elevation, which is essential for 3D drone detection. To address this gap, we propose CubeDN, a single-stage end-to-end radar object detection network specifically designed for flying drones. CubeDN overcomes challenges such as poor elevation resolution by utilizing a dual radar configuration and a novel deep learning pipeline. It simultaneously detects, localizes, and classifies drones of two sizes, achieving decimeter-level tracking accuracy at closer ranges with overall $95\%$ average precision (AP) and $85\%$ average recall (AR). Furthermore, CubeDN completes data processing and inference at 10Hz, making it highly suitable for practical applications. |
| 2025-08-25 | [DRQA: Dynamic Reasoning Quota Allocation for Controlling Overthinking in Reasoning Large Language Models](http://arxiv.org/abs/2508.17803v1) | Kaiwen Yan, Xuanqing Shi et al. | Reasoning large language models (RLLMs), such as OpenAI-O3 and DeepSeek-R1, have recently demonstrated remarkable capabilities by performing structured and multi-step reasoning. However, recent studies reveal that RLLMs often suffer from overthinking, i.e., producing unnecessarily lengthy reasoning chains even for simple questions, leading to excessive token consumption and computational inefficiency. Interestingly, we observe that when processing multiple questions in batch mode, RLLMs exhibit more resource-efficient behavior by dynamically compressing reasoning steps for easier problems, due to implicit resource competition. Inspired by this, we propose Dynamic Reasoning Quota Allocation (DRQA), a novel method that transfers the benefits of resource competition from batch processing to single-question inference. Specifically, DRQA leverages batch-generated preference data and reinforcement learning to train the model to allocate reasoning resources adaptively. By encouraging the model to internalize a preference for responses that are both accurate and concise, DRQA enables it to generate concise answers for simple questions while retaining sufficient reasoning depth for more challenging ones. Extensive experiments on a wide range of mathematical and scientific reasoning benchmarks demonstrate that DRQA significantly reduces token usage while maintaining, and in many cases improving, answer accuracy. By effectively mitigating the overthinking problem, DRQA offers a promising direction for more efficient and scalable deployment of RLLMs, and we hope it inspires further exploration into fine-grained control of reasoning behaviors. |
| 2025-08-25 | [Interpretable Early Failure Detection via Machine Learning and Trace Checking-based Monitoring](http://arxiv.org/abs/2508.17786v1) | Andrea Brunello, Luca Geatti et al. | Monitoring is a runtime verification technique that allows one to check whether an ongoing computation of a system (partial trace) satisfies a given formula. It does not need a complete model of the system, but it typically requires the construction of a deterministic automaton doubly exponential in the size of the formula (in the worst case), which limits its practicality. In this paper, we show that, when considering finite, discrete traces, monitoring of pure past (co)safety fragments of Signal Temporal Logic (STL) can be reduced to trace checking, that is, evaluation of a formula over a trace, that can be performed in time polynomial in the size of the formula and the length of the trace. By exploiting such a result, we develop a GPU-accelerated framework for interpretable early failure detection based on vectorized trace checking, that employs genetic programming to learn temporal properties from historical trace data. The framework shows a 2-10% net improvement in key performance metrics compared to the state-of-the-art methods. |
| 2025-08-25 | [Evaluating the Quality of the Quantified Uncertainty for (Re)Calibration of Data-Driven Regression Models](http://arxiv.org/abs/2508.17761v1) | Jelke Wibbeke, Nico Schönfisch et al. | In safety-critical applications data-driven models must not only be accurate but also provide reliable uncertainty estimates. This property, commonly referred to as calibration, is essential for risk-aware decision-making. In regression a wide variety of calibration metrics and recalibration methods have emerged. However, these metrics differ significantly in their definitions, assumptions and scales, making it difficult to interpret and compare results across studies. Moreover, most recalibration methods have been evaluated using only a small subset of metrics, leaving it unclear whether improvements generalize across different notions of calibration. In this work, we systematically extract and categorize regression calibration metrics from the literature and benchmark these metrics independently of specific modelling methods or recalibration approaches. Through controlled experiments with real-world, synthetic and artificially miscalibrated data, we demonstrate that calibration metrics frequently produce conflicting results. Our analysis reveals substantial inconsistencies: many metrics disagree in their evaluation of the same recalibration result, and some even indicate contradictory conclusions. This inconsistency is particularly concerning as it potentially allows cherry-picking of metrics to create misleading impressions of success. We identify the Expected Normalized Calibration Error (ENCE) and the Coverage Width-based Criterion (CWC) as the most dependable metrics in our tests. Our findings highlight the critical role of metric selection in calibration research. |
| 2025-08-25 | [Talking to Robots: A Practical Examination of Speech Foundation Models for HRI Applications](http://arxiv.org/abs/2508.17753v1) | Theresa Pekarek Rosin, Julia Gachot et al. | Automatic Speech Recognition (ASR) systems in real-world settings need to handle imperfect audio, often degraded by hardware limitations or environmental noise, while accommodating diverse user groups. In human-robot interaction (HRI), these challenges intersect to create a uniquely challenging recognition environment. We evaluate four state-of-the-art ASR systems on eight publicly available datasets that capture six dimensions of difficulty: domain-specific, accented, noisy, age-variant, impaired, and spontaneous speech. Our analysis demonstrates significant variations in performance, hallucination tendencies, and inherent biases, despite similar scores on standard benchmarks. These limitations have serious implications for HRI, where recognition errors can interfere with task performance, user trust, and safety. |
| 2025-08-25 | [Multi-layer Abstraction for Nested Generation of Options (MANGO) in Hierarchical Reinforcement Learning](http://arxiv.org/abs/2508.17751v1) | Alessio Arcudi, Davide Sartor et al. | This paper introduces MANGO (Multilayer Abstraction for Nested Generation of Options), a novel hierarchical reinforcement learning framework designed to address the challenges of long-term sparse reward environments. MANGO decomposes complex tasks into multiple layers of abstraction, where each layer defines an abstract state space and employs options to modularize trajectories into macro-actions. These options are nested across layers, allowing for efficient reuse of learned movements and improved sample efficiency. The framework introduces intra-layer policies that guide the agent's transitions within the abstract state space, and task actions that integrate task-specific components such as reward functions. Experiments conducted in procedurally-generated grid environments demonstrate substantial improvements in both sample efficiency and generalization capabilities compared to standard RL methods. MANGO also enhances interpretability by making the agent's decision-making process transparent across layers, which is particularly valuable in safety-critical and industrial applications. Future work will explore automated discovery of abstractions and abstract actions, adaptation to continuous or fuzzy environments, and more robust multi-layer training strategies. |
| 2025-08-25 | [Speculative Safety-Aware Decoding](http://arxiv.org/abs/2508.17739v1) | Xuekang Wang, Shengyu Zhu et al. | Despite extensive efforts to align Large Language Models (LLMs) with human values and safety rules, jailbreak attacks that exploit certain vulnerabilities continuously emerge, highlighting the need to strengthen existing LLMs with additional safety properties to defend against these attacks. However, tuning large models has become increasingly resource-intensive and may have difficulty ensuring consistent performance. We introduce Speculative Safety-Aware Decoding (SSD), a lightweight decoding-time approach that equips LLMs with the desired safety property while accelerating inference. We assume that there exists a small language model that possesses this desired property. SSD integrates speculative sampling during decoding and leverages the match ratio between the small and composite models to quantify jailbreak risks. This enables SSD to dynamically switch between decoding schemes to prioritize utility or safety, to handle the challenge of different model capacities. The output token is then sampled from a new distribution that combines the distributions of the original and the small models. Experimental results show that SSD successfully equips the large model with the desired safety property, and also allows the model to remain helpful to benign queries. Furthermore, SSD accelerates the inference time, thanks to the speculative sampling design. |
| 2025-08-25 | [Code Difference Guided Fuzzing for FPGA Logic Synthesis Compilers via Bayesian Optimization](http://arxiv.org/abs/2508.17713v1) | Zhihao Xu, Shikai Guo et al. | Field Programmable Gate Arrays (FPGAs) play a crucial role in Electronic Design Automation (EDA) applications, which have been widely used in safety-critical environments, including aerospace, chip manufacturing, and medical devices. A critical step in FPGA development is logic synthesis, which enables developers to translate their software designs into hardware net lists, which facilitates the physical implementation of the chip, detailed timing and power analysis, gate-level simulation, test vector generation, and optimization and consistency checking. However, bugs or incorrect implementations in FPGA logic synthesis compilers may lead to unexpected behaviors in target wapplications, posing security risks. Therefore, it is crucial to eliminate such bugs in FPGA logic synthesis compilers. The effectiveness of existing works is still limited by its simple, blind mutation strategy. To address this challenge, we propose a guided mutation strategy based on Bayesian optimization called LSC-Fuzz to detect bugs in FPGA logic synthesis compilers. Specifically, LSC-Fuzz consists of three components: the test-program generation component, the Bayesian diversity selection component, and the equivalent check component. By performing test-program generation and Bayesian diversity selection, LSC-Fuzz generates diverse and complex HDL code, thoroughly testing the FPGA logic synthesis compilers using equivalent check to detect bugs. Through three months, LSC-Fuzz has found 16 bugs, 12 of these has been confirmed by official technical support. |

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



