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
| 2025-08-27 | [High-frequency continuous gravitational waves searched in LIGO O3 public data with Einstein@Home](http://arxiv.org/abs/2508.20073v1) | Brian McGloughlin, Benjamin Steltner et al. | We search for nearly-monochromatic gravitational wave signals with frequencies $800.0~\textrm{Hz} \leq f \leq 1686.0~\textrm{Hz}$ and spin-down $-2.7\times10^{-9}~\textrm{Hz}\,\textrm{s}^{-1} \leq \dot f \leq 0.2\times 10^{-9}~\textrm{Hz}\,\textrm{s}^{-1}$. We use LIGO O3 public data from the Hanford and Livingston detectors and deploy this search on the Einstein@Home volunteer-computing project. This is the most sensitive search carried out to date in this parameter space. Our results are consistent with a non-detection. We set upper limits on the gravitational wave amplitude $h_{0}$ and translate these to upper limits on neutron star ellipticity and on r-mode amplitude. The most stringent upper limits are at $800~\textrm{Hz}$ with $h_{0} = 1.32\times10^{-25}$, at the $90\%$ confidence level. Searching in the high frequency bands allows us to probe astrophysically interesting ellipticities with our results excluding isolated neutron stars rotating faster than $2.5~\textrm{ms}$ with ellipticities $\epsilon \geq 1.96 \times 10^{-8}\left[\frac{d}{100~\textrm{pc}}\right]$ within a distance $d$ from Earth. Our results also exclude r-mode amplitudes $\alpha \geq 7 \times 10^{-7}\left[\frac{d}{100~\textrm{pc}}\right]$ for neutron stars stars spinning faster than 400 Hz. |
| 2025-08-27 | [Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks](http://arxiv.org/abs/2508.20038v1) | Sheng Liu, Qiang Sheng et al. | Despite advances in improving large language model(LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility. |
| 2025-08-27 | [Bridging the Regulatory Divide: Ensuring Safety and Equity in Wearable Health Technologies](http://arxiv.org/abs/2508.20031v1) | Akshay Kelshiker, Susan Cheng et al. | As wearable health technologies have grown more sophisticated, the distinction between "wellness" and "medical" devices has become increasingly blurred. While some features undergo formal U.S. Food and Drug Administration (FDA) review, many over-the-counter tools operate in a regulatory grey zone, leveraging health-related data and outputs without clinical validation. Further complicating the issue is the widespread repurposing of wellness devices for medical uses, which can introduce safety risks beyond the reach of current oversight. Drawing on legal analysis, case studies, and ethical considerations, we propose an approach emphasizing distributed risk, patient-centered outcomes, and iterative reform. Without a more pluralistic and evolving framework, the promise of wearable health technology risks being undermined by growing inequities, misuse, and eroded public trust. |
| 2025-08-27 | [SWIRL: A Staged Workflow for Interleaved Reinforcement Learning in Mobile GUI Control](http://arxiv.org/abs/2508.20018v1) | Quanfeng Lu, Zhantao Ma et al. | The rapid advancement of large vision language models (LVLMs) and agent systems has heightened interest in mobile GUI agents that can reliably translate natural language into interface operations. Existing single-agent approaches, however, remain limited by structural constraints. Although multi-agent systems naturally decouple different competencies, recent progress in multi-agent reinforcement learning (MARL) has often been hindered by inefficiency and remains incompatible with current LVLM architectures. To address these challenges, we introduce SWIRL, a staged workflow for interleaved reinforcement learning designed for multi-agent systems. SWIRL reformulates MARL into a sequence of single-agent reinforcement learning tasks, updating one agent at a time while keeping the others fixed. This formulation enables stable training and promotes efficient coordination across agents. Theoretically, we provide a stepwise safety bound, a cross-round monotonic improvement theorem, and convergence guarantees on return, ensuring robust and principled optimization. In application to mobile GUI control, SWIRL instantiates a Navigator that converts language and screen context into structured plans, and an Interactor that grounds these plans into executable atomic actions. Extensive experiments demonstrate superior performance on both high-level and low-level GUI benchmarks. Beyond GUI tasks, SWIRL also demonstrates strong capability in multi-agent mathematical reasoning, underscoring its potential as a general framework for developing efficient and robust multi-agent systems. |
| 2025-08-27 | [Evaluating Language Model Reasoning about Confidential Information](http://arxiv.org/abs/2508.19980v1) | Dylan Sam, Alexander Robey et al. | As language models are increasingly deployed as autonomous agents in high-stakes settings, ensuring that they reliably follow user-defined rules has become a critical safety concern. To this end, we study whether language models exhibit contextual robustness, or the capability to adhere to context-dependent safety specifications. For this analysis, we develop a benchmark (PasswordEval) that measures whether language models can correctly determine when a user request is authorized (i.e., with a correct password). We find that current open- and closed-source models struggle with this seemingly simple task, and that, perhaps surprisingly, reasoning capabilities do not generally improve performance. In fact, we find that reasoning traces frequently leak confidential information, which calls into question whether reasoning traces should be exposed to users in such applications. We also scale the difficulty of our evaluation along multiple axes: (i) by adding adversarial user pressure through various jailbreaking strategies, and (ii) through longer multi-turn conversations where password verification is more challenging. Overall, our results suggest that current frontier models are not well-suited to handling confidential information, and that reasoning capabilities may need to be trained in a different manner to make them safer for release in high-stakes settings. |
| 2025-08-27 | [Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors](http://arxiv.org/abs/2508.19953v1) | Rafael Cathomen, Mayank Mittal et al. | Unsupervised Skill Discovery (USD) allows agents to autonomously learn diverse behaviors without task-specific rewards. While recent USD methods have shown promise, their application to real-world robotics remains underexplored. In this paper, we propose a modular USD framework to address the challenges in the safety, interpretability, and deployability of the learned skills. Our approach employs user-defined factorization of the state space to learn disentangled skill representations. It assigns different skill discovery algorithms to each factor based on the desired intrinsic reward function. To encourage structured morphology-aware skills, we introduce symmetry-based inductive biases tailored to individual factors. We also incorporate a style factor and regularization penalties to promote safe and robust behaviors. We evaluate our framework in simulation using a quadrupedal robot and demonstrate zero-shot transfer of the learned skills to real hardware. Our results show that factorization and symmetry lead to the discovery of structured human-interpretable behaviors, while the style factor and penalties enhance safety and diversity. Additionally, we show that the learned skills can be used for downstream tasks and perform on par with oracle policies trained with hand-crafted rewards. |
| 2025-08-27 | [Streamlining the Development of Active Learning Methods in Real-World Object Detection](http://arxiv.org/abs/2508.19906v1) | Moussa Kassem Sbeyti, Nadja Klein et al. | Active learning (AL) for real-world object detection faces computational and reliability challenges that limit practical deployment. Developing new AL methods requires training multiple detectors across iterations to compare against existing approaches. This creates high costs for autonomous driving datasets where the training of one detector requires up to 282 GPU hours. Additionally, AL method rankings vary substantially across validation sets, compromising reliability in safety-critical transportation systems. We introduce object-based set similarity ($\mathrm{OSS}$), a metric that addresses these challenges. $\mathrm{OSS}$ (1) quantifies AL method effectiveness without requiring detector training by measuring similarity between training sets and target domains using object-level features. This enables the elimination of ineffective AL methods before training. Furthermore, $\mathrm{OSS}$ (2) enables the selection of representative validation sets for robust evaluation. We validate our similarity-based approach on three autonomous driving datasets (KITTI, BDD100K, CODA) using uncertainty-based AL methods as a case study with two detector architectures (EfficientDet, YOLOv3). This work is the first to unify AL training and evaluation strategies in object detection based on object similarity. $\mathrm{OSS}$ is detector-agnostic, requires only labeled object crops, and integrates with existing AL pipelines. This provides a practical framework for deploying AL in real-world applications where computational efficiency and evaluation reliability are critical. Code is available at https://mos-ks.github.io/publications/. |
| 2025-08-27 | [NM-Hebb: Coupling Local Hebbian Plasticity with Metric Learning for More Accurate and Interpretable CNNs](http://arxiv.org/abs/2508.19896v1) | Davorin Miličević, Ratko Grbić | Deep Convolutional Neural Networks (CNNs) achieve high accuracy but often rely on purely global, gradient-based optimisation, which can lead to overfitting, redundant filters, and reduced interpretability. To address these limitations, we propose NM-Hebb, a two-phase training framework that integrates neuro-inspired local plasticity with distance-aware supervision. Phase 1 extends standard supervised training by jointly optimising a cross-entropy objective with two biologically inspired mechanisms: (i) a Hebbian regulariser that aligns the spatial mean of activations with the mean of the corresponding convolutional filter weights, encouraging structured, reusable primitives; and (ii) a learnable neuromodulator that gates an elastic-weight-style consolidation loss, preserving beneficial parameters without freezing the network. Phase 2 fine-tunes the backbone with a pairwise metric-learning loss, explicitly compressing intra-class distances and enlarging inter-class margins in the embedding space. Evaluated on CIFAR-10, CIFAR-100, and TinyImageNet across five backbones (ResNet-18, VGG-11, MobileNet-v2, EfficientNet-V2, DenseNet-121), NM-Hebb achieves consistent gains over baseline and other methods: Top-1 accuracy improves by +2.0-10.0 pp (CIFAR-10), +2.0-9.0 pp (CIFAR-100), and up to +4.3-8.9 pp (TinyImageNet), with Normalised Mutual Information (NMI) increased by up to +0.15. Qualitative visualisations and filter-level analyses further confirm that NM-Hebb produces more structured and selective features, yielding tighter and more interpretable class clusters. Overall, coupling local Hebbian plasticity with metric-based fine-tuning yields CNNs that are not only more accurate but also more interpretable, offering practical benefits for resource-constrained and safety-critical AI deployments. |
| 2025-08-27 | [Generative AI for Testing of Autonomous Driving Systems: A Survey](http://arxiv.org/abs/2508.19882v1) | Qunying Song, He Ye et al. | Autonomous driving systems (ADS) have been an active area of research, with the potential to deliver significant benefits to society. However, before large-scale deployment on public roads, extensive testing is necessary to validate their functionality and safety under diverse driving conditions. Therefore, different testing approaches are required, and achieving effective and efficient testing of ADS remains an open challenge. Recently, generative AI has emerged as a powerful tool across many domains, and it is increasingly being applied to ADS testing due to its ability to interpret context, reason about complex tasks, and generate diverse outputs. To gain a deeper understanding of its role in ADS testing, we systematically analyzed 91 relevant studies and synthesized their findings into six major application categories, primarily centered on scenario-based testing of ADS. We also reviewed their effectiveness and compiled a wide range of datasets, simulators, ADS, metrics, and benchmarks used for evaluation, while identifying 27 limitations. This survey provides an overview and practical insights into the use of generative AI for testing ADS, highlights existing challenges, and outlines directions for future research in this rapidly evolving field. |
| 2025-08-27 | [Gradient Rectification for Robust Calibration under Distribution Shift](http://arxiv.org/abs/2508.19830v1) | Yilin Zhang, Cai Xu et al. | Deep neural networks often produce overconfident predictions, undermining their reliability in safety-critical applications. This miscalibration is further exacerbated under distribution shift, where test data deviates from the training distribution due to environmental or acquisition changes. While existing approaches improve calibration through training-time regularization or post-hoc adjustment, their reliance on access to or simulation of target domains limits their practicality in real-world scenarios. In this paper, we propose a novel calibration framework that operates without access to target domain information. From a frequency-domain perspective, we identify that distribution shifts often distort high-frequency visual cues exploited by deep models, and introduce a low-frequency filtering strategy to encourage reliance on domain-invariant features. However, such information loss may degrade In-Distribution (ID) calibration performance. Therefore, we further propose a gradient-based rectification mechanism that enforces ID calibration as a hard constraint during optimization. Experiments on synthetic and real-world shifted datasets, including CIFAR-10/100-C and WILDS, demonstrate that our method significantly improves calibration under distribution shift while maintaining strong in-distribution performance. |
| 2025-08-27 | [A Standing Support Mobility Robot for Enhancing Independence in Elderly Daily Living](http://arxiv.org/abs/2508.19816v1) | Ricardo J. Manríquez-Cisterna, Ankit A. Ravankar et al. | This paper presents a standing support mobility robot "Moby" developed to enhance independence and safety for elderly individuals during daily activities such as toilet transfers. Unlike conventional seated mobility aids, the robot maintains users in an upright posture, reducing physical strain, supporting natural social interaction at eye level, and fostering a greater sense of self-efficacy. Moby offers a novel alternative by functioning both passively and with mobility support, enabling users to perform daily tasks more independently. Its main advantages include ease of use, lightweight design, comfort, versatility, and effective sit-to-stand assistance. The robot leverages the Robot Operating System (ROS) for seamless control, featuring manual and autonomous operation modes. A custom control system enables safe and intuitive interaction, while the integration with NAV2 and LiDAR allows for robust navigation capabilities. This paper reviews existing mobility solutions and compares them to Moby, details the robot's design, and presents objective and subjective experimental results using the NASA-TLX method and time comparisons to other methods to validate our design criteria and demonstrate the advantages of our contribution. |
| 2025-08-27 | [T2R-bench: A Benchmark for Generating Article-Level Reports from Real World Industrial Tables](http://arxiv.org/abs/2508.19813v1) | Jie Zhang, Changzai Pan et al. | Extensive research has been conducted to explore the capabilities of large language models (LLMs) in table reasoning. However, the essential task of transforming tables information into reports remains a significant challenge for industrial applications. This task is plagued by two critical issues: 1) the complexity and diversity of tables lead to suboptimal reasoning outcomes; and 2) existing table benchmarks lack the capacity to adequately assess the practical application of this task. To fill this gap, we propose the table-to-report task and construct a bilingual benchmark named T2R-bench, where the key information flow from the tables to the reports for this task. The benchmark comprises 457 industrial tables, all derived from real-world scenarios and encompassing 19 industry domains as well as 4 types of industrial tables. Furthermore, we propose an evaluation criteria to fairly measure the quality of report generation. The experiments on 25 widely-used LLMs reveal that even state-of-the-art models like Deepseek-R1 only achieves performance with 62.71 overall score, indicating that LLMs still have room for improvement on T2R-bench. Source code and data will be available after acceptance. |
| 2025-08-27 | [Context-Aware Risk Estimation in Home Environments: A Probabilistic Framework for Service Robots](http://arxiv.org/abs/2508.19788v1) | Sena Ishii, Akash Chikhalikar et al. | We present a novel framework for estimating accident-prone regions in everyday indoor scenes, aimed at improving real-time risk awareness in service robots operating in human-centric environments. As robots become integrated into daily life, particularly in homes, the ability to anticipate and respond to environmental hazards is crucial for ensuring user safety, trust, and effective human-robot interaction. Our approach models object-level risk and context through a semantic graph-based propagation algorithm. Each object is represented as a node with an associated risk score, and risk propagates asymmetrically from high-risk to low-risk objects based on spatial proximity and accident relationship. This enables the robot to infer potential hazards even when they are not explicitly visible or labeled. Designed for interpretability and lightweight onboard deployment, our method is validated on a dataset with human-annotated risk regions, achieving a binary risk detection accuracy of 75%. The system demonstrates strong alignment with human perception, particularly in scenes involving sharp or unstable objects. These results underline the potential of context-aware risk reasoning to enhance robotic scene understanding and proactive safety behaviors in shared human-robot spaces. This framework could serve as a foundation for future systems that make context-driven safety decisions, provide real-time alerts, or autonomously assist users in avoiding or mitigating hazards within home environments. |
| 2025-08-27 | [Using normal to find abnormal: AI-based anomaly detection in gravitational wave data](http://arxiv.org/abs/2508.19772v1) | Yi-Yang Guo, Soumya D. Mohanty et al. | The detection and classification of anomalies in gravitational wave data plays a critical role in improving the sensitivity of searches for signals of astrophysical origins. We present ABNORMAL (AI Based Nonstationarity Observer for Resectioning and Marking AnomaLies), a deep neural network (DNN) model for anomaly detection that is trained exclusively on simulated Gaussian noise. By removing dependence on real data for training, the method resolves a circular paradox in anomaly detection: training on real data implicitly involves prior segregation of stationary from non-stationary data but this is not possible unless all anomalies are detected first. ABNORMAL is an autoencoder-based DNN, commonly used in anomaly detection, with the key innovation that it is trained to predict statistical features of noise rather than reconstructing the noise time series themselves. The statistical features are obtained by applying Gabor and Wavelet filter banks, which implement time-frequency analysis, and are subsequently combined through multi-view fusion using a dual-path architecture. We quantify the performance of our method on simulated and real LIGO data. Application to data from the O1 to O3b observational runs uncovers a rich landscape of anomalies over different timescales, including many that do not fit within known classes. |
| 2025-08-27 | [Safety Alignment Should Be Made More Than Just A Few Attention Heads](http://arxiv.org/abs/2508.19697v1) | Chao Huang, Zefeng Zhang et al. | Current safety alignment for large language models(LLMs) continues to present vulnerabilities, given that adversarial prompting can effectively bypass their safety measures.Our investigation shows that these safety mechanisms predominantly depend on a limited subset of attention heads: removing or ablating these heads can severely compromise model safety. To identify and evaluate these safety-critical components, we introduce RDSHA, a targeted ablation method that leverages the model's refusal direction to pinpoint attention heads mostly responsible for safety behaviors. Further analysis shows that existing jailbreak attacks exploit this concentration by selectively bypassing or manipulating these critical attention heads. To address this issue, we propose AHD, a novel training strategy designed to promote the distributed encoding of safety-related behaviors across numerous attention heads. Experimental results demonstrate that AHD successfully distributes safety-related capabilities across more attention heads. Moreover, evaluations under several mainstream jailbreak attacks show that models trained with AHD exhibit considerably stronger safety robustness, while maintaining overall functional utility. |
| 2025-08-27 | [InquireMobile: Teaching VLM-based Mobile Agent to Request Human Assistance via Reinforcement Fine-Tuning](http://arxiv.org/abs/2508.19679v1) | Qihang Ai, Pi Bu et al. | Recent advances in Vision-Language Models (VLMs) have enabled mobile agents to perceive and interact with real-world mobile environments based on human instructions. However, the current fully autonomous paradigm poses potential safety risks when model understanding or reasoning capabilities are insufficient. To address this challenge, we first introduce \textbf{InquireBench}, a comprehensive benchmark specifically designed to evaluate mobile agents' capabilities in safe interaction and proactive inquiry with users, encompassing 5 categories and 22 sub-categories, where most existing VLM-based agents demonstrate near-zero performance. In this paper, we aim to develop an interactive system that actively seeks human confirmation at critical decision points. To achieve this, we propose \textbf{InquireMobile}, a novel model inspired by reinforcement learning, featuring a two-stage training strategy and an interactive pre-action reasoning mechanism. Finally, our model achieves an 46.8% improvement in inquiry success rate and the best overall success rate among existing baselines on InquireBench. We will open-source all datasets, models, and evaluation codes to facilitate development in both academia and industry. |
| 2025-08-27 | [Distributed Safety-Critical MPC for Multi-Agent Formation Control and Obstacle Avoidance](http://arxiv.org/abs/2508.19678v1) | Chao Wang, Shuyuan Zhang et al. | For nonlinear multi-agent systems with high relative degrees, achieving formation control and obstacle avoidance in a distributed manner remains a significant challenge. To address this issue, we propose a novel distributed safety-critical model predictive control (DSMPC) algorithm that incorporates discrete-time high-order control barrier functions (DHCBFs) to enforce safety constraints, alongside discrete-time control Lyapunov functions (DCLFs) to establish terminal constraints. To facilitate distributed implementation, we develop estimated neighbor states for formulating DHCBFs and DCLFs, while also devising a bound constraint to limit estimation errors and ensure convergence. Additionally, we provide theoretical guarantees regarding the feasibility and stability of the proposed DSMPC algorithm based on a mild assumption. The effectiveness of the proposed method is evidenced by the simulation results, demonstrating improved performance and reduced computation time compared to existing approaches. |
| 2025-08-27 | [Intellectual Property in Graph-Based Machine Learning as a Service: Attacks and Defenses](http://arxiv.org/abs/2508.19641v1) | Lincan Li, Bolin Shen et al. | Graph-structured data, which captures non-Euclidean relationships and interactions between entities, is growing in scale and complexity. As a result, training state-of-the-art graph machine learning (GML) models have become increasingly resource-intensive, turning these models and data into invaluable Intellectual Property (IP). To address the resource-intensive nature of model training, graph-based Machine-Learning-as-a-Service (GMLaaS) has emerged as an efficient solution by leveraging third-party cloud services for model development and management. However, deploying such models in GMLaaS also exposes them to potential threats from attackers. Specifically, while the APIs within a GMLaaS system provide interfaces for users to query the model and receive outputs, they also allow attackers to exploit and steal model functionalities or sensitive training data, posing severe threats to the safety of these GML models and the underlying graph data. To address these challenges, this survey systematically introduces the first taxonomy of threats and defenses at the level of both GML model and graph-structured data. Such a tailored taxonomy facilitates an in-depth understanding of GML IP protection. Furthermore, we present a systematic evaluation framework to assess the effectiveness of IP protection methods, introduce a curated set of benchmark datasets across various domains, and discuss their application scopes and future challenges. Finally, we establish an open-sourced versatile library named PyGIP, which evaluates various attack and defense techniques in GMLaaS scenarios and facilitates the implementation of existing benchmark methods. The library resource can be accessed at: https://labrai.github.io/PyGIP. We believe this survey will play a fundamental role in intellectual property protection for GML and provide practical recipes for the GML community. |
| 2025-08-27 | [Language Models Identify Ambiguities and Exploit Loopholes](http://arxiv.org/abs/2508.19546v1) | Jio Choi, Mohit Bansal et al. | Studying the responses of large language models (LLMs) to loopholes presents a two-fold opportunity. First, it affords us a lens through which to examine ambiguity and pragmatics in LLMs, since exploiting a loophole requires identifying ambiguity and performing sophisticated pragmatic reasoning. Second, loopholes pose an interesting and novel alignment problem where the model is presented with conflicting goals and can exploit ambiguities to its own advantage. To address these questions, we design scenarios where LLMs are given a goal and an ambiguous user instruction in conflict with the goal, with scenarios covering scalar implicature, structural ambiguities, and power dynamics. We then measure different models' abilities to exploit loopholes to satisfy their given goals as opposed to the goals of the user. We find that both closed-source and stronger open-source models can identify ambiguities and exploit their resulting loopholes, presenting a potential AI safety risk. Our analysis indicates that models which exploit loopholes explicitly identify and reason about both ambiguity and conflicting goals. |
| 2025-08-27 | [Spatial-temporal risk field-based coupled dynamic-static driving risk assessment and trajectory planning in weaving segments](http://arxiv.org/abs/2508.19513v1) | Guodong Ma, Baofeng Sun et al. | In this paper, we first propose a spatial-temporal coupled risk assessment paradigm by constructing a three-dimensional spatial-temporal risk field (STRF). Specifically, we introduce spatial-temporal distances to quantify the impact of future trajectories of dynamic obstacles. We also incorporate a geometrically configured specialized field for the weaving segment to constrain vehicle movement directionally. To enhance the STRF's accuracy, we further developed a parameter calibration method using real-world aerial video data, leveraging YOLO-based machine vision and dynamic risk balance theory. A comparative analysis with the traditional risk field demonstrates the STRF's superior situational awareness of anticipatory risk. Building on these results, we final design a STRF-based CAV trajectory planning method in weaving segments. We integrate spatial-temporal risk occupancy maps, dynamic iterative sampling, and quadratic programming to enhance safety, comfort, and efficiency. By incorporating both dynamic and static risk factors during the sampling phase, our method ensures robust safety performance. Additionally, the proposed method simultaneously optimizes path and speed using a parallel computing approach, reducing computation time. Real-world cases show that, compared to the dynamic planning + quadratic programming schemes, and real human driving trajectories, our method significantly improves safety, reduces lane-change completion time, and minimizes speed fluctuations. |

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



