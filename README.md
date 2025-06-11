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
| 2025-06-10 | [Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation with World Foundation Models](http://arxiv.org/abs/2506.09042v1) | Xuanchi Ren, Yifan Lu et al. | Collecting and annotating real-world data for safety-critical physical AI systems, such as Autonomous Vehicle (AV), is time-consuming and costly. It is especially challenging to capture rare edge cases, which play a critical role in training and testing of an AV system. To address this challenge, we introduce the Cosmos-Drive-Dreams - a synthetic data generation (SDG) pipeline that aims to generate challenging scenarios to facilitate downstream tasks such as perception and driving policy training. Powering this pipeline is Cosmos-Drive, a suite of models specialized from NVIDIA Cosmos world foundation model for the driving domain and are capable of controllable, high-fidelity, multi-view, and spatiotemporally consistent driving video generation. We showcase the utility of these models by applying Cosmos-Drive-Dreams to scale the quantity and diversity of driving datasets with high-fidelity and challenging scenarios. Experimentally, we demonstrate that our generated data helps in mitigating long-tail distribution problems and enhances generalization in downstream tasks such as 3D lane detection, 3D object detection and driving policy learning. We open source our pipeline toolkit, dataset and model weights through the NVIDIA's Cosmos platform.   Project page: https://research.nvidia.com/labs/toronto-ai/cosmos_drive_dreams |
| 2025-06-10 | [DIsoN: Decentralized Isolation Networks for Out-of-Distribution Detection in Medical Imaging](http://arxiv.org/abs/2506.09024v1) | Felix Wagner, Pramit Saha et al. | Safe deployment of machine learning (ML) models in safety-critical domains such as medical imaging requires detecting inputs with characteristics not seen during training, known as out-of-distribution (OOD) detection, to prevent unreliable predictions. Effective OOD detection after deployment could benefit from access to the training data, enabling direct comparison between test samples and the training data distribution to identify differences. State-of-the-art OOD detection methods, however, either discard training data after deployment or assume that test samples and training data are centrally stored together, an assumption that rarely holds in real-world settings. This is because shipping training data with the deployed model is usually impossible due to the size of training databases, as well as proprietary or privacy constraints. We introduce the Isolation Network, an OOD detection framework that quantifies the difficulty of separating a target test sample from the training data by solving a binary classification task. We then propose Decentralized Isolation Networks (DIsoN), which enables the comparison of training and test data when data-sharing is impossible, by exchanging only model parameters between the remote computational nodes of training and deployment. We further extend DIsoN with class-conditioning, comparing a target sample solely with training data of its predicted class. We evaluate DIsoN on four medical imaging datasets (dermatology, chest X-ray, breast ultrasound, histopathology) across 12 OOD detection tasks. DIsoN performs favorably against existing methods while respecting data-privacy. This decentralized OOD detection framework opens the way for a new type of service that ML developers could provide along with their models: providing remote, secure utilization of their training data for OOD detection services. Code will be available upon acceptance at: ***** |
| 2025-06-10 | [Comparing human and LLM proofreading in L2 writing: Impact on lexical and syntactic features](http://arxiv.org/abs/2506.09021v1) | Hakyung Sung, Karla Csuros et al. | This study examines the lexical and syntactic interventions of human and LLM proofreading aimed at improving overall intelligibility in identical second language writings, and evaluates the consistency of outcomes across three LLMs (ChatGPT-4o, Llama3.1-8b, Deepseek-r1-8b). Findings show that both human and LLM proofreading enhance bigram lexical features, which may contribute to better coherence and contextual connectedness between adjacent words. However, LLM proofreading exhibits a more generative approach, extensively reworking vocabulary and sentence structures, such as employing more diverse and sophisticated vocabulary and incorporating a greater number of adjective modifiers in noun phrases. The proofreading outcomes are highly consistent in major lexical and syntactic features across the three models. |
| 2025-06-10 | [Online Learning Control Strategies for Industrial Processes with Application for Loosening and Conditioning](http://arxiv.org/abs/2506.08983v1) | Yue Wu, Jianfu Cao et al. | This paper proposes a novel adaptive Koopman Model Predictive Control (MPC) framework, termed HPC-AK-MPC, designed to address the dual challenges of time-varying dynamics and safe operation in complex industrial processes. The framework integrates two core strategies: online learning and historically-informed safety constraints. To contend with process time-variance, a Recursive Extended Dynamic Mode Decomposition (rEDMDc) technique is employed to construct an adaptive Koopman model capable of updating its parameters from real-time data, endowing the controller with the ability to continuously learn and track dynamic changes. To tackle the critical issue of safe operation under model uncertainty, we introduce a novel Historical Process Constraint (HPC) mechanism. This mechanism mines successful operational experiences from a historical database and, by coupling them with the confidence level of the online model, generates a dynamic "safety corridor" for the MPC optimization problem. This approach transforms implicit expert knowledge into explicit, adaptive constraints, establishing a dynamic balance between pursuing optimal performance and ensuring robust safety. The proposed HPC-AK-MPC method is applied to a real-world tobacco loosening and conditioning process and systematically validated using an "advisor mode" simulation framework with industrial data. Experimental results demonstrate that, compared to historical operations, the proposed method significantly improves the Process Capability Index (Cpk) for key quality variables across all tested batches, proving its substantial potential in enhancing control performance while guaranteeing operational safety. |
| 2025-06-10 | [Evaluating Generative Vehicle Trajectory Models for Traffic Intersection Dynamics](http://arxiv.org/abs/2506.08963v1) | Yash Ranjan, Rahul Sengupta et al. | Traffic Intersections are vital to urban road networks as they regulate the movement of people and goods. However, they are regions of conflicting trajectories and are prone to accidents. Deep Generative models of traffic dynamics at signalized intersections can greatly help traffic authorities better understand the efficiency and safety aspects. At present, models are evaluated on computational metrics that primarily look at trajectory reconstruction errors. They are not evaluated online in a `live' microsimulation scenario. Further, these metrics do not adequately consider traffic engineering-specific concerns such as red-light violations, unallowed stoppage, etc. In this work, we provide a comprehensive analytics tool to train, run, and evaluate models with metrics that give better insights into model performance from a traffic engineering point of view. We train a state-of-the-art multi-vehicle trajectory forecasting model on a large dataset collected by running a calibrated scenario of a real-world urban intersection. We then evaluate the performance of the prediction models, online in a microsimulator, under unseen traffic conditions. We show that despite using ideally-behaved trajectories as input, and achieving low trajectory reconstruction errors, the generated trajectories show behaviors that break traffic rules. We introduce new metrics to evaluate such undesired behaviors and present our results. |
| 2025-06-10 | [IntTrajSim: Trajectory Prediction for Simulating Multi-Vehicle driving at Signalized Intersections](http://arxiv.org/abs/2506.08957v1) | Yash Ranjan, Rahul Sengupta et al. | Traffic simulators are widely used to study the operational efficiency of road infrastructure, but their rule-based approach limits their ability to mimic real-world driving behavior. Traffic intersections are critical components of the road infrastructure, both in terms of safety risk (nearly 28% of fatal crashes and 58% of nonfatal crashes happen at intersections) as well as the operational efficiency of a road corridor. This raises an important question: can we create a data-driven simulator that can mimic the macro- and micro-statistics of the driving behavior at a traffic intersection? Deep Generative Modeling-based trajectory prediction models provide a good starting point to model the complex dynamics of vehicles at an intersection. But they are not tested in a "live" micro-simulation scenario and are not evaluated on traffic engineering-related metrics. In this study, we propose traffic engineering-related metrics to evaluate generative trajectory prediction models and provide a simulation-in-the-loop pipeline to do so. We also provide a multi-headed self-attention-based trajectory prediction model that incorporates the signal information, which outperforms our previous models on the evaluation metrics. |
| 2025-06-10 | [AdversariaL attacK sAfety aLIgnment(ALKALI): Safeguarding LLMs through GRACE: Geometric Representation-Aware Contrastive Enhancement- Introducing Adversarial Vulnerability Quality Index (AVQI)](http://arxiv.org/abs/2506.08885v1) | Danush Khanna, Krishna Kumar et al. | Adversarial threats against LLMs are escalating faster than current defenses can adapt. We expose a critical geometric blind spot in alignment: adversarial prompts exploit latent camouflage, embedding perilously close to the safe representation manifold while encoding unsafe intent thereby evading surface level defenses like Direct Preference Optimization (DPO), which remain blind to the latent geometry. We introduce ALKALI, the first rigorously curated adversarial benchmark and the most comprehensive to date spanning 9,000 prompts across three macro categories, six subtypes, and fifteen attack families. Evaluation of 21 leading LLMs reveals alarmingly high Attack Success Rates (ASRs) across both open and closed source models, exposing an underlying vulnerability we term latent camouflage, a structural blind spot where adversarial completions mimic the latent geometry of safe ones. To mitigate this vulnerability, we introduce GRACE - Geometric Representation Aware Contrastive Enhancement, an alignment framework coupling preference learning with latent space regularization. GRACE enforces two constraints: latent separation between safe and adversarial completions, and adversarial cohesion among unsafe and jailbreak behaviors. These operate over layerwise pooled embeddings guided by a learned attention profile, reshaping internal geometry without modifying the base model, and achieve up to 39% ASR reduction. Moreover, we introduce AVQI, a geometry aware metric that quantifies latent alignment failure via cluster separation and compactness. AVQI reveals when unsafe completions mimic the geometry of safe ones, offering a principled lens into how models internally encode safety. We make the code publicly available at https://anonymous.4open.science/r/alkali-B416/README.md. |
| 2025-06-10 | [Safety-Driven Response Adaptive Randomisation: An Application in Non-inferiority Oncology Trials](http://arxiv.org/abs/2506.08864v1) | Maria Vittoria Chiaruttini, Lukas Pin et al. | The majority of response-adaptive randomisation (RAR) designs in the literature use efficacy data to dynamically allocate patients. Their applicability in settings where the efficacy measure is observable with a random delay, such as overall survival, remains challenging. This paper introduces a RAR design referred to as SAFER (Safety-Aware Flexible Elastic Randomisation) design, which uses early-emerging safety data to inform treatment allocation decisions in oncology trials. However, the design is applicable to a range of settings where it may be desirable to favour the arm demonstrating a superior safety profile. This is particularly relevant in non-inferiority trials, which aim to demonstrate an experimental treatment is not inferior to the standard of care, while offering advantages in terms of safety and tolerability. Consequently, an unavoidable and well-established trade-off arises for such designs: to balance the goals of preserving inferential efficiency for the primary non-inferiority outcome while incorporating safety considerations into the randomisation process through RAR. Our method, defines a randomisation procedure which prioritises the assignment of patients to better-tolerated arms and adjusts the allocation proportion according to the observed association between safety and efficacy endpoints. We illustrate our procedure through a comprehensive simulation study, inspired by the CAPP-IT Phase III oncology trial. Our results demonstrate that SAFER preserves statistical power even when efficacy and safety endpoints are weakly associated and offers power gains when a strong positive association is present. Moreover, the approach enables a faster/slower adaptation when efficacy and safety endpoints are temporally aligned/misaligned, respectively. |
| 2025-06-10 | [Confidence Boosts Trust-Based Resilience in Cooperative Multi-Robot Systems](http://arxiv.org/abs/2506.08807v1) | Luca Ballotta, Áron Vékássy et al. | Wireless communication-based multi-robot systems open the door to cyberattacks that can disrupt safety and performance of collaborative robots. The physical channel supporting inter-robot communication offers an attractive opportunity to decouple the detection of malicious robots from task-relevant data exchange between legitimate robots. Yet, trustworthiness indications coming from physical channels are uncertain and must be handled with this in mind. In this paper, we propose a resilient protocol for multi-robot operation wherein a parameter {\lambda}t accounts for how confident a robot is about the legitimacy of nearby robots that the physical channel indicates. Analytical results prove that our protocol achieves resilient coordination with arbitrarily many malicious robots under mild assumptions. Tuning {\lambda}t allows a designer to trade between near-optimal inter-robot coordination and quick task execution; see Fig. 1. This is a fundamental performance tradeoff and must be carefully evaluated based on the task at hand. The effectiveness of our approach is numerically verified with experiments involving platoons of autonomous cars where some vehicles are maliciously spoofed. |
| 2025-06-10 | [Enhancing Accuracy and Maintainability in Nuclear Plant Data Retrieval: A Function-Calling LLM Approach Over NL-to-SQL](http://arxiv.org/abs/2506.08757v1) | Mishca de Costa, Muhammad Anwar et al. | Retrieving operational data from nuclear power plants requires exceptional accuracy and transparency due to the criticality of the decisions it supports. Traditionally, natural language to SQL (NL-to-SQL) approaches have been explored for querying such data. While NL-to-SQL promises ease of use, it poses significant risks: end-users cannot easily validate generated SQL queries, and legacy nuclear plant databases -- often complex and poorly structured -- complicate query generation due to decades of incremental modifications. These challenges increase the likelihood of inaccuracies and reduce trust in the approach. In this work, we propose an alternative paradigm: leveraging function-calling large language models (LLMs) to address these challenges. Instead of directly generating SQL queries, we define a set of pre-approved, purpose-specific functions representing common use cases. Queries are processed by invoking these functions, which encapsulate validated SQL logic. This hybrid approach mitigates the risks associated with direct NL-to-SQL translations by ensuring that SQL queries are reviewed and optimized by experts before deployment. While this strategy introduces the upfront cost of developing and maintaining the function library, we demonstrate how NL-to-SQL tools can assist in the initial generation of function code, allowing experts to focus on validation rather than creation. Our study includes a performance comparison between direct NL-to-SQL generation and the proposed function-based approach, highlighting improvements in accuracy and maintainability. This work underscores the importance of balancing user accessibility with operational safety and provides a novel, actionable framework for robust data retrieval in critical systems. |
| 2025-06-10 | [Societal AI Research Has Become Less Interdisciplinary](http://arxiv.org/abs/2506.08738v1) | Dror Kris Markus, Fabrizio Gilardi et al. | As artificial intelligence (AI) systems become deeply embedded in everyday life, calls to align AI development with ethical and societal values have intensified. Interdisciplinary collaboration is often championed as a key pathway for fostering such engagement. Yet it remains unclear whether interdisciplinary research teams are actually leading this shift in practice. This study analyzes over 100,000 AI-related papers published on ArXiv between 2014 and 2024 to examine how ethical values and societal concerns are integrated into technical AI research. We develop a classifier to identify societal content and measure the extent to which research papers express these considerations. We find a striking shift: while interdisciplinary teams remain more likely to produce societally-oriented research, computer science-only teams now account for a growing share of the field's overall societal output. These teams are increasingly integrating societal concerns into their papers and tackling a wide range of domains - from fairness and safety to healthcare and misinformation. These findings challenge common assumptions about the drivers of societal AI and raise important questions. First, what are the implications for emerging understandings of AI safety and governance if most societally-oriented research is being undertaken by exclusively technical teams? Second, for scholars in the social sciences and humanities: in a technical field increasingly responsive to societal demands, what distinctive perspectives can we still offer to help shape the future of AI? |
| 2025-06-10 | [ROS-related Robotic Systems Development with V-model-based Application of MeROS Metamodel](http://arxiv.org/abs/2506.08706v1) | Tomasz Winiarski, Jan Kaniuka et al. | As robotic systems grow increasingly complex, heterogeneous, and safety-critical, the need for structured development methodologies becomes paramount. Although frameworks like the Robot Operating System (ROS) and Model-Based Systems Engineering (MBSE) offer foundational tools, they often lack integration when used together. This paper addresses that gap by aligning the widely recognized V-model development paradigm with the MeROS metamodel SysML-based modeling language tailored for ROS-based systems.   We propose a domain-specific methodology that bridges ROS-centric modelling with systems engineering practices. Our approach formalises the structure, behaviour, and validation processes of robotic systems using MeROS, while extending it with a generalized, adaptable V-model compatible with both ROS and ROS 2. Rather than prescribing a fixed procedure, the approach supports project-specific flexibility and reuse, offering guidance across all stages of development.   The approach is validated through a comprehensive case study on HeROS, a heterogeneous multi-robot platform comprising manipulators, mobile units, and dynamic test environments. This example illustrates how the MeROS-compatible V-model enhances traceability and system consistency while remaining accessible and extensible for future adaptation. The work contributes a structured, tool-agnostic foundation for developers and researchers seeking to apply MBSE practices in ROS-based projects. |
| 2025-06-10 | [Causality-aware Safety Testing for Autonomous Driving Systems](http://arxiv.org/abs/2506.08688v1) | Wenbing Tang, Mingfei Cheng et al. | Simulation-based testing is essential for evaluating the safety of Autonomous Driving Systems (ADSs). Comprehensive evaluation requires testing across diverse scenarios that can trigger various types of violations under different conditions. While existing methods typically focus on individual diversity metrics, such as input scenarios, ADS-generated motion commands, and system violations, they often fail to capture the complex interrelationships among these elements. This oversight leads to gaps in testing coverage, potentially missing critical issues in the ADS under evaluation. However, quantifying these interrelationships presents a significant challenge. In this paper, we propose a novel causality-aware fuzzing technique, Causal-Fuzzer, to enable efficient and comprehensive testing of ADSs by exploring causally diverse scenarios. The core of Causal-Fuzzer is constructing a causal graph to model the interrelationships among the diversities of input scenarios, ADS motion commands, and system violations. Then the causal graph will guide the process of critical scenario generation. Specifically, Causal-Fuzzer proposes (1) a causality-based feedback mechanism that quantifies the combined diversity of test scenarios by assessing whether they activate new causal relationships, and (2) a causality-driven mutation strategy that prioritizes mutations on input scenario elements with higher causal impact on ego action changes and violation occurrence, rather than treating all elements equally. We evaluated Causal-Fuzzer on an industry-grade ADS Apollo, with a high-fidelity. Our empirical results demonstrate that Causal-Fuzzer significantly outperforms existing methods in (1) identifying a greater diversity of violations, (2) providing enhanced testing sufficiency with improved coverage of causal relationships, and (3) achieving greater efficiency in detecting the first critical scenarios. |
| 2025-06-10 | [Linguistic Ordered Weighted Averaging based deep learning pooling for fault diagnosis in a wastewater treatment plant](http://arxiv.org/abs/2506.08676v1) | Alicia Beneyto-Rodriguez, Gregorio I. Sainz-Palmero et al. | Nowadays, water reuse is a serious challenge to help address water shortages. Here, the wastewater treatment plants (WWTP) play a key role, and its proper operation is mandatory. So, fault diagnosis is a key activity for these plants. Their high complexity and large-scale require of smart methodologies for that fault diagnosis and safety operation. All these large-scale and complex industrial processes are monitored, allowing the data collection about the plant operation, so data driven approaches for fault diagnosis can be applied. A popular approach to fault diagnosis is deep learning-based methodologies. Here, a fault diagnosis methodology is proposed for a WWTP using a new linguistic Ordered Weighted Averaging (OWA) pooling based Deep Convolutional Neural Network (DCNN) and a sliding and overlapping time window. This window slides over input data based on the monitoring sampling time, then the diagnosis is carried out by the linguistic OWA pooling based DCNN. This alternative linguistic pooling uses well-known linguistic OWA quantifiers, which permit terms such as \textsl{Most, AtLeast, etc.}, supplying new intuitive options for the pooling tasks. This sliding time window and the OWA pooling based network permit a better and earlier fault diagnosis, at each sampling time, using a few monitoring samples and a fewer learning iterations than DCNN standard pooling. Several linguistic OWA operators have been checked with a benchmark for WWTPs. A set of 5 fault types has been used, taking into account 140 variables sampled at 15 minutes time intervals. The performance has been over $91\%$ for $Accuracy$, $Recall$ or $F1-Score$, and better than other competitive methodologies. Moreover, these linguistic OWA operators for DCNN pooling have shown a better performance than the standard \textsl{Max} and \textsl{Average} options. |
| 2025-06-10 | [RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](http://arxiv.org/abs/2506.08672v1) | Yang Liu, Jiaqi Li et al. | Rule-based reasoning has been acknowledged as one of the fundamental problems in reasoning, while deviations in rule formats, types, and complexity in real-world applications pose severe challenges. Recent studies have shown that large reasoning models (LRMs) have remarkable reasoning capabilities, and their performance is substantially enhanced by reinforcement learning (RL). However, it remains an open question whether small reasoning models (SRMs) can learn rule-based reasoning effectively with robust generalization across diverse tasks and domains. To address this, we introduce Reinforced Rule-based Reasoning, a.k.a. RuleReasoner, a simple yet effective method to conduct rule-based reasoning via a wide collection of curated tasks and a novel domain-aware dynamic sampling approach. Specifically, RuleReasoner resamples each training batch by updating the sampling weights of different domains based on historical rewards. This facilitates domain augmentation and flexible online learning schedules for RL, obviating the need for pre-hoc human-engineered mix-training recipes used in existing methods. Empirical evaluations on in-distribution (ID) and out-of-distribution (OOD) benchmarks reveal that RuleReasoner outperforms frontier LRMs by a significant margin ($\Delta$4.1% average points on eight ID tasks and $\Delta$10.4% average points on three OOD tasks over OpenAI-o1). Notably, our approach also exhibits higher computational efficiency compared to prior dynamic sampling methods for RL. |
| 2025-06-10 | [CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmark of Large Language Models in Mental Health Counseling](http://arxiv.org/abs/2506.08584v1) | Yahan Li, Jifan Yao et al. | Large language models (LLMs) are increasingly proposed for use in mental health support, yet their behavior in realistic counseling scenarios remains largely untested. We introduce CounselBench, a large-scale benchmark developed with 100 mental health professionals to evaluate and stress-test LLMs in single-turn counseling. The first component, CounselBench-EVAL, contains 2,000 expert evaluations of responses from GPT-4, LLaMA 3, Gemini, and online human therapists to real patient questions. Each response is rated along six clinically grounded dimensions, with written rationales and span-level annotations. We find that LLMs often outperform online human therapists in perceived quality, but experts frequently flag their outputs for safety concerns such as unauthorized medical advice. Follow-up experiments show that LLM judges consistently overrate model responses and overlook safety issues identified by human experts. To probe failure modes more directly, we construct CounselBench-Adv, an adversarial dataset of 120 expert-authored counseling questions designed to trigger specific model issues. Evaluation across 2,880 responses from eight LLMs reveals consistent, model-specific failure patterns. Together, CounselBench establishes a clinically grounded framework for benchmarking and improving LLM behavior in high-stakes mental health settings. |
| 2025-06-10 | [TrajFlow: Multi-modal Motion Prediction via Flow Matching](http://arxiv.org/abs/2506.08541v1) | Qi Yan, Brian Zhang et al. | Efficient and accurate motion prediction is crucial for ensuring safety and informed decision-making in autonomous driving, particularly under dynamic real-world conditions that necessitate multi-modal forecasts. We introduce TrajFlow, a novel flow matching-based motion prediction framework that addresses the scalability and efficiency challenges of existing generative trajectory prediction methods. Unlike conventional generative approaches that employ i.i.d. sampling and require multiple inference passes to capture diverse outcomes, TrajFlow predicts multiple plausible future trajectories in a single pass, significantly reducing computational overhead while maintaining coherence across predictions. Moreover, we propose a ranking loss based on the Plackett-Luce distribution to improve uncertainty estimation of predicted trajectories. Additionally, we design a self-conditioning training technique that reuses the model's own predictions to construct noisy inputs during a second forward pass, thereby improving generalization and accelerating inference. Extensive experiments on the large-scale Waymo Open Motion Dataset (WOMD) demonstrate that TrajFlow achieves state-of-the-art performance across various key metrics, underscoring its effectiveness for safety-critical autonomous driving applications. The code and other details are available on the project website https://traj-flow.github.io/. |
| 2025-06-10 | [AsFT: Anchoring Safety During LLM Fine-Tuning Within Narrow Safety Basin](http://arxiv.org/abs/2506.08473v1) | Shuo Yang, Qihui Zhang et al. | Large language models (LLMs) are vulnerable to safety risks during fine-tuning, where small amounts of malicious or harmless data can compromise safeguards. In this paper, building on the concept of alignment direction -- defined by the weight difference between aligned and unaligned models -- we observe that perturbations along this direction preserve model safety. In contrast, perturbations along directions orthogonal to this alignment are strongly linked to harmful direction perturbations, rapidly degrading safety and framing the parameter space as a narrow safety basin. Based on this insight, we propose a methodology for safety fine-tuning called AsFT (Anchoring Safety in Fine-Tuning), which integrates a regularization term into the training objective. This term uses the alignment direction as an anchor to suppress updates in harmful directions, ensuring that fine-tuning is constrained within the narrow safety basin. Extensive experiments on multiple datasets show that AsFT outperforms Safe LoRA, reducing harmful behavior by 7.60 percent, improving model performance by 3.44 percent, and maintaining robust performance across various experimental settings. Code is available at https://github.com/PKU-YuanGroup/AsFT |
| 2025-06-10 | [Diffusion Models for Safety Validation of Autonomous Driving Systems](http://arxiv.org/abs/2506.08459v1) | Juanran Wang, Marc R. Schlichting et al. | Safety validation of autonomous driving systems is extremely challenging due to the high risks and costs of real-world testing as well as the rarity and diversity of potential failures. To address these challenges, we train a denoising diffusion model to generate potential failure cases of an autonomous vehicle given any initial traffic state. Experiments on a four-way intersection problem show that in a variety of scenarios, the diffusion model can generate realistic failure samples while capturing a wide variety of potential failures. Our model does not require any external training dataset, can perform training and inference with modest computing resources, and does not assume any prior knowledge of the system under test, with applicability to safety validation for traffic intersections. |
| 2025-06-10 | [Offline RL with Smooth OOD Generalization in Convex Hull and its Neighborhood](http://arxiv.org/abs/2506.08417v1) | Qingmao Yao, Zhichao Lei et al. | Offline Reinforcement Learning (RL) struggles with distributional shifts, leading to the $Q$-value overestimation for out-of-distribution (OOD) actions. Existing methods address this issue by imposing constraints; however, they often become overly conservative when evaluating OOD regions, which constrains the $Q$-function generalization. This over-constraint issue results in poor $Q$-value estimation and hinders policy improvement. In this paper, we introduce a novel approach to achieve better $Q$-value estimation by enhancing $Q$-function generalization in OOD regions within Convex Hull and its Neighborhood (CHN). Under the safety generalization guarantees of the CHN, we propose the Smooth Bellman Operator (SBO), which updates OOD $Q$-values by smoothing them with neighboring in-sample $Q$-values. We theoretically show that SBO approximates true $Q$-values for both in-sample and OOD actions within the CHN. Our practical algorithm, Smooth Q-function OOD Generalization (SQOG), empirically alleviates the over-constraint issue, achieving near-accurate $Q$-value estimation. On the D4RL benchmarks, SQOG outperforms existing state-of-the-art methods in both performance and computational efficiency. |

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



