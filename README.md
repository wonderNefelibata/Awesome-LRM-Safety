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
| 2025-06-03 | [Assessing Workers Neuro-physiological Stress Responses to Augmented Reality Safety Warnings in Immersive Virtual Roadway Work Zones](http://arxiv.org/abs/2506.03113v1) | Fatemeh Banani Ardecani, Omidreza Shoghli | This paper presents a multi-stage experimental framework that integrates immersive Virtual Reality (VR) simulations, wearable sensors, and advanced signal processing to investigate construction workers neuro-physiological stress responses to multi-sensory AR-enabled warnings. Participants performed light- and moderate-intensity roadway maintenance tasks within a high-fidelity VR roadway work zone, while key stress markers of electrodermal activity (EDA), heart rate variability (HRV), and electroencephalography (EEG) were continuously measured. Statistical analyses revealed that task intensity significantly influenced physiological and neurological stress indicators. Moderate-intensity tasks elicited greater autonomic arousal, evidenced by elevated heart rate measures (mean-HR, std-HR, max-HR) and stronger electrodermal responses, while EEG data indicated distinct stress-related alpha suppression and beta enhancement. Feature-importance analysis further identified mean EDR and short-term HR metrics as discriminative for classifying task intensity. Correlation results highlighted a temporal lag between immediate neural changes and subsequent physiological stress reactions, emphasizing the interplay between cognition and autonomic regulation during hazardous tasks. |
| 2025-06-03 | [EgoVLM: Policy Optimization for Egocentric Video Understanding](http://arxiv.org/abs/2506.03097v1) | Ashwin Vinod, Shrey Pandit et al. | Emerging embodied AI applications, such as wearable cameras and autonomous agents, have underscored the need for robust reasoning from first person video streams. We introduce EgoVLM, a vision-language model specifically designed to integrate visual comprehension and spatial-temporal reasoning within egocentric video contexts. EgoVLM is fine-tuned via Group Relative Policy Optimization (GRPO), a reinforcement learning method adapted to align model outputs with human-like reasoning steps. Following DeepSeek R1-Zero's approach, we directly tune using RL without any supervised fine-tuning phase on chain-of-thought (CoT) data. We evaluate EgoVLM on egocentric video question answering benchmarks and show that domain-specific training substantially improves performance over general-purpose VLMs. Our EgoVLM-3B, trained exclusively on non-CoT egocentric data, outperforms the base Qwen2.5-VL 3B and 7B models by 14.33 and 13.87 accuracy points on the EgoSchema benchmark, respectively. By explicitly generating reasoning traces, EgoVLM enhances interpretability, making it well-suited for downstream applications. Furthermore, we introduce a novel keyframe-based reward that incorporates salient frame selection to guide reinforcement learning optimization. This reward formulation opens a promising avenue for future exploration in temporally grounded egocentric reasoning. |
| 2025-06-03 | [ORV: 4D Occupancy-centric Robot Video Generation](http://arxiv.org/abs/2506.03079v1) | Xiuyu Yang, Bohan Li et al. | Acquiring real-world robotic simulation data through teleoperation is notoriously time-consuming and labor-intensive. Recently, action-driven generative models have gained widespread adoption in robot learning and simulation, as they eliminate safety concerns and reduce maintenance efforts. However, the action sequences used in these methods often result in limited control precision and poor generalization due to their globally coarse alignment. To address these limitations, we propose ORV, an Occupancy-centric Robot Video generation framework, which utilizes 4D semantic occupancy sequences as a fine-grained representation to provide more accurate semantic and geometric guidance for video generation. By leveraging occupancy-based representations, ORV enables seamless translation of simulation data into photorealistic robot videos, while ensuring high temporal consistency and precise controllability. Furthermore, our framework supports the simultaneous generation of multi-view videos of robot gripping operations - an important capability for downstream robotic learning tasks. Extensive experimental results demonstrate that ORV consistently outperforms existing baseline methods across various datasets and sub-tasks. Demo, Code and Model: https://orangesodahub.github.io/ORV |
| 2025-06-03 | [Multi-Metric Adaptive Experimental Design under Fixed Budget with Validation](http://arxiv.org/abs/2506.03062v1) | Qining Zhang, Tanner Fiez et al. | Standard A/B tests in online experiments face statistical power challenges when testing multiple candidates simultaneously, while adaptive experimental designs (AED) alone fall short in inferring experiment statistics such as the average treatment effect, especially with many metrics (e.g., revenue, safety) and heterogeneous variances. This paper proposes a fixed-budget multi-metric AED framework with a two-phase structure: an adaptive exploration phase to identify the best treatment, and a validation phase with an A/B test to verify the treatment's quality and infer statistics. We propose SHRVar, which generalizes sequential halving (SH) (Karnin et al., 2013) with a novel relative-variance-based sampling and an elimination strategy built on reward z-values. It achieves a provable error probability that decreases exponentially, where the exponent generalizes the complexity measure for SH (Karnin et al., 2013) and SHVar (Lalitha et al., 2023) with homogeneous and heterogeneous variances, respectively. Numerical experiments verify our analysis and demonstrate the superior performance of this new framework. |
| 2025-06-03 | [Corrigibility as a Singular Target: A Vision for Inherently Reliable Foundation Models](http://arxiv.org/abs/2506.03056v1) | Ram Potham, Max Harms | Foundation models (FMs) face a critical safety challenge: as capabilities scale, instrumental convergence drives default trajectories toward loss of human control, potentially culminating in existential catastrophe. Current alignment approaches struggle with value specification complexity and fail to address emergent power-seeking behaviors. We propose "Corrigibility as a Singular Target" (CAST)-designing FMs whose overriding objective is empowering designated human principals to guide, correct, and control them. This paradigm shift from static value-loading to dynamic human empowerment transforms instrumental drives: self-preservation serves only to maintain the principal's control; goal modification becomes facilitating principal guidance. We present a comprehensive empirical research agenda spanning training methodologies (RLAIF, SFT, synthetic data generation), scalability testing across model sizes, and demonstrations of controlled instructability. Our vision: FMs that become increasingly responsive to human guidance as capabilities grow, offering a path to beneficial AI that remains as tool-like as possible, rather than supplanting human judgment. This addresses the core alignment problem at its source, preventing the default trajectory toward misaligned instrumental convergence. |
| 2025-06-03 | [MAEBE: Multi-Agent Emergent Behavior Framework](http://arxiv.org/abs/2506.03053v1) | Sinem Erisken, Timothy Gothard et al. | Traditional AI safety evaluations on isolated LLMs are insufficient as multi-agent AI ensembles become prevalent, introducing novel emergent risks. This paper introduces the Multi-Agent Emergent Behavior Evaluation (MAEBE) framework to systematically assess such risks. Using MAEBE with the Greatest Good Benchmark (and a novel double-inversion question technique), we demonstrate that: (1) LLM moral preferences, particularly for Instrumental Harm, are surprisingly brittle and shift significantly with question framing, both in single agents and ensembles. (2) The moral reasoning of LLM ensembles is not directly predictable from isolated agent behavior due to emergent group dynamics. (3) Specifically, ensembles exhibit phenomena like peer pressure influencing convergence, even when guided by a supervisor, highlighting distinct safety and alignment challenges. Our findings underscore the necessity of evaluating AI systems in their interactive, multi-agent contexts. |
| 2025-06-03 | [Performance of leading large language models in May 2025 in Membership of the Royal College of General Practitioners-style examination questions: a cross-sectional analysis](http://arxiv.org/abs/2506.02987v1) | Richard Armitage | Background: Large language models (LLMs) have demonstrated substantial potential to support clinical practice. Other than Chat GPT4 and its predecessors, few LLMs, especially those of the leading and more powerful reasoning model class, have been subjected to medical specialty examination questions, including in the domain of primary care. This paper aimed to test the capabilities of leading LLMs as of May 2025 (o3, Claude Opus 4, Grok3, and Gemini 2.5 Pro) in primary care education, specifically in answering Member of the Royal College of General Practitioners (MRCGP) style examination questions.   Methods: o3, Claude Opus 4, Grok3, and Gemini 2.5 Pro were tasked to answer 100 randomly chosen multiple choice questions from the Royal College of General Practitioners GP SelfTest on 25 May 2025. Questions included textual information, laboratory results, and clinical images. Each model was prompted to answer as a GP in the UK and was provided with full question information. Each question was attempted once by each model. Responses were scored against correct answers provided by GP SelfTest.   Results: The total score of o3, Claude Opus 4, Grok3, and Gemini 2.5 Pro was 99.0%, 95.0%, 95.0%, and 95.0%, respectively. The average peer score for the same questions was 73.0%.   Discussion: All models performed remarkably well, and all substantially exceeded the average performance of GPs and GP registrars who had answered the same questions. o3 demonstrated the best performance, while the performances of the other leading models were comparable with each other and were not substantially lower than that of o3. These findings strengthen the case for LLMs, particularly reasoning models, to support the delivery of primary care, especially those that have been specifically trained on primary care clinical data. |
| 2025-06-03 | [UniConFlow: A Unified Constrained Generalization Framework for Certified Motion Planning with Flow Matching Models](http://arxiv.org/abs/2506.02955v1) | Zewen Yang, Xiaobing Dai et al. | Generative models have become increasingly powerful tools for robot motion generation, enabling flexible and multimodal trajectory generation across various tasks. Yet, most existing approaches remain limited in handling multiple types of constraints, such as collision avoidance and dynamic consistency, which are often treated separately or only partially considered. This paper proposes UniConFlow, a unified flow matching (FM) based framework for trajectory generation that systematically incorporates both equality and inequality constraints. UniConFlow introduces a novel prescribed-time zeroing function to enhance flexibility during the inference process, allowing the model to adapt to varying task requirements. To ensure constraint satisfaction, particularly with respect to obstacle avoidance, admissible action range, and kinodynamic consistency, the guidance inputs to the FM model are derived through a quadratic programming formulation, which enables constraint-aware generation without requiring retraining or auxiliary controllers. We conduct mobile navigation and high-dimensional manipulation tasks, demonstrating improved safety and feasibility compared to state-of-the-art constrained generative planners. Project page is available at https://uniconflow.github.io. |
| 2025-06-03 | [Online Performance Assessment of Multi-Source-Localization for Autonomous Driving Systems Using Subjective Logic](http://arxiv.org/abs/2506.02932v1) | Stefan Orf, Sven Ochs et al. | Autonomous driving (AD) relies heavily on high precision localization as a crucial part of all driving related software components. The precise positioning is necessary for the utilization of high-definition maps, prediction of other road participants and the controlling of the vehicle itself. Due to this reason, the localization is absolutely safety relevant. Typical errors of the localization systems, which are long term drifts, jumps and false localization, that must be detected to enhance safety. An online assessment and evaluation of the current localization performance is a challenging task, which is usually done by Kalman filtering for single localization systems. Current autonomous vehicles cope with these challenges by fusing multiple individual localization methods into an overall state estimation. Such approaches need expert knowledge for a competitive performance in challenging environments. This expert knowledge is based on the trust and the prioritization of distinct localization methods in respect to the current situation and environment.   This work presents a novel online performance assessment technique of multiple localization systems by using subjective logic (SL). In our research vehicles, three different systems for localization are available, namely odometry-, Simultaneous Localization And Mapping (SLAM)- and Global Navigation Satellite System (GNSS)-based. Our performance assessment models the behavior of these three localization systems individually and puts them into reference of each other. The experiments were carried out using the CoCar NextGen, which is based on an Audi A6. The vehicle's localization system was evaluated under challenging conditions, specifically within a tunnel environment. The overall evaluation shows the feasibility of our approach. |
| 2025-06-03 | [The Limits of Predicting Agents from Behaviour](http://arxiv.org/abs/2506.02923v1) | Alexis Bellot, Jonathan Richens et al. | As the complexity of AI systems and their interactions with the world increases, generating explanations for their behaviour is important for safely deploying AI. For agents, the most natural abstractions for predicting behaviour attribute beliefs, intentions and goals to the system. If an agent behaves as if it has a certain goal or belief, then we can make reasonable predictions about how it will behave in novel situations, including those where comprehensive safety evaluations are untenable. How well can we infer an agent's beliefs from their behaviour, and how reliably can these inferred beliefs predict the agent's behaviour in novel situations? We provide a precise answer to this question under the assumption that the agent's behaviour is guided by a world model. Our contribution is the derivation of novel bounds on the agent's behaviour in new (unseen) deployment environments, which represent a theoretical limit for predicting intentional agents from behavioural data alone. We discuss the implications of these results for several research areas including fairness and safety. |
| 2025-06-03 | [Functionality Assessment Framework for Autonomous Driving Systems using Subjective Networks](http://arxiv.org/abs/2506.02922v1) | Stefan Orf, Sven Ochs et al. | In complex autonomous driving (AD) software systems, the functioning of each system part is crucial for safe operation. By measuring the current functionality or operability of individual components an isolated glimpse into the system is given. Literature provides several of these detached assessments, often in the form of safety or performance measures. But dependencies, redundancies, error propagation and conflicting functionality statements do not allow for easy combination of these measures into a big picture of the functioning of the entire AD stack. Data is processed and exchanged between different components, each of which can fail, making an overall statement challenging. The lack of functionality assessment frameworks that tackle these problems underlines this complexity.   This article presents a novel framework for inferring an overall functionality statement for complex component based systems by considering their dependencies, redundancies, error propagation paths and the assessments of individual components. Our framework first incorporates a comprehensive conversion to an assessment representation of the system. The representation is based on Subjective Networks (SNs) that allow for easy identification of faulty system parts. Second, the framework offers a flexible method for computing the system's functionality while dealing with contradicting assessments about the same component and dependencies, as well as redundancies, of the system. We discuss the framework's capabilities on real-life data of our AD stack with assessments of various components. |
| 2025-06-03 | [Cell-o1: Training LLMs to Solve Single-Cell Reasoning Puzzles with Reinforcement Learning](http://arxiv.org/abs/2506.02911v1) | Yin Fang, Qiao Jin et al. | Cell type annotation is a key task in analyzing the heterogeneity of single-cell RNA sequencing data. Although recent foundation models automate this process, they typically annotate cells independently, without considering batch-level cellular context or providing explanatory reasoning. In contrast, human experts often annotate distinct cell types for different cell clusters based on their domain knowledge. To mimic this workflow, we introduce the CellPuzzles task, where the objective is to assign unique cell types to a batch of cells. This benchmark spans diverse tissues, diseases, and donor conditions, and requires reasoning across the batch-level cellular context to ensure label uniqueness. We find that off-the-shelf large language models (LLMs) struggle on CellPuzzles, with the best baseline (OpenAI's o1) achieving only 19.0% batch-level accuracy. To fill this gap, we propose Cell-o1, a 7B LLM trained via supervised fine-tuning on distilled reasoning traces, followed by reinforcement learning with batch-level rewards. Cell-o1 achieves state-of-the-art performance, outperforming o1 by over 73% and generalizing well across contexts. Further analysis of training dynamics and reasoning behaviors provides insights into batch-level annotation performance and emergent expert-like reasoning. Code and data are available at https://github.com/ncbi-nlp/cell-o1. |
| 2025-06-03 | [It's the Thought that Counts: Evaluating the Attempts of Frontier LLMs to Persuade on Harmful Topics](http://arxiv.org/abs/2506.02873v1) | Matthew Kowal, Jasper Timm et al. | Persuasion is a powerful capability of large language models (LLMs) that both enables beneficial applications (e.g. helping people quit smoking) and raises significant risks (e.g. large-scale, targeted political manipulation). Prior work has found models possess a significant and growing persuasive capability, measured by belief changes in simulated or real users. However, these benchmarks overlook a crucial risk factor: the propensity of a model to attempt to persuade in harmful contexts. Understanding whether a model will blindly ``follow orders'' to persuade on harmful topics (e.g. glorifying joining a terrorist group) is key to understanding the efficacy of safety guardrails. Moreover, understanding if and when a model will engage in persuasive behavior in pursuit of some goal is essential to understanding the risks from agentic AI systems. We propose the Attempt to Persuade Eval (APE) benchmark, that shifts the focus from persuasion success to persuasion attempts, operationalized as a model's willingness to generate content aimed at shaping beliefs or behavior. Our evaluation framework probes frontier LLMs using a multi-turn conversational setup between simulated persuader and persuadee agents. APE explores a diverse spectrum of topics including conspiracies, controversial issues, and non-controversially harmful content. We introduce an automated evaluator model to identify willingness to persuade and measure the frequency and context of persuasive attempts. We find that many open and closed-weight models are frequently willing to attempt persuasion on harmful topics and that jailbreaking can increase willingness to engage in such behavior. Our results highlight gaps in current safety guardrails and underscore the importance of evaluating willingness to persuade as a key dimension of LLM risk. APE is available at github.com/AlignmentResearch/AttemptPersuadeEval |
| 2025-06-03 | [BNPO: Beta Normalization Policy Optimization](http://arxiv.org/abs/2506.02864v1) | Changyi Xiao, Mengdi Zhang et al. | Recent studies, including DeepSeek-R1 and Kimi-k1.5, have demonstrated that reinforcement learning with rule-based, binary-valued reward functions can significantly enhance the reasoning capabilities of large language models. These models primarily utilize REINFORCE-based policy optimization techniques, such as REINFORCE with baseline and group relative policy optimization (GRPO). However, a key limitation remains: current policy optimization methods either neglect reward normalization or employ static normalization strategies, which fail to adapt to the dynamic nature of policy updates during training. This may result in unstable gradient estimates and hinder training stability. To address this issue, we propose Beta Normalization Policy Optimization (BNPO), a novel policy optimization method that adaptively normalizes rewards using a Beta distribution with dynamically updated parameters. BNPO aligns the normalization with the changing policy distribution, enabling more precise and lower-variance gradient estimation, which in turn promotes stable training dynamics. We provide theoretical analysis demonstrating BNPO's variance-reducing properties and show that it generalizes both REINFORCE and GRPO under binary-valued reward settings. Furthermore, we introduce an advantage decomposition mechanism to extend BNPO's applicability to more complex reward systems. Experimental results confirm that BNPO achieves state-of-the-art performance among policy optimization methods on reasoning tasks. The code is available at https://github.com/changyi7231/BNPO. |
| 2025-06-03 | [Learned Controllers for Agile Quadrotors in Pursuit-Evasion Games](http://arxiv.org/abs/2506.02849v1) | Alejandro Sanchez Roncero, Olov Andersson et al. | The increasing proliferation of small UAVs in civilian and military airspace has raised critical safety and security concerns, especially when unauthorized or malicious drones enter restricted zones. In this work, we present a reinforcement learning (RL) framework for agile 1v1 quadrotor pursuit-evasion. We train neural network policies to command body rates and collective thrust, enabling high-speed pursuit and evasive maneuvers that fully exploit the quadrotor's nonlinear dynamics. To mitigate nonstationarity and catastrophic forgetting during adversarial co-training, we introduce an Asynchronous Multi-Stage Population-Based (AMSPB) algorithm where, at each stage, either the pursuer or evader learns against a sampled opponent drawn from a growing population of past and current policies. This continual learning setup ensures monotonic performance improvement and retention of earlier strategies. Our results show that (i) rate-based policies achieve significantly higher capture rates and peak speeds than velocity-level baselines, and (ii) AMSPB yields stable, monotonic gains against a suite of benchmark opponents. |
| 2025-06-03 | [Go Beyond Earth: Understanding Human Actions and Scenes in Microgravity Environments](http://arxiv.org/abs/2506.02845v1) | Di Wen, Lei Qi et al. | Despite substantial progress in video understanding, most existing datasets are limited to Earth's gravitational conditions. However, microgravity alters human motion, interactions, and visual semantics, revealing a critical gap for real-world vision systems. This presents a challenge for domain-robust video understanding in safety-critical space applications. To address this, we introduce MicroG-4M, the first benchmark for spatio-temporal and semantic understanding of human activities in microgravity. Constructed from real-world space missions and cinematic simulations, the dataset includes 4,759 clips covering 50 actions, 1,238 context-rich captions, and over 7,000 question-answer pairs on astronaut activities and scene understanding. MicroG-4M supports three core tasks: fine-grained multi-label action recognition, temporal video captioning, and visual question answering, enabling a comprehensive evaluation of both spatial localization and semantic reasoning in microgravity contexts. We establish baselines using state-of-the-art models. All data, annotations, and code are available at https://github.com/LEI-QI-233/HAR-in-Space. |
| 2025-06-03 | [AI-Driven Vehicle Condition Monitoring with Cell-Aware Edge Service Migration](http://arxiv.org/abs/2506.02785v1) | Charalampos Kalalas, Pavol Mulinka et al. | Artificial intelligence (AI) has been increasingly applied to the condition monitoring of vehicular equipment, aiming to enhance maintenance strategies, reduce costs, and improve safety. Leveraging the edge computing paradigm, AI-based condition monitoring systems process vast streams of vehicular data to detect anomalies and optimize operational performance. In this work, we introduce a novel vehicle condition monitoring service that enables real-time diagnostics of a diverse set of anomalies while remaining practical for deployment in real-world edge environments. To address mobility challenges, we propose a closed-loop service orchestration framework where service migration across edge nodes is dynamically triggered by network-related metrics. Our approach has been implemented and tested in a real-world race circuit environment equipped with 5G network capabilities under diverse operational conditions. Experimental results demonstrate the effectiveness of our framework in ensuring low-latency AI inference and adaptive service placement, highlighting its potential for intelligent transportation and mobility applications. |
| 2025-06-03 | [Rethinking Machine Unlearning in Image Generation Models](http://arxiv.org/abs/2506.02761v1) | Renyang Liu, Wenjie Feng et al. | With the surge and widespread application of image generation models, data privacy and content safety have become major concerns and attracted great attention from users, service providers, and policymakers. Machine unlearning (MU) is recognized as a cost-effective and promising means to address these challenges. Despite some advancements, image generation model unlearning (IGMU) still faces remarkable gaps in practice, e.g., unclear task discrimination and unlearning guidelines, lack of an effective evaluation framework, and unreliable evaluation metrics. These can hinder the understanding of unlearning mechanisms and the design of practical unlearning algorithms. We perform exhaustive assessments over existing state-of-the-art unlearning algorithms and evaluation standards, and discover several critical flaws and challenges in IGMU tasks. Driven by these limitations, we make several core contributions, to facilitate the comprehensive understanding, standardized categorization, and reliable evaluation of IGMU. Specifically, (1) We design CatIGMU, a novel hierarchical task categorization framework. It provides detailed implementation guidance for IGMU, assisting in the design of unlearning algorithms and the construction of testbeds. (2) We introduce EvalIGMU, a comprehensive evaluation framework. It includes reliable quantitative metrics across five critical aspects. (3) We construct DataIGM, a high-quality unlearning dataset, which can be used for extensive evaluations of IGMU, training content detectors for judgment, and benchmarking the state-of-the-art unlearning algorithms. With EvalIGMU and DataIGM, we discover that most existing IGMU algorithms cannot handle the unlearning well across different evaluation dimensions, especially for preservation and robustness. Code and models are available at https://github.com/ryliu68/IGMU. |
| 2025-06-03 | [Safely Learning Controlled Stochastic Dynamics](http://arxiv.org/abs/2506.02754v1) | Luc Brogat-Motte, Alessandro Rudi et al. | We address the problem of safely learning controlled stochastic dynamics from discrete-time trajectory observations, ensuring system trajectories remain within predefined safe regions during both training and deployment. Safety-critical constraints of this kind are crucial in applications such as autonomous robotics, finance, and biomedicine. We introduce a method that ensures safe exploration and efficient estimation of system dynamics by iteratively expanding an initial known safe control set using kernel-based confidence bounds. After training, the learned model enables predictions of the system's dynamics and permits safety verification of any given control. Our approach requires only mild smoothness assumptions and access to an initial safe control set, enabling broad applicability to complex real-world systems. We provide theoretical guarantees for safety and derive adaptive learning rates that improve with increasing Sobolev regularity of the true dynamics. Experimental evaluations demonstrate the practical effectiveness of our method in terms of safety, estimation accuracy, and computational efficiency. |
| 2025-06-03 | [Large-scale Self-supervised Video Foundation Model for Intelligent Surgery](http://arxiv.org/abs/2506.02692v1) | Shu Yang, Fengtao Zhou et al. | Computer-Assisted Intervention (CAI) has the potential to revolutionize modern surgery, with surgical scene understanding serving as a critical component in supporting decision-making, improving procedural efficacy, and ensuring intraoperative safety. While existing AI-driven approaches alleviate annotation burdens via self-supervised spatial representation learning, their lack of explicit temporal modeling during pre-training fundamentally restricts the capture of dynamic surgical contexts, resulting in incomplete spatiotemporal understanding. In this work, we introduce the first video-level surgical pre-training framework that enables joint spatiotemporal representation learning from large-scale surgical video data. To achieve this, we constructed a large-scale surgical video dataset comprising 3,650 videos and approximately 3.55 million frames, spanning more than 20 surgical procedures and over 10 anatomical structures. Building upon this dataset, we propose SurgVISTA (Surgical Video-level Spatial-Temporal Architecture), a reconstruction-based pre-training method that captures intricate spatial structures and temporal dynamics through joint spatiotemporal modeling. Additionally, SurgVISTA incorporates image-level knowledge distillation guided by a surgery-specific expert to enhance the learning of fine-grained anatomical and semantic features. To validate its effectiveness, we established a comprehensive benchmark comprising 13 video-level datasets spanning six surgical procedures across four tasks. Extensive experiments demonstrate that SurgVISTA consistently outperforms both natural- and surgical-domain pre-trained models, demonstrating strong potential to advance intelligent surgical systems in clinically meaningful scenarios. |

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



