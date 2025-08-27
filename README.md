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
| 2025-08-26 | [Get Global Guarantees: On the Probabilistic Nature of Perturbation Robustness](http://arxiv.org/abs/2508.19183v1) | Wenchuan Mu, Kwan Hui Lim | In safety-critical deep learning applications, robustness measures the ability of neural models that handle imperceptible perturbations in input data, which may lead to potential safety hazards. Existing pre-deployment robustness assessment methods typically suffer from significant trade-offs between computational cost and measurement precision, limiting their practical utility. To address these limitations, this paper conducts a comprehensive comparative analysis of existing robustness definitions and associated assessment methodologies. We propose tower robustness to evaluate robustness, which is a novel, practical metric based on hypothesis testing to quantitatively evaluate probabilistic robustness, enabling more rigorous and efficient pre-deployment assessments. Our extensive comparative evaluation illustrates the advantages and applicability of our proposed approach, thereby advancing the systematic understanding and enhancement of model robustness in safety-critical deep learning applications. |
| 2025-08-26 | [From Tabula Rasa to Emergent Abilities: Discovering Robot Skills via Real-World Unsupervised Quality-Diversity](http://arxiv.org/abs/2508.19172v1) | Luca Grillotti, Lisa Coiffard et al. | Autonomous skill discovery aims to enable robots to acquire diverse behaviors without explicit supervision. Learning such behaviors directly on physical hardware remains challenging due to safety and data efficiency constraints. Existing methods, including Quality-Diversity Actor-Critic (QDAC), require manually defined skill spaces and carefully tuned heuristics, limiting real-world applicability. We propose Unsupervised Real-world Skill Acquisition (URSA), an extension of QDAC that enables robots to autonomously discover and master diverse, high-performing skills directly in the real world. We demonstrate that URSA successfully discovers diverse locomotion skills on a Unitree A1 quadruped in both simulation and the real world. Our approach supports both heuristic-driven skill discovery and fully unsupervised settings. We also show that the learned skill repertoire can be reused for downstream tasks such as real-world damage adaptation, where URSA outperforms all baselines in 5 out of 9 simulated and 3 out of 5 real-world damage scenarios. Our results establish a new framework for real-world robot learning that enables continuous skill discovery with limited human intervention, representing a significant step toward more autonomous and adaptable robotic systems. Demonstration videos are available at http://adaptive-intelligent-robotics.github.io/URSA . |
| 2025-08-26 | [MATRIX: Multi-Agent simulaTion fRamework for safe Interactions and conteXtual clinical conversational evaluation](http://arxiv.org/abs/2508.19163v1) | Ernest Lim, Yajie Vera He et al. | Despite the growing use of large language models (LLMs) in clinical dialogue systems, existing evaluations focus on task completion or fluency, offering little insight into the behavioral and risk management requirements essential for safety-critical systems. This paper presents MATRIX (Multi-Agent simulaTion fRamework for safe Interactions and conteXtual clinical conversational evaluation), a structured, extensible framework for safety-oriented evaluation of clinical dialogue agents.   MATRIX integrates three components: (1) a safety-aligned taxonomy of clinical scenarios, expected system behaviors and failure modes derived through structured safety engineering methods; (2) BehvJudge, an LLM-based evaluator for detecting safety-relevant dialogue failures, validated against expert clinician annotations; and (3) PatBot, a simulated patient agent capable of producing diverse, scenario-conditioned responses, evaluated for realism and behavioral fidelity with human factors expertise, and a patient-preference study.   Across three experiments, we show that MATRIX enables systematic, scalable safety evaluation. BehvJudge with Gemini 2.5-Pro achieves expert-level hazard detection (F1 0.96, sensitivity 0.999), outperforming clinicians in a blinded assessment of 240 dialogues. We also conducted one of the first realism analyses of LLM-based patient simulation, showing that PatBot reliably simulates realistic patient behavior in quantitative and qualitative evaluations. Using MATRIX, we demonstrate its effectiveness in benchmarking five LLM agents across 2,100 simulated dialogues spanning 14 hazard scenarios and 10 clinical domains.   MATRIX is the first framework to unify structured safety engineering with scalable, validated conversational AI evaluation, enabling regulator-aligned safety auditing. We release all evaluation tools, prompts, structured scenarios, and datasets. |
| 2025-08-26 | [Safe Navigation under State Uncertainty: Online Adaptation for Robust Control Barrier Functions](http://arxiv.org/abs/2508.19159v1) | Ersin Das, Rahal Nanayakkara et al. | Measurements and state estimates are often imperfect in control practice, posing challenges for safety-critical applications, where safety guarantees rely on accurate state information. In the presence of estimation errors, several prior robust control barrier function (R-CBF) formulations have imposed strict conditions on the input. These methods can be overly conservative and can introduce issues such as infeasibility, high control effort, etc. This work proposes a systematic method to improve R-CBFs, and demonstrates its advantages on a tracked vehicle that navigates among multiple obstacles. A primary contribution is a new optimization-based online parameter adaptation scheme that reduces the conservativeness of existing R-CBFs. In order to reduce the complexity of the parameter optimization, we merge several safety constraints into one unified numerical CBF via Poisson's equation. We further address the dual relative degree issue that typically causes difficulty in vehicle tracking. Experimental trials demonstrate the overall performance improvement of our approach over existing formulations. |
| 2025-08-26 | [ZeST: an LLM-based Zero-Shot Traversability Navigation for Unknown Environments](http://arxiv.org/abs/2508.19131v1) | Shreya Gummadi, Mateus V. Gasparino et al. | The advancement of robotics and autonomous navigation systems hinges on the ability to accurately predict terrain traversability. Traditional methods for generating datasets to train these prediction models often involve putting robots into potentially hazardous environments, posing risks to equipment and safety. To solve this problem, we present ZeST, a novel approach leveraging visual reasoning capabilities of Large Language Models (LLMs) to create a traversability map in real-time without exposing robots to danger. Our approach not only performs zero-shot traversability and mitigates the risks associated with real-world data collection but also accelerates the development of advanced navigation systems, offering a cost-effective and scalable solution. To support our findings, we present navigation results, in both controlled indoor and unstructured outdoor environments. As shown in the experiments, our method provides safer navigation when compared to other state-of-the-art methods, constantly reaching the final goal. |
| 2025-08-26 | [Reading minds on the road: decoding perceived risk in automated vehicles through 140K+ ratings](http://arxiv.org/abs/2508.19121v1) | Xiaolin He, Zirui Li et al. | Perceived risk in automated vehicles (AVs) can create the very danger that automation is meant to prevent: a frightened rider may hesitate when seconds matter, misjudge hazards, or disengage. However, measuring how perceived risk evolves in real time during driving remains challenging, leaving a gap in decoding such hidden psychological states. Here, we present a novel method to time-continuously measure and decode perceived risk. We conducted a controlled experiment where 2,164 participants viewed high-fidelity videos of common highway driving scenes and provided 141,628 discrete safety ratings. Through continuous-signal reconstruction of the discrete ratings, we obtained 236 hours of time-continuous perceived risk data - the largest perceived risk dataset to date. Leveraging this dataset, we trained deep neural networks that predict moment-by-moment perceived risk from vehicle kinematics with a mean relative error below $3\%$. Explainable AI analysis uncovers which factors determine perceived risk in real time. Our findings demonstrate a new paradigm for quantifying dynamic passenger experience and psychological constructs in real time. These findings can guide the design of AVs and other machines that operate in close proximity to people, adjusting behaviour before trust erodes, and help realise automation's benefits in transport, healthcare, and service robotics. |
| 2025-08-26 | [SecureV2X: An Efficient and Privacy-Preserving System for Vehicle-to-Everything (V2X) Applications](http://arxiv.org/abs/2508.19115v1) | Joshua Lee, Ali Arastehfard et al. | Autonomous driving and V2X technologies have developed rapidly in the past decade, leading to improved safety and efficiency in modern transportation. These systems interact with extensive networks of vehicles, roadside infrastructure, and cloud resources to support their machine learning capabilities. However, the widespread use of machine learning in V2X systems raises issues over the privacy of the data involved. This is particularly concerning for smart-transit and driver safety applications which can implicitly reveal user locations or explicitly disclose medical data such as EEG signals. To resolve these issues, we propose SecureV2X, a scalable, multi-agent system for secure neural network inferences deployed between the server and each vehicle. Under this setting, we study two multi-agent V2X applications: secure drowsiness detection, and secure red-light violation detection. Our system achieves strong performance relative to baselines, and scales efficiently to support a large number of secure computation interactions simultaneously. For instance, SecureV2X is $9.4 \times$ faster, requires $143\times$ fewer computational rounds, and involves $16.6\times$ less communication on drowsiness detection compared to other secure systems. Moreover, it achieves a runtime nearly $100\times$ faster than state-of-the-art benchmarks in object detection tasks for red light violation detection. |
| 2025-08-26 | [Reasoning LLMs in the Medical Domain: A Literature Survey](http://arxiv.org/abs/2508.19097v1) | Armin Berger, Sarthak Khanna et al. | The emergence of advanced reasoning capabilities in Large Language Models (LLMs) marks a transformative development in healthcare applications. Beyond merely expanding functional capabilities, these reasoning mechanisms enhance decision transparency and explainability-critical requirements in medical contexts. This survey examines the transformation of medical LLMs from basic information retrieval tools to sophisticated clinical reasoning systems capable of supporting complex healthcare decisions. We provide a thorough analysis of the enabling technological foundations, with a particular focus on specialized prompting techniques like Chain-of-Thought and recent breakthroughs in Reinforcement Learning exemplified by DeepSeek-R1. Our investigation evaluates purpose-built medical frameworks while also examining emerging paradigms such as multi-agent collaborative systems and innovative prompting architectures. The survey critically assesses current evaluation methodologies for medical validation and addresses persistent challenges in field interpretation limitations, bias mitigation strategies, patient safety frameworks, and integration of multimodal clinical data. Through this survey, we seek to establish a roadmap for developing reliable LLMs that can serve as effective partners in clinical practice and medical research. |
| 2025-08-26 | [Investigating Advanced Reasoning of Large Language Models via Black-Box Interaction](http://arxiv.org/abs/2508.19035v1) | Congchi Yin, Tianyi Wu et al. | Existing tasks fall short in evaluating reasoning ability of Large Language Models (LLMs) in an interactive, unknown environment. This deficiency leads to the isolated assessment of deductive, inductive, and abductive reasoning, neglecting the integrated reasoning process that is indispensable for humans discovery of real world. We introduce a novel evaluation paradigm, \textit{black-box interaction}, to tackle this challenge. A black-box is defined by a hidden function that maps a specific set of inputs to outputs. LLMs are required to unravel the hidden function behind the black-box by interacting with it in given exploration turns, and reasoning over observed input-output pairs. Leveraging this idea, we build the \textsc{Oracle} benchmark which comprises 6 types of black-box task and 96 black-boxes. 19 modern LLMs are benchmarked. o3 ranks first in 5 of the 6 tasks, achieving over 70\% accuracy on most easy black-boxes. But it still struggles with some hard black-box tasks, where its average performance drops below 40\%. Further analysis indicates a universal difficulty among LLMs: They lack the high-level planning capability to develop efficient and adaptive exploration strategies for hypothesis refinement. |
| 2025-08-26 | [On the Generalisation of Koopman Representations for Chaotic System Control](http://arxiv.org/abs/2508.18954v1) | Kyriakos Hjikakou, Juan Diego Cardenas Cartagena et al. | This paper investigates the generalisability of Koopman-based representations for chaotic dynamical systems, focusing on their transferability across prediction and control tasks. Using the Lorenz system as a testbed, we propose a three-stage methodology: learning Koopman embeddings through autoencoding, pre-training a transformer on next-state prediction, and fine-tuning for safety-critical control. Our results show that Koopman embeddings outperform both standard and physics-informed PCA baselines, achieving accurate and data-efficient performance. Notably, fixing the pre-trained transformer weights during fine-tuning leads to no performance degradation, indicating that the learned representations capture reusable dynamical structure rather than task-specific patterns. These findings support the use of Koopman embeddings as a foundation for multi-task learning in physics-informed machine learning. A project page is available at https://kikisprdx.github.io/. |
| 2025-08-26 | [VisionSafeEnhanced VPC: Cautious Predictive Control with Visibility Constraints under Uncertainty for Autonomous Robotic Surgery](http://arxiv.org/abs/2508.18937v1) | Wang Jiayin, Wei Yanran et al. | Autonomous control of the laparoscope in robot-assisted Minimally Invasive Surgery (MIS) has received considerable research interest due to its potential to improve surgical safety. Despite progress in pixel-level Image-Based Visual Servoing (IBVS) control, the requirement of continuous visibility and the existence of complex disturbances, such as parameterization error, measurement noise, and uncertainties of payloads, could degrade the surgeon's visual experience and compromise procedural safety. To address these limitations, this paper proposes VisionSafeEnhanced Visual Predictive Control (VPC), a robust and uncertainty-adaptive framework for autonomous laparoscope control that guarantees Field of View (FoV) safety under uncertainty. Firstly, Gaussian Process Regression (GPR) is utilized to perform hybrid (deterministic + stochastic) quantification of operational uncertainties including residual model uncertainties, stochastic uncertainties, and external disturbances. Based on uncertainty quantification, a novel safety aware trajectory optimization framework with probabilistic guarantees is proposed, where a uncertainty-adaptive safety Control Barrier Function (CBF) condition is given based on uncertainty propagation, and chance constraints are simultaneously formulated based on probabilistic approximation. This uncertainty aware formulation enables adaptive control effort allocation, minimizing unnecessary camera motion while maintaining robustness. The proposed method is validated through comparative simulations and experiments on a commercial surgical robot platform (MicroPort MedBot Toumai) performing a sequential multi-target lymph node dissection. Compared with baseline methods, the framework maintains near-perfect target visibility (>99.9%), reduces tracking e |
| 2025-08-26 | [Network Calculus Results for TSN: An Introduction](http://arxiv.org/abs/2508.18855v1) | Lisa Maile, Kai-Steffen Hielscher et al. | Time-Sensitive Networking (TSN) is a set of standards that enables the industry to provide real-time guarantees for time-critical communications with Ethernet hardware. TSN supports various queuing and scheduling mechanisms and allows the integration of multiple traffic types in a single network. Network Calculus (NC) can be used to calculate upper bounds for latencies and buffer sizes within these networks, for example, for safety or real-time traffic. We explain the relevance of NC for TSN-based computer communications and potential areas of application. Different NC analysis approaches have been published to examine different parts of TSN and this paper provides a survey of these publications and presents their main results, dependencies, and differences. We present a consistent presentation of the most important results and suggest an improvement to model the output of sending end-devices. To ease access to the current research status, we introduce a common notation to show how all results depend on each other and also identify common assumptions. Thus, we offer a comprehensive overview of NC for industrial networks and identify possible areas for future work. |
| 2025-08-26 | [Closed-Form Input Design for Identification under Output Feedback with Perturbation Constraints](http://arxiv.org/abs/2508.18813v1) | Jingwei Hu, Dave Zachariah et al. | In many applications, system identification experiments must be performed under output feedback to ensure safety or to maintain system operation. In this paper, we consider the online design of informative experiments for ARMAX models by applying a bounded perturbation to the input signal generated by a fixed output feedback controller. Specifically, the design constrains the resulting output perturbation within user-specified limits and can be efficiently computed in closed form. We demonstrate the effectiveness of the method in two numerical experiments. |
| 2025-08-26 | [Governance-as-a-Service: A Multi-Agent Framework for AI System Compliance and Policy Enforcement](http://arxiv.org/abs/2508.18765v1) | Helen Pervez, Suyash Gaurav et al. | As AI systems evolve into distributed ecosystems with autonomous execution, asynchronous reasoning, and multi-agent coordination, the absence of scalable, decoupled governance poses a structural risk. Existing oversight mechanisms are reactive, brittle, and embedded within agent architectures, making them non-auditable and hard to generalize across heterogeneous deployments.   We introduce Governance-as-a-Service (GaaS): a modular, policy-driven enforcement layer that regulates agent outputs at runtime without altering model internals or requiring agent cooperation. GaaS employs declarative rules and a Trust Factor mechanism that scores agents based on compliance and severity-weighted violations. It enables coercive, normative, and adaptive interventions, supporting graduated enforcement and dynamic trust modulation.   To evaluate GaaS, we conduct three simulation regimes with open-source models (LLaMA3, Qwen3, DeepSeek-R1) across content generation and financial decision-making. In the baseline, agents act without governance; in the second, GaaS enforces policies; in the third, adversarial agents probe robustness. All actions are intercepted, evaluated, and logged for analysis. Results show that GaaS reliably blocks or redirects high-risk behaviors while preserving throughput. Trust scores track rule adherence, isolating and penalizing untrustworthy components in multi-agent systems.   By positioning governance as a runtime service akin to compute or storage, GaaS establishes infrastructure-level alignment for interoperable agent ecosystems. It does not teach agents ethics; it enforces them. |
| 2025-08-26 | [Dynamic Count Models with Flexible Innovation Processes for Irregular Maritime Migration](http://arxiv.org/abs/2508.18716v1) | Gregor Zens, Jakub Bijak | Motivated by the dynamics of weekly sea border crossings in the Mediterranean (2015-2025) and the English Channel (2018-2025), we develop a Bayesian dynamic framework for modeling potentially heteroskedastic count time series. Building on theoretical considerations and empirical stylized facts, our approach specifies a latent log-intensity that follows a random walk driven by either heavy-tailed or stochastic volatility innovations, incorporating an explicit mechanism to separate structural from sampling zeros. Posterior inference is carried out via a straightforward Markov chain Monte Carlo algorithm. We compare alternative innovation specifications through a comprehensive out-of-sample density forecasting exercise, evaluating each model using log predictive scores and empirical coverage up to the 99th percentile of the predictive distribution. The results of two case studies reveal strong evidence for stochastic volatility in sea migration innovations, with stochastic volatility models producing particularly well-calibrated forecasts even at extreme quantiles. The model can be used to develop risk indicators and has direct policy implications for improving governance and preparedness for sea migration surges. The presented methodology readily extends to other zero-inflated non-stationary count time series applications, including epidemiological surveillance and public safety incident monitoring. |
| 2025-08-26 | [A Novel Deep Hybrid Framework with Ensemble-Based Feature Optimization for Robust Real-Time Human Activity Recognition](http://arxiv.org/abs/2508.18695v1) | Wasi Ullah, Yasir Noman Khalid et al. | Human Activity Recognition (HAR) plays a pivotal role in various applications, including smart surveillance, healthcare, assistive technologies, sports analytics, etc. However, HAR systems still face critical challenges, including high computational costs, redundant features, and limited scalability in real-time scenarios. An optimized hybrid deep learning framework is introduced that integrates a customized InceptionV3, an LSTM architecture, and a novel ensemble-based feature selection strategy. The proposed framework first extracts spatial descriptors using the customized InceptionV3 model, which captures multilevel contextual patterns, region homogeneity, and fine-grained localization cues. The temporal dependencies across frames are then modeled using LSTMs to effectively encode motion dynamics. Finally, an ensemble-based genetic algorithm with Adaptive Dynamic Fitness Sharing and Attention (ADFSA) is employed to select a compact and optimized feature set by dynamically balancing objectives such as accuracy, redundancy, uniqueness, and complexity reduction. Consequently, the selected feature subsets, which are both diverse and discriminative, enable various lightweight machine learning classifiers to achieve accurate and robust HAR in heterogeneous environments. Experimental results on the robust UCF-YouTube dataset, which presents challenges such as occlusion, cluttered backgrounds, motion dynamics, and poor illumination, demonstrate good performance. The proposed approach achieves 99.65% recognition accuracy, reduces features to as few as 7, and enhances inference time. The lightweight and scalable nature of the HAR system supports real-time deployment on edge devices such as Raspberry Pi, enabling practical applications in intelligent, resource-aware environments, including public safety, assistive technology, and autonomous monitoring systems. |
| 2025-08-26 | [PRISM: Robust VLM Alignment with Principled Reasoning for Integrated Safety in Multimodality](http://arxiv.org/abs/2508.18649v1) | Nanxi Li, Zhengyue Zhao et al. | Safeguarding vision-language models (VLMs) is a critical challenge, as existing methods often suffer from over-defense, which harms utility, or rely on shallow alignment, failing to detect complex threats that require deep reasoning. To this end, we introduce PRISM (Principled Reasoning for Integrated Safety in Multimodality), a system2-like framework that aligns VLMs by embedding a structured, safety-aware reasoning process. Our framework consists of two key components: PRISM-CoT, a dataset that teaches safety-aware chain-of-thought reasoning, and PRISM-DPO, generated via Monte Carlo Tree Search (MCTS) to further refine this reasoning through Direct Preference Optimization to help obtain a delicate safety boundary. Comprehensive evaluations demonstrate PRISM's effectiveness, achieving remarkably low attack success rates including 0.15% on JailbreakV-28K for Qwen2-VL and 90% improvement over the previous best method on VLBreak for LLaVA-1.5. PRISM also exhibits strong robustness against adaptive attacks, significantly increasing computational costs for adversaries, and generalizes effectively to out-of-distribution challenges, reducing attack success rates to just 8.70% on the challenging multi-image MIS benchmark. Remarkably, this robust defense is achieved while preserving, and in some cases enhancing, model utility. To promote reproducibility, we have made our code, data, and model weights available at https://github.com/SaFoLab-WISC/PRISM. |
| 2025-08-25 | [Language Models For Generalised PDDL Planning: Synthesising Sound and Programmatic Policies](http://arxiv.org/abs/2508.18507v1) | Dillon Z. Chen, Johannes Zenn et al. | We study the usage of language models (LMs) for planning over world models specified in the Planning Domain Definition Language (PDDL). We prompt LMs to generate Python programs that serve as generalised policies for solving PDDL problems from a given domain. Notably, our approach synthesises policies that are provably sound relative to the PDDL domain without reliance on external verifiers. We conduct experiments on competition benchmarks which show that our policies can solve more PDDL problems than PDDL planners and recent LM approaches within a fixed time and memory constraint. Our approach manifests in the LMPlan planner which can solve planning problems with several hundreds of relevant objects. Surprisingly, we observe that LMs used in our framework sometimes plan more effectively over PDDL problems written in meaningless symbols in place of natural language; e.g. rewriting (at dog kitchen) as (p2 o1 o3). This finding challenges hypotheses that LMs reason over word semantics and memorise solutions from its training corpus, and is worth further exploration. |
| 2025-08-25 | [Engineering a Digital Twin for the Monitoring and Control of Beer Fermentation Sampling](http://arxiv.org/abs/2508.18452v1) | Pierre-Emmanuel Goffi, Raphaël Tremblay et al. | Successfully engineering interactive industrial DTs is a complex task, especially when implementing services beyond passive monitoring. We present here an experience report on engineering a safety-critical digital twin (DT) for beer fermentation monitoring, which provides continual sampling and reduces manual sampling time by 91%. We document our systematic methodology and practical solutions for implementing bidirectional DTs in industrial environments. This includes our three-phase engineering approach that transforms a passive monitoring system into an interactive Type 2 DT with real-time control capabilities for pressurized systems operating at seven bar. We contribute details of multi-layered safety protocols, hardware-software integration strategies across Arduino controllers and Unity visualization, and real-time synchronization solutions. We document specific engineering challenges and solutions spanning interdisciplinary integration, demonstrating how our use of the constellation reporting framework facilitates cross-domain collaboration. Key findings include the critical importance of safety-first design, simulation-driven development, and progressive implementation strategies. Our work thus provides actionable guidance for practitioners developing DTs requiring bidirectional control in safety-critical applications. |
| 2025-08-25 | [Mining the Long Tail: A Comparative Study of Data-Centric Criticality Metrics for Robust Offline Reinforcement Learning in Autonomous Motion Planning](http://arxiv.org/abs/2508.18397v1) | Antonio Guillen-Perez | Offline Reinforcement Learning (RL) presents a promising paradigm for training autonomous vehicle (AV) planning policies from large-scale, real-world driving logs. However, the extreme data imbalance in these logs, where mundane scenarios vastly outnumber rare "long-tail" events, leads to brittle and unsafe policies when using standard uniform data sampling. In this work, we address this challenge through a systematic, large-scale comparative study of data curation strategies designed to focus the learning process on information-rich samples. We investigate six distinct criticality weighting schemes which are categorized into three families: heuristic-based, uncertainty-based, and behavior-based. These are evaluated at two temporal scales, the individual timestep and the complete scenario. We train seven goal-conditioned Conservative Q-Learning (CQL) agents with a state-of-the-art, attention-based architecture and evaluate them in the high-fidelity Waymax simulator. Our results demonstrate that all data curation methods significantly outperform the baseline. Notably, data-driven curation using model uncertainty as a signal achieves the most significant safety improvements, reducing the collision rate by nearly three-fold (from 16.0% to 5.5%). Furthermore, we identify a clear trade-off where timestep-level weighting excels at reactive safety while scenario-level weighting improves long-horizon planning. Our work provides a comprehensive framework for data curation in Offline RL and underscores that intelligent, non-uniform sampling is a critical component for building safe and reliable autonomous agents. |

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



