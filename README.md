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
| 2025-12-19 | [Humanlike AI Design Increases Anthropomorphism but Yields Divergent Outcomes on Engagement and Trust Globally](http://arxiv.org/abs/2512.17898v1) | Robin Schimmelpfennig, Mark Díaz et al. | Over a billion users across the globe interact with AI systems engineered with increasing sophistication to mimic human traits. This shift has triggered urgent debate regarding Anthropomorphism, the attribution of human characteristics to synthetic agents, and its potential to induce misplaced trust or emotional dependency. However, the causal link between more humanlike AI design and subsequent effects on engagement and trust has not been tested in realistic human-AI interactions with a global user pool. Prevailing safety frameworks continue to rely on theoretical assumptions derived from Western populations, overlooking the global diversity of AI users. Here, we address these gaps through two large-scale cross-national experiments (N=3,500) across 10 diverse nations, involving real-time and open-ended interactions with an AI system. We find that when evaluating an AI's human-likeness, users focus less on the kind of theoretical aspects often cited in policy (e.g., sentience or consciousness), but rather applied, interactional cues like conversation flow or understanding the user's perspective. We also experimentally demonstrate that humanlike design levers can causally increase anthropomorphism among users; however, we do not find that humanlike design universally increases behavioral measures for user engagement and trust, as previous theoretical work suggests. Instead, part of the connection between human-likeness and behavioral outcomes is fractured by culture: specific design choices that foster self-reported trust in AI-systems in some populations (e.g., Brazil) may trigger the opposite result in others (e.g., Japan). Our findings challenge prevailing narratives of inherent risk in humanlike AI design. Instead, we identify a nuanced, culturally mediated landscape of human-AI interaction, which demands that we move beyond a one-size-fits-all approach in AI governance. |
| 2025-12-19 | [SAVeD: A First-Person Social Media Video Dataset for ADAS-equipped vehicle Near-Miss and Crash Event Analyses](http://arxiv.org/abs/2512.17724v1) | Shaoyan Zhai, Mohamed Abdel-Aty et al. | The advancement of safety-critical research in driving behavior in ADAS-equipped vehicles require real-world datasets that not only include diverse traffic scenarios but also capture high-risk edge cases such as near-miss events and system failures. However, existing datasets are largely limited to either simulated environments or human-driven vehicle data, lacking authentic ADAS (Advanced Driver Assistance System) vehicle behavior under risk conditions. To address this gap, this paper introduces SAVeD, a large-scale video dataset curated from publicly available social media content, explicitly focused on ADAS vehicle-related crashes, near-miss incidents, and disengagements. SAVeD features 2,119 first-person videos, capturing ADAS vehicle operations in diverse locations, lighting conditions, and weather scenarios. The dataset includes video frame-level annotations for collisions, evasive maneuvers, and disengagements, enabling analysis of both perception and decision-making failures. We demonstrate SAVeD's utility through multiple analyses and contributions: (1) We propose a novel framework integrating semantic segmentation and monocular depth estimation to compute real-time Time-to-Collision (TTC) for dynamic objects. (2) We utilize the Generalized Extreme Value (GEV) distribution to model and quantify the extreme risk in crash and near-miss events across different roadway types. (3) We establish benchmarks for state-of-the-art VLLMs (VideoLLaMA2 and InternVL2.5 HiCo R16), showing that SAVeD's detailed annotations significantly enhance model performance through domain adaptation in complex near-miss scenarios. |
| 2025-12-19 | [STAMP/STPA informed characterization of Factors Leading to Loss of Control in AI Systems](http://arxiv.org/abs/2512.17600v1) | Steve Barrett, Anna Bruvere et al. | A major concern amongst AI safety practitioners is the possibility of loss of control, whereby humans lose the ability to exert control over increasingly advanced AI systems. The range of concerns is wide, spanning current day risks to future existential risks, and a range of loss of control pathways from rapid AI self-exfiltration scenarios to more gradual disempowerment scenarios. In this work we set out to firstly, provide a more structured framework for discussing and characterizing loss of control and secondly, to use this framework to assist those responsible for the safe operation of AI-containing socio-technical systems to identify causal factors leading to loss of control. We explore how these two needs can be better met by making use of a methodology developed within the safety-critical systems community known as STAMP and its associated hazard analysis technique of STPA. We select the STAMP methodology primarily because it is based around a world-view that socio-technical systems can be functionally modeled as control structures, and that safety issues arise when there is a loss of control in these structures. |
| 2025-12-19 | [Learning Safe Autonomous Driving Policies Using Predictive Safety Representations](http://arxiv.org/abs/2512.17586v1) | Mahesh Keswani, Raunak Bhattacharyya | Safe reinforcement learning (SafeRL) is a prominent paradigm for autonomous driving, where agents are required to optimize performance under strict safety requirements. This dual objective creates a fundamental tension, as overly conservative policies limit driving efficiency while aggressive exploration risks safety violations. The Safety Representations for Safer Policy Learning (SRPL) framework addresses this challenge by equipping agents with a predictive model of future constraint violations and has shown promise in controlled environments. This paper investigates whether SRPL extends to real-world autonomous driving scenarios. Systematic experiments on the Waymo Open Motion Dataset (WOMD) and NuPlan demonstrate that SRPL can improve the reward-safety tradeoff, achieving statistically significant improvements in success rate (effect sizes r = 0.65-0.86) and cost reduction (effect sizes r = 0.70-0.83), with p < 0.05 for observed improvements. However, its effectiveness depends on the underlying policy optimizer and the dataset distribution. The results further show that predictive safety representations play a critical role in improving robustness to observation noise. Additionally, in zero-shot cross-dataset evaluation, SRPL-augmented agents demonstrate improved generalization compared to non-SRPL methods. These findings collectively demonstrate the potential of predictive safety representations to strengthen SafeRL for autonomous driving. |
| 2025-12-19 | [Optimized Scheduling and Positioning of Mobile Manipulators in Collaborative Applications](http://arxiv.org/abs/2512.17584v1) | Christian Cella, Sole Ester Sonnino et al. | The growing integration of mobile robots in shared workspaces requires efficient path planning and coordination between the agents, accounting for safety and productivity. In this work, we propose a digital model-based optimization framework for mobile manipulators in human-robot collaborative environments, in order to determine the sequence of robot base poses and the task scheduling for the robot. The complete problem is treated as black-box, and Particle Swarm Optimization (PSO) is employed to balance conflicting Key-Performance Indicators (KPIs). We demonstrate improvements in cycle time, task sequencing, and adaptation to human presence in a collaborative box-packing scenario. |
| 2025-12-19 | [On Using Neural Networks to Learn Safety Speed Reduction in Human-Robot Collaboration: A Comparative Analysis](http://arxiv.org/abs/2512.17579v1) | Marco Faroni, Alessio Spanò et al. | In Human-Robot Collaboration, safety mechanisms such as Speed and Separation Monitoring and Power and Force Limitation dynamically adjust the robot's speed based on human proximity. While essential for risk reduction, these mechanisms introduce slowdowns that makes cycle time estimation a hard task and impact job scheduling efficiency. Existing methods for estimating cycle times or designing schedulers often rely on predefined safety models, which may not accurately reflect real-world safety implementations, as these depend on case-specific risk assessments. In this paper, we propose a deep learning approach to predict the robot's safety scaling factor directly from process execution data. We analyze multiple neural network architectures and demonstrate that a simple feed-forward network effectively estimates the robot's slowdown. This capability is crucial for improving cycle time predictions and designing more effective scheduling algorithms in collaborative robotic environments. |
| 2025-12-19 | [Learning-Based Safety-Aware Task Scheduling for Efficient Human-Robot Collaboration](http://arxiv.org/abs/2512.17560v1) | M. Faroni, A. Spano et al. | Ensuring human safety in collaborative robotics can compromise efficiency because traditional safety measures increase robot cycle time when human interaction is frequent. This paper proposes a safety-aware approach to mitigate efficiency losses without assuming prior knowledge of safety logic. Using a deep-learning model, the robot learns the relationship between system state and safety-induced speed reductions based on execution data. Our framework does not explicitly predict human motions but directly models the interaction effects on robot speed, simplifying implementation and enhancing generalizability to different safety logics. At runtime, the learned model optimizes task selection to minimize cycle time while adhering to safety requirements. Experiments on a pick-and-packaging scenario demonstrated significant reductions in cycle times. |
| 2025-12-19 | [Deep Learning-based Robust Autonomous Navigation of Aerial Robots in Dense Forests](http://arxiv.org/abs/2512.17553v1) | Guglielmo Del Col, Väinö Karjalainen et al. | Autonomous aerial navigation in dense natural environments remains challenging due to limited visibility, thin and irregular obstacles, GNSS-denied operation, and frequent perceptual degradation. This work presents an improved deep learning-based navigation framework that integrates semantically enhanced depth encoding with neural motion-primitive evaluation for robust flight in cluttered forests. Several modules are incorporated on top of the original sevae-ORACLE algorithm to address limitations observed during real-world deployment, including lateral control for sharper maneuvering, a temporal consistency mechanism to suppress oscillatory planning decisions, a stereo-based visual-inertial odometry solution for drift-resilient state estimation, and a supervisory safety layer that filters unsafe actions in real time. A depth refinement stage is included to improve the representation of thin branches and reduce stereo noise, while GPU optimization increases onboard inference throughput from 4 Hz to 10 Hz.   The proposed approach is evaluated against several existing learning-based navigation methods under identical environmental conditions and hardware constraints. It demonstrates higher success rates, more stable trajectories, and improved collision avoidance, particularly in highly cluttered forest settings. The system is deployed on a custom quadrotor in three boreal forest environments, achieving fully autonomous completion in all flights in moderate and dense clutter, and 12 out of 15 flights in highly dense underbrush. These results demonstrate improved reliability and safety over existing navigation methods in complex natural environments. |
| 2025-12-19 | [GroundingME: Exposing the Visual Grounding Gap in MLLMs through Multi-Dimensional Evaluation](http://arxiv.org/abs/2512.17495v1) | Rang Li, Lei Li et al. | Visual grounding, localizing objects from natural language descriptions, represents a critical bridge between language and vision understanding. While multimodal large language models (MLLMs) achieve impressive scores on existing benchmarks, a fundamental question remains: can MLLMs truly ground language in vision with human-like sophistication, or are they merely pattern-matching on simplified datasets? Current benchmarks fail to capture real-world complexity where humans effortlessly navigate ambiguous references and recognize when grounding is impossible. To rigorously assess MLLMs' true capabilities, we introduce GroundingME, a benchmark that systematically challenges models across four critical dimensions: (1) Discriminative, distinguishing highly similar objects, (2) Spatial, understanding complex relational descriptions, (3) Limited, handling occlusions or tiny objects, and (4) Rejection, recognizing ungroundable queries. Through careful curation combining automated generation with human verification, we create 1,005 challenging examples mirroring real-world complexity. Evaluating 25 state-of-the-art MLLMs reveals a profound capability gap: the best model achieves only 45.1% accuracy, while most score 0% on rejection tasks, reflexively hallucinating objects rather than acknowledging their absence, raising critical safety concerns for deployment. We explore two strategies for improvements: (1) test-time scaling selects optimal response by thinking trajectory to improve complex grounding by up to 2.9%, and (2) data-mixture training teaches models to recognize ungroundable queries, boosting rejection accuracy from 0% to 27.9%. GroundingME thus serves as both a diagnostic tool revealing current limitations in MLLMs and a roadmap toward human-level visual grounding. |
| 2025-12-19 | [Bridging Natural Language and Formal Specification--Automated Translation of Software Requirements to LTL via Hierarchical Semantics Decomposition Using LLMs](http://arxiv.org/abs/2512.17334v1) | Zhi Ma, Cheng Wen et al. | Automating the translation of natural language (NL) software requirements into formal specifications remains a critical challenge in scaling formal verification practices to industrial settings, particularly in safety-critical domains. Existing approaches, both rule-based and learning-based, face significant limitations. While large language models (LLMs) like GPT-4o demonstrate proficiency in semantic extraction, they still encounter difficulties in addressing the complexity, ambiguity, and logical depth of real-world industrial requirements. In this paper, we propose Req2LTL, a modular framework that bridges NL and Linear Temporal Logic (LTL) through a hierarchical intermediate representation called OnionL. Req2LTL leverages LLMs for semantic decomposition and combines them with deterministic rule-based synthesis to ensure both syntactic validity and semantic fidelity. Our comprehensive evaluation demonstrates that Req2LTL achieves 88.4% semantic accuracy and 100% syntactic correctness on real-world aerospace requirements, significantly outperforming existing methods. |
| 2025-12-19 | [CodeDance: A Dynamic Tool-integrated MLLM for Executable Visual Reasoning](http://arxiv.org/abs/2512.17312v1) | Qi Song, Honglin Li et al. | Recent releases such as o3 highlight human-like "thinking with images" reasoning that combines structured tool use with stepwise verification, yet most open-source approaches still rely on text-only chains, rigid visual schemas, or single-step pipelines, limiting flexibility, interpretability, and transferability on complex tasks. We introduce CodeDance, which explores executable code as a general solver for visual reasoning. Unlike fixed-schema calls (e.g., only predicting bounding-box coordinates), CodeDance defines, composes, and executes code to orchestrate multiple tools, compute intermediate results, and render visual artifacts (e.g., boxes, lines, plots) that support transparent, self-checkable reasoning. To guide this process, we introduce a reward for balanced and adaptive tool-call, which balances exploration with efficiency and mitigates tool overuse. Interestingly, beyond the expected capabilities taught by atomic supervision, we empirically observe novel emergent behaviors during RL training: CodeDance demonstrates novel tool invocations, unseen compositions, and cross-task transfer. These behaviors arise without task-specific fine-tuning, suggesting a general and scalable mechanism of executable visual reasoning. Extensive experiments across reasoning benchmarks (e.g., visual search, math, chart QA) show that CodeDance not only consistently outperforms schema-driven and text-only baselines, but also surpasses advanced closed models such as GPT-4o and larger open-source models. |
| 2025-12-19 | [Uncertainty-Aware 3D UAV Tracking Using Single-Anchor UWB Measurements](http://arxiv.org/abs/2512.17249v1) | Yuqi Ping, Junwei Wu et al. | In this letter, we present an uncertainty-aware single-anchor Ultra-Wideband (UWB)-based 3D tracking framework. Specifically, a mobile Unmanned Aerial Vehicle (UAV) maintains a desired standoff distance to a moving target using range and 3D bearing measurements from a multi-antenna UWB anchor rigidly mounted on the UAV. To enhance the stability and safety under measurement degradation and motion uncertainty, we jointly design a robust factor-graph-based target localization method and a covariance-aware control Lyapunov function--control barrier function (CLF--CBF) tracking controller. This controller adaptively adjusts distance bounds and safety margins based on the posterior target covariance provided by the factor graph. The proposed system is evaluated through numerical simulations and real-world experiments carried out in a narrow indoor corridor environment. |
| 2025-12-19 | [CheXPO-v2: Preference Optimization for Chest X-ray VLMs with Knowledge Graph Consistency](http://arxiv.org/abs/2512.17213v1) | Xiao Liang, Yuxuan An et al. | Medical Vision-Language Models (VLMs) are prone to hallucinations, compromising clinical reliability. While reinforcement learning methods like Group Relative Policy Optimization (GRPO) offer a low-cost alignment solution, their reliance on sparse, outcome-based rewards inadvertently encourages models to "overthink" -- generating verbose, convoluted, and unverifiable Chain-of-Thought reasoning to justify answers. This focus on outcomes obscures factual errors and poses significant safety risks. To address this, we propose CheXPO-v2, a novel alignment framework that shifts from outcome to process supervision. Our core innovation is a Knowledge Graph Consistency Reward mechanism driven by Entity-Relation Matching. By explicitly parsing reasoning steps into structured "Disease, Relation, Anatomy" triplets, we provide fine-grained supervision that penalizes incoherent logic and hallucinations at the atomic level. Integrating this with a hard-example mining strategy, our approach significantly outperforms GRPO and state-of-the-art models on benchmarks like MIMIC-CXR-VQA. Crucially, CheXPO-v2 achieves new state-of-the-art accuracy using only 5k samples, demonstrating exceptional data efficiency while producing clinically sound and verifiable reasoning. The project source code is publicly available at: https://github.com/ecoxial2007/CheX-Phi4MM. |
| 2025-12-19 | [A Search for Binary Black Hole Mergers in LIGO O1-O3 Data with Convolutional Neural Networks](http://arxiv.org/abs/2512.17204v1) | Ethan Silver, Plamen Krastev et al. | Since the first detection of gravitational waves in 2015 by LIGO from the binary black hole merger GW150914, gravitational wave astronomy has developed significantly, with over 200 compact binary merger events cataloged. The use of neural networks has the potential to significantly speed up the detection, classification, and especially parameter estimation for gravitational wave events, compared to current techniques, quite important for electromagnetic follow-up of events. In this work, we present a machine learning pipeline using neural networks to detect gravitational wave events. We generate training data using real LIGO data to train and refine neural networks that can detect binary black hole (BBH) mergers, and apply these models to search through LIGO's first three observing runs. We detect 57 out of the 75 total cataloged BBH events with two detectors of data in O1, O2, and O3, with 57 false positives that can mostly be ruled out with parameter inference and human inspection. Finally, we extensively test this pipeline on time-shifted data to characterize its False Alarm Rate (FAR). These results are an important step in developing machine learning-based GW searches, enabling low-latency detection and multi-messenger astronomy. |
| 2025-12-19 | [Conservative Bias in Multi-Teacher Learning: Why Agents Prefer Low-Reward Advisors](http://arxiv.org/abs/2512.17180v1) | Maher Mesto, Francisco Cruz | Interactive reinforcement learning (IRL) has shown promise in enabling autonomous agents and robots to learn complex behaviours from human teachers, yet the dynamics of teacher selection remain poorly understood. This paper reveals an unexpected phenomenon in IRL: when given a choice between teachers with different reward structures, learning agents overwhelmingly prefer conservative, low-reward teachers (93.16% selection rate) over those offering 20x higher rewards. Through 1,250 experimental runs in navigation tasks with multiple expert teachers, we discovered: (1) Conservative bias dominates teacher selection: agents systematically choose the lowest-reward teacher, prioritising consistency over optimality; (2) Critical performance thresholds exist at teacher availability rho >= 0.6 and accuracy omega >= 0.6, below which the framework fails catastrophically; (3) The framework achieves 159% improvement over baseline Q-learning under concept drift. These findings challenge fundamental assumptions about optimal teaching in RL and suggest potential implications for human-robot collaboration, where human preferences for safety and consistency may align with the observed agent selection behaviour, potentially informing training paradigms for safety-critical robotic applications. |
| 2025-12-18 | [Perception-Limited Smooth Safety Filtering](http://arxiv.org/abs/2512.17057v1) | Lyes Smaili, Soulaimane Berkane | This paper develops a smooth safety-filtering framework for nonlinear control-affine systems under limited perception. Classical Control Barrier Function (CBF) filters assume global availability of the safety function - its value and gradient must be known everywhere - an assumption incompatible with sensing-limited settings, and the resulting filters often exhibit nonsmooth switching when constraints activate. We propose two complementary perception-aware safety filters applicable to general control-invariant safety sets. The first introduces a smooth perception gate that modulates barrier constraints based on sensing range, yielding a closed-form Lipschitz-safe controller with forward-invariance guarantees. The second replaces the hard CBF constraint with a differentiable penalty term, leading to a smooth unconstrained optimization-based safety filter consistent with CBF principles. For both designs, we establish existence, uniqueness, and forward invariance of the closed-loop trajectories. Numerical results demonstrate that the proposed smooth filters enable the synthesis of higher-order tracking controllers for systems such as drones and second-order ground robots, offering substantially smoother and more robust safety-critical behaviors than classical CBF-based filters. |
| 2025-12-18 | [Security Risks of Agentic Vehicles: A Systematic Analysis of Cognitive and Cross-Layer Threats](http://arxiv.org/abs/2512.17041v1) | Ali Eslami, Jiangbo Yu | Agentic AI is increasingly being explored and introduced in both manually driven and autonomous vehicles, leading to the notion of Agentic Vehicles (AgVs), with capabilities such as memory-based personalization, goal interpretation, strategic reasoning, and tool-mediated assistance. While frameworks such as the OWASP Agentic AI Security Risks highlight vulnerabilities in reasoning-driven AI systems, they are not designed for safety-critical cyber-physical platforms such as vehicles, nor do they account for interactions with other layers such as perception, communication, and control layers. This paper investigates security threats in AgVs, including OWASP-style risks and cyber-attacks from other layers affecting the agentic layer. By introducing a role-based architecture for agentic vehicles, consisting of a Personal Agent and a Driving Strategy Agent, we will investigate vulnerabilities in both agentic AI layer and cross-layer risks, including risks originating from upstream layers (e.g., perception layer, control layer, etc.). A severity matrix and attack-chain analysis illustrate how small distortions can escalate into misaligned or unsafe behavior in both human-driven and autonomous vehicles. The resulting framework provides the first structured foundation for analyzing security risks of agentic AI in both current and emerging vehicle platforms. |
| 2025-12-18 | [Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning](http://arxiv.org/abs/2512.16917v1) | Qihao Liu, Luoxin Ye et al. | Large language models (LLMs) with explicit reasoning capabilities excel at mathematical reasoning yet still commit process errors, such as incorrect calculations, brittle logic, and superficially plausible but invalid steps. In this paper, we introduce Generative Adversarial Reasoner, an on-policy joint training framework designed to enhance reasoning by co-evolving an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning. A compute-efficient review schedule partitions each reasoning chain into logically complete slices of comparable length, and the discriminator evaluates each slice's soundness with concise, structured justifications. Learning couples complementary signals: the LLM reasoner is rewarded for logically consistent steps that yield correct answers, while the discriminator earns rewards for correctly detecting errors or distinguishing traces in the reasoning process. This produces dense, well-calibrated, on-policy step-level rewards that supplement sparse exact-match signals, improving credit assignment, increasing sample efficiency, and enhancing overall reasoning quality of LLMs. Across various mathematical benchmarks, the method delivers consistent gains over strong baselines with standard RL post-training. Specifically, on AIME24, we improve DeepSeek-R1-Distill-Qwen-7B from 54.0 to 61.3 (+7.3) and DeepSeek-R1-Distill-Llama-8B from 43.7 to 53.7 (+10.0). The modular discriminator also enables flexible reward shaping for objectives such as teacher distillation, preference alignment, and mathematical proof-based reasoning. |
| 2025-12-18 | [SceneDiff: A Benchmark and Method for Multiview Object Change Detection](http://arxiv.org/abs/2512.16908v1) | Yuqun Wu, Chih-hao Lin et al. | We investigate the problem of identifying objects that have been added, removed, or moved between a pair of captures (images or videos) of the same scene at different times. Detecting such changes is important for many applications, such as robotic tidying or construction progress and safety monitoring. A major challenge is that varying viewpoints can cause objects to falsely appear changed. We introduce SceneDiff Benchmark, the first multiview change detection benchmark with object instance annotations, comprising 350 diverse video pairs with thousands of changed objects. We also introduce the SceneDiff method, a new training-free approach for multiview object change detection that leverages pretrained 3D, segmentation, and image encoding models to robustly predict across multiple benchmarks. Our method aligns the captures in 3D, extracts object regions, and compares spatial and semantic region features to detect changes. Experiments on multi-view and two-view benchmarks demonstrate that our method outperforms existing approaches by large margins (94% and 37.4% relative AP improvements). The benchmark and code will be publicly released. |
| 2025-12-18 | [An evacuation simulator for pedestrian dynamics based on the Social Force Model](http://arxiv.org/abs/2512.16887v1) | Julián López, Virginia Mazzone et al. | The evacuation of pedestrians from enclosed spaces represents a key problem in safety engineering and infrastructure design. Analyzing the collective dynamics that emerge during evacuation processes requires simulation tools capable of capturing individual interactions and spatial constraints realistically.   In this work, we present \textit{SiCoBioNa}, an open-source evacuation simulator based on the Social Force Model (SFM). The software provides an intuitive graphical interface that allows users to configure pedestrian properties, spatial geometries, and initial conditions without requiring prior expertise in numerical modeling techniques. The SFM framework enables the representation of goal-oriented motion, interpersonal interactions, and interactions with fixed obstacles.   The simulator generates both quantitative data and visual outputs, facilitating the analysis of evacuation dynamics and the evaluation of different spatial configurations. Due to its modular and extensible design, \textit{SiCoBioNa} serves as a reproducible research tool for studies on pedestrian dynamics providing practical support for evacuation planning. |

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



