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
| 2025-10-14 | [KoALA: KL-L0 Adversarial Detector via Label Agreement](http://arxiv.org/abs/2510.12752v1) | Siqi Li, Yasser Shoukry | Deep neural networks are highly susceptible to adversarial attacks, which pose significant risks to security- and safety-critical applications. We present KoALA (KL-L0 Adversarial detection via Label Agreement), a novel, semantics-free adversarial detector that requires no architectural changes or adversarial retraining. KoALA operates on a simple principle: it detects an adversarial attack when class predictions from two complementary similarity metrics disagree. These metrics-KL divergence and an L0-based similarity-are specifically chosen to detect different types of perturbations. The KL divergence metric is sensitive to dense, low-amplitude shifts, while the L0-based similarity is designed for sparse, high-impact changes. We provide a formal proof of correctness for our approach. The only training required is a simple fine-tuning step on a pre-trained image encoder using clean images to ensure the embeddings align well with both metrics. This makes KOALA a lightweight, plug-and-play solution for existing models and various data modalities. Our extensive experiments on ResNet/CIFAR-10 and CLIP/Tiny-ImageNet confirm our theoretical claims. When the theorem's conditions are met, KoALA consistently and effectively detects adversarial examples. On the full test sets, KoALA achieves a precision of 0.94 and a recall of 0.81 on ResNet/CIFAR-10, and a precision of 0.66 and a recall of 0.85 on CLIP/Tiny-ImageNet. |
| 2025-10-14 | [HYPE: Hybrid Planning with Ego Proposal-Conditioned Predictions](http://arxiv.org/abs/2510.12733v1) | Hang Yu, Julian Jordan et al. | Safe and interpretable motion planning in complex urban environments needs to reason about bidirectional multi-agent interactions. This reasoning requires to estimate the costs of potential ego driving maneuvers. Many existing planners generate initial trajectories with sampling-based methods and refine them by optimizing on learned predictions of future environment states, which requires a cost function that encodes the desired vehicle behavior. Designing such a cost function can be very challenging, especially if a wide range of complex urban scenarios has to be considered. We propose HYPE: HYbrid Planning with Ego proposal-conditioned predictions, a planner that integrates multimodal trajectory proposals from a learned proposal model as heuristic priors into a Monte Carlo Tree Search (MCTS) refinement. To model bidirectional interactions, we introduce an ego-conditioned occupancy prediction model, enabling consistent, scene-aware reasoning. Our design significantly simplifies cost function design in refinement by considering proposal-driven guidance, requiring only minimalistic grid-based cost terms. Evaluations on large-scale real-world benchmarks nuPlan and DeepUrban show that HYPE effectively achieves state-of-the-art performance, especially in safety and adaptability. |
| 2025-10-14 | [Towards Robust Artificial Intelligence: Self-Supervised Learning Approach for Out-of-Distribution Detection](http://arxiv.org/abs/2510.12713v1) | Wissam Salhab, Darine Ameyed et al. | Robustness in AI systems refers to their ability to maintain reliable and accurate performance under various conditions, including out-of-distribution (OOD) samples, adversarial attacks, and environmental changes. This is crucial in safety-critical systems, such as autonomous vehicles, transportation, or healthcare, where malfunctions could have severe consequences. This paper proposes an approach to improve OOD detection without the need of labeled data, thereby increasing the AI systems' robustness. The proposed approach leverages the principles of self-supervised learning, allowing the model to learn useful representations from unlabeled data. Combined with graph-theoretical techniques, this enables the more efficient identification and categorization of OOD samples. Compared to existing state-of-the-art methods, this approach achieved an Area Under the Receiver Operating Characteristic Curve (AUROC) = 0.99. |
| 2025-10-14 | [CAMNet: Leveraging Cooperative Awareness Messages for Vehicle Trajectory Prediction](http://arxiv.org/abs/2510.12703v1) | Mattia Grasselli, Angelo Porrello et al. | Autonomous driving remains a challenging task, particularly due to safety concerns. Modern vehicles are typically equipped with expensive sensors such as LiDAR, cameras, and radars to reduce the risk of accidents. However, these sensors face inherent limitations: their field of view and line of sight can be obstructed by other vehicles, thereby reducing situational awareness. In this context, vehicle-to-vehicle communication plays a crucial role, as it enables cars to share information and remain aware of each other even when sensors are occluded. One way to achieve this is through the use of Cooperative Awareness Messages (CAMs). In this paper, we investigate the use of CAM data for vehicle trajectory prediction. Specifically, we design and train a neural network, Cooperative Awareness Message-based Graph Neural Network (CAMNet), on a widely used motion forecasting dataset. We then evaluate the model on a second dataset that we created from scratch using Cooperative Awareness Messages, in order to assess whether this type of data can be effectively exploited. Our approach demonstrates promising results, showing that CAMs can indeed support vehicle trajectory prediction. At the same time, we discuss several limitations of the approach, which highlight opportunities for future research. |
| 2025-10-14 | [Few Shot Semi-Supervised Learning for Abnormal Stop Detection from Sparse GPS Trajectories](http://arxiv.org/abs/2510.12686v1) | Muhammad Ayub Sabir, Junbiao Pang et al. | Abnormal stop detection (ASD) in intercity coach transportation is critical for ensuring passenger safety, operational reliability, and regulatory compliance. However, two key challenges hinder ASD effectiveness: sparse GPS trajectories, which obscure short or unauthorized stops, and limited labeled data, which restricts supervised learning. Existing methods often assume dense sampling or regular movement patterns, limiting their applicability. To address data sparsity, we propose a Sparsity-Aware Segmentation (SAS) method that adaptively defines segment boundaries based on local spatial-temporal density. Building upon these segments, we introduce three domain-specific indicators to capture abnormal stop behaviors. To further mitigate the impact of sparsity, we develop Locally Temporal-Indicator Guided Adjustment (LTIGA), which smooths these indicators via local similarity graphs. To overcome label scarcity, we construct a spatial-temporal graph where each segment is a node with LTIGA-refined features. We apply label propagation to expand weak supervision across the graph, followed by a GCN to learn relational patterns. A final self-training module incorporates high-confidence pseudo-labels to iteratively improve predictions. Experiments on real-world coach data show an AUC of 0.854 and AP of 0.866 using only 10 labeled instances, outperforming prior methods. The code and dataset are publicly available at \href{https://github.com/pangjunbiao/Abnormal-Stop-Detection-SSL.git} |
| 2025-10-14 | [Autonomous Legged Mobile Manipulation for Lunar Surface Operations via Constrained Reinforcement Learning](http://arxiv.org/abs/2510.12684v1) | Alvaro Belmonte-Baeza, Miguel Cazorla et al. | Robotics plays a pivotal role in planetary science and exploration, where autonomous and reliable systems are crucial due to the risks and challenges inherent to space environments. The establishment of permanent lunar bases demands robotic platforms capable of navigating and manipulating in the harsh lunar terrain. While wheeled rovers have been the mainstay for planetary exploration, their limitations in unstructured and steep terrains motivate the adoption of legged robots, which offer superior mobility and adaptability. This paper introduces a constrained reinforcement learning framework designed for autonomous quadrupedal mobile manipulators operating in lunar environments. The proposed framework integrates whole-body locomotion and manipulation capabilities while explicitly addressing critical safety constraints, including collision avoidance, dynamic stability, and power efficiency, in order to ensure robust performance under lunar-specific conditions, such as reduced gravity and irregular terrain. Experimental results demonstrate the framework's effectiveness in achieving precise 6D task-space end-effector pose tracking, achieving an average positional accuracy of 4 cm and orientation accuracy of 8.1 degrees. The system consistently respects both soft and hard constraints, exhibiting adaptive behaviors optimized for lunar gravity conditions. This work effectively bridges adaptive learning with essential mission-critical safety requirements, paving the way for advanced autonomous robotic explorers for future lunar missions. |
| 2025-10-14 | [Keep Calm and Avoid Harmful Content: Concept Alignment and Latent Manipulation Towards Safer Answers](http://arxiv.org/abs/2510.12672v1) | Ruben Belo, Claudia Soares et al. | Large Language Models are susceptible to jailbreak attacks that bypass built-in safety guardrails (e.g., by tricking the model with adversarial prompts). We propose Concept Alignment and Concept Manipulation \textbf{CALM}, an inference-time method that suppresses harmful concepts by modifying latent representations of the last layer of the model, without retraining. Leveraging \gls*{cw} technique from Computer Vision combined with orthogonal projection, CALM removes unwanted latent directions associated with harmful content while preserving model performance. Experiments show that CALM reduces harmful outputs and outperforms baseline methods in most metrics, offering a lightweight approach to AI safety with no additional training data or model fine-tuning, while incurring only a small computational overhead at inference. |
| 2025-10-14 | [Rethinking Knowledge Distillation: A Data Dependent Regulariser With a Negative Asymmetric Payoff](http://arxiv.org/abs/2510.12615v1) | Israel Mason-Williams, Gabryel Mason-Williams et al. | Knowledge distillation is often considered a compression mechanism when judged on the resulting student's accuracy and loss, yet its functional impact is poorly understood. In this work, we quantify the compression capacity of knowledge distillation and the resulting knowledge transfer from a functional perspective, decoupling compression from architectural reduction, which provides an improved understanding of knowledge distillation. We employ hypothesis testing, controls, and random control distillation to understand knowledge transfer mechanisms across data modalities. To rigorously test the breadth and limits of our analyses, we explore multiple distillation variants and analyse distillation scaling laws across model sizes. Our findings demonstrate that, while there is statistically significant knowledge transfer in some modalities and architectures, the extent of this transfer is less pronounced than anticipated, even under conditions designed to maximise knowledge sharing. Notably, in cases of significant knowledge transfer, we identify a consistent and severe asymmetric transfer of negative knowledge to the student, raising safety concerns in knowledge distillation applications. Across 12 experimental setups, 9 architectures, and 7 datasets, our findings show that knowledge distillation functions less as a compression mechanism and more as a data-dependent regulariser with a negative asymmetric payoff. |
| 2025-10-14 | [Learning Robust Agile Flight Control with Stability Guarantees](http://arxiv.org/abs/2510.12611v1) | Lukas Pries, Markus Ryll | In the evolving landscape of high-speed agile quadrotor flight, achieving precise trajectory tracking at the platform's operational limits is paramount. Controllers must handle actuator constraints, exhibit robustness to disturbances, and remain computationally efficient for safety-critical applications. In this work, we present a novel neural-augmented feedback controller for agile flight control. The controller addresses individual limitations of existing state-of-the-art control paradigms and unifies their strengths. We demonstrate the controller's capabilities, including the accurate tracking of highly aggressive trajectories that surpass the feasibility of the actuators. Notably, the controller provides universal stability guarantees, enhancing its robustness and tracking performance even in exceedingly disturbance-prone settings. Its nonlinear feedback structure is highly efficient enabling fast computation at high update rates. Moreover, the learning process in simulation is both fast and stable, and the controller's inherent robustness allows direct deployment to real-world platforms without the need for training augmentations or fine-tuning. |
| 2025-10-14 | [Optimising Communication Control Factors for Energy Consumption in Rural LOS V2X](http://arxiv.org/abs/2510.12539v1) | Zhanle Zhao, Son Dinh-Van et al. | Connected braking can reduce fatal collisions in connected and autonomous vehicles (CAVs) by using reliable, low-latency 5G New Radio (NR) links, especially NR Sidelink Vehicle-to-Everything (V2X). In rural areas, road side units are sparse and power-constrained or off-grid, so energy efficiency must be considered alongside safety. This paper studies how three communication control factors including subcarrier spacing ($\mathrm{SCS}$), modulation and coding scheme ($\mathrm{MCS}$), and transmit power ($P_{\mathrm{t}}$) should be configured to balance safety and energy consumption in rural line-of-sight (LOS) scenarios in light and heavy traffic scenarios. Safety is quantified by the packet receive ratio ($\mathrm{PRR}$) against the minimum communication distance $D_{\mathrm{comm}}$, defined as the distance that the vehicle travels during the transmission of the safety message. Results show that, under heavy traffic, increasing $P_{\mathrm{t}}$ and selecting a low-rate $\mathrm{MCS}$ at $\mathrm{SCS} = 30$ kHz sustains high $\mathrm{PRR}$ at $D_{\mathrm{comm}}$, albeit with higher energy cost. In light traffic, maintaining lower $P_\mathrm{t}$ with low $\mathrm{MCS}$ levels achieves a favorable reliability-energy trade-off while preserving acceptable $\mathrm{PRR}$ at $D_{\mathrm{comm}}$. These findings demonstrate the necessity of adaptive, energy-aware strategy to guarantee both safety and energy efficiency in rural V2X systems. |
| 2025-10-14 | [A Task-Efficient Reinforcement Learning Task-Motion Planner for Safe Human-Robot Cooperation](http://arxiv.org/abs/2510.12477v1) | Gaoyuan Liu, Joris de Winter et al. | In a Human-Robot Cooperation (HRC) environment, safety and efficiency are the two core properties to evaluate robot performance. However, safety mechanisms usually hinder task efficiency since human intervention will cause backup motions and goal failures of the robot. Frequent motion replanning will increase the computational load and the chance of failure. In this paper, we present a hybrid Reinforcement Learning (RL) planning framework which is comprised of an interactive motion planner and a RL task planner. The RL task planner attempts to choose statistically safe and efficient task sequences based on the feedback from the motion planner, while the motion planner keeps the task execution process collision-free by detecting human arm motions and deploying new paths when the previous path is not valid anymore. Intuitively, the RL agent will learn to avoid dangerous tasks, while the motion planner ensures that the chosen tasks are safe. The proposed framework is validated on the cobot in both simulation and the real world, we compare the planner with hard-coded task motion planning methods. The results show that our planning framework can 1) react to uncertain human motions at both joint and task levels; 2) reduce the times of repeating failed goal commands; 3) reduce the total number of replanning requests. |
| 2025-10-14 | [A Function Centric Perspective On Flat and Sharp Minima](http://arxiv.org/abs/2510.12451v1) | Israel Mason-Williams, Gabryel Mason-Williams et al. | Flat minima are widely believed to correlate with improved generalisation in deep neural networks. However, this connection has proven more nuanced in recent studies, with both theoretical counterexamples and empirical exceptions emerging in the literature. In this paper, we revisit the role of sharpness in model performance, proposing that sharpness is better understood as a function-dependent property rather than a reliable indicator of poor generalisation. We conduct extensive empirical studies, from single-objective optimisation to modern image classification tasks, showing that sharper minima often emerge when models are regularised (e.g., via SAM, weight decay, or data augmentation), and that these sharp minima can coincide with better generalisation, calibration, robustness, and functional consistency. Across a range of models and datasets, we find that baselines without regularisation tend to converge to flatter minima yet often perform worse across all safety metrics. Our findings demonstrate that function complexity, rather than flatness alone, governs the geometry of solutions, and that sharper minima can reflect more appropriate inductive biases (especially under regularisation), calling for a function-centric reappraisal of loss landscape geometry. |
| 2025-10-14 | [Biased-Attention Guided Risk Prediction for Safe Decision-Making at Unsignalized Intersections](http://arxiv.org/abs/2510.12428v1) | Chengyang Dong, Nan Guo | Autonomous driving decision-making at unsignalized intersections is highly challenging due to complex dynamic interactions and high conflict risks. To achieve proactive safety control, this paper proposes a deep reinforcement learning (DRL) decision-making framework integrated with a biased attention mechanism. The framework is built upon the Soft Actor-Critic (SAC) algorithm. Its core innovation lies in the use of biased attention to construct a traffic risk predictor. This predictor assesses the long-term risk of collision for a vehicle entering the intersection and transforms this risk into a dense reward signal to guide the SAC agent in making safe and efficient driving decisions. Finally, the simulation results demonstrate that the proposed method effectively improves both traffic efficiency and vehicle safety at the intersection, thereby proving the effectiveness of the intelligent decision-making framework in complex scenarios. The code of our work is available at https://github.com/hank111525/SAC-RWB. |
| 2025-10-14 | [Shape-Aware Whole-Body Control for Continuum Robots with Application in Endoluminal Surgical Robotics](http://arxiv.org/abs/2510.12332v1) | Mohammadreza Kasaei, Mostafa Ghobadi et al. | This paper presents a shape-aware whole-body control framework for tendon-driven continuum robots with direct application to endoluminal surgical navigation. Endoluminal procedures, such as bronchoscopy, demand precise and safe navigation through tortuous, patient-specific anatomy where conventional tip-only control often leads to wall contact, tissue trauma, or failure to reach distal targets. To address these challenges, our approach combines a physics-informed backbone model with residual learning through an Augmented Neural ODE, enabling accurate shape estimation and efficient Jacobian computation. A sampling-based Model Predictive Path Integral (MPPI) controller leverages this representation to jointly optimize tip tracking, backbone conformance, and obstacle avoidance under actuation constraints. A task manager further enhances adaptability by allowing real-time adjustment of objectives, such as wall clearance or direct advancement, during tele-operation. Extensive simulation studies demonstrate millimeter-level accuracy across diverse scenarios, including trajectory tracking, dynamic obstacle avoidance, and shape-constrained reaching. Real-robot experiments on a bronchoscopy phantom validate the framework, showing improved lumen-following accuracy, reduced wall contacts, and enhanced adaptability compared to joystick-only navigation and existing baselines. These results highlight the potential of the proposed framework to increase safety, reliability, and operator efficiency in minimally invasive endoluminal surgery, with broader applicability to other confined and safety-critical environments. |
| 2025-10-14 | [PAGS: Priority-Adaptive Gaussian Splatting for Dynamic Driving Scenes](http://arxiv.org/abs/2510.12282v1) | Ying A, Wenzhang Sun et al. | Reconstructing dynamic 3D urban scenes is crucial for autonomous driving, yet current methods face a stark trade-off between fidelity and computational cost. This inefficiency stems from their semantically agnostic design, which allocates resources uniformly, treating static backgrounds and safety-critical objects with equal importance. To address this, we introduce Priority-Adaptive Gaussian Splatting (PAGS), a framework that injects task-aware semantic priorities directly into the 3D reconstruction and rendering pipeline. PAGS introduces two core contributions: (1) Semantically-Guided Pruning and Regularization strategy, which employs a hybrid importance metric to aggressively simplify non-critical scene elements while preserving fine-grained details on objects vital for navigation. (2) Priority-Driven Rendering pipeline, which employs a priority-based depth pre-pass to aggressively cull occluded primitives and accelerate the final shading computations. Extensive experiments on the Waymo and KITTI datasets demonstrate that PAGS achieves exceptional reconstruction quality, particularly on safety-critical objects, while significantly reducing training time and boosting rendering speeds to over 350 FPS. |
| 2025-10-14 | [Local Background Features Matter in Out-of-Distribution Detection](http://arxiv.org/abs/2510.12259v1) | Jinlun Ye, Zhuohao Sun et al. | Out-of-distribution (OOD) detection is crucial when deploying deep neural networks in the real world to ensure the reliability and safety of their applications. One main challenge in OOD detection is that neural network models often produce overconfident predictions on OOD data. While some methods using auxiliary OOD datasets or generating fake OOD images have shown promising OOD detection performance, they are limited by the high costs of data collection and training. In this study, we propose a novel and effective OOD detection method that utilizes local background features as fake OOD features for model training. Inspired by the observation that OOD images generally share similar background regions with ID images, the background features are extracted from ID images as simulated OOD visual representations during training based on the local invariance of convolution. Through being optimized to reduce the $L_2$-norm of these background features, the neural networks are able to alleviate the overconfidence issue on OOD data. Extensive experiments on multiple standard OOD detection benchmarks confirm the effectiveness of our method and its wide combinatorial compatibility with existing post-hoc methods, with new state-of-the-art performance achieved from our method. |
| 2025-10-14 | [MedKGEval: A Knowledge Graph-Based Multi-Turn Evaluation Framework for Open-Ended Patient Interactions with Clinical LLMs](http://arxiv.org/abs/2510.12224v1) | Yuechun Yu, Han Ying et al. | The reliable evaluation of large language models (LLMs) in medical applications remains an open challenge, particularly in capturing the complexity of multi-turn doctor-patient interactions that unfold in real clinical environments. Existing evaluation methods typically rely on post hoc review of full conversation transcripts, thereby neglecting the dynamic, context-sensitive nature of medical dialogues and the evolving informational needs of patients. In this work, we present MedKGEval, a novel multi-turn evaluation framework for clinical LLMs grounded in structured medical knowledge. Our approach introduces three key contributions: (1) a knowledge graph-driven patient simulation mechanism, where a dedicated control module retrieves relevant medical facts from a curated knowledge graph, thereby endowing the patient agent with human-like and realistic conversational behavior. This knowledge graph is constructed by integrating open-source resources with additional triples extracted from expert-annotated datasets; (2) an in-situ, turn-level evaluation framework, where each model response is assessed by a Judge Agent for clinical appropriateness, factual correctness, and safety as the dialogue progresses using a suite of fine-grained, task-specific metrics; (3) a comprehensive multi-turn benchmark of eight state-of-the-art LLMs, demonstrating MedKGEval's ability to identify subtle behavioral flaws and safety risks that are often overlooked by conventional evaluation pipelines. Although initially designed for Chinese and English medical applications, our framework can be readily extended to additional languages by switching the input knowledge graphs, ensuring seamless bilingual support and domain-specific applicability. |
| 2025-10-14 | [Learning Social Navigation from Positive and Negative Demonstrations and Rule-Based Specifications](http://arxiv.org/abs/2510.12215v1) | Chanwoo Kim, Jihwan Yoon et al. | Mobile robot navigation in dynamic human environments requires policies that balance adaptability to diverse behaviors with compliance to safety constraints. We hypothesize that integrating data-driven rewards with rule-based objectives enables navigation policies to achieve a more effective balance of adaptability and safety. To this end, we develop a framework that learns a density-based reward from positive and negative demonstrations and augments it with rule-based objectives for obstacle avoidance and goal reaching. A sampling-based lookahead controller produces supervisory actions that are both safe and adaptive, which are subsequently distilled into a compact student policy suitable for real-time operation with uncertainty estimates. Experiments in synthetic and elevator co-boarding simulations show consistent gains in success rate and time efficiency over baselines, and real-world demonstrations with human participants confirm the practicality of deployment. A video illustrating this work can be found on our project page https://chanwookim971024.github.io/PioneeR/. |
| 2025-10-14 | [Controllable Collision Scenario Generation via Collision Pattern Prediction](http://arxiv.org/abs/2510.12206v1) | Pin-Lun Chen, Chi-Hsi Kung et al. | Evaluating the safety of autonomous vehicles (AVs) requires diverse, safety-critical scenarios, with collisions being especially important yet rare and unsafe to collect in the real world. Therefore, the community has been focusing on generating safety-critical scenarios in simulation. However, controlling attributes such as collision type and time-to-accident (TTA) remains challenging. We introduce a new task called controllable collision scenario generation, where the goal is to produce trajectories that realize a user-specified collision type and TTA, to investigate the feasibility of automatically generating desired collision scenarios. To support this task, we present COLLIDE, a large-scale collision scenario dataset constructed by transforming real-world driving logs into diverse collisions, balanced across five representative collision types and different TTA intervals. We propose a framework that predicts Collision Pattern, a compact and interpretable representation that captures the spatial configuration of the ego and the adversarial vehicles at impact, before rolling out full adversarial trajectories. Experiments show that our approach outperforms strong baselines in both collision rate and controllability. Furthermore, generated scenarios consistently induce higher planner failure rates, revealing limitations of existing planners. We demonstrate that these scenarios fine-tune planners for robustness improvements, contributing to safer AV deployment in different collision scenarios. |
| 2025-10-14 | [Sleepy Chauffeur Detection and Alert Techniques for Road Safety](http://arxiv.org/abs/2510.12205v1) | Himel Ghosh, Sayak Chatterjee et al. | The most startling of the contemporary problems is the sleepiness of chauffeur which causes lots of car accidents. Prevention of those impending accidents by detecting and alerting the sleepy chauffeur is vital, otherwise that would lead to loss of lives and various traumas along with severe injuries. The slumber or sleep may be caused by huge stress, pressure, relentless work load or alcoholism, for which sleep deprivation occurs and the chauffeur while driving gets drowsy. So far, considerable amount of systems has been developed to detect drowsiness of drivers, most of which mainly depend on image processing algorithms using cameras. Some of them also incorporate artificial intelligence and machine learning based algorithms. This paper presents a review of the existing systems and also proposes an easy and cheap system using sensors and Arduino, capable of detecting sleepiness and generates siren alarm and send alert message to take precautionary measures. |

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



