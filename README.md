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
| 2025-09-12 | [DeepDive: Advancing Deep Search Agents with Knowledge Graphs and Multi-Turn RL](http://arxiv.org/abs/2509.10446v1) | Rui Lu, Zhenyu Hou et al. | Augmenting large language models (LLMs) with browsing tools substantially improves their potential as deep search agents to solve complex, real-world tasks. Yet, open LLMs still perform poorly in such settings due to limited long-horizon reasoning capacity with browsing tools and the lack of sufficiently difficult supervised data. To address these challenges, we present DeepDive to advance deep search agents. First, we propose a strategy to automatically synthesize complex, difficult, and hard-to-find questions from open knowledge graphs. Second, we apply end-to-end multi-turn reinforcement learning (RL) to enhance LLMs' long-horizon reasoning with deep search. Experiments show that DeepDive-32B achieves a new open-source competitive result on BrowseComp, outperforming WebSailor, DeepSeek-R1-Browse, and Search-o1. We demonstrate that multi-turn RL training improves deep search ability and significantly contributes to the performance improvements across multiple benchmarks. We observe that DeepDive enables test-time scaling of tool calls and parallel sampling. All datasets, models, and code are publicly available at https://github.com/THUDM/DeepDive. |
| 2025-09-12 | [DECAMP: Towards Scene-Consistent Multi-Agent Motion Prediction with Disentangled Context-Aware Pre-Training](http://arxiv.org/abs/2509.10426v1) | Jianxin Shi, Zengqi Peng et al. | Trajectory prediction is a critical component of autonomous driving, essential for ensuring both safety and efficiency on the road. However, traditional approaches often struggle with the scarcity of labeled data and exhibit suboptimal performance in multi-agent prediction scenarios. To address these challenges, we introduce a disentangled context-aware pre-training framework for multi-agent motion prediction, named DECAMP. Unlike existing methods that entangle representation learning with pretext tasks, our framework decouples behavior pattern learning from latent feature reconstruction, prioritizing interpretable dynamics and thereby enhancing scene representation for downstream prediction. Additionally, our framework incorporates context-aware representation learning alongside collaborative spatial-motion pretext tasks, which enables joint optimization of structural and intentional reasoning while capturing the underlying dynamic intentions. Our experiments on the Argoverse 2 benchmark showcase the superior performance of our method, and the results attained underscore its effectiveness in multi-agent motion forecasting. To the best of our knowledge, this is the first context autoencoder framework for multi-agent motion forecasting in autonomous driving. The code and models will be made publicly available. |
| 2025-09-12 | [Run-Time Monitoring of ERTMS/ETCS Control Flow by Process Mining](http://arxiv.org/abs/2509.10419v1) | Francesco Vitale, Tommaso Zoppi et al. | Ensuring the resilience of computer-based railways is increasingly crucial to account for uncertainties and changes due to the growing complexity and criticality of those systems. Although their software relies on strict verification and validation processes following well-established best-practices and certification standards, anomalies can still occur at run-time due to residual faults, system and environmental modifications that were unknown at design-time, or other emergent cyber-threat scenarios. This paper explores run-time control-flow anomaly detection using process mining to enhance the resilience of ERTMS/ETCS L2 (European Rail Traffic Management System / European Train Control System Level 2). Process mining allows learning the actual control flow of the system from its execution traces, thus enabling run-time monitoring through online conformance checking. In addition, anomaly localization is performed through unsupervised machine learning to link relevant deviations to critical system components. We test our approach on a reference ERTMS/ETCS L2 scenario, namely the RBC/RBC Handover, to show its capability to detect and localize anomalies with high accuracy, efficiency, and explainability. |
| 2025-09-12 | [Merging Physics-Based Synthetic Data and Machine Learning for Thermal Monitoring of Lithium-ion Batteries: The Role of Data Fidelity](http://arxiv.org/abs/2509.10380v1) | Yusheng Zheng, Wenxue Liu et al. | Since the internal temperature is less accessible than surface temperature, there is an urgent need to develop accurate and real-time estimation algorithms for better thermal management and safety. This work presents a novel framework for resource-efficient and scalable development of accurate, robust, and adaptive internal temperature estimation algorithms by blending physics-based modeling with machine learning, in order to address the key challenges in data collection, model parameterization, and estimator design that traditionally hinder both approaches. In this framework, a physics-based model is leveraged to generate simulation data that includes different operating scenarios by sweeping the model parameters and input profiles. Such a cheap simulation dataset can be used to pre-train the machine learning algorithm to capture the underlying mapping relationship. To bridge the simulation-to-reality gap resulting from imperfect modeling, transfer learning with unsupervised domain adaptation is applied to fine-tune the pre-trained machine learning model, by using limited operational data (without internal temperature values) from target batteries. The proposed framework is validated under different operating conditions and across multiple cylindrical batteries with convective air cooling, achieving a root mean square error of 0.5 {\deg}C when relying solely on prior knowledge of battery thermal properties, and less than 0.1 {\deg}C when using thermal parameters close to the ground truth. Furthermore, the role of the simulation data quality in the proposed framework has been comprehensively investigated to identify promising ways of synthetic data generation to guarantee the performance of the machine learning model. |
| 2025-09-12 | [Acetrans: An Autonomous Corridor-Based and Efficient UAV Suspended Transport System](http://arxiv.org/abs/2509.10349v1) | Weiyan Lu, Huizhe Li et al. | Unmanned aerial vehicles (UAVs) with suspended payloads offer significant advantages for aerial transportation in complex and cluttered environments. However, existing systems face critical limitations, including unreliable perception of the cable-payload dynamics, inefficient planning in large-scale environments, and the inability to guarantee whole-body safety under cable bending and external disturbances. This paper presents Acetrans, an Autonomous, Corridor-based, and Efficient UAV suspended transport system that addresses these challenges through a unified perception, planning, and control framework. A LiDAR-IMU fusion module is proposed to jointly estimate both payload pose and cable shape under taut and bent modes, enabling robust whole-body state estimation and real-time filtering of cable point clouds. To enhance planning scalability, we introduce the Multi-size-Aware Configuration-space Iterative Regional Inflation (MACIRI) algorithm, which generates safe flight corridors while accounting for varying UAV and payload geometries. A spatio-temporal, corridor-constrained trajectory optimization scheme is then developed to ensure dynamically feasible and collision-free trajectories. Finally, a nonlinear model predictive controller (NMPC) augmented with cable-bending constraints provides robust whole-body safety during execution. Simulation and experimental results validate the effectiveness of Acetrans, demonstrating substantial improvements in perception accuracy, planning efficiency, and control safety compared to state-of-the-art methods. |
| 2025-09-12 | [We Need a New Ethics for a World of AI Agents](http://arxiv.org/abs/2509.10289v1) | Iason Gabriel, Geoff Keeling et al. | The deployment of capable AI agents raises fresh questions about safety, human-machine relationships and social coordination. We argue for greater engagement by scientists, scholars, engineers and policymakers with the implications of a world increasingly populated by AI agents. We explore key challenges that must be addressed to ensure that interactions between humans and agents, and among agents themselves, remain broadly beneficial. |
| 2025-09-12 | [A Holistic Architecture for Monitoring and Optimization of Robust Multi-Agent Path Finding Plan Execution](http://arxiv.org/abs/2509.10284v1) | David Zahrádka, Denisa Mužíková et al. | The goal of Multi-Agent Path Finding (MAPF) is to find a set of paths for a fleet of agents moving in a shared environment such that the agents reach their goals without colliding with each other. In practice, some of the robots executing the plan may get delayed, which can introduce collision risk. Although robust execution methods are used to ensure safety even in the presence of delays, the delays may still have a significant impact on the duration of the execution. At some point, the accumulated delays may become significant enough that instead of continuing with the execution of the original plan, even if it was optimal, there may now exist an alternate plan which will lead to a shorter execution. However, the problem is how to decide when to search for the alternate plan, since it is a costly procedure. In this paper, we propose a holistic architecture for robust execution of MAPF plans, its monitoring and optimization. We exploit a robust execution method called Action Dependency Graph to maintain an estimate of the expected execution duration during the plan's execution. This estimate is used to predict the potential that finding an alternate plan would lead to shorter execution. We empirically evaluate the architecture in experiments in a real-time simulator which we designed to mimic our real-life demonstrator of an autonomous warehouse robotic fleet. |
| 2025-09-12 | [On the Geometric Accuracy of Implicit and Primitive-based Representations Derived from View Rendering Constraints](http://arxiv.org/abs/2509.10241v1) | Elias De Smijter, Renaud Detry et al. | We present the first systematic comparison of implicit and explicit Novel View Synthesis methods for space-based 3D object reconstruction, evaluating the role of appearance embeddings. While embeddings improve photometric fidelity by modeling lighting variation, we show they do not translate into meaningful gains in geometric accuracy - a critical requirement for space robotics applications. Using the SPEED+ dataset, we compare K-Planes, Gaussian Splatting, and Convex Splatting, and demonstrate that embeddings primarily reduce the number of primitives needed for explicit methods rather than enhancing geometric fidelity. Moreover, convex splatting achieves more compact and clutter-free representations than Gaussian splatting, offering advantages for safety-critical applications such as interaction and collision avoidance. Our findings clarify the limits of appearance embeddings for geometry-centric tasks and highlight trade-offs between reconstruction quality and representation efficiency in space scenarios. |
| 2025-09-12 | [Using joint models in phase I dose-finding designs in oncology: considerations for frequentist approaches](http://arxiv.org/abs/2509.10238v1) | Xijin Chen, Pavel Mozgunov et al. | Dose-finding trials for oncology studies are traditionally designed to assess safety in the early stages of drug development. With the rise of molecularly targeted therapies and immuno-oncology compounds, biomarker-driven approaches have gained significant importance. In this paper, we propose a novel approach that incorporates multiple values of a predictive biomarker to assist in evaluating binary toxicity outcomes using the factorization of a joint model in phase I dose-finding oncology trials. The proposed joint model framework, which utilizes additional repeated biomarker values as an early predictive marker for potential toxicity, is compared to the likelihood-based continual reassessment method (CRM) using only binary toxicity data, across various dose-toxicity relationship scenarios. Our findings highlight a critical limitation of likelihood-based approaches in early-phase dose-finding studies with small sample sizes: estimation challenges that have been previously overlooked in the phase I dose-escalation setting. We explore potential remedies to address these challenges and emphasize the appropriate use of likelihood-based methods. Simulation results demonstrate that the proposed joint model framework, by integrating biomarker information, can alleviate estimation problems in the the likelihood-based continual reassessment method (CRM) and improve the proportion of correct selection. However, we highlight that the inherent data limitations in early-phase dose-finding studies remain a significant challenge that cannot fully be overcomed in the frequentist framework. |
| 2025-09-12 | [A Certifiable Machine Learning-Based Pipeline to Predict Fatigue Life of Aircraft Structures](http://arxiv.org/abs/2509.10227v1) | Ángel Ladrón, Miguel Sánchez-Domínguez et al. | Fatigue life prediction is essential in both the design and operational phases of any aircraft, and in this sense safety in the aerospace industry requires early detection of fatigue cracks to prevent in-flight failures. Robust and precise fatigue life predictors are thus essential to ensure safety. Traditional engineering methods, while reliable, are time consuming and involve complex workflows, including steps such as conducting several Finite Element Method (FEM) simulations, deriving the expected loading spectrum, and applying cycle counting techniques like peak-valley or rainflow counting. These steps often require collaboration between multiple teams and tools, added to the computational time and effort required to achieve fatigue life predictions. Machine learning (ML) offers a promising complement to traditional fatigue life estimation methods, enabling faster iterations and generalization, providing quick estimates that guide decisions alongside conventional simulations.   In this paper, we present a ML-based pipeline that aims to estimate the fatigue life of different aircraft wing locations given the flight parameters of the different missions that the aircraft will be operating throughout its operational life. We validate the pipeline in a realistic use case of fatigue life estimation, yielding accurate predictions alongside a thorough statistical validation and uncertainty quantification. Our pipeline constitutes a complement to traditional methodologies by reducing the amount of costly simulations and, thereby, lowering the required computational and human resources. |
| 2025-09-12 | [Virtual Agent Economies](http://arxiv.org/abs/2509.10147v1) | Nenad Tomasev, Matija Franklin et al. | The rapid adoption of autonomous AI agents is giving rise to a new economic layer where agents transact and coordinate at scales and speeds beyond direct human oversight. We propose the "sandbox economy" as a framework for analyzing this emergent system, characterizing it along two key dimensions: its origins (emergent vs. intentional) and its degree of separateness from the established human economy (permeable vs. impermeable). Our current trajectory points toward a spontaneous emergence of a vast and highly permeable AI agent economy, presenting us with opportunities for an unprecedented degree of coordination as well as significant challenges, including systemic economic risk and exacerbated inequality. Here we discuss a number of possible design choices that may lead to safely steerable AI agent markets. In particular, we consider auction mechanisms for fair resource allocation and preference resolution, the design of AI "mission economies" to coordinate around achieving collective goals, and socio-technical infrastructure needed to ensure trust, safety, and accountability. By doing this, we argue for the proactive design of steerable agent markets to ensure the coming technological shift aligns with humanity's long-term collective flourishing. |
| 2025-09-12 | [Scalable Synthesis and Verification of String Stable Neural Certificates for Interconnected Systems](http://arxiv.org/abs/2509.10118v1) | Jingyuan Zhou, Haoze Wu et al. | Ensuring string stability is critical for the safety and efficiency of large-scale interconnected systems. Although learning-based controllers (e.g., those based on reinforcement learning) have demonstrated strong performance in complex control scenarios, their black-box nature hinders formal guarantees of string stability. To address this gap, we propose a novel verification and synthesis framework that integrates discrete-time scalable input-to-state stability (sISS) with neural network verification to formally guarantee string stability in interconnected systems. Our contributions are four-fold. First, we establish a formal framework for synthesizing and robustly verifying discrete-time scalable input-to-state stability (sISS) certificates for neural network-based interconnected systems. Specifically, our approach extends the notion of sISS to discrete-time settings, constructs neural sISS certificates, and introduces a verification procedure that ensures string stability while explicitly accounting for discrepancies between the true dynamics and their neural approximations. Second, we establish theoretical foundations and algorithms to scale the training and verification pipeline to large-scale interconnected systems. Third, we extend the framework to handle systems with external control inputs, thereby allowing the joint synthesis and verification of neural certificates and controllers. Fourth, we validate our approach in scenarios of mixed-autonomy platoons, drone formations, and microgrids. Numerical simulations show that the proposed framework not only guarantees sISS with minimal degradation in control performance but also efficiently trains and verifies controllers for large-scale interconnected systems under specific practical conditions. |
| 2025-09-12 | [VARCO-VISION-2.0 Technical Report](http://arxiv.org/abs/2509.10105v1) | Young-rok Cha, Jeongho Ju et al. | We introduce VARCO-VISION-2.0, an open-weight bilingual vision-language model (VLM) for Korean and English with improved capabilities compared to the previous model VARCO-VISION-14B. The model supports multi-image understanding for complex inputs such as documents, charts, and tables, and delivers layoutaware OCR by predicting both textual content and its spatial location. Trained with a four-stage curriculum with memory-efficient techniques, the model achieves enhanced multimodal alignment, while preserving core language abilities and improving safety via preference optimization. Extensive benchmark evaluations demonstrate strong spatial grounding and competitive results for both languages, with the 14B model achieving 8th place on the OpenCompass VLM leaderboard among models of comparable scale. Alongside the 14B-scale model, we release a 1.7B version optimized for on-device deployment. We believe these models advance the development of bilingual VLMs and their practical applications. Two variants of VARCO-VISION-2.0 are available at Hugging Face: a full-scale 14B model and a lightweight 1.7B model. |
| 2025-09-12 | [Uncertainty-Aware Tabular Prediction: Evaluating VBLL-Enhanced TabPFN in Safety-Critical Medical Data](http://arxiv.org/abs/2509.10048v1) | Madhushan Ramalingam | Predictive models are being increasingly used across a wide range of domains, including safety-critical applications such as medical diagnosis and criminal justice. Reliable uncertainty estimation is a crucial task in such settings. Tabular Prior-data Fitted Network (TabPFN) is a recently proposed machine learning foundation model for tabular dataset, which uses a generative transformer architecture. Variational Bayesian Last Layers (VBLL) is a state-of-the-art lightweight variational formulation that effectively improves uncertainty estimation with minimal computational overhead. In this work we aim to evaluate the performance of VBLL integrated with the recently proposed TabPFN in uncertainty calibration. Our experiments, conducted on three benchmark medical tabular datasets, compare the performance of the original TabPFN and the VBLL-integrated version. Contrary to expectations, we observed that original TabPFN consistently outperforms VBLL integrated TabPFN in uncertainty calibration across all datasets. |
| 2025-09-12 | [Towards simulation-based optimization of compliant fingers for high-speed connector assembly](http://arxiv.org/abs/2509.10012v1) | Richard Matthias Hartisch, Alexander Rother et al. | Mechanical compliance is a key design parameter for dynamic contact-rich manipulation, affecting task success and safety robustness over contact geometry variation. Design of soft robotic structures, such as compliant fingers, requires choosing design parameters which affect geometry and stiffness, and therefore manipulation performance and robustness. Today, these parameters are chosen through either hardware iteration, which takes significant development time, or simplified models (e.g. planar), which can't address complex manipulation task objectives. Improvements in dynamic simulation, especially with contact and friction modeling, present a potential design tool for mechanical compliance. We propose a simulation-based design tool for compliant mechanisms which allows design with respect to task-level objectives, such as success rate. This is applied to optimize design parameters of a structured compliant finger to reduce failure cases inside a tolerance window in insertion tasks. The improvement in robustness is then validated on a real robot using tasks from the benchmark NIST task board. The finger stiffness affects the tolerance window: optimized parameters can increase tolerable ranges by a factor of 2.29, with workpiece variation up to 8.6 mm being compensated. However, the trends remain task-specific. In some tasks, the highest stiffness yields the widest tolerable range, whereas in others the opposite is observed, motivating need for design tools which can consider application-specific geometry and dynamics. |
| 2025-09-12 | [Detection of Anomalous Behavior in Robot Systems Based on Machine Learning](http://arxiv.org/abs/2509.09953v1) | Mahfuzul I. Nissan, Sharmin Aktar | Ensuring the safe and reliable operation of robotic systems is paramount to prevent potential disasters and safeguard human well-being. Despite rigorous design and engineering practices, these systems can still experience malfunctions, leading to safety risks. In this study, we present a machine learning-based approach for detecting anomalies in system logs to enhance the safety and reliability of robotic systems. We collected logs from two distinct scenarios using CoppeliaSim and comparatively evaluated several machine learning models, including Logistic Regression (LR), Support Vector Machine (SVM), and an Autoencoder. Our system was evaluated in a quadcopter context (Context 1) and a Pioneer robot context (Context 2). Results showed that while LR demonstrated superior performance in Context 1, the Autoencoder model proved to be the most effective in Context 2. This highlights that the optimal model choice is context-dependent, likely due to the varying complexity of anomalies across different robotic platforms. This research underscores the value of a comparative approach and demonstrates the particular strengths of autoencoders for detecting complex anomalies in robotic systems. |
| 2025-09-12 | [SmartCoder-R1: Towards Secure and Explainable Smart Contract Generation with Security-Aware Group Relative Policy Optimization](http://arxiv.org/abs/2509.09942v1) | Lei Yu, Jingyuan Zhang et al. | Smart contracts automate the management of high-value assets, where vulnerabilities can lead to catastrophic financial losses. This challenge is amplified in Large Language Models (LLMs) by two interconnected failures: they operate as unauditable "black boxes" lacking a transparent reasoning process, and consequently, generate code riddled with critical security vulnerabilities. To address both issues, we propose SmartCoder-R1 (based on Qwen2.5-Coder-7B), a novel framework for secure and explainable smart contract generation. It begins with Continual Pre-training (CPT) to specialize the model. We then apply Long Chain-of-Thought Supervised Fine-Tuning (L-CoT SFT) on 7,998 expert-validated reasoning-and-code samples to train the model to emulate human security analysis. Finally, to directly mitigate vulnerabilities, we employ Security-Aware Group Relative Policy Optimization (S-GRPO), a reinforcement learning phase that refines the generation policy by optimizing a weighted reward signal for compilation success, security compliance, and format correctness. Evaluated against 17 baselines on a benchmark of 756 real-world functions, SmartCoder-R1 establishes a new state of the art, achieving top performance across five key metrics: a ComPass of 87.70%, a VulRate of 8.60%, a SafeAval of 80.16%, a FuncRate of 53.84%, and a FullRate of 50.53%. This FullRate marks a 45.79% relative improvement over the strongest baseline, DeepSeek-R1. Crucially, its generated reasoning also excels in human evaluations, achieving high-quality ratings for Functionality (82.7%), Security (85.3%), and Clarity (90.7%). |
| 2025-09-11 | [Self-Augmented Robot Trajectory: Efficient Imitation Learning via Safe Self-augmentation with Demonstrator-annotated Precision](http://arxiv.org/abs/2509.09893v1) | Hanbit Oh, Masaki Murooka et al. | Imitation learning is a promising paradigm for training robot agents; however, standard approaches typically require substantial data acquisition -- via numerous demonstrations or random exploration -- to ensure reliable performance. Although exploration reduces human effort, it lacks safety guarantees and often results in frequent collisions -- particularly in clearance-limited tasks (e.g., peg-in-hole) -- thereby, necessitating manual environmental resets and imposing additional human burden. This study proposes Self-Augmented Robot Trajectory (SART), a framework that enables policy learning from a single human demonstration, while safely expanding the dataset through autonomous augmentation. SART consists of two stages: (1) human teaching only once, where a single demonstration is provided and precision boundaries -- represented as spheres around key waypoints -- are annotated, followed by one environment reset; (2) robot self-augmentation, where the robot generates diverse, collision-free trajectories within these boundaries and reconnects to the original demonstration. This design improves the data collection efficiency by minimizing human effort while ensuring safety. Extensive evaluations in simulation and real-world manipulation tasks show that SART achieves substantially higher success rates than policies trained solely on human-collected demonstrations. Video results available at https://sites.google.com/view/sart-il . |
| 2025-09-11 | [Distinguishing Startle from Surprise Events Based on Physiological Signals](http://arxiv.org/abs/2509.09799v1) | Mansi Sharma, Alexandre Duchevet et al. | Unexpected events can impair attention and delay decision-making, posing serious safety risks in high-risk environments such as aviation. In particular, reactions like startle and surprise can impact pilot performance in different ways, yet are often hard to distinguish in practice. Existing research has largely studied these reactions separately, with limited focus on their combined effects or how to differentiate them using physiological data. In this work, we address this gap by distinguishing between startle and surprise events based on physiological signals using machine learning and multi-modal fusion strategies. Our results demonstrate that these events can be reliably predicted, achieving a highest mean accuracy of 85.7% with SVM and Late Fusion. To further validate the robustness of our model, we extended the evaluation to include a baseline condition, successfully differentiating between Startle, Surprise, and Baseline states with a highest mean accuracy of 74.9% with XGBoost and Late Fusion. |
| 2025-09-11 | [Steering MoE LLMs via Expert (De)Activation](http://arxiv.org/abs/2509.09660v1) | Mohsen Fayyaz, Ali Modarressi et al. | Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts. |

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



