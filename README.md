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
| 2025-04-23 | [Meta-Learning Online Dynamics Model Adaptation in Off-Road Autonomous Driving](http://arxiv.org/abs/2504.16923v1) | Jacob Levy, Jason Gibson et al. | High-speed off-road autonomous driving presents unique challenges due to complex, evolving terrain characteristics and the difficulty of accurately modeling terrain-vehicle interactions. While dynamics models used in model-based control can be learned from real-world data, they often struggle to generalize to unseen terrain, making real-time adaptation essential. We propose a novel framework that combines a Kalman filter-based online adaptation scheme with meta-learned parameters to address these challenges. Offline meta-learning optimizes the basis functions along which adaptation occurs, as well as the adaptation parameters, while online adaptation dynamically adjusts the onboard dynamics model in real time for model-based control. We validate our approach through extensive experiments, including real-world testing on a full-scale autonomous off-road vehicle, demonstrating that our method outperforms baseline approaches in prediction accuracy, performance, and safety metrics, particularly in safety-critical scenarios. Our results underscore the effectiveness of meta-learned dynamics model adaptation, advancing the development of reliable autonomous systems capable of navigating diverse and unseen environments. Video is available at: https://youtu.be/cCKHHrDRQEA |
| 2025-04-23 | [Learning Verifiable Control Policies Using Relaxed Verification](http://arxiv.org/abs/2504.16879v1) | Puja Chaudhury, Alexander Estornell et al. | To provide safety guarantees for learning-based control systems, recent work has developed formal verification methods to apply after training ends. However, if the trained policy does not meet the specifications, or there is conservatism in the verification algorithm, establishing these guarantees may not be possible. Instead, this work proposes to perform verification throughout training to ultimately aim for policies whose properties can be evaluated throughout runtime with lightweight, relaxed verification algorithms. The approach is to use differentiable reachability analysis and incorporate new components into the loss function. Numerical experiments on a quadrotor model and unicycle model highlight the ability of this approach to lead to learned control policies that satisfy desired reach-avoid and invariance specifications. |
| 2025-04-23 | [Hybrid Reinforcement Learning and Model Predictive Control for Adaptive Control of Hydrogen-Diesel Dual-Fuel Combustion](http://arxiv.org/abs/2504.16875v1) | Julian Bedei, Murray McBain et al. | Reinforcement Learning (RL) and Machine Learning Integrated Model Predictive Control (ML-MPC) are promising approaches for optimizing hydrogen-diesel dual-fuel engine control, as they can effectively control multiple-input multiple-output systems and nonlinear processes. ML-MPC is advantageous for providing safe and optimal controls, ensuring the engine operates within predefined safety limits. In contrast, RL is distinguished by its adaptability to changing conditions through its learning-based approach. However, the practical implementation of either method alone poses challenges. RL requires high variance in control inputs during early learning phases, which can pose risks to the system by potentially executing unsafe actions, leading to mechanical damage. Conversely, ML-MPC relies on an accurate system model to generate optimal control inputs and has limited adaptability to system drifts, such as injector aging, which naturally occur in engine applications. To address these limitations, this study proposes a hybrid RL and ML-MPC approach that uses an ML-MPC framework while incorporating an RL agent to dynamically adjust the ML-MPC load tracking reference in response to changes in the environment. At the same time, the ML-MPC ensures that actions stay safe throughout the RL agent's exploration. To evaluate the effectiveness of this approach, fuel pressure is deliberately varied to introduce a model-plant mismatch between the ML-MPC and the engine test bench. The result of this mismatch is a root mean square error (RMSE) in indicated mean effective pressure of 0.57 bar when running the ML-MPC. The experimental results demonstrate that RL successfully adapts to changing boundary conditions by altering the tracking reference while ML-MPC ensures safe control inputs. The quantitative improvement in load tracking by implementing RL is an RSME of 0.44 bar. |
| 2025-04-23 | [Improving Significant Wave Height Prediction Using Chronos Models](http://arxiv.org/abs/2504.16834v1) | Yilin Zhai, Hongyuan Shi et al. | Accurate wave height prediction is critical for maritime safety and coastal resilience, yet conventional physics-based models and traditional machine learning methods face challenges in computational efficiency and nonlinear dynamics modeling. This study introduces Chronos, the first implementation of a large language model (LLM)-powered temporal architecture (Chronos) optimized for wave forecasting. Through advanced temporal pattern recognition applied to historical wave data from three strategically chosen marine zones in the Northwest Pacific basin, our framework achieves multimodal improvements: (1) 14.3% reduction in training time with 2.5x faster inference speed compared to PatchTST baselines, achieving 0.575 mean absolute scaled error (MASE) units; (2) superior short-term forecasting (1-24h) across comprehensive metrics; (3) sustained predictive leadership in extended-range forecasts (1-120h); and (4) demonstrated zero-shot capability maintaining median performance (rank 4/12) against specialized operational models. This LLM-enhanced temporal modeling paradigm establishes a new standard in wave prediction, offering both computationally efficient solutions and a transferable framework for complex geophysical systems modeling. |
| 2025-04-23 | [Association-Based Track-Before-Detect with Object Contribution Probabilities](http://arxiv.org/abs/2504.16814v1) | Thomas Kropfreiter, Jason L. Williams et al. | Multiobject tracking provides situational awareness that enables new applications for modern convenience, applied ocean sciences, public safety, and homeland security. In many multiobject tracking applications, including radar and sonar tracking, after coherent prefiltering of the received signal, measurement data is typically structured in cells, where each cell represent, e.g., a different range and bearing value. While conventional detect-then-track (DTT) multiobject tracking approaches convert the cell-structured data within a detection phase into so-called point measurements in order to reduce the amount of data, track-before-detect (TBD) methods process the cell-structured data directly, avoiding a potential information loss. However, many TBD tracking methods are computationally intensive and achieve a reduced tracking accuracy when objects interact, i.e., when they come into close proximity. We here counteract these difficulties by introducing the concept of probabilistic object-to-cell contributions. As many conventional DTT methods, our approach uses a probabilistic association of objects with data cells, and a new object contribution model with corresponding object contribution probabilities to further associate cell contributions to objects that occupy the same data cell. Furthermore, to keep the computational complexity and filter runtimes low, we here use an efficient Poisson multi-Bernoulli filtering approach in combination with the application of belief propagation for fast probabilistic data association. We demonstrate numerically that our method achieves significantly increased tracking performance compared to state-of-the-art TBD tracking approaches, where performance differences are particularly pronounced when multiple objects interact. |
| 2025-04-23 | [Evaluating the Impact of CT-to-RED Calibration Curves on Dosimetric Accuracy in Brain Radiotherapy Dose Distribution](http://arxiv.org/abs/2504.16805v1) | Islam G. Ali, Wael M. Daabis et al. | Accurate dose calculation is crucial in radiotherapy, as tissue relative electron densities (RED) derived from CT scans play a vital role. This study investigated the impact of different CT-to-RED calibration curves on brain cancer treatment plans. Three calibration curves were compared: CIRS phantom-derived, Catphan phantom-derived, and the default curve in the Monaco Treatment Planning System. Ten volumetric modulated arc therapy (VMAT) plans were generated and recalculated using each curve. Dosimetric parameters for Planning Target Volume (PTV) and Organs at Risk (OARs) were analyzed. Results showed significant differences in PTV dose distribution between the CIRS-derived and default curves, while no significant differences were found between Catphan-derived and default curves. The CIRS-derived curve demonstrated superior performance in representing brain tissue electron densities. These findings emphasize the importance of using site-specific CT-to-RED calibration curves for accurate dose calculations in brain radiotherapy, potentially improving treatment safety and efficacy |
| 2025-04-23 | [Reduction of $ε$-expanded Feynman integrals](http://arxiv.org/abs/2504.16766v1) | Yan-Qing Ma, Cong-Hao Qin et al. | Since Feynman integrals (FIs) at higher spacetime dimensions are free of infrared and collinear divergence--and their ultraviolet divergences can be systematically subtracted--this allows us to construct a wide range of locally finite Feynman integrals. Especially, we propose a method named $\bar{R}$-operation to subtract out ultraviolet divergences that at the same time preserves infrared and collinear safety of the original FI. By expressing these locally finite FIs in terms of master integrals and imposing constraints on their $\epsilon$-expanded forms, we reduce the $\epsilon$-expanded master integrals to a minimal basis. We provide an automated package to identify such constraints, offering a tool useful for high-order perturbative computations. |
| 2025-04-23 | [Frequency-Compensated Network for Daily Arctic Sea Ice Concentration Prediction](http://arxiv.org/abs/2504.16745v1) | Jialiang Zhang, Feng Gao et al. | Accurately forecasting sea ice concentration (SIC) in the Arctic is critical to global ecosystem health and navigation safety. However, current methods still is confronted with two challenges: 1) these methods rarely explore the long-term feature dependencies in the frequency domain. 2) they can hardly preserve the high-frequency details, and the changes in the marginal area of the sea ice cannot be accurately captured. To this end, we present a Frequency-Compensated Network (FCNet) for Arctic SIC prediction on a daily basis. In particular, we design a dual-branch network, including branches for frequency feature extraction and convolutional feature extraction. For frequency feature extraction, we design an adaptive frequency filter block, which integrates trainable layers with Fourier-based filters. By adding frequency features, the FCNet can achieve refined prediction of edges and details. For convolutional feature extraction, we propose a high-frequency enhancement block to separate high and low-frequency information. Moreover, high-frequency features are enhanced via channel-wise attention, and temporal attention unit is employed for low-frequency feature extraction to capture long-range sea ice changes. Extensive experiments are conducted on a satellite-derived daily SIC dataset, and the results verify the effectiveness of the proposed FCNet. Our codes and data will be made public available at: https://github.com/oucailab/FCNet . |
| 2025-04-23 | [DYNUS: Uncertainty-aware Trajectory Planner in Dynamic Unknown Environments](http://arxiv.org/abs/2504.16734v1) | Kota Kondo, Mason Peterson et al. | This paper introduces DYNUS, an uncertainty-aware trajectory planner designed for dynamic unknown environments. Operating in such settings presents many challenges -- most notably, because the agent cannot predict the ground-truth future paths of obstacles, a previously planned trajectory can become unsafe at any moment, requiring rapid replanning to avoid collisions.   Recently developed planners have used soft-constraint approaches to achieve the necessary fast computation times; however, these methods do not guarantee collision-free paths even with static obstacles. In contrast, hard-constraint methods ensure collision-free safety, but typically have longer computation times.   To address these issues, we propose three key contributions. First, the DYNUS Global Planner (DGP) and Temporal Safe Corridor Generation operate in spatio-temporal space and handle both static and dynamic obstacles in the 3D environment. Second, the Safe Planning Framework leverages a combination of exploratory, safe, and contingency trajectories to flexibly re-route when potential future collisions with dynamic obstacles are detected. Finally, the Fast Hard-Constraint Local Trajectory Formulation uses a variable elimination approach to reduce the problem size and enable faster computation by pre-computing dependencies between free and dependent variables while still ensuring collision-free trajectories.   We evaluated DYNUS in a variety of simulations, including dense forests, confined office spaces, cave systems, and dynamic environments. Our experiments show that DYNUS achieves a success rate of 100% and travel times that are approximately 25.0% faster than state-of-the-art methods. We also evaluated DYNUS on multiple platforms -- a quadrotor, a wheeled robot, and a quadruped -- in both simulation and hardware experiments. |
| 2025-04-23 | [Offline Robotic World Model: Learning Robotic Policies without a Physics Simulator](http://arxiv.org/abs/2504.16680v1) | Chenhao Li, Andreas Krause et al. | Reinforcement Learning (RL) has demonstrated impressive capabilities in robotic control but remains challenging due to high sample complexity, safety concerns, and the sim-to-real gap. While offline RL eliminates the need for risky real-world exploration by learning from pre-collected data, it suffers from distributional shift, limiting policy generalization. Model-Based RL (MBRL) addresses this by leveraging predictive models for synthetic rollouts, yet existing approaches often lack robust uncertainty estimation, leading to compounding errors in offline settings. We introduce Offline Robotic World Model (RWM-O), a model-based approach that explicitly estimates epistemic uncertainty to improve policy learning without reliance on a physics simulator. By integrating these uncertainty estimates into policy optimization, our approach penalizes unreliable transitions, reducing overfitting to model errors and enhancing stability. Experimental results show that RWM-O improves generalization and safety, enabling policy learning purely from real-world data and advancing scalable, data-efficient RL for robotics. |
| 2025-04-23 | [3D-1D modelling of cranial plate heating induced by low or medium frequency magnetic fields](http://arxiv.org/abs/2504.16600v1) | Alessandro Arduino, Oriano Bottauscio et al. | Safety assessment of patients with one-dimensionally structured passive implants, like cranial plates or stents, exposed to low or medium frequency magnetic fields, like those generated in magnetic resonance imaging or magnetic hyperthermia, can be challenging, because of the different length scales of the implant and the human body. Most of the methods used to estimate the heating induced near such implants neglect the presence of the metallic materials within the body, modeling the metal as thermal seeds. To overcome this limitation, a novel numerical approach that solves three-dimensional and one-dimensional coupled problems is proposed. This method leads to improved results by modelling the thermal diffusion through the highly conductive metallic implants. A comparison of the proposed method predictions with measurements performed on a cranial plate exposed to the magnetic field generated by a gradient coil system for magnetic resonance imaging is presented, showing an improved accuracy up to 25 % with respect to the method based on thermal seeds. The proposed method is finally applied to a magnetic hyperthermia case study in which a patient with a cranial plate is exposed to the magnetic field generated by a collar-type magnetic hyperthermia applicator for neck tumour treatment, predicting a temperature increase in proximity of the implant that is 10 % lower than the one overestimated by relying on thermal seeds. |
| 2025-04-23 | [Streetscape Analysis with Generative AI (SAGAI): Vision-Language Assessment and Mapping of Urban Scenes](http://arxiv.org/abs/2504.16538v1) | Joan Perez, Giovanni Fusco | Streetscapes are an essential component of urban space. Their assessment is presently either limited to morphometric properties of their mass skeleton or requires labor-intensive qualitative evaluations of visually perceived qualities. This paper introduces SAGAI: Streetscape Analysis with Generative Artificial Intelligence, a modular workflow for scoring street-level urban scenes using open-access data and vision-language models. SAGAI integrates OpenStreetMap geometries, Google Street View imagery, and a lightweight version of the LLaVA model to generate structured spatial indicators from images via customizable natural language prompts. The pipeline includes an automated mapping module that aggregates visual scores at both the point and street levels, enabling direct cartographic interpretation. It operates without task-specific training or proprietary software dependencies, supporting scalable and interpretable analysis of urban environments. Two exploratory case studies in Nice and Vienna illustrate SAGAI's capacity to produce geospatial outputs from vision-language inference. The initial results show strong performance for binary urban-rural scene classification, moderate precision in commercial feature detection, and lower estimates, but still informative, of sidewalk width. Fully deployable by any user, SAGAI can be easily adapted to a wide range of urban research themes, such as walkability, safety, or urban design, through prompt modification alone. |
| 2025-04-23 | [SafeSpect: Safety-First Augmented Reality Heads-up Display for Drone Inspections](http://arxiv.org/abs/2504.16533v1) | Peisen Xu, Jérémie Garcia et al. | Current tablet-based interfaces for drone operations often impose a heavy cognitive load on pilots and reduce situational awareness by dividing attention between the video feed and the real world. To address these challenges, we designed a heads-up augmented reality (AR) interface that overlays in-situ information to support drone pilots in safety-critical tasks. Through participatory design workshops with professional pilots, we identified key features and developed an adaptive AR interface that dynamically switches between task and safety views to prevent information overload. We evaluated our prototype by creating a realistic building inspection task and comparing three interfaces: a 2D tablet, a static AR, and our adaptive AR design. A user study with 15 participants showed that the AR interface improved access to safety information, while the adaptive AR interface reduced cognitive load and enhanced situational awareness without compromising task performance. We offer design insights for developing safety-first heads-up AR interfaces. |
| 2025-04-23 | [Amplified Vulnerabilities: Structured Jailbreak Attacks on LLM-based Multi-Agent Debate](http://arxiv.org/abs/2504.16489v1) | Senmao Qi, Yifei Zou et al. | Multi-Agent Debate (MAD), leveraging collaborative interactions among Large Language Models (LLMs), aim to enhance reasoning capabilities in complex tasks. However, the security implications of their iterative dialogues and role-playing characteristics, particularly susceptibility to jailbreak attacks eliciting harmful content, remain critically underexplored. This paper systematically investigates the jailbreak vulnerabilities of four prominent MAD frameworks built upon leading commercial LLMs (GPT-4o, GPT-4, GPT-3.5-turbo, and DeepSeek) without compromising internal agents. We introduce a novel structured prompt-rewriting framework specifically designed to exploit MAD dynamics via narrative encapsulation, role-driven escalation, iterative refinement, and rhetorical obfuscation. Our extensive experiments demonstrate that MAD systems are inherently more vulnerable than single-agent setups. Crucially, our proposed attack methodology significantly amplifies this fragility, increasing average harmfulness from 28.14% to 80.34% and achieving attack success rates as high as 80% in certain scenarios. These findings reveal intrinsic vulnerabilities in MAD architectures and underscore the urgent need for robust, specialized defenses prior to real-world deployment. |
| 2025-04-23 | [The Dodecacopter: a Versatile Multirotor System of Dodecahedron-Shaped Modules](http://arxiv.org/abs/2504.16475v1) | Kévin Garanger, Thanakorn Khamvilai et al. | With the promise of greater safety and adaptability, modular reconfigurable uncrewed air vehicles have been proposed as unique, versatile platforms holding the potential to replace multiple types of monolithic vehicles at once. State-of-the-art rigidly assembled modular vehicles are generally two-dimensional configurations in which the rotors are coplanar and assume the shape of a "flight array". We introduce the Dodecacopter, a new type of modular rotorcraft where all modules take the shape of a regular dodecahedron, allowing the creation of richer sets of configurations beyond flight arrays. In particular, we show how the chosen module design can be used to create three-dimensional and fully actuated configurations. We justify the relevance of these types of configurations in terms of their structural and actuation properties with various performance indicators. Given the broad range of configurations and capabilities that can be achieved with our proposed design, we formulate tractable optimization programs to find optimal configurations given structural and actuation constraints. Finally, a prototype of such a vehicle is presented along with results of performed flights in multiple configurations. |
| 2025-04-23 | [ERASER: Efficient RTL FAult Simulation Framework with Trimmed Execution Redundancy](http://arxiv.org/abs/2504.16473v1) | Jiaping Tang, Jianan Mu et al. | As intelligent computing devices increasingly integrate into human life, ensuring the functional safety of the corresponding electronic chips becomes more critical. A key metric for functional safety is achieving a sufficient fault coverage. To meet this requirement, extensive time-consuming fault simulation of the RTL code is necessary during the chip design phase.The main overhead in RTL fault simulation comes from simulating behavioral nodes (always blocks). Due to the limited fault propagation capacity, fault simulation results often match the good simulation results for many behavioral nodes. A key strategy for accelerating RTL fault simulation is the identification and elimination of redundant simulations. Existing methods detect redundant executions by examining whether the fault inputs to each RTL node are consistent with the good inputs. However, we observe that this input comparison mechanism overlooks a significant amount of implicit redundant execution: although the fault inputs differ from the good inputs, the node's execution results remain unchanged. Our experiments reveal that this overlooked redundant execution constitutes nearly half of the total execution overhead of behavioral nodes, becoming a significant bottleneck in current RTL fault simulation. The underlying reason for this overlooked redundancy is that, in these cases, the true execution paths within the behavioral nodes are not affected by the changes in input values. In this work, we propose a behavior-level redundancy detection algorithm that focuses on the true execution paths. Building on the elimination of redundant executions, we further developed an efficient RTL fault simulation framework, Eraser.Experimental results show that compared to commercial tools, under the same fault coverage, our framework achieves a 3.9 $\times$ improvement in simulation performance on average. |
| 2025-04-23 | [iTFKAN: Interpretable Time Series Forecasting with Kolmogorov-Arnold Network](http://arxiv.org/abs/2504.16432v1) | Ziran Liang, Rui An et al. | As time evolves, data within specific domains exhibit predictability that motivates time series forecasting to predict future trends from historical data. However, current deep forecasting methods can achieve promising performance but generally lack interpretability, hindering trustworthiness and practical deployment in safety-critical applications such as auto-driving and healthcare. In this paper, we propose a novel interpretable model, iTFKAN, for credible time series forecasting. iTFKAN enables further exploration of model decision rationales and underlying data patterns due to its interpretability achieved through model symbolization. Besides, iTFKAN develops two strategies, prior knowledge injection, and time-frequency synergy learning, to effectively guide model learning under complex intertwined time series data. Extensive experimental results demonstrated that iTFKAN can achieve promising forecasting performance while simultaneously possessing high interpretive capabilities. |
| 2025-04-23 | [Anytime Safe Reinforcement Learning](http://arxiv.org/abs/2504.16417v1) | Pol Mestres, Arnau Marzabal et al. | This paper considers the problem of solving constrained   reinforcement learning problems with anytime guarantees, meaning   that the algorithmic solution returns a safe policy regardless of   when it is terminated. Drawing inspiration from anytime constrained   optimization, we introduce Reinforcement Learning-based Safe   Gradient Flow (RL-SGF), an on-policy algorithm which employs   estimates of the value functions and their respective gradients   associated with the objective and safety constraints for the current   policy, and updates the policy parameters by solving a convex   quadratically constrained quadratic program. We show that if the   estimates are computed with a sufficiently large number of episodes   (for which we provide an explicit bound), safe policies are updated   to safe policies with a probability higher than a prescribed   tolerance. We also show that iterates asymptotically converge to a   neighborhood of a KKT point, whose size can be arbitrarily reduced   by refining the estimates of the value function and their gradients.   We illustrate the performance of RL-SGF in a navigation example. |
| 2025-04-23 | [SILM: A Subjective Intent Based Low-Latency Framework for Multiple Traffic Participants Joint Trajectory Prediction](http://arxiv.org/abs/2504.16377v1) | Qu Weiming, Wang Jia et al. | Trajectory prediction is a fundamental technology for advanced autonomous driving systems and represents one of the most challenging problems in the field of cognitive intelligence. Accurately predicting the future trajectories of each traffic participant is a prerequisite for building high safety and high reliability decision-making, planning, and control capabilities in autonomous driving. However, existing methods often focus solely on the motion of other traffic participants without considering the underlying intent behind that motion, which increases the uncertainty in trajectory prediction. Autonomous vehicles operate in real-time environments, meaning that trajectory prediction algorithms must be able to process data and generate predictions in real-time. While many existing methods achieve high accuracy, they often struggle to effectively handle heterogeneous traffic scenarios. In this paper, we propose a Subjective Intent-based Low-latency framework for Multiple traffic participants joint trajectory prediction. Our method explicitly incorporates the subjective intent of traffic participants based on their key points, and predicts the future trajectories jointly without map, which ensures promising performance while significantly reducing the prediction latency. Additionally, we introduce a novel dataset designed specifically for trajectory prediction. Related code and dataset will be available soon. |
| 2025-04-23 | [The Safety-Privacy Tradeoff in Linear Bandits](http://arxiv.org/abs/2504.16371v1) | Arghavan Zibaie, Spencer Hutchinson et al. | We consider a collection of linear stochastic bandit problems, each modeling the random response of different agents to proposed interventions, coupled together by a global safety constraint. We assume a central coordinator must choose actions to play on each bandit with the objective of regret minimization, while also ensuring that the expected response of all agents satisfies the global safety constraints at each round, in spite of uncertainty about the bandits' parameters. The agents consider their observed responses to be private and in order to protect their sensitive information, the data sharing with the central coordinator is performed under local differential privacy (LDP). However, providing higher level of privacy to different agents would have consequences in terms of safety and regret. We formalize these tradeoffs by building on the notion of the sharpness of the safety set - a measure of how the geometric properties of the safe set affects the growth of regret - and propose a unilaterally unimprovable vector of privacy levels for different agents given a maximum regret budget. |

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



