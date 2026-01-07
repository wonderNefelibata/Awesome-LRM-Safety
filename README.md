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
| 2026-01-06 | [The Sonar Moment: Benchmarking Audio-Language Models in Audio Geo-Localization](http://arxiv.org/abs/2601.03227v1) | Ruixing Zhang, Zihan Liu et al. | Geo-localization aims to infer the geographic origin of a given signal. In computer vision, geo-localization has served as a demanding benchmark for compositional reasoning and is relevant to public safety. In contrast, progress on audio geo-localization has been constrained by the lack of high-quality audio-location pairs. To address this gap, we introduce AGL1K, the first audio geo-localization benchmark for audio language models (ALMs), spanning 72 countries and territories. To extract reliably localizable samples from a crowd-sourced platform, we propose the Audio Localizability metric that quantifies the informativeness of each recording, yielding 1,444 curated audio clips. Evaluations on 16 ALMs show that ALMs have emerged with audio geo-localization capability. We find that closed-source models substantially outperform open-source models, and that linguistic clues often dominate as a scaffold for prediction. We further analyze ALMs' reasoning traces, regional bias, error causes, and the interpretability of the localizability metric. Overall, AGL1K establishes a benchmark for audio geo-localization and may advance ALMs with better geospatial reasoning capability. |
| 2026-01-06 | [Wait or cross? Understanding the influence of behavioral tendency, trust, and risk perception on pedestrian gap-acceptance of automated truck platoons](http://arxiv.org/abs/2601.03225v1) | Yun Ye, Yuan Che et al. | Although automated trucks have the potential to improve freight efficiency, reduce costs, and address driver shortages, organizing two or more trucks in a convoy has raised considerable concerns for pedestrian safety. This study conducted a controlled experiment to examine the influence of behavioral tendency, trust, and risk perception on pedestrian intention to cross in front of an automated truck platoon. A total of 603 subjects participated in the virtual reality video-based questionnaire survey. By fusing the merits of structural equation modeling and artificial neural networks, a two-stage, hybrid model was developed to examine complex relationships between latent variables and gap-acceptance behaviors. Our results indicated that subjects watched an average of five vehicle gaps before starting crossing and the average time gap accepted was about 5.35 seconds. Risk perception not only played the most dominant role in shaping pedestrian crossing decisions, but also served as the strong bone, mediating the effects of behavioral tendency and trust on gap-acceptance. Participants who frequently violated traffic rules were more likely to accept a smaller time gap, while those who showed positive behaviors to other road users tended to wait for a larger time gap. Participants who often committed errors, showed aggressive behaviors, and held greater trust in the safety of automated trucks generally reported a lower level of risk for road-crossing in front of automated truck platoons. Built on these findings, a range of tailored countermeasures were proposed to ensure safer and smother interactions between pedestrians and automated truck platoons. |
| 2026-01-06 | [Enhancing Safety in Automated Ports: A Virtual Reality Study of Pedestrian-Autonomous Vehicle Interactions under Time Pressure, Visual Constraints, and Varying Vehicle Size](http://arxiv.org/abs/2601.03218v1) | Yuan Che, Mun On Wong et al. | Autonomous driving improves traffic efficiency but presents safety challenges in complex port environments. This study investigates how environmental factors, traffic factors, and pedestrian characteristics influence interaction safety between autonomous vehicles and pedestrians in ports. Using virtual reality (VR) simulations of typical port scenarios, 33 participants completed pedestrian crossing tasks under varying visibility, vehicle sizes, and time pressure conditions. Results indicate that low-visibility conditions, partial occlusions and larger vehicle sizes significantly increase perceived risk, prompting pedestrians to wait longer and accept larger gaps. Specifically, pedestrians tended to accept larger gaps and waited longer when interacting with large autonomous truck platoons, reflecting heightened caution due to their perceived threat. However, local obstructions also reduce post-encroachment time, compressing safety margins. Individual attributes such as age, gender, and driving experience further shape decision-making, while time pressure undermines compensatory behaviors and increases risk. Based on these findings, safety strategies are proposed, including installing wide-angle cameras at multiple viewpoints, enabling real-time vehicle-infrastructure communication, enhancing port lighting and signage, and strengthening pedestrian safety training. This study offers practical recommendations for improving the safety and deployment of vision-based autonomous systems in port settings. |
| 2026-01-06 | [Predicting Time Pressure of Powered Two-Wheeler Riders for Proactive Safety Interventions](http://arxiv.org/abs/2601.03173v1) | Sumit S. Shevtekar, Chandresh K. Maurya et al. | Time pressure critically influences risky maneuvers and crash proneness among powered two-wheeler riders, yet its prediction remains underexplored in intelligent transportation systems. We present a large-scale dataset of 129,000+ labeled multivariate time-series sequences from 153 rides by 51 participants under No, Low, and High Time Pressure conditions. Each sequence captures 63 features spanning vehicle kinematics, control inputs, behavioral violations, and environmental context. Our empirical analysis shows High Time Pressure induces 48% higher speeds, 36.4% greater speed variability, 58% more risky turns at intersections, 36% more sudden braking, and 50% higher rear brake forces versus No Time Pressure. To benchmark this dataset, we propose MotoTimePressure, a deep learning model combining convolutional preprocessing, dual-stage temporal attention, and Squeeze-and-Excitation feature recalibration, achieving 91.53% accuracy and 98.93% ROC AUC, outperforming eight baselines. Since time pressure cannot be directly measured in real time, we demonstrate its utility in collision prediction and threshold determination. Using MTPS-predicted time pressure as features, improves Informer-based collision risk accuracy from 91.25% to 93.51%, approaching oracle performance (93.72%). Thresholded time pressure states capture rider cognitive stress and enable proactive ITS interventions, including adaptive alerts, haptic feedback, V2I signaling, and speed guidance, supporting safer two-wheeler mobility under the Safe System Approach. |
| 2026-01-06 | [The Anatomy of Conversational Scams: A Topic-Based Red Teaming Analysis of Multi-Turn Interactions in LLMs](http://arxiv.org/abs/2601.03134v1) | Xiangzhe Yuan, Zhenhao Zhang et al. | As LLMs gain persuasive agentic capabilities through extended dialogues, they introduce novel risks in multi-turn conversational scams that single-turn safety evaluations fail to capture. We systematically study these risks using a controlled LLM-to-LLM simulation framework across multi-turn scam scenarios. Evaluating eight state-of-the-art models in English and Chinese, we analyze dialogue outcomes and qualitatively annotate attacker strategies, defensive responses, and failure modes. Results reveal that scam interactions follow recurrent escalation patterns, while defenses employ verification and delay mechanisms. Furthermore, interactional failures frequently stem from safety guardrail activation and role instability. Our findings highlight multi-turn interactional safety as a critical, distinct dimension of LLM behavior. |
| 2026-01-06 | [A Probabilistic Digital Twin of UK En Route Airspace for Training and Evaluating AI Agents for Air Traffic Control](http://arxiv.org/abs/2601.03113v1) | Nick Pepper, Adam Keane et al. | This paper presents the first probabilistic Digital Twin of operational en route airspace, developed for the London Area Control Centre. The Digital Twin is intended to support the development and rigorous human-in-the-loop evaluation of AI agents for Air Traffic Control (ATC), providing a virtual representation of real-world airspace that enables safe exploration of higher levels of ATC automation.   This paper makes three significant contributions: firstly, we demonstrate how historical and live operational data may be combined with a probabilistic, physics-informed machine learning model of aircraft performance to reproduce real-world traffic scenarios, while accurately reflecting the level of uncertainty inherent in ATC. Secondly, we develop a structured assurance case, following the Trustworthy and Ethical Assurance framework, to provide quantitative evidence for the Digital Twin's accuracy and fidelity. This is crucial to building trust in this novel technology within this safety-critical domain. Thirdly, we describe how the Digital Twin forms a unified environment for agent testing and evaluation. This includes fast-time execution (up to x200 real-time), a standardised Python-based ``gym'' interface that supports a range of AI agent designs, and a suite of quantitative metrics for assessing performance. Crucially, the framework facilitates competency-based assessment of AI agents by qualified Air Traffic Control Officers through a Human Machine Interface. We also outline further applications and future extensions of the Digital Twin architecture. |
| 2026-01-06 | [Dual-quaternion learning control for autonomous vehicle trajectory tracking with safety guarantees](http://arxiv.org/abs/2601.03097v1) | Omayra Yago Nieto, Alexandre Anahory Simoes et al. | We propose a learning-based trajectory tracking controller for autonomous robotic platforms whose motion can be described kinematically on $\mathrm{SE}(3)$. The controller is formulated in the dual quaternion framework and operates at the velocity level, assuming direct command of angular and linear velocities, as is standard in many aerial vehicles and omnidirectional mobile robots. Gaussian Process (GP) regression is integrated into a geometric feedback law to learn and compensate online for unknown, state-dependent disturbances and modeling imperfections affecting both attitude and position, while preserving the algebraic structure and coupling properties inherent to rigid-body motion.   The proposed approach does not rely on explicit parametric models of the unknown effects, making it well-suited for robotic systems subject to sensor-induced disturbances, unmodeled actuation couplings, and environmental uncertainties. A Lyapunov-based analysis establishes probabilistic ultimate boundedness of the pose tracking error under bounded GP uncertainty, providing formal stability guarantees for the learning-based controller.   Simulation results demonstrate accurate and smooth trajectory tracking in the presence of realistic, localized disturbances, including correlated rotational and translational effects arising from magnetometer perturbations. These results illustrate the potential of combining geometric modeling and probabilistic learning to achieve robust, data-efficient pose control for autonomous robotic systems. |
| 2026-01-06 | [Fast Surrogate Models for Adaptive Aircraft Trajectory Prediction in En route Airspace](http://arxiv.org/abs/2601.03075v1) | Nick Pepper, Marc Thomas et al. | Trajectory prediction (TP) is crucial for ensuring safety and efficiency in modern air traffic management systems. It is, for example, a core component of conflict detection and resolution tools, arrival sequencing algorithms, capacity planning, as well as several future concepts. However, TP accuracy within operational systems is hampered by a range of epistemic uncertainties such as the mass and performance settings of aircraft and the effect of meteorological conditions on aircraft performance. It can also require considerable computational resources.   This paper proposes a method for adaptive TP that has two components: first, a fast surrogate TP model based on linear state space models (LSSM)s with an execution time that was 6.7 times lower on average than an implementation of the Base of Aircraft Data (BADA) in Python. It is demonstrated that such models can effectively emulate the BADA aircraft performance model, which is based on the numerical solution of a partial differential equation (PDE), and that the LSSMs can be fitted to trajectories in a dataset of historic flight data. Secondly, the paper proposes an algorithm to assimilate radar observations using particle filtering to adaptively refine TP accuracy. Comparison with baselines using BADA and Kalman filtering demonstrate that the proposed framework improves system identification and state estimation for both climb and descent phases, with 46.3% and 64.7% better estimates for time to top of climb and bottom of descent compared to the best performing benchmark model. In particular, the particle filtering approach provides the flexibility to capture non-linear performance effects including the CAS-Mach transition. |
| 2026-01-06 | [When the Coffee Feature Activates on Coffins: An Analysis of Feature Extraction and Steering for Mechanistic Interpretability](http://arxiv.org/abs/2601.03047v1) | Raphael Ronge, Markus Maier et al. | Recent work by Anthropic on Mechanistic interpretability claims to understand and control Large Language Models by extracting human-interpretable features from their neural activation patterns using sparse autoencoders (SAEs). If successful, this approach offers one of the most promising routes for human oversight in AI safety. We conduct an initial stress-test of these claims by replicating their main results with open-source SAEs for Llama 3.1. While we successfully reproduce basic feature extraction and steering capabilities, our investigation suggests that major caution is warranted regarding the generalizability of these claims. We find that feature steering exhibits substantial fragility, with sensitivity to layer selection, steering magnitude, and context. We observe non-standard activation behavior and demonstrate the difficulty to distinguish thematically similar features from one another. While SAE-based interpretability produces compelling demonstrations in selected cases, current methods often fall short of the systematic reliability required for safety-critical applications. This suggests a necessary shift in focus from prioritizing interpretability of internal representations toward reliable prediction and control of model output. Our work contributes to a more nuanced understanding of what mechanistic interpretability has achieved and highlights fundamental challenges for AI safety that remain unresolved. |
| 2026-01-06 | [From inconsistency to decision: explainable operation and maintenance of battery energy storage systems](http://arxiv.org/abs/2601.03007v1) | Jingbo Qu, Yijie Wang et al. | Battery Energy Storage Systems (BESSs) are increasingly critical to power-system stability, yet their operation and maintenance remain dominated by reactive, expert-dependent diagnostics. While cell-level inconsistencies provide early warning signals of degradation and safety risks, the lack of scalable and interpretable decision-support frameworks prevents these signals from being effectively translated into operational actions. Here we introduce an inconsistency-driven operation and maintenance paradigm for large-scale BESSs that systematically transforms routine monitoring data into explainable, decision-oriented guidance. The proposed framework integrates multi-dimensional inconsistency evaluation with large language model-based semantic reasoning to bridge the gap between quantitative diagnostics and practical maintenance decisions. Using eight months of field data from an in-service battery system comprising 3,564 cells, we demonstrate how electrical, thermal, and aging-related inconsistencies can be distilled into structured operational records and converted into actionable maintenance insights through a multi-agent framework. The proposed approach enables accurate and explainable responses to real-world operation and maintenance queries, reducing response time and operational cost by over 80% compared with conventional expert-driven practices. These results establish a scalable pathway for intelligent operation and maintenance of battery energy storage systems, with direct implications for reliability, safety, and cost-effective integration of energy storage into modern power systems. |
| 2026-01-06 | [JPU: Bridging Jailbreak Defense and Unlearning via On-Policy Path Rectification](http://arxiv.org/abs/2601.03005v1) | Xi Wang, Songlei Jian et al. | Despite extensive safety alignment, Large Language Models (LLMs) often fail against jailbreak attacks. While machine unlearning has emerged as a promising defense by erasing specific harmful parameters, current methods remain vulnerable to diverse jailbreaks. We first conduct an empirical study and discover that this failure mechanism is caused by jailbreaks primarily activating non-erased parameters in the intermediate layers. Further, by probing the underlying mechanism through which these circumvented parameters reassemble into the prohibited output, we verify the persistent existence of dynamic $\textbf{jailbreak paths}$ and show that the inability to rectify them constitutes the fundamental gap in existing unlearning defenses. To bridge this gap, we propose $\textbf{J}$ailbreak $\textbf{P}$ath $\textbf{U}$nlearning (JPU), which is the first to rectify dynamic jailbreak paths towards safety anchors by dynamically mining on-policy adversarial samples to expose vulnerabilities and identify jailbreak paths. Extensive experiments demonstrate that JPU significantly enhances jailbreak resistance against dynamic attacks while preserving the model's utility. |
| 2026-01-06 | [Developing and Evaluating Lightweight Cryptographic Algorithms for Secure Embedded Systems in IoT Devices](http://arxiv.org/abs/2601.02981v1) | Brahim Khalil Sedraoui, Abdelmadjid Benmachiche et al. | The high rate of development of Internet of Things (IoT) devices has brought to attention new challenges in the area of data security, especially within the resource-limited realm of RFID tags, sensors, and embedded systems. Traditional cryptographic implementations can be of inappropriate computational complexity and energy usage and hence are not suitable on these platforms. This paper examines the design, implementation, and testing of lightweight cryptographic algorithms that have been specifically designed to be used in secure embedded systems. A comparison of some of the state-of-the-art lightweight encryption algorithms, that is PRESENT, SPECK, and SIMON, focuses on the main performance indicators, i.e., throughput, use of memory, and energy utilization. The study presents novel lightweight algorithms that are founded upon the Feistel-network architecture and their safety under cryptanalytic attacks, e.g., differential and linear cryptanalysis. The proposed solutions are proven through hardware implementation on the FPGA platform. The results have shown that lightweight cryptography is an effective strategy that could be used to establish security and maintain performance in the IoT and other resource-limited settings. |
| 2026-01-06 | [Parameter-Robust MPPI for Safe Online Learning of Unknown Parameters](http://arxiv.org/abs/2601.02948v1) | Matti Vahs, Jaeyoun Choi et al. | Robots deployed in dynamic environments must remain safe even when key physical parameters are uncertain or change over time. We propose Parameter-Robust Model Predictive Path Integral (PRMPPI) control, a framework that integrates online parameter learning with probabilistic safety constraints. PRMPPI maintains a particle-based belief over parameters via Stein Variational Gradient Descent, evaluates safety constraints using Conformal Prediction, and optimizes both a nominal performance-driven and a safety-focused backup trajectory in parallel. This yields a controller that is cautious at first, improves performance as parameters are learned, and ensures safety throughout. Simulation and hardware experiments demonstrate higher success rates, lower tracking error, and more accurate parameter estimates than baselines. |
| 2026-01-06 | [Beyond the Black Box: Theory and Mechanism of Large Language Models](http://arxiv.org/abs/2601.02907v1) | Zeyu Gan, Ruifeng Ren et al. | The rapid emergence of Large Language Models (LLMs) has precipitated a profound paradigm shift in Artificial Intelligence, delivering monumental engineering successes that increasingly impact modern society. However, a critical paradox persists within the current field: despite the empirical efficacy, our theoretical understanding of LLMs remains disproportionately nascent, forcing these systems to be treated largely as ``black boxes''. To address this theoretical fragmentation, this survey proposes a unified lifecycle-based taxonomy that organizes the research landscape into six distinct stages: Data Preparation, Model Preparation, Training, Alignment, Inference, and Evaluation. Within this framework, we provide a systematic review of the foundational theories and internal mechanisms driving LLM performance. Specifically, we analyze core theoretical issues such as the mathematical justification for data mixtures, the representational limits of various architectures, and the optimization dynamics of alignment algorithms. Moving beyond current best practices, we identify critical frontier challenges, including the theoretical limits of synthetic data self-improvement, the mathematical bounds of safety guarantees, and the mechanistic origins of emergent intelligence. By connecting empirical observations with rigorous scientific inquiry, this work provides a structured roadmap for transitioning LLM development from engineering heuristics toward a principled scientific discipline. |
| 2026-01-06 | [Bridging Mechanistic Interpretability and Prompt Engineering with Gradient Ascent for Interpretable Persona Control](http://arxiv.org/abs/2601.02896v1) | Harshvardhan Saini, Yiming Tang et al. | Controlling emergent behavioral personas (e.g., sycophancy, hallucination) in Large Language Models (LLMs) is critical for AI safety, yet remains a persistent challenge. Existing solutions face a dilemma: manual prompt engineering is intuitive but unscalable and imprecise, while automatic optimization methods are effective but operate as "black boxes" with no interpretable connection to model internals. We propose a novel framework that adapts gradient ascent to LLMs, enabling targeted prompt discovery. In specific, we propose two methods, RESGA and SAEGA, that both optimize randomly initialized prompts to achieve better aligned representation with an identified persona direction. We introduce fluent gradient ascent to control the fluency of discovered persona steering prompts. We demonstrate RESGA and SAEGA's effectiveness across Llama 3.1, Qwen 2.5, and Gemma 3 for steering three different personas,sycophancy, hallucination, and myopic reward. Crucially, on sycophancy, our automatically discovered prompts achieve significant improvement (49.90% compared with 79.24%). By grounding prompt discovery in mechanistically meaningful features, our method offers a new paradigm for controllable and interpretable behavior modification. |
| 2026-01-06 | [Soft Responsive Materials Enhance Humanoid Safety](http://arxiv.org/abs/2601.02857v1) | Chunzheng Wang, Yiyuan Zhang et al. | Humanoid robots are envisioned as general-purpose platforms in human-centered environments, yet their deployment is limited by vulnerability to falls and the risks posed by rigid metal-plastic structures to people and surroundings. We introduce a soft-rigid co-design framework that leverages non-Newtonian fluid-based soft responsive materials to enhance humanoid safety. The material remains compliant during normal interaction but rapidly stiffens under impact, absorbing and dissipating fall-induced forces. Physics-based simulations guide protector placement and thickness and enable learning of active fall policies. Applied to a 42 kg life-size humanoid, the protector markedly reduces peak impact and allows repeated falls without hardware damage, including drops from 3 m and tumbles down long staircases. Across diverse scenarios, the approach improves robot robustness and environmental safety. By uniting responsive materials, structural co-design, and learning-based control, this work advances interact-safe, industry-ready humanoid robots. |
| 2026-01-06 | [ML enhanced measurement of the electrostatic charge distribution of powder conveyed through a duct](http://arxiv.org/abs/2601.02852v1) | Christoph Wilms, Wenchao Xu et al. | The electrostatic charge acquired by powders during transport through ducts can cause devastating dust explosions. Our recently developed laser-optical measurement technique can resolve the powder charge along a one-dimensional (1D) path. However, the charge across the duct's complete two-dimensional (2D) cross-section, which is the critical parameter for process safety, is generally unavailable due to limited optical access. To estimate the complete powder charge distribution in a conveying duct, we propose a machine learning (ML) approach using a shallow neural network (SNN). The ML algorithm is trained with cross-sectional data extracted from four different three-dimensional direct numerical simulations of a turbulent duct flow with varying particle size. Through this training with simulation data, the ML algorithm can estimate the powder charge distribution in the duct's cross-section based on only 1D measurements. The results reveal an average $L^1$-error of the reconstructed 2D cross-section of 1.63 %. |
| 2026-01-06 | [Adapting Polyhedral Dominance Cones to Ordinal Preference Structures](http://arxiv.org/abs/2601.02796v1) | Kathrin Klamroth, Michael Stiglmayr et al. | In combinatorial optimization, ordinal costs can be used to model the quality of elements whenever numerical values are not available. When considering, for example, routing problems for cyclists, the safety of a street can be ranked in ordered categories like safe (separate bike lane), medium safe (street with a bike lane) and unsafe (street without a bike lane). However, ordinal optimization may suggest unrealistic solutions with huge detours to avoid unsafe street segments. In this paper, we investigate how partial preference information regarding the relative quality of the ordinal categories can be used to improve the relevance of the computed solutions. By introducing preference weights which describe how much better a category is at least or at most, compared to the subsequent category, we enlarge the ordinal dominance cone. This leads to a smaller set of alternatives, i. e., of ordinally efficient solutions. We show that the corresponding weighted ordinal ordering cone is a polyhedral cone and provide descriptions via its extreme rays and via its facets. The latter implies a linear transformation to an associated multi-objective optimization problem. This paves the way for the application of standard multi-objective solution approaches. We demonstrate the usefulness of the weighted ordinal ordering cone by investigating a safest path problem with different preference weights. Moreover, we investigate the interrelation between the weighted ordering cone to standard dominance concepts of multi-objective optimization, like, e.g., Pareto dominance, lexicographic dominance and weighted sum dominance. |
| 2026-01-06 | [Hierarchical Preemptive Holistic Collaborative Systems for Embodied Multi-Agent Systems: Framework, Hybrid Stability, and Scalability Analysis](http://arxiv.org/abs/2601.02779v1) | Ting Peng | The coordination of Embodied Multi-Agent Systems in constrained physical environments requires a rigorous balance between safety, scalability, and efficiency. Traditional decentralized approaches, e.g., reactive collision avoidance, are prone to local minima or reciprocal yielding standoffs due to the lack of future intent awareness. In contrast, centralized planning suffers from intractable computational complexity and single-point-of-failure vulnerabilities. To address these limitations, we propose the Hierarchical Preemptive Holistic Collaborative (Prollect) framework, which generalizes the Preemptive Holistic Collaborative System (PHCS) by decomposing the global coordination problem into topologically connected subspace optimizations. We formalize the system as a Hybrid Automaton and introduce a three-stage receding horizon mechanism (frozen execution, preliminary planning, proactive look-ahead windows) with explicit padding to prevent races between coordination dissemination and intent updates. Notably, we design a robust timing protocol with a mandatory Idle Buffer that acts as a dwell-time constraint to eliminate Zeno behaviors and ensure computational stability under jitter. Furthermore, we formalize a Shadow Agent protocol to guarantee seamless trajectory consistency across subspace boundaries, which we treat as an Input-to-State Stability (ISS) problem. |
| 2026-01-06 | [Advancing Assistive Robotics: Multi-Modal Navigation and Biophysical Monitoring for Next-Generation Wheelchairs](http://arxiv.org/abs/2601.02766v1) | Md. Anowar Hossain, Mohd. Ehsanul Hoque | Assistive electric-powered wheelchairs (EPWs) have become essential mobility aids for people with disabilities such as amyotrophic lateral sclerosis (ALS), post-stroke hemiplegia, and dementia-related mobility impairment. This work presents a novel multi-modal EPW control system designed to prioritize patient needs while allowing seamless switching between control modes. Four complementary interfaces, namely joystick, speech, hand gesture, and electrooculography (EOG), are integrated with a continuous vital sign monitoring framework measuring heart rate variability, oxygen saturation (SpO2), and skin temperature. This combination enables greater patient independence while allowing caregivers to maintain real-time supervision and early intervention capability.   Two-point calibration of the biophysical sensors against clinical reference devices resulted in root mean square errors of at most 2 bpm for heart rate, 0.5 degree Celsius for skin temperature, and 1 percent for SpO2. Experimental evaluation involved twenty participants with mobility impairments executing a total of 500 indoor navigation commands. The achieved command recognition accuracies were 99 percent for joystick control, 97 percent plus or minus 2 percent for speech, and 95 percent plus or minus 3 percent for hand gesture, with an average closed-loop latency of 20 plus or minus 0.5 milliseconds. Caregivers receive real-time alerts through an Android application following encrypted cloud transmission of physiological data. By integrating multi-modal mobility control with cloud-enabled health monitoring and reporting latency and energy budgets, the proposed prototype addresses key challenges in assistive robotics, contributes toward compliance with ISO 7176-31 and IEC 80601-2-78 safety standards, and establishes a foundation for future adaptive machine learning enhancements. |

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



