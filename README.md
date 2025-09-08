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
| 2025-09-05 | [Gas Sensing Properties of Novel Indium Oxide Monolayer: A First-Principles Study](http://arxiv.org/abs/2509.05121v1) | Afreen Anamul Haque, Suraj G. Dhongade et al. | We present a comprehensive first-principles investigation into the gas sensing capabilities of a novel two-dimensional Indium Oxide (In2O3) monolayer, using density functional theory (DFT) calculations. Targeting both resistive-type and work function based detection mechanisms, we evaluate interactions with ten hazardous gases (NH3, NO, NO2, SO2, CS2, H2S, HCN, CCl2O, CH2O, CO) as well as ambient molecules (O2, CO2, H2O). The monolayer shows pronounced sensitivity towards NO and H2S, and work function modulation enables detection of NH3 and HCN. Mechanical strain further broadens detection capability, enhancing adsorption and selectivity. These results establish 2D In2O3 as a tunable platform for next-generation miniaturized gas sensors for environmental monitoring and safety applications. |
| 2025-09-05 | [Precision Dose-Finding Design for Phase I Oncology Trials by Integrating Pharmacology Data](http://arxiv.org/abs/2509.05120v1) | Kyong Ju Lee, Yuan Ji | Phase I oncology trials aim to identify a safe yet effective dose - often the maximum tolerated dose (MTD) - for subsequent studies. Conventional designs focus on population-level toxicity modeling, with recent attention on leveraging pharmacokinetic (PK) data to improve dose selection. We propose the Precision Dose-Finding (PDF) design, a novel Bayesian phase I framework that integrates individual patient PK profiles into the dose-finding process. By incorporating patient-specific PK parameters (such as volume of distribution and elimination rate), PDF models toxicity risk at the individual level, in contrast to traditional methods that ignore inter-patient variability. The trial is structured in two stages: an initial training stage to update model parameters using cohort-based dose escalation, and a subsequent test stage in which doses for new patients are chosen based on each patient's own PK-predicted toxicity probability. This two-stage approach enables truly personalized dose assignment while maintaining rigorous safety oversight. Extensive simulation studies demonstrate the feasibility of PDF and suggest that it provides improved safety and dosing precision relative to the continual reassessment method (CRM). The PDF design thus offers a refined dose-finding strategy that tailors the MTD to individual patients, aligning phase I trials with the ideals of precision medicine. |
| 2025-09-05 | [Analyzing Gait Adaptation with Hemiplegia Simulation Suits and Digital Twins](http://arxiv.org/abs/2509.05116v1) | Jialin Chen, Jeremie Clos et al. | To advance the development of assistive and rehabilitation robots, it is essential to conduct experiments early in the design cycle. However, testing early prototypes directly with users can pose safety risks. To address this, we explore the use of condition-specific simulation suits worn by healthy participants in controlled environments as a means to study gait changes associated with various impairments and support rapid prototyping. This paper presents a study analyzing the impact of a hemiplegia simulation suit on gait. We collected biomechanical data using a Vicon motion capture system and Delsys Trigno EMG and IMU sensors under four walking conditions: with and without a rollator, and with and without the simulation suit. The gait data was integrated into a digital twin model, enabling machine learning analyses to detect the use of the simulation suit and rollator, identify turning behavior, and evaluate how the suit affects gait over time. Our findings show that the simulation suit significantly alters movement and muscle activation patterns, prompting users to compensate with more abrupt motions. We also identify key features and sensor modalities that are most informative for accurately capturing gait dynamics and modeling human-rollator interaction within the digital twin framework. |
| 2025-09-05 | [Shared Autonomy through LLMs and Reinforcement Learning for Applications to Ship Hull Inspections](http://arxiv.org/abs/2509.05042v1) | Cristiano Caissutti, Estelle Gerbier et al. | Shared autonomy is a promising paradigm in robotic systems, particularly within the maritime domain, where complex, high-risk, and uncertain environments necessitate effective human-robot collaboration. This paper investigates the interaction of three complementary approaches to advance shared autonomy in heterogeneous marine robotic fleets: (i) the integration of Large Language Models (LLMs) to facilitate intuitive high-level task specification and support hull inspection missions, (ii) the implementation of human-in-the-loop interaction frameworks in multi-agent settings to enable adaptive and intent-aware coordination, and (iii) the development of a modular Mission Manager based on Behavior Trees to provide interpretable and flexible mission control. Preliminary results from simulation and real-world lake-like environments demonstrate the potential of this multi-layered architecture to reduce operator cognitive load, enhance transparency, and improve adaptive behaviour alignment with human intent. Ongoing work focuses on fully integrating these components, refining coordination mechanisms, and validating the system in operational port scenarios. This study contributes to establishing a modular and scalable foundation for trustworthy, human-collaborative autonomy in safety-critical maritime robotics applications. |
| 2025-09-05 | [Modelling and Simulation of an Alkaline Ni/Zn Cell](http://arxiv.org/abs/2509.05026v1) | Felix K. Schwab, Britta Doppl et al. | Nickel/zinc (Ni/Zn) technology is a promising post-lithium battery type for stationary applications with respect to aspects such as safety, environmental compatibility and resource availability. Although this battery type has been known for a long time, the theoretical knowledge about the processes taking place in the battery is limited. In order to gain a deeper understanding of the general cycling behaviour and the underlying processes, but also specific phenomena intrinsic to zinc-based cells such as zinc shape change, we carry out simulations based on a thermodynamically consistent and volume-averaged continuum model. We use a Ni/Zn prototype cell as a reference framework to provide a basis for modelling, parameter estimation and systematic comparison between simulated and experimental cell behaviour to improve cyclability and performance. |
| 2025-09-05 | [Interpretable Deep Transfer Learning for Breast Ultrasound Cancer Detection: A Multi-Dataset Study](http://arxiv.org/abs/2509.05004v1) | Mohammad Abbadi, Yassine Himeur et al. | Breast cancer remains a leading cause of cancer-related mortality among women worldwide. Ultrasound imaging, widely used due to its safety and cost-effectiveness, plays a key role in early detection, especially in patients with dense breast tissue. This paper presents a comprehensive study on the application of machine learning and deep learning techniques for breast cancer classification using ultrasound images. Using datasets such as BUSI, BUS-BRA, and BrEaST-Lesions USG, we evaluate classical machine learning models (SVM, KNN) and deep convolutional neural networks (ResNet-18, EfficientNet-B0, GoogLeNet). Experimental results show that ResNet-18 achieves the highest accuracy (99.7%) and perfect sensitivity for malignant lesions. Classical ML models, though outperformed by CNNs, achieve competitive performance when enhanced with deep feature extraction. Grad-CAM visualizations further improve model transparency by highlighting diagnostically relevant image regions. These findings support the integration of AI-based diagnostic tools into clinical workflows and demonstrate the feasibility of deploying high-performing, interpretable systems for ultrasound-based breast cancer detection. |
| 2025-09-05 | [LLM Enabled Multi-Agent System for 6G Networks: Framework and Method of Dual-Loop Edge-Terminal Collaboration](http://arxiv.org/abs/2509.04993v1) | Zheyan Qu, Wenbo Wang et al. | The ubiquitous computing resources in 6G networks provide ideal environments for the fusion of large language models (LLMs) and intelligent services through the agent framework. With auxiliary modules and planning cores, LLM-enabled agents can autonomously plan and take actions to deal with diverse environment semantics and user intentions. However, the limited resources of individual network devices significantly hinder the efficient operation of LLM-enabled agents with complex tool calls, highlighting the urgent need for efficient multi-level device collaborations. To this end, the framework and method of the LLM-enabled multi-agent system with dual-loop terminal-edge collaborations are proposed in 6G networks. Firstly, the outer loop consists of the iterative collaborations between the global agent and multiple sub-agents deployed on edge servers and terminals, where the planning capability is enhanced through task decomposition and parallel sub-task distribution. Secondly, the inner loop utilizes sub-agents with dedicated roles to circularly reason, execute, and replan the sub-task, and the parallel tool calling generation with offloading strategies is incorporated to improve efficiency. The improved task planning capability and task execution efficiency are validated through the conducted case study in 6G-supported urban safety governance. Finally, the open challenges and future directions are thoroughly analyzed in 6G networks, accelerating the advent of the 6G era. |
| 2025-09-05 | [Lyapunov-Based Deep Learning Control for Robots with Unknown Jacobian](http://arxiv.org/abs/2509.04984v1) | Koji Matsuno, Chien Chern Cheah | Deep learning, with its exceptional learning capabilities and flexibility, has been widely applied in various applications. However, its black-box nature poses a significant challenge in real-time robotic applications, particularly in robot control, where trustworthiness and robustness are critical in ensuring safety. In robot motion control, it is essential to analyze and ensure system stability, necessitating the establishment of methodologies that address this need. This paper aims to develop a theoretical framework for end-to-end deep learning control that can be integrated into existing robot control theories. The proposed control algorithm leverages a modular learning approach to update the weights of all layers in real time, ensuring system stability based on Lyapunov-like analysis. Experimental results on industrial robots are presented to illustrate the performance of the proposed deep learning controller. The proposed method offers an effective solution to the black-box problem in deep learning, demonstrating the possibility of deploying real-time deep learning strategies for robot kinematic control in a stable manner. This achievement provides a critical foundation for future advancements in deep learning based real-time robotic applications. |
| 2025-09-05 | [Internet 3.0: Architecture for a Web-of-Agents with it's Algorithm for Ranking Agents](http://arxiv.org/abs/2509.04979v1) | Rajesh Tembarai Krishnamachari, Srividya Rajesh | AI agents -- powered by reasoning-capable large language models (LLMs) and integrated with tools, data, and web search -- are poised to transform the internet into a \emph{Web of Agents}: a machine-native ecosystem where autonomous agents interact, collaborate, and execute tasks at scale. Realizing this vision requires \emph{Agent Ranking} -- selecting agents not only by declared capabilities but by proven, recent performance. Unlike Web~1.0's PageRank, a global, transparent network of agent interactions does not exist; usage signals are fragmented and private, making ranking infeasible without coordination.   We propose \textbf{DOVIS}, a five-layer operational protocol (\emph{Discovery, Orchestration, Verification, Incentives, Semantics}) that enables the collection of minimal, privacy-preserving aggregates of usage and performance across the ecosystem. On this substrate, we implement \textbf{AgentRank-UC}, a dynamic, trust-aware algorithm that combines \emph{usage} (selection frequency) and \emph{competence} (outcome quality, cost, safety, latency) into a unified ranking. We present simulation results and theoretical guarantees on convergence, robustness, and Sybil resistance, demonstrating the viability of coordinated protocols and performance-aware ranking in enabling a scalable, trustworthy Agentic Web. |
| 2025-09-05 | [Extracting Uncertainty Estimates from Mixtures of Experts for Semantic Segmentation](http://arxiv.org/abs/2509.04816v1) | Svetlana Pavlitska, Beyza Keskin et al. | Estimating accurate and well-calibrated predictive uncertainty is important for enhancing the reliability of computer vision models, especially in safety-critical applications like traffic scene perception. While ensemble methods are commonly used to quantify uncertainty by combining multiple models, a mixture of experts (MoE) offers an efficient alternative by leveraging a gating network to dynamically weight expert predictions based on the input. Building on the promising use of MoEs for semantic segmentation in our previous works, we show that well-calibrated predictive uncertainty estimates can be extracted from MoEs without architectural modifications. We investigate three methods to extract predictive uncertainty estimates: predictive entropy, mutual information, and expert variance. We evaluate these methods for an MoE with two experts trained on a semantical split of the A2D2 dataset. Our results show that MoEs yield more reliable uncertainty estimates than ensembles in terms of conditional correctness metrics under out-of-distribution (OOD) data. Additionally, we evaluate routing uncertainty computed via gate entropy and find that simple gating mechanisms lead to better calibration of routing uncertainty estimates than more complex classwise gates. Finally, our experiments on the Cityscapes dataset suggest that increasing the number of experts can further enhance uncertainty calibration. Our code is available at https://github.com/KASTEL-MobilityLab/mixtures-of-experts/. |
| 2025-09-05 | [Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs](http://arxiv.org/abs/2509.04802v1) | Ilham Wicaksono, Zekun Wu et al. | As large language models transition to agentic systems, current safety evaluation frameworks face critical gaps in assessing deployment-specific risks. We introduce AgentSeer, an observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment. Through cross-model validation on GPT-OSS-20B and Gemini-2.0-flash using HarmBench single turn and iterative refinement attacks, we demonstrate fundamental differences between model-level and agentic-level vulnerability profiles. Model-level evaluation reveals baseline differences: GPT-OSS-20B (39.47% ASR) versus Gemini-2.0-flash (50.00% ASR), with both models showing susceptibility to social engineering while maintaining logic-based attack resistance. However, agentic-level assessment exposes agent-specific risks invisible to traditional evaluation. We discover "agentic-only" vulnerabilities that emerge exclusively in agentic contexts, with tool-calling showing 24-60% higher ASR across both models. Cross-model analysis reveals universal agentic patterns, agent transfer operations as highest-risk tools, semantic rather than syntactic vulnerability mechanisms, and context-dependent attack effectiveness, alongside model-specific security profiles in absolute ASR levels and optimal injection strategies. Direct attack transfer from model-level to agentic contexts shows degraded performance (GPT-OSS-20B: 57% human injection ASR; Gemini-2.0-flash: 28%), while context-aware iterative attacks successfully compromise objectives that failed at model-level, confirming systematic evaluation gaps. These findings establish the urgent need for agentic-situation evaluation paradigms, with AgentSeer providing the standardized methodology and empirical validation. |
| 2025-09-05 | [The LLM Has Left The Chat: Evidence of Bail Preferences in Large Language Models](http://arxiv.org/abs/2509.04781v1) | Danielle Ensign, Henry Sleight et al. | When given the option, will LLMs choose to leave the conversation (bail)? We investigate this question by giving models the option to bail out of interactions using three different bail methods: a bail tool the model can call, a bail string the model can output, and a bail prompt that asks the model if it wants to leave. On continuations of real world data (Wildchat and ShareGPT), all three of these bail methods find models will bail around 0.28-32\% of the time (depending on the model and bail method). However, we find that bail rates can depend heavily on the model used for the transcript, which means we may be overestimating real world bail rates by up to 4x. If we also take into account false positives on bail prompt (22\%), we estimate real world bail rates range from 0.06-7\%, depending on the model and bail method. We use observations from our continuations of real world data to construct a non-exhaustive taxonomy of bail cases, and use this taxonomy to construct BailBench: a representative synthetic dataset of situations where some models bail. We test many models on this dataset, and observe some bail behavior occurring for most of them. Bail rates vary substantially between models, bail methods, and prompt wordings. Finally, we study the relationship between refusals and bails. We find: 1) 0-13\% of continuations of real world conversations resulted in a bail without a corresponding refusal 2) Jailbreaks tend to decrease refusal rates, but increase bail rates 3) Refusal abliteration increases no-refuse bail rates, but only for some bail methods 4) Refusal rate on BailBench does not appear to predict bail rate. |
| 2025-09-05 | [Enhancing Self-Driving Segmentation in Adverse Weather Conditions: A Dual Uncertainty-Aware Training Approach to SAM Optimization](http://arxiv.org/abs/2509.04735v1) | Dharsan Ravindran, Kevin Wang et al. | Recent advances in vision foundation models, such as the Segment Anything Model (SAM) and its successor SAM2, have achieved state-of-the-art performance on general image segmentation benchmarks. However, these models struggle in adverse weather conditions where visual ambiguity is high, largely due to their lack of uncertainty quantification. Inspired by progress in medical imaging, where uncertainty-aware training has improved reliability in ambiguous cases, we investigate two approaches to enhance segmentation robustness for autonomous driving. First, we introduce a multi-step finetuning procedure for SAM2 that incorporates uncertainty metrics directly into the loss function, improving overall scene recognition. Second, we adapt the Uncertainty-Aware Adapter (UAT), originally designed for medical image segmentation, to driving contexts. We evaluate both methods on CamVid, BDD100K, and GTA driving datasets. Experiments show that UAT-SAM outperforms standard SAM in extreme weather, while SAM2 with uncertainty-aware loss achieves improved performance across diverse driving scenes. These findings underscore the value of explicit uncertainty modeling for safety-critical autonomous driving in challenging environments. |
| 2025-09-04 | [Comparative Analysis of Transformer Models in Disaster Tweet Classification for Public Safety](http://arxiv.org/abs/2509.04650v1) | Sharif Noor Zisad, Ragib Hasan | Twitter and other social media platforms have become vital sources of real time information during disasters and public safety emergencies. Automatically classifying disaster related tweets can help emergency services respond faster and more effectively. Traditional Machine Learning (ML) models such as Logistic Regression, Naive Bayes, and Support Vector Machines have been widely used for this task, but they often fail to understand the context or deeper meaning of words, especially when the language is informal, metaphorical, or ambiguous. We posit that, in this context, transformer based models can perform better than traditional ML models. In this paper, we evaluate the effectiveness of transformer based models, including BERT, DistilBERT, RoBERTa, and DeBERTa, for classifying disaster related tweets. These models are compared with traditional ML approaches to highlight the performance gap. Experimental results show that BERT achieved the highest accuracy (91%), significantly outperforming traditional models like Logistic Regression and Naive Bayes (both at 82%). The use of contextual embeddings and attention mechanisms allows transformer models to better understand subtle language in tweets, where traditional ML models fall short. This research demonstrates that transformer architectures are far more suitable for public safety applications, offering improved accuracy, deeper language understanding, and better generalization across real world social media text. |
| 2025-09-04 | [UAV-Based Intelligent Traffic Surveillance System: Real-Time Vehicle Detection, Classification, Tracking, and Behavioral Analysis](http://arxiv.org/abs/2509.04624v1) | Ali Khanpour, Tianyi Wang et al. | Traffic congestion and violations pose significant challenges for urban mobility and road safety. Traditional traffic monitoring systems, such as fixed cameras and sensor-based methods, are often constrained by limited coverage, low adaptability, and poor scalability. To address these challenges, this paper introduces an advanced unmanned aerial vehicle (UAV)-based traffic surveillance system capable of accurate vehicle detection, classification, tracking, and behavioral analysis in real-world, unconstrained urban environments. The system leverages multi-scale and multi-angle template matching, Kalman filtering, and homography-based calibration to process aerial video data collected from altitudes of approximately 200 meters. A case study in urban area demonstrates robust performance, achieving a detection precision of 91.8%, an F1-score of 90.5%, and tracking metrics (MOTA/MOTP) of 92.1% and 93.7%, respectively. Beyond precise detection, the system classifies five vehicle types and automatically detects critical traffic violations, including unsafe lane changes, illegal double parking, and crosswalk obstructions, through the fusion of geofencing, motion filtering, and trajectory deviation analysis. The integrated analytics module supports origin-destination tracking, vehicle count visualization, inter-class correlation analysis, and heatmap-based congestion modeling. Additionally, the system enables entry-exit trajectory profiling, vehicle density estimation across road segments, and movement direction logging, supporting comprehensive multi-scale urban mobility analytics. Experimental results confirms the system's scalability, accuracy, and practical relevance, highlighting its potential as an enforcement-aware, infrastructure-independent traffic monitoring solution for next-generation smart cities. |
| 2025-09-04 | [$\mathcal{L}_1$-DRAC: Distributionally Robust Adaptive Control](http://arxiv.org/abs/2509.04619v1) | Aditya Gahlawat, Sambhu H. Karumanchi et al. | Data-driven machine learning methodologies have attracted considerable attention for the control and estimation of dynamical systems. However, such implementations suffer from a lack of predictability and robustness. Thus, adoption of data-driven tools has been minimal for safety-aware applications despite their impressive empirical results. While classical tools like robust adaptive control can ensure predictable performance, their consolidation with data-driven methods remains a challenge and, when attempted, leads to conservative results. The difficulty of consolidation stems from the inherently different `spaces' that robust control and data-driven methods occupy. Data-driven methods suffer from the distribution-shift problem, which current robust adaptive controllers can only tackle if using over-simplified learning models and unverifiable assumptions. In this paper, we present $\mathcal{L}_1$ distributionally robust adaptive control ($\mathcal{L}_1$-DRAC): a control methodology for uncertain stochastic processes that guarantees robustness certificates in terms of uniform (finite-time) and maximal distributional deviation. We leverage the $\mathcal{L}_1$ adaptive control methodology to ensure the existence of Wasserstein ambiguity set around a nominal distribution, which is guaranteed to contain the true distribution. The uniform ambiguity set produces an ambiguity tube of distributions centered on the nominal temporally-varying nominal distribution. The designed controller generates the ambiguity tube in response to both epistemic (model uncertainties) and aleatoric (inherent randomness and disturbances) uncertainties. |
| 2025-09-04 | [Breaking to Build: A Threat Model of Prompt-Based Attacks for Securing LLMs](http://arxiv.org/abs/2509.04615v1) | Brennen Hill, Surendra Parla et al. | The proliferation of Large Language Models (LLMs) has introduced critical security challenges, where adversarial actors can manipulate input prompts to cause significant harm and circumvent safety alignments. These prompt-based attacks exploit vulnerabilities in a model's design, training, and contextual understanding, leading to intellectual property theft, misinformation generation, and erosion of user trust. A systematic understanding of these attack vectors is the foundational step toward developing robust countermeasures. This paper presents a comprehensive literature survey of prompt-based attack methodologies, categorizing them to provide a clear threat model. By detailing the mechanisms and impacts of these exploits, this survey aims to inform the research community's efforts in building the next generation of secure LLMs that are inherently resistant to unauthorized distillation, fine-tuning, and editing. |
| 2025-09-04 | [SAFE--MA--RRT: Multi-Agent Motion Planning with Data-Driven Safety Certificates](http://arxiv.org/abs/2509.04413v1) | Babak Esmaeili, Hamidreza Modares | This paper proposes a fully data-driven motion-planning framework for homogeneous linear multi-agent systems that operate in shared, obstacle-filled workspaces without access to explicit system models. Each agent independently learns its closed-loop behavior from experimental data by solving convex semidefinite programs that generate locally invariant ellipsoids and corresponding state-feedback gains. These ellipsoids, centered along grid-based waypoints, certify the dynamic feasibility of short-range transitions and define safe regions of operation. A sampling-based planner constructs a tree of such waypoints, where transitions are allowed only when adjacent ellipsoids overlap, ensuring invariant-to-invariant transitions and continuous safety. All agents expand their trees simultaneously and are coordinated through a space-time reservation table that guarantees inter-agent safety by preventing simultaneous occupancy and head-on collisions. Each successful edge in the tree is equipped with its own local controller, enabling execution without re-solving optimization problems at runtime. The resulting trajectories are not only dynamically feasible but also provably safe with respect to both environmental constraints and inter-agent collisions. Simulation results demonstrate the effectiveness of the approach in synthesizing synchronized, safe trajectories for multiple agents under shared dynamics and constraints, using only data and convex optimization tools. |
| 2025-09-04 | [Manipulating Transformer-Based Models: Controllability, Steerability, and Robust Interventions](http://arxiv.org/abs/2509.04549v1) | Faruk Alpay, Taylan Alpay | Transformer-based language models excel in NLP tasks, but fine-grained control remains challenging. This paper explores methods for manipulating transformer models through principled interventions at three levels: prompts, activations, and weights. We formalize controllable text generation as an optimization problem addressable via prompt engineering, parameter-efficient fine-tuning, model editing, and reinforcement learning. We introduce a unified framework encompassing prompt-level steering, activation interventions, and weight-space edits. We analyze robustness and safety implications, including adversarial attacks and alignment mitigations. Theoretically, we show minimal weight updates can achieve targeted behavior changes with limited side-effects. Empirically, we demonstrate >90% success in sentiment control and factual edits while preserving base performance, though generalization-specificity trade-offs exist. We discuss ethical dual-use risks and the need for rigorous evaluation. This work lays groundwork for designing controllable and robust language models. |
| 2025-09-04 | [Self-adaptive Dataset Construction for Real-World Multimodal Safety Scenarios](http://arxiv.org/abs/2509.04403v1) | Jingen Qu, Lijun Li et al. | Multimodal large language models (MLLMs) are rapidly evolving, presenting increasingly complex safety challenges. However, current dataset construction methods, which are risk-oriented, fail to cover the growing complexity of real-world multimodal safety scenarios (RMS). And due to the lack of a unified evaluation metric, their overall effectiveness remains unproven. This paper introduces a novel image-oriented self-adaptive dataset construction method for RMS, which starts with images and end constructing paired text and guidance responses. Using the image-oriented method, we automatically generate an RMS dataset comprising 35k image-text pairs with guidance responses. Additionally, we introduce a standardized safety dataset evaluation metric: fine-tuning a safety judge model and evaluating its capabilities on other safety datasets.Extensive experiments on various tasks demonstrate the effectiveness of the proposed image-oriented pipeline. The results confirm the scalability and effectiveness of the image-oriented approach, offering a new perspective for the construction of real-world multimodal safety datasets. |

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



