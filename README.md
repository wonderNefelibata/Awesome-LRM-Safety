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
| 2025-07-23 | [BetterCheck: Towards Safeguarding VLMs for Automotive Perception Systems](http://arxiv.org/abs/2507.17722v1) | Malsha Ashani Mahawatta Dona, Beatriz Cabrero-Daniel et al. | Large language models (LLMs) are growingly extended to process multimodal data such as text and video simultaneously. Their remarkable performance in understanding what is shown in images is surpassing specialized neural networks (NNs) such as Yolo that is supporting only a well-formed but very limited vocabulary, ie., objects that they are able to detect. When being non-restricted, LLMs and in particular state-of-the-art vision language models (VLMs) show impressive performance to describe even complex traffic situations. This is making them potentially suitable components for automotive perception systems to support the understanding of complex traffic situations or edge case situation. However, LLMs and VLMs are prone to hallucination, which mean to either potentially not seeing traffic agents such as vulnerable road users who are present in a situation, or to seeing traffic agents who are not there in reality. While the latter is unwanted making an ADAS or autonomous driving systems (ADS) to unnecessarily slow down, the former could lead to disastrous decisions from an ADS. In our work, we are systematically assessing the performance of 3 state-of-the-art VLMs on a diverse subset of traffic situations sampled from the Waymo Open Dataset to support safety guardrails for capturing such hallucinations in VLM-supported perception systems. We observe that both, proprietary and open VLMs exhibit remarkable image understanding capabilities even paying thorough attention to fine details sometimes difficult to spot for us humans. However, they are also still prone to making up elements in their descriptions to date requiring hallucination detection strategies such as BetterCheck that we propose in our work. |
| 2025-07-23 | [Piecewise Control Barrier Functions for Stochastic Systems](http://arxiv.org/abs/2507.17703v1) | Rayan Mazouz, Luca Laurenti et al. | This paper presents a method for the simultaneous synthesis of a barrier certificate and a safe controller for discrete-time nonlinear stochastic systems. Our approach, based on piecewise stochastic control barrier functions, reduces the synthesis problem to a minimax optimization, which we solve exactly using a dual linear program with zero gap. This enables the joint optimization of the barrier certificate and safe controller within a single formulation. The method accommodates stochastic dynamics with additive noise and a bounded continuous control set. The synthesized controllers and barrier certificates provide a formally guaranteed lower bound on probabilistic safety. Case studies on linear and nonlinear stochastic systems validate the effectiveness of our approach. |
| 2025-07-23 | [Safety Assurance for Quadrotor Kinodynamic Motion Planning](http://arxiv.org/abs/2507.17679v1) | Theodoros Tavoulareas, Marzia Cescon | Autonomous drones have gained considerable attention for applications in real-world scenarios, such as search and rescue, inspection, and delivery. As their use becomes ever more pervasive in civilian applications, failure to ensure safe operation can lead to physical damage to the system, environmental pollution, and even loss of human life. Recent work has demonstrated that motion planning techniques effectively generate a collision-free trajectory during navigation. However, these methods, while creating the motion plans, do not inherently consider the safe operational region of the system, leading to potential safety constraints violation during deployment. In this paper, we propose a method that leverages run time safety assurance in a kinodynamic motion planning scheme to satisfy the system's operational constraints. First, we use a sampling-based geometric planner to determine a high-level collision-free path within a user-defined space. Second, we design a low-level safety assurance filter to provide safety guarantees to the control input of a Linear Quadratic Regulator (LQR) designed with the purpose of trajectory tracking. We demonstrate our proposed approach in a restricted 3D simulation environment using a model of the Crazyflie 2.0 drone. |
| 2025-07-23 | [Insights into experimental evaluation of the non-fourier heat transfer model in biological tissues](http://arxiv.org/abs/2507.17627v1) | Mohammad Azhdari, Ghader Rezazadeh et al. | A comprehensive understanding of heat transfer mechanisms in biological tissues is essential for the advancement of thermal therapeutic techniques and the development of accurate bioheat transfer models. Conventional models often fail to capture the inherently complex thermal behavior of biological media, necessitating more sophisticated approaches for experimental validation and parameter extraction. In this study, the Two-Dimensional Three-Phase Lag (TPL) heat transfer model, implemented via the finite difference method (FDM), was employed to extract key phase lag parameters characterizing heat conduction in bovine skin tissue. Experimental measurements were obtained using a 450 nm laser source and two non-contact infrared sensors. The influence of four critical parameters was systematically investigated: heat flux phase lag ($\tau_{q}$), temperature gradient phase lag ($\tau_{\theta}$), thermal displacement coefficient ($k^*$), and thermal displacement phase lag ($\tau_{v}$). A carefully designed experimental protocol was used to assess each parameter independently. The results revealed that the extracted phase lag values were substantially lower than those previously reported in the literature. This highlights the importance of high-precision measurements and the need to isolate each parameter during analysis. These findings contribute to the refinement of bioheat transfer models and hold potential for improving the efficacy and safety of clinical thermal therapies. |
| 2025-07-23 | [Explainable AI for Collaborative Assessment of 2D/3D Registration Quality](http://arxiv.org/abs/2507.17597v1) | Sue Min Cho, Alexander Do et al. | As surgery embraces digital transformation--integrating sophisticated imaging, advanced algorithms, and robotics to support and automate complex sub-tasks--human judgment of system correctness remains a vital safeguard for patient safety. This shift introduces new "operator-type" roles tasked with verifying complex algorithmic outputs, particularly at critical junctures of the procedure, such as the intermediary check before drilling or implant placement. A prime example is 2D/3D registration, a key enabler of image-based surgical navigation that aligns intraoperative 2D images with preoperative 3D data. Although registration algorithms have advanced significantly, they occasionally yield inaccurate results. Because even small misalignments can lead to revision surgery or irreversible surgical errors, there is a critical need for robust quality assurance. Current visualization-based strategies alone have been found insufficient to enable humans to reliably detect 2D/3D registration misalignments. In response, we propose the first artificial intelligence (AI) framework trained specifically for 2D/3D registration quality verification, augmented by explainability features that clarify the model's decision-making. Our explainable AI (XAI) approach aims to enhance informed decision-making for human operators by providing a second opinion together with a rationale behind it. Through algorithm-centric and human-centered evaluations, we systematically compare four conditions: AI-only, human-only, human-AI, and human-XAI. Our findings reveal that while explainability features modestly improve user trust and willingness to override AI errors, they do not exceed the standalone AI in aggregate performance. Nevertheless, future work extending both the algorithmic design and the human-XAI collaboration elements holds promise for more robust quality assurance of 2D/3D registration. |
| 2025-07-23 | [Anticipate, Simulate, Reason (ASR): A Comprehensive Generative AI Framework for Combating Messaging Scams](http://arxiv.org/abs/2507.17543v1) | Xue Wen Tan, Kenneth See et al. | The rapid growth of messaging scams creates an escalating challenge for user security and financial safety. In this paper, we present the Anticipate, Simulate, Reason (ASR) framework, a generative AI method that enables users to proactively identify and comprehend scams within instant messaging platforms. Using large language models, ASR predicts scammer responses, creates realistic scam conversations, and delivers real-time, interpretable support to end-users. We develop ScamGPT-J, a domain-specific language model fine-tuned on a new, high-quality dataset of scam conversations covering multiple scam types. Thorough experimental evaluation shows that the ASR framework substantially enhances scam detection, particularly in challenging contexts such as job scams, and uncovers important demographic patterns in user vulnerability and perceptions of AI-generated assistance. Our findings reveal a contradiction where those most at risk are often least receptive to AI support, emphasizing the importance of user-centered design in AI-driven fraud prevention. This work advances both the practical and theoretical foundations for interpretable, human-centered AI systems in combating evolving digital threats. |
| 2025-07-23 | [An Uncertainty-Driven Adaptive Self-Alignment Framework for Large Language Models](http://arxiv.org/abs/2507.17477v1) | Haoran Sun, Zekun Zhang et al. | Large Language Models (LLMs) have demonstrated remarkable progress in instruction following and general-purpose reasoning. However, achieving high-quality alignment with human intent and safety norms without human annotations remains a fundamental challenge. In this work, we propose an Uncertainty-Driven Adaptive Self-Alignment (UDASA) framework designed to improve LLM alignment in a fully automated manner. UDASA first generates multiple responses for each input and quantifies output uncertainty across three dimensions: semantics, factuality, and value alignment. Based on these uncertainty scores, the framework constructs preference pairs and categorizes training samples into three stages, conservative, moderate, and exploratory, according to their uncertainty difference. The model is then optimized progressively across these stages. In addition, we conduct a series of preliminary studies to validate the core design assumptions and provide strong empirical motivation for the proposed framework. Experimental results show that UDASA outperforms existing alignment methods across multiple tasks, including harmlessness, helpfulness, truthfulness, and controlled sentiment generation, significantly improving model performance. |
| 2025-07-23 | [RoadBench: A Vision-Language Foundation Model and Benchmark for Road Damage Understanding](http://arxiv.org/abs/2507.17353v1) | Xi Xiao, Yunbei Zhang et al. | Accurate road damage detection is crucial for timely infrastructure maintenance and public safety, but existing vision-only datasets and models lack the rich contextual understanding that textual information can provide. To address this limitation, we introduce RoadBench, the first multimodal benchmark for comprehensive road damage understanding. This dataset pairs high resolution images of road damages with detailed textual descriptions, providing a richer context for model training. We also present RoadCLIP, a novel vision language model that builds upon CLIP by integrating domain specific enhancements. It includes a disease aware positional encoding that captures spatial patterns of road defects and a mechanism for injecting road-condition priors to refine the model's understanding of road damages. We further employ a GPT driven data generation pipeline to expand the image to text pairs in RoadBench, greatly increasing data diversity without exhaustive manual annotation. Experiments demonstrate that RoadCLIP achieves state of the art performance on road damage recognition tasks, significantly outperforming existing vision-only models by 19.2%. These results highlight the advantages of integrating visual and textual information for enhanced road condition analysis, setting new benchmarks for the field and paving the way for more effective infrastructure monitoring through multimodal learning. |
| 2025-07-23 | [DeMo++: Motion Decoupling for Autonomous Driving](http://arxiv.org/abs/2507.17342v1) | Bozhou Zhang, Nan Song et al. | Motion forecasting and planning are tasked with estimating the trajectories of traffic agents and the ego vehicle, respectively, to ensure the safety and efficiency of autonomous driving systems in dynamically changing environments. State-of-the-art methods typically adopt a one-query-one-trajectory paradigm, where each query corresponds to a unique trajectory for predicting multi-mode trajectories. While this paradigm can produce diverse motion intentions, it often falls short in modeling the intricate spatiotemporal evolution of trajectories, which can lead to collisions or suboptimal outcomes. To overcome this limitation, we propose DeMo++, a framework that decouples motion estimation into two distinct components: holistic motion intentions to capture the diverse potential directions of movement, and fine spatiotemporal states to track the agent's dynamic progress within the scene and enable a self-refinement capability. Further, we introduce a cross-scene trajectory interaction mechanism to explore the relationships between motions in adjacent scenes. This allows DeMo++ to comprehensively model both the diversity of motion intentions and the spatiotemporal evolution of each trajectory. To effectively implement this framework, we developed a hybrid model combining Attention and Mamba. This architecture leverages the strengths of both mechanisms for efficient scene information aggregation and precise trajectory state sequence modeling. Extensive experiments demonstrate that DeMo++ achieves state-of-the-art performance across various benchmarks, including motion forecasting (Argoverse 2 and nuScenes), motion planning (nuPlan), and end-to-end planning (NAVSIM). |
| 2025-07-23 | [Identification and Characterization of a New Disruption Regime in ADITYA-U Tokamak](http://arxiv.org/abs/2507.17299v1) | Soumitra Banerjee, Harshita Raj et al. | Disruptions continue to pose a significant challenge to the stable operation and future design of tokamak reactors. A comprehensive statistical investigation carried out on the ADITYA-U tokamak has led to the observation and characterization of a novel disruption regime. In contrast to the conventional Locked Mode Disruption (LMD), the newly identified disruption exhibits a distinctive two-phase evolution: an initial phase characterized by a steady rise in mode frequency with a nonlinearly saturated amplitude, followed by a sudden frequency collapse accompanied by a pronounced increase in amplitude. This behaviour signifies the onset of the precursor phase on a significantly shorter timescale. Clear empirical thresholds have been identified to distinguish this disruption type from conventional LMD events, including edge safety factor, current decay coefficient, current quench (CQ) time, and CQ rate. The newly identified disruption regime is predominantly governed by the (m/n = 2/1) drift-tearing mode (DTM), which, in contrast to typical disruptions in the ADITYA-U tokamak that involve both m/n = 2/1 and 3/1 modes, consistently manifests as the sole dominant instability. Initiated by core temperature hollowing, the growth of this mode is significantly enhanced by a synergistic interplay between a strongly localized pressure gradient and the pronounced steepening of the current density profile in the vicinity of the mode rational surface. |
| 2025-07-23 | [Optimizing Delivery Logistics: Enhancing Speed and Safety with Drone Technology](http://arxiv.org/abs/2507.17253v1) | Maharshi Shastri, Ujjval Shrivastav | The increasing demand for fast and cost effective last mile delivery solutions has catalyzed significant advancements in drone based logistics. This research describes the development of an AI integrated drone delivery system, focusing on route optimization, object detection, secure package handling, and real time tracking. The proposed system leverages YOLOv4 Tiny for object detection, the NEO 6M GPS module for navigation, and the A7670 SIM module for real time communication. A comparative analysis of lightweight AI models and hardware components is conducted to determine the optimal configuration for real time UAV based delivery. Key challenges including battery efficiency, regulatory compliance, and security considerations are addressed through the integration of machine learning techniques, IoT devices, and encryption protocols. Preliminary studies demonstrate improvement in delivery time compared to conventional ground based logistics, along with high accuracy recipient authentication through facial recognition. The study also discusses ethical implications and societal acceptance of drone deliveries, ensuring compliance with FAA, EASA and DGCA regulatory standards. Note: This paper presents the architecture, design, and preliminary simulation results of the proposed system. Experimental results, simulation benchmarks, and deployment statistics are currently being acquired. A comprehensive analysis will be included in the extended version of this work. |
| 2025-07-23 | [On the Construction of Barrier Certificate: A Dynamic Programming Perspective](http://arxiv.org/abs/2507.17222v1) | Yu Chen, Shaoyuan Li et al. | In this paper, we revisit the formal verification problem for stochastic dynamical systems over finite horizon using barrier certificates. Most existing work on this topic focuses on safety properties by constructing barrier certificates based on the notion of $c$-martingales. In this work, we first provide a new insight into the conditions of existing martingale-based barrier certificates from the perspective of dynamic programming operators. Specifically, we show that the existing conditions essentially provide a bound on the dynamic programming solution, which exactly characterizes the safety probability. Based on this new perspective, we demonstrate that the barrier conditions in existing approaches are unnecessarily conservative over unsafe states. To address this, we propose a new set of safety barrier certificate conditions that are strictly less conservative than existing ones, thereby providing tighter probability bounds for safety verification. We further extend our approach to the case of reach-avoid specifications by providing a set of new barrier certificate conditions. We also illustrate how to search for these new barrier certificates using sum-of-squares (SOS) programming. Finally, we use two numerical examples to demonstrate the advantages of our method compared to existing approaches. |
| 2025-07-23 | [Spintronic Bayesian Hardware Driven by Stochastic Magnetic Domain Wall Dynamics](http://arxiv.org/abs/2507.17193v1) | Tianyi Wang, Bingqian Dai et al. | As artificial intelligence (AI) advances into diverse applications, ensuring reliability of AI models is increasingly critical. Conventional neural networks offer strong predictive capabilities but produce deterministic outputs without inherent uncertainty estimation, limiting their reliability in safety-critical domains. Probabilistic neural networks (PNNs), which introduce randomness, have emerged as a powerful approach for enabling intrinsic uncertainty quantification. However, traditional CMOS architectures are inherently designed for deterministic operation and actively suppress intrinsic randomness. This poses a fundamental challenge for implementing PNNs, as probabilistic processing introduces significant computational overhead. To address this challenge, we introduce a Magnetic Probabilistic Computing (MPC) platform-an energy-efficient, scalable hardware accelerator that leverages intrinsic magnetic stochasticity for uncertainty-aware computing. This physics-driven strategy utilizes spintronic systems based on magnetic domain walls (DWs) and their dynamics to establish a new paradigm of physical probabilistic computing for AI. The MPC platform integrates three key mechanisms: thermally induced DW stochasticity, voltage controlled magnetic anisotropy (VCMA), and tunneling magnetoresistance (TMR), enabling fully electrical and tunable probabilistic functionality at the device level. As a representative demonstration, we implement a Bayesian Neural Network (BNN) inference structure and validate its functionality on CIFAR-10 classification tasks. Compared to standard 28nm CMOS implementations, our approach achieves a seven orders of magnitude improvement in the overall figure of merit, with substantial gains in area efficiency, energy consumption, and speed. These results underscore the MPC platform's potential to enable reliable and trustworthy physical AI systems. |
| 2025-07-23 | [SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs](http://arxiv.org/abs/2507.17178v1) | Zhiqiang Liu, Enpei Niu et al. | Although large language models (LLMs) have made significant progress in understanding Structured Knowledge (SK) like KG and Table, existing evaluations for SK understanding are non-rigorous (i.e., lacking evaluations of specific capabilities) and focus on a single type of SK. Therefore, we aim to propose a more comprehensive and rigorous structured knowledge understanding benchmark to diagnose the shortcomings of LLMs. In this paper, we introduce SKA-Bench, a Structured Knowledge Augmented QA Benchmark that encompasses four widely used structured knowledge forms: KG, Table, KG+Text, and Table+Text. We utilize a three-stage pipeline to construct SKA-Bench instances, which includes a question, an answer, positive knowledge units, and noisy knowledge units. To evaluate the SK understanding capabilities of LLMs in a fine-grained manner, we expand the instances into four fundamental ability testbeds: Noise Robustness, Order Insensitivity, Information Integration, and Negative Rejection. Empirical evaluations on 8 representative LLMs, including the advanced DeepSeek-R1, indicate that existing LLMs still face significant challenges in understanding structured knowledge, and their performance is influenced by factors such as the amount of noise, the order of knowledge units, and hallucination phenomenon. Our dataset and code are available at https://github.com/Lza12a/SKA-Bench. |
| 2025-07-23 | [Falconry-like palm landing by a flapping-wing drone based on the human gesture interaction and distance-aware flight planning](http://arxiv.org/abs/2507.17144v1) | Kazuki Numazato, Keiichiro Kan et al. | Flapping-wing drones have attracted significant attention due to their biomimetic flight. They are considered more human-friendly due to their characteristics such as low noise and flexible wings, making them suitable for human-drone interactions. However, few studies have explored the practical interaction between humans and flapping-wing drones. On establishing a physical interaction system with flapping-wing drones, we can acquire inspirations from falconers who guide birds of prey to land on their arms. This interaction interprets the human body as a dynamic landing platform, which can be utilized in various scenarios such as crowded or spatially constrained environments. Thus, in this study, we propose a falconry-like interaction system in which a flapping-wing drone performs a palm landing motion on a human hand. To achieve a safe approach toward humans, we design a trajectory planning method that considers both physical and psychological factors of the human safety such as the drone's velocity and distance from the user. We use a commercial flapping platform with our implemented motion planning and conduct experiments to evaluate the palm landing performance and safety. The results demonstrate that our approach enables safe and smooth hand landing interactions. To the best of our knowledge, it is the first time to achieve a contact-based interaction between flapping-wing drones and humans. |
| 2025-07-23 | [HySafe-AI: Hybrid Safety Architectural Analysis Framework for AI Systems: A Case Study](http://arxiv.org/abs/2507.17118v1) | Mandar Pitale, Jelena Frtunikj et al. | AI has become integral to safety-critical areas like autonomous driving systems (ADS) and robotics. The architecture of recent autonomous systems are trending toward end-to-end (E2E) monolithic architectures such as large language models (LLMs) and vision language models (VLMs). In this paper, we review different architectural solutions and then evaluate the efficacy of common safety analyses such as failure modes and effect analysis (FMEA) and fault tree analysis (FTA). We show how these techniques can be improved for the intricate nature of the foundational models, particularly in how they form and utilize latent representations. We introduce HySAFE-AI, Hybrid Safety Architectural Analysis Framework for AI Systems, a hybrid framework that adapts traditional methods to evaluate the safety of AI systems. Lastly, we offer hints of future work and suggestions to guide the evolution of future AI safety standards. |
| 2025-07-22 | [LoRA is All You Need for Safety Alignment of Reasoning LLMs](http://arxiv.org/abs/2507.17075v1) | Yihao Xue, Baharan Mirzasoleiman | Reasoning LLMs have demonstrated remarkable breakthroughs in solving complex problems that were previously out of reach. To ensure LLMs do not assist with harmful requests, safety alignment fine-tuning is necessary in the post-training phase. However, safety alignment fine-tuning has recently been shown to significantly degrade reasoning abilities, a phenomenon known as the "Safety Tax". In this work, we show that using LoRA for SFT on refusal datasets effectively aligns the model for safety without harming its reasoning capabilities. This is because restricting the safety weight updates to a low-rank space minimizes the interference with the reasoning weights. Our extensive experiments across four benchmarks covering math, science, and coding show that this approach produces highly safe LLMs -- with safety levels comparable to full-model fine-tuning -- without compromising their reasoning abilities. Additionally, we observe that LoRA induces weight updates with smaller overlap with the initial weights compared to full-model fine-tuning. We also explore methods that further reduce such overlap -- via regularization or during weight merging -- and observe some improvement on certain tasks. We hope this result motivates designing approaches that yield more consistent improvements in the reasoning-safety trade-off. |
| 2025-07-22 | [How animal movement influences wildlife-vehicle collision risk: a mathematical framework for range-resident species](http://arxiv.org/abs/2507.17058v1) | Benjamin Garcia de Figueiredo, Inês Silva et al. | Wildlife-vehicle collisions (WVC) threaten both biodiversity and human safety worldwide. Despite empirical efforts to characterize the major determinants of WVC risk and optimize mitigation strategies, we still lack a theoretical framework linking traffic, landscape, and individual movement features to collision risk. Here, we introduce such a framework by leveraging recent advances in movement ecology and reaction-diffusion stochastic processes with partially absorbing boundaries. Focusing on range-resident terrestrial mammals -- responsible for most fatal WVCs -- we model interactions with a single linear road and derive exact expressions for key survival statistics, including mean collision time and road-induced lifespan reduction. These quantities are expressed in terms of measurable parameters, such as traffic intensity or road width, and movement parameters that can be robustly estimated from relocation data, such as home-range crossing times, home-range sizes, or distance between home-range center and road. Therefore, our work provides an effective theoretical framework integrating movement and road ecology, laying the foundation for data-driven, evidence-based strategies to mitigate WVCs and promote safer, more sustainable transportation networks. |
| 2025-07-22 | [Pragmatic Policy Development via Interpretable Behavior Cloning](http://arxiv.org/abs/2507.17056v1) | Anton Matsson, Yaochen Rao et al. | Offline reinforcement learning (RL) holds great promise for deriving optimal policies from observational data, but challenges related to interpretability and evaluation limit its practical use in safety-critical domains. Interpretability is hindered by the black-box nature of unconstrained RL policies, while evaluation -- typically performed off-policy -- is sensitive to large deviations from the data-collecting behavior policy, especially when using methods based on importance sampling. To address these challenges, we propose a simple yet practical alternative: deriving treatment policies from the most frequently chosen actions in each patient state, as estimated by an interpretable model of the behavior policy. By using a tree-based model, which is specifically designed to exploit patterns in the data, we obtain a natural grouping of states with respect to treatment. The tree structure ensures interpretability by design, while varying the number of actions considered controls the degree of overlap with the behavior policy, enabling reliable off-policy evaluation. This pragmatic approach to policy development standardizes frequent treatment patterns, capturing the collective clinical judgment embedded in the data. Using real-world examples in rheumatoid arthritis and sepsis care, we demonstrate that policies derived under this framework can outperform current practice, offering interpretable alternatives to those obtained via offline RL. |
| 2025-07-22 | [Shared Control of Holonomic Wheelchairs through Reinforcement Learning](http://arxiv.org/abs/2507.17055v1) | Jannis Bähler, Diego Paez-Granados et al. | Smart electric wheelchairs can improve user experience by supporting the driver with shared control. State-of-the-art work showed the potential of shared control in improving safety in navigation for non-holonomic robots. However, for holonomic systems, current approaches often lead to unintuitive behavior for the user and fail to utilize the full potential of omnidirectional driving. Therefore, we propose a reinforcement learning-based method, which takes a 2D user input and outputs a 3D motion while ensuring user comfort and reducing cognitive load on the driver. Our approach is trained in Isaac Gym and tested in simulation in Gazebo. We compare different RL agent architectures and reward functions based on metrics considering cognitive load and user comfort. We show that our method ensures collision-free navigation while smartly orienting the wheelchair and showing better or competitive smoothness compared to a previous non-learning-based method. We further perform a sim-to-real transfer and demonstrate, to the best of our knowledge, the first real-world implementation of RL-based shared control for an omnidirectional mobility platform. |

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



