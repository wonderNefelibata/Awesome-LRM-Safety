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
| 2025-09-19 | [Designing Culturally Aligned AI Systems For Social Good in Non-Western Contexts](http://arxiv.org/abs/2509.16158v1) | Deepak Varuvel Dennison, Mohit Jain et al. | AI technologies are increasingly deployed in high-stakes domains such as education, healthcare, law, and agriculture to address complex challenges in non-Western contexts. This paper examines eight real-world deployments spanning seven countries and 18 languages, combining 17 interviews with AI developers and domain experts with secondary research. Our findings identify six cross-cutting factors - Language, Domain, Demography, Institution, Task, and Safety - that structured how systems were designed and deployed. These factors were shaped by sociocultural (diversity, practices), institutional (resources, policies), and technological (capabilities, limits) influences. We find that building AI systems required extensive collaboration between AI developers and domain experts. Notably, human resources proved more critical to achieving safe and effective systems in high-stakes domains than technological expertise alone. We present an analytical framework that synthesizes these dynamics and conclude with recommendations for designing AI for social good systems that are culturally grounded, equitable, and responsive to the needs of non-Western contexts. |
| 2025-09-19 | [Automated Cyber Defense with Generalizable Graph-based Reinforcement Learning Agents](http://arxiv.org/abs/2509.16151v1) | Isaiah J. King, Benjamin Bowman et al. | Deep reinforcement learning (RL) is emerging as a viable strategy for automated cyber defense (ACD). The traditional RL approach represents networks as a list of computers in various states of safety or threat. Unfortunately, these models are forced to overfit to specific network topologies, rendering them ineffective when faced with even small environmental perturbations. In this work, we frame ACD as a two-player context-based partially observable Markov decision problem with observations represented as attributed graphs. This approach allows our agents to reason through the lens of relational inductive bias. Agents learn how to reason about hosts interacting with other system entities in a more general manner, and their actions are understood as edits to the graph representing the environment. By introducing this bias, we will show that our agents can better reason about the states of networks and zero-shot adapt to new ones. We show that this approach outperforms the state-of-the-art by a wide margin, and makes our agents capable of defending never-before-seen networks against a wide range of adversaries in a variety of complex, and multi-agent environments. |
| 2025-09-19 | [Randomized Smoothing Meets Vision-Language Models](http://arxiv.org/abs/2509.16088v1) | Emmanouil Seferis, Changshun Wu et al. | Randomized smoothing (RS) is one of the prominent techniques to ensure the correctness of machine learning models, where point-wise robustness certificates can be derived analytically. While RS is well understood for classification, its application to generative models is unclear, since their outputs are sequences rather than labels. We resolve this by connecting generative outputs to an oracle classification task and showing that RS can still be enabled: the final response can be classified as a discrete action (e.g., service-robot commands in VLAs), as harmful vs. harmless (content moderation or toxicity detection in VLMs), or even applying oracles to cluster answers into semantically equivalent ones. Provided that the error rate for the oracle classifier comparison is bounded, we develop the theory that associates the number of samples with the corresponding robustness radius. We further derive improved scaling laws analytically relating the certified radius and accuracy to the number of samples, showing that the earlier result of 2 to 3 orders of magnitude fewer samples sufficing with minimal loss remains valid even under weaker assumptions. Together, these advances make robustness certification both well-defined and computationally feasible for state-of-the-art VLMs, as validated against recent jailbreak-style adversarial attacks. |
| 2025-09-19 | [Communications to Circulations: 3D Wind Field Retrieval and Real-Time Prediction Using 5G GNSS Signals and Deep Learning](http://arxiv.org/abs/2509.16068v1) | Yuchen Ye, Hong Liang et al. | Accurate atmospheric wind field information is crucial for various applications, including weather forecasting, aviation safety, and disaster risk reduction. However, obtaining high spatiotemporal resolution wind data remains challenging due to limitations in traditional in-situ observations and remote sensing techniques, as well as the computational expense and biases of numerical weather prediction (NWP) models. This paper introduces G-WindCast, a novel deep learning framework that leverages signal strength variations from 5G Global Navigation Satellite System (GNSS) signals to retrieve and forecast three-dimensional (3D) atmospheric wind fields. The framework utilizes Forward Neural Networks (FNN) and Transformer networks to capture complex, nonlinear, and spatiotemporal relationships between GNSS-derived features and wind dynamics. Our preliminary results demonstrate promising accuracy in both wind retrieval and short-term wind forecasting (up to 30 minutes lead time), with skill scores comparable to high-resolution NWP outputs in certain scenarios. The model exhibits robustness across different forecast horizons and pressure levels, and its predictions for wind speed and direction show superior agreement with observations compared to concurrent ERA5 reanalysis data. Furthermore, we show that the system can maintain excellent performance for localized forecasting even with a significantly reduced number of GNSS stations (e.g., around 100), highlighting its cost-effectiveness and scalability. This interdisciplinary approach underscores the transformative potential of exploiting non-traditional data sources and deep learning for advanced environmental monitoring and real-time atmospheric applications. |
| 2025-09-19 | [Latent Conditioned Loco-Manipulation Using Motion Priors](http://arxiv.org/abs/2509.16061v1) | Maciej Stępień, Rafael Kourdis et al. | Although humanoid and quadruped robots provide a wide range of capabilities, current control methods, such as Deep Reinforcement Learning, focus mainly on single skills. This approach is inefficient for solving more complicated tasks where high-level goals, physical robot limitations and desired motion style might all need to be taken into account. A more effective approach is to first train a multipurpose motion policy that acquires low-level skills through imitation, while providing latent space control over skill execution. Then, this policy can be used to efficiently solve downstream tasks. This method has already been successful for controlling characters in computer graphics. In this work, we apply the approach to humanoid and quadrupedal loco-manipulation by imitating either simple synthetic motions or kinematically retargeted dog motions. We extend the original formulation to handle constraints, ensuring deployment safety, and use a diffusion discriminator for better imitation quality. We verify our methods by performing loco-manipulation in simulation for the H1 humanoid and Solo12 quadruped, as well as deploying policies on Solo12 hardware. Videos and code are available at https://gepetto.github.io/LaCoLoco/ |
| 2025-09-19 | [SABER: Uncovering Vulnerabilities in Safety Alignment via Cross-Layer Residual Connection](http://arxiv.org/abs/2509.16060v1) | Maithili Joshi, Palash Nandi et al. | Large Language Models (LLMs) with safe-alignment training are powerful instruments with robust language comprehension capabilities. These models typically undergo meticulous alignment procedures involving human feedback to ensure the acceptance of safe inputs while rejecting harmful or unsafe ones. However, despite their massive scale and alignment efforts, LLMs remain vulnerable to jailbreak attacks, where malicious users manipulate the model to produce harmful outputs that it was explicitly trained to avoid. In this study, we find that the safety mechanisms in LLMs are predominantly embedded in the middle-to-late layers. Building on this insight, we introduce a novel white-box jailbreak method, SABER (Safety Alignment Bypass via Extra Residuals), which connects two intermediate layers $s$ and $e$ such that $s < e$, through a residual connection. Our approach achieves a 51% improvement over the best-performing baseline on the HarmBench test set. Furthermore, SABER induces only a marginal shift in perplexity when evaluated on the HarmBench validation set. The source code is publicly available at https://github.com/PalGitts/SABER. |
| 2025-09-19 | [Learning Safety for Obstacle Avoidance via Control Barrier Functions](http://arxiv.org/abs/2509.16037v1) | Shuo Liu, Zhe Huang et al. | Obstacle avoidance is central to safe navigation, especially for robots with arbitrary and nonconvex geometries operating in cluttered environments. Existing Control Barrier Function (CBF) approaches often rely on analytic clearance computations, which are infeasible for complex geometries, or on polytopic approximations, which become intractable when robot configurations are unknown. To address these limitations, this paper trains a residual neural network on a large dataset of robot-obstacle configurations to enable fast and tractable clearance prediction, even at unseen configurations. The predicted clearance defines the radius of a Local Safety Ball (LSB), which ensures continuous-time collision-free navigation. The LSB boundary is encoded as a Discrete-Time High-Order CBF (DHOCBF), whose constraints are incorporated into a nonlinear optimization framework. To improve feasibility, a novel relaxation technique is applied. The resulting framework ensure that the robot's rigid-body motion between consecutive time steps remains collision-free, effectively bridging discrete-time control and continuous-time safety. We show that the proposed method handles arbitrary, including nonconvex, robot geometries and generates collision-free, dynamically feasible trajectories in cluttered environments. Experiments demonstrate millisecond-level solve times and high prediction accuracy, highlighting both safety and efficiency beyond existing CBF-based methods. |
| 2025-09-19 | [Defining and Monitoring Complex Robot Activities via LLMs and Symbolic Reasoning](http://arxiv.org/abs/2509.16006v1) | Francesco Argenziano, Elena Umili et al. | Recent years have witnessed a growing interest in automating labor-intensive and complex activities, i.e., those consisting of multiple atomic tasks, by deploying robots in dynamic and unpredictable environments such as industrial and agricultural settings. A key characteristic of these contexts is that activities are not predefined: while they involve a limited set of possible tasks, their combinations may vary depending on the situation. Moreover, despite recent advances in robotics, the ability for humans to monitor the progress of high-level activities - in terms of past, present, and future actions - remains fundamental to ensure the correct execution of safety-critical processes. In this paper, we introduce a general architecture that integrates Large Language Models (LLMs) with automated planning, enabling humans to specify high-level activities (also referred to as processes) using natural language, and to monitor their execution by querying a robot. We also present an implementation of this architecture using state-of-the-art components and quantitatively evaluate the approach in a real-world precision agriculture scenario. |
| 2025-09-19 | [Simultaneous real-time multispectral fluorescence and reflectance imaging for enhanced intraoperative guidance](http://arxiv.org/abs/2509.15979v1) | Kyriakos Pentarakis, George Kakavelakis et al. | Intraoperative optical imaging is essential for surgical precision and patient safety, but current systems present anatomical and fluorescence information separately, causing delays and increasing cognitive load. A unified system for simultaneous visualization is critically needed to enhance guidance and efficiency. To develop and validate a multispectral imaging system that overcomes fragmented intraoperative imaging by simultaneously capturing and integrating white light reflectance and multiple fluorescence signals in real-time, enhancing surgical precision and safety. The system integrates two synchronized color cameras with optimized optical components, including custom filters and a beam splitter, tailored for clinically relevant fluorophores protoporphyrin IX (PpIX), fluorescein sodium, and indocyanine green (ICG). Advanced image processing techniques enhance visualization accuracy. Performance was evaluated using phantoms simulating clinical conditions. The system provided real time, simultaneous visualization of white light and multiple fluorescence signals without mode switching. Fluorescence emissions from PpIX, fluorescein sodium, and ICG were accurately separated using linear unmixing algorithms. Reflectance images showed high color fidelity, with Delta E values indicating imperceptible to barely perceptible differences compared to a conventional camera. Latency was minimal, ensuring immediate visual feedback suitable for intraoperative use. This multispectral imaging system addresses the limitations of fragmented intraoperative imaging by enabling continuous, real time, integrated visualization of anatomical and fluorescence information. It streamlines workflow, reduces cognitive load, and supports more precise interventions. By leveraging components similar to conventional surgical microscopes, it offers a practical, cost effective solution aligned with clinical needs. |
| 2025-09-19 | [CoReVLA: A Dual-Stage End-to-End Autonomous Driving Framework for Long-Tail Scenarios via Collect-and-Refine](http://arxiv.org/abs/2509.15968v1) | Shiyu Fang, Yiming Cui et al. | Autonomous Driving (AD) systems have made notable progress, but their performance in long-tail, safety-critical scenarios remains limited. These rare cases contribute a disproportionate number of accidents. Vision-Language Action (VLA) models have strong reasoning abilities and offer a potential solution, but their effectiveness is limited by the lack of high-quality data and inefficient learning in such conditions. To address these challenges, we propose CoReVLA, a continual learning end-to-end autonomous driving framework that improves the performance in long-tail scenarios through a dual-stage process of data Collection and behavior Refinement. First, the model is jointly fine-tuned on a mixture of open-source driving QA datasets, allowing it to acquire a foundational understanding of driving scenarios. Next, CoReVLA is deployed within the Cave Automatic Virtual Environment (CAVE) simulation platform, where driver takeover data is collected from real-time interactions. Each takeover indicates a long-tail scenario that CoReVLA fails to handle reliably. Finally, the model is refined via Direct Preference Optimization (DPO), allowing it to learn directly from human preferences and thereby avoid reward hacking caused by manually designed rewards. Extensive open-loop and closed-loop experiments demonstrate that the proposed CoReVLA model can accurately perceive driving scenarios and make appropriate decisions. On the Bench2Drive benchmark, CoReVLA achieves a Driving Score (DS) of 72.18 and a Success Rate (SR) of 50%, outperforming state-of-the-art methods by 7.96 DS and 15% SR under long-tail, safety-critical scenarios. Furthermore, case studies demonstrate the model's ability to continually improve its performance in similar failure-prone scenarios by leveraging past takeover experiences. All codea and preprocessed datasets are available at: https://github.com/FanGShiYuu/CoReVLA |
| 2025-09-19 | [Proper-Time Approach in Asymptotic Safety via Black Hole Quasinormal Modes and Grey-body Factors](http://arxiv.org/abs/2509.15923v1) | Bekir Can Lütfüoğlu, Erdinç Ulaş Saka et al. | We study the quasinormal mode spectrum and grey-body factors of black holes in an effectively quantum-corrected spacetime, focusing on the influence of near-horizon modifications on observable quantities. Employing scalar, electromagnetic, and Dirac test fields, we analyze the perturbation equations and extract the fundamental quasinormal frequencies using both the 6th-order WKB method with Pad\'e resummation and time-domain integration. Our results show that quantum corrections near the horizon significantly affect the real and imaginary parts of the quasinormal modes, particularly for low multipole numbers and in the near-extremal regime. We also verify the robustness of the correspondence between quasinormal modes and grey-body factors by comparing WKB results with those reconstructed from the dominant quasinormal modes. Across all field types and parameter ranges considered, the WKB method proves accurate within a few percent, confirming its reliability in probing the impact of near-horizon physics. These findings support the use of quasinormal ringing and Hawking radiation spectra as sensitive tools for testing quantum modifications of black hole spacetimes. |
| 2025-09-19 | [Failure Modes and Effects Analysis: An Experience from the E-Bike Domain](http://arxiv.org/abs/2509.15893v1) | Andrea Bombarda, Federico Conti et al. | Software failures can have catastrophic and costly consequences. Functional Failure Mode and Effects Analysis (FMEA) is a standard technique used within Cyber-Physical Systems (CPS) to identify software failures and assess their consequences. Simulation-driven approaches have recently been shown to be effective in supporting FMEA. However, industries need evidence of the effectiveness of these approaches to increase practical adoption. This industrial paper presents our experience with using FMEA to analyze the safety of a CPS from the e-Bike domain. We used Simulink Fault Analyzer, an industrial tool that supports engineers with FMEA. We identified 13 realistic faults, modeled them, and analyzed their effects. We sought expert feedback to analyze the appropriateness of our models and the effectiveness of the faults in detecting safety breaches. Our results reveal that for the faults we identified, our models were accurate or contained minor imprecision that we subsequently corrected. They also confirm that FMEA helps engineers improve their models. Specifically, the output provided by the simulation-driven support for 38.4% (5 out of 13) of the faults did not match the engineers' expectations, helping them discover unexpected effects of the faults. We present a thorough discussion of our results and ten lessons learned. Our findings are useful for software engineers who work as Simulink engineers, use the Simulink Fault Analyzer, or work as safety analysts. |
| 2025-09-19 | [A Comparative Study of Rule-Based and Data-Driven Approaches in Industrial Monitoring](http://arxiv.org/abs/2509.15848v1) | Giovanni De Gasperis, Sante Dino Facchini | Industrial monitoring systems, especially when deployed in Industry 4.0 environments, are experiencing a shift in paradigm from traditional rule-based architectures to data-driven approaches leveraging machine learning and artificial intelligence. This study presents a comparison between these two methodologies, analyzing their respective strengths, limitations, and application scenarios, and proposes a basic framework to evaluate their key properties. Rule-based systems offer high interpretability, deterministic behavior, and ease of implementation in stable environments, making them ideal for regulated industries and safety-critical applications. However, they face challenges with scalability, adaptability, and performance in complex or evolving contexts. Conversely, data-driven systems excel in detecting hidden anomalies, enabling predictive maintenance and dynamic adaptation to new conditions. Despite their high accuracy, these models face challenges related to data availability, explainability, and integration complexity. The paper suggests hybrid solutions as a possible promising direction, combining the transparency of rule-based logic with the analytical power of machine learning. Our hypothesis is that the future of industrial monitoring lies in intelligent, synergic systems that leverage both expert knowledge and data-driven insights. This dual approach enhances resilience, operational efficiency, and trust, paving the way for smarter and more flexible industrial environments. |
| 2025-09-19 | [Hierarchical Reinforcement Learning with Low-Level MPC for Multi-Agent Control](http://arxiv.org/abs/2509.15799v1) | Max Studt, Georg Schildbach | Achieving safe and coordinated behavior in dynamic, constraint-rich environments remains a major challenge for learning-based control. Pure end-to-end learning often suffers from poor sample efficiency and limited reliability, while model-based methods depend on predefined references and struggle to generalize. We propose a hierarchical framework that combines tactical decision-making via reinforcement learning (RL) with low-level execution through Model Predictive Control (MPC). For the case of multi-agent systems this means that high-level policies select abstract targets from structured regions of interest (ROIs), while MPC ensures dynamically feasible and safe motion. Tested on a predator-prey benchmark, our approach outperforms end-to-end and shielding-based RL baselines in terms of reward, safety, and consistency, underscoring the benefits of combining structured learning with model-based control. |
| 2025-09-19 | [All-Electric Heavy-Duty Robotic Manipulator: Actuator Configuration Optimization and Sensorless Control](http://arxiv.org/abs/2509.15778v1) | Mohammad Bahari, Amir Hossein Barjini et al. | This paper presents a unified framework that integrates modeling, optimization, and sensorless control of an all-electric heavy-duty robotic manipulator (HDRM) driven by electromechanical linear actuators (EMLAs). An EMLA model is formulated to capture motor electromechanics and direction-dependent transmission efficiencies, while a mathematical model of the HDRM, incorporating both kinematics and dynamics, is established to generate joint-space motion profiles for prescribed TCP trajectories. A safety-ensured trajectory generator, tailored to this model, maps Cartesian goals to joint space while enforcing joint-limit and velocity margins. Based on the resulting force and velocity demands, a multi-objective Non-dominated Sorting Genetic Algorithm II (NSGA-II) is employed to select the optimal EMLA configuration. To accelerate this optimization, a deep neural network, trained with EMLA parameters, is embedded in the optimization process to predict steady-state actuator efficiency from trajectory profiles. For the chosen EMLA design, a physics-informed Kriging surrogate, anchored to the analytic model and refined with experimental data, learns residuals of EMLA outputs to support force and velocity sensorless control. The actuator model is further embedded in a hierarchical virtual decomposition control (VDC) framework that outputs voltage commands. Experimental validation on a one-degree-of-freedom EMLA testbed confirms accurate trajectory tracking and effective sensorless control under varying loads. |
| 2025-09-19 | [Incremental Multistep Forecasting of Battery Degradation Using Pseudo Targets](http://arxiv.org/abs/2509.15740v1) | Jonathan Adam Rico, Nagarajan Raghavan et al. | Data-driven models accurately perform early battery prognosis to prevent equipment failure and further safety hazards. Most existing machine learning (ML) models work in offline mode which must consider their retraining post-deployment every time new data distribution is encountered. Hence, there is a need for an online ML approach where the model can adapt to varying distributions. However, existing online incremental multistep forecasts are a great challenge as there is no way to correct the model of its forecasts at the current instance. Also, these methods need to wait for a considerable amount of time to acquire enough streaming data before retraining. In this study, we propose iFSNet (incremental Fast and Slow learning Network) which is a modified version of FSNet for a single-pass mode (sample-by-sample) to achieve multistep forecasting using pseudo targets. It uses a simple linear regressor of the input sequence to extrapolate pseudo future samples (pseudo targets) and calculate the loss from the rest of the forecast and keep updating the model. The model benefits from the associative memory and adaptive structure mechanisms of FSNet, at the same time the model incrementally improves by using pseudo targets. The proposed model achieved 0.00197 RMSE and 0.00154 MAE on datasets with smooth degradation trajectories while it achieved 0.01588 RMSE and 0.01234 MAE on datasets having irregular degradation trajectories with capacity regeneration spikes. |
| 2025-09-19 | [SMART: Scalable Multi-Agent Reasoning and Trajectory Planning in Dense Environments](http://arxiv.org/abs/2509.15737v1) | Heye Huang, Yibin Yang et al. | Multi-vehicle trajectory planning is a non-convex problem that becomes increasingly difficult in dense environments due to the rapid growth of collision constraints. Efficient exploration of feasible behaviors and resolution of tight interactions are essential for real-time, large-scale coordination. This paper introduces SMART, Scalable Multi-Agent Reasoning and Trajectory Planning, a hierarchical framework that combines priority-based search with distributed optimization to achieve efficient and feasible multi-vehicle planning. The upper layer explores diverse interaction modes using reinforcement learning-based priority estimation and large-step hybrid A* search, while the lower layer refines solutions via parallelizable convex optimization. By partitioning space among neighboring vehicles and constructing robust feasible corridors, the method decouples the joint non-convex problem into convex subproblems solved efficiently in parallel. This design alleviates the step-size trade-off while ensuring kinematic feasibility and collision avoidance. Experiments show that SMART consistently outperforms baselines. On 50 m x 50 m maps, it sustains over 90% success within 1 s up to 25 vehicles, while baselines often drop below 50%. On 100 m x 100 m maps, SMART achieves above 95% success up to 50 vehicles and remains feasible up to 90 vehicles, with runtimes more than an order of magnitude faster than optimization-only approaches. Built on vehicle-to-everything communication, SMART incorporates vehicle-infrastructure cooperation through roadside sensing and agent coordination, improving scalability and safety. Real-world experiments further validate this design, achieving planning times as low as 0.014 s while preserving cooperative behaviors. |
| 2025-09-19 | [How built environment shapes cycling experience: A multi-scale review in historical urban contexts](http://arxiv.org/abs/2509.15657v1) | Haining Ding, Chenxi Wang et al. | Understanding how built environments shape human experience is central to designing sustainable cities. Cycling provides a critical case: it delivers health and environmental benefits, yet its uptake depends strongly on the experience of cycling rather than infrastructure alone. Research on this relationship has grown rapidly but remains fragmented across disciplines and scales, and has concentrated on network-level analyses of routes and connectivity. This bias is especially problematic in historical cities, where embedding new infrastructure is difficult, and where cycling experience is shaped not only by spatial form but also by how cyclists perceive, interpret, and physically respond to their environment - through psychological factors such as safety and comfort, physiological demands such as stress and fatigue, and perceptual cues in the streetscape. We systematically reviewed 68 studies across urban planning, transportation, behavioural science, neuroscience, and public health. Two scales of analysis were identified: a macro scale addressing the ability to cycle and a micro scale addressing the propensity to cycle. Methods were classified into objective and subjective approaches, with hybrid approaches beginning to emerge. We find a persistent reliance on objective proxies, limited integration of subjective accounts, and insufficient attention to the streetscape as a lived environment. Addressing these gaps is essential to explain why environments enable or deter cycling, and to inform the design of cities that support cycling as both mobility and lived experience. |
| 2025-09-19 | [Bench-RNR: Dataset for Benchmarking Repetitive and Non-repetitive Scanning LiDAR for Infrastructure-based Vehicle Localization](http://arxiv.org/abs/2509.15583v1) | Runxin Zhao, Chunxiang Wang et al. | Vehicle localization using roadside LiDARs can provide centimeter-level accuracy for cloud-controlled vehicles while simultaneously serving multiple vehicles, enhanc-ing safety and efficiency. While most existing studies rely on repetitive scanning LiDARs, non-repetitive scanning LiDAR offers advantages such as eliminating blind zones and being more cost-effective. However, its application in roadside perception and localization remains limited. To address this, we present a dataset for infrastructure-based vehicle localization, with data collected from both repetitive and non-repetitive scanning LiDARs, in order to benchmark the performance of different LiDAR scanning patterns. The dataset contains 5,445 frames of point clouds across eight vehicle trajectory sequences, with diverse trajectory types. Our experiments establish base-lines for infrastructure-based vehicle localization and compare the performance of these methods using both non-repetitive and repetitive scanning LiDARs. This work offers valuable insights for selecting the most suitable LiDAR scanning pattern for infrastruc-ture-based vehicle localization. Our dataset is a signifi-cant contribution to the scientific community, supporting advancements in infrastructure-based perception and vehicle localization. The dataset and source code are publicly available at: https://github.com/sjtu-cyberc3/BenchRNR. |
| 2025-09-19 | [Momentum-constrained Hybrid Heuristic Trajectory Optimization Framework with Residual-enhanced DRL for Visually Impaired Scenarios](http://arxiv.org/abs/2509.15582v1) | Yuting Zeng, Zhiwen Zheng et al. | This paper proposes a momentum-constrained hybrid heuristic trajectory optimization framework (MHHTOF) tailored for assistive navigation in visually impaired scenarios, integrating trajectory sampling generation, optimization and evaluation with residual-enhanced deep reinforcement learning (DRL). In the first stage, heuristic trajectory sampling cluster (HTSC) is generated in the Frenet coordinate system using third-order interpolation with fifth-order polynomials and momentum-constrained trajectory optimization (MTO) constraints to ensure smoothness and feasibility. After first stage cost evaluation, the second stage leverages a residual-enhanced actor-critic network with LSTM-based temporal feature modeling to adaptively refine trajectory selection in the Cartesian coordinate system. A dual-stage cost modeling mechanism (DCMM) with weight transfer aligns semantic priorities across stages, supporting human-centered optimization. Experimental results demonstrate that the proposed LSTM-ResB-PPO achieves significantly faster convergence, attaining stable policy performance in approximately half the training iterations required by the PPO baseline, while simultaneously enhancing both reward outcomes and training stability. Compared to baseline method, the selected model reduces average cost and cost variance by 30.3% and 53.3%, and lowers ego and obstacle risks by over 77%. These findings validate the framework's effectiveness in enhancing robustness, safety, and real-time feasibility in complex assistive planning tasks. |

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



