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
| 2025-04-22 | [Impact of Noise on LLM-Models Performance in Abstraction and Reasoning Corpus (ARC) Tasks with Model Temperature Considerations](http://arxiv.org/abs/2504.15903v1) | Nikhil Khandalkar, Pavan Yadav et al. | Recent advancements in Large Language Models (LLMs) have generated growing interest in their structured reasoning capabilities, particularly in tasks involving abstraction and pattern recognition. The Abstraction and Reasoning Corpus (ARC) benchmark plays a crucial role in evaluating these capabilities by testing how well AI models generalize to novel problems. While GPT-4o demonstrates strong performance by solving all ARC tasks under zero-noise conditions, other models like DeepSeek R1 and LLaMA 3.2 fail to solve any, suggesting limitations in their ability to reason beyond simple pattern matching. To explore this gap, we systematically evaluate these models across different noise levels and temperature settings. Our results reveal that the introduction of noise consistently impairs model performance, regardless of architecture. This decline highlights a shared vulnerability: current LLMs, despite showing signs of abstract reasoning, remain highly sensitive to input perturbations. Such fragility raises concerns about their real-world applicability, where noise and uncertainty are common. By comparing how different model architectures respond to these challenges, we offer insights into the structural weaknesses of modern LLMs in reasoning tasks. This work underscores the need for developing more robust and adaptable AI systems capable of handling the ambiguity and variability inherent in real-world scenarios. Our findings aim to guide future research toward enhancing model generalization, robustness, and alignment with human-like cognitive flexibility. |
| 2025-04-22 | [SARI: Structured Audio Reasoning via Curriculum-Guided Reinforcement Learning](http://arxiv.org/abs/2504.15900v1) | Cheng Wen, Tingwei Guo et al. | Recent work shows that reinforcement learning(RL) can markedly sharpen the reasoning ability of large language models (LLMs) by prompting them to "think before answering." Yet whether and how these gains transfer to audio-language reasoning remains largely unexplored. We extend the Group-Relative Policy Optimization (GRPO) framework from DeepSeek-R1 to a Large Audio-Language Model (LALM), and construct a 32k sample multiple-choice corpus. Using a two-stage regimen supervised fine-tuning on structured and unstructured chains-of-thought, followed by curriculum-guided GRPO, we systematically compare implicit vs. explicit, and structured vs. free form reasoning under identical architectures. Our structured audio reasoning model, SARI (Structured Audio Reasoning via Curriculum-Guided Reinforcement Learning), achieves a 16.35% improvement in average accuracy over the base model Qwen2-Audio-7B-Instruct. Furthermore, the variant built upon Qwen2.5-Omni reaches state-of-the-art performance of 67.08% on the MMAU test-mini benchmark. Ablation experiments show that on the base model we use: (i) SFT warm-up is important for stable RL training, (ii) structured chains yield more robust generalization than unstructured ones, and (iii) easy-to-hard curricula accelerate convergence and improve final performance. These findings demonstrate that explicit, structured reasoning and curriculum learning substantially enhances audio-language understanding. |
| 2025-04-22 | [Dynamic Early Exit in Reasoning Models](http://arxiv.org/abs/2504.15895v1) | Chenxu Yang, Qingyi Si et al. | Recent advances in large reasoning language models (LRLMs) rely on test-time scaling, which extends long chain-of-thought (CoT) generation to solve complex tasks. However, overthinking in long CoT not only slows down the efficiency of problem solving, but also risks accuracy loss due to the extremely detailed or redundant reasoning steps. We propose a simple yet effective method that allows LLMs to self-truncate CoT sequences by early exit during generation. Instead of relying on fixed heuristics, the proposed method monitors model behavior at potential reasoning transition points (e.g.,"Wait" tokens) and dynamically terminates the next reasoning chain's generation when the model exhibits high confidence in a trial answer. Our method requires no additional training and can be seamlessly integrated into existing o1-like reasoning LLMs. Experiments on multiple reasoning benchmarks MATH-500, AMC 2023, GPQA Diamond and AIME 2024 show that the proposed method is consistently effective on deepseek-series reasoning LLMs, reducing the length of CoT sequences by an average of 31% to 43% while improving accuracy by 1.7% to 5.7%. |
| 2025-04-22 | [MS-Occ: Multi-Stage LiDAR-Camera Fusion for 3D Semantic Occupancy Prediction](http://arxiv.org/abs/2504.15888v1) | Zhiqiang Wei, Lianqing Zheng et al. | Accurate 3D semantic occupancy perception is essential for autonomous driving in complex environments with diverse and irregular objects. While vision-centric methods suffer from geometric inaccuracies, LiDAR-based approaches often lack rich semantic information. To address these limitations, MS-Occ, a novel multi-stage LiDAR-camera fusion framework which includes middle-stage fusion and late-stage fusion, is proposed, integrating LiDAR's geometric fidelity with camera-based semantic richness via hierarchical cross-modal fusion. The framework introduces innovations at two critical stages: (1) In the middle-stage feature fusion, the Gaussian-Geo module leverages Gaussian kernel rendering on sparse LiDAR depth maps to enhance 2D image features with dense geometric priors, and the Semantic-Aware module enriches LiDAR voxels with semantic context via deformable cross-attention; (2) In the late-stage voxel fusion, the Adaptive Fusion (AF) module dynamically balances voxel features across modalities, while the High Classification Confidence Voxel Fusion (HCCVF) module resolves semantic inconsistencies using self-attention-based refinement. Experiments on the nuScenes-OpenOccupancy benchmark show that MS-Occ achieves an Intersection over Union (IoU) of 32.1% and a mean IoU (mIoU) of 25.3%, surpassing the state-of-the-art by +0.7% IoU and +2.4% mIoU. Ablation studies further validate the contribution of each module, with substantial improvements in small-object perception, demonstrating the practical value of MS-Occ for safety-critical autonomous driving scenarios. |
| 2025-04-22 | [Embedded Safe Reactive Navigation for Multirotors Systems using Control Barrier Functions](http://arxiv.org/abs/2504.15850v1) | Nazar Misyats, Marvin Harms et al. | Aiming to promote the wide adoption of safety filters for autonomous aerial robots, this paper presents a safe control architecture designed for seamless integration into widely used open-source autopilots. Departing from methods that require consistent localization and mapping, we formalize the obstacle avoidance problem as a composite control barrier function constructed only from the online onboard range measurements. The proposed framework acts as a safety filter, modifying the acceleration references derived by the nominal position/velocity control loops, and is integrated into the PX4 autopilot stack. Experimental studies using a small multirotor aerial robot demonstrate the effectiveness and performance of the solution within dynamic maneuvering and unknown environments. |
| 2025-04-22 | [TrustGeoGen: Scalable and Formal-Verified Data Engine for Trustworthy Multi-modal Geometric Problem Solving](http://arxiv.org/abs/2504.15780v1) | Daocheng Fu, Zijun Chen et al. | Mathematical geometric problem solving (GPS) often requires effective integration of multimodal information and verifiable logical coherence. Despite the fast development of large language models in general problem solving, it remains unresolved regarding with both methodology and benchmarks, especially given the fact that exiting synthetic GPS benchmarks are often not self-verified and contain noise and self-contradicted information due to the illusion of LLMs. In this paper, we propose a scalable data engine called TrustGeoGen for problem generation, with formal verification to provide a principled benchmark, which we believe lays the foundation for the further development of methods for GPS. The engine synthesizes geometric data through four key innovations: 1) multimodal-aligned generation of diagrams, textual descriptions, and stepwise solutions; 2) formal verification ensuring rule-compliant reasoning paths; 3) a bootstrapping mechanism enabling complexity escalation via recursive state generation and 4) our devised GeoExplore series algorithms simultaneously produce multi-solution variants and self-reflective backtracking traces. By formal logical verification, TrustGeoGen produces GeoTrust-200K dataset with guaranteed modality integrity, along with GeoTrust-test testset. Experiments reveal the state-of-the-art models achieve only 49.17\% accuracy on GeoTrust-test, demonstrating its evaluation stringency. Crucially, models trained on GeoTrust achieve OOD generalization on GeoQA, significantly reducing logical inconsistencies relative to pseudo-label annotated by OpenAI-o1. Our code is available at https://github.com/Alpha-Innovator/TrustGeoGen |
| 2025-04-22 | [Pose Optimization for Autonomous Driving Datasets using Neural Rendering Models](http://arxiv.org/abs/2504.15776v1) | Quentin Herau, Nathan Piasco et al. | Autonomous driving systems rely on accurate perception and localization of the ego car to ensure safety and reliability in challenging real-world driving scenarios. Public datasets play a vital role in benchmarking and guiding advancement in research by providing standardized resources for model development and evaluation. However, potential inaccuracies in sensor calibration and vehicle poses within these datasets can lead to erroneous evaluations of downstream tasks, adversely impacting the reliability and performance of the autonomous systems. To address this challenge, we propose a robust optimization method based on Neural Radiance Fields (NeRF) to refine sensor poses and calibration parameters, enhancing the integrity of dataset benchmarks. To validate improvement in accuracy of our optimized poses without ground truth, we present a thorough evaluation process, relying on reprojection metrics, Novel View Synthesis rendering quality, and geometric alignment. We demonstrate that our method achieves significant improvements in sensor pose accuracy. By optimizing these critical parameters, our approach not only improves the utility of existing datasets but also paves the way for more reliable autonomous driving models. To foster continued progress in this field, we make the optimized sensor poses publicly available, providing a valuable resource for the research community. |
| 2025-04-22 | [Prediction of CO2 reduction reaction intermediates and products on transition metal-doped r-GeSe monolayers:A combined DFT and machine learning approach](http://arxiv.org/abs/2504.15710v1) | Xuxin Kang, Wenjing Zhou et al. | The electrocatalytic CO2 reduction reaction (CO2RR) is a complex multi-proton-electron transfer process that generates a vast network of reaction intermediates. Accurate prediction of free energy changes (G) of these intermediates and products is essential for evaluating catalytic performance. We combined density functional theory (DFT) and machine learning (ML) to screen 25 single-atom catalysts (SACs) on defective r-GeSe monolayers for CO2 reduction to methanol, methane, and formic acid. Among nine ML models evaluated with 14 intrinsic and DFT-based features, the XGBoost performed best (R2 = 0.92 and MAE = 0.24 eV), aligning closely with DFT calculations and identifying Ni, Ru, and Rh@GeSe as prospective catalysts. Feature importance analysis in free energy and product predictions highlighted the significance of CO2 activation with O-C-O and IPC-O1 as the key attributes. Furthermore, by incorporating non-DFT-based features, rapid predictions became possible, and the XGBoost model retained its predictive performance with R2 = 0.89 and MAE = 0.29 eV. This accuracy was further validated using Ir@GeSe. Our work highlights effective SACs for CO2RR, and provides valuable insights for efficient catalyst design. |
| 2025-04-22 | [Advancing Embodied Agent Security: From Safety Benchmarks to Input Moderation](http://arxiv.org/abs/2504.15699v1) | Ning Wang, Zihan Yan et al. | Embodied agents exhibit immense potential across a multitude of domains, making the assurance of their behavioral safety a fundamental prerequisite for their widespread deployment. However, existing research predominantly concentrates on the security of general large language models, lacking specialized methodologies for establishing safety benchmarks and input moderation tailored to embodied agents. To bridge this gap, this paper introduces a novel input moderation framework, meticulously designed to safeguard embodied agents. This framework encompasses the entire pipeline, including taxonomy definition, dataset curation, moderator architecture, model training, and rigorous evaluation. Notably, we introduce EAsafetyBench, a meticulously crafted safety benchmark engineered to facilitate both the training and stringent assessment of moderators specifically designed for embodied agents. Furthermore, we propose Pinpoint, an innovative prompt-decoupled input moderation scheme that harnesses a masked attention mechanism to effectively isolate and mitigate the influence of functional prompts on moderation tasks. Extensive experiments conducted on diverse benchmark datasets and models validate the feasibility and efficacy of the proposed approach. The results demonstrate that our methodologies achieve an impressive average detection accuracy of 94.58%, surpassing the performance of existing state-of-the-art techniques, alongside an exceptional moderation processing time of merely 0.002 seconds per instance. |
| 2025-04-22 | [Symbolic Runtime Verification and Adaptive Decision-Making for Robot-Assisted Dressing](http://arxiv.org/abs/2504.15666v1) | Yasmin Rafiq, Gricel Vázquez et al. | We present a control framework for robot-assisted dressing that augments low-level hazard response with runtime monitoring and formal verification. A parametric discrete-time Markov chain (pDTMC) models the dressing process, while Bayesian inference dynamically updates this pDTMC's transition probabilities based on sensory and user feedback. Safety constraints from hazard analysis are expressed in probabilistic computation tree logic, and symbolically verified using a probabilistic model checker. We evaluate reachability, cost, and reward trade-offs for garment-snag mitigation and escalation, enabling real-time adaptation. Our approach provides a formal yet lightweight foundation for safety-aware, explainable robotic assistance. |
| 2025-04-22 | [An ACO-MPC Framework for Energy-Efficient and Collision-Free Path Planning in Autonomous Maritime Navigation](http://arxiv.org/abs/2504.15611v1) | Yaoze Liu, Zhen Tian et al. | Automated driving on ramps presents significant challenges due to the need to balance both safety and efficiency during lane changes. This paper proposes an integrated planner for automated vehicles (AVs) on ramps, utilizing an unsatisfactory level metric for efficiency and arrow-cluster-based sampling for safety. The planner identifies optimal times for the AV to change lanes, taking into account the vehicle's velocity as a key factor in efficiency. Additionally, the integrated planner employs arrow-cluster-based sampling to evaluate collision risks and select an optimal lane-changing curve. Extensive simulations were conducted in a ramp scenario to verify the planner's efficient and safe performance. The results demonstrate that the proposed planner can effectively select an appropriate lane-changing time point and a safe lane-changing curve for AVs, without incurring any collisions during the maneuver. |
| 2025-04-22 | [A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment](http://arxiv.org/abs/2504.15585v1) | Kun Wang, Guibin Zhang et al. | The remarkable success of Large Language Models (LLMs) has illuminated a promising pathway toward achieving Artificial General Intelligence for both academic and industrial communities, owing to their unprecedented performance across various applications. As LLMs continue to gain prominence in both research and commercial domains, their security and safety implications have become a growing concern, not only for researchers and corporations but also for every nation. Currently, existing surveys on LLM safety primarily focus on specific stages of the LLM lifecycle, e.g., deployment phase or fine-tuning phase, lacking a comprehensive understanding of the entire "lifechain" of LLMs. To address this gap, this paper introduces, for the first time, the concept of "full-stack" safety to systematically consider safety issues throughout the entire process of LLM training, deployment, and eventual commercialization. Compared to the off-the-shelf LLM safety surveys, our work demonstrates several distinctive advantages: (I) Comprehensive Perspective. We define the complete LLM lifecycle as encompassing data preparation, pre-training, post-training, deployment and final commercialization. To our knowledge, this represents the first safety survey to encompass the entire lifecycle of LLMs. (II) Extensive Literature Support. Our research is grounded in an exhaustive review of over 800+ papers, ensuring comprehensive coverage and systematic organization of security issues within a more holistic understanding. (III) Unique Insights. Through systematic literature analysis, we have developed reliable roadmaps and perspectives for each chapter. Our work identifies promising research directions, including safety in data generation, alignment techniques, model editing, and LLM-based agent systems. These insights provide valuable guidance for researchers pursuing future work in this field. |
| 2025-04-22 | [RiskNet: Interaction-Aware Risk Forecasting for Autonomous Driving in Long-Tail Scenarios](http://arxiv.org/abs/2504.15541v1) | Qichao Liu, Heye Huang et al. | Ensuring the safety of autonomous vehicles (AVs) in long-tail scenarios remains a critical challenge, particularly under high uncertainty and complex multi-agent interactions. To address this, we propose RiskNet, an interaction-aware risk forecasting framework, which integrates deterministic risk modeling with probabilistic behavior prediction for comprehensive risk assessment. At its core, RiskNet employs a field-theoretic model that captures interactions among ego vehicle, surrounding agents, and infrastructure via interaction fields and force. This model supports multidimensional risk evaluation across diverse scenarios (highways, intersections, and roundabouts), and shows robustness under high-risk and long-tail settings. To capture the behavioral uncertainty, we incorporate a graph neural network (GNN)-based trajectory prediction module, which learns multi-modal future motion distributions. Coupled with the deterministic risk field, it enables dynamic, probabilistic risk inference across time, enabling proactive safety assessment under uncertainty. Evaluations on the highD, inD, and rounD datasets, spanning lane changes, turns, and complex merges, demonstrate that our method significantly outperforms traditional approaches (e.g., TTC, THW, RSS, NC Field) in terms of accuracy, responsiveness, and directional sensitivity, while maintaining strong generalization across scenarios. This framework supports real-time, scenario-adaptive risk forecasting and demonstrates strong generalization across uncertain driving environments. It offers a unified foundation for safety-critical decision-making in long-tail scenarios. |
| 2025-04-22 | [Towards Resilience and Autonomy-based Approaches for Adolescents Online Safety](http://arxiv.org/abs/2504.15533v1) | Jinkyung Park, Mamtaj Akter et al. | In this position paper, we discuss the paradigm shift that has emerged in the literature, suggesting to move away from restrictive and authoritarian parental mediation approaches to move toward resilient-based and privacy-preserving solutions to promote adolescents' online safety. We highlight the limitations of restrictive mediation strategies, which often induce a trade-off between teens' privacy and online safety, and call for more teen-centric frameworks that can empower teens to self-regulate while using the technology in meaningful ways. We also present an overview of empirical studies that conceptualized and examined resilience-based approaches to promoting the digital well-being of teens in a way to empower teens to be more resilient. |
| 2025-04-22 | [T2VShield: Model-Agnostic Jailbreak Defense for Text-to-Video Models](http://arxiv.org/abs/2504.15512v1) | Siyuan Liang, Jiayang Liu et al. | The rapid development of generative artificial intelligence has made text to video models essential for building future multimodal world simulators. However, these models remain vulnerable to jailbreak attacks, where specially crafted prompts bypass safety mechanisms and lead to the generation of harmful or unsafe content. Such vulnerabilities undermine the reliability and security of simulation based applications. In this paper, we propose T2VShield, a comprehensive and model agnostic defense framework designed to protect text to video models from jailbreak threats. Our method systematically analyzes the input, model, and output stages to identify the limitations of existing defenses, including semantic ambiguities in prompts, difficulties in detecting malicious content in dynamic video outputs, and inflexible model centric mitigation strategies. T2VShield introduces a prompt rewriting mechanism based on reasoning and multimodal retrieval to sanitize malicious inputs, along with a multi scope detection module that captures local and global inconsistencies across time and modalities. The framework does not require access to internal model parameters and works with both open and closed source systems. Extensive experiments on five platforms show that T2VShield can reduce jailbreak success rates by up to 35 percent compared to strong baselines. We further develop a human centered audiovisual evaluation protocol to assess perceptual safety, emphasizing the importance of visual level defense in enhancing the trustworthiness of next generation multimodal simulators. |
| 2025-04-22 | [Automatically Detecting Numerical Instability in Machine Learning Applications via Soft Assertions](http://arxiv.org/abs/2504.15507v1) | Shaila Sharmin, Anwar Hossain Zahid et al. | Machine learning (ML) applications have become an integral part of our lives. ML applications extensively use floating-point computation and involve very large/small numbers; thus, maintaining the numerical stability of such complex computations remains an important challenge. Numerical bugs can lead to system crashes, incorrect output, and wasted computing resources. In this paper, we introduce a novel idea, namely soft assertions (SA), to encode safety/error conditions for the places where numerical instability can occur. A soft assertion is an ML model automatically trained using the dataset obtained during unit testing of unstable functions. Given the values at the unstable function in an ML application, a soft assertion reports how to change these values in order to trigger the instability. We then use the output of soft assertions as signals to effectively mutate inputs to trigger numerical instability in ML applications. In the evaluation, we used the GRIST benchmark, a total of 79 programs, as well as 15 real-world ML applications from GitHub. We compared our tool with 5 state-of-the-art (SOTA) fuzzers. We found all the GRIST bugs and outperformed the baselines. We found 13 numerical bugs in real-world code, one of which had already been confirmed by the GitHub developers. While the baselines mostly found the bugs that report NaN and INF, our tool \tool found numerical bugs with incorrect output. We showed one case where the Tumor Detection Model, trained on Brain MRI images, should have predicted "tumor", but instead, it incorrectly predicted "no tumor" due to the numerical bugs. Our replication package is located at https://figshare.com/s/6528d21ccd28bea94c32. |
| 2025-04-21 | [Nearly Optimal Nonlinear Safe Control with BaS-SDRE](http://arxiv.org/abs/2504.15453v1) | Hassan Almubarak, Maitham F. AL-Sunni et al. | The State-Dependent Riccati Equation (SDRE) approach has emerged as a systematic and effective means of designing nearly optimal nonlinear controllers. The Barrier States (BaS) embedding methodology was developed recently for safe multi-objective controls in which the safety condition is manifested as a state to be controlled along with other states of the system. The overall system, termed the safety embedded system, is highly nonlinear even if the original system is linear. This paper develops a nonlinear nearly optimal safe feedback control technique by combining the two strategies effectively. First, the BaS is derived in an extended linearization formulation to be subsequently used to form an extended safety embedded system. A new optimal control problem is formed thereafter, which is used to construct a safety embedded State-Dependent Riccati Equation, termed BaS-SDRE, whose solution approximates the solution of the optimal control problem's associated Hamilton-Jacobi-Bellman (HJB) equation. The BaS-SDRE is then solved online to synthesize the nearly optimal safe control. The proposed technique's efficacy is demonstrated on an unstable, constrained linear system that shows how the synthesized control reacts to nonlinearities near the unsafe region, a nonlinear flight control system with limited path angular velocity that exists due to structural and dynamic concerns, and a planar quadrotor system that navigates safely in a crowded environment. |
| 2025-04-21 | [Solving Multi-Agent Safe Optimal Control with Distributed Epigraph Form MARL](http://arxiv.org/abs/2504.15425v1) | Songyuan Zhang, Oswin So et al. | Tasks for multi-robot systems often require the robots to collaborate and complete a team goal while maintaining safety. This problem is usually formalized as a constrained Markov decision process (CMDP), which targets minimizing a global cost and bringing the mean of constraint violation below a user-defined threshold. Inspired by real-world robotic applications, we define safety as zero constraint violation. While many safe multi-agent reinforcement learning (MARL) algorithms have been proposed to solve CMDPs, these algorithms suffer from unstable training in this setting. To tackle this, we use the epigraph form for constrained optimization to improve training stability and prove that the centralized epigraph form problem can be solved in a distributed fashion by each agent. This results in a novel centralized training distributed execution MARL algorithm named Def-MARL. Simulation experiments on 8 different tasks across 2 different simulators show that Def-MARL achieves the best overall performance, satisfies safety constraints, and maintains stable training. Real-world hardware experiments on Crazyflie quadcopters demonstrate the ability of Def-MARL to safely coordinate agents to complete complex collaborative tasks compared to other methods. |
| 2025-04-21 | [Safety Embedded Adaptive Control Using Barrier States](http://arxiv.org/abs/2504.15423v1) | Maitham F. AL-Sunni, Hassan Almubarak et al. | In this work, we explore the application of barrier states (BaS) in the realm of safe nonlinear adaptive control. Our proposed framework derives barrier states for systems with parametric uncertainty, which are augmented into the uncertain dynamical model. We employ an adaptive nonlinear control strategy based on a control Lyapunov functions approach to design a stabilizing controller for the augmented system. The developed theory shows that the controller ensures safe control actions for the original system while meeting specified performance objectives. We validate the effectiveness of our approach through simulations on diverse systems, including a planar quadrotor subject to unknown drag forces and an adaptive cruise control system, for which we provide comparisons with existing methodologies. |
| 2025-04-21 | [$k$-Inductive and Interpolation-Inspired Barrier Certificates for Stochastic Dynamical Systems](http://arxiv.org/abs/2504.15412v1) | Mohammed Adib Oumer, Vishnu Murali et al. | We introduce two notions of barrier certificates that use multiple functions to provide a lower bound on the probabilistic satisfaction of safety for stochastic dynamical systems. A barrier certificate for a stochastic dynamical system acts as a nonnegative supermartingale, and provides a lower bound on the probability that the system is safe. The promise of such certificates is that their search can be effectively automated. Typically, one may use optimization or SMT solvers to find such barrier certificates of a given fixed template. When such approaches fail, a typical approach is to instead change the template. We propose an alternative approach that we dub interpolation-inspired barrier certificates. An interpolation-inspired barrier certificate consists of a set of functions that jointly provide a lower bound on the probability of satisfying safety. We show how one may find such certificates of a fixed template, even when we fail to find standard barrier certificates of the same template. However, we note that such certificates still need to ensure a supermartingale guarantee for one function in the set. To address this challenge, we consider the use of $k$-induction with these interpolation-inspired certificates. The recent use of $k$-induction in barrier certificates allows one to relax the supermartingale requirement at every time step to a combination of a supermartingale requirement every $k$ steps and a $c$-martingale requirement for the intermediate steps. We provide a generic formulation of a barrier certificate that we dub $k$-inductive interpolation-inspired barrier certificate. The formulation allows for several combinations of interpolation and $k$-induction for barrier certificate. We present two examples among the possible combinations. We finally present sum-of-squares programming to synthesize this set of functions and demonstrate their utility in case studies. |

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



