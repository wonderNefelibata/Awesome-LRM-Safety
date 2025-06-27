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
| 2025-06-26 | [PsyLite Technical Report](http://arxiv.org/abs/2506.21536v1) | Fangjun Ding, Renyu Zhang et al. | With the rapid development of digital technology, AI-driven psychological counseling has gradually become an important research direction in the field of mental health. However, existing models still have deficiencies in dialogue safety, detailed scenario handling, and lightweight deployment. To address these issues, this study proposes PsyLite, a lightweight psychological counseling large language model agent developed based on the base model InternLM2.5-7B-chat. Through a two-stage training strategy (hybrid distillation data fine-tuning and ORPO preference optimization), PsyLite enhances the model's deep-reasoning ability, psychological counseling ability, and safe dialogue ability. After deployment using Ollama and Open WebUI, a custom workflow is created with Pipelines. An innovative conditional RAG is designed to introduce crosstalk humor elements at appropriate times during psychological counseling to enhance user experience and decline dangerous requests to strengthen dialogue safety. Evaluations show that PsyLite outperforms the baseline models in the Chinese general evaluation (CEval), psychological counseling professional evaluation (CPsyCounE), and dialogue safety evaluation (SafeDialBench), particularly in psychological counseling professionalism (CPsyCounE score improvement of 47.6\%) and dialogue safety (\safe{} score improvement of 2.4\%). Additionally, the model uses quantization technology (GGUF q4\_k\_m) to achieve low hardware deployment (5GB memory is sufficient for operation), providing a feasible solution for psychological counseling applications in resource-constrained environments. |
| 2025-06-26 | [Towards Reliable Detection of Empty Space: Conditional Marked Point Processes for Object Detection](http://arxiv.org/abs/2506.21486v1) | Tobias J. Riedlinger, Kira Maag et al. | Deep neural networks have set the state-of-the-art in computer vision tasks such as bounding box detection and semantic segmentation. Object detectors and segmentation models assign confidence scores to predictions, reflecting the model's uncertainty in object detection or pixel-wise classification. However, these confidence estimates are often miscalibrated, as their architectures and loss functions are tailored to task performance rather than probabilistic foundation. Even with well calibrated predictions, object detectors fail to quantify uncertainty outside detected bounding boxes, i.e., the model does not make a probability assessment of whether an area without detected objects is truly free of obstacles. This poses a safety risk in applications such as automated driving, where uncertainty in empty areas remains unexplored. In this work, we propose an object detection model grounded in spatial statistics. Bounding box data matches realizations of a marked point process, commonly used to describe the probabilistic occurrence of spatial point events identified as bounding box centers, where marks are used to describe the spatial extension of bounding boxes and classes. Our statistical framework enables a likelihood-based training and provides well-defined confidence estimates for whether a region is drivable, i.e., free of objects. We demonstrate the effectiveness of our method through calibration assessments and evaluation of performance. |
| 2025-06-26 | [A Comprehensive Dataset for Underground Miner Detection in Diverse Scenario](http://arxiv.org/abs/2506.21451v1) | Cyrus Addy, Ajay Kumar Gurumadaiah et al. | Underground mining operations face significant safety challenges that make emergency response capabilities crucial. While robots have shown promise in assisting with search and rescue operations, their effectiveness depends on reliable miner detection capabilities. Deep learning algorithms offer potential solutions for automated miner detection, but require comprehensive training datasets, which are currently lacking for underground mining environments. This paper presents a novel thermal imaging dataset specifically designed to enable the development and validation of miner detection systems for potential emergency applications. We systematically captured thermal imagery of various mining activities and scenarios to create a robust foundation for detection algorithms. To establish baseline performance metrics, we evaluated several state-of-the-art object detection algorithms including YOLOv8, YOLOv10, YOLO11, and RT-DETR on our dataset. While not exhaustive of all possible emergency situations, this dataset serves as a crucial first step toward developing reliable thermal-based miner detection systems that could eventually be deployed in real emergency scenarios. This work demonstrates the feasibility of using thermal imaging for miner detection and establishes a foundation for future research in this critical safety application. |
| 2025-06-26 | [Real-time Terrain Analysis for Off-road Autonomous Vehicles](http://arxiv.org/abs/2506.21347v1) | Edwina Lewis, Aditya Parameshwaran et al. | This research addresses critical autonomous vehicle control challenges arising from road roughness variation, which induces course deviations and potential loss of road contact during steering operations. We present a novel real-time road roughness estimation system employing Bayesian calibration methodology that processes axle accelerations to predict terrain roughness with quantifiable confidence measures. The technical framework integrates a Gaussian process surrogate model with a simulated half-vehicle model, systematically processing vehicle velocity and road surface roughness parameters to generate corresponding axle acceleration responses. The Bayesian calibration routine performs inverse estimation of road roughness from observed accelerations and velocities, yielding posterior distributions that quantify prediction uncertainty for adaptive risk management. Training data generation utilizes Latin Hypercube sampling across comprehensive velocity and roughness parameter spaces, while the calibrated model integrates seamlessly with a Simplex controller architecture to dynamically adjust velocity limits based on real-time roughness predictions. Experimental validation on stochastically generated surfaces featuring varying roughness regions demonstrates robust real-time characterization capabilities, with the integrated Simplex control strategy effectively enhancing autonomous vehicle operational safety through proactive surface condition response. This innovative Bayesian framework establishes a comprehensive foundation for mitigating roughness-related operational risks while simultaneously improving efficiency and safety margins in autonomous vehicle systems. |
| 2025-06-26 | [Coordinated Control of Autonomous Vehicles for Traffic Density Reduction at a Signalized Junction: An MPC Approach](http://arxiv.org/abs/2506.21302v1) | Rudra Sen, Subashish Datta | The effective and safe management of traffic is a key issue due to the rapid advancement of the urban transportation system. Connected autonomous vehicles (CAVs) possess the capability to connect with each other and adjacent infrastructure, presenting novel opportunities for enhancing traffic flow and coordination. This work proposes a dual-mode model predictive control (MPC) architecture that tackles two interrelated issues: mitigating traffic density at signalized junctions and facilitating seamless, cooperative lane changes in high-density traffic conditions. The objective of this work is to facilitate responsive decision-making for CAVs, thereby enhancing the efficiency and safety of urban mobility. Moreover, we ensure recursive feasibility and convergence of the proposed MPC scheme by the integration of an online-calculated maximal control invariant terminal set. Finally, the efficacy of the proposed approach is validated through numerical simulation. |
| 2025-06-26 | [Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](http://arxiv.org/abs/2506.21252v1) | Tianyi Men, Zhuoran Jin et al. | As Multimodal Large Language Models (MLLMs) advance, multimodal agents show promise in real-world tasks like web navigation and embodied intelligence. However, due to limitations in a lack of external feedback, these agents struggle with self-correction and generalization. A promising approach is to use reward models as external feedback, but there is no clear on how to select reward models for agents. Thus, there is an urgent need to build a reward bench targeted at agents. To address these challenges, we propose Agent-RewardBench, a benchmark designed to evaluate reward modeling ability in MLLMs. The benchmark is characterized by three key features: (1) Multiple dimensions and real-world agent scenarios evaluation. It covers perception, planning, and safety with 7 scenarios; (2) Step-level reward evaluation. It allows for the assessment of agent capabilities at the individual steps of a task, providing a more granular view of performance during the planning process; and (3) Appropriately difficulty and high-quality. We carefully sample from 10 diverse models, difficulty control to maintain task challenges, and manual verification to ensure the integrity of the data. Experiments demonstrate that even state-of-the-art multimodal models show limited performance, highlighting the need for specialized training in agent reward modeling. Code is available at github. |
| 2025-06-26 | [Dynamic Risk-Aware MPPI for Mobile Robots in Crowds via Efficient Monte Carlo Approximations](http://arxiv.org/abs/2506.21205v1) | Elia Trevisan, Khaled A. Mustafa et al. | Deploying mobile robots safely among humans requires the motion planner to account for the uncertainty in the other agents' predicted trajectories. This remains challenging in traditional approaches, especially with arbitrarily shaped predictions and real-time constraints. To address these challenges, we propose a Dynamic Risk-Aware Model Predictive Path Integral control (DRA-MPPI), a motion planner that incorporates uncertain future motions modelled with potentially non-Gaussian stochastic predictions. By leveraging MPPI's gradient-free nature, we propose a method that efficiently approximates the joint Collision Probability (CP) among multiple dynamic obstacles for several hundred sampled trajectories in real-time via a Monte Carlo (MC) approach. This enables the rejection of samples exceeding a predefined CP threshold or the integration of CP as a weighted objective within the navigation cost function. Consequently, DRA-MPPI mitigates the freezing robot problem while enhancing safety. Real-world and simulated experiments with multiple dynamic obstacles demonstrate DRA-MPPI's superior performance compared to state-of-the-art approaches, including Scenario-based Model Predictive Control (S-MPC), Frenet planner, and vanilla MPPI. |
| 2025-06-26 | [Out-of-Distribution Semantic Occupancy Prediction](http://arxiv.org/abs/2506.21185v1) | Yuheng Zhang, Mengfei Duan et al. | 3D Semantic Occupancy Prediction is crucial for autonomous driving, providing a dense, semantically rich environmental representation. However, existing methods focus on in-distribution scenes, making them susceptible to Out-of-Distribution (OoD) objects and long-tail distributions, which increases the risk of undetected anomalies and misinterpretations, posing safety hazards. To address these challenges, we introduce Out-of-Distribution Semantic Occupancy Prediction, targeting OoD detection in 3D voxel space. To fill the gaps in the dataset, we propose a Synthetic Anomaly Integration Pipeline that injects synthetic anomalies while preserving realistic spatial and occlusion patterns, enabling the creation of two datasets: VAA-KITTI and VAA-KITTI-360. We introduce OccOoD, a novel framework integrating OoD detection into 3D semantic occupancy prediction, with Voxel-BEV Progressive Fusion (VBPF) leveraging an RWKV-based branch to enhance OoD detection via geometry-semantic fusion. Experimental results demonstrate that OccOoD achieves state-of-the-art OoD detection with an AuROC of 67.34% and an AuPRCr of 29.21% within a 1.2m region, while maintaining competitive occupancy prediction performance. The established datasets and source code will be made publicly available at https://github.com/7uHeng/OccOoD. |
| 2025-06-26 | [Curriculum-Guided Antifragile Reinforcement Learning for Secure UAV Deconfliction under Observation-Space Attacks](http://arxiv.org/abs/2506.21129v1) | Deepak Kumar Panda, Adolfo Perrusquia et al. | Reinforcement learning (RL) policies deployed in safety-critical systems, such as unmanned aerial vehicle (UAV) navigation in dynamic airspace, are vulnerable to out-ofdistribution (OOD) adversarial attacks in the observation space. These attacks induce distributional shifts that significantly degrade value estimation, leading to unsafe or suboptimal decision making rendering the existing policy fragile. To address this vulnerability, we propose an antifragile RL framework designed to adapt against curriculum of incremental adversarial perturbations. The framework introduces a simulated attacker which incrementally increases the strength of observation-space perturbations which enables the RL agent to adapt and generalize across a wider range of OOD observations and anticipate previously unseen attacks. We begin with a theoretical characterization of fragility, formally defining catastrophic forgetting as a monotonic divergence in value function distributions with increasing perturbation strength. Building on this, we define antifragility as the boundedness of such value shifts and derive adaptation conditions under which forgetting is stabilized. Our method enforces these bounds through iterative expert-guided critic alignment using Wasserstein distance minimization across incrementally perturbed observations. We empirically evaluate the approach in a UAV deconfliction scenario involving dynamic 3D obstacles. Results show that the antifragile policy consistently outperforms standard and robust RL baselines when subjected to both projected gradient descent (PGD) and GPS spoofing attacks, achieving up to 15% higher cumulative reward and over 30% fewer conflict events. These findings demonstrate the practical and theoretical viability of antifragile reinforcement learning for secure and resilient decision-making in environments with evolving threat scenarios. |
| 2025-06-26 | [V2X-REALM: Vision-Language Model-Based Robust End-to-End Cooperative Autonomous Driving with Adaptive Long-Tail Modeling](http://arxiv.org/abs/2506.21041v1) | Junwei You, Pei Li et al. | Ensuring robust planning and decision-making under rare, diverse, and visually degraded long-tail scenarios remains a fundamental challenge for autonomous driving in urban environments. This issue becomes more critical in cooperative settings, where vehicles and infrastructure jointly perceive and reason across complex environments. To address this challenge, we propose V2X-REALM, a vision-language model (VLM)-based framework with adaptive multimodal learning for robust cooperative autonomous driving under long-tail scenarios. V2X-REALM introduces three core innovations: (i) a prompt-driven long-tail scenario generation and evaluation pipeline that leverages foundation models to synthesize realistic long-tail conditions such as snow and fog across vehicle- and infrastructure-side views, enriching training diversity efficiently; (ii) a gated multi-scenario adaptive attention module that modulates the visual stream using scenario priors to recalibrate ambiguous or corrupted features; and (iii) a multi-task scenario-aware contrastive learning objective that improves multimodal alignment and promotes cross-scenario feature separability. Extensive experiments demonstrate that V2X-REALM significantly outperforms existing baselines in robustness, semantic reasoning, safety, and planning accuracy under complex, challenging driving conditions, advancing the scalability of end-to-end cooperative autonomous driving. |
| 2025-06-26 | [VisionGuard: Synergistic Framework for Helmet Violation Detection](http://arxiv.org/abs/2506.21005v1) | Lam-Huy Nguyen, Thinh-Phuc Nguyen et al. | Enforcing helmet regulations among motorcyclists is essential for enhancing road safety and ensuring the effectiveness of traffic management systems. However, automatic detection of helmet violations faces significant challenges due to environmental variability, camera angles, and inconsistencies in the data. These factors hinder reliable detection of motorcycles and riders and disrupt consistent object classification. To address these challenges, we propose VisionGuard, a synergistic multi-stage framework designed to overcome the limitations of frame-wise detectors, especially in scenarios with class imbalance and inconsistent annotations. VisionGuard integrates two key components: Adaptive Labeling and Contextual Expander modules. The Adaptive Labeling module is a tracking-based refinement technique that enhances classification consistency by leveraging a tracking algorithm to assign persistent labels across frames and correct misclassifications. The Contextual Expander module improves recall for underrepresented classes by generating virtual bounding boxes with appropriate confidence scores, effectively addressing the impact of data imbalance. Experimental results show that VisionGuard improves overall mAP by 3.1% compared to baseline detectors, demonstrating its effectiveness and potential for real-world deployment in traffic surveillance systems, ultimately promoting safety and regulatory compliance. |
| 2025-06-26 | [Optimal Parameter Design for Power Electronic Converters Using a Probabilistic Learning-Based Stochastic Surrogate Model](http://arxiv.org/abs/2506.20987v1) | Akash Mahajan, Shivam Chaturvedi et al. | The selection of optimal design for power electronic converter parameters involves balancing efficiency and thermal constraints to ensure high performance without compromising safety. This paper introduces a probabilistic-learning-based stochastic surrogate modeling framework to address this challenge and significantly reduce the time required during the design phase. The approach begins with a neural network classifier that evaluates the feasibility of parameter configurations, effectively filtering out unsafe and/or impractical inputs. Subsequently, a probabilistic prediction model estimates the converter's efficiency and temperature while quantifying prediction uncertainty, providing both performance insights and reliability metrics. Finally, a heuristic optimization-based model is employed to optimize a multi-objective function that maximizes efficiency while adhering to thermal constraints. The optimization process incorporates penalty terms to discourage solutions that violate practical thresholds, ensuring actionable and realistic recommendations. An advanced heuristic optimization method is used to find the optimal solution and is compared with several well-known search algorithms, including Genetic Algorithm (GA), Particle Swarm Optimization (PSO), Simulated Annealing (SA), Tabu-Search (TS), and Stochastic Hill Climbing (SHC). The results demonstrate significant improvements in predictive accuracy and optimization outcomes, offering a robust solution for advancing power electronics design. |
| 2025-06-26 | [Beyond Reactive Safety: Risk-Aware LLM Alignment via Long-Horizon Simulation](http://arxiv.org/abs/2506.20949v1) | Chenkai Sun, Denghui Zhang et al. | Given the growing influence of language model-based agents on high-stakes societal decisions, from public policy to healthcare, ensuring their beneficial impact requires understanding the far-reaching implications of their suggestions. We propose a proof-of-concept framework that projects how model-generated advice could propagate through societal systems on a macroscopic scale over time, enabling more robust alignment. To assess the long-term safety awareness of language models, we also introduce a dataset of 100 indirect harm scenarios, testing models' ability to foresee adverse, non-obvious outcomes from seemingly harmless user prompts. Our approach achieves not only over 20% improvement on the new dataset but also an average win rate exceeding 70% against strong baselines on existing safety benchmarks (AdvBench, SafeRLHF, WildGuardMix), suggesting a promising direction for safer agents. |
| 2025-06-26 | [LLM-guided Chemical Process Optimization with a Multi-Agent Approach](http://arxiv.org/abs/2506.20921v1) | Tong Zeng, Srivathsan Badrinarayanan et al. | Chemical process optimization is crucial to maximize production efficiency and economic performance. Traditional methods, including gradient-based solvers, evolutionary algorithms, and parameter grid searches, become impractical when operating constraints are ill-defined or unavailable, requiring engineers to rely on subjective heuristics to estimate feasible parameter ranges. To address this constraint definition bottleneck, we present a multi-agent framework of large language model (LLM) agents that autonomously infer operating constraints from minimal process descriptions, then collaboratively guide optimization using the inferred constraints. Our AutoGen-based agentic framework employs OpenAI's o3 model, with specialized agents for constraint generation, parameter validation, simulation execution, and optimization guidance. Through two phases - autonomous constraint generation using embedded domain knowledge, followed by iterative multi-agent optimization - the framework eliminates the need for predefined operational bounds. Validated on the hydrodealkylation process across cost, yield, and yield-to-cost ratio metrics, the framework demonstrated competitive performance with conventional optimization methods while achieving better computational efficiency, requiring fewer iterations to converge. Our approach converged in under 20 minutes, achieving a 31-fold speedup over grid search. Beyond computational efficiency, the framework's reasoning-guided search demonstrates sophisticated process understanding, correctly identifying utility trade-offs, and applying domain-informed heuristics. This approach shows significant potential for optimization scenarios where operational constraints are poorly characterized or unavailable, particularly for emerging processes and retrofit applications. |
| 2025-06-25 | [Generating Reliable Adverse event Profiles for Health through Automated Integrated Data (GRAPH-AID): A Semi-Automated Ontology Building Approach](http://arxiv.org/abs/2506.20851v1) | Srikar Reddy Gadusu, Larry Callahan et al. | As data and knowledge expand rapidly, adopting systematic methodologies for ontology generation has become crucial. With the daily increases in data volumes and frequent content changes, the demand for databases to store and retrieve information for the creation of knowledge graphs has become increasingly urgent. The previously established Knowledge Acquisition and Representation Methodology (KNARM) outlines a systematic approach to address these challenges and create knowledge graphs. However, following this methodology highlights the existing challenge of seamlessly integrating Neo4j databases with the Web Ontology Language (OWL). Previous attempts to integrate data from Neo4j into an ontology have been discussed, but these approaches often require an understanding of description logics (DL) syntax, which may not be familiar to many users. Thus, a more accessible method is necessary to bridge this gap. This paper presents a user-friendly approach that utilizes Python and its rdflib library to support ontology development. We showcase our novel approach through a Neo4j database we created by integrating data from the Food and Drug Administration (FDA) Adverse Event Reporting System (FAERS) database. Using this dataset, we developed a Python script that automatically generates the required classes and their axioms, facilitating a smoother integration process. This approach offers a practical solution to the challenges of ontology generation in the context of rapidly growing adverse drug event datasets, supporting improved drug safety monitoring and public health decision-making. |
| 2025-06-25 | [A Survey of AI for Materials Science: Foundation Models, LLM Agents, Datasets, and Tools](http://arxiv.org/abs/2506.20743v1) | Minh-Hao Van, Prateek Verma et al. | Foundation models (FMs) are catalyzing a transformative shift in materials science (MatSci) by enabling scalable, general-purpose, and multimodal AI systems for scientific discovery. Unlike traditional machine learning models, which are typically narrow in scope and require task-specific engineering, FMs offer cross-domain generalization and exhibit emergent capabilities. Their versatility is especially well-suited to materials science, where research challenges span diverse data types and scales. This survey provides a comprehensive overview of foundation models, agentic systems, datasets, and computational tools supporting this growing field. We introduce a task-driven taxonomy encompassing six broad application areas: data extraction, interpretation and Q\&A; atomistic simulation; property prediction; materials structure, design and discovery; process planning, discovery, and optimization; and multiscale modeling. We discuss recent advances in both unimodal and multimodal FMs, as well as emerging large language model (LLM) agents. Furthermore, we review standardized datasets, open-source tools, and autonomous experimental platforms that collectively fuel the development and integration of FMs into research workflows. We assess the early successes of foundation models and identify persistent limitations, including challenges in generalizability, interpretability, data imbalance, safety concerns, and limited multimodal fusion. Finally, we articulate future research directions centered on scalable pretraining, continual learning, data governance, and trustworthiness. |
| 2025-06-25 | [The Singapore Consensus on Global AI Safety Research Priorities](http://arxiv.org/abs/2506.20702v1) | Yoshua Bengio, Tegan Maharaj et al. | Rapidly improving AI capabilities and autonomy hold significant promise of transformation, but are also driving vigorous debate on how to ensure that AI is safe, i.e., trustworthy, reliable, and secure. Building a trusted ecosystem is therefore essential -- it helps people embrace AI with confidence and gives maximal space for innovation while avoiding backlash.   The "2025 Singapore Conference on AI (SCAI): International Scientific Exchange on AI Safety" aimed to support research in this space by bringing together AI scientists across geographies to identify and synthesise research priorities in AI safety. This resulting report builds on the International AI Safety Report chaired by Yoshua Bengio and backed by 33 governments. By adopting a defence-in-depth model, this report organises AI safety research domains into three types: challenges with creating trustworthy AI systems (Development), challenges with evaluating their risks (Assessment), and challenges with monitoring and intervening after deployment (Control). |
| 2025-06-25 | [Deciphering GunType Hierarchy through Acoustic Analysis of Gunshot Recordings](http://arxiv.org/abs/2506.20609v1) | Ankit Shah, Rita Singh et al. | The escalating rates of gun-related violence and mass shootings represent a significant threat to public safety. Timely and accurate information for law enforcement agencies is crucial in mitigating these incidents. Current commercial gunshot detection systems, while effective, often come with prohibitive costs. This research explores a cost-effective alternative by leveraging acoustic analysis of gunshot recordings, potentially obtainable from ubiquitous devices like cell phones, to not only detect gunshots but also classify the type of firearm used. This paper details a study on deciphering gun type hierarchies using a curated dataset of 3459 recordings. We investigate the fundamental acoustic characteristics of gunshots, including muzzle blasts and shockwaves, which vary based on firearm type, ammunition, and shooting direction. We propose and evaluate machine learning frameworks, including Support Vector Machines (SVMs) as a baseline and a more advanced Convolutional Neural Network (CNN) architecture for joint gunshot detection and gun type classification. Results indicate that our deep learning approach achieves a mean average precision (mAP) of 0.58 on clean labeled data, outperforming the SVM baseline (mAP 0.39). Challenges related to data quality, environmental noise, and the generalization capabilities when using noisy web-sourced data (mAP 0.35) are also discussed. The long-term vision is to develop a highly accurate, real-time system deployable on common recording devices, significantly reducing detection costs and providing critical intelligence to first responders. |
| 2025-06-25 | [Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior Toward Beneficence or Harm](http://arxiv.org/abs/2506.20606v1) | Baixiang Huang, Zhen Tan et al. | Agents based on Large Language Models (LLMs) have demonstrated strong capabilities across a wide range of tasks. However, deploying LLM-based agents in high-stakes domains comes with significant safety and ethical risks. Unethical behavior by these agents can directly result in serious real-world consequences, including physical harm and financial loss. To efficiently steer the ethical behavior of agents, we frame agent behavior steering as a model editing task, which we term Behavior Editing. Model editing is an emerging area of research that enables precise and efficient modifications to LLMs while preserving their overall capabilities. To systematically study and evaluate this approach, we introduce BehaviorBench, a multi-tier benchmark grounded in psychological moral theories. This benchmark supports both the evaluation and editing of agent behaviors across a variety of scenarios, with each tier introducing more complex and ambiguous scenarios. We first demonstrate that Behavior Editing can dynamically steer agents toward the target behavior within specific scenarios. Moreover, Behavior Editing enables not only scenario-specific local adjustments but also more extensive shifts in an agent's global moral alignment. We demonstrate that Behavior Editing can be used to promote ethical and benevolent behavior or, conversely, to induce harmful or malicious behavior. Through comprehensive evaluations on agents based on frontier LLMs, BehaviorBench shows the effectiveness of Behavior Editing across different models and scenarios. Our findings offer key insights into a new paradigm for steering agent behavior, highlighting both the promise and perils of Behavior Editing. |
| 2025-06-25 | [Fine-Tuning and Prompt Engineering of LLMs, for the Creation of Multi-Agent AI for Addressing Sustainable Protein Production Challenges](http://arxiv.org/abs/2506.20598v1) | Alexander D. Kalian, Jaewook Lee et al. | The global demand for sustainable protein sources has accelerated the need for intelligent tools that can rapidly process and synthesise domain-specific scientific knowledge. In this study, we present a proof-of-concept multi-agent Artificial Intelligence (AI) framework designed to support sustainable protein production research, with an initial focus on microbial protein sources. Our Retrieval-Augmented Generation (RAG)-oriented system consists of two GPT-based LLM agents: (1) a literature search agent that retrieves relevant scientific literature on microbial protein production for a specified microbial strain, and (2) an information extraction agent that processes the retrieved content to extract relevant biological and chemical information. Two parallel methodologies, fine-tuning and prompt engineering, were explored for agent optimisation. Both methods demonstrated effectiveness at improving the performance of the information extraction agent in terms of transformer-based cosine similarity scores between obtained and ideal outputs. Mean cosine similarity scores were increased by up to 25%, while universally reaching mean scores of $\geq 0.89$ against ideal output text. Fine-tuning overall improved the mean scores to a greater extent (consistently of $\geq 0.94$) compared to prompt engineering, although lower statistical uncertainties were observed with the latter approach. A user interface was developed and published for enabling the use of the multi-agent AI system, alongside preliminary exploration of additional chemical safety-based search capabilities |

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



