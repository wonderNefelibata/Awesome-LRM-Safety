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
| 2025-04-25 | [Examining the Impact of Optical Aberrations to Image Classification and Object Detection Models](http://arxiv.org/abs/2504.18510v1) | Patrick Müller, Alexander Braun et al. | Deep neural networks (DNNs) have proven to be successful in various computer vision applications such that models even infer in safety-critical situations. Therefore, vision models have to behave in a robust way to disturbances such as noise or blur. While seminal benchmarks exist to evaluate model robustness to diverse corruptions, blur is often approximated in an overly simplistic way to model defocus, while ignoring the different blur kernel shapes that result from optical systems. To study model robustness against realistic optical blur effects, this paper proposes two datasets of blur corruptions, which we denote OpticsBench and LensCorruptions. OpticsBench examines primary aberrations such as coma, defocus, and astigmatism, i.e. aberrations that can be represented by varying a single parameter of Zernike polynomials. To go beyond the principled but synthetic setting of primary aberrations, LensCorruptions samples linear combinations in the vector space spanned by Zernike polynomials, corresponding to 100 real lenses. Evaluations for image classification and object detection on ImageNet and MSCOCO show that for a variety of different pre-trained models, the performance on OpticsBench and LensCorruptions varies significantly, indicating the need to consider realistic image corruptions to evaluate a model's robustness against blur. |
| 2025-04-25 | [PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts](http://arxiv.org/abs/2504.18428v1) | Yiming Wang, Pei Zhang et al. | In this paper, we introduce PolyMath, a multilingual mathematical reasoning benchmark covering 18 languages and 4 easy-to-hard difficulty levels. Our benchmark ensures difficulty comprehensiveness, language diversity, and high-quality translation, making it a highly discriminative multilingual mathematical benchmark in the era of reasoning LLMs. We conduct a comprehensive evaluation for advanced LLMs and find that even Deepseek-R1-671B and Qwen-QwQ-32B, achieve only 43.4 and 41.8 benchmark scores, with less than 30% accuracy under the highest level. From a language perspective, our benchmark reveals several key challenges of LLMs in multilingual reasoning: (1) Reasoning performance varies widely across languages for current LLMs; (2) Input-output language consistency is low in reasoning LLMs and may be correlated with performance; (3) The thinking length differs significantly by language for current LLMs. Additionally, we demonstrate that controlling the output language in the instructions has the potential to affect reasoning performance, especially for some low-resource languages, suggesting a promising direction for improving multilingual capabilities in LLMs. |
| 2025-04-25 | [Expressing stigma and inappropriate responses prevents LLMs from safely replacing mental health providers](http://arxiv.org/abs/2504.18412v1) | Jared Moore, Declan Grabb et al. | Should a large language model (LLM) be used as a therapist? In this paper, we investigate the use of LLMs to *replace* mental health providers, a use case promoted in the tech startup and research space. We conduct a mapping review of therapy guides used by major medical institutions to identify crucial aspects of therapeutic relationships, such as the importance of a therapeutic alliance between therapist and client. We then assess the ability of LLMs to reproduce and adhere to these aspects of therapeutic relationships by conducting several experiments investigating the responses of current LLMs, such as `gpt-4o`. Contrary to best practices in the medical community, LLMs 1) express stigma toward those with mental health conditions and 2) respond inappropriately to certain common (and critical) conditions in naturalistic therapy settings -- e.g., LLMs encourage clients' delusional thinking, likely due to their sycophancy. This occurs even with larger and newer LLMs, indicating that current safety practices may not address these gaps. Furthermore, we note foundational and practical barriers to the adoption of LLMs as therapists, such as that a therapeutic alliance requires human characteristics (e.g., identity and stakes). For these reasons, we conclude that LLMs should not replace therapists, and we discuss alternative roles for LLMs in clinical therapy. |
| 2025-04-25 | [Interpretable Affordance Detection on 3D Point Clouds with Probabilistic Prototypes](http://arxiv.org/abs/2504.18355v1) | Maximilian Xiling Li, Korbinian Rudolf et al. | Robotic agents need to understand how to interact with objects in their environment, both autonomously and during human-robot interactions. Affordance detection on 3D point clouds, which identifies object regions that allow specific interactions, has traditionally relied on deep learning models like PointNet++, DGCNN, or PointTransformerV3. However, these models operate as black boxes, offering no insight into their decision-making processes. Prototypical Learning methods, such as ProtoPNet, provide an interpretable alternative to black-box models by employing a "this looks like that" case-based reasoning approach. However, they have been primarily applied to image-based tasks. In this work, we apply prototypical learning to models for affordance detection on 3D point clouds. Experiments on the 3D-AffordanceNet benchmark dataset show that prototypical models achieve competitive performance with state-of-the-art black-box models and offer inherent interpretability. This makes prototypical models a promising candidate for human-robot interaction scenarios that require increased trust and safety. |
| 2025-04-25 | [Unifying Direct and Indirect Learning for Safe Control of Linear Systems](http://arxiv.org/abs/2504.18331v1) | Amir Modares, Niyousha Ghiasi et al. | This paper aims to learn safe controllers for uncertain discrete-time linear systems under disturbances while achieving the following two crucial goals: 1) integration of different sources of information (i.e., prior information in terms of physical knowledge and posterior information in terms of streaming data), and 2) unifying direct learning with indirect learning. These goals are achieved by representing a parametrized data-driven constrained matrix zonotope form of closed-loop systems that is conformant to prior knowledge. To this end, we first leverage collected data to characterize closed-loop systems by a matrix zonotope and then show that the explainability of these closed-loop systems by prior knowledge can be formalized by adding an equality conformity constraint, which refines the matrix zonotope obtained by data to a constrained matrix zonotope. The prior knowledge is further refined by conforming it to the set of models obtained from a novel zonotope-based system identifier. The source of data used for zonotope-based system identification can be different than the one used for closed-loop representation, allowing to perform transfer learning and online adaptation to new data. The parametrized closed-loop set of systems is then leveraged to directly learn a controller that robustly imposes safety on the closed-loop system. We consider both polytope and zonotope safe sets and provide set inclusion conditions using linear programming to impose safety through {\lambda}-contractivity. For polytope safe sets, a primal-dual optimization is developed to formalize a linear programming optimization that certifies the set inclusion. For zonotope safe sets, the constrained zonotope set of all next states is formed, and set inclusion is achieved by ensuring the inclusion of this constrained zonotope in a {\lambda}-scaled level set of the safe set. |
| 2025-04-25 | [AI Safety Assurance for Automated Vehicles: A Survey on Research, Standardization, Regulation](http://arxiv.org/abs/2504.18328v1) | Lars Ullrich, Michael Buchholz et al. | Assuring safety of artificial intelligence (AI) applied to safety-critical systems is of paramount importance. Especially since research in the field of automated driving shows that AI is able to outperform classical approaches, to handle higher complexities, and to reach new levels of autonomy. At the same time, the safety assurance required for the use of AI in such safety-critical systems is still not in place. Due to the dynamic and far-reaching nature of the technology, research on safeguarding AI is being conducted in parallel to AI standardization and regulation. The parallel progress necessitates simultaneous consideration in order to carry out targeted research and development of AI systems in the context of automated driving. Therefore, in contrast to existing surveys that focus primarily on research aspects, this paper considers research, standardization and regulation in a concise way. Accordingly, the survey takes into account the interdependencies arising from the triplet of research, standardization and regulation in a forward-looking perspective and anticipates and discusses open questions and possible future directions. In this way, the survey ultimately serves to provide researchers and safety experts with a compact, holistic perspective that discusses the current status, emerging trends, and possible future developments. |
| 2025-04-25 | [A comprehensive review of classifier probability calibration metrics](http://arxiv.org/abs/2504.18278v1) | Richard Oliver Lane | Probabilities or confidence values produced by artificial intelligence (AI) and machine learning (ML) models often do not reflect their true accuracy, with some models being under or over confident in their predictions. For example, if a model is 80% sure of an outcome, is it correct 80% of the time? Probability calibration metrics measure the discrepancy between confidence and accuracy, providing an independent assessment of model calibration performance that complements traditional accuracy metrics. Understanding calibration is important when the outputs of multiple systems are combined, for assurance in safety or business-critical contexts, and for building user trust in models. This paper provides a comprehensive review of probability calibration metrics for classifier and object detection models, organising them according to a number of different categorisations to highlight their relationships. We identify 82 major metrics, which can be grouped into four classifier families (point-based, bin-based, kernel or curve-based, and cumulative) and an object detection family. For each metric, we provide equations where available, facilitating implementation and comparison by future researchers. |
| 2025-04-25 | [Depth-Constrained ASV Navigation with Deep RL and Limited Sensing](http://arxiv.org/abs/2504.18253v1) | Amirhossein Zhalehmehrabi, Daniele Meli et al. | Autonomous Surface Vehicles (ASVs) play a crucial role in maritime operations, yet their navigation in shallow-water environments remains challenging due to dynamic disturbances and depth constraints. Traditional navigation strategies struggle with limited sensor information, making safe and efficient operation difficult. In this paper, we propose a reinforcement learning (RL) framework for ASV navigation under depth constraints, where the vehicle must reach a target while avoiding unsafe areas with only a single depth measurement per timestep from a downward-facing Single Beam Echosounder (SBES). To enhance environmental awareness, we integrate Gaussian Process (GP) regression into the RL framework, enabling the agent to progressively estimate a bathymetric depth map from sparse sonar readings. This approach improves decision-making by providing a richer representation of the environment. Furthermore, we demonstrate effective sim-to-real transfer, ensuring that trained policies generalize well to real-world aquatic conditions. Experimental results validate our method's capability to improve ASV navigation performance while maintaining safety in challenging shallow-water environments. |
| 2025-04-25 | [Learning to fuse: dynamic integration of multi-source data for accurate battery lifespan prediction](http://arxiv.org/abs/2504.18230v1) | He Shanxuan, Lin Zuhong et al. | Accurate prediction of lithium-ion battery lifespan is vital for ensuring operational reliability and reducing maintenance costs in applications like electric vehicles and smart grids. This study presents a hybrid learning framework for precise battery lifespan prediction, integrating dynamic multi-source data fusion with a stacked ensemble (SE) modeling approach. By leveraging heterogeneous datasets from the National Aeronautics and Space Administration (NASA), Center for Advanced Life Cycle Engineering (CALCE), MIT-Stanford-Toyota Research Institute (TRC), and nickel cobalt aluminum (NCA) chemistries, an entropy-based dynamic weighting mechanism mitigates variability across heterogeneous datasets. The SE model combines Ridge regression, long short-term memory (LSTM) networks, and eXtreme Gradient Boosting (XGBoost), effectively capturing temporal dependencies and nonlinear degradation patterns. It achieves a mean absolute error (MAE) of 0.0058, root mean square error (RMSE) of 0.0092, and coefficient of determination (R2) of 0.9839, outperforming established baseline models with a 46.2% improvement in R2 and an 83.2% reduction in RMSE. Shapley additive explanations (SHAP) analysis identifies differential discharge capacity (Qdlin) and temperature of measurement (Temp_m) as critical aging indicators. This scalable, interpretable framework enhances battery health management, supporting optimized maintenance and safety across diverse energy storage systems, thereby contributing to improved battery health management in energy storage systems. |
| 2025-04-25 | [Reimagining Assistive Walkers: An Exploration of Challenges and Preferences in Older Adults](http://arxiv.org/abs/2504.18169v1) | Victory A. Aruona, Sergio D. Sierra M. et al. | The well-being of older adults relies significantly on maintaining balance and mobility. As physical ability declines, older adults often accept the need for assistive devices. However, existing walkers frequently fail to consider user preferences, leading to perceptions of imposition and reduced acceptance. This research explores the challenges faced by older adults, caregivers, and healthcare professionals when using walkers, assesses their perceptions, and identifies their needs and preferences. A holistic approach was employed, using tailored perception questionnaires for older adults (24 participants), caregivers (30 participants), and healthcare professionals (27 participants), all of whom completed the survey. Over 50% of caregivers and healthcare professionals displayed good knowledge, positive attitudes, and effective practices regarding walkers. However, over 30% of participants perceived current designs as fall risks, citing the need for significant upper body strength, potentially affecting safety and movement. More than 50% highlighted the importance of incorporating fall detection, ergonomic designs, noise reduction, and walker ramps to better meet user needs and preferences. |
| 2025-04-25 | [Combating the Bucket Effect:Multi-Knowledge Alignment for Medication Recommendation](http://arxiv.org/abs/2504.18096v1) | Xiang Li, Haixu Ma et al. | Medication recommendation is crucial in healthcare, offering effective treatments based on patient's electronic health records (EHR). Previous studies show that integrating more medication-related knowledge improves medication representation accuracy. However, not all medications encompass multiple types of knowledge data simultaneously. For instance, some medications provide only textual descriptions without structured data. This imbalance in data availability limits the performance of existing models, a challenge we term the "bucket effect" in medication recommendation. Our data analysis uncovers the severity of the "bucket effect" in medication recommendation. To fill this gap, we introduce a cross-modal medication encoder capable of seamlessly aligning data from different modalities and propose a medication recommendation framework to integrate Multiple types of Knowledge, named MKMed. Specifically, we first pre-train a cross-modal encoder with contrastive learning on five knowledge modalities, aligning them into a unified space. Then, we combine the multi-knowledge medication representations with patient records for recommendations. Extensive experiments on the MIMIC-III and MIMIC-IV datasets demonstrate that MKMed mitigates the "bucket effect" in data, and significantly outperforms state-of-the-art baselines in recommendation accuracy and safety. |
| 2025-04-25 | [DREAM: Disentangling Risks to Enhance Safety Alignment in Multimodal Large Language Models](http://arxiv.org/abs/2504.18053v1) | Jianyu Liu, Hangyu Guo et al. | Multimodal Large Language Models (MLLMs) pose unique safety challenges due to their integration of visual and textual data, thereby introducing new dimensions of potential attacks and complex risk combinations. In this paper, we begin with a detailed analysis aimed at disentangling risks through step-by-step reasoning within multimodal inputs. We find that systematic multimodal risk disentanglement substantially enhances the risk awareness of MLLMs. Via leveraging the strong discriminative abilities of multimodal risk disentanglement, we further introduce \textbf{DREAM} (\textit{\textbf{D}isentangling \textbf{R}isks to \textbf{E}nhance Safety \textbf{A}lignment in \textbf{M}LLMs}), a novel approach that enhances safety alignment in MLLMs through supervised fine-tuning and iterative Reinforcement Learning from AI Feedback (RLAIF). Experimental results show that DREAM significantly boosts safety during both inference and training phases without compromising performance on normal tasks (namely oversafety), achieving a 16.17\% improvement in the SIUO safe\&effective score compared to GPT-4V. The data and code are available at https://github.com/Kizna1ver/DREAM. |
| 2025-04-25 | [RAG LLMs are Not Safer: A Safety Analysis of Retrieval-Augmented Generation for Large Language Models](http://arxiv.org/abs/2504.18041v1) | Bang An, Shiyue Zhang et al. | Efforts to ensure the safety of large language models (LLMs) include safety fine-tuning, evaluation, and red teaming. However, despite the widespread use of the Retrieval-Augmented Generation (RAG) framework, AI safety work focuses on standard LLMs, which means we know little about how RAG use cases change a model's safety profile. We conduct a detailed comparative analysis of RAG and non-RAG frameworks with eleven LLMs. We find that RAG can make models less safe and change their safety profile. We explore the causes of this change and find that even combinations of safe models with safe documents can cause unsafe generations. In addition, we evaluate some existing red teaming methods for RAG settings and show that they are less effective than when used for non-RAG settings. Our work highlights the need for safety research and red-teaming methods specifically tailored for RAG LLMs. |
| 2025-04-25 | [Joint Resource Estimation and Trajectory Optimization for eVTOL-involved CR network: A Monte Carlo Tree Search-based Approach](http://arxiv.org/abs/2504.18031v1) | Kai Xiong, Chenxin Yang et al. | Electric Vertical Take-Off and Landing (eVTOL) aircraft, pivotal to Advanced Air Mobility (AAM), are emerging as a transformative transportation paradigm with the potential to redefine urban and regional mobility. While these systems offer unprecedented efficiency in transporting people and goods, they rely heavily on computation capability, safety-critical operations such as real-time navigation, environmental sensing, and trajectory tracking--necessitating robust offboard computational support. A widely adopted solution involves offloading these tasks to terrestrial base stations (BSs) along the flight path. However, air-to-ground connectivity is often constrained by spectrum conflicts with terrestrial users, which poses a significant challenge to maintaining reliable task execution. Cognitive radio (CR) techniques offer promising capabilities for dynamic spectrum access, making them a natural fit for addressing this issue. Existing studies often overlook the time-varying nature of BS resources, such as spectrum availability and CPU cycles, which leads to inaccurate trajectory planning, suboptimal offloading success rates, excessive energy consumption, and operational delays. To address these challenges, we propose a trajectory optimization framework for eVTOL swarms that maximizes task offloading success probability while minimizing both energy consumption and resource competition (e.g., spectrum and CPU cycles) with primary terrestrial users. The proposed algorithm integrates a Multi-Armed Bandit (MAB) model to dynamically estimate BS resource availability and a Monte Carlo Tree Search (MCTS) algorithm to determine optimal offloading decisions, selecting both the BSs and access time windows that align with energy and temporal constraints. |
| 2025-04-24 | [Virtual Roads, Smarter Safety: A Digital Twin Framework for Mixed Autonomous Traffic Safety Analysis](http://arxiv.org/abs/2504.17968v1) | Hao Zhang, Ximin Yue et al. | This paper presents a digital-twin platform for active safety analysis in mixed traffic environments. The platform is built using a multi-modal data-enabled traffic environment constructed from drone-based aerial LiDAR, OpenStreetMap, and vehicle sensor data (e.g., GPS and inclinometer readings). High-resolution 3D road geometries are generated through AI-powered semantic segmentation and georeferencing of aerial LiDAR data. To simulate real-world driving scenarios, the platform integrates the CAR Learning to Act (CARLA) simulator, Simulation of Urban MObility (SUMO) traffic model, and NVIDIA PhysX vehicle dynamics engine. CARLA provides detailed micro-level sensor and perception data, while SUMO manages macro-level traffic flow. NVIDIA PhysX enables accurate modeling of vehicle behaviors under diverse conditions, accounting for mass distribution, tire friction, and center of mass. This integrated system supports high-fidelity simulations that capture the complex interactions between autonomous and conventional vehicles. Experimental results demonstrate the platform's ability to reproduce realistic vehicle dynamics and traffic scenarios, enhancing the analysis of active safety measures. Overall, the proposed framework advances traffic safety research by enabling in-depth, physics-informed evaluation of vehicle behavior in dynamic and heterogeneous traffic environments. |
| 2025-04-24 | [LLM Agent Swarm for Hypothesis-Driven Drug Discovery](http://arxiv.org/abs/2504.17967v1) | Kevin Song, Andrew Trotter et al. | Drug discovery remains a formidable challenge: more than 90 percent of candidate molecules fail in clinical evaluation, and development costs often exceed one billion dollars per approved therapy. Disparate data streams, from genomics and transcriptomics to chemical libraries and clinical records, hinder coherent mechanistic insight and slow progress. Meanwhile, large language models excel at reasoning and tool integration but lack the modular specialization and iterative memory required for regulated, hypothesis-driven workflows. We introduce PharmaSwarm, a unified multi-agent framework that orchestrates specialized LLM "agents" to propose, validate, and refine hypotheses for novel drug targets and lead compounds. Each agent accesses dedicated functionality--automated genomic and expression analysis; a curated biomedical knowledge graph; pathway enrichment and network simulation; interpretable binding affinity prediction--while a central Evaluator LLM continuously ranks proposals by biological plausibility, novelty, in silico efficacy, and safety. A shared memory layer captures validated insights and fine-tunes underlying submodels over time, yielding a self-improving system. Deployable on low-code platforms or Kubernetes-based microservices, PharmaSwarm supports literature-driven discovery, omics-guided target identification, and market-informed repurposing. We also describe a rigorous four-tier validation pipeline spanning retrospective benchmarking, independent computational assays, experimental testing, and expert user studies to ensure transparency, reproducibility, and real-world impact. By acting as an AI copilot, PharmaSwarm can accelerate translational research and deliver high-confidence hypotheses more efficiently than traditional pipelines. |
| 2025-04-24 | [Integrating Learning-Based Manipulation and Physics-Based Locomotion for Whole-Body Badminton Robot Control](http://arxiv.org/abs/2504.17771v1) | Haochen Wang, Zhiwei Shi et al. | Learning-based methods, such as imitation learning (IL) and reinforcement learning (RL), can produce excel control policies over challenging agile robot tasks, such as sports robot. However, no existing work has harmonized learning-based policy with model-based methods to reduce training complexity and ensure the safety and stability for agile badminton robot control. In this paper, we introduce \ourmethod, a novel hybrid control system for agile badminton robots. Specifically, we propose a model-based strategy for chassis locomotion which provides a base for arm policy. We introduce a physics-informed ``IL+RL'' training framework for learning-based arm policy. In this train framework, a model-based strategy with privileged information is used to guide arm policy training during both IL and RL phases. In addition, we train the critic model during IL phase to alleviate the performance drop issue when transitioning from IL to RL. We present results on our self-engineered badminton robot, achieving 94.5% success rate against the serving machine and 90.7% success rate against human players. Our system can be easily generalized to other agile mobile manipulation tasks such as agile catching and table tennis. Our project website: https://dreamstarring.github.io/HAMLET/. |
| 2025-04-24 | [Integrating Learning-Based Manipulation and Physics-Based Locomotion for Whole-Body Badminton Robot Control](http://arxiv.org/abs/2504.17771v2) | Haochen Wang, Zhiwei Shi et al. | Learning-based methods, such as imitation learning (IL) and reinforcement learning (RL), can produce excel control policies over challenging agile robot tasks, such as sports robot. However, no existing work has harmonized learning-based policy with model-based methods to reduce training complexity and ensure the safety and stability for agile badminton robot control. In this paper, we introduce Hamlet, a novel hybrid control system for agile badminton robots. Specifically, we propose a model-based strategy for chassis locomotion which provides a base for arm policy. We introduce a physics-informed "IL+RL" training framework for learning-based arm policy. In this train framework, a model-based strategy with privileged information is used to guide arm policy training during both IL and RL phases. In addition, we train the critic model during IL phase to alleviate the performance drop issue when transitioning from IL to RL. We present results on our self-engineered badminton robot, achieving 94.5% success rate against the serving machine and 90.7% success rate against human players. Our system can be easily generalized to other agile mobile manipulation tasks such as agile catching and table tennis. Our project website: https://dreamstarring.github.io/HAMLET/. |
| 2025-04-24 | [Conformal Segmentation in Industrial Surface Defect Detection with Statistical Guarantees](http://arxiv.org/abs/2504.17721v1) | Cheng Shen, Yuewei Liu | In industrial settings, surface defects on steel can significantly compromise its service life and elevate potential safety risks. Traditional defect detection methods predominantly rely on manual inspection, which suffers from low efficiency and high costs. Although automated defect detection approaches based on Convolutional Neural Networks(e.g., Mask R-CNN) have advanced rapidly, their reliability remains challenged due to data annotation uncertainties during deep model training and overfitting issues. These limitations may lead to detection deviations when processing the given new test samples, rendering automated detection processes unreliable. To address this challenge, we first evaluate the detection model's practical performance through calibration data that satisfies the independent and identically distributed (i.i.d) condition with test data. Specifically, we define a loss function for each calibration sample to quantify detection error rates, such as the complement of recall rate and false discovery rate. Subsequently, we derive a statistically rigorous threshold based on a user-defined risk level to identify high-probability defective pixels in test images, thereby constructing prediction sets (e.g., defect regions). This methodology ensures that the expected error rate (mean error rate) on the test set remains strictly bounced by the predefined risk level. Additionally, we observe a negative correlation between the average prediction set size and the risk level on the test set, establishing a statistically rigorous metric for assessing detection model uncertainty. Furthermore, our study demonstrates robust and efficient control over the expected test set error rate across varying calibration-to-test partitioning ratios, validating the method's adaptability and operational effectiveness. |
| 2025-04-24 | [Safety in Large Reasoning Models: A Survey](http://arxiv.org/abs/2504.17704v1) | Cheng Wang, Yue Liu et al. | Large Reasoning Models (LRMs) have exhibited extraordinary prowess in tasks like mathematics and coding, leveraging their advanced reasoning capabilities. Nevertheless, as these capabilities progress, significant concerns regarding their vulnerabilities and safety have arisen, which can pose challenges to their deployment and application in real-world settings. This paper presents a comprehensive survey of LRMs, meticulously exploring and summarizing the newly emerged safety risks, attacks, and defense strategies. By organizing these elements into a detailed taxonomy, this work aims to offer a clear and structured understanding of the current safety landscape of LRMs, facilitating future research and development to enhance the security and reliability of these powerful models. |

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



