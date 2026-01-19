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
| 2026-01-16 | [Building Production-Ready Probes For Gemini](http://arxiv.org/abs/2601.11516v1) | János Kramár, Joshua Engels et al. | Frontier language model capabilities are improving rapidly. We thus need stronger mitigations against bad actors misusing increasingly powerful systems. Prior work has shown that activation probes may be a promising misuse mitigation technique, but we identify a key remaining challenge: probes fail to generalize under important production distribution shifts. In particular, we find that the shift from short-context to long-context inputs is difficult for existing probe architectures. We propose several new probe architecture that handle this long-context distribution shift.   We evaluate these probes in the cyber-offensive domain, testing their robustness against various production-relevant shifts, including multi-turn conversations, static jailbreaks, and adaptive red teaming. Our results demonstrate that while multimax addresses context length, a combination of architecture choice and training on diverse distributions is required for broad generalization. Additionally, we show that pairing probes with prompted classifiers achieves optimal accuracy at a low cost due to the computational efficiency of probes.   These findings have informed the successful deployment of misuse mitigation probes in user-facing instances of Gemini, Google's frontier language model. Finally, we find early positive results using AlphaEvolve to automate improvements in both probe architecture search and adaptive red teaming, showing that automating some AI safety research is already possible. |
| 2026-01-16 | [Applying Formal Methods Tools to an Electronic Warfare Codebase (Experience report)](http://arxiv.org/abs/2601.11510v1) | Letitia W. Li, Denley Lam et al. | While using formal methods offers advantages over unit testing, their steep learning curve can be daunting to developers and can be a major impediment to widespread adoption. To support integration into an industrial software engineering workflow, a tool must provide useful information and must be usable with relatively minimal user effort. In this paper, we discuss our experiences associated with identifying and applying formal methods tools on an electronic warfare (EW) system with stringent safety requirements and present perspectives on formal methods tools from EW software engineers who are proficient in development yet lack formal methods training. In addition to a difference in mindset between formal methods and unit testing approaches, some formal methods tools use terminology or annotations that differ from their target programming language, creating another barrier to adoption. Input/output contracts, objects in memory affected by a function, and loop invariants can be difficult to grasp and use. In addition to usability, our findings include a comparison of vulnerabilities detected by different tools. Finally, we present suggestions for improving formal methods usability including better documentation of capabilities, decreased manual effort, and improved handling of library code. |
| 2026-01-16 | [Hierarchical Orthogonal Residual Spread for Precise Massive Editing in Large Language Models](http://arxiv.org/abs/2601.11441v1) | Xiaojie Gu, Guangxu Chen et al. | Large language models (LLMs) exhibit exceptional performance across various domains, yet they face critical safety concerns. Model editing has emerged as an effective approach to mitigate these issues. Existing model editing methods often focus on optimizing an information matrix that blends new and old knowledge. While effective, these approaches can be computationally expensive and may cause conflicts. In contrast, we shift our attention to Hierarchical Orthogonal Residual SprEad of the information matrix, which reduces noisy gradients and enables more stable edits from a different perspective. We demonstrate the effectiveness of our method HORSE through a clear theoretical comparison with several popular methods and extensive experiments conducted on two datasets across multiple LLMs. The results show that HORSE maintains precise massive editing across diverse scenarios. The code is available at https://github.com/XiaojieGu/HORSE |
| 2026-01-16 | [Learning-Based Shrinking Disturbance-Invariant Tubes for State- and Input-Dependent Uncertainty](http://arxiv.org/abs/2601.11426v1) | Abdelrahman Ramadan, Sidney Givigi | We develop a learning-based framework for constructing shrinking disturbance-invariant tubes under state- and input-dependent uncertainty, intended as a building block for tube Model Predictive Control (MPC), and certify safety via a lifted, isotone (order-preserving) fixed-point map. Gaussian Process (GP) posteriors become $(1-α)$ credible ellipsoids, then polytopic outer sets for deterministic set operations. A two-time-scale scheme separates learning epochs, where these polytopes are frozen, from an inner, outside-in iteration that converges to a compact fixed point $Z^\star\!\subseteq\!\mathcal G$; its state projection is RPI for the plant. As data accumulate, disturbance polytopes tighten, and the associated tubes nest monotonically, resolving the circular dependence between the set to be verified and the disturbance model while preserving hard constraints. A double-integrator study illustrates shrinking tube cross-sections in data-rich regions while maintaining invariance. |
| 2026-01-16 | [SME-YOLO: A Real-Time Detector for Tiny Defect Detection on PCB Surfaces](http://arxiv.org/abs/2601.11402v1) | Meng Han | Surface defects on Printed Circuit Boards (PCBs) directly compromise product reliability and safety. However, achieving high-precision detection is challenging because PCB defects are typically characterized by tiny sizes, high texture similarity, and uneven scale distributions. To address these challenges, this paper proposes a novel framework based on YOLOv11n, named SME-YOLO (Small-target Multi-scale Enhanced YOLO). First, we employ the Normalized Wasserstein Distance Loss (NWDLoss). This metric effectively mitigates the sensitivity of Intersection over Union (IoU) to positional deviations in tiny objects. Second, the original upsampling module is replaced by the Efficient Upsampling Convolution Block (EUCB). By utilizing multi-scale convolutions, the EUCB gradually recovers spatial resolution and enhances the preservation of edge and texture details for tiny defects. Finally, this paper proposes the Multi-Scale Focused Attention (MSFA) module. Tailored to the specific spatial distribution of PCB defects, this module adaptively strengthens perception within key scale intervals, achieving efficient fusion of local fine-grained features and global context information. Experimental results on the PKU-PCB dataset demonstrate that SME-YOLO achieves state-of-the-art performance. Specifically, compared to the baseline YOLOv11n, SME-YOLO improves mAP by 2.2% and Precision by 4%, validating the effectiveness of the proposed method. |
| 2026-01-16 | [Understanding Help Seeking for Digital Privacy, Safety, and Security](http://arxiv.org/abs/2601.11398v1) | Kurt Thomas, Sai Teja Peddinti et al. | The complexity of navigating digital privacy, safety, and security threats often falls directly on users. This leads to users seeking help from family and peers, platforms and advice guides, dedicated communities, and even large language models (LLMs). As a precursor to improving resources across this ecosystem, our community needs to understand what help seeking looks like in the wild. To that end, we blend qualitative coding with LLM fine-tuning to sift through over one billion Reddit posts from the last four years to identify where and for what users seek digital privacy, safety, or security help. We isolate three million relevant posts with 93% precision and recall and automatically annotate each with the topics discussed (e.g., security tools, privacy configurations, scams, account compromise, content moderation, and more). We use this dataset to understand the scope and scale of help seeking, the communities that provide help, and the types of help sought. Our work informs the development of better resources for users (e.g., user guides or LLM help-giving agents) while underscoring the inherent challenges of supporting users through complex combinations of threats, platforms, mitigations, context, and emotions. |
| 2026-01-16 | [Cutting Corners on Uncertainty: Zonotope Abstractions for Stream-based Runtime Monitoring](http://arxiv.org/abs/2601.11358v1) | Bernd Finkbeiner, Martin Fränzle et al. | Stream-based monitoring assesses the health of safety-critical systems by transforming input streams of sensor measurements into output streams that determine a verdict. These inputs are often treated as accurate representations of the physical state, although real sensors introduce calibration and measurement errors. Such errors propagate through the monitor's computations and can distort the final verdict. Affine arithmetic with symbolic slack variables can track these errors precisely, but independent measurement noise introduces a fresh slack variable upon each measurement event, causing the monitor's state representation to grow without bound over time. Therefore, any bounded-memory monitoring algorithm must unify slack variables at runtime in a way that generates a sound approximation.   This paper introduces zonotopes as an abstract domain for online monitoring of RLola specifications. We demonstrate that zonotopes precisely capture the affine state of the monitor and that their over-approximation produces a sound bounded-memory monitor. We present a comparison of different zonotope over-approximation strategies in the context of runtime monitoring, evaluating their performance and false-positive rates. |
| 2026-01-16 | [Distributed Control Barrier Functions for Safe Multi-Vehicle Navigation in Heterogeneous USV Fleets](http://arxiv.org/abs/2601.11335v1) | Tyler Paine, Brendan Long et al. | Collision avoidance in heterogeneous fleets of uncrewed vessels is challenging because the decision-making processes and controllers often differ between platforms, and it is further complicated by the limitations on sharing trajectories and control values in real-time. This paper presents a pragmatic approach that addresses these issues by adding a control filter on each autonomous vehicle that assumes worst-case behavior from other contacts, including crewed vessels. This distributed safety control filter is developed using control barrier function (CBF) theory and the application is clearly described to ensure explainability of these safety-critical methods. This work compares the worst-case CBF approach with a Collision Regulations (COLREGS) behavior-based approach in simulated encounters. Real-world experiments with three different uncrewed vessels and a human operated vessel were performed to confirm the approach is effective across a range of platforms and is robust to uncooperative behavior from human operators. Results show that combining both CBF methods and COLREGS behaviors achieves the best safety and efficiency. |
| 2026-01-16 | [TANDEM: Temporal-Aware Neural Detection for Multimodal Hate Speech](http://arxiv.org/abs/2601.11178v1) | Girish A. Koushik, Helen Treharne et al. | Social media platforms are increasingly dominated by long-form multimodal content, where harmful narratives are constructed through a complex interplay of audio, visual, and textual cues. While automated systems can flag hate speech with high accuracy, they often function as "black boxes" that fail to provide the granular, interpretable evidence, such as precise timestamps and target identities, required for effective human-in-the-loop moderation. In this work, we introduce TANDEM, a unified framework that transforms audio-visual hate detection from a binary classification task into a structured reasoning problem. Our approach employs a novel tandem reinforcement learning strategy where vision-language and audio-language models optimize each other through self-constrained cross-modal context, stabilizing reasoning over extended temporal sequences without requiring dense frame-level supervision. Experiments across three benchmark datasets demonstrate that TANDEM significantly outperforms zero-shot and context-augmented baselines, achieving 0.73 F1 in target identification on HateMM (a 30% improvement over state-of-the-art) while maintaining precise temporal grounding. We further observe that while binary detection is robust, differentiating between offensive and hateful content remains challenging in multi-class settings due to inherent label ambiguity and dataset imbalance. More broadly, our findings suggest that structured, interpretable alignment is achievable even in complex multimodal settings, offering a blueprint for the next generation of transparent and actionable online safety moderation tools. |
| 2026-01-16 | [Assesing the Viability of Unsupervised Learning with Autoencoders for Predictive Maintenance in Helicopter Engines](http://arxiv.org/abs/2601.11154v1) | P. Sánchez, K. Reyes et al. | Unplanned engine failures in helicopters can lead to severe operational disruptions, safety hazards, and costly repairs. To mitigate these risks, this study compares two predictive maintenance strategies for helicopter engines: a supervised classification pipeline and an unsupervised anomaly detection approach based on autoencoders (AEs). The supervised method relies on labelled examples of both normal and faulty behaviour, while the unsupervised approach learns a model of normal operation using only healthy engine data, flagging deviations as potential faults. Both methods are evaluated on a real-world dataset comprising labelled snapshots of helicopter engine telemetry. While supervised models demonstrate strong performance when annotated failures are available, the AE achieves effective detection without requiring fault labels, making it particularly well suited for settings where failure data are scarce or incomplete. The comparison highlights the practical trade-offs between accuracy, data availability, and deployment feasibility, and underscores the potential of unsupervised learning as a viable solution for early fault detection in aerospace applications. |
| 2026-01-16 | [Crane Lowering Guidance Using a Attachable Camera Module for Driver Vision Support](http://arxiv.org/abs/2601.11026v1) | HyoJae Kang, SunWoo Ahn et al. | Cranes have long been essential equipment for lifting and placing heavy loads in construction projects. This study focuses on the lowering phase of crane operation, the stage in which the load is moved to the desired location. During this phase, a constant challenge exists: the load obstructs the operator's view of the landing point. As a result, operators traditionally have to rely on verbal or gestural instructions from ground personnel, which significantly impacts site safety. To alleviate this constraint, the proposed system incorporates a attachable camera module designed to be attached directly to the load via a suction cup. This module houses a single-board computer, battery, and compact camera. After installation, it streams and processes images of the ground directly below the load in real time to generate installation guidance. Simultaneously, this guidance is transmitted to and monitored by a host computer. Preliminary experiments were conducted by attaching this module to a test object, confirming the feasibility of real-time image acquisition and transmission. This approach has the potential to significantly improve safety on construction sites by providing crane operators with an instant visual reference of hidden landing zones. |
| 2026-01-16 | [Study of circular cross-section plasmas in HL-2A tokamak: MHD equilibrium, stability and operational \b{eta} limit](http://arxiv.org/abs/2601.11014v1) | SHEN Yong, DONG Jiaqi et al. | Circular cross-section plasma is the most basic form of tokamak plasma and the fundamental configuration for magnetic confinement fusion experiments. Based on the HL-2A limiter discharge experiments, the magnetohydrodynamic (MHD) equilibrium and MHD instability of circular cross-section tokamak plasmas are investigated in this work. The results show that when q_0=0.95, the internal kink mode of m/n=1/1 is always unstable. The increase in plasma \b{eta} (the ratio of thermal pressure to magnetic pressure) can lead to the appearance of external kink modes. The combination of axial safety factor q_0 and edge safety factor q_a determines the equilibrium configuration of the plasma and also affects the MHD stability of the equilibrium, but its growth rate is also related to the size of \b{eta}. Under the condition of q_a>2 and q_0 slightly greater than 1, the internal kink mode and surface kink mode can be easily stabilized. However the plasma becomes unstable again and the instability intensity increases as q_0 continues to increase when q_0 exceeds 1. As the poloidal beta (\b{eta}_p) increases, the MHD instability develops, the equilibrium configuration of MHD elongates laterally, and the Shafranov displacement increases, which in turn has the effect on suppressing instability. Calculations have shown that the maximum \b{eta} value imposed by the ideal MHD mode in a plasma with free boundary in tokamak experiments is proportional to the normalized current I_N (I_N=I_p (MA)/a(m)B_0 (T)), and the achievable maximum beta \b{eta}(max) is calibrated to be 2.01I_N,i.e. \b{eta}(max)~2.01I_N. The operational \b{eta} limit of HL-2A circular cross-section plasma is approximately \b{eta}_N^c~2.0. Too high a value of q_0 is not conducive to MHD stability and leads the \b{eta} limit value to decrease. When q_0=1.3, we obtain a maximum value of \b{eta}_N of approximately 1.8. |
| 2026-01-16 | [Data-driven Prediction of Ionic Conductivity in Solid-State Electrolytes with Machine Learning and Large Language Models](http://arxiv.org/abs/2601.10997v1) | Haewon Kim, Taekgi Lee et al. | Solid-state electrolytes (SSEs) are attractive for next-generation lithium-ion batteries due to improved safety and stability but their low room-temperature ionic conductivity hinders practical application. Experimental synthesis and testing of new SSEs remain time-consuming and resource intensive. Machine learning (ML) offers an accelerated route for SSE discovery; however, composition-only models neglect structural factors important for ion transport while graph neural networks (GNNs) are challenged by the scarcity of structure-labeled conductivity data and the prevalence of crystallographic disorder in CIFs. Here, we train two complementary predictors on the same room-temperature, structure-labeled dataset (n = 499). A gradient-boosted tree regressor (GBR) combining stoichiometric and geometric descriptors achieves best performance (MAE = 0.543 in log(S cm-1)), and Shapley Additive exPlanations (SHAP) identifies probe-occupiable volume (POAV) and lattice parameters as key correlations for conductivity. In parallel, we fine-tune large language models (LLMs) using compact text prompts derived from CIF metadata (formula with optional symmetry and disorder tags), avoiding direct use of raw atomic coordinates. Notably, Llama-3.1-8B-Instruct achieves high accuracy (MAE = 0.657 in log(S cm-1)) using formula and symmetry information, eliminating the need for numerical feature extraction from CIF files. Together, these results show that global geometric descriptors improve tree-based predictions and enable interpretable structure-property analysis, while LLMs provide a competitive low-preprocessing alternative for rapid SSE screening. |
| 2026-01-16 | [Analyzing Residential Speeding Using Connected Vehicle Data: A Case Study in Charlottesville, VA Area](http://arxiv.org/abs/2601.10974v1) | Shi Feng, B. Brian Park et al. | This study uses connected vehicle data to analyze speeding behavior on residential roads. A scalable pipeline processes trajectory data and supplements missing speed limits to generate summaries at OpenStreetMap's way ID level. The findings reveal a highly skewed distribution of both aggressive and reckless speeding. Based on a case study of Charlottesville, VA's connected vehicle data on residential roads, we found that 38% of segments had at least one instance of aggressive speeding, and 20% had at least one instance of reckless speeding. In addition, night time speeding is 27 times more prevalent than day time, and extreme violations on specific road segments highlight how severe the issue can be. Several segments rank among the top 10 for both aggressive and reckless speedings, indicating that there exist high-risk residential roads. These findings support the need for both spatial and behavioral interventions. The analysis provides a rich foundation for policy and planning, offering a valuable complement to traditional enforcement and planning tools. In conclusion, this framework sets the foundation for future applications in traffic safety analytics, demonstrating the growing potential of telematics data to inform safer, more livable communities. |
| 2026-01-16 | [AJAR: Adaptive Jailbreak Architecture for Red-teaming](http://arxiv.org/abs/2601.10971v1) | Yipu Dou, Wang Yang | As Large Language Models (LLMs) evolve from static chatbots into autonomous agents capable of tool execution, the landscape of AI safety is shifting from content moderation to action security. However, existing red-teaming frameworks remain bifurcated: they either focus on rigid, script-based text attacks or lack the architectural modularity to simulate complex, multi-turn agentic exploitations. In this paper, we introduce AJAR (Adaptive Jailbreak Architecture for Red-teaming), a proof-of-concept framework designed to bridge this gap through Protocol-driven Cognitive Orchestration. Built upon the robust runtime of Petri, AJAR leverages the Model Context Protocol (MCP) to decouple adversarial logic from the execution loop, encapsulating state-of-the-art algorithms like X-Teaming as standardized, plug-and-play services. We validate the architectural feasibility of AJAR through a controlled qualitative case study, demonstrating its ability to perform stateful backtracking within a tool-use environment. Furthermore, our preliminary exploration of the "Agentic Gap" reveals a complex safety dynamic: while tool usage introduces new injection vectors via code execution, the cognitive load of parameter formatting can inadvertently disrupt persona-based attacks. AJAR is open-sourced to facilitate the standardized, environment-aware evaluation of this emerging attack surface. The code and data are available at https://github.com/douyipu/ajar. |
| 2026-01-16 | [Secure Data Bridging in Industry 4.0: An OPC UA Aggregation Approach for Including Insecure Legacy Systems](http://arxiv.org/abs/2601.10929v1) | Dalibor Sain, Thomas Rosenstatter et al. | The increased connectivity of industrial networks has led to a surge in cyberattacks, emphasizing the need for cybersecurity measures tailored to the specific requirements of industrial systems. Modern Industry 4.0 technologies, such as OPC UA, offer enhanced resilience against these threats. However, widespread adoption remains limited due to long installation times, proprietary technology, restricted flexibility, and formal process requirements (e.g. safety certifications). Consequently, many systems do not yet implement these technologies, or only partially. This leads to the challenge of dealing with so-called brownfield systems, which are often placed in isolated security zones to mitigate risks. However, the need for data exchange between secure and insecure zones persists.   This paper reviews existing solutions to address this challenge by analysing their approaches, advantages, and limitations. Building on these insights, we identify three key concepts, evaluate their suitability and compatibility, and ultimately introduce the SigmaServer, a novel TCP-level aggregation method. The developed proof-of-principle implementation is evaluated in an operational technology (OT) testbed, demonstrating its applicability and effectiveness in bridging secure and insecure zones. |
| 2026-01-15 | [Realistic Curriculum Reinforcement Learning for Autonomous and Sustainable Marine Vessel Navigation](http://arxiv.org/abs/2601.10911v1) | Zhang Xiaocai, Xiao Zhe et al. | Sustainability is becoming increasingly critical in the maritime transport, encompassing both environmental and social impacts, such as Greenhouse Gas (GHG) emissions and navigational safety. Traditional vessel navigation heavily relies on human experience, often lacking autonomy and emission awareness, and is prone to human errors that may compromise safety. In this paper, we propose a Curriculum Reinforcement Learning (CRL) framework integrated with a realistic, data-driven marine simulation environment and a machine learning-based fuel consumption prediction module. The simulation environment is constructed using real-world vessel movement data and enhanced with a Diffusion Model to simulate dynamic maritime conditions. Vessel fuel consumption is estimated using historical operational data and learning-based regression. The surrounding environment is represented as image-based inputs to capture spatial complexity. We design a lightweight, policy-based CRL agent with a comprehensive reward mechanism that considers safety, emissions, timeliness, and goal completion. This framework effectively handles complex tasks progressively while ensuring stable and efficient learning in continuous action spaces. We validate the proposed approach in a sea area of the Indian Ocean, demonstrating its efficacy in enabling sustainable and safe vessel navigation. |
| 2026-01-15 | [Can Vision-Language Models Understand Construction Workers? An Exploratory Study](http://arxiv.org/abs/2601.10835v1) | Hieu Bui, Nathaniel E. Chodosh et al. | As robotics become increasingly integrated into construction workflows, their ability to interpret and respond to human behavior will be essential for enabling safe and effective collaboration. Vision-Language Models (VLMs) have emerged as a promising tool for visual understanding tasks and offer the potential to recognize human behaviors without extensive domain-specific training. This capability makes them particularly appealing in the construction domain, where labeled data is scarce and monitoring worker actions and emotional states is critical for safety and productivity. In this study, we evaluate the performance of three leading VLMs, GPT-4o, Florence 2, and LLaVa-1.5, in detecting construction worker actions and emotions from static site images. Using a curated dataset of 1,000 images annotated across ten action and ten emotion categories, we assess each model's outputs through standardized inference pipelines and multiple evaluation metrics. GPT-4o consistently achieved the highest scores across both tasks, with an average F1-score of 0.756 and accuracy of 0.799 in action recognition, and an F1-score of 0.712 and accuracy of 0.773 in emotion recognition. Florence 2 performed moderately, with F1-scores of 0.497 for action and 0.414 for emotion, while LLaVa-1.5 showed the lowest overall performance, with F1-scores of 0.466 for action and 0.461 for emotion. Confusion matrix analyses revealed that all models struggled to distinguish semantically close categories, such as collaborating in teams versus communicating with supervisors. While the results indicate that general-purpose VLMs can offer a baseline capability for human behavior recognition in construction environments, further improvements, such as domain adaptation, temporal modeling, or multimodal sensing, may be needed for real-world reliability. |
| 2026-01-15 | [IMU-based Real-Time Crutch Gait Phase and Step Detections in Lower-Limb Exoskeletons](http://arxiv.org/abs/2601.10832v1) | Anis R. Shakkour, David Hexner et al. | Lower limb exoskeletons and prostheses require precise, real time gait phase and step detections to ensure synchronized motion and user safety. Conventional methods often rely on complex force sensing hardware that introduces control latency. This paper presents a minimalist framework utilizing a single, low cost Inertial-Measurement Unit (IMU) integrated into the crutch hand grip, eliminating the need for mechanical modifications. We propose a five phase classification system, including standard gait phases and a non locomotor auxiliary state, to prevent undesired motion. Three deep learning architectures were benchmarked on both a PC and an embedded system. To improve performance under data constrained conditions, models were augmented with a Finite State Machine (FSM) to enforce biomechanical consistency. The Temporal Convolutional Network (TCN) emerged as the superior architecture, yielding the highest success rates and lowest latency. Notably, the model generalized to a paralyzed user despite being trained exclusively on healthy participants. Achieving a 94% success rate in detecting crutch steps, this system provides a high performance, cost effective solution for real time exoskeleton control. |
| 2026-01-15 | [Reasoning Models Generate Societies of Thought](http://arxiv.org/abs/2601.10825v1) | Junsol Kim, Shiyang Lai et al. | Large language models have achieved remarkable capabilities across domains, yet mechanisms underlying sophisticated reasoning remain elusive. Recent reasoning models outperform comparable instruction-tuned models on complex cognitive tasks, attributed to extended computation through longer chains of thought. Here we show that enhanced reasoning emerges not from extended computation alone, but from simulating multi-agent-like interactions -- a society of thought -- which enables diversification and debate among internal cognitive perspectives characterized by distinct personality traits and domain expertise. Through quantitative analysis and mechanistic interpretability methods applied to reasoning traces, we find that reasoning models like DeepSeek-R1 and QwQ-32B exhibit much greater perspective diversity than instruction-tuned models, activating broader conflict between heterogeneous personality- and expertise-related features during reasoning. This multi-agent structure manifests in conversational behaviors, including question-answering, perspective shifts, and the reconciliation of conflicting views, and in socio-emotional roles that characterize sharp back-and-forth conversations, together accounting for the accuracy advantage in reasoning tasks. Controlled reinforcement learning experiments reveal that base models increase conversational behaviors when rewarded solely for reasoning accuracy, and fine-tuning models with conversational scaffolding accelerates reasoning improvement over base models. These findings indicate that the social organization of thought enables effective exploration of solution spaces. We suggest that reasoning models establish a computational parallel to collective intelligence in human groups, where diversity enables superior problem-solving when systematically structured, which suggests new opportunities for agent organization to harness the wisdom of crowds. |

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



