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
| 2026-03-05 | [Safe-SAGE: Social-Semantic Adaptive Guidance for Safe Engagement through Laplace-Modulated Poisson Safety Functions](http://arxiv.org/abs/2603.05497v1) | Lizhi Yang, Ryan M. Bena et al. | Traditional safety-critical control methods, such as control barrier functions, suffer from semantic blindness, exhibiting the same behavior around obstacles regardless of contextual significance. This limitation leads to the uniform treatment of all obstacles, despite their differing semantic meanings. We present Safe-SAGE (Social-Semantic Adaptive Guidance for Safe Engagement), a unified framework that bridges the gap between high-level semantic understanding and low-level safety-critical control through a Poisson safety function (PSF) modulated using a Laplace guidance field. Our approach perceives the environment by fusing multi-sensor point clouds with vision-based instance segmentation and persistent object tracking to maintain up-to-date semantics beyond the camera's field of view. A multi-layer safety filter is then used to modulate system inputs to achieve safe navigation using this semantic understanding of the environment. This safety filter consists of both a model predictive control layer and a control barrier function layer. Both layers utilize the PSF and flux modulation of the guidance field to introduce varying levels of conservatism and multi-agent passing norms for different obstacles in the environment. Our framework enables legged robots to navigate semantically rich, dynamic environments with context-dependent safety margins while maintaining rigorous safety guarantees. |
| 2026-03-05 | [Censored LLMs as a Natural Testbed for Secret Knowledge Elicitation](http://arxiv.org/abs/2603.05494v1) | Helena Casademunt, Bartosz Cywiński et al. | Large language models sometimes produce false or misleading responses. Two approaches to this problem are honesty elicitation -- modifying prompts or weights so that the model answers truthfully -- and lie detection -- classifying whether a given response is false. Prior work evaluates such methods on models specifically trained to lie or conceal information, but these artificial constructions may not resemble naturally-occurring dishonesty. We instead study open-weights LLMs from Chinese developers, which are trained to censor politically sensitive topics: Qwen3 models frequently produce falsehoods about subjects like Falun Gong or the Tiananmen protests while occasionally answering correctly, indicating they possess knowledge they are trained to suppress. Using this as a testbed, we evaluate a suite of elicitation and lie detection techniques. For honesty elicitation, sampling without a chat template, few-shot prompting, and fine-tuning on generic honesty data most reliably increase truthful responses. For lie detection, prompting the censored model to classify its own responses performs near an uncensored-model upper bound, and linear probes trained on unrelated data offer a cheaper alternative. The strongest honesty elicitation techniques also transfer to frontier open-weights models including DeepSeek R1. Notably, no technique fully eliminates false responses. We release all prompts, code, and transcripts. |
| 2026-03-05 | [Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought](http://arxiv.org/abs/2603.05488v1) | Siddharth Boppana, Annabel Ma et al. | We provide evidence of performative chain-of-thought (CoT) in reasoning models, where a model becomes strongly confident in its final answer, but continues generating tokens without revealing its internal belief. Our analysis compares activation probing, early forced answering, and a CoT monitor across two large models (DeepSeek-R1 671B & GPT-OSS 120B) and find task difficulty-specific differences: The model's final answer is decodable from activations far earlier in CoT than a monitor is able to say, especially for easy recall-based MMLU questions. We contrast this with genuine reasoning in difficult multihop GPQA-Diamond questions. Despite this, inflection points (e.g., backtracking, 'aha' moments) occur almost exclusively in responses where probes show large belief shifts, suggesting these behaviors track genuine uncertainty rather than learned "reasoning theater." Finally, probe-guided early exit reduces tokens by up to 80% on MMLU and 30% on GPQA-Diamond with similar accuracy, positioning attention probing as an efficient tool for detecting performative reasoning and enabling adaptive computation. |
| 2026-03-05 | [HALP: Detecting Hallucinations in Vision-Language Models without Generating a Single Token](http://arxiv.org/abs/2603.05465v1) | Sai Akhil Kogilathota, Sripadha Vallabha E G et al. | Hallucinations remain a persistent challenge for vision-language models (VLMs), which often describe nonexistent objects or fabricate facts. Existing detection methods typically operate after text generation, making intervention both costly and untimely. We investigate whether hallucination risk can instead be predicted before any token is generated by probing a model's internal representations in a single forward pass. Across a diverse set of vision-language tasks and eight modern VLMs, including Llama-3.2-Vision, Gemma-3, Phi-4-VL, and Qwen2.5-VL, we examine three families of internal representations: (i) visual-only features without multimodal fusion, (ii) vision-token representations within the text decoder, and (iii) query-token representations that integrate visual and textual information before generation. Probes trained on these representations achieve strong hallucination-detection performance without decoding, reaching up to 0.93 AUROC on Gemma-3-12B, Phi-4-VL 5.6B, and Molmo 7B. Late query-token states are the most predictive for most models, while visual or mid-layer features dominate in a few architectures (e.g., ~0.79 AUROC for Qwen2.5-VL-7B using visual-only features). These results demonstrate that (1) hallucination risk is detectable pre-generation, (2) the most informative layer and modality vary across architectures, and (3) lightweight probes have the potential to enable early abstention, selective routing, and adaptive decoding to improve both safety and efficiency. |
| 2026-03-05 | [O^3-LSM: Maximizing Disaggregated LSM Write Performance via Three-Layer Offloading](http://arxiv.org/abs/2603.05439v1) | Qi Lin, Gangqi Huang et al. | Log-Structured Merge-tree-based Key-Value Stores (LSM-KVS) have been optimized and redesigned for disaggregated storage via techniques such as compaction offloading to reduce the network I/Os between compute and storage. However, the constrained memory space and slow flush at the compute node severely limit the overall write throughput of existing optimizations. In this paper, we propose O3-LSM, a fundamental new LSM-KVS architecture, that leverages the shared Disaggregated Memory (DM) to support a three-layer offloading, i.e., memtable Offloading, flush Offloading, and the existing compaction Offloading. Compared to the existing disaggregated LSM-KVS with compaction offloading only, O3-LSM maximizes the write performance by addressing the above issues.   O3-LSM first leverages a novel DM-Optimized Memtable to achieve dynamic memtable offloading, which extends the write buffer while enabling fast, asynchronous, and parallel memtable transmission. Second, we propose Collaborative Flush Offloading that decouples the flush control plane from execution and supports memtable flush offloading at any node with dedicated scheduling and global optimizations. Third, O3-LSM is further improved with the Shard-Level Optimization, which partitions the memtable into shards based on disjoint key-ranges that can be transferred and flushed independently, unlocking parallelism across shards. Besides, to mitigate slow lookups in the disaggregated setting, O3-LSM also employs an adaptive Cache-Enhanced Read Delegation mechanism to combine a compact local cache with DM-assisted memtable delegated read. Our evaluation shows that O3-LSM achieves up to 4.5X write, 5.2X range query, and 1.8X point lookup throughput improvement, and up to 76% P99 latency reduction compared with Disaggregated-RocksDB, CaaS-LSM, and Nova-LSM. |
| 2026-03-05 | [Judge Reliability Harness: Stress Testing the Reliability of LLM Judges](http://arxiv.org/abs/2603.05399v1) | Sunishchal Dev, Andrew Sloan et al. | We present the Judge Reliability Harness, an open source library for constructing validation suites that test the reliability of LLM judges. As LLM based scoring is widely deployed in AI benchmarks, more tooling is needed to efficiently assess the reliability of these methods. Given a benchmark dataset and an LLM judge configuration, the harness generates reliability tests that evaluate both binary judgment accuracy and ordinal grading performance for free-response and agentic task formats. We evaluate four state-of-the-art judges across four benchmarks spanning safety, persuasion, misuse, and agentic behavior, and find meaningful variation in performance across models and perturbation types, highlighting opportunities to improve the robustness of LLM judges. No judge that we evaluated is uniformly reliable across benchmarks using our harness. For example, our preliminary experiments on judges revealed consistency issues as measured by accuracy in judging another LLM's ability to complete a task due to simple text formatting changes, paraphrasing, changes in verbosity, and flipping the ground truth label in LLM-produced responses. The code for this tool is available at: https://github.com/RANDCorporation/judge-reliability-harness |
| 2026-03-05 | [Omni-Manip: Beyond-FOV Large-Workspace Humanoid Manipulation with Omnidirectional 3D Perception](http://arxiv.org/abs/2603.05355v1) | Pei Qu, Zheng Li et al. | The deployment of humanoid robots for dexterous manipulation in unstructured environments remains challenging due to perceptual limitations that constrain the effective workspace. In scenarios where physical constraints prevent the robot from repositioning itself, maintaining omnidirectional awareness becomes far more critical than color or semantic information. While recent advances in visuomotor policy learning have improved manipulation capabilities, conventional RGB-D solutions suffer from narrow fields of view (FOV) and self-occlusion, requiring frequent base movements that introduce motion uncertainty and safety risks. Existing approaches to expanding perception, including active vision systems and third-view cameras, introduce mechanical complexity, calibration dependencies, and latency that hinder reliable real-time performance. In this work, We propose Omni-Manip, an end-to-end LiDAR-driven 3D visuomotor policy that enables robust manipulation in large workspaces. Our method processes panoramic point clouds through a Time-Aware Attention Pooling mechanism, efficiently encoding sparse 3D data while capturing temporal dependencies. This 360° perception allows the robot to interact with objects across wide areas without frequent repositioning. To support policy learning, we develop a whole-body teleoperation system for efficient data collection on full-body coordination. Extensive experiments in simulation and real-world environments show that Omni-Manip achieves robust performance in large-workspace and cluttered scenarios, outperforming baselines that rely on egocentric depth cameras. |
| 2026-03-05 | [Building AI Coding Agents for the Terminal: Scaffolding, Harness, Context Engineering, and Lessons Learned](http://arxiv.org/abs/2603.05344v1) | Nghi D. Q. Bui | The landscape of AI coding assistance is undergoing a fundamental shift from complex IDE plugins to versatile, terminal-native agents. Operating directly where developers manage source control, execute builds, and deploy environments, CLI-based agents offer unprecedented autonomy for long-horizon development tasks. In this paper, we present OPENDEV, an open-source, command-line coding agent engineered specifically for this new paradigm. Effective autonomous assistance requires strict safety controls and highly efficient context management to prevent context bloat and reasoning degradation. OPENDEV overcomes these challenges through a compound AI system architecture with workload-specialized model routing, a dual-agent architecture separating planning from execution, lazy tool discovery, and adaptive context compaction that progressively reduces older observations. Furthermore, it employs an automated memory system to accumulate project-specific knowledge across sessions and counteracts instruction fade-out through event-driven system reminders. By enforcing explicit reasoning phases and prioritizing context efficiency, OPENDEV provides a secure, extensible foundation for terminal-first AI assistance, offering a blueprint for robust autonomous software engineering. |
| 2026-03-05 | [CT-Enabled Patient-Specific Simulation and Contact-Aware Robotic Planning for Cochlear Implantation](http://arxiv.org/abs/2603.05333v1) | Lingxiao Xun, Gang Zheng et al. | Robotic cochlear-implant (CI) insertion requires precise prediction and regulation of contact forces to minimize intracochlear trauma and prevent failure modes such as locking and buckling. Aligned with the integration of advanced medical imaging and robotics for autonomous, precision interventions, this paper presents a unified CT-to-simulation pipeline for contact-aware insertion planning and validation. We develop a low-dimensional, differentiable Cosserat-rod model of the electrode array coupled with frictional contact and pseudo-dynamics regularization to ensure continuous stick-slip transitions. Patient-specific cochlear anatomy is reconstructed from CT imaging and encoded via an analytic parametrization of the scala-tympani lumen, enabling efficient and differentiable contact queries through closest-point projection. Based on a differentiated equilibrium-constraint formulation, we derive an online direction-update law under an RCM-like constraint that suppresses lateral insertion forces while maintaining axial advancement. Simulations and benchtop experiments validate deformation and force trends, demonstrating reduced locking/buckling risk and improved insertion depth. The study highlights how CT-based imaging enhances modeling, planning, and safety capabilities in robot-assisted inner-ear procedures. |
| 2026-03-05 | [Knowledge Divergence and the Value of Debate for Scalable Oversight](http://arxiv.org/abs/2603.05293v1) | Robin Young | AI safety via debate and reinforcement learning from AI feedback (RLAIF) are both proposed methods for scalable oversight of advanced AI systems, yet no formal framework relates them or characterizes when debate offers an advantage. We analyze this by parameterizing debate's value through the geometry of knowledge divergence between debating models. Using principal angles between models' representation subspaces, we prove that the debate advantage admits an exact closed form. When models share identical training corpora, debate reduces to RLAIF-like where a single-agent method recovers the same optimum. When models possess divergent knowledge, debate advantage scales with a phase transition from quadratic regime (debate offers negligible benefit) to linear regime (debate is essential). We classify three regimes of knowledge divergence (shared, one-sided, and compositional) and provide existence results showing that debate can achieve outcomes inaccessible to either model alone, alongside a negative result showing that sufficiently strong adversarial incentives cause coordination failure in the compositional regime, with a sharp threshold separating effective from ineffective debate. We offer the first formal connection between debate and RLAIF, a geometric foundation for understanding when adversarial oversight protocols are justified, and connection to the problem of eliciting latent knowledge across models with complementary information. |
| 2026-03-05 | [Beyond Word Error Rate: Auditing the Diversity Tax in Speech Recognition through Dataset Cartography](http://arxiv.org/abs/2603.05267v1) | Ting-Hui Cheng, Line H. Clemmensen et al. | Automatic speech recognition (ASR) systems are predominantly evaluated using the Word Error Rate (WER). However, raw token-level metrics fail to capture semantic fidelity and routinely obscures the `diversity tax', the disproportionate burden on marginalized and atypical speaker due to systematic recognition failures. In this paper, we explore the limitations of relying solely on lexical counts by systematically evaluating a broader class of non-linear and semantic metrics. To enable rigorous model auditing, we introduce the sample difficulty index (SDI), a novel metric that quantifies how intrinsic demographic and acoustic factors drive model failure. By mapping SDI on data cartography, we demonstrate that metrics EmbER and SemDist expose hidden systemic biases and inter-model disagreements that WER ignores. Finally, our findings are the first steps towards a robust audit framework for prospective safety analysis, empowering developers to audit and mitigate ASR disparities prior to deployment. |
| 2026-03-05 | [Rethinking the Role of Collaborative Robots in Rehabilitation](http://arxiv.org/abs/2603.05252v1) | Vivek Gupte, Shalutha Rajapakshe et al. | Current research on collaborative robots (cobots) in physical rehabilitation largely focuses on repeated motion training for people undergoing physical therapy (PuPT), even though these sessions include phases that could benefit from robotic collaboration and assistance. Meanwhile, access to physical therapy remains limited for people with disabilities and chronic illnesses. Cobots could support both PuPT and therapists, and improve access to therapy, yet their broader potential remains underexplored. We propose extending the scope of cobots by imagining their role in assisting therapists and PuPT before, during, and after a therapy session. We discuss how cobot assistance may lift access barriers by promoting ability-based therapy design and helping therapists manage their time and effort. Finally, we highlight challenges to realizing these roles, including advancing user-state understanding, ensuring safety, and integrating cobots into therapists' workflow. This view opens new research questions and opportunities to draw from the HRI community's advances in assistive robotics. |
| 2026-03-05 | [Early Warning of Intraoperative Adverse Events via Transformer-Driven Multi-Label Learning](http://arxiv.org/abs/2603.05212v1) | Xueyao Wang, Xiuding Cai et al. | Early warning of intraoperative adverse events plays a vital role in reducing surgical risk and improving patient safety. While deep learning has shown promise in predicting the single adverse event, several key challenges remain: overlooking adverse event dependencies, underutilizing heterogeneous clinical data, and suffering from the class imbalance inherent in medical datasets. To address these issues, we construct the first Multi-label Adverse Events dataset (MuAE) for intraoperative adverse events prediction, covering six critical events. Next, we propose a novel Transformerbased multi-label learning framework (IAENet) that combines an improved Time-Aware Feature-wise Linear Modulation (TAFiLM) module for static covariates and dynamic variables robust fusion and complex temporal dependencies modeling. Furthermore, we introduce a Label-Constrained Reweighting Loss (LCRLoss) with co-occurrence regularization to effectively mitigate intra-event imbalance and enforce structured consistency among frequently co-occurring events. Extensive experiments demonstrate that IAENet consistently outperforms strong baselines on 5, 10, and 15-minute early warning tasks, achieving improvements of +5.05%, +2.82%, and +7.57% on average F1 score. These results highlight the potential of IAENet for supporting intelligent intraoperative decision-making in clinical practice. |
| 2026-03-05 | [Logi-PAR: Logic-Infused Patient Activity Recognition via Differentiable Rule](http://arxiv.org/abs/2603.05184v1) | Muhammad Zarar, MingZheng Zhang et al. | Patient Activity Recognition (PAR) in clinical settings uses activity data to improve safety and quality of care. Although significant progress has been made, current models mainly identify which activity is occurring. They often spatially compose sub-sparse visual cues using global and local attention mechanisms, yet only learn logically implicit patterns due to their neural-pipeline. Advancing clinical safety requires methods that can infer why a set of visual cues implies a risk, and how these can be compositionally reasoned through explicit logic beyond mere classification. To address this, we proposed Logi-PAR, the first Logic-Infused Patient Activity Recognition Framework that integrates contextual fact fusion as a multi-view primitive extractor and injects neural-guided differentiable rules. Our method automatically learns rules from visual cues, optimizing them end-to-end while enabling the implicit emergence patterns to be explicitly labelled during training. To the best of our knowledge, Logi-PAR is the first framework to recognize patient activity by applying learnable logic rules to symbolic mappings. It produces auditable why explanations as rule traces and supports counterfactual interventions (e.g., risk would decrease by 65% if assistance were present). Extensive evaluation on clinical benchmarks (VAST and OmniFall) demonstrates state-of-the-art performance, significantly outperforming Vision-Language Models and transformer baselines. The code is available via: https://github.com/zararkhan985/Logi-PAR.git} |
| 2026-03-05 | [V2N-Based Algorithm and Communication Protocol for Autonomous Non-Stop Intersections](http://arxiv.org/abs/2603.05165v1) | Lorenzo Farina, Lorenzo Mario Amorosa et al. | Intersections are critical areas for road safety and traffic efficiency, accounting for a significant portion of vehicle crashes and fatalities. While connected and autonomous vehicle (CAV) technologies offer a promising solution for autonomous intersection management, many existing proposals either rely on computationally heavy centralized controllers or overlook the practical impairments of real-world communication networks. This paper introduces seamless mobility of vehicles over intersections (Moveover), a novel algorithm comprising a vehicle-to-network (V2N) communication protocol designed to let vehicles cross autonomous intersections without stopping. Moveover delegates trajectory and speed profile selection to individual vehicles, allowing each CAV to optimize them according to its unique kinematic characteristics. Simultaneously, a local intersection controller prevents collisions through deterministic conflict zone reservations. The algorithm is rigorously evaluated under both ideal and non-ideal networking conditions, specifically modeling 4G and 5G communication delays, across multiple layouts including single-lane, multi-lane, and roundabouts. Furthermore, we test Moveover on a real urban map with multiple intersections. Simulation results demonstrate that Moveover significantly outperforms baseline strategies, offering substantial improvements in travel times and reduced pollutant emissions. |
| 2026-03-05 | [Trajectory Tracking for Uncrewed Surface Vessels with Input Saturation and Dynamic Motion Constraints](http://arxiv.org/abs/2603.05115v1) | Ram Milan Kumar Verma, Shashi Ranjan Kumar et al. | This work addresses the problem of constrained motion control of the uncrewed surface vessels. The constraints are imposed on states/inputs of the vehicles due to the physical limitations, mission requirements, and safety considerations. We develop a nonlinear feedback controller utilizing log-type Barrier Lyapunov Functions to enforce static and dynamic motion constraints. The proposed scheme uniquely addresses asymmetric constraints on position and heading alongside symmetric constraints on surge, sway, and yaw rates. Additionally, a smooth input saturation model is incorporated in the design to guarantee stability even under actuator bounds, which, if unaccounted for, can lead to severe performance degradation and poor tracking. Rigorous Lyapunov stability analysis shows that the closed-loop system remains stable and that all state variables remain within their prescribed bounds at all times, provided the initial conditions also lie within those bounds. Numerical simulations demonstrate the effectiveness of the proposed strategies for surface vessels without violating the motion and actuator constraints. |
| 2026-03-05 | [SPIRIT: Perceptive Shared Autonomy for Robust Robotic Manipulation under Deep Learning Uncertainty](http://arxiv.org/abs/2603.05111v1) | Jongseok Lee, Ribin Balachandran et al. | Deep learning (DL) has enabled impressive advances in robotic perception, yet its limited robustness and lack of interpretability hinder reliable deployment in safety critical applications. We propose a concept termed perceptive shared autonomy, in which uncertainty estimates from DL based perception are used to regulate the level of autonomy. Specifically, when the robot's perception is confident, semi-autonomous manipulation is enabled to improve performance; when uncertainty increases, control transitions to haptic teleoperation for maintaining robustness. In this way, high-performing but uninterpretable DL methods can be integrated safely into robotic systems. A key technical enabler is an uncertainty aware DL based point cloud registration approach based on the so called Neural Tangent Kernels (NTK). We evaluate perceptive shared autonomy on challenging aerial manipulation tasks through a user study of 15 participants and realization of mock-up industrial scenarios, demonstrating reliable robotic manipulation despite failures in DL based perception. The resulting system, named SPIRIT, improves both manipulation performance and system reliability. SPIRIT was selected as a finalist of a major industrial innovation award. |
| 2026-03-05 | [Aura: Universal Multi-dimensional Exogenous Integration for Aviation Time Series](http://arxiv.org/abs/2603.05092v1) | Jiafeng Lin, Mengren Zheng et al. | Time series forecasting has witnessed an increasing demand across diverse industrial applications, where accurate predictions are pivotal for informed decision-making. Beyond numerical time series data, reliable forecasting in practical scenarios requires integrating diverse exogenous factors. Such exogenous information is often multi-dimensional or even multimodal, introducing heterogeneous interactions that unimodal time series models struggle to capture. In this paper, we delve into an aviation maintenance scenario and identify three distinct types of exogenous factors that influence temporal dynamics through distinct interaction modes. Based on this empirical insight, we propose Aura, a universal framework that explicitly organizes and encodes heterogeneous external information according to its interaction mode with the target time series. Specifically, Aura utilizes a tailored tripartite encoding mechanism to embed heterogeneous features into well-established time series models, ensuring seamless integration of non-sequential context. Extensive experiments on a large-scale, three-year industrial dataset from China Southern Airlines, covering the Boeing 777 and Airbus A320 fleets, demonstrate that Aura consistently achieves state-of-the-art performance across all baselines and exhibits superior adaptability. Our findings highlight Aura's potential as a general-purpose enhancement for aviation safety and reliability. |
| 2026-03-05 | [A 360-degree Multi-camera System for Blue Emergency Light Detection Using Color Attention RT-DETR and the ABLDataset](http://arxiv.org/abs/2603.05058v1) | Francisco Vacalebri-Lloret, Lucas Banchero et al. | This study presents an advanced system for detecting blue lights on emergency vehicles, developed using ABLDataset, a curated dataset that includes images of European emergency vehicles under various climatic and geographic conditions. The system employs a configuration of four fisheye cameras, each with a 180-degree horizontal field of view, mounted on the sides of the vehicle. A calibration process enables the azimuthal localization of the detections. Additionally, a comparative analysis of major deep neural network algorithms was conducted, including YOLO (v5, v8, and v10), RetinaNet, Faster R-CNN, and RT-DETR. RT-DETR was selected as the base model and enhanced through the incorporation of a color attention block, achieving an accuracy of 94.7 percent and a recall of 94.1 percent on the test set, with field test detections reaching up to 70 meters. Furthermore, the system estimates the approach angle of the emergency vehicle relative to the center of the car using geometric transformations. Designed for integration into a multimodal system that combines visual and acoustic data, this system has demonstrated high efficiency, offering a promising approach to enhancing Advanced Driver Assistance Systems (ADAS) and road safety. |
| 2026-03-05 | [S5-SHB Agent: Society 5.0 enabled Multi-model Agentic Blockchain Framework for Smart Home](http://arxiv.org/abs/2603.05027v1) | Janani Rangila, Akila Siriweera et al. | The smart home is a key application domain within the Society 5.0 vision for a human-centered society. As smart home ecosystems expand with heterogeneous IoT protocols, diverse devices, and evolving threats, autonomous systems must manage comfort, security, energy, and safety for residents. Such autonomous decision-making requires a trust anchor, making blockchain a preferred foundation for transparent and accountable smart home governance. However, realizing this vision requires blockchain-governed smart homes to simultaneously address adaptive consensus, intelligent multi-agent coordination, and resident-controlled governance aligned with the principles of Society 5.0. Existing frameworks rely solely on rigid smart contracts with fixed consensus protocols, employ at most a single AI model without multi-agent coordination, and offer no governance mechanism for residents to control automation behaviour. To address these limitations, this paper presents the Society 5.0-driven human-centered governance-enabled smart home blockchain agent (S5-SHB-Agent). The framework orchestrates ten specialized agents using interchangeable large language models to make decisions across the safety, security, comfort, energy, privacy, and health domains. An adaptive PoW blockchain adjusts mining difficulty based on transaction volume and emergency conditions, with digital signatures and Merkle tree anchoring to ensure tamper evident auditability. A four-tier governance model enables residents to control automation through tiered preferences from routine adjustments to immutable safety thresholds. Evaluation confirms that resident governance correctly separates adjustable comfort priorities from immutable safety thresholds across all tested configurations, while adaptive consensus commits emergency blocks. |

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



