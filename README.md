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
| 2025-12-10 | [STACHE: Local Black-Box Explanations for Reinforcement Learning Policies](http://arxiv.org/abs/2512.09909v1) | Andrew Elashkin, Orna Grumberg | Reinforcement learning agents often behave unexpectedly in sparse-reward or safety-critical environments, creating a strong need for reliable debugging and verification tools. In this paper, we propose STACHE, a comprehensive framework for generating local, black-box explanations for an agent's specific action within discrete Markov games. Our method produces a Composite Explanation consisting of two complementary components: (1) a Robustness Region, the connected neighborhood of states where the agent's action remains invariant, and (2) Minimal Counterfactuals, the smallest state perturbations required to alter that decision. By exploiting the structure of factored state spaces, we introduce an exact, search-based algorithm that circumvents the fidelity gaps of surrogate models. Empirical validation on Gymnasium environments demonstrates that our framework not only explains policy actions, but also effectively captures the evolution of policy logic during training - from erratic, unstable behavior to optimized, robust strategies - providing actionable insights into agent sensitivity and decision boundaries. |
| 2025-12-10 | [CHEM: Estimating and Understanding Hallucinations in Deep Learning for Image Processing](http://arxiv.org/abs/2512.09806v1) | Jianfei Li, Ines Rosellon-Inclan et al. | U-Net and other U-shaped architectures have achieved significant success in image deconvolution tasks. However, challenges have emerged, as these methods might generate unrealistic artifacts or hallucinations, which can interfere with analysis in safety-critical scenarios. This paper introduces a novel approach for quantifying and comprehending hallucination artifacts to ensure trustworthy computer vision models. Our method, termed the Conformal Hallucination Estimation Metric (CHEM), is applicable to any image reconstruction model, enabling efficient identification and quantification of hallucination artifacts. It offers two key advantages: it leverages wavelet and shearlet representations to efficiently extract hallucinations of image features and uses conformalized quantile regression to assess hallucination levels in a distribution-free manner. Furthermore, from an approximation theoretical perspective, we explore the reasons why U-shaped networks are prone to hallucinations. We test the proposed approach on the CANDELS astronomical image dataset with models such as U-Net, SwinUNet, and Learnlets, and provide new perspectives on hallucination from different aspects in deep learning-based image processing. |
| 2025-12-10 | [Explainable Verification of Hierarchical Workflows Mined from Event Logs with Shapley Values](http://arxiv.org/abs/2512.09562v1) | Radoslaw Klimek, Jakub Blazowski | Workflow mining discovers hierarchical process trees from event logs, but it remains unclear why such models satisfy or violate logical properties, or how individual elements contribute to overall behavior. We propose to translate mined workflows into logical specifications and analyze properties such as satisfiability, liveness, and safety with automated theorem provers. On this basis, we adapt Shapley values from cooperative game theory to attribute outcomes to workflow elements and quantify their contributions. Experiments on benchmark datasets show that this combination identifies critical nodes, reveals redundancies, and exposes harmful structures. This outlines a novel direction for explainable workflow analysis with direct relevance to software engineering practice, supporting compliance checks, process optimization, redundancy reduction, and the design of next-generation process mining tools. |
| 2025-12-10 | [REASAN: Learning Reactive Safe Navigation for Legged Robots](http://arxiv.org/abs/2512.09537v1) | Qihao Yuan, Ziyu Cao et al. | We present a novel modularized end-to-end framework for legged reactive navigation in complex dynamic environments using a single light detection and ranging (LiDAR) sensor. The system comprises four simulation-trained modules: three reinforcement-learning (RL) policies for locomotion, safety shielding, and navigation, and a transformer-based exteroceptive estimator that processes raw point-cloud inputs. This modular decomposition of complex legged motor-control tasks enables lightweight neural networks with simple architectures, trained using standard RL practices with targeted reward shaping and curriculum design, without reliance on heuristics or sophisticated policy-switching mechanisms. We conduct comprehensive ablations to validate our design choices and demonstrate improved robustness compared to existing approaches in challenging navigation tasks. The resulting reactive safe navigation (REASAN) system achieves fully onboard and real-time reactive navigation across both single- and multi-robot settings in complex environments. We release our training and deployment code at https://github.com/ASIG-X/REASAN. |
| 2025-12-10 | [CNFinBench: A Benchmark for Safety and Compliance of Large Language Models in Finance](http://arxiv.org/abs/2512.09506v1) | Jinru Ding, Chao Ding et al. | Large language models are increasingly deployed across the financial sector for tasks such as research, compliance, risk analysis, and customer service, which makes rigorous safety evaluation essential. However, existing financial benchmarks primarily focus on textbook-style question answering and numerical problem solving, but fail to evaluate models' real-world safety behaviors. They weakly assess regulatory compliance and investor-protection norms, rarely stress-test multi-turn adversarial tactics such as jailbreaks or prompt injection, inconsistently ground answers in long filings, ignore tool- or RAG-induced over-reach risks, and rely on opaque or non-auditable evaluation protocols. To close these gaps, we introduce CNFinBench, a benchmark that employs finance-tailored red-team dialogues and is structured around a Capability-Compliance-Safety triad, including evidence-grounded reasoning over long reports and jurisdiction-aware rule/tax compliance tasks. For systematic safety quantification, we introduce the Harmful Instruction Compliance Score (HICS) to measure how consistently models resist harmful prompts across multi-turn adversarial dialogues. To ensure auditability, CNFinBench enforces strict output formats with dynamic option perturbation for objective tasks and employs a hybrid LLM-ensemble plus human-calibrated judge for open-ended evaluations. Experiments on 21 models across 15 subtasks confirm a persistent capability-compliance gap: models achieve an average score of 61.0 on capability tasks but fall to 34.18 on compliance and risk-control evaluations. Under multi-turn adversarial dialogue tests, most systems reach only partial resistance (HICS 60-79), demonstrating that refusal alone is not a reliable proxy for safety without cited and verifiable reasoning. |
| 2025-12-10 | [Source Coverage and Citation Bias in LLM-based vs. Traditional Search Engines](http://arxiv.org/abs/2512.09483v1) | Peixian Zhang, Qiming Ye et al. | LLM-based Search Engines (LLM-SEs) introduces a new paradigm for information seeking. Unlike Traditional Search Engines (TSEs) (e.g., Google), these systems summarize results, often providing limited citation transparency. The implications of this shift remain largely unexplored, yet raises key questions regarding trust and transparency. In this paper, we present a large-scale empirical study of LLM-SEs, analyzing 55,936 queries and the corresponding search results across six LLM-SEs and two TSEs. We confirm that LLM-SEs cites domain resources with greater diversity than TSEs. Indeed, 37% of domains are unique to LLM-SEs. However, certain risks still persist: LLM-SEs do not outperform TSEs in credibility, political neutrality and safety metrics. Finally, to understand the selection criteria of LLM-SEs, we perform a feature-based analysis to identify key factors influencing source choice. Our findings provide actionable insights for end users, website owners, and developers. |
| 2025-12-10 | [An Efficient Interaction Human-AI Synergy System Bridging Visual Awareness and Large Language Model for Intensive Care Units](http://arxiv.org/abs/2512.09473v1) | Yibowen Zhao, Yiming Cao et al. | Intensive Care Units (ICUs) are critical environments characterized by high-stakes monitoring and complex data management. However, current practices often rely on manual data transcription and fragmented information systems, introducing potential risks to patient safety and operational efficiency. To address these issues, we propose a human-AI synergy system based on a cloud-edge-end architecture, which integrates visual-aware data extraction and semantic interaction mechanisms. Specifically, a visual-aware edge module non-invasively captures real-time physiological data from bedside monitors, reducing manual entry errors. To improve accessibility to fragmented data sources, a semantic interaction module, powered by a Large Language Model (LLM), enables physicians to perform efficient and intuitive voice-based queries over structured patient data. The hierarchical cloud-edge-end deployment ensures low-latency communication and scalable system performance. Our system reduces the cognitive burden on ICU nurses and physicians and demonstrates promising potential for broader applications in intelligent healthcare systems. |
| 2025-12-10 | [Architectures for Building Agentic AI](http://arxiv.org/abs/2512.09458v1) | Sławomir Nowaczyk | This chapter argues that the reliability of agentic and generative AI is chiefly an architectural property. We define agentic systems as goal-directed, tool-using decision makers operating in closed loops, and show how reliability emerges from principled componentisation (goal manager, planner, tool-router, executor, memory, verifiers, safety monitor, telemetry), disciplined interfaces (schema-constrained, validated, least-privilege tool calls), and explicit control and assurance loops. Building on classical foundations, we propose a practical taxonomy-tool-using agents, memory-augmented agents, planning and self-improvement agents, multi-agent systems, and embodied or web agents - and analyse how each pattern reshapes the reliability envelope and failure modes. We distil design guidance on typed schemas, idempotency, permissioning, transactional semantics, memory provenance and hygiene, runtime governance (budgets, termination conditions), and simulate-before-actuate safeguards. |
| 2025-12-10 | [ODMA: On-Demand Memory Allocation Framework for LLM Serving on LPDDR-Class Accelerators](http://arxiv.org/abs/2512.09427v1) | Guoqiang Zou, Wanyu Wang et al. | Serving large language models (LLMs) on accelerators with poor random-access bandwidth (e.g., LPDDR5-based) is limited by current memory managers. Static pre-allocation wastes memory, while fine-grained paging (e.g., PagedAttention) is ill-suited due to high random-access costs. Existing HBM-centric solutions do not exploit the characteristics of random-access-constrained memory (RACM) accelerators like Cambricon MLU370. We present ODMA, an on-demand memory allocation framework for RACM. ODMA addresses distribution drift and heavy-tailed requests by coupling a lightweight length predictor with dynamic bucket partitioning and a large-bucket safeguard. Boundaries are periodically updated from live traces to maximize utilization. On Alpaca and Google-NQ, ODMA improves prediction accuracy of prior work significantly (e.g., from 82.68% to 93.36%). Serving DeepSeek-R1-Distill-Qwen-7B on Cambricon MLU370-X4, ODMA raises memory utilization from 55.05% to 72.45% and improves RPS and TPS by 29% and 27% over static baselines. This demonstrates that hardware-aware allocation unlocks efficient LLM serving on RACM platforms. |
| 2025-12-10 | [Time-Discretized Simulation of Vehicle Platoons for Safety Analysis with Guaranteed Error Bounds](http://arxiv.org/abs/2512.09416v1) | Yuhao Chen, Ahmet Cetinkaya | Wireless communication is essential to achieve coordinated control in vehicle platoons. However, packet losses in wireless communication can cause critical safety issues when they occur in conjunction with sudden brakes. In this paper, we propose simulation-based methods that allow the study of such safety issues by determining the absolute minimum distance between vehicles over time for various control parameters that guarantee string stability. For our proposed time-discretized simulations, we provide two methods for selecting different time-step intervals to ensure that the error in distance approximation remains within specified bounds at all times. Through numerical examples we demonstrate that among control parameters that guarantee string stability some perform better than others under simultaneously occurring packet losses and sudden brakes. |
| 2025-12-10 | [Black-Box Behavioral Distillation Breaks Safety Alignment in Medical LLMs](http://arxiv.org/abs/2512.09403v1) | Sohely Jahan, Ruimin Sun | As medical large language models (LLMs) become increasingly integrated into clinical workflows, concerns around alignment robustness, and safety are escalating. Prior work on model extraction has focused on classification models or memorization leakage, leaving the vulnerability of safety-aligned generative medical LLMs underexplored.   We present a black-box distillation attack that replicates the domain-specific reasoning of safety-aligned medical LLMs using only output-level access. By issuing 48,000 instruction queries to Meditron-7B and collecting 25,000 benign instruction response pairs, we fine-tune a LLaMA3 8B surrogate via parameter efficient LoRA under a zero-alignment supervision setting, requiring no access to model weights, safety filters, or training data. With a cost of $12, the surrogate achieves strong fidelity on benign inputs while producing unsafe completions for 86% of adversarial prompts, far exceeding both Meditron-7B (66%) and the untuned base model (46%). This reveals a pronounced functional-ethical gap, task utility transfers, while alignment collapses. To analyze this collapse, we develop a dynamic adversarial evaluation framework combining Generative Query (GQ)-based harmful prompt generation, verifier filtering, category-wise failure analysis, and adaptive Random Search (RS) jailbreak attacks. We also propose a layered defense system, as a prototype detector for real-time alignment drift in black-box deployments.   Our findings show that benign-only black-box distillation exposes a practical and under-recognized threat: adversaries can cheaply replicate medical LLM capabilities while stripping safety mechanisms, underscoring the need for extraction-aware safety monitoring. |
| 2025-12-10 | [CFLight: Enhancing Safety with Traffic Signal Control through Counterfactual Learning](http://arxiv.org/abs/2512.09368v1) | Mingyuan Li, Chunyu Liu et al. | Traffic accidents result in millions of injuries and fatalities globally, with a significant number occurring at intersections each year. Traffic Signal Control (TSC) is an effective strategy for enhancing safety at these urban junctures. Despite the growing popularity of Reinforcement Learning (RL) methods in optimizing TSC, these methods often prioritize driving efficiency over safety, thus failing to address the critical balance between these two aspects. Additionally, these methods usually need more interpretability. CounterFactual (CF) learning is a promising approach for various causal analysis fields. In this study, we introduce a novel framework to improve RL for safety aspects in TSC. This framework introduces a novel method based on CF learning to address the question: ``What if, when an unsafe event occurs, we backtrack to perform alternative actions, and will this unsafe event still occur in the subsequent period?'' To answer this question, we propose a new structure causal model to predict the result after executing different actions, and we propose a new CF module that integrates with additional ``X'' modules to promote safe RL practices. Our new algorithm, CFLight, which is derived from this framework, effectively tackles challenging safety events and significantly improves safety at intersections through a near-zero collision control strategy. Through extensive numerical experiments on both real-world and synthetic datasets, we demonstrate that CFLight reduces collisions and improves overall traffic performance compared to conventional RL methods and the recent safe RL model. Moreover, our method represents a generalized and safe framework for RL methods, opening possibilities for applications in other domains. The data and code are available in the github https://github.com/MJLee00/CFLight-Enhancing-Safety-with-Traffic-Signal-Control-through-Counterfactual-Learning. |
| 2025-12-10 | [Efficiency-Aware Computational Intelligence for Resource-Constrained Manufacturing Toward Edge-Ready Deployment](http://arxiv.org/abs/2512.09319v1) | Qianyu Zhou | Industrial cyber physical systems operate under heterogeneous sensing, stochastic dynamics, and shifting process conditions, producing data that are often incomplete, unlabeled, imbalanced, and domain shifted. High-fidelity datasets remain costly, confidential, and slow to obtain, while edge devices face strict limits on latency, bandwidth, and energy. These factors restrict the practicality of centralized deep learning, hinder the development of reliable digital twins, and increase the risk of error escape in safety-critical applications. Motivated by these challenges, this dissertation develops an efficiency grounded computational framework that enables data lean, physics-aware, and deployment ready intelligence for modern manufacturing environments. The research advances methods that collectively address core bottlenecks across multimodal and multiscale industrial scenarios. Generative strategies mitigate data scarcity and imbalance, while semi-supervised learning integrates unlabeled information to reduce annotation and simulation demands. Physics-informed representation learning strengthens interpretability and improves condition monitoring under small-data regimes. Spatially aware graph-based surrogate modeling provides efficient approximation of complex processes, and an edge cloud collaborative compression scheme supports real-time signal analytics under resource constraints. The dissertation also extends visual understanding through zero-shot vision language reasoning augmented by domain specific retrieval, enabling generalizable assessment in previously unseen scenarios. Together, these developments establish a unified paradigm of data efficient and resource aware intelligence that bridges laboratory learning with industrial deployment, supporting reliable decision-making across diverse manufacturing systems. |
| 2025-12-10 | [Transformer-Driven Multimodal Fusion for Explainable Suspiciousness Estimation in Visual Surveillance](http://arxiv.org/abs/2512.09311v1) | Kuldeep Singh Yadav, Lalan Kumar | Suspiciousness estimation is critical for proactive threat detection and ensuring public safety in complex environments. This work introduces a large-scale annotated dataset, USE50k, along with a computationally efficient vision-based framework for real-time suspiciousness analysis. The USE50k dataset contains 65,500 images captured from diverse and uncontrolled environments, such as airports, railway stations, restaurants, parks, and other public areas, covering a broad spectrum of cues including weapons, fire, crowd density, abnormal facial expressions, and unusual body postures. Building on this dataset, we present DeepUSEvision, a lightweight and modular system integrating three key components, i.e., a Suspicious Object Detector based on an enhanced YOLOv12 architecture, dual Deep Convolutional Neural Networks (DCNN-I and DCNN-II) for facial expression and body-language recognition using image and landmark features, and a transformer-based Discriminator Network that adaptively fuses multimodal outputs to yield an interpretable suspiciousness score. Extensive experiments confirm the superior accuracy, robustness, and interpretability of the proposed framework compared to state-of-the-art approaches. Collectively, the USE50k dataset and the DeepUSEvision framework establish a strong and scalable foundation for intelligent surveillance and real-time risk assessment in safety-critical applications. |
| 2025-12-09 | [Exposing Vulnerabilities in Counterfeit Prevention Systems Utilizing Physically Unclonable Surface Features](http://arxiv.org/abs/2512.09150v1) | Anirudh Nakra, Nayeeb Rashid et al. | Counterfeit products pose significant risks to public health and safety through infiltrating untrusted supply chains. Among numerous anti-counterfeiting techniques, leveraging inherent, unclonable microscopic irregularities of paper surfaces is an accurate and cost-effective solution. Prior work of this approach has focused on enabling ubiquitous acquisition of these physically unclonable features (PUFs). However, we will show that existing authentication methods relying on paper surface PUFs may be vulnerable to adversaries, resulting in a gap between technological feasibility and secure real-world deployment. This gap is investigated through formalizing an operational framework for paper-PUF-based authentication. Informed by this framework, we reveal system-level vulnerabilities across both physical and digital domains, designing physical denial-of-service and digital forgery attacks to disrupt proper authentication. The effectiveness of the designed attacks underscores the strong need for security countermeasures for reliable and resilient authentication based on paper PUFs. The proposed framework further facilitates a comprehensive, stage-by-stage security analysis, guiding the design of future counterfeit prevention systems. This analysis delves into potential attack strategies, offering a foundational understanding of how various system components, such as physical features and verification processes, might be exploited by adversaries. |
| 2025-12-09 | [Knowledge-Guided Large Language Model for Automatic Pediatric Dental Record Understanding and Safe Antibiotic Recommendation](http://arxiv.org/abs/2512.09127v1) | Zihan Han, Junyan Ge et al. | Accurate interpretation of pediatric dental clinical records and safe antibiotic prescribing remain persistent challenges in dental informatics. Traditional rule-based clinical decision support systems struggle with unstructured dental narratives, incomplete radiographic descriptions, and complex safety constraints. To address these limitations, this study proposes a Knowledge-Guided Large Language Model (KG-LLM) that integrates a pediatric dental knowledge graph, retrieval-augmented generation (RAG), and a multi-stage safety validation pipeline for evidence-grounded antibiotic recommendation. The framework first employs a clinical NER/RE module to extract structured entities and relations from dental notes and radiology reports. Relevant guidelines, drug-safety rules, and analogous historical cases are subsequently retrieved from the knowledge graph and supplied to the LLM for diagnostic summarization and dose-drug-duration prediction. Safety assurance is achieved through a dual-layer validation mechanism combining deterministic rule checking with a learned classifier for detecting allergies, contraindications, and dosing errors. Experiments on 32,000 de-identified pediatric dental visit records demonstrate the effectiveness of the proposed approach. Compared with a domain-adapted Llama-2 clinical baseline, KG-LLM improves record-understanding performance (F1: 0.914 vs. 0.867), drug-dose-duration accuracy (Top-1: 0.782 vs. 0.716), and reduces unsafe antibiotic suggestions by 50%. Additional evaluation across summary quality, recommendation accuracy, and global safety scores further confirms the robustness of the system. Ablation analyses indicate that the knowledge graph, RAG, and safety modules each contribute substantially to clinical reliability and interpretability. |
| 2025-12-09 | [Semantic Trajectory Generation for Goal-Oriented Spacecraft Rendezvous](http://arxiv.org/abs/2512.09111v1) | Yuji Takubo, Arpit Dwivedi et al. | Reliable real-time trajectory generation is essential for future autonomous spacecraft. While recent progress in nonconvex guidance and control is paving the way for onboard autonomous trajectory optimization, these methods still rely on extensive expert input (e.g., waypoints, constraints, mission timelines, etc.), which limits the operational scalability in real rendezvous missions.This paper introduces SAGES (Semantic Autonomous Guidance Engine for Space), a trajectory-generation framework that translates natural-language commands into spacecraft trajectories that reflect high-level intent while respecting nonconvex constraints. Experiments in two settings -- fault-tolerant proximity operations with continuous-time constraint enforcement and a free-flying robotic platform -- demonstrate that SAGES reliably produces trajectories aligned with human commands, achieving over 90\% semantic-behavioral consistency across diverse behavior modes. Ultimately, this work marks an initial step toward language-conditioned, constraint-aware spacecraft trajectory generation, enabling operators to interactively guide both safety and behavior through intuitive natural-language commands with reduced expert burden. |
| 2025-12-09 | [SIP: Site in Pieces- A Dataset of Disaggregated Construction-Phase 3D Scans for Semantic Segmentation and Scene Understanding](http://arxiv.org/abs/2512.09062v1) | Seongyong Kim, Yong Kwon Cho | Accurate 3D scene interpretation in active construction sites is essential for progress monitoring, safety assessment, and digital twin development. LiDAR is widely used in construction because it offers advantages over camera-based systems, performing reliably in cluttered and dynamically changing conditions. Yet most public datasets for 3D perception are derived from densely fused scans with uniform sampling and complete visibility, conditions that do not reflect real construction sites. Field data are often collected as isolated single-station LiDAR views, constrained by safety requirements, limited access, and ongoing operations. These factors lead to radial density decay, fragmented geometry, and view-dependent visibility-characteristics that remain underrepresented in existing datasets. This paper presents SIP, Site in Pieces, a dataset created to reflect the practical constraints of LiDAR acquisition during construction. SIP provides indoor and outdoor scenes captured with a terrestrial LiDAR scanner and annotated at the point level using a taxonomy tailored to construction environments: A. Built Environment, B. Construction Operations, and C. Site Surroundings. The dataset includes both structural components and slender temporary objects such as scaffolding, MEP piping, and scissor lifts, where sparsity caused by occlusion and fragmented geometry make segmentation particularly challenging. The scanning protocol, annotation workflow, and quality control procedures establish a consistent foundation for the dataset. SIP is openly available with a supporting Git repository, offering adaptable class configurations that streamline adoption within modern 3D deep learning frameworks. By providing field data that retain real-world sensing characteristics, SIP enables robust benchmarking and contributes to advancing construction-oriented 3D vision tasks. |
| 2025-12-09 | [Monitoring Deployed AI Systems in Health Care](http://arxiv.org/abs/2512.09048v1) | Timothy Keyes, Alison Callahan et al. | Post-deployment monitoring of artificial intelligence (AI) systems in health care is essential to ensure their safety, quality, and sustained benefit-and to support governance decisions about which systems to update, modify, or decommission. Motivated by these needs, we developed a framework for monitoring deployed AI systems grounded in the mandate to take specific actions when they fail to behave as intended. This framework, which is now actively used at Stanford Health Care, is organized around three complementary principles: system integrity, performance, and impact. System integrity monitoring focuses on maximizing system uptime, detecting runtime errors, and identifying when changes to the surrounding IT ecosystem have unintended effects. Performance monitoring focuses on maintaining accurate system behavior in the face of changing health care practices (and thus input data) over time. Impact monitoring assesses whether a deployed system continues to have value in the form of benefit to clinicians and patients. Drawing on examples of deployed AI systems at our academic medical center, we provide practical guidance for creating monitoring plans based on these principles that specify which metrics to measure, when those metrics should be reviewed, who is responsible for acting when metrics change, and what concrete follow-up actions should be taken-for both traditional and generative AI. We also discuss challenges to implementing this framework, including the effort and cost of monitoring for health systems with limited resources and the difficulty of incorporating data-driven monitoring practices into complex organizations where conflicting priorities and definitions of success often coexist. This framework offers a practical template and starting point for health systems seeking to ensure that AI deployments remain safe and effective over time. |
| 2025-12-09 | [Secure and Privacy-Preserving Federated Learning for Next-Generation Underground Mine Safety](http://arxiv.org/abs/2512.08862v1) | Mohamed Elmahallawy, Sanjay Madria et al. | Underground mining operations depend on sensor networks to monitor critical parameters such as temperature, gas concentration, and miner movement, enabling timely hazard detection and safety decisions. However, transmitting raw sensor data to a centralized server for machine learning (ML) model training raises serious privacy and security concerns. Federated Learning (FL) offers a promising alternative by enabling decentralized model training without exposing sensitive local data. Yet, applying FL in underground mining presents unique challenges: (i) Adversaries may eavesdrop on shared model updates to launch model inversion or membership inference attacks, compromising data privacy and operational safety; (ii) Non-IID data distributions across mines and sensor noise can hinder model convergence. To address these issues, we propose FedMining--a privacy-preserving FL framework tailored for underground mining. FedMining introduces two core innovations: (1) a Decentralized Functional Encryption (DFE) scheme that keeps local models encrypted, thwarting unauthorized access and inference attacks; and (2) a balancing aggregation mechanism to mitigate data heterogeneity and enhance convergence. Evaluations on real-world mining datasets demonstrate FedMining's ability to safeguard privacy while maintaining high model accuracy and achieving rapid convergence with reduced communication and computation overhead. These advantages make FedMining both secure and practical for real-time underground safety monitoring. |

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



