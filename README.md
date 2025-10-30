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
| 2025-10-29 | [Gaperon: A Peppered English-French Generative Language Model Suite](http://arxiv.org/abs/2510.25771v1) | Nathan Godey, Wissam Antoun et al. | We release Gaperon, a fully open suite of French-English-coding language models designed to advance transparency and reproducibility in large-scale model training. The Gaperon family includes 1.5B, 8B, and 24B parameter models trained on 2-4 trillion tokens, released with all elements of the training pipeline: French and English datasets filtered with a neural quality classifier, an efficient data curation and training framework, and hundreds of intermediate checkpoints. Through this work, we study how data filtering and contamination interact to shape both benchmark and generative performance. We find that filtering for linguistic quality enhances text fluency and coherence but yields subpar benchmark results, and that late deliberate contamination -- continuing training on data mixes that include test sets -- recovers competitive scores while only reasonably harming generation quality. We discuss how usual neural filtering can unintentionally amplify benchmark leakage. To support further research, we also introduce harmless data poisoning during pretraining, providing a realistic testbed for safety studies. By openly releasing all models, datasets, code, and checkpoints, Gaperon establishes a reproducible foundation for exploring the trade-offs between data curation, evaluation, safety, and openness in multilingual language model development. |
| 2025-10-29 | [PairUni: Pairwise Training for Unified Multimodal Language Models](http://arxiv.org/abs/2510.25682v1) | Jiani Zheng, Zhiyang Teng et al. | Unified vision-language models (UVLMs) must perform both understanding and generation within a single architecture, but these tasks rely on heterogeneous data and supervision, making it difficult to balance them during reinforcement learning (RL). We propose PairUni, a unified framework that reorganizes data into understanding-generation (UG) pairs and aligns optimization accordingly. We first use GPT-o3 to augment single-task data, generating captions for understanding samples and question-answer (QA) pairs for generation samples, forming aligned pairs from the same instance. Additionally, for each generation sample, we retrieve a semantically related understanding example to form a retrieved pair, linking different but related data points. These paired structures expose cross-task semantic correspondences and support consistent policy learning. To leverage this structure, we present Pair-GPRO, a pair-aware variant based on Group Relative Policy Optimization. It assigns a similarity score to each pair to modulate the advantage, strengthening learning from well-aligned examples and reducing task interference. We curate a high-quality dataset of 16K UG pairs named PairUG for RL fine-tuning and evaluate PairUni on the powerful Janus-Pro UVLMs. Our approach achieves balanced improvements on various UVLMs, outperforming strong UVLM RL baselines. Code: \href{https://github.com/Haochen-Wang409/PairUni}{github.com/Haochen-Wang409/PairUni} |
| 2025-10-29 | [Uncertainty Quantification for Regression: A Unified Framework based on kernel scores](http://arxiv.org/abs/2510.25599v1) | Christopher Bülte, Yusuf Sale et al. | Regression tasks, notably in safety-critical domains, require proper uncertainty quantification, yet the literature remains largely classification-focused. In this light, we introduce a family of measures for total, aleatoric, and epistemic uncertainty based on proper scoring rules, with a particular emphasis on kernel scores. The framework unifies several well-known measures and provides a principled recipe for designing new ones whose behavior, such as tail sensitivity, robustness, and out-of-distribution responsiveness, is governed by the choice of kernel. We prove explicit correspondences between kernel-score characteristics and downstream behavior, yielding concrete design guidelines for task-specific measures. Extensive experiments demonstrate that these measures are effective in downstream tasks and reveal clear trade-offs among instantiations, including robustness and out-of-distribution detection performance. |
| 2025-10-29 | [Incorporating Social Awareness into Control of Unknown Multi-Agent Systems: A Real-Time Spatiotemporal Tubes Approach](http://arxiv.org/abs/2510.25597v1) | Siddhartha Upadhyay, Ratnangshu Das et al. | This paper presents a decentralized control framework that incorporates social awareness into multi-agent systems with unknown dynamics to achieve prescribed-time reach-avoid-stay tasks in dynamic environments. Each agent is assigned a social awareness index that quantifies its level of cooperation or self-interest, allowing heterogeneous social behaviors within the system. Building on the spatiotemporal tube (STT) framework, we propose a real-time STT framework that synthesizes tubes online for each agent while capturing its social interactions with others. A closed-form, approximation-free control law is derived to ensure that each agent remains within its evolving STT, thereby avoiding dynamic obstacles while also preventing inter-agent collisions in a socially aware manner, and reaching the target within a prescribed time. The proposed approach provides formal guarantees on safety and timing, and is computationally lightweight, model-free, and robust to unknown disturbances. The effectiveness and scalability of the framework are validated through simulation and hardware experiments on a 2D omnidirectional |
| 2025-10-29 | [Psychoacoustic assessment of synthetic sounds for electric vehicles in a virtual reality experiment](http://arxiv.org/abs/2510.25593v1) | Pavlo Bazilinskyy, Md Shadab Alam et al. | The growing adoption of electric vehicles, known for their quieter operation compared to internal combustion engine vehicles, raises concerns about their detectability, particularly for vulnerable road users. To address this, regulations mandate the inclusion of exterior sound signals for electric vehicles, specifying minimum sound pressure levels at low speeds. These synthetic exterior sounds are often used in noisy urban environments, creating the challenge of enhancing detectability without introducing excessive noise annoyance. This study investigates the design of synthetic exterior sound signals that balance high noticeability with low annoyance. An audiovisual experiment with 14 participants was conducted using 15 virtual reality scenarios featuring a passing car. The scenarios included various sound signals, such as pure, intermittent, and complex tones at different frequencies. Two baseline cases, a diesel engine and only tyre noise, were also tested. Participants rated sounds for annoyance, noticeability, and informativeness using 11-point ICBEN scales. The findings highlight how psychoacoustic sound quality metrics predict annoyance ratings better than conventional sound metrics, providing insight into optimising sound design for electric vehicles. By improving pedestrian safety while minimising noise pollution, this research supports the development of effective and user-friendly exterior sound standards for electric vehicles. |
| 2025-10-29 | [Agentic AI: A Comprehensive Survey of Architectures, Applications, and Future Directions](http://arxiv.org/abs/2510.25445v1) | Mohamad Abou Ali, Fadi Dornaika | Agentic AI represents a transformative shift in artificial intelligence, but its rapid advancement has led to a fragmented understanding, often conflating modern neural systems with outdated symbolic models -- a practice known as conceptual retrofitting. This survey cuts through this confusion by introducing a novel dual-paradigm framework that categorizes agentic systems into two distinct lineages: the Symbolic/Classical (relying on algorithmic planning and persistent state) and the Neural/Generative (leveraging stochastic generation and prompt-driven orchestration). Through a systematic PRISMA-based review of 90 studies (2018--2025), we provide a comprehensive analysis structured around this framework across three dimensions: (1) the theoretical foundations and architectural principles defining each paradigm; (2) domain-specific implementations in healthcare, finance, and robotics, demonstrating how application constraints dictate paradigm selection; and (3) paradigm-specific ethical and governance challenges, revealing divergent risks and mitigation strategies. Our analysis reveals that the choice of paradigm is strategic: symbolic systems dominate safety-critical domains (e.g., healthcare), while neural systems prevail in adaptive, data-rich environments (e.g., finance). Furthermore, we identify critical research gaps, including a significant deficit in governance models for symbolic systems and a pressing need for hybrid neuro-symbolic architectures. The findings culminate in a strategic roadmap arguing that the future of Agentic AI lies not in the dominance of one paradigm, but in their intentional integration to create systems that are both adaptable and reliable. This work provides the essential conceptual toolkit to guide future research, development, and policy toward robust and trustworthy hybrid intelligent systems. |
| 2025-10-29 | [Small Talk, Big Impact? LLM-based Conversational Agents to Mitigate Passive Fatigue in Conditional Automated Driving](http://arxiv.org/abs/2510.25421v1) | Lewis Cockram, Yueteng Yu et al. | Passive fatigue during conditional automated driving can compromise driver readiness and safety. This paper presents findings from a test-track study with 40 participants in a real-world rural automated driving scenario. In this scenario, a Large Language Model (LLM) based conversational agent (CA) was designed to check in with drivers and re-engage them with their surroundings. Drawing on in-car video recordings, sleepiness ratings and interviews, we analysed how drivers interacted with the agent and how these interactions shaped alertness. Users found the CA helpful for supporting vigilance during passive fatigue. Thematic analysis of acceptability further revealed three user preference profiles that implicate future intention to use CAs. Positioning empirically observed profiles within existing CA archetype frameworks highlights the need for adaptive design sensitive to diverse user groups. This work underscores the potential of CAs as proactive Human-Machine Interface (HMI) interventions, demonstrating how natural language can support context-aware interaction during automated driving. |
| 2025-10-29 | [TECS/Rust: Memory-safe Component Framework for Embedded Systems](http://arxiv.org/abs/2510.25270v1) | Nao Yoshimura, Hiroshi Oyama et al. | As embedded systems grow in complexity and scale due to increased functional diversity, component-based development (CBD) emerges as a solution to streamline their architecture and enhance functionality reuse. CBD typically utilizes the C programming language for its direct hardware access and low-level operations, despite its susceptibility to memory-related issues. To address these concerns, this paper proposes TECS/Rust, a Rust-based framework specifically designed for TECS, which is a component framework for embedded systems. It leverages Rust's compile-time memory-safe features, such as lifetime and borrowing, to mitigate memory vulnerabilities common with C. The proposed framework not only ensures memory safety but also maintains the flexibility of CBD, automates Rust code generation for CBD components, and supports efficient integration with real-time operating systems. An evaluation of the amount of generated code indicates that the code generated by this paper framework accounts for a large percentage of the actual code. Compared to code developed without the proposed framework, the difference in execution time is minimal, indicating that the overhead introduced by the proposed framework is negligible. |
| 2025-10-29 | [TECS/Rust-OE: Optimizing Exclusive Control in Rust-based Component Systems for Embedded Devices](http://arxiv.org/abs/2510.25242v1) | Nao Yoshimura, Hiroshi Oyama et al. | The diversification of functionalities and the development of the IoT are making embedded systems larger and more complex in structure. Ensuring system reliability, especially in terms of security, necessitates selecting an appropriate programming language. As part of existing research, TECS/Rust has been proposed as a framework that combines Rust and component-based development (CBD) to enable scalable system design and enhanced reliability. This framework represents system structures using static mutable variables, but excessive exclusive controls applied to ensure thread safety have led to performance degradation. This paper proposes TECS/Rust-OE, a memory-safe CBD framework utilizing call flows to address these limitations. The proposed Rust code leverages real-time OS exclusive control mechanisms, optimizing performance without compromising reusability. Rust code is automatically generated based on component descriptions. Evaluations demonstrate reduced overhead due to optimized exclusion control and high reusability of the generated code. |
| 2025-10-29 | [Human Resilience in the AI Era -- What Machines Can't Replace](http://arxiv.org/abs/2510.25218v1) | Shaoshan Liu, Anina Schwarzenbach et al. | AI is displacing tasks, mediating high-stakes decisions, and flooding communication with synthetic content, unsettling work, identity, and social trust. We argue that the decisive human countermeasure is resilience. We define resilience across three layers: psychological, including emotion regulation, meaning-making, cognitive flexibility; social, including trust, social capital, coordinated response; organizational, including psychological safety, feedback mechanisms, and graceful degradation. We synthesize early evidence that these capacities buffer individual strain, reduce burnout through social support, and lower silent failure in AI-mediated workflows through team norms and risk-responsive governance. We also show that resilience can be cultivated through training that complements rather than substitutes for structural safeguards. By reframing the AI debate around actionable human resilience, this article offers policymakers, educators, and operators a practical lens to preserve human agency and steer responsible adoption. |
| 2025-10-29 | [RoadSens-4M: A Multimodal Smartphone & Camera Dataset for Holistic Road-way Analysis](http://arxiv.org/abs/2510.25211v1) | Amith Khandakar, David Michelson et al. | It's important to monitor road issues such as bumps and potholes to enhance safety and improve road conditions. Smartphones are equipped with various built-in sensors that offer a cost-effective and straightforward way to assess road quality. However, progress in this area has been slow due to the lack of high-quality, standardized datasets. This paper discusses a new dataset created by a mobile app that collects sensor data from devices like GPS, accelerometers, gyroscopes, magnetometers, gravity sensors, and orientation sensors. This dataset is one of the few that integrates Geographic Information System (GIS) data with weather information and video footage of road conditions, providing a comprehensive understanding of road issues with geographic context. The dataset allows for a clearer analysis of road conditions by compiling essential data, including vehicle speed, acceleration, rotation rates, and magnetic field intensity, along with the visual and spatial context provided by GIS, weather, and video data. Its goal is to provide funding for initiatives that enhance traffic management, infrastructure development, road safety, and urban planning. Additionally, the dataset will be publicly accessible to promote further research and innovation in smart transportation systems. |
| 2025-10-29 | [Agentic Moderation: Multi-Agent Design for Safer Vision-Language Models](http://arxiv.org/abs/2510.25179v1) | Juan Ren, Mark Dras et al. | Agentic methods have emerged as a powerful and autonomous paradigm that enhances reasoning, collaboration, and adaptive control, enabling systems to coordinate and independently solve complex tasks. We extend this paradigm to safety alignment by introducing Agentic Moderation, a model-agnostic framework that leverages specialised agents to defend multimodal systems against jailbreak attacks. Unlike prior approaches that apply as a static layer over inputs or outputs and provide only binary classifications (safe or unsafe), our method integrates dynamic, cooperative agents, including Shield, Responder, Evaluator, and Reflector, to achieve context-aware and interpretable moderation. Extensive experiments across five datasets and four representative Large Vision-Language Models (LVLMs) demonstrate that our approach reduces the Attack Success Rate (ASR) by 7-19%, maintains a stable Non-Following Rate (NF), and improves the Refusal Rate (RR) by 4-20%, achieving robust, interpretable, and well-balanced safety performance. By harnessing the flexibility and reasoning capacity of agentic architectures, Agentic Moderation provides modular, scalable, and fine-grained safety enforcement, highlighting the broader potential of agentic systems as a foundation for automated safety governance. |
| 2025-10-29 | [DINO-YOLO: Self-Supervised Pre-training for Data-Efficient Object Detection in Civil Engineering Applications](http://arxiv.org/abs/2510.25140v1) | Malaisree P, Youwai S et al. | Object detection in civil engineering applications is constrained by limited annotated data in specialized domains. We introduce DINO-YOLO, a hybrid architecture combining YOLOv12 with DINOv3 self-supervised vision transformers for data-efficient detection. DINOv3 features are strategically integrated at two locations: input preprocessing (P0) and mid-backbone enhancement (P3). Experimental validation demonstrates substantial improvements: Tunnel Segment Crack detection (648 images) achieves 12.4% improvement, Construction PPE (1K images) gains 13.7%, and KITTI (7K images) shows 88.6% improvement, while maintaining real-time inference (30-47 FPS). Systematic ablation across five YOLO scales and nine DINOv3 variants reveals that Medium-scale architectures achieve optimal performance with DualP0P3 integration (55.77% mAP@0.5), while Small-scale requires Triple Integration (53.63%). The 2-4x inference overhead (21-33ms versus 8-16ms baseline) remains acceptable for field deployment on NVIDIA RTX 5090. DINO-YOLO establishes state-of-the-art performance for civil engineering datasets (<10K images) while preserving computational efficiency, providing practical solutions for construction safety monitoring and infrastructure inspection in data-constrained environments. |
| 2025-10-29 | [Automated Supervised Identification of Thunderstorm Ground Enhancements (TGEs)](http://arxiv.org/abs/2510.25125v1) | Davit Aslanyan | Thunderstorm Ground Enhancements (TGEs) are bursts of high-energy particle fluxes detected at Earth's surface, linked to the Relativistic Runaway Electron Avalanche (RREA) mechanism within thunderclouds. Accurate detection of TGEs is vital for advancing atmospheric physics and radiation safety, but event selection methods heavily rely on expert-defined thresholds. In this study, we use an automated supervised classification approach on a newly curated dataset of 2024 events from the Aragats Space Environment Center (ASEC). By combining a Tabular Prior-data Fitted Network (TabPFN) with SHAP-based interpretability, we attain 94.79% classification accuracy with 96% precision for TGEs. The analysis reveals data-driven thresholds for particle flux increases and environmental parameters that closely match the empirically established criteria used over the last 15 years. Our results demonstrate that modest but concurrent increases across multiple particle detectors, along with strong near-surface electric fields, are reliable indicators of TGEs. The framework we propose offers a scalable method for automated, interpretable TGE detection, with potential uses in real-time radiation hazard monitoring and multi-site atmospheric research. |
| 2025-10-29 | [SeeingEye: Agentic Information Flow Unlocks Multimodal Reasoning In Text-only LLMs](http://arxiv.org/abs/2510.25092v1) | Weijia Zhang, Zijia Liu et al. | Recent advances in text-only large language models (LLMs), such as DeepSeek-R1, demonstrate remarkable reasoning ability. However, these models remain fragile or entirely incapable when extended to multi-modal tasks. Existing approaches largely rely on single-form captions, which lack diversity and often fail to adapt across different types of Visual Question Answering (VQA) benchmarks. As a result, they provide no principled or efficient channel for transmitting fine-grained visual information. We introduce Seeing Eye, a modular framework that unlocks multimodal reasoning in text-only LLMs through an agent-based small VLM translator. This translator acts as a perception agent: it can invoke specialized tools (e.g., OCR and crop) and iteratively distill multimodal inputs into structured intermediate representations (SIRs) tailored to the question. These SIRs are then passed to the text-only LLM, which serves as a reasoning agent. Crucially, the translator and reasoner engage in multi-round feedback and interaction, enabling the extraction of targeted visual details and yielding more confident answers. Experiments on knowledge-intensive VQA benchmarks, including MMMU and MIA-Bench, demonstrate that Seeing Eye not only reduces inference cost but also surpasses much larger end-to-end VLMs. For example, an instantiation combining a 3B-parameter vision translator with an 8B-parameter language reasoner outperforms a monolithic 32B VLM on challenging knowledge-based questions. Our results highlight that decoupling perception from reasoning via agent information flow offers a scalable and plug-and-play pathway to multimodal reasoning, allowing strong text-only LLMs to fully leverage their reasoning capabilities. Code is available at: https://github.com/ulab-uiuc/SeeingEye |
| 2025-10-28 | [SLIP-SEC: Formalizing Secure Protocols for Model IP Protection](http://arxiv.org/abs/2510.24999v1) | Racchit Jain, Satya Lokam et al. | Large Language Models (LLMs) represent valuable intellectual property (IP), reflecting significant investments in training data, compute, and expertise. Deploying these models on partially trusted or insecure devices introduces substantial risk of model theft, making it essential to design inference protocols with provable security guarantees.   We present the formal framework and security foundations of SLIP, a hybrid inference protocol that splits model computation between a trusted and an untrusted resource. We define and analyze the key notions of model decomposition and hybrid inference protocols, and introduce formal properties including safety, correctness, efficiency, and t-soundness. We construct secure inference protocols based on additive decompositions of weight matrices, combined with masking and probabilistic verification techniques. We prove that these protocols achieve information-theoretic security against honest-but-curious adversaries, and provide robustness against malicious adversaries with negligible soundness error.   This paper focuses on the theoretical underpinnings of SLIP: precise definitions, formal protocols, and proofs of security. Empirical validation and decomposition heuristics appear in the companion SLIP paper. Together, the two works provide a complete account of securing LLM IP via hybrid inference, bridging both practice and theory. |
| 2025-10-28 | [Smooth path planning with safety margins using Piece-Wise Bezier curves](http://arxiv.org/abs/2510.24972v1) | Iancu Andrei, Marius Kloetzer et al. | In this paper, we propose a computationally efficient quadratic programming (QP) approach for generating smooth, $C^1$ continuous paths for mobile robots using piece-wise quadratic Bezier (PWB) curves. Our method explicitly incorporates safety margins within a structured optimization framework, balancing trajectory smoothness and robustness with manageable numerical complexity suitable for real-time and embedded applications. Comparative simulations demonstrate clear advantages over traditional piece-wise linear (PWL) path planning methods, showing reduced trajectory deviations, enhanced robustness, and improved overall path quality. These benefits are validated through simulations using a Pure-Pursuit controller in representative scenarios, highlighting the practical effectiveness and scalability of our approach for safe navigation. |
| 2025-10-28 | [RiddleBench: A New Generative Reasoning Benchmark for LLMs](http://arxiv.org/abs/2510.24932v1) | Deepon Halder, Alan Saji et al. | Large Language Models have demonstrated strong performance on many established reasoning benchmarks. However, these benchmarks primarily evaluate structured skills like quantitative problem-solving, leaving a gap in assessing flexible, multifaceted reasoning abilities that are central to human intelligence. These abilities require integrating logical deduction with spatial awareness and constraint satisfaction, which current evaluations do not measure well. To address this, we introduce RiddleBench, a benchmark of 1,737 challenging puzzles in English designed to probe these core reasoning capabilities. Evaluation of state-of-the-art models on RiddleBench shows fundamental weaknesses. Even top proprietary models like Gemini 2.5 Pro, o3, and Claude 4 Sonnet achieve accuracy just above 60% (60.30%, 63.37%, and 63.16%). Analysis further reveals deep failures, including hallucination cascades (accepting flawed reasoning from other models) and poor self-correction due to a strong self-confirmation bias. Their reasoning is also fragile, with performance degrading significantly when constraints are reordered or irrelevant information is introduced. RiddleBench functions as a diagnostic tool for these issues and as a resource for guiding the development of more robust and reliable language models. |
| 2025-10-28 | [A Hamilton-Jacobi Reachability Framework with Soft Constraints for Safety-Critical Systems](http://arxiv.org/abs/2510.24933v1) | Chams Eddine Mballo, Donggun Lee et al. | Traditional reachability methods provide formal guarantees of safety under bounded disturbances. However, they strictly enforce state constraints as inviolable, which can result in overly conservative or infeasible solutions in complex operational scenarios. Many constraints encountered in practice, such as bounds on battery state of charge in electric vehicles, recommended speed envelopes, and comfort constraints in passenger-carrying vehicles, are inherently soft. Soft constraints allow temporary violations within predefined safety margins to accommodate uncertainty and competing operational demands, albeit at a cost such as increased wear or higher operational expenses. This paper introduces a novel soft-constrained reachability framework that extends Hamilton-Jacobi reachability analysis for the formal verification of safety-critical systems subject to both hard and soft constraints. Specifically, the framework characterizes a subset of the state space, referred to as the soft-constrained reach-avoid set, from which the system is guaranteed to reach a desired set safely, under worst-case disturbances, while ensuring that cumulative soft-constraint violations remain within a user-specified budget. The framework comprises two principal components: (i) an augmented-state model with an auxiliary budget state that tracks soft-constraint violations, and (ii) a regularization-based approximation of the discontinuous Hamilton-Jacobi value function associated with the reach-avoid differential game studied herein. The effectiveness of the proposed framework is demonstrated through numerical examples involving the landing of a simple point-mass model and a fixed-wing aircraft executing an emergency descent, both under wind disturbances. The simulation results validate the framework's ability to simultaneously manage both hard and soft constraints in safety-critical settings |
| 2025-10-28 | [Delay Tolerant Control for Autonomous Driving Using CDOB](http://arxiv.org/abs/2510.24898v1) | Xincheng Cao, Haochong Chen et al. | With the rapid growth of autonomous vehicle technologies, effective path-tracking control has become a critical component in ensuring safety and efficiency in complex traffic scenarios. When a high level decision making agent generates a collision free path, a robust low level controller is required to precisely follow this trajectory. However, connected autonomous vehicles (CAV) are inherently affected by communication delays and computation delays, which significantly degrade the performance of conventional controllers such as PID or other more advanced controllers like disturbance observers (DOB). While DOB-based designs have shown effectiveness in rejecting disturbances under nominal conditions, their performance deteriorates considerably in the presence of unknown time delays. To address this challenge, this paper proposes a delay-tolerant communication disturbance observer (CDOB) framework for path-tracking control in delayed systems. The proposed CDOB compensates for the adverse effects of time delays, maintaining accurate trajectory tracking even under uncertain and varying delay conditions. It is shown through a simulation study that the proposed control architecture maintains close alignment with the reference trajectory across various scenarios, including single lane change, double-= lane change, and Elastic Band generated collision avoidance paths under various time delays. Simulation results further demonstrate that the proposed method outperforms conventional approaches in both tracking accuracy and delay robustness, making it well suited for autonomous driving applications. |

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



