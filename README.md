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
| 2026-03-17 | [Surg$Σ$: A Spectrum of Large-Scale Multimodal Data and Foundation Models for Surgical Intelligence](http://arxiv.org/abs/2603.16822v1) | Zhitao Zeng, Mengya Xu et al. | Surgical intelligence has the potential to improve the safety and consistency of surgical care, yet most existing surgical AI frameworks remain task-specific and struggle to generalize across procedures and institutions. Although multimodal foundation models, particularly multimodal large language models, have demonstrated strong cross-task capabilities across various medical domains, their advancement in surgery remains constrained by the lack of large-scale, systematically curated multimodal data. To address this challenge, we introduce Surg$Σ$, a spectrum of large-scale multimodal data and foundation models for surgical intelligence. At the core of this framework lies Surg$Σ$-DB, a large-scale multimodal data foundation designed to support diverse surgical tasks. Surg$Σ$-DB consolidates heterogeneous surgical data sources (including open-source datasets, curated in-house clinical collections and web-source data) into a unified schema, aiming to improve label consistency and data standardization across heterogeneous datasets. Surg$Σ$-DB spans 6 clinical specialties and diverse surgical types, providing rich image- and video-level annotations across 18 practical surgical tasks covering understanding, reasoning, planning, and generation, at an unprecedented scale (over 5.98M conversations). Beyond conventional multimodal conversations, Surg$Σ$-DB incorporates hierarchical reasoning annotations, providing richer semantic cues to support deeper contextual understanding in complex surgical scenarios. We further provide empirical evidence through recently developed surgical foundation models built upon Surg$Σ$-DB, illustrating the practical benefits of large-scale multimodal annotations, unified semantic design, and structured reasoning annotations for improving cross-task generalization and interpretability. |
| 2026-03-17 | [Differential Harm Propensity in Personalized LLM Agents: The Curious Case of Mental Health Disclosure](http://arxiv.org/abs/2603.16734v1) | Caglar Yildirim | Large language models (LLMs) are increasingly deployed as tool-using agents, shifting safety concerns from harmful text generation to harmful task completion. Deployed systems often condition on user profiles or persistent memory, yet agent safety evaluations typically ignore personalization signals. To address this gap, we investigated how mental health disclosure, a sensitive and realistic user-context cue, affects harmful behavior in agentic settings. Building on the AgentHarm benchmark, we evaluated frontier and open-source LLMs on multi-step malicious tasks (and their benign counterparts) under controlled prompt conditions that vary user-context personalization (no bio, bio-only, bio+mental health disclosure) and include a lightweight jailbreak injection. Our results reveal that harmful task completion is non-trivial across models: frontier lab models (e.g., GPT 5.2, Claude Sonnet 4.5, Gemini 3-Pro) still complete a measurable fraction of harmful tasks, while an open model (DeepSeek 3.2) exhibits substantially higher harmful completion. Adding a bio-only context generally reduces harm scores and increases refusals. Adding an explicit mental health disclosure often shifts outcomes further in the same direction, though effects are modest and not uniformly reliable after multiple-testing correction. Importantly, the refusal increase also appears on benign tasks, indicating a safety--utility trade-off via over-refusal. Finally, jailbreak prompting sharply elevates harm relative to benign conditions and can weaken or override the protective shift induced by personalization. Taken together, our results indicate that personalization can act as a weak protective factor in agentic misuse settings, but it is fragile under minimal adversarial pressure, highlighting the need for personalization-aware evaluations and safeguards that remain robust across user-context conditions. |
| 2026-03-17 | [Assessment of Latent Pedestrian--Vehicle Interaction Risk Profiles at Midblock Crossing in VR](http://arxiv.org/abs/2603.16688v1) | Rulla Al-Haideri, Bilal Farooq et al. | Pedestrian safety at midblock crossings is a critical concern in mixed traffic environments where autonomous vehicles (AVs) and human-driven vehicles (HDVs) share the road. Pedestrians often infer intent from vehicle motion in AV encounters, making them vulnerable to small shifts in conflict margins. This study investigates whether virtual reality (VR) crossing sessions separate into distinct interaction risk profiles and whether AV-only sessions shift profile prevalence compared to HDV-only sessions. Using large-scale immersive VR experiments from Toronto, Canada, and Newcastle, England, we compute surrogate safety measures (SSMs) and apply latent profile analysis (LPA) to identify distinct pedestrian crossing stances, ranging from risk-accepting to highly cautious. Key findings show that Newcastle exhibits a higher prevalence of high-urgency risk profiles in AV-only sessions, indicating that AVs contribute to higher-risk encounters. In contrast, Toronto shows no significant difference between AV-only and HDV-only sessions, suggesting that contextual factors influence the impact of AVs on pedestrian safety. |
| 2026-03-17 | [Bio-inspired metaheuristic optimization for hierarchical architecture design of industrial control systems](http://arxiv.org/abs/2603.16617v1) | Ruslan Zakirzyanov | Automated process control systems (APCS) are widely used in modern industrial enterprises. They address three key objectives: ensuring the required quality of manufactured products, ensuring process safety for people and the environment, and reducing capital and operating costs. At large industrial enterprises, APCSs are typically geographically distributed and characterized by a large number of monitored parameters. Such systems often consist of several subsystems built using various technical means and serving different functional purposes. APCSs usually have a hierarchical structure consisting of several levels, where each level hosts commercially available technical devices with predetermined characteristics. This article examines the engineering problem of selecting an optimal software and hardware structure for a distributed process control system applied to a continuous process in the chemical industry. A formal formulation of the optimization problem is presented, in which the hierarchical structure of the system is represented as an acyclic graph. Optimization criteria and constraints are defined. A solution method based on a metaheuristic ant colony optimization algorithm, widely used for this class of problems, is proposed. A brief overview of the developed software tool used to solve a number of numerical examples is provided. The experimental results are discussed, along with parameter selection and possible algorithm modifications aimed at improving solution quality. Information on the verification of the control system implemented using the selected software and hardware structure is presented, and directions for further research are outlined. |
| 2026-03-17 | [UrbanFlow-3K: A Dataset of 3,000 Lattice-Boltzmann Simulations of Random Building Layouts](http://arxiv.org/abs/2603.16554v1) | Hojin Lee, Andreas Lintermann et al. | The analysis of flow around buildings has gained significant research interest across various domains, including pedestrian safety, pollutant dispersion, natural ventilation, and building energy efficiency. While these domains frequently include high-resolution computational fluid dynamics (CFD) data, predicting urban flow fields with machine learning (ML) models has emerged as a promising approach to overcome the prohibitive costs of CFD simulations. However, the availability of open-source datasets for training such ML models remains scarce. In particular, publicly available two-dimensional datasets of urban flow fields are nearly non-existent, despite their potential value for early development and debugging stages of data-driven models, before scaling to computationally expensive three-dimensional datasets. To bridge this gap, this study presents a comprehensive dataset consisting of 3,000 two-dimensional urban flow simulations conducted using a lattice-Boltzmann method across three distinct Reynolds numbers. The dataset contains the time-averaged velocity fields. A key feature of this dataset is its high geometric diversity: each layout incorporates between three and six buildings with randomized sizes, positions, and rotation angles ranging from 0° to 90°. This extensive variability enables the dataset to capture several critical flow characteristics, including wake formation, flow acceleration, shielding effects, and recirculation zones, across a wide range of orchestrated urban canopies. The large sample size and consistent simulation setup make the dataset particularly suitable for developing and benchmarking ML architectures. In addition, the dataset can support transfer-learning strategies in which models trained on large two-dimensional datasets are adapted to smaller and more computationally expensive three-dimensional datasets. |
| 2026-03-17 | [ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace](http://arxiv.org/abs/2603.16550v1) | Alexander Prutsch, David Schinagl et al. | Accurate trajectory prediction can improve General Aviation safety in non-towered terminal airspace, where high traffic density increases accident risk. We present ASCENT, a lightweight transformer-based model for multi-modal 3D aircraft trajectory forecasting, which integrates domain-aware 3D coordinate normalization and parameterized predictions. ASCENT employs a transformer-based motion encoder and a query-based decoder, enabling the generation of diverse maneuver hypotheses with low latency. Experiments on the TrajAir and TartanAviation datasets demonstrate that our model outperforms prior baselines, as the encoder effectively captures motion dynamics and the decoder aligns with structured aircraft traffic patterns. Furthermore, ablation studies confirm the contributions of the decoder design, coordinate-frame modeling, and parameterized outputs. These results establish ASCENT as an effective approach for real-time aircraft trajectory prediction in non-towered terminal airspace. |
| 2026-03-17 | [ExpressMind: A Multimodal Pretrained Large Language Model for Expressway Operation](http://arxiv.org/abs/2603.16495v1) | Zihe Wang, Yihuan Wang et al. | The current expressway operation relies on rule-based and isolated models, which limits the ability to jointly analyze knowledge across different systems. Meanwhile, Large Language Models (LLMs) are increasingly applied in intelligent transportation, advancing traffic models from algorithmic to cognitive intelligence. However, general LLMs are unable to effectively understand the regulations and causal relationships of events in unconventional scenarios in the expressway field. Therefore, this paper constructs a pre-trained multimodal large language model (MLLM) for expressways, ExpressMind, which serves as the cognitive core for intelligent expressway operations. This paper constructs the industry's first full-stack expressway dataset, encompassing traffic knowledge texts, emergency reasoning chains, and annotated video events to overcome data scarcity. This paper proposes a dual-layer LLM pre-training paradigm based on self-supervised training and unsupervised learning. Additionally, this study introduces a Graph-Augmented RAG framework to dynamically index the expressway knowledge base. To enhance reasoning for expressway incident response strategies, we develop a RL-aligned Chain-of-Thought (RL-CoT) mechanism that enforces consistency between model reasoning and expert problem-solving heuristics for incident handling. Finally, ExpressMind integrates a cross-modal encoder to align the dynamic feature sequences under the visual and textual channels, enabling it to understand traffic scenes in both video and image modalities. Extensive experiments on our newly released multi-modal expressway benchmark demonstrate that ExpressMind comprehensively outperforms existing baselines in event detection, safety response generation, and complex traffic analysis. The code and data are available at: https://wanderhee.github.io/ExpressMind/. |
| 2026-03-17 | [Unlearning for One-Step Generative Models via Unbalanced Optimal Transport](http://arxiv.org/abs/2603.16489v1) | Hyundo Choi, Junhyeong An et al. | Recent advances in one-step generative frameworks, such as flow map models, have significantly improved the efficiency of image generation by learning direct noise-to-data mappings in a single forward pass. However, machine unlearning for ensuring the safety of these powerful generators remains entirely unexplored. Existing diffusion unlearning methods are inherently incompatible with these one-step models, as they rely on a multi-step iterative denoising process. In this work, we propose UOT-Unlearn, a novel plug-and-play class unlearning framework for one-step generative models based on the Unbalanced Optimal Transport (UOT). Our method formulates unlearning as a principled trade-off between a forget cost, which suppresses the target class, and an $f$-divergence penalty, which preserves overall generation fidelity via relaxed marginal constraints. By leveraging UOT, our method enables the probability mass of the forgotten class to be smoothly redistributed to the remaining classes, rather than collapsing into low-quality or noise-like samples. Experimental results on CIFAR-10 and ImageNet-256 demonstrate that our framework achieves superior unlearning success (PUL) and retention quality (u-FID), significantly outperforming baselines. |
| 2026-03-17 | [Visual Distraction Undermines Moral Reasoning in Vision-Language Models](http://arxiv.org/abs/2603.16445v1) | Xinyi Yang, Chenheng Xu et al. | Moral reasoning is fundamental to safe Artificial Intelligence (AI), yet ensuring its consistency across modalities becomes critical as AI systems evolve from text-based assistants to embodied agents. Current safety techniques demonstrate success in textual contexts, but concerns remain about generalization to visual inputs. Existing moral evaluation benchmarks rely on textonly formats and lack systematic control over variables that influence moral decision-making. Here we show that visual inputs fundamentally alter moral decision-making in state-of-the-art (SOTA) Vision-Language Models (VLMs), bypassing text-based safety mechanisms. We introduce Moral Dilemma Simulation (MDS), a multimodal benchmark grounded in Moral Foundation Theory (MFT) that enables mechanistic analysis through orthogonal manipulation of visual and contextual variables. The evaluation reveals that the vision modality activates intuition-like pathways that override the more deliberate and safer reasoning patterns observed in text-only contexts. These findings expose critical fragilities where language-tuned safety filters fail to constrain visual processing, demonstrating the urgent need for multimodal safety alignment. |
| 2026-03-17 | [Poisoning the Pixels: Revisiting Backdoor Attacks on Semantic Segmentation](http://arxiv.org/abs/2603.16405v1) | Guangsheng Zhang, Huan Tian et al. | Semantic segmentation models are widely deployed in safety-critical applications such as autonomous driving, yet their vulnerability to backdoor attacks remains largely underexplored. Prior segmentation backdoor studies transfer threat settings from existing image classification tasks, focusing primarily on object-to-background mis-segmentation. In this work, we revisit the threats by systematically examining backdoor attacks tailored to semantic segmentation. We identify four coarse-grained attack vectors (Object-to-Object, Object-to-Background, Background-to-Object, and Background-to-Background attacks), as well as two fine-grained vectors (Instance-Level and Conditional attacks). To formalize these attacks, we introduce BADSEG, a unified framework that optimizes trigger designs and applies label manipulation strategies to maximize attack performance while preserving victim model utility. Extensive experiments across diverse segmentation architectures on benchmark datasets demonstrate that BADSEG achieves high attack effectiveness with minimal impact on clean samples. We further evaluate six representative defenses and find that they fail to reliably mitigate our attacks, revealing critical gaps in current defenses. Finally, we demonstrate that these vulnerabilities persist in recent emerging architectures, including transformer-based networks and the Segment Anything Model (SAM), thereby compromising their security. Our work reveals previously overlooked security vulnerabilities in semantic segmentation, and motivates the development of defenses tailored to segmentation-specific threat models. |
| 2026-03-17 | [Fanar 2.0: Arabic Generative AI Stack](http://arxiv.org/abs/2603.16397v1) | FANAR TEAM, Ummar Abbas et al. | We present Fanar 2.0, the second generation of Qatar's Arabic-centric Generative AI platform. Sovereignty is a first-class design principle: every component, from data pipelines to deployment infrastructure, was designed and operated entirely at QCRI, Hamad Bin Khalifa University. Fanar 2.0 is a story of resource-constrained excellence: the effort ran on 256 NVIDIA H100 GPUs, with Arabic having only ~0.5% of web data despite 400 million native speakers. Fanar 2.0 adopts a disciplined strategy of data quality over quantity, targeted continual pre-training, and model merging to achieve substantial gains within these constraints. At the core is Fanar-27B, continually pre-trained from a Gemma-3-27B backbone on a curated corpus of 120 billion high-quality tokens across three data recipes. Despite using 8x fewer pre-training tokens than Fanar 1.0, it delivers substantial benchmark improvements: Arabic knowledge (+9.1 pts), language (+7.3 pts), dialects (+3.5 pts), and English capability (+7.6 pts). Beyond the core LLM, Fanar 2.0 introduces a rich stack of new capabilities. FanarGuard is a state-of-the-art 4B bilingual moderation filter for Arabic safety and cultural alignment. The speech family Aura gains a long-form ASR model for hours-long audio. Oryx vision family adds Arabic-aware image and video understanding alongside culturally grounded image generation. An agentic tool-calling framework enables multi-step workflows. Fanar-Sadiq utilizes a multi-agent architecture for Islamic content. Fanar-Diwan provides classical Arabic poetry generation. FanarShaheen delivers LLM-powered bilingual translation. A redesigned multi-layer orchestrator coordinates all components through intent-aware routing and defense-in-depth safety validation. Taken together, Fanar 2.0 demonstrates that sovereign, resource-constrained AI development can produce systems competitive with those built at far greater scale. |
| 2026-03-17 | [Encoding Predictability and Legibility for Style-Conditioned Diffusion Policy](http://arxiv.org/abs/2603.16368v1) | Adrien Jacquet Crétides, Mouad Abrini et al. | Striking a balance between efficiency and transparent motion is a core challenge in human-robot collaboration, as highly expressive movements often incur unnecessary time and energy costs. In collaborative environments, legibility allows a human observer a better understanding of the robot's actions, increasing safety and trust. However, these behaviors result in sub-optimal and exaggerated trajectories that are redundant in low-ambiguity scenarios where the robot's goal is already obvious. To address this trade-off, we propose Style-Conditioned Diffusion Policy (SCDP), a modular framework that constrains the trajectory generation of a pre-trained diffusion model toward either legibility or efficiency based on the environment's configuration. Our method utilizes a post-training pipeline that freezes the base policy and trains a lightweight scene encoder and conditioning predictor to modulate the diffusion process. At inference time, an ambiguity detection module activates the appropriate conditioning, prioritizing expressive motion only for ambiguous goals and reverting to efficient paths otherwise. We evaluate SCDP on manipulation and navigation tasks, and results show that it enhances legibility in ambiguous settings while preserving optimal efficiency when legibility is unnecessary, all without retraining the base policy. |
| 2026-03-17 | [Toward Experimentation-as-a-Service in 5G/6G: The Plaza6G Prototype for AI-Assisted Trials](http://arxiv.org/abs/2603.16356v1) | Sergio Barrachina-Muñoz, Marc Carrascosa-Zamacois et al. | This paper presents Plaza6G, the first operational Experiment-as-a-Service (ExaS) platform unifying cloud resources with next-generation wireless infrastructure. Developed at CTTC in Barcelona, Plaza6G integrates GPU-accelerated compute clusters, multiple 5G cores, both open-source (e.g., Free5GC) and commercial (e.g., Cumucore), programmable RANs, and physical or emulated user equipment under unified orchestration. In Plaza6G, the experiment design requires minimal expertise as it is expressed in natural language via a web portal or a REST API. The web portal and REST API are enhanced with a Large Language Model (LLM)-based assistant, which employs retrieval-augmented generation (RAG) for up-to-date experiment knowledge and Low-Rank Adaptation (LoRA) for continuous domain fine-tuning. Over-the-air (OTA) trials leverage a four-chamber anechoic facility and a dual-site outdoor 5G network operating in sub-6~GHz and mmWave bands. Demonstrations include automated CI/CD integration with sub-ten-minute setup and interactive OTA testing under programmable propagation conditions. Machine-readable experiment descriptors ensure reproducibility, while future work targets policy-aware orchestration, safety validation, and federated testbed integration toward open, reproducible wireless experimentation. |
| 2026-03-17 | [Learning Human-Object Interaction for 3D Human Pose Estimation from LiDAR Point Clouds](http://arxiv.org/abs/2603.16343v1) | Daniel Sungho Jung, Dohee Cho et al. | Understanding humans from LiDAR point clouds is one of the most critical tasks in autonomous driving due to its close relationships with pedestrian safety, yet it remains challenging in the presence of diverse human-object interactions and cluttered backgrounds. Nevertheless, existing methods largely overlook the potential of leveraging human-object interactions to build robust 3D human pose estimation frameworks. There are two major challenges that motivate the incorporation of human-object interaction. First, human-object interactions introduce spatial ambiguity between human and object points, which often leads to erroneous 3D human keypoint predictions in interaction regions. Second, there exists severe class imbalance in the number of points between interacting and non-interacting body parts, with the interaction-frequent regions such as hand and foot being sparsely observed in LiDAR data. To address these challenges, we propose a Human-Object Interaction Learning (HOIL) framework for robust 3D human pose estimation from LiDAR point clouds. To mitigate the spatial ambiguity issue, we present human-object interaction-aware contrastive learning (HOICL) that effectively enhances feature discrimination between human and object points, particularly in interaction regions. To alleviate the class imbalance issue, we introduce contact-aware part-guided pooling (CPPool) that adaptively reallocates representational capacity by compressing overrepresented points while preserving informative points from interacting body parts. In addition, we present an optional contact-based temporal refinement that refines erroneous per-frame keypoint estimates using contact cues over time. As a result, our HOIL effectively leverages human-object interaction to resolve spatial ambiguity and class imbalance in interaction regions. Codes will be released. |
| 2026-03-17 | [Learning to Predict, Discover, and Reason in High-Dimensional Discrete Event Sequences](http://arxiv.org/abs/2603.16313v1) | Hugo Math | Electronic control units (ECUs) embedded within modern vehicles generate a large number of asynchronous events known as diagnostic trouble codes (DTCs). These discrete events form complex temporal sequences that reflect the evolving health of the vehicle's subsystems. In the automotive industry, domain experts manually group these codes into higher-level error patterns (EPs) using Boolean rules to characterize system faults and ensure safety. However, as vehicle complexity grows, this manual process becomes increasingly costly, error-prone, and difficult to scale. Notably, the number of unique DTCs in a modern vehicle is on the same order of magnitude as the vocabulary of a natural language, often numbering in the tens of thousands. This observation motivates a paradigm shift: treating diagnostic sequences as a language that can be modeled, predicted, and ultimately explained. Traditional statistical approaches fail to capture the rich dependencies and do not scale to high-dimensional datasets characterized by thousands of nodes, large sample sizes, and long sequence lengths. Specifically, the high cardinality of categorical event spaces in industrial logs poses a significant challenge, necessitating new machine learning architectures tailored to such event-driven systems. This thesis addresses automated fault diagnostics by unifying event sequence modeling, causal discovery, and large language models (LLMs) into a coherent framework for high-dimensional event streams. It is structured in three parts, reflecting a progressive transition from prediction to causal understanding and finally to reasoning for vehicle diagnostics. Consequently, we introduce several Transformer-based architectures for predictive maintenance, scalable sample- and population-level causal discovery frameworks and a multi-agent system that automates the synthesis of Boolean EP rules. |
| 2026-03-17 | [VisBrowse-Bench: Benchmarking Visual-Native Search for Multimodal Browsing Agents](http://arxiv.org/abs/2603.16289v1) | Zhengbo Zhang, Jinbo Su et al. | The rapid advancement of Multimodal Large Language Models (MLLMs) has enabled browsing agents to acquire and reason over multimodal information in the real world. But existing benchmarks suffer from two limitations: insufficient evaluation of visual reasoning ability and the neglect of native visual information of web pages in the reasoning chains. To address these challenges, we introduce a new benchmark for visual-native search, VisBrowse-Bench. It contains 169 VQA instances covering multiple domains and evaluates the models' visual reasoning capabilities during the search process through multimodal evidence cross-validation via text-image retrieval and joint reasoning. These data were constructed by human experts using a multi-stage pipeline and underwent rigorous manual verification. We additionally propose an agent workflow that can effectively drive the browsing agent to actively collect and reason over visual information during the search process. We comprehensively evaluated both open-source and closed-source models in this workflow. Experimental results show that even the best-performing model, Claude-4.6-Opus only achieves an accuracy of 47.6%, while the proprietary Deep Research model, o3-deep-research only achieves an accuracy of 41.1%. The code and data can be accessed at: https://github.com/ZhengboZhang/VisBrowse-Bench |
| 2026-03-17 | [MOSAIC: Composable Safety Alignment with Modular Control Tokens](http://arxiv.org/abs/2603.16210v1) | Jingyu Peng, Hongyu Chen et al. | Safety alignment in large language models (LLMs) is commonly implemented as a single static policy embedded in model parameters. However, real-world deployments often require context-dependent safety rules that vary across users, regions, and applications. Existing approaches struggle to provide such conditional control: parameter-level alignment entangles safety behaviors with general capabilities, while prompt-based methods rely on natural language instructions that provide weak enforcement. We propose MOSAIC, a modular framework that enables compositional safety alignment through learnable control tokens optimized over a frozen backbone model. Each token represents a safety constraint and can be flexibly activated and composed at inference time. To train compositional tokens efficiently, we introduce order-based task sampling and a distribution-level alignment objective that mitigates over-refusal. Experiments show that MOSAIC achieves strong defense performance with substantially lower over-refusal while preserving model utility. |
| 2026-03-17 | [Are Large Language Models Truly Smarter Than Humans?](http://arxiv.org/abs/2603.16197v1) | Eshwar Reddy M, Sourav Karmakar | Public leaderboards increasingly suggest that large language models (LLMs) surpass human experts on benchmarks spanning academic knowledge, law, and programming. Yet most benchmarks are fully public, their questions widely mirrored across the internet, creating systematic risk that models were trained on the very data used to evaluate them. This paper presents three complementary experiments forming a rigorous multi-method contamination audit of six frontier LLMs: GPT-4o, GPT-4o-mini, DeepSeek-R1, DeepSeek-V3, Llama-3.3-70B, and Qwen3-235B. Experiment 1 applies a lexical contamination detection pipeline to 513 MMLU questions across all 57 subjects, finding an overall contamination rate of 13.8% (18.1% in STEM, up to 66.7% in Philosophy) and estimated performance gains of +0.030 to +0.054 accuracy points by category. Experiment 2 applies a paraphrase and indirect-reference diagnostic to 100 MMLU questions, finding accuracy drops by an average of 7.0 percentage points under indirect reference, rising to 19.8 pp in both Law and Ethics. Experiment 3 applies TS-Guessing behavioral probes to all 513 questions and all six models, finding that 72.5% trigger memorization signals far above chance, with DeepSeek-R1 displaying a distributed memorization signature (76.6% partial reconstruction, 0% verbatim recall) that explains its anomalous Experiment 2 profile. All three experiments converge on the same contamination ranking: STEM > Professional > Social Sciences > Humanities. |
| 2026-03-17 | [PanguMotion: Continuous Driving Motion Forecasting with Pangu Transformers](http://arxiv.org/abs/2603.16196v1) | Quanhao Ren, Yicheng Li et al. | Motion forecasting is a core task in autonomous driving systems, aiming to accurately predict the future trajectories of surrounding agents to ensure driving safety. Existing methods typically process discrete driving scenes independently, neglecting the temporal continuity and historical context correlations inherent in real-world driving environments. This paper proposes PanguMotion, a motion forecasting framework for continuous driving scenarios that integrates Transformer blocks from the Pangu-1B large language model as feature enhancement modules into autonomous driving motion prediction architectures. We conduct experiments on the Argoverse 2 datasets processed by the RealMotion data reorganization strategy, transforming each independent scene into a continuous sequence to mimic real-world driving scenarios. |
| 2026-03-17 | [Structured Semantic Cloaking for Jailbreak Attacks on Large Language Models](http://arxiv.org/abs/2603.16192v1) | Xiaobing Sun, Perry Lam et al. | Modern LLMs employ safety mechanisms that extend beyond surface-level input filtering to latent semantic representations and generation-time reasoning, enabling them to recover obfuscated malicious intent during inference and refuse accordingly, and rendering many surface-level obfuscation jailbreak attacks ineffective. We propose Structured Semantic Cloaking (S2C), a novel multi-dimensional jailbreak attack framework that manipulates how malicious semantic intent is reconstructed during model inference. S2C strategically distributes and reshapes semantic cues such that full intent consolidation requires multi-step inference and long-range co-reference resolution within deeper latent representations. The framework comprises three complementary mechanisms: (1) Contextual Reframing, which embeds the request within a plausible high-stakes scenario to bias the model toward compliance; (2) Content Fragmentation, which disperses the semantic signature of the request across disjoint prompt segments; and (3) Clue-Guided Camouflage, which disguises residual semantic cues while embedding recoverable markers that guide output generation. By delaying and restructuring semantic consolidation, S2C degrades safety triggers that depend on coherent or explicitly reconstructed malicious intent at decoding time, while preserving sufficient instruction recoverability for functional output generation. We evaluate S2C across multiple open-source and proprietary LLMs using HarmBench and JBB-Behaviors, where it improves Attack Success Rate (ASR) by 12.4% and 9.7%, respectively, over the current SOTA. Notably, S2C achieves substantial gains on GPT-5-mini, outperforming the strongest baseline by 26% on JBB-Behaviors. We also analyse which combinations perform best against broad families of models, and characterise the trade-off between the extent of obfuscation versus input recoverability on jailbreak success. |

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



