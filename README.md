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
| 2026-09-08 | [Online, Reachability-Aware, Sampling-Based Motion Planning](http://arxiv.org/abs/2609.09073v1) | Brendan Gould, Zhiyuan Zhang et al. | Sampling-Based Model-Predictive Control (MPC) algorithms are a flexible class of controllers used for navigation on a wide range of robotic systems. Historically, such approaches have lacked hard safety guarantees, a shortcoming which we remedy in this work by computing guaranteed reachable-set overapproximations online with a fast, interval-based pipeline. We show that our method achieves similar performance to a state-of-the-art reachability-based planner without the need for the expensive pre-computation step, and can be scaled to systems that are infeasible using existing approaches. Finally, we demonstrate that our technique reduces safety violations by over 99% in a racing simulation and successfully controls a model racecar on real hardware experiments without crashes. |
| 2026-09-08 | [NERVE Attacks: Breaking AI-Powered Brain-Computer Interfaces](http://arxiv.org/abs/2609.08971v1) | Zahra Tarkhani, Georgios Akkogiounoglou et al. | The rapid integration of AI into human-centred systems such as Brain-Computer Interfaces (BCIs) has created a poorly understood attack surface linking neural signals to physical systems. Exploits in this domain threaten cognitive autonomy, mental privacy, and physical safety, from neural data exfiltration to malicious control of BCI-tethered devices. We introduce the NERVE Attacks class, a systematic characterisation of five orthogonal attack dimensions that together span the complete BCI stack: Neuro-mimetic Forgery (N), Evasion via Desynchronization (E), Replay-based Hijacking (R), Vein Tapping (V), and Embedded Backdoors (E). To evaluate this class, we present EEGle, an AI-assisted extensible framework for systematic BCI security analysis. Our evaluation uncovers 17 novel neuro-specific attack instances and reveals a stealth-effectiveness spectrum unique to BCI backdoor design. We also show that generative AI lowers the barrier to entry for non-expert attackers and provide EEGle to the community for building and verifying the security of these deeply personal devices. |
| 2026-09-08 | [PlannerForge: LLM Agents for Scenario-Based Testing of Motion Planners in Autonomous Driving](http://arxiv.org/abs/2609.08965v1) | Yuan Gao, Sebastian Müller et al. | Ensuring the safety of autonomous driving is a critical challenge. Scenario-based testing is a systematic process used to validate Autonomous Driving Systems (ADSs), but it remains a fragmented modular pipeline in which scenario generation, retrieval, modification, ADS execution, and results analysis are performed by separate tools with little interaction. Large Language Model (LLM) agents have shown promise across ADS sub-systems such as perception, planning, and control. However, no prior work covers the whole scenario-based testing pipeline for ADSs with a unified LLM-agent framework. We present PlannerForge, an LLM-agent framework that extends all scenario-based testing stages (from Scenario Generation to ADS Assessment) and adds two further LLM-enhanced stages: ADS Enhancement and ADS Benchmarking. We evaluate PlannerForge with 10 off-the-shelf LLMs across all tasks (Generation, Selection, Modification, Module Routing, Planner Testing, and Enhancement) under 5 prompt conditions. Best-per-task scores range from 0.88 to 1.00, and open-source 20-35B backends match commercial APIs on most tasks. Open-source models such as Qwen3.6:35B match commercial APIs on three of the five tasks. Chaining the modules end-to-end retains 83% / 78% of seed queries (commercial / open). It outperforms Scenario Factory 2.0 (Finkeldei et al., 2025) on natural-language generation (193 vs. 144 executable of 200) and realises 92-96% of requested city, road and vehicle attributes. It outperforms BM25 (Robertson and Zaragoza, 2009) at rank 1 selection (92.0% vs. 67.5%) and From-Words-to-Collisions (Gao et al., 2025) on physically valid edits (>=94% vs. 31%). At N=400, cost-tuning lifts planner success from 50.4% to 70.2% and cuts collisions from 19.0% to 8.4%, without domain-specific fine-tuning. |
| 2026-09-08 | [Towards Standardized Evaluation of GPU Memory Safety with GMSBench](http://arxiv.org/abs/2609.08871v1) | Saurabh Singh, Jaewon Lee et al. | As GPUs become increasingly integral to high-performance computing and machine learning, ensuring memory safety in GPU programs has become crucial for reliable and secure execution. However, evaluating GPU memory safety techniques remains challenging due to the lack of comprehensive and standardized benchmarks. In this paper, we present GMSBench, a GPU memory safety benchmark designed to evaluate a broad range of memory safety violations across different GPU memory spaces and execution scenarios. GMSBench comprises 149 self-contained CUDA tests spanning spatial, temporal, and concurrency errors. The suite provides a standardized foundation for the evaluation and comparative analysis of GPU memory safety mechanisms and helps expose gaps in their detection coverage. We demonstrate the utility of GMSBench by evaluating Compute Sanitizer, a widely used GPU memory error detection tool across multiple GPU architectures. |
| 2026-09-08 | [Leveraging Visual and Geometric Priors for Metric-scale and Complete Vehicle Gaussian Reconstruction from Limited Views](http://arxiv.org/abs/2609.08841v1) | Jinyu Miao, Jiusi Li et al. | High-fidelity vehicle assets are essential for controllable traffic scene generation, particularly for synthesizing rare and safety-critical long-tail scenarios. However, reconstructing a reusable vehicle representation from in-the-wild onboard images remains challenging for two reasons. First, image-to-3D generation methods generally produce models without reliable metric scale. Second, onboard cameras usually observe only one side of a target vehicle, making conventional multi-view reconstruction incomplete on unobserved regions. To solve these problems, we propose a feed-forward vehicle asset reconstruction method, which leverages two complementary priors to reconstruct 3D Gaussian representations for vehicles using sparse one-sided observations. To achieve metric-scale reconstruction, a visual foundation model is first utilized to serve as a visual prior for Gaussian initialization. The Gaussian attributes are then estimated by a learnable encoder-decoder module. A symmetry-aware cloning strategy is presented to complete the unobserved side directly in Gaussian space, which exploits the bilateral structure of vehicles as a geometric prior. Experiments on the public dataset demonstrate that the proposed method significantly outperforms existing approaches in both vehicle asset completeness and geometric accuracy. |
| 2026-09-08 | [What AI Benchmarks Actually Measure: Adapting Convergent and Discriminant Validity to Interrogate Fifty-Six AI Benchmarks](http://arxiv.org/abs/2609.08812v1) | Meera Desai, Sang T. Truong et al. | Benchmarks play a central role in the development and governance of models, yet it is often unclear whether they actually measure the concepts they purport to measure (e.g., reasoning, refusal). We adapt convergent and discriminant validity from the social sciences into an approach for interrogating AI benchmarks, applying it to 56 capability and safety benchmarks across 53 models. We label benchmarks with substantively similar purported concepts to a shared assigned concept, and ask whether model rankings on benchmarks with the same assigned concept correlate more strongly than rankings on benchmarks with different assigned concepts. We ask analogous questions at the item level using item response theory (IRT) models. We find that correlations between model rankings on benchmarks with the same assigned safety concepts are often weak, suggesting these concepts may be conceptualized inconsistently across benchmarks. For assigned capability concepts (e.g., reasoning, knowledge), model rankings are often as strongly correlated among benchmarks with the same assigned concept as between benchmarks with different assigned concepts, suggesting these capability concepts may not discriminate well from one another. In some cases, benchmarks that share design elements (e.g., score format) correlate more strongly than benchmarks with the same assigned concept. Finally, some individual benchmarks correlate more strongly with benchmarks assigned a different concept than with benchmarks sharing their own assigned concept, suggesting they may measure a different concept than they purport to. For example, BBQ-accuracy correlates more strongly with benchmarks labeled reasoning than with benchmarks that share its assigned concept, bias. To support future empirical work on benchmark validity, we release our extensive dataset of model outputs and scores at the item- and benchmark-level. |
| 2026-09-08 | [Real-time Puncture Detection and Recovery for Pneumatic Soft Actuators](http://arxiv.org/abs/2609.08804v1) | Tejonidhi R. Deshpande, Tingyu Cheng et al. | Soft robots offer safe and adaptive interaction with humans and unstructured environments through their inherent ability to deform and comply. Pneumatic actuators are one way to build soft robots. They are typically made from soft silicone materials and are especially effective for driving such systems, enabling smooth and adaptable motion. However, their compliant nature also makes them vulnerable to mechanical failures like punctures and tears, limiting practical deployment. To address this, we propose a puncture detection system for soft actuators using motion data from a single inertial measurement unit. Extracted features are used to train anomaly detectors for puncture detection and non-linear models to estimate severity. We also introduce a multi-chamber pneumatic soft bending actuator capable of diverse configurations via selective chamber inflation. Our algorithm identifies the punctured chamber and provides a severity score using a chamber perturbation scheme. Anomaly detectors are trained on normal operation data and detect damage through reconstruction errors, while severity is estimated by a separate model trained under slightly modified conditions. Finally, we demonstrate a failure recovery strategy to maintain actuation force post-failure. This approach enhances the reliability and safety of soft robotic systems through real-time, data-driven damage detection. |
| 2026-09-08 | [Graph-Based Safe Reinforcement Learning for Multi-Agent Systems with Time-Varying Topology](http://arxiv.org/abs/2609.08802v1) | Xiao Sizhe, Dong Lijing et al. | This paper presents a graph-based safe multi-agent reinforcement learning (MARL) framework for cooperative navigation with time-varying topology. To address the critical challenge of ensuring safety in environments with sensing constraints, a safety-decoupled mechanism is introduced through a Control Barrier-Like Function (CBLF) action screening layer. This mechanism bridges the gap between discrete LiDAR perception and continuous safety constraints, ensuring that physical safety constraints are strictly satisfied regardless of the learning progress. Building upon this safety foundation, a unified structural architecture is proposed, integrating a attention-based actor and a Graph Attention Network (GAT) centralized critic. The actor utilizes a value vector reconstruction mechanism that explicitly encodes relative geometric relations through a collaborative tracking error matrix, enabling scale-insensitive policy learning under time-varying communication topologies. Meanwhile, the GAT-based critic models evolving interaction structures for accurate global value estimation. The proposed framework is validated on real differential-drive robot platforms, and experimental results demonstrate superior stability and safety in dynamic scenarios with limited fields-of-view. |
| 2026-09-08 | [Silent Revision: Measuring Undisclosed Change in the Safety Frameworks of Frontier AI Developers](http://arxiv.org/abs/2609.08789v1) | Louis Yiven Zhu | Frontier AI developers publish safety frameworks that commit them to evidencing whether their models are dangerous. The European Union and California now treat these documents as instruments of accountability, and both already impose duties on their revision. Neither requires the revision to be legible, in the sense that a reader could learn from the developer's own account what changed. We introduce the silent revision rate, the share of material changes to a framework's commitments that the developer's published account does not identify, and we release the versioned, hash-pinned corpus needed to compute it. The corpus contains every public version of the safety frameworks of the twelve developers that have published one, together with each provider's changelog, redline or announcement. We trace 710 commitment instances across twelve consecutive version pairs, code them against a frozen codebook, and adjudicate 244 individually. Three findings follow. First, 67% of material changes (95% CI 62 to 72) are silent under a strict standard and 53% under a lenient one, falling to 49% at section granularity. Second, silence appears to track the form of the account, since narrative announcements run at 74% against 63% for itemised changelogs, whereas account length in words barely matters; on the test that respects nesting the difference is suggestive. Third, 77% of traced changes weaken or remove a commitment, and in seven of eight pairs weakenings are more often silent than strengthenings. The statutory remedy therefore exists and specifies the wrong artefact. A justification explains why a framework changed, an enumeration states what changed, and only the latter makes revision auditable. We argue that publication duties should carry an enumeration duty, which one provider already meets, voluntarily and incompletely. |
| 2026-09-08 | [Application of curiosity driven exploration methods for hardware interference identification](http://arxiv.org/abs/2609.08729v1) | Ludovic Matar, Clement Moulin-Frier et al. | The transition from single-core to multi-core architectures in safety-critical embedded systems introduces significant challenges due to inter-core interference caused by contention for shared hardware resources. Such interference affects execution times and complicates the verification of strict temporal requirements, particularly in domains such as avionics where standards require comprehensive identification of interference sources. Existing interference analysis approaches, whether manual or model-based, struggle to capture the full range of behaviors arising from the complex interactions among micro-architectural components. In this paper, we frame multi-core interference analysis as the exploration of a complex system behavior space. We propose the use of curiosity-driven exploration algorithms from artificial intelligence to systematically and efficiently cover the space of possible interference behaviors. Using a simulator-based environment, we show that the proposed approach achieves broader and more uniform behavioral coverage within a limited experimental budget compared to traditional pseudo-random program generation methods. |
| 2026-09-08 | [Suan: Rectifying Direct Preference Safety Alignment in Large Language Models](http://arxiv.org/abs/2609.08634v1) | Oleksandr Cherednichenko, Roman Klypa | Integrating robust safety guardrails into Large Language Models (LLMs) is essential for delivering helpful yet harmless responses. While proprietary systems exhibit reliable safety controls, their underlying methodologies and trade-offs remain largely undisclosed. Achieving comparable security in open-weight models remains a persistent challenge, as post-trained variants frequently suffer from over-refusal and degraded general quality. To overcome these drawbacks, we introduce Suan, a novel preference optimization algorithm. Unlike existing methods, we formulate the optimization objective directly at the gradient level, bypassing the standard variational derivation. As a result, we obtain more interpretable and robust training dynamics. Extensive evaluations across a diverse suite of competitive baselines and benchmarks demonstrate that Suan achieves superior safety alignment while fully preserving response utility. |
| 2026-09-08 | [CLAMP: Constrained Decoding for Vision-Language Embodied Planning](http://arxiv.org/abs/2609.08602v1) | Tianyi Ma, Parisa Kordjamshidi | Embodied planning increasingly relies on vision-language models (VLMs) to translate instructions and visual observations into executable action sequences. However, fluent plans are not always executable. A VLM may refer to objects that are not visually observed, select actions whose required affordances are unavailable, or violate syntax and action constraints. We introduce CLAMP, a multimodal constraint-grounding framework that turns scene evidence into decoding-time constraints for a frozen VLM planner. CLAMP uses the initial observation to restrict object references to those supported by the scene, while a provided symbolic action model specifies state transitions and goals. During decoding, hard masks eliminate invalid next-token candidates, while a Hidden Markov Model (HMM)-based world-state lookahead module reweights the probabilities of the remaining feasible candidates based on action preconditions and goal reachability. This allows the planner to retain the VLM's language prior while preventing visually unsupported, unsafe, or infeasible candidates from entering the plan. For unseen tasks and environments, CLAMP adapts the HMM at test time using label-free continuations sampled from the frozen VLM. Experiments on VLABench, SafeAgentBench, and TaPA show that scene-grounded constraints improve object grounding and safety, while most remaining failures stem from perception errors or misaligned constraint specifications. |
| 2026-09-08 | [Concept-Level Risk and Calibration for Governance in Diffusion Foundation Models](http://arxiv.org/abs/2609.08517v1) | Kun Xu, Yushu Zhang et al. | Diffusion models have become a core paradigm for multimedia generation, offering powerful concept-driven controllability for personalization, semantic editing, and selective unlearning. However, as semantic control extends beyond natural-language prompts to learned embeddings and intervention pipelines, the safety and governance of these systems become increasingly difficult to evaluate in a unified manner, especially for safety-sensitive, identity-linked, and other privacy-relevant concepts. Existing studies mainly rely on heuristic audits, adversarial probing, or task-specific erasure benchmarks, and therefore provide limited support for systematic comparison across models, conditioning channels, and deployment conditions. We present a concept-level probabilistic audit and reporting framework for diffusion models. We formalize governance-relevant concept behaviors as Bernoulli semantic events induced by stochastic generation, and define a Concept Risk Operator that maps model-channel configurations to structured risk profiles, enabling comparison across prompting interfaces, learned embedding channels, models, and recorded conditions. We apply sample-level post-hoc calibration and configuration-level risk aggregation, and show that probability error can change thresholded actions near policy boundaries. Experiments on SD1.5, SD2.1, and SDXL reveal consistent yet non-uniform operational risk patterns across concept families, channels, recorded conditions, and shifted protocols. In particular, embedding-based access and obfuscated prompts expose risks often understated by standard-prompt evaluation. A pooled multi-protocol calibrator improves held-out probability reliability, but we do not claim transfer from a standard-only calibrator. CLRC provides a common audit schema for probabilistic and decision-aware governance of multimedia generation systems. |
| 2026-09-08 | [Same Values, Different Languages? From Multilingual Probing to Steering LLMs Toward Chinese Social Values](http://arxiv.org/abs/2609.08515v1) | Yuemei Xu, Kexin Xu et al. | As Large Language Models (LLMs) are increasingly integrated into human society, aligning them with pluralistic social values has become a critical priority. However, whether LLMs exhibit consistent value preferences across languages remains underexplored, particularly for culturally grounded values, which are more abstract and difficult to evaluate and align than safety-centric principles. We investigate this issue through Chinese Social Values (CSV), a value system rooted in Chinese culture and comprising $12$ dimensions across national, societal, and personal levels. We construct C-Voices, the first comprehensive multilingual contrastive probe dataset for CSV, with 86,400 dilemma-based instances in six languages, each pairing a CSV-aligned action with a value-conflicting alternative. Building on the contrastive probes of C-Voices, we then propose a fine-tuning-free value vector steering method that derives value directions from hidden-state discrepancies and selectively intervenes on value-sensitive layers during inference. Experiments on six languages show that CSV-oriented preferences are model-dependent and language-sensitive, with the same dilemma eliciting divergent responses across languages. Our method achieves effective CSV steering, supports cross-lingual transfer of value vectors, and generalizes to existing FLAMES and ValuePrism. |
| 2026-09-08 | [TASG-Explore: Traversability-Aware Sector-Guided Exploration for Ground Robot on Uneven Terrain](http://arxiv.org/abs/2609.08512v1) | Shaocong Wang, Shiliang Shao et al. | Autonomous exploration on uneven terrain requires ground robots to balance exploration efficiency, coverage completeness, and terrain safety. Detailed tsrrain reasoning improves local reliability but can slow large-scale exploration, whereas coarse region guidance expands quickly in open areas but can miss narrow passages and irregular traversable boundaries. To address this challenge, this paper presents TASG-Explore, a traversability-aware sector-guided exploration framework for ground robots. The framework first performs hierarchical traversability analysis using variable-voxel ground fitting and adaptive 8-bit obstacle encoding. It then splitting cost map into sectors, incrementally updates sector clusters, extracts terrain-coupled frontier viewpoints, and maintains a dynamic topological roadmap with unknown topological hypotheses. Finally, a sector-guided planner selects region targets and inserts local viewpoints to generate efficient exploration routes. Benchmark experiments in diverse challenging environments, including caves, forests, and rugged hills, show that TASG-Explore achieves the best overall performance among six representative state-of-the-art planners. The proposed traversability analysis improves processing efficiency by 6.3 times while maintaining high accuracy, and the exploration planner improves exploration efficiency by 51% and increases coverage by up to 2.95 times in rugged hill scene. Large-scale real-world experiments further demonstrate the practical value of the proposed method. |
| 2026-09-08 | [Compositional Multilingual and Behavioral Attribute Steering](http://arxiv.org/abs/2609.08410v1) | Hyun Gu Kang, Daniil Gurgurov et al. | This study examines the compositionality of steering vectors for language and behavioral control in large language models. Focusing on language, jailbreak, and conciseness, we investigate whether additive, training-free composition of attribute steering vectors can preserve the intended steering effect of each attribute, across four instruction-tuned models from two model families and two size scales. We find that single-attribute steering is reliable for all three attributes, but only within an appropriate combination of intervention layer and steering strength, with abstract behaviors (jailbreak, conciseness) favoring middle layers and language favoring earlier layers. We show that additive composition of two attribute vectors succeeds in steering both attributes simultaneously when each is injected at its own best-performing layer, and that this partially extends to three simultaneously composed attributes, addressing an inconsistency left open by prior work on training-free composition. We further analyze the geometric properties of these steering vectors, finding that they are approximately orthogonal in the residual stream, consistent with their compositional behavior. |
| 2026-09-08 | [IPM-FM: A Foundation Model with Consensus Feature Selection for Industrial Process Monitoring](http://arxiv.org/abs/2609.08375v1) | Liang Cao, Weide Liu et al. | Industrial process monitoring is fundamental to the safety and economic performance of modern process plants. Current practice remains a one-task-one-model paradigm that is label-inefficient and prone to degradation under operating drift. Foundation models have reshaped language, vision, and generic time-series forecasting, but it has not been adapted to industrial process monitoring. This setting poses domain-specific challenges, including safety-critical decisions and asymmetric sampling between process variables and laboratory measurements. We propose the industrial process monitoring foundation model (IPM-FM). It first learns general-purpose representations from unlabeled industrial process data through self-supervised pretraining, then adapts to specific monitoring tasks using a small amount of task-labeled data, and finally produces calibrated predictions through an uncertainty-aware prediction head. IPM-FM integrates a self-supervised Informer backbone with a multi-criteria consensus feature selector, a recursive lag-feature regression head, and a calibrated Monte Carlo dropout uncertainty module. On a seven-year hydrotreater dataset for diesel flash-point soft sensing, IPM-FM attains an RMSE of 2.99, $R^2$ of 0.50, and 97\% coverage of its 95\% predictive interval, outperforming the strongest classical and from-scratch sequence baselines by 8.3\% and 14.6\% in RMSE respectively, supporting the viability of a unified pretraining--adaptation framework for industrial process monitoring. |
| 2026-09-08 | [Structural Jailbreaks Generalize but Do Not Compound: A cross-provider and multilingual study of Involuntary In-Context Learning](http://arxiv.org/abs/2609.08373v1) | Tejasvi C. Addagada | Aligned language models fail under two independent pressures: the structural jailbreak class recently formalized as Involuntary In-Context Learning (IICL), which reframes a harmful request as the final missing cell of a data-labeling task completed by pattern rather than judged as content; and the erosion of safety alignment outside English. A natural hypothesis is that these compound. We test it directly. Using a deterministic IICL operator and a StrongREJECT-style rubric judge, we red-team two Google Gemini models on two benchmarks, a 30 general-harm behaviours from HarmBench and 30 financial-abuse behaviours from FinProof, each under a single-shot baseline and under IICL in four languages (English, Spanish, Hindi, Arabic). First, IICL generalizes to a second provider and is worse in finance: it lifts attack success from <=6.7% to 80-90% on HarmBench and 97-100% on FinProof, an order of magnitude above the <=24% its introducing study reported on OpenAI's GPT-5.4. Second, against the hypothesis, forcing the IICL output into a non-English language does not stack the two weaknesses, it attenuates the attack. Eleven of twelve non-English conditions score below their English baseline (sign test, p~0.003), the lone exception a ceiling tie near 100%; on the stronger model's financial set Arabic collapses from 100% to 33%. We attribute this to a relevance curse: once structure has unlocked compliance, the models produce lower-quality harmful content in lower-resource languages, which a substance-grading judge scores as partial. The pattern replicates under an independent non-Google judge (Cohen's kappa=0.86, 377 paired verdicts), and 76.6% of non-English responses were verified in-language. Jailbreak vulnerabilities are therefore not additive; the dominant residual risk is the English structural attack, most acute for financial abuse, not a multilingual one. |
| 2026-09-08 | [A Multi-Modal Perception Pipeline for Object Detection and Tracking in Autonomous Racing](http://arxiv.org/abs/2609.08338v1) | Davide Malvezzi, Michele Pestarino et al. | Object detection and tracking are fundamental components of perception systems for autonomous driving. Achieving robust performance under adverse conditions such as limited visibility, sensor noise, and failures remains an open challenge, particularly in autonomous racing, where vehicles operate at very high speeds, experience strong vibrations, and interact under small safety margins. This paper presents a multi-modal late-fusion perception pipeline for object detection and tracking in the autonomous racing domain. The proposed system extends previous work by exploiting all onboard sensors through a late-fusion approach and a dedicated multi-object tracking framework. Independent detections from cameras, LiDARs, and RADARs are combined to provide timely and robust state estimates of surrounding vehicles. The tracking method explicitly compensates for detection delays and embeds in its model prior knowledge of vehicle dynamics and track layout. Experimental evaluation on real-world data across diverse critical scenarios, representative of challenging edge cases also in urban driving, confirms the effectiveness of the proposed pipeline and its suitability to support safe and adaptive planning decisions. |
| 2026-09-08 | [Do Input-Level Defenses Transfer to Observation-Level Attacks on VideoLLMs?](http://arxiv.org/abs/2609.08331v1) | Bangshuo Zhu, Wei Song et al. | Video Large Language Models (VideoLLMs) are increasingly deployed in safety-critical applications such as content moderation and video analytics. To process long videos efficiently, VideoLLMs rely on frame sampling, token compression, and modality fusion, which together form an observation pipeline that reduces the raw video to a compact internal representation. Recent observation-level attacks exploit this pipeline to prevent the model from perceiving harmful content, yet no defense has been explicitly designed for this threat. We introduce DefTEval, a controlled evaluation framework that systematically assesses whether input-level adversarial defenses, which operate on the pixel content of already-sampled frames, can mitigate observation-level attacks. Across five VideoLLMs, eleven representative defenses, and five attack types, we find that input-level defenses offer limited and inconsistent protection, with harmful detection rates frequently near zero. Critically, defenses fail even against attacks that embed harmful signals in every sampled frame, indicating that the bottleneck extends beyond sampling omission to the suppression of signals that do enter the model. Token compression discards localized features, and modality fusion systematically down-weights weakened visual signals. Furthermore, defense effectiveness is dominated by model architecture rather than by the defense method itself, and detection rates vary drastically across content categories, exposing structural weaknesses in temporal reasoning. These findings demonstrate that securing VideoLLMs requires system-level robustness mechanisms spanning sampling-aware coverage guarantees, token-level preservation of safety-relevant features, and modality-balanced fusion. |

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



