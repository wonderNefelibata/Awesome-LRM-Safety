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
| 2025-11-21 | [Masked-and-Reordered Self-Supervision for Reinforcement Learning from Verifiable Rewards](http://arxiv.org/abs/2511.17473v1) | Zhen Wang, Zhifeng Gao et al. | Test-time scaling has been shown to substantially improve large language models' (LLMs) mathematical reasoning. However, for a large portion of mathematical corpora, especially theorem proving, RLVR's scalability is limited: intermediate reasoning is crucial, while final answers are difficult to directly and reliably verify. Meanwhile, token-level SFT often degenerates into rote memorization rather than inducing longer chains of thought. Inspired by BERT's self-supervised tasks, we propose MR-RLVR (Masked-and-Reordered RLVR), which constructs process-level self-supervised rewards via "masked-then-fill" and "step reordering" to extract learnable signals from intermediate reasoning. Our training pipeline comprises two stages: we first perform self-supervised training on sampled mathematical calculation and proof data; we then conduct RLVR fine-tuning on mathematical calculation datasets where only outcomes are verifiable. We implement MR-RLVR on Qwen2.5-3B and DeepSeek-R1-Distill-Qwen-1.5B, and evaluate on AIME24, AIME25, AMC23, and MATH500. Under a fixed sampling and decoding budget, MR-RLVR achieves average relative gains over the original RLVR of +9.86% Pass@1, +5.27% Pass@5, and +4.00% Pass@8. These results indicate that incorporating process-aware self-supervised signals can effectively enhance RLVR's scalability and performance in only outcome-verifiable settings. |
| 2025-11-21 | [SRA-CP: Spontaneous Risk-Aware Selective Cooperative Perception](http://arxiv.org/abs/2511.17461v1) | Jiaxi Liu, Chengyuan Ma et al. | Cooperative perception (CP) offers significant potential to overcome the limitations of single-vehicle sensing by enabling information sharing among connected vehicles (CVs). However, existing generic CP approaches need to transmit large volumes of perception data that are irrelevant to the driving safety, exceeding available communication bandwidth. Moreover, most CP frameworks rely on pre-defined communication partners, making them unsuitable for dynamic traffic environments. This paper proposes a Spontaneous Risk-Aware Selective Cooperative Perception (SRA-CP) framework to address these challenges. SRA-CP introduces a decentralized protocol where connected agents continuously broadcast lightweight perception coverage summaries and initiate targeted cooperation only when risk-relevant blind zones are detected. A perceptual risk identification module enables each CV to locally assess the impact of occlusions on its driving task and determine whether cooperation is necessary. When CP is triggered, the ego vehicle selects appropriate peers based on shared perception coverage and engages in selective information exchange through a fusion module that prioritizes safety-critical content and adapts to bandwidth constraints. We evaluate SRA-CP on a public dataset against several representative baselines. Results show that SRA-CP achieves less than 1% average precision (AP) loss for safety-critical objects compared to generic CP, while using only 20% of the communication bandwidth. Moreover, it improves the perception performance by 15% over existing selective CP methods that do not incorporate risk awareness. |
| 2025-11-21 | [MMT-ARD: Multimodal Multi-Teacher Adversarial Distillation for Robust Vision-Language Models](http://arxiv.org/abs/2511.17448v1) | Yuqi Li, Junhao Dong et al. | Vision-Language Models (VLMs) are increasingly deployed in safety-critical applications, making their adversarial robustness a crucial concern. While adversarial knowledge distillation has shown promise in transferring robustness from teacher to student models, traditional single-teacher approaches suffer from limited knowledge diversity, slow convergence, and difficulty in balancing robustness and accuracy. To address these challenges, we propose MMT-ARD: a Multimodal Multi-Teacher Adversarial Robust Distillation framework. Our key innovation is a dual-teacher knowledge fusion architecture that collaboratively optimizes clean feature preservation and robust feature enhancement. To better handle challenging adversarial examples, we introduce a dynamic weight allocation strategy based on teacher confidence, enabling adaptive focus on harder samples. Moreover, to mitigate bias among teachers, we design an adaptive sigmoid-based weighting function that balances the strength of knowledge transfer across modalities. Extensive experiments on ImageNet and zero-shot benchmarks demonstrate that MMT-ARD improves robust accuracy by +4.32% and zero-shot accuracy by +3.5% on the ViT-B-32 model, while achieving a 2.3x increase in training efficiency over traditional single-teacher methods. These results highlight the effectiveness and scalability of MMT-ARD in enhancing the adversarial robustness of multimodal large models. Our codes are available at https://github.com/itsnotacie/MMT-ARD. |
| 2025-11-21 | [IndustryNav: Exploring Spatial Reasoning of Embodied Agents in Dynamic Industrial Navigation](http://arxiv.org/abs/2511.17384v1) | Yifan Li, Lichi Li et al. | While Visual Large Language Models (VLLMs) show great promise as embodied agents, they continue to face substantial challenges in spatial reasoning. Existing embodied benchmarks largely focus on passive, static household environments and evaluate only isolated capabilities, failing to capture holistic performance in dynamic, real-world complexity. To fill this gap, we present IndustryNav, the first dynamic industrial navigation benchmark for active spatial reasoning. IndustryNav leverages 12 manually created, high-fidelity Unity warehouse scenarios featuring dynamic objects and human movement. Our evaluation employs a PointGoal navigation pipeline that effectively combines egocentric vision with global odometry to assess holistic local-global planning. Crucially, we introduce the "collision rate" and "warning rate" metrics to measure safety-oriented behaviors and distance estimation. A comprehensive study of nine state-of-the-art VLLMs (including models such as GPT-5-mini, Claude-4.5, and Gemini-2.5) reveals that closed-source models maintain a consistent advantage; however, all agents exhibit notable deficiencies in robust path planning, collision avoidance and active exploration. This highlights a critical need for embodied research to move beyond passive perception and toward tasks that demand stable planning, active exploration, and safe behavior in dynamic, real-world environment. |
| 2025-11-21 | [U-DESPE: a Bayesian Utility-based methodology for dosing regimen optimization in early-phase oncology trials based on Dose-Exposure, Safety, Pharmacodynamics, Efficacy](http://arxiv.org/abs/2511.17376v1) | Anaïs Andrillon, Sandrine Micallef et al. | With the development of novel therapies such as molecularly targeted agents and immunotherapy, the maximum tolerated dose paradigm that "more is better" does not necessarily hold anymore. In this context, doses and schedules of novel therapies may be inadequately characterized and oncology drug dose-finding approaches should be revised. This is increasingly recognized by health authorities, notably through the Optimus project. We developed a Bayesian dose-finding design, called U-DESPE, which allows to either determine the optimal dosing regimen at the end of the dose-escalation phase, or use of dedicated cohorts for randomizing patients to candidate optimal dosing regimens after that safe dosing regimens have been found. U-DESPE design relies on a dose-exposure model built from pharmacokinetic data using non-linear mixed-effect modeling approaches. Then three models are built to assess the relationships between exposure and the probability of selected relevant endpoints on safety, efficacy, and pharmacodynamics. These models are then combined to predict the different endpoints for every candidate dosing regimens. Finally, a utility function is proposed to quantify the trade-off between these endpoints and to determine the optimal dosing regimen. We applied the proposed method on a clinical trial case study and performed an extensive simulation study to evaluate the operating characteristics of the method. |
| 2025-11-21 | [FORWARD: Dataset of a forwarder operating in rough terrain](http://arxiv.org/abs/2511.17318v1) | Mikael Lundbäck, Erik Wallin et al. | We present FORWARD, a high-resolution multimodal dataset of a cut-to-length forwarder operating in rough terrain on two harvest sites in the middle part of Sweden. The forwarder is a large Komatsu model equipped with a variety of sensors, including RTK-GNSS, 360-camera, operator vibration sensors, internal CAN-bus signal recording, and multiple IMUs. The data includes event time logs recorded in 5 Hz with e.g., driving speed, fuel consumption, vehicle position with centimeter accuracy, and crane use while the vehicle operates in forest areas laser-scanned with very high-resolution, $\sim$1500 points per square meter. Production log files (StanForD standard) with time-stamped machine events, extensive video material, and terrain data in various formats are included as well. About 18 hours of regular wood extraction work during three days is annotated from 360-video material into individual work elements and included in the dataset. We also include scenario specifications of conducted experiments on forest roads and in terrain. Scenarios include repeatedly driving the same routes with and without steel tracks, different load weight, and different target driving speeds. The dataset is intended for developing models and algorithms for trafficability, perception, and autonomous control of forest machines using artificial intelligence, simulation, and experiments on physical testbeds. In part, we focus on forwarders traversing terrain, avoiding obstacles, and loading or unloading logs, with consideration for efficiency, fuel consumption, safety, and environmental impact. Other benefits of the open dataset include the ability to explore auto-generation and calibration of forestry machine simulators and automation scenario descriptions using the data recorded in the field. |
| 2025-11-21 | [Navigating in the Dark: A Multimodal Framework and Dataset for Nighttime Traffic Sign Recognition](http://arxiv.org/abs/2511.17183v1) | Aditya Mishra, Akshay Agarwal et al. | Traffic signboards are vital for road safety and intelligent transportation systems, enabling navigation and autonomous driving. Yet, recognizing traffic signs at night remains challenging due to visual noise and scarcity of public nighttime datasets. Despite advances in vision architectures, existing methods struggle with robustness under low illumination and fail to leverage complementary mutlimodal cues effectively. To overcome these limitations, firstly, we introduce INTSD, a large-scale dataset comprising street-level night-time images of traffic signboards collected across diverse regions of India. The dataset spans 41 traffic signboard classes captured under varying lighting and weather conditions, providing a comprehensive benchmark for both detection and classification tasks. To benchmark INTSD for night-time sign recognition, we conduct extensive evaluations using state-of-the-art detection and classification models. Secondly, we propose LENS-Net, which integrates an adaptive image enhancement detector for joint illumination correction and sign localization, followed by a structured multimodal CLIP-GCNN classifier that leverages cross-modal attention and graph-based reasoning for robust and semantically consistent recognition. Our method surpasses existing frameworks, with ablation studies confirming the effectiveness of its key components. The dataset and code for LENS-Net is publicly available for research. |
| 2025-11-21 | [The Promotion Wall: Efficiency-Equity Trade-offs of Direct Promotion Regimes in Engineering Education](http://arxiv.org/abs/2511.17182v1) | H. R. Paz | Progression and assessment rules are often treated as administrative details, yet they fundamentally shape who is allowed to remain in higher education, and on what terms. This article uses a calibrated agent-based model to examine how alternative progression regimes reconfigure dropout, time-to-degree, equity and students' psychological experience in a long, tightly sequenced engineering programme. Building on a leakage-aware longitudinal dataset of 1,343 students and a Kaplan-Meier survival analysis of time-to-dropout, we simulate three policy scenarios: (A) a historical "regularity + finals" regime, where students accumulate exam debt; (B) a direct-promotion regime that removes regularity and finals but requires full course completion each term; and (C) a direct-promotion regime complemented by a capacity-limited remedial "safety net" for marginal failures in bottleneck courses. The model is empirically calibrated to reproduce the observed dropout curve under Scenario A and then used to explore counterfactuals. Results show that direct promotion creates a "promotion wall": attrition becomes sharply front-loaded in the first two years, overall dropout rises, and equity gaps between low- and high-resilience students widen, even as exam debt disappears. The safety-net scenario partially dismantles this wall: it reduces dropout and equity gaps relative to pure direct promotion and yields the lowest final stress levels, at the cost of additional, targeted teaching capacity. These findings position progression rules as central objects of assessment policy rather than neutral background. The article argues that claims of improved efficiency are incomplete unless they are evaluated jointly with inclusion, equity and students' psychological wellbeing, and it illustrates how simulation-based decision support can help institutions rehearse assessment reforms before implementing them. |
| 2025-11-21 | [Super-Resolution ISAC Receivers: An MCMC-Based Gridless Sparse Bayesian Learning Approach](http://arxiv.org/abs/2511.17062v1) | Keying Zhu, Xingyu Zhou et al. | Integrated sensing and communication (ISAC) is crucial for low-altitude wireless networks (LAWNs), where the safety-critical demand for high-accuracy sensing creates a trade-off between precision and complexity for conventional methods. To address this, we propose a novel gridless sparse Bayesian learning (SBL) framework for joint super-resolution multi-target detection and high-accuracy parameter estimation with manageable computational cost. Our model treats target parameters as continuous variables to bypass the grid limitations of conventional approaches. This SBL formulation, however, transforms the estimation task into a challenging high-dimensional inference problem, which we address by developing an enhanced gradient-based Markov chain Monte Carlo algorithm. Our method integrates mini-batch sampling and the Adam optimizer to ensure computational efficiency and rapid convergence. Finally, we validate the framework's robustness in strong clutter and provide a theoretical benchmark by deriving the corresponding Bayesian Cramer-Rao bound. Simulation results demonstrate remarkable super-resolution capabilities, successfully resolving multiple targets separated by merely 50% of the Rayleigh limit in range, 17% in velocity, and 52% in angle. At a signal-to-noise ratio of 20 dB, the algorithm achieves a multi-target detection probability exceeding 90% while concurrently delivering ultra-high accuracy, with root mean square error of 0.07 m, 0.024 m/s, and 0.015 degree for range, velocity, and angle, respectively. This robust performance, demonstrated against strong clutter, showcases its suitability for practical ISAC-LAWNs applications. |
| 2025-11-21 | [Supervised Fine Tuning of Large Language Models for Domain Specific Knowledge Graph Construction:A Case Study on Hunan's Historical Celebrities](http://arxiv.org/abs/2511.17012v1) | Junjie Hao, Chun Wang et al. | Large language models and knowledge graphs offer strong potential for advancing research on historical culture by supporting the extraction, analysis, and interpretation of cultural heritage. Using Hunan's modern historical celebrities shaped by Huxiang culture as a case study, pre-trained large models can help researchers efficiently extract key information, including biographical attributes, life events, and social relationships, from textual sources and construct structured knowledge graphs. However, systematic data resources for Hunan's historical celebrities remain limited, and general-purpose models often underperform in domain knowledge extraction and structured output generation in such low-resource settings. To address these issues, this study proposes a supervised fine-tuning approach for enhancing domain-specific information extraction. First, we design a fine-grained, schema-guided instruction template tailored to the Hunan historical celebrities domain and build an instruction-tuning dataset to mitigate the lack of domain-specific training corpora. Second, we apply parameter-efficient instruction fine-tuning to four publicly available large language models - Qwen2.5-7B, Qwen3-8B, DeepSeek-R1-Distill-Qwen-7B, and Llama-3.1-8B-Instruct - and develop evaluation criteria for assessing their extraction performance. Experimental results show that all models exhibit substantial performance gains after fine-tuning. Among them, Qwen3-8B achieves the strongest results, reaching a score of 89.3866 with 100 samples and 50 training iterations. This study provides new insights into fine-tuning vertical large language models for regional historical and cultural domains and highlights their potential for cost-effective applications in cultural heritage knowledge extraction and knowledge graph construction. |
| 2025-11-21 | [Real Option AI: Reversibility, Silence, and the Release Ladder](http://arxiv.org/abs/2511.16958v1) | I. Sebastian Buhai | We model the cadence of AI product releases, i.e. quiet spells, reversible patches, and rarer pivots, as optimal exercise of strategic real options under reputational learning. A privately observed technical state follows a diffusion. The firm controls two upgrade options with asymmetric costs and reversibility (a cheap patch and a costly pivot) and a publication-frequency clock, a Cox process whose intensity governs when noisy public performance and safety signals are disclosed. For sufficiently low clock costs the optimal policy posts observable clock-off windows around knife-edge regions. These windows shut down the martingale part of public beliefs, eliminate knife-edge mixing, and collapse behavior to a two-rung release ladder with endogenous triggers, jump targets, and no interior mixing. Within stationary Markov strategies we show that this ladder is uniquely characterized by a boundary-value system with value matching and smooth pasting at triggers and target optimality at jump targets. We endogenize market or platform adoption as a threshold rule in public beliefs and show that leverage creates an irreversibility wedge: the gap between first-best and levered surplus is bounded by the takeover switching cost of the least reversible rung. Patches are debt-insensitive; pivots can be distorted, but only up to that bound. The framework predicts telemetry signatures in firm-authored disclosures: a pre-release cadence dip in publication intensity and intra-month dispersion as the clock is shut off before a major reset; two post-release plateaus in disclosed performance, consistent with patch versus pivot jump targets; and debt-insensitive patch timing in high-reversibility regimes, with leverage effects concentrated in pivots. Unlike option-implied volatility spikes, these patterns reflect the firm's own throttling of technical signals rather than market pricing of event risk. |
| 2025-11-21 | [MultiPriv: Benchmarking Individual-Level Privacy Reasoning in Vision-Language Models](http://arxiv.org/abs/2511.16940v1) | Xiongtao Sun, Hui Li et al. | Modern Vision-Language Models (VLMs) demonstrate sophisticated reasoning, escalating privacy risks beyond simple attribute perception to individual-level linkage. Current privacy benchmarks are structurally insufficient for this new threat, as they primarily evaluate privacy perception while failing to address the more critical risk of privacy reasoning: a VLM's ability to infer and link distributed information to construct individual profiles. To address this critical gap, we propose \textbf{MultiPriv}, the first benchmark designed to systematically evaluate individual-level privacy reasoning in VLMs. We introduce the \textbf{Privacy Perception and Reasoning (PPR)} framework and construct a novel, bilingual multimodal dataset to support it. The dataset uniquely features a core component of synthetic individual profiles where identifiers (e.g., faces, names) are meticulously linked to sensitive attributes. This design enables nine challenging tasks evaluating the full PPR spectrum, from attribute detection to cross-image re-identification and chained inference. We conduct a large-scale evaluation of over 50 foundational and commercial VLMs. Our analysis reveals: (1) Many VLMs possess significant, unmeasured reasoning-based privacy risks. (2) Perception-level metrics are poor predictors of these reasoning risks, revealing a critical evaluation gap. (3) Existing safety alignments are inconsistent and ineffective against such reasoning-based attacks. MultiPriv exposes systemic vulnerabilities and provides the necessary framework for developing robust, privacy-preserving VLMs. |
| 2025-11-21 | [Hybrid Differential Reward: Combining Temporal Difference and Action Gradients for Efficient Multi-Agent Reinforcement Learning in Cooperative Driving](http://arxiv.org/abs/2511.16916v1) | Ye Han, Lijun Zhang et al. | In multi-vehicle cooperative driving tasks involving high-frequency continuous control, traditional state-based reward functions suffer from the issue of vanishing reward differences. This phenomenon results in a low signal-to-noise ratio (SNR) for policy gradients, significantly hindering algorithm convergence and performance improvement. To address this challenge, this paper proposes a novel Hybrid Differential Reward (HDR) mechanism. We first theoretically elucidate how the temporal quasi-steady nature of traffic states and the physical proximity of actions lead to the failure of traditional reward signals. Building on this analysis, the HDR framework innovatively integrates two complementary components: (1) a Temporal Difference Reward (TRD) based on a global potential function, which utilizes the evolutionary trend of potential energy to ensure optimal policy invariance and consistency with long-term objectives; and (2) an Action Gradient Reward (ARG), which directly measures the marginal utility of actions to provide a local guidance signal with a high SNR. Furthermore, we formulate the cooperative driving problem as a Multi-Agent Partially Observable Markov Game (POMDPG) with a time-varying agent set and provide a complete instantiation scheme for HDR within this framework. Extensive experiments conducted using both online planning (MCTS) and Multi-Agent Reinforcement Learning (QMIX, MAPPO, MADDPG) algorithms demonstrate that the HDR mechanism significantly improves convergence speed and policy stability. The results confirm that HDR guides agents to learn high-quality cooperative policies that effectively balance traffic efficiency and safety. |
| 2025-11-20 | [Monte Carlo Expected Threat (MOCET) Scoring](http://arxiv.org/abs/2511.16823v1) | Joseph Kim, Saahith Potluri | Evaluating and measuring AI Safety Level (ASL) threats are crucial for guiding stakeholders to implement safeguards that keep risks within acceptable limits. ASL-3+ models present a unique risk in their ability to uplift novice non-state actors, especially in the realm of biosecurity. Existing evaluation metrics, such as LAB-Bench, BioLP-bench, and WMDP, can reliably assess model uplift and domain knowledge. However, metrics that better contextualize "real-world risks" are needed to inform the safety case for LLMs, along with scalable, open-ended metrics to keep pace with their rapid advancements. To address both gaps, we introduce MOCET, an interpretable and doubly-scalable metric (automatable and open-ended) that can quantify real-world risks. |
| 2025-11-20 | [Lord of the (sub-)Rings : Mapping the surface reflectance and spin-axis of Ajisai](http://arxiv.org/abs/2511.16780v1) | Robert J. S Airey, Paul Chote et al. | Active debris removal techniques are posed to become an important tool in maintaining the safety of the near-Earth space environment. These techniques rely on a clear understanding of the rotational motion of the debris targets, which is challenging to constrain from unresolved imaging. The Ajisai satellite provides an ideal test case for developing and demonstrating these techniques due to its simple geometry and well constrained spin behaviour. We present four observations of the Ajisai satellite taken with SuperWASP in August of 2019, where high cadence photometry was extracted from streaked images as a part of a larger survey of Low Earth Orbit. We develop an MCMC-driven method to determine the spin-state of Ajisai by comparing the alignment between a map of modelled mirror positions and a novel derived map of surface reflectivity. We generally find good agreement within the expectation and uncertainties set by empirical models and our determined spin-state solutions align the surface reflectivity map and modelled mirror location well. Our results show that streak photometry can be used to recover the spin-axis and rotation period of fast-spinning objects such as Ajisai using modest ground-based instrumentation, making it readily scalable to a wider range of targets and observatories. |
| 2025-11-20 | [SafeR-CLIP: Mitigating NSFW Content in Vision-Language Models While Preserving Pre-Trained Knowledge](http://arxiv.org/abs/2511.16743v1) | Adeel Yousaf, Joseph Fioresi et al. | Improving the safety of vision-language models like CLIP via fine-tuning often comes at a steep price, causing significant drops in their generalization performance. We find this trade-off stems from rigid alignment strategies that force unsafe concepts toward single, predefined safe targets, disrupting the model's learned semantic structure. To address this, we propose a proximity-aware approach: redirecting unsafe concepts to their semantically closest safe alternatives to minimize representational change. We introduce SaFeR-CLIP, a fine-tuning framework that applies this principle of minimal intervention. SaFeR-CLIP successfully reconciles safety and performance, recovering up to 8.0% in zero-shot accuracy over prior methods while maintaining robust safety. To support more rigorous evaluation, we also contribute NSFW-Caps, a new benchmark of 1,000 highly-aligned pairs for testing safety under distributional shift. Our work shows that respecting the geometry of pretrained representations is key to achieving safety without sacrificing performance. |
| 2025-11-20 | [Formal Abductive Latent Explanations for Prototype-Based Networks](http://arxiv.org/abs/2511.16588v1) | Jules Soria, Zakaria Chihani et al. | Case-based reasoning networks are machine-learning models that make predictions based on similarity between the input and prototypical parts of training samples, called prototypes. Such models are able to explain each decision by pointing to the prototypes that contributed the most to the final outcome. As the explanation is a core part of the prediction, they are often qualified as ``interpretable by design". While promising, we show that such explanations are sometimes misleading, which hampers their usefulness in safety-critical contexts. In particular, several instances may lead to different predictions and yet have the same explanation. Drawing inspiration from the field of formal eXplainable AI (FXAI), we propose Abductive Latent Explanations (ALEs), a formalism to express sufficient conditions on the intermediate (latent) representation of the instance that imply the prediction. Our approach combines the inherent interpretability of case-based reasoning models and the guarantees provided by formal XAI. We propose a solver-free and scalable algorithm for generating ALEs based on three distinct paradigms, compare them, and present the feasibility of our approach on diverse datasets for both standard and fine-grained image classification. The associated code can be found at https://github.com/julsoria/ale |
| 2025-11-20 | [Synthesis of Safety Specifications for Probabilistic Systems](http://arxiv.org/abs/2511.16579v1) | Gaspard Ohlmann, Edwin Hamel-De le Court et al. | Ensuring that agents satisfy safety specifications can be crucial in safety-critical environments. While methods exist for controller synthesis with safe temporal specifications, most existing methods restrict safe temporal specifications to probabilistic-avoidance constraints. Formal methods typically offer more expressive ways to express safety in probabilistic systems, such as Probabilistic Computation Tree Logic (PCTL) formulas. Thus, in this paper, we develop a new approach that supports more general temporal properties expressed in PCTL. Our contribution is twofold. First, we develop a theoretical framework for the Synthesis of safe-PCTL specifications. We show how the reducing global specification satisfaction to local constraints, and define CPCTL, a fragment of safe-PCTL. We demonstrate how the expressiveness of CPCTL makes it a relevant fragment for the Synthesis Problem. Second, we leverage these results and propose a new Value Iteration-based algorithm to solve the synthesis problem for these more general temporal properties, and we prove the soundness and completeness of our method. |
| 2025-11-20 | [WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient Facing Dialogue](http://arxiv.org/abs/2511.16544v1) | Zachary Ellis, Jared Joselowitz et al. | As Automatic Speech Recognition (ASR) is increasingly deployed in clinical dialogue, standard evaluations still rely heavily on Word Error Rate (WER). This paper challenges that standard, investigating whether WER or other common metrics correlate with the clinical impact of transcription errors. We establish a gold-standard benchmark by having expert clinicians compare ground-truth utterances to their ASR-generated counterparts, labeling the clinical impact of any discrepancies found in two distinct doctor-patient dialogue datasets. Our analysis reveals that WER and a comprehensive suite of existing metrics correlate poorly with the clinician-assigned risk labels (No, Minimal, or Significant Impact). To bridge this evaluation gap, we introduce an LLM-as-a-Judge, programmatically optimized using GEPA to replicate expert clinical assessment. The optimized judge (Gemini-2.5-Pro) achieves human-comparable performance, obtaining 90% accuracy and a strong Cohen's $κ$ of 0.816. This work provides a validated, automated framework for moving ASR evaluation beyond simple textual fidelity to a necessary, scalable assessment of safety in clinical dialogue. |
| 2025-11-20 | [WER is Unaware: Assessing How ASR Errors Distort Clinical Understanding in Patient Facing Dialogue](http://arxiv.org/abs/2511.16544v2) | Zachary Ellis, Jared Joselowitz et al. | As Automatic Speech Recognition (ASR) is increasingly deployed in clinical dialogue, standard evaluations still rely heavily on Word Error Rate (WER). This paper challenges that standard, investigating whether WER or other common metrics correlate with the clinical impact of transcription errors. We establish a gold-standard benchmark by having expert clinicians compare ground-truth utterances to their ASR-generated counterparts, labeling the clinical impact of any discrepancies found in two distinct doctor-patient dialogue datasets. Our analysis reveals that WER and a comprehensive suite of existing metrics correlate poorly with the clinician-assigned risk labels (No, Minimal, or Significant Impact). To bridge this evaluation gap, we introduce an LLM-as-a-Judge, programmatically optimized using GEPA through DSPy to replicate expert clinical assessment. The optimized judge (Gemini-2.5-Pro) achieves human-comparable performance, obtaining 90% accuracy and a strong Cohen's $κ$ of 0.816. This work provides a validated, automated framework for moving ASR evaluation beyond simple textual fidelity to a necessary, scalable assessment of safety in clinical dialogue. |

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



