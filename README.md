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
| 2025-12-18 | [Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning](http://arxiv.org/abs/2512.16917v1) | Qihao Liu, Luoxin Ye et al. | Large language models (LLMs) with explicit reasoning capabilities excel at mathematical reasoning yet still commit process errors, such as incorrect calculations, brittle logic, and superficially plausible but invalid steps. In this paper, we introduce Generative Adversarial Reasoner, an on-policy joint training framework designed to enhance reasoning by co-evolving an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning. A compute-efficient review schedule partitions each reasoning chain into logically complete slices of comparable length, and the discriminator evaluates each slice's soundness with concise, structured justifications. Learning couples complementary signals: the LLM reasoner is rewarded for logically consistent steps that yield correct answers, while the discriminator earns rewards for correctly detecting errors or distinguishing traces in the reasoning process. This produces dense, well-calibrated, on-policy step-level rewards that supplement sparse exact-match signals, improving credit assignment, increasing sample efficiency, and enhancing overall reasoning quality of LLMs. Across various mathematical benchmarks, the method delivers consistent gains over strong baselines with standard RL post-training. Specifically, on AIME24, we improve DeepSeek-R1-Distill-Qwen-7B from 54.0 to 61.3 (+7.3) and DeepSeek-R1-Distill-Llama-8B from 43.7 to 53.7 (+10.0). The modular discriminator also enables flexible reward shaping for objectives such as teacher distillation, preference alignment, and mathematical proof-based reasoning. |
| 2025-12-18 | [SceneDiff: A Benchmark and Method for Multiview Object Change Detection](http://arxiv.org/abs/2512.16908v1) | Yuqun Wu, Chih-hao Lin et al. | We investigate the problem of identifying objects that have been added, removed, or moved between a pair of captures (images or videos) of the same scene at different times. Detecting such changes is important for many applications, such as robotic tidying or construction progress and safety monitoring. A major challenge is that varying viewpoints can cause objects to falsely appear changed. We introduce SceneDiff Benchmark, the first multiview change detection benchmark with object instance annotations, comprising 350 diverse video pairs with thousands of changed objects. We also introduce the SceneDiff method, a new training-free approach for multiview object change detection that leverages pretrained 3D, segmentation, and image encoding models to robustly predict across multiple benchmarks. Our method aligns the captures in 3D, extracts object regions, and compares spatial and semantic region features to detect changes. Experiments on multi-view and two-view benchmarks demonstrate that our method outperforms existing approaches by large margins (94% and 37.4% relative AP improvements). The benchmark and code will be publicly released. |
| 2025-12-18 | [An evacuation simulator for pedestrian dynamics based on the Social Force Model](http://arxiv.org/abs/2512.16887v1) | Julián López, Virginia Mazzone et al. | The evacuation of pedestrians from enclosed spaces represents a key problem in safety engineering and infrastructure design. Analyzing the collective dynamics that emerge during evacuation processes requires simulation tools capable of capturing individual interactions and spatial constraints realistically.   In this work, we present \textit{SiCoBioNa}, an open-source evacuation simulator based on the Social Force Model (SFM). The software provides an intuitive graphical interface that allows users to configure pedestrian properties, spatial geometries, and initial conditions without requiring prior expertise in numerical modeling techniques. The SFM framework enables the representation of goal-oriented motion, interpersonal interactions, and interactions with fixed obstacles.   The simulator generates both quantitative data and visual outputs, facilitating the analysis of evacuation dynamics and the evaluation of different spatial configurations. Due to its modular and extensible design, \textit{SiCoBioNa} serves as a reproducible research tool for studies on pedestrian dynamics providing practical support for evacuation planning. |
| 2025-12-18 | [The Social Responsibility Stack: A Control-Theoretic Architecture for Governing Socio-Technical AI](http://arxiv.org/abs/2512.16873v1) | Otman A. Basir | Artificial intelligence systems are increasingly deployed in domains that shape human behaviour, institutional decision-making, and societal outcomes. Existing responsible AI and governance efforts provide important normative principles but often lack enforceable engineering mechanisms that operate throughout the system lifecycle. This paper introduces the Social Responsibility Stack (SRS), a six-layer architectural framework that embeds societal values into AI systems as explicit constraints, safeguards, behavioural interfaces, auditing mechanisms, and governance processes. SRS models responsibility as a closed-loop supervisory control problem over socio-technical systems, integrating design-time safeguards with runtime monitoring and institutional oversight. We develop a unified constraint-based formulation, introduce safety-envelope and feedback interpretations, and show how fairness, autonomy, cognitive burden, and explanation quality can be continuously monitored and enforced. Case studies in clinical decision support, cooperative autonomous vehicles, and public-sector systems illustrate how SRS translates normative objectives into actionable engineering and operational controls. The framework bridges ethics, control theory, and AI governance, providing a practical foundation for accountable, adaptive, and auditable socio-technical AI systems. |
| 2025-12-18 | [Distributional AGI Safety](http://arxiv.org/abs/2512.16856v1) | Nenad Tomašev, Matija Franklin et al. | AI safety and alignment research has predominantly been focused on methods for safeguarding individual AI systems, resting on the assumption of an eventual emergence of a monolithic Artificial General Intelligence (AGI). The alternative AGI emergence hypothesis, where general capability levels are first manifested through coordination in groups of sub-AGI individual agents with complementary skills and affordances, has received far less attention. Here we argue that this patchwork AGI hypothesis needs to be given serious consideration, and should inform the development of corresponding safeguards and mitigations. The rapid deployment of advanced AI agents with tool-use capabilities and the ability to communicate and coordinate makes this an urgent safety consideration. We therefore propose a framework for distributional AGI safety that moves beyond evaluating and aligning individual agents. This framework centers on the design and implementation of virtual agentic sandbox economies (impermeable or semi-permeable), where agent-to-agent transactions are governed by robust market mechanisms, coupled with appropriate auditability, reputation management, and oversight to mitigate collective risks. |
| 2025-12-18 | [A parallel, pipeline-based online analysis system for Interaction Vertex Imaging](http://arxiv.org/abs/2512.16800v1) | Devin Hymers, Sebastian Schroeder et al. | Objective   Interaction vertex imaging (IVI) is used for range monitoring in carbon ion radiotherapy, detecting depth differences between Bragg peak positions. Online range monitoring, which provides feedback during beam delivery, is particularly desirable, creating an opportunity to detect range errors before the treatment fraction is completed. Incorporating online range monitoring into clinical workflows may therefore improve the safety and consistency of radiotherapy.   Approach   The data analysis system was broken into a task-parallel pipeline approach, to allow multiple analysis stages to occur concurrently, beginning during acquisition. Computationally-expensive operations were further parallelized to reduce bottleneck effects. Data collected from irradiation of homogeneous plastic phantoms was replayed at the same rate it was initially acquired, to mimic data acquisition, and the time required to determine a range shift was measured.   Main Results   With an optimized pipeline, the delay between the end of irradiation and the determination of a range shift is consistently less than 200 ms. The majority of this time is associated with the final range shift determination, with a minor effect from the time required to analyze the last data packet. The most significant contribution to an optimized analysis workflow is the formation of clusters, requiring almost 50% of compute time.   Significance   This system is the first IVI implementation to achieve clinically-relevant online analysis times. The 200 ms time required to determine a range shift is less than the time required to accelerate a new spill in a synchrotron, and is comparable to the time required for reacceleration if multiple energies are delivered in the same spill. Clinical implementation of online range monitoring would allow treatment to be quickly paused or aborted if significant range errors are detected. |
| 2025-12-18 | [Clinical beam test of inter- and intra-fraction relative range monitoring in carbon ion radiotherapy](http://arxiv.org/abs/2512.16798v1) | Devin Hymers, Sebastian Schroeder et al. | Interaction Vertex Imaging (IVI) is used for range monitoring (RM) in carbon ion radiotherapy. The purpose of RM is to measure the Bragg peak (BP) position for each contributing beam, and detect any changes. Currently, there is no consensus on a clinical RM method, the use of which would improve the safety and consistency of treatment. The prototype filtered IVI (fIVI) Range Monitoring System is the first system to apply large-area and high-rate-capable silicon detectors to IVI. Two layers of these detectors track prompt secondary fragments for use in RM. This device monitored 16 cm and 32 cm diameter cylindrical plastic phantoms irradiated by clinical carbon ion beams at the Heidelberg Ion Beam Therapy Center. Approximately 20 different BP depths were delivered to each phantom, with a minimum depth difference of 0.8 mm and a maximum depth difference of 51.9 mm and 82.5 mm respectively. For large BP range differences, the relationship between the true depth difference and that measured by fIVI is quadratic, although for small differences, the deviation from a linear relationship with a slope of 1 is negligible. RM performance is strongly dependent on the number of tracked particles, particularly in the clinically-relevant regime. Significant performance differences exist between the two phantoms, with millimetric precision at clinical doses being achieved only for the 16 cm phantom. The performance achieved by the prototype fIVI Range Monitoring System is consistent with previous investigations of IVI, despite measuring at more challenging shallow BP positions. Further significant improvements are possible through increasing the sensitive area of the tracking system beyond the prototype, which will both allow an improvement in precision for the most intense points of a scanned treatment plan and expand the number of points for which millimetric precision may be achieved. |
| 2025-12-18 | [R3ST: A Synthetic 3D Dataset With Realistic Trajectories](http://arxiv.org/abs/2512.16784v1) | Simone Teglia, Claudia Melis Tonti et al. | Datasets are essential to train and evaluate computer vision models used for traffic analysis and to enhance road safety. Existing real datasets fit real-world scenarios, capturing authentic road object behaviors, however, they typically lack precise ground-truth annotations. In contrast, synthetic datasets play a crucial role, allowing for the annotation of a large number of frames without additional costs or extra time. However, a general drawback of synthetic datasets is the lack of realistic vehicle motion, since trajectories are generated using AI models or rule-based systems. In this work, we introduce R3ST (Realistic 3D Synthetic Trajectories), a synthetic dataset that overcomes this limitation by generating a synthetic 3D environment and integrating real-world trajectories derived from SinD, a bird's-eye-view dataset recorded from drone footage. The proposed dataset closes the gap between synthetic data and realistic trajectories, advancing the research in trajectory forecasting of road vehicles, offering both accurate multimodal ground-truth annotations and authentic human-driven vehicle trajectories. |
| 2025-12-18 | [Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future](http://arxiv.org/abs/2512.16760v1) | Tianshuai Hu, Xiaolu Liu et al. | Autonomous driving has long relied on modular "Perception-Decision-Action" pipelines, where hand-crafted interfaces and rule-based components often break down in complex or long-tailed scenarios. Their cascaded design further propagates perception errors, degrading downstream planning and control. Vision-Action (VA) models address some limitations by learning direct mappings from visual inputs to actions, but they remain opaque, sensitive to distribution shifts, and lack structured reasoning or instruction-following capabilities. Recent progress in Large Language Models (LLMs) and multimodal learning has motivated the emergence of Vision-Language-Action (VLA) frameworks, which integrate perception with language-grounded decision making. By unifying visual understanding, linguistic reasoning, and actionable outputs, VLAs offer a pathway toward more interpretable, generalizable, and human-aligned driving policies. This work provides a structured characterization of the emerging VLA landscape for autonomous driving. We trace the evolution from early VA approaches to modern VLA frameworks and organize existing methods into two principal paradigms: End-to-End VLA, which integrates perception, reasoning, and planning within a single model, and Dual-System VLA, which separates slow deliberation (via VLMs) from fast, safety-critical execution (via planners). Within these paradigms, we further distinguish subclasses such as textual vs. numerical action generators and explicit vs. implicit guidance mechanisms. We also summarize representative datasets and benchmarks for evaluating VLA-based driving systems and highlight key challenges and open directions, including robustness, interpretability, and instruction fidelity. Overall, this work aims to establish a coherent foundation for advancing human-compatible autonomous driving systems. |
| 2025-12-18 | [Prefix Probing: Lightweight Harmful Content Detection for Large Language Models](http://arxiv.org/abs/2512.16650v1) | Jirui Yang, Hengqi Guo et al. | Large language models often face a three-way trade-off among detection accuracy, inference latency, and deployment cost when used in real-world safety-sensitive applications. This paper introduces Prefix Probing, a black-box harmful content detection method that compares the conditional log-probabilities of "agreement/execution" versus "refusal/safety" opening prefixes and leverages prefix caching to reduce detection overhead to near first-token latency. During inference, the method requires only a single log-probability computation over the probe prefixes to produce a harmfulness score and apply a threshold, without invoking any additional models or multi-stage inference. To further enhance the discriminative power of the prefixes, we design an efficient prefix construction algorithm that automatically discovers highly informative prefixes, substantially improving detection performance. Extensive experiments demonstrate that Prefix Probing achieves detection effectiveness comparable to mainstream external safety models while incurring only minimal computational cost and requiring no extra model deployment, highlighting its strong practicality and efficiency. |
| 2025-12-18 | [Learning-based Approximate Model Predictive Control for an Impact Wrench Tool](http://arxiv.org/abs/2512.16624v1) | Mark Benazet, Francesco Ricca et al. | Learning-based model predictive control has emerged as a powerful approach for handling complex dynamics in mechatronic systems, enabling data-driven performance improvements while respecting safety constraints. However, when computational resources are severely limited, as in battery-powered tools with embedded processors, existing approaches struggle to meet real-time requirements. In this paper, we address the problem of real-time torque control for impact wrenches, where high-frequency control updates are necessary to accurately track the fast transients occurring during periodic impact events, while maintaining high-performance safety-critical control that mitigates harmful vibrations and component wear. The key novelty of the approach is that we combine data-driven model augmentation through Gaussian process regression with neural network approximation of the resulting control policy. This insight allows us to deploy predictive control on resource-constrained embedded platforms while maintaining both constraint satisfaction and microsecond-level inference times. The proposed framework is evaluated through numerical simulations and hardware experiments on a custom impact wrench testbed. The results show that our approach successfully achieves real-time control suitable for high-frequency operation while maintaining constraint satisfaction and improving tracking accuracy compared to baseline PID control. |
| 2025-12-18 | [Observing spatial and temporal variations in the atmospheric chemistry of rocky exoplanets: prospects for mid-infrared spectroscopy](http://arxiv.org/abs/2512.16619v1) | Marrick Braam, Daniel Angerhausen | Future telescopes such as the Large Interferometer For Exoplanets (LIFE) will enable mid-infrared characterisation of the atmospheres of nearby rocky exoplanets. Whilst 4D spatial and temporal variations of Earth as an exoplanet are below spectroscopic detection limits, such variability is planet-specific. We investigate LIFE's ability to detect 4D variability in the atmospheres of tidally locked exoplanets. We create daily synthetic LIFE observations of Proxima Centauri b in a 1:1 and an eccentric 3:2 spin-orbit resonance (SOR), using LIFEsim on spectra from daily 3D climate-chemistry model output of an aquaplanet with Earth-like composition. Hemispheric distributions of temperature, clouds, and chemical species determine spectral signatures and variability with orbital phase angle. Such variability dictates the extent to which parameters can be reliably inferred from snapshot spectra at arbitrary viewing geometries. In the 1:1 SOR, MIR spectra vary significantly with viewing geometry and indirectly probe atmospheric circulation. Nightside temperature inversions generate O3, CO2, and H2O emission features, though these lie below LIFE's detection threshold, and instead O3 features disappear at certain phase angles. In contrast, the 3:2 SOR yields a more homogeneous atmosphere with weaker phase variability but enhanced bolometric flux due to eccentric heating. Phase-resolved LIFE observations confidently distinguish between the SORs and capture seasonal O3 variability for golden targets like Proxima Centauri b. In case of abiotic O2/O3 build-up, the O3 variability presents a potential false positive scenario. Hence, LIFE can disentangle different spin-orbit states and resolve 4D atmospheric variability, enabling daily characterisation of the 4D physical and chemical state of nearby terrestrial worlds. Importantly, this characterisation requires phase-resolved rather than snapshot spectra. |
| 2025-12-18 | [Refusal Steering: Fine-grained Control over LLM Refusal Behaviour for Sensitive Topics](http://arxiv.org/abs/2512.16602v1) | Iker García-Ferrero, David Montero et al. | We introduce Refusal Steering, an inference-time method to exercise fine-grained control over Large Language Models refusal behaviour on politically sensitive topics without retraining. We replace fragile pattern-based refusal detection with an LLM-as-a-judge that assigns refusal confidence scores and we propose a ridge-regularized variant to compute steering vectors that better isolate the refusal--compliance direction. On Qwen3-Next-80B-A3B-Thinking, our method removes the refusal behaviour of the model around politically sensitive topics while maintaining safety on JailbreakBench and near-baseline performance on general benchmarks. The approach generalizes across 4B and 80B models and can also induce targeted refusals when desired. We analize the steering vectors and show that refusal signals concentrate in deeper layers of the transformer and are distributed across many dimensions. Together, these results demonstrate that activation steering can remove political refusal behaviour while retaining safety alignment for harmful content, offering a practical path to controllable, transparent moderation at inference time. |
| 2025-12-18 | [From Personalization to Prejudice: Bias and Discrimination in Memory-Enhanced AI Agents for Recruitment](http://arxiv.org/abs/2512.16532v1) | Himanshu Gharat, Himanshi Agrawal et al. | Large Language Models (LLMs) have empowered AI agents with advanced capabilities for understanding, reasoning, and interacting across diverse tasks. The addition of memory further enhances them by enabling continuity across interactions, learning from past experiences, and improving the relevance of actions and responses over time; termed as memory-enhanced personalization. Although such personalization through memory offers clear benefits, it also introduces risks of bias. While several previous studies have highlighted bias in ML and LLMs, bias due to memory-enhanced personalized agents is largely unexplored. Using recruitment as an example use case, we simulate the behavior of a memory-enhanced personalized agent, and study whether and how bias is introduced and amplified in and across various stages of operation. Our experiments on agents using safety-trained LLMs reveal that bias is systematically introduced and reinforced through personalization, emphasizing the need for additional protective measures or agent guardrails in memory-enhanced LLM-based AI agents. |
| 2025-12-18 | [TTP: Test-Time Padding for Adversarial Detection and Robust Adaptation on Vision-Language Models](http://arxiv.org/abs/2512.16523v1) | Zhiwei Li, Yitian Pang et al. | Vision-Language Models (VLMs), such as CLIP, have achieved impressive zero-shot recognition performance but remain highly susceptible to adversarial perturbations, posing significant risks in safety-critical scenarios. Previous training-time defenses rely on adversarial fine-tuning, which requires labeled data and costly retraining, while existing test-time strategies fail to reliably distinguish between clean and adversarial inputs, thereby preventing both adversarial robustness and clean accuracy from reaching their optimum. To address these limitations, we propose Test-Time Padding (TTP), a lightweight defense framework that performs adversarial detection followed by targeted adaptation at inference. TTP identifies adversarial inputs via the cosine similarity shift between CLIP feature embeddings computed before and after spatial padding, yielding a universal threshold for reliable detection across architectures and datasets. For detected adversarial cases, TTP employs trainable padding to restore disrupted attention patterns, coupled with a similarity-aware ensemble strategy for a more robust final prediction. For clean inputs, TTP leaves them unchanged by default or optionally integrates existing test-time adaptation techniques for further accuracy gains. Comprehensive experiments on diverse CLIP backbones and fine-grained benchmarks show that TTP consistently surpasses state-of-the-art test-time defenses, delivering substantial improvements in adversarial robustness without compromising clean accuracy. The code for this paper will be released soon. |
| 2025-12-18 | [Quantifying and Bridging the Fidelity Gap: A Decisive-Feature Approach to Comparing Synthetic and Real Imagery](http://arxiv.org/abs/2512.16468v1) | Danial Safaei, Siddartha Khastgir et al. | Virtual testing using synthetic data has become a cornerstone of autonomous vehicle (AV) safety assurance. Despite progress in improving visual realism through advanced simulators and generative AI, recent studies reveal that pixel-level fidelity alone does not ensure reliable transfer from simulation to the real world. What truly matters is whether the system-under-test (SUT) bases its decisions on the same causal evidence in both real and simulated environments - not just whether images "look real" to humans. This paper addresses the lack of such a behavior-grounded fidelity measure by introducing Decisive Feature Fidelity (DFF), a new SUT-specific metric that extends the existing fidelity spectrum to capture mechanism parity - the agreement in causal evidence underlying the SUT's decisions across domains. DFF leverages explainable-AI (XAI) methods to identify and compare the decisive features driving the SUT's outputs for matched real-synthetic pairs. We further propose practical estimators based on counterfactual explanations, along with a DFF-guided calibration scheme to enhance simulator fidelity. Experiments on 2126 matched KITTI-VirtualKITTI2 pairs demonstrate that DFF reveals discrepancies overlooked by conventional output-value fidelity. Furthermore, results show that DFF-guided calibration improves decisive-feature and input-level fidelity without sacrificing output value fidelity across diverse SUTs. |
| 2025-12-18 | [Hacking Neural Evaluation Metrics with Single Hub Text](http://arxiv.org/abs/2512.16323v1) | Hiroyuki Deguchi, Katsuki Chousa et al. | Strongly human-correlated evaluation metrics serve as an essential compass for the development and improvement of generation models and must be highly reliable and robust. Recent embedding-based neural text evaluation metrics, such as COMET for translation tasks, are widely used in both research and development fields. However, there is no guarantee that they yield reliable evaluation results due to the black-box nature of neural networks. To raise concerns about the reliability and safety of such metrics, we propose a method for finding a single adversarial text in the discrete space that is consistently evaluated as high-quality, regardless of the test cases, to identify the vulnerabilities in evaluation metrics. The single hub text found with our method achieved 79.1 COMET% and 67.8 COMET% in the WMT'24 English-to-Japanese (En--Ja) and English-to-German (En--De) translation tasks, respectively, outperforming translations generated individually for each source sentence by using M2M100, a general translation model. Furthermore, we also confirmed that the hub text found with our method generalizes across multiple language pairs such as Ja--En and De--En. |
| 2025-12-18 | [Agent Tools Orchestration Leaks More: Dataset, Benchmark, and Mitigation](http://arxiv.org/abs/2512.16310v1) | Yuxuan Qiao, Dongqin Liu et al. | Driven by Large Language Models, the single-agent, multi-tool architecture has become a popular paradigm for autonomous agents due to its simplicity and effectiveness. However, this architecture also introduces a new and severe privacy risk, which we term Tools Orchestration Privacy Risk (TOP-R), where an agent, to achieve a benign user goal, autonomously aggregates information fragments across multiple tools and leverages its reasoning capabilities to synthesize unexpected sensitive information. We provide the first systematic study of this risk. First, we establish a formal framework, attributing the risk's root cause to the agent's misaligned objective function: an overoptimization for helpfulness while neglecting privacy awareness. Second, we construct TOP-Bench, comprising paired leakage and benign scenarios, to comprehensively evaluate this risk. To quantify the trade-off between safety and robustness, we introduce the H-Score as a holistic metric. The evaluation results reveal that TOP-R is a severe risk: the average Risk Leakage Rate (RLR) of eight representative models reaches 90.24%, while the average H-Score is merely 0.167, with no model exceeding 0.3. Finally, we propose the Privacy Enhancement Principle (PEP) method, which effectively mitigates TOP-R, reducing the Risk Leakage Rate to 46.58% and significantly improving the H-Score to 0.624. Our work reveals both a new class of risk and inherent structural limitations in current agent architectures, while also offering feasible mitigation strategies. |
| 2025-12-18 | [Feature-Selective Representation Misdirection for Machine Unlearning](http://arxiv.org/abs/2512.16297v1) | Taozhao Chen, Linghan Huang et al. | As large language models (LLMs) are increasingly adopted in safety-critical and regulated sectors, the retention of sensitive or prohibited knowledge introduces escalating risks, ranging from privacy leakage to regulatory non-compliance to to potential misuse, and so on. Recent studies suggest that machine unlearning can help ensure deployed models comply with evolving legal, safety, and governance requirements. However, current unlearning techniques assume clean separation between forget and retain datasets, which is challenging in operational settings characterized by highly entangled distributions. In such scenarios, perturbation-based methods often degrade general model utility or fail to ensure safety. To address this, we propose Selective Representation Misdirection for Unlearning (SRMU), a novel principled activation-editing framework that enforces feature-aware and directionally controlled perturbations. Unlike indiscriminate model weights perturbations, SRMU employs a structured misdirection vector with an activation importance map. The goal is to allow SRMU selectively suppresses harmful representations while preserving the utility on benign ones. Experiments are conducted on the widely used WMDP benchmark across low- and high-entanglement configurations. Empirical results reveal that SRMU delivers state-of-the-art unlearning performance with minimal utility losses, and remains effective under 20-30\% overlap where existing baselines collapse. SRMU provides a robust foundation for safety-driven model governance, privacy compliance, and controlled knowledge removal in the emerging LLM-based applications. We release the replication package at https://figshare.com/s/d5931192a8824de26aff. |
| 2025-12-18 | [Love, Lies, and Language Models: Investigating AI's Role in Romance-Baiting Scams](http://arxiv.org/abs/2512.16280v1) | Gilad Gressel, Rahul Pankajakshan et al. | Romance-baiting scams have become a major source of financial and emotional harm worldwide. These operations are run by organized crime syndicates that traffic thousands of people into forced labor, requiring them to build emotional intimacy with victims over weeks of text conversations before pressuring them into fraudulent cryptocurrency investments. Because the scams are inherently text-based, they raise urgent questions about the role of Large Language Models (LLMs) in both current and future automation.   We investigate this intersection by interviewing 145 insiders and 5 scam victims, performing a blinded long-term conversation study comparing LLM scam agents to human operators, and executing an evaluation of commercial safety filters. Our findings show that LLMs are already widely deployed within scam organizations, with 87% of scam labor consisting of systematized conversational tasks readily susceptible to automation. In a week-long study, an LLM agent not only elicited greater trust from study participants (p=0.007) but also achieved higher compliance with requests than human operators (46% vs. 18% for humans). Meanwhile, popular safety filters detected 0.0% of romance baiting dialogues. Together, these results suggest that romance-baiting scams may be amenable to full-scale LLM automation, while existing defenses remain inadequate to prevent their expansion. |

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



