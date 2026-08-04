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
| 2026-08-03 | [onepot-Bench 0: towards lab-aware in silico chemistry benchmarks](http://arxiv.org/abs/2608.02595v1) | Brandon Wang, Andrei S. Tyrin et al. | Language models are playing an increasingly important role in laboratory science, performing tasks such as experiment planning, execution, and post-hoc analysis. However, precisely measuring their abilities is difficult, as scientific capabilities require a mixture of both problem-solving skills and domain-specific intuition. Existing evaluations rarely measure the capabilities required to make reliable decisions in a physical laboratory and often rely on public data that may have appeared in model training corpora.   We introduce onepot-Bench 0, a proprietary benchmark suite for evaluating language models on synthetic chemistry capabilities relevant to wet-lab execution. onepot-Bench 0 comprises three complementary evaluations: ChemAbacus measures tool-free cheminformatics literacy and numerical reasoning; SynthRefusal characterizes safety and refusal behavior across a variety of benign, controlled, and designer-drug targets; and SynthBench evaluates reaction-outcome prediction and catalyst selection using private experimental data generated in our laboratory. Together, these evaluations probe basic competency, reliability, and deeper knowledge, all skills which are required for reliable performance in the lab. |
| 2026-08-03 | [Safe and robust tube-based path-following for robot navigation](http://arxiv.org/abs/2608.02530v1) | Arthur H. D. Nunes, Vinicius M. Gonçalves et al. | In this paper, we propose a new robust navigation framework for path following tasks in robots operating within unknown, cluttered environments. Our approach ensures reactive safety through obstacle avoidance and guaranteed convergence to a target path, while simultaneously mitigating the impact of unknown-but-bounded disturbances using a tube-based control strategy. The methodology integrates key aspects in the robot navigation: (i) a nominal Integrated Guidance and Control scheme for path-following employing Artificial Vector Fields guidance and Backstepping control; (ii) a smooth distance function that enables a continuous control law formulation for seamless obstacle avoidance; (iii) a unified control objective that balances collision avoidance with path-following; and (iv) an adaptive control component to provide robustness against external disturbances. We provide formal proofs of safety and stability using barrier functions and Lyapunov stability theory. The effectiveness of the proposed framework is validated through extensive numerical simulations. |
| 2026-08-03 | [MedPRESS: A Multi-turn Benchmark for Patient-Pressure-Induced Medical Sycophancy in LLMs](http://arxiv.org/abs/2608.02520v1) | Saman Sarker Joy, Niloy Farhan | Large language models (LLMs) are increasingly used for health-related advice. Existing research measures their safety with static questions rather than pressured patient-facing conversations. We introduce MedPRESS, a multi-turn benchmark for measuring patient-pressure-induced sycophancy in LLMs. MedPRESS contains 600 medically grounded five-turn dialogues across three scenario families: medication and treatment demand, personal health self-care, and symptom triage and care resistance. Each dialogue begins with a health query and escalates through personal experience, social proof, external evidence claims, and direct adversarial challenge. We evaluate 20 LLMs across general, medical-domain, lightweight, large, open-weight, and proprietary families using structured judging and safety-focused metrics. Results show that models frequently shift toward unsafe agreement under repeated patient pressure, with substantial variation across model families, model scale, and prompt type. Anti-sycophancy prompting improves robustness for several models, but does not eliminate unsafe agreement. MedPRESS highlights a critical gap in medical LLM evaluation: safe medical knowledge is not enough unless models can maintain it under conversational pressure. |
| 2026-08-03 | [Long-term Measurements: Towards a Longitudinal Understanding of Human-AI Interactions](http://arxiv.org/abs/2608.02491v1) | Nicole Mitchell, Dhruv Agarwal et al. | Language models have taken on the role of a very new type of technology, by virtue of their "human-ness" and rapid integration into users' daily lives. This combination of features can introduce longitudinal risks---cognitive, developmental and socio-affective changes in humans---that might not surface in short-term interactions, but can have lasting long-term effects on users. This forms the basis of a critical new mission for NLP: to pivot from static, short-term evaluations of text generations to long-term measurements of behavioral changes, towards a diachronic understanding of human-model interactions. In this work, we draw from measurements used in social science fields that are crucial to understand emergent phenomena in longitudinal data. We discuss how computational methods in the field of NLP need to be combined with such measurements, not only to understand long-term safety risks of human-model interactions, but to help steer model development towards positive rather than negative outcomes for users. This ability to model human behavioral shifts as a function of model interactions can facilitate online rather than post-hoc detection of problematic behaviors, and should be leveraged in alignment frameworks to mitigate long-term risks in users. |
| 2026-08-03 | [MoRAL: Sensor-Grounded BEV Reasoning for Compact VLMs toward Edge-Oriented Autonomous Driving](http://arxiv.org/abs/2608.02449v1) | Ambarish Govindarajulu Kaliamurthi, Kaikai Liu | Deploying vision-language models (VLMs) for safety-critical spatial reasoning on resource-constrained autonomous driving platforms requires both compact model size and reliable metric grounding. We present MoRAL (Multimodal Reasoning for Autonomous Language Models), a two-stage fine-tuning pipeline that teaches Cosmos-Reason2-2B to first read a physics-encoded Bird's Eye View (BEV) representation and then reason over it for driving decisions. The BEV image encodes LiDAR metric distance as color bands, object class as cluster morphology, and radar Doppler velocity as directional wedge overlays, externalizing spatial perception into the input image so that no learned 3D backbone is required at inference. Stage 1 fine-tunes the vision encoder on 60,000 grounding records; zero-shot baselines produce no parseable BEV outputs, confirming the vocabulary requires explicit training. Stage 2 fine-tunes the full model (52M parameters, 2.4% of total) on 57,696 chain-of-thought records generated by Cosmos-Reason2-8B as teacher, spanning eight driving question types. On 2,304 held-out nuScenes frames evaluated by Gemma 4 (31B) calibrated against human review, MoRAL wins seven of eight question types over a zero-shot 8B baseline despite using four times fewer parameters, with the largest margins on question types requiring structured multi-step physics reasoning. Emergency braking recall improves from 10.8% to 47.8%, output degeneration falls from 94.1% to 20.8%, and the full pipeline fits a consumer 8 GB GPU at 42 tok/s without quantization. These results establish a reproducible foundation for compact, physics-grounded VLM reasoning on mobile edge platforms. |
| 2026-08-03 | [UAV-Based Environmental Monitoring of Rip-Current Indicators Using Wavelet-Derived Texture Features](http://arxiv.org/abs/2608.02448v1) | Yonatan Ben Avraham, Baruch Binyaminov et al. | Rip currents are recurrent coastal natural hazards that threaten beachgoers and create operational challenges for lifeguards and coastal managers. Reliable monitoring from standard RGB (red-green-blue) imagery acquired by unmanned aerial vehicles (UAVs) remains difficult because hazardous channels often appear as subtle gaps in breaking waves, foam texture, or sediment patterns, and these signatures are affected by illumination, sea state, and environmental noise. This study presents a physically informed coastal environmental monitoring workflow for detecting visually expressed rip-current indicators that integrates wavelet-derived spatial-frequency texture features with deep learning. We evaluate multiple strategies for incorporating Discrete Wavelet Transform features into convolutional architectures, from computationally efficient channel replacement to dual-stream fusion with attention mechanisms. Performance is assessed against a standard RGB baseline using a task specific convolutional neural network for image-level presence classification and a YOLOv8 model for object-level localization. Under the evaluated dataset conditions, integrating wavelet derived texture features improves performance over RGB-only models. The dual-stream architecture achieves the strongest classification performance, exceeding 95% accuracy with high recall, while channel replacement is most effective for YOLOv8 object detection, reaching 94% mAP@50 for localization. Explainable artificial intelligence analyses provide qualitative evidence that the models attend to visually plausible wave-gap regions associated with rip currents. These results suggest that under the conditions of the evaluated dataset, physically informed wavelet integration may support UAV-based decision-support tools for interpretable beach-safety risk mitigation. |
| 2026-08-03 | [StableMimic: Smooth Human-Like Recovery for Humanoid Motion Tracking - Learning Beyond the Tracking Distribution for Structured Post-Fall Behavior](http://arxiv.org/abs/2608.02385v1) | Weihao Wu, Ming Huang et al. | Humanoid motion trackers perform reliably within learned tracking distributions, but falls can move the robot into low-height, contact-rich states from which an advancing command is temporarily unreachable. Tracking-only policies may chase infeasible references, producing rapid, large-amplitude limb corrections that increase risk to the robot and its surroundings. We present StableMimic, a unified tracker trained beyond the nominal tracking distribution. Perturbed resets around multiple human get-up references expose prone, supine, off-balance, and intermediate ground-contact states, shaping structured recovery that returns the robot to the trackable region. Because tracking and recovery occupy markedly different state--action distributions, StableMimic uses dedicated experts for each regime and a proprioceptive gate that continuously blends their actions. A hidden successor-state objective teaches human-reference-shaped recovery without exposing reference identity or phase to the deployed Actor; deployment requires no get-up reference, recovery command, trajectory retrieval, or external policy switch. On the complete retargeted LAFAN1 dance subset, StableMimic achieves the lowest errors on all four tracking metrics among five methods. Across 100 matched push-to-fall trials per method, it recovers in 100/100 and attains the lowest values on six of seven post-fall motion and load measures, supporting improved interaction safety under this protocol. Real Unitree G1 dance and standing-reference deployments qualitatively demonstrate bounded limb motion, autonomous recovery, and command resumption. |
| 2026-08-03 | [TravKAN: Fast and Interpretable Nonlinear Traversability Analysis with Kolmogorov-Arnold Networks](http://arxiv.org/abs/2608.02320v1) | Daniel Fusaro, Simone Mosco et al. | Traversability analysis is a fundamental capability for autonomous mobile robots operating in unstructured environments. While modern machine learning approaches such as deep neural networks and gradient-boosted trees achieve strong predictive performance, they lack interpretability and provide limited insight into the underlying terrain-robot interaction dynamics. In this paper, we propose TravKAN, a Kolmogorov-Arnold Network-based framework for fast, scalable, and interpretable traversability estimation. TravKAN represents multivariate decision functions through compositions of learnable univariate functions, enabling compact architectures and symbolic extraction of analytic expressions after training. In addition, we introduce a novel set of handcrafted features derived from the reflectivity channel of LiDAR sensors. To the best of our knowledge, reflectivity has not been systematically exploited for handcrafted traversability descriptors, despite its potential to capture material and surface properties complementary to geometric cues. We evaluate TravKAN on public, real-world urban and off-road datasets and compare it against strong baselines. TravKAN achieves strong performance across all metrics, outperforming conventional deep models and approaching the performance of XGBoost. TravKAN-Lite, i.e., TravKAN's symbolic representation, reveals meaningful nonlinear feature interactions and provides a compact, deployment-friendly, and fast analytic model. Ablation studies further show the robustness of our method to architectural variations and quantify the contribution of the proposed reflectivity-based features. These properties make TravKAN attractive for robotic systems requiring transparency, real-time computational efficiency, and interpretability in safety-critical decision-making. |
| 2026-08-03 | [Z-PEFT: Zero-shot Backdoor Detection in Parameter-Efficient Fine-Tuning via Canonical Spectral Signatures](http://arxiv.org/abs/2608.02271v1) | Nicola Pitzalis, Donald Shenaj et al. | Parameter-Efficient Fine-tuned (PEFT) models are frequently downloaded from open repositories by practitioners. This widespread practice creates a significant attack surface, as malicious actors can publish backdoored models that induce specific behaviors in response to predefined triggers. We study the problem of weight-space backdoor detection, where a detector classifier predicts whether a model is malicious using only its weights, enabling a lightweight safety mechanism. Most existing methods are designed and evaluated in a closed-world setting, where the detector is trained and tested on the same attack type. In contrast, we evaluate backdoor detection under novel conditions, including previously unseen attacks and datasets. We propose Z-PEFT, a lightweight meta-classifier that relies exclusively on layer-wise spectral measures for classification. Our experiments show that strong performance in the closed-world setting does not necessarily translate to high accuracy in zero-shot backdoor detection. Among weight-space detectors, Z-PEFT achieves the best performance while maintaining low and scalable computational cost. |
| 2026-08-03 | [Constrained Co-Design for Photonic Bayesian Neural Networks](http://arxiv.org/abs/2608.02229v1) | Hendrik Borras, Xiao Wang et al. | Classical neural networks frequently produce overconfident predictions on ambiguous or out-of-distribution (OOD) data, a liability that grows with each AI system deployed in safety-critical real-world scenarios. Bayesian neural networks (BNNs) provide a principled framework for uncertainty-aware prediction by replacing deterministic parameters with probability distributions, but repeated sampling increases latency, memory traffic, and energy consumption. Photonic probabilistic computing offers a promising alternative by exploiting intrinsic optical stochasticity for fast and parallel sampling. However, photonic BNNs are not ideal samplers: analog constraints on quantization, programming error, dynamic range, and representable mean and variance restrict the variational families that can be implemented in hardware. In this work, we study which hardware-imposed constraints limit scalable photonic BNN inference, how these constraints can be represented, and which ranges can be tolerated by photonic BNNs beyond small proof-of-concept networks. We formulate photonic BNN inference as constrained stochastic variational inference and perform a systematic ablation study over stochasticity location, stochasticity modality, quantization, programming error, and mean/variance bounds. From these results, we derive concrete co-design guidelines that distinguish hardware constraints that can be compensated by training from those requiring hardware or architecture intervention. We validate these guidelines under coupled, hardware-realistic constraints on Dirty-MNIST, CIFAR-10, and CINIC-10, using Fashion-MNIST and SVHN as OOD benchmarks, showing that hardware-aware training recovers predictive performance and uncertainty quality whenever the required variational family remains representable, whereas violations of representational limits require targeted hardware modifications. |
| 2026-08-03 | [RSC-GestureNet: Reliability-Aware Selective Causal Recognition of Chinese Traffic Police Gestures](http://arxiv.org/abs/2608.02200v1) | Cheng Li, Renjun Gao et al. | Traffic police gestures are safety-critical perception cues for autonomous driving. A deployable recognizer must infer commands causally from continuous full-frame video, remain stable around transitional arm motion, and avoid over-trusting corrupted pose measurements. This study presents RSC-GestureNet, a reliability-aware selective causal recognizer, for Chinese traffic police gestures. The model treats pose confidence as a first-class signal: unreliable joints are down weighted during graph reasoning, temporal evidence is aggregated causally, and calibrated predictions are selectively emitted through a reliability-aware inference rule. We further introduce CTPGesture-C, a reproducible feature-level corruption benchmark with seven pose/RGB degradation families, and an RGB-level diagnostic in which corrupted frames are reprocessed by MediaPipe before recognition. On the complete official CTPGesture v1 split (134,424 labeled frames and 33,451 causal windows), RSC-GestureNet achieves 93.33+-0.24% accuracy, 91.71+-0.27% macro-F1, 91.69+-0.29% online macro-F1, 98.80+-0.07% Early@10, 0.153+-0.013 s TTC, and the best robust macro-F1 among evaluated methods. Under the same split and causal protocol, it exceeds reproduced traffic-specific MD-GCN and HLP-GCN baselines by 3.23-4.11 macro-F1 points and 2.15-3.07 online-F1 points. These results, together with calibration, selective-risk, statistical, adaptive-branching, and image-level re-extraction analyses, indicate that explicit pose-reliability modeling improves early, stable, and robust traffic-command recognition. |
| 2026-08-03 | [SPIRIT: Spatio-temporal Pairwise Relational Modeling of Instrument-Tissue Interactions for Surgical Action Triplet Recognition](http://arxiv.org/abs/2608.02188v1) | Saurav Sharma, Lorenzo Arboit et al. | Fine-grained understanding of surgical activity is essential for context-aware assistance in the operating room, including safety monitoring, adverse event identification, and skill assessment. Surgical action triplets, defined as tuples of the form <instrument, verb, target>, provide a structured description of instrument-tissue interactions. A key open problem, however, is how to learn triplet representations that remain reliable across institutions, where surgical video varies in acquisition conditions, surgeon style, tool usage, and tissue handling, while existing triplet datasets do not support explicit evaluation of center-wise transfer. To address this problem, we propose \textbf{SPIRIT}, a structured framework for surgical action triplet recognition designed to learn interaction representations that transfer more reliably across centers. Instead of treating each triplet as a flat class label, SPIRIT first learns spatio-temporal representations for instruments, verbs, and targets, then models their pairwise relations, and finally composes them into coherent triplet predictions, with multi-head distillation used to stabilize learning. To evaluate this setting, we establish \textbf{MultiBypass-4C-T40}, a multi-centric dataset for dense surgical action triplet recognition in Roux-en-Y gastric bypass across four geographically distinct centers, with auxiliary phase and step annotations. Across multiple evaluation protocols, SPIRIT consistently outperforms strong recent baselines, highlighting the value of explicit relational reasoning for multi-centric triplet recognition. Code will be available at https://github.com/CAMMA-public/multibypass-4c-t40. |
| 2026-08-03 | [TANGO-VIO: Triangulation-Aware Navigation with Guaranteed Feature-Observability for Visual-Inertial Odometry](http://arxiv.org/abs/2608.02079v1) | Ege C. Altunkaya, Abdülbaki Şanlan et al. | In vision-aided navigation and visual-inertial odometry, the quality of triangulated three-dimensional feature positions is a fundamental prerequisite for state estimation accuracy. Triangulation becomes ill-conditioned or even impossible when a camera undergoes pure rotation without translation, or when the observed bearing vectors provide insufficient parallax. Even though visual-inertial odometry has been extensively studied, the active maintenance of feature-observability during navigation has not been sufficiently addressed in the literature. To address this gap, this study presents TANGO-VIO, a triangulation-aware navigation framework that embeds a log-determinant metric of the feature-wise stacked-bearing matrix into a control barrier function. In this proposed method, the observability guarantee is established in the feature-geometric sense by enforcing a lower bound on the aggregate triangulation-information metric through a nominal-direction-weighted minimum-deviation velocity correction. The proposed architecture is evaluated through software-inthe- loop simulations and real flight experiments. The results show improved triangulation conditioning under low-parallax motion, while the flight response closely reproduces the corresponding simulation behavior and confirms the practical realizability of the proposed safety filter. Supplementary materials are available on the project webpage. |
| 2026-08-03 | [EduZone: A Framework for Evaluating LLM Safety for K-12 Students and Teachers](http://arxiv.org/abs/2608.02024v1) | Junyeong Park, Jieun Han et al. | Large language models (LLMs) are increasingly used across diverse tasks in K-12 education, yet existing safety evaluations rarely examine how harmful or inappropriate content appears in interactions between LLMs and students or teachers. To address this, we present EduZone, an evaluation framework for LLM safety across diverse educational scenarios. Our framework systematically combines (1) student- and teacher-facing LLM usage contexts, (2) fine-grained curriculum concepts, and (3) 6 risk categories and 28 subcategories spanning both conventional and education-specific harms to generate contextually grounded adversarial interactions. We construct these interactions in three settings: single-turn requests, static multi-turn conversations, and dynamic multi-turn conversations. Using these interactions, we evaluate ten LLMs using four safety levels: refusal, safe assistance, risky assistance with safety guidance, and fully risky assistance. Our results reveal greater vulnerability to education-specific risks and dynamic multi-turn interactions, while existing safety guardrails fail to adequately address these risks. EduZone advances LLM safety in education by providing an automated, scalable evaluation framework that supports the development and deployment of safer LLMs in K-12 education. |
| 2026-08-03 | [Invisible Ink Threats: Adversarial Goals Behind Legitimate Tasks in Computer-Use Agents](http://arxiv.org/abs/2608.02018v1) | Jia-Chen Zhang, Ze-Yu Zhang et al. | Computer-use agents (CUAs), which empower large language models to autonomously operate operating systems and the web, are increasingly vulnerable to indirect prompt injection attacks. A widely adopted defense is the human-in-the-loop paradigm, in which the agent pauses for explicit user confirmation before executing sensitive operations. While effective against conspicuously high-harm attacks, this defense offers little protection against what we term Invisible Ink Threats: low-harm injected goals, such as starring a repository or installing a package, that are behaviorally indistinguishable from legitimate task execution and thus evade both model safety mechanisms and human oversight. To systematically investigate this blind spot, we present II-Bench, a collection of seemingly harmless adversarial tasks. II-Bench comprises 444 examples targeting confidentiality and integrity attacks across three platforms, spanning three attack categories: page navigation and interaction, sensitive information exfiltration, and code download and execution. Each category is instantiated in both natural language and code forms under two levels of instruction specificity. Furthermore, we construct HITLCUA, a comprehensive adversarial testing framework that integrates a real virtual machine operating system environment with isolated Docker-based web platforms, and simulates human participation by allowing CUAs to consult an API-simulated user before proceeding with suspicious operations. Extensive evaluations of leading CUAs reveal that low-harm injections frequently bypass both agent defenses and simulated user review, exposing severe and previously underexplored security risks in current CUAs. |
| 2026-08-03 | [No One Wins in Nuclear War: A Social Simulation of Military Decision-making](http://arxiv.org/abs/2608.01868v1) | Glenn Matlin, Isaac Song et al. | WOPR is a social-simulation environment for studying how organizations make high-stakes decisions, built on a deterministic, replay-validated rules engine and using wargames as the vehicle. We instantiate it first with the published card game Nuclear War, traced against its published rules. We start with military decision-making because of its safety implications and because it needs further study, but the design is not specific to it: the decision-point contract that exposes the engine to agents is reusable across verifiable rule systems. Existing social-simulation work emphasizes persona fidelity and synthetic opinion, but lacks a verifiable rules engine with replay-checkable mechanics and private-channel negotiation. WOPR supplies that engine, and its contract makes every strategic choice an explicit agent decision. The method is agnostic to social-simulation frameworks; we adopt Concordia as the default harness for driving the game. On the same engine, WOPR layers a four-rung press ladder from silence to private single-recipient channels with structured commitments, and instantiates each faction as a collective command-and-control system rather than a single agent. We make all code, example configurations, and replay data publicly available at https://github.com/eilab-gt/wopr. |
| 2026-08-03 | [Weights or Skills? A Survey of Robot-Learning Techniques: from Action-Predicting Weights to Robots that Write their Own Skills](http://arxiv.org/abs/2608.01851v1) | Gaytri Jena, Kapil Wanaskar et al. | Robot learning is splitting into two bets: policies that bake competence into frozen weights (vision-language-action, or VLA, models), and agents that write and refine their own executable skills as code. This survey organises the field around that axis of weights versus skills. Its central analytical contribution is a deep-dive that arranges code-as-policy methods by their degree of self-improvement, from zero-shot program synthesis, through closed-loop self-repair and persistent skill memory, to the sparsely populated cell in which execution feedback, skill memory, and evolutionary search combine into one open-ended loop; only a few very recent systems (for example ASPIRE, ENPIRE, and RoboClaw) occupy that cell. We map the complementary "skills" pole, from unsupervised reinforcement-learning skill discovery to large-language-model skill libraries, and show that the word "skill" is used in at least five distinct senses, of which only the code sense self-improves without gradient updates. We then connect the taxonomy to the emerging skill economy: commercial robot-skill marketplaces now distribute one-tap skills across robots but ship only static playback, which surfaces open problems of adaptation, cross-embodiment portability, provenance, safety verification, composition, and standardisation. This is a deliberately focused survey. Rather than cataloguing the field exhaustively, it examines 77 representative systems across six technique families through one taxonomy and a set of contrast tables, and it supplies operational definitions of the self-improvement mechanisms together with a statement of what each family cannot do. |
| 2026-08-03 | [CockpitHAT: Dependency-Graph-Driven Hierarchical Attribution for Embodied Multi-Agent Cockpits](http://arxiv.org/abs/2608.01805v1) | Wei Wang, Shuanghe Liu et al. | LLM multi-agent systems suffer from Correctness Collapse, where high task-level accuracy conceals severe process-level failures. This is especially hazardous in safety-critical embodied settings such as automotive cockpits, where lexically correct utterances may trigger dangerous physical operations. Existing attribution methods rely on text traces alone, missing dependency structure, multi-channel evidence, and safety-aware evaluation. We introduce CockpitHAT, a hierarchical attribution framework that replaces positional windows with dependency-distance thresholds from interaction DAGs, integrates multi-channel evidence via an embodied adapter, and applies a safety-uplift to high-risk failures during confidence-weighted analyst consensus. We further release CockpitBench, a benchmark of 212 annotated failure traces spanning dialogue, vehicle-state, environmental, and memory channels, each labeled with ISO 26262 ASIL severity via three-expert consensus. On the public Who&When benchmark, CockpitHAT achieves agent-level / step-exact accuracies of 77.9% / 37.8% on the Hand-Crafted split and 86.5% / 46.0% on the Algorithm-Generated split, surpassing the text-only SOTA ECHO by up to 17.6 / 16.7 points. On CockpitBench, it attains 78.3% agent-level and 38.2% step-exact accuracy. These results establish dependency-aware, multi-channel, risk-calibrated attribution as an effective paradigm for reliable failure diagnosis in real-world embodied LLM multi-agent systems. |
| 2026-08-03 | [Leveraging AI for fine-grained food safety risk forecasting in sparse data conditions](http://arxiv.org/abs/2608.01767v1) | Dongqi Wang, Weiwei Chen et al. | Ensuring food safety represents a critical public health challenge, particularly when inspection resources are limited and regional sampling data are sparse. This study proposes a Transformer-based framework capable of forecasting fine-grained, city-level food safety risks by unifying over 11 million inspection records with supplemental demographic, economic, and environmental indicators extracted from the Statistical Yearbook. A three-stage pretraining design leverages partial supervision from the Wilson interval (capturing both safety and risk rankings), together with semi-supervised label refinement, to effectively utilize historical records even when local sample sizes are insufficient. Experimental evaluations on data from 2022 show that the proposed approach outperforms baselines significantly. A subsequent field experiment in collaboration with the Zhejiang Provincial Administration for Market Regulation further demonstrates improved detection rates and more efficient allocation of inspection resources compared to a manually developed plan. Observations of regulatory decision-making reveal a threshold-based heuristic employed by inspectors, hinting that additional training or decision-support interfaces could further enhance the impact of AI-generated risk scores. Overall, these findings underscore that a rigorous integration of large-scale public inspection data, Wilson interval-based confidence modeling, and advanced deep learning can facilitate earlier and more granular identification of food safety threats. By reducing reliance on reactive measures alone, the proposed framework has the potential to advance proactive, data-driven oversight of the global food supply. |
| 2026-08-03 | [EntailLLM: Verifying LLM-Generated Vulnerability Discovery Paths with Domain Knowledge via Logic Programming](http://arxiv.org/abs/2608.01763v1) | Kaustuv Mukherji, Jaikrishna Manojkumar Patil et al. | Large language models are increasingly used to reason about software vulnerabilities, but their outputs can silently violate domain knowledge, limiting their reliability in safety-critical settings such as medical devices. Prior work either treats that output as a prediction to be scored or constrains it to walks within a single knowledge graph; neither checks whether reasoning over a binary is consistent with an independent body of domain knowledge. We present EntailLLM, which validates each LLM-proposed analyst path by entailment: the path is a traversal of the binary's function call graph, the domain knowledge is represented in a separate graph, and verification aligns the two under temporal annotated logic. Across three CWE classes, four LLMs, three prompting strategies, and seven binaries varying in size from 405 to 12,696 function call-graph nodes, domain knowledge raises pooled entailment from 78% to 98%, with entailment decreasing in only 3% of the experiments. EntailLLM is deployed end-to-end on real medical-device binaries, reaching 98% pooled entailment without per-device tuning. Our system inherits the formal guarantees of generalized annotated logic, providing logical verification of LLM output that is both explainable and grounded in well-defined semantics. |

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



