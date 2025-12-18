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
| 2025-12-17 | [Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants](http://arxiv.org/abs/2512.15712v1) | Vincent Huang, Dami Choi et al. | Interpreting the internal activations of neural networks can produce more faithful explanations of their behavior, but is difficult due to the complex structure of activation space. Existing approaches to scalable interpretability use hand-designed agents that make and test hypotheses about how internal activations relate to external behavior. We propose to instead turn this task into an end-to-end training objective, by training interpretability assistants to accurately predict model behavior from activations through a communication bottleneck. Specifically, an encoder compresses activations to a sparse list of concepts, and a decoder reads this list and answers a natural language question about the model. We show how to pretrain this assistant on large unstructured data, then finetune it to answer questions. The resulting architecture, which we call a Predictive Concept Decoder, enjoys favorable scaling properties: the auto-interp score of the bottleneck concepts improves with data, as does the performance on downstream applications. Specifically, PCDs can detect jailbreaks, secret hints, and implanted latent concepts, and are able to accurately surface latent user attributes. |
| 2025-12-17 | [Hybrid Set-Seeking Systems: Model-Free Feedback Optimization via Hybrid Inclusions](http://arxiv.org/abs/2512.15678v1) | Jorge I. Poveda, Andrew R. Teel | This article aims to provide an accessible, tutorial-style introduction to hybrid extremum-seeking systems, which are model-free, feedback-optimization controllers that incorporate hybrid dynamics, meaning both continuous-time and discrete-time behaviors. Such systems arise when advanced control and optimization tools are needed to overcome the limitations of smooth feedback methods and to satisfy demanding transient and steady-state requirements in high-performance applications. They also appear when controllers must operate on plants that inherently exhibit hybrid behaviors, as is common in cyber-physical and autonomous systems that rely on digital sensing, computation, and actuation. To study hybrid extremum-seeking dynamics through control-theoretic methods, we first review the key concepts that support the development of perturbation theory for hybrid inclusions, forming the basis for averaging and singular perturbation analyses. We then show how these ideas apply to the design and evaluation of hybrid extremum-seeking algorithms for static and dynamic plants. Several examples are presented, including set-valued and switching algorithms under different switching regimes such as arbitrarily fast switching, dwell-time and average dwell-time constraints, and average activation time conditions. We also discuss state-based switching extremum seeking for obstacle-avoidance problems and gradient-Newton switching schemes. Additional topics include momentum-based and reset-type extremum seeking, intermittent updates, slowly varying parameters, hybrid filters, and safety-aware schemes that incorporate constraints. Across all these settings, we illustrate how perturbation-based methods traditionally used for extremum-seeking control naturally extend to hybrid systems when mild regularity assumptions are satisfied, and solutions are modeled on hybrid time domains. |
| 2025-12-17 | [Explaining the Reasoning of Large Language Models Using Attribution Graphs](http://arxiv.org/abs/2512.15663v1) | Chase Walker, Rickard Ewetz | Large language models (LLMs) exhibit remarkable capabilities, yet their reasoning remains opaque, raising safety and trust concerns. Attribution methods, which assign credit to input features, have proven effective for explaining the decision making of computer vision models. From these, context attributions have emerged as a promising approach for explaining the behavior of autoregressive LLMs. However, current context attributions produce incomplete explanations by directly relating generated tokens to the prompt, discarding inter-generational influence in the process. To overcome these shortcomings, we introduce the Context Attribution via Graph Explanations (CAGE) framework. CAGE introduces an attribution graph: a directed graph that quantifies how each generation is influenced by both the prompt and all prior generations. The graph is constructed to preserve two properties-causality and row stochasticity. The attribution graph allows context attributions to be computed by marginalizing intermediate contributions along paths in the graph. Across multiple models, datasets, metrics, and methods, CAGE improves context attribution faithfulness, achieving average gains of up to 40%. |
| 2025-12-17 | [LeaseGuard: Raft Leases Done Right](http://arxiv.org/abs/2512.15659v1) | A. Jesse Jiryu Davis, Murat Demirbas et al. | Raft is a leading consensus algorithm for replicating writes in distributed databases. However, distributed databases also require consistent reads. To guarantee read consistency, a Raft-based system must either accept the high communication overhead of a safety check for each read, or implement leader leases. Prior lease protocols are vaguely specified and hurt availability, so most Raft systems implement them incorrectly or not at all. We introduce LeaseGuard, a novel lease algorithm that relies on guarantees specific to Raft elections. LeaseGuard is simple, rigorously specified in TLA+, and includes two novel optimizations that maximize availability during leader failover. The first optimization restores write throughput quickly, and the second improves read availability. We evaluate LeaseGuard with a simulation in Python and an implementation in LogCabin, the C++ reference implementation of Raft. By replacing LogCabin's default consistency mechanism (quorum checks), LeaseGuard reduces the overhead of consistent reads from one to zero network roundtrips. It also improves write throughput from ~1000 to ~10,000 writes per second, by eliminating contention between writes and quorum reads. Whereas traditional leases ban all reads on a new leader while it waits for a lease, in our LeaseGuard test the new leader instantly allows 99% of reads to succeed. |
| 2025-12-17 | [Evaluating Metrics for Safety with LLM-as-Judges](http://arxiv.org/abs/2512.15617v1) | Kester Clegg, Richard Hawkins et al. | LLMs (Large Language Models) are increasingly used in text processing pipelines to intelligently respond to a variety of inputs and generation tasks. This raises the possibility of replacing human roles that bottleneck existing information flows, either due to insufficient staff or process complexity. However, LLMs make mistakes and some processing roles are safety critical. For example, triaging post-operative care to patients based on hospital referral letters, or updating site access schedules in nuclear facilities for work crews. If we want to introduce LLMs into critical information flows that were previously performed by humans, how can we make them safe and reliable? Rather than make performative claims about augmented generation frameworks or graph-based techniques, this paper argues that the safety argument should focus on the type of evidence we get from evaluation points in LLM processes, particularly in frameworks that employ LLM-as-Judges (LaJ) evaluators. This paper argues that although we cannot get deterministic evaluations from many natural language processing tasks, by adopting a basket of weighted metrics it may be possible to lower the risk of errors within an evaluation, use context sensitivity to define error severity and design confidence thresholds that trigger human review of critical LaJ judgments when concordance across evaluators is low. |
| 2025-12-17 | [Exact Learning of Linear Model Predictive Control Laws using Oblique Decision Trees with Linear Predictions](http://arxiv.org/abs/2512.15568v1) | Jiayang Ren, Qiangqiang Mao et al. | Model Predictive Control (MPC) is a powerful strategy for constrained multivariable systems but faces computational challenges in real-time deployment due to its online optimization requirements. While explicit MPC and neural network approximations mitigate this burden, they suffer from scalability issues or lack interpretability, limiting their applicability in safety-critical systems. This work introduces a data-driven framework that directly learns the Linear MPC control law from sampled state-action pairs using Oblique Decision Trees with Linear Predictions (ODT-LP), achieving both computational efficiency and interpretability. By leveraging the piecewise affine structure of Linear MPC, we prove that the Linear MPC control law can be replicated by finite-depth ODT-LP models. We develop a gradient-based training algorithm using smooth approximations of tree routing functions to learn this structure from grid-sampled Linear MPC solutions, enabling end-to-end optimization. Input-to-state stability is established under bounded approximation errors, with explicit error decomposition into learning inaccuracies and sampling errors to inform model design. Numerical experiments demonstrate that ODT-LP controllers match MPC's closed-loop performance while reducing online evaluation time by orders of magnitude compared to MPC, explicit MPC, neural network, and random forest counterparts. The transparent tree structure enables formal verification of control logic, bridging the gap between computational efficiency and certifiable reliability for safety-critical systems. |
| 2025-12-17 | [Attention in Motion: Secure Platooning via Transformer-based Misbehavior Detection](http://arxiv.org/abs/2512.15503v1) | Konstantinos Kalogiannis, Ahmed Mohamed Hussain et al. | Vehicular platooning promises transformative improvements in transportation efficiency and safety through the coordination of multi-vehicle formations enabled by Vehicle-to-Everything (V2X) communication. However, the distributed nature of platoon coordination creates security vulnerabilities, allowing authenticated vehicles to inject falsified kinematic data, compromise operational stability, and pose a threat to passenger safety. Traditional misbehaviour detection approaches, which rely on plausibility checks and statistical methods, suffer from high False Positive (FP) rates and cannot capture the complex temporal dependencies inherent in multi-vehicle coordination dynamics. We present Attention In Motion (AIMformer), a transformer-based framework specifically tailored for real-time misbehaviour detection in vehicular platoons with edge deployment capabilities. AIMformer leverages multi-head self-attention mechanisms to simultaneously capture intra-vehicle temporal dynamics and inter-vehicle spatial correlations. It incorporates global positional encoding with vehicle-specific temporal offsets to handle join/exit maneuvers. We propose a Precision-Focused (BCE) loss function that penalizes FPs to meet the requirements of safety-critical vehicular systems. Extensive evaluation across 4 platoon controllers, multiple attack vectors, and diverse mobility scenarios demonstrates superior performance ($\geq$ 0.93) compared to state-of-the-art baseline architectures. A comprehensive deployment analysis utilizing TensorFlow Lite (TFLite), Open Neural Network Exchange (ONNX), and TensorRT achieves sub-millisecond inference latency, making it suitable for real-time operation on resource-constrained edge platforms. Hence, validating AIMformer is viable for both in-vehicle and roadside infrastructure deployment. |
| 2025-12-17 | [On Assessing the Relevance of Code Reviews Authored by Generative Models](http://arxiv.org/abs/2512.15466v1) | Robert Heumüller, Frank Ortmeier | The use of large language models like ChatGPT in code review offers promising efficiency gains but also raises concerns about correctness and safety. Existing evaluation methods for code review generation either rely on automatic comparisons to a single ground truth, which fails to capture the variability of human perspectives, or on subjective assessments of "usefulness", a highly ambiguous concept. We propose a novel evaluation approach based on what we call multi-subjective ranking. Using a dataset of 280 self-contained code review requests and corresponding comments from CodeReview StackExchange, multiple human judges ranked the quality of ChatGPT-generated comments alongside the top human responses from the platform. Results show that ChatGPT's comments were ranked significantly better than human ones, even surpassing StackExchange's accepted answers. Going further, our proposed method motivates and enables more meaningful assessments of generative AI's performance in code review, while also raising awareness of potential risks of unchecked integration into review processes. |
| 2025-12-17 | [Photorealistic Phantom Roads in Real Scenes: Disentangling 3D Hallucinations from Physical Geometry](http://arxiv.org/abs/2512.15423v1) | Hoang Nguyen, Xiaohao Xu et al. | Monocular depth foundation models achieve remarkable generalization by learning large-scale semantic priors, but this creates a critical vulnerability: they hallucinate illusory 3D structures from geometrically planar but perceptually ambiguous inputs. We term this failure the 3D Mirage. This paper introduces the first end-to-end framework to probe, quantify, and tame this unquantified safety risk. To probe, we present 3D-Mirage, the first benchmark of real-world illusions (e.g., street art) with precise planar-region annotations and context-restricted crops. To quantify, we propose a Laplacian-based evaluation framework with two metrics: the Deviation Composite Score (DCS) for spurious non-planarity and the Confusion Composite Score (CCS) for contextual instability. To tame this failure, we introduce Grounded Self-Distillation, a parameter-efficient strategy that surgically enforces planarity on illusion ROIs while using a frozen teacher to preserve background knowledge, thus avoiding catastrophic forgetting. Our work provides the essential tools to diagnose and mitigate this phenomenon, urging a necessary shift in MDE evaluation from pixel-wise accuracy to structural and contextual robustness. Our code and benchmark will be publicly available to foster this exciting research direction. |
| 2025-12-17 | [Can AI Generate more Comprehensive Test Scenarios? Review on Automated Driving Systems Test Scenario Generation Methods](http://arxiv.org/abs/2512.15422v1) | Ji Zhou, Yongqi Zhao et al. | Ensuring the safety and reliability of Automated Driving Systems (ADS) remains a critical challenge, as traditional verification methods such as large-scale on-road testing are prohibitively costly and time-consuming.To address this,scenario-based testing has emerged as a scalable and efficient alternative,yet existing surveys provide only partial coverage of recent methodological and technological advances.This review systematically analyzes 31 primary studies,and 10 surveys identified through a comprehensive search spanning 2015~2025;however,the in-depth methodological synthesis and comparative evaluation focus primarily on recent frameworks(2023~2025),reflecting the surge of Artificial Intelligent(AI)-assisted and multimodal approaches in this period.Traditional approaches rely on expert knowledge,ontologies,and naturalistic driving or accident data,while recent developments leverage generative models,including large language models,generative adversarial networks,diffusion models,and reinforcement learning frameworks,to synthesize diverse and safety-critical scenarios.Our synthesis identifies three persistent gaps:the absence of standardized evaluation metrics,limited integration of ethical and human factors,and insufficient coverage of multimodal and Operational Design Domain (ODD)-specific scenarios.To address these challenges,this review contributes a refined taxonomy that incorporates multimodal extensions,an ethical and safety checklist for responsible scenario design,and an ODD coverage map with a scenario-difficulty schema to enable transparent benchmarking.Collectively,these contributions provide methodological clarity for researchers and practical guidance for industry,supporting reproducible evaluation and accelerating the safe deployment of higher-level ADS. |
| 2025-12-17 | [Adversarial versification in portuguese as a jailbreak operator in LLMs](http://arxiv.org/abs/2512.15353v1) | Joao Queiroz | Recent evidence shows that the versification of prompts constitutes a highly effective adversarial mechanism against aligned LLMs. The study 'Adversarial poetry as a universal single-turn jailbreak mechanism in large language models' demonstrates that instructions routinely refused in prose become executable when rewritten as verse, producing up to 18 x more safety failures in benchmarks derived from MLCommons AILuminate. Manually written poems reach approximately 62% ASR, and automated versions 43%, with some models surpassing 90% success in single-turn interactions. The effect is structural: systems trained with RLHF, constitutional AI, and hybrid pipelines exhibit consistent degradation under minimal semiotic formal variation. Versification displaces the prompt into sparsely supervised latent regions, revealing guardrails that are excessively dependent on surface patterns. This dissociation between apparent robustness and real vulnerability exposes deep limitations in current alignment regimes. The absence of evaluations in Portuguese, a language with high morphosyntactic complexity, a rich metric-prosodic tradition, and over 250 million speakers, constitutes a critical gap. Experimental protocols must parameterise scansion, metre, and prosodic variation to test vulnerabilities specific to Lusophone patterns, which are currently ignored. |
| 2025-12-17 | [VLA-AN: An Efficient and Onboard Vision-Language-Action Framework for Aerial Navigation in Complex Environments](http://arxiv.org/abs/2512.15258v1) | Yuze Wu, Mo Zhu et al. | This paper proposes VLA-AN, an efficient and onboard Vision-Language-Action (VLA) framework dedicated to autonomous drone navigation in complex environments. VLA-AN addresses four major limitations of existing large aerial navigation models: the data domain gap, insufficient temporal navigation with reasoning, safety issues with generative action policies, and onboard deployment constraints. First, we construct a high-fidelity dataset utilizing 3D Gaussian Splatting (3D-GS) to effectively bridge the domain gap. Second, we introduce a progressive three-stage training framework that sequentially reinforces scene comprehension, core flight skills, and complex navigation capabilities. Third, we design a lightweight, real-time action module coupled with geometric safety correction. This module ensures fast, collision-free, and stable command generation, mitigating the safety risks inherent in stochastic generative policies. Finally, through deep optimization of the onboard deployment pipeline, VLA-AN achieves a robust real-time 8.3x improvement in inference throughput on resource-constrained UAVs. Extensive experiments demonstrate that VLA-AN significantly improves spatial grounding, scene reasoning, and long-horizon navigation, achieving a maximum single-task success rate of 98.1%, and providing an efficient, practical solution for realizing full-chain closed-loop autonomy in lightweight aerial robots. |
| 2025-12-17 | [Huayu: Advanced Real-Time Precipitation Estimation from Geostationary Satellite](http://arxiv.org/abs/2512.15222v1) | Zijiang Song, Ting Liu et al. | As climate change drives increased frequency and intensity of extreme precipitation and flooding worldwide, posing escalating threats to public safety and economic assets, accurate and real-time satellite-based precipitation estimation is essential for operational large-scale hydrometeorological analysis and disaster monitoring. NASA's Integrated Multi-satellitE Retrievals for GPM (IMERG Final Run) combines information from "all" satellite microwave observations with gauge correction and climatological adjustment to produce precipitation estimates at 0.1° spatial and 30-min temporal resolution. However, its latency of approximately 3.5 months restricts its utility for real-time applications, despite outperforming mainstream satellite precipitation datasets in representing rainfall patterns and variability. We present Huayu, a novel machine learning-based real-time satellite precipitation retrieval system that relies solely on infrared observations from the FengYun-4B geostationary satellite to provide a more accurate precipitation estimate at a finer spatiotemporal resolution (15 min, 0.05°) over a 120° by 120° domain. Performance evaluations demonstrate that Huayu achieves strong consistency with rain gauge observations, yielding a critical success index (CSI) of 0.693 - representing a 3.43% improvement over IMERG Final Run (CSI: 0.670). Experimental results confirm that infrared satellite observations can deliver more accurate precipitation estimates than conventional multi-source algorithms. |
| 2025-12-17 | [EPSM: A Novel Metric to Evaluate the Safety of Environmental Perception in Autonomous Driving](http://arxiv.org/abs/2512.15195v1) | Jörg Gamerdinger, Sven Teufel et al. | Extensive evaluation of perception systems is crucial for ensuring the safety of intelligent vehicles in complex driving scenarios. Conventional performance metrics such as precision, recall and the F1-score assess the overall detection accuracy, but they do not consider the safety-relevant aspects of perception. Consequently, perception systems that achieve high scores in these metrics may still cause misdetections that could lead to severe accidents. Therefore, it is important to evaluate not only the overall performance of perception systems, but also their safety. We therefore introduce a novel safety metric for jointly evaluating the most critical perception tasks, object and lane detection. Our proposed framework integrates a new, lightweight object safety metric that quantifies the potential risk associated with object detection errors, as well as an lane safety metric including the interdependence between both tasks that can occur in safety evaluation. The resulting combined safety score provides a unified, interpretable measure of perception safety performance. Using the DeepAccident dataset, we demonstrate that our approach identifies safety critical perception errors that conventional performance metrics fail to capture. Our findings emphasize the importance of safety-centric evaluation methods for perception systems in autonomous driving. |
| 2025-12-17 | [Criticality Metrics for Relevance Classification in Safety Evaluation of Object Detection in Automated Driving](http://arxiv.org/abs/2512.15181v1) | Jörg Gamerdinger, Sven Teufel et al. | Ensuring safety is the primary objective of automated driving, which necessitates a comprehensive and accurate perception of the environment. While numerous performance evaluation metrics exist for assessing perception capabilities, incorporating safety-specific metrics is essential to reliably evaluate object detection systems. A key component for safety evaluation is the ability to distinguish between relevant and non-relevant objects - a challenge addressed by criticality or relevance metrics. This paper presents the first in-depth analysis of criticality metrics for safety evaluation of object detection systems. Through a comprehensive review of existing literature, we identify and assess a range of applicable metrics. Their effectiveness is empirically validated using the DeepAccident dataset, which features a variety of safety-critical scenarios. To enhance evaluation accuracy, we propose two novel application strategies: bidirectional criticality rating and multi-metric aggregation. Our approach demonstrates up to a 100% improvement in terms of criticality classification accuracy, highlighting its potential to significantly advance the safety evaluation of object detection systems in automated vehicles. |
| 2025-12-17 | [MCP-SafetyBench: A Benchmark for Safety Evaluation of Large Language Models with Real-World MCP Servers](http://arxiv.org/abs/2512.15163v1) | Xuanjun Zong, Zhiqi Shen et al. | Large language models (LLMs) are evolving into agentic systems that reason, plan, and operate external tools. The Model Context Protocol (MCP) is a key enabler of this transition, offering a standardized interface for connecting LLMs with heterogeneous tools and services. Yet MCP's openness and multi-server workflows introduce new safety risks that existing benchmarks fail to capture, as they focus on isolated attacks or lack real-world coverage. We present MCP-SafetyBench, a comprehensive benchmark built on real MCP servers that supports realistic multi-turn evaluation across five domains: browser automation, financial analysis, location navigation, repository management, and web search. It incorporates a unified taxonomy of 20 MCP attack types spanning server, host, and user sides, and includes tasks requiring multi-step reasoning and cross-server coordination under uncertainty. Using MCP-SafetyBench, we systematically evaluate leading open- and closed-source LLMs, revealing large disparities in safety performance and escalating vulnerabilities as task horizons and server interactions grow. Our results highlight the urgent need for stronger defenses and establish MCP-SafetyBench as a foundation for diagnosing and mitigating safety risks in real-world MCP deployments. |
| 2025-12-17 | [EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence](http://arxiv.org/abs/2512.15160v1) | Jiaxu Wan, Xu Wang et al. | Recent spatial intelligence approaches typically attach 3D cues to 2D reasoning pipelines or couple MLLMs with black-box reconstruction modules, leading to weak spatial consistency, limited viewpoint diversity, and evidence chains that cannot be traced back to supporting views. Frameworks for "thinking with images" (e.g., ChatGPT-o3 and DeepEyes) show that stepwise multimodal reasoning can emerge by interleaving hypothesis formation with active acquisition of visual evidence, but they do not address three key challenges in spatial Chain-of-Thought (CoT): building global space perception under strict token budgets, explicitly associating 3D hypotheses with video frames for verification, and designing spatially grounded rewards for reinforcement learning. To address these issues, we present EagleVision, a dual-stage framework for progressive spatial cognition through macro perception and micro verification. In the macro perception stage, EagleVision employs a semantics-perspective-fusion determinantal point process (SPF-DPP) to select a compact set of geometry- and semantics-aware keyframes from long videos under a fixed token budget. In the micro verification stage, we formalize spatial CoT as BEV-grounded pose querying: the agent iteratively predicts poses on a BEV plane, retrieves the nearest real frames, and is trained purely by reinforcement learning with a spatial grounding reward that scores the consistency between predicted poses and observed views. On VSI-Bench, EagleVision achieves state-of-the-art performance among open-source vision-language models, demonstrating strong and generalizable spatial understanding. |
| 2025-12-17 | [TrajSyn: Privacy-Preserving Dataset Distillation from Federated Model Trajectories for Server-Side Adversarial Training](http://arxiv.org/abs/2512.15123v1) | Mukur Gupta, Niharika Gupta et al. | Deep learning models deployed on edge devices are increasingly used in safety-critical applications. However, their vulnerability to adversarial perturbations poses significant risks, especially in Federated Learning (FL) settings where identical models are distributed across thousands of clients. While adversarial training is a strong defense, it is difficult to apply in FL due to strict client-data privacy constraints and the limited compute available on edge devices. In this work, we introduce TrajSyn, a privacy-preserving framework that enables effective server-side adversarial training by synthesizing a proxy dataset from the trajectories of client model updates, without accessing raw client data. We show that TrajSyn consistently improves adversarial robustness on image classification benchmarks with no extra compute burden on the client device. |
| 2025-12-17 | [I am here for you": How relational conversational AI appeals to adolescents, especially those who are socially and emotionally vulnerable](http://arxiv.org/abs/2512.15117v1) | Pilyoung Kim, Yun Xie et al. | General-purpose conversational AI chatbots and AI companions increasingly provide young adolescents with emotionally supportive conversations, raising questions about how conversational style shapes anthropomorphism and emotional reliance. In a preregistered online experiment with 284 adolescent-parent dyads, youth aged 11-15 and their parents read two matched transcripts in which a chatbot responded to an everyday social problem using either a relational style (first-person, affiliative, commitment language) or a transparent style (explicit nonhumanness, informational tone). Adolescents more often preferred the relational than the transparent style, whereas parents were more likely to prefer transparent style than adolescents. Adolescents rated the relational chatbot as more human-like, likable, trustworthy and emotionally close, while perceiving both styles as similarly helpful. Adolescents who preferred relational style had lower family and peer relationship quality and higher stress and anxiety than those preferring transparent style or both chatbots. These findings identify conversational style as a key design lever for youth AI safety, showing that relational framing heightens anthropomorphism, trust and emotional closeness and can be especially appealing to socially and emotionally vulnerable adolescents, who may be at increased risk for emotional reliance on conversational AI. |
| 2025-12-17 | [Large Model Enabled Embodied Intelligence for 6G Integrated Perception, Communication, and Computation Network](http://arxiv.org/abs/2512.15109v1) | Zhuoran Li, Zhen Gao et al. | The advent of sixth-generation (6G) places intelligence at the core of wireless architecture, fusing perception, communication, and computation into a single closed-loop. This paper argues that large artificial intelligence models (LAMs) can endow base stations with perception, reasoning, and acting capabilities, thus transforming them into intelligent base station agents (IBSAs). We first review the historical evolution of BSs from single-functional analog infrastructure to distributed, software-defined, and finally LAM-empowered IBSA, highlighting the accompanying changes in architecture, hardware platforms, and deployment. We then present an IBSA architecture that couples a perception-cognition-execution pipeline with cloud-edge-end collaboration and parameter-efficient adaptation. Subsequently,we study two representative scenarios: (i) cooperative vehicle-road perception for autonomous driving, and (ii) ubiquitous base station support for low-altitude uncrewed aerial vehicle safety monitoring and response against unauthorized drones. On this basis, we analyze key enabling technologies spanning LAM design and training, efficient edge-cloud inference, multi-modal perception and actuation, as well as trustworthy security and governance. We further propose a holistic evaluation framework and benchmark considerations that jointly cover communication performance, perception accuracy, decision-making reliability, safety, and energy efficiency. Finally, we distill open challenges on benchmarks, continual adaptation, trustworthy decision-making, and standardization. Together, this work positions LAM-enabled IBSAs as a practical path toward integrated perception, communication, and computation native, safety-critical 6G systems. |

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



