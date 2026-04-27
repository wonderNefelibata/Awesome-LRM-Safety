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
| 2026-04-24 | [A Unified Framework for Multiple Exposure Distributed Lag Non-Linear Models for Air Pollution Epidemiology](http://arxiv.org/abs/2604.22692v1) | Tianyi Pan, Hwashin Hyun Shin et al. | This study quantifies the association between air pollution and mortality in Ontario, Canada. Exposure-response relationships in air pollution epidemiology are complex due to three features: time-lagged associations, non-linear associations, and multiple pollutants. To address the first two features, two distinct classes of distributed lag non-linear model (DLNM) have been proposed, but extending them to multiple exposures and selecting an appropriate model remain challenging. We propose a unified framework for multiple exposure DLNMs, integrating model specification, estimation, selection and stacking. The framework applies to four different model structures: two additive and two proposed single-index DLNMs, all applicable to general outcome types, including the mortality counts in the motivating application. We develop an estimation approach that applies to all four models. Choosing among the candidate DLNMs is challenging a priori, and we derive an AIC to select among them. As an alternative to selecting a single model, we also extend a model stacking approach to combine inferences across the four DLNMs and propose an implementation scalable to our dataset with 106,346 observations. In the motivating analysis, the four DLNMs yield different estimates, and the proposed stacking approach identifies significant associations between respiratory mortality and a mixture of PM2.5, O3 and NO2. |
| 2026-04-24 | [Inferring Equivalence Classes from Legacy Undocumented Embedded Binaries for ISO 26262-Compliant Testing](http://arxiv.org/abs/2604.22673v1) | Marco De Luca, Domenico Francesco De Angelis et al. | Equivalence class partitioning is a well-established test design technique mandated by safety standards such as ISO~26262 for systematic testing of safety software. In industrial practice, however, its application to legacy undocumented embedded firmware is often hindered by incomplete or outdated functional specifications.   This paper proposes a binary-level methodology for inferring output-oriented equivalence classes directly from compiled firmware, without relying on source-level annotations or external documentation. The approach combines control-flow reconstruction and guided symbolic execution to analyze individual functions and group execution paths according to indistinguishable observable behavior, including return values and output parameters. An optional post-processing step produces human-readable representations to support comprehension and documentation.   The methodology is evaluated in an industrial automotive context through a practitioner-based study assessing correctness and interpretability. Results indicate strong alignment with expert expectations and a positive perception of readability and usefulness for supporting function understanding and test design. These findings demonstrate the feasibility and practical relevance of binary-level equivalence class inference for systematic testing of legacy undocumented safety-embedded software. |
| 2026-04-24 | [Compositional Online Learning for Multi-Objective System Co-Design](http://arxiv.org/abs/2604.22624v1) | Meshal Alharbi, Munther A. Dahleh et al. | Many engineered systems must balance competing objectives, such as performance and safety, cost and reliability, or efficiency and sustainability, and are naturally modeled as compositions of interacting subsystems. We study online multi-objective decision-making in monotone co-design, where functionalities and resources are partially ordered, and the goal is to identify the target-feasible antichain of non-dominated trade-offs using few expensive evaluations. We introduce optimistic evaluators: history-dependent bounds on functionality and resource mappings that enable safe elimination of implementations before full evaluation. Based on these evaluators, we develop an elimination-based rejection-sampling algorithm, prove its soundness, and show that the admissible region shrinks monotonically as information accumulates. We instantiate the framework under monotonicity, Lipschitz continuity, and linear-parametric structure. For compositional co-design problems modeled by multigraphs, we show how local optimistic certificates propagate through the tractable remainder of the graph to yield system-level optimistic feasibility and resource bounds. Experiments on multi-robot fleet design, intermodal mobility systems, and synthetic monotone and Lipschitz benchmarks show substantial sample-efficiency gains over uniform sampling, Bayesian optimization, and multi-objective evolutionary algorithms. |
| 2026-04-24 | [Chamelio: A Fast Shared Cloud Network Stack for Isolated Tenant-Defined Protocols](http://arxiv.org/abs/2604.22603v1) | Matheus Stolet, Simon Peter et al. | Conventional cloud network virtualization sends packets through multiple guest and host layers, inflating CPU cost and tail latency. Shared host datapaths collapse this layering into one optimized path across tenants, but existing shared stacks are fixed-function: tenants cannot specialize their protocols. eBPF is the natural vehicle for restoring programmability to a shared datapath, but today's extensions are hook-sized, and its verifier provides safety -- not performance isolation: one tenant's per-packet work can inflate every other tenant's tail latency.   Chamelio is a programmable shared network stack that lets tenants implement full protocols through a bounded eBPF fast path and a tenant slow path, while approaching the performance and preserving the strong isolation of fixed shared stacks. It combines three ideas: a shared-stack architecture for tenant-defined protocols; joint optimisation of tenant handlers with provider infrastructure and co-resident tenants in the shared fast path; and a bounded fast path contract with runtime cycle accounting that keeps tenant programmability compatible with strong performance isolation. A tenant programmable TCP on Chamelio reaches 9.2 Mreq/s, matching the hand-tuned TAS stack; joint compilation shrinks the programmability tax from 23.9% to 3.8%; and under a scaling TCP adversary that drives uninstrumented stacks to 154 microseconds, Chamelio bounds victim tail latency at 46 microseconds. |
| 2026-04-24 | [RedVLA: Physical Red Teaming for Vision-Language-Action Models](http://arxiv.org/abs/2604.22591v1) | Yuhao Zhang, Borong Zhang et al. | The real-world deployment of Vision-Language-Action (VLA) models remains limited by the risk of unpredictable and irreversible physical harm. However, we currently lack effective mechanisms to proactively detect these physical safety risks before deployment. To address this gap, we propose \textbf{RedVLA}, the first red teaming framework for physical safety in VLA models. We systematically uncover unsafe behaviors through a two-stage process: (I) \textbf{Risk Scenario Synthesis} constructs a valid and task-feasible initial risk scene. Specifically, it identifies critical interaction regions from benign trajectories and positions the risk factor within these regions, aiming to entangle it with the VLA's execution flow and elicit a target unsafe behavior. (II) \textbf{Risk Amplification} ensures stable elicitation across heterogeneous models. It iteratively refines the risk factor state through gradient-free optimization guided by trajectory features. Experiments on six representative VLA models show that RedVLA uncovers diverse unsafe behaviors and achieves the ASR up to 95.5\% within 10 optimization iterations. To mitigate these risks, we further propose SimpleVLA-Guard, a lightweight safety guard built from RedVLA-generated data. Our data, assets, and code are available \href{https://redvla.github.io}{here}. |
| 2026-04-24 | [Transferable Physical-World Adversarial Patches Against Pedestrian Detection Models](http://arxiv.org/abs/2604.22552v1) | Shihui Yan, Ziqi Zhou et al. | Physical adversarial patch attacks critically threaten pedestrian detection, causing surveillance and autonomous driving systems to miss pedestrians and creating severe safety risks. Despite their effectiveness in controlled settings, existing physical attacks face two major limitations in practice: they lack systematic disruption of the multi-stage decision pipeline, enabling residual modules to offset perturbations, and they fail to model complex physical variations, leading to poor robustness. To overcome these limitations, we propose a novel pedestrian adversarial patch generation method that combines multi-stage collaborative attacks with robustness enhancement under physical diversity, called TriPatch. Specifically, we design a triplet loss consisting of detection confidence suppression, bounding-box offset amplification, and non-maximum suppression (NMS) disruption, which jointly act across different stages of the detection pipeline. In addition, we introduce an appearance consistency loss to constrain the color distribution of the patch, thereby improving its adaptability under diverse imaging conditions, and incorporate data augmentation to further enhance robustness against complex physical perturbations. Extensive experiments demonstrate that TriPatch achieves a higher attack success rate across multiple detector models compared to existing approaches. |
| 2026-04-24 | [Multi-output Extreme Spatial Model for Complex Aircraft Production Systems](http://arxiv.org/abs/2604.22548v1) | Cheolhei Lee, Xing Wang et al. | Problem definition: Data-driven models in machine learning have enabled efficient management of production systems. However, a majority of machine learning models are devoted to modeling the mean response or average pattern, which is inappropriate for studying abnormal extreme events that are often of primary interest in aircraft manufacturing. Since extreme events from heavy-tailed distributions give rise to prohibitive expenditures in system management, sophisticated extreme models are urgently needed to analyze complex extreme risks. Engineering applications of extreme models usually focus on individual extreme events, which is insufficient for complex systems with correlations. Methodology/results: We introduce an extreme spatial model for multi-output response control systems that efficiently captures the dynamics using a bilinear function on two spatial domains for control variables and measurement locations. Marginal parameter modeling and extremal dependence have been investigated. In addition, an efficient graph-assisted composite likelihood estimation and corresponding computational algorithms are developed to cope with high-dimensional outputs. The application to composite aircraft production shows that the proposed model enables comprehensive analyses with superior predictive performance on extreme events compared to canonical methods. Managerial implications: Our method shows how to use an extreme spatial model for predicting extreme events and managing extreme risks in complex production systems such as aircraft. This can help achieve better quality management and operation safety in aircraft production systems and beyond. |
| 2026-04-24 | [On the Properties of Feature Attribution for Supervised Contrastive Learning](http://arxiv.org/abs/2604.22540v1) | Leonardo Arrighi, Julia Eva Belloni et al. | Most Neural Networks (NNs) for classification are trained using Cross-Entropy as a loss function. This approach requires the model to have an explicit classification layer. However, there exist alternative approaches, such as Contrastive Learning (CL). Instead of explicitly operating a classification, CL has the NN produce an embedding space where projections of similar data are pulled together, while projections of dissimilar data are pushed apart. In the case of Supervised CL (SCL), labels are adopted as similarity criteria, thus creating an embedding space where the projected data points are well-clustered. SCL provides crucial advantages over CE with regard to adversarial robustness and out-of-distribution detection, thus making it a more natural choice in safety-critical scenarios. In the present paper, we empirically show that NNs for image classification trained with SCL present higher-quality feature attribution explanations than CL with regard to faithfulness, complexity, and continuity. These results reinforce previous findings about CL-based approaches when targeting more trustworthy and transparent NNs and can guide practitioners in the selection of training objectives targeting not only accuracy, but also transparency of the models. |
| 2026-04-24 | [Benchmarking LLM-Driven Network Configuration Repair](http://arxiv.org/abs/2604.22513v1) | Ioannis Protogeros, Rufat Asadli et al. | There is a rapidly growing interest in using Large Language Models (LLMs) to automate complex network operations, but their reliable adoption requires rigorous assessment of their effectiveness and safety. Existing benchmarks do not address whether LLMs can successfully resolve errors in large-scale, interdependent network configurations without introducing new disruptions. Developing such a benchmark is challenging: scenarios must be diverse and increasingly complex, yet their evaluation must be straightforward and meaningful. In this paper, we present Cornetto, the first benchmark to evaluate LLM-driven network configuration repair functionally and at scale. Cornetto features a generation pipeline that synthesizes representative and plausible misconfiguration scenarios, coupled with an evaluation framework that uses formal verification to assess functional correctness of proposed fixes against ground-truth specifications. Using this pipeline, we synthesize a dataset of 231 problems for fixing configurations across varying network topologies (20--754 nodes) and diverse protocols. We evaluate 9 state-of-the-art LLMs and find that while they show promise, they often introduce regressions and their performance degrades at scale. Our results indicate that reliable LLM-powered network automation requires integrating LLMs into iterative workflows guided by formal verification. |
| 2026-04-24 | [Improving Driver Drowsiness Detection via Personalized EAR/MAR Thresholds and CNN-Based Classification](http://arxiv.org/abs/2604.22479v1) | Gökdeniz Ersoy, Mehmet Alper Tatar et al. | Driver drowsiness is a major cause of traffic accidents worldwide, posing a serious threat to public safety. Vision-based driver monitoring systems often rely on fixed Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) thresholds; however, such fixed values frequently fail to generalize across individuals due to variations in facial structure, illumination, and driving conditions. This paper proposes a personalized driver drowsiness detection system that monitors eyelid movements, head position, and yawning behavior in real time and provides warnings when signs of fatigue are detected. The system employs driver-specific EAR and MAR thresholds, calibrated before driving, to improve classical metric-based detection. In addition, deep learning-based Convolutional Neural Network (CNN) models are integrated to enhance accuracy in challenging scenarios. The system is evaluated using publicly available datasets as well as a custom dataset collected under diverse lighting conditions, head poses, and user characteristics. Experimental results show that personalized thresholding improves detection accuracy by 2-3% compared to fixed thresholds, while CNN-based classification achieves 99.1% accuracy for eye state detection and 98.8% for yawning detection, demonstrating the effectiveness of combining classical metrics with deep learning for robust real-time driver monitoring. |
| 2026-04-24 | [Automation-Exploit: A Multi-Agent LLM Framework for Adaptive Offensive Security with Digital Twin-Based Risk-Mitigated Exploitation](http://arxiv.org/abs/2604.22427v1) | Biagio Andreucci, Arcangelo Castiglione | The offensive security landscape is highly fragmented: enterprise platforms avoid memory-corruption vulnerabilities due to Denial of Service (DoS) risks, Automatic Exploit Generation (AEG) systems suffer from semantic blindness, and Large Language Model (LLM) agents face safety alignment filters and "Live Fire" execution hazards. We introduce Automation-Exploit, a fully autonomous Multi-Agent System (MAS) framework designed for adaptive offensive security in complex black-box scenarios. It bridges the abstraction gap between reconnaissance and exploitation by autonomously exfiltrating executables and contextual intelligence across multiple protocols, using this data to fuel both logical and binary attack chains. The framework introduces an adaptive safety architecture to mitigate DoS risks. While it natively resolves logical and web-based vulnerabilities, it employs a conditional isomorphic validation for high-risk memory-corruption flaws: if the target binary is successfully exfiltrated, it dynamically instantiates a cross-platform digital twin. By enforcing strict state synchronization, including libc alignment and runtime file descriptor hooking, potentially destructive payloads are iteratively debugged in an isolated replica. This enables a highly risk-mitigated "one-shot" execution on the physical target. Empirical evaluations across eight scenarios, including undocumented zero-day environments to rule out LLM data contamination, validate the framework's architectural resilience, demonstrating its ability to prevent "live fire" crashes and execute risk-mitigated compromises on actual targets. |
| 2026-04-24 | [Rethinking AI-Mediated Minority Support in Power-Imbalanced Group Decision-Making: From Anonymity To Authenticity](http://arxiv.org/abs/2604.22319v1) | Soohwan Lee, Kyungho Lee | AI-mediated Communication (AIMC) systems increasingly aim to protect minority voices by anonymizing or proxying their input, but anonymity and authenticity are not the same construct. This position paper draws on an ongoing empirical study comparing two LLM-powered minority support strategies in hierarchical group decision-making. We found that relaying minority input anonymously through AI increased participation but significantly reduced psychological safety and satisfaction, while generating only autonomous counterarguments improved satisfaction and reduced marginalization. These counterintuitive findings reveal three provocations for AIMC design in hierarchical contexts: the inherent trade-offs among anonymity, authenticity, agency, and accountability; the risk that power asymmetry reverses intended effects; and the need for AI to facilitate group reflection rather than substitute for human responsibility. These findings and provocations are offered as a contribution to the Restoring Human Authenticity in AI-Mediated Communication workshop. |
| 2026-04-24 | [Introducing the Cyber-Physical Data Flow Diagram to Improve Threat Modelling of Internet of Things Devices](http://arxiv.org/abs/2604.22307v1) | Simon Liebl, Ian Ferguson et al. | A growing number of Internet of Things (IoT) devices are used across consumer, medical, and industrial domains. They interact with their environment through sensors and actuators and connect to networks such as the Internet. Because sensors may collect sensitive data and actuators can trigger physical actions, security, privacy, and safety are major challenges. Threat modelling can help identify risks, but established IT-focused methods transfer to the IoT only to a limited extent. In this paper, a new modelling technique specifically for IoT devices called Cyber-Physical Data Flow Diagram (CPDFD) is proposed that also allows modelling of hardware with the aim to support manufacturers in identifying threats and developing countermeasures. The technique was examined through an experimental study and a survey with interviews. The results suggest that numerous other attack scenarios can be found through the modelling technique, improving the identification of threats to IoT devices. |
| 2026-04-24 | [Train in Vain: Functionality-Preserving Poisoning to Prevent Unauthorized Use of Code Datasets](http://arxiv.org/abs/2604.22291v1) | Yuan Xiao, Jiaming Wang et al. | The widespread availability of large-scale code datasets has accelerated the development of code large language models (CodeLLMs), raising concerns about unauthorized dataset usage. Dataset poisoning offers a proactive defense by reducing the utility of such unauthorized training. However, existing poisoning methods often require full dataset poisoning and introduce transformations that break code compilability. In this paper, we introduce FunPoison, a functionality-preserving poisoning approach that injects short, compilable weak-use fragments into executed code paths. FunPoison leverages reusable statement-level templates with automatic repair and conservative safety checking to ensure side-effect freedom, while a type-aware synthesis module suppresses static analysis warnings and enhances stealth. Extensive experiments show that FunPoison achieves effective poisoning by contaminating only 10% of the dataset, while maintaining 100% compilability and functional correctness, and remains robust against various advanced code sanitization techniques. |
| 2026-04-24 | [When Does LLM Self-Correction Help? A Control-Theoretic Markov Diagnostic and Verify-First Intervention](http://arxiv.org/abs/2604.22273v1) | Aofan Liu, Jingxiang Meng | Iterative self-correction is widely used in agentic LLM systems, but when repeated refinement helps versus hurts remains unclear. We frame self-correction as a cybernetic feedback loop in which the same language model serves as both controller and plant, and use a two-state Markov model over {Correct, Incorrect} to operationalize a simple deployment diagnostic: iterate only when ECR/EIR > Acc/(1 - Acc). In this view, EIR functions as a stability margin and prompting functions as lightweight controller design. Across 7 models and 3 datasets (GSM8K, MATH, StrategyQA), we find a sharp near-zero EIR threshold (<= 0.5%) separating beneficial from harmful self-correction. Only o3-mini (+3.4 pp, EIR = 0%), Claude Opus 4.6 (+0.6 pp, EIR ~ 0.2%), and o4-mini (+/-0 pp) remain non-degrading; GPT-5 degrades by -1.8 pp. A verify-first prompt ablation provides causal evidence that this threshold is actionable through prompting alone: on GPT-4o-mini it reduces EIR from 2% to 0% and turns -6.2 pp degradation into +0.2 pp (paired McNemar p < 10^-4), while producing little change on already-sub-threshold models. ASC further illustrates the stopping trade-off: it halts harmful refinement but incurs a 3.8 pp confidence-elicitation cost. Overall, the paper argues that self-correction should be treated not as a default behavior, but as a control decision governed by measurable error dynamics. |
| 2026-04-24 | [Towards Safe Mobility: A Unified Transportation Foundation Model enabled by Open-Ended Vision-Language Dataset](http://arxiv.org/abs/2604.22260v1) | Wenhui Huang, Songyan Zhang et al. | Urban transportation systems face growing safety challenges that require scalable intelligence for emerging smart mobility infrastructures. While recent advances in foundation models and large-scale multimodal datasets have strengthened perception and reasoning in intelligent transportation systems (ITS), existing research remains largely centered on microscopic autonomous driving (AD), with limited attention to city-scale traffic analysis. In particular, open-ended safety-oriented visual question answering (VQA) and corresponding foundation models for reasoning over heterogeneous roadside camera observations remain underexplored. To address this gap, we introduce the Land Transportation Dataset (LTD), a large-scale open-source vision-language dataset for open-ended reasoning in urban traffic environments. LTD contains 11.6K high-quality VQA pairs collected from heterogeneous roadside cameras, spanning diverse road geometries, traffic participants, illumination conditions, and adverse weather. The dataset integrates three complementary tasks: fine-grained multi-object grounding, multi-image camera selection, and multi-image risk analysis, requiring joint reasoning over minimally correlated views to infer hazardous objects, contributing factors, and risky road directions. To ensure annotation fidelity, we combine multi-model vision-language generation with cross-validation and human-in-the-loop refinement. Building upon LTD, we further propose UniVLT, a transportation foundation model trained via curriculum-based knowledge transfer to unify microscopic AD reasoning and macroscopic traffic analysis within a single architecture. Extensive experiments on LTD and multiple AD benchmarks demonstrate that UniVLT achieves SOTA performance on open-ended reasoning tasks across diverse domains, while exposing limitations of existing foundation models in complex multi-view traffic scenarios. |
| 2026-04-24 | [Learning Control Policies to Provably Satisfy Hard Affine Constraints for Black-Box Hybrid Dynamical Systems](http://arxiv.org/abs/2604.22244v1) | Aayushi Shrivastava, Kartik Nagpal et al. | Ensuring safety for black-box hybrid dynamical systems presents significant challenges due to their instantaneous state jumps and unknown explicit nonlinear dynamics. Existing solutions for strict safety constraint satisfaction, like control barrier functions (CBFs) and reachability analysis, rely on direct knowledge of the dynamics. Similarly, safe reinforcement learning (RL) approaches often rely on known system dynamics or merely discourage safety violations through reward shaping. In this work, we want to learn RL policies which provably satisfy affine state constraints in closed loop for black-box hybrid dynamical systems with affine reset maps. Our key insight is forcing the RL policy to be affine and repulsive near the constraint boundaries for the unknown nonlinear dynamics of the system, providing guarantees that the trajectories will not violate the constraint. We further account for constraint violation due to instantaneous state jumps that occur due to impacts or reset maps in the hybrid system by introducing a second repulsive affine region before the reset that prevents post-reset states from violating the constraint. We derive sufficient conditions under which these policies satisfy safety constraints in closed loop. We also compare our approach with state-of-the-art reward shaping and learned-CBF methods on hybrid dynamical systems like the constrained pendulum and paddle juggler environments. In both scenarios, we show that our methodology learns higher quality policies while always satisfying the safety constraints. |
| 2026-04-24 | [Learning-augmented robotic automation for real-world manufacturing](http://arxiv.org/abs/2604.22235v1) | Yunho Kim, Quan Nguyen et al. | Industrial robots are widely used in manufacturing, yet most manipulation still depends on fixed waypoint scripts that are brittle to environmental changes. Learning-based control offers a more adaptive alternative, but it remains unclear whether such methods, still mostly confined to laboratory demonstrations, can sustain hours of reliable operation, deliver consistent quality, and behave safely around people on a live production line. Here we present Learning-Augmented Robotic Automation, a hybrid system that integrates learned task controllers and a neural 3D safety monitor into conventional industrial workflows. We deployed the system on an electric-motor production line to automate deformable cable insertion and soldering under real manufacturing constraints, a step previously performed manually by human workers. With less than 20 min of real-world data per task, the system operated continuously for 5 h 10 min, producing 108 motors without physical fencing and achieving a 99.4% pass rate on product-level quality-control tests. It maintained near-human takt time while reducing variability in solder-joint quality and cycle time. These results establish a practical pathway for extending industrial automation with learning-based methods. |
| 2026-04-24 | [A Co-Evolutionary Theory of Human-AI Coexistence: Mutualism, Governance, and Dynamics in Complex Societies](http://arxiv.org/abs/2604.22227v1) | Somyajit Chakraborty | Classical robot ethics is often framed around obedience, most famously through Asimov's laws. This framing is too narrow for contemporary AI systems, which are increasingly adaptive, generative, embodied, and embedded in physical, psychological, and social worlds. We argue that future human-AI relations should not be understood as master-tool obedience. A better framework is conditional mutualism under governance: a co-evolutionary relationship in which humans and AI systems can develop, specialize, and coordinate, while institutions keep the relationship reciprocal, reversible, psychologically safe, and socially legitimate. We synthesize work from computability, automata theory, statistical machine learning, neural networks, deep learning, transformers, generative and foundation models, world models, embodied AI, alignment, human-robot interaction, ecological mutualism, biological markets, coevolution, and polycentric governance. We then formalize coexistence as a multiplex dynamical system across physical, psychological, and social layers, with reciprocal supply-demand coupling, conflict penalties, developmental freedom, and governance regularization. The framework yields a coexistence model with conditions for existence, uniqueness, and global asymptotic stability of equilibria. It shows that reciprocal complementarity can strengthen stable coexistence, while ungoverned coupling can produce fragility, lock-in, polarization, and domination basins. Human-AI coexistence should therefore be designed as a co-evolutionary governance problem, not as a one-shot obedience problem. This shift supports a scientifically grounded and normatively defensible charter of coexistence: one that permits bounded AI development while preserving human dignity, contestability, collective safety, and fair distribution of gains. |
| 2026-04-24 | [V-STC: A Time-Efficient Multi-Vehicle Coordinated Trajectory Planning Approach](http://arxiv.org/abs/2604.22196v1) | Pengfei Liu, Jialing Zhou et al. | Coordinating the motions of multiple autonomous vehicles (AVs) requires planning frameworks that ensure safety while making efficient use of space and time. This paper presents a new approach, termed variable-time-step spatio-temporal corridor (V-STC), that enhances the temporal efficiency of multi-vehicle coordination. An optimization model is formulated to construct a V-STC for each AV, in which both the spatial configuration of the corridor cubes and their time durations are treated as decision variables. By allowing the corridor's spatial position and time step to vary, the constructed V-STC reduces the overall temporal occupancy of each AV while maintaining collision-free separation in the spatio-temporal domain. Based on the generated V-STC, a dynamically feasible trajectory is then planned independently for each AV. Simulation studies demonstrate that the proposed method achieves safe multi-vehicle coordination and yields more time-efficient motion compared with existing STC approaches. |

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



