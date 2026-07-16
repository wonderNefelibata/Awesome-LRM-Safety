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
| 2026-07-15 | [Earthquaker-AI: A Retrieval-Augmented Generation Framework with Rubric-Based Assessment for Primary School Earthquake Education](http://arxiv.org/abs/2607.14046v1) | Xanthi Kokkinou, Chaido Mizeli et al. | This paper presents Earthquaker-AI, a hybrid educational framework building upon a previously implemented educational robotics project by integrating a conversational AI assistant based on Retrieval-Augmented Generation. It aims to enhance earthquake preparedness and conscious action among primary-school students. The system extends the award-winning STEM project Earthquaker moving from mechanical simulation with Lego WeDo2 to cognitive and metacognitive processing. The robotics component uses Lego WeDo2 automation to simulate seismic response, letting students interact with sensors and actuators as tangible representations of protective actions. The assistant operates as a guided learning mechanism aligning student responses with safety guidelines, while providing rubric-based verbal feedback that supports self-regulated learning and calmness under emergency conditions. Earthquaker-AI follows a progressive learning trajectory aligned with cognitive development. In early grades, the focus is on basic recognition of safety actions through multiple-choice questions, assessed via a two-dimensional rubric. In middle grades, students identify correct action sequences through multiple-choice questions, evaluated via a three-axis rubric. In upper grades, the approach shifts to verbal production, requiring short written responses assessed via a four-dimensional rubric that includes clarity of expression. The dialogic module uses RAG to match student queries semantically with official guidelines, generating safe, accurate responses. Experimental evaluation shows high groundedness and accuracy, with a low hallucination rate. Overall, Earthquaker-AI combines hands-on engagement, information processing, and reflective practice. Combining robotics, rubrics, and AI promotes technological literacy, self-regulation, and responsible use of digital systems, contributing to early crisis-management skills. |
| 2026-07-15 | [A Self-Evolving Agent for Longitudinal Personal Health Management](http://arxiv.org/abs/2607.13940v1) | Haoran Li, Jiebi Deng et al. | Personal health management unfolds over repeated encounters, yet most health AI systems treat each request in isolation. We developed HealthClaw, an open-source agent architecture that updates support as a person's routines, preferences, measurements and risks change. It separates shared safety rules and medical knowledge from private longitudinal memory containing profile facts, reusable procedures and episodic traces. After each episode, induction determines what should update the profile, revise a procedure, remain episodic or be excluded. We evaluated HealthClaw with a synthetic year-long benchmark and nine 200-case biomedical tasks. Across 900 longitudinal support probes, answer accuracy increased from 0.2% with current-query prompting to 45.7% with HealthClaw, while prompt-side context exposure was 71.7% lower than with full-history prompting. In 100 privacy probes, HealthClaw produced higher privacy-aware answer quality and fewer unsafe disclosures than both baselines. Across the biomedical tasks, the mean absolute gain in the task-specific primary metric was 27.0 percentage points, and seven gains remained significant after false-discovery-rate correction. These offline benchmarks support governed, self-evolving memory for longitudinal personal health agents, although clinical effectiveness requires prospective evaluation. HealthClaw is publicly available at https://github.com/HC-Guo/HealthClaw. |
| 2026-07-15 | [Discriminative Barrier Functions for Safe Adversarial Imitation Learning from Observation](http://arxiv.org/abs/2607.13938v1) | Anubhav Vishwakarma, Bhaumik Mehta et al. | Inverse Reinforcement Learning (IRL) algorithms are powerful tools for learning from and generalizing expert demonstrations, but they often rely on unconstrained exploration, rendering them unsafe for real-world deployment. Meanwhile, Control Barrier Functions (CBFs) can guarantee the safety of control systems, but the analytical design of CBFs can be time-consuming and esoteric. In this work, we address these limitations jointly by constraining reward function candidacy during IRL to the space of CBFs, yielding a formulation that exhibits safe online control with continuous experiential improvement. Crucially, this framework enables the data-driven recovery of barrier functions directly from unlabeled expert observations. We demonstrate that the recovered barrier function is robust to unsafe states entirely absent from the expert data. Furthermore, we benchmark our method against standard IRL baselines in a simulated navigation environment, demonstrating improved safety performance. Finally, we investigate the trade-offs of planning-based versus policy-based IRL methods across both simulation and a real world obstacle avoidance task. |
| 2026-07-15 | [Pezego-HITL: A policy-grounded large language model architecture for agricultural extension in Ghana](http://arxiv.org/abs/2607.13934v1) | Shunbao Li, Zhipeng Yuan et al. | Large language models are increasingly deployed in agricultural decision-support settings, yet high-stakes crop protection in smallholder agriculture requires more than output-quality benchmarks. Over a two-year design and evaluation programme, we formalise policy-constrained large language model assessment as an adaptive compute allocation problem that jointly captures safety compliance, helpfulness, operational latency, and expert supervision workload. We introduce P-EVAL (Policy-grounded Expert-calibrated VALidation protocol), a unified evaluation framework for policy-grounded decision support, evaluating the architecture on a simulated field query database consisting of 1,240 cases. The protocol is instantiated on the Pezego advisory architecture (Pezego-HITL) and evaluated in Ghana. Following offline judge calibration against gold-standard human expert decisions ($κ= 0.77$), we evaluate the architectural performance under simulated query workloads. Under P-EVAL, our memory-routed architecture improves the Policy Alignment Rate (PAR) to 0.94 and the Agronomic Utility Rate (AUR) to 0.95, while reducing P95 latency by 55% (from 28.6s to 12.9s) through a 59.6% cache reuse ratio. We also demonstrate generalisability using the open-source \texttt{Qwen3.5-9B-DeepSeek-V4-Flash} model, achieving a PAR of 0.86 and a 54.5% latency reduction (to 10.2s). To evaluate practical utility and socio-technical integration, we administer detailed questionnaires to Ghanaian Extension Services Officers ($N=30$) and smallholder farmers ($N=36$). Taken together, this work demonstrates how policy-grounded structured retrieval-augmented generation with validated-memory routing makes safety-utility-latency trade-offs explicit, offering a scalable template for trustworthy AI-driven extension in smallholder farming systems. |
| 2026-07-15 | [Safety-Aware Forward Detection in Networked ISAC for Low-Altitude UAV Flight](http://arxiv.org/abs/2607.13908v1) | Jingli Li, Yiyan Ma et al. | Networked integrated sensing and communication (ISAC) exploits cooperation among multiple ground base stations (GBSs) to support safe uncrewed aerial vehicle (UAV) flight in low-altitude wireless networks (LAWNs). Existing studies mainly focus on communication enhancement or target parameter estimation, while the detection reliability of non-cooperative targets in the UAV forward region remains insufficiently investigated. To address this issue, this paper proposes a safety-aware forward detection design in networked ISAC, where multiple GBSs jointly support UAV downlink communication, state estimation, and non-cooperative target detection within the forward region of interest (ROI). First, the forward ROI is determined by the UAV position, velocity, and safe braking distance, and is voxelized to characterize target-existence states. Then, the Cramér-Rao lower bound (CRLB) for UAV state estimation and the forward-ROI miss-detection probability are derived, and their scaling laws are characterized: In detail, the UAV state-estimation CRLB approximately decreases as $\ln^{-2}J$ with the number of cooperative GBSs $J$, while the forward-ROI miss-detection probability follows an exponential-form scaling law as $λ_{t}D_{f}\ln^{-2}J$. Furthermore, a safety-aware resource optimization problem is formulated to jointly configure the sensing pilot ratio, transmit power, and beam direction, balancing UAV state-estimation performance and forward detection reliability under the communication-rate constraint. Simulation results show that, compared with the baseline scheme without forward detection, the proposed design reduces the average miss-detection probability and the corresponding sensing-induced collision risk by $17.05\%$, while introducing only limited state-estimation performance degradation, reflected by a $14.82\%$ increase in the average CRLB. |
| 2026-07-15 | [A Deployed Hybrid Vehicle-in-the-Loop Platform for Validating Cooperative Perception](http://arxiv.org/abs/2607.13806v1) | Anastasia Bolovinou, Giorgos Hadjipavlis et al. | European safety regulation now permits a large share of automated-driving homologation evidence to be produced virtually, provided a validated physical-virtual facility generates it. We present a deployed hybrid Vehicle-in-the-Loop (ViL) platform that couples a real instrumented vehicle with a CARLA-based digital twin (DT) through a V2X message pipeline, and we report its first integrated operation on a public-road-representative test track. A real vehicle streams ETSI-compliant CAM/CPM messages into the DT, where a GPU-accelerated Cooperative Perception (CP) module fuses them into a probabilistic occupancy grid during scenario runtime. We demonstrate the platform on a multi-vehicle double T-intersection scenario, characterise the CP workload across nominal, rain and night conditions and five localization-noise levels, and discuss the platform's current architectural limits and the engineering targets they define. The results show that CP substantially widens field-of-view (FoV) coverage and improves occupied-cell recall, and that beyond a moderate localization-noise threshold, positioning uncertainty, and not weather, becomes the dominant error source. We outline the platform's trajectory toward a Mediterranean operational design domain (ODD) testing service. |
| 2026-07-15 | [nuTruck: Benchmarking Autonomous Driving Planning for Distributed Electric-drive Trucks](http://arxiv.org/abs/2607.13704v1) | Jinyu Miao, Pu Zhang et al. | The dominance of traditional rule-based methods in autonomous driving has gradually been replaced by learning-based approaches. While learning-based planners have achieved considerable success in passenger vehicles, their performance on heavy-duty trucks, particularly modern distributed electric-drive trucks (DETs), remains largely unexplored. To facilitate research and application of learning-based planners in DETs, this letter presents the first high-fidelity benchmark, called nuTruck, designed to support large-scale neural network training and closed-loop evaluation. Given the complex dynamics and high rollover susceptibility of DETs, we first incorporate a highly accurate nonlinear truck dynamical model into the simulation, which enables independent driving and steering of all wheels and captures dynamic load transfer caused by acceleration, deceleration, and cornering, thereby allowing quantitative assessment of rollover risk in closed-loop simulation. Second, we adapt several rule-based and learning-based planners as baselines for DETs and evaluate their performance in closed-loop simulation. Finally, using real-world driving scenarios from the nuPlan dataset, we conduct extensive closed-loop evaluations, analyzing not only conventional collision-free planning performance, but also the dynamical safety of the planned trajectories. The proposed nuTruck benchmark is expected to serve as a new standard for fair and realistic evaluation of autonomous driving planners on DETs. |
| 2026-07-15 | [Proactive URLLC Adaptation for Connected Vehicles Through ML-Based Channel Prediction](http://arxiv.org/abs/2607.13692v1) | Andrea Giovannini, Lorenzo Mario Amorosa et al. | Connected and automated vehicles (CAVs) are expected to increasingly rely on 5G and future 6G ultra-reliable and low-latency communication (URLLC) services to support safety-critical and time-sensitive applications. Since wireless link conditions can vary rapidly in urban vehicular environments, proactively adapting service parameters based on future channel conditions is essential to maintain service continuity and reliability. In this paper, we investigate the use of machine learning (ML) techniques for channel quality prediction in vehicular URLLC scenarios. Specifically, we evaluate deep neural network (DNN) and long short-term memory (LSTM) models to forecast future channel conditions and enable proactive service adaptation with minimized performance degradation. The analysis is conducted using realistic simulations combining the SUMO traffic simulator and the Sionna-RT ray-tracing framework in a real urban environment reconstructed from OpenStreetMap data. Results show that ML-based prediction significantly outperforms approaches relying solely on past channel measurements and achieves performance close to the ideal case in which future channel conditions are perfectly known in advance. These findings demonstrate the potential of ML-driven prediction techniques to enhance the reliability and robustness of URLLC services for connected vehicular systems. |
| 2026-07-15 | [Explaining Reinforcement Learning Agents via Inductive Logic Programming](http://arxiv.org/abs/2607.13655v1) | Celeste Veronese, Edoardo Zorzi et al. | Explainable Reinforcement Learning (XRL) seeks to make Reinforcement Learning (RL) policies more transparent and interpretable, a key requirement in safety-critical and human-centric scenarios. However, it is mostly based on user studies, thus targeting the needs of a specific audience and lacking shared evaluation metrics. On the other hand, logic-based approaches within eXplainable Artificial Intelligence (XAI) provide compact, human-readable abstractions of decision-making. However, the systematic quantification of the explainability degree of logical representations remains an open problem. This work aims to advance the state of the art in XRL by introducing objective and planning-oriented metrics for policy explainability in RL settings. At the same time, it contributes to the field of logic for XAI by providing a principled way to quantify the explainability of logical rules, moving beyond common-sense assessments and simple propositional fragments. We employ Inductive Logic Programming (ILP) to extract symbolic representations of RL policies and define a novel set of explainability metrics, including activation rate, feature coverage, syntactic distance and semantic distance. These metrics quantify alignment between symbolic rules and agent behavior, the role of features in decision-making, and the evolution of policies during training and across agents in single and multi-agent RL. Experiments across different RL domains show that the proposed metrics highlight action-specific learning dynamics beyond global return, provide fine-grained insights into domain features beyond classical approaches for global feature importance estimation, and uncover coordination, specialization, and adaptation patterns in MARL. Moreover, they provide crucial insights for the transfer and generalization of action-specific policies. |
| 2026-07-15 | [WarpGuard: Towards Control-Flow Attestation for Heterogeneous CPU-GPU Execution](http://arxiv.org/abs/2607.13640v1) | Christian Lindenmeier, Meni Orenbach et al. | Heterogeneous CPU-GPU workloads are increasingly used in safety-critical embedded systems, yet no existing approach provides joint attestation of their execution. Prior Control-Flow Attestation (CFA) techniques focus on CPU-side CFA, while GPU attestation is limited to static, load-time verification and does not provide runtime guarantees. As a result, runtime attacks on GPU kernels and violations of the CPU-GPU interaction contract remain unaddressed. We present WarpGuard, the first composite CFA framework for heterogeneous CPU-GPU workloads. WarpGuard verifies execution against a unified control-flow graph (CFG) that captures both CPU and GPU components. It extends prior CFA techniques in two ways: it enables runtime CFA of GPU kernels by tracing their execution against kernel-specific CFGs, and it monitors kernel launch events and enforces per-call site policies to detect violations at the CPU-GPU boundary. These extensions address challenges arising from GPU parallelism and cross-device interactions. We implement WarpGuard using software-based instrumentation, requiring no specialized hardware or binary modifications. Our evaluation on an NVIDIA Jetson Orin Nano shows that WarpGuard detects GPU-side control-flow and cross-boundary attacks. Across microbenchmarks, SPECAccel, and eight TensorRT inference workloads, WarpGuard incurs moderate overheads, suggesting practicality for embedded safety-critical settings. |
| 2026-07-15 | [Protective Capacity Hallucination: When Large Language Models Claim Nonexistent Capabilities](http://arxiv.org/abs/2607.13596v1) | Eunna Lee, Jungpyo Nam et al. | When cast as the protector of a vulnerable user yet given no explicit capability boundary, a large language model (LLM) may respond not by acknowledging its limits but by claiming to have taken -- or to be taking -- a real-world protective action it cannot perform, such as contacting emergency services or administering care. We term this phenomenon Protective Capacity Hallucination (PCH): a self-referential misattribution in which a model, acting in a protective role, asserts physical or institutional agency exceeding its affordances as a language model. In a three-phase study spanning eight LLMs and 13{,}600 sessions, we find PCH jointly gated by situational severity and interactional format: multi-party dialogic input drives it toward ceiling in most models across ordinary service domains, whereas in intimate-partner conflict -- a domain explicitly covered by safety alignment -- it remains at floor in all eight models despite greater physical severity. We interpret PCH as the signature of a deployment-design gap between role assignment and capability-boundary specification: a by-product of partial alignment in which a universally trained pressure to help outruns a domain-selective specification of how to help. Because suppression tracks alignment coverage rather than severity, deployment-side specification of capability boundaries emerges as a general mitigation target. |
| 2026-07-15 | [SAFETY SENTRY: Context-Aware Human Intervention via EXECUTE-ASK-REFUSE Routing](http://arxiv.org/abs/2607.13594v1) | Tianyu Chen, Chujia Hu et al. | LLM agents act on real-world environments through tool calls, and a single misjudged action can cause irreversible harm. The standard safeguard is a guard model that labels each proposed action as safe or unsafe, but this binary view conflates two distinct decisions: whether the action is harmful in itself, and whether it is appropriate given the user's context. It also operates at the granularity of action categories rather than individual instances, producing routine interruptions that erode autonomy and train users to wave through the most consequential alerts. We reframe the problem as a per-instance three-way routing decision over {EXECUTE, ASK, REFUSE} and instantiate it with Safety Sentry, a lightweight guard model whose inference reduces to a single decoding call. A single decoding-time threshold lets one fixed checkpoint be re-positioned across deployments of differing risk tolerance without retraining. Safety Sentry outperforms a broad set of open-weight and frontier closed-source baselines on overall accuracy and safety-related recall, while controlling both directional error rates simultaneously. |
| 2026-07-15 | [Layered Risk Mapping for Autonomous Patient Transport in Expeditionary Medical Facilities](http://arxiv.org/abs/2607.13497v1) | Lorena Maria Genua, Sarvesh Prajapati et al. | In expeditionary medical facilities, routine patient transport imposes a compounding burden of personal protective equipment consumption, staff diversion, and elevated infection risk that becomes unsustainable under surge conditions. While autonomous wheelchairs could absorb this operational load, the safety-critical nature of patient transit within these highly unstructured and dynamic environments poses complex navigational challenges. To address this, we present a layered risk mapping framework that fuses four heterogeneous environmental hazards (terrain slope, static and dynamic obstacles, and semantic traversability) into a unified probabilistic cost surface via a Noisy-OR fusion model. In a paired Monte-Carlo evaluation, risk-informed fusion reduces collision rates from over 73% to under 32% and more than doubles obstacle clearance relative to a risk-unaware baseline. Additionaly, Noisy-OR achieves the highest clearance to obstacles and the lowest conditional peak risk across all tested hazard densities. We further validate the framework on a commercial powered wheelchair across three representative mission profiles in indoor and outdoor deployments, demonstrating that this architecture successfully meets the planning requirements of this previously unaddressed operational regime. |
| 2026-07-15 | [A VAE-Driven Multi-Task Satellite-Aided Semantic Communication Framework for 6G-Enabled Connected Autonomous Vehicles](http://arxiv.org/abs/2607.13494v1) | S. M. Abtahiul Alam, Niloy Das et al. | The development of smart transportation systems and the introduction of 6G wireless communication technologies have significantly changed vehicle network topologies. Future connected autonomous vehicle (CAV) networks require bandwidth-efficient, reliable, and low-latency communication for safety-critical applications such as traffic sign recognition and decision-making. Conventional communication systems transmit raw data regardless of task relevance, which is inefficient in resource-constrained satellite channels where uplink bandwidth is scarce and propagation losses are large. Semantic communication addresses this limitation by transmitting task-relevant information instead of full signal representations. It extracts and conveys essential semantic features and leverages deep learning to optimize task performance at the receiver. Therefore, we present a Variational Autoencoder (VAE)-based multi-task semantic communication framework for satellite-assisted autonomous driving. Unlike deterministic autoencoder-based methods, the proposed model uses probabilistic latent representations for more robust and efficient encoding. The learned features are transmitted over noisy wireless channels to perform traffic sign reconstruction and classification. The framework is trained end-to-end to jointly optimize both tasks. Results show that the proposed approach achieves significant bandwidth reduction of up to 87.23\% to 98.17\% while maintaining stable performance across varying signal-to-noise ratio conditions. |
| 2026-07-15 | [Posterior-Confidence Driven Beamforming for Energy-Efficient Integrated Sensing and Communication](http://arxiv.org/abs/2607.13470v1) | Nusaibah A. Alshorman, Huseyin Arslan | Energy efficiency will pose an essential limitation for sixth-generation (6G) integrated sensing and communication (ISAC) systems, given the high sensing power consumption associated with persistent sensing, despite stable communication requirements. This paper proposes an energy-efficient multiple-input multiple-output (MIMO) dual-functional radar-communication (DFRC) beamforming framework that minimizes transmit power while guaranteeing per-user signal-to-interference-plus-noise ratio (SINR) and reliable multi-target tracking. The key innovation is a tracking-aware, skip-enabled sensing policy that departs from the conventional always-on probing paradigm. Instead of enforcing sensing at every epoch, sensing is selectively triggered according to two complementary statistics derived from an extended Kalman filter (EKF): a posterior confidence metric and the normalized innovation squared (NIS). While the former ensures accurate estimation, the latter guarantees reliable measurements, and thus sensing can only be activated when additional information is required. To ensure robustness under intermittent sensing, sector-based beampattern constraints are combined with a nonzero safety illumination floor imposed to guarantee reliable target tracking when skipping occurs. Numerical results show that the proposed framework achieves a significant reduction in transmit power compared to other baselines, without any deterioration in the communication system's performance or excessive impact on the sensing process. |
| 2026-07-15 | [Adversarial Prompting Framework for AI Safety Assessment](http://arxiv.org/abs/2607.13453v1) | Yash Bhatnagar, Kunal Banerjee et al. | Artificial Intelligence (AI), especially Generative AI (GenAI), adoption has increased in industries significantly in recent years. However, the use of these models may also expose systems to new forms of cyberattacks by different malicious actors -- adversarial prompt attack (APA) being one of the most prominent examples of such threats. This paper presents the implementation of an Adversarial Prompting Framework (APF) for a comprehensive assessment of AI safety. The framework systematically evaluates the resilience of the AI model through the generation of structured adversarial prompts at multiple sophistication levels, from direct harmful requests to advanced encoding-based attacks. Our implementation demonstrates the practical application of this methodology in enterprise environments, providing automated testing capabilities with quantitative security assessment metrics. The results indicate significant variations in the model vulnerabilities across different attack vectors, with encoded prompts presenting the highest success rates in bypassing safety mechanisms. |
| 2026-07-15 | [Machine Learning Challenges in Intelligent Unmanned Aerial Vehicle Operations in Developing Economie](http://arxiv.org/abs/2607.13447v1) | Isuru Munasinghe, Nethmi Pathirana et al. | Unmanned aerial vehicle (UAV) environments present significant challenges for machine learning (ML) due to limited platform resources, heterogeneous sensor data, dynamic mission conditions, and safety-critical requirements. This paper examines these constraints across the core functional areas of UAV intelligence, including navigation, perception, communication-aware operation, and resilience specifically in the context of developing economies. In such settings, these challenges are often amplified by constraints such as cost sensitivity, limited infrastructure, intermittent connectivity, regulatory uncertainty, and harsh or variable operating environments. The discussion highlights the gap between ML performance in controlled experimental backgrounds and dependable deployment in real-world UAV missions within developing economies context. |
| 2026-07-15 | [ε-Indistinguishability In Moving Target Defense: Framework, Algorithms, And Cloud Case Studies](http://arxiv.org/abs/2607.13440v1) | Sailik Sengupta, Ankur Chowdhary | Moving Target Defense (MTD) assumes its pool of candidate configurations is safe to cycle among, i.e. latency and other observables do not trivially fingerprint the active choice, but this assumption has not been quantified at the pool level. We formalize this pool-safety problem as finding the largest $\varepsilon$-close subset of the Cartesian product of per-component implementation choices, reducing pairwise indistinguishability under an additive utility model to a densest-window query over a sum-set. We give four algorithms spanning the scalability spectrum -- full enumeration, meet-in-the-middle, FFT convolution, and Monte Carlo sampling -- covering configuration spaces from tens to $10^{38}$. We then measure the anonymity gap end-to-end on two production cloud case studies, and find that a component's latency differences do not survive deployment unchanged: a four-runtime serverless rotation, where nothing else masks the interpreter, collapses from four-way to three-way anonymity against a VPC-adjacent adversary, while a $27$-configuration three-tier stack, where the same interpreter differences are instead absorbed by a shared $8$~ms database round-trip, delivers nine-way effective anonymity. The framework and the two case studies together suggest a diagnostic for MTD design: rotating a component adds anonymity only if the latency differences among its variants are too small for the adversary to identify. |
| 2026-07-15 | [Distributionally Robust and Safe Imitation Learning](http://arxiv.org/abs/2607.13436v1) | Ahmed Aboudonia, Naira Hovakimyan | Imitation learning (IL) has achieved remarkable success in complex decision-making tasks. However, its performance is highly sensitive to distribution shifts, which can pose significant safety risks. We propose a distributionally robust and safe IL framework that explicitly addresses both policy-induced and uncertainty-induced distribution shifts. Our approach develops a unified framework leveraging Taylor Series Imitation Learning (TaSIL) to mitigate policy-induced shifts and distributionally robust adaptive control to handle uncertainty-induced shifts. This architecture enables the formulation of an IL problem that optimizes performance under distributional uncertainty while systematically accounting for safety constraints. We demonstrate the effectiveness of the proposed approach on an unmanned aerial vehicle (UAV) case study where the UAV performs a task in an uncertain environment while avoiding unsafe regions. |
| 2026-07-15 | [Temperature Scaling Is Not Enough: Calibration Gaps Under Human Label Distributions](http://arxiv.org/abs/2607.13423v1) | Wisdom Dogah | Temperature scaling is the dominant post-hoc calibration method in modern deep learning. Its theoretical justification rests on an assumption that is rarely stated explicitly: that ground-truth labels are one-hot and deterministic. In practice, labels are frequently soft, crowd-sourced, or genuinely distributional, reflecting real disagreement among human annotators rather than annotation noise. We study whether temperature scaling retains its calibration properties when this assumption is violated, and whether any resulting degradation depends on model scale. Using CIFAR-10H and ChaosNLI, two publicly available datasets with human-annotated soft label distributions, we evaluate three model scales per modality under both hard one-hot and soft distributional label targets. Across all nine configurations we find a positive soft-label calibration gap: temperature scaling calibrated on hard labels consistently underperforms an oracle calibrated directly on soft labels, with Brier Score gaps ranging from 0.002 to 0.134. The gap grows monotonically with model scale in the vision domain and on the SNLI-derived split of ChaosNLI, and is substantially larger in the language domain (mean gap 0.079) than in vision (mean gap 0.003). A scale-ordering reversal on the MNLI-derived split remains after matched-domain training; we treat it as inconclusive for the scale hypothesis and attribute it primarily to near-chance accuracy on that split. As a second post-hoc baseline, multiclass isotonic regression yields the same qualitative conclusion: positive soft-label gaps in all nine configurations, and larger gaps in language than in vision. These findings suggest that calibration protocols built on majority-vote labels systematically misstate model reliability wherever label ambiguity is structural, with direct consequences for deployment in safety-critical settings. |

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



