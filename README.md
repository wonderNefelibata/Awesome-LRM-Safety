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
| 2026-03-06 | [BEVLM: Distilling Semantic Knowledge from LLMs into Bird's-Eye View Representations](http://arxiv.org/abs/2603.06576v1) | Thomas Monninger, Shaoyuan Xie et al. | The integration of Large Language Models (LLMs) into autonomous driving has attracted growing interest for their strong reasoning and semantic understanding abilities, which are essential for handling complex decision-making and long-tail scenarios. However, existing methods typically feed LLMs with tokens from multi-view and multi-frame images independently, leading to redundant computation and limited spatial consistency. This separation in visual processing hinders accurate 3D spatial reasoning and fails to maintain geometric coherence across views. On the other hand, Bird's-Eye View (BEV) representations learned from geometrically annotated tasks (e.g., object detection) provide spatial structure but lack the semantic richness of foundation vision encoders. To bridge this gap, we propose BEVLM, a framework that connects a spatially consistent and semantically distilled BEV representation with LLMs. Through extensive experiments, we show that BEVLM enables LLMs to reason more effectively in cross-view driving scenes, improving accuracy by 46%, by leveraging BEV features as unified inputs. Furthermore, by distilling semantic knowledge from LLMs into BEV representations, BEVLM significantly improves closed-loop end-to-end driving performance by 29% in safety-critical scenarios. |
| 2026-03-06 | [SUREON: A Benchmark and Vision-Language-Model for Surgical Reasoning](http://arxiv.org/abs/2603.06570v1) | Alejandra Perez, Anita Rau et al. | Surgeons don't just see -- they interpret. When an expert observes a surgical scene, they understand not only what instrument is being used, but why it was chosen, what risk it poses, and what comes next. Current surgical AI cannot answer such questions, largely because training data that explicitly encodes surgical reasoning is immensely difficult to annotate at scale. Yet surgical video lectures already contain exactly this -- explanations of intent, rationale, and anticipation, narrated by experts for the purpose of teaching. Though inherently noisy and unstructured, these narrations encode the reasoning that surgical AI currently lacks. We introduce SUREON, a large-scale video QA dataset that systematically harvests this training signal from surgical academic videos. SUREON defines 12 question categories covering safety assessment, decision rationale, and forecasting, and uses a multi-agent pipeline to extract and structure supervision at scale. Across 134.7K clips and 170 procedure types, SUREON yields 206.8k QA pairs and an expert-validated benchmark of 354 examples. To evaluate the extent to which this supervision translates to surgical reasoning ability, we introduce two models: SureonVLM, a vision-language model adapted through supervised fine-tuning, and SureonVLM-R1, a reasoning model trained with Group Relative Policy Optimization. Both models can answer complex questions about surgery and substantially outperform larger general-domain models, exceeding 84% accuracy on the SUREON benchmark while outperforming general-domain models on standard surgical perception tasks. Qualitative analysis of SureonVLM-R1 reveals explicit reasoning behavior, such as inferring operative intent from visual context. |
| 2026-03-06 | [Control Barrier Corridors: From Safety Functions to Safe Sets](http://arxiv.org/abs/2603.06494v1) | Ömür Arslan, Nikolay Atanasov | Safe autonomy is a critical requirement and a key enabler for robots to operate safely in unstructured complex environments. Control barrier functions and safe motion corridors are two widely used but technically distinct safety methods, functional and geometric, respectively, for safe motion planning and control. Control barrier functions are applied to the safety filtering of control inputs to limit the decay rate of system safety, whereas safe motion corridors are geometrically constructed to define a local safe zone around the system state for use in motion optimization and reference-governor design. This paper introduces a new notion of control barrier corridors, which unifies these two approaches by converting control barrier functions into local safe goal regions for reference goal selection in feedback control systems. We show, with examples on fully actuated systems, kinematic unicycles, and linear output regulation systems, that individual state safety can be extended locally over control barrier corridors for convex barrier functions, provided the control convergence rate matches the barrier decay rate, highlighting a trade-off between safety and reactiveness. Such safe control barrier corridors enable safely reachable persistent goal selection over continuously changing barrier corridors during system motion, which we demonstrate for verifiably safe and persistent path following in autonomous exploration of unknown environments. |
| 2026-03-06 | [What if? Emulative Simulation with World Models for Situated Reasoning](http://arxiv.org/abs/2603.06445v1) | Ruiping Liu, Yufan Chen et al. | Situated reasoning often relies on active exploration, yet in many real-world scenarios such exploration is infeasible due to physical constraints of robots or safety concerns of visually impaired users. Given only a limited observation, can an agent mentally simulate a future trajectory toward a target situation and answer spatial what-if questions? We introduce WanderDream, the first large-scale dataset designed for the emulative simulation of mental exploration, enabling models to reason without active exploration. WanderDream-Gen comprises 15.8K panoramic videos across 1,088 real scenes from HM3D, ScanNet++, and real-world captures, depicting imagined trajectories from current viewpoints to target situations. WanderDream-QA contains 158K question-answer pairs, covering starting states, paths, and end states along each trajectory to comprehensively evaluate exploration-based reasoning. Extensive experiments with world models and MLLMs demonstrate (1) that mental exploration is essential for situated reasoning, (2) that world models achieve compelling performance on WanderDream-Gen, (3) that imagination substantially facilitates reasoning on WanderDream-QA, and (4) that WanderDream data exhibit remarkable transferability to real-world scenarios. The source code and all data will be released. |
| 2026-03-06 | [Safe Consensus of Cooperative Manipulation with Hierarchical Event-Triggered Control Barrier Functions](http://arxiv.org/abs/2603.06356v1) | Simiao Zhuang, Bingkun Huang et al. | Cooperative transport and manipulation of heavy or bulky payloads by multiple manipulators requires coordinated formation tracking, while simultaneously enforcing strict safety constraints in varying environments with limited communication and real-time computation budgets. This paper presents a distributed control framework that achieves consensus coordination with safety guarantees via hierarchical event-triggered control barrier functions (CBFs). We first develop a consensus-based protocol that relies solely on local neighbor information to enforce both translational and rotational consistency in task space. Building on this coordination layer, we propose a three-level hierarchical event-triggered safety architecture with CBFs, which is integrated with a risk-aware leader selection and smooth switching strategy to reduce online computation. The proposed approach is validated through real-world hardware experiments using two Franka manipulators operating with static obstacles, as well as comprehensive simulations demonstrating scalable multi-arm cooperation with dynamic obstacles. Results demonstrate higher precision cooperation under strict safety constraints, achieving substantially reduced computational cost and communication frequency compared to baseline methods. |
| 2026-03-06 | [SAHOO: Safeguarded Alignment for High-Order Optimization Objectives in Recursive Self-Improvement](http://arxiv.org/abs/2603.06333v1) | Subramanyam Sahoo, Aman Chadha et al. | Recursive self-improvement is moving from theory to practice: modern systems can critique, revise, and evaluate their own outputs, yet iterative self-modification risks subtle alignment drift. We introduce SAHOO, a practical framework to monitor and control drift through three safeguards: (i) the Goal Drift Index (GDI), a learned multi-signal detector combining semantic, lexical, structural, and distributional measures; (ii) constraint preservation checks that enforce safety-critical invariants such as syntactic correctness and non-hallucination; and (iii) regression-risk quantification to flag improvement cycles that undo prior gains. Across 189 tasks in code generation, mathematical reasoning, and truthfulness, SAHOO produces substantial quality gains, including 18.3 percent improvement in code tasks and 16.8 percent in reasoning, while preserving constraints in two domains and maintaining low violations in truthfulness. Thresholds are calibrated on a small validation set of 18 tasks across three cycles. We further map the capability-alignment frontier, showing efficient early improvement cycles but rising alignment costs later and exposing domain-specific tensions such as fluency versus factuality. SAHOO therefore makes alignment preservation during recursive self-improvement measurable, deployable, and systematically validated at scale. |
| 2026-03-06 | [Polarized Direct Cross-Attention Message Passing in GNNs for Machinery Fault Diagnosis](http://arxiv.org/abs/2603.06303v1) | Zongyu Shi, Laibin Zhang et al. | The reliability of safety-critical industrial systems hinges on accurate and robust fault diagnosis in rotating machinery. Conventional graph neural networks (GNNs) for machinery fault diagnosis face limitations in modeling complex dynamic interactions due to their reliance on predefined static graph structures and homogeneous aggregation schemes. To overcome these challenges, this paper introduces polarized direct cross-attention (PolaDCA), a novel relational learning framework that enables adaptive message passing through data-driven graph construction. Our approach builds upon a direct cross-attention (DCA) mechanism that dynamically infers attention weights from three semantically distinct node features (such as individual characteristics, neighborhood consensus, and neighborhood diversity) without requiring fixed adjacency matrices. Theoretical analysis establishes PolaDCA's superior noise robustness over conventional GNNs. Extensive experiments on industrial datasets (i.e., XJTUSuprgear, CWRUBearing and Three-Phase Flow Facility datasets) demonstrate state-of-the-art diagnostic accuracy and enhanced generalization under varying noise conditions, outperforming seven competitive baseline methods. The proposed framework provides an effective solution for safety-critical industrial applications. |
| 2026-03-06 | [An Integrated Failure and Threat Mode and Effect Analysis (FTMEA) Framework with Quantified Cross-Domain Correlation Factors for Automotive Semiconductors](http://arxiv.org/abs/2603.06299v1) | Antonino Armato, Marzana Khatun et al. | The automotive industry faces increasing challenges in ensuring both functional safety (FuSa) and cybersecurity for complex semiconductor devices. Traditional Failure Mode and Effects Analysis (FMEA) primarily addresses safety-related failure modes, often overlooking synergistic vulnerabilities and shared consequences with cybersecurity threats. This paper introduces an Integrated Failure and Threat Mode and Effect Analysis (FTMEA) framework that systematically co-analyzes FuSa and cybersecurity. A cornerstone of this framework is the introduction of rigorously defined Cross-Domain Correlation Factors (CDCFs), which quantify the interdependencies and mutual influences between safety-related failures and cybersecurity threats. These factors are derived from a combination of structured expert knowledge, static structural analysis metrics (e.g., Controllability/Observability), and validated against empirical data from fault/attack injection campaigns. We propose a modified Risk Priority Number (RPN) calculation that systematically integrates these correlation factors, enabling a more accurate and transparent prioritization of risks that span both domains. A detailed case study involving an automotive ASIC configuration register proves the practical application of the FTMEA. We present explicit mapping tables, quantitative CDCF values, and a comparative analysis against a baseline FMEA/TARA (Threat Analysis and Risk Assessment), illustrating how the integrated approach uncovers previously masked cross-domain risks, improves mitigation strategy effectiveness, and provides a clear quantitative justification for the derived correlation values. This framework offers a unified, traceable, methodology for risk assessment in critical automotive systems, thereby overcoming the limitations of conventional analyses and promoting optimized, cross-disciplinary development. |
| 2026-03-06 | [Can we Trust Unreliable Voxels? Exploring 3D Semantic Occupancy Prediction under Label Noise](http://arxiv.org/abs/2603.06279v1) | Wenxin Li, Kunyu Peng et al. | 3D semantic occupancy prediction is a cornerstone of robotic perception, yet real-world voxel annotations are inherently corrupted by structural artifacts and dynamic trailing effects. This raises a critical but underexplored question: can autonomous systems safely rely on such unreliable occupancy supervision? To systematically investigate this issue, we establish OccNL, the first benchmark dedicated to 3D occupancy under occupancy-asymmetric and dynamic trailing noise. Our analysis reveals a fundamental domain gap: state-of-the-art 2D label noise learning strategies collapse catastrophically in sparse 3D voxel spaces, exposing a critical vulnerability in existing paradigms. To address this challenge, we propose DPR-Occ, a principled label noise-robust framework that constructs reliable supervision through dual-source partial label reasoning. By synergizing temporal model memory with representation-level structural affinity, DPR-Occ dynamically expands and prunes candidate label sets to preserve true semantics while suppressing noise propagation. Extensive experiments on SemanticKITTI demonstrate that DPR-Occ prevents geometric and semantic collapse under extreme corruption. Notably, even at 90% label noise, our method achieves significant performance gains (up to 2.57% mIoU and 13.91% IoU) over existing label noise learning baselines adapted to the 3D occupancy prediction task. By bridging label noise learning and 3D perception, OccNL and DPR-Occ provide a reliable foundation for safety-critical robotic perception in dynamic environments. The benchmark and source code will be made publicly available at https://github.com/mylwx/OccNL. |
| 2026-03-06 | [CRIMSON: A Clinically-Grounded LLM-Based Metric for Generative Radiology Report Evaluation](http://arxiv.org/abs/2603.06183v1) | Mohammed Baharoon, Thibault Heintz et al. | We introduce CRIMSON, a clinically grounded evaluation framework for chest X-ray report generation that assesses reports based on diagnostic correctness, contextual relevance, and patient safety. Unlike prior metrics, CRIMSON incorporates full clinical context, including patient age, indication, and guideline-based decision rules, and prevents normal or clinically insignificant findings from exerting disproportionate influence on the overall score. The framework categorizes errors into a comprehensive taxonomy covering false findings, missing findings, and eight attribute-level errors (e.g., location, severity, measurement, and diagnostic overinterpretation). Each finding is assigned a clinical significance level (urgent, actionable non-urgent, non-actionable, or expected/benign), based on a guideline developed in collaboration with attending cardiothoracic radiologists, enabling severity-aware weighting that prioritizes clinically consequential mistakes over benign discrepancies. CRIMSON is validated through strong alignment with clinically significant error counts annotated by six board-certified radiologists in ReXVal (Kendalls tau = 0.61-0.71; Pearsons r = 0.71-0.84), and through two additional benchmarks that we introduce. In RadJudge, a targeted suite of clinically challenging pass-fail scenarios, CRIMSON shows consistent agreement with expert judgment. In RadPref, a larger radiologist preference benchmark of over 100 pairwise cases with structured error categorization, severity modeling, and 1-5 overall quality ratings from three cardiothoracic radiologists, CRIMSON achieves the strongest alignment with radiologist preferences. We release the metric, the evaluation benchmarks, RadJudge and RadPref, and a fine-tuned MedGemma model to enable reproducible evaluation of report generation, all available at https://github.com/rajpurkarlab/CRIMSON. |
| 2026-03-06 | [A Hazard-Informed Data Pipeline for Robotics Physical Safety](http://arxiv.org/abs/2603.06130v1) | Alexei Odinokov, Rostislav Yavorskiy | This report presents a structured Robotics Physical Safety Framework based on explicit asset declaration, systematic vulnerability enumeration, and hazard-driven synthetic data generation. The approach bridges classical risk engineering with modern machine learning pipelines, enabling safety envelope learning grounded in a formalized hazard ontology. The key contribution of this framework is the alignment between classical safety engineering, digital twin simulation, synthetic data generation, and machine learning model training. |
| 2026-03-06 | [Evaluating Austrian A-Level German Essays with Large Language Models for Automated Essay Scoring](http://arxiv.org/abs/2603.06066v1) | Jonas Kubesch, Lena Huber et al. | Automated Essay Scoring (AES) has been explored for decades with the goal to support teachers by reducing grading workload and mitigating subjective biases. While early systems relied on handcrafted features and statistical models, recent advances in Large Language Models (LLMs) have made it possible to evaluate student writing with unprecedented flexibility. This paper investigates the application of state-of-the-art open-weight LLMs for the grading of Austrian A-level German texts, with a particular focus on rubric-based evaluation. A dataset of 101 anonymised student exams across three text types was processed and evaluated. Four LLMs, DeepSeek-R1 32b, Qwen3 30b, Mixtral 8x7b and LLama3.3 70b, were evaluated with different contexts and prompting strategies. The LLMs were able to reach a maximum of 40.6% agreement with the human rater in the rubric-provided sub-dimensions, and only 32.8% of final grades matched the ones given by a human expert. The results indicate that even though smaller models are able to use standardised rubrics for German essay grading, they are not accurate enough to be used in a real-world grading environment. |
| 2026-03-06 | [Moving Through Clutter: Scaling Data Collection and Benchmarking for 3D Scene-Aware Humanoid Locomotion via Virtual Reality](http://arxiv.org/abs/2603.05993v1) | Beichen Wang, Yuanjie Lu et al. | Recent advances in humanoid locomotion have enabled dynamic behaviors such as dancing, martial arts, and parkour, yet these capabilities are predominantly demonstrated in open, flat, and obstacle-free settings. In contrast, real-world environments such as homes, offices, and public spaces, are densely cluttered, three-dimensional, and geometrically constrained, requiring scene-aware whole-body coordination, precise balance control, and reasoning over spatial constraints imposed by furniture and household objects. However, humanoid locomotion in cluttered 3D environments remains underexplored, and no public dataset systematically couples full-body human locomotion with the scene geometry that shapes it. To address this gap, we present Moving Through Clutter (MTC), an opensource Virtual Reality (VR) based data collection and evaluation framework for scene-aware humanoid locomotion in cluttered environments. Our system procedurally generates scenes with controllable clutter levels and captures embodiment-consistent, whole-body human motion through immersive VR navigation, which is then automatically retargeted to a humanoid robot model. We further introduce benchmarks that quantify environment clutter level and locomotion performance, including stability and collision safety. Using this framework, we compile a dataset of 348 trajectories across 145 diverse 3D cluttered scenes. The dataset provides a foundation for studying geometry-induced adaptation in humanoid locomotion and developing scene-aware planning and control methods. |
| 2026-03-06 | [Technical Report: Automated Optical Inspection of Surgical Instruments](http://arxiv.org/abs/2603.05987v1) | Zunaira Shafqat, Atif Aftab Ahmed Jilani et al. | In the dynamic landscape of modern healthcare, maintaining the highest standards in surgical instruments is critical for clinical success. This report explores the diverse realm of surgical instruments and their associated manufacturing defects, emphasizing their pivotal role in ensuring the safety of surgical procedures. With potentially fatal consequences arising from even minor defects, precision in manufacturing is paramount.The report addresses the identification and rectification of critical defects such as cracks, rust, and structural irregularities. Such scrutiny prevents substantial financial losses for manufacturers and, more crucially, safeguards patient lives. The collaboration with industry leaders Daddy D Pro and Dr. Frigz International, renowned trailblazers in the Sialkot surgical cluster, provides invaluable insights into the analysis of defects in Pakistani-made instruments. This partnership signifies a commitment to advancing automated defect detection methodologies, specifically through the integration of deep learning architectures including YOLOv8, ResNet-152, and EfficientNet-b4, thereby elevating quality standards in the manufacturing process. The scope of this report is to identify various surgical instruments manufactured in Pakistan and analyze their associated defects using a newly developed dataset of 4,414 high-resolution images. By focusing on quality assurance through Automated Optical Inspection (AOI) tools, this document serves as a resource for manufacturers, healthcare professionals, and regulatory bodies. The insights gained contribute to the enhancement of instrument standards, ensuring a more reliable healthcare environment through industry expertise and cutting-edge technology. |
| 2026-03-06 | [OD-RASE: Ontology-Driven Risk Assessment and Safety Enhancement for Autonomous Driving](http://arxiv.org/abs/2603.05936v1) | Kota Shimomura, Masaki Nambata et al. | Although autonomous driving systems demonstrate high perception performance, they still face limitations when handling rare situations or complex road structures. Such road infrastructures are designed for human drivers, safety improvements are typically introduced only after accidents occur. This reactive approach poses a significant challenge for autonomous systems, which require proactive risk mitigation. To address this issue, we propose OD-RASE, a framework for enhancing the safety of autonomous driving systems by detecting road structures that cause traffic accidents and connecting these findings to infrastructure development. First, we formalize an ontology based on specialized domain knowledge of road traffic systems. In parallel, we generate infrastructure improvement proposals using a large-scale visual language model (LVLM) and use ontology-driven data filtering to enhance their reliability. This process automatically annotates improvement proposals on pre-accident road images, leading to the construction of a new dataset. Furthermore, we introduce the Baseline approach (OD-RASE model), which leverages LVLM and a diffusion model to produce both infrastructure improvement proposals and generated images of the improved road environment. Our experiments demonstrate that ontology-driven data filtering enables highly accurate prediction of accident-causing road structures and the corresponding improvement plans. We believe that this work contributes to the overall safety of traffic environments and marks an important step toward the broader adoption of autonomous driving systems. |
| 2026-03-06 | [Iterative Convex Optimization with Control Barrier Functions for Obstacle Avoidance among Polytopes](http://arxiv.org/abs/2603.05916v1) | Shuo Liu, Zhe Huang et al. | Obstacle avoidance of polytopic obstacles by polytopic robots is a challenging problem in optimization-based control and trajectory planning. Many existing methods rely on smooth geometric approximations, such as hyperspheres or ellipsoids, which allow differentiable distance expressions but distort the true geometry and restrict the feasible set. Other approaches integrate exact polytope distances into nonlinear model predictive control (MPC), resulting in nonconvex programs that limit real-time performance. In this paper, we construct linear discrete-time control barrier function (DCBF) constraints by deriving supporting hyperplanes from exact closest-point computations between convex polytopes. We then propose a novel iterative convex MPC-DCBF framework, where local linearization of system dynamics and robot geometry ensures convexity of the finite-horizon optimization at each iteration. The resulting formulation reduces computational complexity and enables fast online implementation for safety-critical control and trajectory planning of general nonlinear dynamics. The framework extends to multi-robot and three-dimensional environments. Numerical experiments demonstrate collision-free navigation in cluttered maze scenarios with millisecond-level solve times. |
| 2026-03-06 | [Bayesian Linear Programming under Learned Uncertainty: Posterior Feasibility Guarantees, Scenario Certification, and Applications](http://arxiv.org/abs/2603.05885v1) | Debashis Chatterjee | Linear programming is widely used for decision-making in science, engineering, and operations research, yet in many modern applications the coefficients entering the constraints and objective are not known exactly and must be learned from data. Classical stochastic and robust optimization offer two influential paradigms for handling such uncertainty, but they typically treat the underlying uncertainty description as given and do not directly integrate priors and updated to posteriors guarantees. This paper develops a Bayesian framework for linear programming in which uncertain quantities are modeled probabilistically, updated through observed data, and propagated into optimization through posterior feasibility requirements. We present two complementary computational strategies: a credible-region robustification that converts posterior uncertainty into deterministic protection, and a posterior-scenario approach that uses sampled posterior realizations to construct tractable optimization problems with finite-sample interpretability. We also propose a Monte Carlo certification procedure that provides conservative, data-conditioned assessments of residual infeasibility. Simulation experiments show that the proposed framework substantially improves safety relative to naive plug-in decisions, while a real-data study on single-cell transcriptomic data demonstrates that the approach can produce scientifically interpretable decisions together with explicit uncertainty-aware feasibility diagnostics. The proposed methodology offers a unified bridge between Bayesian learning, optimization under uncertainty, and practical decision certification. |
| 2026-03-06 | [Expert Knowledge-driven Reinforcement Learning for Autonomous Racing via Trajectory Guidance and Dynamics Constraints](http://arxiv.org/abs/2603.05842v1) | Bo Leng, Weiqi Zhang et al. | Reinforcement learning has demonstrated significant potential in the field of autonomous driving. However, it suffers from defects such as training instability and unsafe action outputs when faced with autonomous racing environments characterized by high dynamics and strong nonlinearities. To this end, this paper proposes a trajectory guidance and dynamics constraints Reinforcement Learning (TraD-RL) method for autonomous racing. The key features of this method are as follows: 1) leveraging the prior expert racing line to construct an augmented state representation and facilitate reward shaping, thereby integrating domain knowledge to stabilize early-stage policy learning; 2) embedding explicit vehicle dynamic priors into a safe operating envelope formulated via control barrier functions to enable safety-constrained learning; and 3) adopting a multi-stage curriculum learning strategy that shifts from expert-guided learning to autonomous exploration, allowing the learned policy to surpass expert-level performance. The proposed method is evaluated in a high-fidelity simulation environment modeled after the Tempelhof Airport Street Circuit. Experimental results demonstrate that TraD-RL effectively improves both lap speed and driving stability of the autonomous racing vehicle, achieving a synergistic optimization of racing performance and safety. |
| 2026-03-06 | [Proof-of-Guardrail in AI Agents and What (Not) to Trust from It](http://arxiv.org/abs/2603.05786v1) | Xisen Jin, Michael Duan et al. | As AI agents become widely deployed as online services, users often rely on an agent developer's claim about how safety is enforced, which introduces a threat where safety measures are falsely advertised. To address the threat, we propose proof-of-guardrail, a system that enables developers to provide cryptographic proof that a response is generated after a specific open-source guardrail. To generate proof, the developer runs the agent and guardrail inside a Trusted Execution Environment (TEE), which produces a TEE-signed attestation of guardrail code execution verifiable by any user offline. We implement proof-of-guardrail for OpenClaw agents and evaluate latency overhead and deployment cost. Proof-of-guardrail ensures integrity of guardrail execution while keeping the developer's agent private, but we also highlight a risk of deception about safety, for example, when malicious developers actively jailbreak the guardrail. Code and demo video: https://github.com/SaharaLabsAI/Verifiable-ClawGuard |
| 2026-03-06 | [Knowing without Acting: The Disentangled Geometry of Safety Mechanisms in Large Language Models](http://arxiv.org/abs/2603.05773v1) | Jinman Wu, Yi Xie et al. | Safety alignment is often conceptualized as a monolithic process wherein harmfulness detection automatically triggers refusal. However, the persistence of jailbreak attacks suggests a fundamental mechanistic decoupling. We propose the \textbf{\underline{D}}isentangled \textbf{\underline{S}}afety \textbf{\underline{H}}ypothesis \textbf{(DSH)}, positing that safety computation operates on two distinct subspaces: a \textit{Recognition Axis} ($\mathbf{v}_H$, ``Knowing'') and an \textit{Execution Axis} ($\mathbf{v}_R$, ``Acting''). Our geometric analysis reveals a universal ``Reflex-to-Dissociation'' evolution, where these signals transition from antagonistic entanglement in early layers to structural independence in deep layers. To validate this, we introduce \textit{Double-Difference Extraction} and \textit{Adaptive Causal Steering}. Using our curated \textsc{AmbiguityBench}, we demonstrate a causal double dissociation, effectively creating a state of ``Knowing without Acting.'' Crucially, we leverage this disentanglement to propose the \textbf{Refusal Erasure Attack (REA)}, which achieves State-of-the-Art attack success rates by surgically lobotomizing the refusal mechanism. Furthermore, we uncover a critical architectural divergence, contrasting the \textit{Explicit Semantic Control} of Llama3.1 with the \textit{Latent Distributed Control} of Qwen2.5. The code and dataset are available at https://anonymous.4open.science/r/DSH. |

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



