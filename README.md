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
| 2026-02-13 | [Conversational Image Segmentation: Grounding Abstract Concepts with Scalable Supervision](http://arxiv.org/abs/2602.13195v1) | Aadarsh Sahoo, Georgia Gkioxari | Conversational image segmentation grounds abstract, intent-driven concepts into pixel-accurate masks. Prior work on referring image grounding focuses on categorical and spatial queries (e.g., "left-most apple") and overlooks functional and physical reasoning (e.g., "where can I safely store the knife?"). We address this gap and introduce Conversational Image Segmentation (CIS) and ConverSeg, a benchmark spanning entities, spatial relations, intent, affordances, functions, safety, and physical reasoning. We also present ConverSeg-Net, which fuses strong segmentation priors with language understanding, and an AI-powered data engine that generates prompt-mask pairs without human supervision. We show that current language-guided segmentation models are inadequate for CIS, while ConverSeg-Net trained on our data engine achieves significant gains on ConverSeg and maintains strong performance on existing language-guided segmentation benchmarks. Project webpage: https://glab-caltech.github.io/converseg/ |
| 2026-02-13 | [Realistic Face Reconstruction from Facial Embeddings via Diffusion Models](http://arxiv.org/abs/2602.13168v1) | Dong Han, Yong Li et al. | With the advancement of face recognition (FR) systems, privacy-preserving face recognition (PPFR) systems have gained popularity for their accurate recognition, enhanced facial privacy protection, and robustness to various attacks. However, there are limited studies to further verify privacy risks by reconstructing realistic high-resolution face images from embeddings of these systems, especially for PPFR. In this work, we propose the face embedding mapping (FEM), a general framework that explores Kolmogorov-Arnold Network (KAN) for conducting the embedding-to-face attack by leveraging pre-trained Identity-Preserving diffusion model against state-of-the-art (SOTA) FR and PPFR systems. Based on extensive experiments, we verify that reconstructed faces can be used for accessing other real-word FR systems. Besides, the proposed method shows the robustness in reconstructing faces from the partial and protected face embeddings. Moreover, FEM can be utilized as a tool for evaluating safety of FR and PPFR systems in terms of privacy leakage. All images used in this work are from public datasets. |
| 2026-02-13 | [Optimal Take-off under Fuzzy Clearances](http://arxiv.org/abs/2602.13166v1) | Hugo Henry, Arthur Tsai et al. | This paper presents a hybrid obstacle avoidance architecture that integrates Optimal Control under clearance with a Fuzzy Rule Based System (FRBS) to enable adaptive constraint handling for unmanned aircraft. Motivated by the limitations of classical optimal control under uncertainty and the need for interpretable decision making in safety critical aviation systems, we design a three stage Takagi Sugeno Kang fuzzy layer that modulates constraint radii, urgency levels, and activation decisions based on regulatory separation minima and airworthiness guidelines from FAA and EASA. These fuzzy-derived clearances are then incorporated as soft constraints into an optimal control problem solved using the FALCON toolbox and IPOPT. The framework aims to reduce unnecessary recomputations by selectively activating obstacle avoidance updates while maintaining compliance with aviation procedures. A proof of concept implementation using a simplified aircraft model demonstrates that the approach can generate optimal trajectories with computation times of 2,3 seconds per iteration in a single threaded MATLAB environment, suggesting feasibility for near real time applications. However, our experiments revealed a critical software incompatibility in the latest versions of FALCON and IPOPT, in which the Lagrangian penalty term remained identically zero, preventing proper constraint enforcement. This behavior was consistent across scenarios and indicates a solver toolbox regression rather than a modeling flaw. Future work includes validating this effect by reverting to earlier software versions, optimizing the fuzzy membership functions using evolutionary methods, and extending the system to higher fidelity aircraft models and stochastic obstacle environments. |
| 2026-02-13 | [Human Emotion-Mediated Soft Robotic Arts: Exploring the Intersection of Human Emotions, Soft Robotics and Arts](http://arxiv.org/abs/2602.13163v1) | Saitarun Nadipineni, Chenhao Hong et al. | Soft robotics has emerged as a versatile field with applications across various domains, from healthcare to industrial automation, and more recently, art and interactive installations. The inherent flexibility, adaptability, and safety of soft robots make them ideal for applications that require delicate, organic, and lifelike movement, allowing for immersive and responsive interactions. This study explores the intersection of human emotions, soft robotics, and art to establish and create new forms of human emotion-mediated soft robotic art. In this paper, we introduce two soft embodiments: a soft character and a soft flower as an art display that dynamically responds to brain signals based on alpha waves, reflecting different emotion levels. We present how human emotions can be measured as alpha waves based on brain/EEG signals, how we map the alpha waves to the dynamic movements of the two soft embodiments, and demonstrate our proposed concept using experiments. The findings of this study highlight how soft robotics can embody human emotional states, offering a new medium for insightful artistic expression and interaction, and demonstrating how art displays can be embodied. |
| 2026-02-13 | [Temporally-Sampled Efficiently Adaptive State Lattices for Autonomous Ground Robot Navigation in Partially Observed Environments](http://arxiv.org/abs/2602.13159v1) | Ashwin Satish Menon, Eric R. Damm et al. | Due to sensor limitations, environments that off-road mobile robots operate in are often only partially observable. As the robots move throughout the environment and towards their goal, the optimal route is continuously revised as the sensors perceive new information. In traditional autonomous navigation architectures, a regional motion planner will consume the environment map and output a trajectory for the local motion planner to use as a reference. Due to the continuous revision of the regional plan guidance as a result of changing map information, the reference trajectories which are passed down to the local planner can differ significantly across sequential planning cycles. This rapidly changing guidance can result in unsafe navigation behavior, often requiring manual safety interventions during autonomous traversals in off-road environments. To remedy this problem, we propose Temporally-Sampled Efficiently Adaptive State Lattices (TSEASL), which is a regional planner arbitration architecture that considers updated and optimized versions of previously generated trajectories against the currently generated trajectory. When tested on a Clearpath Robotics Warthog Unmanned Ground Vehicle as well as real map data collected from the Warthog, results indicate that when running TSEASL, the robot did not require manual interventions in the same locations where the robot was running the baseline planner. Additionally, higher levels of planner stability were recorded with TSEASL over the baseline. The paper concludes with a discussion of further improvements to TSEASL in order to make it more generalizable to various off-road autonomy scenarios. |
| 2026-02-13 | [Eventizing Traditionally Opaque Binary Neural Networks as 1-safe Petri net Models](http://arxiv.org/abs/2602.13128v1) | Mohamed Tarraf, Alex Chan et al. | Binary Neural Networks (BNNs) offer a low-complexity and energy-efficient alternative to traditional full-precision neural networks by constraining their weights and activations to binary values. However, their discrete, highly non-linear behavior makes them difficult to explain, validate and formally verify. As a result, BNNs remain largely opaque, limiting their suitability in safety-critical domains, where causal transparency and behavioral guarantees are essential. In this work, we introduce a Petri net (PN)-based framework that captures the BNN's internal operations as event-driven processes. By "eventizing" their operations, we expose their causal relationships and dependencies for a fine-grained analysis of concurrency, ordering, and state evolution. Here, we construct modular PN blueprints for core BNN components including activation, gradient computation and weight updates, and compose them into a complete system-level model. We then validate the composed PN against a reference software-based BNN, verify it against reachability and structural checks to establish 1-safeness, deadlock-freeness, mutual exclusion and correct-by-construction causal sequencing, before we assess its scalability and complexity at segment, component, and system levels using the automated measurement tools in Workcraft. Overall, this framework enables causal introspection of transparent and event-driven BNNs that are amenable to formal reasoning and verification. |
| 2026-02-13 | [UniManip: General-Purpose Zero-Shot Robotic Manipulation with Agentic Operational Graph](http://arxiv.org/abs/2602.13086v1) | Haichao Liu, Yuanjiang Xue et al. | Achieving general-purpose robotic manipulation requires robots to seamlessly bridge high-level semantic intent with low-level physical interaction in unstructured environments. However, existing approaches falter in zero-shot generalization: end-to-end Vision-Language-Action (VLA) models often lack the precision required for long-horizon tasks, while traditional hierarchical planners suffer from semantic rigidity when facing open-world variations. To address this, we present UniManip, a framework grounded in a Bi-level Agentic Operational Graph (AOG) that unifies semantic reasoning and physical grounding. By coupling a high-level Agentic Layer for task orchestration with a low-level Scene Layer for dynamic state representation, the system continuously aligns abstract planning with geometric constraints, enabling robust zero-shot execution. Unlike static pipelines, UniManip operates as a dynamic agentic loop: it actively instantiates object-centric scene graphs from unstructured perception, parameterizes these representations into collision-free trajectories via a safety-aware local planner, and exploits structured memory to autonomously diagnose and recover from execution failures. Extensive experiments validate the system's robust zero-shot capability on unseen objects and tasks, demonstrating a 22.5% and 25.0% higher success rate compared to state-of-the-art VLA and hierarchical baselines, respectively. Notably, the system enables direct zero-shot transfer from fixed-base setups to mobile manipulation without fine-tuning or reconfiguration. Our open-source project page can be found at https://henryhcliu.github.io/unimanip. |
| 2026-02-13 | [Diverging Flows: Detecting Extrapolations in Conditional Generation](http://arxiv.org/abs/2602.13061v1) | Constantinos Tsakonas, Serena Ivaldi et al. | The ability of Flow Matching (FM) to model complex conditional distributions has established it as the state-of-the-art for prediction tasks (e.g., robotics, weather forecasting). However, deployment in safety-critical settings is hindered by a critical extrapolation hazard: driven by smoothness biases, flow models yield plausible outputs even for off-manifold conditions, resulting in silent failures indistinguishable from valid predictions. In this work, we introduce Diverging Flows, a novel approach that enables a single model to simultaneously perform conditional generation and native extrapolation detection by structurally enforcing inefficient transport for off-manifold inputs. We evaluate our method on synthetic manifolds, cross-domain style transfer, and weather temperature forecasting, demonstrating that it achieves effective detection of extrapolations without compromising predictive fidelity or inference latency. These results establish Diverging Flows as a robust solution for trustworthy flow models, paving the way for reliable deployment in domains such as medicine, robotics, and climate science. |
| 2026-02-13 | [TCRL: Temporal-Coupled Adversarial Training for Robust Constrained Reinforcement Learning in Worst-Case Scenarios](http://arxiv.org/abs/2602.13040v1) | Wentao Xu, Zhongming Yao et al. | Constrained Reinforcement Learning (CRL) aims to optimize decision-making policies under constraint conditions, making it highly applicable to safety-critical domains such as autonomous driving, robotics, and power grid management. However, existing robust CRL approaches predominantly focus on single-step perturbations and temporally independent adversarial models, lacking explicit modeling of robustness against temporally coupled perturbations. To tackle these challenges, we propose TCRL, a novel temporal-coupled adversarial training framework for robust constrained reinforcement learning (TCRL) in worst-case scenarios. First, TCRL introduces a worst-case-perceived cost constraint function that estimates safety costs under temporally coupled perturbations without the need to explicitly model adversarial attackers. Second, TCRL establishes a dual-constraint defense mechanism on the reward to counter temporally coupled adversaries while maintaining reward unpredictability. Experimental results demonstrate that TCRL consistently outperforms existing methods in terms of robustness against temporally coupled perturbation attacks across a variety of CRL tasks. |
| 2026-02-13 | [Buy versus Build an LLM: A Decision Framework for Governments](http://arxiv.org/abs/2602.13033v1) | Jiahao Lu, Ziwei Xu et al. | Large Language Models (LLMs) represent a new frontier of digital infrastructure that can support a wide range of public-sector applications, from general purpose citizen services to specialized and sensitive state functions. When expanding AI access, governments face a set of strategic choices over whether to buy existing services, build domestic capabilities, or adopt hybrid approaches across different domains and use cases. These are critical decisions especially when leading model providers are often foreign corporations, and LLM outputs are increasingly treated as trusted inputs to public decision-making and public discourse. In practice, these decisions are not intended to mandate a single approach across all domains; instead, national AI strategies are typically pluralistic, with sovereign, commercial and open-source models coexisting to serve different purposes. Governments may rely on commercial models for non-sensitive or commodity tasks, while pursuing greater control for critical, high-risk or strategically important applications.   This paper provides a strategic framework for making this decision by evaluating these options across dimensions including sovereignty, safety, cost, resource capability, cultural fit, and sustainability. Importantly, "building" does not imply that governments must act alone: domestic capabilities may be developed through public research institutions, universities, state-owned enterprises, joint ventures, or broader national ecosystems. By detailing the technical requirements and practical challenges of each pathway, this work aims to serve as a reference for policy-makers to determine whether a buy or build approach best aligns with their specific national needs and societal goals. |
| 2026-02-13 | [Detecting Object Tracking Failure via Sequential Hypothesis Testing](http://arxiv.org/abs/2602.12983v1) | Alejandro Monroy Muñoz, Rajeev Verma et al. | Real-time online object tracking in videos constitutes a core task in computer vision, with wide-ranging applications including video surveillance, motion capture, and robotics. Deployed tracking systems usually lack formal safety assurances to convey when tracking is reliable and when it may fail, at best relying on heuristic measures of model confidence to raise alerts. To obtain such assurances we propose interpreting object tracking as a sequential hypothesis test, wherein evidence for or against tracking failures is gradually accumulated over time. Leveraging recent advancements in the field, our sequential test (formalized as an e-process) quickly identifies when tracking failures set in whilst provably containing false alerts at a desired rate, and thus limiting potentially costly re-calibration or intervention steps. The approach is computationally light-weight, requires no extra training or fine-tuning, and is in principle model-agnostic. We propose both supervised and unsupervised variants by leveraging either ground-truth or solely internal tracking information, and demonstrate its effectiveness for two established tracking models across four video benchmarks. As such, sequential testing can offer a statistically grounded and efficient mechanism to incorporate safety assurances into real-time tracking systems. |
| 2026-02-13 | [Solving Qualitative Multi-Objective Stochastic Games](http://arxiv.org/abs/2602.12927v1) | Moritz Graf, Anthony Lin et al. | Many problems in compositional synthesis and verification of multi-agent systems -- such as rational verification and assume-guarantee verification in probabilistic systems -- reduce to reasoning about two-player multi-objective stochastic games. This motivates us to study the problem of characterizing the complexity and memory requirements for two-player stochastic games with Boolean combinations of qualitative reachability and safety objectives. Reachability objectives require that a given set of states is reached; safety requires that a given set is invariant. A qualitative winning condition asks that an objective is satisfied almost surely (AS) or (in negated form) with non-zero (NZ) probability.   We study the determinacy and complexity landscape of the problem. We show that games with conjunctions of AS and NZ reachability and safety objectives are determined, and determining the winner is PSPACE-complete. The same holds for positive boolean combinations of AS reachability and safety, as well as for negations thereof. On the other hand, games with full Boolean combinations of qualitative objectives are not determined, and are NEXPTIME-hard. Our hardness results show a connection between stochastic games and logics with partially-ordered quantification. Our results shed light on the relationship between determinacy and complexity, and extend the complexity landscape for stochastic games in the multi-objective setting. |
| 2026-02-13 | [Robustness of Object Detection of Autonomous Vehicles in Adverse Weather Conditions](http://arxiv.org/abs/2602.12902v1) | Fox Pettersen, Hong Zhu | As self-driving technology advances toward widespread adoption, determining safe operational thresholds across varying environmental conditions becomes critical for public safety. This paper proposes a method for evaluating the robustness of object detection ML models in autonomous vehicles under adverse weather conditions. It employs data augmentation operators to generate synthetic data that simulates different severance degrees of the adverse operation conditions at progressive intensity levels to find the lowest intensity of the adverse conditions at which the object detection model fails. The robustness of the object detection model is measured by the average first failure coefficients (AFFC) over the input images in the benchmark. The paper reports an experiment with four object detection models: YOLOv5s, YOLOv11s, Faster R-CNN, and Detectron2, utilising seven data augmentation operators that simulate weather conditions fog, rain, and snow, and lighting conditions of dark, bright, flaring, and shadow. The experiment data show that the method is feasible, effective, and efficient to evaluate and compare the robustness of object detection models in various adverse operation conditions. In particular, the Faster R-CNN model achieved the highest robustness with an overall average AFFC of 71.9% over all seven adverse conditions, while YOLO variants showed the AFFC values of 43%. The method is also applied to assess the impact of model training that targets adverse operation conditions using synthetic data on model robustness. It is observed that such training can improve robustness in adverse conditions but may suffer from diminishing returns and forgetting phenomena (i.e., decline in robustness) if overtrained. |
| 2026-02-13 | [X-VORTEX: Spatio-Temporal Contrastive Learning for Wake Vortex Trajectory Forecasting](http://arxiv.org/abs/2602.12869v1) | Zhan Qu, Michael Färber | Wake vortices are strong, coherent air turbulences created by aircraft, and they pose a major safety and capacity challenge for air traffic management. Tracking how vortices move, weaken, and dissipate over time from LiDAR measurements is still difficult because scans are sparse, vortex signatures fade as the flow breaks down under atmospheric turbulence and instabilities, and point-wise annotation is prohibitively expensive. Existing approaches largely treat each scan as an independent, fully supervised segmentation problem, which overlooks temporal structure and does not scale to the vast unlabeled archives collected in practice. We present X-VORTEX, a spatio-temporal contrastive learning framework grounded in Augmentation Overlap Theory that learns physics-aware representations from unlabeled LiDAR point cloud sequences. X-VORTEX addresses two core challenges: sensor sparsity and time-varying vortex dynamics. It constructs paired inputs from the same underlying flight event by combining a weakly perturbed sequence with a strongly augmented counterpart produced via temporal subsampling and spatial masking, encouraging the model to align representations across missing frames and partial observations. Architecturally, a time-distributed geometric encoder extracts per-scan features and a sequential aggregator models the evolving vortex state across variable-length sequences. We evaluate on a real-world dataset of over one million LiDAR scans. X-VORTEX achieves superior vortex center localization while using only 1% of the labeled data required by supervised baselines, and the learned representations support accurate trajectory forecasting. |
| 2026-02-13 | [TRACE: Temporal Reasoning via Agentic Context Evolution for Streaming Electronic Health Records (EHRs)](http://arxiv.org/abs/2602.12833v1) | Zhan Qu, Michael Färber | Large Language Models (LLMs) encode extensive medical knowledge but struggle to apply it reliably to longitudinal patient trajectories, where evolving clinical states, irregular timing, and heterogeneous events degrade performance over time. Existing adaptation strategies rely on fine-tuning or retrieval-based augmentation, which introduce computational overhead, privacy constraints, or instability under long contexts. We introduce TRACE (Temporal Reasoning via Agentic Context Evolution), a framework that enables temporal clinical reasoning with frozen LLMs by explicitly structuring and maintaining context rather than extending context windows or updating parameters. TRACE operates over a dual-memory architecture consisting of a static Global Protocol encoding institutional clinical rules and a dynamic Individual Protocol tracking patient-specific state. Four agentic components, Router, Reasoner, Auditor, and Steward, coordinate over this structured memory to support temporal inference and state evolution. The framework maintains bounded inference cost via structured state compression and selectively audits safety-critical clinical decisions. Evaluated on longitudinal clinical event streams from MIMIC-IV, TRACE significantly improves next-event prediction accuracy, protocol adherence, and clinical safety over long-context and retrieval-augmented baselines, while producing interpretable and auditable reasoning traces. |
| 2026-02-13 | [SafeFlowMPC: Predictive and Safe Trajectory Planning for Robot Manipulators with Learning-based Policies](http://arxiv.org/abs/2602.12794v1) | Thies Oelerich, Gerald Ebmer et al. | The emerging integration of robots into everyday life brings several major challenges. Compared to classical industrial applications, more flexibility is needed in combination with real-time reactivity. Learning-based methods can train powerful policies based on demonstrated trajectories, such that the robot generalizes a task to similar situations. However, these black-box models lack interpretability and rigorous safety guarantees. Optimization-based methods provide these guarantees but lack the required flexibility and generalization capabilities. This work proposes SafeFlowMPC, a combination of flow matching and online optimization to combine the strengths of learning and optimization. This method guarantees safety at all times and is designed to meet the demands of real-time execution by using a suboptimal model-predictive control formulation. SafeFlowMPC achieves strong performance in three real-world experiments on a KUKA 7-DoF manipulator, namely two grasping experiment and a dynamic human-robot object handover experiment. A video of the experiments is available at http://www.acin.tuwien.ac.at/42d6. The code is available at https://github.com/TU-Wien-ACIN-CDS/SafeFlowMPC. |
| 2026-02-13 | [Media Framing Moderates Risk-Benefit Perceptions and Value Tradeoffs in Human-Robot Collaboration](http://arxiv.org/abs/2602.12785v1) | Philipp Brauner, Felix Glawe et al. | Public acceptance of industrial human-robot collaboration (HRC) is shaped by how risks and benefits are perceived by affected employees. Positive or negative media framing may shape and shift how individuals evaluate HRC. This study examines how message framing moderates the effects of perceived risks and perceived benefits on overall attributed value. In a pre-registered study, participants (N = 1150) were randomly assigned to read either a positively or negatively framed newspaper article in one of three industrial contexts (autonomy, employment, safety) about HRC in production. Subsequently, perceived risks, benefits, and value were measured using reliable and publicly available psychometric scales. Two multiple regressions (one per framing condition) tested for main and interaction effects. Framing influenced absolute evaluations of risk, benefits, and value. In both frames, risks and benefits significantly predicted attributed value. Under positive framing, only main effects were observed (risks: beta = -0.52; benefits: beta = 0.45). Under negative framing, both predictors had stronger main effects (risks: beta = -0.69; benefits: beta = 0.63) along with a significant negative interaction (beta = -0.32), indicating that higher perceived risk diminishes the positive effect of perceived benefits. Model fit was higher for the positive frame (R^2 = 0.715) than for the negative frame (R^2 = 0.583), indicating greater explained variance in value attributions. Framing shapes the absolute evaluation of HRC and how risks and benefits are cognitively integrated in trade-offs. Negative framing produces stronger but interdependent effects, whereas positive framing supports additive evaluations. These findings highlight the role of strategic communication in fostering acceptance of HRC and underscore the need to consider framing in future HRC research. |
| 2026-02-13 | [GNSS Jamming Detection with Automatic Gain Control (AGC) and Carrier-to-Noise Ratio Density (CNO) Observables from a COTS receiver](http://arxiv.org/abs/2602.12688v1) | Syed Ali Kazim, Anas Darwich et al. | As rail transport moves toward higher degrees of automation under initiatives like the R2DATO project [1], accurate and reliable train localization has become essential. Global Satellite Navigation System (GNSS) is considered as a main technology in enabling operational advancements including Automatic Train Operation (ATO), moving block signaling, and virtual coupling, which are the core components of the Horizon Europe 2024 rail digitalization agenda. However, GNSS signal integrity is increasingly threatened by intentional and unintentional radio frequency interference (RFI). This include jamming and spoofing, which are particularly concerning as the broadcasted signal can deliberately disrupt or manipulate the GNSS signal. - Jamming refers to an intentional form of interference that induces disturbances in the GNSS band, causing performance degradation or can even entirely block the receiver from acquiring the satellite signals. - Spoofing involves broadcasting counterfeit satellite signals to deceive the GNSS receiver, leading to inaccurate estimation of position, navigation and timing information. This concern about interference is not unique to rail applications. The aeronautical sector has long recognized the risks posed by GNSS interference, with extensive documentation on its impact on navigation, landing procedures, and surveillance systems. In recent years, awareness of these risks has expanded to other transport sectors. Within the automotive industry, particularly in Intelligent Transport Systems (ITS), several studies [2][3][4] have addressed the vulnerability of GNSS against interference. Similar concerns are now emerging in the rail domain [5][6][7], especially as GNSS is increasingly adopted in safety-critical applications. In literature, several levels of actions have been explored, ranging from merely the detection of a malicious signal at the initial phase to the application of advanced signal processing methods aimed at suppressing the effects of interference [8]. In alignment with the goal of the R2DATO project, we evaluated the impact of various classes of interference signals such as amplitude modulation (AM), frequency modulation (FM), pulsed, frequency hopping and chirp signals on the GNSS observables including Automatic Gain Control (AGC) and Carrier to Noise Ratio (CNO) as measured by a Commercial Off-The-Shelf (COTS). However, in this work, the analysis is only limited to impact of chirp interference on GPS L1 receiver observables and detection performance. |
| 2026-02-13 | [Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents](http://arxiv.org/abs/2602.12662v1) | Ruihan Yang, Fanghua Ye et al. | Large language models (LLMs) are increasingly deployed as autonomous agents for multi-turn decision-making tasks. However, current agents typically rely on fixed cognitive patterns: non-thinking models generate immediate responses, while thinking models engage in deep reasoning uniformly. This rigidity is inefficient for long-horizon tasks, where cognitive demands vary significantly from step to step, with some requiring strategic planning and others only routine execution. In this paper, we introduce CogRouter, a framework that trains agents to dynamically adapt cognitive depth at each step. Grounded in ACT-R theory, we design four hierarchical cognitive levels ranging from instinctive responses to strategic planning. Our two-stage training approach includes Cognition-aware Supervised Fine-tuning (CoSFT) to instill stable level-specific patterns, and Cognition-aware Policy Optimization (CoPO) for step-level credit assignment via confidence-aware advantage reweighting. The key insight is that appropriate cognitive depth should maximize the confidence of the resulting action. Experiments on ALFWorld and ScienceWorld demonstrate that CogRouter achieves state-of-the-art performance with superior efficiency. With Qwen2.5-7B, it reaches an 82.3% success rate, outperforming GPT-4o (+40.3%), OpenAI-o3 (+18.3%), and GRPO (+14.0%), while using 62% fewer tokens. |
| 2026-02-13 | [Safe Controller Synthesis Using Lyapunov-based Barriers for Linear Hybrid Systems with Simplex Architecture](http://arxiv.org/abs/2602.12638v1) | Sunandan Adhikary, Soumyajit Dey | Modern cyber-physical systems often have a two-layered design, where the primary controller is AI-enabled or an analytical controller optimising some specific cost function. If the resulting control action is perceived as unsafe, a secondary safety-focused backup controller is activated. The existing backup controller design schemes do not consider a real-time deadline for the course correction of a potentially unsafe system trajectory or constrain maximisation of the safe operating region as a synthesis criterion. This essentially implies an eventual safety guarantee over a small operating region.   This paper proposes a novel design method for backup safe controllers (BSCs) that ensure invariance across the largest possible region in the safe state space, along with a guarantee for timely recovery when the system states deviate from their usual behaviour. This is the first work to synthesise safe controllers that ensure maximal safety and timely recovery while aiming at minimal resource usage by switching between BSCs with different execution rates. An online safe controller activation policy is also proposed to switch between BSCs (and the primary optimal controller) to optimise processing bandwidth for control computation. To establish the efficacy of the proposed method, we evaluate the safety and recovery time of the proposed safe controllers, as well as the activation policy, in closed loops with linear hybrid dynamical systems under budgeted bandwidth. |

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



