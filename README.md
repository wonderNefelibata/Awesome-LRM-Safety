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
| 2026-03-09 | [Cybersecurity AI: Hacking Consumer Robots in the AI Era](http://arxiv.org/abs/2603.08665v1) | Víctor Mayoral-Vilches, Unai Ayucar-Carbajo et al. | Is robot cybersecurity broken by AI? Consumer robots -- from autonomous lawnmowers to powered exoskeletons and window cleaners -- are rapidly entering homes and workplaces, yet their security remains rooted in assumptions of specialized attacker expertise. This paper presents evidence that Generative AI has fundamentally disrupted robot cybersecurity: what historically required deep knowledge of ROS, ROS 2, and robotic system internals can now be automated by anyone with access to state-of-the-art GenAI tools spearheaded by the open source CAI (Cybersecurity AI). We provide empirical evidence through three case studies: (1) compromising a Hookii autonomous lawnmower robot, uncovering fleet-wide vulnerabilities and data protection violations affecting 267+ connected devices, (2) exploiting a Hypershell powered exoskeleton, demonstrating safety-critical motor control weaknesses and credential exposure including access to over 3,300 internal support emails, and (3) breaching a HOBOT S7 Pro window cleaning robot, achieving unauthenticated BLE command injection and OTA firmware exploitation. Across these platforms, CAI discovered in an automated manner 38 vulnerabilities that would have previously required months of specialized security research. Our findings reveal a stark asymmetry: while offensive capabilities have been democratized through AI, defensive measures often remain lagging behind. We argue that traditional defense-in-depth architectures like the Robot Immune System (RIS) must evolve toward GenAI-native defensive agents capable of matching the speed and adaptability of AI-powered attacks. |
| 2026-03-09 | [FOMO-3D: Using Vision Foundation Models for Long-Tailed 3D Object Detection](http://arxiv.org/abs/2603.08611v1) | Anqi Joyce Yang, James Tu et al. | In order to navigate complex traffic environments, self-driving vehicles must recognize many semantic classes pertaining to vulnerable road users or traffic control devices. However, many safety-critical objects (e.g., construction worker) appear infrequently in nominal traffic conditions, leading to a severe shortage of training examples from driving data alone. Recent vision foundation models, which are trained on a large corpus of data, can serve as a good source of external prior knowledge to improve generalization. We propose FOMO-3D, the first multi-modal 3D detector to leverage vision foundation models for long-tailed 3D detection. Specifically, FOMO-3D exploits rich semantic and depth priors from OWLv2 and Metric3Dv2 within a two-stage detection paradigm that first generates proposals with a LiDAR-based branch and a novel camera-based branch, and refines them with attention especially to image features from OWL. Evaluations on real-world driving data show that using rich priors from vision foundation models with careful multi-modal fusion designs leads to large gains for long-tailed 3D detection. Project website is at https://waabi.ai/fomo3d/. |
| 2026-03-09 | [Drift-to-Action Controllers: Budgeted Interventions with Online Risk Certificates](http://arxiv.org/abs/2603.08578v1) | Ismail Lamaakal, Chaymae Yahyati et al. | Deployed machine learning systems face distribution drift, yet most monitoring pipelines stop at alarms and leave the response underspecified under labeling, compute, and latency constraints. We introduce Drift2Act, a drift-to-action controller that treats monitoring as constrained decision-making with explicit safety. Drift2Act combines a sensing layer that maps unlabeled monitoring signals to a belief over drift types with an active risk certificate that queries a small set of delayed labels from a recent window to produce an anytime-valid upper bound $U_t(δ)$ on current risk. The certificate gates operation: if $U_t(δ) \le τ$, the controller selects low-cost actions (e.g., recalibration or test-time adaptation); if $U_t(δ) > τ$, it activates abstain/handoff and escalates to rollback or retraining under cooldowns. In a realistic streaming protocol with label delay and explicit intervention costs, Drift2Act achieves near-zero safety violations and fast recovery at moderate cost on WILDS Camelyon17, DomainNet, and a controlled synthetic drift stream, outperforming alarm-only monitoring, adapt-always adaptation, schedule-based retraining, selective prediction alone, and an ablation without certification. Overall, online risk certification enables reliable drift response and reframes monitoring as decision-making with safety. |
| 2026-03-09 | [SCAFFOLD-CEGIS: Preventing Latent Security Degradation in LLM-Driven Iterative Code Refinement](http://arxiv.org/abs/2603.08520v1) | Yi Chen, Yun Bian et al. | The application of large language models to code generation has evolved from one-shot generation to iterative refinement, yet the evolution of security throughout iteration remains insufficiently understood. Through comparative experiments on three mainstream LLMs, this paper reveals the iterative refinement paradox: specification drift during multi-objective optimization causes security to degrade gradually over successive iterations. Taking GPT-4o as an example, 43.7 % of iteration chains contain more vulnerabilities than the baseline after ten rounds, and cross-model experiments show that this phenomenon is prevalent. Further analysis shows that simply introducing static application security testing (SAST) gating cannot effectively suppress degradation; instead, it increases the latent security degradation rate from 12.5% under the unprotected baseline to 20.8 %. The root cause is that static-analysis rules cannot cover structural degradations such as the removal of defensive logic or the weakening of exception handling. To address this problem, we propose the SCAFFOLD-CEGIS framework. Drawing on the counterexample-guided inductive synthesis (CEGIS) paradigm, the framework adopts a multi-agent collaborative architecture that transforms security constraints from implicit prompts into explicit verifiable constraints. It automatically identifies and solidifies security-critical elements as hard constraints through semantic anchoring, enforces safety monotonicity through four-layer gated verification, and continuously assimilates experience from failures. Comparative experiments against six existing defense methods show that the full framework reduces the latent security degradation rate to 2.1% and achieves a safety monotonicity rate of 100%. |
| 2026-03-09 | [Oracle-Guided Soft Shielding for Safe Move Prediction in Chess](http://arxiv.org/abs/2603.08506v1) | Prajit T Rajendran, Fabio Arnez et al. | In high stakes environments, agents relying purely on imitation learning or reinforcement learning often struggle to avoid safety-critical errors during exploration. Existing reinforcement learning approaches for environments such as chess require hundreds of thousands of episodes and substantial computational resources to converge. Imitation learning, on the other hand, is more sample efficient but is brittle under distributional shift and lacks mechanisms for proactive risk avoidance. In this work, we propose Oracle-Guided Soft Shielding (OGSS), a simple yet effective framework for safer decision-making, enabling safe exploration by learning a probabilistic safety model from oracle feedback in an imitation learning setting. Focusing on the domain of chess, we train a model to predict strong moves based on past games, and separately learn a blunder prediction model from Stockfish evaluations to estimate the tactical risk of each move. During inference, the agent first generates a set of candidate moves and then uses the blunder model to determine high-risk options, and uses a utility function combining the predicted move likelihood from the policy model and the blunder probability to select actions that strike a balance between performance and safety. This enables the agent to explore and play competitively while significantly reducing the chance of tactical mistakes. Across hundreds of games against a strong chess engine, we compare our approach with other methods in the literature, such as action pruning, SafeDAgger, and uncertainty-based sampling. Our results demonstrate that OGSS variants maintain a lower blunder rate even as the agent's exploration ratio is increased by several folds, highlighting its ability to support broader exploration without compromising tactical soundness. |
| 2026-03-09 | [Efficient Credal Prediction through Decalibration](http://arxiv.org/abs/2603.08495v1) | Paul Hofman, Timo Löhr et al. | A reliable representation of uncertainty is essential for the application of modern machine learning methods in safety-critical settings. In this regard, the use of credal sets (i.e., convex sets of probability distributions) has recently been proposed as a suitable approach to representing epistemic uncertainty. However, as with other approaches to epistemic uncertainty, training credal predictors is computationally complex and usually involves (re-)training an ensemble of models. The resulting computational complexity prevents their adoption for complex models such as foundation models and multi-modal systems. To address this problem, we propose an efficient method for credal prediction that is grounded in the notion of relative likelihood and inspired by techniques for the calibration of probabilistic classifiers. For each class label, our method predicts a range of plausible probabilities in the form of an interval. To produce the lower and upper bounds of these intervals, we propose a technique that we refer to as decalibration. Extensive experiments show that our method yields credal sets with strong performance across diverse tasks, including coverage-efficiency evaluation, out-of-distribution detection, and in-context learning. Notably, we demonstrate credal prediction on models such as TabPFN and CLIP -- architectures for which the construction of credal sets was previously infeasible. |
| 2026-03-09 | [An Open-Source Robotics Research Platform for Autonomous Laparoscopic Surgery](http://arxiv.org/abs/2603.08490v1) | Ariel Rodriguez, Lorenzo Mazza et al. | Autonomous robot-assisted surgery demands reliable, high-precision platforms that strictly adhere to the safety and kinematic constraints of minimally invasive procedures. Existing research platforms, primarily based on the da Vinci Research Kit, suffer from cable-driven mechanical limitations that degrade state-space consistency and hinder the downstream training of reliable autonomous policies. We present an open-source, robot-agnostic Remote Center of Motion (RCM) controller based on a closed-form analytical velocity solver that enforces the trocar constraint deterministically without iterative optimization. The controller operates in Cartesian space, enabling any industrial manipulator to function as a surgical robot. We provide implementations for the UR5e and Franka Emika Panda manipulators, and integrate stereoscopic 3D perception. We integrate the robot control into a full-stack ROS-based surgical robotics platform supporting teleoperation, demonstration recording, and deployment of learned policies via a decoupled server-client architecture. We validate the system on a bowel grasping and retraction task across phantom, ex vivo, and in vivo porcine laparoscopic procedures. RCM deviations remain sub-millimeter across all conditions, and trajectory smoothness metrics (SPARC, LDLJ) are comparable to expert demonstrations from the JIGSAWS benchmark recorded on the da Vinci system. These results demonstrate that the platform provides the precision and robustness required for teleoperation, data collection and autonomous policy deployment in realistic surgical scenarios. |
| 2026-03-09 | [Visual Self-Fulfilling Alignment: Shaping Safety-Oriented Personas via Threat-Related Images](http://arxiv.org/abs/2603.08486v1) | Qishun Yang, Shu Yang et al. | Multimodal large language models (MLLMs) face safety misalignment, where visual inputs enable harmful outputs. To address this, existing methods require explicit safety labels or contrastive data; yet, threat-related concepts are concrete and visually depictable, while safety concepts, like helpfulness, are abstract and lack visual referents. Inspired by the Self-Fulfilling mechanism underlying emergent misalignment, we propose Visual Self-Fulfilling Alignment (VSFA). VSFA fine-tunes vision-language models (VLMs) on neutral VQA tasks constructed around threat-related images, without any safety labels. Through repeated exposure to threat-related visual content, models internalize the implicit semantics of vigilance and caution, shaping safety-oriented personas. Experiments across multiple VLMs and safety benchmarks demonstrate that VSFA reduces the attack success rate, improves response quality, and mitigates over-refusal while preserving general capabilities. Our work extends the self-fulfilling mechanism from text to visual modalities, offering a label-free approach to VLMs alignment. |
| 2026-03-09 | [A prospective clinical feasibility study of a conversational diagnostic AI in an ambulatory primary care clinic](http://arxiv.org/abs/2603.08448v1) | Peter Brodeur, Jacob M. Koshy et al. | Large language model (LLM)-based AI systems have shown promise for patient-facing diagnostic and management conversations in simulated settings. Translating these systems into clinical practice requires assessment in real-world workflows with rigorous safety oversight. We report a prospective, single-arm feasibility study of an LLM-based conversational AI, the Articulate Medical Intelligence Explorer (AMIE), conducting clinical history taking and presentation of potential diagnoses for patients to discuss with their provider at urgent care appointments at a leading academic medical center. 100 adult patients completed an AMIE text-chat interaction up to 5 days before their appointment. We sought to assess the conversational safety and quality, patient and clinician experience, and clinical reasoning capabilities compared to primary care providers (PCPs). Human safety supervisors monitored all patient-AMIE interactions in real time and did not need to intervene to stop any consultations based on pre-defined criteria. Patients reported high satisfaction and their attitudes towards AI improved after interacting with AMIE (p < 0.001). PCPs found AMIE's output useful with a positive impact on preparedness. AMIE's differential diagnosis (DDx) included the final diagnosis, per chart review 8 weeks post-encounter, in 90% of cases, with 75% top-3 accuracy. Blinded assessment of AMIE and PCP DDx and management (Mx) plans suggested similar overall DDx and Mx plan quality, without significant differences for DDx (p = 0.6) and appropriateness and safety of Mx (p = 0.1 and 1.0, respectively). PCPs outperformed AMIE in the practicality (p = 0.003) and cost effectiveness (p = 0.004) of Mx. While further research is needed, this study demonstrates the initial feasibility, safety, and user acceptance of conversational AI in a real-world setting, representing crucial steps towards clinical translation. |
| 2026-03-09 | [Graph Based Semantic Encoder Decoder Framework for Task Oriented Communications in Connected Autonomous Vehicles](http://arxiv.org/abs/2603.08438v1) | Soheyb Ribouh, Phil Polo Ditsia Di Ngoma | Connected autonomous vehicles (CAVs) require reliable and efficient communication frameworks to support safety critical and task-oriented applications such as collision avoidance, cooperative perception, and traffic risk assessment. Traditional communication paradigms, which focus on transmitting raw bits, often incur excessive bandwidth consumption and fail to preserve the semantic relevance of transmitted information. To bridge this gap, we propose a Graph-Based Semantic Encoder-Decoder (GBSED) architecture tailored for task-oriented communications in CAV networks. The encoder leverages scene graphs to capture spatial and semantic relationships among road entities, combined with a semantic compression algorithm that reduces the size of the extracted graph based representations by up to 99% compared to raw images, while the decoder reconstructs task relevant representations rather than raw data. This design enables a significant reduction in communication overhead while maintaining high semantic fidelity, exceeding 0.9 at SNR levels above 10dB, for downstream vehicular tasks. We evaluate the proposed framework through simulations in autonomous driving scenarios, where the semantic encoder and decoder are integrated into a MIMO OFDM physical layer system. The results demonstrate high prediction success rates for risk assessment, improved robustness under the 3GPP CDL channel, and significant compression gains, confirming that the proposed semantic communication framework is a promising solution for future 6G systems. |
| 2026-03-09 | [IronEngine: Towards General AI Assistant](http://arxiv.org/abs/2603.08425v1) | Xi Mo | This paper presents IronEngine, a general AI assistant platform organized around a unified orchestration core that connects a desktop user interface, REST and WebSocket APIs, Python clients, local and cloud model backends, persistent memory, task scheduling, reusable skills, 24-category tool execution, MCP-compatible extensibility, and hardware-facing integration. IronEngine introduces a three-phase pipeline -- Discussion (Planner--Reviewer collaboration), Model Switch (VRAM-aware transition), and Execution (tool-augmented action loop) -- that separates planning quality from execution capability. The system features a hierarchical memory architecture with multi-level consolidation, a vectorized skill repository backed by ChromaDB, an adaptive model management layer supporting 92 model profiles with VRAM-aware context budgeting, and an intelligent tool routing system with 130+ alias normalization and automatic error correction. We present experimental results on file operation benchmarks achieving 100\% task completion with a mean total time of 1541 seconds across four heterogeneous tasks, and provide detailed comparisons with representative AI assistant systems including ChatGPT, Claude Desktop, Cursor, Windsurf, and open-source agent frameworks. Without disclosing proprietary prompts or core algorithms, this paper analyzes the platform's architectural decomposition, subsystem design, experimental performance, safety boundaries, and comparative engineering advantages. The resulting study positions IronEngine as a system-oriented foundation for general-purpose personal assistants, automation frameworks, and future human-centered agent platforms. |
| 2026-03-09 | [CORE-Acu: Structured Reasoning Traces and Knowledge Graph Safety Verification for Acupuncture Clinical Decision Support](http://arxiv.org/abs/2603.08321v1) | Liuyi Xu, Yun Guo et al. | Large language models (LLMs) show significant potential for clinical decision support (CDS), yet their black-box nature -- characterized by untraceable reasoning and probabilistic hallucinations -- poses severe challenges in acupuncture, a field demanding rigorous interpretability and safety. To address this, we propose CORE-Acu, a neuro-symbolic framework for acupuncture clinical decision support that integrates Structured Chain-of-Thought (S-CoT) with knowledge graph (KG) safety verification. First, we construct the first acupuncture Structured Reasoning Trace dataset and a schema-constrained fine-tuning framework. By enforcing an explicit causal chain from pattern identification to treatment principles, treatment plans, and acupoint selection, we transform implicit Traditional Chinese Medicine (TCM) reasoning into interpretable generation constraints, mitigating the opacity of LLM-based CDS. Furthermore, we construct a TCM safety knowledge graph and establish a ``Generate--Verify--Revise'' closed-loop inference system based on a Symbolic Veto Mechanism, employing deterministic rules to intercept hallucinations and enforce hard safety boundaries. Finally, we introduce the Lexicon-Matched Entity-Reweighted Loss (LMERL), which corrects terminology drift caused by the frequency--importance mismatch in general optimization by adaptively amplifying gradient contributions of high-risk entities during fine-tuning. Experiments on 1,000 held-out cases demonstrate CORE-Acu's superior entity fidelity and reasoning quality. Crucially, CORE-Acu achieved 0/1,000 observed safety violations (95\% CI: 0--0.37\%), whereas GPT-4o exhibited an 8.5\% violation rate under identical rules. These results establish CORE-Acu as a robust neuro-symbolic framework for acupuncture clinical decision support, guaranteeing both reasoning auditability and strict safety compliance. |
| 2026-03-09 | [AdaCultureSafe: Adaptive Cultural Safety Grounded by Cultural Knowledge in Large Language Models](http://arxiv.org/abs/2603.08275v1) | Hankun Kang, Di Lin et al. | With the widespread adoption of Large Language Models (LLMs), respecting indigenous cultures becomes essential for models' culturally safety and responsible global applications. Existing studies separately consider cultural safety and cultural knowledge and neglect that the former should be grounded by the latter. This severely prevents LLMs from yielding culture-specific respectful responses. Consequently, adaptive cultural safety remains a formidable task. In this work, we propose to jointly model cultural safety and knowledge. First and foremost, cultural-safety and knowledge-paired data serve as the key prerequisite to conduct this research. However, the cultural diversity across regions and the subtlety of cultural differences pose significant challenges to the creation of such paired evaluation data. To address this issue, we propose a novel framework that integrates authoritative cultural knowledge descriptions curation, LLM-automated query generation, and heavy manual verification. Accordingly, we obtain a dataset named AdaCultureSafe containing 4.8K manually decomposed fine-grained cultural descriptions and the corresponding 48K manually verified safety- and knowledge-oriented queries. Upon the constructed dataset, we evaluate three families of popular LLMs on their cultural safety and knowledge proficiency, via which we make a critical discovery: no significant correlation exists between their cultural safety and knowledge proficiency. We then delve into the utility-related neuron activations within LLMs to investigate the potential cause of the absence of correlation, which can be attributed to the difference of the objectives of pre-training and post-alignment. We finally present a knowledge-grounded method, which significantly enhances cultural safety by enforcing the integration of knowledge into the LLM response generation process. |
| 2026-03-09 | [The Struggle Between Continuation and Refusal: A Mechanistic Analysis of the Continuation-Triggered Jailbreak in LLMs](http://arxiv.org/abs/2603.08234v1) | Yonghong Deng, Zhen Yang et al. | With the rapid advancement of large language models (LLMs), the safety of LLMs has become a critical concern. Despite significant efforts in safety alignment, current LLMs remain vulnerable to jailbreaking attacks. However, the root causes of such vulnerabilities are still poorly understood, necessitating a rigorous investigation into jailbreak mechanisms across both academic and industrial communities. In this work, we focus on a continuation-triggered jailbreak phenomenon, whereby simply relocating a continuation-triggered instruction suffix can substantially increase jailbreak success rates. To uncover the intrinsic mechanisms of this phenomenon, we conduct a comprehensive mechanistic interpretability analysis at the level of attention heads. Through causal interventions and activation scaling, we show that this jailbreak behavior primarily arises from an inherent competition between the model's intrinsic continuation drive and the safety defenses acquired through alignment training. Furthermore, we perform a detailed behavioral analysis of the identified safety-critical attention heads, revealing notable differences in the functions and behaviors of safety heads across different model architectures. These findings provide a novel mechanistic perspective for understanding and interpreting jailbreak behaviors in LLMs, offering both theoretical insights and practical implications for improving model safety. |
| 2026-03-09 | [A Comparative Study of Recent Advances in Internet of Intrusion Detection Things](http://arxiv.org/abs/2603.08218v1) | Marianna Rezk, Hassan Harb et al. | The Internet of Things (IoT) has revolutionized the way devices communicate and interact with each other, but it has also created new challenges in terms of security. In this context, intrusion detection has become a crucial mechanism to ensure the safety of IoT systems. To address this issue, a comprehensive comparative study of advanced techniques and types of IoT intrusion detection systems (IDS) has been conducted. The study delves into various architectures, classifications, and evaluation methodologies of IoT IDS. This paper provides a valuable resource for researchers and practitioners interested in IoT security and intrusion detection. |
| 2026-03-09 | [ALOOD: Exploiting Language Representations for LiDAR-based Out-of-Distribution Object Detection](http://arxiv.org/abs/2603.08180v1) | Michael Kösel, Marcel Schreiber et al. | LiDAR-based 3D object detection plays a critical role for reliable and safe autonomous driving systems. However, existing detectors often produce overly confident predictions for objects not belonging to known categories, posing significant safety risks. This is caused by so-called out-of-distribution (OOD) objects, which were not part of the training data, resulting in incorrect predictions. To address this challenge, we propose ALOOD (Aligned LiDAR representations for Out-Of-Distribution Detection), a novel approach that incorporates language representations from a vision-language model (VLM). By aligning the object features from the object detector to the feature space of the VLM, we can treat the detection of OOD objects as a zero-shot classification task. We demonstrate competitive performance on the nuScenes OOD benchmark, establishing a novel approach to OOD object detection in LiDAR using language representations. The source code is available at https://github.com/uulm-mrm/mmood3d. |
| 2026-03-09 | [An explainable hybrid deep learning-enabled intelligent fault detection and diagnosis approach for automotive software systems validation](http://arxiv.org/abs/2603.08165v1) | Mohammad Abboush, Ehab Ghannoum et al. | Advancements in data-driven machine learning have emerged as a pivotal element in supporting automotive software systems (ASSs) engineering across various levels of the V-development process. Duringsystemverificationandvalidation,theintegrationofanintelligent fault detection anddiagnosis (FDD) model with test recordings analysis process serves as a powerful tool for efficiency ensuring functional safety. However, the lack of interpretability of the black-box FDD models developed not only hinders understanding of the cause underlying the prediction, but also prevents the model from being adapted based on the prediction result. This, in turn, increases the computational cost required for developingacomplexFDDmodelandlimitsconfidenceinreal-timesafety-criticalapplications.To address this challenge, a novel explainable method for fault detection, identification, and localization is proposed in this article with the aim of providing a clear understanding of the logic behind the prediction outcome. To this end, a hybrid 1dCNN-GRU-based intelligent model was developed to analyze the recordings from the real-time validation process of ASSs. The employment of explainable AI techniques, i.e., IGs, DeepLIFT, Gradient SHAP, and DeepLIFT SHAP, was instrumental in enabling model adaptation and facilitating the root cause analysis (RCA). The proposed approach is applied to the real time dataset collected during a virtual test drive performed by the user on hardware in the loop system. |
| 2026-03-09 | [Optimal Embedding of Wiring Diagrams in Constrained Three-Dimensional Spaces](http://arxiv.org/abs/2603.08157v1) | Víctor Blanco, Gabriel González et al. | This paper investigates the \emph{Wiring Diagram Problem} (WDP), a three-dimensional layout design problem arising in industrial applications such as cable harness design and pipeline routing in constrained environments. In these settings, hierarchical tree-like systems composed of supply units, intermediate devices (e.g., valves or junctions), and terminal components must be spatially arranged and interconnected while satisfying stringent engineering requirements, including safety separation distances, obstacle avoidance, geometric feasibility, and constructibility constraints.   We develop an optimization-based framework that formulates the WDP as a mixed-integer linear programming model capturing both topological and spatial design requirements within a unified formulation. To address the combinatorial and geometric complexity of three-dimensional routing, the feasible design space is discretized into structured network graphs that preserve engineering constraints while reducing dimensionality.   The resulting model minimizes total cable or pipeline length while ensuring compliance with all technical specifications. Computational experiments on representative industrial instances demonstrate the robustness and practical applicability of the proposed approach for automated layout generation. |
| 2026-03-09 | [Toward Governing Perception in Safety-Critical Mediated Reality on the Move](http://arxiv.org/abs/2603.08138v1) | Pascal Jansen | Wearable Augmented Reality (AR) is increasingly deployed in on-the-move contexts such as automated driving, cycling, and pedestrian navigation. To date, most systems rely on additive overlays that highlight hazards, intentions, or predictions without altering the scene itself. However, advances in head-mounted displays and computer vision now enable Diminished and Modified Reality techniques that suppress, transform, or substitute scene elements. These capabilities conceptually extend AR into Mediated Reality (MR), shifting the design space from "what to add" to "what is perceptually available." Because such mediation reshapes the evidential basis for situation awareness and trust calibration, it raises novel interaction challenges. This position paper argues that MR on the move must become governable, as users need mechanisms to configure, inspect, and understand mediation without compromising safety. Additionally, this position paper outlines design challenges related to governance granularity, epistemic signaling, and accountability, and frames MR on the move as a research agenda for governable perceptual mediation in dynamic, safety-critical environments. |
| 2026-03-09 | [Explainable Condition Monitoring via Probabilistic Anomaly Detection Applied to Helicopter Transmissions](http://arxiv.org/abs/2603.08130v1) | Aurelio Raffa Ugolini, Jessica Leoni et al. | We present a novel Explainable methodology for Condition Monitoring, relying on healthy data only. Since faults are rare events, we propose to focus on learning the probability distribution of healthy observations only, and detect Anomalies at runtime. This objective is achieved via the definition of probabilistic measures of deviation from nominality, which allow to detect and anticipate faults. The Bayesian perspective underpinning our approach allows us to perform Uncertainty Quantification to inform decisions. At the same time, we provide descriptive tools to enhance the interpretability of the results, supporting the deployment of the proposed strategy also in safety-critical applications. The methodology is validated experimentally on two use cases: a publicly available benchmark for Predictive Maintenance, and a real-world Helicopter Transmission dataset collected over multiple years. In both applications, the method achieves competitive detection performance with respect to state-of-the-art anomaly detection methods. |

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



