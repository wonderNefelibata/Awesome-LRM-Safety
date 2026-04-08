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
| 2026-04-07 | [Exclusive Unlearning](http://arxiv.org/abs/2604.06154v1) | Mutsumi Sasaki, Kouta Nakayama et al. | When introducing Large Language Models (LLMs) into industrial applications, such as healthcare and education, the risk of generating harmful content becomes a significant challenge. While existing machine unlearning methods can erase specific harmful knowledge and expressions, diverse harmful content makes comprehensive removal difficult. In this study, instead of individually listing targets for forgetting, we propose Exclusive Unlearning (EU), which aims for broad harm removal by extensively forgetting everything except for the knowledge and expressions we wish to retain. We demonstrate that through Exclusive Unlearning, it is possible to obtain a model that ensures safety against a wide range of inputs, including jailbreaks, while maintaining the ability to respond to diverse instructions related to specific domains such as medicine and mathematics. |
| 2026-04-07 | [Claw-Eval: Toward Trustworthy Evaluation of Autonomous Agents](http://arxiv.org/abs/2604.06132v1) | Bowen Ye, Rang Li et al. | Large language models are increasingly deployed as autonomous agents executing multi-step workflows in real-world software environments. However, existing agent benchmarks suffer from three critical limitations: (1) trajectory-opaque grading that checks only final outputs, (2) underspecified safety and robustness evaluation, and (3) narrow modality coverage and interaction paradigms. We introduce Claw-Eval, an end-to-end evaluation suite addressing all three gaps. It comprises 300 human-verified tasks spanning 9 categories across three groups (general service orchestration, multimodal perception and generation, and multi-turn professional dialogue). Every agent action is recorded through three independent evidence channels (execution traces, audit logs, and environment snapshots), enabling trajectory-aware grading over 2,159 fine-grained rubric items. The scoring protocol evaluates Completion, Safety, and Robustness, reporting Average Score, Pass@k, and Pass^k across three trials to distinguish genuine capability from lucky outcomes. Experiments on 14 frontier models reveal that: (1) trajectory-opaque evaluation is systematically unreliable, missing 44% of safety violations and 13% of robustness failures that our hybrid pipeline catches; (2) controlled error injection primarily degrades consistency rather than peak capability, with Pass^3 dropping up to 24% while Pass@3 remains stable; (3) multimodal performance varies sharply, with most models performing poorer on video than on document or image, and no single model dominating across all modalities. Beyond benchmarking, Claw-Eval highlights actionable directions for agent development, shedding light on what it takes to build agents that are not only capable but reliably deployable. |
| 2026-04-07 | [eVTOL Aircraft Energy Overhead Estimation under Conflict Resolution in High-Density Airspaces](http://arxiv.org/abs/2604.06093v1) | Alex Zongo, Peng Wei | Electric vertical takeoff and landing (eVTOL) aircraft operating in high-density urban airspace must maintain safe separation through tactical conflict resolution, yet the energy cost of such maneuvers has not been systematically quantified. This paper investigates how conflict-resolution maneuvers under the Modified Voltage Potential (MVP) algorithm affect eVTOL energy consumption. Using a physics-based power model integrated within a traffic simulation, we analyze approximately 71,767 en route sections within a sector, across traffic densities of 10-60 simultaneous aircraft. The main finding is that MVP-based deconfliction is energy-efficient: median energy overhead remains below 1.5% across all density levels, and the majority of en route flights within the sector incur negligible penalty. However, the distribution exhibits pronounced right-skewness, with tail cases reaching 44% overhead at the highest densities due to sustained multi-aircraft conflicts. The 95th percentile ranges from 3.84% to 5.3%, suggesting that a 4-5% reserve margin accommodates the vast majority of tactical deconfliction scenarios. To support operational planning, we develop a machine learning model that estimates energy overhead at mission initiation. Because conflict outcomes depend on future traffic interactions that cannot be known in advance, the model provides both point estimates and uncertainty bounds. These bounds are conservative; actual outcomes fall within the predicted range more often than the stated confidence level, making them suitable for safety-critical reserve planning. Together, these results validate MVP's suitability for energy-constrained eVTOL operations and provide quantitative guidance for reserve energy determination in Advanced Air Mobility. |
| 2026-04-07 | [Opportunistic Network-Level ISAC with Cooperative Sensing: A Meta-Distribution Analysis](http://arxiv.org/abs/2604.06069v1) | Yasser Nabil, Hesham ElSawy et al. | We propose a cooperative sensing framework for mmWave ISAC networks in which a target is sensed by its nearest BS while opportunistically exploiting bistatic echoes from neighboring BSs. Cooperation requires no dedicated resources or exchange of sensing results, and is realized via non-coherent echo-power combining. Using stochastic geometry, we characterize sensing/communication coverage and rates and, for the first time, the cooperative sensing meta-distribution to quantify reliability across targets. Results show substantial sensing gains with limited communication loss and improved high-reliability tail, increasing the fraction of targets meeting stringent reliability guarantees crucial for safety-critical applications. |
| 2026-04-07 | [Staggered Integral Online Conformal Prediction for Safe Dynamics Adaptation with Multi-Step Coverage Guarantees](http://arxiv.org/abs/2604.06058v1) | Daniel M. Cherenson, Dimitra Panagou | Safety-critical control of uncertain, adaptive systems often relies on conservative, worst-case uncertainty bounds that limit closed-loop performance. Online conformal prediction is a powerful data-driven method for quantifying uncertainty when truth values of predicted outputs are revealed online; however, for systems that adapt the dynamics without measurements of the state derivatives, standard online conformal prediction is insufficient to quantify the model uncertainty. We propose Staggered Integral Online Conformal Prediction (SI-OCP), an algorithm utilizing an integral score function to quantify the lumped effect of disturbance and learning error. This approach provides long-run coverage guarantees, resulting in long-run safety when synthesized with safety-critical controllers, including robust tube model predictive control. Finally, we validate the proposed approach through a numerical simulation of an all-layer deep neural network (DNN) adaptive quadcopter using robust tube MPC, highlighting the applicability of our method to complex learning parameterizations and control strategies. |
| 2026-04-07 | [Incremental Risk Assessment for Cascading Failures in Large-Scale Multi-Agent Systems](http://arxiv.org/abs/2604.06024v1) | Guangyi Liu, Vivek Pandey et al. | We develop a framework for studying and quantifying the risk of cascading failures in time-delay consensus networks, motivated by a team of agents attempting temporal rendezvous under stochastic disturbances and communication delays. To assess how failures at one or multiple agents amplify the risk of deviation across the network, we employ the Average Value-at-Risk as a systemic measure of cascading uncertainty. Closed-form expressions reveal explicit dependencies of the risk of cascading failure on the Laplacian spectrum, communication delay, and noise statistics. We further establish fundamental lower bounds that characterize the best-achievable network performance under time-delay constraints. These bounds serve as feasibility certificates for assessing whether a desired safety or performance goal can be achieved without exhaustive search across all possible topologies. In addition, we develop an efficient single-step update law that enables scalable propagation of conditional risk as new failures are detected. Analytical and numerical studies demonstrate significant computational savings and confirm the tightness of the theoretical limits across diverse network configurations. |
| 2026-04-07 | [Arch: An AI-Native Hardware Description Language for Register-Transfer Clocked Hardware Design](http://arxiv.org/abs/2604.05983v1) | Shuqing Zhao | We present Arch (AI-native Register-transfer Clocked Hardware), a hardware description language designed from first principles for micro-architecture specification and AI-assisted code generation. Arch introduces first-class language constructs for pipelines, FSMs, FIFOs, arbiters, register files, buses, and clock-domain crossings -- structures that existing HDLs express only as user-defined patterns prone to subtle errors.   A central design choice is that clocks and resets are themselves parameterized types (Clock<D>, Reset<S,P,D?>) rather than ordinary nets, converting clock-domain crossing (CDC) and reset-domain crossing (RDC) analysis from external linter passes into compile-time typing rules. Combined with simultaneous tracking of bit widths, port directions, single-driver ownership, and combinational acyclicity, the type system catches multiple drivers, undriven ports, implicit latches, width mismatches, combinational loops, and unsynchronized domain crossings before any simulator runs.   Every syntactic choice is governed by an AI-generatability contract: an LL(1) grammar requiring no backtracking or multi-token lookahead, no preprocessor or macros, a uniform declaration schema, named block endings, explicit directional connect arrows, and a todo! escape hatch enable LLMs to produce structurally correct, type-safe Arch from natural-language specifications without fine-tuning.   The Arch compiler emits deterministic, lint-clean IEEE 1800-2017 SystemVerilog and provides an integrated simulation toolchain that generates compiled C++ models for cycle-accurate simulation. We present case studies of an 8-way set-associative L1 data cache and a synthesizable PG021-compatible AXI DMA controller (with Yosys and OpenSTA results on Sky130), and compare Arch to SystemVerilog, VHDL, Chisel, Bluespec, and other modern HDLs across expressiveness, safety, and AI suitability dimensions. |
| 2026-04-07 | [Automating Manual Tasks through Intuitive Robot Programming and Cognitive Robotics](http://arxiv.org/abs/2604.05978v1) | Bijan Kavousian, Petar Tesic et al. | This paper presents a novel concept for intuitive end-user programming of robots, inspired by natural interaction between humans. Natural language and supportive gestures are translated into robot programs using large language models (LLMs) and computer vision (CV). Through equally natural system feedback in the form of clarification questions and visual representations, the generated program can be reviewed and adjusted, thereby ensuring safety, transparency, and user acceptance. |
| 2026-04-07 | [Dialogue based Interactive Explanations for Safety Decisions in Human Robot Collaboration](http://arxiv.org/abs/2604.05896v1) | Yifan Xu, Xiao Zhan et al. | As robots increasingly operate in shared, safety critical environments, acting safely is no longer sufficient robots must also make their safety decisions intelligible to human collaborators. In human robot collaboration (HRC), behaviours such as stopping or switching modes are often triggered by internal safety constraints that remain opaque to nearby workers. We present a dialogue based framework for interactive explanation of safety decisions in HRC. The approach tightly couples explanation with constraint based safety evaluation, grounding dialogue in the same state and constraint representations that govern behaviour selection. Explanations are derived directly from the recorded decision trace, enabling users to pose causal ("Why?"), contrastive ("Why not?"), and counterfactual ("What if?") queries about safety interventions. Counterfactual reasoning is evaluated in a bounded manner under fixed, certified safety parameters, ensuring that interactive exploration does not relax operational guarantees. We instantiate the framework in a construction robotics scenario and provide a structured operational trace illustrating how constraint aware dialogue clarifies safety interventions and supports coordinated task recovery. By treating explanation as an operational interface to safety control, this work advances a design perspective for interactive, safety aware autonomy in HRC. |
| 2026-04-07 | [Understanding Performance Gap Between Parallel and Sequential Sampling in Large Reasoning Models](http://arxiv.org/abs/2604.05868v1) | Xiangming Gu, Soham De et al. | Large Reasoning Models (LRMs) have shown remarkable performance on challenging questions, such as math and coding. However, to obtain a high quality solution, one may need to sample more than once. In principal, there are two sampling strategies that can be composed to form more complex processes: sequential sampling and parallel sampling. In this paper, we first compare these two approaches with rigor, and observe, aligned with previous works, that parallel sampling seems to outperform sequential sampling even though the latter should have more representation power. To understand the underline reasons, we make three hypothesis on the reason behind this behavior: (i) parallel sampling outperforms due to the aggregator operator; (ii) sequential sampling is harmed by needing to use longer contexts; (iii) sequential sampling leads to less exploration due to conditioning on previous answers. The empirical evidence on various model families and sizes (Qwen3, DeepSeek-R1 distilled models, Gemini 2.5) and question domains (math and coding) suggests that the aggregation and context length do not seem to be the main culprit behind the performance gap. In contrast, the lack of exploration seems to play a considerably larger role, and we argue that this is one main cause for the performance gap. |
| 2026-04-07 | [Reading Between the Pixels: An Inscriptive Jailbreak Attack on Text-to-Image Models](http://arxiv.org/abs/2604.05853v1) | Zonghao Ying, Haowen Dai et al. | Modern text-to-image (T2I) models can now render legible, paragraph-length text, enabling a fundamentally new class of misuse. We identify and formalize the inscriptive jailbreak, where an adversary coerces a T2I system into generating images containing harmful textual payloads (e.g., fraudulent documents) embedded within visually benign scenes. Unlike traditional depictive jailbreaks that elicit visually objectionable imagery, inscriptive attacks weaponize the text-rendering capability itself. Because existing jailbreak techniques are designed for coarse visual manipulation, they struggle to bypass multi-stage safety filters while maintaining character-level fidelity. To expose this vulnerability, we propose Etch, a black-box attack framework that decomposes the adversarial prompt into three functionally orthogonal layers: semantic camouflage, visual-spatial anchoring, and typographic encoding. This decomposition reduces joint optimization over the full prompt space to tractable sub-problems, which are iteratively refined through a zero-order loop. In this process, a vision-language model critiques each generated image, localizes failures to specific layers, and prescribes targeted revisions. Extensive evaluations across 7 models on the 2 benchmarks demonstrate that Etch achieves an average attack success rate of 65.57% (peaking at 91.00%), significantly outperforming existing baselines. Our results reveal a critical blind spot in current T2I safety alignments and underscore the urgent need for typography-aware defense multimodal mechanisms. |
| 2026-04-07 | [Learn to Rank: Visual Attribution by Learning Importance Ranking](http://arxiv.org/abs/2604.05819v1) | David Schinagl, Christian Fruhwirth-Reisinger et al. | Interpreting the decisions of complex computer vision models is crucial to establish trust and accountability, especially in safety-critical domains. An established approach to interpretability is generating visual attribution maps that highlight regions of the input most relevant to the model's prediction. However, existing methods face a three-way trade-off. Propagation-based approaches are efficient, but they can be biased and architecture-specific. Meanwhile, perturbation-based methods are causally grounded, yet they are expensive and for vision transformers often yield coarse, patch-level explanations. Learning-based explainers are fast but usually optimize surrogate objectives or distill from heuristic teachers. We propose a learning scheme that instead optimizes deletion and insertion metrics directly. Since these metrics depend on non-differentiable sorting and ranking, we frame them as permutation learning and replace the hard sorting with a differentiable relaxation using Gumbel-Sinkhorn. This enables end-to-end training through attribution-guided perturbations of the target model. During inference, our method produces dense, pixel-level attributions in a single forward pass with optional, few-step gradient refinement. Our experiments demonstrate consistent quantitative improvements and sharper, boundary-aligned explanations, particularly for transformer-based vision models. |
| 2026-04-07 | [From Points to Sets: Set-Based Safety Verification in the Latent Space](http://arxiv.org/abs/2604.05799v1) | Wenyuan Wu, Peng Xie et al. | We extend latent representation methods for safety control design to set-valued states. Recent work has shown that barrier functions designed in a learned latent space can transfer safety guarantees back to the original system, but these methods evaluate certificates at single state points, ignoring state uncertainty. A fixed safety margin can partially address this but cannot adapt to the anisotropic and time-varying nature of the uncertainty gap across different safety constraints. We instead represent the system state as a zonotope, propagate it through the encoder to obtain a latent zonotope, and evaluate certificates over the worst case of the entire set. On a 16-dimensional quadrotor suspended-load gate passage task, set-valued evaluation achieves 5/5 collision-free passages, compared to 1/5 for point-based evaluation and 2/5 for a fixed-margin baseline. Set evaluation reports safety in 44.4% of per-head evaluations versus 48.5% for point-based, and this greater conservatism detects 4.1% blind spots where point evaluation falsely certifies safety, enabling earlier corrective control. The safety gap between point and set evaluation varies up to $12\times$ across certificate heads, explaining why no single fixed margin suffices and confirming the need for per-head, per-timestep adaptation, which set evaluation provides by construction. |
| 2026-04-07 | [Near-Field Integrated Sensing, Computing and Semantic Communication in Digital Twin-Assisted Vehicular Networks](http://arxiv.org/abs/2604.05797v1) | Yinchao Yang, Yahao Ding et al. | Digital twin (DT) technology offers transformative potential for vehicular networks, enabling high-fidelity virtual representations for enhanced safety and automation. However, seamless DT synchronization in dynamic environments faces challenges such as massive data transmission, precision sensing, and strict computational constraints. This paper proposes an integrated sensing, computing, and semantic communication (ISCSC) framework tailored for DT-assisted vehicular networks in the near-field (NF) regime. Leveraging a multi-user multiple-input multiple-output (MU-MIMO) configuration, each roadside unit (RSU) employs semantic communication to serve vehicles while simultaneously utilizing millimeter-wave (mmWave) radar for environmental mapping. We implement particle filtering at RSUs to achieve high-precision vehicle tracking. To optimize performance, we formulate a joint optimization problem balancing semantic communication rates and sensing accuracy under limited computational resources and power budget. Our solution includes a hybrid heuristic algorithm for vehicle-to-RSU assignment and an alternating optimization approach for determining semantic extraction ratios and beamforming matrices. Performance is extensively evaluated via the Cramér-Rao bound (CRB) for angle and distance estimation, semantic transmission rates, and resource utilization. Numerical results demonstrate that the proposed ISCSC framework achieves a 20% improvement in transmission rate while maintaining the sensing accuracy of existing integrated sensing and communication (ISAC) schemes under constrained resource conditions. |
| 2026-04-07 | [Beyond the Beep: Scalable Collision Anticipation and Real-Time Explainability with BADAS-2.0](http://arxiv.org/abs/2604.05767v1) | Roni Goldshmidt, Hamish Scott et al. | We present BADAS-2.0, the second generation of our collision anticipation system, building on BADAS-1.0 [7], which showed that fine-tuning V-JEPA2 [1] on large-scale ego-centric dashcam data outperforms both academic baselines and production ADAS systems.   BADAS-2.0 advances the state of the art along three axes.   (i) Long-tail benchmark and accuracy: We introduce a 10-group long-tail benchmark targeting rare and safety-critical scenarios. To construct it, BADAS-1.0 is used as an active oracle to score millions of unlabeled drives and surface high-risk candidates for annotation. Combined with Nexar's Atlas platform [13] for targeted data collection, this expands the dataset from 40k to 178,500 labeled videos (~2M clips), yielding consistent gains across all subgroups, with the largest improvements on the hardest long-tail cases.   (ii) Knowledge distillation to edge: Domain-specific self-supervised pre-training on 2.25M unlabeled driving videos enables distillation into compact models, BADAS-2.0-Flash (86M) and BADAS-2.0-Flash-Lite (22M), achieving 7-12x speedup with near-parity accuracy, enabling real-time edge deployment.   (iii) Explainability: BADAS-2.0 produces real-time object-centric attention heatmaps that localize the evidence behind predictions. BADAS-Reason [17] extends this with a vision-language model that consumes the last frame and heatmap to generate driver actions and structured textual reasoning.   Inference code and evaluation benchmarks are publicly available. |
| 2026-04-07 | [Physics-Informed Neural Optimal Control for Precision Immobilization Technique in Emergency Scenarios](http://arxiv.org/abs/2604.05758v1) | Yangye Jiang, Jiachen Wang et al. | Precision Immobilization Technique (PIT) is a potentially effective intervention maneuver for emergency out-of-control vehicle, but its automation is challenged by highly nonlinear collision dynamics, strict safety constraints, and real-time computation requirements. This work presents a PIT-oriented neural optimal-control framework built around PicoPINN (Planning-Informed Compact Physics-Informed Neural Network), a compact physics-informed surrogate obtained through knowledge distillation, hierarchical parameter clustering, and relation-matrix-based parameter reconstruction. A hierarchical neural-OCP (Optimal Control Problem) architecture is then developed, in which an upper virtual decision layer generates PIT decision packages under scenario constraints and a lower coupled-MPC (Model Predictive Control) layer executes interaction-aware control. To evaluate the framework, we construct a PIT Scenario Dataset and conduct surrogate-model comparison, planning-structure ablation, and multi-fidelity assessment from simulation to scaled by-wire vehicle tests. In simulation, adding the upper planning layer improves PIT success rate from 63.8% to 76.7%, and PicoPINN reduces the original PINN parameter count from 8965 to 812 and achieves the smallest average heading error among the learned surrogates (0.112 rad). Scaled vehicle experiments are further used as evidence of control feasibility, with 3 of 4 low-speed controllable-contact PIT trials achieving successful yaw reversal. |
| 2026-04-07 | [Hazard Management in Robot-Assisted Mammography Support](http://arxiv.org/abs/2604.05749v1) | Ioannis Stefanakos, Roisin Bradley et al. | Robotic and embodied-AI systems have the potential to improve accessibility and quality of care in clinical settings, but their deployment in close physical contact with vulnerable patients introduces significant safety risks. This paper presents a hazard management methodology for MammoBot, an assistive robotic system designed to support patients during X-ray mammography. To ensure safety from early development stages, we combine stakeholder-guided process modelling with Software Hazard Analysis and Resolution in Design (SHARD) and System-Theoretic Process Analysis (STPA). The robot-assisted workflow is defined collaboratively with clinicians, roboticists, and patient representatives to capture key human-robot interactions. SHARD is applied to identify technical and procedural deviations, while STPA is used to analyse unsafe control actions arising from user interaction. The results show that many hazards arise not from component failures, but from timing mismatches, premature actions, and misinterpretation of system state. These hazards are translated into refined and additional safety requirements that constrain system behaviour and reduce reliance on correct human timing or interpretation alone. The work demonstrates a structured and traceable approach to safety-driven design with potential applicability to assistive robotic systems in clinical environments. |
| 2026-04-07 | [Understanding: reframing automation and assurance](http://arxiv.org/abs/2604.05662v1) | Robin Bloomfield | Safety and assurance cases risk becoming detached from the understanding needed for responsible engineering and governance decisions. More broadly, the production and evaluation of critical socio-technical systems increasingly face an understanding challenge: pressures for increased tempo, reduced scrutiny, software complexity, and growing use of AI generated artefacts may produce outputs that appear coherent without supporting genuine human comprehension. We argue that understanding should become an explicit, assessable, and defensible component of decision making: what developers, assessors, and decision makers grasp about system behavior, evidence, assumptions, risks, and residual uncertainty. Drawing on Catherine Elgin's epistemology of understanding, we outline a conceptual foundation and then use Assurance 2.0 as an engineering route to operationalize using structured argumentation, evidence, confidence, defeaters, and theory based automation. This leads to two linked artefacts: an Understanding Basis, which justifies why available understanding is sufficient for a decision, and a Personal Understanding Statement, through which participants make their grasp explicit and challengeable. We also identify risks that automation may improve artefact production while weakening understanding, and we propose initial directions for evaluating both efficacy and epistemic impact. |
| 2026-04-07 | [Uncovering Linguistic Fragility in Vision-Language-Action Models via Diversity-Aware Red Teaming](http://arxiv.org/abs/2604.05595v1) | Baoshun Tong, Haoran He et al. | Vision-Language-Action (VLA) models have achieved remarkable success in robotic manipulation. However, their robustness to linguistic nuances remains a critical, under-explored safety concern, posing a significant safety risk to real-world deployment. Red teaming, or identifying environmental scenarios that elicit catastrophic behaviors, is an important step in ensuring the safe deployment of embodied AI agents. Reinforcement learning (RL) has emerged as a promising approach in automated red teaming that aims to uncover these vulnerabilities. However, standard RL-based adversaries often suffer from severe mode collapse due to their reward-maximizing nature, which tends to converge to a narrow set of trivial or repetitive failure patterns, failing to reveal the comprehensive landscape of meaningful risks. To bridge this gap, we propose a novel \textbf{D}iversity-\textbf{A}ware \textbf{E}mbodied \textbf{R}ed \textbf{T}eaming (\textbf{DAERT}) framework, to expose the vulnerabilities of VLAs against linguistic variations. Our design is based on evaluating a uniform policy, which is able to generate a diverse set of challenging instructions while ensuring its attack effectiveness, measured by execution failures in a physical simulator. We conduct extensive experiments across different robotic benchmarks against two state-of-the-art VLAs, including $π_0$ and OpenVLA. Our method consistently discovers a wider range of more effective adversarial instructions that reduce the average task success rate from 93.33\% to 5.85\%, demonstrating a scalable approach to stress-testing VLA agents and exposing critical safety blind spots before real-world deployment. |
| 2026-04-07 | [CrowdVLA: Embodied Vision-Language-Action Agents for Context-Aware Crowd Simulation](http://arxiv.org/abs/2604.05525v1) | Juyeong Hwang, Seong-Eun Hong et al. | Crowds do not merely move; they decide. Human navigation is inherently contextual: people interpret the meaning of space, social norms, and potential consequences before acting. Sidewalks invite walking, crosswalks invite crossing, and deviations are weighed against urgency and safety. Yet most crowd simulation methods reduce navigation to geometry and collision avoidance, producing motion that is plausible but rarely intentional. We introduce CrowdVLA, a new formulation of crowd simulation that models each pedestrian as a Vision-Language-Action (VLA) agent. Instead of replaying recorded trajectories, CrowdVLA enables agents to interpret scene semantics and social norms from visual observations and language instructions, and to select actions through consequence-aware reasoning. CrowdVLA addresses three key challenges-limited agent-centric supervision in crowd datasets, unstable per-frame control, and success-biased datasets-through: (i) agent-centric visual supervision via semantically reconstructed environments and Low-Rank Adaptation (LoRA) fine-tuning of a pretrained vision-language model, (ii) a motion skill action space that bridges symbolic decision making and continuous locomotion, and (iii) exploration-based question answering that exposes agents to counterfactual actions and their outcomes through simulation rollouts. Our results shift crowd simulation from motion-centric synthesis toward perception-driven, consequence-aware decision making, enabling crowds that move not just realistically, but meaningfully. |

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



