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
| 2026-06-08 | [Deep learning reveals a stronger fossil fuel influence than biomass burning in shaping remote tropospheric ozone](http://arxiv.org/abs/2606.09793v1) | Chaoqun Ma, Hang Su et al. | Tropospheric ozone (O3) is a key greenhouse gas and atmospheric oxidant, yet its sources in the remote troposphere remain strongly debated. Observation-based tracer analyses suggest that O3 attributed to biomass burning is much greater than that from fossil fuel sources (by a factor of ~2-10), contradicting state-of-the-art global models. Here we show that this discrepancy primarily arises from the strong sensitivity of tracer methods to differences in tracer lifetimes, especially after extended transport to the remote regions. To resolve this discrepancy, we develop a deep learning (DL) framework that synthesizes global observations and chemical transport model simulations. The DL approach accurately infers source contributions and reveals that fossil fuel emissions contribute over three times more O3 to the remote troposphere than biomass burning. Our findings underscore that phasing out fossil fuels remains the most powerful lever for mitigating remote tropospheric ozone. |
| 2026-06-08 | [Who Earns the Safety? Intervention-Aware Quantum Predictive Control with Safety Attribution](http://arxiv.org/abs/2606.09778v1) | Yifan Wang | Hard safety filters are increasingly placed downstream of learned controllers to guarantee constraint satisfaction at run time. Yet a filtered controller that never violates a constraint may still have learned nothing about safety: the filter can silently repair an incompetent upstream policy, so that post-filter success measures the filter, not the policy. We argue that safe policy learning should ask who earns the safety - the policy or its protective layers - and we make this question measurable. We introduce Intervention-Aware Variational Quantum Differentiable Predictive Control (IA-VQC-DPC), which (i) trains a compact variational quantum circuit (VQC) policy under a primal-dual intervention budget that penalizes reliance on a differentiable Control-Barrier-Function (CBF) projection, and (ii) is evaluated with a safety-attribution protocol that decomposes the executed-trajectory correction into a CBF term and a deployment runtime-guard term, and stress-tests the policy with guard-off evaluation. On closed-loop, high-fidelity BOPTEST building-control emulators (5 seeds, 60 episodes per method), intervention-aware training significantly lowers the quantum policy's raw pre-filter violation and total safety-layer reliance (both p < 10^-4) with no significant energy regression; at an equal approximately 400-parameter budget the quantum policy is significantly safer and more comfortable than a matched classical policy. Guard-off evaluation confirms the improvement is policy-level and exposes a valuable negative result: a learned differentiable energy head is only safe when paired with a distribution-aware runtime guard. The attribution protocol is general beyond quantum policies and buildings. |
| 2026-06-08 | [Your Model Already Knows: Attention-Guided Safety Filter for Vision-Language-Action Models](http://arxiv.org/abs/2606.09749v1) | Seongbin Park, Fan Zhang et al. | Vision-Language-Action (VLA) models have demonstrated impressive end-to-end performance across a variety of robotic manipulation tasks. However, these policies offer no guarantees against collisions with task-irrelevant objects in the scene. Existing safety filters sidestep this problem by querying a vision-language model (VLM) to identify obstacles and their locations. This, however, is too slow to run in the control loop and can only be invoked at episode initialization, leaving the filter unable to track moving obstacles. We discover that a small number of attention heads within a VLA model reliably localize the object the policy intends to approach. These heads can be exploited within a training-free safety framework that obtains the active target from the attention heads at every step, treats the remainder of the scene as obstacles, and feeds these into a Control Barrier Function (CBF) filter. Together with a lightweight real-time object tracker, this allows for collision avoidance for non-static obstacles. We evaluate our framework on SafeLIBERO, which we extend with moving obstacles. On the original static benchmark, our method performs comparably to an oracle that uses privileged simulator state to identify the target, emulating a VLM-based identification step run once at episode initialization. On the dynamic variant, where the oracle's init-time target assignment becomes stale, our method substantially outperforms it by 43%, on average. Our findings suggest that the perceptual signals needed for real-time safety filtering are already present within VLA policies and can be exploited without additional training or heavy auxiliary models. |
| 2026-06-08 | [Hybrid Robustness Verification for Spatio-Temporal Neural Networks](http://arxiv.org/abs/2606.09746v1) | Sherwin Varghese, Matthew Wicker et al. | With AI increasingly deployed in safety-critical systems, providing formal robustness guarantees for the underlying models is essential. Existing verification methods either rely on overly conservative approximations or incur prohibitive computational costs. For example, the use of lp-norm perturbations in video settings encodes the belief that the adversary can inject noise in every video frame. In practice, adversarial perturbations exhibit structured spatial and temporal correlations, constrained to lower-dimensional, semantically meaningful subspaces. In this work, we study robustness verification of 3D CNNs processing video and volumetric inputs, targeting applications in action recognition (UCF-101), autonomous driving (Udacity), and medical imaging (MedMNIST) exploiting realistic assumptions on adversarial strength by modelling them as spatio-temporal constraints - where the attacker can modify either a subset of frames or patches within a set of consecutive frames. We demonstrate that modelling realistic constraints enables tighter approximations. We introduce Spatio-Temporal Bound Propagation (STBP), a verification framework that computes an exact closed-form characterization of the first convolutional layer and propagates certified bounds through subsequent layers using scalable approximations. Computing the exact closed form provides the tightest bounds for the first convolutional layer. Thus, we utilise approximation methods in the remainder of the network. To spur further progress in this field, we propose ST-Bench, a verification benchmark for autonomous driving and activity recognition, to systematically evaluate verifiable robustness. Compared to existing verification-based approaches, STBP provides stronger robustness guarantees with significantly improved scalability, achieving 1.7x higher certified robust accuracy under identical perturbation budgets. |
| 2026-06-08 | [ProbeAct: Probe-Guided Training-Free Failure Recovery in Vision-Language-Action Models](http://arxiv.org/abs/2606.09740v1) | Fan Zhang, Seongbin Park et al. | Vision-Language-Action (VLA) models demonstrate strong perfor-1 mance on language-conditioned robotic manipulation within their training dis-2 tribution, yet their generalization capabilities remain fundamentally limited. They3 lack the robustness required to handle perturbations, frequently failing when con-4 fronted with lighting changes, altered camera viewpoints, or small initial-state5 variations. We propose PROBEACT, a training-free runtime intervention frame-6 work that detects and recovers from grasping and placement failures in pre-7 trained VLA policies without modifying their weights or requiring additional8 demonstrations. PROBEACT combines three components: (i) a lightweight multi-9 target hidden-state probe that predicts the 3D positions of task-relevant objects10 from intermediate VLA features, with Hungarian-matched identity tracking for11 multi-object scenes; (ii) an object-agnostic kinematic state machine that detects12 grasp, transport, and placement failures using only gripper-internal signals and13 end-effector kinematics; and (iii) a hierarchical Control Barrier Function (CBF)14 filter that encodes repeated-failure locations as soft safe-set constraints, mini-15 mally correcting VLA actions while preserving baseline behavior. As a plug-and-16 play, training-free intervention loop, PROBEACT is orthogonal to existing train-17 ing pipelines. Evaluated on the LIBERO-plus benchmark, our framework acts as18 a universal safety net, improving the success rate of the OpenVLA-OFT model19 from 69.6% to 74.1%, while demonstrating broad applicability to both base and20 fine-tuned VLA policies. |
| 2026-06-08 | [Safe Polytope-in-Polytope Motion Planning and Control with Control Barrier Functions](http://arxiv.org/abs/2606.09719v1) | Alejandro Gonzalez-Garcia, Dries Dirckx et al. | Autonomous mobile robots operating in tight environments require motion planning frameworks that account for the physical footprint of the robot. Simplifying the geometry to a point or a circle is conservative and discards information needed to successfully and safely traverse narrow passages. This work proposes a safe local motion planning and control method that guarantees that a polytopic robot footprint stays inside a continuously updated convex free-space region. The containment condition is formulated as a set of discrete-time control barrier function constraints within a model predictive controller. The number of safety constraints depends on the complexity of the local free-space geometry and the robot shape, instead of the number of obstacles. The proposed free-space formulation does not need any obstacle detection or segmentation. A comparative analysis against a polytope-based obstacle avoidance formulation confirms favorable scaling up to a reduction of 91$\times$ in computation time as the number of obstacles increases. The approach is validated in simulation with an autonomous surface vehicle and on hardware with a non-holonomic mobile robot, using both occupancy grids and LiDAR sensing. The experiments demonstrate safe real-time motion planning and control at 10~Hz on an onboard embedded computer, including reactive avoidance of dynamic obstacles. |
| 2026-06-08 | [Learning to Attack and Defend: Adaptive Red Teaming of Language Models via GRPO](http://arxiv.org/abs/2606.09701v1) | Blake Bullwinkel, Eugenia Kim et al. | AI red teaming must continually adapt to evolving attackers and defenders. Reinforcement learning offers a promising approach to discovering novel attacks, and co-training methods can produce more robust defenders in tandem. Recent works have demonstrated the efficacy of attacker-defender co-training by applying PPO and DPO, but report that GRPO is unstable in this setting. We introduce AdvGRPO, a co-training framework that makes GRPO viable for joint attacker-defender optimization using dense multi-channel rewards and decoupled advantage normalization. Training progresses through a curriculum from single-turn to closed-loop multi-turn attacks before bootstrapping co-training, where attacker and defender models are updated in alternation. We show that our method can produce highly effective and transferable attacks and that co-trained defenders outperform baselines on safety benchmarks. |
| 2026-06-08 | [Gradient-Guided Reward Optimization for Inference-time Alignment](http://arxiv.org/abs/2606.09635v1) | Hankun Lin, Ruqi Zhang | Ensuring the reliability of Large Language Models (LLMs) under distribution drift requires inference-time adaptation. While inference-time alignment methods such as Best-of-$N$ and rejection sampling are widely used, they frame the task as a sampling-intensive, reward-guided search, leading to two key limitations: their performance is bounded by the base model's generation quality, and their reliance on imperfect reward models makes them vulnerable to reward hacking. To address these challenges, we introduce Gradient-Guided Reward Optimization (GGRO), a lightweight inference-time method that performs targeted, minimal intervention during decoding via gradient guidance. Specifically, GGRO monitors token-level entropy to identify high-uncertainty regions indicative of drift or misalignment. Upon detection, it responds by injecting nudging tokens, generated using gradient signals from an off-the-shelf reward model, to steer the generation trajectory rather than merely re-ranking samples. Experiments show that GGRO consistently improves inference-time alignment across safety, helpfulness, and reasoning benchmarks. It also increases coverage of high-quality responses and robustness to reward hacking, with minimal computational overhead. Code is available at https://github.com/lhk2004/GGRO. |
| 2026-06-08 | [Conceptual and Geometric Foundations for a Teleparallel Approach to Quantum Gravity](http://arxiv.org/abs/2606.09592v1) | Alexandre Landry | We revisit quantum field theory in curved spacetime (QFTCS) as a semi-classical framework for quantum matter on classical geometries, emphasizing its limitations, including vacuum ambiguity and background dependence. We briefly review major approaches to quantum gravity (QG), including Loop Quantum Gravity (LQG), string theory, and asymptotic safety, highlighting their conceptual challenges. Motivated by these issues, we outline a teleparallel framework based on coframe and spin-connection variables, where gravity is encoded in torsion rather than curvature. This framework naturally incorporates local Lorentz symmetry and fermionic couplings while displaying a gauge-like structure. We argue that the coframe/spin-connection pair provides an alternative and geometrically refined description of gravitational variables, which may serve as a useful starting point for future investigations of QG. The purpose of this work is not to provide a complete quantization of teleparallel gravity but to identify the geometric and conceptual ingredients that such a formulation would require. |
| 2026-06-08 | [Code Is More Than Text: Uncertainty Estimation for Code Generation](http://arxiv.org/abs/2606.09577v1) | Yuling Shi, Caiqi Zhang et al. | Large language models (LLMs) are increasingly deployed as code generators, where silently wrong programs pose real safety and reliability risks. Reliable uncertainty estimation (UE) is essential for selective prediction, human-in-the-loop review, and downstream agentic decisions. Yet most existing code UE methods are inherited from natural language (NL) generation and ignore properties that make code distinct. We argue that code differs from NL in three ways: a single wrong token can break an entire program (token fragility); algorithmic intent and concrete implementation can disagree independently (intent-code gap); and programs can be executed (executability). We instantiate these properties as three orthogonal uncertainty axes: lexical (Top-K token entropy), algorithmic (pseudo-code consistency), and functional (behavioral consistency). Across five code LLMs, our three-axis ensemble improves average AUROC from 0.696 for the strongest NL-derived baseline to 0.776 (+8.1 points). Notably, on Qwen3-14B, our single-pass Top-K token entropy matches the strongest multi-pass baseline while being over 3x cheaper; across models, it remains a competitive low-cost signal. These results suggest that code UE deserves code-specific design rather than direct NL ports. |
| 2026-06-08 | [Safe-RULE: Safe Reinforcement UnLEarning](http://arxiv.org/abs/2606.09559v1) | Shixiong Jiang, Taozheng Zhu et al. | Offline safe reinforcement learning (Safe RL) enables policy learning without online interactions, making it suitable for safety-critical systems such as robotics systems. However, its reliance on static datasets exposes offline Safe RL to data poisoning attacks, where adversaries inject malicious samples that compromise safety and induce unsafe policy behavior. In this work, we propose a new learning paradigm, named safe reinforcement unlearning (Safe-RULE), used as a defense framework to remove the influence of poisoned data without retraining from scratch or requiring access to the original training environment. We further extend reinforcement unlearning to offline Safe RL by explicitly accounting for both task performance and safety constraints during the unlearning process. Experiments across benchmark Safe RL tasks demonstrate that our approach effectively enhances safety performance against data poisoning attacks. |
| 2026-06-08 | [A VideoMAE-v2 Approach to Zero-Shot Traffic Accident Anticipation](http://arxiv.org/abs/2606.09542v1) | Siyuan Li, Xiaoyang Bi et al. | Traffic accident anticipation -- predicting the likelihood of an imminent collision at every frame of a dashcam video -- is safety-critical yet difficult to scale, because collecting in-domain annotated accident footage for every deployment scenario is prohibitively expensive. We study this task under a zero-shot setting where no target-domain training data is available: the model must learn exclusively from a publicly available binary-labelled driving-accident dataset and generalise to unseen dashcam footage. We propose a framework that bridges the gap between the frame-level temporal risk estimation task and coarsely labelled binary accident datasets by coupling a VideoMAE-v2 backbone with a per-frame prediction head under a sliding-window protocol. Our method achieves 2nd place in the 2026 CVPR@AUTOPILOT Zero-Shot Accident Anticipation competition. Code is available at https://github.com/TimeSouth/zero-shot-taa-solution. |
| 2026-06-08 | [Emergent alignment and the projectability of ethical personas](http://arxiv.org/abs/2606.09475v1) | Guillermo Del Pinal, Youngchan Lee et al. | Work on `emergent misalignment' shows that finetuning LLMs on narrow tasks can induce broadly misaligned behavior. This supports the `persona selection' (PSM) hypothesis: during pre-training, LLMs learn to simulate different characters and perspectives, which can be elicited and refined during post-training. This paper investigates the converse phenomenon, `emergent alignment', and uses it to support and refine the PSM and motivate a novel desideratum for alignment. We finetune a helpful-only model on broad and narrow safety tasks. To create SFT samples, we follow the `Constitutional AI' (CAI) approach and use four constitutions which encode reasonable alignment strategies: deontology, consequentialism, virtue ethics, and aligning AIs as subordinate to human authority. For each of those models, we show that finetuning on two narrow safety sub-categories reliably induces emergent alignment over a representative set of general safety categories, and on safety subcategories that we directly filtered-out of the data sets used for narrow alignment. To test the `PSM' using a more fine-grained evaluation, we used a multidimensional `ethical persona' diagnostic. For each constitutionally finetuned (broad/narrow) model, we evaluate how well their behavior matches their expected signature profile. Our results show that our CAI models acquire their expected ``ethical persona'' -- e.g., the model narrowly fine-tuned on SFT samples created using the consequentialist constitution agrees significantly more with utilitarian than deontological beliefs. Yet our coarse and fine-grained evaluations show that there are significant differences across our (broad/narrow) finetuned CAI models in how well they project. We conclude that alignment strategies should be evaluated, not just on their (in-distribution) general safety performance, but also specifically on their degree of projectability. |
| 2026-06-08 | [AI Assurance in UK Defence: Challenges in Operationalising JSP 936](http://arxiv.org/abs/2606.09414v1) | Callum Cockburn, Sam Farrow | This report examines practical challenges in operationalising JSP 936 Part 1 for AI assurance in UK Defence. Using a structured interpretive review of the directive's requirements, the analysis identifies eight thematic challenge areas adequacy of evidence and argument, management of human interaction with AI, definition of the operational environment, integration of AI within systems of systems, assessment and maintenance of AI performance, analysis of safety and security, measurement of ethicality, and mitigation of the inherent complexities of AI. The report argues that JSP 936 provides a useful governance basis, but that implementation depends on unresolved technical, organisational, and assurance questions. These challenges stem from the socio-technical nature of AI-enabled systems, uncertainty in real-world deployment contexts, limitations in current assurance methodologies, and tensions between performance, safety, human oversight, security, and ethical acceptability. The report identifies areas where further methods, guidance, and organisational capability are needed for the ambitious, safe, and responsible adoption of AI across Defence. This is consistent with MOD's own framing of JSP 936 as requiring iterative implementation and supporting guidance. |
| 2026-06-08 | [Towards Post-Quantum Secure Pharmacovigilance with ML-KEM and ML-DSA](http://arxiv.org/abs/2606.09412v1) | Saee Desai, Tom Shimoni et al. | Pharmacovigilance systems handle sensitive healthcare and drug-safety data, including adverse event reports and clinical observations. As quantum computing advances, classical public-key cryptographic systems such as RSA and elliptic-curve cryptography may become vulnerable, creating long-term risks for healthcare data that must remain confidential for many years. This paper presents an educational prototype of a post-quantum secure pharmacovigilance data pipeline. The system uses ML-KEM-768 for post-quantum key establishment, HKDF-SHA-256 for deriving an AES key, AES-256-GCM for efficient file encryption, and ML-DSA-65 for digital signatures and tamper detection. The pipeline supports multiple file formats, including TXT, CSV, JSON, and PDF, by treating files as raw bytes and preserving metadata for reconstruction at the receiver. The prototype includes separate hospital, gateway, pharma receiver, attacker, benchmarking, and dashboard components. We evaluate the system using synthetic pharmacovigilance datasets of different sizes and formats. Our results show that ML-KEM adds a small constant overhead, while AES encryption and ML-DSA signing dominate runtime as file size increases. This work is not a production-ready healthcare system, but rather an educational systems-level exploration of how post-quantum cryptographic primitives can be integrated into healthcare-style data pipelines. |
| 2026-06-08 | [Can Data Work be Reparative?](http://arxiv.org/abs/2606.09408v1) | Srravya Chandhiramowuli, Ding Wang et al. | We present an ethnographic study of an alternative approach to data work, developed by a civic-tech initiative that builds datasets for training and benchmarking online safety systems. They aim to respond to online safety concerns from a feminist perspective, by building safety datasets collaboratively with those most impacted by online harms. In this paper, we examine how this approach aims to reorient data work as a site for repair and redress, and trace the struggles they encounter in the process. Specifically, we draw attention to the challenges and tensions involved in advancing just reward for data work and collective governance of AI datasets. Examining these challenges through an STS-informed lens of reparative justice and repair, we argue that the work of repairing data work (and AI) lies, fundamentally, in resetting the ties of accountability. At a time heightened emphasis on efforts like safety evaluations and red teaming to make AI more responsible, we highlight the need to confront foundational questions about how the humans involved in these efforts relate to the datasets and systems they help produce. A reparative lens demands that we interrupt prevailing norms of data work and place at their centre, not AI or datasets, but those most harmed by the neglect, oversight and exclusion animated in the current modes of dataset production. This, we argue, offers a bold vision for responsibility and contributes towards a critical agenda for building alternative futures of data and AI practice. |
| 2026-06-08 | [Distilling Safe LLM Systems via Soft Prompts for On Device Settings](http://arxiv.org/abs/2606.09388v1) | Motasem Alfarra, Cristina Pinneri et al. | Deploying safe large language models (LLMs) on resource-constrained edge devices presents a critical challenge: while dual-model systems combining LLMs with guard models provide effective safety guarantees, their substantial memory and computational demands make them prohibitively expensive for on-device deployment. This paper presents a comprehensive study of parameter-efficient safety alignment methods for resource-constrained settings. Through systematic evaluation across multiple LLM architectures, training objectives, and parameter-efficient fine-tuning approaches, we identify that soft prompts combined with distillation-based training consistently outperform alternative methods. We introduce distillation frameworks based on total variation and KL divergence that effectively transfer safety behaviors from guard models into learned soft prompts. Our evaluations on various benchmarks demonstrate that this combination achieves superior safety-usefulness trade-offs compared to LoRA adapters, steering vectors, and direct optimization methods, while requiring minimal additional memory and compute at inference time. These findings establish soft prompt distillation as the preferred approach for safety alignment in on-device LLM deployment. |
| 2026-06-08 | [Scaling Neural Network Verification with Tensor Parallelism and Fully Sharded Data Parallelism](http://arxiv.org/abs/2606.09377v1) | Sergei Vorobyov, Eugene Ilyushin | Formal neural network verification -- proving that a network satisfies safety properties for \emph{all} inputs in a specified domain -- is bounded in practice by GPU memory: standard implementations of bound-propagation algorithms (IBP, CROWN, $α$-CROWN) require weight and relaxation-coefficient matrices to reside entirely on one accelerator. We adapt two parallelism techniques originally developed for large-scale model training to the \texttt{auto\_LiRPA}\,/\,$α,β$-CROWN verification framework. \textbf{Tensor Parallelism (TP)} shards both weight and $A$-matrices across GPUs, achieving ${\approx}2\times$ peak-memory reduction at $P{=}2$; soundness is confirmed on VNN-COMP 2022 MNIST-FC benchmarks, though bound tightness degrades with the number of sharded zones due to forced IBP substitution for intermediate bounds inside sharded zones. \textbf{Fully Sharded Data Parallelism (FSDP)} shards only weight matrices with a per-layer \texttt{AllGather}, producing bounds that are \emph{bitwise identical} to the single-GPU baseline: baseline memory drops by 80--90\%, peak memory by 34--39\% on wide MLPs. FSDP integrates cleanly with complete verification ($β$-CROWN + Branch-and-Bound) and with convolutional layers (\texttt{BoundConv}); a complete \emph{unsat} result is obtained for CIFAR-100 ResNet-large (VNN-COMP 2024) under FSDP. Across all experiments the memory bottleneck in $α$-CROWN+BaB mode proves to be per-neuron alpha tensors, not weight matrices, pointing to the key direction for future work. |
| 2026-06-08 | [Brain-Prompt Injection: A Route-Safety Audit for BCI-LLM Agents](http://arxiv.org/abs/2606.09315v1) | Jianwei Tai | BCI-to-agent pipelines turn decoded neural activity into an authorization channel for tool-use agents, exposing a new attack surface we call \emph{brain-prompt injection}: signal-side perturbations, context-only injections, and adaptive dual-decoder attacks can all change the routed action while EEG-side or text-side monitors remain blind. Route safety in this stack depends on what the audit log can observe, not on decoder accuracy or agreement alone. We define a Route-Safety Audit Contract: a minimal log schema, denominator hierarchy, and endpoint specification, and prove an audit-schema separation theorem together with a C3 attacked-dependence decomposition; clean agreement and marginal robustness do not identify the joint term that controls C3 routing. As a calibration layer on top of the contract, we apply split-conformal calibration to a non-oracle EEG confirmation channel and report the resulting false-accept frontier under an explicit threat-archetype matrix. We instantiate the contract on EEGMMI native left/right command-control over 5{,}400 events, harmless tool stubs, and seed/case denominators. Provenance blocks C2 routes ($0.000$); agreement-plus-provenance routes C3 flips ($1.000$); confirmation-plus-provenance routes them ($0.000$). The conformal frontier reaches FAR $0.000$ at clean utility $0.150$ for $α=.005$ and FAR $0.119$ at clean utility $0.452$ for $α=.10$ under acquisition isolation; an attacker-controllable confirmation channel breaks the bound to $\approx\!1$. Subject-cluster bootstrap confirms these intervals on $60$ subjects; cross-architecture (TinyEEGNet, EEGNetV4) and capacity-sweep results show within-regime saturation. Mediation and confirmation reduce risk; they are not intent certificates. |
| 2026-06-08 | [RPO-PDT: Demonstrating Role-Play-Based Knowledge Adaptation for Student Support Dialogue (Demonstration System)](http://arxiv.org/abs/2606.09255v1) | Filip Janik, Ewa Olton et al. | We present RPO-PDT: a retrieval-grounded, role-play-based dialogue system for adaptive student support in higher education. RPO-PDT is: (1) able to provide institution-specific Personal Development Tutor (PDT) guidance using structured knowledge sources; (2) constrained by explicit persona, boundary, confidentiality, and safety policies; and (3) designed around a reverse-roleplay loop where unresolved interactions are replayed from the student perspective, enabling alternative tutor strategies to be generated and stored as reusable strategy memory. RPO-PDT supports both text-based and Furhat-based embodied interaction for demonstrating grounded, safe, and adaptive student-support dialogue. |

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



