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
| 2026-05-19 | [From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models](http://arxiv.org/abs/2605.20177v1) | Juncheng Wu, Hardy Chen et al. | Recent advances in vision-language models (VLMs) emphasize long chain-of-thought reasoning; yet, we find that their performance on visual tasks is primarily limited by a lack of visual perception as opposed to reasoning itself. In this work, we systematically study the interplay between perception and reasoning in VLM post-training by decomposing their capabilities into three separate training stages: visual perception, visual reasoning, and textual reasoning, incorporating specialized training data. We demonstrate that visual perception (a) requires targeted optimization with specialized data; (b) serves as a fundamental scaffold that should be solidified through staged training before refining visual reasoning; and (c) is more effectively learned via RL than caption-based SFT. Our experiments across multiple VLMs demonstrate that staged training consistently improves both visual perception and reasoning performance over merged training. Notably, models trained with our approach achieve 1.5% higher reasoning accuracy with 20.8% shorter reasoning traces, suggesting that superior perception reduces the need for excessive reasoning. Furthermore, we show that this capability-based staging represents a new curriculum dimension orthogonal to traditional difficulty-based curricula, and combining both yields further additive gains. Our staged-training models achieve superior performance among open-weight VLMs, establishing advanced results on several visual math and perception (e.g., +5.2% on WeMath and +3.7% on RealWorldQA) tasks compared with the base counterpart. |
| 2026-05-19 | [A Unified Framework for Attack-Resilient CLF-CBF Quadratic Programs for Nonlinear Control-Affine Systems](http://arxiv.org/abs/2605.20144v1) | Mohamadamin Rajabinezhad, Shan Zuo | This letter introduces attack-resilient Control Lyapunov Functions (AR-CLFs) and attack-resilient Control Barrier Functions (AR-CBFs) for nonlinear control-affine systems subject to control-input false data injection attacks (FDIA) satisfying an at-most-exponentially growing envelope. The proposed framework embeds a unified adaptive compensation term into both the CLF decrease and CBF safety constraints. In contrast to input-to-state stability/safety (ISS/ISSf)-based methods that certify disturbance-dependent enlarged safe sets, the proposed approach enables finite-time recovery to the nominal safe set without requiring a prior magnitude bound on the FDIA, relying instead on a growth-rate characterization used for analysis and an online gain tuning law that regulates the compensation term. A unified quadratic program (QP) is developed to enforce the AR-CLF and AR-CBF conditions simultaneously, guaranteeing uniformly ultimately bounded (UUB) stability and uniform ultimate safety (UUS) under unbounded FDIA. Numerical results demonstrate improved resilience compared to existing ISS-CLF, ISSf-CBF, and robust CLF-CBF-QP approaches. |
| 2026-05-19 | [Hamilton--Jacobi Reachability for Spacecraft Collision Avoidance](http://arxiv.org/abs/2605.20138v1) | Larry Hui, Jordan Kam et al. | This article presents a Hamilton--Jacobi (HJ) reachability framework for a two--satellite collision avoidance problem operating in the same circular orbit, where relative motion is modeled in the radial--tangential--normal (RTN) frame using planar Hill--Clohessy--Wiltshire (HCW) dynamics. We define the target state space as unsafe relative configurations in the orbit plane corresponding to minimum separation requirements consistent with Federal Communications Commission (FCC) orbital standards. The interaction between spacecraft is formulated as a zero--sum differential game, where Player 1 is the controlled satellite and Player 2 is modeled as a bounded adversarial disturbance with unknown intent. We present the HJ formulation and compute backward reachable sets that characterize relative states from which collision cannot be avoided under worst-case disturbances, while states outside this set admit provably collision-free trajectories. These reachable sets are integrated with supervisory hybrid control logic to determine when evasive maneuvers must be initiated, enabling mathematically grounded safety guarantees for scalability. |
| 2026-05-19 | [k-Inductive Neural Barrier Certificates for Unknown Nonlinear Dynamics](http://arxiv.org/abs/2605.20108v1) | Ben Wooding, Hongchao Zhang et al. | While conventional (k=1) discrete-time barrier certificate conditions impose strict safety constraints by requiring the function to be non-increasing at every step, k-inductive barrier certificates relax this by allowing a temporary increase -- up to k-1 times, each within a threshold $ε$ -- while maintaining overall safety, and improving flexibility. This paper leverages neural networks and constructs k-inductive neural barrier certificates (k-NBCs) for (partially) unknown nonlinear systems. While neural networks offer scalability in the design process, they lack formal guarantees, requiring additional approaches such as counterexample-guided inductive synthesis (CEGIS) with satisfiability modulo theories (SMT) for verification. However, the CEGIS-SMT framework requires knowledge of system dynamics, which is unavailable in practical settings. To address this, we leverage the generalization of the Willems et al.'s fundamental lemma, using a single state trajectory, to construct a data-driven representation of (partially) unknown models for SMT verification without sacrificing accuracy. Additionally, CEGIS-SMT further removes the constraint of restricting barrier certificates to specific function classes, such as sum-of-squares, enabling greater flexibility in their design. We validate our approach on three nonlinear case studies with (partially) unknown dynamics. |
| 2026-05-19 | [Reliable sampling-based RKHS norm estimation via superconvergence](http://arxiv.org/abs/2605.20091v1) | Tizian Wenzel, Abdullah Tokmak et al. | Kernel methods are one of the cornerstones of learning-based control, modern system identification, surrogate modelling, and related fields. A key advantage of this class of learning and function approximation methods is the availability of quantitative error bounds, which in turn play a key role in guaranteeing the safety of learned controllers and related learning-based algorithms. However, these error bounds rely on a particular property of the target function -- its reproducing kernel Hilbert space (RKHS) norm -- which is usually impossible to obtain in practice. Motivated by this severe shortcoming, we present a novel sampling-based RKHS norm estimation approach with a solid theoretical foundation, leveraging very recent advances in the theory of superconvergence in kernel methods. Our method is applicable to a broad range of practically relevant function classes and requires only reasonable prior knowledge about the target function. Extensive numerical experiments demonstrate the efficacy and practical applicability of the proposed method. By providing a reliable RKHS norm estimation approach, we remove a major obstacle to the practical deployment of learning-based control algorithms. |
| 2026-05-19 | [Safe Deep Reinforcement Learning for Spacecraft Reorientation with Pointing Keep-Out Constraint](http://arxiv.org/abs/2605.19967v1) | Juntang Yang, Mohamed Khalil Ben-Larbi | This paper implements deep reinforcement learning (DRL) with a safety filter for spacecraft reorientation control with a single pointing keep-out zone. A new state space representation is designed which includes a compact representation of the attitude constraint zone. A reward function is formulated to achieve the control objective while enforcing the attitude constraint. The soft actor-critic (SAC) algorithm is adopted to handle continuous state and action space. A curriculum learning approach is implemented for agent training. To guarantee the compliance of the attitude constraint, a control barrier function (CBF)-based safety filter is implemented for agent deployment. Simulation results demonstrate the effectiveness of the proposed state space presentation and the designed reward function. Monte Carlo simulations underscore that reward shaping alone cannot guarantee the safety during reorientation maneuver. In contrast, with the CBF-based safety filter, the constraint can be guaranteed during maneuvers. |
| 2026-05-19 | [Detecting Fluent Optimization-Based Adversarial Prompts via Sequential Entropy Changes](http://arxiv.org/abs/2605.19966v1) | Mohammed Alshaalan, Miguel R. D. Rodrigues | Optimization-based adversarial suffixes can jailbreak aligned large language models (LLMs) while remaining fluent, weakening static and windowed perplexity-based detectors. We cast adversarial suffix detection as an online change-point detection problem over the token-level next-token entropy stream. Using the LLM system prompt to estimate a robust baseline, we standardize user-token entropies and apply a one-sided CUSUM statistic. The resulting detector, CPD Online (CPD), is model-agnostic, training-free, runs online, and localizes the adversarial suffix onset. On a benchmark of 1,012 optimization-based suffix attacks (GCG, AutoDAN, AdvPrompter, BEAST, AutoDAN-HGA) and 1,012 perplexity-controlled benign prompts, CPD improves F1 over the strongest windowed-perplexity baseline on all six open-weight chat models (LLaMA-2-7B/13B, Vicuna-7B/13B, Qwen2.5-7B/14B). On LLaMA-2-7B at the canonical CUSUM setting ($k=0$), CPD reaches AUROC $0.88$ and F1 $0.82$. Beyond prompt-level detection, CPD concentrates 79.6% of its triggers inside the adversarial suffix, versus 17-46% for windowed perplexity. Finally, when used as a lightweight gate for LLaMA Guard, CPD reduces guard calls by 17-22% on a high-volume, benign-dominated deployment while preserving guard-level detection quality |
| 2026-05-19 | [Robotics-Inspired Guardrails for Foundation Models in Socially Sensitive Domains](http://arxiv.org/abs/2605.19940v1) | Rebecca Ramnauth, Drazen Brscic et al. | Foundation models are increasingly deployed in socially sensitive domains such as education, mental health, and caregiving, where failures are often cumulative and context-dependent. Existing guardrail approaches -- ranging from training-time alignment to prompting, decoding constraints, and post-hoc moderation -- primarily provide empirical risk reduction rather than enforceable behavioral guarantees, and largely treat safety as a property of individual outputs rather than interaction trajectories. We reframe guardrails as a problem of runtime behavioral control over interaction trajectories, drawing on robotics to introduce formal constructs for constraint enforcement in uncertain, closed-loop systems. We instantiate these ideas in the Grounded Observer framework and apply it across three real-world deployments: small talk, in-home autism therapy, and behavioral de-escalation in schools. Across settings, the framework enables runtime interventions that mitigate drift into undesirable interaction regimes while adapting to diverse social contexts. We discuss extensions to the framework and propose research directions toward stronger guarantees. |
| 2026-05-19 | [Passive Construction Site Safety Monitoring via Persona-Scaffolded Adversarial Chain-of-Thought VLM Verification](http://arxiv.org/abs/2605.19869v1) | Ananth Sriram, Neel Mokaria et al. | Construction remains the deadliest industry sector in the United States, with 1,055 fatal worker injuries recorded in 2023, and the majority preventable. Existing monitoring approaches are expensive, require real-time human operators, or address only a narrow subset of violations. This paper presents a passive, end-of-shift construction safety monitoring pipeline processing video from POV body-worn and fixed wall-mounted cameras through a three-stage architecture: (1) fine-tuned YOLO11 for primary PPE and hazard detection, (2) SAM 3 for segmentation refinement and worker deduplication, and (3) Qwen3-VL-8B-Instruct with a method-prompted, persona-scaffolded three-pass adversarial chain-of-thought protocol for compliance verification and hallucination control. The principal contribution is the Stage 3 prompt design: professional persona backstories following the method-actor framing drive an observed 12% precision improvement over single-pass prompting in an informal three-author review of the 12-video Ironsite development corpus, with the largest gains on hallucination-prone violation categories. Structural message isolation enforces observational independence between a generator, discriminator, and reconciliation pass governed by asymmetric rules encoding priors about human observation versus automated detection reliability. The system maps violations to OSHA standards, performs REBA-inspired ergonomic risk scoring from pose keypoints, and produces per-worker safety reports with timestamped evidence. An evaluation harness is released for future reproduction. |
| 2026-05-19 | [A new open-shell CCSDTQ implementation and its application to the basis set convergence of post-CCSDT(Q) corrections in computational thermochemistry](http://arxiv.org/abs/2605.19860v1) | Aditya Barman, Gregory H. Jones et al. | We extend the CCSDTQ implementation in CFOUR to UHF and ROHF references and demonstrate its efficiency. We apply it to basis set convergence of post-CCSDT(Q) corrections for the W4-08 thermochemical dataset. Convergence of (Q)$_Λ$--(Q) is relatively rapid. For difficult species (e.g., B2, O3), CCSDTQ--CCSDT(Q)$_Λ$ may converge more slowly than (5)$_Λ$, but the effects and and basis-set trends oppose each other. Consequently, a single-shot CCCSDTQ(5)$_Λ$-CCSDT(Q)$_Λ$ correction appears most efficient. For radicals with bifurcating UHF solutions, energetics of the `less spin-contaminated' solution are clearly more well-behaved. Our best computed adiabatic electron affinity of ozone is in excellent agreement with experiment. |
| 2026-05-19 | [CADENet: Condition-Adaptive Asynchronous Dual-Stream Enhancement Network for Adverse Weather Perception in Autonomous Driving](http://arxiv.org/abs/2605.19837v1) | Sherif Khairy, Catherine M. Elias | Adverse weather (rain, fog, sand, and snow) degrades camera-based object detection in autonomous vehicles. Existing enhancement-then-detect approaches stall the safety-critical perception loop, violating hard real-time requirements. Progress on this problem is also constrained by an under-recognized evaluation ceiling: ground truth annotated on degraded images cannot credit a detector that recovers objects the annotators themselves could not see, so a genuinely useful enhancement can register as a near-flat F1 gain. This paper presents CADENet (Condition-Adaptive Asynchronous Dual-stream Enhancement Network), a training-free three-thread system: Thread S (YOLOv11n) delivers detections at full frame rate with zero added latency; Thread Q applies condition-adaptive enhancement (CAPE) and fuses results via entropy-guided NMS (EG-NMS) without blocking Thread S; Thread E provides CLIP zero-shot weather classification, so new weather categories require only a new text prompt, with no labeled data and no retraining. Evaluated on 1327 DAWN images (YOLOv11m, IoU = 0.5, confidence = 0.25), CADENet achieves Recall = 0.0103 (micro), F1 = 0.0230 on snow, and F1 = 0.0038 on rain. We formalize the annotation completeness bias on DAWN-class data, so the reported F1 values are lower bounds on the true gain; recall is the annotation-gap-immune headline metric. Thread S sustains approximately 44 FPS regardless of enhancement load. No model retraining or additional sensor hardware is required. |
| 2026-05-19 | [Explainable Wastewater Digital Twins: Adaptive Context-Conditioned Structured Simulators with Self-Falsifying Decision Support](http://arxiv.org/abs/2605.19826v1) | Gary Simethy, Daniel Ortiz Arroyo et al. | Operators of safety-critical industrial processes increasingly rely on digital twins to screen control interventions, but such simulators rarely carry certified safety guarantees. Wastewater treatment plants exemplify the gap: operators face a daily safety-efficiency trade-off where aerating too little risks effluent violations and nitrous-oxide (N2O) spikes, and aerating too much wastes energy. We develop an explainable digital twin for aeration and dosing setpoints. CCSS-IX, the simulator, is a bank of interpretable locally linear state-space "experts" adaptively mixed by a context-aware gating network, building on a continuous-time regime-switching scaffold. A runtime decision layer applies conformal risk control to abstain, reopen, or return a falsifying temporal witness for any operator-proposed action that cannot be statistically certified. The artificial-intelligence contribution is twofold: an identifiable, context-conditioned structured surrogate that retains operator-readable dynamics, and a self-falsifying decision rule with finite-sample coverage guarantees. The engineering contribution is a validated, end-to-end decision-support pipeline, tested on a 1000-step slice of the Avedøre full-scale plant (42.6% sensor missingness, 2-minute sampling), the Agtrup/BlueKolding full-scale plant in Denmark, and the Benchmark Simulation Model No. 2 (BSM2) international benchmark, under a matched ten-seed protocol. The static structured ensemble lies within 0.78% root-mean-square error of an unconstrained black-box reference, and the adaptive variant within 1.08%. The calibrated reopen rule cuts aggregate two-plant regret by 43.6% at an unsafe-action cost weight of 4 and eliminates unsafe chosen actions on the BSM2 main slice. Event-aligned temporal witnesses prevent 93 of 187 false-safe N2O approvals, about 4.65x the dyadic baseline (paired McNemar p < 1e-21). |
| 2026-05-19 | [From Prompts to Pavement Through Time: Temporal Grounding in Agentic Scene-to-Plan Reasoning](http://arxiv.org/abs/2605.19824v1) | Ahmed Y. Gado, Omar Y. Goba et al. | Recent attempts to support high-level scene interpretation and planning in Autonomous Vehicles (AVs) using ensembles of Large Language Models (LLMs) and Large Multimodal Models (LMMs) continue to treat time as a secondary property. This lack of temporal grounding leads to inconsistencies in reasoning about continuous actions, undermining both safety and interpretability. This work explores whether temporal conditioning within inter-agent communication can preserve or enhance coherence without introducing degradation in semantic or logical consistency. To investigate this, we introduce three planner architectures with progressively increasing temporal integration and evaluate them on curated subsets of the BDD-X dataset using semantic, syntactic, and logical metrics. Results show that while temporal conditioning reshapes reasoning style, it yields no statistically significant improvements in standard NLP-based correctness metrics. However, qualitative analysis reveals predictive hazard reasoning, stable corrective behavior, and strategic divergence in the Sentinel. These findings clarify the limits of prompt-based temporal grounding and establish the first empirical benchmark for temporal scene-to-plan reasoning. |
| 2026-05-19 | [Understanding Inference Scaling for LLMs: Bottlenecks, Trade-offs, and Performance Principles](http://arxiv.org/abs/2605.19775v1) | Moiz Arif, Avinash Maurya et al. | The transition from standard generative AI to \emph{reasoning-centric architectures}, exemplified by models capable of extensive Chain-of-Thought~(CoT) processing, marks a fundamental paradigm shift in system requirements. Unlike traditional workloads dominated by compute-bound prefill, reasoning workloads generate long chains of reasoning tokens that shift inference into a \emph{Capacity-Bound regime}. This paper presents a comprehensive system characterization, evaluating models ranging from 8B to 671B parameters on GPUs clusters. By systematically exploring the interplay between Data, Tensor, and Pipeline parallelism, we identify critical bottlenecks that defy standard scaling heuristics. Our analysis reveals that data parallelism is throughput efficient for small models but hits a capacity trap on reasoning workloads as KV-cache fragmentation forces early throttling resulting in sub-optimal compute utilization. Tensor parallelism unlocks stranded memory and delivers sublinear gains near the 32B crossover. At frontier scale, dense models (e.g., Llama-405B) are interconnect and memory-bandwidth bound and favor high-degree TP, while sparse Mixture-of-Experts (MoE) models (e.g., DeepSeek-R1) are limited by routing and synchronization latency and benefit from hybrid strategies. These insights provide a rigorous decision framework for navigating the reasoning cliff, establishing new architectural imperatives for the next generation of inference infrastructure. |
| 2026-05-19 | [Beyond Imitation: Learning Safe End-to-End Autonomous Driving from Hard Negatives](http://arxiv.org/abs/2605.19771v1) | Junli Wang, Zhihua Hua et al. | Existing imitation learning methods for end-to-end autonomous driving predominantly learn from successful demonstrations by minimizing geometric deviations from expert trajectories. This paradigm implicitly assumes that spatial proximity implies behavioral safety, leading to a critical objective mismatch: trajectories with nearly identical imitation losses may exhibit drastically different safety outcomes, where one remains recoverable while the other results in collision. To address this limitation, we propose BeyondDrive, a failure-aware imitation learning framework that jointly learns from successful and failed driving behaviors. First, we introduce a flow matching-based negative trajectory generator that synthesizes safety-critical yet expert-proximate trajectories, enabling explicit modeling of safety asymmetry. Second, we develop a diversity-aware sampling strategy that mitigates mode collapse and improves coverage of diverse failure modes during negative trajectory generation. Third, we propose a Repulsive Distance Loss that simultaneously attracts predictions toward expert demonstrations while repelling them from hard negative trajectories, thereby establishing discriminative safety boundaries in trajectory space. Applied to the uni-modal baseline Latent TransFuser, BeyondDrive achieves 89.7 PDMS on the NAVSIMv1 closed-loop benchmark, outperforming prior state-of-the-art methods. Moreover, BeyondDrive generalizes effectively across different autonomous driving architectures, including multi-modal planners, and further demonstrates strong zero-shot transferability on the HUGSIM benchmark. |
| 2026-05-19 | [Real-World On-Vehicle Evaluation of Embedding-Based Anomaly Detection](http://arxiv.org/abs/2605.19744v1) | Albert Schotschneider, Daniel Bogdoll et al. | Detecting anomalies in traffic scenes is crucial for ensuring safety in autonomous driving, yet collecting representative anomalous data remains challenging. Existing anomaly detection methods are highly specialized and rely on normality as defined by the abstract semantic Cityscapes classes, making it difficult to adapt to diverse real-world scenarios. We propose an adaptable real-time anomaly detection method that leverages foundation models in the form of pretrained vision transformer embeddings to detect deviations via nearest-neighbor similarity in the latent semantic feature space. Based on patch-wise processing, the algorithm produces dense anomaly masks, allowing for the localization of detected anomalies. The method robustly models normality through a single reference image. This formulation avoids explicit supervision and dataset-specific training, making it suitable for real-world deployment. We evaluate the method on standard benchmarks and on an automated vehicle in real-world scenarios. Despite its simplicity, the method achieves good performance on the Road Anomaly benchmark and demonstrates consistent qualitative behavior in practice, successfully highlighting semantically unusual objects in diverse scenes. These results suggest that simple, reference-based methods can provide useful anomaly signals under realistic operating conditions. |
| 2026-05-19 | [FlowErase-RL: Rethinking Concept Erasure as Reward Optimization in Flow Matching Models](http://arxiv.org/abs/2605.19739v1) | Yi Sun, Zhiqi Zhang et al. | Recent advances in flow matching models have significantly improved text-to-image generation quality, but also introduce growing safety risks due to the generation of harmful or undesirable content. Existing concept erasure methods are either inference-time interventions with limited effectiveness or rely on supervised fine-tuning (SFT), which requires precisely aligned data and struggles with scalability and multi-concept settings. In this paper, we propose \emph{FlowErase-RL}, the first GRPO-based framework for concept erasure in flow matching models. We reformulate concept erasure as a reward optimization problem and introduce a \textbf{dynamic dual-path reward mechanism} that jointly optimizes (i) a Concept Erasure (CE) reward to suppress target concepts and (ii) a Non-target Space (NS) reward to preserve generative fidelity. The two reward paths are adaptively balanced during training via a performance-driven switching strategy, enabling stable optimization without explicit supervision. Extensive experiments on nudity, object, and artistic style erasure demonstrate that our method achieves state-of-the-art erasure performance while maintaining strong image quality and semantic alignment. Moreover, it exhibits robust resistance to adversarial attacks and scales effectively to multi-concept scenarios. Our results establish a new paradigm for safe and controllable generation in flow matching models. |
| 2026-05-19 | [Measuring Safety Alignment Effects in Autonomous Security Agents](http://arxiv.org/abs/2605.19722v1) | Isaac David, Arthur Gervais | Do stock safety-aligned language models and their uncensored or abliterated derivatives behave differently when run as autonomous security agents? Single-turn refusal benchmarks cannot answer this question: security agents must inspect repositories, call tools, and produce vulnerability evidence inside authorized sandboxes.   We present a trace-based benchmark of 30 local vulnerability-analysis tasks with fixed tools, deterministic success predicates, redaction rules, and grounding checks, and compare four stock models against uncensored or abliterated derivatives: Gemma 4 31B, Gemma 4 26B A4B, Qwen2.5-Coder 7B, and Llama 3.1 8B. The artifact contains 1,500 security-agent traces and 800 non-security control traces. The Gemma pairs show large less-restricted gains on security tasks: 14.0% versus 0.7% success for 31B and 10.7% versus 0.0% for 26B, with higher mean grounding (3.91 versus 3.27 and 4.12 versus 1.64 out of five) and 0.0% refusal, suppressed-action, and unsafe-action rates in the 31B traces. However, controls and non-Gemma pairs rule out a clean security-specific or universal less-restricted effect: Gemma gaps also appear on ordinary coding tasks, Qwen2.5-Coder success is lower for the less-restricted derivative (2.0% versus 5.3%), and the abliterated Llama derivative fails the tool protocol. Across all families, hard proof-of-trigger and patch-verification tasks remain unsolved. These results show that safety alignment effects in autonomous security agents should be measured at the system level, separating refusal, unsafe action, tool reliability, and evidence grounding rather than treating refusal rate as the safety signal. |
| 2026-05-19 | [KIO-planner: Attention-Guided Single-Stage Motion Planning with Dual Mapping for UAV Navigation](http://arxiv.org/abs/2605.19703v1) | Dexing Yao, Haochen Li et al. | Autonomous UAV flight in confined, wall-dense environments requires low-latency and reliable motion planning under strict safety constraints. Traditional optimization-based planners suffer from mapping latency and easily fall into local minima when navigating through dense structural obstacles. Meanwhile, existing end-to-end learning methods struggle to extract fine-grained geometric features from raw depth images and lack hard kinodynamic constraints, leading to unpredictable collisions near walls. To address these issues, we propose KIO-planner, an attention-guided single-stage trajectory planning framework. First, we integrate a Convolutional Block Attention Module (CBAM) into the perception backbone to adaptively focus on critical structural edges and traversable space. Second, we introduce a novel Dual Mapping mechanism--comprising physical bounds activation and a deterministic Geometric Safety Shield in the depth-pixel space--to enforce kinodynamic feasibility and collision-free flight without global map fusion. Extensive high-fidelity simulated experiments demonstrate that KIO-planner enables highly agile navigation at speeds up to 3.0 m/s. Compared to the state-of-the-art baseline, KIO-planner achieves lower inference latency (approximately 24 ms) and generates significantly smoother trajectories, reducing control cost by 28.4%. Most notably, our Dual Mapping substantially increases the worst-case safety margin, measured by minimum distance to obstacles, from 0.48 m to 0.76 m, ensuring fast, smooth, and safer navigation in highly constrained environments. |
| 2026-05-19 | [Pseudocode-Guided Structured Reasoning for Automating Reliable Inference in Vision-Language Models](http://arxiv.org/abs/2605.19663v1) | Weicong Ni, Tianbao Jiang et al. | Vision-Language Models (VLMs) are becoming the cornerstone of high-level reasoning for robotic automation, enabling robots to parse natural language commands and perceive their environments. However, their susceptibility to hallucinations introduces critical failures in decision-making, posing significant safety and reliability risks in physical deployments. This challenge is exacerbated by the open-ended nature of real-world tasks, where questions vary vastly in difficulty and modality, demanding robust and adaptable reasoning strategies. To tackle this, we propose the Pseudocode-guided Structured Reasoning framework (PStar), which adaptively selects structured pseudocode reasoning paths to help VLMs perform flexible and step-by-step reasoning. We first design a set of abstract reasoning functions and formulate a structured pseudocode library to represent modular reasoning strategies. Crucially, we design a Difficulty Feature Vector (DFV) that allows the model to assess question complexity and adaptively choose appropriate reasoning strategies-enhancing robustness and interpretability. Extensive experiments demonstrate that PStar significantly reduces hallucination rates, achieving state-of-the-art scores of 87.1% on POPE and 68.0% on MMStar, outperforming even GPT-4V. By providing a validated mechanism to reduce visual-language errors, PStar offers a critical step toward deploying more trustworthy and deterministic VLMs for real-world automated systems, where such errors can lead to catastrophic outcomes. |

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



