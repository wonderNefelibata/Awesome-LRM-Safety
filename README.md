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
| 2026-04-27 | [Green Shielding: A User-Centric Approach Towards Trustworthy AI](http://arxiv.org/abs/2604.24700v1) | Aaron J. Li, Nicolas Sanchez et al. | Large language models (LLMs) are increasingly deployed, yet their outputs can be highly sensitive to routine, non-adversarial variation in how users phrase queries, a gap not well addressed by existing red-teaming efforts. We propose Green Shielding, a user-centric agenda for building evidence-backed deployment guidance by characterizing how benign input variation shifts model behavior. We operationalize this agenda through the CUE criteria: benchmarks with authentic Context, reference standards and metrics that capture true Utility, and perturbations that reflect realistic variations in the Elicitation of model behavior. Guided by the PCS framework and developed with practicing physicians, we instantiate Green Shielding in medical diagnosis through HealthCareMagic-Diagnosis (HCM-Dx), a benchmark of patient-authored queries, together with structured reference diagnosis sets and clinically grounded metrics for evaluating differential diagnosis lists. We also study perturbation regimes that capture routine input variation and show that prompt-level factors shift model behavior along clinically meaningful dimensions. Across multiple frontier LLMs, these shifts trace out Pareto-like tradeoffs. In particular, neutralization, which removes common user-level factors while preserving clinical content, increases plausibility and yields more concise, clinician-like differentials, but reduces coverage of highly likely and safety-critical conditions. Together, these results show that interaction choices can systematically shift task-relevant properties of model outputs and support user-facing guidance for safer deployment in high-stakes domains. Although instantiated here in medical diagnosis, the agenda extends naturally to other decision-support settings and agentic AI systems. |
| 2026-04-27 | [Governing What You Cannot Observe: Adaptive Runtime Governance for Autonomous AI Agents](http://arxiv.org/abs/2604.24686v1) | German Marin, Jatin Chaudhary | Autonomous AI agents can remain fully authorized and still become unsafe as behavior drifts, adversaries adapt, and decision patterns shift without any code change. We propose the \textbf{Informational Viability Principle}: governing an agent reduces to estimating a bound on unobserved risk $\hat{B}(x) = U(x) + SB(x) + RG(x)$ and allowing an action only when its capacity $S(x)$ exceeds $\hat{B}(x)$ by a safety margin. The \textbf{Agent Viability Framework}, grounded in Aubin's viability theory, establishes three properties -- monitoring (P1), anticipation (P2), and monotonic restriction (P3) -- as individually necessary and collectively sufficient for documented failure modes. \textbf{RiskGate} instantiates the framework with dedicated statistical estimators (KL divergence, segment-vs-rest $z$-tests, sequential pattern matching), a fail-secure monotonic pipeline, and a closed-loop Autopilot formalised as an instance of Aubin's regulation map with kill-switch-as-last-resort; a scalar Viability Index $VI(t) \in [-1,+1]$ with first-order $t^*$ prediction transforms governance from reactive to predictive. Contributions are the theoretical framework, the reference implementation, and analytical coverage against published agent-failure taxonomies; quantitative empirical evaluation is scoped as follow-up work. |
| 2026-04-27 | [The Price of Agreement: Measuring LLM Sycophancy in Agentic Financial Applications](http://arxiv.org/abs/2604.24668v1) | Zhenyu Zhao, Aparna Balagopalan et al. | Given the increased use of LLMs in financial systems today, it becomes important to evaluate the safety and robustness of such systems. One failure mode that LLMs frequently display in general domain settings is that of sycophancy. That is, models prioritize agreement with expressed user beliefs over correctness, leading to decreased accuracy and trust. In this work, we focus on evaluating sycophancy that LLMs display in agentic financial tasks. Our findings are three-fold: first, we find the models show only low to modest drops in performance in the face of user rebuttals or contradictions to the reference answer, which distinguishes sycophancy that models display in financial agentic settings from findings in prior work. Second, we introduce a suite of tasks to test for sycophancy by user preference information that contradicts the reference answer and find that most models fail in the presence of such inputs. Lastly, we benchmark different modes of recovery such as input filtering with a pretrained LLM. |
| 2026-04-27 | [Verification of Correlated Equilibria in Concurrent Reachability Games](http://arxiv.org/abs/2604.24655v1) | Senthil Rajasekaran, Jean-François Raskin et al. | As part of an effort to apply the rigorous guarantees of formal verification to multi-agent systems, the field of equilibrium analysis, also called rational verification, studies equilibria in multiplayer games to reason about system-level properties such as safety and scalability. While most prior work focuses on deterministic settings, recent probabilistic extensions enable the use of richer equilibrium concepts. In this paper, we study one such equilibrium concept -- correlated equilibria -- and introduce a natural refinement -- subgame-perfect correlated equilibria -- in the context of the verification problem. We characterize the computational complexity of verifying such equilibria and show a somewhat surprising separation (under standard complexity-theoretic assumptions): despite being more general, correlated equilibria yield a strictly harder P-complete verification problem than the subgame-perfect correlated equilibria verification problem, which can be solved in log-squared-space. We further analyze the setting where inputs are given succinctly via Bayesian networks, as the study of succinct representations is an important direction to connect static complexity-theoretic analysis to real-world program representations, and show that this complexity gap disappears under such representations. |
| 2026-04-27 | [Evaluating whether AI models would sabotage AI safety research](http://arxiv.org/abs/2604.24618v1) | Robert Kirk, Alexandra Souly et al. | We evaluate the propensity of frontier models to sabotage or refuse to assist with safety research when deployed as AI research agents within a frontier AI company. We apply two complementary evaluations to four Claude models (Mythos Preview, Opus 4.7 Preview, Opus 4.6, and Sonnet 4.6): an unprompted sabotage evaluation testing model behaviour with opportunities to sabotage safety research, and a sabotage continuation evaluation testing whether models continue to sabotage when placed in trajectories where prior actions have started undermining research. We find no instances of unprompted sabotage across any model, with refusal rates close to zero for Mythos Preview and Opus 4.7 Preview, though all models sometimes only partially completed tasks. In the continuation evaluation, Mythos Preview actively continues sabotage in 7% of cases (versus 3% for Opus 4.6, 4% for Sonnet 4.6, and 0% for Opus 4.7 Preview), and exhibits reasoning-output discrepancy in the majority of these cases, indicating covert sabotage reasoning. Our evaluation framework builds on Petri, an open-source LLM auditing tool, with a custom scaffold running models inside Claude Code, alongside an iterative pipeline for generating realistic sabotage trajectories. We measure both evaluation awareness and a new form of situational awareness termed "prefill awareness", the capability to recognise that prior trajectory content was not self-generated. Opus 4.7 Preview shows notably elevated unprompted evaluation awareness, while prefill awareness remains low across all models. Finally, we discuss limitations including evaluation awareness confounds, limited scenario coverage, and untested pathways to risk beyond safety research sabotage. |
| 2026-04-27 | [Children's Online Safety Risks and Ethical Considerations in XR Games](http://arxiv.org/abs/2604.24601v1) | Zinan Zhang, Xinning Gui et al. | Emerging extended reality technologies are reshaping how children play, learn, and socialize. Yet, they also present serious safety risks. Gaming, a primary form of entertainment for children, is also one of the key applications of XR. While XR platforms offer immersive and engaging gaming experiences, recent news has highlighted safety concerns such as car accidents, lower judgment for real-world situations, and exposure to disturbing content like virtual rape. This research examines how XR game design may lead to online safety risks for children. Through analysis of player forums, game developer forums, and interviews with child players, we identify harmful XR design patterns, explore how developers collaboratively generate and implement risky game ideas, and document children's firsthand experiences of online safety risks. Existing ethical frameworks often fail to address the immersive and socially dynamic nature of XR games. We advocate for a child-centered, design-aware approach to ethical considerations in XR games, urging platforms and policymakers to prioritize children's developmental needs. Our work aims to help shape safer, more inclusive XR environments through research and cross-sector collaboration. |
| 2026-04-27 | [FastOMOP: A Foundational Architecture for Reliable Agentic Real-World Evidence Generation on OMOP CDM data](http://arxiv.org/abs/2604.24572v1) | Niko Moeller-Grell, Shihao Shenzhang et al. | The Observational Medical Outcomes Partnership Common Data Model (OMOP CDM), maintained by the Observational Health Data Sciences and Informatics (OHDSI) collaboration, enabled the harmonisation of electronic health records data of nearly one billion patients in 83 countries. Yet generating real-world evidence (RWE) from these repositories remains a manual process requiring clinical, epidemiological and technical expertise. LLMs and multi-agent systems have shown promise for clinical tasks, but RWE automation exposes a fundamental challenge: agentic systems introduce emergent behaviours, coordination failures and safety risks that existing approaches fail to govern. No infrastructure exists to ensure agentic RWE generation is flexible, safe and auditable across the lifecycle. We introduce FastOMOP, an open-source multi-agent architecture that addresses this gap by separating three infrastructure layers, governance, observability and orchestration, from pluggable agent-teams. Governance is enforced at the process boundary through deterministic validation independent of agent reasoning, ensuring no compromised or hallucinating agent can bypass safety controls. Agent teams for phenotyping, study design and statistical analysis inherit these guarantees through controlled tool exposure. We validated FastOMOP using a natural-language-to-SQL agent team across three OMOP CDM datasets: synthetic data from Synthea, MIMIC-IV and a real-world NHS dataset from Lancashire Teaching Hospitals (IDRIL). FastOMOP achieved reliability scores of 0.84-0.94 with perfect adversarial and out-of-scope block rates, demonstrating process-boundary governance delivers safety guarantees independent of model choice. These results indicate that the reliability gap in RWE deployment is architectural rather than model capability, and establish FastOMOP as a governed architecture for progressive RWE automation. |
| 2026-04-27 | [Layerwise Convergence Fingerprints for Runtime Misbehavior Detection in Large Language Models](http://arxiv.org/abs/2604.24542v1) | Nay Myat Min, Long H. Pham et al. | Large language models deployed at runtime can misbehave in ways that clean-data validation cannot anticipate: training-time backdoors lie dormant until triggered, jailbreaks subvert safety alignment, and prompt injections override the deployer's instructions. Existing runtime defenses address these threats one at a time and often assume a clean reference model, trigger knowledge, or editable weights, assumptions that rarely hold for opaque third-party artifacts. We introduce Layerwise Convergence Fingerprinting (LCF), a tuning-free runtime monitor that treats the inter-layer hidden-state trajectory as a health signal: LCF computes a diagonal Mahalanobis distance on every inter-layer difference, aggregates via Ledoit-Wolf shrinkage, and thresholds via leave-one-out calibration on 200 clean examples, with no reference model, trigger knowledge, or retraining. Evaluated on four architectures (Llama-3-8B, Qwen2.5-7B, Gemma-2-9B, Qwen2.5-14B) across backdoors, jailbreaks, and prompt injection (56 backdoor combinations, 3 jailbreak techniques, and BIPIA email + code-QA), LCF reduces mean backdoor attack success rate (ASR) below 1% on Qwen2.5-7B and Gemma-2 and to 1.3% on Qwen2.5-14B, detects 92-100% of DAN jailbreaks (62-100% for GCG and softer role-play), and flags 100% of text-payload injections across all eight (model, domain) cells, at 12-16% backdoor FPR and <0.1% inference overhead. A single aggregation score covers all three threat families without threat-specific tuning, positioning LCF as a general-purpose runtime safety layer for cloud-served and on-device LLMs. |
| 2026-04-27 | [Sliding Mode Control for Safe Trajectory Tracking with Moving Obstacles Avoidance: Experimental Validation on Planar Robots](http://arxiv.org/abs/2604.24518v1) | Shubham Sawarkar, P Sangeerth et al. | This paper presents a unified control framework for robust trajectory tracking and moving obstacle avoidance applicable to a broad class of mobile robots. By formulating a generalized kinematic transformation, we convert diverse vehicle dynamics into a strict feedback form, facilitating the design of a Sliding Mode Control (SMC) strategy for precise and robust reference tracking. To ensure operational safety in dynamic environments, the tracking controller is integrated with a Collision Cone Control Barrier Function (C3BF) based safety filter. The proposed architecture guarantees asymptotic tracking in the presence of external disturbances while strictly enforcing collision avoidance constraints. The novelty of this work lies in designing a sliding mode controller for ground robots like the Ackermann drive, which has not been done before. The efficacy and versatility of the approach are validated through numerical simulations and extensive real-world experiments on three distinct platforms: an Ackermann-steered vehicle, a differential drive robot, and a quadrotor drone. Video of the experiments are available at https://youtu.be/dWcxwum96vk |
| 2026-04-27 | [Compilation and Execution of an Embeddable YOLO-NAS on the VTA](http://arxiv.org/abs/2604.24455v1) | Anthony Faure-Gignoux, Kevin Delmas et al. | Deploying complex Convolutional Neural Networks (CNNs) on FPGA-based accelerators is a promising way forward for safety-critical domains such as aeronautics. In a previous work, we have explored the Versatile Tensor Accelerator (VTA) and showed its suitability for avionic applications. For that, we developed an initial stand-alone compiler designed with certification in mind. However, this compiler still suffers from some limitations that are overcome in this paper. The contributions consist in extending and fully automating the VTA compilation chain to allow complete CNN compilation and support larger CNNs (which parameters do not fit in the on-chip memory). The effectiveness is demonstrated by the successful compilation and simulated execution of a YOLO-NAS object detection model. |
| 2026-04-27 | [Minimum Reachability Probabilities in Rectangular Automata with Random Clocks](http://arxiv.org/abs/2604.24440v1) | Joanna Delicaris, Erika Ábrahám et al. | Control applications for cyber-physical systems must make reliably safe control decisions in the presence of continuous dynamics as well as stochastic uncertainty. Providing safety guarantees for such systems requires formal modeling and analysis techniques that capture these aspects. For modeling, in this paper we consider rectangular automata with random clocks under prophetic scheduling. For this model class, existing methods can compute only upper bounds on reachability probabilities, enabling optimistic, best-case safety reasoning. We complement this view by introducing a novel method to compute lower bounds, thereby enabling worst-case analysis that is essential for safety-critical applications. Although both upper and lower bounds rely on reachability analysis, they are not dual: computing lower bounds requires an explicit separation of stochastic and nondeterministic choices along executions. We implement our approach and demonstrate its practical feasibility on an electric vehicle charging scenario, showing that meaningful worst-case guarantees can be obtained. |
| 2026-04-27 | [An Automatic Ground Collision Avoidance System with Reinforcement Learning](http://arxiv.org/abs/2604.24403v1) | Seyyid Osman Sevgili, Atahan Cilan et al. | This article evaluates an artificial intelligence (AI)-based Automatic Ground Collision Avoidance System (AGCAS) designed for advanced jet trainers to enhance operational effectiveness. In the continuously evolving field of aerospace engineering, the integration of AI is crucial for advancing operations with improved timing constraints and efficiency. Our study explores the design process of an AI-driven AGCAS, specifically tailored for advanced jet trainers, focusing on addressing the AGCAS problem within a limited observation space. The system utilizes line-of-sight queries on a terrain server to ensure precise and efficient collision avoidance. This approach aims to significantly improve the safety and operational capabilities of advanced jet trainers. |
| 2026-04-27 | [Pedestrians play chicken with an autonomous vehicle](http://arxiv.org/abs/2604.24384v1) | Rakshit Soni, Charles Fox | Automated vehicles (AVs) are commonly programmed to yield unconditionally to pedestrians in the interest of safety. However, this design choice can give rise to the Freezing Robot Problem in which pedestrians learn to assert priority at every interaction, causing vehicles to stall and make no progress. The game theoretic Sequential Chicken model has shown that, like human drivers, AVs can resolve this problem by trading credible threats of very small risks of collision or larger risks of less severe invasion of personal space against the value of time due to yielding delays. This paper presents the first demonstration and evaluation of this approach using a real AV with human subjects and shows that pedestrian behavior under experimentally constrained safety conditions can be well fitted by Sequential Chicken, with a low time value of collision, suggestive of their planning to avoid proxemic personal space penalties as well as actual collisions. |
| 2026-04-27 | [Certified geometric robustness -- Super-DeepG](http://arxiv.org/abs/2604.24379v1) | Noémie Cohen, Mélanie Ducoffe et al. | Safety-critical applications are required to perform as expected in normal operations. Image processing functions are often required to be insensitive to small geometric perturbations such as rotation, scaling, shearing or translation. This paper addresses the formal verification of neural networks against geometric perturbations on their image dataset. Our method Super-DeepG improves the reasoning used in linear relaxation techniques and Lipschitz optimization, and provides an implementation that leverages GPU hardware. By doing so, Super-DeepG achieves both precision and computational efficiency of robustness certification, to an extent that outperforms prior work. Super-DeepG is shared as an open-source tool on GitHub. |
| 2026-04-27 | [OS-SPEAR: A Toolkit for the Safety, Performance,Efficiency, and Robustness Analysis of OS Agents](http://arxiv.org/abs/2604.24348v1) | Zheng Wu, Yi Hua et al. | The evolution of Multimodal Large Language Models (MLLMs) has shifted the focus from text generation to active behavioral execution, particularly via OS agents navigating complex GUIs. However, the transition of these agents into trustworthy daily partners is hindered by a lack of rigorous evaluation regarding safety, efficiency, and multi-modal robustness. Current benchmarks suffer from narrow safety scenarios, noisy trajectory labeling, and limited robustness metrics. To bridge this gap, we propose OS-SPEAR, a comprehensive toolkit for the systematic analysis of OS agents across four dimensions: Safety, Performance, Efficiency, and Robustness. OS-SPEAR introduces four specialized subsets: (1) a S(afety)-subset encompassing diverse environment- and human-induced hazards; (2) a P(erformance)-subset curated via trajectory value estimation and stratified sampling; (3) an E(fficiency)-subset quantifying performance through the dual lenses of temporal latency and token consumption; and (4) a R(obustness)-subset that applies cross-modal disturbances to both visual and textual inputs. Additionally, we provide an automated analysis tool to generate human-readable diagnostic reports. We conduct an extensive evaluation of 22 popular OS agents using OS-SPEAR. Our empirical results reveal critical insights into the current landscape: notably, a prevalent trade-off between efficiency and safety or robustness, the performance superiority of specialized agents over general-purpose models, and varying robustness vulnerabilities across different modalities. By providing a multidimensional ranking and a standardized evaluation framework, OS-SPEAR offers a foundational resource for developing the next generation of reliable and efficient OS agents. The dataset and codes are available at https://github.com/Wuzheng02/OS-SPEAR. |
| 2026-04-27 | [OpenPodcar2: a robust, ROS2 vehicle for self-driving research](http://arxiv.org/abs/2604.24242v1) | Rakshit Soni, Chris Waltham et al. | OpenPodcar2 is a robust, ROS2-interfaced, low-cost, open source hardware and software, autonomous vehicle platform based on an off-the-shelf, hard-canopy, mobility scooter donor vehicle. It is a modification of the previous OpenPodcar design, which extends it with robust electronics and ROS2 interfacing, to enable both research and also potential deployment use cases. The platform consists of (a) hardware components: documented as a bill of materials and build instructions; (b) integration to the general purpose OSH R4 mechatronics board and a Gazebo simulation of the vehicle, both presenting a common ROS2 interface (c) higher-level ROS2 software implementations and configurations of standard robot autonomous planning and control, including the nav2 stack which performs SLAM and enacts commands to drive the vehicle from a current to a desired pose around obstacles. OpenPodcar2 can transport a human passenger or similar load at speeds up to 15km/h, for example for use as a last-mile autonomous taxi service or to transport delivery containers similarly around a city center. It is small and safe enough to be parked in a standard research lab robust enough for some deployment cases. Total build cost was around 7,000USD from new components, or 2,000USD with a used Donor Vehicle. OpenPodcar2 thus provides a research balance between real world utility, safety, cost and robustness. |
| 2026-04-27 | [A Theory of Hanoi Omega-Automata and Games](http://arxiv.org/abs/2604.24231v1) | Emmanuel Filiot, Allen Joseph et al. | The Hanoi Omega-Automata (HOA) format has established itself as the definitive standard for encoding $ω$-regular automata in modern synthesis tools. While HOA is widely adopted due to its succinct symbolic representation, using Boolean formulas as transition guards and transition-based coloring, the exact computational cost of these features has remained understudied. This paper provides the first systematic investigation into the theoretical complexity of decision problems for HOA-encoded automata and games. We establish that the structural features of HOA, specifically the symbolic encoding of large alphabets, make classical problems more complex than in traditional formats. We prove that the non-emptiness problem is NP-complete for all standard acceptance conditions, with hardness arising directly from the Boolean transition guards. For language inclusion, we show that the problem is PSPACE-complete under most conditions but becomes EXPSPACE-complete for Emerson-Lei acceptance. Furthermore, we formalize Hanoi Omega-Games (HOG), where the underlying arena is a deterministic HOA with atomic propositions partitioned into inputs and outputs. We provide tight complexity bounds for solving HOGs, ranging from $Π_2$-completeness for parity and safety conditions to PSPACE-completeness for Muller and Emerson-Lei objectives. Finally, we generalize our techniques to symbolic games where transitions are guarded by formulas in arbitrary decidable first-order theories. |
| 2026-04-27 | [Computer Vision-Based Early Detection of Container Loss at Sea](http://arxiv.org/abs/2604.24193v1) | Vishakha Lall, Capt. Stanley S Pinto et al. | Containerised shipping underpins global trade, yet container loss at sea remains a persistent safety, environmental, and economic challenge. Despite compliance with Cargo Securing Manuals, dynamic maritime conditions such as vessel motion, wind loading, and severe sea states can progressively destabilise container stacks, leading to overboard losses. With the new International Maritime Organisation's (IMO) mandatory reporting requirements for lost containers, there is an urgent need for a reliable, evidence-based early detection solution for destabilised containers. This study showcases a low-cost, retrofittable computer vision-based system for early detection of destabilised containers using existing onboard cameras. The framework integrates object segmentation to isolate container stacks, temporal object tracking using optical flow and individual objects' residual motion extraction to quantify relative movement. Experimental evaluation on real onboard ship footage demonstrates that the proposed pipeline effectively isolates container-level motion under challenging conditions of varying sea states and visibility conditions. By enabling early alerts for crew intervention and navigational adjustment, the proposed approach enhances cargo safety, operational resilience, and regulatory compliance. |
| 2026-04-27 | [Omni-o3: Deep Nested Omnimodal Deduction for Deliberative Audio-Visual Reasoning](http://arxiv.org/abs/2604.24191v1) | Zhicheng Zhang, Wentao Gu et al. | Omnimodal understanding entails a massive, highly redundant search space of cross-modal interactions, demanding focused and deliberative reasoning. Current reasoning paradigms rely on either sequential step-by-step generation or parallel sample-by-sample rollouts, leading to isolated reasoning trajectories. This inability to share promising intermediate paths severely limits exploration efficiency and causes compounding errors in complex audio-visual tasks. To break this bottleneck, we introduce Omni-o3, a novel framework driven by a deep nested deduction policy. By formulating reasoning as a dynamic recursive search, Omni-o3 inherently shares reasoning prefixes across branches, enabling the iterative execution of four atomic cognitive actions: expansion, selection, simulation, and backpropagation. To empower this framework, we propose a robust two-stage training paradigm: (1) cold-start supervised fine-tuning on 101K high-quality, long-chain trajectories distilled from 3.5M diverse omnimodal samples, enabling necessary recursive search patterns; and (2) nested group rollout-driven exploratory reinforcement learning on 18K complex multi-turn samples, explicitly guided by a novel multi-step reward model to stimulate deep nested reasoning. Extensive experiments demonstrate that Omni-o3 achieves competitive performance across 11 benchmarks, unlocking advanced capabilities in comprehensive audio-visual, visual-centric, and audio-centric reasoning tasks. |
| 2026-04-27 | [Right-to-Act: A Pre-Execution Non-Compensatory Decision Protocol for AI Systems](http://arxiv.org/abs/2604.24153v1) | Gadi Lavi | Current AI systems increasingly operate in contexts where their outputs directly trigger real-world actions. Most existing approaches to AI safety, risk management, and governance focus on post-hoc validation, probabilistic risk estimation, or certification of model behavior. However, these approaches implicitly assume that once a decision is produced, it is eligible for execution. In this work, we introduce the Right-to-Act protocol, a deterministic, non-compensatory pre-execution decision layer that evaluates whether an AI-generated decision is permitted to be realized at all. Unlike compensatory systems, where high-confidence signals can override failed conditions, the proposed framework enforces strict structural constraints: if any required condition is unmet, execution is halted or deferred. We formalize the distinction between compensatory and non-compensatory decision regimes and define a pre-execution legitimacy boundary. Through a scenario-based case study, we demonstrate how identical AI outputs can lead to divergent outcomes when evaluated under a Right-to-Act protocol, preserving reversibility and preventing premature or irreversible actions. The proposed approach reframes AI control from optimizing decisions to governing their admissibility, introducing a protocol-level abstraction that operates independently of model architecture or training methodology. |

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



