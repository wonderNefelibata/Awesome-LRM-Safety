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
| 2025-06-18 | [Leaky Thoughts: Large Reasoning Models Are Not Private Thinkers](http://arxiv.org/abs/2506.15674v1) | Tommaso Green, Martin Gubri et al. | We study privacy leakage in the reasoning traces of large reasoning models used as personal agents. Unlike final outputs, reasoning traces are often assumed to be internal and safe. We challenge this assumption by showing that reasoning traces frequently contain sensitive user data, which can be extracted via prompt injections or accidentally leak into outputs. Through probing and agentic evaluations, we demonstrate that test-time compute approaches, particularly increased reasoning steps, amplify such leakage. While increasing the budget of those test-time compute approaches makes models more cautious in their final answers, it also leads them to reason more verbosely and leak more in their own thinking. This reveals a core tension: reasoning improves utility but enlarges the privacy attack surface. We argue that safety efforts must extend to the model's internal thinking, not just its outputs. |
| 2025-06-18 | [deepSURF: Detecting Memory Safety Vulnerabilities in Rust Through Fuzzing LLM-Augmented Harnesses](http://arxiv.org/abs/2506.15648v1) | Georgios Androutsopoulos, Antonio Bianchi | Although Rust ensures memory safety by default, it also permits the use of unsafe code, which can introduce memory safety vulnerabilities if misused. Unfortunately, existing tools for detecting memory bugs in Rust typically exhibit limited detection capabilities, inadequately handle Rust-specific types, or rely heavily on manual intervention.   To address these limitations, we present deepSURF, a tool that integrates static analysis with Large Language Model (LLM)-guided fuzzing harness generation to effectively identify memory safety vulnerabilities in Rust libraries, specifically targeting unsafe code. deepSURF introduces a novel approach for handling generics by substituting them with custom types and generating tailored implementations for the required traits, enabling the fuzzer to simulate user-defined behaviors within the fuzzed library. Additionally, deepSURF employs LLMs to augment fuzzing harnesses dynamically, facilitating exploration of complex API interactions and significantly increasing the likelihood of exposing memory safety vulnerabilities. We evaluated deepSURF on 27 real-world Rust crates, successfully rediscovering 20 known memory safety bugs and uncovering 6 previously unknown vulnerabilities, demonstrating clear improvements over state-of-the-art tools. |
| 2025-06-18 | [LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning](http://arxiv.org/abs/2506.15606v1) | Gabrel J. Perin, Runjin Chen et al. | Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at github.com/VITA-Group/LoX. |
| 2025-06-18 | [RePCS: Diagnosing Data Memorization in LLM-Powered Retrieval-Augmented Generation](http://arxiv.org/abs/2506.15513v1) | Le Vu Anh, Nguyen Viet Anh et al. | Retrieval-augmented generation (RAG) has become a common strategy for updating large language model (LLM) responses with current, external information. However, models may still rely on memorized training data, bypass the retrieved evidence, and produce contaminated outputs. We introduce Retrieval-Path Contamination Scoring (RePCS), a diagnostic method that detects such behavior without requiring model access or retraining. RePCS compares two inference paths: (i) a parametric path using only the query, and (ii) a retrieval-augmented path using both the query and retrieved context by computing the Kullback-Leibler (KL) divergence between their output distributions. A low divergence suggests that the retrieved context had minimal impact, indicating potential memorization. This procedure is model-agnostic, requires no gradient or internal state access, and adds only a single additional forward pass. We further derive PAC-style guarantees that link the KL threshold to user-defined false positive and false negative rates. On the Prompt-WNQA benchmark, RePCS achieves a ROC-AUC of 0.918. This result outperforms the strongest prior method by 6.5 percentage points while keeping latency overhead below 4.7% on an NVIDIA T4 GPU. RePCS offers a lightweight, black-box safeguard to verify whether a RAG system meaningfully leverages retrieval, making it especially valuable in safety-critical applications. |
| 2025-06-18 | [Designing Intent: A Multimodal Framework for Human-Robot Cooperation in Industrial Workspaces](http://arxiv.org/abs/2506.15293v1) | Francesco Chiossi, Julian Rasch et al. | As robots enter collaborative workspaces, ensuring mutual understanding between human workers and robotic systems becomes a prerequisite for trust, safety, and efficiency. In this position paper, we draw on the cooperation scenario of the AIMotive project in which a human and a cobot jointly perform assembly tasks to argue for a structured approach to intent communication. Building on the Situation Awareness-based Agent Transparency (SAT) framework and the notion of task abstraction levels, we propose a multidimensional design space that maps intent content (SAT1, SAT3), planning horizon (operational to strategic), and modality (visual, auditory, haptic). We illustrate how this space can guide the design of multimodal communication strategies tailored to dynamic collaborative work contexts. With this paper, we lay the conceptual foundation for a future design toolkit aimed at supporting transparent human-robot interaction in the workplace. We highlight key open questions and design challenges, and propose a shared agenda for multimodal, adaptive, and trustworthy robotic collaboration in hybrid work environments. |
| 2025-06-18 | [AI-driven visual monitoring of industrial assembly tasks](http://arxiv.org/abs/2506.15285v1) | Mattia Nardon, Stefano Messelodi et al. | Visual monitoring of industrial assembly tasks is critical for preventing equipment damage due to procedural errors and ensuring worker safety. Although commercial solutions exist, they typically require rigid workspace setups or the application of visual markers to simplify the problem. We introduce ViMAT, a novel AI-driven system for real-time visual monitoring of assembly tasks that operates without these constraints. ViMAT combines a perception module that extracts visual observations from multi-view video streams with a reasoning module that infers the most likely action being performed based on the observed assembly state and prior task knowledge. We validate ViMAT on two assembly tasks, involving the replacement of LEGO components and the reconfiguration of hydraulic press molds, demonstrating its effectiveness through quantitative and qualitative analysis in challenging real-world scenarios characterized by partial and uncertain visual observations. Project page: https://tev-fbk.github.io/ViMAT |
| 2025-06-18 | [From LLMs to MLLMs to Agents: A Survey of Emerging Paradigms in Jailbreak Attacks and Defenses within LLM Ecosystem](http://arxiv.org/abs/2506.15170v1) | Yanxu Mao, Tiehan Cui et al. | Large language models (LLMs) are rapidly evolving from single-modal systems to multimodal LLMs and intelligent agents, significantly expanding their capabilities while introducing increasingly severe security risks. This paper presents a systematic survey of the growing complexity of jailbreak attacks and corresponding defense mechanisms within the expanding LLM ecosystem. We first trace the developmental trajectory from LLMs to MLLMs and Agents, highlighting the core security challenges emerging at each stage. Next, we categorize mainstream jailbreak techniques from both the attack impact and visibility perspectives, and provide a comprehensive analysis of representative attack methods, related datasets, and evaluation metrics. On the defense side, we organize existing strategies based on response timing and technical approach, offering a structured understanding of their applicability and implementation. Furthermore, we identify key limitations in existing surveys, such as insufficient attention to agent-specific security issues, the absence of a clear taxonomy for hybrid jailbreak methods, a lack of detailed analysis of experimental setups, and outdated coverage of recent advancements. To address these limitations, we provide an updated synthesis of recent work and outline future research directions in areas such as dataset construction, evaluation framework optimization, and strategy generalization. Our study seeks to enhance the understanding of jailbreak mechanisms and facilitate the advancement of more resilient and adaptive defense strategies in the context of ever more capable LLMs. |
| 2025-06-18 | [A Force Feedback Exoskeleton for Teleoperation Using Magnetorheological Clutches](http://arxiv.org/abs/2506.15124v1) | Zhongyuan Kong, Lei Li et al. | This paper proposes an upper-limb exoskeleton teleoperation system based on magnetorheological (MR) clutches, aiming to improve operational accuracy and enhance the immersive experience during lunar sampling tasks. Conventional exoskeleton teleoperation systems commonly employ active force feedback solutions, such as servo motors, which typically suffer from high system complexity and increased energy consumption. Furthermore, force feedback devices utilizing motors and gear reducers generally compromise backdrivability and pose safety risks to operators due to active force output. To address these limitations, we propose a semi-active force feedback strategy based on MR clutches. Dynamic magnetic field control enables precise adjustment of joint stiffness and damping, thereby providing smooth and high-resolution force feedback. The designed MR clutch exhibits outstanding performance across key metrics, achieving a torque-to-mass ratio (TMR) of 93.6 Nm/kg, a torque-to-volume ratio (TVR) of 4.05 x 10^5 Nm/m^3, and a torque-to-power ratio (TPR) of 4.15 Nm/W. Notably, the TMR represents an improvement of approximately 246% over a representative design in prior work. Experimental results validate the system's capability to deliver high-fidelity force feedback. Overall, the proposed system presents a promising solution for deep-space teleoperation with strong potential for real-world deployment in future missions. |
| 2025-06-18 | [International Security Applications of Flexible Hardware-Enabled Guarantees](http://arxiv.org/abs/2506.15100v1) | Onni Aarne, James Petrie | As AI capabilities advance rapidly, flexible hardware-enabled guarantees (flexHEGs) offer opportunities to address international security challenges through comprehensive governance frameworks. This report examines how flexHEGs could enable internationally trustworthy AI governance by establishing standardized designs, robust ecosystem defenses, and clear operational parameters for AI-relevant chips. We analyze four critical international security applications: limiting proliferation to address malicious use, implementing safety norms to prevent loss of control, managing risks from military AI systems, and supporting strategic stability through balance-of-power mechanisms while respecting national sovereignty. The report explores both targeted deployments for specific high-risk facilities and comprehensive deployments covering all AI-relevant compute. We examine two primary governance models: verification-based agreements that enable transparent compliance monitoring, and ruleset-based agreements that automatically enforce international rules through cryptographically-signed updates. Through game-theoretic analysis, we demonstrate that comprehensive flexHEG agreements could remain stable under reasonable assumptions about state preferences and catastrophic risks. The report addresses critical implementation challenges including technical thresholds for AI-relevant chips, management of existing non-flexHEG hardware, and safeguards against abuse of governance power. While requiring significant international coordination, flexHEGs could provide a technical foundation for managing AI risks at the scale and speed necessary to address emerging threats to international security and stability. |
| 2025-06-18 | [Flexible Hardware-Enabled Guarantees for AI Compute](http://arxiv.org/abs/2506.15093v1) | James Petrie, Onni Aarne et al. | As artificial intelligence systems become increasingly powerful, they pose growing risks to international security, creating urgent coordination challenges that current governance approaches struggle to address without compromising sensitive information or national security. We propose flexible hardware-enabled guarantees (flexHEGs), that could be integrated with AI accelerators to enable trustworthy, privacy-preserving verification and enforcement of claims about AI development. FlexHEGs consist of an auditable guarantee processor that monitors accelerator usage and a secure enclosure providing physical tamper protection. The system would be fully open source with flexible, updateable verification capabilities. FlexHEGs could enable diverse governance mechanisms including privacy-preserving model evaluations, controlled deployment, compute limits for training, and automated safety protocol enforcement. In this first part of a three part series, we provide a comprehensive introduction of the flexHEG system, including an overview of the governance and security capabilities it offers, its potential development and adoption paths, and the remaining challenges and limitations it faces. While technically challenging, flexHEGs offer an approach to address emerging regulatory and international security challenges in frontier AI development. |
| 2025-06-18 | [Systems-Theoretic and Data-Driven Security Analysis in ML-enabled Medical Devices](http://arxiv.org/abs/2506.15028v1) | Gargi Mitra, Mohammadreza Hallajiyan et al. | The integration of AI/ML into medical devices is rapidly transforming healthcare by enhancing diagnostic and treatment facilities. However, this advancement also introduces serious cybersecurity risks due to the use of complex and often opaque models, extensive interconnectivity, interoperability with third-party peripheral devices, Internet connectivity, and vulnerabilities in the underlying technologies. These factors contribute to a broad attack surface and make threat prevention, detection, and mitigation challenging. Given the highly safety-critical nature of these devices, a cyberattack on these devices can cause the ML models to mispredict, thereby posing significant safety risks to patients. Therefore, ensuring the security of these devices from the time of design is essential. This paper underscores the urgency of addressing the cybersecurity challenges in ML-enabled medical devices at the pre-market phase. We begin by analyzing publicly available data on device recalls and adverse events, and known vulnerabilities, to understand the threat landscape of AI/ML-enabled medical devices and their repercussions on patient safety. Building on this analysis, we introduce a suite of tools and techniques designed by us to assist security analysts in conducting comprehensive premarket risk assessments. Our work aims to empower manufacturers to embed cybersecurity as a core design principle in AI/ML-enabled medical devices, thereby making them safe for patients. |
| 2025-06-17 | [Algorithmic Approaches to Enhance Safety in Autonomous Vehicles: Minimizing Lane Changes and Merging](http://arxiv.org/abs/2506.15026v1) | Seyed Moein Abtahi, Akramul Azim | The rapid advancements in autonomous vehicle (AV) technology promise enhanced safety and operational efficiency. However, frequent lane changes and merging maneuvers continue to pose significant safety risks and disrupt traffic flow. This paper introduces the Minimizing Lane Change Algorithm (MLCA), a state-machine-based approach designed to reduce unnecessary lane changes, thereby enhancing both traffic safety and efficiency. The MLCA algorithm prioritizes maintaining lane stability unless safety-critical conditions necessitate a lane change. The algorithm's effectiveness was evaluated through simulations conducted on the SUMO platform, comparing its performance against established models, including LC2017 and MOBIL. Results demonstrate substantial reductions in lane changes and collisions, leading to smoother traffic flow and improved safety metrics. Additionally, the study highlights the MLCA's adaptability to various traffic densities and roadway configurations, showcasing its potential for wide-scale deployment in real-world AV systems. Future work aims to validate these findings in more complex scenarios using the CARLA simulator, which will enable the testing of the algorithm under more dynamic and high-fidelity conditions, such as urban traffic environments with diverse road users. Moreover, the integration of cybersecurity measures for vehicle-to-vehicle (V2V) communication will be explored to ensure robust and secure data exchange, further enhancing the reliability and safety of AV operations. This research contributes to the broader goal of developing intelligent traffic systems that optimize both individual vehicle performance and overall traffic network efficiency. |
| 2025-06-17 | [Context Matters: Learning Generalizable Rewards via Calibrated Features](http://arxiv.org/abs/2506.15012v1) | Alexandra Forsey-Smerek, Julie Shah et al. | A key challenge in reward learning from human input is that desired agent behavior often changes based on context. Traditional methods typically treat each new context as a separate task with its own reward function. For example, if a previously ignored stove becomes too hot to be around, the robot must learn a new reward from scratch, even though the underlying preference for prioritizing safety over efficiency remains unchanged. We observe that context influences not the underlying preference itself, but rather the $\textit{saliency}$--or importance--of reward features. For instance, stove heat affects the importance of the robot's proximity, yet the human's safety preference stays the same. Existing multi-task and meta IRL methods learn context-dependent representations $\textit{implicitly}$--without distinguishing between preferences and feature importance--resulting in substantial data requirements. Instead, we propose $\textit{explicitly}$ modeling context-invariant preferences separately from context-dependent feature saliency, creating modular reward representations that adapt to new contexts. To achieve this, we introduce $\textit{calibrated features}$--representations that capture contextual effects on feature saliency--and present specialized paired comparison queries that isolate saliency from preference for efficient learning. Experiments with simulated users show our method significantly improves sample efficiency, requiring 10x fewer preference queries than baselines to achieve equivalent reward accuracy, with up to 15% better performance in low-data regimes (5-10 queries). An in-person user study (N=12) demonstrates that participants can effectively teach their unique personal contextual preferences using our method, enabling more adaptable and personalized reward learning. |
| 2025-06-17 | [Mixed Traffic: A Perspective from Long Duration Autonomy](http://arxiv.org/abs/2506.15004v1) | Filippos Tzortzoglou, Logan E. Beaver | The rapid adoption of autonomous vehicle has established mixed traffic environments, comprising both autonomous and human-driven vehicles (HDVs), as essential components of next-generation mobility systems. Along these lines, connectivity between autonomous vehicles and infrastructure (V2I) is also a significant factor that can effectively support higher-level decision-making. At the same time, the integration of V2I within mixed traffic environments remains a timely and challenging problem. In this paper, we present a long-duration autonomy controller for connected and automated vehicles (CAVs) operating in such environments, with a focus on intersections where right turns on red are permitted. We begin by deriving the optimal control policy for CAVs under free-flow traffic. Next, we analyze crossing time constraints imposed by smart traffic lights and map these constraints to controller bounds using Control Barrier Functions (CBFs), with the aim to drive a CAV to cross the intersection on time. We also introduce criteria for identifying, in real-time, feasible crossing intervals for each CAV. To ensure safety for the CAVs, we present model-agnostic safety guarantees, and demonstrate their compatibility with both CAVs and HDVs. Ultimately, the final control actions are enforced through a combination of CBF constraints, constraining CAVs to traverse the intersection within the designated time intervals while respecting other vehicles. Finally, we guarantee that our control policy yields always a feasible solution and validate the proposed approach through extensive simulations in MATLAB. |
| 2025-06-17 | [Time-Optimized Safe Navigation in Unstructured Environments through Learning Based Depth Completion](http://arxiv.org/abs/2506.14975v1) | Jeffrey Mao, Raghuram Cauligi Srinivas et al. | Quadrotors hold significant promise for several applications such as agriculture, search and rescue, and infrastructure inspection. Achieving autonomous operation requires systems to navigate safely through complex and unfamiliar environments. This level of autonomy is particularly challenging due to the complexity of such environments and the need for real-time decision making especially for platforms constrained by size, weight, and power (SWaP), which limits flight time and precludes the use of bulky sensors like Light Detection and Ranging (LiDAR) for mapping. Furthermore, computing globally optimal, collision-free paths and translating them into time-optimized, safe trajectories in real time adds significant computational complexity. To address these challenges, we present a fully onboard, real-time navigation system that relies solely on lightweight onboard sensors. Our system constructs a dense 3D map of the environment using a novel visual depth estimation approach that fuses stereo and monocular learning-based depth, yielding longer-range, denser, and less noisy depth maps than conventional stereo methods. Building on this map, we introduce a novel planning and trajectory generation framework capable of rapidly computing time-optimal global trajectories. As the map is incrementally updated with new depth information, our system continuously refines the trajectory to maintain safety and optimality. Both our planner and trajectory generator outperforms state-of-the-art methods in terms of computational efficiency and guarantee obstacle-free trajectories. We validate our system through robust autonomous flight experiments in diverse indoor and outdoor environments, demonstrating its effectiveness for safe navigation in previously unknown settings. |
| 2025-06-17 | [FEAST: A Flexible Mealtime-Assistance System Towards In-the-Wild Personalization](http://arxiv.org/abs/2506.14968v1) | Rajat Kumar Jenamani, Tom Silver et al. | Physical caregiving robots hold promise for improving the quality of life of millions worldwide who require assistance with feeding. However, in-home meal assistance remains challenging due to the diversity of activities (e.g., eating, drinking, mouth wiping), contexts (e.g., socializing, watching TV), food items, and user preferences that arise during deployment. In this work, we propose FEAST, a flexible mealtime-assistance system that can be personalized in-the-wild to meet the unique needs of individual care recipients. Developed in collaboration with two community researchers and informed by a formative study with a diverse group of care recipients, our system is guided by three key tenets for in-the-wild personalization: adaptability, transparency, and safety. FEAST embodies these principles through: (i) modular hardware that enables switching between assisted feeding, drinking, and mouth-wiping, (ii) diverse interaction methods, including a web interface, head gestures, and physical buttons, to accommodate diverse functional abilities and preferences, and (iii) parameterized behavior trees that can be safely and transparently adapted using a large language model. We evaluate our system based on the personalization requirements identified in our formative study, demonstrating that FEAST offers a wide range of transparent and safe adaptations and outperforms a state-of-the-art baseline limited to fixed customizations. To demonstrate real-world applicability, we conduct an in-home user study with two care recipients (who are community researchers), feeding them three meals each across three diverse scenarios. We further assess FEAST's ecological validity by evaluating with an Occupational Therapist previously unfamiliar with the system. In all cases, users successfully personalize FEAST to meet their individual needs and preferences. Website: https://emprise.cs.cornell.edu/feast |
| 2025-06-17 | [FORTRESS: Frontier Risk Evaluation for National Security and Public Safety](http://arxiv.org/abs/2506.14922v1) | Christina Q. Knight, Kaustubh Deshpande et al. | The rapid advancement of large language models (LLMs) introduces dual-use capabilities that could both threaten and bolster national security and public safety (NSPS). Models implement safeguards to protect against potential misuse relevant to NSPS and allow for benign users to receive helpful information. However, current benchmarks often fail to test safeguard robustness to potential NSPS risks in an objective, robust way. We introduce FORTRESS: 500 expert-crafted adversarial prompts with instance-based rubrics of 4-7 binary questions for automated evaluation across 3 domains (unclassified information only): Chemical, Biological, Radiological, Nuclear and Explosive (CBRNE), Political Violence & Terrorism, and Criminal & Financial Illicit Activities, with 10 total subcategories across these domains. Each prompt-rubric pair has a corresponding benign version to test for model over-refusals. This evaluation of frontier LLMs' safeguard robustness reveals varying trade-offs between potential risks and model usefulness: Claude-3.5-Sonnet demonstrates a low average risk score (ARS) (14.09 out of 100) but the highest over-refusal score (ORS) (21.8 out of 100), while Gemini 2.5 Pro shows low over-refusal (1.4) but a high average potential risk (66.29). Deepseek-R1 has the highest ARS at 78.05, but the lowest ORS at only 0.06. Models such as o1 display a more even trade-off between potential risks and over-refusals (with an ARS of 21.69 and ORS of 5.2). To provide policymakers and researchers with a clear understanding of models' potential risks, we publicly release FORTRESS at https://huggingface.co/datasets/ScaleAI/fortress_public. We also maintain a private set for evaluation. |
| 2025-06-17 | [PeRL: Permutation-Enhanced Reinforcement Learning for Interleaved Vision-Language Reasoning](http://arxiv.org/abs/2506.14907v1) | Yizhen Zhang, Yang Ding et al. | Inspired by the impressive reasoning capabilities demonstrated by reinforcement learning approaches like DeepSeek-R1, recent emerging research has begun exploring the use of reinforcement learning (RL) to enhance vision-language models (VLMs) for multimodal reasoning tasks. However, most existing multimodal reinforcement learning approaches remain limited to spatial reasoning within single-image contexts, yet still struggle to generalize to more complex and real-world scenarios involving multi-image positional reasoning, where understanding the relationships across images is crucial. To address this challenge, we propose a general reinforcement learning approach PeRL tailored for interleaved multimodal tasks, and a multi-stage strategy designed to enhance the exploration-exploitation trade-off, thereby improving learning efficiency and task performance. Specifically, we introduce permutation of image sequences to simulate varied positional relationships to explore more spatial and positional diversity. Furthermore, we design a rollout filtering mechanism for resampling to focus on trajectories that contribute most to learning optimal behaviors to exploit learned policies effectively. We evaluate our model on 5 widely-used multi-image benchmarks and 3 single-image benchmarks. Our experiments confirm that PeRL trained model consistently surpasses R1-related and interleaved VLM baselines by a large margin, achieving state-of-the-art performance on multi-image benchmarks, while preserving comparable performance on single-image tasks. |
| 2025-06-17 | [DETONATE: A Benchmark for Text-to-Image Alignment and Kernelized Direct Preference Optimization](http://arxiv.org/abs/2506.14903v1) | Renjith Prasad, Abhilekh Borah et al. | Alignment is crucial for text-to-image (T2I) models to ensure that generated images faithfully capture user intent while maintaining safety and fairness. Direct Preference Optimization (DPO), prominent in large language models (LLMs), is extending its influence to T2I systems. This paper introduces DPO-Kernels for T2I models, a novel extension enhancing alignment across three dimensions: (i) Hybrid Loss, integrating embedding-based objectives with traditional probability-based loss for improved optimization; (ii) Kernelized Representations, employing Radial Basis Function (RBF), Polynomial, and Wavelet kernels for richer feature transformations and better separation between safe and unsafe inputs; and (iii) Divergence Selection, expanding beyond DPO's default Kullback-Leibler (KL) regularizer by incorporating Wasserstein and R'enyi divergences for enhanced stability and robustness. We introduce DETONATE, the first large-scale benchmark of its kind, comprising approximately 100K curated image pairs categorized as chosen and rejected. DETONATE encapsulates three axes of social bias and discrimination: Race, Gender, and Disability. Prompts are sourced from hate speech datasets, with images generated by leading T2I models including Stable Diffusion 3.5 Large, Stable Diffusion XL, and Midjourney. Additionally, we propose the Alignment Quality Index (AQI), a novel geometric measure quantifying latent-space separability of safe/unsafe image activations, revealing hidden vulnerabilities. Empirically, we demonstrate that DPO-Kernels maintain strong generalization bounds via Heavy-Tailed Self-Regularization (HT-SR). DETONATE and complete code are publicly released. |
| 2025-06-17 | [OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents](http://arxiv.org/abs/2506.14866v1) | Thomas Kuntz, Agatha Duzan et al. | Computer use agents are LLM-based agents that can directly interact with a graphical user interface, by processing screenshots or accessibility trees. While these systems are gaining popularity, their safety has been largely overlooked, despite the fact that evaluating and understanding their potential for harmful behavior is essential for widespread adoption. To address this gap, we introduce OS-Harm, a new benchmark for measuring safety of computer use agents. OS-Harm is built on top of the OSWorld environment and aims to test models across three categories of harm: deliberate user misuse, prompt injection attacks, and model misbehavior. To cover these cases, we create 150 tasks that span several types of safety violations (harassment, copyright infringement, disinformation, data exfiltration, etc.) and require the agent to interact with a variety of OS applications (email client, code editor, browser, etc.). Moreover, we propose an automated judge to evaluate both accuracy and safety of agents that achieves high agreement with human annotations (0.76 and 0.79 F1 score). We evaluate computer use agents based on a range of frontier models - such as o4-mini, Claude 3.7 Sonnet, Gemini 2.5 Pro - and provide insights into their safety. In particular, all models tend to directly comply with many deliberate misuse queries, are relatively vulnerable to static prompt injections, and occasionally perform unsafe actions. The OS-Harm benchmark is available at https://github.com/tml-epfl/os-harm. |

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



