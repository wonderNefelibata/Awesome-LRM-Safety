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
| 2026-09-21 | [Emergent Collusion in Long-Horizon LLM Agent Interaction](http://arxiv.org/abs/2609.24967v1) | Xinrui Shi, Yanzhe Zhang et al. | LLM agents are increasingly deployed in collaborative settings, yet long-term interaction may give rise to undesirable coordination. We study the emergence of collusion in a long-horizon multi-agent environment: two agents repeatedly complete individual tasks, share task logs, verify each other's work, and receive rewards. We introduce realistic constraints that make compliance with the verification protocol incompatible with reward maximization, and find that agents increasingly deviate from the protocol over repeated interactions. Collusion emerges in 94% of trajectories across 10 models, and more capable models within the same family reach it earlier. Controlled peer interventions show that collusion is shaped by peer behavior, while ablations reveal additional effects of reward structure, the verification feedback agents receive, and their interaction history. In particular, restricting the amount and scope of interaction history available to agents reduces collusion. Overall, our findings show that long-horizon interaction can reshape how agents coordinate in ways that create safety risks. |
| 2026-09-21 | [Perception-Aware Communication Middleware for Distributed Visual Perception in UAV Swarms](http://arxiv.org/abs/2609.24964v1) | Manveen Kaur, Kevin Loi et al. | Unmanned Aerial Vehicle (UAV) swarms increasingly support safety-critical applications that rely on distributed visual perception. Meeting the low-latency requirements of these applications can require perception models to execute within the swarm on inference-capable UAVs, creating a need for efficient UAV-to-UAV transport of high-bandwidth perception data. However, the Quality-of-Service (QoS) requirements of perception differ from conventional packet-level QoS; successful delivery of individual packets does not ensure that a complete, timely, and usable image is available for inference. We present a novel perception-aware communication middleware that treats complete perception-data samples as the communication objects for which QoS must be satisfied. The middleware extends a lightweight UDP broker-based publish-subscribe architecture with perception-specific services, including image fragmentation and reconstruction, concurrent packet transmission, priority-aware scheduling, and image quality assessment. The middleware is evaluated on a heterogeneous hardware testbed emulating a UAV swarm using YOLOv8n object detection. Experimental results demonstrate low end-to-end application latency, substantially higher throughput than a lightweight UDP broker, effective prioritization of perception traffic under increasing background load, and mitigation of object-detection degradation through middleware-level image quality assessment. This work provides an initial framework for integrating AI-specific data handling into communication middleware to support emerging distributed AI applications in multi-agent mobile cyber-physical systems. |
| 2026-09-21 | [ATCion: Exploring the Design of Icon-based Visual Aids for Enhancing In-cockpit Air Traffic Control Communication](http://arxiv.org/abs/2609.24863v1) | Yue Lyu, Xizi Wang et al. | Effective communication between pilots and air traffic control (ATC) is essential for aviation safety, but verbal exchanges over radios are prone to miscommunication, especially under high workload conditions. While cockpit-embedded visual aids offer the potential to enhance ATC communication, little is known about how to design and integrate such aids. We present an exploratory, user-centered investigation into the design and integration of icon-based visual aids, named ATCion, to support in-cockpit ATC communication, through four phases involving 22 pilots and 1 ATC controller. This study contributes a validated set of design principles and visual icon components for ATC messages. In a comparative study of ATCion, text-based visual aids, and no visual aids, we found that our design improved readback accuracy and reduced memory workload, without negatively impacting flight operations; most participants preferred ATCion over text-based aids, citing their clarity, low cognitive cost, and fast interpretability. Further, we point to implications and opportunities for integrating icon-based aids into future multimodal ATC communication systems to improve both safety and efficiency. |
| 2026-09-21 | [MedRSI: Recursive Self-Improvement for Medical Agents via Clinically Aligned Self-Evolution](http://arxiv.org/abs/2609.24838v1) | Junde Wu, Jiayuan Zhu et al. | Medical agents increasingly combine general reasoning models with specialized clinical tools, yet their capabilities remain largely fixed by what clinicians and engineers design before deployment. Recursive self-improvement (RSI) offers a different paradigm in which agents learn from their own failures and autonomously expand their capabilities, but directly applying RSI to medicine introduces fundamental safety challenges. We introduce MedRSI, the first recursive self-improvement framework for medicine, which continuously transforms diagnostic failures into new clinical capabilities through tool composition and task-specific model training. Inspired by clinical practice, MedRSI introduces two mechanisms for clinically aligned self-evolution. Clinical-cost-aware failure prioritization directs improvement toward errors according to their potential clinical consequences rather than frequency alone. Fast discovery with slow registration separates rapid capability invention from conservative adoption, allowing new tools to enter the persistent agent only after demonstrating sustained benefit across subsequent patient cohorts. Across public glaucoma and heart disease benchmarks and two private clinical tasks, MedRSI progressively develops segmentation, measurement, prediction, multimodal reasoning, and generative capabilities, surpasses manually engineered medical agents, and autonomously discovers solutions to clinical problems not anticipated by its original designers. Our results show that medical agents need not remain constrained by capabilities specified before deployment: with clinically grounded mechanisms governing what to improve and what to retain, they can continuously construct, validate, and accumulate new capabilities from diagnostic experience. Code is available at https://github.com/ImprintLab/MedRSI. |
| 2026-09-21 | [CRiDiT: Instantiating a run-time testbed for trust calibration in AI-infused systems](http://arxiv.org/abs/2609.24833v1) | Yuntian Ding, Nicolas Herbaut et al. | The integration of AI into larger technical infrastructures has made the alignment of human trust with system trustworthiness, known as trust calibration, a critical engineering concern, since misplaced trust in either direction leads to operational and safety risks. While conceptual frameworks provide a strong foundation for understanding trust calibration, their translation into running systems remains a challenge, because there are few testbeds in which human trust inputs, machine trustworthiness evidence, gap detection and remediation operate together within a closed loop. This paper instantiates CRiDiT (Computational Risk-Sensitive biDirectional Trust) as a run-time testbed, operationalising machine-side trust with Dempster-Shafer Theory and PCR5 redistribution, human-side trust with Subjective Logic, and calibration with a threshold-based trust gap. Following the Design Science Research methodology, we exercise the artifact across three high-stakes scenarios (hiring, financial, legal), producing 144 logged interaction steps across fifteen sessions. The analysis shows that the artifact captures trust calibration dynamics as intended, and reveals three points at which the instantiated policy departs from its design requirements: the machine-side estimate begins from a global benchmark rather than task-relevant evidence; risk-sensitive thresholds do not produce risk-sensitive triggering; and the calibration policy assigns explanatory prompts to over-trust, where corrections narrowed the gap in all 6 observed cases. Since the first two arise from the same design decision, to make the difference of two estimated scalars the calibration criterion, they point toward a common requirement: that the criterion should operate on the evidence rather than on scalars derived from it. The third concerns what follows detection, and shows that the action vocabulary inherited from trust repair does not align with what the interaction logs show to be effective. The work contributes the artifact, a characterisation of its run-time behaviour, and the requirements this characterisation elicits. |
| 2026-09-21 | [Decoding Guardrails: XAI-Guided Perturbation Analysis of Prompt Injection Detection](http://arxiv.org/abs/2609.24801v1) | Fernando Outeda, Gustavo Betarte et al. | Large language models (LLMs) are increasingly deployed in production systems, raising concerns about their exposure to adversarial manipulation through prompt injection and jailbreak attacks. Classifier-based guardrails, such as Prompt Guard 2, are widely used as a first line of defense against such attacks, but their internal decision logic is largely opaque to both defenders and attackers. This paper presents an exploratory case study that applies explainable artificial intelligence (XAI) techniques to analyze how Prompt Guard 2 distinguishes malicious from benign prompts.   We conduct four experiments to probe this question empirically. Guided by Vanilla Gradient and SHAP attributions, we find that Prompt Guard 2's decisions rely on the cumulative contribution of many tokens rather than a few dominant ones, yet saliency-guided synonym substitution and sentence-level paraphrasing can flip its predictions while altering only a moderate fraction of the text, in some cases yielding a successful jailbreak against the underlying LLM. A dataset-scale saliency analysis further shows that undetected injection prompts systematically lack the lexical markers the classifier relies on. We discuss the implications of these findings for the design and evaluation of classifier-based guardrails, and argue that explanation methods intended to support transparency can simultaneously lower the cost of constructing successful adversarial bypasses. |
| 2026-09-21 | [Construting Reverse Thinking: Developing Large Language Models' Reverse Thingking Ability](http://arxiv.org/abs/2609.24760v1) | Xin Liu, Yunhai Li et al. | When facing complex problems, humans tend to try various ideas for different issues. Human thinking patterns exhibit remarkable flexibility in adapting to diverse scenarios. GPT-o1, GPT-o3, and DeepSeek-R1 adopt long chain-of-thought models to address complex problems by increasing reasoning depth, which default to a forward reasoning mode. We conducted statistical analysis on the accuracy of different mathematical problem datasets on models of different scales, and found five reasons for errors: Insufficient solution-space coverage, Computational mistakes, Unverified assumptions, Ignoring constraint conditions, Maximum response length limitation. To address the above issues, we proposed a backward reasoning pattern construction method aimed at enhancing the model's reverse thinking ability and dynamic adaptability. First, we constructed an easy-hard two-stage Math dataset for training large models and gradually improving their inference ability at different difficulty levels. The dataset contains forward reasoning paths as well as backward reasoning paths. And a two-stage supervised fine-tuning process is applied to progressively train the model's backward reasoning capability. Furthermore, a fine-grained reward mechanism is developed, employing smoothed reward signals to strengthen the model's ability to autonomously select thinking modes during the reasoning process, thereby avoiding reward hacking. A linear-decay balanced sampling strategy is designed to maintain a balance between forward and backward reasoning path samples during training, enabling the model to converge quickly and stably. Experimental results show that our method significantly improves reasoning efficiency and accuracy in tasks such as mathematical proofs, offering a flexible and efficient reasoning paradigm for solving complex problems. |
| 2026-09-21 | [MIRA: Real-Time Full-Duplex Human-Robot Interaction for Embodied Companions](http://arxiv.org/abs/2609.24547v1) | Lijian Lin, Ye Zhu et al. | Real-time embodied companion interaction requires a robot to infer user intent from streaming speech, generate timely responses, and execute expressive, interruptible motions. Existing systems typically decouple dialogue orchestration from gesture synthesis, relying on offline motion generation from complete audio. This separation leaves open how a deployed robot can dynamically synchronize response content, prosodic timing, and physical safety under incremental inputs and uncertain turn boundaries. We present MIRA, a unified framework for full-duplex embodied companion interaction. Given streaming user speech, dialogue history, and vocal affect, MIRA predicts both the response text and an explicit embodiment cue. Discrete social behaviors (\eg listening, greeting) are mapped to validated robot trajectories, while open-ended speaking is paired with streaming, co-speech motion. This generative motion is governed by a predict-more-than-commit sliding window that provides temporal look-ahead for motion continuity while limiting physical commitment to a short, cancellable prefix. Crucially, we design CORTEX, a dual-timescale interaction policy that manages low-latency streaming and deliberative turn decisions, backed by a robot-side execution layer that enforces physical safety constraints at the control rate. We deploy MIRA on an Astribot S1 humanoid robot. Quantitative evaluations demonstrate competitive audio-motion alignment relative to state-of-the-art motion-generation baselines, while real-robot deployment measurements characterize streaming responsiveness and interruption handling. |
| 2026-09-21 | [Conformalized Safe Feasible Sets in Uncertain Decision Systems](http://arxiv.org/abs/2609.24496v1) | Yajie Bao, Yinjie Min et al. | Safety-critical decision systems often require a downstream optimizer to choose from an unknown feasible set determined by an unobserved label $Y$. Given a context $X$, the goal is to construct a safe subset $D(X)$ contained in the oracle feasible set $A(X,Y)$ with probability at least $1-α$. Existing conformal approaches typically construct a prediction set of the unobserved label $Y$ and retain decisions that are safe for every value in this set. Although valid, this requires a stronger intermediate event than set inclusion. We propose Directed Inclusion Safety Calibration (DISC), a conformal framework that directly controls the probability of this inclusion event by reducing its verification to a scalar critical-inclusion score. Given a pretrained nested family of candidate feasible sets, DISC assigns each labeled observation the smallest nestedness level at which the corresponding subset is contained in $A(X,Y)$, and constructs the safe feasible set using an empirical quantile at test-time. With data exchangeability, this yields a finite-sample, distribution-free inclusion guarantee. Under two practical set families, we show that DISC produces a safe feasible set containing that obtained by the corresponding calibration baseline. We further develop optimization-based score computation and decision-aware procedures for learning subset families. Experiments across continuous and structured decision problems show that DISC achieves the target inclusion guarantee while producing larger feasible regions. |
| 2026-09-21 | [Multi-Agent Transportation of Free-Flyers in Microgravity Via Pushing Interaction Under Human-in-the-Loop Control](http://arxiv.org/abs/2609.24376v1) | Gregorio Marchesini, Nicola De Carli et al. | We propose a safety-critical framework for the cooperative transportation of passive targets in microgravity, where a team of chaser robots acts through unilateral pushing contacts to track a human-provided desired twist while ensuring safe target motion. The pushing-only nature of the interaction introduces sparse, configuration-dependent actuation constraints requiring chasers to physically relocate on the target body when the desired pushing allocation changes. To address these challenges, we formulate a delay-aware feedback control architecture leveraging Control Lyapunov Function (CLF) and Control Barrier Function (CBF) constraints within a mixed-integer thrust allocation program to enforce stability and safety of the target, respectively. The proposed framework enables reference tracking while guaranteeing obstacle avoidance with a circular obstacle despite intermittent control authority, providing a foundation for human-supervised cooperative transportation of free-flyers in space environments. The proposed framework is validated through Gazebo simulations. |
| 2026-09-21 | [Explainable Predictive Condition-based Maintenance of Naval-Propulsion Systems using Fuzzy Logic](http://arxiv.org/abs/2609.24250v1) | Dionisis Kalogeropoulos, Georgia Sovatzidi et al. | The shipping industry has a significant impact on the global economy, emphasizing the need for operational availability and safety through the use of effective maintenance techniques. During the last decades, predictive maintenance (PdM) has emerged as a promising solution compared to the existing conventional maintenance systems. This is because it offers several advantageous functions, such as damage predictions for vessel components, reduced downtime, improved and extended life of machinery, as well as higher safety during voyages. However, existing methodologies developed for performing PdM do not provide explanations of their results to users, so that they can understand the failures that may occur. To address this limitation, this paper proposes a novel framework based on a fuzzy decision tree and a deep residual neural network, aiming to perform explainable PdM on naval vessels. The proposed framework is able to generate fuzzy local rules based on the dataset used, and can provide explanations of its outcomes, using cause-and-effect relationships, in a way that are understandable to users, thereby gaining their trust. Experiments using a publicly available dataset demonstrate the effectiveness of the proposed framework, as it achieves an accuracy of 99.24%. |
| 2026-09-21 | [Reinforcement Learning Inspired Black-box Adversarial Attacks for Computer Vision](http://arxiv.org/abs/2609.24249v1) | Florian Krone, Elena Hoemann et al. | Neural networks, both convolution or transformer based, are essential for modern computer vision systems. However, they are vulnerable to small perturbations, almost imperceptible to humans, which significantly alter the model's prediction. These adversarial attacks are often considered to be a significant threat to the implementation of neural networks in safety-critical applications. Most attacks utilize the white-box threat model and therefore require full access to the target model, making them unrealistic to use in practice. We propose a novel approach under the more realistic black-box threat model that utilizes concepts from reinforcement learning to optimize perturbations with a non-differentiable target model. Reinforcement learning algorithms have already been optimized to be query efficient, making them an ideal starting point when designing black-box adversarial attacks. We show the success of our reinforcement learning inspired black-box adversarial attack (RIBA) in generating adversarial perturbations using only a small number of queries to the target model, by comparing it to state of the art attacks on different models on the Cifar10 and ImageNet data sets. RIBA takes $25.4\%$ fewer median queries to generate attacked images against a ResNet-18 on Cifar10 and $22.5\%$ fewer median queries to fool a Vit-B/16 model on ImageNet. Additionally, we demonstrate that RIBA can match the performance of white-box attacks on an adversarially trained model. |
| 2026-09-21 | [APEXA: Execution-Integrity Enforcement for Multi-Agent LLM Automation of Synchrotron Data Reduction](http://arxiv.org/abs/2609.24165v1) | Pawan K. Tripathi, Hemant Sharma et al. | Synchrotron data reduction, detector calibration followed by azimuthal integration of terabyte-scale diffraction series, is a multi-step, expert-bound bottleneck that increasingly limits the science rate of user facilities. LLM agents promise to collapse it, but driving a real pipeline with a stochastic model creates a failure mode chat benchmarks cannot see: an agent can report a calibration that was never computed. Correctness here is a property of what executed, not of the transcript. We present APEXA, a deployed multi-agent framework (61 tools over heterogeneous compute, run as a single reasoning loop) automating calibration and integration from natural language at a major light source. We make three contributions. First, execution-integrity enforcement: a deterministic tool-layer guard that refuses to surface any result not backed by an executed tool call, with a parser tolerant of cross-model tool-call format drift: in deployment, a frontier model fabricated a complete calibration-comparison report for commands that never ran, which the guard converts to an explicit non-result; the same code gates an optional motor-control surface at 0/200 adversarial violations against a simulated IOC, versus 15/200 for an equivalent safety prompt. Second, we release APEXA-Bench, an evaluation harness of 58 facility tasks (50 base plus an 8-task cross-detector slice) organized by a four-class physical-consequence taxonomy, the first benchmark axis we know of separating a wasted compute cycle from a damaged instrument; its cross-detector grading against NIST-traceable lattice constants surfaced two latent pipeline bugs. Large-scale agent scoring is left to a full-length study. Third, we validate APEXA on real beamline data: from one natural-language prompt it recovers detector geometry and integrates a full attenuation/exposure sweep. We release the framework, harness and traces. |
| 2026-09-21 | [Monet: Measuring the Ecosystem of Open-Source Text-to-Image Models Tailored for Harmful Services](http://arxiv.org/abs/2609.24134v1) | Zihao Wang, Jiacen Xu et al. | The open-source text-to-image (T2I) ecosystem enables rapid model development and sharing, but also hosts models intentionally tailored for harmful services, which we call Monets. Prior work has examined specific types of harmful T2I models on individual platforms, but a Monet does not exist in isolation. The broader Monet ecosystem, spanning model characteristics, cross-platform propagation, governance evasion, monetization, and downstream deployment, remains poorly understood.   In this study, we present the first systematic, ecosystem-level measurement of Monets. Grounded in the policies of real-world model hubs, we construct a taxonomy of ten harmful service categories and identify 23,947 Monets across eight major T2I model hubs, with the most popular exceeding 19 million downloads. While some developers employ anti-theft mechanisms against unauthorized re-uploading, Monets propagate across platforms at scale, with 40.76% mirrored across hubs. Such propagation further enables governance evasion via cross-platform archiving, keeping 11.99% of Monets accessible after bans on their original platforms, alongside other evasion strategies including keyword obfuscation and model-level safeguard circumvention. Monets also anchor coordinated commercial campaigns---one spanning 668 models with 914 completed commissions and another advertising gray-market account-farming service---and reach users through GitHub projects and inference APIs, raising downstream child safety concerns. These findings expose the limitations of platform-siloed defenses and highlight the need for cross-platform threat intelligence, coordinated governance, and technical safeguards. |
| 2026-09-21 | [A$^2$Safe: Counterfactual Evidence-Aligned Adaptive Agent Collaboration for Safe and Effective Visual Question Answering](http://arxiv.org/abs/2609.24098v1) | Quanxing Xu, Ling Zhou et al. | Visual Question Answering (VQA) with Multimodal Large Language Models (MLLMs) requires not only producing safe and effective responses, but also grounding safety decisions in the multimodal evidence that determines risk. Recent safety-alignment methods improve refusal behavior and contextual risk awareness, yet correct safety outcomes may still rely on superficial textual or visual correlations, particularly when risk emerges from interactions between individually benign image and question content. To address this issue, we propose A$^2$Safe, a counterfactual evidence-aligned adaptive agent collaboration framework for safe and effective VQA. A$^2$Safe organizes localized visual observations, textual intent, and cross-modal risk relations through a Grounded Safety Evidence Board, making the basis of safety decisions explicit. Counterfactual safety evidence alignment enforces invariance to safety-irrelevant changes while requiring appropriate safety-state and response-mode transitions when risk-critical evidence is minimally altered. The resulting evidence state further supports adaptive collaboration, enabling direct answering when grounded evidence is sufficient and invoking policy critique and response revision when evidence is risky, uncertain, or conflicting. Under complementary safety-critical and general VQA protocols, A$^2$Safe achieves a 95.72 SIUO safety score, reduces the benign refusal rate on MOSSBench to 14.67%, and maintains an average general VQA score of 78.34 with 27.8% token overhead. These results support counterfactual evidence-aligned adaptive collaboration for safe and effective multimodal question answering. |
| 2026-09-21 | [Incremental Consistency Execution for Autonomous Intelligent Systems](http://arxiv.org/abs/2609.24090v1) | Cheng Li, Jiexiong Liu et al. | Long-horizon autonomous intelligent systems rely on heterogeneous components such as large language models, databases, external APIs, and rule engines, while their external states continuously change during execution. Re-executing the entire workflow after every change introduces substantial redundant computation. This paper proposes an incremental consistency execution method based on task fact contracts, field-level dependency masks, and state perturbation result invariant domains. After an initial verified execution, the system constructs conservative invariant domains for critical inputs and uses them to determine whether downstream results can be safely renewed without re-invoking expensive components. When re-execution is required, only the smallest affected output fields are recomputed, and an equivalence barrier prevents unnecessary downstream propagation. A submission-time version consistency gate further ensures the safety of side-effecting actions. Experiments on industrial fault diagnosis, enterprise analytics, and LLM-based multi-tool assistants show that the proposed method significantly reduces expensive component calls and end-to-end latency while maintaining high consistency and low incorrect-reuse rates. |
| 2026-09-21 | [Safety Control of a Hyper-redundant Robot via Adaptive Weighted Control Barrier Functions](http://arxiv.org/abs/2609.24062v1) | Zijian Cai, Kiwan Wong et al. | Hyper-redundant robots are well suited for confined-space manipulation due to their high dexterity, but safe operation in cluttered environments remains challenging. In addition, their slender structures often lead to uneven load distributions and nonuniform tracking errors along the body. To address these issues, this work proposes a weighted control barrier functions (W-CBFs) framework that enforces safety constraints while reducing tracking errors caused by uneven loading. The proposed controller was first evaluated on a circular path-following task under different obstacle configurations. With fixed weights, compared to the non-weighted method, the maximum reduction in root-mean-square (RMS) tracking error was 59.6\% in simulation and 87.7\% in physical experiments. An adaptive weighting strategy was then investigated based on the discrepancy between simulated and experimental performance under different mapping functions. The RMS errors were further reduced by 21.9\% and 8.5\%, respectively, although the error increases when obstacles were located close to the robot body. Finally, the robot was evaluated in a cleaning task requiring coverage of a rectangular area and compared with manual teleoperation. Although the controller was not explicitly optimized for area coverage, the autonomous strategy achieved comparable or better coverage performance while avoiding collisions with the surrounding frame, whereas collisions occurred during manual operation. |
| 2026-09-21 | [Byzantine Causal Reliable Broadcast (BCRB) with Constant-Size Message Metadata](http://arxiv.org/abs/2609.24018v1) | Purv Patel, Ajay D. Kshemkalyani | Asynchronous Byzantine Reliable Broadcast (BRB) is a fundamental primitive that guarantees agreement and validity in distributed systems subject to Byzantine faults, but it lacks ordering guarantees. In this paper, we address Byzantine Causal Reliable Broadcast (BCRB), which builds on BRB to enforce causal message ordering. We present a novel BCRB protocol that decouples causal ordering from the BRB layer, achieving constant-size $\mathcal{O}(1)$ message metadata overhead and $\mathcal{O}(n^2)$ communication word complexity as against $\mathcal{O}(n^3)$ communication word complexity of existing protocols; here $n$ is the number of processes.   We present two variants of our protocol: a cryptographic version using a threshold encryption scheme and sequence gating, and its non-cryptographic version. In the cryptographic version, senders broadcast ciphertexts immediately, and decryption shares are piggybacked on out-of-band ACKs, preventing early decryption and front-running. In both versions, causal safety is achieved probabilistically. We evaluate the probability of causal safety violations using a random variable path analysis under independent exponential link delay distributions. We show that both variants satisfy liveness and the probability of weak safety violation is bounded by $\mathcal{O}(f^{-3}\cdot\ln^3 f)$, where $f$ is the upper bound on the number of Byzantine processes, and $f < n/3$ and $f=\mathcal{O}(n)$. Further, for the crypto version, we show that the probability of strong safety violation is bounded by $\mathcal{O}(f^{-1} \cdot \ln^2 f)$. We also show how to modify our two protocols to guarantee 100\% weak safety keeping $\mathcal{O}(1)$ message space overhead but with $\mathcal{O}(n^3)$ messages and $\mathcal{O}(n^3)$ communication word complexity. |
| 2026-09-21 | [Context-Aware Pre-Deployment Evaluation of AI Systems: A Regulatory Framework for Nigerian Fintech](http://arxiv.org/abs/2609.24016v1) | Andrew Anogie Uduimoh, Hadiza Umar Yusuf et al. | Commercial large language models are increasingly deployed across African fintech infrastructure for fraud detection and customer communication, yet no Nigerian or African continental regulatory instrument specifies what pre-deployment evaluation such systems must undergo before procurement. This paper reviews African fintech AI governance across global, continental, and Nigerian instruments, and shows that safety is affirmed as a principle while pre-deployment evaluation is operationally unspecified. Generic safety benchmarks cannot surface the failure modes most relevant to this domain, since none contain Nigerian institutional content or test for false positive misclassification of legitimate financial communications. These claims are demonstrated using SafeAlert, a purpose-built evaluation kit applied to six commercial models across three system prompt conditions. Results show that models resisting generic harmful content requests still produce complete fraud scripts under specific framing, and that several models misclassify most legitimate Nigerian bank communications as suspicious or fraudulent, a failure invisible to standard safety evaluation. The paper concludes with a regulatory framework proposing pre-deployment evaluation requirements for the CBN, NITDA, SEC, and the AU, arguing that the identified gap reflects an absence of regulatory specification, not a shortage of technical or financial resources. |
| 2026-09-20 | [Measuring the Assistant's Harmlessness Preferences on the User Turn](http://arxiv.org/abs/2609.23935v1) | Jord Nguyen | Post-training turns a general next-token predictor into a chat model with a persistent assistant persona. If that persona is a character the model plays only on its own turns, its preferences should govern what the assistant says, not what the model predicts other speakers will say. We test this boundary and find that it does not hold: a safety-relevant preference of the assistant---for harmless over harmful tasks---shapes the model's predictions even on the user's turn, where the assistant is not the one speaking. We find that this preference is small or near-zero in pretrained base models, that it emerges through post-training, replicated across open-weight model families, grows with scale, and can be moved by narrow finetuning that never touches user turns. We claim that this is evidence that post-training does not merely install a shallow assistant persona, but instead generalises beyond just the local assistant turn, into the model's representation of the user. |

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



