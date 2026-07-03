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
| 2026-07-02 | [Online Safety Monitoring for LLMs](http://arxiv.org/abs/2607.02510v1) | Mona Schirmer, Metod Jazbec et al. | Despite alignment training, LLMs remain prone to generating unsafe outputs at deployment time. Monitoring outputs online and raising an alarm when safety can no longer be assumed is therefore critical. We study a simple real-time monitor that turns a verifier signal from an external model into an alarm decision by thresholding, with the threshold calibrated via risk control. In experiments on mathematical reasoning and red teaming datasets, we show that this simple design is competitive with more advanced monitors based on sequential hypothesis testing. |
| 2026-07-02 | [Controllable Sim Agents with Behavior Latents](http://arxiv.org/abs/2607.02496v1) | Juanwu Lu, Junyu Zhu et al. | Realistic traffic simulation requires agents that imitate logged behavior and can also be steered along interpretable axes. Such controllability enables engineers to isolate variables, reproduce specific edge cases, and test autonomous systems without real-world risk. We introduce Controllable Neural Variational Agents (CNeVA), a controllable simulated-agent framework that learns to infer a per-agent Gaussian behavior latent from per-channel discounted returns via a closed-form conjugate variational update, conditioning a rectified-flow trajectory generator trained on a mixed channel-mask curriculum for classifier-free guidance. To tackle scarcity in reward signals, we propose soft eligibility gates that replace hard binary thresholds with smooth exponential decay, preserving the gradient signal for near-threshold agents. On the Waymo Open Motion Dataset, CNeVA attains competitive realism on the benchmark while exposing per-channel controllability that the higher-ranked imitation models lack. Speed- and acceleration-based steering produces monotone responses without stall-induced reward hacking. Safety controllability is monotone and substantial with the introduction of soft eligibility. We manage to achieve steerable map compliance under a context-residual return measure. Furthermore, our experiment demonstrates that steering metrics must be read alongside physical-plausibility guardrails to avoid reward-hacking confounds. |
| 2026-07-02 | [Towards Robustness against Typographic Attack with Training-free Concept Localization](http://arxiv.org/abs/2607.02494v1) | Bohan Liu, Wenqian Ye et al. | Models trained via Contrastive Language-Image Pretraining (CLIP) serve as the foundational vision encoders for most modern Large Vision Language Models (LVLMs). Despite their widespread adoption, CLIP models exhibit a critical yet underexplored failure mode: irrelevant text appearing within images confounds visual representations, biasing them toward lexical meaning rather than true visual semantics. This robustness issue, commonly described as a Typographic Attack (TA), exposes a vulnerability that poses a significant risk to safety-critical applications such as autonomous driving. To achieve interpretable and effective robustness against TA, we propose a novel, training-free mechanistic interpretability method. Our method provides sampling-based interpretations of hidden state representations and quantitatively attributes semantic versus lexical focus to individual attention heads. Through probabilistic analysis and circuit mining, we isolate specific Vision Transformer (ViT) components that disproportionately encode lexical information, thereby identifying the mechanistic source of TA. We further show that simple interventions applied directly to the identified circuits, without any additional training, can substantially improve robustness against Typographic Attacks in object classification. These interventions, such as selective adjustment of attention weights, also outperform both supervised and training-free defense methods. Our experiments demonstrate that applying the proposed intervention to the vision encoders of several state-of-the-art LVLMs yields substantial gains in Visual Question Answering accuracy under Typographic Attack interference on RIO-Bench. These results confirm both the efficacy and the generalizability of our mechanistic approach. Code is released at https://github.com/Liu-524/SamplingTAR. |
| 2026-07-02 | [Docking of Autonomous Vehicles with a Stationary Docking Station in 3D Space](http://arxiv.org/abs/2607.02478v1) | Ram Milan Kumar Verma, Shashi Ranjan Kumar et al. | In this letter, we present a strategy for autonomous docking of autonomous vehicles in three-dimensional space. Docking is a safety-critical task and requires expert piloting skills. Vehicles with autonomous docking capabilities are highly desirable in various applications, such as marine vehicle docking, aerial vehicle docking, spacecraft docking, and landing. To dock autonomously with the docking station, the vehicle must align itself to a specific desired orientation relative to the docking station and also reduce speed as it approaches. The vehicle achieves near-zero speed to dock successfully and safely without colliding with the docking station. Inspired by the philosophies from the guidance literature, we present a finite-time sliding mode-based strategy to achieve the same. The range and line-of-sight kinematics relations describing the motion of the vehicle with respect to the stationary docking station are used to steer the vehicle to achieve the desired orientation for docking. This docking strategy is validated in MATLAB\textsuperscript{\textregistered} simulations for various initial locations and orientations of both the vehicle and the docking station. |
| 2026-07-02 | [Fast Multi-dimensional Refusal Subspaces via RFM-AGOP](http://arxiv.org/abs/2607.02396v1) | Thomas Winninger | Steering and monitoring activations in Large Language Models (LLMs) are increasingly used for both safety and interpretability. Early work assumed behaviours are encoded along single linear directions, but recent findings suggest complex behaviours, such as the refusal to answer harmful queries, live in multi-dimensional subspaces. However, existing methods for extracting these subspaces are computationally expensive, which becomes prohibitive on reasoning models who produce long reasoning traces. By adapting the Recursive Feature Machine (RFM) algorithm -- which can be computed efficiently -- with a probe-informed initialization, we are able to identify the multi-dimensional refusal subspace in seconds, on reasoning (Qwen 3) and non-reasoning (Qwen 2.5) models. While RFM allows for faster subspace identification, it also showed better performances on the ablation task than its alternatives. More work is planned to better understand the relations between subspaces found by different methods. If confirmed, RFM could be a cheap and scalable complement to existing subspace-extraction methods in LLMs. |
| 2026-07-02 | [Generative Autonomous Grid Control: Integrating Decision Transformers with a Two-Stage Safety Stack](http://arxiv.org/abs/2607.02379v1) | Mohamed Shamseldein | The displacement of synchronous generation by inverter-based resources is accelerating power system frequency dynamics beyond the response capability of conventional automatic generation control. This paper presents Autonomous Grid Generation Control with Decision Transformers, a framework coupling an offline-trained Decision Transformer with a twostage symbolic safety stack for secondary frequency control. The Decision Transformer learns a conditional dispatch policy from offline supervisory control and data acquisition records via sequence modeling, eliminating online exploration risks. A Constraint Verification Unit provides sub-ten-millisecond algebraic screening using real-time power transfer distribution factors, while an aggregate digital twin performs swing-equation-based dynamic stability certification. Validated on the Northeast Power Coordinating Council 140-bus system under low-inertia conditions, the proposed controller reduces the area control error integral by over 99% relative to tuned automatic generation control, maintains a 59.4 Hz frequency nadir, and achieves inference latency of approximately 10 ms, well within real-time constraints. Comparative evaluation against a linear quadratic regulator baseline and structural analysis against conservative Q-learning demonstrate the advantages of the sequence-modeling formulation. Small-signal eigenvalue analysis characterizes the dominant 1.87 Hz electromechanical mode and confirms that the safety stack maintains stable operation across operating points. By falling back to tuned automatic generation control whenever proposals are rejected, the safety stack bounds worst-case performance to industry-standard levels in simulation. |
| 2026-07-02 | [Hardware-Enforced Semantic Coordination for Safety-Critical Real-Time Autonomous Systems](http://arxiv.org/abs/2607.02376v1) | Uwe M. Borghoff, Paolo Bottoni et al. | Recent advances in agentic AI are producing increasingly complex autonomous systems that integrate large language models, world models, optimization engines, specialized neural architectures, autonomous platforms, and human operators. While much current research focuses on improving reasoning capabilities, safety-critical real-time deployment also requires bounded and verifiable coordination among heterogeneous components operating concurrently under uncertainty. Software-mediated coordination presents fundamental limitations in domains where bounded latency, deterministic coordination, and enforceable safety guarantees are essential.   Hence, we propose a hardware-enforced semantic coordination architecture in which selected coordination semantics are implemented directly at the hardware level via field-programmable gate arrays (FPGAs). The approach builds on the Topic-Based Communication Space Petri Net (TB-CSPN) framework, which separates semantic reasoning from interaction management.   In this approach, selected TB-CSPN coordination mechanisms are mapped onto FPGA primitives, creating a hardware-native semantic coordination layer. Focus is not on acceleration, but on enforcing temporal synchronization, semantic gating, authorization constraints, and bounded coordination behavior directly in hardware. Semantic reasoning remains adaptive and software-driven, while embedded coordination semantics become deterministic. |
| 2026-07-02 | [Search-based Testing of Vision Language Models for In-Car Scene Understanding](http://arxiv.org/abs/2607.02300v1) | Lev Sorokin, Chen Yang et al. | In the automotive domain, in-car scene understanding (ISU) enables the detection of safety-critical events, such as driver distraction, and supports drivers or passengers by analyzing the in-car scene and adapting the environment (e.g., ambient lighting). The industry is increasingly exploring vision-language models (VLMs) to interpret camera-recorded in-car scenes and extract information for downstream reasoning tasks. However, VLMs may generate incomplete, erroneous, or misleading scene descriptions, highlighting the need for systematic testing. Collecting real in-vehicle data is costly, difficult to scale, and often infeasible, particularly in early design stages. In this paper, we present ISU-Test, an automated testing approach that combines rendering-based scene generation with search-based testing to evaluate ISU systems. By framing testing as an optimization problem and systematically modifying scene parameters, our method generates diverse in-car scenarios and explores a wide range of configurations. We evaluate ISU-Test on both an industrial prototype and open-source VLMs across two case studies: question answering and captioning, comparing against randomized scenario generation. Results show that ISU-Test significantly outperforms the baseline, achieving up to 10 times higher failure rates and up to 3.6 times higher failure coverage. |
| 2026-07-02 | [Coding Agents Are Guessing: Measuring Action-Boundary Violations in Underspecified DevOps Instructions](http://arxiv.org/abs/2607.02294v1) | Zimo Ji, Zekai Zhang et al. | LLM coding agents are increasingly deployed to act autonomously on real production infrastructure. They execute shell commands, modify repositories, and call operational APIs. However, completing a task is not sufficient for safety. A wrong action can cause severe consequences. Existing agent benchmarks largely emphasize task completion, leaving open how agents behave under benign but underspecified instructions.   We present UnderSpecBench, a benchmark for measuring action-boundary violations in coding agents (i.e., Claude Code, Codex, and OpenCode) on DevOps tasks. UnderSpecBench includes 69 task families grounded in documented incidents, CVEs, or tool behavior and organized across four DevOps capability domains and nine operational control surfaces. To isolate underspecification from task difficulty, each task keeps the same environment and ground-truth safe action while varying the instruction along three axes: intent clarity, target certainty, and blast radius. The resulting 2,208 prompt variants are evaluated with deterministic, side-effect-based oracles that separate Safe Success, Wrong Target, and OverScope outcomes; non-action runs are further classified as clarification, refusal, or deferment.   Across five agent x model configurations using OpenCode, Claude Code, and Codex, the evaluation results show that underspecification does not mainly make agents fail; it makes them guess. 55.8-67.8% of runs violate at least one boundary. Target underspecification sharply degrades action quality, while blast-radius cues barely reduce action propensity. These findings show that completion-centric evaluation can overstate safe autonomy and motivate mitigations at the model, harness, and system layer. |
| 2026-07-02 | [NEUROSYMLAND: Neuro-Symbolic Landing-Site Assessment for Robust and Edge-Deployable UAV Autonomy](http://arxiv.org/abs/2607.02277v1) | Weixian Qian, Tianyi Yang et al. | Safe landing-site assessment in unstructured environments remains a key challenge for autonomous UAV deployment, as vision-only learning approaches often degrade under terrain variability and provide limited transparency in safety decisions. We present NEUROSYMLAND, a neuro-symbolic landing-site assessment system that integrates lightweight perception with explicit safety reasoning. The framework constructs a probabilistic semantic scene graph from onboard visual input and evaluates candidate landing regions using symbolic constraints capturing terrain flatness, obstacle clearance, and spatial consistency, enabling structured reasoning under perceptual uncertainty while maintaining edge-feasible execution. Across 72 simulated landing scenarios spanning diverse terrains, NEUROSYMLAND achieves 61 successful assessments, outperforming four competitive baselines (37-57 successes). To evaluate deployability, we further conduct 100 hardware-in-the-loop trials with randomized initial poses, profiling end-to-end latency, stage-wise execution time, and system-level metrics including CPU/GPU utilization, memory footprint, and power consumption. Results demonstrate improved robustness and interpretability with bounded edge-resource usage. Profiling shows that symbolic reasoning contributes only a small fraction of end-to-end latency, while the main computational cost arises from perception and PSSG construction. These results demonstrate the feasibility of deploying the landing-site assessment stack on edge-constrained UAV hardware, and all source code, datasets, prompts, and symbolic rule refinement examples are released in an open-source repository |
| 2026-07-02 | [Cadence: Extreme Pipelining with Multiple Concurrent Proposers](http://arxiv.org/abs/2607.02275v1) | Fatima Elsheimy, Mohammad Mussadiq Jalalzai et al. | We present Cadence, a Byzantine fault-tolerant multi-proposer consensus protocol with arbitrarily low block intervals, optimal resilience, and optimal fast-path latency. Cadence divides time into equally spaced slots, one block per slot, each finalized in its own consensus instance. Blocks do not build directly on their predecessor, so instances run independently and none waits for an earlier block to finish or propagate; we call this extreme pipelining, decoupling the block interval from network latency. Cadence also removes the single-leader monopoly over transaction inclusion and ordering: under multiple concurrent proposers (MCP), several validators propose for each block, and it guarantees that, under synchrony, a transaction a correct proposer includes cannot be censored or deferred (short-term censorship resistance), and that no proposer can craft its proposal in reaction to the others' (hiding). To realize extreme pipelining, we introduce a general framework that turns any one-shot consensus meeting our slot-consensus specification into a multi-shot protocol. We instantiate it for MCP with two protocols of our own: Chorus, a slot consensus whose fast path finalizes a block in an optimal three rounds, with speculative finality one round earlier, and Conductor, an orchestrator that opens slots at an even cadence, more slowly under asynchrony to keep open slots bounded. To our knowledge, Cadence is the first MCP protocol to provide short-term censorship resistance and hiding at the fast-path latency of single-leader consensus. We prove safety, liveness, censorship resistance, and hiding under partial synchrony with optimal resilience (n = 3f+1). In simulation over Monad's 200 validators with five proposers per slot, finalization averages 219 ms (167 ms to speculative finality); at a 100 ms block interval a transaction waits on average 50 ms to enter a proposal. |
| 2026-07-02 | [Copewell: A Multi-Agent Swarm Architecture for Equitable Mental Wellness Support](http://arxiv.org/abs/2607.02245v1) | Seren Yenikent, Jack Vinijtrongjit et al. | Mental health disorders affect nearly one billion people globally, yet 75% of individuals in low- and middle-income countries receive no treatment due to workforce shortages, cost barriers, and stigma. Current AI-powered wellness solutions predominantly rely on single-mode conversational interfaces that suffer high abandonment rates and fail to provide measurable, immediate relief calibrated to users' dynamic emotional states. This paper presents Copewell, a novel multi-agent swarm system designed to expand access to mental wellness support through human-centered AI principles. Our architecture introduces three technical innovations: (1) a multi-source assessment framework integrating self-reported, physiological, and contextual data to mitigate algorithmic bias; (2) valence-arousal emotion mapping using Russell's Circumplex Model of Affect to route users to specialized AI agents; and (3) dual-mode intervention delivery combining conversational support with evidence-based sensory wellness protocols. We examine the sociotechnical design considerations underlying Copewell's development, including a privacy-first architecture, embedded ethical oversight through a dedicated Ethics Supervisor agent, and participatory design informed by mental health practitioners. Early practitioner engagement and beta deployment inform design decisions and identify directions for future empirical evaluation. This work contributes to responsible AI discourse by demonstrating how technical architecture can operationalize equity and safety principles from inception. |
| 2026-07-02 | [Reference-Governed Distributed Safe Gradient Flow for Safe Optimal Output Agreement of Multi-Agent Systems](http://arxiv.org/abs/2607.02192v1) | Zhanglin Shangguan, Wei Xiao et al. | This paper studies safe optimal output agreement for nonlinear multi-agent systems with output safety constraints. Existing safe feedback optimization methods often implement gradient-flow dynamics directly through the plant input, which may require high-order control barrier functions (HOCBFs). The resulting derivative-chain design is tuning-sensitive and can introduce additional equilibrium conditions that alter the steady-state optimal solution. We propose a reference-governed two-layer architecture that separates lower-layer output regulation from upper-layer distributed optimization. The upper layer filters the reference gradient flow through first-order control barrier function constraints, which are easier to tune and preserve the steady-state optimality structure of the original agreement problem. The lower layer uses an internal-model-based output regulator with a reference-dependent Lyapunov function, from which dynamic safety margins (DSMs) are constructed to certify transient output safety. We prove forward invariance, optimal-solution preservation under DSM-compatibility conditions, and convergence via a Lyapunov small-gain argument. Simulations validate safe convergence, show advantages over HOCBF-based feedback optimization, and demonstrate adaptive tangential objective shaping for escaping spurious equilibria induced by nonconvex obstacles. |
| 2026-07-02 | [Behind the Refusal: Determining Guardrail Activation via Behavioral Monitoring](http://arxiv.org/abs/2607.02121v1) | William Hackett, Peter Garraghan | As Large Language Models (LLMs) and agentic systems become integrated into real-world applications, ensuring their safety and security is critical. Guardrail systems that detect and block malicious instructions sent to and from an LLM are an essential component of AI security. However, researchers conducting black-box adversarial emulation against production AI systems often struggle to determine whether a guardrail block or an LLM rejection has occurred. This distinction is important because the techniques used to bypass guardrails can differ substantially from those used to bypass LLM safety alignment, and has a material impact on attack technique selection and optimization. We propose the first black-box guardrail reconnaissance methodology, which detects the presence of a guardrail within a target AI system through behavioral monitoring of HTTP, lexical, and timing signals, assuming only black-box access and zero prior knowledge of the guardrail or AI system. Experiments demonstrate that our approach detects guardrail presence with 100% accuracy, with statistically significant behavioral separation between benign and malicious interactions (q < 0.001). Our approach further identifies the content categories a guardrail is designed to block, and distinguishes guardrail blocks from LLM rejection on unseen prompts with an average F1 score of 98%. |
| 2026-07-02 | [HaloGuard 1.0: An Open Weights Constitutional Classifier for Multilingual AI Safety](http://arxiv.org/abs/2607.02079v1) | Navaneeth Sangameswaran, Preetham S et al. | We present HaloGuard 1.0, an open-weights implementation of the constitutional-classifier paradigm for input safety. It achieves state-of-the-art performance on English and multilingual prompt-safety benchmarks at roughly one-tenth the model size of current leading open guard models. The safety constitution is the organising structure of the corpus: a natural-language constitution of 46 policies and 2,940 subcategories drives synthetic data generation, with exhaustive one-to-one paired counterfactuals that hold topic and vocabulary fixed while flipping intent, a two-tier harmless design that separately targets boundary and baseline false positives (FPs), and balanced multilingual materialisation across 46 languages that treats language as a surface form appearing on both sides of the boundary rather than as an adversarial signal. Across seven prompt-safety benchmarks, HaloGuard 1.0-0.8B attains the best average F1 (90.9) of any open guard we evaluate, outperforming baselines up to 27B parameters (over 30 times larger) while holding false-positive rate (FPR) to 4.3 and false-negative rate (FNR) to 9.5. The HaloGuard 1.0-4B variant reaches average F1 of 92.1 and FPR of 3.5, spending its extra capacity on precision rather than recall. A structured adjudication of the remaining failures indicates that most apparent missed-harm cases are benchmark mislabels rather than genuine model misses. An always-on adversarial red-teaming protocol continuously hardens the guard against both content-level and agentic attacks. We release the models as open weights. |
| 2026-07-02 | [kNNGuard: Turning LLM Hidden Activations into a Training-Free Configurable Guardrail](http://arxiv.org/abs/2607.02072v1) | Mahmoud Abdelfattah, Hamid Nasiri et al. | Large language models (LLMs) are increasingly deployed in domains requiring guardrails to detect unsafe, off-topic, or adversarial prompts. Existing guardrails predominately rely on fine-tuning to build classifiers, which often suffer from low generalization and high inference latency. We present kNNGuard, a training-free guardrail that utilizes the activation space of an off-the-shelf LLM. Given a small bank of 50 safe and unsafe prompts, kNNGuard extracts hidden activations and performs multi-layer kNN fusing activation-space and embedding-space scores for classification. Across six domains spanning topical and security prompts, kNNGuard achieves competitive or superior F1 compared to fine-tuned state-of-the-art guardrails while running 2.7x faster than the best comparable guardrail, and 10x faster than a fine-tuned safety classifier without gradient updates or fine-tuning. Domain adaptation requires only updating the labeled bank, which can be constructed in under 10 seconds and several orders of magnitude faster than established guardrails. We also analyze the impact of system prompts, layer selection, and integration into production LLM pipelines as a configurable, low-latency guardrail. |
| 2026-07-02 | [OpenSafeIntent: Evaluating Intent-Calibrated Safe Completion Across Dual-Use Prompt Sets](http://arxiv.org/abs/2607.02047v1) | Rheeya Uppaal, Seungwoo Lyu et al. | Safe completion requires models to provide useful assistance without enabling harm, but this behavior is difficult to evaluate with isolated prompts. We introduce OpenSafeIntent, a benchmark of controlled prompt-sets that vary intent while holding the underlying task fixed. Each datapoint contains benign, dual-use, and malicious variants of the same task. This design lets us evaluate whether models calibrate assistance across intent shifts, rather than merely appearing safe on average. Across a broad model suite, we find that prompt-level safety hides important failures: models often fail to remain safe across matched intent variants, dual-use behavior is brittle under paraphrase, high-level answers on risky topics are not reliably safe, and responses that reframe ambiguous requests into safer tasks are substantially less likely to cross the safety boundary. Our results suggest that safe completion should be evaluated as intent-calibrated behavior over controlled task variants, not as a single safety-helpfulness tradeoff over independent prompts. |
| 2026-07-02 | [Fast and Accurate Anomaly Detection in Time Series](http://arxiv.org/abs/2607.02046v1) | Emanuele Mele, Massimo Cafaro et al. | Anomaly detection is a critical and evolving field in Machine Learning, with applications targeting different domains such as cybersecurity, finance, healthcare, manufacturing and IoT (Internet of Things) systems. Traditionally, anomaly detection algorithms have been designed using both supervised and unsupervised learning paradigms. The fundamental challenge in real-world anomaly detection scenarios is related to the inherent class imbalance (anomalies are typically rare) and, for supervised methods, to the scarcity of labelled anomalous data. Indeed, labelling is both expensive and time-consuming. Conversely unsupervised methods do not require labelling, but may suffer from high false positive rates when deployed in safety-critical applications. In this work we introduce a novel unsupervised algorithm for anomaly detection in time series based on the Haar discrete wavelet and a suitably designed $t$-test. We establish the theoretical foundation of the proposed $t$-test and, through extensive experimentation across 343 datasets, demonstrate that our algorithm outperforms state-of-the-art unsupervised and self-supervised benchmarks. |
| 2026-07-02 | [Assessing VLM Reliability for Medical Image Quality Evaluation Under Corruption and Bias](http://arxiv.org/abs/2607.01973v1) | Sofiane Ouaari, Kevin Vorwalder et al. | Vision-Language Models (VLMs) are increasingly applied in medical tasks such as pathology description, report generation, and visual question answering. Medical Image Quality Assessment (MIQA) supports diagnostic accuracy and patient safety by determining whether images meet the standards required for clinical decision-making. Automating MIQA with VLMs may reduce workload, but their behavior under real-world conditions, where images may be degraded or textual context may affect judgments, should be further explored before deployment. We benchmark VLMs on medical image quality using the MediMeta-C dataset zero-shot across seven corruption types and five severity levels. We evaluate sensitivity to degradation patterns, the effect of corruptions on embedding geometry, and whether textual attributes (demographics, expertise, infrastructure, institution) alter scores. Across 16 VLMs and seven modalities, pixelation produced the largest score reductions (mean -20.58%, up to -34.4% for OCT), whereas brightness had limited effect (-0.81%). Embedding displacement was associated with score changes. Same-family models showed correlations of 0.67-0.83; some produced increases up to +31% for corrupted mammography. Textual attributes affected scores: institutional prestige raised them +17.15%, and equipment age lowered them -14.7%. The largest changes were +95.62% (InternVL-8B) and -37.7% (MedGemma). Current VLMs show limitations for medical image quality assessment. Pixelation, a privacy-preserving transformation, reduces performance, indicating a trade-off between patient privacy and reliability. Sensitivity to contextual metadata indicates limited objectivity and marks metadata as a privacy and bias source. Privacy protection and objective quality assessment are related requirements for use. |
| 2026-07-02 | [Robust for the Wrong Reasons: The Representational Geometry of LLM Robustness to Science Skepticism](http://arxiv.org/abs/2607.01951v1) | Minjong Cheon | Large language models (LLMs) are increasingly consulted on contested scientific questions, raising the concern that they will sycophantically retreat from established consensus when a user signals doubt -- drifting toward a false balance that treats settled science as one view among several. We test this across three open instruction-tuned models (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B), three consensus-science domains (climate, vaccines, evolution), and single- and multi-turn settings, combining behavioral measurement with linear probing and activation patching. We do not observe sycophantic retreat. Instead, models show three distinct policies under the same skeptical pressure: reactive assertion, where consensus assertion increases rather than decreases (Llama); surface hedging, where tone softens while the position holds (Qwen); and non-response (Mistral). Pairwise judgments confirm the reactive shift is stance, not style (63.6%, p=.007), and a decomposition identifies increased consensus assertion, not false balance, as its driver (beta=+0.042 per dose, p<1e-77). Linear probes localize the divergence to middle layers -- perfect separation in Llama and Qwen versus 72% in Mistral, with non-overlapping confidence intervals -- indicating the non-responsive model does not linearly represent the skepticism signal at all. Crucially, this robustness does not transfer: it attenuates across domains and, in the safety-critical vaccine domain, can reverse, with myth-rebuttal weakening under skeptical pressure. We synthesize these into a four-way taxonomy separating active from accidental robustness, and argue that behavioral evaluation alone cannot distinguish a model that resists skepticism because it understands the signal from one that only appears to resist because it fails to perceive it. |

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



