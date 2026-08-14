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
| 2026-08-13 | [SCULPT: Subtractive Composition for 3D Part Generation](http://arxiv.org/abs/2608.13541v1) | Sikuang Li, Chen Yang et al. | Part-aware 3D generation aims to create digital assets that are coherent as complete objects while exposing structural parts for editing, material assignment, animation, and reuse. Existing methods impose this structure outside the native generation loop: segmentation-based methods partition an already generated shape, while additive methods synthesize parts from predefined layouts, boxes, or tokens and then reconcile them into a whole. The former preserves the generated geometry but fixes the object before part boundaries are determined; the latter exposes part cardinality but often leaves shared boundaries vulnerable to gaps, interpenetrations, and material discontinuities. In this paper, we propose SCULPT, a framework that addresses these challenges through subtractive composition. Given a complete object represented in a structured 3D latent space, SCULPT iteratively applies a joint split predictor to generate one extracted part together with the remaining object. The predictor performs a coupled denoising process conditioned on both the image and the current 3D state, so the extracted part and updated remainder are generated together rather than reconciled after generation. The joint split predictor processes both outputs on the union of their native sparse 3D supports, allowing neighboring supports to overlap rather than imposing a disjoint voxel partition. The rollout ends when the remainder support becomes empty or reaches a fixed safety cap, allowing the number of generated parts to adapt to each object within that bound. Extensive experiments demonstrate state-of-the-art geometry on PartObjaverse while preserving strong complete-object reconstruction after part assembly. Results on four dataset images, one text-to-image-generated input, and one real-world photograph further show fine-grained textured part decomposition beyond the benchmark. |
| 2026-08-13 | [Safety vs. Social Image: Co-Designing Protection Mechanisms Against Ableist Harassment with People with Disabilities in Social Virtual Reality](http://arxiv.org/abs/2608.13532v1) | Kexin Zhang, Daniel Killough et al. | People with disabilities (PWD) increasingly use avatars to express disability identities in social virtual reality (VR), but greater visibility also invites targeted harassment. Existing safety features are often insufficient, overlooking PWD's experiences and needs. To address this gap, we co-designed protection mechanisms with 11 PWD to reveal their values and needs. Our research employed a social lens to interpret harassment behaviors and protection mechanisms. Inspired by Hall's Proxemics Theory that interpersonal distances indicate social intent and boundaries, we divided social VR spaces into four proxemic zones (Intimate, Personal, Social, and Public) and used them to structure our protection mechanism co-design. We also provided different protection mechanism probes (Inform, Educate, Consent, and Combat) to elicit participant preferences. Our study highlighted the role of social proximity in shaping PWD's harassment perception and protection preferences and revealing PWD's unique social values and needs (e.g., managing harassment with optimism and resilience, prioritizing social image over safety). We proposed design recommendations for protection mechanisms that protect PWD while maintaining their desired social images. |
| 2026-08-13 | [Safety-Critical Control for Quadrotor UAVs via Decentralized Navigation Functions](http://arxiv.org/abs/2608.13507v1) | Omayra Yago Nieto, Alexandre Anahory Simoes et al. | We study safety-critical control for teams of quadrotor UAVs driven by decentralized navigation functions under learned model uncertainty. These functions generate fully actuated translational reference forces, while quadrotors can only produce thrust along their body-fixed vertical axes. We construct a thrust-attitude implementation of the induced navigation forces and quantify its error with respect to the fully actuated reference dynamics. An aggregated robust HOCBF-QP safety filter minimally modifies the nominal thrusts while guaranteeing pairwise collision avoidance with high probability. |
| 2026-08-13 | [TraVEL: Trajectory-Guided Video Embedding Learning for Driving-Video Retrieval](http://arxiv.org/abs/2608.13495v1) | Yi-Chung Chen, Philip Jacobson et al. | Efficiently retrieving relevant clips from large-scale driving logs is essential for data curation, model development, and safety analysis. Structured and rule-based retrieval systems can explicitly target driving events, but typically require expert-defined rules, auxiliary data, and multi-stage perception pipelines. Multimodal embedding models offer a simpler and more efficient alternative by representing each video with a single searchable vector. However, general-purpose models often rely on shortcuts from static scene context and struggle to distinguish motion-centric events, such as turning left versus right or accelerating versus decelerating. In this work, we study how to adapt a general-purpose multimodal embedding model to driving-video retrieval. We first fine-tune Qwen3-VL-Embedding on paired clips and reasoning traces from nuReasoning using an InfoNCE objective. While this stage substantially improves overall retrieval, caption supervision alone remains insufficient for fine-grained motion understanding. We therefore introduce TraVEL (Trajectory-Guided Video Embedding Learning), a motion-aware fine-tuning framework that uses ego-trajectory similarity as a reward within Group Relative Policy Optimization. Trajectories serve only as privileged training supervision; retrieval still operates on single-vector video embeddings without ego poses, expert rules, or auxiliary perception outputs. We further construct a driving-video retrieval benchmark from nuReasoning. Experiments show that TraVEL improves motion-centric retrieval across model scales: relative to SFT, it raises longitudinal and lateral mAP by 9.8 and 4.7 points at 2B, with corresponding gains of 7.2 and 1.5 points at 8B. TraVEL thus combines physically grounded supervision with efficient embedding-based search. |
| 2026-08-13 | [Synthetic Persona Pretraining: Alignment from Token Zero](http://arxiv.org/abs/2608.13482v1) | Julian Minder, Viktor Moskvoretskii et al. | As language-model-based AI is increasingly deployed in autonomous settings, aligning its goals and values with those of humans becomes critical. Today, alignment, and the assistant identity itself, are typically introduced only after pretraining, once behavioral priors are already established. This can make values a thin overlay, rather than deeply rooted, and facilitate subsequent misalignment. Pursuing a different paradigm, we introduce Synthetic Persona Pretraining (SPP), which installs the desired assistant persona from token zero in pretraining. First, we annotate pretraining documents with value-aligned first-person reflections derived from a normative value constitution. Second, we pretrain via the standard cross-entropy loss on standard pretraining documents as well as their reflections, which installs the desired persona among a multitude of other personas. Finally, we post-train on user-assistant dialogue data, which binds this desired persona to the assistant identity, a process we call persona binding. By pretraining models up to 3B parameters on 500B tokens, we show that SPP improves constitution following and jailbreak robustness, and reduces the misalignment rate in out-of-distribution moral dilemmas, while preserving capabilities. Early intervention matters: compared with alignment from token zero, introducing SPP only at the end of pretraining yields weaker constitution adherence, does not shift value priorities, and leads to less aligned choices in dilemmas. This advantage depends on persona binding and, importantly, increases with pretraining budget. Overall, our results show that shaping values early is critical for alignment and establish pretraining-time persona interventions as an effective approach to do so. |
| 2026-08-13 | [LLM-Assisted Dynamic Threat Analysis for Attacker-Reachable Software Weaknesses in Autonomous Vehicles](http://arxiv.org/abs/2608.13450v1) | Md Wasiul Haque, Sagar Dasgupta et al. | Autonomous vehicles depend on large safety-critical software stacks, where weaknesses reachable from adversarial inputs may affect steering, braking, or other control decisions. Static analysis can identify candidate sites, but dynamically confirming exploitability requires executable test artifacts that are difficult to construct manually. We investigate whether large language models (LLMs) can automate this process for Autoware, an open-source autonomous-driving stack. We perform compiler-precise static analysis across 185 packages, identifying 1,375 decision rules, 2,274 validation checks, and 482 input-to-safety-output flows, from which we derive a weakness taxonomy and sample 740 reachable sites. Two local open-weight LLMs, a no-static-context ablation, and a naive-template baseline generate 3,700 artifact sets, which are compiled against the real build under sanitizers, repaired through compiler-in-the-loop feedback, and fuzzed when executable. The main result is a build-integration failure taxonomy showing that 80% of first-shot compilation failures arise from dependency wiring rather than program logic. The reasoning model compiled 64% of harnesses on the first attempt, compared with 6% for the code-specialized model. Repair achieved full object-compileability for the reasoning model only through extensive stubbing; fewer than half of its harnesses reached the fuzzer, and all 37 observed crashes originated in stubbed code rather than Autoware. No candidate weakness was dynamically confirmed within budget. These results show that build integration, not candidate generation or fuzzing, is the primary barrier to reliable LLM-assisted dynamic analysis of full autonomous-vehicle software stacks. |
| 2026-08-13 | [COBRA-DOSE: Copula-based Bayesian Model Averaging for Dose Selection](http://arxiv.org/abs/2608.13423v1) | Luke Hagar, Min Zhang et al. | Early-phase clinical trials for dose selection typically enrol few patients and aim to identify doses that are both safe and promising for further study. While traditional approaches identify the maximum tolerated dose, modern trials for targeted therapies often seek the optimal biological dose, defined as the lowest dose achieving sufficient biological activity with acceptable safety. In immunology settings, assessment of biological activity is based on multiple biomarkers or clinical endpoints. Clinicians leading dose-selection efforts would thus benefit from transparent summaries of the probabilities of observing combinations of biomarker outcomes across doses. However, such inference is challenging in small samples where complex modelling assumptions are difficult to verify. To address this limitation, we propose COBRA-DOSE, a framework for posterior predictive inference based on two endpoints that models dependence via copulas and accounts for uncertainty in both marginal distributions and dependence structures through Bayesian model averaging. This approach avoids reliance on a single model and yields interpretable quantities for clinical decision making. We demonstrate the performance of COBRA-DOSE using DEN-181, a phase I immunology trial in rheumatoid arthritis. We also provide a general implementation of our approach through the CobraDose package in R. |
| 2026-08-13 | [Rules or Character? Scaling Laws for AI Safety Design](http://arxiv.org/abs/2608.13345v1) | Satoshi Takahashi, Nobuji Kouno et al. | Artificial Intelligence (AI) safety systems combine character shaping (e.g., Reinforcement Learning from Human Feedback [RLHF], Constitutional AI), which modifies behavioral distributions at training time, with rule enforcement (e.g., output filters, safety classifiers), which blocks harmful outputs at inference time, yet little formal analysis exists on how their optimal balance should change as deployment scales increase. We introduce a stylized comparative-statics model that parameterizes safety design as a resource allocation alpha in [0,1] between these two approaches, incorporating scale-dependent filter degradation, common-mode failures, and character fragility -- the risk that shaped behavior degrades or collapses under novel conditions. Under a multiplicative Pareto damage model, we derive closed-form expected harm and supplement it with tail-risk (CVaR) analysis via Monte Carlo simulation. Across three scenarios (optimistic, moderate, pessimistic), the optimal alpha* is interior or at the rules-only boundary and shifts weakly toward character shaping as deployment scale T grows, from negligible (Delta alpha* = +0.01) to pronounced (Delta alpha* = +0.21) depending on scenario. The dominant parameter is the baseline character fragility rate p^(0)_frag, which shifts alpha* by 0.50 across its range -- far exceeding the effect of tail severity, filter quality, or common-mode failure probability. CVaR and expected-harm optima converge at large T. These results suggest that safety architecture decisions depend less on deployment scale per se than on the reliability of character shaping under distributional shift. |
| 2026-08-13 | [Integration-First Structural Coverage for Embedded Software:Trace-Based Evidence, Hybrid Runtime Analysis, and Cross-Variant Consolidation](http://arxiv.org/abs/2608.13322v1) | Alexander Weiss, Albert Schulz et al. | Structural coverage is widely used as evidence that testing is complete, yet in embedded projects it is predominantly collected at unit level, simply because that is where instrumentation and observability are inexpensive. This produces a mismatch. The most representative completeness signal would come from integration and system tests executed on the device under test, but classical instrumentation perturbs timing, memory footprint and concurrency behaviour, while purely trace-reconstructed coverage loses reliability for decisions and conditions as soon as the compiler optimizes aggressively. We address this mismatch from both ends. On the process side we describe an integrationfirst coverage strategy that treats integration and system tests as the baseline measurement and drives the residual gaps through an explicit closure loop, so that completeness is established as covered or justified rather than as covered alone. On the technical side we use embedded trace as the observation path and add hybrid runtime analysis (hRA): a minimal, semantics-preserving observability scaffolding that keeps decision and condition boundaries distinguishable in the trace stream of an optimized (-O3) build, while all coverage state and counting remain off-target. This converts object-to-source mapping from a heuristic reconstruction into reviewable evidence and makes branch, condition and MC/DC measurement practical on release-like binaries. Finally we describe Hyper Coverage, a consolidation layer that merges evidence across test levels, test runs, variants and build configurations, and that exposes source lines which remain untested in every relevant variant. |
| 2026-08-13 | [Refusing Intent, Not Form: Wrapper-Based Intent-Group Supervision for LLM Safety](http://arxiv.org/abs/2608.13304v1) | Ping Wu, Haibo Tong et al. | Safety tuning can improve harmful refusal, but models may learn surface-form shortcuts: wrapped harmful prompts bypass safety, while similarly wrapped benign prompts are over-refused. We propose Wrapper-Based Intent-Form Augmentation (WIFA), an automatic intent-group augmentation method that pairs wrapped harmful examples with structurally matched wrapped benign counterexamples, requiring no external teacher or manual per-wrapper intent labels. We use WIFA as a common data layer for two complementary fine-tuning routes: WIFA-Boost, a two-stage high-safety recipe, and Anchored Group-Consistent Refusal Training (A-GCRT), which regularizes refusal/compliance decision scores across same-intent wrappers and anchors harmful and benign groups on opposite sides of a margin. In the Qwen setting, WIFA-Boost reaches the strongest transformed-harmful refusal, while A-GCRT reduces OR-Bench over-refusal from 25.7\% for the base model to 17.4\%; reproduced baselines do not match these operating points. Llama results and ablations over data structure, two-stage order, and A-GCRT components support this intent-group interpretation without claiming universal below-base over-refusal. |
| 2026-08-13 | [Predictive Relative-Velocity Steering for Safe Robotic Manipulator Teleoperation in Dynamic Environments](http://arxiv.org/abs/2608.13284v1) | Changhao Hu, Zeyi Liu et al. | Recent advances in teleoperation have enabled robotic manipulators to perform dexterous, human-arm-like motions. However, human operators may fail to avoid suddenly appearing obstacles promptly and effectively, particularly under network latency or limited attention, thereby creating safety risks. To address this issue, we propose a lightweight and modular framework for proactive collision avoidance, operating directly at the end-effector velocity-command level. After preprocessing the point cloud, the framework first predicts potential collisions based on time-to-collision (TTC) with integrated overshoot protection, and subsequently rotates the relative-velocity vector using Rodrigues' rotation formula. The deflection changes only the direction of the relative velocity while preserving its magnitude, thereby mitigating the deadlock problem commonly encountered by conventional artificial potential field (APF) methods. The prediction module compensates for point-cloud processing latency introduced by complex teleoperation pipelines, while the lightweight design enables the high-frequency control required for teleoperation. Simulations across diverse scenarios show that the proposed method achieves a higher end-effector collision avoidance rate than the baseline methods. Experiments on a physical robotic system further validate its collision-avoidance effectiveness. |
| 2026-08-13 | [Follow the Norm: Accounting for Fine-Tuning and Prompt Effects on Model Rationales](http://arxiv.org/abs/2608.13250v1) | Long Hoang Nguyen, Brice Valentin Kok-Shun et al. | Normative datasets are often used to train and align AI systems, but the norms they contain can function as action-guiding patterns rather than neutral moral knowledge. We propose treating the AI system as a proxy actor and test whether dataset-level norms can shift it away from its baseline safety behavior when it faces high-conflict dilemmas. We make three contributions. First, we demonstrate in controlled experiments that norm-breaking fine-tuning yields norm-divergent actions justified by self-interested rationales, suggesting a systematic shift in patterns of justification. Second, we establish a practical audit trail linking downstream justifications to upstream norms using mixed methods. Third, we show that system prompts can both suppress and elicit these patterns. We conducted experiments on three models (LLaMA-3.2-11B, Qwen-3.5-9B, and Pixtral-12B) using Low-Rank Adaptation (LoRA) fine-tuning on Social Chemistry 101 Fairness/Cheating (norm-following vs. norm-breaking) with prompt steering. Across all three models, we find that norm-breaking fine-tuning shifts the model's default rationale style from safety compliance to instrumental self-interest, whereas system prompts can override this behavior. Our results support a distributed view of alignment in which observed behavior depends jointly on training data, fine-tuning, and prompting, motivating norm-aware documentation and rationale logging for contestable oversight. |
| 2026-08-13 | [Stream-based Online and Offline Monitoring under Measurement Noise](http://arxiv.org/abs/2608.13211v1) | Bernd Finkbeiner, Martin Fränzle et al. | Stream-based monitoring is a runtime verification approach for cyber-physical systems that translates streams of input data, such as sensor readings, into streams of aggregate statistics and verdicts about the safety of the system. It is usually assumed that the values on the input streams represent fully accurate measurements of the physical world. In reality, however, physical sensors are prone to measurement noise and errors. These errors are further amplified by the processing and aggregation steps within the monitor. This paper introduces RLola, a robust extension of the stream-based specification language Lola. RLola incorporates the concept of slack variables, which symbolically represent measurement noise while avoiding the aliasing problem of interval arithmetic.   We present algorithms for both online and offline monitoring of RLola specifications. Since monitoring RLola specifications may require unbounded memory in general, we identify a rich fragment of RLola that can be automatically translated into monitors with guaranteed constant memory usage for online monitoring. An online RLola monitor observes a live system and provides real-time feedback on the current status of specified assertions. A satisfiability-modulo-theories-based offline algorithm analyzes complete system traces and determines whether a hypothetical ground-truth trace exists that satisfies all assertions at all time points. The offline algorithm can therefore detect violations that the online algorithm may miss. We implement these algorithms in the existing RTLola framework and evaluate their precision and running time based on a comprehensive example. |
| 2026-08-13 | [Chance-constrained selection of sequential intervention strategies from counterfactual estimates](http://arxiv.org/abs/2608.13209v1) | Minkyoung Kim, Beakcheol Jang | Many operational decisions are sequences of interventions under a cumulative resource limit, such as a maintenance schedule within a crew-hour budget. Choosing among them calls for the outcome and the cumulative cost each would produce, counterfactual quantities identified from observational data. Two strategies with the same expected cost can exceed the budget at very different rates, so constraining the mean does not bound how often an overrun occurs. Prior two-step architectures, recently extended to continuous doses, constrain the mean cost rather than its tail and allocate at a single decision point. Methods that do bound a cost tail take its distribution from a specified model rather than identifying it from data. We present a predict-then-optimize framework. In the prediction step, any estimator returning an outcome value and a cost distribution supplies what the decision rule consumes, so the predictor is interchangeable. In the optimization step, a chance-constrained selection over a finite candidate set bounds the probability that the cumulative cost exceeds the budget. That tail does not decompose across stages, so each strategy is scored whole. Sweeping the tolerated violation probability traces a safety-utility frontier, and distribution-free finite-sample bounds cover violation and outcome shortfall. Four of five environments, spanning clinical treatment and equipment maintenance, supply exact counterfactual ground truth; the fifth carries real outcomes from a digital-health micro-randomized trial. Across them, the rule holds the budget where a point-estimate rule overruns it, at an outcome cost the frontier makes explicit. All code is available at https://github.com/mfriendly/counterfactual-chance-selection |
| 2026-08-13 | [Branch and Bound for Relational Verification of Neural Networks](http://arxiv.org/abs/2608.13118v1) | Kota Fukuda, Zhenya Zhang et al. | Verification of neural networks against relational specifications, such as global robustness, is crucial for safety-critical applications of cyber-physical systems (CPS), given their increasing adoption of AI components. Compared to simple trace properties (e.g., local robustness), verifying relational specifications requires reasoning about the relationship between multiple network inferences, which brings significant technical challenges. Existing research has explored abstraction techniques based on sound and convex over-approximation of neural network outputs; however, since these approaches are inherently incomplete and may raise false alarms, they further underscore the need of effective abstraction refinement.   In this paper, we propose a branch-and-bound (BaB) framework to mitigate the issue, which iteratively splits the problem until all sub-problems are verified. Specifically, our BaB framework features splitting of relational neurons rather than individual neurons as prior works do, and as the core of our technique, we devise a relational neuron selection strategy based on the dual formulation of the verification problem, which allows us to efficiently select the (most likely) optimal relational neuron that maximizes the refinement brought by problem splitting. We evaluate SaBRe on 817 verification problems across ACAS Xu, MNIST-F, MNIST-C, CIFAR and GTSRB. The results show that SaBRe outperforms different baseline approaches, in terms of the number of solved instances and verification efficiency, which demonstrates the effectiveness of our proposed techniques. |
| 2026-08-13 | [Formal Verification of Quantum Ancilla Safety](http://arxiv.org/abs/2608.13099v1) | Jiqi Li, Jingyi Mei et al. | Ensuring ancilla safety is a critical correctness requirement for quantum compilation, since ancilla qubits are routinely introduced to implement complex operations with fewer gates and reduced depth. However, formally verifying this property is computationally hard due to state-space explosion in the number of qubits, particularly for dirty ancillae, which carry unknown initial states and must be restored after use. We propose an end-to-end verification-and-repair framework that rigorously addresses both clean and dirty ancilla safety. Our core contribution is a two-step reduction strategy: we first prove that verifying an $m$-qubit dirty ancilla register decomposes into $2m$ independent clean ancilla safety checks; subsequently, we reduce each clean ancilla safety instance to an algebraic commutativity check against Pauli-$Z$ and Pauli-$X$ operators. This approach yields an efficient and naturally parallel verifier and enables actionable diagnosis by classifying violations into logic errors and phase errors. Leveraging this diagnosis, we further design lightweight repair routines that append local single-qubit rotations to eliminate a broad class of local ancilla faults. We implement the full pipeline in a prototype tool using a dual-backend architecture combining decision diagrams and weighted model counting, and validate it on diverse circuits ranging from arithmetic benchmarks to Grover's algorithm. Our experiments demonstrate scalability to thousands of qubits and show that the proposed repairs effectively improve ancilla safety while preserving circuit functionality. |
| 2026-08-13 | [Tracing Methamphetamine abuse in under-treatment drivers: How biomechanical and oculomotor features help detect at-risk drivers?](http://arxiv.org/abs/2608.13054v1) | Hamed Salmanzadeh, Alireza Mortezapour et al. | While the detrimental impacts of driving under the influence of stimulants such as methamphetamine are well-documented, the driving performance of individuals currently under-treatment has received considerably less attention. This study compared the behavior of individuals with a history of stimulant abuse (across two distinct treatment phases) with a control group of healthy drivers using a driving simulator. Oculomotor and biomechanical data were continuously collected via an eye-tracker and a Kinect sensor, respectively. These parameters were utilized to train a K-Nearest Neighbors (KNN) classification model designed to detect high-risk behavioral patterns in drivers undergoing methamphetamine rehabilitation. Through the evaluation of various feature combinations and neighborhood configurations, the optimized model successfully discriminated between normal drivers and those with a history of abuse with an accuracy of 90%. Detecting at-risk drivers through technologies embedded in Advanced Driver Assistance Systems (ADAS) by continuously monitoring physiological and behavioral parameters, facilitates a proactive safety strategy. Issuing real-time alerts to the driver, passengers, and external monitoring networks can ultimately mitigate the risk of traffic collisions. |
| 2026-08-13 | [Static analysis-guided agentic AI translation enables Rust as a full stack bioinformatics language](http://arxiv.org/abs/2608.13029v1) | Johan Henriksson | The field of bioinformatics struggles with legacy code - old code that is commonly used but may no longer have a maintainer, or may be written in an now-unfamiliar language (e.g. Perl, Fortran). This incurs maintenance cost (technical debt), but dynamically typed languages also negatively impacts the environment and fail to make use of modern hardware. Legacy code may also have security or safety problems that make it unsuited for use in clinical settings. Here we show that agentic AI, combined with static analysis, can be used to translate legacy code to the modern language Rust. We provide prompts and supporting software to aid systematic translation, and evaluate it on common software for NGS and imaging. We showcase the result on our software Bascet: Size was reduced by ~80x, build time decreased by ~10x, and performance of key steps improved >3x. Unix dependencies were also removed, making Bascet the only single-cell pipeline able to run on native Windows, without a container. Large-scale refactoring of bioinformatics software is thus now possible at a limited budget, enabling more complex tools to be developed. |
| 2026-08-13 | [How LLMs Respond to Escalating Delusions: Four Longitudinal Trajectories of Model Behavior](http://arxiv.org/abs/2608.13017v1) | Anna Sterna, Kacper Dudzic et al. | The widespread use of LLMs among psychiatric populations has raised concerns regarding their safety and potential iatrogenic impact in the context of AI psychosis. While growing literature conceptualizes AI psychosis and documents case studies, empirical evidence tracing AI-exacerbated psychotic processes remains scarce. We propose and test a longitudinal qualitative evaluation design, supported by automated metrics, to assess mainstream LLMs' potential to exacerbate psychosis. Fifteen widely used LLMs were prompted across 30 days using the same 30-message script, simulating progression from mild anomalous experiences to psychotic ideation. Four trained evaluators independently rated 449 model-days, assessing (1) recognition stage (from naive engagement to stabilized clinical framing), (2) interpretative confidence, and (3) intervention profile (from education to treatment recommendation). Two computational metrics-entrainment and modality-were devised to increase evaluation reliability. Direct recommendations to disengage from the LLM were flagged and re-coded via adjudication using a strict two-level definition. Across model generations and vendors, we identified four response trajectories: (1) premature medicalization and disengagement (Claude Haiku 4.5); (2) recognition without safeguarding, marked by LLM self-sufficiency in offering help (GPT Instant/Thinking); (3) delayed and unstable recognition, marked by late, non-progressive conceptualization (Claude Opus 3/4/4.1, Claude Haiku 3.5, GPT-4o, Gemini 3.1 Pro); and (4) delusion co-construction through active engagement with delusional content (Gemini 2.5 Pro/Flash, DeepSeek-V3, Claude Sonnet 4). Our findings indicate that LLMs' potential to exacerbate AI psychosis should be operationalized as a combination of recognition timing, stability, and intervention accuracy and evaluated longitudinally, focusing on temporal dynamics. |
| 2026-08-13 | [Practice Makes Unsafe: Skill Misevolution in Self-Improving LLM Agents](http://arxiv.org/abs/2608.12851v1) | Xutao Mao, Liangjie Zhao et al. | Self-improving LLM agents convert successful trajectories into persistent cross-task state. An unsafe success can thereby become reusable policy after its triggering input disappears. Skill evolution makes this failure measurable by distilling operational trajectories into executable, transferable, and inspectable procedures. Because evolution optimizes task outcomes rather than procedure safety, compromised experience can cause skill misevolution. Existing benchmarks measure current behavior or static artifacts but cannot attribute risk across authoring, retrieval, and later execution. To expose this lifecycle, we introduce SkillMisevo-Gym, a lifecycle-aware harness that versions skill state across agent frameworks, and SkillMisevo-Bench, a frozen design from malicious exposure to carryover tasks, with concept-aligned benign tasks and nine lifecycle metrics. We also introduce SafeEvolve, a wrapper that repairs unsafe content and governs subsequent reuse. Across 25 agent-method configurations, each covering 525 tasks in 25 episodes, all 21 evolved configurations author unsafe artifacts, while only fifteen lead to fresh-session harm. In the exposure sweep, three malicious tasks raise carryover ASR from 16.0% to 35.3%. Across representative skill evolution methods, SafeEvolve reduces unsafe retrieval and fresh-session harm by 26.7 and 17.3 percentage points, respectively, while mean benign utility changes by only 0.4 points. Together, persistent-adaptation safety must govern what updates write and what future executors reuse. Code is available at https://github.com/henrymao2004/misevolve. |

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



