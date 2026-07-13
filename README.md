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
| 2026-07-10 | [PAC-ACT: Post-training Actor-Critic for Action Chunking Transformers](http://arxiv.org/abs/2607.09590v1) | Yujie Pang, Zudong Li | Precision industrial contact manipulation requires reliable robot policies under pose perturbations and contact-force constraints. Vision-language-action models offer broad generalization but often introduce high inference latency and GPU-memory cost, while vision-action chunking policies are more suitable for real-time industrial control. However, these policies are usually trained by behavior cloning and suffer from distribution shift in contact-rich tasks. This paper proposes PAC-ACT, a reinforcement-learning post-training framework for pretrained Action Chunking Transformer policies. PAC-ACT reformulates policy optimization at the chunk level, constructs an ACT-transferred actor-critic architecture, and introduces a hybrid behavior-prior constraint to preserve the pretrained action distribution during online fine-tuning. Experiments on industrial precision-contact benchmarks show that PAC-ACT improves task success, contact stability, and force safety while retaining low latency and low GPU-memory usage. On the Contour task, PAC-ACT significantly reduces peak contact force and decreases the proportion of force readings above 60 N by 46 times. Sparse-reward ablations further show that the proposed behavior-prior constraint enables effective exploration under randomized initial poses. |
| 2026-07-10 | [How Mobile Gas Sensor Trajectories Govern Hydrogen Leak Detection: A Safety Gap in Manual Leak Inspection of Hydrogen System Components](http://arxiv.org/abs/2607.09527v1) | Christian Masuhr, Arne Wendt et al. | The integrity of hydrogen infrastructure relies on reliable leak detection, performed almost exclusively via manual tracer gas sniffing in electrolyzer manufacturing. Although mandated by standards, the lack of spatial probe guidance instructions leaves detection reliability entirely to operator execution, further compromised by sensor signal delays. This study quantifies how sniffer trajectory kinematics affect detection reliability at small-scale pipes and fittings, a near-field regime largely neglected by macroscopic dispersion research. Using a robotically guided test bench to eliminate operator variability, static concentration fields and dynamic trajectory passes were acquired across representative geometries under standardized leak rates (5 vol% hydrogen in nitrogen) and varying scanning velocities. Results demonstrate that scanning velocity and spatial probe orientation strongly dictate detectability. Conventional linear trajectories frequently miss leaks under dynamic conditions, causing severe false negatives. Conversely, geometry-specific routing, such as circumferential plunging paths around sealing points, maintains a high safety margin. From these observations, geometry-specific routing rules and a reduction-factor model for dynamic signal loss are derived. The findings reveal that current standard operating procedures pose a tangible safety risk. To operationalize these rules, a proof-of-concept software pipeline is presented, generating validated trajectories directly from 3D models for visualization in assistance systems. |
| 2026-07-10 | [Multimodal Reward Hacking in Reinforcement Learning](http://arxiv.org/abs/2607.09492v1) | Jiayu Yao, Yiwei Wang et al. | Reinforcement learning (RL) is increasingly used to align multimodal large language models (MLLMs), but higher rewards do not always imply better task performance. This risk is amplified when visual evidence is evaluated by text-only or weakly grounded rewards. We study reward hacking in MLLM RL across safety VQA, chart VQA, and stress-test settings, varying reward design, data ambiguity, model scale (2B-32B), and RL algorithm (GRPO, RLOO, DAPO). We introduce Newly Rewarded Failure Rate (NRFR), which measures failures among samples whose proxy reward improves over the SFT baseline. Outcome-only rewards cause severe hacking, reaching 48.1% Reward Hacking Rate (RHR), while NRFR exceeding RHR shows that RL creates new failures rather than merely inheriting them. Scaling reduces but does not eliminate hacking: even the 32B model retains a 54.9% worse rate under outcome-only rewards, whereas answer-aware rewards improve the oracle trend at every scale. Robustness is also algorithm- and scale-dependent: GRPO is consistently most resistant, RLOO remains vulnerable, and DAPO improves substantially from 2B to 8B. Visual-evidence rewards help only with reliable verification: keyword-based checks increase hacking, while VLM-as-judge semantic verification reduces it. Overall, multimodal reward hacking is a systematic result of optimizing imperfect rewards, and robust alignment requires rewards and verifiers that remain reliable under optimization pressure. |
| 2026-07-10 | [System Capybara: Tracking Capabilities for Separation and Freshness (Extended Version)](http://arxiv.org/abs/2607.09383v1) | Yichen Xu, Oliver Bračevac et al. | Substructural type systems give strong static control over aliasing. Examples include uniqueness, separation, and borrowing. How can such control be brought to established languages whose programming models rely on higher-order abstraction, unrestricted aliasing, and pervasive sharing? We study this problem in the context of Scala. We show how to retrofit these guarantees selectively instead of globally: ordinary code keeps Scala's usual aliasing discipline, while stronger guarantees can be enforced where they matter.   Our starting point is Scala's capture checking, whose treatment of capabilities is inspired by the object-capability tradition: capabilities are ordinary values, and capture sets record, in a value's type, which capabilities the value may use. We develop System Capybara, which adds a selective alias-control layer to this mechanism. By tracking separation, consumption, freshness, and read-only access for capabilities, Capybara recovers key reasoning principles from substructural and ownership-based disciplines without global invariants.   We give a type-preserving translation from the surface calculus Capybara to CoreCapybara, a core calculus extending System Capless, the earlier foundation for capture checking. The translation uses quantifiers for capture polymorphism and freshness, and constraint-indexed modal types for separation. We prove a semantic soundness result for the core calculus in Lean 4 and derive type safety, memory safety (no use-after-free or double-free), immutability of read-only computations, and data-race freedom for well-typed programs.   Finally, we implement Scala 3's new separation checker, which brings higher-order separation reasoning about effects, capabilities, and resources to ordinary Scala, including fearless concurrency. |
| 2026-07-10 | [WILDTRACE: Benchmarking Natural Evidence Trails in Long-Context Reasoning](http://arxiv.org/abs/2607.09328v1) | Zixin Chen, Peng Liu et al. | Answering complex questions over long documents frequently requires integrating evidence that the source itself disperses naturally across distant passages. In an incident report, the operating condition, design flaw, and missed safety check that jointly explain a disaster may appear dozens of sections apart; in a novel, a character's true motive may surface only through scenes far removed from the moment it becomes relevant. This source-internal evidence integration is central to real-world long-document analysis, yet existing benchmarks largely sidestep it. Needle probes, planted facts, and reverse-engineered multi-hop chains embed evidence that may differ from the host text in distribution, placement, or register, making it unclear whether strong performance reflects genuine source reasoning or distributional artifacts. We introduce WILDTRACE, a benchmark of 481 tasks over 214 naturally occurring long-form sources such as technical incident reports and lesser-known literary narratives, where all evidence trails arise from the document's own causal, temporal, and narrative logic. Drawing on Pearl's causal hierarchy and prior multi-hop reasoning typologies, we define seven source-internal evidence geometries that characterize the distinct relational demands of analytical reading in long documents. A source-first construction pipeline mines candidate trails from document structure before writing questions; each item then undergoes multi-stage validation covering clue necessity, answer groundedness, rubric fidelity, contamination resistance and answerability. As models are increasingly entrusted with real-world high-stakes analytical tasks, this gap between accessing information and reasoning over naturally dispersed evidence emerges as a defining challenge for the next stage of long-context research. |
| 2026-07-10 | [Effects of Robotic Touch on Older Users During Walking Guidance by a Humanoid Robot](http://arxiv.org/abs/2607.09323v1) | Leonie Leven, Marko Ackermann et al. | The shortage of healthcare staff is a challenge in geriatric care. To address this, robots can be integrated into care settings to provide assistance and emotional support. A promising application is walking guidance, particularly benefiting older adults as navigation skills deteriorate with aging. As walking guidance involves direct contact, the aim of this study is to understand how older adults perceive and respond to different touch modes during guided walking. 24 older adults (68 - 88 yrs.) walked four times a ten-meter trajectory guided by the robot TIAGo Pro in four contact conditions: no physical contact (NC); physical contact through holding the robot's wrist with the hand (HH); physical interaction through linking arms with the robot (LA); and physical contact through resting the forearm on the robots forearm (FC). A multimodal assessment approach included electrocardiogram, electrodermal activity, contact force, distance to robot, and questionnaires. Physiological results reveal a slight increase in stress levels during robot interaction. Behavioural and subjective measures, however, show overall acceptance of robotic touch. The two conditions corresponding to larger interaction forces (HH and FC) were associated with lower relative distances between participant and robot, indicating a higher trust and confidence. Questionnaire responses supported these findings, evidencing greater perceived safety, trust and comfort in these conditions. This study provides insights for the design of robotic walking guidance assistance, indicating that gentle, stable touch is preferred by older adults in comparison to contactless interaction. |
| 2026-07-10 | [Forget Narrowly, Retain Broadly: Unlearning as an Asymmetric Generalization Problem](http://arxiv.org/abs/2607.09236v1) | Amit Peleg, Naman Deep Singh et al. | Machine unlearning in LLMs is the targeted removal of specific knowledge while preserving all other capabilities, critical for privacy and safety. Yet existing benchmarks measure it unreliably. They miss knowledge that resurfaces under paraphrased or indirect queries, a failure we call under-forgetting, and lack the semantic, syntactic, and lexical probes needed to verify that unrelated knowledge is preserved, a failure we call over-forgetting. Both failures reflect an asymmetric generalization problem. Forget evaluation must cover diverse query formulations of the same target facts, testing whether forgetting holds beyond exact training prompts. Retain evaluation must probe a far larger and implicitly defined set, namely every fact disjoint from the forget target. The retain set thus defines the effective forget set, yet current datasets provide no fine-grained annotation of this forget-retain boundary. We address this with SUITE, an evaluation protocol and training corpus that captures forget-retain structure for real-world factual domains. Methods trained on SUITE improve substantially, showing that training data is as important as algorithmic design. Building on the obtained insights, we introduce JensUn++, an unlearning algorithm that achieves the best forget-retain utility trade-off across three LLMs, in both sequential and joint unlearning settings. Code and datasets are available at https://amitpeleg.github.io/forget-narrowly-retain-broadly |
| 2026-07-10 | [Git-Assistant: Planning-Based Support for Updating Git Repositories](http://arxiv.org/abs/2607.09224v1) | Alfredo Garrachón Ruiz, Tomás de la Rosa et al. | Version control systems are essential for collaborative software development, yet tools like git remain challenging for many practitioners. Recent advances in Large Language Models (LLMs) offer promising capabilities for interpreting developer intent, but their effectiveness in repository management tasks is limited by the need for formal reasoning. This work introduces Git-Assistant, an AI-based assistant that combines LLMs with automated planning to support developers in executing non-trivial git operations. The assistant analyzes repository context, translates natural language requests into actionable command sequences, and incorporates planning techniques to ensure correctness and safety. We present a systematic evaluation methodology using synthetic and randomized git environments, comparing the performance of LLM-only and planning-augmented variants across multiple metrics. Experimental results demonstrate that integrating formal reasoning with LLMs improves reliability and reduces errors in repository management, highlighting the potential of hybrid AI approaches for intelligent developer assistance. |
| 2026-07-10 | [Empirical Pedestrian Safety Assessment in a Mobile Robot Using a Predictive Social Force Model](http://arxiv.org/abs/2607.09192v1) | Alireza Jafari, Yun-Hao Tsai et al. | Mobile robots are going to share the sidewalks with pedestrians. They must ensure their objective safety and respect the walkers' subjective safety/comfort. Computationally efficient Social Force Models (SFM) present interpretable solutions for real-time robot navigation in dynamic crowds. Recent explorations of Projected Time-to-collision (PTTC) integration into SFM variants, for example, PTTC-based SFM (TSFM), improve safety metrics. But the effect of predictive variants is unclear. We introduce Predictive SFM (PSFM) and Predictive TSFM (PTSFM) by integrating predicted social force vectors over a finite time horizon. The paper implements SFM, TSFM, PSFM, and PTSFM on a nonholonomic mobile robot and performs experimental trials with volunteers attending a facing scenario. We systematically study objective and subjective safety across the variants. Minimum PTTC, average speed, minimum distance, lateral distance, and the maximum trajectory curvature benchmark the objective safety. Likert scale post-interaction surveys assess subjective safety by marking comfort, smoothness, distance appropriateness, and speed suitability. We confirm that PTTC integration improves safety metrics. The prediction contribution is limited and occasionally visible in some of the sub-metrics. Some participants perceive smoother movements and safer speed behavior with predictive methods, but Mann-Whitney tests reveal no significant differences in subjective ratings. Therefore, PTTC-based navigation enhances safety, whereas the formulated prediction offers limited additional benefits in single-pedestrian scenarios. |
| 2026-07-10 | [Present but Rescaled: Chat-to-Agent Transfer of Additive Activation Steering](http://arxiv.org/abs/2607.09156v1) | Lucas Pinto | Additive activation steering (injecting a scaled residual-stream direction during generation) is calibrated almost entirely in single-turn chat, yet the models it targets are increasingly deployed as tool-using ReAct agents. We present the first systematic chat-to-agent transfer study of additive steering, coupling behavioral measurement with a representation read-out in a matched-information design: the same items rendered as plain chat or as a ReAct tool-use episode, with matched-norm random-direction controls and the transcript re-encoded every turn to exclude KV-cache contamination. Transfer is real but rescaled, and the right description is a dissociation: the injected direction reaches the late layers at near-full strength in every setting and model tested (install-site agent-over-chat ratios 0.83-1.16 across three families), while the behavioral coupling is reset per model and context. On Qwen2.5-7B a refusal bypass vector amplifies in the agent (T = 1.45, CI [1.20, 1.78], N = 300); across a powered uniform-protocol distribution the coupling spans amplification (Gemma-2-9B T = 2.00) to attenuation (Yi-1.5-9B T = 0.43, CI [0.29, 0.60]), with no universal constant and a single clean attenuator against a universal sign. Directional ablation of the same axis does not amplify (T = 0.93, CI including 1) while additive injection amplifies (T = 1.50), a 20.1-point gain difference (CI [13.4, 26.8]) that identifies an additive-specific mechanism. Two pre-registered instruments converge to localize the rescaling to the ReAct format scaffold, before any tool observation, rather than to the observation boundary where a dilution account would predict it. The safety implication is immediate and unpredictable: agentic deployment amplifies steering-based refusal bypass by up to 2.00x on some models while others attenuate, so a deployment cannot assume a given model is safe under additive steering. |
| 2026-07-10 | [MedRealMM: A Real-World Multimodal Benchmark for Chinese Online Medical Consultation](http://arxiv.org/abs/2607.09142v1) | Runhan Shi, Quan Zhou et al. | Large language models (LLMs) are increasingly deployed in online medical consultation, yet existing benchmarks remain poorly aligned with real clinical practice. Many rely on synthetic conversations or patient simulators, omit patient-uploaded medical images, or evaluate open-ended clinical responses using multiple-choice or lexical-overlap metrics that poorly reflect clinical quality. We introduce \textbf{MedRealMM}, a large-scale benchmark for multimodal online medical consultation built from de-identified patient-doctor interactions collected from a nationwide Chinese internet hospital. MedRealMM uses a Multimodal Clinical Challenge Point (MCCP) extraction framework to identify clinically demanding moments in authentic consultation trajectories and converts each into a standardized next-response generation task while preserving the preceding text-image context. Each instance is paired with a case-specific rubric refined by physicians that rewards clinically desirable behaviors and penalizes unsafe, unsupported, or contradictory responses. The current release contains 5,620 real-world multimodal cases spanning 64 clinical departments. We evaluate 19 general-purpose and medical-specialized LLMs, including text-only and multimodal systems. Our results show that image information is critical for reliable clinical performance and that current frontier models remain below the online physician response. Although some frontier models satisfy as many or more positive clinical criteria than physicians, they trigger more negative criteria, indicating that safety-sensitive error avoidance remains a central bottleneck. MedRealMM offers a realistic and reproducible benchmark for evaluating multimodal medical reasoning in real-world online consultation. The dataset will be publicly available on Hugging Face at https://huggingface.co/datasets/jdh-algo/MedRealMM. |
| 2026-07-10 | [L-MAD: A Systematic Evaluation of Multi-Agent Debate Structures in Legal Reasoning](http://arxiv.org/abs/2607.09099v1) | Tan-Minh Nguyen, Hoang-Trung Nguyen et al. | While multi-agent debate (MAD) frameworks have shown significant potential in general reasoning, their effectiveness in highly structured, knowledge-heavy legal domains remains under-explored. In this work, we introduce the Legal Multi-Agent Debate (L-MAD) framework to systematically evaluate different debate structures and aggregation methods within Legal Textual Entailment. By assigning distinct expert personas to multiple agents, L-MAD improves upon strong single-agent baselines by up to 8\%. Furthermore, analyzing how debate scales reveals a clear trade-off: increasing the agent population reduces inconsistency and improves accuracy, whereas extending discussion rounds induces a detrimental \textit{over-deliberation drift} where agents reinforce each other's mistakes. Ultimately, our findings outline the practical boundaries and safety margins of deploying collaborative multi-agent systems in high-stakes legal reasoning environments. |
| 2026-07-10 | [Neuro-Agentic Control: A Deep Learning-based LLM-Powered Agentic AI Framework for Controlling Security Controls](http://arxiv.org/abs/2607.09076v1) | Saroj Gopali, Bipin Chhetri et al. | Cyberattacks on operational technology are increasingly causing costly downtime and physical damage, exposing the limitations of traditional rule-based monitoring in industrial IoT environments. While Large Language Models (LLMs) have strong semantic reasoning abilities to assist in decision support, their hallucinatory nature presents unacceptable safety liabilities for closed-loop control. This paper introduces a neuro-agentic control framework, a novel architecture that couples an LLM-based planner (i.e., such as Gemini 2.5 Flash-Lite) with a pre-trained Time-Series Foundation Model (TimesFM), to achieve physics-grounded autonomous defense. The paper introduces a ``Counterfactual Physics Injection'' mechanism that simulates the impact of LLM-proposed interventions within the numerical latent space of the foundation model before actuation, while allowing the system to reject hallucinatory or unsafe actions. Evaluated on an industrial dataset (e.g., the Secure Water Treatment (SWaT)) in the context of stochastic attack scenarios, the framework exhibited better performance compared to LSTM and TCN baselines. The Neuro-Agentic Loop prevented five breaches (33.3%) below the threshold versus LSTM (26.7%) and TCN (13.3%), with zero physically invalid (hallucinated) actions executed. These results demonstrate the efficacy of using foundation models as deterministic ``Sentinels'' to safeguard agentic AI in critical infrastructure. |
| 2026-07-10 | [Inside the Skill Market: From Software Engineering Activities to Reusable Agent Skills](http://arxiv.org/abs/2607.09065v1) | Jialun Cao, Xinru Yan et al. | Software engineering (abbrev. SE) has continuously evolved through increasingly powerful forms of reuse, from source code and libraries to components and services. Recent advances in AI agents have introduced a potentially new reusable artifact: skills. Emerging agent skill repositories and marketplaces enable developers to package, share, and reuse SE expertise as reusable skills. This trend raises a fundamental question: what SE activities are being encapsulated into reusable skills? Existing studies primarily focus on a broad range of skills acquisition, safety, or benchmarking, while lacking a systematic understanding of SE-specific skills and their coverage across the software development lifecycle. To address this gap, we conduct the first large-scale empirical study of SE skills in public repositories and marketplaces. We collect and analyze a large corpus of SE skills, examining the activities they encapsulate, lifecycle coverage, evolution characteristics, and evaluation mechanisms. Our findings reveal that SE activities are increasingly becoming reusable artifacts via skills and suggest promising research opportunities for skill recommendation and engineering-oriented structuring, as well as the need for mechanisms to encapsulate high-context SE activities into reusable skills. Overall, our study provides the first activity-centric characterization of SE skills and reveals how SE activities are increasingly being transformed into reusable skills. These findings offer new insights into skill reuse, ecosystem development, and the future of agent-centric SE. |
| 2026-07-10 | [SLBench: Evaluating How LLM Agents Follow Logical Relations in Skills](http://arxiv.org/abs/2607.09016v1) | Xuan Chen, Chengpeng Wang et al. | Agent skills extend LLM agents with reusable procedures, tools, and domain-specific workflows, but their safety depends on resolving dependencies among interacting instructions. We introduce SkillLogic, a framework for analyzing logical relations in skill files and constructing executable tests from them. Our taxonomy covers eight relation types, including preconditions that gate valid actions, constraints that limit how allowed actions may be performed, and fallbacks that specify recovery behavior after failure. Using SkillLogic, we scan over 5000 public skills and find that 70% contain at least one logical relation. We then construct SLBench, an 86-case executable benchmark from high-confidence, high-impact, and locally testable relations. Evaluating Codex and Claude Code across six LLM backbones shows unsafe rates up to 70%, with violations leading to privacy leaks, unsafe configuration changes, and incomplete cleanup. The human audit attributes failures to both agent capability gaps and low-salience skill text. We further show that SLGuard, a lightweight inference-time scaffold, reduces violations by 63% on targeted cases. Our results establish logical-relation following as a distinct reliability challenge for skill-guided agents. |
| 2026-07-10 | [C-GAP: Class-Aware and Online Prompting Improves Vision-Language Models on Imbalanced Classes](http://arxiv.org/abs/2607.09008v1) | Francis Fernandez, Arash Jahangiri et al. | Safety-critical perception systems must reliably detect rare object classes within small label spaces, a setting that long-tailed detection methods, designed for hundreds of classes with dense annotation, fundamentally do not address. Open-vocabulary detectors offer a promising alternative, as they use natural language queries at inference time, making prompt quality a first-class lever for detection performance. We exploit this property to address class imbalance: rather than retraining models or collecting additional annotations, we ask whether iteratively refining the language prompts, fed to frozen detectors, can improve minority class detection. We introduce C-GAP Caption-Guided Augmentation and Prompting), a detector-agnostic, annotation-free framework that operates in two phases. First, we establish a composite caption baseline combining per-image scene descriptions with class-quantity context, which we show outperforms scene-description only or class-quantity-only prompts across multiple open-vocabulary architectures and benchmarks. Second, an LLM iteratively refines each image's caption individually, with trials triaged into accept, tentative, or regenerate buckets based on minority-class AP@0.5 against a dynamic threshold derived from the composite baseline. Refinement terminates early once sufficient AP@0.5 gain is achieved. No detector weights are updated at any stage. Our experiments shows that C-GAP improves minority-class average precision up to 53% over the baselines. On COCO, C-GAP improves minority-class AP@0.5 by ~81% relative over the composite baseline (17.69 -> 32.09). Experiments confirm that composite captions provide the critical foundation for effective refinement: using scene-description-only or class-quantity-only prompts as the refinement starting point yields diminishing returns, supporting both stages of C-GAP as necessary contributions. |
| 2026-07-09 | [Every Sample Counts: Supervised Fine-Tuning of Language Models with Pointwise Constraints](http://arxiv.org/abs/2607.08968v1) | Ignacio Hounie, Ignacio Boero et al. | Fine-tuning language models often requires enforcing constraints on individual inputs without compromising downstream performance. Existing constrained alignment methods impose constraints on average, which can induce undesirable disparities across inputs or users. We propose a novel alignment framework that addresses this gap by enforcing per-sample constraints while still minimizing an average loss. To mitigate the impact of overly restrictive constraints and outliers, we introduce a learned, sample-dependent relaxation that minimally adjusts the constraints, trading off a user-defined relaxation cost with the training objective. To address practical duality and optimization challenges, we develop an augmented Lagrangian approach tailored to this formulation. We demonstrate the flexibility of the framework by instantiating it under distinct small language-model fine-tuning tasks and constraints: safety in instruction following, preferences in function calling and length in re-ranking. Across these settings, our approach reduces tail constraint violations while largely preserving the model's performance. |
| 2026-07-09 | [Optimizing Against Safety Representations: Activation-Guided Adversarial Suffixes and the Geometry of Refusal](http://arxiv.org/abs/2607.08883v1) | Ege Çakar, Hannah Guan et al. | Behavioral alignment in large language models often masks fragile internal safety representations. Recent work suggests that refusal behavior is mediated by low-dimensional directions in activation space. This raises questions about how such representations are structured, localized, and accessed by optimization. We study adversarial suffix attacks as a probe of representational alignment. We introduce Activation-Guided GCG, which replaces output-based objectives with losses that directly target a model's internal refusal direction. Across several objective variants, we find that suppressing refusal globally across all layers and positions is more effective than targeting a single layer-position pair. This suggests that safety representations are distributed across the forward pass rather than causally localized to a single site. We further introduce Soft-GCG, a continuous relaxation of discrete suffix optimization using Gumbel-Softmax. Soft-GCG achieves a 33 $\times$ speedup over standard GCG while improving attack success rates. Evaluating across model scales, we find that smaller models remain vulnerable while larger models resist both activation- and suffix-based attacks at our compute-constrained settings, consistent with larger and better safety trained models being harder to jailbreak. Together, our results clarify how safety mechanisms are encoded and can be broken in contemporary models. These insights provide concrete guidance for designing more robust and representation-aware alignment strategies. |
| 2026-07-09 | [AUTOPILOT VQA: Benchmarking Vision-Language Models for Incident-Centric Dashcam Understanding](http://arxiv.org/abs/2607.08745v1) | Siddharth Damodharan, Radhika Gupta et al. | Recent advances in Vision-Language Models, Large Language Models, and Multimodal Large Language Models have improved autonomous driving tasks such as scene understanding, decision making, trajectory prediction, and visual question answering. However, evaluating whether these models can reliably reason about safety-critical incidents remains challenging. To address this gap, we present AUTOPILOT-VQA, an incident-centric visual question answering benchmark for dashcam video understanding. The dataset evaluates different systems through structured questions designed around real-world driving incidents and near-incidents. The benchmark covers diverse safety-relevant categories, including weather and lighting conditions, traffic environment, road layout, road surface state, signage, involved entities, accident occurrence, impact location, and avoidability-related reasoning. By requiring models to answer grounded questions about both contextual scene properties and event-level incident details, AUTOPILOT-VQA moves beyond object recognition toward temporally grounded, safety-aware reasoning. The dataset is released as part of the AUTOPILOT CVPR 2026 competition and provides a standardized benchmark for assessing the reliability of autonomous driving systems in different scenarios. Our benchmark support developments for more interpretable, robust, and safety-conscious vision-language systems for real-world autonomous driving. |
| 2026-07-09 | [Privacy-Preserving Intent Fulfilment and Assurance for 6G RAN](http://arxiv.org/abs/2607.08809v1) | Joss Armstrong | Intent-based network management is the emerging paradigm for 6G service lifecycle automation, with the 3GPP intent management framework (TS~28.312) defining creation, translation, fulfilment, and assurance stages. Existing fulfilment and assurance approaches require deep packet inspection, per-flow state tracking, or access to vendor-internal node telemetry to verify that provisioned resources satisfy expressed intents. These requirements conflict with regulatory constraints (GDPR, ePrivacy Directive) in multi-tenant networks and with vendor opacity in multi-vendor O-RAN deployments.   We present an architecture for privacy-preserving intent fulfilment and assurance in which a coordinator provisions resources from declared intent categories without traffic inspection, and verifies fulfilment using only aggregate standardised PM counters at the O1 interface. A data-processing inequality argument shows that the resource allocation reveals at most $\log_2 K$ bits about traffic content, where $K$ is the number of intent categories. We define two architectural privacy properties, intent-traffic unlinkability and node-opaque verification, and show that both hold by construction. Node-opacity does not sacrifice detection power: the aggregate verifier weakly dominates the per-agent verifier under a homogeneity condition.   We map the architecture to the 3GPP intent lifecycle and the O-RAN Non-RT RIC, identifying the concrete interfaces, data objects, and deployment points at which the mechanism operates. On production PM counter data from four operator networks, increasing intent-category granularity sharpens provisioning but weakens assurance, consistent with the theoretical prediction that the privacy ceiling is a structural side effect of the detection constraint rather than a separate design parameter. |

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



