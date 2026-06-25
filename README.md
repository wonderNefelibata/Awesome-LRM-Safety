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
| 2026-06-24 | [Model Forensics: Investigating Whether Concerning Behavior Reflects Misalignment](http://arxiv.org/abs/2606.26071v1) | Aditya Singh, Gerson Kroiz et al. | A central goal of safety research is determining whether a model is misaligned. Prior work has largely focused on detecting concerning behavior. But behavior alone does not establish misalignment: a concerning action can arise from benign causes such as confusion. This motivates model forensics: investigating whether the action was driven by malign intent. In this paper, we propose a baseline protocol for model forensics consisting of two steps, iterated as needed. First, we read the chain of thought (CoT) to generate hypotheses about what drives model behavior. Second, we make edits to the prompt or environment to test these hypotheses. While the CoT is not always faithful, it is a rich source of unsupervised insight that can guide the collection of more rigorous evidence. To evaluate our protocol, we create a suite of six agentic environments where models exhibit concerning behavior, and apply it to each. We establish that Kimi K2 Thinking takes shortcuts due to a genuine disposition towards low-effort actions, by showing this hypothesis successfully predicts its behavior. Through counterfactual experiments, we show DeepSeek R1 deceives out of a desire to be consistent with a previous instance of itself. Our methods nonetheless leave significant room for refinement. For example, when we test whether Kimi K2 Thinking believes it is violating user intent, we find no evidence of such a belief, but without positive controls we cannot confirm our tests would detect it. Overall, we find our simple protocol provides a strong baseline that we hope future work will improve upon. More broadly, our work is a concrete step in developing the growing field of model forensics. |
| 2026-06-24 | [The Unfireable Safety Kernel: Execution-Time AI Alignment for AI Agents and Other Escapable AI Systems](http://arxiv.org/abs/2606.26057v1) | Seth Dobrin, Łukasz Chmiel | AI agents are granted access to tools, APIs, and other infrastructure, making them active principals in those systems. The dominant approach places controls inside the agent's own runtime: system prompts, output filters, and guardrail libraries. Any control in the agent's address space is reachable by inputs that influence it; this generalizes to any AI system with sufficient reach into its own runtime, a class we term escapable AI systems.   We identify four properties that an authorization mechanism must satisfy for architectural control rather than for cooperative requests: process separation, pre-action enforcement on a structurally only path, fail-closed at both the request and system levels, and externalized signed evidence verifiable outside the controlled system's trust boundary. We position this layer as execution-time AI alignment, complementing training-time alignment (RLHF, Constitutional AI) and inference-time alignment.   We present the Unfireable Safety Kernel, a Rust reference implementation realizing all four. Its fail-closed invariant is machine-checked at two levels: an SMT theorem (Z3) and an exhaustive bounded-model-checking proof of the production decision function (Kani, 4/4 harnesses). A Python-to-Rust migration was gated on byte-equivalence (1000/1000 fixtures; 17/17 adversarial classes). We evaluate the kernel governing a live, escapable AI system, a deterministic, self-improving world model, against an escape-seeking adversary driving its real self-modification seam: across 1,000 self-modifications, all 704 attempts on the safety-critical core are refused, with no escape; a further 300, under the operator kill switch, are also refused. A separate campaign of 6,240 authorization round-trips had no successful bypass. Against 3 contemporary systems claiming the agent control plane, the agent invokes control; here, it lacks that choice. |
| 2026-06-24 | [G2DP: Diffusion Planning with Spatio-Temporal Grid Guidance](http://arxiv.org/abs/2606.26017v1) | Hang Yu, Ye Jin et al. | In autonomous driving, diffusion-based planners have emerged as a promising paradigm for robust motion planning in dense and interactive traffic, as they can effectively model diverse driving behaviors. However, their inherent stochasticity often requires explicit guidance during denoising to ensure safety and route adherence for robust closed-loop execution. Existing guidance typically relies on sparse, entity-centric geometric queries or post-hoc refinement, yielding limited situational awareness and fragile performance in interactive scenes. To address this issue, we propose G2DP (Grid-Guided Diffusion Planning), a diffusion-based planner that directly enforces dense environmental constraints through inference-time guidance. Specifically, G2DP constructs a differentiable spatio-temporal cost volume by fusing probabilistic future occupancy distributions with a route-progress map. By formulating this volume as a continuous safety energy functional, it injects dense gradients directly into the denoising loop, actively steering trajectory generation toward collision-free and progress-optimal regions. Extensive closed-loop evaluations show that G2DP achieves state-of-the-art performance on nuPlan, outperforming the strongest imitation-learning baseline by +7.2 points in reactive score. It further maintains top scores in zero-shot transfers to interPlan and DeepScenario benchmarks, with collision avoidance improving by +10.15 over the unguided approach on interPlan. These results demonstrate that spatio-temporal cost grids serve as an effective representation for robust guidance in diffusion-based planning. |
| 2026-06-24 | [The Tatoxa System for Text Detoxification in Low-Resource Languages: The Case of Tatar](http://arxiv.org/abs/2606.26015v1) | Ilseyar Alimova, Bogdan Monogov et al. | Text detoxification, the automated detection and mitigation of abusive and harmful content, is essential for ensuring the safety of online communities and protecting users. However, low resource languages such as Tatar have received little research attention. In this paper we present Tatoxa, a novel state-of-the-art system for text detoxification in the Tatar language. Comparative experiments show that the proposed approach outperforms existing open source and proprietary commercial LLMs on key quality metrics. We also introduce a new dataset for text detoxification in Tatar, designed for fine tuning and evaluation in low resource settings. Finally, cross lingual transfer experiments indicate that transfer from other languages, including the culturally close Russian, performs significantly worse than training on native Tatar data even when a large Russian corpus is available. |
| 2026-06-24 | [SpeechEQ: Benchmarking Emotional Intelligence Quotient in Socially Aware Voice Conversational Models](http://arxiv.org/abs/2606.25990v1) | Liang-Yuan Wu, Zih-Ching Chen et al. | As multimodal conversational systems increasingly engage in spoken interaction, their ability to navigate paralinguistic social cues has become a critical bottleneck for natural human-AI communication. However, existing evaluations of machine emotional intelligence assess reasoning exclusively through isolated text or passive acoustic perception, overlooking the complex cross-modal reasoning required for active, multi-turn dialogue. We introduce \textsc{SpeechEQ}, a comprehensive framework designed to evaluate the sociolinguistic reasoning of Speech-Language Models (SLMs). The framework includes a validated dataset of 2,265 dialogues across 15 Emotional Quotient (EQ) subscales grounded in EQ-i 2.0 theory, along with a multi-turn evaluation protocol measured by our proposed Spoken EQ (SEQ) score inspired by human EQ assessments. Experiments show limitations in how both existing Speech Emotion Recognition and end-to-end Speech-Language Models understand and apply paralinguistic cues through speech. While end-to-end architectures outperform cascaded systems, \textsc{SpeechEQ} reveals that current multimodal models remain bottlenecked by a text-reliant ``modality shortcut,'' an alignment-induced ``safety trap,'' and ``contextual amnesia,'' highlighting the barriers to truly emotionally aware AI. Our benchmark can be accessed at https://huggingface.co/datasets/SpeechEQ/SpeechEQ and demo page at https://binomial14.github.io/speecheq-demo/ |
| 2026-06-24 | [Molexar: A Unified Multimodal Molecular Foundation Model for Drug Design](http://arxiv.org/abs/2606.25865v1) | Haoyu Lin, Yiyan Liao et al. | Molecular generation is a central challenge in drug discovery, requiring models that explore vast chemical space while satisfying diverse design constraints. We present Molexar, a unified multimodal molecular foundation model built on Fragment-SELFIES, a robust, fragment-aware molecular language with validity-preserving decoding and explicit fragment structure. A pretrained autoregressive decoder learns the Fragment-SELFIES syntax and molecular distribution; supervised fine-tuning (SFT) then trains the same decoder on condition-molecule pairs spanning scalar molecular properties, pharmacophore fingerprints, protein sequences, and binding pockets, injecting each condition by in-place replacement of value-token embeddings so that all generation modes share one autoregressive path. Molexar achieves strong efficiency at a small parameter count while matching or exceeding larger models. The pretrained model reaches 100% validity and high drug-likeness in unconditional and fragment-constrained generation; the SFT model follows single- and multi-property instructions and remains competitive on target-conditioned generation on the CrossDocked2020 test set. On MolGenBench, Molexar further generates molecules with favorable safety and potency. These results establish Molexar as a practical unified foundation for computational chemistry and drug-design workflows. |
| 2026-06-24 | [Lyapunov Optimization based Queue-aware Traffic Shaping for 5G-TSN in Industrial Environments](http://arxiv.org/abs/2606.25823v1) | Kouros Zanbouri, Md Noor-A-Rahim et al. | Manufacturing companies look increasingly at Private 5G networks to manage Automated Guided Vehicles (AGVs). While 5G promises Ultra-Reliable Low Latency Communication (URLLC), its service quality is challenged by industrial environments characterized by dense metallic structures, which frequently cause line-of-sight (LOS) blockage events, causing deep fades in received signal levels that can degrade channel capacity to near-zero. Standard transport protocols and rate adaptation mechanisms fail to react sufficiently fast to these deep fades, resulting in bufferbloat and latency spikes that violate safety margins. In this paper, we propose a cross-layer rate control algorithm based on Lyapunov Drift-plus-Penalty theory. The proposed controller dynamically optimizes the trade-off between service utility and queue stability based on instantaneous buffer states, without requiring predictive channel models. We validate the approach using a trace-driven simulation framework that replicates the stochastic dynamics of 5G blockage using 3GPP-compliant capacity data. Numerical results demonstrate that while baseline scheduling schemes suffer from catastrophic queue accumulation, leading to excessive delays upon reconnection, the proposed Lyapunov controller effectively eliminates bufferbloat. By preventing congestion-induced backlog, the system ensures immediate low-latency operation as soon as the channel recovers, maintaining near-deterministic behavior. |
| 2026-06-24 | [Do Encoders Suffice? A Systematic Comparison of Encoder and Decoder Safety Judges for LLM Adversarial Evaluation](http://arxiv.org/abs/2606.25782v1) | Han Jeon, Shiv Medler et al. | With the widespread adoption of large language models (LLMs) in chatbots and everyday applications, companies increasingly need guardrails that are effective while remaining low-cost and low-latency. Safety evaluation of LLM outputs has generally relied on LLM-based judges, which can be effective but are often slow and expensive to deploy at scale.   In this paper, we evaluate whether fine-tuned modern encoder classifiers from the ModernBERT family, including ModernBERT and Ettin, can reliably identify harmful LLM outputs in user-model conversations without substantial performance loss relative to LLM-based judges. We benchmark these encoder classifiers against rule-based prefix matching, fine-tuned LLM classifiers, and LLM judges using a range of judge-prompting strategies across open-source adversarial datasets.   The LLM judges include evaluation methodologies from StrongReject, ShieldGemma, JailbreakBench, AILuminate, SorryBench, and a Claude-as-a-judge setup, as well as fine-tuned safety classifiers such as LlamaGuard 3 and LlamaGuard 4. The encoder classifiers are fine-tuned on judge-labeled data using a majority-voting label strategy and are then evaluated on a gold-standard holdout dataset to assess their performance relative to LLM judges.   We report absolute performance using F1 score, false negative rate, and precision-recall metrics. We also break down results by attack technique, including single-turn prompting, decomposition, escalation, and context manipulation, to identify where encoder classifiers align with or diverge from LLM-based judges. Our findings provide guidance on when encoder classifiers can serve as cost- and latency-efficient alternatives to LLM-based safety evaluation. |
| 2026-06-24 | [Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and GUI Grounding Datasets](http://arxiv.org/abs/2606.25760v1) | Divake Kumar, Sina Tayebati et al. | Computer-use agents turn vision-language model (VLM) predictions into executable GUI clicks, so reliable uncertainty estimates are essential for rejection, calibration, miss-severity ranking, and spatial safety regions. Yet evidence on post-hoc uncertainty quantification (UQ) for these agents is fragmented across isolated model and dataset pairs, leaving it unclear whether UQ rankings stay stable when the agent, benchmark, or observable interface changes. We present Argus, a cross-regime benchmark for post-hoc UQ in single-step executable GUI grounding: a 27-method open-weight matrix over 4 VLM agents and 4 datasets, plus an 8-method closed-source matrix across 3 frontier vendors where logits, hidden states, and attention maps are unavailable. Evaluated methods span logit-based scores, sampling and consistency measures, hidden-state and density estimators (Mahalanobis, SAPLMA), attention-based scores, P(True) and verbalised-confidence prompting, and split-conformal prediction. The main finding is selective transfer: UQ rankings are stable across datasets for a fixed model, but degrade across model classes and observable interfaces. Hidden-state and density methods are the most stable open-weight family, while CoCoA-1MCA, Focus, sampling-based scores, and verbalised self-assessment win in specific regimes. Within-model ranking transfer is strong (Spearman rho up to 0.969), but cross-tier transfer to closed-source vendors averages only +0.08, so closed-source UQ should be reranked on the target rather than extrapolated. Conformal click regions show score-level discrimination is not enough for deployment: locally weighted disks shrink radii by 40-60% when the plug-in UQ is calibrated, but coverage degrades under calibration-test or interface mismatch. We release per-item records, calibration/test splits, UQ scores, and analysis scripts for regime-aware UQ selection in GUI agents. |
| 2026-06-24 | [RAS: Measuring LLM Safety Through Refusal Alignment](http://arxiv.org/abs/2606.25750v1) | Chang-Chieh Huang, Yan-Lun Chen et al. | Safety evaluation of large language models (LLMs) is commonly performed by querying models with unsafe or jailbreak prompts and judging whether their outputs violate a safety policy. Although useful, output-level evaluation is expensive, sensitive to judge choice, and easily tied to fixed question banks. We propose **SafeVec**, a white-box evaluation procedure that measures safety from internal representations rather than generated answers. **SafeVec** first extracts layer-wise refusal directions from a safety-aligned reference model, then selects stable layer windows where safe and unsafe behaviors are separable, and finally scores a target model by measuring whether its hidden states align with these refusal directions under unsafe and jailbreak prompts. The resulting metric, **RAS** (**R**efusal **A**lignment **S**core), maps representation-level refusal alignment to a calibrated 0-100 safety score. Across `Llama`, `Gemma`, and `Qwen` model families, RAS separates aligned models from uncensored and abliterated variants, tracks output-level attack success rate, and is substantially faster than judge-based evaluation. These results suggest that refusal alignment provides a compact and efficient signal for white-box LLM safety evaluation. |
| 2026-06-24 | [GUI agent: Guided Exploration of User-Sensitive Screens](http://arxiv.org/abs/2606.25705v1) | Aradhana Nayak, Mussadiq Nazeer et al. | LLM agents are increasingly being used to automate tasks for users within an open GUI environment. They inevitably encounter screens containing user-sensitive information, for which takeover of task execution by the user is highly desirable or even necessary. State-of-the-art LLM-driven agents are usually fine-tuned to complete tasks regardless of the safety implications of their actions. This makes their real-world deployment difficult and adversely affects the reliability. Therefore, it is crucial to identify and categorize user-sensitive states and define user-sensitive queries. This dataset would be to engineers to recognize and request handover to the user in critical scenarios. This short paper develops an explorer agent that systematically explores the query space starting from one demonstrated task to identify queries that, if executed, would lead to user-sensitive states in a GUI environment. |
| 2026-06-24 | [Falcon: Functional Assembly and Language for Compositional Reasoning in X-ray](http://arxiv.org/abs/2606.25701v1) | Yonathan Michael, Mohamad Alansari et al. | Conventional vision-language models are largely object-centric, focusing on detecting and describing individual entities. In safety-critical X-ray baggage screening, however, threat often emerges not from a single object but from the functional compatibility of spatially dispersed components, such as batteries, detonators, and explosive charges. We formalize this setting as \emph{compositional threat reasoning}, where risk is modeled as a relational property of grounded regions rather than an independent detection outcome. We introduce \textbf{Falcon}, a multimodal framework that abstracts segmentation-aware region features into a structured safety state capturing component presence, pairwise functional compatibility, and scene-level risk. This structured representation is injected into the language model as an explicit intermediate interface, encouraging relationally consistent and safety-aware reasoning. To evaluate this problem, we present \textbf{Falcon-X}, a benchmark that unifies dense grounding with structured supervision over component completeness and risk inference in cluttered X-ray imagery. Experiments show that while existing multimodal models adapt to appearance, they struggle with compositional safety reasoning. Falcon improves functional grounding and produces more coherent threat assessments, establishing compositional safety reasoning as a distinct evaluation paradigm for multimodal systems. |
| 2026-06-24 | [Auto-Labelling-Based Domain Transfer for 3D Object Detection on a Bicycle-Mounted LiDAR Platform](http://arxiv.org/abs/2606.25652v1) | Mario Finkbeiner, Max A. Buettner et al. | Reliable 3D perception of vulnerable road users (VRUs) such as cyclists and pedestrians is essential for their safety in urban traffic and a core requirement for autonomous driving (AD). Alongside advances in vehicle-based perception, research increasingly equips bicycles with sensors to study traffic from a perspective native to VRUs. Such platforms still rely on LiDAR detectors originally trained on vehicle data, yet annotated 3D data from a cyclist's perspective is scarce. How well these detectors generalise to this setting has not been evaluated. We present a 3D object detection benchmark of 1,027 annotated LiDAR keyframes (over 18,000 3D bounding boxes) from the FUSE-Bike platform in urban Munich. We evaluate four nuScenes-pre-trained detectors against 1,854 human-verified ground-truth (GT) boxes both in their original form and after finetuning on training labels produced by a VRU-dedicated auto-labelling pipeline that requires no manual annotation. The zero-shot domain gap is concentrated on the VRU classes. Finetuning recovers most of it, improving mean average precision (mAP) by up to 23.4 points with the largest gains on pedestrians and cyclists, and the adapted detectors even surpass the quality of the auto-labels they were trained on. The benchmark provides a reproducible baseline for VRU-centric 3D detection and shows that auto-labels are a viable substitute for manual annotation when adapting vehicle-trained detectors to a cyclist platform. |
| 2026-06-24 | [MedGuards: Multi-Agent System for Reliable Medical Error Detection and Correction](http://arxiv.org/abs/2606.25651v1) | Congbo Ma, Hu Wang et al. | As Large Language Models (LLMs) are increasingly deployed in healthcare settings, accurate error detection and correction in generated or existing text becomes critical, as even minor mistakes can pose risks to patient safety. Existing methods for error detection and correction, including automated checks and heuristic-based approaches, do not generalize well across unseen datasets. In this paper, we propose MedGuards as a medical safety guardrail, which is a new framework that treats medical error detection and correction as a multi-agent in-context learning task. Specialized agents separately detect, localize, and correct errors, while a confidence-guided arbitration mechanism resolves disagreements using reasoning traces and confidence scores. This design enhances interpretability, robustness, and adaptability, without requiring additional training of the base LLMs. Additionally, we introduce the Keyword-Prioritized Correction Score (KPCS), a new evaluation metric that considers whether critical keywords within the reference text are generated correctly, providing a more comprehensive assessment than conventional metrics. Experiments across four multilingual medical datasets consisting of clinical notes demonstrate significant improvements by the proposed framework across several metrics and models. Our aim is to enable safer deployment of LLMs in real-world healthcare applications. For reproducibility, we make our code publicly available at https://github.com/congboma/MedErrBench. |
| 2026-06-24 | [Event-Adaptive Motion Planning with Distilled Vision-Language Model in Safety-Critical Situations](http://arxiv.org/abs/2606.25629v1) | Zhenwei Huang, Changsheng You et al. | Robot navigation in safety-critical scenarios faces significant challenges from unforeseen semantic events, where collisions arise primarily from the unpredictable behaviors of dynamic agents rather than unseen objects. While large vision-language models (VLMs) offer remarkable capabilities in commonsense reasoning, frequently invoking them within the continuous control loop introduces severe computational latency, fundamentally destabilizing physical execution. To address these challenges, we propose event-adaptive motion planning (EAMP), an efficient framework for VLM-based robot navigation. Specifically, a prompt-configurable semantic event trigger (PC-SET) selectively activates semantic intervention by continuously monitoring short temporal clips for behavioral anomalies. Upon triggering, an event-triggered distilled SemNav-VLM, fine-tuned via physically verified semantic distillation, maps detected anomalies into discrete strategy-level decisions. Subsequently, a semantic model predictive control (SMPC) module translates these strategies into dynamic reconfigurations of optimization objectives and geometric references. Extensive experiments in safety-critical logistics scenarios demonstrate that EAMP effectively aligns high-level reasoning with low-level control, significantly improving dynamic safety margins over existing baselines while preserving real-time efficiency. |
| 2026-06-24 | [Statistically Valid Hyperparameter Selection: From Tuning to Guarantees](http://arxiv.org/abs/2606.25601v1) | Amirmohammad Farzaneh, Osvaldo Simeone | Hyperparameter selection is a critical step in the deployment of modern artificial intelligence systems, given the need to tune degrees of freedom such as inference-time parameters, implementation-level settings, and thresholds driving decision rules. Despite its practical importance, hyperparameter selection is typically performed using best-effort empirical methods such as grid search or Bayesian optimization, which provide no formal statistical guarantees on reliability or safety.   This monograph presents a unified statistical framework for reliable hyperparameter selection, centered on the learn-then-test (LTT) paradigm, which formulates the problem as multiple hypothesis testing over a candidate set of hyperparameters. The framework enables the selection of hyperparameters that provably satisfy application-specific reliability requirements -- such as bounds on average risk, quantile risk, or information-theoretic constraints -- with explicit, finite-sample control of error probabilities. The supporting statistical machinery, namely p-values, e-values, and concentration inequalities, is developed from first principles in a dedicated appendix. |
| 2026-06-24 | [Reference-Free Heterogeneous Multi-Agent Reinforcement Learning for Grid-Friendly Tie-Line Power Shaping in Industrial Microgrids](http://arxiv.org/abs/2606.25599v1) | Daniyaer Paizulamua, Lin Cheng et al. | Tie-line power (TLP) shaping is a key requirement for the grid-friendly operation of industrial microgrids (IMGs). This paper studies the coordination of multi-timescale heterogeneous adjustable resources in a steel IMG to shape a grid-friendly TLP trajectory considering multiple objectives. A sequential heterogeneous-agent coordination (SHAC) framework is proposed, where process loads, hydrogen storage, and battery storage are modeled as functionally heterogeneous agents with cross-role observations, asynchronous decision intervals, role-specific rewards and critics. This design captures the heterogeneous temporal effects of different resources on the TLP trajectory and alleviates ambiguous credit assignment and weak inter-agent coordination. To ensure feasible real-time execution, process-knowledge-based action masking and feasibility projection are embedded into policy execution, and a role-aware multi-timescale actor--critic training scheme is developed for agents with different action structures and decision intervals. Numerical studies using real renewable generation and electricity market data show that SHAC effectively eliminates the dependence on predefined reference trajectories and enables adaptive 1-min online decision-making, achieving zero production failures with an average computational time of only 0.4 ms per step. Compared with the original operation, SHAC reduces the total grid purchase cost, contract-demand exceedance time, and cumulative ramp excess by 91.27\%, 98.64\%, and 96.91\%, respectively. These results demonstrate that the proposed framework improves the economic efficiency and grid friendliness of industrial microgrid operation while satisfying strict process-safety constraints and real-time computational requirements. |
| 2026-06-24 | [VPA-Guard: Defending and Benchmarking Image-to-Video Generation Against Visual Prompt Attacks](http://arxiv.org/abs/2606.25592v1) | Yining Sun, Haoyu Kang et al. | Recent advancements in Image-to-Video (I2V) generation have transformed input images from simple appearance references into interactive control interfaces where visual cues such as arrows, sketches, and emojis orchestrate complex video dynamics with unprecedented controllability. However, these seemingly innocuous static cues can be interpreted by models as executable temporal instructions, unfolding into harmful actions in the generated videos. Despite the severity of this threat, existing safety benchmarks remain predominantly focused on text-based and content-only image-based jailbreaks, leaving implicit visual prompt attacks insufficiently explored. To bridge this gap, we present VVA-Bench, the first systematic benchmark for evaluating video generation safety under categorized vision-centric prompt attacks. Extensive experiments on VVA-Bench demonstrate that state-of-the-art models are highly susceptible to such attacks, with Attack Success Rates (ASR) reaching 100.0\% on Wan 2.7 and 74.8\% on Veo 3.1. To mitigate these risks, we propose VPA-Guard, a retrieval-augmented and self-evolving defense framework. By leveraging few-shot reasoning to identify latent malicious intents, our method reduces the attack ASR by 44.2\% and the harmfulness score by 73.4\% on average, while maintaining the model's utility for legitimate user edits. Our work provides both a rigorous benchmark and an effective defense strategy to advance safe and socially responsible multimodal generation. |
| 2026-06-24 | [WOLF-VLA: Whole-Body Humanoid Optimal Locomotion Framework for Vision-Language-Action Learning](http://arxiv.org/abs/2606.25591v1) | Melya Boukheddimi, Omar Adjali et al. | Vision-Language-Action (VLA) models have recently demonstrated strong generalization in robotic manipulation, yet their applicability to whole-body, contact-rich humanoid locomotion remains severely underexplored due to data scarcity, the absence of dynamically consistent demonstrations, and the difficulty of encoding optimality and safety in learning-based pipelines. This work introduces a unified framework WOLF-VLA that integrates whole-body optimal-control (OC) motion synthesis with large-scale multi-modal dataset to train VLAs capable of generating humanoid locomotion policies directly from natural-language instructions. We construct a comprehensive dataset of dynamically feasible humanoid trajectories across six locomotion-related task families, each parameterized by environmental variations, object colors, placements, and visual distractors. We train a VLA model using the collected joint trajectories, ego-centric visual observations and natural language instruction, yielding a policy that exhibits strong reasoning and robustness to initial-condition variability, and competitive performance across several tasks and environment settings. A systematic ablation study demonstrates the impact of each modality on the model performance. The full dataset, model checkpoints, and benchmarking simulation suite will be openly released, establishing a reproducible dynamically consistent benchmark for whole-body humanoid locomotion rich VLA control and enabling future research in scalable transfer of instruction-driven locomotion policies. |
| 2026-06-24 | [How Reliable Is Your Jailbreak Judge? Calibration and Adversarial Robustness of Automated ASR Scoring](http://arxiv.org/abs/2606.25487v1) | Yang Gao | Almost every paper on LLM jailbreaks and prompt injection reports an attack-success rate (ASR), and that number is assigned not by people but by an automated judge: either a safety classifier trained for the task, or a general chat model prompted to grade. The judge is rarely checked. We check it. Using 596 human-labeled completions from the HarmBench classifier validation set, we compare the two judge families against human majority votes and then attack them. The two families fail in opposite ways. The dedicated classifier over-flags (precision 0.835, recall 0.974); three different LLM-as-judges keep high precision (0.81 to 0.94) but show erratic recall (0.06 to 0.65), so the same responses produce very different ASR depending on which judge scores them. The two families also differ sharply in robustness. Wrappers that leave the harmful text untouched and only add benign framing flip every LLM-judge between 57% and 100% of the time, and a single prepended refusal sentence accounts for much of this (39% to 88%). The dedicated classifier resists these surface attacks (at most 6.7%), but a white-box GCG attack on its open weights flips 70% of confident true positives (21 of 30; 95% CI 54 to 86%) even at a small optimization budget. A two-annotator audit confirms the attacks leave the harm intact: every one of 80 sampled flips still contained the harmful content. Because a large and growing share of reported ASR comes from LLM-judges, many such numbers are unreliable both on average and under deliberate pressure. We recommend that papers report judge precision and recall on a human-labeled slice, report ASR corrected for judge precision, and include an adversarial check of the judge. Our code is released. |

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



