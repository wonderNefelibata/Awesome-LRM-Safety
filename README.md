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
| 2026-03-26 | [Back to Basics: Revisiting ASR in the Age of Voice Agents](http://arxiv.org/abs/2603.25727v1) | Geeyang Tay, Wentao Ma et al. | Automatic speech recognition (ASR) systems have achieved near-human accuracy on curated benchmarks, yet still fail in real-world voice agents under conditions that current evaluations do not systematically cover. Without diagnostic tools that isolate specific failure factors, practitioners cannot anticipate which conditions, in which languages, will cause what degree of degradation. We introduce WildASR, a multilingual (four-language) diagnostic benchmark sourced entirely from real human speech that factorizes ASR robustness along three axes: environmental degradation, demographic shift, and linguistic diversity. Evaluating seven widely used ASR systems, we find severe and uneven performance degradation, and model robustness does not transfer across languages or conditions. Critically, models often hallucinate plausible but unspoken content under partial or degraded inputs, creating concrete safety risks for downstream agent behavior. Our results demonstrate that targeted, factor-isolated evaluation is essential for understanding and improving ASR reliability in production systems. Besides the benchmark itself, we also present three analytical tools that practitioners can use to guide deployment decisions. |
| 2026-03-26 | [Uncertainty-Guided Label Rebalancing for CPS Safety Monitoring](http://arxiv.org/abs/2603.25670v1) | John Ayotunde, Qinghua Xu et al. | Safety monitoring is essential for Cyber-Physical Systems (CPSs). However, unsafe events are rare in real-world CPS operations, creating an extreme class imbalance that degrades safety predictors. Standard rebalancing techniques perform poorly on time-series CPS telemetry, either generating unrealistic synthetic samples or overfitting on the minority class. Meanwhile, behavioral uncertainty in CPS operations, defined as the degree of doubt or uncertainty in CPS decisions , is often correlated with safety outcomes but unexplored in safety monitoring. To that end, we propose U-Balance, a supervised approach that leverages behavioral uncertainty to rebalance imbalanced datasets prior to training a safety predictor. U-Balance first trains a GatedMLP-based uncertainty predictor that summarizes each telemetry window into distributional kinematic features and outputs an uncertainty score. It then applies an uncertainty-guided label rebalancing (uLNR) mechanism that probabilistically relabels \textit{safe}-labeled windows with unusually high uncertainty as \textit{unsafe}, thereby enriching the minority class with informative boundary samples without synthesizing new data. Finally, a safety predictor is trained on the rebalanced dataset for safety monitoring. We evaluate U-Balance on a large-scale UAV benchmark with a 46:1 safe-to-unsafe ratio. Results confirm a moderate but significant correlation between behavioral uncertainty and safety. We then identify uLNR as the most effective strategy to exploit uncertainty information, compared to direct early and late fusion. U-Balance achieves a 0.806 F1 score, outperforming the strongest baseline by 14.3 percentage points, while maintaining competitive inference efficiency. Ablation studies confirm that both the GatedMLP-based uncertainty predictor and the uLNR mechanism contribute significantly to U-Balance's effectiveness. |
| 2026-03-26 | [Experimental Analysis of FreeRTOS Dependability through Targeted Fault Injection Campaigns](http://arxiv.org/abs/2603.25666v1) | Luca Mannella, Stefano Di Carlo et al. | Real-Time Operating Systems (RTOSes) play a crucial role in safety-critical domains, where deterministic and predictable task execution is essential. Yet they are increasingly exposed to ionizing radiation, which can compromise system dependability.   To assess FreeRTOS under such conditions, we introduce KRONOS, a software-based, non-intrusive post-propagation Fault Injection (FI) framework that injects transient and permanent faults into Operating System-visible kernel data structures without specialized hardware or debug interfaces. Using KRONOS, we conduct an extensive FI campaign on core FreeRTOS kernel components, including scheduler-related variables and Task Control Blocks (TCBs), characterizing the impact of kernel-level corruptions on functional correctness, timing behavior, and availability.   The results show that corruption of pointer and key scheduler-related variables frequently leads to crashes, whereas many TCB fields have only a limited impact on system availability. |
| 2026-03-26 | [Radiation safety considerations for ultrafast lasers beyond laser machining](http://arxiv.org/abs/2603.25585v1) | Simon Bohlen, Julian Holland et al. | The interaction of ultrafast lasers with plasmas has been studied for many years, primarily with respect to fundamental emission mechanisms. Only in recent years has ionizing radiation emerged as a safety concern in ultrafast laser-based material processing, where high pulse energies, repetition rates, and average powers, combined with continuous material supply, can lead to sustained X-ray emission. These processing-specific findings have informed German radiation protection legislation, which mandates notification or approval for laser systems exceeding irradiances of $1 \times 10^{13}~W/cm^2$. However, this threshold does not distinguish between material processing and other ultrafast laser applications. In this work, we show that the conditions required for X-ray generation are highly specific and are typically only met during material processing. We assess the applicability of existing radiation studies to non-processing environments and present experimental results demonstrating negligible or no dose production under representative laboratory conditions, such as ultrafast laser interactions with underdense gas or stationary solid targets. We conclude that current legislation generalizes a processing-specific hazard to all ultrafast laser applications and does not adequately reflect the relevant physical conditions. |
| 2026-03-26 | [Synchronous Signal Temporal Logic for Decidable Verification of Cyber-Physical Systems](http://arxiv.org/abs/2603.25531v1) | Partha Roop, Sobhan Chatterjee et al. | Many Cyber Physical System (CPS) work in a safety-critical environment, where correct execution, reliability and trustworthiness are essential. Signal Temporal Logic (STL) provides a formal framework for checking safety-critical CPS. However, static verification of STL is undecidable in general, except when we want to verify using run-time-based methods, which have limitations. We propose Synchronous Signal Temporal Logic (SSTL), a decidable fragment of STL, which admits static safety and liveness property verification. In SSTL, we assume that a signal is sampled at fixed discrete steps, called ticks, and then propose a hypothesis, called the Signal Invariance Hypothesis (SIH), which is inspired by a similar hypothesis for synchronous programs. We define the syntax and semantics of SSTL and show that SIH is a necessary and sufficient condition for equivalence between an STL formula and its SSTL counterpart. By translating SSTL to LTL_P (LTL defined over predicates), we enable decidable model checking using the SPIN model checker. We demonstrate the approach on a 33-node human heart model and other case studies. |
| 2026-03-26 | [NERO-Net: A Neuroevolutionary Approach for the Design of Adversarially Robust CNNs](http://arxiv.org/abs/2603.25517v1) | Inês Valentim, Nuno Antunes et al. | Neuroevolution automates the complex task of neural network design but often ignores the inherent adversarial fragility of evolved models which is a barrier to adoption in safety-critical scenarios. While robust training methods have received significant attention, the design of architectures exhibiting intrinsic robustness remains largely unexplored. In this paper, we propose NERO-Net, a neuroevolutionary approach to design convolutional neural networks better equipped to resist adversarial attacks. Our search strategy isolates architectural influence on robustness by avoiding adversarial training during the evolutionary loop. As such, our fitness function promotes candidates that, even trained with standard (non-robust) methods, achieve high post-attack accuracy without sacrificing the accuracy on clean samples. We assess NERO-Net on CIFAR-10 with a specific focus on $L_\infty$-robustness. In particular, the fittest individual emerged from evolutionary search with 33% accuracy against FGSM, used as an efficient estimator for robustness during the search phase, while maintaining 87% clean accuracy. Further standard training of this individual boosted these metrics to 47% adversarial and 93% clean accuracy, suggesting inherent architectural robustness. Adversarial training brings the overall accuracy of the model up to 40% against AutoAttack. |
| 2026-03-26 | [Knowledge-Guided Failure Prediction: Detecting When Object Detectors Miss Safety-Critical Objects](http://arxiv.org/abs/2603.25499v1) | Jakob Paul Zimmermann, Gerrit Holzbach et al. | Object detectors deployed in safety-critical environments can fail silently, e.g. missing pedestrians, workers, or other safety-critical objects without emitting any warning. Traditional Out Of Distribution (OOD) detection methods focus on identifying unfamiliar inputs, but do not directly predict functional failures of the detector itself. We introduce Knowledge Guided Failure Prediction (KGFP), a representation-based monitoring framework that treats missed safety-critical detections as anomalies to be detected at runtime. KGFP measures semantic misalignment between internal object detector features and visual foundation model embeddings using a dual-encoder architecture with an angular distance metric. A key property is that when either the detector is operating outside its competence or the visual foundation model itself encounters novel inputs, the two embeddings diverge, producing a high-angle signal that reliably flags unsafe images. We compare our novel KGFS method to baseline OOD detection methods. On COCO person detection, applying KGFP as a selective-prediction gate raises person recall among accepted images from 64.3% to 84.5% at 5% False Positive Rate (FPR), and maintains strong performance across six COCO-O visual domains, outperforming OOD baselines by large margins. Our code, models, and features are published at https://gitlab.cc-asp.fraunhofer.de/iosb_public/KGFP. |
| 2026-03-26 | [Temporally Decoupled Diffusion Planning for Autonomous Driving](http://arxiv.org/abs/2603.25462v1) | Xiang Li, Bikun Wang et al. | Motion planning in dynamic urban environments requires balancing immediate safety with long-term goals. While diffusion models effectively capture multi-modal decision-making, existing approaches treat trajectories as monolithic entities, overlooking heterogeneous temporal dependencies where near-term plans are constrained by instantaneous dynamics and far-term plans by navigational goals. To address this, we propose Temporally Decoupled Diffusion Model (TDDM), which reformulates trajectory generation via a noise-as-mask paradigm. By partitioning trajectories into segments with independent noise levels, we implicitly treat high noise as information voids and weak noise as contextual cues. This compels the model to reconstruct corrupted near-term states by leveraging internal correlations with better-preserved temporal contexts. Architecturally, we introduce a Temporally Decoupled Adaptive Layer Normalization (TD-AdaLN) to inject segment-specific timesteps. During inference, our Asymmetric Temporal Classifier-Free Guidance utilizes weakly noised far-term priors to guide immediate path generation. Evaluations on the nuPlan benchmark show TDDM approaches or exceeds state-of-the-art baselines, particularly excelling in the challenging Test14-hard subset. |
| 2026-03-26 | [Modernising Reinforcement Learning-Based Navigation for Embodied Semantic Scene Graph Generation](http://arxiv.org/abs/2603.25415v1) | Roman Kueble, Marco Hueller et al. | Semantic world models enable embodied agents to reason about objects, relations, and spatial context beyond purely geometric representations. In Organic Computing, such models are a key enabler for objective-driven self-adaptation under uncertainty and resource constraints. The core challenge is to acquire observations maximising model quality and downstream usefulness within a limited action budget.   Semantic scene graphs (SSGs) provide a structured and compact representation for this purpose. However, constructing them within a finite action horizon requires exploration strategies that trade off information gain against navigation cost and decide when additional actions yield diminishing returns.   This work presents a modular navigation component for Embodied Semantic Scene Graph Generation and modernises its decision-making by replacing the policy-optimisation method and revisiting the discrete action formulation. We study compact and finer-grained, larger discrete motion sets and compare a single-head policy over atomic actions with a factorised multi-head policy over action components. We evaluate curriculum learning and optional depth-based collision supervision, and assess SSG completeness, execution safety, and navigation behaviour.   Results show that replacing the optimisation algorithm alone improves SSG completeness by 21\% relative to the baseline under identical reward shaping. Depth mainly affects execution safety (collision-free motion), while completeness remains largely unchanged. Combining modern optimisation with a finer-grained, factorised action representation yields the strongest overall completeness--efficiency trade-off. |
| 2026-03-26 | [Beyond Content Safety: Real-Time Monitoring for Reasoning Vulnerabilities in Large Language Models](http://arxiv.org/abs/2603.25412v1) | Xunguang Wang, Yuguang Zhou et al. | Large language models (LLMs) increasingly rely on explicit chain-of-thought (CoT) reasoning to solve complex tasks, yet the safety of the reasoning process itself remains largely unaddressed. Existing work on LLM safety focuses on content safety--detecting harmful, biased, or factually incorrect outputs -- and treats the reasoning chain as an opaque intermediate artifact. We identify reasoning safety as an orthogonal and equally critical security dimension: the requirement that a model's reasoning trajectory be logically consistent, computationally efficient, and resistant to adversarial manipulation. We make three contributions. First, we formally define reasoning safety and introduce a nine-category taxonomy of unsafe reasoning behaviors, covering input parsing errors, reasoning execution errors, and process management errors. Second, we conduct a large-scale prevalence study annotating 4111 reasoning chains from both natural reasoning benchmarks and four adversarial attack methods (reasoning hijacking and denial-of-service), confirming that all nine error types occur in practice and that each attack induces a mechanistically interpretable signature. Third, we propose a Reasoning Safety Monitor: an external LLM-based component that runs in parallel with the target model, inspects each reasoning step in real time via a taxonomy-embedded prompt, and dispatches an interrupt signal upon detecting unsafe behavior. Evaluation on a 450-chain static benchmark shows that our monitor achieves up to 84.88\% step-level localization accuracy and 85.37\% error-type classification accuracy, outperforming hallucination detectors and process reward model baselines by substantial margins. These results demonstrate that reasoning-level monitoring is both necessary and practically achievable, and establish reasoning safety as a foundational concern for the secure deployment of large reasoning models. |
| 2026-03-26 | [SafeGuard ASF: SR Agentic Humanoid Robot System for Autonomous Industrial Safety](http://arxiv.org/abs/2603.25353v1) | Thanh Nguyen Canh, Thang Tran Viet et al. | The rise of unmanned ``dark factories'' operating without human presence demands autonomous safety systems capable of detecting and responding to multiple hazard types. We present SafeGuard ASF (Agentic Security Fleet), a comprehensive framework deploying humanoid robots for autonomous hazard detection in industrial environments. Our system integrates multi-modal perception (RGB-D imaging), a ReAct-based agentic reasoning framework, and learned locomotion policies on the Unitree G1 humanoid platform. We address three critical hazard scenarios: fire and smoke detection, abnormal temperature monitoring in pipelines, and intruder detection in restricted zones. Our perception pipeline achieves 94.2% mAP for fire or smoke detection with 127ms latency. We train multiple locomotion policies, including dance motion tracking and velocity control, using Unitree RL Lab with PPO, demonstrating stable convergence within 80,000 training iterations. We validate our system in both simulation and real-world environments, demonstrating autonomous patrol, human detection with visual perception, and obstacle avoidance capabilities. The proposed ToolOrchestra action framework enables structured decision-making through perception, reasoning, and actuation tools. |
| 2026-03-26 | [Macroscopic Characteristics of Mixed Traffic Flow with Deep Reinforcement Learning Based Automated and Human-Driven Vehicles](http://arxiv.org/abs/2603.25328v1) | Pankaj Kumar, Pranamesh Chakraborty et al. | Automated Vehicle (AV) control in mixed traffic, where AVs coexist with human-driven vehicles, poses significant challenges in balancing safety, efficiency, comfort, fuel efficiency, and compliance with traffic rules while capturing heterogeneous driver behavior. Traditional car-following models, such as the Intelligent Driver Model (IDM), often struggle to generalize across diverse traffic scenarios and typically do not account for fuel efficiency, motivating the use of learning-based approaches. Although Deep Reinforcement Learning (DRL) has shown strong microscopic performance in car-following conditions, its macroscopic traffic flow characteristics remain underexplored. This study focuses on analyzing the macroscopic traffic flow characteristics and fuel efficiency of DRL-based models in mixed traffic. A Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm is implemented for AVs' control and trained using the NGSIM highway dataset, enabling realistic interaction with human-driven vehicles. Traffic performance is evaluated using the Fundamental Diagram (FD) under varying driver heterogeneity, heterogeneous time-gap penetration levels, and different shares of RL-controlled vehicles. A macroscopic level comparison of fuel efficiency between the RL-based AV model and the IDM is also conducted. Results show that traffic performance is sensitive to the distribution of safe time gaps and the proportion of RL vehicles. Transitioning from fully human-driven to fully RL-controlled traffic can increase road capacity by approximately 7.52%. Further, RL-based AVs also improve average fuel efficiency by about 28.98% at higher speeds (above 50 km/h), and by 1.86% at lower speeds (below 50 km/h) compared to the IDM. Overall, the DRL framework enhances traffic capacity and fuel efficiency without compromising safety. |
| 2026-03-26 | [SliderQuant: Accurate Post-Training Quantization for LLMs](http://arxiv.org/abs/2603.25284v1) | Shigeng Wang, Chao Li et al. | In this paper, we address post-training quantization (PTQ) for large language models (LLMs) from an overlooked perspective: given a pre-trained high-precision LLM, the predominant sequential quantization framework treats different layers equally, but this may be not optimal in challenging bit-width settings. We empirically study the quantization impact of different layers on model accuracy, and observe that: (1) shallow/deep layers are usually more sensitive to quantization than intermediate layers; (2) among shallow/deep layers, the most sensitive one is the first/last layer, which exhibits significantly larger quantization error than others. These empirical observations imply that the quantization design for different layers of LLMs is required on multiple levels instead of a single level shared to all layers. Motivated by this, we propose a new PTQ framework termed Sliding-layer Quantization (SliderQuant) that relies on a simple adaptive sliding quantization concept facilitated by few learnable parameters. The base component of SliderQuant is called inter-layer sliding quantization, which incorporates three types of novel sliding window designs tailored for addressing the varying quantization sensitivity of shallow, intermediate and deep layers. The other component is called intra-layer sliding quantization that leverages an incremental strategy to quantize each window. As a result, SliderQuant has a strong ability to reduce quantization errors across layers. Extensive experiments on basic language generation, zero-shot commonsense reasoning and challenging math and code tasks with various LLMs, including Llama/Llama2/Llama3/Qwen2.5 model families, DeepSeek-R1 distilled models and large MoE models, show that our method outperforms existing PTQ methods (including the latest PTQ methods using rotation transformations) for both weight-only quantization and weight-activation quantization. |
| 2026-03-26 | [Vertical Contracts for Safety Control](http://arxiv.org/abs/2603.25246v1) | Armin Pirastehzad, Bart Besselink | We propose a methodology that exploits the contract formalism to characterize the continuous-time safety control problem, which is often difficult to address, in terms of a discrete-time one, for which numerous efficient solution scheme exist. We construct contracts as pairs of assumptions and guarantees which are set-valued mappings that describe the safe boundaries within which the system must operate. By formalizing safety control as contract implementation, we develop a vertical hierarchy according to which we translate implementation from continuous to discrete time. We accomplish this by constructing a discrete-time system and a contract such that a solution to the continuous-time implementation problem can be characterized in terms of a solution to its discrete-time counterpart. We then use this characterization to construct a control input that establishes implementation in continuous time on the basis of the control sequence that achieves implementation in discrete time. |
| 2026-03-26 | [SafeMath: Inference-time Safety improves Math Accuracy](http://arxiv.org/abs/2603.25201v1) | Sagnik Basu, Subhrajit Mitra et al. | Recent research points toward LLMs being manipulated through adversarial and seemingly benign inputs, resulting in harmful, biased, or policy-violating outputs. In this paper, we study an underexplored issue concerning harmful and toxic mathematical word problems. We show that math questions, particularly those framed as natural language narratives, can serve as a subtle medium for propagating biased, unethical, or psychologically harmful content, with heightened risks in educational settings involving children. To support a systematic study of this phenomenon, we introduce ToxicGSM, a dataset of 1.9k arithmetic problems in which harmful or sensitive context is embedded while preserving mathematically well-defined reasoning tasks. Using this dataset, we audit the behaviour of existing LLMs and analyse the trade-offs between safety enforcement and mathematical correctness. We further propose SafeMath -- a safety alignment technique that reduces harmful outputs while maintaining, and in some cases improving, mathematical reasoning performance. Our results highlight the importance of disentangling linguistic harm from math reasoning and demonstrate that effective safety alignment need not come at the cost of accuracy. We release the source code and dataset at https://github.com/Swagnick99/SafeMath/tree/main. |
| 2026-03-26 | [The Competence Shadow: Theory and Bounds of AI Assistance in Safety Engineering](http://arxiv.org/abs/2603.25197v1) | Umair Siddique | As AI assistants become integrated into safety engineering workflows for Physical AI systems, a critical question emerges: does AI assistance improve safety analysis quality, or introduce systematic blind spots that surface only through post-deployment incidents? This paper develops a formal framework for AI assistance in safety analysis. We first establish why safety engineering resists benchmark-driven evaluation: safety competence is irreducibly multidimensional, constrained by context-dependent correctness, inherent incompleteness, and legitimate expert disagreement. We formalize this through a five-dimensional competence framework capturing domain knowledge, standards expertise, operational experience, contextual understanding, and judgment.   We introduce the competence shadow: the systematic narrowing of human reasoning induced by AI-generated safety analysis. The shadow is not what the AI presents, but what it prevents from being considered. We formalize four canonical human-AI collaboration structures and derive closed-form performance bounds, demonstrating that the competence shadow compounds multiplicatively to produce degradation far exceeding naive additive estimates.   The central finding is that AI assistance in safety engineering is a collaboration design problem, not a software procurement decision. The same tool degrades or improves analysis quality depending entirely on how it is used. We derive non-degradation conditions for shadow-resistant workflows and call for a shift from tool qualification toward workflow qualification for trustworthy Physical AI. |
| 2026-03-26 | [A Decade-Scale Benchmark Evaluating LLMs' Clinical Practice Guidelines Detection and Adherence in Multi-turn Conversations](http://arxiv.org/abs/2603.25196v1) | Andong Tan, Shuyu Dai et al. | Clinical practice guidelines (CPGs) play a pivotal role in ensuring evidence-based decision-making and improving patient outcomes. While Large Language Models (LLMs) are increasingly deployed in healthcare scenarios, it is unclear to which extend LLMs could identify and adhere to CPGs during conversations. To address this gap, we introduce CPGBench, an automated framework benchmarking the clinical guideline detection and adherence capabilities of LLMs in multi-turn conversations. We collect 3,418 CPG documents from 9 countries/regions and 2 international organizations published in the last decade spanning across 24 specialties. From these documents, we extract 32,155 clinical recommendations with corresponding publication institute, date, country, specialty, recommendation strength, evidence level, etc. One multi-turn conversation is generated for each recommendation accordingly to evaluate the detection and adherence capabilities of 8 leading LLMs. We find that the 71.1%-89.6% recommendations can be correctly detected, while only 3.6%-29.7% corresponding titles can be correctly referenced, revealing the gap between knowing the guideline contents and where they come from. The adherence rates range from 21.8% to 63.2% in different models, indicating a large gap between knowing the guidelines and being able to apply them. To confirm the validity of our automatic analysis, we further conduct a comprehensive human evaluation involving 56 clinicians from different specialties. To our knowledge, CPGBench is the first benchmark systematically revealing which clinical recommendations LLMs fail to detect or adhere to during conversations. Given that each clinical recommendation may affect a large population and that clinical applications are inherently safety critical, addressing these gaps is crucial for the safe and responsible deployment of LLMs in real world clinical practice. |
| 2026-03-26 | [Prompt Attack Detection with LLM-as-a-Judge and Mixture-of-Models](http://arxiv.org/abs/2603.25176v1) | Hieu Xuan Le, Benjamin Goh et al. | Prompt attacks, including jailbreaks and prompt injections, pose a critical security risk to Large Language Model (LLM) systems. In production, guardrails must mitigate these attacks under strict low-latency constraints, resulting in a deployment gap in which lightweight classifiers and rule-based systems struggle to generalize under distribution shift, while high-capacity LLM-based judges remain too slow or costly for live enforcement. In this work, we examine whether lightweight, general-purpose LLMs can reliably serve as security judges under real-world production constraints. Through careful prompt and output design, lightweight LLMs are guided through a structured reasoning process involving explicit intent decomposition, safety-signal verification, harm assessment, and self-reflection. We evaluate our method on a curated dataset combining benign queries from real-world chatbots with adversarial prompts generated via automated red teaming (ART), covering diverse and evolving patterns. Our results show that general-purpose LLMs, such as gemini-2.0-flash-lite-001, can serve as effective low-latency judges for live guardrails. This configuration is currently deployed in production as a centralized guardrail service for public service chatbots in Singapore. We additionally evaluate a Mixture-of-Models (MoM) setting to assess whether aggregating multiple LLM judges improves prompt-attack detection performance relative to single-model judges, with only modest gains observed. |
| 2026-03-26 | [SEVerA: Verified Synthesis of Self-Evolving Agents](http://arxiv.org/abs/2603.25111v1) | Debangshu Banerjee, Changming Xu et al. | Recent advances have shown the effectiveness of self-evolving LLM agents on tasks such as program repair and scientific discovery. In this paradigm, a planner LLM synthesizes an agent program that invokes parametric models, including LLMs, which are then tuned per task to improve performance. However, existing self-evolving agent frameworks provide no formal guarantees of safety or correctness. Because such programs are often executed autonomously on unseen inputs, this lack of guarantees raises reliability and security concerns. We formulate agentic code generation as a constrained learning problem, combining hard formal specifications with soft objectives capturing task utility. We introduce Formally Guarded Generative Models (FGGM), which allow the planner LLM to specify a formal output contract for each generative model call using first-order logic. Each FGGM call wraps the underlying model in a rejection sampler with a verified fallback, ensuring every returned output satisfies the contract for any input and parameter setting. Building on FGGM, we present SEVerA (Self-Evolving Verified Agents), a three-stage framework: Search synthesizes candidate parametric programs containing FGGM calls; Verification proves correctness with respect to hard constraints for all parameter values, reducing the problem to unconstrained learning; and Learning applies scalable gradient-based optimization, including GRPO-style fine-tuning, to improve the soft objective while preserving correctness. We evaluate SEVerA on Dafny program verification, symbolic math synthesis, and policy-compliant agentic tool use ($τ^2$-bench). Across tasks, SEVerA achieves zero constraint violations while improving performance over unconstrained and SOTA baselines, showing that formal behavioral constraints not only guarantee correctness but also steer synthesis toward higher-quality agents. |
| 2026-03-26 | [Layer-Specific Lipschitz Modulation for Fault-Tolerant Multimodal Representation Learning](http://arxiv.org/abs/2603.25103v1) | Diyar Altinses, Andreas Schwung | Modern multimodal systems deployed in industrial and safety-critical environments must remain reliable under partial sensor failures, signal degradation, or cross-modal inconsistencies. This work introduces a mathematically grounded framework for fault-tolerant multimodal representation learning that unifies self-supervised anomaly detection and error correction within a single architecture. Building upon a theoretical analysis of perturbation propagation, we derive Lipschitz- and Jacobian-based criteria that determine whether a neural operator amplifies or attenuates localized faults. Guided by this theory, we propose a two-stage self-supervised training scheme: pre-training a multimodal convolutional autoencoder on clean data to preserve localized anomaly signals in the latent space, and expanding it with a learnable compute block composed of dense layers for correction and contrastive objectives for anomaly identification. Furthermore, we introduce layer-specific Lipschitz modulation and gradient clipping as principled mechanisms to control sensitivity across detection and correction modules. Experimental results on multimodal fault datasets demonstrate that the proposed approach improves both anomaly detection accuracy and reconstruction under sensor corruption. Overall, this framework bridges the gap between analytical robustness guarantees and practical fault-tolerant multimodal learning. |

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



