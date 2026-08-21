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
| 2026-08-20 | [ConceptGuard: Benchmarking Context-Sensitive Unlearning in Large Language Models](http://arxiv.org/abs/2608.20338v1) | Sahil Kale, Ian Harris | Large Language Models (LLMs) increasingly require selective removal of harmful or sensitive knowledge, called unlearning, yet existing methods and benchmarks fail to evaluate this capability completely. Current approaches rely on disjoint forget and retain sets composed of independent facts, and measure success using simple and direct factual recall. This framing fails to capture a key requirement of unlearning, namely the ability to eliminate harmful behaviors while preserving benign and beneficial knowledge. We argue that effective unlearning must operate at the level of concepts, ensuring complete removal of unsafe applications while maintaining their correct and useful usage, thereby achieving conceptually meaningful and complete unlearning. To better evaluate unlearning techniques from such a practical viewpoint, we introduce the notion of dual-use concepts: concepts that can be used in both harmful and benign contexts. Building on these concepts, we construct a benchmark called ConceptGuard where forget and retain sets are explicitly complementary in concept usage. Our benchmark uniquely enables unlearning to be explored and gauged at the level of concepts, instead of sparse facts, and evaluation is intent-sensitive with the goal of maximizing contextual separation to promote safer behavior. We demonstrate that current unlearning techniques perform poorly under this setting, showing weak contextual separation alongside poor performance in ROUGE and concept-level metrics. Our results reveal strong forgetting-utility trade-offs, limited gains in contextual sensitivity, and poor consistency in concept-level control across methods, and provide ideas for unlearning approaches that better align with real-world safety requirements. Our dataset is publicly available. |
| 2026-08-20 | [Electronic Navigational Chart Change Classification](http://arxiv.org/abs/2608.20218v1) | Jacob Arndt, Abhishek Potnis et al. | Electronic Navigational Charts (ENCs) are geospatial vector datasets used in maritime navigation systems that represent hydrographic and navigational information such as depths, navigational aids, traffic schemes, and hazards. A major challenge for hydrographic offices is determining whether a given chart change poses a critical or non-critical risk to maritime safety. Existing workflows rely heavily on manual review and verification, which is labor-intensive, scales poorly with the volume of incoming chart updates, and introduces inter-analyst inconsistencies. To address this challenge, we propose a method for automated classification of ENC changes. We establish a baseline encoding scheme to translate complex vector data changes into a structured tabular format for classification models. The two crucial components of the encoding scheme include a spatial context encoder to enrich the change representations with surrounding geographic features, and an ENC attribute encoder to represent nuanced attribute-value descriptions of the modified objects. We evaluate the proposed approach across two distinct operational datasets, comprising 1,308 chart pairs containing over 100,000 individual chart modifications. Tuned gradient-boosted trees leveraging the proposed encoding schemes achieve accuracies of 90% and 94% on the two datasets, yielding a 5-7% improvement over default hyperparameterized models trained on encodings without spatial context and attribute embeddings. These results demonstrate the viability of integrating machine learning into operational geospatial pipelines to improve ENC maintenance and enhance maritime safety. Finally, our experiments demonstrate the effectiveness of simple location and spatial aggregation methods, providing a foundation for evaluating more sophisticated spatial representation learning techniques for this application. |
| 2026-08-20 | [Multi-Agent Orchestration with the Common-Sense Reasoning Capabilities of LLMs for Autonomous Driving](http://arxiv.org/abs/2608.20129v1) | Mehdi Azarafza, Faezeh Pasandideh et al. | Autonomous vehicles require robust perception and decision-making capabilities to operate in diverse and unseen scenarios. While reinforcement learning and rule-based methods can provide effective control and safety mechanisms, their performance may degrade in situations requiring contextual reasoning. Large Language Models (LLMs) have demonstrated strong capabilities in understanding multimodal information and generating contextual reasoning, however, their use for direct vehicle control can introduce latency and hallucination risks. To address these limitations, a hybrid framework is proposed. This system uses an orchestrator to coordinate PPO-trained reinforcement learning and PID control, with LLM common-sense reasoning applied throughout the framework. LLM reasoning is further employed iteratively to refine the RL reward function for dynamic driving environments. The proposed framework is evaluated in highly randomized CARLA scenarios under diverse environmental and traffic conditions. The results demonstrate the potential of integrating LLM-based reasoning with conventional autonomous driving methods while retaining structured control and safety mechanism. |
| 2026-08-20 | [Worst-Case Probability Bounds for Finite-Horizon Safety under Moment Uncertainty](http://arxiv.org/abs/2608.20121v1) | Renato Loureiro, Torbjørn Cunis | This paper addresses the problem of estimating upper bounds on the probability that a dynamical system will enter an undesirable region at some point within a finite time horizon. The primary source of uncertainty lies in the system's initial state, for which only a finite set of moments is known or within a prescribed interval. To tackle this problem, we formulate a measure-based program and propose its relaxation using the moment-sum-of-squares (moment-SOS) framework. The corresponding dual problem is introduced as a functional program, which is subsequently strengthened into a sum-of-squares (SOS) program. Notably, this dual formulation bears a structural resemblance to classical barrier function techniques for certifying system safety, with the key distinction that it yields a probabilistic certificate. The effectiveness of the proposed approach is demonstrated through multiple case studies, including a case involving an object in orbit. |
| 2026-08-20 | [Planning-Oriented End-to-End Autonomous Driving: Architectures, Evaluation, and Emerging Paradigms](http://arxiv.org/abs/2608.20111v1) | Yanchen Guan, Xingcheng Liu et al. | End-to-end autonomous driving has evolved from camera-to-control regression toward planning-oriented systems that use structured representations, trajectory-level outputs, and increasingly realistic evaluation protocols. This survey reviews this transition across behavior cloning, conditional imitation learning, privileged distillation, BEV and vectorized planning, unified perception-prediction-planning architectures, world-model-based planners, and vision-language-action systems. We argue that the key distinction in modern end-to-end driving is not whether intermediate representations are used, but whether they are learned, supervised, and evaluated to support safe, feasible, and route-compliant planning. To organize the literature, we synthesize existing methods along four axes: input representation, planning output, supervision signal, and evaluation protocol. We further examine the benchmark shift from open-loop trajectory matching to closed-loop simulation, non-reactive real-log evaluation, long-tail testing, and human-preference-aware metrics. Our analysis highlights that architectural progress is difficult to interpret without benchmark-consistent evaluation, and that displacement-based open-loop metrics alone provide limited evidence for safe and human-aligned driving. We conclude with open challenges in uncertainty-aware planning, learner-expert mismatch, runtime safety assurance, language-action grounding, world-model validation, and reproducible benchmarking. |
| 2026-08-20 | [OenoBench: A Wine-Domain Benchmark for Knowledge-Grounded Evaluation of Large Language Models](http://arxiv.org/abs/2608.20106v1) | Nikita Khudov | We introduce OenoBench, a wine-domain knowledge benchmark of 3,266 multiple-choice questions across six pillars (regions, grape varieties, viticulture, winemaking, producers, business) and four difficulty tiers. The corpus is built from 38,104 atomic, source-anchored facts extracted by 35 provenance-verified scrapers from government registries (INAO, TTB, OIV), peer-reviewed journals, and Wikipedia/Wikidata. Our methodological contribution is an LLM-driven pipeline in which language models reformat verified facts and audit the result, but never serve as the source of truth: every claim traces to a URL, every question is generated by one of five strategies across five generator families, and every question is scored by a nine-agent audit calibrated against a human gold sheet via Cohen's $κ$. Evaluating sixteen frontier configurations, we find: (i) overall accuracy spans 53%-84%, led by o3 at 83.6%; (ii) reasoning-mode lift concentrates in DeepSeek R1 (+6.8pp) and is absent in Claude Opus and Gemini Pro; (iii) Anthropic shows +9pp self preference on its own questions while Google shows -8pp inverse preference; (iv) frontier open-weight models share the cost-vs-accuracy Pareto frontier with proprietary reasoning models; and (v) every config gains around 33pp on closed-book solvable items, revealing a parametric-recall ceiling that only the contextual slice avoids. We release corpus, audit findings, human-review app, and construction code under CC-BY-SA-4.0. |
| 2026-08-20 | [On the Applicability of Safety Nets: A Safety-By-Design Solution for Certifying Neural Networks](http://arxiv.org/abs/2608.20053v1) | Johann Maximilian Christensen, Thomas Stefani et al. | The integration of Artificial Intelligence (AI) in safety-critical aviation systems presents significant challenges for certification and deployment. Aviation, often regarded as the safest form of transportation, relies on numerous safety-critical systems. For future safety-critical AI-based systems, EASA requires a Safety-by-Design approach, which can be achieved by using Safety Nets that combine neural network compression with lookup tables to ensure 100 % correct runtime behavior across the discretized operational design domain. Although Safety Nets have been studied, no comprehensive study of their performance characteristics and system design trade-offs has been conducted. This work presents the first systematic analysis of the trade-off between neural network and lookup table size in Safety Nets. By systematically comparing neural networks with diverse architectures, this study identifies optimal design parameters that minimize overall storage and memory requirements while maintaining certification compliance. Results demonstrate that architectures with 3 to 5 hidden layers, each with approximately 50 to 100 nodes, combined with one-hot encoding, achieve the best balance. In these configurations, neural networks accurately represent at least 97 % of the data, while compact lookup tables handle the remaining errors. The resulting Safety Nets reduce the system size by almost three orders of magnitude, fitting within the memory budget of current avionics hardware while guaranteeing 100 % correct outputs across the entire discretized input space, as required by EASA guidelines. This work provides the first-ever open-source implementation of Safety Nets for HCAS and VCAS with replicable results, demonstrating a practical pathway toward certifiable AI-based systems in aviation and establishing Safety Nets as a viable Safety-by-Design solution for safety-critical applications. |
| 2026-08-20 | [G-MARK: Grounded Multi-Agent Reasoning for Cooperative Driving via Knowledge Graphs](http://arxiv.org/abs/2608.19964v1) | Bhavya Gupta, Onat Gungor et al. | Autonomous driving systems must operate under partial observability, where safety-critical objects may be occluded or visible only to neighboring connected vehicles. Vehicle-to-vehicle cooperation can reduce this uncertainty, but existing cooperative driving methods often compress multi-agent evidence into latent features or hidden multimodal states. As a result, they obscure which agent observed each object, whether the object is visible to the ego vehicle, and how conflicting evidence affects downstream decisions. We propose G-MARK, a grounded multi-agent reasoning framework that converts cooperative object-centric observations into explicit provenance-aware knowledge graphs (KGs). The resulting KGs preserve object hypotheses together with their source attribution, ego-versus-partner visibility, uncertainty, conflicts, spatial relations, and planning-relevant context. G-MARK then derives a shared feature representation from these KGs, enabling lightweight task heads to support object reasoning, motion prediction, control selection, and trajectory forecasting. Compared with the state-of-the-art baseline, GMARK improves occlusion reasoning accuracy by 42.2%, reduces control-selection error by 13.1%, and achieves comparable trajectory-planning accuracy with a 25.6x smaller structured communication payload. Our code is available at https://github.com/bhavyagupta98/g-mark. |
| 2026-08-20 | [A knowledge-guided agentic framework for mitigating patient-context ambiguity in health queries](http://arxiv.org/abs/2608.19875v1) | Mahyar Abbasian, Saba A. Farahani et al. | Patients often submit short, underspecified queries to healthcare chatbots that lack the patient-specific information needed to determine an appropriate response. Although these queries may be linguistically clear, they can support multiple plausible answers depending on undisclosed factors such as symptoms, diagnoses, medications, allergies, or dietary restrictions. A language model answering such a query directly may therefore rely on unsupported assumptions about the patient. We introduce a knowledge-guided agentic framework for mitigating patient-context ambiguity before final response generation. The framework operates between the patient and an otherwise unchanged downstream language model. It interprets the initial query, uses a task-specific knowledge graph to construct a set of plausible hypotheses, identifies the missing patient-context variables needed to distinguish among them, and asks targeted follow-up questions. The original query and the acquired context are then combined into a clarified prompt for the downstream model. We evaluated the framework across five language models using two controlled ambiguity-mitigation benchmarks: diagnosis retrieval from 1,034 symptom queries with clinically relevant evidence systematically masked, and dietary-safety classification from 487 queries with decisive health context omitted. The framework was compared with direct answering of the underspecified query and with rephrasing the same query without acquiring new patient information. In diagnosis retrieval, it increased overall exact Top-1 accuracy by at least 57.1 percentage points and selective exact Recall@5 by at least 77.7 percentage points across the five evaluated models compared with direct prompting. In dietary-safety classification, it improved accuracy across all five models and achieved the highest Matthews correlation coefficient for four... |
| 2026-08-20 | [Adaptive Probabilistic Shielding by Learning MDPs for Safe Reinforcement Learning](http://arxiv.org/abs/2608.19836v1) | Astrid Horn Brorholt, Maris F. L. Galesloot et al. | Probabilistic shielding is a technique for safe reinforcement learning (RL). Typically, a static observer -- called the shield -- constrains the learning agent's actions to those for which acting safely remains feasible. Traditionally, the shield is computed from the transition probabilities of the underlying Markov decision process (MDP). Thus, this technique is not applicable when the MDP model is not given a priori, which, unfortunately, is the case in typical RL applications. In this paper, we study the problem of computing a shield in the setting where the transition graph of the MDP is known, but the transition probabilities are unknown. Our approach integrates probabilistic shielding with online model learning: as the RL agent explores the environment, we estimate the transition probabilities. From this estimate, we compute a shield. While the shield may be conservative initially, it adapts as the model estimate becomes more precise. Thus, the shield improves in tandem with the RL agent. This paradigm of adaptive probabilistic shielding raises a number of challenges, such as when to recompute the shield and how to balance between exploration and safety during learning. We empirically evaluate multiple variants of this paradigm across several environments. |
| 2026-08-20 | [Large reasoning models for abnormal situation management in safety-critical industrial processes](http://arxiv.org/abs/2608.19819v1) | Khalid Alhazmi | Automation operates safety-critical processes inside their design envelope and leaves abnormal situations to human operators. Mismanagement of these situations is a leading contributor to process-safety incidents and a hindrance to achieving autonomy. Here we show that a general-purpose large reasoning model, with no task-specific training and only the information available to an operator, manages abnormal situations at run time through a bounded, programmatically verified action interface. Across 39 abnormal situations and operating-point changes on a plant-wide industrial benchmark process, the reasoning model maintained the plant within all hard constraints in all 39, while basic regulatory control failed in 15. It matched the plant's expert-engineered advanced control and diagnosed the root-cause fault in 15 of 15 safety-critical situations. Three independently developed models spanning a thirty-fold cost range exceeded the baseline. In a fully auditable evaluation, these results demonstrate run-time abnormal situation management without a human in the loop. |
| 2026-08-20 | [Understanding as an Explicit and Assessable Component of Frontier AI Safety Decisions](http://arxiv.org/abs/2608.19816v1) | Stephen Barrett, Robin Bloomfield et al. | Decision makers need sufficient understanding to make good decisions about complex AI systems. However, AI deployment decisions are increasingly made under time-pressure, and this combined with the use of AI generated artefact creation, can mean that the existence of safety cases and system cards may no longer demonstrate that sufficient understanding exists. Our provisional methodology for making understanding explicit and assessable requires the production of an explicit description of 4 objects of understanding (decision, decision-frame, safety justification, system-in-context) and a justification for the adequacy of this understanding. In addition, the methodology provides a mechanism for describing and evaluating the adequacy of the decision-maker representation of this understanding. It builds on recent developments in safety cases using the Assurance 2.0 framework to operationalise the philosophical basis of understanding from Elgin and Arendt. To assess the methodology we trialled two different scenarios. One scenario, which we investigated through role-based analysis, concerned the risk of scheming in the deployment of an AI coding agent in a robotics company and the other scenario was for the higher uncertainty, more decision-critical argument of 'If Anyone Builds It, Everyone Dies' (Yudkowsky and Soares). The trial's central finding, for these two scenarios, is that the methodology could be applied and was found to be generative: we found the analyses that justify sufficiency of understanding (internal coherence, tethering, felicitous falsehoods, external coherence) drives the engineering. |
| 2026-08-20 | [Trustworthy Decisions in Reliability Set Estimation under Insufficient Model Information](http://arxiv.org/abs/2608.19815v1) | Holger Dette, Zhengfu Liu et al. | Reliability set estimation identifies input regions where a response probability exceeds a target level, bridging estimation and safety-critical decisions. Practitioners typically start with a working model, an imperfect approximation of the true response surface. Relying on this imperfect model may incur decision risk, potentially certifying unsafe regions as safe. We develop a unified framework that turns such a working model into a trustworthy decision rule. First, a modeling-then-calibration procedure decouples estimation from decision. Since the true set is unobservable, we introduce an asymmetric, observable surrogate loss and use a separate calibration set to select a bias-correcting threshold, reducing decision risk and achieving $O_P(1/n)$ volume convergence. Second, we leverage conformal risk control with the surrogate loss to control false inclusion risk, which is the most safety-critical error, at a pre-specified level regardless of working model quality. Together, these calibration procedures show that a separate calibration set is necessary for risk control. Third, an adaptive design concentrates observations on the reliability set and its boundary, improving model quality where errors most affect decisions while controlling budget elsewhere. Numerical studies show not only more accurate set estimates but also calibrated finite-sample risk control that classical plug-in methods lack. |
| 2026-08-20 | [Stopping and Routing LLM Judge Panels](http://arxiv.org/abs/2608.19802v1) | Bin Zhu, Yi Xie et al. | LLM evaluation pipelines often have many candidate judges: general LLM-as-a-judge prompts, reward models, safety classifiers, confidence variants, and task-specific verifiers. The deployment question is not only which judge is best, but which judges should be called, on which examples, and when panel construction should stop. We formulate judge-panel design as a role-conditioned allocation problem. From a small labeled audit set, declared slices, and judge costs, the method estimates target-relative roles: copies add no conditional information, complements improve the global panel, and specialists help only on slices. These roles induce a policy: drop copies, add complements globally, route specialists conditionally, and stop when validation gain falls below a threshold. Across reasoning, code, safety, preference, reward-model, summarization, and math audits, the method is compared with single judges, flat panels, matched diversity heuristics, full-call stacking, reliability juries, and frugal cascades. The result is a regime map for judge calls: route specialists on deployable slices, stop in saturated verifier regimes, keep broad ensembles when their risk benefit is worth the cost, and ignore conditional copies. The output is a reusable, auditable call plan for the next evaluation batch. |
| 2026-08-20 | [A Fully Automated, Deployment-Aware Testing Pipeline for IoT-Based Automotive Applications](http://arxiv.org/abs/2608.19752v1) | Denesa Zyberaj, Roman Vintonyak et al. | Testing embedded software in modern vehicles is challenging due to system complexity, decentralized architectures, and strict safety and performance constraints. In this work, we present an end-to-end, deployment-aware testing pipeline for IoT-based automotive applications. The pipeline combines requirement-driven test and code generation with large language model (LLM) and vision-language model (VLM) assistance, and human-in-the-loop curation to reduce manual effort and improve consistency. Using Eclipse openDuT, it supports flexible, distributed deployment across geographically separated cyber-physical and IoT infrastructures, optimizing for node availability and cross-organizational coordination. For validation, we conduct a case study using a Child Presence Detection System (CPDS), achieving full functional requirement coverage across all 9 requirements and 100% Gherkin generation accuracy on the controlled requirement set. Distributed test execution across geographically separated ECUs via Eclipse openDuT confirms the pipeline's applicability to OEM--supplier testing workflows. |
| 2026-08-20 | [TempJail: Temporal Jailbreak Attack against Large Vision-Language Models via Subtitle Scheduling](http://arxiv.org/abs/2608.19737v1) | Ling Zhou, Yihao Huang et al. | Large vision-language models (LVLMs) have achieved remarkable progress in video understanding and reasoning. Despite extensive studies on text- and image-based jailbreaks, video jailbreaks against LVLMs remain largely unexplored. Existing video jailbreak methods mainly manipulate textual content embedded in videos, while overlooking how such information is organized over time. Our analysis reveals that jailbreak effectiveness depends not only on the semantics of textual information but also on its temporal presentation, including duration and timing-slot allocation. Motivated by this finding, we use subtitles, which are common in real-world videos and allow semantic content to be presented under precise temporal control without appearing visually intrusive, as a natural attack medium. Based on this insight, we propose TempJail, a black-box video-based jailbreak framework that constructs query-aligned dialogue-style subtitle sequences and optimizes their temporal scheduling to exploit temporal vulnerabilities in LVLMs and elicit responses that satisfy the harmful intent of the source query. Extensive experiments on four representative LVLMs and two datasets demonstrate that TempJail achieves the highest attack success rate across all evaluated model--dataset settings, outperforming the strongest baseline by 53 and 18 percentage points in dataset-averaged ASR on GPT-5 and Gemini 3.5-Flash, respectively. |
| 2026-08-20 | [SafeBranch: Branch-Pair Safety Alignment for Embodied Agents](http://arxiv.org/abs/2608.19729v1) | Hyunse Lee, Jiwoo Jeong et al. | Vision-language-model-based embodied agents can complete instructed tasks but often violate safety constraints in the process, a problem recently framed as interactive safety. Training such agents to act safely is difficult, since safety and task success are distinct objectives, and safety arises only at a small number of safety-critical steps within a trajectory. Standard supervision is insufficient: imitating safe trajectories teaches behavior without explaining why it is safe, and contrasting arbitrary safe and unsafe trajectories mixes the safety signal with unrelated differences. We propose SafeBranch, a framework that aligns an embodied actor on safety through branch pairs constructed from the actor's own unsafe rollouts via environment rollback. SafeBranch rolls each unsafe rollout back to the safety-critical step that caused the violation, queries the actor for a safe alternative, and pairs the original action with the alternative so that the two branches differ only at that step. The trained actor acts safely at deployment with no critic in the loop. On IS-Bench, SafetyALFRED, and out-of-distribution variants with unseen tasks and objects, it handles safety reliably without sacrificing task success, achieving roughly ten times more safe successes than the untrained baseline on the unseen-object variant. |
| 2026-08-20 | [Variational Goal-Oriented Optimal Experimental Design for Mixed-Distribution Quantities of Interest: Application to Ship Roll Safety](http://arxiv.org/abs/2608.19631v1) | Chen Cheng, Xun Huan et al. | Goal-oriented optimal experimental design (GO-OED) selects experiments according to the expected information gain (EIG) about a quantity of interest (QoI) rather than the full parameter vector. This work develops a variational GO-OED formulation for mixed discrete-continuous QoI laws arising in probabilistic mechanics when thresholding or event-based transformations map a positive-probability set of uncertain inputs to a common value while other inputs produce continuously varying responses. The motivating application is ship roll safety assessment in random waves, where the QoI is the temporal exceedance probability above a prescribed roll-angle threshold. This quantity is zero when no exceedance occurs and varies continuously over positive values otherwise. A purely continuous variational approximation does not dominate a posterior QoI law containing an atom, yielding an infinite Kullback-Leibler divergence and a trivial Barber-Agakov lower bound of $-\infty$. Scoring atom samples using continuous density values instead changes the objective and does not produce a valid lower-bound estimator. We introduce a mixed variational approximation that models the conditional atom probability and continuous component separately, with a normalizing flow used for the latter. An analytical example recovers the correct EIG landscape, while the ship roll application provides stable EIG lower-bound estimates and identifies informative wave conditions for temporal-exceedance-probability inference. |
| 2026-08-20 | [Mix&Fix-Net: A Dual-Stage Trajectory Prediction Model for AIS and Vision-Derived Vessel Data](http://arxiv.org/abs/2608.19580v1) | Md Mahmuddun Nabi Murad, Bora San Turgut et al. | Vessel trajectory prediction is critical for maritime safety and accident prevention. While most existing trajectory prediction models rely on Automatic Identification System (AIS) data due to its precision and availability, small vessels mostly operate without AIS, resulting in a significant monitoring gap. To address this, we propose Mix&Fix-Net, a dual-stage mixer-based trajectory prediction model designed to handle vessel trajectory time-series data derived from both AIS and (non-AIS) vision data. Our architecture integrates a Primary Trajectory Predictor with a Residual Trajectory Adjuster, enabling more refined trajectory prediction. Additionally, we introduce a new video-based dataset derived from webcam streams, from which vessel trajectories are extracted to represent non-AIS data. Extensive evaluations on both AIS and non-AIS datasets across six metrics (mean squared error, mean absolute error, symmetric mean absolute percentage error, final displacement error, Frechet distance, and average Euclidean distance) demonstrate that Mix&Fix-Net consistently outperforms existing baselines across most metrics and datasets. |
| 2026-08-20 | [Enforcing LLM Safety through DMD-based Classification of Prompt-Response Embedding Dynamics](http://arxiv.org/abs/2608.19579v1) | Mohamed Akrout, Olivera Kotevska et al. | Large Language Models (LLMs) are increasingly deployed in high-stakes applications, yet their tendency to generate toxic, harmful, or policy-violating content poses significant risks. Detecting these unsafe outputs efficiently in a black-box manner remains an open challenge. In this paper, we extend a recently proposed dynamical systems framework designed for hallucination detection to LLM safety classification. By projecting both prompts and responses into high-dimensional embedding spaces and fitting separate Koopman-based predictive models for safe and unsafe regimes, we classify new outputs using a new differential residual score that compares prediction errors of the safe and unsafe regimes. A key contribution is the incorporation of the prompt and response embedding dynamics, yielding fitted Koopman operators that capture crucial interaction patterns. We evaluate our black-box method across three safety benchmarks using three embedding models. Our results show that incorporating prompt embeddings yields consistent improvements, particularly for interaction-dependent violations when paired with causal decoders (e.g., in Llama-3), while response-only violations benefit more from dense semantic embedding representations. These findings opens the door for using dynamical systems to analyze AI systems rather than the dominant paradigm of using AI to model dynamical systems. |

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



