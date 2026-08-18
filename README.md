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
| 2026-08-17 | [Q-based Variational Inverse Reinforcement Learning](http://arxiv.org/abs/2608.16888v1) | Ondrej Bajgar, Peter Tisnikar et al. | The development of safe and beneficial AI requires that systems can learn and act in accordance with human preferences. However, explicitly specifying these preferences by hand is often infeasible. Inverse reinforcement learning (IRL) addresses this challenge by inferring preferences, represented as reward functions, from expert behaviour. We introduce Q-based Variational IRL (QVIRL), a novel Bayesian IRL method that recovers a posterior distribution over rewards from expert demonstrations via primarily learning a variational distribution over optimal Q-values. Unlike previous approaches, QVIRL combines scalability with uncertainty quantification, important for safety-critical applications as well as active learning. We demonstrate QVIRL's strong performance in apprenticeship learning across various tasks, including gridworlds, Lunar Lander, the Highway Environment, and two ATARI games both with static expert data and with active learning. It is the first method for Bayesian IRL that demonstrates training from raw pixel observations. |
| 2026-08-17 | [Security of Foundation-Model-Powered Embodied Agents: Attack Surfaces, Attacks, Defenses, and Evaluation](http://arxiv.org/abs/2608.16843v1) | Jiawei Liu, Jiacheng Guo et al. | Foundation models are increasingly used for perception, reasoning, planning, and action generation in embodied agents, creating security risks that can propagate from digital inputs to physical behavior. Existing surveys often organize threats by mechanisms such as jailbreaks, prompt injection, backdoors, poisoning, or adversarial examples, but these categories do not consistently identify where an adversary first enters the embodied control loop. We present a trust-boundary-centric survey of foundation-model-powered embodied-agent security. Using a first-compromised-trust-boundary principle, we separate attack surface from attack mechanism and organize the system into five layers and twelve attack surfaces spanning the model supply chain, user instructions, context and memory, physical semantic environments, multimodal perception, world state, internal reasoning, task planning, action interfaces, middleware, multi-agent communication, and execution control. Based on 58 attack records and 61 defense records collected through August 15, 2026, we analyze representative attacks, cross-layer propagation, defense placement, and evaluation practices. Our quantitative analysis shows that attack research is concentrated on multimodal perception and action interfaces, while defenses are especially concentrated on action-level and runtime protection. Context and long-term memory, middleware and networking, world-state integrity, and multi-agent trust remain comparatively underexplored. We conclude with open challenges in state provenance, compositional defenses, long-horizon attack propagation, physical realizability, Byzantine multi-robot behavior, and unified closed-loop evaluation. |
| 2026-08-17 | [HAF: Adapting Generalist VLAs to Humanoid Whole-Body Loco-manipulation via Hierarchical Action Flow and Spectral Latent RL](http://arxiv.org/abs/2608.16837v1) | Langzhe Gu, Chengkai Hou et al. | Humanoid robots hold great promise as general-purpose agents in human-centered environments, yet generalist vision-language-action (VLA) foundation models are not readily applicable to humanoid whole-body loco-manipulation. The high dimensionality and interdependence of humanoid motions make it challenging for conventional single-stage VLA architectures to coordinate locomotion, waist posture, and dual-arm manipulation effectively. Moreover, policies trained through offline behavior cloning can remain suboptimal during real-world deployment. Although online reinforcement learning can refine policies through real-world interaction, directly tuning large VLA backbones demands excessive computation and may introduce safety risks during real-robot exploration. To address these bottlenecks, we introduce HAF (Humanoid Adaptation Framework), a two-part framework consisting of HAF-VLA and HAF-Steer that transfers off-the-shelf generalist VLA foundation models to humanoid whole-body loco-manipulation. HAF-VLA is a hierarchical action-flow generator built on a pretrained flow-matching VLA. It splits full-body action denoising into three sequential stages with stage embeddings and cross-stage KV caches that retain kinematic dependencies, avoiding incoherent whole-body actions from one-shot generation. On top of the frozen HAF-VLA, HAF-Steer is a latent offline-to-online RL pipeline that leverages flow-matching invertibility and DCT-based dimensionality reduction to restrict RL optimization to a compact noise subspace and train a regularized SAC policy. This avoids updating the large VLA backbone and enables efficient real-world policy refinement. Evaluated on seven real-world humanoid loco-manipulation tasks, HAF surpasses vanilla single-stage VLA baselines and improves whole-body coordination and task performance. Project website: https://grange007.github.io/HAF . |
| 2026-08-17 | [Model Hypnosis: Strong control of AI via additive subliminal effects](http://arxiv.org/abs/2608.16834v1) | Enric Boix-Adsera, Benedict Tessler | We demonstrate that AI models are broadly susceptible to a phenomenon we call model hypnosis, in which individually weak and seemingly irrelevant cues in the prompt can be systematically combined to strongly control model behavior. Model hypnosis occurs across model families and scales, including in frontier reasoning models, and hypnotic prompts can transfer between models. Because the model is controlled by inconspicuous textual choices, such as paraphrases and typos, model hypnosis presents new challenges and avenues for AI safety, and is a major hurdle for AI interpretability. |
| 2026-08-17 | [HAPS through the Lens of Satellites and UAVs: A Function-Level Perspective on the Emerging High Altitude Economy](http://arxiv.org/abs/2608.16828v1) | Mukhtiar Ahmad, Mohamed-Slim Alouini | High-Altitude Platform Stations (HAPS) operate in the lower stratosphere at 17-27 km, between satellites and Unmanned Aerial Vehicles (UAVs). For this third tier the architectural case has long outpaced the flight evidence, but a wave of 2020-2026 stratospheric flights now permits a direct comparison. We evaluate HAPS function by function across sensing, navigation, and communication, taking operational satellite and UAV implementations as the reference. We define a strict evidence rule, counting a function as flight-validated only on operationally relevant stratospheric data return at or above 18 km, and apply it to nineteen functions. The resulting count is lower than the literature implies: five functions have credibly crossed over (optical Earth observation, hyperspectral imaging, methane imaging, RF/SIGINT, and broadband relay), yet these rest on only four flight programs, with at most one carrying peer-reviewed flight evidence. One function is ground-demonstrated, two are partially demonstrated, three are conceptual, and eight remain unflown. Four engineering domains (size, weight, and power; station-keeping; aperture; and viewing geometry), bounded by an operational envelope of platform stability and payload operability, explain the pattern. The governing advantage is persistence at close range, not altitude. Eight use cases, supported by same-sensor forward simulations, translate the pattern into missions, led by resilient public-safety mission-critical services (MCX). On this evidence, HAPS fits as a persistent regional tier in a multi-tier non-terrestrial network and as the seed of an emerging High Altitude Economy. Carrier-grade service, station-keeping precision, and regulation remain the principal open problems, and we pose the persistent-tier reading as a testable hypothesis with dated 2030 markers. |
| 2026-08-17 | [Calibration-Free Vehicle Speed Estimation: A Monocular Keypoint-Template Approach](http://arxiv.org/abs/2608.16785v1) | Gaofeng Su, Keya Li et al. | This paper proposes a calibration-free framework for reliably and effectively estimating vehicle speeds from monocular videos, without relying on roadway features, camera calibration, or roadway-feature-based reference objects. The proposed framework estimates vehicle speeds using a 36-keypoint vehicle template and a homography matrix updated at each frame. A YOLO-based keypoint detection module is trained on diverse datasets, and two estimation strategies are compared: keypoint-only tracking and warped optical flow with dense spatial aggregation. Speed is estimated by projecting displacements into metric space using the homography, with validation conducted on over 400 video clips from roadside and overhead datasets, covering speeds from 30 to 100 mph. The method achieves reliable speed estimation on the VS13 and BrnoCompSpeed datasets, with the warped optical flow method delivering MAEs of 15.0% and 9.7%, respectively, and 77.9% and 93.1% of estimates falling within +/-20% error. After applying a 10% trim to remove edge-of-frame outliers, performance improves to MAEs of 11.7% and 7.6%, with within-+/-20% accuracy increasing to 85.3% and 95.4%. This work addresses key limitations of existing vision-based approaches and enables low-cost and efficient speed enforcement using portable devices such as dashcams and smartphones, thereby supporting citizen-based enforcement programs for traffic safety. |
| 2026-08-17 | [Beyond $L_2$: Generalizing Abductive Latent Explanations to Diverse Prototype-Based Architectures](http://arxiv.org/abs/2608.16773v1) | Jules Soria, Alban Grastien et al. | Prototype-based neural networks are hailed as interpretable-by-design architectures. Recently, Abductive Latent Explanations (ALE) were introduced to provide formal, mathematically guaranteed explanations that leverage the intrinsic structure of these networks to ensure both predictive safety and human readability. ALEs rely on computing tight bounds on latent space distances to produce formal explanations. However, existing ALE formulations are rigidly confined to Euclidean latent spaces. This leaves a critical gap: modern state-of-the-art architectures increasingly rely on non-Euclidean representations - such as spherical metrics, Gaussian densities, and dimensional projections - rendering current formal explanation methods incompatible. In this work, we generalize the ALE framework to support non-Euclidean prototype architectures. For each geometric variant, we systematically derive how to either map the architecture to existing bounds or construct novel, architecture-specific bounding algorithms. We validate our theoretical constructions by computing subset-minimal formal explanations on fully trained image classifiers. By unifying these diverse models under a single formal framework, we enable the first rigorous, cross-architecture comparison of their interpretability. |
| 2026-08-17 | [Unsupervised Anomaly Detection for Image Dataset Quality Assurance in Multi-Center Breast MRI](http://arxiv.org/abs/2608.16725v1) | Chiara Tappermann, Steffen Renisch et al. | Corrupted, inconsistent, or anomalous data silently threatens the safety and reliability of medical AI. Despite growing regulatory recognition of dataset quality assurance (QA) for high-risk medical AI, scalable automated detection remains underdeveloped. We employ unsupervised anomaly detection (AD) and out-of-distribution (OOD) detection as an automated dataset QA mechanism for multi-center dynamic contrast-enhanced breast MRI.   We build a controlled AD benchmark of 17 realistic QA-relevant anomaly types from six public datasets (protocol violations, processing errors, incorrect anatomical regions) and propose a taxonomy of radiological image anomalies based on human visual perception, enabling fine-grained analysis of AD failure modes. The benchmark includes near-, medium-far-, far-OOD samples, as well as in-distribution and external normal data. Four methods are evaluated: a projection-based method extended with a domain-specific feature extractor and a novel positional encoding, a reconstruction-based approach extended to full 3D volumes with an augmented training objective, and two unmodified hybrid OOD detection methods.   Medium-far- and far-OOD samples are detected reliably, whereas near-OOD samples and external normal data from unseen institutions expose method-specific differences. The 3D reconstruction-based approach best balances detection performance (AUROC: 0.936) and generalization to unseen institutions. The projection-based method with positional encoding achieves the highest overall detection performance (AUROC: 0.954). Both hybrid methods exhibit critical failure modes, confirming that methods validated for one modality or anatomy may not generalize without domain-specific adaptation. Implants and mastectomies remain an open challenge for all methods. Our results establish a foundation and practical guidance on scalable unsupervised QA in medical AI pipelines. |
| 2026-08-17 | [The Ethical Decision Head: Operationalizing Normative Ethics in Autonomous Vehicles via Reinforcement Learning from Human Feedback](http://arxiv.org/abs/2608.16710v1) | Thomas Mbrice, Ammar Ali et al. | As autonomous vehicles (AVs) approach Level 4 and Level 5 operational capability [SAE International, 2018], their on- board decision systems must handle not only safety-critical locomotion but also their subsequent moral weight. This paper details the Ethical Decision Head (EDH), a deep re- inforcement learning (RL) framework that encodes ethical reasoning as a differentiable reward signal, enabling a pol- icy gradient agent to learn morally-aligned driving behavior in scenarios whose state representation is aligned with the CARLA simulation environment [Dosovitskiy et al., 2017]. Two normative frameworks are instantiated and evaluated: a Utilitarian framework minimizing total casualties and a Kan- tian framework enforcing course maintenance as a categori- cal imperative. The EDH is trained via Proximal Policy Op- timization (PPO) [Schulman et al., 2017] against a Bradley- Terry reward model [Bradley and Terry, 1952] learned from pairwise human preference annotations over 200 collision- imminent scenarios. Results reveal an asymmetry in the learnability of normative ethical frameworks under human su- pervision. The Kantian condition, which reduces to a con- stant prediction task under the codebook, serves as a pipeline control: it confirms training stability and rules out infrastruc- ture failure as an explanation for the utilitarian result. The Utilitarian agent learned something more unsettling: human raters rewarded self-sacrifice over casualty minimization, and the model learned that preference faithfully. This divergence between what humans prescribe in theory and what they re- ward in practice suggests that RLHF does not learn ethics as philosophers define it, but as humans live it. |
| 2026-08-17 | [Toward Better Assessment of LLMs' Performance in Clinical Error Detection](http://arxiv.org/abs/2608.16643v1) | Yifan Zhang, Rahmatollah Beheshti | Automated detection of errors in clinical documentation is a promising application of large language models (LLMs), yet decisions to deploy such models rest on benchmarks that evaluate each clinical note in isolation. Error-detection benchmarks are typically constructed by injecting errors into notes, such that each erroneous note has a natural counterpart. Aggregate discriminative metrics (e.g., balanced accuracy or F1) do not exploit this structure. We show that this omission is consequential. In particular, evaluating 15 diverse LLMs on 4 standardized clinical error-detection test sets across 3 languages, we find that 13 of 15 models fall below the level of random pairwise discrimination, even while achieving F1 scores that standard practice would read as moderate. We also observe that the underlying bias patterns differ across languages: the same model can default to "no error" on one language and over-flag errors on another. To diagnose where discrimination breaks down, we further introduce a procedure to score the evidence models cite in their outputs. We find that while models consistently locate error-relevant content, they fail to produce the corresponding correct verdict on the clean counterpart. Finally, we show that F1 and pairwise accuracy are driven in opposite directions by the same underlying bias, so that ranking models by F1 may systematically promote the weakest discriminators. For safety-critical clinical NLP applications, we advocate for supplementing aggregate metrics with paired evaluations in benchmark reporting. Code and analysis scripts are available at https://github.com/healthylaife/paired-clinical-eval. |
| 2026-08-17 | [Palmyra x6 Technical Report: An Agentic, Tool-Use Model Post-Trained via Anchored Supervised Fine-Tuning](http://arxiv.org/abs/2608.16620v1) | Peng Du, Kiran Kamble et al. | Palmyra x6 is a large language model optimized for use with enterprise-oriented agentic tasks. The model was built by post-training a Mixture-of-Experts base model with Anchored Supervised Fine-Tuning on a compact corpus of verified, synthetic tool-use trajectories, optimized with a Muon + Adam hybrid. The recipe is deliberately conservative and deliberately controlled: 626 trajectories, a single epoch, a low learning rate, and a KL anchor to the frozen base. The model shows substantial gains over the previous default model for Writer Agent, and compares favorably with several recent models on public benchmarks, scoring the highest on BFCL Core at $0.785$ and posts the highest six-benchmark mean of the cohort. Furthermore, the model has shown itself to be competitive or leading relative to comparators in our bias and safety evaluations. |
| 2026-08-17 | [BabelSteering: Multilingual Safety Alignment via English Steering Vectors](http://arxiv.org/abs/2608.16577v1) | Emma V. Stein, Dominik Meier et al. | Large language models (LLMs) are deployed globally in high-stakes settings, yet most safety research and alignment efforts remain concentrated on English. Thus, users interacting with LLMs in other languages may encounter weaker safeguards despite relying on the same systems for similarly sensitive tasks. In this work, we investigate whether safety signals learned from a high-resource language, like English, can improve multilingual safety. We propose BabelSteering, an activation steering method that acts as a lightweight inference- time intervention, using refusal directions derived from English safety supervision to generalize across languages. Our evaluation includes eight languages and jointly measures refusal of harmful requests, over-refusal, and general task utility. The results show that BabelSteering increases the refusal of harmful requests across languages, with only a marginal to no reduction in task utility but with some increase in refusal of pseudo-harmful prompts. For example, for Gemma 7B, we see an average increase in the refusal of harmful prompts across languages of 11 percentage points (pp), with individual languages like Bengali seeing an increase of 17 pp, with no loss of utility on Global MMLU, while pseudo-harmful refusals increase by 13 pp on average. We also introduce a multilingual translation-and-evaluation pipeline to facilitate future work on cross-lingual safety interventions. Overall, our findings suggest that activation steering may provide a practical, low- cost mechanism for extending English-derived safety signals to other languages. Warning: this paper contains examples with unsafe content |
| 2026-08-17 | [CUBICS: Situation-aware performance estimation for safety-relevant ML components](http://arxiv.org/abs/2608.16564v1) | Benjamin Herd, Jessica Kelly et al. | Machine learning (ML) is a key technology driving innovation today, but ensuring ML safety remains a major challenge for safety-related applications. A promising idea is to build proven-in-use arguments from field data, e.g. by running ML components (MLCs) in shadow mode or within safety envelopes so that their outputs can be monitored as 'safe probes' without affecting safety. These probes can then be used to build a statistical argument about field performance in a Bayesian way. However, many Bayesian field-data approaches in safety engineering model failures as a simple Bernoulli (or binomial) process with a single global failure probability and i.i.d. trials, which is rarely adequate for MLCs whose performance depends strongly on context. Statistical evidence is also about coverage of relevant situations, including edge cases, and building a single integrated statistical model for the entire system is usually not feasible. To address these challenges, this paper introduces CUBICS, a context-modular framework for per-component, situation-aware performance estimation of safety-relevant ML components. CUBICS partitions the operational design domain into situations and, for each safety-relevant component, defines a set of situation-specific assumptions and probabilistic guarantees that are represented and updated in a Bayesian manner using Subjective Logic (SL). By combining these guarantees with beliefs about how often each situation occurs, CUBICS derives an overall risk estimate for each component without requiring a monolithic system-level statistical model, and thus provides a building block for modular, field-data based safety assurance. |
| 2026-08-17 | [Vantage: Availability-Graded Broadcast for Signature-Free BFT](http://arxiv.org/abs/2608.16504v1) | Nikita Polyanskii | Digital signatures make blocks and votes transferable evidence: one party can prove to another what a third party said. Authenticated channels convince only the direct receiver, so existing high-throughput signature-free protocols complete an availability vote or a broadcast instance for each data block before any proposer may order it. We present Vantage, a partially synchronous Byzantine fault-tolerant protocol for $n \ge 3f+1$ parties that uses only authenticated channels and collision-resistant hashing. Parties publish blocks on hash-linked author lanes; each view's proposer pairs a quorum-available core manifest of lane frontiers with an optimistic tip manifest of freshly received blocks. A new primitive, Availability-Graded Broadcast (AGB), makes the core irrevocable on a quorum of first-hand responses while grading, rather than blocking on, the tip. Unresolved tips are sealed later through a signature-free control log, by resolutions each correct party checks against its own recorded responses; a crash-only silent view is skipped by a quorum of skip votes without the log. AGB makes a published block proposal-eligible one message delay after publication, matching signed optimistic-tip designs. When all parties are correct and message delays are $δ$, a data-only proposal seals within $2δ$ of its send on all $n$ first-hand acknowledgments, so a block is sequenced within $4δ$ of publication, and within $3δ$ when publication aligns with the next proposal. We prove safety under asynchrony; liveness holds after the Global Stabilization Time. On an emulated ten-region WAN with 100 parties, Vantage has the lowest median latency among the nearest signature-based and signature-free protocols at every accepted load and sequences 239k 512-byte transactions per second with median latency below 500 ms. |
| 2026-08-17 | [Graph Machine Learning: An Opportunity for Power Systems](http://arxiv.org/abs/2608.16494v1) | Martin Sadric, Sebastian Pütz et al. | Modern power systems face growing operational complexity driven by the integration of renewable energy sources, decentralization, and the need for real-time decision-making across a wide range of timescales. Addressing these challenges traditionally relies on model-based methods that, while accurate, can be too slow for operational demands. Machine learning (ML) has therefore emerged as a faster, data-driven alternative. As grid topology plays a central role in power system operation, graph machine learning (GML) methods offer a natural framework for incorporating topological dependencies as an inductive bias. We survey nearly 800 papers at the intersection of GML and power systems, covering forecasting, state estimation, optimization, control, fault diagnosis, and cybersecurity. Power systems constitute an unusually rich benchmark setting for GML, as they combine hard physical constraints, multi-scale dynamics, safety-critical requirements, and scarce labeled data within a single, well-defined domain. Conversely, power systems can benefit from utilizing GML to complement classical solvers, as GML provide scalable, topology-aware approximations with promising generalization and computational efficiency. We identify open challenges, including limited real-world deployment and the need for interpretable models in safety-critical settings. Despite the rapidly growing number of publications, standardized benchmarks and open datasets remain scarce, leaving many results difficult to reproduce and undermining the long-term scientific credibility of the field. We further derive a structured requirements catalog for ML-ready power grid benchmarks, intended to guide future dataset development and improve reproducibility across studies. We call on the community to prioritize dedicated benchmark studies and the release of open datasets and models. |
| 2026-08-17 | [Exposing the Long-tail in Embodied Urban Navigation via Scalable Learning from In-the-Wild Videos](http://arxiv.org/abs/2608.16476v1) | Bingyi Xia, Han Bao et al. | Learning embodied urban navigation policies from real-world data is constrained by the cost of task-specific data collection and the limited coverage of rare yet safety-critical scenarios. To address these challenges, we present a scalable framework for learning point-goal urban navigation from web-scale in-the-wild egocentric videos while systematically exposing its long tail. The framework automatically annotates uncurated web videos with metric trajectories and structured navigation semantics, which are then used to train a vision-language-action policy for interpretable navigation planning. We characterize the long tail based on model performance and the distribution of perception-motion patterns, and employ reflection-based analysis to diagnose recurring failure modes. Experiments on web-video data and real-world urban navigation tasks demonstrate effective knowledge transfer from unconstrained videos and reveal coherent long-tail structures beyond aggregate navigation performance. |
| 2026-08-17 | [Visualizing Uncertainty-to-Action Composition for Human Oversight](http://arxiv.org/abs/2608.16428v1) | Chisom Anyabolu, Akshat Dubey et al. | Artificial intelligence systems often disclose uncertainty, yet they rarely make clear what response that uncertainty should trigger. Most uncertainty visualizations encode uncertainty in model outputs, leaving users to discern the most appropriate course of action. A second region of the design space--uncertainty in the decision process itself, including how multiple uncertainty conditions compose into an oversight response-- remains comparatively underexplored. We address this gap with two coupled contributions. First, we introduce an uncertainty-to-action binding framework that composes multiple uncertainty conditions into a single oversight response under a precedence policy with a contextual safety modifier. That response concerns whether and how an AI-supported decision may proceed, not the substantive domain decision itself. Second, we present ActionCue, a process-transparency visualization that renders that composition explicit. We demonstrate the approach through a three-way comparison with confidence-only and data-level uncertainty displays, using worked cases from healthcare, credit assessment, and disaster forecasting. Together, the framework specifies how uncertainty conditions are resolved into an oversight response, and the visualization makes that resolution inspectable rather than implicit. |
| 2026-08-17 | [Scalable Gaussian Process Regression via Deterministic Trigonometric Features: Uniform Bounds for Safe Model Predictive Control](http://arxiv.org/abs/2608.16415v1) | Julius Jagdt, Johanna Menn et al. | Learning-based Model Predictive Control (MPC) using Gaussian processes (GPs) is an effective approach for safe control in the presence of model mismatch. High-probability safety guarantees typically require uncertainty bounds that hold uniformly over the entire state--input domain, but existing bounds are available only for full GP regression. Since exact GP inference scales poorly with the number of data points, its deployment is impractical in large-data regimes. We close this gap by developing a scalable GP framework that admits the derivation of uniform uncertainty bounds. We formalize a deterministic trigonometric feature Gaussian process (DTF-GP), a finite-dimensional kernel approximation based on discretized trigonometric features that reduces GP regression to Bayesian linear regression in feature space. We derive a high-probability uniform uncertainty bound for the proposed DTF-GP and provide its closed-form solution for the squared-exponential kernel case. Finally, we integrate the DTF-GP into a learning-based MPC scheme and demonstrate that it provides high-probability safety guarantees and exploration performance comparable to a full GP while improving computational efficiency in large-data regimes. |
| 2026-08-17 | [Think Inside the Chunk: RegulaRAG for Regulation-Compliant Scenario Generation using LLMs: A Case Study of UN Regulation No. 152](http://arxiv.org/abs/2608.16394v1) | Vahid Zolfaghari, Nenad Petrovic et al. | Generating regulation-compliant test scenarios is essential for validating safety-critical automotive systems, yet Large Language Models (LLMs) struggle to ground outputs in long, hierarchical standards. We present RegulaRAG, a Retrieval-Augmented Generation (RAG) pipeline that couples SmartChunking, reference-aware enrichment of paragraphs and tables via graph traversal, with Smart Retrieve & Rerank over these enriched units. To test our system, we evaluate on a manually curated dataset covering all scenarios in UN Regulation No. 152 (AEBS). Our study comprises: (i) a three-step progressive search that identifies near-optimal retrieval parameters without exhaustive grid search; (ii) head-to-head comparisons against five baseline RAG systems; and (iii) a robustness stress test that scales the source corpus with distractor content. Outputs are evaluated using a customized penalized scoring metric. Across all experiments, RegulaRAG achieves the highest average Meta-Score (82.99), outperforming the next-best system by 43% (NoRAG: 57.94), while operating at 14k-25k tokens per query versus up to 500k for graphcentric baselines. It maintains strong performance, remaining stable even as the number of regulatory sources grows, whereas competing RAG systems degrade sharply in both quality and robustness. |
| 2026-08-17 | [AeroCopilotBench: A Two-Tier Benchmark for Evaluating LLM Agents as Aviation Copilots in an Interactive Virtual Cockpit Environment](http://arxiv.org/abs/2608.16349v1) | Yuchen Yuan, Zhenghuang Wu et al. | Large language model (LLM) agents may assist flight crews with complex decisions and task execution, but existing aviation evaluations centered on static knowledge do not support systematic testing of procedural execution and safety compliance in interactive environments. This paper presents the AeroCopilot Operational Environment (ACOE), a reproducible interactive virtual-cockpit test environment, and AeroCopilotBench, a two-tier aviation agent evaluation benchmark. Tier-1 evaluates aviation knowledge using 1,200 multiple-choice questions, while Tier-2 comprises 73 emergency and abnormal tasks derived from the manufacturers' Pilot's Operating Handbooks (POHs) and instantiated in ACOE. ACOE converts natural-language procedures into executable state transitions, final-state goal conditions, and hard safety constraints, enabling models to interpret cockpit state, diagnose faults, and operate aircraft systems through standardized tool interfaces. We establish a safety-gated evaluation framework in which a trajectory succeeds only when all task goals are achieved without violating any hard safety constraint, while safe goal progress and trajectory safety are measured separately. Across 12 models, the highest Tier-2 success rate is 72.6%, while static knowledge performance does not consistently translate into procedural execution. Analysis of 451 failed episodes from 3 representative models identifies recurring failures in procedural completeness, use of state feedback, and long-horizon execution management. These findings motivate state-aware agent orchestration, joint assessment of task completion and trajectory safety, and repeated regression testing. ACOE and AeroCopilotBench provide a reproducible foundation for testing knowledge application, interactive execution, and operational safety in aviation agents. |

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



