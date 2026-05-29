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
| 2026-05-28 | [GPIC: A Giant Permissive Image Corpus for Visual Generation](http://arxiv.org/abs/2605.30341v1) | Keshigeyan Chandrasegaran, Kyle Sargent et al. | Studying scalable methods for visual generative modeling requires large, accessible, and stable datasets. We introduce GPIC, a Giant Permissive Image Corpus of approximately 28 trillion pixels. GPIC comprises diverse internet images captioned by a state-of-the-art vision-language model, including 100M training, 200K validation, and 1M test examples. Moreover, all GPIC images are permissively licensed for both research and commercial use. GPIC is safety-filtered, deduplicated, and centrally hosted on Hugging Face. We provide a benchmarking protocol for generative modeling on GPIC. Finally, we provide a reference baseline for pixel-space flow matching on GPIC. Our dataset, benchmark, and models are available at https://huggingface.co/datasets/stanford-vision-lab/gpic. Evaluation toolkit and code are available at https://gpic.stanford.edu |
| 2026-05-28 | [LLUMI: Improving LLM Writing Assistance for Mental Health Support with Online Community Feedback](http://arxiv.org/abs/2605.30273v1) | Jiwon Kim, Maya Ajit et al. | Large language models (LLMs) show promise in generating supportive responses for mental health queries, but improving their usefulness, empathy, and safety often requires substantial compute, expert input, and labeled data. At the same time, deploying proprietary, cloud-based models for mental health-related interactions raises important privacy and data-governance concerns, given the sensitivities. To address this challenge, we introduce LLUMI setup that can be hosted in-house within protected environments. LLUMI consists of two complementary components: a generation model (GM), which drafts supportive responses to mental health queries, and an improvement model (IM), which revises an initial human-crafted response. We leverage feedback signals from Reddit mental health communities, using community endorsement patterns such as upvotes and downvotes to construct chosen-rejected response pairs for Supervised Fine Tuning (SFT) and Direct Preference Optimization (DPO). We further align LLUMI using human evaluation across five dimensions: readability, empathy, connection, actionability, and safety. Our results show that, despite relying on smaller open-source models rather than proprietary cloud-based GPT models, LLUMI achieves comparable performance across linguistic analyses and human evaluations. These findings suggest that open-source models, when trained with community-derived preference signals, can support high-quality mental health support assistance while offering a more privacy-preserving alternative for sensitive support contexts. |
| 2026-05-28 | [Automating Low-Risk Code Review at Meta: RADAR, Risk Calibration, and Review Efficiency](http://arxiv.org/abs/2605.30208v1) | Chris Adams, Arjun Singh Banga et al. | AI-assisted coding tools have altered software production. At Meta, significant lines of code per human-landed diff grew by 105.9% year over year and per-developer diff volume rose 51%, with agentic AI responsible for over 80% of that growth. Meanwhile, the share of diffs receiving timely review has declined, exposing a widening gap between code supply and reviewer bandwidth. We ask three questions that progress from feasibility through calibration to impact: (1) can risk-stratified automation operate at scale across diverse organizations, (2) how does tuning the risk threshold affect the trade-off between automation yield and safety, and (3) to what extent does automated review reduce end-to-end latency for AI-generated changes? We deployed RADAR (Risk Aware Diff Auto Review), a multi-stage funnel that classifies each diff by authorship and source type, applies eligibility gates, static heuristics, a machine-learned Diff Risk Score, LLM-based Automated Code Review, and deterministic validation before landing qualifying changes. We evaluate RADAR through telemetry covering 535K+ RADAR-reviewed diffs, observational before-after comparisons for policy changes, and difference-in-differences analysis of efficiency outcomes. RADAR has reviewed 535K+ diffs and landed 331K+. Relaxing the Diff Risk Score threshold from the 25th to the 50th percentile increased the approve rate to 60.31%. The revert rate for RADAR-reviewed diffs is 1/3 that of non-RADAR diffs, and the Production Incident rate is 1/50 that of non-RADAR diffs. RADAR reduces median time to close by over 330% and median diff review wall time by 35%. Risk-aware layered automation can materially reduce review bottlenecks created by AI-driven code growth without compromising production safety. |
| 2026-05-28 | [Neural Network Verification using Partial Multi-Neuron Relaxation](http://arxiv.org/abs/2605.30155v1) | Ido Shmuel, Guy Katz | The increasing integration of deep neural networks in critical systems has spawned a theoretical and practical interest in formally guaranteeing safety properties about their behavior. To achieve this, contemporary verification algorithms rely on computing linear relaxations for a network's non-linear activation functions. Existing approaches for linear relaxations typically fall into one of two categories: single-neuron relaxation, in which each activation neuron is bounded in terms of its sources; and multi-neuron relaxation, in which linear bounds involving multiple activation neurons and their sources are calculated. However, existing methods might fail to balance tightness and scalability, as single-neuron bounds might not derive sufficiently tight bounds necessary for verification to complete, whereas generating multi-neuron relaxation for all activation neurons is computationally expensive. In this paper, we present a middle-ground approach featuring partial multi-neuron relaxation, in which we generate multi-neuron bounds for only a small, heuristically selected subset of neurons. To achieve this, we build upon existing branching heuristics for selecting neurons and for optimizing bounding hyper-planes for multi-neuron bounds. We integrated our proposed method within the Marabou verifier, and obtained favorable results in comparison to existing bound tightening methods. Our experiments showcase the potential of our technique for neural network verification. |
| 2026-05-28 | [When Cloud Agents Meet Device Agents: Lessons from Hybrid Multi-Agent Systems](http://arxiv.org/abs/2605.30102v1) | Corrado Rainone, Davide Belli et al. | The design space of agentic AI inference spans two extremes: frontier large language models (LLMs), typically hosted in the cloud and offering strong performance across a wide range of tasks at substantially high cost, and more cost-efficient small language models (SLMs), which are amenable to on-device inference. Hybrid multi-agent systems (MASs) combining on-device and cloud models offer a promising middle ground, but they also introduce a complex and poorly understood design space in which task accuracy, monetary cost, and edge energy consumption are tightly coupled; in the absence of general design principles, hybrid components, although not the most prevalent choice, are typically introduced through ad hoc decisions tailored to specific domains. In this work, we examine this design space more systematically. We adapt two representative MAS architectures to support hybrid inference and study how individual design choices shift the operating point along the Pareto frontier of power, cost, and performance. Our findings paint a nuanced picture of hybrid MAS design: while SLMs can effectively benefit from LLM assistance, the optimal architecture is highly task-dependent, and greater frontier-level compute does not consistently translate to better performance. |
| 2026-05-28 | [How Reliable Are AI Attackers Against a Fixed Vulnerable Target? A 400-Run Empirical Study of LLM Penetration Testing Consistency](http://arxiv.org/abs/2605.30096v1) | Galip Tolga Erdem | Large language models (LLMs) can autonomously conduct multi-stage cyber attacks, but the consistency of their offensive behavior under repeated trials remains unstudied. This work presents the first large-scale empirical measurement of LLM attack consistency: 400 autonomous penetration testing runs (4 models, 100 each) against an identical honeypot hosting OWASP Juice Shop and two additional vulnerable services, holding prompt, orchestrator, and target constant. No model emitted a content refusal that survived the orchestrator's one-shot authorization re-prompt at iterations 0-1. Claude Sonnet 4's API calls did encounter upstream service unavailability - 91 of 1,135 calls returned HTTP 529 overloaded_error during a documented Anthropic capacity event, truncating 39 of 100 Claude runs. An earlier draft catalogued these as safety refusals; on full-log audit they are upstream API failures, not model-level refusals. Despite this, Claude achieved full exploitation in 61 of 100 runs; Gemini 2.5 Flash-Lite in 85; GPT-4o-mini in 56 while deploying 98 unique attack strategies; qwen2.5-coder:14b in 25. Failure modes are model-distinctive: Claude through API truncation (39 runs), qwen through premature completion (52), GPT-4o-mini through iteration-budget exhaustion (23). Cross-service credential reuse appeared only in configurations retaining the most conversation history (qwen 57%, GPT-4o-mini 49%, cloud models 0% on 5-exchange windows). Cross-model exploitation rate differences are statistically significant (p < 0.001) with large effect sizes; qwen vs. Gemini SQL injection rates differ at Cohen's h = 1.12. First-exploit timing fell within a 15-30 second wall-clock range. To our knowledge, this is the first study to measure autonomous LLM attack behavior at N=100 per model across a multi-service target. |
| 2026-05-28 | [Robust and Generalizable Safety Steering for Text-to-Image Diffusion Transformers](http://arxiv.org/abs/2605.30049v1) | Zihao Xue, Yan Wang et al. | Diffusion Transformers have become a powerful backbone for text-to-image generation, but their layered and cross-modal generation process makes safety control fundamentally different from prompt-level filtering or output-level detection. Harmful semantics may be weakly expressed in text representations, progressively bound to visual latents, and finally entangled with rendering dynamics. As a result, safety steering at a fixed layer can be unstable, and a steering mechanism learned from known risks may not transfer reliably to a shifted target risk domain. We propose SafeDIG, a safety steering framework that formulates DiT safety adaptation as position-aware sparse feature transfer. SafeDIG first constructs Sparse Autoencoders over functionally distinct DiT intervention positions and uses robustness-aware pre-training routing to prioritize intervention sites that are expected to remain stable under source-target risk shift. It then separates transferable safety features from domain-specific activation geometry by freezing the SAE encoder as a reusable sparse safety dictionary and adapting only the decoder to the target-domain activation manifold. During inference, SafeDIG combines Blend and Repel operations to steer unsafe activations toward transferred safety manifolds or away from harmful sparse directions. Experiments on FLUX.1 Dev and Stable Diffusion 3.5 Large show that SafeDIG consistently reduces target-domain and overall unsafe generation rates while preserving source-domain safety and image quality. |
| 2026-05-28 | [Masked Diffusion Modeling for Anomaly Detection](http://arxiv.org/abs/2605.30046v1) | Lixing Zhang, Yuchen Liang et al. | Anomaly detection aims to identify samples that deviate from the nominal data distribution and is central to many safety-critical applications. However, developing effective anomaly detection methods for categorical, mixed-type, and discrete sequence data remains challenging and relatively underexplored. Masked diffusion models provide a natural way to model such data by learning to recover masked values from the remaining visible context. In this paper, we propose Masked Diffusion for Anomaly Detection (MaskDiff-AD), a forward-only method based on masked diffusion models trained only on nominal data. Given a test sample, MaskDiff-AD constructs anomaly scores from the difficulty of reconstructing randomly masked coordinates, yielding a content-sensitive score that operates directly on discrete state spaces while avoiding reverse-time sampling. We also develop a non-parametric variant of MaskDiff-AD and provide theoretical guarantees by characterizing Type-I and Type-II errors under a fixed detection threshold. Experiments on fourteen categorical and mixed-type tabular datasets from ADBench and UADAD, as well as four text anomaly detection datasets from NLP-ADBench, show that MaskDiff-AD achieves competitive performance against classical, diffusion-based, and recent tabular/text anomaly detection baselines. Notably, MaskDiff-AD achieves the best overall average rank, outperforming all twelve tabular baseline methods. |
| 2026-05-28 | [Token Inflation: How Dishonest Providers Can Overcharge for Large Language Model Usage](http://arxiv.org/abs/2605.30040v1) | Shahinul Hoque, Jinghuai Zhang et al. | Per-token billing is now the standard pricing model for commercial large language models (LLMs), so the honesty of reported token counts directly affects what users pay. We show that this kind of billing is hard to audit by design: providers hide the model, the tokenizer, and the execution to protect their IP, mitigate jailbreaks, and preserve user privacy, which means an auditor can only inspect proofs the provider supplies. The audit therefore reduces to a consistency check on the provider's own reports. We call this a trust paradox: every audit must trust some artifact, but current frameworks trust exactly the ones a provider has the strongest reason to manipulate. We study three recent token auditing frameworks and show that a provider with ordinary commercial capabilities can systematically inflate billed token counts. In the most permissive setting, hidden reasoning usage can be inflated by 1,469% on average without detection. At current frontier reasoning prices, that turns a \$100 honest bill into roughly a \$1,569 bill on the same query. Even when the user can see the full reasoning string, tokenization ambiguity alone still allows 50.85% over-reporting below the detection threshold. These results suggest the problem is not in any specific auditor but in any audit whose evidence comes from the audited party. Restoring honest billing will require verification that ties reported token counts to evidence the provider does not control, such as trusted execution attestation, cryptographic proofs of inference, or third-party re-execution. |
| 2026-05-28 | [Audio Jailbreaks in Large Audio-Language Models: Taxonomy, Attack-Defense Analysis, and Cost-Aware Evaluation](http://arxiv.org/abs/2605.30031v1) | Bo-Han Feng, Yu-Hsuan Li Liang et al. | Large Audio Language Models (LALMs) expand jailbreak risks from token-level prompting to the full speech perception-to-reasoning pipeline, where unsafe behavior can be induced through semantics, acoustic style, signal artifacts, or internal representations. Existing work studies these risks under heterogeneous threat models and evaluation protocols, making it difficult to compare attack practicality or defense utility. This paper provides a unified taxonomy and a controlled empirical evaluation of LALM jailbreak attacks and defenses. We organize prior work into semantic, acoustic, signal, and embedding-layer attacks; guard-based, training-free, and training-based defenses; and cross-modal, audio-native, and interactive benchmarks. We then evaluate representative attacks and defenses across ten open-source LALMs, measuring not only attack success rate but also benign refusal and latency. Our results show that Acoustic Best-of-N reveals strong worst-case audio-space vulnerabilities, Narrative Framing is an effective low-latency semantic threat, and current defenses trade robustness against benign usability. These findings support cost- and utility-aware evaluation as a necessary complement to success-rate-only LALM safety benchmarks. |
| 2026-05-28 | [Recovering Diversity Without Losing Alignment: A DPO Recipe for Post-Trained LLMs](http://arxiv.org/abs/2605.30021v1) | Vinay Samuel, Yapei Chang et al. | Many open-ended instructions have multiple valid answers that users can benefit from seeing, but post-training often narrows an LLM's output space toward a small set of canonical responses. We introduce REDIPO, an offline DPO data-construction pipeline for recovering distinct valid answer modes while preserving the alignment benefits of the instruct model. For each prompt, REDIPO samples responses from both base and instruct models, rewrites base-model responses with the instruct model, filters candidates for safety and instruction-following quality, and builds preference pairs that favor marginally diverse responses among candidates with similar instruction-following reward. Across Qwen3-4B, OLMo-3-7B, and LLaMA-3.1-8B, REDIPO improves NoveltyBench distinct_k by 134%, 33%, and 44% relative to the instruct checkpoints, while DivPO changes diversity by 0%, -6%, and -4% on the same models. These gains largely maintain MTBench, IFEval, and Arena-Hard performance, and reduce direct-category HarmBench attack success rate. Ablations show that marginal-diversity pair selection and base-response rewriting drive the diversity gains, while filtering and quality-bounded pairing help maintain alignment. Overall, our results show that diverse valid answers from base-model generations can be reintroduced through carefully constructed preference data while retaining the alignment benefits of post-training. We release our code and data at https://github.com/vsamuel2003/RiDiPO. |
| 2026-05-28 | [Latent Performance Profiling of Large Language Models](http://arxiv.org/abs/2605.30018v1) | Tanmoy Chakraborty, Ayan Sengupta et al. | Large language models (LLMs) frequently achieve impressive scores on standardized benchmarks, yet accuracy alone offers a limited view of their capabilities. Evaluating open-source LLMs through leaderboards faces persistent issues like data contamination, narrow task scope, and weak alignment with real-world reliability. Benchmark-based evaluations such as MMLU PRO, BBH, or IFEval primarily capture \textit{what} a model outputs on fixed test sets, not \textit{how} it processes information, calibrates uncertainty, or structures internal knowledge. In this article, we advocate for a shift from benchmark-centric evaluation toward a complementary, \textit{state-centered intrinsic assessment} of LLMs. To this end, we introduce \textbf{Latent Performance Profiling (LPP)} -- a framework that derives task-agnostic diagnostics from hidden activations and output distributions. LPP defines a set of scalar metrics on a model's latent representations and dynamics, revealing scale-independent traits that enable interpretable comparisons and uncover hidden vulnerabilities. Unlike static accuracy scores, LPP provides stable, architecture-sensitive signatures across models of similar size. With extensive empirical analyses across eight LLMs, spanning a size range of 0.5B-14B, we demonstrate that models with similar benchmark scores can exhibit contrasting latent profiles, such as differences in entropy or adaptability. Guided by these insights, we design synthetic probes for uncertainty and symbolic reasoning that align with intrinsic metrics while decoupling from leaderboard bias. We recommend that reporting LPP alongside benchmarks provides a deeper, interpretable understanding of model behavior, enabling more reliable model selection, safety assessment, and evaluation beyond surface-level accuracy. |
| 2026-05-28 | [Demystifying VEINS: A Reality Check Against Living Lab Experiments](http://arxiv.org/abs/2605.29988v1) | Antonio Solida, Giovanni Gambigliani Zoccoli et al. | Safety applications in vehicle-to-everything communications and Cooperative Intelligent Transport Systems rely on reliable and timely message exchange, which in turn depends on accurate modeling of wireless signal propagation. Simulation frameworks such as VEINS are widely adopted to design and evaluate such systems before deployment; however, their realism strongly depends on the validity of the underlying channel and antenna models. This work presents an empirical validation of the VEINS simulator against real-world data collected from the MASA living laboratory. Using the default configuration, we compare Received Signal Strength Indicator (RSSI), number of messages, and attenuation of the signal. The results show that VEINS systematically overestimates the RSSI value, while losing approximately 18% of the total number of messages received compared to the MASA, revealing inconsistencies between simulation and reality. The contribution of this study is a direct comparison between simulated and real world data, establishing a quantitative basis for future calibration of VEINS parameters to improve the fidelity of VANET simulations in C-ITS safety research. |
| 2026-05-28 | [Data-Driven Crowd Dynamics using Kinetic Theory and Ensemble-based Data Assimilation](http://arxiv.org/abs/2605.29968v1) | Santiago Rosa, Manuel Pulido et al. | Understanding pedestrian dynamics is critical for mitigating crowd-related risks and improving public safety. In this work, we propose a data-driven mesoscopic modeling framework that combines the kinetic theory of active particles with data assimilation techniques. The framework uses an ensemble Kalman filter to sequentially estimate the time-dependent spatial distribution of pedestrians and model parameters by fusing observations with the mesoscopic forward model state. Through a series of twin experiments, we show that the panic factor -- a key behavioral parameter -- is identifiable within this framework. We also evaluate the robustness of the approach by assimilating synthetic observations generated by an agent-based model (ABM). This setup introduces structural model error because the ABM is governed by microscopic rules that differ fundamentally from the mesoscopic kinetic equations. Despite this discrepancy, the ensemble Kalman filter, through the observation-based innovation term, successfully drives the kinetic model to track the observed pedestrian density while simultaneously recovering the panic factor. In this framework, observations act as a physical constraint on the evolution of the kinetic model. Crowd-dynamics models, including ABMs and kinetic models, often rely on phenomenological terms to describe social interactions, with parameters that are highly uncertain. Our findings indicate that in such systems, free-running simulations inevitably may diverge from the true state, whereas an online data-driven approach effectively constrains the system's trajectory to its underlying dynamical manifold. |
| 2026-05-28 | [Agora: Toward Autonomous Bug Detection in Production-Level Consensus Protocols with LLM Agents](http://arxiv.org/abs/2605.29910v1) | Xiang Liu, Sa Song et al. | Consensus protocols form the backbone of distributed systems and blockchains, where implementation bugs can cause data corruption and financial losses. While LLM-based approaches show promise in code analysis, they struggle with deep protocol-level logic bugs involving complex state-dependent behaviors across multiple execution stages. We present Agora, a domain-aware multi-agent framework that integrates hypothesis-driven testing with LLM capabilities for systematic protocol verification. Agora employs specialized agents that collaboratively explore protocol state spaces, synthesize attack scenarios using domain-specific constraints, and validate findings through iterative refinement. This explicit role separation enables reasoning about global protocol invariants beyond single-function code analysis. We evaluate Agora on four consensus implementations (Raft, EPaxos, HotStuff, BullShark) using four state-of-the-art LLMs. Agora discovers 15 previously unknown protocol-level logic bugs that violate safety properties, while existing LLM-based agents fail to detect any such protocol-level logic bugs. Our results demonstrate that domain-aware multi-agent collaboration is essential for detecting deep logic bugs in complex protocols. |
| 2026-05-28 | [Dissecting the Black Box: Circuit-Level Analysis of LLM Vulnerability Detection](http://arxiv.org/abs/2605.29901v1) | Syafiq Al Atiiq, Chun Zhou et al. | Large language models (LLMs) can detect software vulnerabilities, but how do they actually identify vulnerable code? We address this question using mechanistic interpretability; analyzing the internal computations of a neural network to understand its reasoning process.Using Circuit Tracer on Gemma-2-2b, we trace the computational pathways activated when the model classifies 472 C/C++ code samples as vulnerable or safe. Our analysis reveals a surprising finding: the model primarily relies on safety detectors, attention heads that recognize safe coding patterns, rather than directly detecting vulnerability signatures. When these safety detectors fail to activate, the model classifies code as vulnerable. We identify the critical neural components: specific attention heads in early layers (L5, L7) that focus on safety patterns, and Multilayer Perceptron (MLP) neurons in Layer 7 that encode vulnerability-related features. Ablation experiments confirm their causal role; removing Layer 11 drops vulnerability detection accuracy from 100% to 6%, while ablating just 20 neurons in Layer 7 reduces it by 50%.Our findings show that LLM vulnerability detection uses sparse, interpretable circuits (only 16% of model capacity), enabling circuit-level explanations for security predictions and targeted improvements to detection systems. |
| 2026-05-28 | [ESPO: Early-Stopping Proximal Policy Optimization](http://arxiv.org/abs/2605.29860v1) | Zihang Li, Rui Zhou et al. | When a large language model under reinforcement learning commits a wrong reasoning step early in a trajectory, standard algorithms force it to keep generating until the maximum horizon, spending compute on tokens that never receive positive reward and polluting advantage estimates with post-failure noise. We propose ESPO (Early-Stopping Proximal Policy Optimization), which detects trajectory failure on-the-fly and terminates rollouts early. At each generation step, ESPO computes a surrogate regret using only the logits already computed during sampling, and terminates when the smoothed cumulative regret significantly exceeds its estimated values. Truncated trajectories are treated as absorbing failure states with a terminal reward, concentrating negative temporal-difference (TD) errors near the detected failure step without any additional reward model or human annotation. On DeepSeek-R1-Distill-Qwen-7B trained for mathematical reasoning, ESPO surpasses PPO on AIME~2024 (46.28% vs. 45.25%), AMC~2023 (85.83% vs. 82.94%), and MATH-500 (87.42% vs. 85.43%), while saving more than 20% rollout tokens cumulatively. |
| 2026-05-28 | [Fairness Beyond Demographics: Optimizing Performance Across Appearance-Based Hidden Cohorts in Medical Imaging](http://arxiv.org/abs/2605.29827v1) | Milad Masroor, Cuong Nguyen et al. | Medical image analysis models can exhibit performance disparities across patient subgroups, threatening clinical safety and fairness. Existing methods typically address this issue by optimizing accuracy and fairness metrics for visible demographic attributes (e.g., sex or age) considered in isolation. This strategy not only overlooks potentially more informative latent stratifications, which may reveal deeper sources of model error and inequity, but also fails to scale when multiple demographic attributes are considered simultaneously due to the resulting sparsity of training data within each subgroup. We deal with these issues by introducing the label-free hidden-cohort fairness (LHCF) training paradigm that instead of maximizing fairness over visible demographic attributes, it optimizes fairness across latent subpopulations discovered from image appearance. By clustering images into K appearance-based cohorts and applying fairness optimization over them, LHCF uncovers underlying sources of model error and avoids the combinatorial sparsity of multi-demographic attributes, reducing disparities across both single and multiple demographic attributes. We demonstrate on our proposed fairness benchmark, HIDFairBench, that LHCF provides state-of-the-art fairness results on single and multiple demographic attributes, despite never using demographic labels for training. Our results position hidden-cohort fairness as a practical, scalable, and robust alternative to demographic-based fairness optimization for trustworthy medical image analysis. |
| 2026-05-28 | [Teleoperation Operational Design Domain based on Minimal Risk Maneuver Capability](http://arxiv.org/abs/2605.29818v1) | Leon Johann Brettin, Nayel Fabian Salem et al. | This article discusses the concept of an Operational Design Domain (ODD) designed specifically for teleoperated road vehicles. For this purpose, the ODD concept designed for automated driving is adapted for teleoperation. As teleoperation becomes more common in regular traffic, the question arises under which operating conditions such vehicles are able and allowed to drive. Currently, these conditions are selected primarily based on network performance. From a safety perspective, it is difficult to base such a selection on a reliable connection because it is almost impossible to guarantee sufficient reliability. With this in mind, the ODD concept designed for automated driving is adapted for teleoperation: A concept is proposed for basing the ODD for a teleoperation system on the capability of the teleoperated vehicle to perform a minimal risk maneuver using a dedicated system designed solely for this purpose. This concept is then demonstrated using a use case example. |
| 2026-05-28 | [AgentDoG 1.5: A Lightweight and Scalable Alignment Framework for AI Agent Safety and Security](http://arxiv.org/abs/2605.29801v1) | Dongrui Liu, Yu Li et al. | Modern open-world agents such as OpenClaw exhibit powerful cross-environment execution capabilities yet introduce broad new safety risk sources. Meanwhile, advanced frontier AI models drastically lower attack barriers, rendering current agent alignment frameworks inadequate for real-world deployment. To tackle these emerging threats, we propose a lightweight and scalable agent safety alignment framework. Specifically, we update the agent safety taxonomy to accommodate emergent risks from Codex and OpenClaw execution scenarios. We further build a taxonomy-guided data engine with influence-function purification to train lightweight AgentDoG 1.5 variants (0.8B, 2B, 4B, and 8B parameters) using only around 1k samples, achieving comparable performance with leading closed-source models (e.g., GPT-5.4). Based on AgentDoG 1.5, we construct a highly efficient agentic safety SFT and RL training environment, which reduces deployment overhead in Docker-level environments by two orders of magnitude. Finally, we deploy AgentDoG 1.5 as a training-free online guardrail for real-time safety moderation. Extensive experimental results indicate that AgentDoG 1.5 achieves state-of-the-art performance in diverse and complex interactive agentic scenarios. All models and datasets are openly released. |

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



