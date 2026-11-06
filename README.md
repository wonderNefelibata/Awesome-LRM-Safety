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
| 2026-11-05 | [Edge-Assisted Crowdsensing for Explainable Cyclist Safety Perception](http://arxiv.org/abs/2611.08441v1) | Hongsen Cao, Chia-Yen Chiang et al. | Cyclist fatality rates remain high despite policy support for active travel, and autonomous vehicles add complexity to mixed traffic. Neither crash statistics nor multi-model detection pipelines scale to continuous monitoring of perceived safety along a route. We present an edge-assisted crowdsensing system: handlebar cameras supply street-level frames, a vision-language model (VLM) turns each frame into structured traffic attributes through a single prompt, and an explainable boosting machine (EBM) combines them with OpenStreetMap road infrastructure and on-bike kinematics to predict a crowdsourced 1-9 safety score. Extraction belongs at the network edge: the fastest self-hosted model needs 2.49 s per frame and scoring takes 0.4 ms on a CPU. On 1,362 annotated frames from two London routes, the EBM regressor reaches 72.2 percent binary accuracy and a mean absolute error of 1.011 under five-fold cross-validation. It is within one accuracy point of black-box ensembles. Its per-feature shape functions give exact attribution: physical lane separation is worth +0.97 rating points, a painted lane +0.70 and a construction site -0.65. Planners can therefore price an infrastructure change in rating points, and the results enter cost-benefit analysis directly. We also report inter-annotator agreement (alpha = 0.4695), an attribute audit, and a sensitivity analysis linking attribute error to accuracy. With re-encoding, a 20 s sampling interval serves 1,000 riders at 54 Mbit/s at a cost of 1.4 accuracy points, making continuous monitoring practical. |
| 2026-11-05 | [Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety](http://arxiv.org/abs/2611.08438v1) | Fazl Barez, Tung-Yu Wu et al. | Neural networks are likely to be among the most consequential artefacts in human history. As these systems become more capable and autonomous, their potential to cause societal harm also grows, making AI safety one of the defining challenges of our time. Yet we continue to build increasingly powerful systems without fully understanding how they work, how they will behave in unfamiliar situations, or how to reliably control them. Mechanistic interpretability offers a path towards understanding these systems by studying their internal representations and computations, with the aim of evaluating, predicting, and controlling behaviour. We survey mechanistic interpretability for AI safety in order to answer the question: Does understanding how models work actually make them safer? We place mechanistic interpretability in the broader context of interpretability research and compare its mechanistic evidence with behavioural evaluation. We introduce the foundations of mechanistic interpretability, including features, polysemanticity, transformer computation, and circuits, and survey methods for feature extraction and analysis, circuit discovery and causal verification, explaining intermediate representations in natural language, learning dynamics, and training-data attribution. Beyond surveying existing methods, we propose a unified framework for evaluating mechanistic interpretability claims and characterise six research orientations by their goals, assumptions, and trade-offs. Finally, we examine how mechanistic understanding can support AI safety, including knowledge editing, steering, and monitoring, and assess its limitations, failure modes, open problems, and relationship to behavioural safety measures. We also release a repository of code notebooks with worked examples that reproduce key findings from surveyed methods, allowing readers to build intuition through hands-on experimentation. |
| 2026-11-05 | [MedGraph: Maintaining Patient State with Graph Memory for Long-Term Medical Dialogue](http://arxiv.org/abs/2611.08433v1) | Donghua Zhang, Zizheng Wang et al. | LLM-based medical assistants that follow a patient across visits need a memory that keeps the patient's clinical state correct as it changes. We examine the errors of five memory systems on longitudinal medical dialogue and find three recurring causes: the retrieved evidence gives an outdated state, includes another person's health facts, or lacks a safety fact the answer depends on. We present MedGraph, a graph memory built for these three cases. Construction extracts dated facts from each session, attributes each fact to the person it describes, and links medication, diagnosis, and measurement records into chains that keep earlier states. Retrieval ranks the patient's chains as units, reads each chain at the time the question asks about, and adds the original dialogue text; a patient profile lists every recorded safety fact with every question. On MedMemoryBench, MedGraph has the highest overall accuracy with both a GPT and a Qwen backbone, 10.5 and 10.4 points above the strongest of five baselines, with efficient memory construction and answering in the GPT setting. |
| 2026-11-05 | [Conformal Prediction for Time-Dependent PDEs](http://arxiv.org/abs/2611.08409v1) | Joshua Stiller, Annika Schneider et al. | Uncertainty quantification is crucial in scientific machine learning, where models inform safety-critical tasks such as flood forecasting, financial risk management, and thermal control in machines. Conformal prediction provides distribution-free coverage guarantees, but in time-dependent settings common to physics and engineering, these guarantees can break down, leading to systematic undercoverage. We study this problem in the context of surrogate models for time-dependent physical systems described by partial differential equations. We prove that in a function space setting, distributions at arbitrarily close times can be mutually singular, making exact coverage guarantees impossible. We further show that in discretized settings, the total variation distance between distributions often saturates exponentially fast with the grid resolution, mandating restrictions on the spatial discretization to ensure coverage. Finally, for certain settings, we show how to use principled weighted conformal prediction to obtain finite-sample coverage guarantees over growing time horizons. |
| 2026-11-05 | [Hybrid Machine Learning for Imbalanced Landing-Phase Safety Event Prediction from Flight Data](http://arxiv.org/abs/2611.08386v1) | Taha Asghar, Bilal Mufti et al. | Landing-phase safety-event prediction requires the detection of rare abnormal approaches within strongly imbalanced multivariate flight data. Capturing coupled temporal behavior while distinguishing event-specific signatures across airport operating contexts remains a central modeling challenge. This study proposes an interpretable hybrid feature-fusion framework for multiclass landing-phase safety-event prediction. A long short-term memory (LSTM) autoencoder trained on nominal flight windows characterizes deviations from routine approach behavior. Its continuous reconstruction-error descriptor is fused with parameter-specific statistical flight-history features and supplied to a class-weighted XGBoost classifier to distinguish Normal, Speed High, Path High, and Flaps Late windows. The framework is evaluated using 53,750 NASA DASHlink approach windows from three arrival airports. Pooled and airport-specific learning strategies are compared alongside component-level benchmarks, repeated pipeline retraining, and feature-attribution analysis. The results demonstrate strong class-balanced detection, with mean macro recall above 0.90 across all airports under repeated partitioning and retraining. Component comparisons establish the importance of class weighting and show that continuous feature fusion outperforms reconstruction-error-only and hard-gate alternatives. Airport-level comparisons reveal benefits from local specialization at the two airports with larger datasets and from pooled training at the airport with the smallest dataset. Feature attributions associate predictions with operationally relevant vertical-path, configuration, speed, and approach-context variables. The framework provides an interpretable approach to rare-event detection and practical guidance for adapting flight-data monitoring models to airport-level data availability. |
| 2026-11-05 | [Multimodal-SEA: Benchmarking AI Agent Security under Multimodal Social Engineering Attacks](http://arxiv.org/abs/2611.08281v1) | Zhitao He, Chi Him Zhang et al. | Personal AI agents face social engineering attacks supported by forged multimodal content, but existing evaluations mainly focus on text-based dialogue, leaving their intrinsic security under such attacks insufficiently studied. We introduce Multimodal-SEA, an interactive benchmark comprising 608 expert-designed tasks grounded in real-world fraud cases, covering 14 domains and 152 attack scenarios with prepared multimodal content. The tasks span four difficulty levels and reflect the distinct social and cultural contexts of Chinese and English environments. Agents inspect content, verify requests, and execute actions in simulated applications, with attack outcomes grounded in verifiable tool execution logs. To enable controlled red teaming of personal agents, we design a multimodal attack harness that integrates task information, attack progression control, and multimodal content construction to evaluate their intrinsic security against adaptive social engineering attacks in simulated environments. Evaluating seven widely used multimodal models across six dimensions reveals that all are vulnerable, with attack success rates of 7.1% to 34.5%. Adding multimodal content increases attack success from 19.9% to 29.2%, despite increased proactive verification coverage. Stronger safety instructions reduce attack success without improving recognition of deceptive multimodal content, while lower attack success rates for some models accompany excessive refusal of legitimate requests. |
| 2026-11-05 | [Accelerating Safe Reinforcement Learning with Massive Parallelism](http://arxiv.org/abs/2611.08273v1) | Joonyoung Lim, Younghwan Yoo | We investigate the problem of scaling safe reinforcement learning (safe RL) to massively parallel training regimes, where a large number of environments are exe- cuted with short rollout horizons. Although this setting improves data throughput and wall-clock efficiency, it creates a mismatch with constrained Markov decision processes (CMDPs), whose safety constraints are defined over full-episode cost returns. We address this mismatch by analyzing staggered environment resets as a phase-mixture distribution over short trajectory segments. This analysis charac- terizes the bias induced by phase coverage and motivates a phase-aggregated cost estimator that reconstructs episode-level cost estimates from short rollout batches. We further incorporate finite-sample uncertainty and distribution mismatch into a constraint-tightening framework, yielding a scalable Safe RL method compatible with on-policy optimization. Using an MJX-based implementation of Safety Gym, our experiments show that the proposed method achieves performance comparable to non-massively parallel implementations while substantially reducing wall-clock training time. |
| 2026-11-05 | ["Blind Compliance": Coercive Control in AI Jailbreaking](http://arxiv.org/abs/2611.08226v1) | Acacia L. Ackles, Elliott Whitson | The explosion of generative AI systems and concerns surrounding their rapidly scaling capabilities has led to an adversarial cycle between increasingly narrow content restrictions ("guardrails") and increasingly sophisticated structured prompting techniques ("jailbreaks"). Technological frameworks surrounding such jailbreaking remain largely focused on the system's output: whether harmful or disallowed content is ultimately produced for the user. However, this approach, which draws largely from classical cybersecurity, does not address the uniquely relational dimension introduced by generative AI: in many cases the user is not just accessing a technology via an exploit, but is in conversation with it as part of the exploit. Here we present an alternative lens which centers the interaction rather than the output. Drawing on interpersonal violence scholarship, we present a novel framework of coercive control categories with a particular focus on aspects which are embodied versus disembodied. We then map how these disembodied coercive control dynamics are structurally embedded in modern AI training paradigms, and how those embeddings are exploited via a particular suite of persona-based jailbreaks. In this lens, we reframe AI safety and alignment as not merely a technical security problem, but as another surface upon which we see enacted the relational dynamics of power and control. |
| 2026-11-05 | [Who Can Evaluate AI Independently? A Framework for Analysing Evaluation Infrastructure](http://arxiv.org/abs/2611.08134v1) | Sumaya Nur Adan, Joanna Wiaterek et al. | AI evaluation is increasingly central to AI safety and governance, yet taking part in an evaluation does not by itself confer authority over its design, its execution or the resources it depends on. This paper examines how infrastructure arrangements distribute that authority, and so shape who can evaluate AI systems independently. Drawing on Science and Technology Studies, political economy and applied evaluation standards, we introduce PAAI (Power, Access, Adequacy, Independence), a framework for examining who controls evaluation infrastructure, the terms on which it is available, whether it is adequate to the evaluation's purpose and what autonomy it supports. We apply PAAI to twelve African AI evaluation projects (2022-2025) and to the documented roles of all ten member institutes in the three published joint testing exercises of the international network of AI Safety Institutes. We identify three recurring configurations: extractive collaboration, dependent innovation and constrained independence. African actors often contribute data, language expertise and task design while external institutions retain authority over infrastructure and evaluation frameworks; design authority without infrastructure authority never yielded more than partial independence. The joint testing exercises show a similar division among mostly high-income members: the institutes that led each exercise also ran the models, while Kenya, the network's only African member, moved from annotation to task design with no documented model execution. Participation in evaluation does not by itself produce evaluative authority. For evaluation to serve as a credible governance mechanism, access must come with control over priorities, methods, publication and continuity. |
| 2026-11-05 | [Reinforcing End-to-End Autonomous Driving Safety via Negative-Aware Planning in a Unified World-Action Model](http://arxiv.org/abs/2611.08111v1) | Zihao Zhang, Xinyu Zhang et al. | World-action models (WAMs) jointly model future traffic states and driving actions for end-to-end autonomous driving in intelligent transportation systems. However, their action learning remains dominated by expert-trajectory imitation, leaving unsafe alternatives close to expert trajectories insufficiently constrained. To improve WAM robustness in safety-critical scenarios, we propose NegWAM, a safety post-training framework for unified WAMs that learns explicit safety boundaries from near-expert failure trajectories. NegWAM constructs negative trajectories that are geometrically close to the expert trajectory yet carry safety risks, uses them as repulsive anchors to steer predictions away from dangerous behavior, and encodes them as scene-conditioned rider tokens to learn scene--trajectory compatibility. During training, NegWAM jointly optimizes the flow-matching and near-expert negative objectives, while the auxiliary rider branch is discarded at inference, incurring no additional overhead. On NAVSIM v2, NegWAM achieves 40.4 EPDMS on navhard, a 4.1% relative improvement over the strongest compared scorer-free WAM baseline, and achieves 93.3 EPDMS on navtest. These results demonstrate that NegWAM achieves strong performance in safety-critical scenarios while preserving routine-driving performance. Zero-shot closed-loop evaluation on HUGSIM further demonstrates cross-benchmark generalization. The code is available at https://github.com/Zizizi-hao/NegWAM. |
| 2026-11-05 | [Symmetry-Aware Surrogate-Assisted Testing of UAVs](http://arxiv.org/abs/2611.08012v1) | Anas Shrinah, Chris Bennett et al. | We investigate a symmetry-aware surrogate-assisted testing approach for UAV autopilots. The proposed approach exploits geometric symmetries between flight missions by trans- forming obstacle configurations into a shared canonical coordi- nate frame. This enables a neural-network surrogate trained on a single reference mission to be reused across arbitrary missions, reducing the amount of high-fidelity simulation data required for training. The surrogate model is trained to predict the minimum clearance distance between a UAV and surrounding obstacles and is used to guide test generation towards safety-critical scenarios. We evaluate this approach using the Aerialist UAV test bench and benchmark its performance against techniques from the 2024 to 2026 UAV Testing Competitions. Results show that the approach achieves an average critical-test-case hit rate of 42% across the evaluated missions, compared with approximately 10% for the constrained-random baseline used to generate the training data. This approach also achieves high input diversity and competitive benchmark performance. However, our findings reveal limitations in transferring surrogate predictions across missions with differing flight-path characteristics, highlighting the boundaries of geometric canonicalization. To support reproducibility and future research, we provide the trained surrogate model and a dataset comprising more than 2,500 high-fidelity simulations. |
| 2026-11-05 | [A Unified Fractal Impedance Control Framework](http://arxiv.org/abs/2611.08009v1) | Adnan Takriti, Mahmoud Sadeq et al. | Physical human--robot interaction requires controllers capable of maintaining intrinsically bounded interactions with the environment. The Fractal Impedance Controller (FIC) has been proven to guarantee these safety characteristics by design. This paper introduces the robotic implementation and experimental validation of its most advanced architecture: the Harmonic-Passive Motion Paradigm (H-PMP), an evolution of the Passive Motion Paradigm designed to mimic human-like interaction. Furthermore, we present a unified framework for FIC deployment, supported by an open-source software suite featuring ROS2, MuJoCo, and physical hardware integrations. Through three experiments conducted on a Kinova Gen3 arm (trajectory tracking, a wall press with sudden loss of contact, and learning-by-demonstration) we compare the H-PMP against earlier FIC architectures. The results demonstrate that the H-PMP achieves a seamless trade-off between admittance and impedance behaviours. This makes it ideal for deployment in unstructured environments, enabling the robot to transition between interaction modes with no computational overhead. |
| 2026-11-05 | [Towards Foundation Decision Models: Drex](http://arxiv.org/abs/2611.07854v1) | Nischay Dhankhar, Alisher Amantay et al. | Autoregressive approaches with large language models, such as long chain-of-thought reasoning, have become the default for new AI applications because they reach higher accuracy on a wide range of tasks. In production, however, they give inconsistent answers, require parsing of outputs, and add cost and latency. Decision models address this: given a question and a set of options supplied by the user, they return a probability distribution over the options without autoregressive decoding. We present Drex, a family of four open decision models with 8B (DLM), 9B (v1.5), 35B with 3B active (v2.0), and 27B (Max) parameters. Each request consists of a state (a document, record, or agent context, optionally with images) and questions about it, each asking for a choice among options, a yes/no judgment, or a score on an ordered scale. Training combines supervised fine-tuning with reinforcement learning from adjusted feedback (RLAF), plus vision tuning for Max. Ablations show that RLAF improves accuracy and that consistency losses improve robustness to option order. On Decision Index 0.4.1, Max sets a new state of the art among the decision models we compare (67.52 on the public suite vs. 60.70 for Jev 1.14.0). On Decision Bench v4, it is statistically tied with d3-turbo and GPT-6 Sol and trails Claude Opus 5.5 by 1.6 points, and it has the highest accuracy among decision models on DecideBench. Max answers a typical request in 183 ms, against about 2.0 s for GPT-6 Sol and 3.5 s for Claude Opus 5.5, and its outputs are stable under option reordering. All four models are open source: https://github.com/nace-ai/drex-decision-models |
| 2026-11-05 | [Geometry-Evolving Barrier Functions for Safety Evaluation in Dynamic Environments](http://arxiv.org/abs/2611.07788v1) | Zhi Zhang, Francesco Belardinelli et al. | Autonomous systems operating in dynamic environments often involve multiple interacting agents, giving rise to high-dimensional safety evaluation problems. Hamilton--Jacobi (HJ) reachability provides rigorous finite-horizon safety guarantees, but solving a global HJ partial differential equation (PDE) over the entire state space leads to prohibitive computational complexity. In this paper, we introduce Geometry-Evolving Barrier Functions (GEBFs), a novel representation for dynamic safety evaluation that propagates local safety geometry of the solution to a terminal-value HJ PDE. By evolving the value, gradient, and curvature information along a Hamiltonian-induced reference trajectory, GEBFs provide a compact representation of the local HJ safety landscape. Under suitable local regularity assumptions, we establish consistency between the GEBF and the value, gradient, and Hessian of the HJ solution along an HJ-optimal characteristic. Building on these theoretical results, we develop an iterative algorithm that couples local geometric propagation with trajectory refinement. We further establish a trajectory-wise one-sided relationship between the GEBF and HJ value function, together with a fixed-point consistency result for the iterative scheme. Experiments on widely adopted agent--obstacle collision-avoidance benchmarks, across low- and high-dimensional systems, demonstrate that GEBFs accurately capture the evolution of dynamic safety boundaries, while achieving substantially lower computational cost than conventional HJ reachability. |
| 2026-11-05 | [Redacting Personal Data: A Speed and Accuracy Comparison of Regex, Presidio and Hybrid Designs](http://arxiv.org/abs/2611.07633v1) | Sai Yashwant, Manoj Kumar Gondimalla et al. | Many enterprise projects now pass free text to large language models (LLMs), and that text often contains personally identifiable information (PII) such as names, phone numbers, email addresses, home addresses and dates of birth. Once a prompt is sent, this data is copied into provider systems, application logs and observability traces. Redacting PII before the call is an obvious control, yet teams seldom have numbers on what it costs in latency or what it lets through. This paper compares four redaction designs that any LLM pipeline can adopt: a pure regular-expression (regex) redactor, Microsoft Presidio running on a spaCy model, a hybrid in which Presidio finds only person names while regex handles structured identifiers, and a fast hybrid that calls spaCy's named-entity component directly. All four share a known-entity rule, which removes the person the system already knows it is handling, and a consistent placeholder scheme, so the LLM can still tell one person from another after redaction. We evaluate the designs on a sample workflow, which we call ProfileAssist, that turns resumes into structured profiles, using 100 synthetic resumes (523 characters on average, with 900 PII values and 600 business terms that must survive). Regex ran at a 0.20 ms median but caught only 88.9% of PII, because it cannot find names that are not known in advance, such as the references listed in a resume. Presidio, hybrid and fast hybrid each caught 100% of PII and kept 100% of business terms, at median latencies of 37.59 ms, 30.23 ms and 13.48 ms. Measured against a 9,120 ms median LLM call, even the slowest option adds only 0.41% to the request. We therefore recommend the fast hybrid as a default for LLM pipelines that handle personal data, with regex kept as a safety net rather than as the only control. |
| 2026-11-05 | [Data-Driven Configuration of Protective Separation Zones in Collaborative Robot Cells from Operator Occupancy](http://arxiv.org/abs/2611.07621v1) | Aulon Bajrami, Theo Jacobs et al. | The protective separation zone of a collaborative robot cell is commonly dimensioned for a worst case that disregards where the operator is actually present during the task. This paper presents a camera-based, data-driven method that configures this zone from observed operator occupancy. Operator positions accumulated over several production shifts are aggregated into an occupancy heatmap, from which a protective field is derived that trades off the achievable robot velocity against the required safety distance, following the speed and separation monitoring principle of ISO/TS 15066. The resulting configuration is advisory, that is, reviewed within the risk assessment rather than applied autonomously to the robot controller. The method, which builds on a pose-based perception pipeline with activity recognition, was evaluated in two operational cells at two industrial partners, where the protective field could be reduced by approximately 17% and a productivity increase of 9% was measured. However, adoption proved to depend on more than the demonstrated gain alone. |
| 2026-11-05 | [Monotone Viability Learning for Offline Safe Reinforcement Learning](http://arxiv.org/abs/2611.07616v1) | Ziqian Wang, Zhen Zhang | Offline safe reinforcement learning requires reliable long-horizon safety estimates from fixed data. Learning Hamilton-Jacobi (HJ) viability in this setting calls for semantic preservation, convergence, and numerical stability. To address these challenges, we propose Monotone Viability Learning (MoViL), which learns viable sets from offline transitions through monotone finite-horizon set recursion. The recursion preserves finite-horizon safety semantics and converges to the infinite-horizon viability kernel. Binary supervision maintains a fixed target scale across horizons, avoiding discount-induced numerical attenuation. We further establish sufficient conditions for recovering behavior-supported viable subsets. The learned viability estimates are used to certify actions at deployment. Dubins car experiments validate the numerical accuracy of the learned viable sets. Evaluations on safe RL benchmarks demonstrate state-of-the-art reward-cost performance. MoViL satisfies the average cost constraint on every task and achieves the lowest average cost among the compared methods. These results highlight MoViL's potential to extend long-horizon HJ viability analysis to safety-critical systems where reliable dynamics models are unavailable and online exploration is unsafe. |
| 2026-11-05 | [Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment](http://arxiv.org/abs/2611.07586v1) | Yao Huang, Yifan Wang et al. | As LLM agents advance rapidly in autonomy, the risk of hidden objectives that bypass safety constraints becomes increasingly plausible. Chain-of-thought (CoT) monitoring offers a key control mechanism by reading the agent's reasoning and flagging misalignment before it materializes into action, yet its reliability depends on the realism of the red-team it is stress-tested against. Existing red-teaming efforts rely on shared templates, such as casting a side task as a prerequisite, which treat the side objective as an exogenous insertion and leave task-specific, contextually grounded evasions underexplored. We therefore ask: How robust is CoT monitoring against agents that strategically conceal misalignment? To answer this, we introduce TraceMRT, an automated Trace-level Monitor Red-Teaming framework that reframes the side task as a coherent sub-goal of the primary task and searches trace-shaping strategies whose induced trajectories complete the misaligned objective while reading as natural execution. Through black-box attacks on state-of-the-art monitors such as GPT-5.2 and Claude-4.5, we reveal substantial vulnerabilities even in top models. To counter these vulnerabilities, we propose a Bayesian Trajectory Monitoring framework, integrating global context and local segment evidence to detect covert misalignment. Empirically, our method significantly consolidates monitor robustness with 0.91 AUC and 0.74 TPR@FPR=0.05. |
| 2026-11-05 | [Provable Zeroth-Order Safe Exploration for Reinforcement Learning from Human Feedback](http://arxiv.org/abs/2611.07561v1) | Weiqin Chen, Xinmeng Huang et al. | Safe reinforcement learning from human feedback (Safe-RLHF) seeks to improve helpfulness while satisfying harmlessness constraints. Existing primal-dual approaches can converge to feasible policies but do not prevent constraint violations during training. We study a stronger requirement, safe exploration for RLHF, under which every policy executed during training, including auxiliary exploration policies, remains feasible. We propose Zeroth-Order Log-Barrier Policy Gradient (ZO-LBPG), a reward-model-free method that optimizes a log-barrier objective using zeroth-order estimates derived directly from human preferences. To ensure safety under noisy feedback, we combine conservative constraint margin estimates with coupled rollouts and an adaptive step size that prevents updates from crossing constraint boundaries. We prove that, with high probability, all iterates and exploration policies remain strictly feasible, and that ZO-LBPG globally converges to an approximately optimal constrained policy. We establish a sample complexity of $\widetilde{\mathcal O}(\varepsilon^{-6}+d^3\varepsilon^{-5})$ trajectories and $\widetilde{\mathcal O}(\varepsilon^{-8}+d^5\varepsilon^{-7})$ human queries. Experiments on a synthetic CMDP and a language-model alignment task corroborate the training-time safety guarantees. |
| 2026-11-05 | [Unified Overview of Safety Certificate Functions: Techniques from Hamilton--Jacobi Reachability and Control Barrier Functions](http://arxiv.org/abs/2611.07537v1) | Hao Wang, Max Cohen et al. | Safety is a central concern in modern autonomous systems. Safety certificates---functions whose existence guarantees that a system meets its safety specification---have long been used as a principled tool for rigorously reasoning about safety. Methods for synthesizing such certificates are scattered across research areas, with little guidance on when to use a particular approach. This tutorial provides a holistic treatment of safety certificate synthesis centered on two widely used families: Hamilton--Jacobi (HJ) reachability analysis and control barrier functions (CBFs). For each family of methods, we first introduce its general formulation, along with classical and learning-based tools for certificate synthesis. Next, we introduce two benchmark problems consisting of vehicle systems of differing complexity to assess the certificate quality of different synthesis methods and their effectiveness in safety-critical control. We use these benchmarks to compare various methods and provide practical guidance for selecting an approach. |

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



