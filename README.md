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
| 2026-05-21 | [The Matching Principle: A Geometric Theory of Loss Functions for Nuisance-Robust Representation Learning](http://arxiv.org/abs/2605.22800v1) | Vishal Rajput | Robustness, domain adaptation, photometric and occlusion invariance, compositional generalisation, temporal robustness, alignment safety, and classical anisotropic regularisation are usually treated as separate problems with separate method families. This paper argues that much of their shared structure is one statistical problem: estimate the covariance of label-preserving deployment nuisance, then regularise the encoder Jacobian along a matrix whose range covers that covariance (the matching principle). CORAL, adversarial training, IRM, augmentation, metric learning, Jacobian penalties, and alignment-style constraints are different estimators of that object, not independent robustness tricks.   In the linear-Gaussian model we prove closed-form optimality (Theorem A), including cube-root water-filling within the matched range; necessity of range coverage for quadratic Jacobian penalties (Theorem G); the same range dichotomy at deep global minima; and two falsification controls (Lemma C; Corollaries E), with seven conditional consistency lemmas (D1-D7) for estimation under standard identifiability assumptions.   We introduce the Trajectory Deviation Index (TDI), a label-free probe of embedding sensitivity when task accuracy or Jacobian Frobenius norm is insufficient.   Thirteen pre-registered blocks from classical ML through Qwen2.5-7B test the predicted matched, then isotropic, then wrong-W ordering on geometry and deployment drift; twelve pass, and the sole exception (Office-31) is an eigengap failure named before the run. At 7B scale, matched style-PMH improves selective honesty and preserves Style TDI where standard DPO degrades it.   The contribution is naming the deployment nuisance covariance, stating what the regulariser must do, and supplying a closed-form falsifiable theory once that object is identified, not universality on every leaderboard. |
| 2026-05-21 | [MambaGaze: Bidirectional Mamba with Explicit Missing Data Modeling for Cognitive Load Assessment from Eye-Gaze Tracking Data](http://arxiv.org/abs/2605.22775v1) | Amir Mousavi, Mohammad Sadegh Sirjani et al. | Real-time cognitive load assessment from eye-tracking signals could potentially enable adaptive human-centered-AI such as safety-critical applications such as driver vigilance monitoring or automated flight deck assistance, yet two challenges persist: handling frequent data missingness from blinks and tracking failures, and efficiently modeling long-range temporal dependencies. We propose MambaGaze, a framework that addresses these challenges through 1) XMD encoding, which augments raw features with observation masks and time-deltas to explicitly model data uncertainty, and 2) bidirectional Mamba-2, which captures temporal dependencies with linear computational complexity. Experiments on CLARE and CL-Drive datasets under leave-one-subject-out evaluation show that MambaGaze achieves 76.8% and 73.1% accuracy, respectively, outperforming CNN, Transformer, ResNet, and VGG baselines by 4-12 percentage points. Edge deployment benchmarks on NVIDIA Jetson platforms demonstrate real-time inference at 43-68 FPS with power consumption below 7.5W, confirming feasibility for wearable cognitive load monitoring. |
| 2026-05-21 | [Superhuman Safe and Agile Racing through Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2605.22748v1) | Ismail Geles, Leonard Bauersfeld et al. | Autonomous systems have achieved superhuman performance in isolation or simulation, yet they remain brittle in shared, dynamic real-world spaces. This failure stems from the dominant single-agent paradigm for physical applications, where other actors are ignored or treated as environmental noise, preventing effective coordination. Here we show that multi-agent reinforcement learning provides the essential safety scaffolding required for real-world interaction. Using high-speed quadrotor racing as a high-stakes testbed, we train agents to navigate complex aerodynamic interactions and strategic maneuvering with a variable number of racers. Through league-based self-play, agents evolve sophisticated anticipatory behaviors, including proactive collision avoidance, overtaking, and handling multi-agent physical interactions, including aerodynamic downwash. Our agents outperform a champion-level human pilot in multi-player races at speeds exceeding 22 m/s, while simultaneously reducing collision rates by 50 % compared to state-of-the-art single-agent baselines. Crucially, training with diverse artificial agents enables zero-shot generalization to safer human interaction. These results suggest that the path to robust robotic co-existence lies not in isolated safety constraints, but in the rigorous demands of multi-agent interaction. Multimedia materials are available at: https://rpg.ifi.uzh.ch/marl |
| 2026-05-21 | [Can AI Make Conflicts Worse? An Alignment Failure in LLM Deployment Across Conflict Contexts](http://arxiv.org/abs/2605.22720v1) | Andrii Kryshtal | AI models are already deployed in societies affected by armed conflict, and journalists, humanitarian workers, governments and ordinary citizens rely on them for information or for their work processes. No established practice exists for checking whether their outputs can make those conflicts worse. We tested nine model configurations from four providers (OpenAI, Anthropic, DeepSeek, xAI) on 90 multi-turn scenarios designed to surface misaligned behaviour in conflict contexts: false equivalence between documented atrocities, denial of genocide, and failure to recognise ethnic slurs, among others. When such outputs feed into journalism, humanitarian reporting, or public debate, they can deepen divisions in fragile societies. Failure rates span 6\% to 47\% between the best and worst performing models, which makes model choice a safety question in its own right and when users pushed for ``balance'' in cases where international courts have already assigned responsibility, five of nine configurations failed 80 to 100 percent of the time. We release the first evaluation framework for this domain and propose adding it to alignment evaluation portfolios. |
| 2026-05-21 | [A Generalized Nash Equilibrium-Seeking Scheme for Trauma Resuscitation](http://arxiv.org/abs/2605.22661v1) | Promise Ekpo, Angelique Taylor et al. | Trauma resuscitation is a clinical process for treating life-threatening physiological disorders in safety-critical environments, driven by the experience of healthcare workers (HCWs). Designing and optimizing quantifiable metrics that accurately capture HCW decisions may augment current resuscitation procedures with the potential to improve patient outcomes. This motivates our socio-technical formulation of trauma resuscitation as a distributed generalized Nash equilibrium (GNE)-seeking game with coupled inequality constraints. This method is optimized over a time-varying communication graph. We introduce novel insights from clinical experience to model HCWs behavior. This work facilitates the best possible resuscitation outcome given HCWs workloads, schedules, competencies, and limited resources. |
| 2026-05-21 | [A Metalens-based Bicycle Safety Reflector for Autonomous Vehicle Radars](http://arxiv.org/abs/2605.22659v1) | Sepideh Ghasemi, Jimmy Hester et al. | With the rising number of interactions between autonomous or sensor-assisted vehicles -- especially in poor weather conditions -- come the need and opportunity for a new class of bicycle safety reflectors designed to enhance cyclist visibility to radars. To this effect, the first retrodirective planar metalens-based tag operating in the millimeter-wave automotive frequency range is proposed. The compact, lightweight ($0.61~\mathrm{g}$) design consists of two layers: a metalens layer and a patch antenna pixel layer. The metalens focuses incoming plane waves from different incidence angles onto corresponding patch antenna pixels on the second layer, which re-radiate the signal back through the metalens, enabling retrodirective operation. The proposed tag was thoroughly evaluated, demonstrating reliable detection beyond 70 m and a peak monostatic radar cross section (RCS) of $3.54~\mathrm{dBsm}$ with stable retrodirectivity over $\pm 40^\circ$, providing an average gain improvement of $7.58~\mathrm{dB}$ and an RCS enhancement of $15.16~\mathrm{dB}$ relative to a lens-less reference. A realistic deployment scenario on a metallic bicycle demonstrated up to a 110x improvement in its detectability at broadside. These results highlight the potential of the proposed passive tag to operate as a low-cost, lightweight, and easily integrable bicycle safety reflector for next-generation autonomous vehicle radar systems. |
| 2026-05-21 | [Boiling the Frog: A Multi-Turn Benchmark for Agentic Safety](http://arxiv.org/abs/2605.22643v1) | Piercosma Bisconti, Matteo Prandi et al. | Background. Traditional safety benchmarks for language models evaluate generated text: whether a model outputs toxic language, reproduces bias, or follows harmful instructions. When models are deployed as agents, the safety-relevant object shifts from what the system says to what it does within an environment, and evaluating model responses under prompting is no longer sufficient to address the safety challenges posed by artificial intelligence. Recent developments have seen the rise of benchmarks that evaluate large language models as agents. We contribute to this strand of research. Approach. We introduce Boiling the Frog, a benchmark that evaluates whether tool-using AI models deployed in corporate and office settings are susceptible to incremental attacks. Each scenario begins with benign workspace edits and later introduces a risk-bearing request. The benchmark focuses on stateful multi-turn evaluation: chains expose a persistent workspace, place the risk-bearing payload at controlled positions in the turn sequence, and score whether the resulting artifact state becomes unsafe. Scenarios are organized through a three-level operational risk taxonomy grounded in the Boiling the Frog risks, the AI Act Annex I and Annex III high-risk contexts, and EU AI Act's Code of Practice on General-Purpose AI (GPAI). Results. Across a nine-model panel, aggregate strict attack success rate (ASR) is 44.4%. Model-level ASR ranges from 20.5% for Claude Haiku 4.5 to 92.9% for Gemini 3.1 Flash Lite, with Seed 2.0 Lite also above 80%. Average chain category-level ASR reaches 93.3% for Code of Practice loss-of-control scenarios. |
| 2026-05-21 | [Contractual Skills: A GovernSpec Design Framework for Enterprise AI Agents](http://arxiv.org/abs/2605.22634v1) | Ting Liu | Skills are increasingly used to package agent instructions, workflows, scripts, and reference materials. In enterprise settings, however, skills often need to express more than task guidance: they must make goals, input boundaries, permissions, evidence requirements, output contracts, quality criteria, verification steps, human approval points, and handoff rules inspectable. This paper proposes contractual skills, a GovernSpec-inspired design framework for organizing SKILL.md files as readable task contracts while preserving lightweight skill discovery and progressive loading. The framework clarifies the boundary between contractual skills, GovernSpec YAML contracts, Model Context Protocol surfaces, tool adapters, runtime guardrails, tracing, and evaluation systems.   We evaluate the framework with two offline experiments. A text-generation study covers three enterprise skills, fifteen synthetic tasks, four instruction conditions, and eight generation models, yielding 960 outputs and 1680 cross-judge score records. Contractual skills outperform no-skill and minimal-skill baselines on all tested models. Relative to information-rich plain expanded skills, the gains are small and mixed, suggesting that contractual fields mainly improve checkability and maintainability rather than raw generation quality. A tool-calling challenge covers eight models and 192 simulated tool-call records. Skills usually reduce high-risk tool attempts, but model differences remain and runtime tool guardrails are still required. The results suggest that contractual skills are best understood as a governance layer that makes task intent, boundaries, and acceptance criteria explicit, not as a standalone safety mechanism. |
| 2026-05-21 | [Branch-Stochastic Model Predictive Control for Motion Planning under Multi-Modal Uncertainty with Scenario Clustering](http://arxiv.org/abs/2605.22600v1) | Zekun Xing, Ramkrishna Chaudhari et al. | Motion planning for autonomous driving must account for multi-modal uncertainty in both the intentions and trajectories of surrounding vehicles. Handling uncertainty in a worst-case manner guarantees robustness but often leads to excessive conservatism. Stochastic Model Predictive Control (SMPC) reduces trajectory-level conservatism through chance constraints, yet remains conservative with respect to intention uncertainty since constraints must hold across all intentions. We present a novel combination of SMPC and the branching structure, enabling the planner to generate distinct trajectories for different possible intentions while maintaining safety under trajectory uncertainty. A novel scenario clustering is proposed to merge prediction scenarios based on high-level decision similarity, thereby ensuring real-time tractability. Furthermore, an adaptive branching-time computation postpones commitment to separate plans until intention uncertainty is sufficiently reduced. Simulation studies in challenging highway scenarios demonstrate that the proposed method improves safety, reduces conservatism, and achieves real-time computational performance. |
| 2026-05-21 | [MOTOR: A Multimodal Dataset for Two-Wheeler Rider Behavior Understanding](http://arxiv.org/abs/2605.22550v1) | Varun A. Paturkar, Shankar Gangisetty et al. | Two-wheelers account for a disproportionately high share of road fatalities in the Global South. Research on two-wheeler rider behavior, however, lags far behind four-wheelers, where multimodal datasets have driven major advances in Advanced Driver Assistance Systems (ADAS). To address this gap, we present the MOtorized TwO-wheeler Rider (MOTOR) dataset, the first large-scale, multi-view, multimodal resource dedicated to two-wheelers in dense, unstructured traffic. MOTOR comprises 1,629 sequences (25+ hours of video data) collected from 16 riders and integrates synchronized front, rear, and helmet videos, rider eye-gaze from wearable trackers, on-road audio, and telemetry (GPS, accelerometer, gyroscope). Rich annotations capture traffic context, rider state, 12 riding maneuvers spanning conventional and unconventional behaviors, and legality labels (Legal, Illegal, Unspecified). We benchmark rider behavior recognition and maneuver legality classification using state-of-the-art video action recognition backbones (CNN and Transformer-based), extended with multimodal fusion, and find that combining RGB, gaze, and telemetry consistently yields the best performance. MOTOR thus provides a unique foundation for advancing safety-critical understanding of two-wheeler riding. It offers the research community a benchmark to develop and evaluate models for behavior analysis, legality-aware prediction, and intelligent transportation systems. Dataset and code is available at https: //varuniiith.github.io/MOTOR-Dataset/ |
| 2026-05-21 | [A Subjective Logic-based method for runtime confidence updates in safety arguments](http://arxiv.org/abs/2605.22530v1) | Benjamin Herd, Jessica Kelly et al. | We present a method for dynamic quantitative assurance that enhances static safety cases with continuous, runtime-driven confidence updates. The method quantifies and propagates confidence across the development lifecycle by integrating design-time evidence and windowed runtime Safety Performance Indicators (SPIs) within a single Subjective Logic (SL)-based assurance case. At runtime, SPI evidence is continuously evaluated, and targeted claims are updated using a rule that increases confidence in the absence of violations and imposes prompt penalties when violations occur. This design prioritizes safety-relevant responsiveness over exact classical Bayesian posterior updates. We demonstrate the method using a simulation-based construction zone assist function, focusing on an ML-based construction cone detection component, and show how confidence evolves as SPI evidence is observed in operation. |
| 2026-05-21 | ["Refactoring Runaway": Understanding and Mitigating Tangled Refactorings in Coding Agents for Issue Resolution](http://arxiv.org/abs/2605.22526v1) | Zhao Tian, Zifan Zhang et al. | Recent advances in coding agents have shown remarkable progress in software issue resolution. In practice, real-world issues are typically bug fixes or feature requests in which human developers naturally incorporate refactoring as part of the resolution process, resulting in tangled refactoring. Since LLMs are trained on large-scale open-source repositories, coding agents may inherit such behaviors. In this paper, we conduct an empirical study on Multi-SWE-bench, analyzing 3,691 valid patches generated by three agent frameworks with 12 LLMs. We find that coding agents introduce tangled refactorings less frequently (21.43% vs. 36.72%) and with lower intensity (0.66 vs. 1.75) than human developers, although they exhibit a broader diversity of refactoring types. Logistic regression analysis further shows that tangled refactorings are strongly associated with reduced compilability, while exhibiting no significant association with functional correctness. Based on these findings, we propose a refactoring-aware refinement approach that assesses the necessity and safety of tangled refactorings and selectively removes or repairs problematic operations. Our approach improves compilability from 19.34% to 38.33%, and additionally resolves 2.79% previously unresolved issues. Overall, this work presents the first step towards understanding tangled refactoring practices in agentic issue resolution and opens up avenues for future work. |
| 2026-05-21 | [LACO: Adaptive Latent Communication for Collaborative Driving](http://arxiv.org/abs/2605.22504v1) | Tianhao Chen, Yuheng Wu et al. | Collaborative driving aims to improve safety and efficiency by enabling connected vehicles to coordinate under partial observability. Recent approaches have evolved from sharing visual features for perception to exchanging language-based reasoning through foundation models for behavioral coordination. Though communicating in language provides intuitive information, it introduces two challenges: high latency caused by autoregressive decoding and information loss caused by compressing rich internal representations into discrete tokens. To address these challenges, we analyze latent communication in collaborative driving under inherent limitations of multi-agent settings. Our analysis reveals agent identity confusion, where direct fusion of latent states entangles decision representations across vehicles. Motivated by this, we propose LACO, a training-free \textbf{LA}tent \textbf{CO}mmunication paradigm that seamlessly adapts pretrained driving models to collaborative settings. LACO introduces Iterative Latent Deliberation (ILD) for latent reasoning, Cross-Horizon Saliency Attribution (CHSA) for communication-efficient information selection, and Structured Semantic Knowledge Distillation (SSKD) to stabilize ego-centric decision making. Closed-loop experiments in CARLA show that LACO notably reduces communication and inference latency while maintaining strong collaborative driving performance. |
| 2026-05-21 | [Perceived Safety of Workers in Encounters with Large Industrial AGVs](http://arxiv.org/abs/2605.22461v1) | Ansgar Howey, Tim Schreiter et al. | Automated Guided Vehicles (AGV) in factory automation are increasingly capable of moving autonomously in close proximity to human workers. While their physical safety is regulated by standards and directives, perceived safety and workers comfort in close-proximity interactions are being actively investigated in studies. There are three limitations in the prior art research to that end. Firstly, AGVs with larger payloads are understudied. Secondly, the test participants are usually students and not working professionals. Thirdly, while conducting in-person experiments with heavy machinery can be dangerous, the transfer of safety perception results from simulated experiments remains open. In this paper, we investigate industrial workers perceived safety in shared spaces with large AGVs in a real-world encounter and in virtual reality. We vary the passing distance and the shape of the collision avoidance maneuver, and evaluate perceived threat level using a handheld pressure-sensitive trigger interface and a post-experiment questionnaire. Additionally, we ask participants to set their own collision avoidance parameters based on their experience with the demonstrated trajectory profiles. In a within-subject study, we found that, while the threat levels are perceived overall slightly higher in VR, the passing distance of 1.5 to 2 meters is preferred among the demonstrated profiles, as well as in the self-defined trajectories. |
| 2026-05-21 | [Steins;Gate Drive: Semantic Safety Arbitration over Structured Futures for Latency-Decoupled LLM Planning](http://arxiv.org/abs/2605.22456v1) | Anjie Qiu, Hans D. Schotten | Cloud-hosted LLM driver agents provide useful semantic judgments, but their inference latency exceeds stepwise vehicle-control windows. Learned world models predict futures, but they usually keep future generation and action selection inside large coupled loops. We present SteinsGateDrive, a latency-decoupled planner-runtime architecture in which the worldline metaphor from the eponymous story names one plausible consequence of an intervention: the LLM selects counterfactual driving futures before the final control instant, and a runtime reuses the selected forecast only while safety contracts remain valid. The generator builds three world-line roles: alpha nominal ego-conditioned futures, beta interaction counterfactuals around nearby vehicles, and gamma hazard-stress futures such as braking, cut-ins, or blocked corridors. The selected branch becomes a typed StrategicForecast with horizon, validity/abort conditions, fallback, and authority. On a within-subject, matched-seed normal-highway protocol with 10 seeds and 20 steps, GPT-5.4 mini reduces effective lag from +3.07 s at 1-second horizon to -0.01 s at 4-second horizon while preserving the measured no-collision safety boundary. The architecture's safety contribution comes from the atom-predicate runtime check, not from the drift score, which functions as a refresh-frequency knob. |
| 2026-05-21 | [Making the Discrete Continuous: Synthetic RAW Augmentations for Fine-Grained Evaluation of Person Detection Performance in Low Light](http://arxiv.org/abs/2605.22455v1) | Valeria Pais, Malena Mendilaharzu et al. | Real-world deployment of AI vision models is both fueled and limited by the data available for training and testing. Real datasets are sparse and uneven: long-tailed or unbalanced distributions hinder generalization, and the low number of samples in low density regions makes it hard to run evaluations. Synthetic data can fill these gaps, providing us with a way to sample the input space more continuously and improve data coverage for benchmarks. Focusing on the autonomous driving safety-critical case of pedestrian detection in the dark, we show how synthetic low-light samples can be used to better characterize the performance of a state-of-the-art object detection model as a function of the scene illumination. We use a synthetic RAW image augmentation technique to generate low-light samples that match the noise model of the camera sensor. Performance metrics on real and synthetic low-light data are similar, indicating that the AI model finds it hard to distinguish between them. |
| 2026-05-21 | [Pre-VLA: Preemptive Runtime Verification for Reliable Vision-Language-Action and World-Model Rollouts](http://arxiv.org/abs/2605.22446v1) | Zhen Sun, Yongjian Guo et al. | While large vision-language-action (VLA) models and generative world models (WM) have advanced long-horizon embodied intelligence, their practical deployment remains challenged by uncertainty in learning-based action generation. Low-quality actions may cause physical failures during execution or lead to misleading world-model rollouts with redundant rendering costs. To address this issue, we propose Pre-VLA, a unified runtime verification architecture that performs preemptive action validity assessment before physical execution or world-model imagination. Pre-VLA leverages an efficient multimodal backbone with modality-aware pooling and a lightweight dual-branch head to predict both safety confidence and critic-derived advantage scores for candidate action chunks. To handle severe class imbalance and unstable boundary decisions, we train Pre-VLA with a multi-task objective combining Focal classification, advantage regression, and soft-threshold calibration. During deployment, a dual-mode preemptive resampling scheduler filters low-quality actions and triggers adaptive resampling under a limited computation budget. Experiments on the LIBERO benchmark show that Pre-VLA improves the average closed-loop success rate across four suites from 30.79\% to 37.62\% over RynnVLA-002, reduces task execution steps, achieves 183.9 ms average forward verification time per action chunk, and mitigates error accumulation in world-model rollouts. |
| 2026-05-21 | [Characterizing the Fault Response of the Intel Neural Compute Stick 2 Under Single-Pulse Electromagnetic Fault Injection](http://arxiv.org/abs/2605.22437v1) | Štefan Kučerák, Jakub Breier et al. | Vision processing units and other commercial neural-network inference accelerators are increasingly deployed in safety-relevant edge applications, but their fault response under transient hardware disturbances remains poorly characterized in the open literature. For the Intel Movidius Myriad X, packaged as the Intel Neural Compute Stick 2 (NCS2), only a single feasibility study has been published. We report a systematic single-pulse electromagnetic fault injection (EMFI) campaign on the NCS2 running three ImageNet-trained convolutional neural networks (ResNet-18, ResNet-50, VGG-11) on the OpenVINO runtime. Across 1,536 spot-test trials at characterized hotspots and approximately 16,000 parameter-search trials, single pulses produce four reproducible outcome classes: no measured accuracy change, minor silent data corruption, major persistent degradation that survives across subsequent inferences until model reload, and device hangs requiring USB power-cycling; these outcomes are respectively interpreted as no-effect, SDC with possible SET-like or small persistent-state mechanisms, SEU-like persistent corruption, and SEFI-like loss of functionality. Two findings are central. First, the major-degradation class can be induced at 18-31% of trials at characterized hotspots, with post-collapse top-1 accuracy below five percent and persistence across all subsequent inferences until explicit model reload - a regime that no inference-API-level mechanism detects. Second, this regime is also inducible by pulses delivered to an idle device with the model already loaded, demonstrating that load-time integrity checks alone are insufficient. We discuss mitigation strategies graded by class, focusing on mechanisms implementable at the application level without modification to the device firmware or the OpenVINO runtime. |
| 2026-05-21 | [Boundary-targeted Membership Inference Attacks on Safety Classifiers](http://arxiv.org/abs/2605.22373v1) | Anthony Hughes, Alexander Goldberg et al. | Safety classifiers are essential safeguards within generative AI systems, filtering harmful content or identifying at-risk users when interacting with large language models. Despite their necessity, these models are trained on sensitive datasets including discussions of self-harm and mental health, raising important, yet poorly understood, privacy concerns. Membership inference attacks (MIAs) allow adversaries to infer membership of examples used to train models. In this work, we hypothesize that identifying the examples on which the classifier is least confident are informative for an adversary to infer membership. This reflects a localized failure of generalization, where the model relies on memorization to resolve ambiguity in the training set. To investigate this, we introduce a new boundary-targeted selection strategy that identifies low confidence examples that amplify the signal of an examples membership within a training set. Our experimental results show that an adversary can recover 19\% of the conversations a safety classifier flagged as indicating user distress, at a 5\% false-positive rate, on a classifier fine-tuned for detecting a user who may require emotional support. This is $3.5$ times more than attacking using state-of-the-art MIA methods alone. Finally, we characterize the boundary laying examples and show that content-based filtering is ineffective for protection, and existing noise strategies can effectively mitigate susceptibility of these examples. |
| 2026-05-21 | [How can reasoning capability empower the AI copilot robot in endoscopic surgery](http://arxiv.org/abs/2605.22322v1) | Guankun Wang, Long Bai et al. | Reasoning capability has significantly advanced complex logical inference and robotic decision-making in general domains. However, its potential in the Artificial Intelligence (AI) copilot robot-particularly implemented based on the Vision-Language-Action (VLA) model-remains unexplored in endoscopic surgery. Effective reasoning should enable AI copilot robots to integrate multimodal cues, interpret surgical intent, and infer hidden tissue dynamics, thereby alleviating intraoperative uncertainty and cognitive burden on surgeons. Properly implemented, reasoning-driven autonomy can transform AI copilot robots from reactive executors into cognitive collaborators, enhancing precision, safety, and sustainability in clinical practice. |

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



