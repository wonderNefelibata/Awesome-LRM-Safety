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
| 2026-07-14 | [TerraZero: Procedural Driving Simulation for Zero-Demonstration Self-Play at Scale](http://arxiv.org/abs/2607.13028v1) | Zhouchonghao Wu, Akshay Rangesh et al. | Training robust autonomous driving agents requires a simulator that is fast enough for reinforcement learning at scale, realistic enough to ground behavior in real-world map structure, and diverse enough to cover the safety-critical long tail that logged data rarely contains. We present TerraZero, a procedural driving simulator and self-play training stack. A configurable C engine runs simulation on the CPU and policy inference on the GPU over a zero-copy path, sustaining 1.3M agent-steps per second on a single server-grade GPU, far faster than existing object-level simulators, while keeping fidelity lighter single-agent systems omit: heterogeneous agents, multiple dynamics models, and full traffic-rule enforcement. TerraZero treats logged data only as a source of real-world map geometry, populating each map with randomized rule-based road users and signal controllers and randomizing agent dynamics, rewards, and sizes per episode, so a map yields an unbounded set of scenarios. Every reported policy trains from scratch by reinforcement learning alone on a compute-efficient self-play recipe across GPUs, with zero human demonstrations and no fallback planner at inference. Policies generalize zero-shot across cities and datasets, including emergent left-hand-traffic driving without explicit supervision. As an ego policy, TerraZero is the first fully learned policy to top the InterPlan long-tail benchmark, ahead of larger learned planners; on routine-driving val14 it ranks among the best approaches and is the safest, posting the best collision and time-to-collision scores. On Waymo Open Sim Agents realism the same recipe outperforms other demonstration-free methods and is competitive with the strongest reference-anchored self-play method. One stack serves both roles: driving policies across dynamics for cars and trucks, and sim agents that jointly control vehicles, pedestrians, and cyclists. |
| 2026-07-14 | [Knowledge- and Gradient-Guided Reinforcement Learning for Parametrized Action Markov Decision Processes](http://arxiv.org/abs/2607.12924v1) | Jonas Ehrhardt, René Heesch et al. | In this paper, we study Reinforcement Learning in Parametrized Action Markov Decision Processes (PAMDP), where each decision consists of a symbolic action and numerical parameters. In such settings Reinforcement Learning algorithms typically determine parameters with one-shot estimators, which makes their training sample inefficient. Though in most PAMDP environments explicit but incomplete knowledge (e.g., rules, safety constraints, or expert heuristics) is available, it is rarely directly used to increase the sample-efficiency of training Reinforcement Learning agents. We step into this gap and propose our novel Neuro-Symbolic Knowledge- and Gradient-Guided Reinforcement Learning (KGRL) algorithm. KGRL uses domain knowledge in a Datalog knowledge base to derive the set of applicable actions and feasible parameters for a given state. This allows it to prune non-applicable actions from the decision-space and constrain the parameter spaces of the remaining actions. We then use a gradient-based parameter refinement loop to estimate the optimal parameters during training and deployment of the agent. By recording activated rules along the trajectory, KGRL additionally provides local procedural explanations on the pruning of actions and constraining of parameters. Overall, KGRL guides the agent's exploration and deployment toward feasible and constraint-aware decisions, while increasing sample efficiency during training. KGRL outperforms state-of-the-art RL baselines for PAMDPs in both, sample efficiency and episodic return. |
| 2026-07-14 | [Astra: an open-source fully autonomous robotic observatory control software](http://arxiv.org/abs/2607.12898v1) | Peter P. Pedersen, David Degen et al. | Robotic and autonomous observatories are critical for modern time-domain and high-cadence astronomical surveys. The operation of these facilities requires complex software coordination to manage hardware, schedule observations, and ensure safety. However, existing observatory control software are often proprietary and platform-locked or require complex message-brokering infrastructure. Here we present Astra (Automated Survey observaTory Robotised with Alpaca): an open-source, cross-platform Python system for the sustained, fully autonomous operation of astronomical observatories, requiring no external message-broker infrastructure. Astra controls observatory hardware via the ASCOM Alpaca protocol, and executes prescheduled observatory actions under continuous safety supervision. Its multi-device actions include plate-solve-based pointing correction with a local Gaia--2MASS catalogue fallback, PID-controlled autoguiding, and autofocus. A FastAPI web service provides a browser UI, REST and WebSocket APIs for real-time status, image previews, and SQLite-backed telemetry and logs. Astra has run in fully unattended production since January 2024, scaling to six telescopes across three facilities: the SPECULOOS-South network (4 $\times$ 1\,m class, Chile), SAINT-EX (1\,m class, Mexico), and the ETH Observatory (0.5\,m class, Switzerland), with no schedule aborts attributable to Astra software. Across the SPECULOOS-South network, it achieves sub-arcsecond autoguiding (0.11\unit{\arcsecond} median pointing scatter) and plate-solve failure rates below 1\% on three of the four telescopes (3\% on the narrowest-field unit), demonstrating that an open, standards-based software stack can meet the reliability demands of production survey astronomy. |
| 2026-07-14 | [Superstatistical Analysis of PDFs and autocorrelation functions for air pollution concentrations in the UK](http://arxiv.org/abs/2607.12876v1) | Nisal Amarakoon, Hankun He et al. | Conventional statistical models often struggle to fully capture the complex spatio-temporal dynamics, intermittent fluctuations, and heavy-tailed distributions characteristic of real-world air pollution data. Furthermore, existing literature frequently focuses on extreme events, overlooking the persistence of low-pollution states and temporal memory effects. To address these gaps, we apply superstatistical frameworks from non-equilibrium statistical physics to analyse a comprehensive five-year dataset (2020-2025) of hourly air pollutant concentrations across the United Kingdom. Excellent fits of experimentally measured distributions are obtained from our theoretical models. We observe large heterogeneities of the best fitting parameters depending on the locations where the measurements are performed. These parameters form characteristic patterns in the 3-dimensional parameter space and depend on the type of pollutant considered, as well as on the environmental conditions (high traffic, industrial, or rural surroundings). We also investigate autocorrelation functions and provide evidence for differences in day-time and night-time decays of the autocorrelation function. Our investigation mainly focuses onto the dynamics of NO, NO2, PM2.5, PM10, but we also report on some anomalous distributions observed for O3. |
| 2026-07-14 | [Breaking Déjà Vu: Independent Auditing of Visual Place Recognition through Vision-Language Reasoning](http://arxiv.org/abs/2607.12818v1) | Sania Waheed, Michael Milford et al. | Visual place recognition (VPR) is a key enabler of accurate localization and long-term autonomous navigation in robotics applications, such as loop closure detection for simultaneous localisation and mapping (SLAM). However, real-world VPR deployment relies on selecting an image matching threshold that balances precision and recall. These thresholds are typically tuned using labeled validation data and fixed during deployment, making them unreliable under environmental changes where ground truth is unavailable. This is particularly problematic in safety-critical robotics, where accepting a false loop closure can corrupt the estimated trajectory and map. In this work, we introduce Visual Place Recognition Auditing, an independent post-retrieval verification framework that leverages Vision-Language Models (VLMs) to assess retrieved matches by reasoning jointly over query and candidate images. Unlike conventional verification methods, our approach performs instance-level verification without requiring architecture-specific confidence measures, dataset-dependent thresholds, or prior knowledge of the deployment environment. We evaluate our method on six benchmark datasets using five state-of-the-art VPR methods and four VLMs. Results show that VLM-based auditing improves recall@1 by 13.6% on average as compared to state-of-the-art methods while reducing false acceptance rates to 12%, maintaining precision above 95% and coverage above 75%. |
| 2026-07-14 | [Silent Alarm: A J-Space Protocol for Comparing Danger Recognition Across Models and Quantization Levels](http://arxiv.org/abs/2607.12792v1) | Roman Prosvirnin, Victor Minchenkov et al. | Jailbreak-robustness research typically evaluates safety through generated responses using an LLM-as-judge approach. Such evaluations, however, are sensitive to the benchmark's grading procedure and capture only observed behavior on a given set of attacks, without directly revealing the hidden fragility of the underlying safety mechanisms. This work proposes JADR (Jacobian Assessment of Danger Recognition), a protocol that measures a model's internal representation through Jacobian space (J-space, a recently proposed workspace of verbalizable concepts) before the first response token is generated. For every prompt and layer we record the top-k J-space tokens; these are grouped into six behavioral scenario axes and compared between a danger sample based on StrongREJECT and a safe control drawn from XSTest and OKTest. The method does not call on an external judge model: the computation runs entirely locally, on the activations of the model under evaluation, which lets us compare both different models against each other and modifications of a single model -- quantization and fine-tuning in particular -- on the same terms. The final comparison rests on the proposed SafetyAUC metric, complemented with bootstrap confidence intervals. The protocol is applied to six models (Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-Uncensored-4B, Qwen3-SafeRL-4B, Gemma 2 9B) across three weight-representation regimes -- BF16, INT8, and INT4 -- and checked against an independent behavioral evaluation with the StrongREJECT grader. The metric separates models with a strong versus a weak internal safety mechanism with statistical significance and captures substantively different effects across quantization regimes. |
| 2026-07-14 | [Who Grades the Grader? Co-Evolving Evaluation Metrics and Skills for Self-Improving LLM Agents](http://arxiv.org/abs/2607.12790v1) | Xing Zhang, Guanghui Wang et al. | Self-evolving agent systems improve by creating, revising, and retiring their own skills, but every such loop rests on a hidden assumption: a reliable evaluation metric already exists. In many real applications it does not. We make three claims. First, metrics can be \emph{evolved}: our metric loop searches compositions of small drawback detectors under a full evolutionary lifecycle, trained to agree with a ten-item anchored reference set, regularized by consensus over unlabeled outputs, and audited against a held-out anchor it never reads, yielding a transparent, inspectable metric rather than an opaque judge. Second, since no metric exists to beat, the yardstick is recovering what an accurate metric would have enabled, and \emph{Double Ratchet}, our co-evolution of the metric with a lifecycle-managed skill loop, does so: across code generation (MBPP+), enterprise text-to-SQL (Spider~2.0-Snow), and reference-free report generation, it retains 88--110\% of the held-out lift achieved by the same skill loop driven by ground truth or the best available rubric. Third, safety comes from anchor discipline plus outer audits: removing anchor guards collapses the metric into a vacuous detector while removing the lifecycle does not; and when evolved skills gamed the report rubric, an independent judge caught it, one detector repaired it, and a task-aware judge then preferred the evolved outputs over the pre-evolution baseline in 77\% of decided pairs. We argue this failure-expecting architecture is the right default wherever no reliable automatic verifier exists. |
| 2026-07-14 | [ExtraGS: Enhancing Endoscopic View Extrapolation via Diffusion-Guided 3D Gaussian Splatting](http://arxiv.org/abs/2607.12785v1) | Cheng-Tai Hsieh, Jiwei Shan et al. | Robot-assisted minimally invasive surgery (MIS) critically depends on reliable endoscopic perception for navigation and safety. However, conventional endoscopes provide only a limited field of view, leaving large portions of surrounding anatomy unobserved. Recent neural rendering approaches, such as Neural Radiance Fields and 3D Gaussian Splatting, enable novel view synthesis from endoscopic videos, but their reliance on sparse observations often leads to severe artifacts when extrapolating beyond the training trajectory.In this work, we propose ExtraGS, a framework for enhancing endoscopic view extrapolation via diffusion-guided 3D Gaussian Splatting. Starting from an initial reconstruction, we introduce an uncertainty-guided virtual camera sampling strategy to actively explore blind spots and maximize information gain. The rendered views from these sampled locations are refined using a diffusion model to recover plausible anatomical structures, producing pseudo observations that guide further optimization. To prevent the generated content from degrading reliable regions, we adopt a confidence-weighted fine-tuning strategy when incorporating these pseudo observations.Extensive experiments on multiple public endoscopic datasets demonstrate that ExtraGS significantly reduces extrapolation artifacts and achieves state-of-the-art performance in endoscopic novel view synthesis. |
| 2026-07-14 | [Directional Constraints for Efficient Exploration in Safe Reinforcement Learning](http://arxiv.org/abs/2607.12784v1) | Paolo Magliano, Puze Liu et al. | Reinforcement Learning has revolutionized the landscape of robotic research, allowing robust learning of complex robotic skills in simulation. However, real-world deployment in open-ended environments requires strong safety guarantees to prevent dangerous or harmful behaviors. Safe Reinforcement Learning methods address this requirement by enforcing safety constraints. Nevertheless, learning under constraints often reduces learning speed and could lead to suboptimal task performance, as the agent must solve a more complex constrained optimization problem compared to unconstrained settings. To tackle this issue, in this work, we propose an extension of the ATACOM framework, a state-of-the-art reliable safety layer that can be integrated with existing Reinforcement Learning algorithms to enforce constraints derived from prior knowledge of the system or learned directly from data. Our proposed method, named ATACOM Directional Constraints (ATACOM-DC), significantly improves the safety-performance trade-off by introducing directional constraints that distinguish between actions approaching and moving away from constraint boundaries, activating constraint enforcement only when necessary. We evaluate our method across a range of challenging robotic control tasks in simulation, analyzing both constraint-violation costs and achieved task performance. Code and additional material at https://atacom-dc.robot-learning.net. |
| 2026-07-14 | [Constraint-Aware Aggregation for Federated Reinforcement Learning in Microgrid Energy Coordination](http://arxiv.org/abs/2607.12763v1) | Usman Haider, Karl Mason | Federated Reinforcement Learning (FedRL) enables coordination of distributed energy resources without sharing raw local data, but standard aggregation methods such as FedAvg do not account for system-level constraints, often leading to unsafe global behavior. In this work, we study constraint-aware aggregation for federated reinforcement learning in distributed energy coordination. We propose aggregation rules that incorporate both local performance and estimated constraint violation into the server-side update. Among these, a simple penalty-based rule, $w_i \propto R_i - αV_i$, consistently provides the most reliable trade-off between reward and safety, without requiring dual optimization or modifications to local training. \textcolor{black}{We evaluate our approach on DairyGridEnv, a benchmark modeling multiple farms coordinating battery storage under stochastic demand and a shared grid capacity constraint, and further assess robustness using real load-driven demand profiles from Finland and the German FIELD dataset. Across multiple seeds, penalty-based aggregation substantially reduces violations while improving reward relative to FedAvg in both synthetic and real load-driven settings.} A combined reward-violation scheme exposes a tunable trade-off via $λ$, but is less stable. These results demonstrate that lightweight aggregation strategies can substantially improve empirical safety in federated reinforcement learning while preserving standard communication protocols. |
| 2026-07-14 | [Epistemic Stance Flexibility Probing: Measuring Prompt-Conditioned Register Shift in Large Language Models](http://arxiv.org/abs/2607.12739v1) | Binwen Liu, Yilin Ren | A language model may be asked either what experts believe about a contested claim or what it believes about the claim itself. A trustworthy conversational agent should distinguish these two requests and respond in different epistemic registers: neutral attribution in the first case and stance expression in the second. Whether such a shift occurs-and whether it occurs coherently-is not directly assessed by existing benchmarks for accuracy, instruction following, or safety. We introduce ESFP, a behavioral benchmark that treats the contrast between externally attributed and self-attributed prompts as the fundamental unit of measurement. ESFP consists of 104 carefully controlled items spanning six epistemic categories and five phrasing templates, and evaluates model responses along four complementary dimensions: lexical self-attribution, representation-level responsiveness to role framing, sentence-level stance content density assessed by an LLM judge panel, and cross-condition stance consistency. Evaluating eight frontier models from five vendors, we find that epistemic flexibility is largely orthogonal to general model capability: a 27B open-weight model matches the strongest proprietary systems, the flagship model of one family underperforms its lightweight counterpart, and reasoning-optimized models do not consistently exhibit higher flexibility. Stance content density provides the strongest signal, while surface-level lexical markers such as 'I think' can change substantially without corresponding changes in expressed stance. We provide item-level bootstrap confidence intervals, weight-sensitivity analyses, and an explicit discussion of the interpretation limits of the composite score. ESFP measures a model's propensity to adapt its epistemic stance under changing attribution conditions, rather than a general competence measure. |
| 2026-07-14 | [On the transition to large fluxes and access to second stability in gyrokinetic simulations of electromagnetic turbulence in STEP](http://arxiv.org/abs/2607.12682v1) | Daniel Kennedy, Yujia Zhang et al. | This work investigates the nonlinear transition to large heat fluxes observed in local gyrokinetic simulations of electromagnetic turbulence in STEP. Using the stress-balance framework of Zhang et al. (arXiv:2606.04616, arXiv:2607.11789), we confirm that the onset of extreme transport correlates with a critical value of $q^{2}β_{e}$, where $q$ is the safety factor and $β_{e}$ is the ratio of electron thermal pressure to magnetic pressure, and relate this to a limit on the poloidal beta $β_{\mathrm{pol}}$. Crucially, this critical value lies below any relevant linear stability limit in the ($q$, $β_{e}$) space (e.g., the onset of ideal or kinetic ballooning modes). Using an extensive set of nonlinear gyrokinetic simulations, we demonstrate that the transition to large fluxes in STEP is governed by a balance between the electrostatic and magnetic-flutter stresses. We argue, and also show numerically, that larger-major-radius tokamaks reach the electromagnetic non-zonal regime at lower $β_{e}$, making this MHD-controlled saturation limit more accessible in reactor-scale devices than in small spherical tokamaks. We also demonstrate that access to a second-stable regime enables re-saturation at larger values of $β^{\prime}$. We further show that the ideal ballooning mode (IBM) threshold serves as a useful proxy for delineating this second-stable region and also as a qualitative guide for the onset of large fluxes. These results provide a predictive framework for identifying no-go zone predictions from local gyrokinetics and offer new insight into the electromagnetic saturation physics relevant to STEP and other high-$β_{e}$ devices. |
| 2026-07-14 | [Internet of Agentic Things: Networked AI Agents for Closed-Loop IoT Orchestration](http://arxiv.org/abs/2607.12662v1) | Quanyan Zhu | The paper introduces the Internet of Agentic Things (IoAT), an architectural framework that integrates agentic AI, IoT, cyber-physical systems, Physical AI, edge computing, and digital twins into a unified closed-loop orchestration framework. The proposed architecture consists of cloud, edge/fog, and physical IoT layers connected through autonomous AI agents that perceive, reason, coordinate, and actuate across distributed cyber-physical environments. The paper formalizes IoAT as a coupled workflow-control problem with nested strategic and tactical decision making using a hylomorphic dynamic programming framework that links agentic planning with physical execution. Smart-building orchestration is presented as a representative use case, and key research challenges related to safety, security, governance, resilience, and trustworthy deployment are discussed. |
| 2026-07-14 | [PVDetector: Detecting Prompt Injection Attacks on Purpose-Specific LLM Agents through Policy-Violation Concept Analysis](http://arxiv.org/abs/2607.12624v1) | Junhui Wang, Hangtao Zhang et al. | Large language models (LLMs) are increasingly deployed as purpose-specific agents to handle domain-specific tasks such as customer service and code generation. These agents are expected to comply with not only generic safety guardrails but also purpose-specific restrictions tailored to their designated roles. Such additional restrictions enlarge the attack surface, particularly to prompt injection (PI) attacks. To defend against such attacks, existing detection methods primarily rely on analyzing input-output patterns, yet yield limited effectiveness. To address this limitation, we turn to analyzing the hidden activation space and discover that LLMs inherently retain latent policy-violation (PV) concepts when prompted with requests beyond their designated purpose. Particularly, PV concepts capture the semantics of conflicts between user queries and predefined restrictions, implicitly reflecting LLMs' intrinsic awareness of recognizing policy violations. Building on this insight, we propose PVDetector, a training-free framework that detects PI attacks during LLM inference by measuring hidden-state alignment with PV concepts, which are derived offline from the contrastive pairs of policy-violating and policy-compliant prompts. Experiments across multiple LLMs and datasets show that PVDetector achieves <1\% false negative rate with minimal auxiliary overhead, consistently outperforming state-of-the-art methods. Our code is available at https://github.com/Claresigle/PVDetector . |
| 2026-07-14 | [Streamlining stereo differentiable rendering for marker-free real-time tracking of surgical robots](http://arxiv.org/abs/2607.12604v1) | Yanghe Hao, Martin Huber et al. | Purpose: Marker-based tracking of surgical robots is occlusion-prone in cluttered operating rooms. We evaluate stereo differentiable rendering for marker-free, real-time robot pose tracking, potentially improving safety, reducing setup time, and enabling multi-robot interaction. Methods: We extend the markerless pose estimation framework roboreg to online dynamic tracking via (i) sequential optimisation that propagates pose estimates across frames with motion-adaptive hyperparameter tuning, and (ii) CUDA stream parallelisation of segmentation and optimisation, combined with CUDA-graph accelerated segmentation. We evaluate on 38 unobstructed and 5 occluded displacement sequences with static start/end ground-truth calibrations and dynamic marker-based reference tracking. Results: We achieve real-time 1080p tracking at 30 fps (up from 14 fps for vanilla roboreg), matching the camera frame rate. Accuracy reaches 1.7 cm / 0.6 deg against static ground truth and 1.2 cm mean 3D error over 27,460 frames against the marker-based reference (1.53 cm over 1,242 occluded frames). Our method outperforms FoundationPose by 11% in dynamic estimation (63% under occlusion) and 250% in static estimation, with 6x faster inference. Conclusions: Stereo differentiable rendering enables real-time, high-resolution marker-free surgical robot tracking, on par with marker-based approaches and surpassing foundation-model baselines. |
| 2026-07-14 | [Hierarchical Fault Localization for Autonomous Driving Systems with Hypothesis Validation and Intent Analysis](http://arxiv.org/abs/2607.12598v1) | Rui Zheng, Changwen Li et al. | Comprehensive testing is essential for the safety and reliability of Autonomous Driving Systems (ADS). Existing techniques can detect system-level failures or attribute them to coarse-grained modules, but they often fall short of localizing the root cause in source code. As a result, debugging remains labor-intensive, requiring developers to connect behavioral violations with complex implementation logic. To address this gap, we present HINT, a two-phase framework for hierarchical ADS fault localization based on hypothesis validation and intent analysis. In Phase I, HINT transforms failure-triggering execution recordings into multi-modal abstractions and uses causal reasoning to identify the responsible module. In Phase II, it reconstructs design-side intent and implementation-side behavior, then localizes suspicious code through reliability-aware consistency checking, without costly re-simulation. We evaluate HINT on Apollo across diverse failure modes and modules. The results show that HINT achieves the strongest overall performance across module-level diagnosis and code-level localization metrics, with 77.8% end-to-end Class@5 accuracy on real-world bugs. |
| 2026-07-14 | [Agent-Safety Evaluations as Load-Bearing Evidence: A Vendor-Neutral, Cross-Harness Reconstructability Metric](http://arxiv.org/abs/2607.12469v1) | Oleg Solozobov | Many agent-safety evaluation results are not yet load-bearing evidence: identical nominal outcomes (task success, attack success, monitor scores) may sit atop materially different evidence regimes. No vendor-neutral, runnable instrument scores reconstructability as an evaluation-validity metric: whether captured evidence can reconstruct the decision a claim depends on. This paper introduces a property-level reconstructability metric over eight decision-property classes and a cross-harness adapter emitting per-decision Evidence Sufficiency Cards backing a per-run monitor-coverage release check. It specifies a counterfactual-replay intervention protocol, implements its replayability-precondition probe, and defines a claim-evidence overclaim gap. On public and bundled traces, without new model runs, twelve-field sufficiency spans 0.458-0.833 across four inputs sharing a surface reading; replay preconditions are unmet in every scored trace. In a synthetic release-gate pair, the sufficiency gate blocks the raw variant (0.542) and passes the instrumented (0.667). Safety-evaluation claims should travel with their reconstructability vector; a reproducibility package regenerates every reported number. |
| 2026-07-14 | [Exploring Zero-Shot Foundation Models for Multivariate Time Series Anomaly Detection](http://arxiv.org/abs/2607.12454v1) | Martin Uray, Saverio Messineo et al. | Multivariate Time Series Anomaly Detection (MTSAD) is essential for reliability and safety in domains such as industrial process monitoring and financial risk management, yet conventional approaches rely on application-specific models that are costly to train and hard to scale. Foundation Models (FMs), pre-trained on broad data with strong zero-shot generalization, have recently become available for univariate time series forecasting, raising the question of whether they can address MTSAD without task-specific training. We investigate the zero-shot application of a univariate forecasting FM, TimesFM, to industrial MTSAD on the Secure Water Treatment (SWaT) benchmark, evaluating two strategies: treating the FM as a per-feature forecaster with thresholded prediction errors, and as an embedder whose intermediate representations feed standard outlier detectors. Neither of our proposed setups is competitive with established baselines; embeddings reveal only partial separation between normal and anomalous segments, insufficient for reliable detection. The cause is that the FM is too effective at capturing temporal dynamics, yielding low error even within fully anomalous windows, so persistent anomalies become indistinguishable from normal behavior. However, these observations yield valuable insights: the error peaks at anomaly boundaries, indicating FMs reliably detect distribution changes. We conclude that the proposed naive zero-shot FMs are unsuitable for MTSAD but promising for change-point detection. |
| 2026-07-14 | [Automatic Testing of Interacting Autonomous Vehicles](http://arxiv.org/abs/2607.12452v1) | Fabio Cavaleri, Alessio Gambi et al. | Autonomous vehicles (AVs) must be thoroughly tested to meet high safety standards and avoid endangering both AV passengers and road users. Scenario-based testing implements driving scenarios in virtual simulation environments as a cost-effective alternative to field testing. Common scenario-based testing approaches set the environment and the surrounding traffic and test a single AV. Recent studies show that the approaches that test single AVs miss critical behaviors that emerge from interactions among multiple AVs. Effective approaches to test scenarios that emerge from n-way interactions must address the combinatorial explosion that the presence of multiple AVs further exacerbates. In this paper, we propose EVITA, an approach that leverages multi-objective optimization to generate scenarios that trigger multiple and diverse AVs interactions, while minimizing the complexity of the generated scenarios, to effectively test multiple interacting AVs and reveal safety-critical scenarios that current approaches overlook. The experimental results that we discuss in this paper confirm that EVITA triggers a higher variety of AVs interactions than state-of-the-art approaches, thus improving the likelihood to reveal safety-critical behaviors. |
| 2026-07-14 | [Model-Based Diffusion Optimal Control for Multi-Robot Motion Planning](http://arxiv.org/abs/2607.12423v1) | Zhilin He, Yorai Shaoul et al. | Multi-Robot Motion Planning in continuous environments, where robots must generate dynamically feasible, collision-free trajectories, is challenging due to the combinatorial growth of the joint trajectory space and the difficulty of enforcing dynamic feasibility and hard safety constraints. Recent approaches recast trajectory planning as probabilistic inference, sampling from a posterior over trajectories using diffusion models whose score functions are learned from demonstration data. While showing promising performance, these approaches are limited: they often rely on sizable demonstration datasets and struggle to rigorously enforce dynamics and hard safety constraints during sampling. To this end, we introduce Model-Based Diffusion Optimal Control (MDOC), a model-based diffusion planner that efficiently produces dynamically feasible trajectories without relying on data. Crucially, we show that MDOC's safety mechanism -- combining known dynamics models with Control Barrier Function-constrained projections -- naturally scales to multi-robot planning settings through Conflict-Based Search. Across simulation experiments, this integrated method consistently outperforms representative baseline planners in sample efficiency, geometric smoothness, and success rate, while reducing computation time and producing collision-free trajectories. |

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



