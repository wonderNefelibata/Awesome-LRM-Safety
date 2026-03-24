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
| 2026-03-23 | [Human-Inspired Pavlovian and Instrumental Learning for Autonomous Agent Navigation](http://arxiv.org/abs/2603.22170v1) | Jingfeng Shan, Francesco Guidi et al. | Autonomous agents operating in uncertain environments must balance fast responses with goal-directed planning. Classical MF RL often converges slowly and may induce unsafe exploration, whereas MB methods are computationally expensive and sensitive to model mismatch. This paper presents a human-inspired hybrid RL architecture integrating Pavlovian, Instrumental MF, and Instrumental MB components. Inspired by Pavlovian and Instrumental learning from neuroscience, the framework considers contextual radio cues, here intended as georeferenced environmental features acting as CS, to shape intrinsic value signals and bias decision-making. Learning is further modulated by internal motivational drives through a dedicated motivational signal. A Bayesian arbitration mechanism adaptively blends MF and MB estimates based on predicted reliability. Simulation results show that the hybrid approach accelerates learning, improves operational safety, and reduces navigation in high-uncertainty regions compared to standard RL baselines. Pavlovian conditioning promotes safer exploration and faster convergence, while arbitration enables a smooth transition from exploration to efficient, plan-driven exploitation. Overall, the results highlight the benefits of biologically inspired modularity for robust and adaptive autonomous systems under uncertainty. |
| 2026-03-23 | [Computationally lightweight classifiers with frequentist bounds on predictions](http://arxiv.org/abs/2603.22128v1) | Shreeram Murali, Cristian R. Rojas et al. | While both classical and neural network classifiers can achieve high accuracy, they fall short on offering uncertainty bounds on their predictions, making them unfit for safety-critical applications. Existing kernel-based classifiers that provide such bounds scale with $\mathcal O (n^{\sim3})$ in time, making them computationally intractable for large datasets. To address this, we propose a novel, computationally efficient classification algorithm based on the Nadaraya-Watson estimator, for whose estimates we derive frequentist uncertainty intervals. We evaluate our classifier on synthetically generated data and on electrocardiographic heartbeat signals from the MIT-BIH Arrhythmia database. We show that the method achieves competitive accuracy $>$\SI{96}{\percent} at $\mathcal O(n)$ and $\mathcal O(\log n)$ operations, while providing actionable uncertainty bounds. These bounds can, e.g., aid in flagging low-confidence predictions, making them suitable for real-time settings with resource constraints, such as diagnostic monitoring or implantable devices. |
| 2026-03-23 | [End-to-End Differentiable Predictive Control with Guaranteed Constraint Satisfaction and feasibility for Building Demand Response](http://arxiv.org/abs/2603.22104v1) | Kaipeng Xu, Zhuo Zhi et al. | The high energy consumption of buildings presents a critical need for advanced control strategies like Demand Response (DR). Differentiable Predictive Control (DPC) has emerged as a promising method for learning explicit control policies, yet conventional DPC frameworks are hindered by three key limitations: the use of simplistic dynamics models with limited expressiveness, a decoupled training paradigm that fails to optimize for closed-loop performance, and a lack of practical safety guarantees under realistic assumptions. To address these shortcomings, this paper proposes a novel End-to-End Differentiable Predictive Control (E2E-DPC) framework. Our approach utilizes an Encoder-Only Transformer to model the complex system dynamics and employs a unified, performance-oriented loss to jointly train the model and the control policy. Crucially, we introduce an online tube-based constraint tightening method that provides theoretical guarantees for recursive feasibility and constraint satisfaction without requiring complex offline computation of terminal sets. The framework is validated in a high-fidelity EnergyPlus simulation, controlling a multi-zone building for a DR task. The results demonstrate that the proposed method with guarantees achieves near-perfect constraint satisfaction - a reduction of over 99% in violations compared to the baseline - at the cost of only a minor increase in electricity expenditure. This work provides a deployable, performance-driven control solution for building energy management and establishes a new pathway for developing verifiable learning-based control systems under milder assumptions. |
| 2026-03-23 | [Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](http://arxiv.org/abs/2603.22094v1) | Xingyu Zhu, Beier Zhu et al. | As vision-language models (VLMs) are increasingly deployed in open-world scenarios, they can be easily induced by visual jailbreak attacks to generate harmful content, posing serious risks to model safety and trustworthy usage. Recent activation steering methods inject directional vectors into model activations during inference to induce refusal behaviors and have demonstrated effectiveness. However, a steering vector may both enhance refusal ability and cause over-refusal, thereby degrading model performance on benign inputs. Moreover, due to the lack of theoretical interpretability, these methods still suffer from limited robustness and effectiveness. To better balance safety and utility, we propose NullSteer, a null-space projected activation defense framework. Our method constructs refusal directions within model activations through a linear transformation: it maintains zero perturbation within the benign subspace while dynamically inducing refusal along potentially harmful directions, thereby theoretically achieving safety enhancement without impairing the model's general capabilities. Extensive experiments show that NullSteer significantly reduces harmful outputs under various jailbreak attacks (average ASR reduction over 15 percent on MiniGPT-4) while maintaining comparable performance to the original model on general benchmarks. |
| 2026-03-23 | [DTVI: Dual-Stage Textual and Visual Intervention for Safe Text-to-Image Generation](http://arxiv.org/abs/2603.22041v1) | Binhong Tan, Zhaoxin Wang et al. | Text-to-Image (T2I) diffusion models have demonstrated strong generation ability, but their potential to generate unsafe content raises significant safety concerns. Existing inference-time defense methods typically perform category-agnostic token-level intervention in the text embedding space, which fails to capture malicious semantics distributed across the full token sequence and remains vulnerable to adversarial prompts. In this paper, we propose DTVI, a dual-stage inference-time defense framework for safe T2I generation. Unlike existing methods that intervene on specific token embeddings, our method introduces category-aware sequence-level intervention on the full prompt embedding to better capture distributed malicious semantics, and further attenuates the remaining unsafe influences during the visual generation stage. Experimental results on real-world unsafe prompts, adversarial prompts, and multiple harmful categories show that our method achieves effective and robust defense while preserving reasonable generation quality on benign prompts, obtaining an average Defense Success Rate (DSR) of 94.43% across sexual-category benchmarks and 88.56 across seven unsafe categories, while maintaining generation quality on benign prompts. |
| 2026-03-23 | [SecureBreak -- A dataset towards safe and secure models](http://arxiv.org/abs/2603.21975v1) | Marco Arazzi, Vignesh Kumar Kembu et al. | Large language models are becoming pervasive core components in many real-world applications. As a consequence, security alignment represents a critical requirement for their safe deployment. Although previous related works focused primarily on model architectures and alignment methodologies, these approaches alone cannot ensure the complete elimination of harmful generations. This concern is reinforced by the growing body of scientific literature showing that attacks, such as jailbreaking and prompt injection, can bypass existing security alignment mechanisms. As a consequence, additional security strategies are needed both to provide qualitative feedback on the robustness of the obtained security alignment at the training stage, and to create an ``ultimate'' defense layer to block unsafe outputs possibly produced by deployed models. To provide a contribution in this scenario, this paper introduces SecureBreak, a safety-oriented dataset designed to support the development of AI-driven solutions for detecting harmful LLM outputs caused by residual weaknesses in security alignment. The dataset is highly reliable due to careful manual annotation, where labels are assigned conservatively to ensure safety. It performs well in detecting unsafe content across multiple risk categories. Tests with pre-trained LLMs show improved results after fine-tuning on SecureBreak. Overall, the dataset is useful both for post-generation safety filtering and for guiding further model alignment and security improvements. |
| 2026-03-23 | [Interaction-Aware Predictive Environmental Control Barrier Function for Emergency Lane Change](http://arxiv.org/abs/2603.21958v1) | Ying Shuai Quan, Paolo Falcone et al. | Safety-critical motion planning in mixed traffic remains challenging for autonomous vehicles, especially when it involves interactions between the ego vehicle (EV) and surrounding vehicles (SVs). In dense traffic, the feasibility of a lane change depends strongly on how SVs respond to the EV motion. This paper presents an interaction-aware safety framework that incorporates such interactions into a control barrier function (CBF)-based safety assessment. The proposed method predicts near-future vehicle positions over a finite horizon, thereby capturing reactive SV behavior and embedding it into the CBF-based safety constraint. To address uncertainty in the SV response model, a robust extension is developed by treating the model mismatch as a bounded disturbance and incorporating an online uncertainty estimate into the barrier condition. Compared with classical environmental CBF methods that neglect SV reactions, the proposed approach provides a less conservative and more informative safety representation for interactive traffic scenarios, while improving robustness to uncertainty in the modeled SV behavior. |
| 2026-03-23 | [Disengagement Analysis and Field Tests of a Prototypical Open-Source Level 4 Autonomous Driving System](http://arxiv.org/abs/2603.21926v1) | Marvin Seegert, Christian Oefinger et al. | Proprietary Autonomous Driving Systems are typically evaluated through disengagements, unplanned manual interventions to alter vehicle behavior, as annually reported by the California Department of Motor Vehicles. However, the real-world capabilities of prototypical open-source Level 4 vehicles over substantial distances remain largely unexplored. This study evaluates a research vehicle running an Autoware-based software stack across 236 km of mixed traffic. By classifying 30 disengagements across 26 rides with a novel five-level criticality framework, we observed a spatial disengagement rate of 0.127 1/km. Interventions predominantly occurred at lower speeds near static objects and traffic lights. Perception and Planning failures accounted for 40% and 26.7% of disengagements, respectively, largely due to object-tracking losses and operational deadlocks caused by parked vehicles. Frequent, unnecessary interventions highlighted a lack of trust on the part of the safety driver. These results show that while open-source software enables extensive operations, disengagement analysis is vital for uncovering robustness issues missed by standard metrics. |
| 2026-03-23 | [Collision-Free Velocity Scheduling for Multi-Agent Systems on Predefined Routes via Inexact-Projection ADMM](http://arxiv.org/abs/2603.21913v1) | Seungyeop Lee, Jong-Han Kim | In structured multi-agent transportation systems, agents often must follow predefined routes, making spatial rerouting undesirable or impossible. This paper addresses route-constrained multi-agent coordination by optimizing waypoint passage times while preserving each agent's assigned waypoint order and nominal route assignment. A differentiable surrogate trajectory model maps waypoint timings to smooth position profiles and captures first-order tracking lag, enabling pairwise safety to be encoded through distance-based penalties evaluated on a dense temporal grid spanning the mission horizon. The resulting nonlinear and nonconvex velocity-scheduling problem is solved using an inexact-projection Alternating Direction Method of Multipliers (ADMM) algorithm that combines structured timing updates with gradient-based collision-correction steps and avoids explicit integer sequencing variables. Numerical experiments on random-crossing, bottleneck, and graph-based network scenarios show that the proposed method computes feasible and time-efficient schedules across a range of congestion levels and yields shorter mission completion times than a representative hierarchical baseline in the tested bottleneck cases. |
| 2026-03-23 | [Partial Attention in Deep Reinforcement Learning for Safe Multi-Agent Control](http://arxiv.org/abs/2603.21810v1) | Turki Bin Mohaya, Peter Seiler | Attention mechanisms excel at learning sequential patterns by discriminating data based on relevance and importance. This provides state-of-the-art performance in advanced generative artificial intelligence models. This paper applies this concept of an attention mechanism for multi-agent safe control. We specifically consider the design of a neural network to control autonomous vehicles in a highway merging scenario. The environment is modeled as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP). Within a QMIX framework, we include partial attention for each autonomous vehicle, thus allowing each ego vehicle to focus on the most relevant neighboring vehicles. Moreover, we propose a comprehensive reward signal that considers the global objectives of the environment (e.g., safety and vehicle flow) and the individual interests of each agent. Simulations are conducted in the Simulation of Urban Mobility (SUMO). The results show better performance compared to other driving algorithms in terms of safety, driving speed, and reward. |
| 2026-03-23 | [Memory-Efficient Boundary Map for Large-Scale Occupancy Grid Mapping](http://arxiv.org/abs/2603.21774v1) | Benxu Tang, Yunfan Ren et al. | Determining the occupancy status of locations in the environment is a fundamental task for safety-critical robotic applications. Traditional occupancy grid mapping methods subdivide the environment into a grid of voxels, each associated with one of three occupancy states: free, occupied, or unknown. These methods explicitly maintain all voxels within the mapped volume and determine the occupancy state of a location by directly querying the corresponding voxel that the location falls within. However, maintaining all grid voxels in high-resolution and large-scale scenarios requires substantial memory resources. In this paper, we introduce a novel representation that only maintains the boundary of the mapped volume. Specifically, we explicitly represent the boundary voxels, such as the occupied voxels and frontier voxels, while free and unknown voxels are automatically represented by volumes within or outside the boundary, respectively. As our representation maintains only a closed surface in two-dimensional (2D) space, instead of the entire volume in three-dimensional (3D) space, it significantly reduces memory consumption. Then, based on this 2D representation, we propose a method to determine the occupancy state of arbitrary locations in the 3D environment. We term this method as boundary map. Besides, we design a novel data structure for maintaining the boundary map, supporting efficient occupancy state queries. Theoretical analyses of the occupancy state query algorithm are also provided. Furthermore, to enable efficient construction and updates of the boundary map from the real-time sensor measurements, we propose a global-local mapping framework and corresponding update algorithms. Finally, we will make our implementation of the boundary map open-source on GitHub to benefit the community:https://github.com/hku-mars/BDM. |
| 2026-03-23 | [Quantifying Uncertainty in FMEDA Safety Metrics: An Error Propagation Approach for Enhanced ASIC Verification](http://arxiv.org/abs/2603.21770v1) | Antonino Armato, Christian Kehl et al. | Accurate and reliable safety metrics are paramount for functional safety verification of ASICs in automotive systems. Traditional FMEDA (Failure Modes, Effects, and Diagnostic Analysis) metrics, such as SPFM (Single Point Fault Metric) and LFM (Latent Fault Metric), depend on the precision of failure mode distribution (FMD) and diagnostic coverage (DC) estimations. This reliance can often leads to significant, unquantified uncertainties and a dependency on expert judgment, compromising the quality of the safety analysis. This paper proposes a novel approach that introduces error propagation theory into the calculation of FMEDA safety metrics. By quantifying the maximum deviation and providing confidence intervals for SPFM and LFM, our method offers a direct measure of analysis quality. Furthermore, we introduce an Error Importance Identifier (EII) to pinpoint the primary sources of uncertainty, guiding targeted improvements. This approach significantly enhances the transparency and trustworthiness of FMEDA, enabling more robust ASIC safety verification for ISO 26262 compliance, addressing a longstanding open question in the functional safety community. |
| 2026-03-23 | [Extending Precipitation Nowcasting Horizons via Spectral Fusion of Radar Observations and Foundation Model Priors](http://arxiv.org/abs/2603.21768v1) | Yuze Qin, Qingyong Li et al. | Precipitation nowcasting is critical for disaster mitigation and aviation safety. However, radar-only models frequently suffer from a lack of large-scale atmospheric context, leading to performance degradation at longer lead times. While integrating meteorological variables predicted by weather foundation models offers a potential remedy, existing architectures fail to reconcile the profound representational heterogeneities between radar imagery and meteorological data. To bridge this gap, we propose PW-FouCast, a novel frequency-domain fusion framework that leverages Pangu-Weather forecasts as spectral priors within a Fourier-based backbone. Our architecture introduces three key innovations: (i) Pangu-Weather-guided Frequency Modulation to align spectral magnitudes and phases with meteorological priors; (ii) Frequency Memory to correct phase discrepancies and preserve temporal evolution; and (iii) Inverted Frequency Attention to reconstruct high-frequency details typically lost in spectral filtering. Extensive experiments on the SEVIR and MeteoNet benchmarks demonstrate that PW-FouCast achieves state-of-the-art performance, effectively extending the reliable forecast horizon while maintaining structural fidelity. Our code is available at https://github.com/Onemissed/PW-FouCast. |
| 2026-03-23 | [Comprehensive Dosimetric Verification and Positional Sensitivity Analysis in Brachytherapy: A Unified ESAPI Tool for HDR and LDR Treatments](http://arxiv.org/abs/2603.21706v1) | J. A. Valgoma | This study presents the development and validation of an independent software tool based on the Varian Eclipse Scripting API (ESAPI) for multi-modal brachytherapy Quality Assurance (QA). The tool addresses GEC-ESTRO HDR protocols and LDR positional uncertainty analysis. Engineered in C#, the application interfaces with BrachyVision, Vitesse, and Variseed, enabling independent TG-43 dose calculations -- comparing point and line source models -- integrated with EQD2-based radiobiological summation. In HDR cervical cancer, the tool successfully automated EMBRACE II protocol reporting, streamlining clinical workflows by combining dosimetric QA with predictive and prospective planning. For LDR prostate treatments, a stochastic simulation module quantified the impact of systematic (rigid-body) versus random seed displacements on target coverage ($D_{90\%}$) and Organs at Risk (OAR) safety ($D_{0.1cc}$). Sensitivity analysis in LDR prostate implants was benchmarked using two clinical cases (prostate volumes 31 cc and 71.3 cc). LDR simulations revealed that systematic displacements ($\pm$ 2 mm) yielded significantly higher dosimetric deviations than stochastic movements. In the 31 cc case, systematic shifts resulted in a rectal ($D_{0.1cc}$) standard deviation (SD) of 24.3 Gy, whereas random displacements reduced this to 12.4 Gy. In the 71.3 cc case, random displacements resulted in a rectal $D_{0.1cc}$ SD of 7.6 Gy, confirming that smaller volumes exhibit heightened sensitivity to errors. Technical analysis demonstrated that the point source model overestimated bladder $D_{10\%}$ by 8% relative to the line source model. Our findings confirm that systematic rigid-body shifts represent a greater clinical risk for OAR toxicity than stochastic migration. Integrating predictive sensitivity analysis into the clinical workflow significantly enhances patient safety through robust plan verification. |
| 2026-03-23 | [Data-Free Layer-Adaptive Merging via Fisher Information for Long-to-Short Reasoning LLMs](http://arxiv.org/abs/2603.21705v1) | Tian Xia | Model merging has emerged as a practical approach to combine capabilities of specialized large language models (LLMs) without additional training. In the Long-to-Short (L2S) scenario, merging a base model with a long-chain-of-thought reasoning model aims to preserve reasoning accuracy while reducing output length. Existing methods rely on Task Arithmetic and its variants, which implicitly assume that model outputs vary linearly with the merging coefficient -- an assumption we show is systematically violated in L2S settings. We provide the first theoretical justification for layer-adaptive merging: we prove that merging error is bounded by a term proportional to the per-layer Hessian norm (Proposition~1), and establish that the Fisher Information Matrix (FIM) is a principled, computable proxy for this bound via the Fisher-Hessian equivalence at local optima. Building on this theory, we propose \textbf{FIM-Merging}, which computes diagonal FIM using only random token inputs (no domain-specific calibration data required) and uses it to assign per-layer merging coefficients. On the 7B L2S benchmark, FIM-TIES achieves state-of-the-art performance on five out of six evaluation benchmarks, including a \textbf{+6.2} point gain on MATH500 over ACM-TIES (90.2 vs.\ 84.0), while requiring no calibration data. On the 1.5B benchmark, FIM-TIES achieves an average accuracy of \textbf{47.3}, surpassing the previous best ACM-TIES (43.3) by \textbf{+3.9} points, while reducing average response length by \textbf{91.9\%} relative to the long-CoT model. Our framework also provides a unified theoretical explanation for why existing layer-adaptive methods such as ACM empirically outperform uniform merging. |
| 2026-03-23 | [A Blueprint for Self-Evolving Coding Agents in Vehicle Aerodynamic Drag Prediction](http://arxiv.org/abs/2603.21698v1) | Jinhui Ren, Huaiming Li et al. | High-fidelity vehicle drag evaluation is constrained less by solver runtime than by workflow friction: geometry cleanup, meshing retries, queue contention, and reproducibility failures across teams. We present a contract-centric blueprint for self-evolving coding agents that discover executable surrogate pipelines for predicting drag coefficient $C_d$ under industrial constraints. The method formulates surrogate discovery as constrained optimization over programs, not static model instances, and combines Famou-Agent-style evaluator feedback with population-based island evolution, structured mutations (data, model, loss, and split policies), and multi-objective selection balancing ranking quality, stability, and cost. A hard evaluation contract enforces leakage prevention, deterministic replay, multi-seed robustness, and resource budgets before any candidate is admitted. Across eight anonymized evolutionary operators, the best system reaches a Combined Score of 0.9335 with sign-accuracy 0.9180, while trajectory and ablation analyses show that adaptive sampling and island migration are primary drivers of convergence quality. The deployment model is explicitly ``screen-and-escalate'': surrogates provide high-throughput ranking for design exploration, but low-confidence or out-of-distribution cases are automatically escalated to high-fidelity CFD. The resulting contribution is an auditable, reusable workflow for accelerating aerodynamic design iteration while preserving decision-grade reliability, governance traceability, and safety boundaries. |
| 2026-03-23 | [Structured Visual Narratives Undermine Safety Alignment in Multimodal Large Language Models](http://arxiv.org/abs/2603.21697v1) | Rui Yang Tan, Yujia Hu et al. | Multimodal Large Language Models (MLLMs) extend text-only LLMs with visual reasoning, but also introduce new safety failure modes under visually grounded instructions. We study comic-template jailbreaks that embed harmful goals inside simple three-panel visual narratives and prompt the model to role-play and "complete the comic." Building on JailbreakBench and JailbreakV, we introduce ComicJailbreak, a comic-based jailbreak benchmark with 1,167 attack instances spanning 10 harm categories and 5 task setups. Across 15 state-of-the-art MLLMs (six commercial and nine open-source), comic-based attacks achieve success rates comparable to strong rule-based jailbreaks and substantially outperform plain-text and random-image baselines, with ensemble success rates exceeding 90% on several commercial models. Then, with the existing defense methodologies, we show that these methods are effective against the harmful comics, they will induce a high refusal rate when prompted with benign prompts. Finally, using automatic judging and targeted human evaluation, we show that current safety evaluators can be unreliable on sensitive but non-harmful content. Our findings highlight the need for safety alignment robust to narrative-driven multimodal jailbreaks. |
| 2026-03-23 | [RTD-RAX: Fast, Safe Trajectory Planning for Systems under Unknown Disturbances](http://arxiv.org/abs/2603.21635v1) | Evanns Morales-Cuadrado, Long Kiu Chung et al. | Reachability-based Trajectory Design (RTD) is a provably safe, real-time trajectory planning framework that combines offline reachable-set computation with online trajectory optimization. However, standard RTD implementations suffer from two key limitations: conservatism induced by worst-case reachable-set overapproximations, and an inability to account for real-time disturbances during execution. This paper presents RTD-RAX, a runtime-assurance extension of RTD that utilizes a non-conservative RTD formulation to rapidly generate goal-directed candidate trajectories, and utilizes mixed monotone reachability for fast, disturbance-aware online safety certification. When proposed trajectories fail safety certification under real-time uncertainty, a repair procedure finds nearby safe trajectories that preserve progress toward the goal while guaranteeing safety under real-time disturbances. |
| 2026-03-23 | [Conformal Koopman for Embedded Nonlinear Control with Statistical Robustness: Theory and Real-World Validation](http://arxiv.org/abs/2603.21580v1) | Koki Hirano, Hiroyasu Tsukamoto | We propose a fully data-driven, Koopman-based framework for statistically robust control of discrete-time nonlinear systems with linear embeddings. Establishing a connection between the Koopman operator and contraction theory, it offers distribution-free probabilistic bounds on the state tracking error under Koopman modeling uncertainty. Conformal prediction is employed here to rigorously derive a bound on the state-dependent modeling uncertainty throughout the trajectory, ensuring safety and robustness without assuming a specific error prediction structure or distribution. Unlike prior approaches that merely combine conformal prediction with Koopman-based control in an open-loop setting, our method establishes a closed-loop control architecture with formal guarantees that explicitly account for both forward and inverse modeling errors. Also, by expressing the tracking error bound in terms of the control parameters and the modeling errors, our framework offers a quantitative means to formally enhance the performance of arbitrary Koopman-based control. We validate our method both in numerical simulations with the Dubins car and in real-world experiments with a highly nonlinear flapping-wing drone. The results demonstrate that our method indeed provides formal safety guarantees while maintaining accurate tracking performance under Koopman modeling uncertainty. |
| 2026-03-23 | [PROBE: Diagnosing Residual Concept Capacity in Erased Text-to-Video Diffusion Models](http://arxiv.org/abs/2603.21547v1) | Yiwei Xie, Zheng Zhang et al. | Concept erasure techniques for text-to-video (T2V) diffusion models report substantial suppression of sensitive content, yet current evaluation is limited to checking whether the target concept is absent from generated frames, treating output-level suppression as evidence of representational removal. We introduce PROBE, a diagnostic protocol that quantifies the \textit{reactivation potential} of erased concepts in T2V models. With all model parameters frozen, PROBE optimizes a lightweight pseudo-token embedding through a denoising reconstruction objective combined with a novel latent alignment constraint that anchors recovery to the spatiotemporal structure of the original concept. We make three contributions: (1) a multi-level evaluation framework spanning classifier-based detection, semantic similarity, temporal reactivation analysis, and human validation; (2) systematic experiments across three T2V architectures, three concept categories, and three erasure strategies revealing that all tested methods leave measurable residual capacity whose robustness correlates with intervention depth; and (3) the identification of temporal re-emergence, a video-specific failure mode where suppressed concepts progressively resurface across frames, invisible to frame-level metrics. These findings suggest that current erasure methods achieve output-level suppression rather than representational removal. We release our protocol to support reproducible safety auditing. Our code is available at https://github.com/YiweiXie/PRObingBasedEvaluation. |

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



