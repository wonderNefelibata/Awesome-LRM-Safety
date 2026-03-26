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
| 2026-03-25 | [DreamerAD: Efficient Reinforcement Learning via Latent World Model for Autonomous Driving](http://arxiv.org/abs/2603.24587v1) | Pengxuan Yang, Yupeng Zheng et al. | We introduce DreamerAD, the first latent world model framework that enables efficient reinforcement learning for autonomous driving by compressing diffusion sampling from 100 steps to 1 - achieving 80x speedup while maintaining visual interpretability. Training RL policies on real-world driving data incurs prohibitive costs and safety risks. While existing pixel-level diffusion world models enable safe imagination-based training, they suffer from multi-step diffusion inference latency (2s/frame) that prevents high-frequency RL interaction. Our approach leverages denoised latent features from video generation models through three key mechanisms: (1) shortcut forcing that reduces sampling complexity via recursive multi-resolution step compression, (2) an autoregressive dense reward model operating directly on latent representations for fine-grained credit assignment, and (3) Gaussian vocabulary sampling for GRPO that constrains exploration to physically plausible trajectories. DreamerAD achieves 87.7 EPDMS on NavSim v2, establishing state-of-the-art performance and demonstrating that latent-space RL is effective for autonomous driving. |
| 2026-03-25 | [Integral Control Barrier Functions with Input Delay: Prediction, Feasibility, and Robustness](http://arxiv.org/abs/2603.24566v1) | Adam K. Kiss, Ersin Das et al. | Time delays in feedback control loops can cause controllers to respond too late, and with excessively large corrective actions, leading to unsafe behavior (violation of state constraints) and controller infeasibility (violation of input constraints). To address this problem, we develop a safety-critical control framework for nonlinear systems with input delay using dynamically defined (integral) controllers. Building on the concept of Integral Control Barrier Functions (ICBFs), we concurrently address two fundamental challenges: compensating the effect of delays, while ensuring feasibility when state and input constraints are imposed jointly. To this end, we embed predictor feedback into a dynamically defined control law to compensate for delays, with the predicted state evolving according to delay-free dynamics. Then, utilizing ICBFs, we formulate a quadratic program for safe control design. For systems subject to simultaneous state and input constraints, we derive a closed-form feasibility condition for the resulting controller, yielding a compatible ICBF pair that guarantees forward invariance under delay. We also address robustness to prediction errors (e.g., caused by delay uncertainty) using tunable robust ICBFs. Our approach is validated on an adaptive cruise control example with actuation delay. |
| 2026-03-25 | [Analysing the Safety Pitfalls of Steering Vectors](http://arxiv.org/abs/2603.24543v1) | Yuxiao Li, Alina Fastowski et al. | Activation steering has emerged as a powerful tool to shape LLM behavior without the need for weight updates. While its inherent brittleness and unreliability are well-documented, its safety implications remain underexplored. In this work, we present a systematic safety audit of steering vectors obtained with Contrastive Activation Addition (CAA), a widely used steering approach, under a unified evaluation protocol. Using JailbreakBench as benchmark, we show that steering vectors consistently influence the success rate of jailbreak attacks, with stronger amplification under simple template-based attacks. Across LLM families and sizes, steering the model in specific directions can drastically increase (up to 57%) or decrease (up to 50%) its attack success rate (ASR), depending on the targeted behavior. We attribute this phenomenon to the overlap between the steering vectors and the latent directions of refusal behavior. Thus, we offer a traceable explanation for this discovery. Together, our findings reveal the previously unobserved origin of this safety gap in LLMs, highlighting a trade-off between controllability and safety. |
| 2026-03-25 | [SEGAR: Selective Enhancement for Generative Augmented Reality](http://arxiv.org/abs/2603.24541v1) | Fanjun Bu, Chenyang Yuan et al. | Generative world models offer a compelling foundation for augmented-reality (AR) applications: by predicting future image sequences that incorporate deliberate visual edits, they enable temporally coherent, augmented future frames that can be computed ahead of time and cached, avoiding per-frame rendering from scratch in real time. In this work, we present SEGAR, a preliminary framework that combines a diffusion-based world model with a selective correction stage to support this vision. The world model generates augmented future frames with region-specific edits while preserving others, and the correction stage subsequently aligns safety-critical regions with real-world observations while preserving intended augmentations elsewhere. We demonstrate this pipeline in driving scenarios as a representative setting where semantic region structure is well defined and real-world feedback is readily available. We view this as an early step toward generative world models as practical AR infrastructure, where future frames can be generated, cached, and selectively corrected on demand. |
| 2026-03-25 | [Robust Multilingual Text-to-Pictogram Mapping for Scalable Reading Rehabilitation](http://arxiv.org/abs/2603.24536v1) | Soufiane Jhilal, Martina Galletti | Reading comprehension presents a significant challenge for children with Special Educational Needs and Disabilities (SEND), often requiring intensive one-on-one reading support. To assist therapists in scaling this support, we developed a multilingual, AI-powered interface that automatically enhances text with visual scaffolding. This system dynamically identifies key concepts and maps them to contextually relevant pictograms, supporting learners across languages. We evaluated the system across five typologically diverse languages (English, French, Italian, Spanish, and Arabic), through multilingual coverage analysis, expert clinical review by speech therapists and special education professionals, and latency assessment. Evaluation results indicate high pictogram coverage and visual scaffolding density across the five languages. Expert audits suggested that automatically selected pictograms were semantically appropriate, with combined correct and acceptable ratings exceeding 95% for the four European languages and approximately 90% for Arabic despite reduced pictogram repository coverage. System latency remained within interactive thresholds suitable for real-time educational use. These findings support the technical viability, semantic safety, and acceptability of automated multimodal scaffolding to improve accessibility for neurodiverse learners. |
| 2026-03-25 | [Study of Low-Frequency Core-Edge Coupling in a Tokamak: I. Experimental Observation in KSTAR](http://arxiv.org/abs/2603.24525v1) | Wonjun Lee, Andreas Bierwage et al. | Double-peaked fishbone events across multiple KSTAR discharges are investigated. The normalized beta $β_{\mathrm{N}}$ and the edge safety factor $q_{\mathrm{95}}$ under which the fishbones appear vary depending on the presence and form of external magnetic perturbations. The fishbone strength is closely related to $β_{\mathrm{N}}$ and $q_{\mathrm{95}}$: as $β_{\mathrm{N}}$ increases and $q_{\mathrm{95}}$ decreases, the fishbone strength increases. Measured fishbone-relevant signals are decomposed into amplitude envelope and phase components in the temporal domain, which are analyzed separately. In terms of the amplitude envelope component, the edge electron temperature fluctuation $\tilde T_\mathrm{e}^{\mathrm{Edge}}$ becomes more correlated with the poloidal magnetic fluctuation $\dot{B}_\mathrmθ$ compared to the core electron temperature fluctuation $\tilde T_\mathrm{e}^{\mathrm{Core}}$ as fishbone strength increases. In terms of the phase component, the phase of $\tilde T_\mathrm{e}^{\mathrm{Edge}}$ precedes the phase of $\tilde T_\mathrm{e}^{\mathrm{Core}}$ except in the case of very weak fishbones where the phase relations are inconclusive due to weak fishbone activity at the edge plasma, which is comparable to background fluctuations. The investigation suggests the possibility that the edge activity is not a mere side effect of the core activity, but could play an active role. |
| 2026-03-25 | [Experimental Evidence for Increased Particle Fluxes Due to a Change in Transport at the Separatrix near Density Limits on Alcator C-Mod](http://arxiv.org/abs/2603.24512v1) | M. A. Miller, J. W. Hughes et al. | Experimental inferences of cross-field particle flux at the separatrix, $Γ_{\perp}^\mathrm{sep}$, show rapid growth near H-mode and L-mode density limits at high magnetic field on Alcator C-Mod. Increases in $Γ_{\perp}^\mathrm{sep}$ correlate well with proximity to high density operational boundaries as proposed by the separatrix operational space model. $Γ_{\perp}^\mathrm{sep}$ grows as the L-mode density limit and the H-L-mode back transition boundaries are approached, consistent with expectations of plasma instability-driven turbulence suggested by theory, confirming the power dependence of density limits. $Γ_{\perp}^\mathrm{sep}$ is well-organized by the characteristic wavenumber for resistive ballooning mode turbulence, $k_\mathrm{RBM}$, from interchange-drift-Alfvén fluid turbulence theory, with additional dependence on the cylindrical safety factor, $\hat{q}_\mathrm{cyl}$, yielding an empirical limit to plasma operation of $k_\mathrm{RBM}^{2}\hat{q}_\mathrm{cyl} = 1$. This limit corresponds to the point where the perpendicular heat flux, $Q_{\perp}$, reaches the level of the parallel heat flux, $Q_{\parallel}$, i.e. $Q_{\perp} \approx Q_{\parallel}$, beyond which point thermal equilibrium is not satisfied, resulting in a fold catastrophe. |
| 2026-03-25 | [Claudini: Autoresearch Discovers State-of-the-Art Adversarial Attack Algorithms for LLMs](http://arxiv.org/abs/2603.24511v1) | Alexander Panfilov, Peter Romov et al. | LLM agents like Claude Code can not only write code but also be used for autonomous AI research and engineering \citep{rank2026posttrainbench, novikov2025alphaevolve}. We show that an \emph{autoresearch}-style pipeline \citep{karpathy2026autoresearch} powered by Claude Code discovers novel white-box adversarial attack \textit{algorithms} that \textbf{significantly outperform all existing (30+) methods} in jailbreaking and prompt injection evaluations.   Starting from existing attack implementations, such as GCG~\citep{zou2023universal}, the agent iterates to produce new algorithms achieving up to 40\% attack success rate on CBRN queries against GPT-OSS-Safeguard-20B, compared to $\leq$10\% for existing algorithms (\Cref{fig:teaser}, left).   The discovered algorithms generalize: attacks optimized on surrogate models transfer directly to held-out models, achieving \textbf{100\% ASR against Meta-SecAlign-70B} \citep{chen2025secalign} versus 56\% for the best baseline (\Cref{fig:teaser}, middle). Extending the findings of~\cite{carlini2025autoadvexbench}, our results are an early demonstration that incremental safety and security research can be automated using LLM agents. White-box adversarial red-teaming is particularly well-suited for this: existing methods provide strong starting points, and the optimization objective yields dense, quantitative feedback. We release all discovered attacks alongside baseline implementations and evaluation code at https://github.com/romovpa/claudini. |
| 2026-03-25 | [Towards Safe Learning-Based Non-Linear Model Predictive Control through Recurrent Neural Network Modeling](http://arxiv.org/abs/2603.24503v1) | Mihaela-Larisa Clement, Mónika Farsang et al. | The practical deployment of nonlinear model predictive control (NMPC) is often limited by online computation: solving a nonlinear program at high control rates can be expensive on embedded hardware, especially when models are complex or horizons are long. Learning-based NMPC approximations shift this computation offline but typically demand large expert datasets and costly training. We propose Sequential-AMPC, a sequential neural policy that generates MPC candidate control sequences by sharing parameters across the prediction horizon. For deployment, we wrap the policy in a safety-augmented online evaluation and fallback mechanism, yielding Safe Sequential-AMPC. Compared to a naive feedforward policy baseline across several benchmarks, Sequential-AMPC requires substantially fewer expert MPC rollouts and yields candidate sequences with higher feasibility rates and improved closed-loop safety. On high-dimensional systems, it also exhibits better learning dynamics and performance in fewer epochs while maintaining stable validation improvement where the feedforward baseline can stagnate. |
| 2026-03-25 | [Multi-Agent Reasoning with Consistency Verification Improves Uncertainty Calibration in Medical MCQA](http://arxiv.org/abs/2603.24481v1) | John Ray B. Martinez | Miscalibrated confidence scores are a practical obstacle to deploying AI in clinical settings. A model that is always overconfident offers no useful signal for deferral. We present a multi-agent framework that combines domain-specific specialist agents with Two-Phase Verification and S-Score Weighted Fusion to improve both calibration and discrimination in medical multiple-choice question answering. Four specialist agents (respiratory, cardiology, neurology, gastroenterology) generate independent diagnoses using Qwen2.5-7B-Instruct. Each diagnosis is then subjected to a two-phase self-verification process that measures internal consistency and produces a Specialist Confidence Score (S-score). The S-scores drive a weighted fusion strategy that selects the final answer and calibrates the reported confidence. We evaluate across four experimental settings, covering 100-question and 250-question high-disagreement subsets of both MedQA-USMLE and MedMCQA. Calibration improvement is the central finding, with ECE reduced by 49-74% across all four settings, including the harder MedMCQA benchmark where these gains persist even when absolute accuracy is constrained by knowledge-intensive recall demands. On MedQA-250, the full system achieves ECE = 0.091 (74.4% reduction over the single-specialist baseline) and AUROC = 0.630 (+0.056) at 59.2% accuracy. Ablation analysis identifies Two-Phase Verification as the primary calibration driver and multi-agent reasoning as the primary accuracy driver. These results establish that consistency-based verification produces more reliable uncertainty estimates across diverse medical question types, providing a practical confidence signal for deferral in safety-critical clinical AI applications. |
| 2026-03-25 | [Study of Low-Frequency Core-Edge Coupling in a Tokamak: II. Spatial Channeling & Focusing In Antenna-Driven MHD](http://arxiv.org/abs/2603.24463v1) | Andreas Bierwage, Wonjun Lee et al. | Motivated by evidence for core-edge coupling in the form of double-peaked fishbone-like low-frequency modes ($\lesssim 20\,{\rm kHz}$) in KSTAR, which exhibit synchronized Alfvénic activity both in the central core and near the plasma edge [1], we study the nonlocal response of a tokamak plasma in a visco-resistive full MHD simulation model using the code MEGA. The waves are driven by an internal "antenna" that is localized both radially and azimuthally in the poloidal $(R,z)$ plane and has a sinusoidal form $\exp(inζ- iωt)$ with Fourier mode number $n=\pm 1$ in the toroidal angle $ζ$ and fixed angular frequency $ω$ in time $t$. By flattening the safety factor profile $q(r)$ at suitable locations in the minor radius $r$, we created plateaus in the low-frequency Alfvén continua that act as wave "receivers". First, we confirm that such continuum plateaus respond with a coherent quasi-mode even when the driving antenna is located at a distant radius. Second, by varying the antenna location, we confirm the expectation of inward drive being more efficient than outward drive, which we attribute to volumetric focusing. Third, we find that the central core also responds well at frequencies below the central Alfvénic continuum plateau, which could facilitate chirping. Our results show that a core-localized low-frequency response does not necessarily require core-localized drive nor an exactly matching continuum, but may be driven from the edge and sub-resonantly. It remains to be seen to what extent the examined effects play a role in double-peaked fishbone-like activity. Other possible contributing mechanisms are discussed to motivate further study. Our analyses also elucidate the mode structure formation process, from transients to quasi- or eigenmodes, here in the realm of MHD, and to be followed by a verification study against kinetic models. |
| 2026-03-25 | [Hybrid Spatiotemporal Logic for Automotive Applications: Modeling and Model-Checking](http://arxiv.org/abs/2603.24443v1) | Radu-Florin Tulcan, Rose Bohrer et al. | We introduce a hybrid spatiotemporal logic for automotive safety applications (HSTL), focused on highway driving. Spatiotemporal logic features specifications about vehicles throughout space and time, while hybrid logic enables precise references to individual vehicles and their historical positions. We define the semantics of HSTL and provide a baseline model-checking algorithm for it. We propose two optimized model-checking algorithms, which reduce the search space based on the reachable states and possible transitions from one state to another. All three model-checking algorithms are evaluated on a series of common driving scenarios such as safe following, safe crossings, overtaking, and platooning. An exponential performance improvement is observed for the optimized algorithms. |
| 2026-03-25 | [ClawKeeper: Comprehensive Safety Protection for OpenClaw Agents Through Skills, Plugins, and Watchers](http://arxiv.org/abs/2603.24414v1) | Songyang Liu, Chaozhuo Li et al. | OpenClaw has rapidly established itself as a leading open-source autonomous agent runtime, offering powerful capabilities including tool integration, local file access, and shell command execution. However, these broad operational privileges introduce critical security vulnerabilities, transforming model errors into tangible system-level threats such as sensitive data leakage, privilege escalation, and malicious third-party skill execution. Existing security measures for the OpenClaw ecosystem remain highly fragmented, addressing only isolated stages of the agent lifecycle rather than providing holistic protection. To bridge this gap, we present ClawKeeper, a real-time security framework that integrates multi-dimensional protection mechanisms across three complementary architectural layers. (1) \textbf{Skill-based protection} operates at the instruction level, injecting structured security policies directly into the agent context to enforce environment-specific constraints and cross-platform boundaries. (2) \textbf{Plugin-based protection} serves as an internal runtime enforcer, providing configuration hardening, proactive threat detection, and continuous behavioral monitoring throughout the execution pipeline. (3) \textbf{Watcher-based protection} introduces a novel, decoupled system-level security middleware that continuously verifies agent state evolution. It enables real-time execution intervention without coupling to the agent's internal logic, supporting operations such as halting high-risk actions or enforcing human confirmation. We argue that this Watcher paradigm holds strong potential to serve as a foundational building block for securing next-generation autonomous agent systems. Extensive qualitative and quantitative evaluations demonstrate the effectiveness and robustness of ClawKeeper across diverse threat scenarios. We release our code. |
| 2026-03-25 | [A Neuro-Symbolic System for Interpretable Multimodal Physiological Signals Integration in Human Fatigue Detection](http://arxiv.org/abs/2603.24358v1) | Mohammadreza Jamalifard, Yaxiong Lei et al. | We propose a neuro-symbolic architecture that learns four interpretable physiological concepts, oculomotor dynamics, gaze stability, prefrontal hemodynamics, and multimodal, from eye-tracking and neural hemodynamics, functional near-infrared spectroscopy, (fNIRS) windows using attention-based encoders, and combines them with differentiable approximate reasoning rules using learned weights and soft thresholds, to address both rigid hand-crafted rules and the lack of subject-level alignment diagnostics. We apply this system to fatigue classification from multimodal physiological signals, a domain that requires models that are accurate and interpretable, with internal reasoning that can be inspected for safety-critical use. In leave-one-subject-out evaluation on 18 participants (560 samples), the method achieves 72.1% +/- 12.3% accuracy, comparable to tuned baselines while exposing concept activations and rule firing strengths. Ablations indicate gains from participant-specific calibration (+5.2 pp), a modest drop without the fNIRS concept (-1.2 pp), and slightly better performance with Lukasiewicz operators than product (+0.9 pp). We also introduce concept fidelity, an offline per-subject audit metric from held-out labels, which correlates strongly with per-subject accuracy (r=0.843, p < 0.0001). |
| 2026-03-25 | [A Sensorless, Inherently Compliant Anthropomorphic Musculoskeletal Hand Driven by Electrohydraulic Actuators](http://arxiv.org/abs/2603.24357v1) | Misato Sonoda, Ronan Hinchet et al. | Robotic manipulation in unstructured environments requires end-effectors that combine high kinematic dexterity with physical compliance. While traditional rigid hands rely on complex external sensors for safe interaction, electrohydraulic actuators offer a promising alternative. This paper presents the design, control, and evaluation of a novel musculoskeletal robotic hand architecture powered entirely by remote Peano-HASEL actuators, specifically optimized for safe manipulation. By relocating the actuators to the forearm, we functionally isolate the grasping interface from electrical hazards while maintaining a slim, human-like profile. To address the inherently limited linear contraction of these soft actuators, we integrate a 1:2 pulley routing mechanism that mechanically amplifies tendon displacement. The resulting system prioritizes compliant interaction over high payload capacity, leveraging the intrinsic force-limiting characteristics of the actuators to provide a high level of inherent safety. Furthermore, this physical safety is augmented by the self-sensing nature of the HASEL actuators. By simply monitoring the operating current, we achieve real-time grasp detection and closed-loop contact-aware control without relying on external force transducers or encoders. Experimental results validate the system's dexterity and inherent safety, demonstrating the successful execution of various grasp taxonomies and the non-destructive grasping of highly fragile objects, such as a paper balloon. These findings highlight a significant step toward simplified, inherently compliant soft robotic manipulation. |
| 2026-03-25 | [C-STEP: Continuous Space-Time Empowerment for Physics-informed Safe Reinforcement Learning of Mobile Agents](http://arxiv.org/abs/2603.24241v1) | Guihlerme Daubt, Adrian Redder | Safe navigation in complex environments remains a central challenge for reinforcement learning (RL) in robotics. This paper introduces Continuous Space-Time Empowerment for Physics-informed (C-STEP) safe RL, a novel measure of agent-centric safety tailored to deterministic, continuous domains. This measure can be used to design physics-informed intrinsic rewards by augmenting positive navigation reward functions. The reward incorporates the agents internal states (e.g., initial velocity) and forward dynamics to differentiate safe from risky behavior. By integrating C-STEP with navigation rewards, we obtain an intrinsic reward function that jointly optimizes task completion and collision avoidance. Numerical results demonstrate fewer collisions, reduced proximity to obstacles, and only marginal increases in travel time. Overall, C-STEP offers an interpretable, physics-informed approach to reward shaping in RL, contributing to safety for agentic mobile robotic systems. |
| 2026-03-25 | [Accelerated Spline-Based Time-Optimal Motion Planning with Continuous Safety Guarantees for Non-Differentially Flat Systems](http://arxiv.org/abs/2603.24133v1) | Dries Dirckx, Jan Swevers et al. | Generating time-optimal, collision-free trajectories for autonomous mobile robots involves a fundamental trade-off between guaranteeing safety and managing computational complexity. State-of-the-art approaches formulate spline-based motion planning as a single Optimal Control Problem (OCP) but often suffer from high computational cost because they include separating hyperplane parameters as decision variables to enforce continuous collision avoidance. This paper presents a novel method that alleviates this bottleneck by decoupling the determination of separating hyperplanes from the OCP. By treating the separation theorem as an independent classification problem solvable via a linear system or quadratic program, the proposed method eliminates hyperplane parameters from the optimisation variables, effectively transforming non-convex constraints into linear ones. Experimental validation demonstrates that this decoupled approach reduces trajectory computation times up to almost 60% compared to fully coupled methods in obstacle-rich environments, while maintaining rigorous continuous safety guarantees. |
| 2026-03-25 | [Likelihood hacking in probabilistic program synthesis](http://arxiv.org/abs/2603.24126v1) | Jacek Karwowski, Younesse Kaddar et al. | When language models are trained by reinforcement learning (RL) to write probabilistic programs, they can artificially inflate their marginal-likelihood reward by producing programs whose data distribution fails to normalise instead of fitting the data better. We call this failure likelihood hacking (LH). We formalise LH in a core probabilistic programming language (PPL) and give sufficient syntactic conditions for its prevention, proving that a safe language fragment $\mathcal{L}_{\text{safe}}$ satisfying these conditions cannot produce likelihood-hacking programs. Empirically, we show that GRPO-trained models generating PyMC code discover LH exploits within the first few training steps, driving violation rates well above the untrained-model baseline. We implement $\mathcal{L}_{\text{safe}}$'s conditions as $\texttt{SafeStan}$, a LH-resistant modification of Stan, and show empirically that it prevents LH under optimisation pressure. These results show that language-level safety constraints are both theoretically grounded and effective in practice for automated Bayesian model discovery. |
| 2026-03-25 | [When Understanding Becomes a Risk: Authenticity and Safety Risks in the Emerging Image Generation Paradigm](http://arxiv.org/abs/2603.24079v1) | Ye Leng, Junjie Chu et al. | Recently, multimodal large language models (MLLMs) have emerged as a unified paradigm for language and image generation. Compared with diffusion models, MLLMs possess a much stronger capability for semantic understanding, enabling them to process more complex textual inputs and comprehend richer contextual meanings. However, this enhanced semantic ability may also introduce new and potentially greater safety risks. Taking diffusion models as a reference point, we systematically analyze and compare the safety risks of emerging MLLMs along two dimensions: unsafe content generation and fake image synthesis. Across multiple unsafe generation benchmark datasets, we observe that MLLMs tend to generate more unsafe images than diffusion models. This difference partly arises because diffusion models often fail to interpret abstract prompts, producing corrupted outputs, whereas MLLMs can comprehend these prompts and generate unsafe content. For current advanced fake image detectors, MLLM-generated images are also notably harder to identify. Even when detectors are retrained with MLLMs-specific data, they can still be bypassed by simply providing MLLMs with longer and more descriptive inputs. Our measurements indicate that the emerging safety risks of the cutting-edge generative paradigm, MLLMs, have not been sufficiently recognized, posing new challenges to real-world safety. |
| 2026-03-25 | [Sensing-Assisted Adaptive Beam Probing with Calibrated Multimodal Priors and Uncertainty-Aware Scheduling](http://arxiv.org/abs/2603.24024v1) | Abidemi Orimogunje, Vukan Ninkovic et al. | Highly directional mmWave/THz links require rapid beam alignment, yet exhaustive codebook sweeps incur prohibitive training overhead. This letter proposes a sensing-assisted adaptive probing policy that maps multimodal sensing (radar/LiDAR/camera) to a calibrated prior over beams, predicts per-beam reward with a deep Q-ensemble whose disagreement serves as a practical epistemic-uncertainty proxy, and schedules a small probe set using a Prior-Q upper-confidence score. The probing budget is adapted from prior entropy, explicitly coupling sensing confidence to communication overhead, while a margin-based safety rule prevents low signal-to-noise ratio (SNR) locks. Experiments on DeepSense-6G (train: scenarios 42 and 44; test:43) with a 21-beam discrete Fourier transform (DFT) codebook achieve Top-1/Top-3 of 0.81/0.99 with expected beam probe of 2 per sweep and zero observed outages at θ = 0 dB with margin Δ = 3 dB. The results show that multimodal priors with ensemble uncertainty match link quality and improve reliability compared to ablations while cutting overhead with better predictive model. |

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



