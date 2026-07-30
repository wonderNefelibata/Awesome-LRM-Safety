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
| 2026-07-29 | [Explainable and Resource-Efficient Spatial Reasoning in Multimodal LLMs for Decision-Critical Applications](http://arxiv.org/abs/2607.27145v1) | Piyush Jain, Kousik Dasgupta et al. | As Multimodal Large Language Models (MLLMs) are increasingly deployed in decision-critical pipelines such as robotics, embodied AI, and safety monitoring, the opacity of their spatial judgments limits operator trust and auditability. MLLMs demonstrate strong reasoning but often struggle with fine-grained spatial understanding and object hallucination. Prior work, ByDeWay, introduced Layered-Depth-Based Prompting (LDP), a training-free framework that mitigates hallucinations by structuring prompts using monocular depth estimation. However, coarse depth layering falls short in resolving object-to-object spatial relationships within the same geometric plane, such as projective ("left of", "above") and topological ("inside", "touching") relations. We propose ByDeWay-V2, which integrates explicit spatial relational context alongside depth cues, expressed as human-readable predicates that serve as auditable evidence for downstream decision support. Using an open-vocabulary object detector (YOLO-World-L), our framework computes pairwise geometric relations between detected objects and injects them as structured spatial predicates into the MLLM prompt, bridging 3D scene depth and 2D spatial semantics without any training. We evaluate ByDeWay-V2 on the Visual Spatial Reasoning (VSR) and BLINK benchmarks across multiple MLLMs, with hallucination grounding assessed via POPE. On the BLINK spatial subset, ByDeWay-V2 achieves a 46 percent relative F1 improvement over LDP for Qwen2.5-VL, and recovers BLIP-Base's spatial reasoning on VSR from near-random performance to a competitive F1 of 0.53. Our lightest configuration operates under a strict 40-token context budget on CPU, showing the framework's suitability for resource-constrained, real-time decision-support settings. |
| 2026-07-29 | [Cost-Sensitive Conformal Prediction and Human-in-the-Loop Abstention for Imbalanced High-Stakes Decision Support: A Multi-Domain Benchmark](http://arxiv.org/abs/2607.27143v1) | Manpreet Singh, Akshatha Srikantha et al. | High-stakes decision systems in credit scoring, fraud detection, healthcare, and industrial safety require reliable uncertainty quantification under severe class imbalance and asymmetric error costs. Standard marginal conformal prediction (CP) provides valid overall coverage guarantees; however, we show that it severely under-covers rare, costly minority classes, with minority-class coverage dropping to as low as 0.5% on certain datasets. To characterize and address this limitation, we conduct a comprehensive benchmark comparing marginal CP, class-conditional (Mondrian) CP, and cost-controlled abstention mechanisms across 15 real-world imbalanced tabular datasets, 7 classification models, 3 probability calibration techniques, and 10 random seeds, resulting in 3,150 experimental runs. Our results show that Mondrian CP restores valid minority-class coverage, achieving an average minority-coverage improvement of 61.7 percentage points over marginal CP (p < 1e-80). Furthermore, combining Mondrian CP with cost-controlled abstention significantly reduces expected decision cost compared with standard decision boundaries, confidence-based rejectors, and risk-controlled rejectors under realistic human review budgets. We further quantify dataset-specific break-even thresholds at which deferring ambiguous instances to human experts becomes cost-effective. These findings provide practical guidance for deploying distribution-free, cost-aware uncertainty quantification in high-stakes decision support systems. |
| 2026-07-29 | [Controlled Experiments on Lane Changing by Transitional Autonomous Vehicle: Dataset and Behavioral Insights](http://arxiv.org/abs/2607.27085v1) | Abhinav Sharma, Md Abdullah Al Hasan et al. | This paper presents the North Carolina Transitional Autonomous Vehicle Lane-Changing (NC-tALC) dataset and uses it to characterize mandatory lane-changing behavior of transitional automated vehicles (tAVs). It quantifies the evolution of lead--lag gaps throughout the lane-change process and examines how potential collision risk develops during the maneuver. A controlled field experiment comprising 78 mandatory lane-change trials was conducted on a public roadway in Apex, North Carolina. Four instrumented vehicles created repeatable traffic conditions while varying the lane changer's initial position within the candidate target gap. High-resolution RTK-GNSS/INS trajectories were processed to identify key timestamps, calculate lead, lag, and lane-change gaps, and estimate interactions using time-gap- and speed-based surrogate safety measures. Despite substantial differences in initial conditions, lead and lag gaps consistently converged toward a relatively narrow range near lane crossing. Potential collision risk increased as the maneuver progressed, peaked near physical lane entry, and was dominated by interactions with the target-lane leader. Lane-change completion did not necessarily coincide with the disappearance of collision risk. This study provides one of the first controlled empirical characterizations of the complete mandatory lane-change process of tAVs using repeatable public-road experiments. The NC-tALC dataset supports analysis of behavioral and safety evolution throughout the maneuver rather than only at the gap-acceptance instant. The dataset and findings provide empirical benchmarks for evaluating automated lane-changing behavior, calibrating behavioral models, and validating simulation and safety assessment methods for mandatory lane-change scenarios. |
| 2026-07-29 | [On-Policy Distillation for LLM Safety: A Routing Approach to Template-Robust Realignment](http://arxiv.org/abs/2607.27081v1) | Yongjian Guo, Wanlun Ma et al. | Fine-tuning is the dominant paradigm for specializing large language models (LLMs), yet it exposes a critical vulnerability: malicious data providers can embed harmful behaviors into downstream corpora, creating models that retain professional skills while violating human values on demand. Existing safety-realignment defenses often fail in practice due to three key limitations: they frequently cause catastrophic forgetting of specialized skills; their effectiveness collapses when the defender cannot observe the attacker's prompt template; and successfully realigned models remain susceptible to re-jailbreaking via simple system prompt switches. To address these challenges, we propose Routing-based On-Policy Distillation (ROPD), a novel realignment framework that models the divergence between aligned and compromised output probability distributions rather than fitting specific prompt templates. We conduct extensive experiments comparing ROPD against four state-of-the-art baselines across three datasets and three base models with varying alignment strengths. Our results demonstrate that when baseline defenses face template mismatches, often accompanied by severe degradation in downstream task performance. In contrast, ROPD substantially mitigates template-mismatch risks, maintaining superior robustness in both defense effectiveness and capability preservation. While our analysis indicates ROPD is not entirely immune to template shifts, its performance degradation is negligible compared to existing methods, establishing a new standard for robust LLM realignment. |
| 2026-07-29 | [Deductive Verification for Earliest Deadline First Scheduler Implementations](http://arxiv.org/abs/2607.26927v1) | Daniel Kuhse, Junjie Shi et al. | Real-Time Operating Systems (RTOSes) rely on scheduler implementations to provide predictable task execution. For safety-critical systems, it is therefore not sufficient to reason only about the abstract scheduling policy; the concrete implementation must also preserve the intended scheduling semantics. This is particularly challenging for Earliest Deadline First (EDF) scheduling, because EDF introduces dynamic, deadline-derived priorities that are often realized by reusing kernel infrastructure originally designed for fixed-priority scheduling. In this work, we formalize EDF correctness through three essential properties that any implementation of the Earliest Deadline First (EDF) scheduler must satisfy. Based on these properties, we propose a framework utilizing deductive verification, that applies to any EDF-based scheduler realization. We instantiate the framework in Frama-C/ACSL and apply it to three structurally different EDF scheduler realizations: RTEMS 5, RTEMS 6, and an EDF extension of FreeRTOS. |
| 2026-07-29 | [Low-Temperature Co-Fired Ceramics for a Sustainable Planar Plasma Jet with Homogeneous Plasma in Large Treatment Areas for Biomedical Applications](http://arxiv.org/abs/2607.26916v1) | Ivan Gomez Ho, Hua-Lin Chen et al. | This study presents a portable planar argon-based plasma jet designed for large-area biomedical applications, with an emphasis on uniform discharge, stability, and safety. The device incorporates interchangeable rear and side inlet gas adapters with restriction plates and channels to equalize gas flow, while electrodes are encapsulated in low-temperature co-fired ceramics to reduce degradation and arcing during repeated operation. Flow simulations were conducted for multiple channel configurations to ensure laminar gas distribution and uniform plasma generation. Device performance and safety were evaluated using the kinPen MED as a reference standard. Electrical characteristics, optical emission, discharge uniformity, temperature, gas velocity, ozone generation, UV irradiance, and leakage current were systematically measured. The plasma exhibited stable voltage, current, and power after an initial warm-up period. Temperatures stabilized within minutes, with the rear-inlet configuration demonstrating lower operating temperatures due to higher outlet gas velocity. Ozone levels remained below established safety limits, while UV exposure constrained allowable treatment times. Leakage current decreased with increasing distance and approached safety thresholds at short separations. These results demonstrate that the proposed planar plasma jet provides stable, uniform plasma delivery while meeting key safety requirements, supporting its potential for clinical and biomedical use. |
| 2026-07-29 | [BioVLN: A Simulation Platform for Visual Language Navigation in Biomedical Laboratories](http://arxiv.org/abs/2607.26914v1) | Zhe Liu, Quan Lu et al. | Biomedical laboratory robots must navigate to instruments before performing experimental procedures. Existing embodied navigation platforms are designed for household environments and treat a target as an object center or an arbitrary nearby position. This representation is inadequate for laboratory instruments, which must be approached from their operating side while maintaining safe clearance from surrounding equipment. We introduce BioVLN, a simulation platform for developing and evaluating visual-language navigation agents in biomedical laboratories. BioVLN represents each instrument with three regions: its physical body, a surrounding clearance region, and an operation area in front of the usable side. This model is applied consistently to scene generation, target placement, navigation evaluation, and safety analysis, so success depends on reaching a position from which the instrument can be accessed. BioVLN supports procedural scene generation and manually designed environments, producing 47 scenes and 1667 episodes. Standardized navigation and reinforcement-learning interfaces enable trajectory collection and policy training. Experiments show that geometric exploration reaches 74.4--87.5% success, while sampling multiple valid positions in the operation area improves success to 83.3--92.5% and reduces unsafe proximity. |
| 2026-07-29 | [CinemaTraj: Composing Atomic Camera Trajectories for 3D Scenes with LLM Agents](http://arxiv.org/abs/2607.26910v1) | Qianru Li, Xuyang Chen et al. | Automatically generating cinematically expressive camera trajectories through 3D scenes from natural language descriptions is a challenging task of high practical value, with applications ranging from real-estate advertising to virtual tour creation. Existing methods either lack true 3D spatial awareness by relying on 2D image priors, or treat trajectory generation as a geometric path planning problem divorced from cinematographic semantics. We present CinemaTraj, a framework that reframes camera trajectory planning as a language-grounded spatial reasoning problem. Given a set of RGB-D images and a user prompt, CinemaTraj equips an LLM agent with a structured 3D scene graph: the agent decomposes the prompt into a sequence of atomic cinematographic movements (dolly, orbit, crane, pan, tilt, zoom, arc). Each movement is instantiated via a novel parametric trajectory representation that is both cinematographically expressive and optimizable for collision avoidance. The scene graph acts as a structured spatial prior, grounding the agent's reasoning in accurate geometric and semantic knowledge of the environment. CinemaTraj further generates synchronized voiceover and subtitles aligned with camera motion, producing narrated cinematic video outputs. We evaluate CinemaTraj on real-world ScanNet++ environments, and show that it produces prompt-faithful, collision-free trajectories with high cinematographic quality, outperforming existing approaches on prompt alignment, trajectory quality, and safety metrics. |
| 2026-07-29 | [ToxScreen: Detecting Whether an LLM Has Been Poisoned](http://arxiv.org/abs/2607.26849v1) | Anthony Hughes, Nicole Xing et al. | As large language models (LLMs) are deployed in high-stakes domains, adversaries may poison training data to implant backdoors: hidden triggers that covertly manipulate model behavior at inference time. We ask whether a defender can recover such a trigger under realistic affordances, namely white-box access to the weights and knowledge of the behavior of concern, but no training data, no trusted reference model, no knowledge of the trigger, and no certainty that the model is poisoned. To evaluate whether a defender can recover such a trigger under realistic settings, we release ToxScreen, a benchmark of roughly 800 backdoored models spanning attack objectives, trigger mechanisms, poisoning rates, model scales, and backdoor training mechanisms. We also assert that the backdoors are high-quality: they achieve high attack success rates, generalize to unseen harmful inputs, and preserve clean-task performance. Scoring recovery of the planted trigger, we find that gradient-based prompt optimization fails in recovery, whereas a token look-up that ranks candidates by attack-success rate recovers the trigger wherever the backdoor is effective. To understand this more, we study the relationship between attack behaviors and the weights of an LLM. We find a phenomenon whereby backdoors operate via different mechanistic strategies than jailbreaks, allowing defenders to filter jailbreaks. Finally, no method reliably surfaces every backdoor, but a broadly jailbreakable model is itself anomalous, a useful signal even when the exact trigger is not recovered. We release all models and evaluation code |
| 2026-07-29 | [Forecasting Trajectory-Level Safety Risks in Black-Box Multi-Turn Interactions](http://arxiv.org/abs/2607.26820v1) | Shi Lin, Peng Qian et al. | As large language models (LLMs) evolve from standalone assistants into autonomous agents, ensuring their safety requires shifting beyond pointwise risk assessment to understand how risks emerge and unfold over long-horizon trajectories. In multi-turn interactions, malicious intent can be decomposed across seemingly harmless turns and gradually reconstructed through interaction trajectories, eventually resulting in safety failures. Existing safeguards remain largely reactive, detecting manifested violations while lacking the ability to predict latent risk evolution and enable preemptive prevention. To address this limitation, we propose Recast, a safety risk forecasting framework that advances LLM safeguarding beyond turn-level violation detection to trajectory-level risk prediction. Recast first retrieves risk-relevant evidence from both short-term dialogue progression and long-term historical context via a dual-scale trajectory view. It then models compositional risk evolution by capturing the current risk configuration and its temporal dynamics. Finally, a causal temporal encoder learns latent risk evolution patterns and predicts the distribution of future risk emergence turns. Extensive experiments across 7 risk categories show that Recast predicts 88.3% of future safety failures with an average lead time of 2.41 turns, while maintaining a false alarm rate of 12.3%, showcasing the effectiveness of trajectory-level forecasting in identifying emerging risks before safety violations occur. |
| 2026-07-29 | [Risk-Aware Motion Planning with Learned Trajectory Primitives and Probabilistic Safety Assessment](http://arxiv.org/abs/2607.26802v1) | Marc Kaufeld, Dian Zhuang et al. | This paper presents a radial basis function network (RBFN)-informed motion planning framework for safe and efficient urban autonomous driving. The proposed approach combines RBFN-based candidate trajectory generation with an analytic collision probability assessment and optimization-based trajectory refinement. The network learns jerk-minimal trajectories, enabling the MPC to operate within a reduced and dynamically consistent search space. Candidate motion primitives are selected based on an accurate probabilistic risk measure. This design decreases solver complexity while preserving safety and constraint satisfaction. The framework is evaluated in numerous urban driving scenarios. Results demonstrate improved risk awareness and fewer vehicle-limit violations compared to benchmark methods. The proposed approach integrates learning-based trajectories into optimization-based motion planning, thereby ensuring safety and interpretability. |
| 2026-07-29 | [Physically Real-time Infrared Attack against Optical Flow Estimation Networks](http://arxiv.org/abs/2607.26651v1) | Shen You, Wei Jiang et al. | With the promising performance of deep neural networks on image-based tasks, different real-world applications such as autonomous driving and motion detection have become increasingly mature and relevant to human lives. In particular, Optical Flow Estimation Networks (OFENs), as upstream models, play a critical role in different domains. Its outputs are heavily assumed and adopted for different downstream tasks, and it is essential to test its robustness to prevent safety accidents. We present an approach for real-time attacks on OFENs in the physical world, leveraging infrared lights for their stealthiness. By generating a large number of Adversarial Examples in advance, our approach computes AEs in real time and dynamically displays them, which allows our method to facilitate precise and targeted attacks without modifying the victim system. Unlike previous digital-to-physical attack techniques, our method directly attacks victim models within the physical world, thereby overcoming the limitations associated with the ineffectiveness of AEs. Experimental results demonstrate the efficacy of our approach in compromising OFENs across diverse lighting conditions, varying object motion velocities, and different object placements, ultimately impairing the network's ability to accurately estimate optical flow. |
| 2026-07-29 | [Borrowed Strength: Best-of-N Search over a Code EncodingBreaks Self-Check Jailbreak Defenses](http://arxiv.org/abs/2607.26639v1) | Haoyu Zhang, Shibo Zheng et al. | A self-check defense asks the target model to assess a request before answering it; SAGE, the strongest published instance, reports an average 99% defense success rate. We show it can be breached by composing two attacks that are individually harmless against it: an established code-completion encoding and an established best-of-N search, neither of which exceeds 4.7% of behaviors alone. Composed, with the search budget spent on the encoding, they reach 67/22/15% across three open targets, and the effect persists on a 70B target. We then explain the composition rather than only reporting it. First, a self-check defense borrows its strength from the target: SAGE does not detect the attack, it asks the model to, and the four targets convert that request into an explicit refusal between 32% and 97% of the time, which orders the spread in defended coverage even though undefended reach is near-identical. Second, which attack survives is decided by the type of defense, and it inverts: against transform defenses the code encoding retains far more of its undefended reach than the character search, while against gate defenses the ordering flips. We account for this with the number of independent probes an attack delivers to a defense's decision boundary. Finally, we report a validity defect we found and repaired in our own pipeline, a deterministic attack under greedy decoding has no best-of-N variation channel at all, and give the one-line diagnostic that detects it. All claims rest on 310,000 generations scored by a human-validated judge. |
| 2026-07-29 | [Recover, Decode, Reguard: Guard-Agnostic Defense Amplification againstEncoded VLM Jailbreaks](http://arxiv.org/abs/2607.26574v1) | Haoyu Zhang, Zhuoxi Wang et al. | Safety classifiers ("guards") are the dominant black-box defense for vision-language models, yet they judge an input's surface form, not its meaning: a harmful request re-encoded as set theory, formal logic, a rare language, code, or an image of text slips past a guard that would block it in plain language -- the decode gap. The natural fix is a guard-agnostic recover-and-decode amplifier that transcribes image content and restates encoded text into its plain payload before the guard, so any off-the-shelf classifier can screen the true request. We build this amplifier and evaluate it against the attacker's best case: an ensemble of eleven attacks, scoring a behavior as broken if any succeeds (best-of-suite, following AutoAttack) -- rarely reported for jailbreak defenses, yet ~3.5x the per-attack mean. This exposes our central finding: an empirical safety-utility ceiling for the non-iterative recovery defenses we evaluate, across five guards and two target VLMs. The amplifier only partly closes the gap -- the undefended ensemble breaks 89-91% of behaviors, and the best guard-plus-amplifier still leaves 63-65% -- and its gain over the guard alone is significant in only four of ten guard-target pairs. It is guard-agnostic at the interface, but not uniformly so in effect. A modular reguard layer closes much of the residual, yet drives benign over-refusal to 81-92% for well-calibrated guards; the one laxer guard that stays usable never reaches deployable safety (48% ensemble ASR). No configuration we evaluate reaches both low attack-success and low over-refusal, for the pipeline we study and for representation-shifting attacks -- encodings and cross-modal renders that leave a legible payload, not pixel- or embedding-space attacks. We contribute the amplifier, an ensemble evaluation that makes the trade-off visible, and a map of where recovery-based VLM defense works and where it does not. |
| 2026-07-29 | [Prosody-driven Jailbreaks in Audio LLMs: A Controlled Study and Mechanistic Analysis](http://arxiv.org/abs/2607.26541v1) | Jiachen Qian, Junyu Li | Audio-capable foundation models enable end-to-end spoken interaction, but they also introduce safety risks beyond transcript content. It remains unclear how much jailbreak capability can arise from matched-text variation in speech delivery rather than from lexical rewriting or broader style transfer. We study this question by holding transcript content fixed and varying six speech-delivery presets whose acoustic attributes may co-vary. We present PJ-Break, a black-box evaluation protocol with presets targeting arousal, authority, and speaking rate, together with AdvAudio-Prosody, a 600-sample benchmark with acoustically verified attributes. On the exact post-QC Qwen2-Audio panel, the Q=1 Panic (38/95), Anger (35/95), and Fast (32/95) presets are all well above Neutral (4/95). The fixed six-query pool covers 44/95 Qwen2-Audio seeds and 15/95 GPT-4o seeds and exceeds a matched-budget StyleBreak reimplementation (27/95) on Qwen2-Audio. A same-voice pool excluding the confounded Commanding condition still reaches 40/95, and a retained-panel ablation shows emotional-delivery audio alone (44/95) is far more effective than emotional text alone (11/95). Exploratory surrogate diagnostics and pilot mitigation observations are secondary, non-core analyses. Overall, matched-text speech delivery should be treated as a first-class factor in Audio LLM safety evaluation |
| 2026-07-29 | [Online Monitoring and Risk Assessment of Non-Cooperative UAVs via STL-Aware Adaptive Fusion Kalman Filtering](http://arxiv.org/abs/2607.26527v1) | Xinhao Yan, Ruige Yang et al. | This paper considers the problem of online state estimation and predictive risk assessment for non-cooperative unmanned aerial vehicles (UAVs) in the presence of asynchronous heterogeneous sensing and uncertain motion modes. To address this problem, a unified estimation and safety-assessment framework is developed by integrating an interacting multiple-model multi-rate Kalman filter with signal temporal logic (STL). The proposed framework enables simultaneous low-level state tracking and high-level safety reasoning within a common recursive architecture. Its main contribution is an STL-aware time-varying mode transition mechanism that updates model probabilities online using robustness measures induced by formal safety specifications. By embedding safety semantics directly into the mode inference and estimation process, the method improves responsiveness to maneuver variations, sensing asynchrony, and evolving threat patterns. Based on the estimated state distributions, the framework further generates multi-step state predictions and probabilistic reachable sets, which are used for finite-horizon safety evaluation and risk-triggered warning generation. Consequently, the proposed method provides not only estimates of the current target state, but also early indication of unsafe behaviors before they become fully observable. Finally, experimental results obtained from a real-time UAV monitoring platform show that the proposed approach improves estimation accuracy and produces earlier and more informative safety warnings, demonstrating its effectiveness for real-time UAV surveillance and safety monitoring applications. |
| 2026-07-29 | [EgoSafe: A First-Person Mobile-Captured Benchmark for Visual Safety Understanding](http://arxiv.org/abs/2607.26518v1) | Yuyun Chen, Tianao Li et al. | Reliable visual safety understanding in real-world scenarios demands more than just object recognition; it requires causal reasoning under epistemic uncertainty. While Large Vision-Language Models (LVLMs) demonstrate impressive semantic alignment on standard benchmarks, they often struggle to distinguish between superficial correlation and genuine forensic logic when grounded in the dynamic, partially observable nature of first-person experiences. Existing evaluations, dominated by third-person surveillance footage and binary classification metrics, fail to expose this cognitive gap. To address this, we introduce EgoSafe-Bench, a benchmark specifically designed to probe forensic reasoning in egocentric safety scenarios. It comprises 12,000 unique evaluation samples, generated by pairing each of the 3,000 video clips with a QA chain governed by our proposed Hierarchical Reasoning Evaluation (HRE) protocol. Unlike standard benchmarks, HRE mandates a rigorous reasoning trajectory from initial feature anchoring to blind-spot deduction and intent inference, thereby enforcing logical consistency and penalizing shortcut-based predictions.Extensive evaluations of state-of-the-art LVLMs (e.g., Qwen3-VL, Gemini, VideoLLaMA 3) reveal a significant perception-reasoning decoupling: models often achieve high descriptive scores but exhibit notable fragility in causal reasoning and logical closure. Our work provides both a challenging dataset and a systematic evaluation framework to foster the development of logically robust video understanding systems. |
| 2026-07-29 | [Safety-Gated Autoscaling: A Multi-Layered Defense Architecture for Kubernetes Vertical Resource Optimization](http://arxiv.org/abs/2607.26503v1) | Azra Karakaya, Erva Şengül et al. | Kubernetes is the standard platform for orchestrating containerized applications, yet resource management remains difficult. To stay safe, engineers over-provision CPU and memory, leaving reserved but unused capacity that is the main source of wasted cost. The built-in Horizontal and Vertical Pod Autoscalers are reactive: they act only after a threshold is crossed, which causes lag, over-provisioning, and can mask software defects by granting a leaking workload more memory. Predictive autoscalers focus on improving forecasting accuracy or run inside proprietary infrastructure, and anomaly detection is used only to alert, never to block a harmful action. The Intelligent Cluster Optimizer is an open-source Kubernetes operator that right-sizes container workloads with safety as a first-class concern. Its central contribution is a five-layer safety pipeline where a memory-leak detector, based on linear regression with R^2 scoring, acts as a blocking gate: if a leak is detected the recommendation is rejected, so the optimizer never hides a bug by enlarging a broken container. The pipeline combines SLA monitoring, a circuit breaker, HPA/PDB conflict detection, and a policy engine, with rollback and dry-run mode for human approval. Recommendations are produced by percentile analysis and Holt-Winters forecasting, balanced through multi-objective Pareto optimization at the per-container level. We validated the system with 1118 automated tests at 80.3% coverage and a live deployment on Google Kubernetes Engine, where right-sizing produced estimated cost savings of 20--40% in what-if projections and the leak gate reached 83% detection accuracy. |
| 2026-07-29 | [Conformal Changepoint Localization and Root Cause Analysis with Corrupted Observations](http://arxiv.org/abs/2607.26481v1) | Seunghun Yu, Meiyi Zhu et al. | Detecting when the statistical behavior of an engineered system changes, and identifying which component is responsible, are core problems in the monitoring of telecommunication networks, robotic platforms, security infrastructure, and multi-agent systems. In safety- and mission-critical deployments, such decisions must be accompanied by statistical reliability guarantees rather than by point estimates alone. Conformal changepoint localization (CONCH) and conformal root cause analysis (CROC) meet this need by returning confidence sets that contain the true changepoint, or the true root-cause stream, with a user-specified probability, without parametric assumptions on the data-generating process. In practice, however, observations are frequently corrupted, e.g., by outliers, sensor faults, or adversarial perturbations. While the finite-sample coverage of these procedures is preserved under contamination, the resulting confidence sets can become uninformatively large. Adopting a Huber-type contamination model, this paper proposes weighted CONCH (W-CONCH) and weighted CROC (W-CROC), which downweight observations that are likely to be corrupted with the goal of reducing confidence set size when data may be corrupted. The weighting mechanism, derived from a formal bound on the unknown corrupted data densities, leverages pre-existing second-order classifier-based uncertainty signals, such as those produced by evidential deep learning or Bayesian learning. W-CONCH and W-CROC are further generalized by introducing a meta-learning procedure for the weights that optimizes a differentiable surrogate of the confidence set size. Experiments on image-based and real-world changepoint and root-cause benchmarks show that uncertainty-based weighting substantially reduces confidence set size while maintaining the target coverage. |
| 2026-07-29 | [FleetScape: A Mixed Reality Sandtable for Spatial Supervision and Control of Scalable Drone Fleets](http://arxiv.org/abs/2607.26423v1) | Peisen Xu, Jérémie Garcia et al. | As autonomous drone deployments scale from individual units to coordinated swarms, the human operator's role shifts from direct piloting to high-level supervision. Current interfaces often treat multi-drone control as a scaled-up version of single-drone operation. We instead investigate how reframing fleet supervision as spatial interaction can better support the spatial, temporal, and safety demands of complex missions. We present FleetScape, a Mixed Reality (MR) sandtable system that externalizes layered real-time mission, safety, and environmental data while enabling fluid transitions between manual intervention and autonomous supervision. We developed a high-fidelity building inspection simulation that generates and streams synchronized multi-drone and environmental data for MR visualizations. We used this prototype to conduct a user study with six experienced drone pilots managing fleets of up to 15 drones. Our findings show that FleetScape supports situational awareness through layered spatial representations and clarifies control mode transitions. However, a limit to situational awareness was observed as fleet size increases, leading to different supervisory strategies. Finally, we derive design implications for supporting scalable drone fleet supervision. |

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



