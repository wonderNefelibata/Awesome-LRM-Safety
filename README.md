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
| 2026-11-04 | [BEAKER: An Expert-Curated Benchmark for Embodied Brains in Self-Driving Chemical Laboratories](http://arxiv.org/abs/2611.06911v1) | Fei Lin, Tengchao Zhang et al. | Self-Driving Chemical Laboratories (SDCLs) are moving chemical experimentation toward embodied automation, where machines must perceive laboratory scenes, reason about experimental states, and act under physical, procedural, and safety constraints. Although Multimodal Large Language Models (MLLMs) are increasingly considered as embodied brains for such systems, existing laboratory benchmarks mainly focus on safety, anomaly detection, or general scientific understanding, leaving their execution-oriented embodied capabilities underexplored. To fill this gap, we introduce BEAKER, a Benchmark for evaluating Embodied Actionability, Knowledge, Experimentation, and Reasoning in SDCLs. BEAKER contains 1,000 expert-curated question-answer samples across 11 image and video task types, covering key judgments required for chemical laboratory execution, including affordance grounding, trajectory planning, equipment, state, and safety recognition, apparatus setup, operation and step reasoning, and temporal procedure reasoning. All questions and annotations are manually created and reviewed by chemistry and embodied Artificial Intelligence (AI) experts, without AI-assisted generation or annotation. We evaluate 28 general-purpose, science-oriented, chemistry-specialized, and embodied MLLMs in an offline, single-turn setting. Results show that current models can follow coarse experimental procedures in videos, but still struggle to ground actions into operable regions and feasible trajectories, to recognize fine-grained laboratory apparatus, states, and safety norms, and to order steps in mid-length procedures. BEAKER thus reveals a clear gap between coarse procedural understanding and the precise, action-grounded perception required for laboratory execution, providing a diagnostic benchmark for future MLLMs in SDCLs. |
| 2026-11-04 | [Persona Perspectives: Adopting Viewpoints Reorganizes Representations in LLMs](http://arxiv.org/abs/2611.06848v1) | Austin C. Kozlowski, Avital Mintz et al. | Common methods for monitoring features in large language models (LLMs) such as linear probes and sparse autoencoders (SAEs) assume that features maintain a consistent representation across contexts. LLMs, however, can flexibly adopt different perspectives or "personas" that suggest different underlying systems of conceptual relations. In this paper, we examine whether prompting an LLM to adopt a persona alters the representational geometry of the model's latent space. We extract representations of over 9,000 common words and 33 semantic axes from models adopting liberal and conservative political personas and compare these against two neutral baselines. We find that persona conditioning systematically alters both the projections of words onto semantic axes and the alignments among the axes themselves in directions consistent with the specified ideology. We then test the implications of this reorganization for two safety-relevant features: truthfulness and morality. Linear probes, SAE-based classifiers, and classifiers built from manually selected word representations, all trained without a persona, produce divergent results when applied to models conditioned on political personas, producing politically slanted ratings of truth and morality on controversial statements. These findings suggest that internal representations, as they are conventionally extracted, can exhibit different geometric relations when the model adopts personas, potentially undermining monitoring efforts. |
| 2026-11-04 | [Before You Continue: Evaluating Time-Sensitive Safety Awareness in Conversational Assistants](http://arxiv.org/abs/2611.06839v1) | Peidi Dong, Yunze Xiao et al. | Large language model (LLM) assistants increasingly help users with everyday activities that involve time-dependent constraints. However, users may not mention these constraints in later requests, requiring assistants to recognize and adjust their advice accordingly. To evaluate such capability, we introduce TASC (Temporal Action Safety in Conversation), a benchmark that evaluates whether LLMs apply time-dependent constraints when responding to conversational requests. TASC consists of 3,418 inputs derived from 128 templates, along with a separate set of 33 scenarios designed to compare models' ability to calculate elapsed time with their ability to apply that information. Across 13 models, assistants account for time-dependent constraints in only 30.5% of cases where the proposed action violates a rule, while rarely warning users unnecessarily. More importantly, models fail to apply these constraints in 62.1% of cases despite correctly calculating the elapsed time. Explicitly providing the timing resolves only about half of these failures, revealing a gap between temporal reasoning and its application in conversational advice. |
| 2026-11-04 | [Excitability of High-Frequency Positive Feedback](http://arxiv.org/abs/2611.06822v1) | Gustave Bainier, Arthur Fyon et al. | What is the threshold of an excitable system, i.e. the criterion that distinguishes rest from excitation? Previous research has shown that this question is far from trivial to answer. This paper studies the threshold of a simple excitable primitive in which excitability arises from the positive feedback interconnection of a first-order high-pass filter and a static, strictly monotone and strictly convex activation function. This threshold is a simple inequality on the difference between the instantaneous input and the motif's internal memory. While this result may help neuro-engineers characterize spiking in biological and artificial neurons, we argue that excitability is not always a desirable property, and that our threshold may also be relevant to safety considerations in fragile systems with rate-induced tipping points. We illustrate our framework with three examples: the FitzHugh-Nagumo model of neuronal dynamics, a model of stop-loss cascades in the stock market, and a social contagion model. |
| 2026-11-04 | [Measuring Risk, Not Harm: Detecting Psychological Harm Risk Factors in LLM Conversations](http://arxiv.org/abs/2611.06806v1) | Nil-Jana Akpinar, Eugenia Kim et al. | Psychological harms from interactions with large language models cannot be monitored directly because they may emerge over repeated interactions and depend strongly on subjective individual user context. We propose measuring risk, not harm: detecting observable AI behaviors that constitute risk factors for psychological harm at the AI response level and aggregating these signals to support longitudinal monitoring. We operationalize eight such behaviors and construct a synthetic multi-label corpus of 47,062 AI responses and 373,873 response-factor labels, with targeted human annotations for validation and evaluation. We then compare trained embedding-based classifiers, direct LLM judging, and a classifier--judge cascade. All detectors are model-agnostic and operate only on LLM outputs, requiring no access to internal model states. We apply these detectors to several out-of-distribution human-AI conversation data sets from prior research. On held-out human-labeled data, the best cascade achieves macro precision of 0.86, recall of 0.76, and F1 of 0.79. On WildChat, it reduces LLM-judge calls by 97.1\% relative to direct judging. On DelusionEval, risk-factor detections cluster within conversations. Together, our results show that response-level behavioral risk-factor detection can provide a scalable measurement layer for longitudinal psychological-safety monitoring without conflating observable AI behavior with latent user harm. |
| 2026-11-04 | [ConDR: Conservative Reachability through Neural Hamilton-Jacobi Subsolutions](http://arxiv.org/abs/2611.06802v1) | Kendra Givens, Taylor T. Johnson et al. | Hamilton-Jacobi (HJ) Reachability identifies the unsafe states of a dynamical system from which it can be forced into failure under worst-case disturbance. However, classical grid-based solvers scale exponentially with state dimension. DeepReach improves scalability by replacing the grid with a neural value function trained on the HJ-Isaacs (HJI) variational inequality (VI), but penalizes both sides of the residual equally. This allows the learned value to lie above the true one and unsafe states to be misclassified as safe. We propose Conservative DeepReach (ConDR) which aims to learn a subsolution of the HJI VI such that the learned unsafe set contains the true one. ConDR enforces the obstacle branch and terminal condition by construction, pushes the HJI residual to a safety margin $ε$ below zero with a one-sided loss, and prevents the degenerate solution of classifying everything as unsafe. We prove that the exact solution of the margin-shifted VI that ConDR trains toward is a subsolution of the original VI so its unsafe set contains the true one and its value lies within $ετ$ of the true value function. We compare against DeepReach and MADR in the established systems of Air3D, Air6D, a 9D multi-vehicle system, and a 13D quadrotor. ConDR reports the lowest rate of unsafe state misclassified and the lowest closed-loop collision rates with at most a small increase in intervention. On Air3D and Air6D, ConDR tolerates model mismatch that scales with $ε$. At matched false-negative rates, ConDR's unsafe sets are smaller than the baselines showing the safety is not obtained by trivial over-approximation. |
| 2026-11-04 | [Cognitive and Behavioural Effects of Haptic Feedback for ROV Teleoperation in Hazardous Environments](http://arxiv.org/abs/2611.06689v1) | Joshua Brown, Ildar Farkhatdinov et al. | Teleoperating a Remotely Operated Vehicle (ROV) in complex underwater environments presents unique challenges. Dynamic, three dimensional obstacles and unseen water currents pose real risks to expensive robotic hardware in environments in which recovery can be next to impossible. This paper explores the cognitive effects of haptic feedback to improve vehicle safety in such environments. An in-water study is analysed with a ROV exposed to controlled cross-currents, with environmental information presented via a soft and vibrating haptic touchpad.   Results demonstrate that while encountering an unseen cross-current significantly increases an operator's physical and temporal demand, the introduction of haptic feedback mitigates these factors and improves self-assessed performance. Interestingly, the haptic condition also produced the highest levels of mental demand and operator frustration. Despite this increased mental workload, operators drove the ROV significantly faster when receiving haptic cues compared to the current-only condition, indicating a higher level of user confidence. Ultimately, these findings suggest that while haptic feedback may initially require more cognitive processing to interpret unseen hazards, it successfully enhances operator awareness and promotes more confident navigation in hazardous environments. |
| 2026-11-04 | [Detecting Silent Action-Interface Mismatch with Calibrated Relational World Models](http://arxiv.org/abs/2611.06653v1) | Rui Zhu, Xurui Li et al. | A robot action can satisfy shape and range constraints yet carry the wrong semantics: axes may be permuted, signs inverted, rotations rescaled, or gripper polarity reversed. We study learned state-action-effect relations as offline contracts for detecting these mismatches and ranking their likely fields. Clean held-out transitions calibrate alarms; a fixed inverse vocabulary supports closed-set localization. On ten LIBERO Spatial tasks comprising 432 source episodes, a capacity-controlled comparison at 2.5% empirical clean false-positive rate gives 46.1% recall for a ridge forward residual, 61.0% for a four-component joint Gaussian mixture, and 65.7% for the exact conditional of that same mixture. The joint score retains a 14.9 percentage-point advantage over ridge, while conditioning improves recall by a further 4.7 points. Joint scoring instead has higher localization: 95.8% versus 94.6%. These matched-rate comparisons are retrospective diagnostics, not deployment calibration guarantees. Earlier Object/Goal results broaden tasks within the LIBERO convention; DROID, all three HYDRA tasks, short temporal faults, small offsets, exact repair, and simulated mitigation fail their respective gates, while xArm contributes one task only. The evidence supports a bounded offline diagnostic and a detection-localization tradeoff, rather than joint-density superiority or cross-robot safety. |
| 2026-11-04 | [FOSCo-Net: Ego-centric Voxel Completion for Forest LiDAR Occlusions](http://arxiv.org/abs/2611.06588v1) | Iman Esfandiyar, Dominik Belter | Ground robots operating in temperate forests suffer from systematic LiDAR occlusions caused by tree trunks, creating persistent blind regions that degrade traversability estimation. In this paper, we propose FOSCo-Net, a self-supervised voxel completion framework that learns to reconstruct occluded geometry from a single scan using temporally accumulated observations as automatic supervision, and we evaluate cylindrical and Cartesian ego-centric grids with multiple decoder designs. FOSCo-Net consistently improves reconstruction quality and traversal safety, with a gated skip decoder achieving the most reliable performance and significantly reducing false obstacle predictions compared to baselines. |
| 2026-11-04 | [Addition by Subtraction: Toward Mitigating Fragile Safety Alignment in Large Language Models](http://arxiv.org/abs/2611.06553v1) | Hamid Dashtbani, Alexandra Dmitrienko | Large language models (LLMs) undergo safety alignment to prevent harmful generations, yet they remain vulnerable to jailbreak attacks. Recent work reveals a deeper structural weakness: safety alignment can be highly sparse, such that pruning a small number of safety-critical neurons or ablating a single refusal direction can substantially undermine aligned behavior. In this work, we take the first steps toward strengthening safety alignment such that it is less dependent on sparse model components. We propose Addition by Subtraction, which gradually removes identified safety-critical neurons during finetuning and uses preference optimization to re-establish safety-aligned behavior through other parts of the model. Our results show that the resulting models are substantially more robust to pruning-based attacks. Remarkably, these benefits generalize beyond the white-box attacks motivating our design: Addition by Subtraction improves robustness against other sparsity-based interventions, including refusal-direction ablation, as well as a broad range of black-box jailbreak attacks. The finetuned models also exhibit improved truthfulness while largely preserving general capabilities. Comprehensive experiments across four models from three vendors demonstrate consistent improvements in safety robustness, reducing attack success rates by up to 90% for some attacks. |
| 2026-11-04 | [Timed Games under Environmental Interference with Real-Time Objectives](http://arxiv.org/abs/2611.06450v1) | Andrea Manini, Matteo Rossi et al. | Classical Timed Games formulations may be unsuitable, or require substantial modeling effort, to capture complex interaction patterns between a controller and its surrounding environment, in which non-determinism must be resolved after the controller has chosen which action to perform. In this paper, we develop the theory of Timed CLTLoc Games (TCGs), a Timed Game variant designed to facilitate the modeling of such patterns. Unlike classical Timed Games formulations, TCGs partition locations rather than transitions, and use Constraint Linear Temporal Logic over clocks (CLTLoc) to specify the controller's objectives. We develop algorithms for solving TCGs and synthesizing winning controller strategies for games whose objectives include formal properties such as reachability and safety. For reachability objectives, our synthesis algorithm yields winning controller strategies that minimize the number of moves required to reach a goal state. Solving reachability TCGs is EXPTIME-complete, matching the complexity of solving reachability Timed Games. All of our algorithms operate backwards and are implemented in our C++ region-based library TARZAN, leveraging OpenMP for efficient parallelization. Experimental results demonstrate the feasibility of TARZAN. |
| 2026-11-04 | [Verified Implementation of the Next-gen Borrow Checking Algorithm for Rust via Abstract Interpretation](http://arxiv.org/abs/2611.06319v1) | Jinhua Wu, Yuting Wang et al. | Borrow checking is a central mechanism for Rust to ensure temporal memory safety. Its correctness is based on the intuition that the Aliasing XOR Mutability discipline guarantees that programs without borrow errors inherently could not access invalid references (i.e., pointers). However, existing work on formalizing borrow checking treats it either as a component of type systems or as symbolic execution describing resource transfer. None provides a direct explanation of the correctness of borrow checking algorithms such as Non-Lexical Lifetimes (NLL) or Polonius in the official Rust compiler that function more like static pointer analysis.   We take a first step toward solving this problem by formalizing Polonius, the next-generation borrow- checking algorithm for the Rust compiler, on a core subset of Rust. We prove its correctness by showing that any program accepted by Polonius does not access invalid references at runtime. The key observation is that, when borrow checking is performed lazily, Polonius could be treated as a forward abstract interpretation on abstract domains recording pointer aliasing (borrowing) relations. By providing a concrete semantics that formally describes the Aliasing XOR Mutability discipline, we prove the soundness of this abstract interpretation, from which we derive memory safety of checked programs. This formulation of Polonius is implemented and formally verified in the abstract interpretation framework of CompCert. The correctness proofs are generated by an agentic AI under human guidance using paper proofs as proof outline. To the best of our knowledge, this is the first comprehensive formalization and verification of a borrow checking algorithm as manifested in the Rust compiler. |
| 2026-11-04 | [Corridor-Based Monitoring and Emergency Braking Decision for Automated Vehicle Safety](http://arxiv.org/abs/2611.06287v1) | Junnan Pan, Vladislav Nenchev et al. | This paper addresses emergency-braking decisions for a redundant safety channel in automated vehicles. The channel is responsible for identifying hazards from sensor data and determining whether braking is necessary, given the constrained computational resources of embedded hardware. We present an emergency-braking approach that uses a predicted motion corridor to filter \ac{lidar} measurements and then decide whether the ego vehicle must brake. \ac{pmcm} constructs a curvature-aware boundary from the planned path and vehicle geometry so that the monitored region contains the full swept body on both straight and curved paths. \ac{gdeta} then decides whether the ego vehicle must command an emergency brake for objects inside this boundary, scoring geometric occupancy and temporal urgency independently. Hard gates permit braking when either source is conclusive; soft-OR arbitration combines partial evidence when neither gate is active. Consecutive-scan confirmation reduces sensitivity to transient measurement noise. Corridor coverage is evaluated in controlled collision simulations and recorded campus driving. Braking decisions are evaluated in Monte Carlo simulations of lead-vehicle braking and transverse-crossing encounters. |
| 2026-11-04 | [Understanding the Perception of Transliterated Offensive Social Media Posts Among Banglish and Hinglish Users](http://arxiv.org/abs/2611.06226v1) | Dhiman Goswami, Zhicong Lu et al. | Effective content moderation depends on understanding not just what harmful content looks like, but how users perceive and respond to it. When offensive language is written in transliterated forms such as Banglish or Hinglish (Bangla or Hindi in Latin script), it may be interpreted very differently than in native scripts, with direct consequences for whether users report, ignore, or recognize it as harmful. Yet little is known about how script choice relates to user perception, or how these perceptions vary across demographic groups. To investigate these perceptions, we conducted a user study with $310$ participants ($193$ Hindi, $60$ Bangla, $57$ Bangla-Hindi bilingual speakers) who evaluated offensive posts in native scripts and transliterations. Participants provided offensiveness scores ($4.03$ for Banglish, $3.86$ for Hinglish) to transliterated posts, yet reporting fell from $20.9\%$ (native - Bangla, Hindi) to $18.2\%$ (transliterated - Banglish, Hinglish). Ignoring remained the dominant response (range $25.38\%-44\%$ overall). Women reported at higher probability than men for native-script ($23.6\%$ vs. $17.7\%$) and showed almost identical reporting probability in comparison to men for transliterated posts ($17.5\%$ vs. $17.9\%$), and younger participants ($<40$) rated Banglish content as more offensive than older participants ($\geq 40$), (Overall offensive score: $3.99$ vs. $3.78$). These findings identify transliteration as critical blind spot for moderation, underscoring the need for linguistically sensitive approaches to online safety. |
| 2026-11-04 | [CoSafe: Cooperative Runtime Safety Assurance for Mixed-Traffic Autonomous Driving](http://arxiv.org/abs/2611.06204v1) | Dingji Wang, You Lu et al. | With the rapid development of autonomous driving systems (ADSes), real-world roads are evolving into mixed-traffic environments where autonomous vehicles (AVs) and other traffic participants (e.g., human-driven vehicles (HDVs), cyclists, and pedestrians) coexist and interact with each other. However, current ADSes still incur various safety violations in mixed traffic scenarios, raising a strong demand for runtime safety assurance frameworks that can continuously monitor hazards and prevent potential violations. Existing ego-centric assurance approaches are insufficient for mixed traffic, where violations often arise from locally reasonable yet jointly incompatible decisions among multiple vehicles and cannot be reliably resolved through independent takeover.   To bridge this gap, we propose CoSafe, a modular and ADS-agnostic cooperative runtime safety assurance framework for mixed-traffic autonomous driving to extend runtime safety assurance from ego-centric takeover to scene-level cooperative intervention. CoSafe leverages vehicle-to-everything (V2X) communication to enable roadside units (RSUs) to monitor scene-level hazards and jointly mitigate hazards involving multiple vehicles. We integrate CoSafe with seven state-of-the-art ADSes, and our evaluation on the InterDrive benchmark demonstrates that CoSafe improves the driving score of AVs by 70.03% on average, reduces safety violations by 87.80%, and accurately identifies unsafe AVs with an F3 score of 0.930, while satisfying real-time constraints. |
| 2026-11-04 | [Beyond Terminal Constraints: Reliable Robot Control with Learned Safe Sets](http://arxiv.org/abs/2611.06196v1) | Gianni Lunardi, Elias Fontanari et al. | Ensuring safety in advanced robot control methods such as Model Predictive Control (MPC) and Reinforcement Learning remains an open challenge. A common approach relies on control-invariant sets, yet their exact computation for nonlinear systems is generally intractable, and numerical approximations compromise theoretical guarantees. In this work, we show that safety does not need to rely exclusively on control invariance. We introduce novel classes of safe sets, each broader than classical control-invariant sets, and analyze corresponding MPC schemes that guarantee recursive feasibility by constraining different nodes of the prediction horizon rather than only the terminal state. Furthermore, when the safe set is an inner approximation of a suitable control-invariant set, our methods can ensure safety through a safe-abort strategy, even in the absence of recursive feasibility. We validate our approaches in extensive simulations on a 6-joint robot manipulator, showing fewer constraint violations than state-of-the-art methods while maintaining comparable tracking performance and computational efficiency. Hardware experiments further demonstrate the practicality of our schemes for real-world robotic applications. |
| 2026-11-04 | [SafeStart: Mitigating the Overlooked Safety Cost of Active Expert Reduction in MoE LLM Serving](http://arxiv.org/abs/2611.06147v1) | Erfan Baghaei Potraghloo, Saeid Shokoufa et al. | Mixture-of-Experts (MoE) language models send each token through a few of many expert subnetworks. Serving methods make these models faster by running even fewer experts per token, and they are validated on accuracy benchmarks. We show that these methods can weaken the model's refusal of harmful requests, and we introduce SafeStart, which largely restores refusal while keeping most of the speedup. In vLLM on one RTX A6000, keeping about two thirds of the selected experts speeds up offline decoding by 15 to 28% at batch size 8. We find that, at this setting, under a jailbreak that tells the model not to refuse, the share of harmful requests that the model answers rises by 18.4 percentage points on Qwen1.5-MoE, 9.5 on OLMoE, and 6.6 on DeepSeek-V2-Lite, the three of nine models where it rises significantly, while MMLU accuracy drops by less than two points. We discover that most of this increase arises while the model processes the prompt and generates its first few tokens. SafeStart therefore runs all experts for the prompt and the first eight generated tokens, and the reduced set of experts afterwards, through a fused kernel in vLLM. With experts removed, SafeStart keeps at least 85% of the offline decode speedup at batch sizes 8 and 32 and brings the change under this jailbreak to between -2.1 and +1.3 points for three expert-reduction methods on these three models. On 889 held-out requests, in an analysis fixed before generation, truncation raises harmful compliance by 5.8 to 21.4 points, and SafeStart leaves a rise of 0.6 to 2.6 points, non-inferior to the original model on two of the three models. Our results show that faster MoE serving need not cost safety: a serving system can keep most of the gain from running fewer experts by running all of them at the start of each response, and it should test refusal, not only accuracy, before it enables such a method. |
| 2026-11-04 | [KVGuard: A KV Cache Safety Checker for LLM Serving Systems](http://arxiv.org/abs/2611.06138v1) | Jonguk Cheong, Myungbin Hwang et al. | Modern LLM serving systems manage the KV cache through PagedAttention, a custom GPU allocator that partitions a single CUDA-allocated region into fixed-size physical blocks and maps each request to those blocks through a CPU-built block table. This block-based design exposes a previously unstudied class of memory safety bugs in the KV cache, including double-free, use-after-free, out-of-bounds access, uninitialized read, and permission violation, which silently corrupt outputs or break request isolation without ever crashing the system. Existing memory sanitizers cannot detect them, because they reason about CUDA-level addresses rather than KV cache blocks, requests, and block tables, and their redzone- and quarantine-based designs do not fit the scarce GPU memory budget of LLM serving. This paper presents KVGuard, the first systematic memory-safety abstraction for KV cache management in LLM serving systems. KVGuard maintains a compact shadow block for each physical KV block, recording per-request permission and per-entry initialization, and uses this metadata to perform five safety checks across the PagedAttention workflow, spanning CPU allocation, block table construction and translation, and GPU-side block table lookup and KV entry access. Each check fires at the boundary that introduced the violation, enabling early detection during development rather than reporting downstream output corruption many requests later. A prototype of KVGuard on vLLM and SGLang covers all 29 historical real-world KV cache safety bugs collected from the two systems' patch histories. KVGuard further found 8 previously unknown safety violations while reporting no false positives. It adds 3.0% and 7.4% geomean TPOT overhead on Qwen 2.5 14B on vLLM and SGLang and at most 0.11% of the KV cache in memory overhead, two orders of magnitude below NVIDIA Compute Sanitizer on the same workloads. |
| 2026-11-04 | [UniCompress: From Multi-LoRA Experts to One Compact Model for Edge Intelligence](http://arxiv.org/abs/2611.06075v1) | Mingkang Liu, Peipei Jiao et al. | Embodied and edge systems increasingly rely on vision-language models (VLMs) to support heterogeneous skills, yet deploying a separate task adapter for each skill while retaining a high-precision backbone creates substantial memory and management costs. We present UniCompress, a deployment-oriented framework that converts multi-task expertise into one compact, adapter-free model. Starting from a consolidated BF16 Qwen3-VL-4B teacher, UniCompress constructs an asymmetric INT2 student using GPTQ initialization and teacher-guided quantization recovery. The approach jointly quantizes and recovers the projection weights in the vision encoder and language backbone, enabling aggressive compression of the complete vision-language model while preserving its multi-task behavior. We evaluate UniCompress on four heterogeneous embodied tasks: posture recognition, household-safety understanding, multimodal dialogue routing, and mobile-GUI action prediction. UniCompress reduces the approximate weight storage from 8.9 GB in BF16 to 2.1 GB in INT2, corresponding to a reduction of approximately 76%. It improves the INT2 macro task score by 34.3 percentage points over GPTQ-INT2 while remaining within 0.9 points of the consolidated BF16 teacher. These results demonstrate the potential of teacher-guided joint vision-language INT2 recovery for deploying a single multi-skill VLM on resource-constrained embodied platforms. |
| 2026-11-04 | [DeepfakeGenome: Toward Next-Generation Deepfake Attribution](http://arxiv.org/abs/2611.06046v1) | Shunli Wang, Xinyu Zhou et al. | In recent years, AIGC technologies are capable of generating hyper-realistic forged facial images, posing severe threats to facial security safety. To tackle these challenges, early Deepfake research primarily centered on real-fake classification. Recently, growing efforts have been devoted to the Deepfake Attribution (DFA) of generated content. Nevertheless, existing research on Deepfake detection and attribution faces saturated performance in binary classification, limited diversity in datasets and algorithms, and imperfect evaluation protocols, which severely impede practical application. To address these limitations, we propose a comprehensive deepfake detection and attribution benchmark named DeepfakeGenome (DFG). It contains 100 facial forgery algorithms and 2M images in total, achieving 4x to 100x larger than prior DFA benchmarks. We further design 4 protocols for practical evaluation, including a novel retrieval-based attribution paradigm. Unlike previous open-set evaluation metrics, the proposed retrieval metrics are more aligned with the real-world active defense situation of blacklist registration mechanisms. Based on these elaborate designs, we investigate the performance ceiling of deepfake attribution task. Over 2k experimental evaluations are conducted, and 10 insightful findings are derived. We believe this work can provide new insights into the DFA research field. The Code and Benchmark of DFG are avaliable on https://huggingface.co/spaces/DeepfakeGenome/DFG_Page. |

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



