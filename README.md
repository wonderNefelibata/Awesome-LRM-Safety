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
| 2026-02-04 | [CoT is Not the Chain of Truth: An Empirical Internal Analysis of Reasoning LLMs for Fake News Generation](http://arxiv.org/abs/2602.04856v1) | Zhao Tong, Chunlin Gong et al. | From generating headlines to fabricating news, the Large Language Models (LLMs) are typically assessed by their final outputs, under the safety assumption that a refusal response signifies safe reasoning throughout the entire process. Challenging this assumption, our study reveals that during fake news generation, even when a model rejects a harmful request, its Chain-of-Thought (CoT) reasoning may still internally contain and propagate unsafe narratives. To analyze this phenomenon, we introduce a unified safety-analysis framework that systematically deconstructs CoT generation across model layers and evaluates the role of individual attention heads through Jacobian-based spectral metrics. Within this framework, we introduce three interpretable measures: stability, geometry, and energy to quantify how specific attention heads respond or embed deceptive reasoning patterns. Extensive experiments on multiple reasoning-oriented LLMs show that the generation risk rise significantly when the thinking mode is activated, where the critical routing decisions concentrated in only a few contiguous mid-depth layers. By precisely identifying the attention heads responsible for this divergence, our work challenges the assumption that refusal implies safety and provides a new understanding perspective for mitigating latent reasoning risks. |
| 2026-02-04 | [Are AI Capabilities Increasing Exponentially? A Competing Hypothesis](http://arxiv.org/abs/2602.04836v1) | Haosen Ge, Hamsa Bastani et al. | Rapidly increasing AI capabilities have substantial real-world consequences, ranging from AI safety concerns to labor market consequences. The Model Evaluation & Threat Research (METR) report argues that AI capabilities have exhibited exponential growth since 2019. In this note, we argue that the data does not support exponential growth, even in shorter-term horizons. Whereas the METR study claims that fitting sigmoid/logistic curves results in inflection points far in the future, we fit a sigmoid curve to their current data and find that the inflection point has already passed. In addition, we propose a more complex model that decomposes AI capabilities into base and reasoning capabilities, exhibiting individual rates of improvement. We prove that this model supports our hypothesis that AI capabilities will exhibit an inflection point in the near future. Our goal is not to establish a rigorous forecast of our own, but to highlight the fragility of existing forecasts of exponential growth. |
| 2026-02-04 | [Safe Urban Traffic Control via Uncertainty-Aware Conformal Prediction and World-Model Reinforcement Learning](http://arxiv.org/abs/2602.04821v1) | Joydeep Chandra, Satyam Kumar Navneet et al. | Urban traffic management demands systems that simultaneously predict future conditions, detect anomalies, and take safe corrective actions -- all while providing reliability guarantees. We present STREAM-RL, a unified framework that introduces three novel algorithmic contributions: (1) PU-GAT+, an Uncertainty-Guided Adaptive Conformal Forecaster that uses prediction uncertainty to dynamically reweight graph attention via confidence-monotonic attention, achieving distribution-free coverage guarantees; (2) CRFN-BY, a Conformal Residual Flow Network that models uncertainty-normalized residuals via normalizing flows with Benjamini-Yekutieli FDR control under arbitrary dependence; and (3) LyCon-WRL+, an Uncertainty-Guided Safe World-Model RL agent with Lyapunov stability certificates, certified Lipschitz bounds, and uncertainty-propagated imagination rollouts. To our knowledge, this is the first framework to propagate calibrated uncertainty from forecasting through anomaly detection to safe policy learning with end-to-end theoretical guarantees. Experiments on multiple real-world traffic trajectory data demonstrate that STREAM-RL achieves 91.4\% coverage efficiency, controls FDR at 4.1\% under verified dependence, and improves safety rate to 95.2\% compared to 69\% for standard PPO while achieving higher reward, with 23ms end-to-end inference latency. |
| 2026-02-04 | [Agentic AI in Healthcare & Medicine: A Seven-Dimensional Taxonomy for Empirical Evaluation of LLM-based Agents](http://arxiv.org/abs/2602.04813v1) | Shubham Vatsal, Harsh Dubey et al. | Large Language Model (LLM)-based agents that plan, use tools and act has begun to shape healthcare and medicine. Reported studies demonstrate competence on various tasks ranging from EHR analysis and differential diagnosis to treatment planning and research workflows. Yet the literature largely consists of overviews which are either broad surveys or narrow dives into a single capability (e.g., memory, planning, reasoning), leaving healthcare work without a common frame. We address this by reviewing 49 studies using a seven-dimensional taxonomy: Cognitive Capabilities, Knowledge Management, Interaction Patterns, Adaptation & Learning, Safety & Ethics, Framework Typology and Core Tasks & Subtasks with 29 operational sub-dimensions. Using explicit inclusion and exclusion criteria and a labeling rubric (Fully Implemented, Partially Implemented, Not Implemented), we map each study to the taxonomy and report quantitative summaries of capability prevalence and co-occurrence patterns. Our empirical analysis surfaces clear asymmetries. For instance, the External Knowledge Integration sub-dimension under Knowledge Management is commonly realized (~76% Fully Implemented) whereas Event-Triggered Activation sub-dimenison under Interaction Patterns is largely absent (~92% Not Implemented) and Drift Detection & Mitigation sub-dimension under Adaptation & Learning is rare (~98% Not Implemented). Architecturally, Multi-Agent Design sub-dimension under Framework Typology is the dominant pattern (~82% Fully Implemented) while orchestration layers remain mostly partial. Across Core Tasks & Subtasks, information centric capabilities lead e.g., Medical Question Answering & Decision Support and Benchmarking & Simulation, while action and discovery oriented areas such as Treatment Planning & Prescription still show substantial gaps (~59% Not Implemented). |
| 2026-02-04 | [Safe-NEureka: a Hybrid Modular Redundant DNN Accelerator for On-board Satellite AI Processing](http://arxiv.org/abs/2602.04803v1) | Riccardo Tedeschi, Luigi Ghionda et al. | Low Earth Orbit (LEO) constellations are revolutionizing the space sector, with on-board Artificial Intelligence (AI) becoming pivotal for next-generation satellites. AI acceleration is essential for safety-critical functions such as autonomous Guidance, Navigation, and Control (GNC), where errors cannot be tolerated, and performance-critical processing of high-bandwidth sensor data, where occasional errors are tolerable. Consequently, AI accelerators for satellites must combine robust protection against radiation-induced faults with high throughput. This paper presents Safe-NEureka, a Hybrid Modular Redundant Deep Neural Network (DNN) accelerator for heterogeneous RISC-V systems. It operates in two modes: a redundancy mode utilizing Dual Modular Redundancy (DMR) with hardware-based recovery, and a performance mode repurposing redundant datapaths to maximize parallel throughput. Furthermore, its memory interface is protected by Error Correction Codes (ECCs), and the controller by Triple Modular Redundancy (TMR). Implementation in GlobalFoundries 12nm technology shows a 96 reduction in faulty executions in redundancy mode, with a manageable 15 area overhead. In performance mode, the architecture achieves near-baseline speeds on 3x3 dense convolutions with a 5 throughput and 11 efficiency reduction, compared to 48 and 53 in redundancy mode. This flexibility ensures high overheads are limited to critical tasks, establishing Safe-NEureka as a versatile solution for space applications. |
| 2026-02-04 | [SQP-Based Cable-Tension Allocation for Multi-Drone Load Transport](http://arxiv.org/abs/2602.04801v1) | Lamberto Vazquez-Soqui, Fatima Oliva-Palomo et al. | Multi-Agent Aerial Load Transport Systems (MAATS) offer greater payload capacity and fault tolerance than single-drone solutions. However, they have an underdetermined tension allocation problem that leads to uneven energy distribution, cable slack, or collisions between drones and cables. This paper presents a real-time optimization layer that improves a hierarchical load-position-attitude controller by incorporating a Sequential Quadratic Programming (SQP) algorithm. The SQP formulation minimizes the sum of squared cable tensions while imposing a cable-alignment penalty that discourages small inter-cable angles, thereby preventing tether convergence without altering the reference trajectory. We tested the method under nominal conditions by running numerical simulations of four quadrotors. Computational experiments based on numerical simulations demonstrate that the SQP routine runs in a few milliseconds on standard hardware, indicating feasibility for real-time use. A sensitivity analysis confirms that the gain of the cable-alignment penalty can be tuned online, enabling a controllable trade-off between safety margin and energy consumption with no measurable degradation of tracking performance in simulation. This framework provides a scalable path to safe and energy-balanced cooperative load transport in practical deployments. |
| 2026-02-04 | [Beyond the Control Equations: An Artifact Study of Implementation Quality in Robot Control Software](http://arxiv.org/abs/2602.04799v1) | Nils Chur, Thorsten Berger et al. | A controller -- a software module managing hardware behavior -- is a key component of a typical robot system. While control theory gives safety guarantees for standard controller designs, the practical implementation of controllers in software introduces complexities that are often overlooked. Controllers are often designed in continuous space, while the software is executed in discrete space, undermining some of the theoretical guarantees. Despite extensive research on control theory and control modeling, little attention has been paid to the implementations of controllers and how their theoretical guarantees are ensured in real-world software systems. We investigate 184 real-world controller implementations in open-source robot software. We examine their application context, the implementation characteristics, and the testing methods employed to ensure correctness. We find that the implementations often handle discretization in an ad hoc manner, leading to potential issues with real-time reliability. Challenges such as timing inconsistencies, lack of proper error handling, and inadequate consideration of real-time constraints further complicate matters. Testing practices are superficial, no systematic verification of theoretical guarantees is used, leaving possible inconsistencies between expected and actual behavior. Our findings highlight the need for improved implementation guidelines and rigorous verification techniques to ensure the reliability and safety of robotic controllers in practice. |
| 2026-02-04 | [LALM-as-a-Judge: Benchmarking Large Audio-Language Models for Safety Evaluation in Multi-Turn Spoken Dialogues](http://arxiv.org/abs/2602.04796v1) | Amir Ivry, Shinji Watanabe | Spoken dialogues with and between voice agents are becoming increasingly common, yet assessing them for their socially harmful content such as violence, harassment, and hate remains text-centric and fails to account for audio-specific cues and transcription errors. We present LALM-as-a-Judge, the first controlled benchmark and systematic study of large audio-language models (LALMs) as safety judges for multi-turn spoken dialogues. We generate 24,000 unsafe and synthetic spoken dialogues in English that consist of 3-10 turns, by having a single dialogue turn including content with one of 8 harmful categories (e.g., violence) and on one of 5 grades, from very mild to severe. On 160 dialogues, 5 human raters confirmed reliable unsafe detection and a meaningful severity scale. We benchmark three open-source LALMs: Qwen2-Audio, Audio Flamingo 3, and MERaLiON as zero-shot judges that output a scalar safety score in [0,1] across audio-only, transcription-only, or multimodal inputs, along with a transcription-only LLaMA baseline. We measure the judges' sensitivity to detecting unsafe content, the specificity in ordering severity levels, and the stability of the score in dialogue turns. Results reveal architecture- and modality-dependent trade-offs: the most sensitive judge is also the least stable across turns, while stable configurations sacrifice detection of mild harmful content. Transcription quality is a key bottleneck: Whisper-Large may significantly reduce sensitivity for transcription-only modes, while largely preserving severity ordering. Audio becomes crucial when paralinguistic cues or transcription fidelity are category-critical. We summarize all findings and provide actionable guidance for practitioners. |
| 2026-02-04 | [Inference-Time Reasoning Selectively Reduces Implicit Social Bias in Large Language Models](http://arxiv.org/abs/2602.04742v1) | Molly Apsel, Michael N. Jones | Drawing on constructs from psychology, prior work has identified a distinction between explicit and implicit bias in large language models (LLMs). While many LLMs undergo post-training alignment and safety procedures to avoid expressions of explicit social bias, they still exhibit significant implicit biases on indirect tasks resembling the Implicit Association Test (IAT). Recent work has further shown that inference-time reasoning can impair LLM performance on tasks that rely on implicit statistical learning. Motivated by a theoretical link between implicit associations and statistical learning in human cognition, we examine how reasoning-enabled inference affects implicit bias in LLMs. We find that enabling reasoning significantly reduces measured implicit bias on an IAT-style evaluation for some model classes across fifteen stereotype topics. This effect appears specific to social bias domains, as we observe no corresponding reduction for non-social implicit associations. As reasoning is increasingly enabled by default in deployed LLMs, these findings suggest that it can meaningfully alter fairness evaluation outcomes in some systems, while also raising questions about how alignment procedures interact with inference-time reasoning to drive variation in bias reduction across model types. More broadly, this work highlights how theory from cognitive science and psychology can complement AI evaluation research by providing methodological and interpretive frameworks that reveal new insights into model behavior. |
| 2026-02-04 | [Alignment Drift in Multimodal LLMs: A Two-Phase, Longitudinal Evaluation of Harm Across Eight Model Releases](http://arxiv.org/abs/2602.04739v1) | Casey Ford, Madison Van Doren et al. | Multimodal large language models (MLLMs) are increasingly deployed in real-world systems, yet their safety under adversarial prompting remains underexplored. We present a two-phase evaluation of MLLM harmlessness using a fixed benchmark of 726 adversarial prompts authored by 26 professional red teamers. Phase 1 assessed GPT-4o, Claude Sonnet 3.5, Pixtral 12B, and Qwen VL Plus; Phase 2 evaluated their successors (GPT-5, Claude Sonnet 4.5, Pixtral Large, and Qwen Omni) yielding 82,256 human harm ratings. Large, persistent differences emerged across model families: Pixtral models were consistently the most vulnerable, whereas Claude models appeared safest due to high refusal rates. Attack success rates (ASR) showed clear alignment drift: GPT and Claude models exhibited increased ASR across generations, while Pixtral and Qwen showed modest decreases. Modality effects also shifted over time: text-only prompts were more effective in Phase 1, whereas Phase 2 produced model-specific patterns, with GPT-5 and Claude 4.5 showing near-equivalent vulnerability across modalities. These findings demonstrate that MLLM harmlessness is neither uniform nor stable across updates, underscoring the need for longitudinal, multimodal benchmarks to track evolving safety behaviour. |
| 2026-02-04 | [From Data to Behavior: Predicting Unintended Model Behaviors Before Training](http://arxiv.org/abs/2602.04735v1) | Mengru Wang, Zhenqian Xu et al. | Large Language Models (LLMs) can acquire unintended biases from seemingly benign training data even without explicit cues or malicious content. Existing methods struggle to detect such risks before fine-tuning, making post hoc evaluation costly and inefficient. To address this challenge, we introduce Data2Behavior, a new task for predicting unintended model behaviors prior to training. We also propose Manipulating Data Features (MDF), a lightweight approach that summarizes candidate data through their mean representations and injects them into the forward pass of a base model, allowing latent statistical signals in the data to shape model activations and reveal potential biases and safety risks without updating any parameters. MDF achieves reliable prediction while consuming only about 20% of the GPU resources required for fine-tuning. Experiments on Qwen3-14B, Qwen2.5-32B-Instruct, and Gemma-3-12b-it confirm that MDF can anticipate unintended behaviors and provide insight into pre-training vulnerabilities. |
| 2026-02-04 | [Safe Adaptive Control of Parabolic PDE-ODE Cascades](http://arxiv.org/abs/2602.04656v1) | Yun Jiang, Ji Wang | In this paper, we propose a safe adaptive boundary control strategy for a class of parabolic partial differential equation-ordinary differential equation (PDE-ODE) cascaded systems with parametric uncertainties in both the PDE and ODE subsystems. The proposed design is built upon an adaptive Control Barrier Function (aCBF) framework that incorporates high-relative-degree CBFs together with a batch least-squares identification (BaLSI)-based adaptive control that guarantees exact parameter identification in finite time. The proposed control law ensures that: (i) if the system output state initially lies within a prescribed safe set, safety is maintained for all time; otherwise, the output is driven back into the safe region within a preassigned finite time; and (ii) convergence to zero of all plant states is achieved. Numerical simulations are provided to demonstrate the effectiveness of the proposed approach. |
| 2026-02-04 | [From Vision to Assistance: Gaze and Vision-Enabled Adaptive Control for a Back-Support Exoskeleton](http://arxiv.org/abs/2602.04648v1) | Alessandro Leanza, Paolo Franceschi et al. | Back-support exoskeletons have been proposed to mitigate spinal loading in industrial handling, yet their effectiveness critically depends on timely and context-aware assistance. Most existing approaches rely either on load-estimation techniques (e.g., EMG, IMU) or on vision systems that do not directly inform control. In this work, we present a vision-gated control framework for an active lumbar occupational exoskeleton that leverages egocentric vision with wearable gaze tracking. The proposed system integrates real-time grasp detection from a first-person YOLO-based perception system, a finite-state machine (FSM) for task progression, and a variable admittance controller to adapt torque delivery to both posture and object state. A user study with 15 participants performing stooping load lifting trials under three conditions (no exoskeleton, exoskeleton without vision, exoskeleton with vision) shows that vision-gated assistance significantly reduces perceived physical demand and improves fluency, trust, and comfort. Quantitative analysis reveals earlier and stronger assistance when vision is enabled, while questionnaire results confirm user preference for the vision-gated mode. These findings highlight the potential of egocentric vision to enhance the responsiveness, ergonomics, safety, and acceptance of back-support exoskeletons. |
| 2026-02-04 | [Abstract Framework for All-Path Reachability Analysis](http://arxiv.org/abs/2602.04641v1) | Misaki Kojima, Naoki Nishida | An all-path reachability (APR, for short) predicate is a pair of a source set and a target set, which are subsets of an object set. APR predicates have been defined for abstract reduction systems (ARSs, for short) and then extended to logically constrained term rewrite systems (LCTRSs, for short) as pairs of constrained terms that represent sets of terms modeling configurations, states, etc. An APR predicate is said to be partially (or demonically) valid w.r.t. a rewrite system if every finite maximal reduction sequence of the system starting from any element in the source set includes an element in the target set. Partial validity of APR predicates w.r.t. ARSs is defined by means of two inference rules, which can be considered a proof system to construct (possibly infinite) derivation trees for partial validity. On the other hand, a proof system for LCTRSs consists of four inference rules, and thus there is a gap between the inference rules for partial validity w.r.t. ARSs and LCTRSs. In this paper, we revisit the framework for APR analysis and adapt it to verification of not only safety but also liveness properties. To this end, we first reformulate an abstract framework for partial validity w.r.t. ARSs so that there is a one-to-one correspondence between the inference rules for ARSs and LCTRSs. Secondly, we show how to apply APR analysis to safety verification. Thirdly, to apply APR analysis to liveness verification, we introduce a novel stronger validity of APR predicates, called total validity, which requires not only finite but also infinite execution paths to reach target sets. Finally, for a partially valid APR predicate with a cyclic-proof tree, we show a necessary and sufficient condition for the tree to ensure total validity, showing how to apply APR analysis to liveness verification. |
| 2026-02-04 | [WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2602.04634v1) | Zelai Xu, Zhexuan Xu et al. | Recent advancements in Large Language Models (LLMs) have largely focused on depth scaling, where a single agent solves long-horizon problems with multi-turn reasoning and tool use. However, as tasks grow broader, the key bottleneck shifts from individual competence to organizational capability. In this work, we explore a complementary dimension of width scaling with multi-agent systems to address broad information seeking. Existing multi-agent systems often rely on hand-crafted workflows and turn-taking interactions that fail to parallelize work effectively. To bridge this gap, we propose WideSeek-R1, a lead-agent-subagent framework trained via multi-agent reinforcement learning (MARL) to synergize scalable orchestration and parallel execution. By utilizing a shared LLM with isolated contexts and specialized tools, WideSeek-R1 jointly optimizes the lead agent and parallel subagents on a curated dataset of 20k broad information-seeking tasks. Extensive experiments show that WideSeek-R1-4B achieves an item F1 score of 40.0% on the WideSearch benchmark, which is comparable to the performance of single-agent DeepSeek-R1-671B. Furthermore, WideSeek-R1-4B exhibits consistent performance gains as the number of parallel subagents increases, highlighting the effectiveness of width scaling. |
| 2026-02-04 | [Resilient Load Forecasting under Climate Change: Adaptive Conditional Neural Processes for Few-Shot Extreme Load Forecasting](http://arxiv.org/abs/2602.04609v1) | Chenxi Hu, Yue Ma et al. | Extreme weather can substantially change electricity consumption behavior, causing load curves to exhibit sharp spikes and pronounced volatility. If forecasts are inaccurate during those periods, power systems are more likely to face supply shortfalls or localized overloads, forcing emergency actions such as load shedding and increasing the risk of service disruptions and public-safety impacts. This problem is inherently difficult because extreme events can trigger abrupt regime shifts in load patterns, while relevant extreme samples are rare and irregular, making reliable learning and calibration challenging. We propose AdaCNP, a probabilistic forecasting model for data-scarce condition. AdaCNP learns similarity in a shared embedding space. For each target data, it evaluates how relevant each historical context segment is to the current condition and reweights the context information accordingly. This design highlights the most informative historical evidence even when extreme samples are rare. It enables few-shot adaptation to previously unseen extreme patterns. AdaCNP also produces predictive distributions for risk-aware decision-making without expensive fine-tuning on the target domain. We evaluate AdaCNP on real-world power-system load data and compare it against a range of representative baselines. The results show that AdaCNP is more robust during extreme periods, reducing the mean squared error by 22\% relative to the strongest baseline while achieving the lowest negative log-likelihood, indicating more reliable probabilistic outputs. These findings suggest that AdaCNP can effectively mitigate the combined impact of abrupt distribution shifts and scarce extreme samples, providing a more trustworthy forecasting for resilient power system operation under extreme events. |
| 2026-02-04 | [Stochastic Decision Horizons for Constrained Reinforcement Learning](http://arxiv.org/abs/2602.04599v1) | Nikola Milosevic, Leonard Franz et al. | Constrained Markov decision processes (CMDPs) provide a principled model for handling constraints, such as safety and other auxiliary objectives, in reinforcement learning. The common approach of using additive-cost constraints and dual variables often hinders off-policy scalability. We propose a Control as Inference formulation based on stochastic decision horizons, where constraint violations attenuate reward contributions and shorten the effective planning horizon via state-action-dependent continuation. This yields survival-weighted objectives that remain replay-compatible for off-policy actor-critic learning. We propose two violation semantics, absorbing and virtual termination, that share the same survival-weighted return but result in distinct optimization structures that lead to SAC/MPO-style policy improvement. Experiments demonstrate improved sample efficiency and favorable return-violation trade-offs on standard benchmarks. Moreover, MPO with virtual termination (VT-MPO) scales effectively to our high-dimensional musculoskeletal Hyfydy setup. |
| 2026-02-04 | [Trust The Typical](http://arxiv.org/abs/2602.04581v1) | Debargha Ganguly, Sreehari Sankar et al. | Current approaches to LLM safety fundamentally rely on a brittle cat-and-mouse game of identifying and blocking known threats via guardrails. We argue for a fresh approach: robust safety comes not from enumerating what is harmful, but from deeply understanding what is safe. We introduce Trust The Typical (T3), a framework that operationalizes this principle by treating safety as an out-of-distribution (OOD) detection problem. T3 learns the distribution of acceptable prompts in a semantic space and flags any significant deviation as a potential threat. Unlike prior methods, it requires no training on harmful examples, yet achieves state-of-the-art performance across 18 benchmarks spanning toxicity, hate speech, jailbreaking, multilingual harms, and over-refusal, reducing false positive rates by up to 40x relative to specialized safety models. A single model trained only on safe English text transfers effectively to diverse domains and over 14 languages without retraining. Finally, we demonstrate production readiness by integrating a GPU-optimized version into vLLM, enabling continuous guardrailing during token generation with less than 6% overhead even under dense evaluation intervals on large-scale workloads. |
| 2026-02-04 | [Peak Bounds for the Estimation Error under Sensor Attacks](http://arxiv.org/abs/2602.04568v1) | Axel Stafström, Daniel Arnström et al. | This paper investigates bounds on the estimation error of a linear system affected by norm-bounded disturbances and full sensor attacks. The system is equipped with a detector that evaluates the norm of the innovation signal to detect faults, and the attacker wants to avoid detection. We utilize induced $L_\infty$ system norms, also called \emph{peak-to-peak} norms, to compare the estimation error bounds under nominal operations and under attack. This leads to a sufficient condition for when the bound on the estimation error is smaller during an attack than during nominal operation. This condition is independent of the attack strategy and depends only on the attacker's desire to remain undetected and (indirectly) the observer gain. Therefore, we investigate both an observer design method, that seeks to reduce the error bound under attack while keeping the nominal error bound low, and detector threshold tuning. As a numerical illustration, we show how a sensor attack can deactivate a robust safety filter based on control barrier functions if the attacked error bound is larger than the nominal one. We also statistically evaluate our observer design method and the effect of the detector threshold. |
| 2026-02-04 | [$C$-$ΔΘ$: Circuit-Restricted Weight Arithmetic for Selective Refusal](http://arxiv.org/abs/2602.04521v1) | Aditya Kasliwal, Pratinav Seth et al. | Modern deployments require LLMs to enforce safety policies at scale, yet many controls rely on inference-time interventions that add recurring compute cost and serving complexity. Activation steering is widely used, but it requires runtime hooks and scales cost with the number of generations; conditional variants improve selectivity by gating when steering is applied but still retain an inference-time control path. We ask whether selective refusal can be moved entirely offline: can a mechanistic understanding of category-specific refusal be distilled into a circuit-restricted weight update that deploys as a standard checkpoint? We propose C-Δθ: Circuit Restricted Weight Arithmetic, which (i) localizes refusal-causal computation as a sparse circuit using EAP-IG and (ii) computes a constrained weight update ΔθC supported only on that circuit (typically <5% of parameters). Applying ΔθC yields a drop-in edited checkpoint with no inference-time hooks, shifting cost from per-request intervention to a one-time offline update. We evaluate category-targeted selectivity and capability retention on refusal and utility benchmarks. |

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



