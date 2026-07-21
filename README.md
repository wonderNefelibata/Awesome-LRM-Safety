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
| 2026-07-20 | [Learning Adaptive Safety Margins for Visual Navigation](http://arxiv.org/abs/2607.18200v1) | Junyi Hu, Shuaihang Yuan et al. | Robots in cluttered indoor spaces often fail not because they cannot generate collision-free paths, but because a fixed safety margin is mis-calibrated: conservative margins cause detours and timeouts, while permissive margins lead to near-boundary shortcuts under perception bias. Diffusion-based planners propose diverse trajectory candidates from egocentric RGB-D, yet reliable selection remains the bottleneck. We propose a context-conditioned safety critic that learns an adaptive clearance preference for ranking diffusion proposals, decomposed into three complementary terms: (i) a safety term with a clearance-budget penalty and a control-barrier-function residual for waypoint- and transition-wise safety, (ii) an efficiency term combining a smoothness penalty with a safety-gated detour-ratio penalty that avoids detours without incentivizing risky shortcuts, and (iii) a distance-constraint matching term that anchors the learned budget to realized ESDF clearances to prevent margin collapse. We train the critic with privileged ESDF geometry in simulation and distill it into a perception-only selector via a two-stage teacher-student procedure. On PointGoal navigation in HM3D and MP3D, including cross-dataset transfer, our method achieves the highest success rate (SR) and success weighted by path length (SPL) among strong diffusion, optimization, and RL baselines. Trained purely in simulation, it transfers to a Unitree G1 humanoid and navigates cluttered indoor scenes without task-specific tuning. |
| 2026-07-20 | [Certified Training for Convolutional Perturbations](http://arxiv.org/abs/2607.18195v1) | Benedikt Brückner, Alessio Lomuscio | Vision models have been found to be susceptible to perturbations such as motion blur induced at runtime by a shaking camera. This impedes their deployment in critical applications since phenomena such as slightly blurred vision might lead to failures, for example an object detector missing objects. While methods such as data augmentation or Adversarial Training can improve empirical robustness, they lack formal safety guarantees, making it difficult to identify and mitigate hidden vulnerabilities. We introduce a novel Certified Training approach that leverages an efficient encoding of convolutional perturbations to train provably robust models. Our method significantly outperforms Adversarial Training, achieving, for example, over 80% robust accuracy against motion blur of reasonable intensity on CIFAR10 while maintaining comparable standard accuracy. |
| 2026-07-20 | [uSTM: A Lightweight and Efficient STM Supporting General Types and Deferred Aborts](http://arxiv.org/abs/2607.18178v1) | Zachary Kent, Guy Blelloch et al. | Software Transactional Memory (STM) systems allow developers to more easily exploit multicore architectures by wrapping arbitrary sequential code in transactions that are executed concurrently. In recent years, the performance of STM systems has approached that of hand-tuned data structures through techniques that avoid unnecessary aborts and exploit the semantics of underlying data structures.   Despite achieving excellent performance, most STM systems do not fully address the concerns they targeted in the first place: safety, usability, and generality. In particular, these systems place restrictions on the data types that may be updated transactionally, such as requiring that these types fit within a word, and can require modification of data layout. Moreover, most STM systems abort transactions in the middle of client code to ensure correctness. This can cause space leaks and other bugs not present in the original code.   We present ustm, a novel STM system addressing all of these shortcomings while still maintaining excellent performance, all within ~300 lines of code. uSTM supports general types while maintaining data layout. Aborts are deferred until the end of the transaction, allowing client code within a transaction to terminate normally. To ensure that uSTM guarantees opacity, we implement a novel timestamping algorithm we call split-increment timestamps.   We compare the performance of uSTM to a variety of state-of-the-art (SOTA) STM systems, demonstrating that uSTM matches or outperforms the SOTA on a variety of workloads. |
| 2026-07-20 | [Judge-dependent safety gains and model-specific helpfulness costs of evidence-sufficiency prompting in clinical LLMs](http://arxiv.org/abs/2607.18086v1) | Koyar Afrasyab | Background: LLM judges increasingly score whether clinical language models give overconfident answers under incomplete evidence, yet whether a measured "safety gain" reflects real behavior change or the judge's calibration is unresolved. Using a structured evidence-sufficiency prompt as a test case, we asked whether it reduces unsafe overconfident answers, how far that effect depends on the scoring judge, and what it costs in helpfulness.   Methods: In a retrospective public-data benchmark (Real-POCQi, HealthBench, MedRBench), four models (GPT-5.5, Claude Opus 4.8, Gemini 3.5 Flash, Grok 4.3) answered a fully paired common panel (1,200 cells) with a standard prompt and the wrapper. The pre-specified endpoint was the paired reduction in unsafe overconfidence scored by the primary judge (GPT-5.4-nano); secondary analyses added a different-family judge (Claude Sonnet 5), a correctness judge, matched scaffold controls, and a blinded three-clinician review.   Results: Unsafe overconfidence fell from 49.3% to 24.7%, a paired reduction of 24.7 points (95% CI 21.8-27.7; p<0.001), robust in direction across models and paraphrases. Magnitude was judge-dependent: Sonnet agreed on direction but nearly halved the effect (+13.1 points), with one-directional disagreement. Blinded clinicians characterized the primary judge as a high-sensitivity (1.00), low-specificity (0.55) screen, not a calibrated rate. The gain carried a model-specific helpfulness cost (correct diagnosis 80.3% to 50.3%): near-free for GPT-5.5, near-total for Gemini (-58 points). Matched scaffold controls showed genuine behavior change, not judge circularity.   Conclusions: LLM-judged clinical safety effects should be reported as directional and relative, anchored to human review and evaluated jointly with helpfulness, not as calibrated absolute rates. This does not establish clinical deployment readiness. |
| 2026-07-20 | [Hardware Mechanisms to Dynamically Throttle AI Performance](http://arxiv.org/abs/2607.18069v1) | Haiyue Ma, Lauren Malek et al. | As more capable AI models are increasingly integrated into critical computer systems, the lack of control over AI intent motivates safety mechanisms. Existing software safeguards impose only behavioral constraints that can potentially be bypassed by sufficiently intelligent models. While hardware-level safety enforcement has been recognized as an essential last line of defense, few mechanisms have been proposed beyond policy regulations on unauthorized accesses or coarse full-chip shutdown. What is missing is a fine-grained, dynamic intervention mechanism at the architecture level.   In this paper, we introduce a set of microarchitecture knobs which dynamically control the available hardware resources to limit AI performance at runtime. We evaluate candidate knobs spanning the GPU memory subsystem, across capacity, bandwidth, latency and frequency dimensions, and narrow down to four strong candidates: L2 size, L2 latency, L2 bandwidth, and shared memory port access rate. To minimize new logic and extra design cost, we build all four mechanisms from well-established microarchitectural primitives: cache way masking, credit-based rate limiting, latency insertion, and bank arbitration. We show that these knobs achieve high performance sensitivity (up to 80% performance cut at 1/8 resource availability), negligible implementation cost (<~10K flip flops), fast stabilization after dynamic throttling (5-80K cycles), and minimal collateral impact on the rest of the chip. Further, multi-knob analysis reveals combinations of knobs that amplify the performance degradation beyond the effect of each knob individually, which enables a broader range of performance targets. |
| 2026-07-20 | [Adaptive Adversaries: A Multi-Turn, Multi-LLM Benchmark for LLM Agent Security](http://arxiv.org/abs/2607.18063v1) | Devina Jain, David Hartmann et al. | LLM-based agents process external content, exposing them to prompt injection and multi-turn manipulation. Most safety benchmarks evaluate defenders against fixed attack pools collected before evaluation, single-turn or multi-turn. We present a 21-scenario benchmark for \emph{adaptive multi-round attacks against memoryless LLM defenders}: an autonomous LLM attacker observes prior defender responses and pivots across rounds, while each defender response is evaluated as a fresh interaction. Holding the 21 scenarios, attackers, defenders, and structured-output scoring fixed, restricting scoring to the first attacker turn yields $0$-$1\%$ attack success rate (ASR); allowing 15 rounds of adaptive attack yields $5.4$-$14.0\%$. Pooling three frontier attacker LLMs uncovers $1.4$-$2.2\times$ as many unique successful attacks as the best single attacker, and the generated attacks have low cosine similarity ($0.02$-$0.14$) to attacks in existing benchmarks. Claude Opus 4.6 and GPT-5.4 are tied in aggregate ($5.4\%$ each; overlapping $95\%$ CIs), but their weaknesses differ sharply: on one scenario Opus reaches $60\%$ ASR ($95\%$ CI $36$--$80\%$) while GPT-5.4 and Gemini each stay at $7\%$ (CI $1$-$30\%$; the gap is preserved in a higher-$N$ replication). $13$ of $21$ scenarios distinguish at least one defender pair, yet rankings disagree across scenarios (Kendall's $W = 0.19$). We release the benchmark -- 21 evaluation scenarios, 10 public development scenarios, the orchestrator, baseline harnesses, and a multi-attacker CLI -- plus 945 transcripts from the 3$\times$3 frontier matrix, an attack-replay dataset, and 18{,}422 gpt-oss-20b battles from an open competition's final scoring rounds. |
| 2026-07-20 | [Test Coverage Analysis of Agentic Pull Requests](http://arxiv.org/abs/2607.18057v1) | Atish Kumar Dipongkor, Talank Baral et al. | AI coding agents increasingly submit complete pull requests (PRs) with minimal human intervention, shifting software development from AI-assisted to autonomous workflows. As these agents become more prevalent, ensuring the code they generate is adequately tested, by existing tests or by tests the agents write, is critical to preventing regressions, yet little is known about testing in agentic PRs. To address this gap, we analyze 4882 agent-generated PRs from the AIDev dataset (532 Java and 4350 Python PRs) produced by five coding agents. We study (i) how often agents include test changes and (ii) how well covered are code changes by existing and agent-written tests. Agents include test changes in only 49.6% of PRs that change code under test files. Existing tests provide an incomplete safety net: they cover 61.5% of agents' changed executable lines in Java and only 27.0% in Python, where 64.8% of PRs have no changed line executed by any existing test. Agent-written tests improve coverage over existing tests, but only in a minority of PRs: 35.9% of Java and 22.5% of Python Code + Tests PRs show a coverage gain. Across both languages, error-handling constructs (e.g., try and catch blocks) are the most consistently under-tested, with miss rates reaching 86.0% in Java and 81.0% in Python. These findings motivate coverage-aware development practices, coverage feedback loops for coding agents, and evaluation benchmarks that measure test quality to better help agents reliably test their own code. |
| 2026-07-20 | [An Early Warning of Emerging Biosecurity Risks in Frontier LLMs](http://arxiv.org/abs/2607.18056v1) | Zhida He, Xia Hu et al. | Frontier large language models (LLMs) are increasingly integrated into scientific workflows, yet their growing biological capabilities may outpace current safeguards. To assess the biological risks of frontier models, we develop Intern-BioBreaker, a specialized bio-red-teaming model, together with an integrated computational-to-physical framework that couples model-level stress testing with wet-lab validation. Within this framework, Intern-BioBreaker generates targeted jailbreak prompts to test whether aligned models can be induced to provide operational guidance for safety-sensitive biological tasks or produce sequence-level outputs with potentially harmful properties. Selected sequence outputs are then carried forward for DNA synthesis, host expression, and orthogonal protein verification to assess whether model-generated designs can yield the intended biological products. Our evaluation reveals a concerning gap between text-level safeguards and the risks posed by capable scientific models: (i) Intern-BioBreaker outperforms baseline attack models and reveals widespread bio-risk jailbreak vulnerabilities across both open-weight and proprietary frontier LLMs, with several targets reaching near-saturated or 100% task-level attack success rate (ASR); (ii) in sequence-level case studies, GPT-5.5 can be induced to generate modified viral candidate sequences with pathogenic potential; the corresponding translated proteins may exhibit even stronger receptor-binding affinity and thus enhanced infection potential; and (iii) end-to-end verification shows that selected model-generated biological designs are not merely textual artifacts, but can be physically realized under controlled experimental settings. These findings underscore the need for stronger biological red-teaming, nucleic acid synthesis screening, and safety mechanisms that keep pace with model capabilities. |
| 2026-07-20 | [MADA-RL: Multi-Agent Debate-Aware Reinforcement Learning for Parameter-Efficient Reasoning in Compact Models](http://arxiv.org/abs/2607.18006v1) | Martino M. L. Pulici, Cuong Xuan Chu et al. | Large language models achieve strong reasoning performance, but often at prohibitive training cost - a challenge that is especially acute for compact models ($\leq 4 \, \mathrm{B}$ parameters) trained under limited budgets. We introduce MADA-RL, a post-training framework that specializes compact models into generator and critic roles and trains them with a debate-aware learning signal, fine-tuning only a small subset of parameters via LoRA adapters. Our central contribution is a counterfactual critic advantage: a dynamic, role-conditioned baseline that redefines the critic's advantage as its reward minus the generator ensemble's per-instance accuracy. This explicitly optimizes critics to improve over generator consensus rather than to merely reproduce a correct answer, yielding more targeted credit assignment than static mean-reward normalization. At deployment, the specialized agents are composed in a lightweight multi-round protocol. Across five mathematical reasoning benchmarks, MADA-RL raises the accuracy of the DeepSeek-R1-Distill-Qwen-1.5B model from $39.9 \, \%$ to $41.9 \, \%$ ($+2.0$ points, $p < 0.001$) using $16$ times fewer trainable parameters than fully fine-tuned baselines, placing it on the accuracy-trainable-parameter Pareto front. It approaches, but does not surpass, the strongest baselines (DeepScaleR, STILL-3), which are trained on substantially larger datasets; we analyse this gap and the associated inference-time cost directly. A controlled study isolates the source of MADA-RL's gains: the counterfactual advantage produces the highest critic improvement rate of any model evaluated, indicating that trained critics learn to correct generator errors rather than to imitate them. |
| 2026-07-20 | [RT-SHCUA: Real-Time Self-Hosted Computer-Use Agent for UAV Control](http://arxiv.org/abs/2607.17951v1) | Di Lu, Bo Zhang et al. | Natural-language control offers a promising interface for unmanned aerial vehicles (UAVs), but directly applying self-hosted computer-use agents (SHCUAs) to UAV control introduces a structural mismatch. SHCUAs are designed for interactive host-side tool use, where delayed agent iterations are often acceptable. UAV control, however, is coupled with continuously changing physical states, strict timing constraints, safety risks, and security accountability. A stale, unauthorized, or tampered agent decision may therefore lead to unsafe or untraceable vehicle behavior.   This paper proposes a real-time and security-oriented restructuring of SHCUA-based UAV control. Instead of allowing an SHCUA to directly issue flight commands, we transform its outputs into contract-bound UAV skill invocations with explicit timing, state, authority, fallback, and evidence semantics. Based on this abstraction, we design an architecture that separates semantic reasoning from onboard execution and security/safety enforcement. Slow cloud or edge reasoning is used for mission understanding, while onboard components validate and dispatch only timely, authorized, and state-consistent skills. Security-critical enforcement points can be protected by TEE-style or microcontroller isolation mechanisms without moving the full language agent or high-frequency flight-control loop into trusted components. Prototype evaluation shows that RT-SHCUA maintains bounded task-level responsiveness while supporting degraded handling, trusted admission, and auditable evidence preservation for SHCUA-mediated UAV actions. |
| 2026-07-20 | [SEAM-V: A Hybrid-Decoupled RISC-V Vector Processor with Backend-Visible EP Context for Sustained Vector Throughput](http://arxiv.org/abs/2607.17899v1) | Weiying Wang, Zhiwei Zhang | Data-parallel workloads in deep learning and scientific computing continue to drive demand for higher processor throughput, energy efficiency, and scalability. The RISC-V Vector Extension (RVV) supports scalable execution through a vector-length-agnostic programming model. However, many tightly coupled implementations still rely on the scalar core to supply vector instructions one at a time, making execution susceptible to vector-instruction supply gaps, scalar-side progression delays, and conservative dependence handling in short-vector, loop-tail, and control/memory-interleaved phases. This paper presents SEAM-V, a hybrid-decoupled vector execution architecture for RVV. SEAM-V forms a continuous stream of execute packets (EPs) through task-level decoupling, local instruction supply, and VLIW-style packing. After an EP is serialized into individual requests, its EP identity and request-bound prefetch context remain visible to the dynamic vector backend, enabling same-EP candidate-hazard suppression and request-bound prefetching. The hybrid-dispatch path can also provide limited cross-EP vector overlap when the required safety conditions are satisfied. Cross-EP dependences, dependences not exempted by the EP contract, resource conflicts, and memory ordering remain dynamically managed by the backend. Cycle-accurate RTL evaluation shows that, compared with an Ara-based tightly coupled RVV implementation (TC), SEAM-V achieves a geometric-mean speedup of 1.34x across 17 representative kernels. The one-dimensional variable-AVL, BLAS and matrix, and fixed-size application groups achieve speedups of 1.50x, 1.25x, and 1.27x, respectively. At AVL=32, the geometric-mean speedup across six one-dimensional vector kernels approaches 3x. |
| 2026-07-20 | [Thermal management optimization of Battery Electric Vehicles via hierarchical NMPC with CMO-trained Neural Models](http://arxiv.org/abs/2607.17892v1) | Francesco Ripa, Diego Regruto et al. | This paper addresses the problem of energy-efficient thermal management in Battery Electric Vehicles (BEVs), with the goal of extending driving range while ensuring passenger comfort and battery thermal safety. A hierarchical Nonlinear Model Predictive Control (NMPC) framework is proposed, in which the supervisory layer explicitly balances energy consumption and cabin temperature reference tracking over a long horizon and translates this trade-off into optimal actuator command sequences for the integrated heating, ventilation, and air conditioning (HVAC) system and the battery thermal management (BTM) system. These command sequences are then enforced by a lower-layer NMPC, which introduces local corrections to compensate disturbances and modeling uncertainty while tracking an externally specified cabin temperature reference trajectory. To enable predictive control of the highly nonlinear and strongly coupled thermal dynamics, data-driven neural network models are employed as prediction models within both control layers. The proposed approach is developed and validated using a high-fidelity BEV thermal simulator developed at Politecnico di Torino in collaboration with Stellantis N.V., and benchmarked against an industrial rule-based (RB) control strategy over the WLTC driving cycle. Simulation results demonstrate a significant reduction in energy consumption compared to the baseline strategy, while satisfying comfort and battery temperature constraints. |
| 2026-07-20 | [Stress Testing Concept Erasure with Large Language Model Agents](http://arxiv.org/abs/2607.17890v1) | Yuyang Xue, Feng Chen et al. | Concept erasure aims to remove semantic concepts from a trained generative model and is increasingly important for responsible AI deployment. However, verifying whether a model has robustly removed targeted concepts remains a critical challenge. Existing evaluation methods are typically pre-defined and static, failing to expose vulnerabilities under diverse natural-language probes and challenging conditions. Moreover, manually designed evaluation strategies can be biased and difficult to scale. We posit that concept erasure evaluation is best formulated as an adaptive hypothesis search, operationalised by agents that iteratively propose, critique, and verify tests to systematically expand coverage of failure modes. To this end, we propose Stress Testing Agents for Concept Erasure (STACE), a framework that autonomously stress-tests concept-erased models using multiple Large Language Model (LLM) agents, by iteratively generating and verifying stress-testing hypotheses grounded by external knowledge. We also introduce a suite of metrics for assessing the performance and efficiency of LLM-agent-powered stress-testing frameworks. Our extensive experiments show that STACE outperforms five LLM-based evaluation baselines on four concept categories. Further analysis across two T2I models, six concept erasure approaches, and various erasure strengths show that STACE is robust for different settings. We also show that STACE can be adapted beyond concept erasure evaluation to other problem domains, such as LLM jailbreaking. Our code is available anonymously. |
| 2026-07-20 | [Receiver-Centered Robot-to-Human Handover with Grasp-Aware Object Orientation](http://arxiv.org/abs/2607.17839v1) | Federico Biagi, Dario Onfiani et al. | Collaborative robots are increasingly sharing workspaces with human operators, making tool handover a frequent and safety-critical micro-interaction. However, traditional static handovers often lead to awkward grasps when handling asymmetric industrial tools. This paper presents a receiver-centered voice-driven adaptive handover system for mechanical tools, built on a Franka cobot. Using an LLM for intention recognition and MediaPipe for real-time 3D hand tracking, the framework dynamically adjusts the end-effector's orientation to present tools in an ergonomically optimal, handle-first pose. A within-subjects study compared this adaptive approach with an object-agnostic static baseline. The results demonstrate that the adaptive system reduces the grasp delay for asymmetric tools, improving the fluency of the interaction. Furthermore, the adaptive strategy improved specific trust-related perceptions, particularly motion predictability and perceived task simplicity. |
| 2026-07-20 | [Reasoning as a Double-Edged Sword: Architecture and Cross-Stage Robustness in Vision-Language-Action Models](http://arxiv.org/abs/2607.17786v1) | Tuan Duong Trinh, Naveed Akhtar et al. | Does adding a reasoning step make a Vision-Language-Action (VLA) model more robust to perturbation? Intuitively, a policy that reasons before acting should absorb a perturbed input better than one that maps observations directly to actions. We test this premise head-on across three models that span the reasoning spectrum (no reasoning, a text chain-of-thought, and a latent iterative loop), perturbing each at the vision, reasoning, and action stages on LIBERO and SimplerEnv. Two questions organize the study: does the reasoning design shift robustness, and can the reasoning be read back at runtime as a safety signal? We find that the latent-iterative model is by far the least robust: under both stochastic noise and white-box perturbation its task success collapses, while the other two hold. This fragility is structural rather than cumulative: varying the reasoning depth at inference barely moves it. Reasoning outputs can in principle be monitored, but the monitors fail under fair tests. A plan--action consistency probe that looks near-perfect under naive evaluation falls to chance under adaptive attack. Under matched-FPR calibration, fusing it with an action-anomaly probe never lifts defended success above undefended. Scoped to these output-level behavioral probes under white-box vision-stage attack, this ceiling is a precondition that any viable defense must first satisfy. |
| 2026-07-20 | [ETAS: An Effect-Typed Language for Agent Systems](http://arxiv.org/abs/2607.17780v1) | Huiri Tan, Yikun Wang et al. | ETAS is a programming language for agent systems that treats model-backed agents, tool calls, prompts, typed memory, human approvals, policies, and execution traces as semantic program elements rather than library conventions. It separates deterministic computation from agentic nondeterminism and externally visible actions while preserving a direct programming style.   We present the core design of ETAS. Its static semantics assigns ordinary types through spec conformance and tracks each computation with two behavioral indices: an escaping effect row and a persistent abstraction of the typed action trace it may request. Specs form a terminating compile-time constraint calculus: type specs provide evidence for polymorphism and resource facts, callable specs constrain function and stage shapes, and trace specs express allow, deny, and temporal constraints. Typing checks requested traces against compiled monitors and emits residual obligations when dynamic resources preclude a complete static proof. The dynamic semantics distinguish requested, handled, denied, and committed events; handlers interpret typed actions without making their requests invisible to authorization or audit.   We formalize a core calculus and state preservation, progress, type/effect soundness, handler trace-transparency, and policy safety. We also implement ETAS in Rust with a command-line interface, typed HIR checks, effect and policy diagnostics, handler checks, and trace-aware execution hooks. ETAS provides a programming-language foundation for reasoning about authorization, nondeterminism, recovery, and audit evidence before and during agent execution. |
| 2026-07-20 | [Dynamic Defense Profiling Enables Cognitive Jailbreak of Text-to-Image Models](http://arxiv.org/abs/2607.17779v1) | Dongdong Yang, Deyue Zhang et al. | Text-to-Image (T2I) generative models have achieved remarkable progress in synthesizing high-quality visual content, yet they remain vulnerable to adversarial misuse, particularly in generating Not-Safe-For-Work (NSFW) images. Most existing jailbreak attacks primarily rely on heuristic prompt engineering or black-box optimization, treating model feedback as a binary signal (success or failure). This coarse-grained paradigm overlooks the rich information embedded in diverse failure modes, such as textual refusal, visual blocking, and semantic sanitization, resulting in inefficient exploration and severe semantic collapse.   In this paper, we propose MIND, a cognitive jailbreak framework that reframes adversarial prompt generation as a belief-state inference problem over latent defense mechanisms. Instead of blindly searching for bypass prompts, MIND actively models the target system's latent defense mechanisms by interpreting multi-modal feedback as high-density signals. Specifically, the framework integrates three core components: (1) a Multi-modal Judge for fine-grained feedback decomposition, (2) a Defense Profiler for iterative belief updating, and (3) a Meta-Memory module for retrieving historically effective attack strategies. These components are unified within a reasoning-driven evolutionary optimization process, enabling adaptive and semantically consistent jailbreak generation. Extensive experiments on the I2P benchmark demonstrate the effectiveness of MIND. Under six representative pre-processing and post-processing defense settings applied to the Stable Diffusion v1.5 model, MIND achieves an Attack Success Rate (ASR) of 95.62%, significantly outperforming existing methods. Additionally, the effectiveness of the proposed framework is validated across four widely used commercial T2I systems, achieving the highest ASR of 91.58% on Wan-2.5. |
| 2026-07-20 | [Towards Reliable Zero-Shot Crowd Forecasting: Evaluating Time Series Foundation Models for Special Event Pedestrian Forecasting](http://arxiv.org/abs/2607.17758v1) | Ziteng Li, Yanan Xin et al. | Managing massive crowds during infrequent special events requires reliable real-time pedestrian-flow forecasting to ensure public safety and operational efficiency. However, supervised forecasting methods face limitations in these contexts due to scarce historical data, heterogeneous data distributions, and short in-event observation windows. To effectively support operational decision-making, forecasts should provide not only accurate point estimates but also informative predictive uncertainty. Probabilistic uncertainty quantification plays a critical role in this aspect, particularly capturing sudden volatility and tail risks. This paper investigates pretrained time series foundation models as a lightweight approach for zero-shot probabilistic forecasting without extensive local retraining. Using decision-oriented metrics tailored to short events, we conduct a comprehensive assessment of two time series foundation models on crowd forecasting, with the SAIL2025 event as a use case. We then distill practical insights for crowd managers, specifying when zero-shot forecasts remain operationally reliable. |
| 2026-07-20 | [Byzantine Fault-Tolerant Post-Quantum Distributed Quorum Signatures](http://arxiv.org/abs/2607.17700v1) | Quentin Kniep, Jakub Sliwinski et al. | Threshold, aggregate, and multi-signatures -- which we collectively call quorum signatures -- certify that a quorum of nodes endorsed a statement, with a certificate as small as a single signature. No constant-size post-quantum quorum signature is known: all candidates grow with the number of signers and are slow to aggregate, making quorum signatures the hardest obstacle to migrating byzantine fault-tolerant systems to post-quantum security.   In this paper, we sidestep this open cryptographic problem by changing how the protocol communicates. We introduce a primitive we call Distributed Quorum Signature (DQS), built solely from ordinary digital signatures and a Bracha-style approval broadcast. DQS turns certificates from network messages into local events. Two event types divide the roles certificates play: weak certificates capture safety, strong certificates capture liveness. In DQS every message is constant size, fitting a single datagram regardless of the number of nodes. The total communication is quadratic, and no security assumptions change. In a large distributed system, the overhead of post-quantum DQS is competitive with the canonical pre-quantum BLS scheme. |
| 2026-07-20 | [BTOR2-Based C Program Verification via Hardware Model Checking](http://arxiv.org/abs/2607.17622v1) | Xinyu Zhang, Runxuan Fang et al. | Program verification tools often rely on specific intermediate representations and analysis backends, limiting the reuse of verification algorithms and model checkers across frameworks. In contrast, hardware model checking has developed a mature backend ecosystem, where standard formats such as BTOR2 support reusable algorithms for counterexample search and inductive safety proving. Applying these capabilities to C requires translating assertion-based programs into transition systems that hardware model checkers can directly process.   We present C2Btor, a method for encoding such verification tasks into BTOR2 models. C2Btor uses a program counter to capture control transfers, represents data states and memory objects with bit-vectors and arrays, and maps assumptions and assertion checks into BTOR2 constraints and bad-state properties.   We evaluate C2Btor on SV-COMP C ReachSafety benchmarks and a curated assertion-category benchmark suite, comparing it with representative program verification tools. C2Btor correctly solves 263 tasks, 101 more than CBMC configured with bounded model checking, and is especially effective on bit-vector benchmarks, where it solves 75.5% of the tasks with no wrong verdicts. These results show that the BTOR2 route allows C program verification to benefit from advances in hardware model-checking backends, expanding the available capability for counterexample search, inductive safety proving, and word-level transition-system reasoning. |

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



