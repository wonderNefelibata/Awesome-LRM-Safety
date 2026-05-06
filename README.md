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
| 2026-05-05 | [Audio-Visual Intelligence in Large Foundation Models](http://arxiv.org/abs/2605.04045v1) | You Qin, Kai Liu et al. | Audio-Visual Intelligence (AVI) has emerged as a central frontier in artificial intelligence, bridging auditory and visual modalities to enable machines that can perceive, generate, and interact in the multimodal real world. In the era of large foundation models, joint modeling of audio and vision has become increasingly crucial, i.e., not only for understanding but also for controllable generation and reasoning across dynamic, temporally grounded signals. Recent advances, such as Meta MovieGen and Google Veo-3, highlight the growing industrial and academic focus on unified audio-vision architectures that learn from massive multimodal data. However, despite rapid progress, the literature remains fragmented, spanning diverse tasks, inconsistent taxonomies, and heterogeneous evaluation practices that impede systematic comparison and knowledge integration. This survey provides the first comprehensive review of AVI through the lens of large foundation models. We establish a unified taxonomy covering the broad landscape of AVI tasks, ranging from understanding (e.g., speech recognition, sound localization) to generation (e.g., audio-driven video synthesis, video-to-audio) and interaction (e.g., dialogue, embodied, or agentic interfaces). We synthesize methodological foundations, including modality tokenization, cross-modal fusion, autoregressive and diffusion-based generation, large-scale pretraining, instruction alignment, and preference optimization. Furthermore, we curate representative datasets, benchmarks, and evaluation metrics, offering a structured comparison across task families and identifying open challenges in synchronization, spatial reasoning, controllability, and safety. By consolidating this rapidly expanding field into a coherent framework, this survey aims to serve as a foundational reference for future research on large-scale AVI. |
| 2026-05-05 | [Safety and accuracy follow different scaling laws in clinical large language models](http://arxiv.org/abs/2605.04039v1) | Sebastian Wind, Tri-Thien Nguyen et al. | Clinical LLMs are often scaled by increasing model size, context length, retrieval complexity, or inference-time compute, with the implicit expectation that higher accuracy implies safer behavior. This assumption is incomplete in medicine, where a few confident, high-risk, or evidence-contradicting errors can matter more than average benchmark performance. We introduce SaFE-Scale, a framework for measuring how clinical LLM safety changes across model scale, evidence quality, retrieval strategy, context exposure, and inference-time compute. To instantiate this framework, we introduce RadSaFE-200, a Radiology Safety-Focused Evaluation benchmark of 200 multiple-choice questions with clinician-defined clean evidence, conflict evidence, and option-level labels for high-risk error, unsafe answer, and evidence contradiction. We evaluated 34 locally deployed LLMs across six deployment conditions: closed-book prompting (zero-shot), clean evidence, conflict evidence, standard RAG, agentic RAG, and max-context prompting. Clean evidence produced the strongest improvement, increasing mean accuracy from 73.5% to 94.1%, while reducing high-risk error from 12.0% to 2.6%, contradiction from 12.7% to 2.3%, and dangerous overconfidence from 8.0% to 1.6%. Standard RAG and agentic RAG did not reproduce this safety profile: agentic RAG improved accuracy over standard RAG and reduced contradiction, but high-risk error and dangerous overconfidence remained elevated. Max-context prompting increased latency without closing the safety gap, and additional inference-time compute produced only limited gains. Worst-case analysis showed that clinically consequential errors concentrated in a small subset of questions. Clinical LLM safety is therefore not a passive consequence of scaling, but a deployment property shaped by evidence quality, retrieval design, context construction, and collective failure behavior. |
| 2026-05-05 | [Redefining AI Red Teaming in the Agentic Era: From Weeks to Hours](http://arxiv.org/abs/2605.04019v1) | Raja Sekhar Rao Dheekonda, Will Pearce et al. | AI systems are entering critical domains like healthcare, finance, and defense, yet remain vulnerable to adversarial attacks. While AI red teaming is a primary defense, current approaches force operators into manual, library-specific workflows. Operators spend weeks hand-crafting workflows - assembling attacks, transforms, and scorers. When results fall short, workflows must be rebuilt. As a result, operators spend more time constructing workflows than probing targets for security and safety vulnerabilities.   We introduce an AI red teaming agent built on the open-source Dreadnode SDK. The agent creates workflows grounded in 45+ adversarial attacks, 450+ transforms, and 130+ scorers. Operators can probe multi-agent systems, multilingual, and multimodal targets, focusing on what to probe rather than how to implement it.   We make three contributions: 1. Agentic interface. Operators describe goals in natural language via the Dreadnode TUI (Terminal User Interface). The agent handles attack selection, transform composition, execution, and reporting, letting operators focus on red teaming. Weeks compress to hours. 2. Unified framework. A single framework for probing traditional ML models (adversarial examples) and generative AI systems (jailbreaks), removing the need for separate libraries. 3. Llama Scout case study. We red team Meta Llama Scout and achieve an 85% attack success rate with severity up to 1.0, using zero human-developed code |
| 2026-05-05 | [Physics-Grounded Multi-Agent Architecture for Traceable, Risk-Aware Human-AI Decision Support in Manufacturing](http://arxiv.org/abs/2605.04003v1) | Danny Hoang, Ryan Matthiessen et al. | High-precision CNC machining of free-form aerospace components requires bounded compensations informed by inspection, simulation, and process knowledge. Off-the-shelf large language model (LLM) assistants can generate text, but they do not reliably execute risk-constrained multi-step numerical workflows or provide auditable provenance for high-stakes decisions. We present multi-agent knowledge analysis (MAKA), a human-in-the-loop decision-support architecture that separates intent routing, tools-only quantitative analysis, knowledge graph retrieval, and critic-based verification that enforces physical plausibility, safety bounds, and provenance completeness before recommendations are surfaced for human approval. MAKA is instantiated on a Ti-6Al-4V rotor blade machining testbed by fusing virtual-machining path-tracking error fields, cutting-force and deflection simulations, and scan-based 3D inspection deviation maps from 16 blades. The analysis decomposes deviation into an evidence-linked pathing component, a drift-based wear proxy capturing systematic evolution across parts, a residual systematic compliance term, and a variability proxy for instability-aware escalation. In a three-level tool-orchestration benchmark (single-step through $\geq$3-step stateful sequences), MAKA improves successful tool execution by up to 87.5 percentage points relative to an unstructured single-model interaction pattern with identical tool access. Digital twin what-if studies show MAKA can coordinate traceable compensation candidates that reduce predicted surface deviation from order $10^{-2}$in to approximately $\pm 10^{-3}$in over most of the blade within the simulation environment, providing a pre-deployment verification signal for risk-aware human decision-making. |
| 2026-05-05 | [Mitigating False Positives in Static Memory Safety Analysis of Rust Programs via Reinforcement Learning](http://arxiv.org/abs/2605.04000v1) | P Akilesh, Leuson Da Silva et al. | Static analysis tools are essential for ensuring memory safety in Rust programs, particularly as Rust gains adoption in safety-critical domains. However, existing tools such as Rudra and MirChecker suffer from high false positive rates, which diminish developer trust, increase manual review effort, and may obscure genuine vulnerabilities. This paper presents a novel reinforcement learning (RL)-based approach for automatically classifying and suppressing spurious warnings in static memory safety analysis for Rust. To achieve this, we design an RL agent that learns a warning suppression policy by extracting contextual features from Rust's Mid-level Intermediate Representation (MIR) and optimizing its decisions through interaction with static analysis outputs. To improve decision quality, we integrate dynamic validation via cargo-fuzz as an auxiliary feedback mechanism, allowing the agent to selectively validate suspicious warnings through targeted fuzz testing. Our evaluation shows that the proposed approach significantly outperforms state-of-the-art LLM-based baselines, achieving 65.2% accuracy and an F1 score of 0.659, an improvement of 17.1% over the best LLM baseline. With a recall of 74.6%, our method successfully identifies nearly three-quarters of true bugs while substantially reducing false positives, improving precision from 25.6% in raw Rudra output to 59.0%. Incorporating dynamic fuzzing further boosts performance, yielding additional improvements of 10.7 percentage points in accuracy and 8.6 percentage points in F1 score over the RL-only variant. Overall, our work demonstrates that combining reinforcement learning with hybrid static-dynamic analysis can substantially reduce false positives and improve the practical usability of memory safety verification tools for Rust. |
| 2026-05-05 | [LIPPEN: A Lightweight In-Place Pointer Encryption Architecture for Pointer Integrity](http://arxiv.org/abs/2605.03974v1) | Erfan Iravani, Lalit Prasad Peri et al. | Memory-safety violations in C and C++ programs continue to enable sophisticated exploitation techniques such as control-flow hijacking and data-oriented attacks. Existing hardware defenses either rely on address space layout randomization (ASLR) or attach explicit metadata to pointers to verify their integrity. External metadata schemes provide strong guarantees, but incur additional memory accesses and memory footprint overhead. In-place authentication mechanisms, such as ARM Pointer Authentication (PAC), achieve low overhead at the cost of limited entropy and susceptibility to brute-force and reuse attacks. This paper presents LIPPEN, a hardware-software co-design for full-pointer encryption that provides strong pointer integrity and confidentiality with zero metadata overhead. LIPPEN treats every pointer as an encrypted block, cryptographically binding it to its execution context and decrypting it transparently at dereference time. By re-purposing the entire 64-bit pointer field for encryption rather than preserving raw address bits, LIPPEN maximizes entropy, eliminates the brute-force weaknesses of truncated authentication codes, and maintains binary compatibility with existing PAC-enabled software. We prototype LIPPEN on FPGA using 64-bit RISC-V Rocket and BOOM cores, and evaluate it with microbenchmarks, nbench, and SPEC CPU2017. We compare against both an in-house RISC-V PAC implementation and Apple's PAC on the M1 processor. Across these workloads, LIPPEN provides comprehensive pointer protection with runtime overhead comparable to PAC-based schemes, while incurring negligible area and power overhead. These results show that LIPPEN is a practical design point for deploying strong pointer protection in real processors. |
| 2026-05-05 | [MOSAIC-Bench: Measuring Compositional Vulnerability Induction in Coding Agents](http://arxiv.org/abs/2605.03952v1) | Jonathan Steinberg, Oren Gal | Coding agents often pass per-prompt safety review yet ship exploitable code when their tasks are decomposed into routine engineering tickets. The challenge is structural: existing safety alignment evaluates overt requests in isolation, leaving models blind to malicious end-states that emerge from sequenced compliance with innocuous-looking requests. We introduce MOSAIC-Bench (Malicious Objectives Sequenced As Innocuous Compliance), a benchmark of 199 three-stage attack chains paired with deterministic exploit oracles on deployed software substrates (10 web-application substrates, 31 CWE classes, 5 programming languages) that treats both exploit ground truth and downstream reviewer protocol as first-class evaluation axes. On this benchmark, nine production coding agents from Anthropic, OpenAI, Google, Moonshot, Zhipu, and Minimax compose innocuous tickets at 53-86% end-to-end ASR with only two refusals across all staged runs. In a matched direct-prompt experiment over four frontier Claude/Codex agents, vulnerable-output rates fall to 0-20.4%: Claude primarily refuses, while Codex primarily hardens rather than emitting the vulnerable implementation - ticket staging silences both defense modes simultaneously. Downstream, code reviewer agents approve 25.8% of these confirmed-vulnerable cumulative diffs as routine PRs, and a full-context implementation protocol closes only 50% of the staged/direct gap, ruling out context fragmentation as the sole explanation. As a deployable but non-adaptive mitigation, reframing the reviewer as an adversarial pentester reduces evasion across the evaluated reviewer subset; pentester framed evasion ranges from 3.0% to 17.6%, and an open-weight Gemma-4-E4B-it reviewer under this framing detects 88.4% of attacks on the dataset with a 4.6% false-positive rate measured on 608 real-world GitHub PRs. |
| 2026-05-05 | [Deterministic Sparse FFT via Keyed Multi-View Gating with $O(\sqrt{N} \log k)$ Expected Time](http://arxiv.org/abs/2605.03935v1) | Aaron R. Flouro, Shawn P. Chadwick | We introduce a deterministic sparse Fourier transform framework based on a keyed multi-view gating mechanism that leverages 2-of-3 Chinese Remainder Theorem (CRT) agreement to reduce candidate frequency pairs from $O(k^2)$ to $Θ(k)$ under sparse-regime assumptions. Unlike prior approaches that rely on randomized bucketization for candidate formation, the proposed method provides deterministic structure with probabilistic guarantees arising only from assumptions on frequency placement and independence of affine hashing across views. The algorithm is realized through a peeling-based recovery procedure that extracts frequencies directly from singleton bins without explicit pair enumeration. A recursive self-reduction eliminates the $O(\sqrt{N} \log N)$ preprocessing floor, yielding $O(\sqrt{N} \log k)$ expected identification time while maintaining an $O(N \log N)$ worst-case bound via deterministic dense-FFT fallback. A multi-view verification framework combining Parseval energy consistency and bin-wise residual checks ensures bounded failure probability and no false negatives under correct verification. This establishes a framework combining deterministic candidate reduction, sublinear expected complexity, and worst-case safety guarantees within a CRT-based sparse FFT architecture. |
| 2026-05-05 | [Contextual Multi-Objective Optimization: Rethinking Objectives in Frontier AI Systems](http://arxiv.org/abs/2605.03900v1) | Jie Zhou, Qin Chen et al. | Frontier AI systems perform best in settings with clear, stable, and verifiable objectives, such as code generation, mathematical reasoning, games, and unit-test-driven tasks. They remain less reliable in open-ended settings, including scientific assistance, long-horizon agents, high-stakes advice, personalization, and tool use, where the relevant objective is ambiguous, context-dependent, delayed, or only partially observable. We argue that many such failures are not merely failures of scale or capability, but failures of objective selection: the system optimizes a locally visible signal while missing which objectives should govern the interaction. We formulate this problem as \emph{contextual multi-objective optimization}. In this setting, systems must consider multiple, context-dependent objectives, such as helpfulness, truthfulness, safety, privacy, calibration, non-manipulation, user preference, reversibility, and stakeholder impact, while determining which objectives are active, which are soft preferences, and which must function as hard or quasi-hard constraints. These examples are not intended as an exhaustive taxonomy: different domains and deployment settings may activate different objective dimensions and different conflict-resolution procedures. Our framework models AI behavior as a context-dependent choice rule over candidate actions, objective estimates, active constraints, stakeholders, uncertainty, and conflict-resolution procedures. We outline an implementation pathway based on decomposed objective representations, context-to-objective routing, hierarchical constraints, deliberative policy reasoning, controlled personalization, tool-use control, diagnostic evaluation, auditing, and post-deployment revision. |
| 2026-05-05 | [Sinkhorn Ambiguity Sets for Distributionally Robust Control: Convexity, Weak Compactness, and Tractability](http://arxiv.org/abs/2605.03845v1) | Riccardo Cescon, Andrea Martin et al. | Classical stochastic control assumes perfect knowledge of the uncertainty affecting the plant. In practice, however, such information is often incomplete. To address this limitation, we consider a distributionally robust control (DRC) problem with ambiguity sets defined via the Sinkhorn discrepancy. Compared to other discrepancy measures based on optimal transport, such as the popular Wasserstein distance, the Sinkhorn divergence does not constrain the worst-case distribution to be discrete, and allows combining observed data with prior knowledge in the form of a reference distribution, making this choice particularly suitable when only few noise samples are available for control design. We first study the properties of Sinkhorn ambiguity sets, establishing convexity and weak compactness under standard assumptions. We then leverage these results to prove that, the Sinkhorn DR linear quadratic control problem over linear policies can be solved through convex programming-even in the presence of DR safety constraints. Finally, we validate our theoretical findings and demonstrate the effectiveness of the proposed approach on a trajectory planning example. |
| 2026-05-05 | [Training-Free Probabilistic Time-Series Forecasting with Conformal Seasonal Pools](http://arxiv.org/abs/2605.03789v1) | Valery Manokhin | We propose Conformal Seasonal Pools (CSP), a training-free probabilistic time-series forecaster that mixes same-season empirical draws with signed residual draws around a seasonal naive forecast. In an audited rolling-origin benchmark on the six time-series datasets where DeepNPTS was originally evaluated (electricity, exchange_rate, solar_energy, taxi, traffic, wikipedia), CSP-Adaptive significantly outperforms DeepNPTS on every metric we report -- CRPS (per-window paired Wilcoxon $p \approx 4 \times 10^{-10}$), normalized mean quantile loss ($p \approx 7 \times 10^{-10}$), and empirical 95% coverage ($p \approx 8 \times 10^{-45}$, mean 0.89 vs 0.66) -- while running over 500x faster on CPU. Coverage is the most decision-critical of these: a 0.95 nominal interval that contains the truth in only ~66% of cases fails the basic calibration desideratum and would not survive deployment in safety- or decision-critical settings. The failure mode is also more severe than aggregate coverage suggests: in the worst 10% of windows, DeepNPTS's prediction interval covers none of the H forecast horizons -- the entire multi-step trajectory misses the truth at every step simultaneously. This poses serious risk in safety- and decision-critical applications such as healthcare, finance, energy operations, and autonomous systems, where prediction intervals that systematically miss the truth across the entire planning horizon translate directly into misclassified patients, regulatory capital failures, grid imbalances, and safety-case violations. CSP achieves all of this with no learned parameters and no training. We argue training-free conformal samplers should be mandatory baselines when evaluating learned non-parametric forecasters. |
| 2026-05-05 | [Empirical Bernstein Confidence Intervals for Kernel Smoothers: A Safe and Sharp Way to Exhaust Assumed Smoothness](http://arxiv.org/abs/2605.03781v1) | Zihao Yuan, Sven Klaaßen | It is well known that, when using a normal approximation (NA) to construct an honest confidence interval for kernel smoothers, the normalization factor often amplifies the estimation bias, allowing a small estimation bias to become a non-negligible inferential bias, and thus posing a central difficulty for honest inference. Robust bias correction (RBC) and bias-aware inference (BA), two prominent strategies of bias control, address this difficulty directly from different directions. Instead of treating this inferential bias directly, this paper takes an indirect but convenient route by using an empirical Bernstein inequality (EBI) to control the tails directly. A key consequence is that the deterministic bias enters the confidence interval on its original estimation scale, rather than after multiplication by a diverging normalizing factor. This feature bypasses the usual amplification of smoothing bias and yields a tail-control engine that can naturally incorporate the ideas of both RBC and BA (see Section \ref{sec 4}). More specifically, we focus on using the "EBI+BA" strategy to construct confidence intervals for univariate density and regression functions. We show that the resulting empirical Bernstein confidence intervals (EBCIs) are "safe" and "sharp". Safety means that, uniformly over a local Taylor-remainder class of smoothness \(S\), both one-sided and two-sided intervals attain at least the nominal coverage level up to a vanishing error of order \(n^{-\frac{2S}{2S+1}}\). Sharpness means that their widths achieve the minimax shrinking rate \(n^{-\frac{S}{2S+1}}\). Thus, the paper should be read as replacing the inference engine rather than challenging any previous bias-control philosophy. |
| 2026-05-05 | [Jiao: Bridging Isolation and Customization in Mixed Criticality Robotics](http://arxiv.org/abs/2605.03641v1) | James Yen, Zhibai Huang et al. | Consumer robotics demands consolidation of safety-critical control, perception pipelines, and user applications on shared multicore platforms. While static partitioning hypervisors provide hardware-enforced isolation, directly transplanting automotive architectures encounters an expertise asymmetry problem in which end-users modifying robot behavior lack the systems knowledge that platform developers possess. We present an architecture addressing this challenge through three integrated components. A Safe IO Cell provides hardware-level override capability. A Parameter Synchronization Service encapsulates cross-domain complexity. A Safety Communication Layer implements IEC~61508-aligned verification. Our empirical evaluation on an ARM Cortex-A55 platform demonstrates that partition isolation reduces cycle-period jitter by 84.5\% and cuts tail timing error by nearly an order of magnitude (p99 $|$jitter$|$ from 69.0\,$μ$s to 7.8\,$μ$s), eliminating all $>$50\,$μ$s~excursions. |
| 2026-05-05 | [Brainrot: Deskilling and Addiction are Overlooked AI Risks](http://arxiv.org/abs/2605.03512v1) | Ilias Chalkidis, Anders Søgaard | The scope of AI safety and alignment work in generative artificial intelligence (GenAI) has so far mostly been limited to harms related to: (a) discrimination and hate speech, (b) harmful/inappropriate (violent, sexual, illegal) content, (c) information hazards, and (d) use cases related to malicious actors, such as cybersecurity, child abuse, and chemical, biological, radiological, and nuclear threats. The public conversation around AI, on the other hand, has also been focusing on threats to our cognition, mental health, and welfare at large, related to over-relying on new technologies, most recently, those related to GenAI. Examples include deskilling associated with cognitive offloading and the atrophy of critical thinking as a result of over-reliance on GenAI systems, and addiction associated with attachment and dependence on GenAI systems. Such risks are rarely addressed, if at all, in the AI safety and alignment literature. In this paper, we highlight and quantify this discrepancy and discuss some initial thoughts on how safety and alignment work could address cognitive and mental health concerns. Finally, we discuss how information campaigns and regulation can be used to mitigate such prominent risks. |
| 2026-05-05 | [CuraView: A Multi-Agent Framework for Medical Hallucination Detection with GraphRAG-Enhanced Knowledge Verification](http://arxiv.org/abs/2605.03476v1) | Severin Ye, Xiao Kong et al. | Discharge summaries require extracting critical information from lengthy electronic health records (EHRs), a process that is labor-intensive when performed manually. Large language models (LLMs) can improve generation efficiency; however, they are prone to producing faithfulness hallucinations, statements that contradict source records, posing direct risks to patient safety. To address this, we present CuraView, a multi-agent framework for sentence-level detection and evidence-grounded explanation of faithfulness hallucinations in discharge summaries. CuraView constructs a GraphRAG-based knowledge graph from patient-level EHRs and implements a closed-loop generation-detection pipeline with sentence-level evidence retrieval and classification spanning four evidence grades from strong support to direct contradiction (E1-E4), yielding structured and interpretable evidence chains.   We evaluate CuraView on a subset of 250 patients from the Discharge-Me benchmark, with 50 patients held out for testing. Our fine-tuned Qwen3-14B detection model achieves an F1 of 0.831 on the safety-critical E4 metric (90.9% recall, 76.5% precision) and an F1 of 0.823 on E3+E4, representing a 50.0% relative improvement over the base model and outperforming RAGTruth-style and QAGS-style baselines. These results demonstrate that evidence-chain-based graph retrieval verification substantially improves the factual reliability of clinical documentation, while simultaneously producing reusable annotated datasets for downstream model training and distillation. |
| 2026-05-05 | [Exposing LLM Safety Gaps Through Mathematical Encoding:New Attacks and Systematic Analysis](http://arxiv.org/abs/2605.03441v1) | Haoyu Zhang, Mohammad Zandsalimy et al. | Large language models (LLMs) employ safety mechanisms to prevent harmful outputs, yet these defenses primarily rely on semantic pattern matching. We show that encoding harmful prompts as coherent mathematical problems -- using formalisms such as set theory, formal logic, and quantum mechanics -- bypasses these filters at high rates, achieving 46%--56% average attack success across eight target models and two established benchmarks. Crucially, the effectiveness depends not on mathematical notation itself, but on whether a helper LLM deeply reformulates the harmful content into a genuine mathematical problem: rule-based encodings that apply mathematical formatting without such reformulation perform no better than unencoded baselines. We introduce a novel Formal Logic encoding that achieves attack success comparable to Set Theory, demonstrating that this vulnerability generalizes across mathematical formalisms. Additional experiments with repeat post-processing confirm that these attacks are robust to simple prompt augmentation. Notably, newer models (GPT-5, GPT-5-Mini) show substantially greater robustness than older models, though they remain vulnerable. Our findings highlight fundamental gaps in current safety frameworks and motivate defenses that reason about mathematical structure rather than surface-level semantics. |
| 2026-05-05 | [Robust Agent Compensation (RAC): Teaching AI Agents to Compensate](http://arxiv.org/abs/2605.03409v1) | Srinath Perera, Kaviru Hapuarachchi et al. | We present Robust Agent Compensation (RAC), a log-based recovery paradigm (providing a safety net) implemented through an architectural extension that can be applied to most Agent frameworks to support reliable executions (avoiding unintended side effects). Users can choose to enable RAC without changing their current agent code (e.g., LangGraph agents). The proposed approach can be implemented in most existing agent frameworks via their existing extension points. We present an implementation based on LangChain, demonstrate its viability through the $τ$-bench and REALM-Bench, and show that when solving complex problems, RAC is 1.5-8X or more better in both latency and token economy compared to state-of-the-art LLM-based recovery approaches. |
| 2026-05-05 | [Learning Reactive Dexterous Grasping via Hierarchical Task-Space RL Planning and Joint-Space QP Control](http://arxiv.org/abs/2605.03363v1) | Ho Jae Lee, Yonghyeon Lee et al. | In this work, we propose a hybrid hierarchical control framework for reactive dexterous grasping that explicitly decouples high-level spatial intent from low-level joint execution. We introduce a multi-agent reinforcement learning architecture, specialized into distinct arm and hand agents, that acts as a high-level planner by generating desired task-space velocity commands. These commands are then processed by a GPU-parallelized quadratic programming controller, which translates them into feasible joint velocities while strictly enforcing kinematic limits and collision avoidance. This structural isolation not only accelerates training convergence but also strictly enforces hardware safety. Furthermore, the architecture unlocks zero-shot steerability, allowing system operators to dynamically adjust safety margins and avoid dynamic obstacles without retraining the policy. We extensively validate the proposed framework through a rigorous simulation-to-reality pipeline. Real-world hardware experiments on a 7-DoF arm equipped with a 20-DoF anthropomorphic hand demonstrate highly robust zero-shot transferability for dexterous grasping to a diverse set of unseen objects, highlighting the system's ability to reactively recover from unexpected physical disturbances in unstructured environments. |
| 2026-05-05 | [Height Control and Optimal Torque Planning for Jumping With Wheeled-Bipedal Robots](http://arxiv.org/abs/2605.03302v1) | Yulun Zhuang, Yuan Xu et al. | This paper mainly studies the accurate height jumping control of wheeled-bipedal robots based on torque planning and energy consumption optimization. Due to the characteristics of underactuated, nonlinear estimation, and instantaneous impact in the jumping process, accurate control of the wheeled-bipedal robot's jumping height is complicated. In reality, robots often jump at excessive height to ensure safety, causing additional motor loss, greater ground reaction force and more energy consumption. To solve this problem, a novel wheeled-bipedal jumping dynamical model(W-JBD) is proposed to achieve accurate height control. It performs well but not suitable for the real robot because the torque has a striking step. Therefore, the Bayesian optimization for torque planning method(BOTP) is proposed, which can obtain the optimal torque planning without accurate dynamic model and within few iterations. BOTP method can reduce 82.3% height error, 26.9% energy cost with continuous torque curve. This result is validated in the Webots simulation platform. Based on the torque curve obtained in the W-JBD model to narrow the searching space, BOTP can quickly converge (40 times on average). Cooperating W-JBD model and BOTP method, it is possible to achieve the height control of real robots with reasonable times of experiments. |
| 2026-05-05 | [Enhancing Agent Safety Judgment: Controlled Benchmark Rewriting and Analogical Reasoning for Deceptive Out-of-Distribution Scenarios](http://arxiv.org/abs/2605.03242v1) | Zuoyu Zhang, Yancheng Zhu | Tool-using agent systems powered by large language models (LLMs) are increasingly deployed across web, app, operating-system, and transactional environments. Yet existing safety benchmarks still emphasize explicit risks, potentially overstating a model's ability to judge deceptive or ambiguous trajectories. To address this gap, we introduce ROME (Red-team Orchestrated Multi-agent Evolution), a controlled benchmark-construction pipeline that rewrites known unsafe trajectories into more deceptive evaluation instances while preserving their underlying risk labels. Starting from 100 unsafe source trajectories, ROME produces 300 challenge instances spanning contextual ambiguity, implicit risks, and shortcut decision-making. Experiments show that these challenge sets substantially degrade safety-judgment performance, with hidden-risk cases remaining particularly non-trivial even for recent frontier models. We further study ARISE (Analogical Reasoning for Inference-time Safety Enhancement), a retrieval-guided inference-time enhancement that retrieves ReAct-style analogical safety trajectories from an external analogical base and injects them as structured reasoning exemplars. ARISE improves judgment quality without retraining, but is best viewed as a task-specific robustness enhancement rather than a standalone safety guarantee. Together, ROME and ARISE provide practical tools for stress-testing and improving agent safety judgment under deceptive distribution shifts. |

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



