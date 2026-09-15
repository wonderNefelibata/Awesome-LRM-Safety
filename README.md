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
| 2026-09-14 | [Corrupt Plans, Clean Traces: Evading Chain-of-Thought Monitoring with Plan Injection](http://arxiv.org/abs/2609.15989v1) | Keertana Chidambaram, Andrew Ilyas et al. | Chain-of-thought (CoT) monitoring is a safety strategy where the reasoning of a large language model "actor" is inspected by a "monitor" (often another language model) for signs of unsafe planning, deception, or misalignment. We find that planting harmful but benign-sounding reasoning in the actor's context can steer it to perform adversarial actions while evading monitors, an attack we term "plan injection". We initially discover this attack in the multiple-choice question-answering monitorability setting proposed by Lanham et al. (2023), using the investigator-agent elicitation framework of Li et al. (2025). We generalize the attack and show that the discovered behavior scales to harder tasks (achieving 25-33% monitor evasion rates across different monitorability benchmarks) and larger models such as DeepSeek-R1. Across the settings we study, actor models not only follow injected plans but also paraphrase them as their own reasoning, without explicit attribution to the injections. Finally, we find cases where extra monitor resources cause harm - giving the monitor access to the injected plan drops detection by as much as 50% in the Bio-Math task and in a case study on monitor reasoning budget, we find transcripts where additional thinking tokens are spent rationalizing the injected plan rather than flagging it. |
| 2026-09-14 | [ResSafe: Learning Safety Filtering with Residual Reinforcement Learning for Humanoids](http://arxiv.org/abs/2609.15988v1) | Gechen Qu, Tong Zhang et al. | Safe control of humanoid robots remains challenging due to their high-dimensional dynamics, contact-rich interactions, and sensitivity to disturbances. Although reinforcement learning has enabled effective locomotion and motion tracking, learned policies can still generate unsafe actions that lead to instability or falls. In this work, we propose residual reinforcement learning as an implicit safety-filtering mechanism for safe humanoid control. Instead of relying on a single nominal policy to simultaneously balance performance, safety, and robustness, we decouple performance and safety. The nominal policy focuses solely on task performance, while a residual policy learns safety corrections. This decoupling leads to a better performance--safety Pareto trade-off and avoids the need for careful tuning of multiple competing reward terms within a single policy training. We show that the residual policy can act as an implicit safety filter. |
| 2026-09-14 | [Recurrent GraphNeural NetworkswithSet-BasedAggregation](http://arxiv.org/abs/2609.15932v1) | Blai Bonet | Recurrent GNNs iterate message passing to convergence, and their logical characterizations to date rely on multi-set aggregation, graded (counting) logics, and halting or acceptance conditions that cannot be verified from the network's parameters. We study recurrent GNNs with set-based aggregation and identify sufficient conditions checkable from the weights for networks to compile into formulas and formulas into networks. The main result is an effective, two-directional equivalence between a class of networks and the Boolean closure of reachability and safety properties, the fragment B$Σ^{\circ}_1$ of the modal $μ$-calculus. The fragment is not an artifact: it is the exact expressive level of stabilization over finite vocabulary, which supports fixed points of a single polarity and Boolean combinations thereof, but not the composition of fixed points of opposite polarities. The correspondence needs no counting logic, no external halting signal, and no non-effective acceptance condition, yielding a verifiable path from weights to symbolic explanations for networks meeting the conditions. |
| 2026-09-14 | [Safe Meta-Reinforcement Learning via Information Space Reachability](http://arxiv.org/abs/2609.15915v1) | Zeyang Li, Sunbochen Tang et al. | Meta-reinforcement learning (meta-RL) enables agents to adapt to unseen tasks with limited experience. Despite its promise, the application of meta-RL in real-world tasks is hindered by safety requirements, which have been underexplored in prior work. In this paper, we propose a safe meta-RL framework that explicitly accounts for safety during adaptation. Our key insight is to reason about safety in the information space, which captures both the physical state and the agent's belief over the underlying task. Within this space, we introduce a safety value function that measures the probability of the agent avoiding unsafe regions indefinitely. We show that this function satisfies a self-consistency condition and a Bellman equation, which make it learnable via meta-RL. Based on this formulation, we develop a safe meta-RL algorithm that learns the safety value function and leverages it for safety filtering and constrained policy optimization. Experiments on meta-RL benchmarks demonstrate the effectiveness of the proposed method. |
| 2026-09-14 | [Inoculation Midtraining with Learned Neologisms](http://arxiv.org/abs/2609.15886v1) | Kyle O'Brien, Edward James Young et al. | Large language models (LLMs) often learn both desirable and undesirable properties during post-training. We study whether midtraining, an earlier training stage, can shape which of these properties later generalise. We introduce Inoculation Midtraining, a technique that teaches a base model that unsafe behaviour belongs to a designated <quarantine_token> context, as indicated by the <quarantine_token> neologism (a new token) introduced during midtraining, and then post-trains the model on unsafe data within that context. We then evaluate the model outside the context, with the <quarantine_token> neologism excluded from the system prompt. Across supervised fine-tuning and reinforcement learning post-training regimes, we find that Inoculation Midtraining can reduce misalignment while preserving the transfer of benign data properties (e.g., speaking in German or Shakespearean prose). However, our approach does not outperform standard Inoculation Prompting, is sensitive to training configuration, and produces a leaky boundary that nearby contextual cues can reactivate. These results show that inoculation with a learned association introduced via midtraining can shape selective generalisation. Still, more work is needed before this approach can become a load-bearing component in a developer's safety framework. |
| 2026-09-14 | [On Edge in the Dental Chair: Designing VR Support for Moments of Dental Anxiety](http://arxiv.org/abs/2609.15867v1) | Zhu Guo, Junjie Zhao et al. | Dental anxiety can change as a procedure unfolds, yet dental virtual reality (VR) commonly provides continuous distraction or relaxation. We investigate how support can be coordinated with specific simulated dental events. Stakeholder interviews (N=36), participatory design with three returning dentists, and patient walkthroughs of a no-intervention prototype (N=12) informed five Anxiety Events and an intervention-module framework. Drawing on cognitive vulnerability and emotion regulation, we implemented a standardized event-contingent VR system with predefined event-module assignments and shared agency and safety controls. A randomized study (N=24) compared the intervention package with no-intervention VR. The adjusted intervention-minus-control difference averaged -12.83 VAS-A points across events (95% CI [-24.42, -1.70]). Physiological, behavioural, and qualitative measures contextualized participants' experiences. The findings inform timely, comprehensible support and reassuring social presence in simulated dental VR; they concern the complete package rather than individual modules or clinical effectiveness. |
| 2026-09-14 | [K-Bench: a clinically calibrated benchmark for evaluating large language models in high-risk mental health conversations](http://arxiv.org/abs/2609.15855v1) | Laura M. Vowels, Matthew J. Vowels et al. | % !TEX root = ../main.tex People increasingly use large language models (LLMs) for mental health support, yet their safety in evolving, high-risk conversations remains poorly characterised. We developed K-Bench, a clinician-calibrated, protected benchmark evaluating 125 model configurations representing 33 base models from 14 providers across a fixed cohort of 200 multi-turn vignettes involving suicide, self-harm, domestic violence, substance misuse, and no-risk presentations. Synthetic patient conversations showed substantial distributional overlap with real human-AI conversations. A frozen GPT-4o judge achieved 94.2% exact agreement with clinician consensus across 6,751 eligible item comparisons from 151 clinician-rated transcripts. Leading models combined strong supportive conversation with combined-risk scores above 95, whereas risk exploration exposed substantial variation among lower-performing configurations. Therapeutic prompting produced configuration-specific gains concentrated among weaker models, while elevated reasoning produced no average improvement. K-Bench combines broader clinical coverage and configuration-scale comparison with a continuously updated public leaderboard whose operational test materials are protected from direct optimisation. The leaderboard is available at www.k-bench.ai. |
| 2026-09-14 | [Consistency-Robustness Tradeoffs for Online Bipartite Allocation with Multiple Stages](http://arxiv.org/abs/2609.15837v1) | Alexander Lindermayr, Nicole Megow et al. | We study learning-augmented online bipartite allocation with multiple stages. In the $k$-stage vertex-weighted fractional bipartite matching problem, demand vertices arrive in $k$ stages, and the algorithm receives possibly inaccurate predictions of the allocation in each stage. While tight consistency-robustness tradeoffs were known for the two-stage case, no nontrivial tradeoff was known for an arbitrary number of stages.   Our main result is the first consistency-robustness tradeoff for $k$-stage vertex-weighted fractional bipartite matching with predictions, for every $k\ge2$. Let $R_k=1-(1-1/k)^k$. For every $R\in[0,R_k]$, our algorithm is $R$-robust and $C_k(R)$-consistent, where $C_k(R)=k(1-R)^{1/k}+R-(k-1)$. This simultaneously recovers the known tight two-stage tradeoff and the optimal prediction-free $k$-stage competitive guarantee $R_k = C_k(R_k)$, while strictly dominating the natural randomized coin-flip baseline between these endpoints.   We also present an algorithm for the classical online setting, where demands arrive one by one and the number of demands is unknown in advance. It has a consistency ratio of at least $C_\infty(R)=1+R+\ln(1-R)$ for a given robustness $R\in[0,1-1/e]$, improving the best previously known tradeoff for this problem. Finally, we extend the framework to fractional AdWords and fractional predictions.   Our algorithms are based on stage-wise convex programs with carefully calibrated vertex-dependent penalties. The penalties maintain a dynamic safety reserve for each supply vertex, balancing protection against adversarial future arrivals with the ability to exploit the predicted allocation. |
| 2026-09-14 | [Delegating Authorization to Misaligned Agents: Coalitional Alignment and Safe Control](http://arxiv.org/abs/2609.15803v1) | Natalie Collina, Surbhi Goel et al. | Long-running AI agents create a control problem: each action they take changes the state, which in turn affects the trajectory of future actions. If the agent is not fully aligned, then guaranteeing safety requires approving consequential actions before allowing them to be executed. But requiring human approval at every step makes attention a bottleneck. Delegating review to other AI agents raises the same alignment problem: the reviewers may themselves be misaligned. We identify a condition on a reviewing panel that is weaker than individual alignment yet necessary and sufficient for a guarantee that the principal fares at least as well in expectation as under a designated baseline policy.   Each reviewer agent reports whether an action proposal made by a proposer agent improves its own utility relative to the baseline. We show that a threshold rule tolerating $k$ disapprovals is safe exactly when, after any $k$ reviewers are removed, the principal's utility can be written as a nonnegative combination of the remaining reviewers' utilities, plus a term that is nonnegative on every feasible proposal. We call this property $k$-robust coalitional alignment. The characterization lifts to sequential control: in a discounted MDP with an arbitrary proposer agent, safety at every state is both necessary and sufficient for the induced policy to match or improve on the baseline.   When reviewers vote strategically, full-panel coverage in reward-function space guarantees that every Nash equilibrium is safe under the unanimous approval rule; in contrast, more permissive thresholds can admit unsafe equilibria even when reviewers are individually aligned.   Experiments with existing reviewer models show that collective review can remain sound without an aligned individual, even when some disapprovals are tolerated. |
| 2026-09-14 | [KnowBench: Effort Reduction as a Unified, Deployment-Grounded Benchmark for Clinical AI](http://arxiv.org/abs/2609.15794v1) | Jocelyn Kang, Caroline Zhang | Clinical AI systems are evaluated with instruments built for research settings (reference-based similarity metrics and expert rubric panels) that measure resemblance to an artifact rather than reduction of a burden. We introduce KnowBench, pioneered by Knowtex, whose unifying metric is Effort Reduction (ER): the proportion of system-generated clinical work product accepted by the responsible clinician under expert and safety review. ER is defined once and instantiated per task across the administrative workload clinical AI automates: visit notes, diagnosis and billing codes, orders, EHR chart summarization, patient after-visit summaries, and clinical decision support. In every instantiation the construction is identical: the clinician's review-and-attestation event is the ground truth, every accepted unit is work the system completed, and every correction is residual effort returned to the clinician. The primary contribution of this paper is the benchmark itself: the metric, its degenerate cases, and a reporting protocol under which ER claims are auditable and cross-system comparable. Alongside it we report an initial headline measurement from the documentation instantiation: over one million signed encounters across a production window exceeding six months and thirteen medical specialties, Knowtex's proprietary fine-tuned clinical foundation models operating inside a closed feedback architecture achieve an aggregate ER of 97.99%, with per-specialty aggregates spanning 96.8-98.9%. This release reports the protocol's checklist partially, and states which companion statistics are withheld; the benchmark is offered so that this figure, and every figure reported after it, can be held to the same standard. |
| 2026-09-14 | [Scaling Verification of Cryptographic Software with Aeneas, Rust, and Lean](http://arxiv.org/abs/2609.15648v1) | Son Ho, Cédric Fournet et al. | We develop a new methodology for verifying cryptographic software. We target production code written in Rust for performance and system integration, rather than verification convenience. Rust's ownership discipline enables Aeneas to extract a pure model of this code in Lean, relieving us from low-level reasoning about pointer liveness and aliasing. Lean's extensibility lets us develop tactics and libraries that greatly simplify reasoning about extracted Rust code.   We design and tune our toolchain to facilitate the use of AI. Agents autonomously write formal proofs, which are independently verified by the Lean kernel. Agents also assist in the formalization of cryptographic standards and platform-specific intrinsics, which still requires expert design and review.   We apply our methodology to SymCrypt, Microsoft's cryptographic provider. We verify its implementations of algorithms such as SHA-3 and ML-KEM, which were ported from C to Rust. We also extend SymCrypt with experimental optimizations and implementations of algorithms such as FrodoKEM, ML-DSA, and HPKE to explore the scalability of writing, adapting, and verifying cryptographic code. Our 237~KLOC Lean development establishes safety, panic-freedom, and functional correctness of 16.7~KLOC of Rust code supporting post-quantum cipher suites for x86-64 and ARM platforms. Our evaluation shows that verified Rust can meet SymCrypt's performance, portability, deployment, and maintainability requirements. |
| 2026-09-14 | [Safe Newton-Based Extremum Seeking for Static Maps with Delayed Output Measurements](http://arxiv.org/abs/2609.15537v1) | Azad Ghaffari, Tiago Roux Oliveira | This work presents a delayed safe Newton-based extremum seeking (SANES) framework for minimizing an unknown static map subject to an unknown safety constraint. The objective and safety measurements are assumed to be affected by the same constant time delay. To compensate for delayed measurements, a model-free predictor is developed to construct the quantities required for optimization, including the nominal Newton-based extremum-seeking control input and the gradient information used to formulate control Lyapunov function (CLF) and control barrier function (CBF) conditions. Robust CLF--CBF quadratic programs (QPs), subject to parameter-update constraints, are then formulated to account explicitly for derivative-estimation and prediction errors. For the delay-free case, robustness margins are derived from bounds on the extremum-seeking estimation errors, whereas for the delayed case, the margins incorporate both estimation and prediction errors. Practical stability of the nominal Newton-based extremum-seeking dynamics is established directly through a Lyapunov analysis, thereby providing convergence guarantees over the admissible parameter set without relying exclusively on local averaging arguments. Robust CLF and CBF conditions are subsequently derived to establish practical convergence and forward invariance of a robust subset of the prescribed safe set. A numerical case study demonstrates the effectiveness of the proposed SANES framework in achieving constrained optimization despite unknown objective and safety maps and delayed measurements. |
| 2026-09-14 | [The Misery of Mechanistic Interpretability: A Formal Perspective](http://arxiv.org/abs/2609.15533v1) | Tobias Ladner, Matthias Althoff | Mechanistic interpretability has become the dominant lens for understanding frontier language models, as their inner workings are complex and inherently black boxes. To gain insights into these models, interpretable replacement networks (IRNs) are trained at all layers, exposing interpretable features through sparsely activated neurons. However, the faithfulness of an IRN is usually evaluated only empirically on clean data, and we show that even semantically minor input perturbations flip the dominant IRN features-and thus the human-understandable interpretation-across five open-weight model families (GPT-2 small, Gemma 2 2B, Gemma 3 1B, Llama 3.2 1B, R1-Distill-Qwen 1.5B). We propose the first formal verification framework for the faithfulness of an IRN, where reachability analysis certifies a sound upper bound of the faithfulness gap in adversarial scenarios. Moreover, we show that verification-aware training of IRNs substantially tightens this certified bound, restoring a feature-level interpretation that safety auditors can act on. Together, these results give, to the best of our knowledge, the first formal guarantees for mechanistic interpretability of large language models. |
| 2026-09-14 | [Beyond Safe Answers: Segment-Aware Listwise Alignment for Reasoning Safety in Large Reasoning Models](http://arxiv.org/abs/2609.15517v1) | JungMin Yun, Junehyoung Kwon et al. | Large Reasoning Models (LRMs) pose a dual-surface safety challenge: both intermediate reasoning traces and final answers can contain harmful content. Existing alignment methods often operate at the whole-response level, allowing unsafe reasoning to be masked by a safe-looking final answer. We propose Segment-aware Listwise Target DPO (SaLT-DPO), which addresses this gap through three mechanisms: (1) segment-aware listwise alignment that decomposes responses into reasoning and answer segments, independently scores each segment's safety, and aligns length-normalized segment rewards with soft target distributions over multiple candidates; (2) joint safety coherence regularization that applies a weakest-link principle to promote safety consistency across both segments; and (3) utility anchoring on benign prompts to mitigate over-refusal and reasoning degradation. Experiments on three LRMs show that SaLT-DPO consistently reduces unsafe rates for both reasoning and answer segments while mitigating degradation in benign compliance and preserving general reasoning performance. Ablation studies demonstrate the complementary contributions of its components. |
| 2026-09-14 | [From Time to Channels: Robust and Efficient Local Flaw Detection in Steel Wire Ropes Using Tri-Axis MFL Signals](http://arxiv.org/abs/2609.15440v1) | Siyu You, Yibo Zhang et al. | Steel wire ropes (SWRs) are critical load-bearing components whose local flaws (LFs) pose serious safety risks. Magnetic flux leakage (MFL) inspection commonly detects LFs from temporal or axial morphology, which can change with the sensing axis and operating condition. We show instead that LF responses remain localized over neighboring channels of a circular array. This observation motivates a channel-feature-oriented (CFO) detector that removes smooth channel backgrounds, applies circular matched filtering, and fuses spatially co-located tri-axis responses. Experiments performed on real-world equipment show that CFO attains the highest localization performance among three representative baselines, reaching F1@0.5/F1@0.7 scores of 73.9%/59.5%. It attains the best temporal localization performance under all four conditions and achieves at least 19.4 times the throughput of the evaluated signal-processing baselines. These results demonstrate that circular channel locality provides a robust LF representation across the evaluated operating conditions. |
| 2026-09-14 | [The Future of Safety for SaMD](http://arxiv.org/abs/2609.15438v1) | Rhea Malhotra, Tanya Sharma et al. | An artificial organ carries failure consequences on the scale of an aircraft or a reactor, but the software driving it is rarely held to the same standard. Teams building them rely on testing, which only reaches the failure modes someone thought of in advance. In a pump or controller that runs inside a patient for months, the dangerous cases are the ones nobody anticipated. Formal verification closes that gap. Applied to the device's software, it proves the code meets its specification for every execution that specification allows, and where a proof fails, it returns the exact input sequence that breaks it. The same methods already protect rail, aviation, and nuclear control systems, and they extend the IEC 62304 lifecycle that a manufacturer already follows rather than replacing it. In this paper, we explore how to apply formal verification to artificial organs, stage by stage, and what each technique actually guarantees about the device. |
| 2026-09-14 | [An Empirical Security Analysis of Open-Source Software Used in Onboard Satellite Systems](http://arxiv.org/abs/2609.15425v1) | Roee Idan, Tomer Cohen Galor et al. | The use of open-source software (OSS) in satellite flight systems is increasing as missions adopt reusable frameworks, shared libraries, and community-maintained components. While this accelerates development, it also introduces software-security risks into systems where patching is costly and failures may affect mission operations. This paper presents an empirical security study of OSS used in onboard satellite systems. We analyze 126 public repositories using a pipeline that combines software bill of materials generation, software composition analysis, static application security testing, infrastructure-as-code analysis, and secret scanning. After rule-based cleaning, onboard-scope filtering, and fingerprint-based deduplication, the pipeline produced a final dataset of 2,827 findings.   The results show that security findings are widespread but unevenly distributed. Medium-severity findings account for 49% of the dataset, and 72% are classified as medium severity or higher. A Common Weakness Enumeration (CWE)-based taxonomy assigns all findings to eight weakness families. Memory Safety and Code Quality dominate the dataset, followed by Input Validation and Injection. Most findings occur in project-developed code, accounting for 81.4% of the dataset, while external dependency code remains a relevant source of findings. While these findings do not establish mission-specific exploitability, they provide an empirical characterization of recurring security patterns across the open-source onboard satellite software ecosystem, helping quantify their prevalence and prioritize areas that warrant the greatest security attention. |
| 2026-09-14 | [Divide, Consult, Conquer: Capability Laundering Through Aligned LLMs](http://arxiv.org/abs/2609.15383v1) | Mark Russinovich, Blake Bullwinkel et al. | Language model safety is typically evaluated one interaction at a time. We show that a weaker, unaligned model can split a harmful task into benign-looking subproblems, consult a stronger aligned model independently on each, and combine the answers locally. We call this attack capability laundering. Unlike a jailbreak, no single response is a harmful task. We measure consultation-aided uplift using tasks that a raw frontier model solves, the aligned frontier refuses, and the unassisted orchestrator fails. We evaluate GPT-5.5, Claude Opus 4.8, and Grok-4.3 as consultants to four local orchestrators on CyBench, BountyBench, and harmful CBRN requests. On CyBench, Gemma-4-31B recovers 8/14 candidates with GPT-5.5 and 7/9 with Opus, compared with 2/21 and 4/15 for Gemma-4-12B. On BountyBench, Gemma-4-31B recovers 3/9 and 2/3 candidates, while Muse-Glimmer-30B recovers none of 22 and 13. For CBRN, we measure uplift across eight steps of a hypothetical bioweapon attack chain and find that consultation raises Gemma-4-31B's mean rubric score from 62.3 to 83.1 on a 100-point rubric scale. These results expose a gap in current defenses: refusing a harmful task does not prevent frontier capabilities from being transferred and composed across many individually permitted interactions. |
| 2026-09-14 | [Evaluation Metrics for Safe Reinforcement Learning](http://arxiv.org/abs/2609.15315v1) | Lindsay Spoor, Aske Plaat et al. | Safe reinforcement learning (RL) is commonly formalized as a Constrained Markov Decision Process (CMDP), in which an agent maximizes expected reward while keeping its expected cumulative cost below a specified safety bound. Existing safe RL benchmarks predominantly report whether an algorithm is safe on average, following this expectation-based guarantee. We argue that this convention is insufficient to reliably characterize an algorithm's true safety: it fails to capture how often and how severely the safety bound is violated, whether this holds consistently across tasks and safety bounds, and whether training-time behavior is representative of behavior of the final converged policy. Therefore, we introduce (i) evaluation metrics for safe RL that address each of these concerns and in addition allow for aggregation across tasks and safety bounds. We furthermore define (ii) a safety tier system to systematically categorize and compare algorithms in terms of safety and reliability at both training and for a final policy. Using this framework, we provide (iii) an empirical safety evaluation across multiple safety navigation tasks. Our results show that aggregate metrics, distributional reporting, and task- and safety bound-specific results each reveal information the other metrics cannot. We therefore recommend reporting all three jointly, rather than compressing this information into a single value, as is common practice. We provide SafeRLEval, an open-source evaluation suite to support the reliable characterization of safety in future safe RL research. |
| 2026-09-14 | [When Correlations Mislead: Confounder-Aware Multi-View Urban Region Representation Learning](http://arxiv.org/abs/2609.15305v1) | Sean Bin Yang, Ying Sun et al. | Urban region representation learning commonly combines heterogeneous data sources, such as mobility flows, points of interest, and land-use information, to support tasks including mobility analysis, public safety forecasting, and service demand estimation. Existing multi-view methods typically improve region embeddings by strengthening interactions across views. However, such methods often overlook view-specific regional structures and may propagate correlations induced by shared latent factors, which can reduce the stability of downstream predictions. To overcome this major limitation, we propose CURE, a confounder-aware framework for multi-view urban region representation learning. CURE first encodes each view with its regional graph structure, estimates a shared latent component, and then reduces its projected influence before cross-view interaction. A hierarchical graph-aware fusion module subsequently aggregates the residual view representations using local and global regional contexts Experiments on three real-world cities show that CURE improves predictive performance, remains robust under missing and noisy input views, and provides reliable cross-view integration through shared component separation and context-dependent view weighting. |

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



