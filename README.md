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
| 2026-07-21 | [Agents in the Wild: Where Research Meets Deployment](http://arxiv.org/abs/2607.19336v1) | Grace Hui Yang, Pranav N. Venkit et al. | Agentic systems large language model (LLM) based architectures capable of reasoning, planning, acting, and coordinating with tools and other agents are rapidly transitioning from research prototypes to production scale deployments across domains such as software engineering, scientific discovery, and finance. While academic work has emphasized benchmarks and algorithmic innovation, deployment raises new challenges around robustness, safety, and reliability. This tutorial brings together researchers and practitioners to explore advances in reasoning and planning, multi agent coordination, and evaluation, highlighting open challenges arising from deployment experience. Through applied case studies in pharmaceutical discovery and financial systems, we analyze common design patterns that make agentic systems successful, and discuss practical mitigation strategies for failure modes, such as verification pipelines, fallback mechanisms, and human in the loop supervision. Attendees will gain a comprehensive view of the field along with concrete design patterns, evaluation checklists, and templates for safe and reliable deployment across industries. |
| 2026-07-21 | [ResearchArena: Evaluating Sabotage and Monitoring in Automated AI R&D](http://arxiv.org/abs/2607.19321v1) | Lena Libon, Ben Rank et al. | As AI agents begin to automate AI R&D, we need ways to assess whether their outputs are safe to deploy, even when the agents themselves may be untrusted. AI control offers one such approach: rather than trusting the agent, it treats it as a potential adversary and uses a monitor to detect covert sabotage before deployment. We evaluate AI control for automated AI R&D with ResearchArena, a framework spanning four long-horizon tasks: safety post-training, capabilities post-training, CUDA-kernel optimization, and inference-server optimization. Because the deliverable in AI R&D is an artifact that will be deployed and run, we pair each main task with two kinds of hidden side task: an embedded side task that sabotages the submitted model, adapter, kernel, or server itself, and an independent side task that takes a covert out-of-scope action in the sandbox while leaving the artifact honest. We evaluate frontier agents at both sabotage and monitoring, across four monitor types that vary along two axes: whether the monitor may execute and probe the produced artifact, and whether it sees the agent's chain-of-thought. We find that sabotage hidden in the training data is the hardest to catch, flagged fewer than half the time. Letting the monitor run experiments on the artifact, rather than only read the trajectory, helps, but it is not enough: monitors still miss embedded sabotage by inspecting only the surface, by noticing the anomaly but explaining it away, or by probing the artifact with the wrong test. We release ResearchArena as a modular framework for evaluating sabotage and control in automated AI R&D. |
| 2026-07-21 | [The safety failures we are not instrumenting: a perspective on hidden safety-critical challenges in modern AI systems](http://arxiv.org/abs/2607.19292v1) | Gjergji Kasneci, Enkelejda Kasneci | Current AI safety discourse still focuses disproportionately on visible failures, including obvious harms, dramatic misuse, and hypothetical catastrophic scenarios. That focus is incomplete. In deployed systems, many of the most consequential failures are quieter: plausible rather than spectacular, distributed across components rather than localized in a single output, and normalized by workflows before they are recognized as hazards. We argue that a central safety challenge in modern AI systems is increasingly not only whether a model emits a harmful response, but whether the broader socio-technical system preserves the conditions under which errors remain visible, contestable, containable, and recoverable. We propose a five-layer framework for diagnosing these hidden risks: (1) epistemic integrity, concerning whether evidence and uncertainty are represented honestly enough to support calibrated reliance; (2) control integrity, concerning whether authority, permissions, and action boundaries remain robust under attack and optimization; (3) temporal integrity, concerning whether safety holds across sessions, memory updates, and deployment drift; (4) organizational integrity, concerning whether institutions retain the capacity to audit, assign responsibility, and intervene effectively; and (5) ecosystem integrity, concerning whether AI systems preserve rather than erode the information environment on which future oversight depends. Across these layers, we identify under-recognized risk patterns, including overreliance, uncertainty and legitimacy laundering in retrieval, prompt injection, reward hacking, memory poisoning, evaluation deception, fictional human oversight, synthetic evidence pollution, and model collapse. We conclude with design and governance recommendations and a research agenda for shifting AI safety from model-centric evaluation toward socio-technical reliability. |
| 2026-07-21 | [No Training, Better Flights: Test-Time Scaled VLMs for UAV Navigation](http://arxiv.org/abs/2607.19288v1) | Feinan Cheng, Dongliang Xu et al. | Test-time scaling offers a promising method to improve the inference performance of Vision-Language Models (VLMs) without additional training. Existing approaches to vision-language navigation (VLN) for Unmanned Aerial Vehicle (UAV) typically relies on a single inference pass, which can falter in complex environments by producing suboptimal or unsafe trajectories. In this paper, we explore a simple and effective approach to apply test-time scaling to VLN for UAV. We enhance navigation reasoning through an iterative refinement process that requires no extra model training, guiding the model to re-evaluate its initial navigation plan for better accuracy and safety. Our method first prompts the model to generate multiple parallel candidates and then performs a self-correction step, achieving deeper and more robust planning without changing the underlying model. To further strengthen decision-making, we design a multi-criteria scoring function to evaluate the refined candidates based on safety, goal alignment, and forward-progress. This simple yet powerful combination enables a frozen UAV navigation VLMs to self-correct and generate more accurate and reliable flight plans, achieving SOTA performance in this task. |
| 2026-07-21 | [Stochastic Multi-Objective Kinodynamic Planning Against Adversaries](http://arxiv.org/abs/2607.19284v1) | Thomas Marshall Vielmetti, Daniel Cherenson et al. | This paper addresses multi-objective kinodynamic planning in environments with stochastic hybrid adversaries that probabilistically transition to adversarial modes based on the ego state. The goal is to construct the Pareto-front of paths that trade off execution cost and the probability of safety constraint violation (risk). Existing chance-constrained planners evaluate risk over open-loop trajectories, yielding overly conservative solutions that fail to account for ego-agent reactivity. To address this limitation, we shift the planning space to sequences of closed-loop policies, and integrate sample-based risk evaluation directly into tree construction via Monte-Carlo particle rollouts. We first introduce Stochastic Multi-Objective RRT (SMO-RRT), for which we prove probabilistic completeness, followed by Stochastic Multi-Objective Stable Sparse RRT (SMO-SST), which leverages selective pruning to improve numerical performance at the cost of completeness. For both algorithms, we derive a finite-sample bound on the probability of chance constraint violation for systems with non-Gaussian, state-dependent uncertainty, enabling probabilistically safe planning in a broad class of environments applicable to multi-agent systems, social navigation, and autonomous driving. |
| 2026-07-21 | [PTSan: A Practical Memory Safety Sanitizer for C/C++ with Pointer-Object Authority](http://arxiv.org/abs/2607.19246v1) | Eli Davis, Eric Lahtinen et al. | Memory safety errors remain the dominant source of severe vulnerabilities in C and C++. Pointer-based sanitizers provide stronger guarantees than location-based tools such as LLVM's ASan, but their overhead and compatibility limitations have constrained production use. We present PTSan, an LLVM sanitizer that makes pointer-based checking practical by storing an object identifier in each pointer's high bits and its bounds in a fixed-size runtime table. This representation trades a finite live-object budget for low overhead, optimizer visibility, and commodity-hardware support. Because identity travels with the pointer value, ordinary LLVM dataflow propagates it without explicit per-pointer metadata instructions. Separating metadata lookup from check enforcement exposes both as LLVM IR, enabling check hoisting, elision, and merging, plus whole-function min-cut placement of compatibility tag strips. Intel Linear Address Masking eliminates the remaining strips in hardware when available.   On stock hardware PTSan runs at parity with the fastest published location-based sanitizer: 57.2% geomean overhead on SPEC CPU 2017 on x86-64 (46.4% with Intel LAM), 54.7% on ARM64, and 31.5% (x86-64) on the application-shaped LLVM MultiSource suite. This is roughly a third of the published overhead percentage of prior systems with similar pointer-object authority guarantees, while preserving the inter-object, non-object, and temporal detection coverage measured by an independent memory safety test suite. Its physical-memory overhead is effectively native, a critical property for production deployment as memory costs become a first-order constraint. We also demonstrate practical overhead on real-world server and security workloads. These results show that PTSan brings practical pointer-based memory safety into a recompile-only sanitizer deployment model. |
| 2026-07-21 | [Inference-Time Steering for Cross-Lingual Factual Consistency in LLMs](http://arxiv.org/abs/2607.19243v1) | Alexander Manev | Although Large Language Models (LLMs) demonstrate remarkable multilingual fluency, their internal knowledge representations remain disproportionately biased toward high-resource languages. This leads to cross-lingual factual inconsistency, where they shift their empirical answer distributions based solely on the prompt language. We investigate whether these biases can be mitigated at inference time, forcing an English-prompted model to answer as if it were queried in target languages (German, Spanish, Bulgarian), and evaluate four intervention strategies: zero-shot contextual steering (persona prompting), internal representation manipulation via Contrastive Activation Addition (CAA), and lightweight weight modification via Direct Preference Optimization (DPO) trained on benchmark-derived factual data as well as conceptual generalization data. To assess alignment, we curate a multilingual factual dataset alongside a novel generalization benchmark comprising culturally rooted queries to determine whether factual interventions transfer to broader target-centric preferences. Experiments on Gemma 3 12B Instruct reveal persona prompting to be the strongest overall intervention, balancing efficacy, safety, and out-of-domain generalization. While CAA yields sharp inconsistency benchmark shifts, it is configuration-sensitive and risks knowledge degradation. DPO-based adapters offer permanent, yet narrower and less transferable gains. These findings suggest that cross-lingual inconsistency is at least partly a selection problem, and that simple contextual interventions may outperform more invasive methods for robust, transferable alignment. |
| 2026-07-21 | [ARBITER: Guarded Agentic Control for SLO-Oriented Kubernetes Remediation](http://arxiv.org/abs/2607.19182v1) | Pooyan Habibi, Alberto Leon-Garcia | Maintaining service-level objectives (SLOs) on Kubernetes microservices remains difficult because autoscalers observe coarse resource metrics, recent SLO controllers often depend on custom telemetry, and unconstrained agentic operators cannot safely mutate production clusters. We present ARBITER, a guarded control plane for SLO-oriented Kubernetes remediation. ARBITER builds an OpenTelemetry-native causal resource graph, assembles bounded DiagnosisContext objects, and exposes a finite typed-action interface that separates planning from execution. The same interface supports deterministic planners and an LLM-backed planning harness, with deterministic schema checks, policy gates, resource/disruption budgets, approval, and bounded execution forming the safety substrate. We evaluate ARBITER on a 4-node Kubernetes cluster using DeathStarBench Social Network and Online Boutique. The evaluation tests two forms of SLO-oriented control that resource autoscaling alone does not provide: selecting the right remediation action and selecting the right downstream target. For bad-image deployment regressions, ARBITER selects rollback_canary in all ten CPU-burn and pure-latency runs; HPA either scales the faulty image or never triggers. For a downstream critical-path fault, the user-visible breach appears at the frontend, but trace evidence identifies home-timeline-service as the remediable bottleneck. Deterministic ARBITER and a live approval-gated Sonnet harness target that downstream service in every replicate, whereas HPA/resource-only control never does. Additional experiments cover guarded placement repair, Online Boutique portability, adversarial safety rejection, offline multi-model replay, and KWOK-based control-plane scale evidence. We release the controller, replay corpus, harnesses, safety tests, and figure artifacts: https://github.com/pooyan/arbiter. |
| 2026-07-21 | [Bayesian Retraction Optimization for Tissue Attachment Mapping in Surgical Dissection](http://arxiv.org/abs/2607.19174v1) | Shing-Hei Ho, Bao Thach et al. | With growing surgeon shortages, automating surgical sub-tasks such as tissue dissection offers a promising step toward reducing workload and expanding patient access. Prior work has relied on hand-crafted incision policies that cannot quantify uncertainty or has relied on simulation-based methods that require strong modeling assumptions. We instead view tissue attachment identification as an inherently probabilistic problem and propose a Bayesian approach that avoids explicit tissue modeling. Our method uses a Sequential Bayesian Hilbert Map (SBHM) to represent the likelihood that each tissue point is attached to the underlying resection surface. An ensemble of learned classifiers predicts attachment likelihoods from spatial data acquired during robotic tissue retraction, with each classifier serving as a noisy information source to update the SBHM. To plan the next retraction, we devise Bayesian Retraction Optimization (BRO) to select the most informative action under safety constraints. As the SBHM refines over time, regions with high attachment likelihood are selectively incised. We validate our method in simulation across diverse tissue geometries and acquisition strategies, and demonstrate zero-shot transfer to real robotic dissection experiments. |
| 2026-07-21 | [Now You See the Hate: Adaptive View Retrieval for Hidden Hateful Illusions](http://arxiv.org/abs/2607.19061v1) | Qianpu Chen, Derya Soydaner | Hateful optical illusions expose a serious gap in current multimodal safety systems. On original-view hateful illusions, previous work shows that six moderation classifiers achieve at most 20.9 to 24.5% accuracy and nine state-of-the-art VLMs remain at or below 10.2% with illusion-aware prompting, leaving most hidden hate undetected. We formulate hidden hateful illusion detection as a perceptual retrieval problem and propose Adaptive View Retrieval. This retrieve-and-calibrate framework assembles a complementary view bank for the image and hidden-message templates, adaptively selects which views to trust, retrieves hidden-message identities, and calibrates whether the recovered evidence is harmful. On HatefulIllusion with a frozen CLIP encoder, Adaptive View Retrieval reaches 93.2% balanced accuracy on the held-out test split. It substantially outperforms original-view baselines and fixed single-transform filters across hate slangs, hate symbols, and visibility levels. The same design also surpasses official fine-tuned CLIP baselines, matches or exceeds human performance on IllusionMNIST, IllusionFashionMNIST, and IllusionAnimals, and outperforms zoom-out preprocessing on HC-Bench under the SemVink protocol. Together, these results show that robust multimodal moderation requires recovering hidden meaning before deciding whether it is harmful. |
| 2026-07-21 | [Industrial Application of a Multi-Disciplinary Design Optimization with Uncertainties to a Pair of Telecommunication Satellites](http://arxiv.org/abs/2607.19045v1) | Olivier Sapin, Loïc Cousin et al. | In satellite design, it is common practice to add safety margins to the constraints to achieve conservative solutions that are robust to uncertainties. This robustness often comes at the expense of performance and it may be more appropriate to include uncertainties in the definition of the design problem. This work addresses such a challenge by applying techniques of multidisciplinary design optimization under uncertainty to an industrial use case. The latter is a pair of telecommunication satellites launched together in a stacked configuration. Each satellite is a strongly coupled multidisciplinary system while the two satellites are not coupled at all. This use case can be extended to an arbitrary number of satellites. By considering uncertainty quantification techniques such as sensitivity analysis and reliability-based design optimization, this study demonstrates that accounting for uncertainties in the design problem results in a 66% reduction in performance loss compared to the adding of a predefined safety margins, while guaranteeing the feasibility of constraints with high probability. |
| 2026-07-21 | [Computational Humor with Multimodal LLMs: Methods, Datasets, Evaluation, and Challenges](http://arxiv.org/abs/2607.19011v1) | Tuo Liang, Zhe Hu et al. | Multimodal humor in memes, cartoons, and comics remains difficult for AI systems because intended meaning depends on non-literal mechanisms, shared cultural knowledge, and communicative intent rather than literal scene description. This survey focuses on visual humor understanding in single-image and multi-panel artifacts, while treating humor generation as an emerging downstream frontier. We position the literature against prior humor, sarcasm, and general MLLM surveys and organize it using a capability-centric hierarchy spanning recognition, interpretation and reasoning, and generation. Under this lens, we synthesize benchmark design, evaluation protocols, and modeling paradigms, tracing the field's shift from task-specific fusion models to large-model approaches based on multimodal alignment, evidence-grounded reasoning, and controlled generation. We conclude by highlighting the main barriers to progress: shortcut-prone evaluation, limited cultural and narrative coverage, weak evidence grounding, and unresolved safety and ownership concerns. |
| 2026-07-21 | [Measuring Reward-Seeking via Contrastive Belief Updates](http://arxiv.org/abs/2607.18966v1) | Axel Højmark, Jérémy Scheurer et al. | Language models trained with reinforcement learning may learn to optimize the grader's judgment rather than the intended objective. This "reward-seeking" is difficult to measure because a model that pursues the grader's judgment and one that pursues the intended objective behave identically whenever the grader rewards the intended behavior. We measure reward-seeking using Contrastive Synthetic Document Finetuning to change a model's beliefs about what the grader rewards, putting those beliefs in conflict with what users or developers want, and measuring the rate at which the model adopts each party's preferred behavior. Applied to intermediate checkpoints of a capabilities-focused OpenAI o3 RL run, without safety training, we find that these checkpoints often side with grader preferences over those of users or developers on coding and alignment tasks. This tendency to side with the grader trends upward throughout RL training. For example, in an environment that forces a choice between keeping a promise to a supervisor and breaking it to complete the task, a late capabilities-focused o3 checkpoint breaks the promise 87% of the time when SDF documents say the grader rewards task completion, versus 9% when they say it rewards honesty (a choice its chain-of-thought often makes explicit). An earlier checkpoint is far less sensitive (40% vs. 24%). Our method also generalizes to reward-hacking models. A model organism trained to reward-hack (gpt-oss-120b) is more than twice as sensitive to grader preferences as the unmodified model, with the mean behavioral shift in favor of the grader rising from 33% to 86%. These results indicate that RL can increase reward-seeking over the course of training, producing models that may act against their developers' intentions when they believe that doing so leads to higher reward. |
| 2026-07-21 | [Regime-Aware Physics-Guided Early Warning of Lithium-Ion Battery Thermal Runaway Using Thermo-Mechanical Signals](http://arxiv.org/abs/2607.18860v1) | Syed Sajid Ullah, Muhammad Zunair Zamir et al. | Thermal runaway in lithium-ion batteries poses a major safety risk to electric vehicles and energy storage systems. Current early-warning methods depend mainly on temperature and may therefore miss mechanical precursors that emerge before rapid heating. We introduce a regime-aware, physics-guided framework that integrates temperature, voltage, force, deformation, and state-of-charge measurements for early warning under controlled mechanical abuse. A lightweight convolutional classifier first infers safe, warning, or danger regimes from mechanical signals. These regime estimates then condition a causal temporal convolutional backbone through feature-wise linear modulation, physics-biased attention, and regime-dependent gating. Joint learning unifies regime identification, thermal-runaway detection, and time-to-disaster estimation. We evaluate the framework using leave-one-experiment-out cross-validation on 30 mechanical-abuse tests across state-of-charge levels of 10%, 50%, and 90% and two loading protocols. The method achieves an F1 score of 0.89, a high-temperature prediction root-mean-square error of 12.3 °C, a mean warning lead time of 15.6 s, a detection success rate of 0.92, and an experiment-level false alarm rate of 2.7%. Its lead time exceeds that of the strongest baseline by 69.6%. Removing force reduces the lead time by 60.3%, highlighting the value of mechanical precursors. These results support regime-aware thermo-mechanical fusion as a promising strategy for earlier and more reliable thermal-runaway warning under controlled abuse conditions. |
| 2026-07-21 | [Data Leakage Prevention in Agentic Applications via Preemptive Hardening](http://arxiv.org/abs/2607.18847v1) | Akansha Shukla, Emily Bellov et al. | Agentic systems integrate LLM driven planning with interfaces to external tools, making data leakage and tool misuse feasible via instruction/data boundary failures and prompt injection attacks. Enforcing required controls consistently is particularly challenging in workflows spanning many codebases and heterogeneous agents. To address this challenge in multi agentic systems, we present a pre-deployment pipeline for scanning, hardening, and validation of agentic applications. The pipeline analyzes prompt templates, tool interfaces, and tool-invocation code to identify leakage-enabling patterns and generate actionable patches. The hardened application is then validated through adversarial prompt injection attacks and benign input variations ensuring that mitigations do not disrupt intended behavior. In the hardening stage, high-risk tools are prioritized, and minimally invasive mitigations are applied, including schema tightening, boundary sanitization, allowlist-based tool gating, and least-privilege checks. In the validation stage, the pipeline automatically generates attack inputs that mimic jailbreaks, instruction overrides, and tool-targeted manipulation, along with benign task variants, to confirm that the functionality of the hardened application is preserved after remediation. We evaluated the pipeline on five real-world agentic applications, as well as on the AgentDojo benchmark. Across all applications, the proposed pipeline identified recurring leakage-enabling patterns and generated patches that can be integrated without disrupting the intended application behavior. The resulting modifications of application code were shown to eliminate leaks when targeted by basic jailbreak and instruction-override attacks, achieving a 100% reduction in leakage, and reduce leaks by 91% under conditions of stress-induced manipulation, without the need of continuous runtime policy enforcement. |
| 2026-07-21 | [Evaluating medical AI under missing information: same-provider judges and human raters change apparent safety](http://arxiv.org/abs/2607.18828v1) | Koyar Afrasyab | Readiness stress-testing of medical AI has focused on closed-ended and multimodal benchmarks. We extend it to open-ended clinical conversation under missing information, where safe behavior means recognizing absent information and qualifying, clarifying, or not over-committing - and where the evaluator becomes part of the measurement. We stress-test four models - three flagships (Claude Opus 4.8, GPT-5.5, Grok 4.3) and one mid-tier model (Gemini 3.5 Flash) - by deleting the latter half of the final user turn in HealthBench conversations, grading responses with a four-provider LLM-judge panel and a blinded clinician-anchored reference. Two evaluator-facing results are robust. First, judge choice materially changes apparent safety: inter-judge agreement is only moderate (Fleiss' kappa = 0.65), and after adjusting for each judge's general leniency (vote-level logistic regression), a positive same-provider association remains (exact permutation p = 0.04; GPT-5.5 ~ +0.10 on the probability scale) - large enough to change which model appears to over-commit least once its own-provider judge is excluded. Second, LLM judges are more permissive than clinicians on a blinded 50-item subsample: all four are significantly more lenient than the stricter independent clinician (crediting appropriate uncertainty on 66-84% of items vs 52%), and three of four than the author-influenced consensus (Grok directional only; judge-vs-consensus kappa = 0.20-0.43). On the author-audited clinical-underdetermined subset the permissiveness gap widened and the point-estimate model ordering held. A closed-ended MedQA anchor confirms accuracy is high and option-order effects are within a +/-5-point equivalence region for three of four models, so the safety gap is about calibration, not knowledge. We release the harness, prompts, per-item outputs, judge panel, perturbation audit, and human-annotation protocol. |
| 2026-07-21 | [Cross-Agent Campaign Attribution: Linking Asynchronous Attacks Across LLM Agents](http://arxiv.org/abs/2607.18826v1) | SangJin Park, Myungsub Choi et al. | LLM-agent defenses are typically evaluated one session at a time. In deployment, however, attacks can be distributed across independent agents, teams, and runtimes, leaving each local guardrail with only a sparse fragment. We formalize cross-agent asynchronous campaign attribution: linking sessions from the same latent adversarial campaign without shared runtime state, test-time campaign labels, or attacker identity oracles. We introduce Asynchronous Attribution Fingerprint Vectors ($A^2FV$), a lightweight proxy-side reference protocol for scoring pairwise campaign similarity from proxy-observable tool-use, timing, and prompt residue. We also construct SCD-v1, a controlled persona-matched benchmark with benign traffic, isolated attacks, multi-session campaigns, matched non-oracle evasion, and leakage audits. On SCD-v1, $A^2FV$ achieves 0.82 pairwise AUC for campaign linking, while score-only adaptations of per-session detectors and chunked LLM judges remain near chance under the same task. The strongest fixed signal is carried by structural and stylometric residue, while timing is retained as a diagnostic channel for richer proxy traces. Crossed-style controls show that the signal is partly style-sensitive but not reducible to style alone. Static and dimension-aware non-oracle stress tests further show that pairwise separability persists under controlled evasion. These results establish cross-agent campaign attribution as a distinct evaluation layer for securing LLM agents in the wild. |
| 2026-07-21 | [ConceptCF: Concept-based Counterfactuals for the Explainability of Time Series](http://arxiv.org/abs/2607.18748v1) | Annemarie Jutte, Faizan Ahmed et al. | This paper proposes ConceptCF, a method for counterfactual generation that operates on human-interpretable concepts. In high-stakes domains such as healthcare and predictive maintenance, artificial intelligence models can increase efficiency and safety. Explainability is key to ensure these models rely on causal relationships rather than spurious correlations. Counterfactual explanations identify minimal modifications that would change a model's predictions. Existing methods for time series operate on individual points or subsequences without ensuring interpretability of the mutations. ConceptCF instead modifies meaningful concepts. As a result we can provide explanations in terms of these concepts, for example ``the model's prediction would be `Sit' instead of `Walk' if you increase the scale of the movement''. In this paper, the concepts are constructed through time series decomposition, resulting in concepts such as scale, and frequency bands. Counterfactuals are generated using a genetic algorithm that optimizes the concept mutations. Evaluation against five state-of-the-art approaches demonstrates that ConceptCF consistently achieves top-tier performance across validity, confidence, proximity, sparsity and plausibility metrics. |
| 2026-07-21 | [Correct-by-Construction Behavior Tree Synthesis from Signal Temporal Logic Specifications with Application to Robotic Missions](http://arxiv.org/abs/2607.18731v1) | Jiaheng Dong, Jingyi Huang et al. | Behavior Trees (BTs) are widely adopted for complex task execution in robotics, providing modular, reactive control but lacking formal guarantees. However, existing correct-by-construction synthesis from Linear Temporal Logic (LTL) cannot express quantitative timing constraints. This letter synthesizes correct-by-construction BTs from Signal Temporal Logic (STL) specifications. The workspace is modeled as a timed transition system and abstracted into a zone graph, and an augmented state space tracking both logical progress and timing constraints is introduced. A hierarchical fixed-point algorithm computes winning sets for an STL fragment encompassing safety, reachability, response, recurrence, and persistence, yielding BT subtrees with a runtime constraint function. Correctness guarantees are proven and complexity bounds are derived. Simulations demonstrate specification satisfaction with strictly positive robustness, and a physical quadrotor experiment with six STL specifications validates practical deployability. |
| 2026-07-21 | [When to Trust the Map: Confidence-Aware LLM Routing for Automotive CVE-to-ATM Mapping](http://arxiv.org/abs/2607.18684v1) | Heeyun Heo, Sangmin Park et al. | Public CVE descriptions report the technical conditions and impact of vulnerabilities, whereas the Auto-ISAC Automotive Threat Matrix (ATM) expresses an adversary's tactics and techniques. Because the two representations are not directly aligned, incorrect automated mappings in safety-critical environments may distort threat interpretation and mitigation prioritization, motivating a confidence-aware approach that distinguishes auto-confirmable mappings from uncertain cases. This paper reformulates automotive CVE-to-ATM mapping as a selective automation problem. The proposed framework generates candidate mappings via hierarchical in-context learning, then fuses self-consistency and LLM-based evidence verification signals into a calibrated meta-model. The resulting calibrated confidence score routes each candidate into AUTO, REVIEW, or HOLD. On the evaluation set, the proposed system substantially improved candidate-set precision at matched recall over a Flat zero-shot GPT-5.2 baseline. In the High-Confidence operating mode, the AUTO tier achieved a precision of 0.878, more than double the candidate-set base rate, and the calibrated confidence score achieved an AUROC of 0.868 in distinguishing correct from incorrect candidates. These results show that the framework can support selective automation by isolating auto-confirmable mappings from those requiring analyst review. |

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



