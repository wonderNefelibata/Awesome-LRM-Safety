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
| 2026-05-04 | [CityOS: Privacy Architecture for Urban Sensing](http://arxiv.org/abs/2605.02886v1) | Giorgio Cavicchioli, Mark Chen et al. | Cities are rapidly deploying sensing infrastructure -- cameras, environmental sensors, and connected kiosks -- that continuously observe public spaces, yet they lack a system architecture governing how applications access, aggregate, and retain this data, creating privacy risks and preventing consistent policy enforcement. We present CityOS, an operating system for urban sensing that mediates application access to sensor data through a three-tier API inspired by structured, privacy-conscious web interfaces. The tiers expand the spatial scope of data access while imposing progressively stronger privacy constraints: On-Scene supports real-time sensing with raw data confined to the local context; Single-Locality Aggregation enables differentially private longitudinal statistics at a fixed location; and Cross-Locality Aggregation supports citywide analytics via aggregation across locations, with user devices enforcing per-user privacy budgets. CityOS runs as an edge runtime that executes untrusted applications in ephemeral containers, enforcing these policies and providing transparency via broadcasts of differential privacy loss. We implement CityOS and applications across all tiers -- including pedestrian safety alerts, real-time and forecast parking availability, traffic dashboards, and subway trajectory measurement -- and show that it supports practical streetscape applications while enforcing strong privacy. |
| 2026-05-04 | [Semantic Risk-Aware Heuristic Planning for Robotic Navigation in Dynamic Environments: An LLM-Inspired Approach](http://arxiv.org/abs/2605.02862v1) | Hamza Ahmed Durrani, Rafay Suleman Durrani | The integration of Large Language Model (LLM) reasoning principles into classical robot path planning represents a rapidly emerging research direction. In this paper, we propose a Semantic Risk-Aware Heuristic (SRAH) planner that encodes LLM-inspired cost functions penalising geometrically cluttered or high-risk zones into an A$^*$ search framework, augmented with closed-loop replanning upon dynamic obstacle detection. We evaluate SRAH against two established baselines Breadth-First Search (BFS) with replanning and a Greedy heuristic without replanning across 200 randomised trials in a $15{\times}15$ grid-world with 20\% static obstacle density and stochastic dynamic obstacles. SRAH achieves a task success rate of 62.0\%, outperforming BFS (56.5\%) by 9.7\% relative improvement and Greedy (4.0\%) by a large margin. We further analyse the trade-off between planning overhead, path efficiency, and failure-recovery count, and demonstrate via an obstacle-density ablation that semantic cost shaping consistently improves navigation across environments of varying difficulty. Our results suggest that even lightweight, LLM-inspired heuristics provide measurable safety and robustness gains for autonomous robot navigation. |
| 2026-05-04 | [Standing on the Shoulders of Giants: Stabilized Knowledge Distillation for Cross--Language Code Clone Detection](http://arxiv.org/abs/2605.02860v1) | Mohamad Khajezade, Fatemeh H. Fard et al. | Cross-language code clone detection (X-CCD) is challenging because semantically equivalent programs written in different languages often share little surface similarity. Although large language models (LLMs) have shown promise for semantic clone detection, their use as black-box systems raises concerns about cost, reproducibility, privacy, and unreliable output formatting. In particular, compact open-source models often struggle to follow reasoning-oriented prompts and to produce outputs that can be consistently mapped to binary clone labels.   To address these limitations, we propose a knowledge distillation framework that transfers reasoning capabilities from DeepSeek-R1 into compact open-source student models for X-CCD. Using cross-language code pairs derived from Project CodeNet, we construct reasoning-oriented synthetic training data and fine-tune Phi3 and Qwen-Coder with LoRA adapters. We further introduce response stabilization methods, including forced conclusion prompting, a binary classification head, and a contrastive classification head, and evaluate model behavior using both predictive metrics and response rate.   Experiments on Python--Java, Rust--Java, Rust--Python, and Rust--Ruby show that knowledge distillation consistently improves the reliability of compact models and often improves predictive performance, especially under distribution shift. In addition, classification-head variants substantially reduce inference time compared to generation-based inference. Overall, our results show that reasoning-oriented distillation combined with response stabilization makes compact open-source models more practical and reliable for X-CCD detection. |
| 2026-05-04 | [FlexSQL: Flexible Exploration and Execution Make Better Text-to-SQL Agents](http://arxiv.org/abs/2605.02815v1) | Quang Hieu Pham, Yang He et al. | Text-to-SQL over large analytical databases requires navigating complex schemas, resolving ambiguous queries, and grounding decisions in actual data. Most current systems follow a fixed pipeline where schema elements are retrieved once upfront and the database is only revisited for post-hoc repair, limiting recovery from early mistakes. We present FlexSQL, a text-to-SQL agent whose core design principle is flexible database interaction: the agent can explore schema structure, inspect data values, and run verification queries at any point during reasoning. FlexSQL generates diverse execution plans to cover multiple query interpretations, implements each plan in either SQL or Python depending on the task, and uses a two-tiered repair mechanism that can backtrack from code-level errors to plan-level revisions. On Spider2-Snow, using gpt-oss-120b, FlexSQL achieves a 65.4\% score, outperforming strong open-source baselines that use stronger, larger models such as gpt-o3 and DeepSeek-R1. When integrated into a general-purpose coding agent (as skills in Claude Code), our approach yields over 10\% relative improvement on Spider2-Snow. Further analysis shows that flexible exploration and flexible execution jointly contribute to the effectiveness of our approach, highlighting flexibility as a key design principle. Our code is available at: https://github.com/StringNLPLAB/FlexSQL |
| 2026-05-04 | [The Activist's Guide to the Decentralized Social Universe: A Framework for Exploring How Decentralized Social Networks Can Support Collective Action](http://arxiv.org/abs/2605.02800v1) | Sybille Légitime, Harini Suresh | The overreaches of mainstream social media platforms have been extensively reported and studied. For activist communities, these platforms pose risks of surveillance, censorship, or erasure. Decentralized social networks (DSNs) serve as alternative online spaces that appear to prioritize values such as user privacy, free speech, and community control. However, the decentralized ecosystem is vast and complex, making it difficult for communities to understand how to best use these platforms for their organizing aims. We aim to fill this gap by proposing a conceptual framework for navigating the DSN landscape that defines core activist community needs -- minimal overhead, community building and reach, on- and off-line safety, and operational sustainability -- and links them to concrete platform affordances such as resource efficiency, interoperability, and data ownership. We apply the framework to (1) evaluate and compare the sociotechnical tradeoffs of two contemporary DSNs (Mastodon and Bluesky), (2) understand broader community configurations that emerge across different DSN infrastructures and their implications for collective action, and (3) explore how two distinct activist communities facing infrastructural and political constraints might use the framework to find platforms that align with their needs. We conclude by reflecting on the theoretical promises of DSNs and the structural conditions that shape and constrain participation across them. |
| 2026-05-04 | [Compositional Neural-Cyber-Physical System Verification in the Interactive Theorem Prover of Your Choice](http://arxiv.org/abs/2605.02790v1) | Matthew L. Daggit, Alistair Sirman et al. | Formal verification of neuro-symbolic cyber-physical systems, such as drones, medical devices and robots, is complicated. Neural components must be trained to be optimal with respect to the available data as well as the safety specifications, and then verified using specialised solvers. Symbolic models of the "cyber" and "physical" behaviour of the system must be constructed and verified in interactive theorem provers (ITPs), often requiring mature mathematical libraries to reason about the interplay of discrete and continuous dynamics, preferably obtaining infinite time-horizon guarantees. Finally, the results of the two already challenging verification tasks need to be integrated into a single proof in a coherent and consistent way, whilst preserving deployability of the resulting model.   In this paper we present a compositional methodology for constructing such proofs. The Vehicle framework provides a functional, domain-specific language for specifying, training, and verifying neural components. We extend Vehicle to allow integration with any ITP with minimal effort. First, we describe how Vehicle's standard bidirectional type checker can be reused to transpile neural specifications into an intermediate representation targeting multiple theorem provers. Second, we integrate Vehicle with Rocq, Isabelle/HOL, Agda and the industrial prover Imandra; and showcase a generic infinite time-horizon safety proof of a discrete cyber-physical system with a neural network controller in each ITP. Finally, we use the Mathematical Components libraries in Rocq to verify infinite time-horizon safety of a medical device, modelled as a continuous cyber-physical system with a neural controller. To our knowledge, this is the first result of this kind in a general purpose ITP; and a result that was only feasible thanks to the compositionality provided by Vehicle's functional interface. |
| 2026-05-04 | [A decoupled diffusion planner that adapts to changing cost limits by using cost-conditioned generation for safety and reward gradients for performance](http://arxiv.org/abs/2605.02777v1) | Rufeng Chen, Zhaofan Zhang et al. | Offline safe reinforcement learning often requires policies to adapt at deployment time to safety budgets that vary across episodes or change within a single episode. While diffusion-based planners enable flexible trajectory generation, existing guidance schemes often treat reward improvement and constraint satisfaction as competing gradient objectives, which can lead to unreliable safety compliance under cost limits. We reinterpret adaptive safe trajectory generation as sampling from a constrained trajectory distribution, where the budget restricts the trajectory region, and reward shapes preferences within that region. This perspective motivates Safe Decoupled Guidance Diffusion (SDGD), which conditions classifier-free guidance on the cost limit to bias sampling toward trajectories satisfying the specified limit, while using reward-gradient guidance to refine trajectories for higher return. Because direct reward guidance can increase return while also steering samples toward trajectories with higher cumulative cost, we introduce Feasible Trajectory Relabeling (FTR) to reshape reward targets and discourage such directions. We further provide a first-order sampling-time analysis showing that FTR suppresses reward-induced cost drift under a prefix-restorative alignment condition. Extensive evaluations on the DSRL benchmark show that SDGD achieves the strongest safety compliance among baselines, satisfying the constraint on 94.7% of tasks (36/38), while obtaining the highest reward among safe methods on 21 tasks. |
| 2026-05-04 | [The Streaming Reservoir Convergence Theorem: A Prospect-Theoretic Framework for Multi-Provider Adaptive Streaming](http://arxiv.org/abs/2605.02761v1) | Justice Owusu Agyemang, Jerry John Kponyo et al. | We present the Streaming Reservoir Convergence Theorem (SRCT), a novel mathematical framework for multi-provider adaptive bitrate streaming that addresses three fundamental structural weaknesses in current systems: linear provider probing, reactive failover, and cold standby transitions. SRCT models stream acquisition as a concurrent reservoir filling problem$-$probing all $N$ providers simultaneously rather than in batches$-$and maintains $k$ pre-verified, pre-fetched standby streams alongside the active stream to enable sub-second failover with zero user-visible disruption.   We prove four principal results: (1) a harmonic lower bound on reservoir safety showing that $k$ independent streams provide $H_k / \barλ$ expected uptime where $H_k$ is the $k$-th harmonic number; (2) a concurrent acquisition speedup $S(N,b) = (N/b) \cdot (1-F^b)/(1-F^N)$ over batched probing, yielding $3$-$5\times$ practical improvement; (3) monotonic non-decreasing quality under lazy-refill with convergence to the Pareto-optimal frontier; and (4) a prospect-weighted switching rule$-$using Kahneman-Tversky value functions with $α=β=0.88$, $λ=2.25$ $-$ that provably eliminates thrashing between similar-quality streams via a no-thrash bound on the expected switch count.   We implement SRCT across two production streaming pipelines: a primary movie/TV system serving 12+ HLS providers with $k=3$ reservoir slots, and a live sports system with multi-format DASH/HLS failover. Empirical verification via Monte Carlo simulation (5000 trials) confirms all four theorems across 22 independent checks. The reservoir of $k=3$ streams achieves $9.15\times$ mean time to depletion versus a single stream, and concurrent probing of 12 providers at 40% failure rate yields a $4.27\times$ speedup over the current batched-by-3 default. |
| 2026-05-04 | [DynoSLAM: Dynamic SLAM with Generative Graph Neural Networks for Real-World Social Navigation](http://arxiv.org/abs/2605.02759v1) | Danil Tokhchukov, Veronika Morozova et al. | Traditional Simultaneous Localization and Mapping (SLAM) algorithms rely heavily on the static environment assumption, which severely limits their applicability in real-world spaces populated by moving entities, such as pedestrians. In this work, we propose DynoSLAM, a tightly-coupled Dynamic GraphSLAM architecture that integrates socially-aware Graph Neural Networks (GNNs) directly into the factor graph optimization. Unlike conventional approaches that use rigid constant-velocity heuristics or deterministic single-agent neural priors, our framework formulates pedestrian motion forecasting as a stochastic World Model. By utilizing Monte Carlo rollouts from a trained GNN, we capture the multimodal epistemic uncertainty of human interactions and embed it into the SLAM graph via a dynamic Mahalanobis distance factor. We demonstrate through extensive simulated experiments that this stochastic formulation not only maintains highly accurate retrospective tracking but also prevents the optimization failures caused by the deterministic "argmax problem". Ultimately, extracting the empirical mean and covariance matrices of future pedestrian states provides a mathematically rigorous, probabilistic safety envelope for downstream local planners, enabling anticipatory and collision-free robot navigation in densely crowded environments. |
| 2026-05-04 | [Parking Assistance for Trailer-Truck Transport Vehicles Using Sensor Fusion and Motion Planning](http://arxiv.org/abs/2605.02716v1) | George Alenchery, Thomas Jeske et al. | Autonomous driving technology has rapidly evolved over the past decade, offering significant improvements in transportation efficiency, safety, and cost reduction. While much of the progress has focused on highway driving and obstacle avoidance, low-speed maneuvers such as parking remain among the most difficult challenges for autonomous systems. This challenge is especially pronounced in trailer-truck transport vehicles due to their articulated motion and environmental constraints. This paper presents a proposed framework for autonomous truck parking that integrates perception, motion planning, control systems, and infrastructure awareness. By combining sensor fusion, Hybrid A* path planning, nonlinear model predictive control (NMPC), and data-driven parking systems, this work highlights the importance of system-level coordination for reliable and scalable autonomous parking solutions. As a proof-of-concept implementation, we adapted an open-source A* path planning simulation to incorporate a tractor-trailer kinematic model, demonstrating articulated vehicle path planning within a command-line simulation environment, with jackknife prevention identified as an area requiring further development. |
| 2026-05-04 | [An Empirical Study of Agent Skills for Healthcare: Practice, Gaps, and Governance](http://arxiv.org/abs/2605.02709v1) | Gelei Xu, Ningzhi Tang et al. | Healthcare automation is shaped by local procedures and organizational constraints, so agent capabilities rarely transfer unchanged across settings. Agent skills, self-contained directories that package reusable procedures for AI agents, are emerging as a procedural layer for adapting healthcare agents across diverse healthcare settings. We present the first empirical analysis of healthcare agent skills, drawing on 557 healthcare-related skills filtered from 58,159 public skills on ClawHub and annotated along ten dimensions covering function, deployment context, autonomy, and safety. We find that public healthcare skills emphasize patient-facing workflow automation and monitoring rather than the diagnostic and treatment-oriented tasks foregrounded in healthcare-agent research; coverage of the healthcare lifecycle and specialized clinical inputs remains uneven; and general technical risk does not reliably capture clinical risk. These findings position healthcare skills as a procedural layer not yet addressed by current benchmarks and risk frameworks. |
| 2026-05-04 | [Executor-Side Progressive Risk-Gated Actuation for Agentic AI in Wireless Supervisory Control](http://arxiv.org/abs/2605.02697v1) | Zhenyu Liu, Yi Ma et al. | Agentic artificial intelligence (AI) shows promise for automating O-RAN wireless supervisory control, but translated intents still require an executor-side decision before live network actuation. Existing control flows lack explicit semantics for whether an intent should commit, gate for evidence, or reject under stale telemetry, concurrent policies, deadline and bandwidth limits, and rollback constraints. We propose Progressive Risk-Gated Actuation (PRGA), an executor-side contract for risk-gated wireless intent execution. PRGA structures each intent into executable local triage (C0), on-demand coordination evidence (C1), and post-hoc provenance support (C2), with C2 kept off the online safety path. A deterministic two-stage policy checks expiry, freshness, rollback-handle validity, local conflict, blocking preconditions, and planner-executor risk divergence from C0, then retrieves C1 only for gated intents when deadline and bandwidth budgets allow; evidence-mandatory gates reject when required C1 is unavailable. On two 3GPP-parameterized energy-saving and slice-SLA benchmarks, PRGA reduces time-to-first-safe-action by 23.3-27.4% and per-commit control-plane bytes by 52.7-54.2% against a decision-identical eager full-evidence cost-overlay comparator, thereby isolating retrieval-cost accounting; remains non-inferior within a pre-declared 0.5 percentage-point unsafe-action margin against an invariant-respecting static-threshold comparator; and rejects 100% of injected over-threshold stale inputs in the stale-state fault campaign. On these benchmarks, PRGA improves supervisory responsiveness and control-plane efficiency within the evaluated unsafe-action boundary. |
| 2026-05-04 | [AcademiClaw: When Students Set Challenges for AI Agents](http://arxiv.org/abs/2605.02661v1) | Junjie Yu, Pengrui Lu et al. | Benchmarks within the OpenClaw ecosystem have thus far evaluated exclusively assistant-level tasks, leaving the academic-level capabilities of OpenClaw largely unexamined. We introduce AcademiClaw, a bilingual benchmark of 80 complex, long-horizon tasks sourced directly from university students' real academic workflows -- homework, research projects, competitions, and personal projects -- that they found current AI agents unable to solve effectively. Curated from 230 student-submitted candidates through rigorous expert review, the final task set spans 25+ professional domains, ranging from olympiad-level mathematics and linguistics problems to GPU-intensive reinforcement learning and full-stack system debugging, with 16 tasks requiring CUDA GPU execution. Each task executes in an isolated Docker sandbox and is scored on task completion by multi-dimensional rubrics combining six complementary techniques, with an independent five-category safety audit providing additional behavioral analysis. Experiments on six frontier models show that even the best achieves only a 55\% pass rate. Further analysis uncovers sharp capability boundaries across task domains, divergent behavioral strategies among models, and a disconnect between token consumption and output quality, providing fine-grained diagnostic signals beyond what aggregate metrics reveal. We hope that AcademiClaw and its open-sourced data and code can serve as a useful resource for the OpenClaw community, driving progress toward agents that are more capable and versatile across the full breadth of real-world academic demands. All data and code are available at https://github.com/GAIR-NLP/AcademiClaw. |
| 2026-05-04 | [ContextualJailbreak: Evolutionary Red-Teaming via Simulated Conversational Priming](http://arxiv.org/abs/2605.02647v1) | Mario Rodríguez Béjar, Francisco J. Cortés-Delgado et al. | Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety alignment and elicit harmful responses. A growing body of work shows that contextual priming, where earlier turns covertly bias later replies, constitutes a powerful attack surface, with hand-crafted multi-turn scaffolds consistently outperforming single-turn manipulations on capable models.   However, automated optimization-based red-teaming has remained largely limited to the single-turn setting, iterating over static prompts and lacking the ability to reason about which forms of conversational priming induce compliance. While recent multi-turn, search-based approaches have begun to bridge this gap, the mutator design space underlying effective primed dialogues remains largely unexplored.   We present ContextualJailbreak, a black-box red-teaming strategy that performs evolutionary search over a simulated multi-turn primed dialogue. The strategy leverages a graded 0-5 harm score from a two-level judge as an in-loop signal, enabling partially harmful responses to guide the search process rather than being discarded.   Search is driven by five semantically defined mutation operators: roleplay, scenario, expand, troubleshooting, and mechanistic, of which the last two are novel contributions of this work.   Across 50 representative HarmBench behaviors, ContextualJailbreak achieves an ASR of 100% on gpt-oss:20B, 100% on qwen3-8B, 100% on llama3.1:70B, and 90% on gpt-oss:120B, outperforming four single- and multi-turn baselines by 31-96 percentage points on average.   The 40 maximally harmful attacks discovered against gpt-oss:120B transfer without adaptation to closed frontier models, achieving 90.0% on gpt-4o-mini, 70.0% on gpt-5, and 70.0% on gemini-3-flash, but only 17.5% on claude-opus-4-7 and 15.0% on claude-sonnet-4-6, revealing a pronounced provider-level asymmetry in alignment robustness. |
| 2026-05-04 | [Mamoda2.5: Enhancing Unified Multimodal Model with DiT-MoE](http://arxiv.org/abs/2605.02641v1) | Yangming Shi, Shixiang Zhu et al. | We present Mamoda2.5, a unified AR-Diffusion framework that seamlessly integrates multimodal understanding and generation within a single architecture. To efficiently enhance the model's generation capability, we equip the Diffusion Transformer backbone with a fine-grained Mixture-of-Experts (MoE) design (128 experts, Top-8 routing), yielding a 25B-parameter model that activates only 3B parameters, significantly reducing training costs while scaling up the model capacity. Mamoda2.5 achieves top-tier generation performance on VBench 2.0 and sets a new record in video editing quality, surpassing evaluated open-source models and matching the performance of current top-tier proprietary models, including the Kling O1 on OpenVE-Bench. Furthermore, we introduce a joint few-step distillation and reinforcement learning framework that compresses the 30-step editing model into a 4-step model and greatly accelerates model inference. Compared to open-source baselines, Mamoda2.5 achieves up to $95.9\times$ faster video editing inference. In real-world applications, Mamoda2.5 has been successfully deployed for content moderation and creative restoration tasks in advertising scenarios, achieving a 98% success rate in internal advertising video editing scenario. |
| 2026-05-04 | [Mitigation of Boundary Sampling Artifacts in Phase Space Generation for Electron FLASH Radiotherapy](http://arxiv.org/abs/2605.02605v1) | Rafael Carballeira, Rongxiao Zhang et al. | Applicator-specific phase space (PHSP) files recorded at the aperture exit reduce Monte Carlo dose calculation time by 30-50% for electron FLASH radiotherapy. However, positioning PHSP scoring planes coincident with the applicator-air interface introduces boundary sampling artifacts. This study characterizes these artifacts in Geant4-based simulations and demonstrates their mitigation. PHSP files were generated using GAMOS 6.2.0 for a 9 MeV Mobetron UHDR model across twelve clinical aperture configurations (2.5-10 cm diameter). Three scoring plane positions were evaluated relative to the physical aperture exit: coincident with the interface, 0.1 mm downstream, and 1 mm downstream. Scoring at the exact interface produced proximal R50 shifts of up to 2.2 mm and Distance-to-Agreement (DTA) values of 4-6 mm, exceeding clinical acceptance criteria. Artifact severity scaled inversely with aperture diameter, with the smallest configurations most severely affected. A 0.1 mm offset partially restored the primary electron energy spectrum but failed to recover the bremsstrahlung tail. A 1 mm offset fully resolved all artifacts, achieving mean DTA values within 2.0 mm, equivalent to or better than linac-exit references. These artifacts arise from the degenerate behavior of the fUseSafety step-limitation algorithm when safety equals zero at exact material boundaries, producing incomplete secondary electron equilibration and suppressed bremsstrahlung production. Angular distribution analysis revealed a near-forward particle pileup in the 0 mm PHSP and a deficit of large-angle secondaries recovered by the 1 mm offset. A 1 mm downstream offset fully mitigates these artifacts while introducing negligible perturbation to primary beam characteristics. This requirement applies to any Geant4-based framework (including TOPAS and GATE) scoring PHSP files at material exit surfaces. |
| 2026-05-04 | [Hyp2Former: Hierarchy-Aware Hyperbolic Embeddings for Open-Set Panoptic Segmentation](http://arxiv.org/abs/2605.02580v1) | Yao Lu, Rohit Mohan et al. | Recognizing unknown objects is crucial for safety-critical applications such as autonomous driving and robotics. Open-Set Panoptic Segmentation (OPS) aims to segment known thing and stuff classes while identifying valid unknown objects as separate instances. Prior OPS approaches largely treat known categories as a flat label set, ignoring the semantic hierarchy that provides valuable structural priors for distinguishing unknown objects from in-distribution classes. In this work, we propose Hyp2Former, an end-to-end framework for OPS that does not require explicit modeling of unknowns during training, and instead learns hierarchical semantic similarities continuously in hyperbolic space. By explicitly encoding hierarchical relationships among known categories, the model learns a structured embedding space that captures multiple levels of semantic abstraction. As a result, unknown objects that cannot be confidently classified as known categories still remain in close proximity to higher-level concepts (e.g., an unknown animal remains closer to "animal" or "object" than to unrelated concepts such as "electronics" or "stuff") and can therefore be reliably detected, even if their fine-grained category was not represented during training. Empirical evaluations across multiple public datasets such as MS COCO, Cityscapes, and Lost&Found demonstrate that Hyp2Former outperforms existing methods on OPS, achieving the best balance between unknown object discovery and in-distribution robustness. |
| 2026-05-04 | [Improving Model Safety by Targeted Error Correction](http://arxiv.org/abs/2605.02544v1) | Abolfazl Mohammadi-Seif, Ricardo Baeza-Yates | The widespread adoption of machine learning in critical applications demands techniques to mitigate high-consequence errors. Our method utilizes a dual-classifier GBDT pipeline to distinguish routine human-like errors from high-risk non-human misclassifications. Evaluated across three domains, animal breed classification, skin lesion diagnosis (ISIC 2018), and prostate histopathology (SICAPv2), our framework demonstrates robust safety improvements. To address real-world deployment concerns, our results confirm the pipeline introduces negligible inference latency (1.60% overhead for the animal dataset, 1.84% for ISIC, and 1.70% for SICAPv2) while outperforming traditional Maximum Class Probability (MCP) baselines in correction precision. Our conservative correction strategy successfully reduced dangerous non-human errors by 34.1% in ISIC and 12.57% in SICAPv2, improving super-class diagnostic safety to 90.41% and 92.13% respectively. This proves that safety-critical reliability can be substantially enhanced post-hoc without expensive model retraining.   keywords: Error Analysis, Post-hoc Correction, Trustworthy AI. |
| 2026-05-04 | [Orchestrating Spatial Semantics via a Zone-Graph Paradigm for Intricate Indoor Scene Generation](http://arxiv.org/abs/2605.02537v1) | Meisheng Zhang, Shizhao Sun et al. | Autonomous 3D indoor scene synthesis breaks down in non-convex rooms with tightly coupled spatial constraints. Data-driven generators lack topological priors for long-horizon planning, while iterative agents fragment semantics and become geometrically brittle. We present ZoneMaestro, a unified framework that shifts the paradigm from object-centric synthesis to Zone-Graph Orchestration. By internalizing a novel zone-based logic, ZoneMaestro translates high-level semantic intent into functional zones and topological constraints, enabling robust adaptation to diverse architectural forms. To support this, we construct Zone-Scene-10K, a large-scale dataset enriched with explicit Zone-Graph annotations. We further introduce an Alternating Alignment Strategy that cycles between reasoning internalization and Zone-Aware Group Relative Policy Optimization (Z-GRPO), effectively reconciling the tension between semantic richness and geometric validity without relying on external physics engines. To rigorously evaluate spatial intelligence beyond convex primitives, we formally define the task of Intricate Spatial Orchestration and release SCALE, a stress-test benchmark for irregular indoor scenarios with complex, dense spatial relations. Extensive experiments demonstrate that ZoneMaestro resolves the density-safety dichotomy, significantly outperforming state-of-the-art baselines in both structural coherence and intent adherence. |
| 2026-05-04 | [Set-Based Training of Neural Barrier Certificates for Safety Verification of Dynamical Systems](http://arxiv.org/abs/2605.02526v1) | Miriam Kranzlmüller, Lukas Koller et al. | Barrier certificates are scalar functions over the state space of dynamical systems that separate all unsafe states from all reachable states. The existence of a barrier certificate formally verifies the safety of the dynamical system. Recent approaches synthesize barrier certificates by iteratively training a neural network. In each iteration, the candidate is formally verified - if successful, the barrier certificate is found. Instead, we propose a set-based training approach that tightly integrates verification into training via a set-based loss function that soundly encodes all barrier certificate properties. A loss of zero formally proves the validity of the barrier certificate, collapsing the iterative training and verification into a single training procedure. Our experiments demonstrate that our set-based training approach scales well with the system dimension and naturally handles complex nonlinear dynamics. |

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



