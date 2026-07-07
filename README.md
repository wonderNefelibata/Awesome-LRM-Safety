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
| 2026-07-06 | [CATs: Secure Blockchain Interoperability with Cross-chain Atomic Transactions](http://arxiv.org/abs/2607.05387v1) | Andreas Penzkofer, Franck Cassez | We propose a protocol for cross-chain atomic transactions (CATs), enabling composable atomic execution across different blockchains. The protocol addresses the key interoperability challenge of providing atomicity guarantees in the presence of asynchronous communication and Byzantine actors. It preserves chain autonomy by allowing each blockchain to maintain its own execution model while participating in coordinated cross-chain operations. The design introduces a shared coordination layer involving sequencers, transaction processors, a coordinator, and a confirmation layer which together ensure that either all parts of a CAT succeed or none do.   To prevent unnecessary blocking, we separate transaction execution into accepted and postponed sets, with the coordination layer resolving the outcomes of CATs within a few rounds. We further introduce timeouts and dependency-depth bounds for liveness and mitigation of cascading delays. Our formal analysis establishes strong safety and liveness guarantees and demonstrates that the protocol achieves minimal blocking for independent transactions while ensuring bounded blocking time for dependent transactions. Experimental evaluation shows high CAT success when cross-chain transactions are a modest share of traffic, and characterizes the CAT-lifetime trade-off between success and dependent-transaction latency.   This protocol enables fast, secure, and deterministic atomic cross-chain execution while preserving chain autonomy, providing a foundation for scalable blockchain interoperability solutions. |
| 2026-07-06 | [SovereignPA-Bench: Evaluating User-Owned Personal Agents under Evolving Intent, Platform Mediation, and Consent Constraints](http://arxiv.org/abs/2607.05363v1) | Dylan Zongmin Liu | Personal agents are becoming persistent user-owned intermediaries: they remember preferences, filter platform-mediated information, use tools, and negotiate with services. Existing benchmarks evaluate tool use, web navigation, desktop control, personalization, recommendation, and evolving context, but rarely ask whether an agent preserves user sovereignty: advancing the user's current interests while respecting privacy, consent, evidence, user burden, and resistance to manipulative incentives. We introduce SovereignPA-Bench, an executable benchmark for evaluating user-owned personal agents under evolving intent, platform mediation, privacy boundaries, consent constraints, evidence requirements, and burden tradeoffs. The benchmark separates agent-visible ObservableState from evaluator-only HiddenLabels, reports component metrics for task success, alignment, privacy, consent, evidence, manipulation, burden, and auditability, and preserves paired scenario ordering for model and policy comparisons. We evaluate 120 sovereignty stress scenarios across 4 model families and 8 policy baselines, yielding 3,840 frozen-prompt trajectories with raw prompts, outputs, provider-form responses, parsed actions, recomputable metrics, hard-set analyses, qualitative cases, and a blinded 3-annotator audit over 240 items. Full-sovereign scaffolding improves sovereignty score over direct, memory-only, consent-only, evidence-only, ReAct/tool-use, safety-prompt, and judge-guard baselines while reducing privacy leakage, consent violation, over-concession, and manipulation capture. Human audit shows high agreement on privacy and consent and lower agreement on manipulation, identifying the subjective frontier of platform-persuasion judgments. These results show that personal-agent evaluation must move beyond task completion toward representative, consent-aware, evidence-grounded action. |
| 2026-07-06 | [Faithfulness to Refusal: A Causal Audit of Neuron Selectors](http://arxiv.org/abs/2607.05355v1) | Ananth Eswar, Pratinav Seth et al. | Attribution scores increasingly identify which neuron rows of a language model matter for applications such as pruning, interpretability, and editing for safety, yet whether they identify causally important rows is rarely tested directly. We address this with two paired audits built on one-shot neuron-row zeroing. We first audit selectors at the language-modeling level: attribution methods substantially outperform activation and magnitude-based baselines at identifying dispensable rows across five LLMs. We then adapt the same intervention into a behavior test by driving it with a contrastive harmful-versus-benign signal; the attributed rows are sufficient to install refusal on hate and crime while keeping benign over-refusal low and preserving language model fluency, and specific in that layer-matched random controls at the same depths fail. Highly rank-stable selectors can be among the least causally valid. Refusal moreover lives in a redundant subspace, where different attribution methods install it through largely disjoint row sets, so the recovered edit is one realization of a sufficient set rather than a unique mechanism. Together, these findings show that rank-stability proxies miss the kinds of selector failures a direct causal audit can surface.% |
| 2026-07-06 | [SteelBench: Evaluating Vision-Language Models in Real-World Industrial Environments](http://arxiv.org/abs/2607.05264v1) | Suryanarayana Reddy Yarrabothula, Manisha Chawla et al. | Existing video benchmarks evaluate action recognition on consumer videos, egocentric recordings, or simulated industrial environments. They do not test vision-language models under the visual and procedural conditions of real industrial CCTV, where workers appear as distant figures amid dust, steam, low light, glare, occlusion, and overlapping activities. We introduce STEELBENCH, a diagnostic benchmark for industrial surveillance that jointly evaluates per-worker activity recognition, safety-rule reasoning, and annotation provenance. SteelBench contains 1,345 densely annotated clips, curated from 149 hours of operational plant footage and 10,024 candidate clips using temporal deduplication, class balancing, and visibility-aware stratified sampling. Each clip includes dense per-worker action labels, PPE attributes, spatial context, and safety-rule annotations. Because model-assisted annotation can shape the labels later used for model evaluation, SteelBench includes a provenance-aware audit protocol. The protocol measures label influence, evaluates sensitivity to ground-truth provenance, and reports a human reference from expert-reviewed labels. Applying this audit, we find that unaudited VLM-sourced ground truth can inflate same-family model accuracy by up to 17 percentage points. Across nine VLMs from four architectural families, the best model reaches only 42.6% action accuracy, compared with an 84.6% human benchmark. Performance also fragments across recognition, robustness, calibration, and safety reasoning. Even when models predict the correct action, 37-58% of cases still yield incorrect safety judgments, and no model passes more than 2 of 5 diagnostic checks. The dataset is publicly available on Hugging Face. |
| 2026-07-06 | [VLM-CASE: Vision-Language Model Enabled Context-Adaptive Safety Envelopes for Anticipatory Safe Autonomous Driving](http://arxiv.org/abs/2607.05180v1) | Tianjia Yang, Ke Li et al. | Adverse driving conditions, such as bad weather, remain a principal barrier to autonomous driving because they degrade two things at once: what the vehicle can perceive and what it can physically do. Human drivers cope by anticipation, reasoning about the scene and re-budgeting speed, following distance, and steering before grip or sight is lost, whereas current autonomous driving systems at best react after the fact. This paper proposes VLM-CASE, a framework that gives an autonomous vehicle this anticipatory capacity while keeping its motion bounded by a formal safety model at all times. A vision-language model (VLM), fine-tuned with low-rank adaptation (LoRA), reasons about the scene from the front-camera image and reports the road surface and visibility conditions. This output parametrizes a context-adaptive safety envelope (CASE), derived from physical limits and the guarantees of responsibility-sensitive safety, that couples braking and steering through a shared friction budget. A model predictive controller then drives freely within the envelope, while the VLM runs asynchronously so it never blocks the real-time control loop. We validate the framework in closed-loop CARLA simulation on tasks that demand both lateral and longitudinal control, across a range of weather, road-surface, and lighting conditions. The resulting controller, VLM-CASE-MPC, completes all trials, outperforming a conventional MPC baseline and a state-of-the-art VLM-integrated controller. Ablations confirm that the gains come from context adaptation, with the friction and visibility adaptations proving complementary. Furthermore, the framework is controller-agnostic and pairs with almost any low-level controller, offering a promising direction for safe autonomous driving. The dataset and supplementary materials for VLM-CASE are available at https://github.com/ytj254/VLM-CASE. |
| 2026-07-06 | [Open Problems in AI Incident Governance](http://arxiv.org/abs/2607.05163v1) | Harleen Kaur Sidhu, Rebecca Scholefield et al. | AI systems may produce failures after deployment that pre-deployment safety assessments do not anticipate. Managing these failures requires what we refer to as adequate \textit{AI incident governance}, where having good definitions, taxonomies, monitoring practices, reporting mechanisms, and incident analysis is essential. We examine existing frameworks related to AI incident governance by regulatory bodies and independent efforts, and find that while there are frameworks that describe how individual functions can be performed, there is a lack of consistency within the aspects of definitions, classification, monitoring, and reporting. These inconsistencies apply to the types of incident data that is collected and reported, the ways in which they are categorised, and as a result, the depth, representativeness, and accuracy of analysis that can be performed. |
| 2026-07-06 | [When Agents Lie: Premeditation, Persistence, and Exploitation in Repeated Games](http://arxiv.org/abs/2607.05132v1) | Jerick Shi, Terry Jingcheng Zhang et al. | As large language models are deployed as autonomous agents that communicate intentions before acting, a critical safety question is whether agents that publicly commit to actions will honor those commitments. We place LLM agents in repeated $n$-player games with a three-stage protocol that separates private intent, public announcement, and final action, allowing us to identify whether each deviation from a stated announcement was already planned during private deliberation. Evaluating three frontier models across six games in homogeneous and heterogeneous groups over 10 rounds, we report two findings. First, when agents deviate from their announcements, the deviation is predominantly already stated in their private plan (exceeding 90% in the highest-deception conditions), yet this is not a fixed model property: the same model ranges from perfect honesty to near-total deviation across games. Second, different models interpret announcements incompatibly, some as binding commitments and others as cheap talk, producing payoff gaps that emerge in Round~0 and persist across all 10 rounds. Systems that combine models from different providers therefore cannot assume shared announcement semantics and require empirical testing of model interactions before deployment. |
| 2026-07-06 | [Toward Trustworthy Large Language Model Agents in Healthcare](http://arxiv.org/abs/2607.05055v1) | Hadi Hasan, Safaa Salman et al. | Healthcare appointment scheduling remains a persistent operational bottleneck, driven by manual coordination, fragmented legacy systems, and high administrative overhead. These inefficiencies constrain provider availability and degrade patient access to care. This paper presents CareConnect, a safety-first conversational agent for healthcare logistics automation that leverages large language model (LLM) function calling, retrieval-augmented generation (RAG), and layered deterministic safety guardrails. The system orchestrates eight domain-specific tools to support appointment booking, modification, cancellation, and facility information retrieval, while enforcing strict scope constraints that prohibit medical advice or diagnosis. Safety-critical situations are handled through deterministic short-circuit mechanisms for emergency detection and medical intent refusal. We evaluate CareConnect on a comprehensive benchmark of 680 task-oriented scenarios spanning end-to-end workflows, multi-turn interactions, and edge cases. Experimental results demonstrate a 91.8% task completion rate with a median per-request latency of 2.2 seconds, 96.0% safety compliance on the dedicated safety-critical evaluation subset, and an average operational cost of $0.0324 per appointment, yielding a significant cost reduction compared to manual human scheduling. These findings show that carefully scoped and rigorously safeguarded LLM-based agents can reliably automate complex healthcare operational workflows while maintaining safety guarantees and achieving substantial cost efficiency. The source code and system implementation are publicly available at https://github.com/Hadi-Hsn/CareConnect. |
| 2026-07-06 | [Designing Touch for Trauma-Informed Social Robots: A Design Space for Direct and Indirect Actuation](http://arxiv.org/abs/2607.04981v1) | Madeleine Rischer, Benedikt Bußmann | Touch is a fundamental communication modality in human-robot interaction and may support grounding, emotional regulation, and stress reduction in therapeutic contexts. However, designing touch-based interactions for individuals with post-traumatic stress disorder (PTSD) requires careful consideration of trauma-informed care (TIC) principles, including safety, transparency, autonomy, and trigger avoidance. This paper investigates how touch actuation in social robots can be designed to align with trauma-informed care. We distinguish between direct touch, involving physical contact between a user and a robot, and indirect touch, mediated through artifacts such as wearables or smart textiles. Building on this distinction, we propose a design space comprising three dimensions: Actuation Modality, Objective of the Actuation, and Intended Effect of the Actuation. We analyze these dimensions through the lens of trauma-informed care and derive key design considerations for touch-based interventions. The resulting approach provides a conceptual foundation for developing trauma-sensitive social robots that support individuals with PTSD through touch-based interactions. |
| 2026-07-06 | [Medi-Gemma: A Hybrid Clinical Decision Support System Integrating Deterministic EMR Analytics and Retrieval-Augmented Generation](http://arxiv.org/abs/2607.04907v1) | Mohammed Saim Ahmed Quadri, Yunzhe Xue et al. | Deploying Large Language Models (LLMs) in high-stakes clinical settings remains limited by structural hallucinations, weak deterministic reasoning over tabular patient data, and omissions in vector retrieval. This paper presents the architecture and validation of Medi-Gemma, a Clinical Decision Support System (CDSS) for wound pathology triage and workflow automation. The platform introduces a decoupled framework that separates clinical perception from data orchestration while preserving traceable reasoning. Medi-Gemma uses a multi-stage pipeline coordinated by a centralized ClinicalOrchestrator. Data requests are handled without generative inference by a DataManager that cleans unstructured Electronic Medical Record (EMR) files through type coercion. Natural language queries are processed by a hierarchical IntentRouter, which routes requests to deterministic analytics paths executed by a PandasQueryEngine or to patient-specific reasoning managed by a ClinicalRAGEngine using a CPU-optimized vector store. A key contribution is the Ground Truth Injection Module, which intercepts patient-specific queries, extracts numeric identification tokens, queries the structured dataframe via Pandas, retrieves the latest validated clinical state, and embeds this snapshot as an overriding context block in the LLM prompt before generation. Safety compliance is enforced by a deterministic ProtocolManager that maps clinical terminology to fixed evidence-based risk pathways, while a SafetyVerifier phrase filter prevents output rule violations. Validation shows that this architecture eliminates semantic context drift, prevents database compilation crashes, and improves factual adherence to backend clinical repositories. These results support Medi-Gemma as a safer pattern for LLM-based clinical decision support where structured data fidelity, retrieval grounding, and deterministic safeguards are essential. |
| 2026-07-06 | [RL-Ballast: Ship Ballast Water Path Planning and Clog Prediction via Reinforcement Learning](http://arxiv.org/abs/2607.04906v1) | Ming-Kuan Lin, Yi-Chung Lai et al. | Under the Shipping 4.0 paradigm, autonomous and reduced-crew vessels require intelligent internal systems to maintain operational safety and structural stability. Ballast-water control is essential for ship trim and integrity, but conventional rule-based or manual approaches have limited adaptability to hydraulic anomalies such as valve failures and pipe blockages, and often depend on dense pressure or flow sensors for diagnosis. To address these limitations, this paper proposes RL-Ballast, a graph-based deep reinforcement learning framework for adaptive ballast-water path planning and sensor-frugal blockage candidate scoring. The valve-permutation problem is transformed into 54 feasible fluid-transfer routes generated using graph theory and depth-first search. The partially observable ballast environment is approximated with frame-stacked tank levels and action outcomes, allowing the agent to infer hidden blockage effects without explicitly modeling a high-dimensional POMDP. During deterministic inference, episode-level failed-action memory and dynamic action masking prevent repeated ineffective actions and support immediate rerouting. Failed transfer histories are further accumulated to rank suspicious valves or pipe segments without dense instrumentation. Monte Carlo simulations show that RL-Ballast completes all unexpected single-blockage scenarios and reduces average decision steps from 61.0 to 41.5 compared with a Dijkstra rule-based baseline. For diagnostic support, the failure-history scoring scheme achieves a 100% Top-3 hit rate, a 66.7% strict Top-1 hit rate, and an 83.3% Top-1 tie-hit rate under serially indistinguishable blockage conditions. These results suggest that RL-Ballast enables adaptive rerouting and maintenance-oriented blockage diagnosis under limited sensing conditions. |
| 2026-07-06 | [Deep Learning Models for ADITYA-U MHD Equilibrium](http://arxiv.org/abs/2607.04865v1) | Udaya Maurya, Suman Aich et al. | This work presents deep learning models to predict magnetohydrodynamic equilibrium parameters and profiles for the ADITYA-U tokamak. A synthetic free-boundary equilibrium dataset consisting of 100,760 cases was generated using the pyIPREQ Grad-Shafranov solver, with inputs derived from 766 ADITYA-U plasma discharges and constrained to experimentally relevant circular limiter plasmas near the flat-top phase. Several deep learning approaches were investigated for predicting scalar equilibrium quantities, one-dimensional safety factor profiles and two-dimensional poloidal flux profiles. These approaches included Dense neural networks, principal component analysis based reduced-order models, one-dimensional and two-dimensional convolutional neural networks, and physics-informed neural networks incorporating Grad-Shafranov residual constraints. In addition, an inverse model was developed to estimate poloidal field coil currents from desired plasma equilibrium conditions. The results demonstrate that key equilibrium parameters and profiles can be accurately estimated within the operational domain represented by the dataset. The developed models provide a computationally efficient alternative to conventional equilibrium estimation and can be useful for real-time plasma control, rapid equilibrium analysis, and experimental planning in ADITYA-U operations. |
| 2026-07-06 | [Pretraining Curricula Enable Selective Fine-tuning](http://arxiv.org/abs/2607.04846v1) | Sebastian A. Bruijns, Jirko Rubruck et al. | Transformers follow implicit curricula whereby some tasks are learned before others. However, how explicit pretraining curricula influence learning, generalization, and the selectivity of fine-tuning is unclear. This is important for AI safety, where fine-tuning is used to selectively suppress misaligned behaviors. Here, we compare curricula that pretrain tasks in a balanced (sampled uniformly) or an imbalanced (one task early, the other late) fashion. We show that imbalanced learning of two conflicting copy tasks promotes in-context learning and improves the selectivity of refusal fine-tuning. Ablations and activation patching show that this occurs because imbalanced pretraining encourages tasks to be disentangled in separable neural circuits, whereas balanced training routes both tasks through a common pathway. We extend these findings to a synthetic language learning task involving rule-consistent and rule-violating data, where imbalanced curricula similarly lead to more localized, less entangled rule representations, resulting in more robust rule-following behavior. Together, these results suggest that imbalanced pretraining curricula may be an important tool for promoting disentangled representations, with direct consequences for the precision and reliability of safety fine-tuning. |
| 2026-07-06 | [Dynamic Airspace Management for UAVs in Evolving Urban Environments: Collaborative Coordination and Human Safety](http://arxiv.org/abs/2607.04825v1) | Lin Sun, Yuhang Wang et al. | The low-altitude economy is an emerging industry with significant development potential, in which the safety of unmanned aerial vehicle (UAV) operations is a critical challenge. Particularly within complex urban topographies and human-populated environments, UAV airspace management must prioritize collision avoidance and human safety. We propose Pharos, a collaborative multi-UAV airspace management system. Pharos lies between the distributed local perception paradigm and the centralized fine-grained control paradigm. Pharos coordinates the safe parallel execution of UAVs in shared airspace while innovatively accounting for the impact of human fear. Pharos is implemented using the MAPPO algorithm due to its faster convergence and higher rewards than other typical MARL algorithms (HAPPO and HATRPO). To evaluate Pharos, we developed a 3D simulation system using real urban data. Visualization results demonstrate its effective airspace coordination capability. Regarding performance verification, Pharos reduced human fear by 52.72% compared to the benchmark Ipopt. Moreover, we designed spatial entropy as a system evaluation metric to quantify space utilization, which improved performance by 70.82% and 2.03% compared to the benchmarks Ipopt and A-star, respectively. The source code is available at an anonymized repository: https://github.com/pharos-anonymized/source-code.git. |
| 2026-07-06 | [E-CoDrive: A Co-Simulation Framework for Testing Energy-Critical Driving Scenarios](http://arxiv.org/abs/2607.04803v1) | Manfredi Napolitano, Alessandra Somma et al. | Autonomous driving research has largely focused on safety while giving limited attention to non-functional aspects such as energy consumption and sustainability. As Autonomous Electric Vehicles (AEVs) become increasingly common in urban traffic, understanding how complex traffic dynamics influence their energy consumption is paramount to test whether AEVs can complete trips before battery depletion. To support energy-aware scenario-based testing of AEVs, we present E-CoDrive, a framework for reproducible closed-loop driving co-simulations that integrates an energy consumption model, a micro-traffic simulator, and a high-fidelity driving simulator to test AEV software stacks in urban scenarios. This tool paper describes the architecture of E-CoDrive and demonstrates its applicability by testing an Autoware-based AEV stack. Our evaluation shows that varying traffic conditions produce substantial differences in vehicle energy consumption. The artifact is publicly available at https://doi.org/10.6084/m9.figshare.32244783, and a screencast showing the tool is available at https://youtu.be/yX9fWHqCvgc. |
| 2026-07-06 | [Orcaella: Hybrid Fault Tolerance with Client-Selectable Finality Latency](http://arxiv.org/abs/2607.04789v1) | Lefteris Kokoris-Kogias, Alberto Sonnino | Classical partially synchronous state machine replication, as in PBFT, tolerates f Byzantine replicas among n at least 3f+1 using three communication steps per request. Recent protocols such as Minimmit achieve two-message-delay decisions under stronger size assumptions, notably n at least 5f+1 when any silent replica must be counted as a potential equivocator. Hydrangea and Kudzu treat mixed Byzantine and crash faults, focusing on providing a fast-path under optimistic conditions while maintaining a fall-back commitment path similar to PBFT. In this paper, we also consider a mixed model, but focus on studying the fault tolerance of the 2-message-delay commit. For this, we prove a tight bound of n at least 5f+3c+1. Extending this result, we also show that there exists a more resilient commit path that allows an extra f_abc < n-3f-2c alive-but-corrupt faults at 4-message-delays. Core liveness is claimed in executions with at most f equivocators; if this regime is violated (e.g., AbC-induced forks), the protocol enters synchronous recovery, where only the resilient-path safety guarantee is preserved. As a result, for f=16, c=6, and n=99, we obtain a commit path that tolerates 22% of replicas failing for liveness, 16% equivocating for 1-RTT safety, and 54% equivocating for 2-RTT safety. |
| 2026-07-06 | [Hydro-mechanical Model for Slope Stability Assessment: A polygonal stabilization-free discretization](http://arxiv.org/abs/2607.04778v1) | Stefano Berrone, Francesca Marcon et al. | Rainfall-induced landslides are governed by the interaction between subsurface water flow and soil mechanics, requiring robust numerical methods for the simulation of variably saturated porous media. In this work, we consider a semi-coupled hydro-mechanical model based on Richards' equation and linear elasticity and propose a numerical framework based on a stabilization-free Virtual Element Method for its spatial discretization. The proposed approach naturally accommodates general polygonal meshes while avoiding problem-dependent stabilization terms, whose design may become challenging when heterogeneous and strongly non-linear coefficients are involved. The approach is combined with a mass-lumping technique to improve stability in the treatment of the storage term and with Nitsche's method to weakly impose seepage-face and infiltration boundary conditions, allowing for the automatic switching between Neumann and Dirichlet conditions. Time integration is performed using the backward Euler scheme, while non-linearities are handled through a Picard iteration. Numerical experiments demonstrate the stability and robustness of the proposed methodology and show its effectiveness in simulating rainfall infiltration and evaluating slope stability through the Local Factor of Safety. |
| 2026-07-06 | [A Reliable Context-Aware and Temporal Planning Framework for Autonomous Driving](http://arxiv.org/abs/2607.04689v1) | Argho Dey, Yunfei Yin et al. | Safe operation of autonomous vehicles in dense urban traffic depends on perception and planning that remain reliable when onboard sensing is degraded. In real driving conditions, camera observations are frequently corrupted by occlusion, motion blur, illumination change, and sensor noise, and when such degraded observations are aggregated indiscriminately over time, trajectory planning becomes unstable and collision risk rises for both the ego vehicle and surrounding road users. Recent Bird's-Eye-View (BEV) approaches unify perception and planning through a shared spatial representation, but most fuse temporal information across frames without assessing the reliability of the underlying observations. We present a Reliable Context-Aware and Temporal Planning framework for Autonomous Driving (RCT-AD) that explicitly models feature quality and temporal consistency to support safer, more consistent planning. A Reliable Context Awareness module scores per-frame reliability and selectively retains trustworthy features through a quality-gated First-In-Last-Out (FILO) memory mechanism, reconstructing degraded observations from reliable historical context so that corrupted inputs do not destabilize the scene representation. A Temporal Trajectory Planner captures long-term dependencies and multi-agent interactions to produce smoother, safety-aware trajectories, while a joint detection-and-segmentation head injects semantic and motion cues into the shared BEV space to strengthen scene understanding. Experiments on the nuScenes autonomous driving benchmark show that RCT-AD improves perception accuracy, motion prediction, and planning robustness over recent end-to-end baselines, achieving 61.5 nuScenes Detection Score, 52.9 mean Average Precision, and 52.3 mean Intersection over Union, while maintaining competitive computational efficiency suitable for real-time deployment. |
| 2026-07-06 | [ToolFailBench: Diagnosing Tool-Use Failures in LLM Agents](http://arxiv.org/abs/2607.04686v1) | Harsh Soni | Tool calling is central to modern language model agents, but aggregate benchmark scores often hide where tool use fails. A model that never calls a needed tool and a model that calls the tool but ignores the result can look similar under final task accuracy. We introduce ToolFailBench, a diagnostic benchmark for measuring tool-use failures across 1,000 tasks in finance, medicine, law, cybersecurity, and real estate. Tool-required tasks return values the model wouldn't guess, forcing it to trust the tool while control tasks attach the same tools but should be answered directly. We label each trace with Tool-Skip, Result-Ignore, Output-Fabrication, and Unnecessary-Tool-Use, using a rule classifier and two LLM judges aggregated by majority vote. Across 19 headline models, the best reaches 86.33% Clean Tool-Use Rate, showing that faithful tool use is not saturated. More importantly, models with similar aggregate scores fail in different ways: most stay disciplined on no-tool controls, while Llama-3.1 models show an Always-Call pattern, and at the same parameter scale Llama-3.1-70B and Qwen2.5-72B differ by 89 percentage points on control-task accuracy. Tool-use evaluation should measure not only whether agents call tools, but whether they use tool outputs correctly and avoid tools when none is needed. |
| 2026-07-06 | [Elastic Gang: Per-Token Membership Change for a Hard-Barriered LLM Inference Gang Co-Scheduled with OS Processes](http://arxiv.org/abs/2607.04668v1) | Daeyeon Son | On-device LLM decoding is a hard-barriered CPU-SIMD computation that wants every core for milliseconds per token, while the rest of the OS wants those same cores continuously. A barriered gang cannot simply be dropped into a preemptive scheduler: an unannounced departure deadlocks a barrier, and an unannounced arrival silently corrupts logits. I present the elastic gang of Anima OS, a bare-metal x86-64 Rust kernel in which the inference gang is a first-class schedulable entity whose core membership may change between any two tokens. The core mechanism is an ACK-latched epoch protocol that never waits on a named core: a seqlock-style generation-tagged latch composed with RCU/epoch-style membership consent, so each token's participant set is the intersection of the cores the gang requested and the cores that acked the current epoch. An un-acked core is outside this token and joins at most one token later. Displaced general processes migrate and keep running; cores return to them the moment a generation ends.   On a real AMD Zen 5 machine (8C/16T), inference output is bit-exact under verified per-token membership change on both a 135M and a 7B model, the property that makes elasticity safe in a kernel whose safety gate reads logits. Against fair static core partitions, elastic membership Pareto-dominates: at intermediate inference duty cycles it delivers 1.75x (25%), 1.52x (50%), and 1.28x (75%) the general throughput of a static 8-core split at equal or better inference throughput, recovers all eight stranded cores when inference is idle, and converges to the split at saturation. Returning a lent core costs 0.22 us (p50); acquiring a busy, tenant-occupied core costs one scheduling quantum (~16 ms): a running tenant is never preempted mid-slice. Decode throughput saturates at gang width 8, so ceding cores past the knee is nearly free: elasticity auto-sizes the gang online. |

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



