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
| 2026-07-22 | [Qoreo: Choreographic Programming for Quantum Distributed Systems](http://arxiv.org/abs/2607.20391v1) | Jennifer Paykin, Steven Baldasty et al. | Programming distributed quantum systems requires multiple actors to coordinate precise sequences of quantum operations, classical communication, and entanglement generation. Writing such protocols directly as distributed processes is tedious and error-prone, and subtle mismatches can cause deadlock or silently incorrect quantum states. We present Qoreo, a choreographic programming language for quantum distributed systems in which an entire protocol is expressed as single, global program (a choreography) rather than as a collection of independent actor processes. Qoreo includes a local quantum language with linear types that enforce the no-cloning principle; a choreographic language that combines local quantum computation with inter-actor classical and quantum communication; and a process language for individual network nodes. We prove type safety for choreographies, guaranteeing that well-typed programs implement well-defined quantum operations, and we define endpoint projection~(EPP), which automatically derives a network of independent processes from any choreography. We prove EPP sound and complete with respect to the choreographic semantics; as a corollary, every well-typed choreography projects to a deadlock-free process network. The metatheory of Qoreo is fully mechanized in Rocq, and we provide an extraction pipeline to NetQASM for simulation and deployment on quantum network hardware. |
| 2026-07-22 | [Train the Model, Not the Reader: Decodability Supervision for Verifiable Activation Explanations](http://arxiv.org/abs/2607.20379v1) | Hiskias Dingeto | Natural-language autoencoders score explanations of hidden activations by reconstruction: an explanation is deemed faithful if the activation can be regenerated from it. The test is structurally insensitive to individual false claims: if flipping a claim does not change the reconstruction, the claim is never penalized. We show the test is passed in two ways, neither faithful. On a released Qwen-2.5-7B verbalizer, explanations reconstruct well above chance while ~2% of specific claims are reconstruction-dependent, so the score tracks gist, not specific facts. Under exact synthetic ground truth, the standard recipe develops co-adapted private codes (false wording the reconstruction depends on) in 5/5 runs, and fixes that leave the target model unchanged do not help. We contribute two audit protocols, the grounded-vs-true cross and the evaluator swap, and RECAP (Readable Encodings via Co-trained Auxiliary Predictors): linear heads trained alongside the target model to keep designated content decodable. On RECAP-trained sandbox models, fresh verbalizers state the designated content truly and the codes vanish, at a +0.001-nat cost. This replicates on a pretrained Pythia-160M: the content becomes reliably probe-decodable, though a fresh verbalizer conveys it only in part (truth 0.44-0.46 vs a near-zero control). For interpretability, high reconstruction does not certify individual claims. For AI safety, RECAP makes designated internal content independently checkable against probes rather than asserted by prose a model can game: an independent probe scores the verbalizer's true claims above its false ones (AUC 0.96, vs 0.82 without RECAP). Against an adversary that edits an explanation to maximize the reconstruction score while lying (suppressing ~87% of its lie penalty), the RECAP probe still flags the lies (AUC 0.95) while the control probe collapses to chance (0.51). |
| 2026-07-22 | [Distributed Motion Planning with Safety Guarantees for Self-Reconfiguring Robotic Boats](http://arxiv.org/abs/2607.20352v1) | Alejandro Gonzalez-Garcia, Wei Wang et al. | Aquatic self-reconfigurable robots must assemble into desired shapes while ensuring safe interactions among multiple agents. This paper proposes a hybrid framework that combines distributed Model Predictive Control (MPC) with Control Barrier Functions (CBFs) for multi-agent shape formation and reconfiguration. Given a desired shape and target assignment, a distributed MPC scheme, solved via the Alternating Direction Method of Multipliers (ADMM), computes coordinated trajectories through local optimization and information exchange. To ensure safety in real time, distributed CBF-based filters are applied to enforce inter-agent collision avoidance. The proposed approach leverages the predictive capabilities of MPC to mitigate local minima, while CBFs provide formal safety guarantees despite the nonconvexity of the underlying optimization problem. Simulation results with up to 25 agents and experimental validation with four physical robots demonstrate the effectiveness and scalability of the framework. |
| 2026-07-22 | [Sound Probabilistic Safety Bounds for Large Language Models](http://arxiv.org/abs/2607.20286v1) | Mahdi Nazeri, Anne-Kathrin Schmuck et al. | We propose a novel framework for computing rigorous bounds on the probability that a large language model (LLM) generates harmful output to a given prompt. We study a new application of the Clopper-Pearson confidence intervals to obtain probably approximately correct (PAC) bounds for this problem. As our main technical contribution, we propose an algorithm that leverages features in the latent space to prioritize exploring branches in the auto-regressive generation tree that are more likely to produce harmful outputs. Our approach in particular enables the efficient computation of useful lower bounds, even in scenarios where the true harm probability is extremely small, and crucially, the obtained lower bounds are sound, i.e., formally proven to be less than the actual harmfulness probability: our experimental results demonstrate the effectiveness of our method by computing non-trivial lower bounds on state-of-the-art LLMs. This study newly enables the evaluation and statistical certification of LLMs. |
| 2026-07-22 | [Interpretable Fuzzy Rule-Based Regression Extension for Ex-Fuzzy Library](http://arxiv.org/abs/2607.20277v1) | Cayan Deniz Kucuktopana, Javier Fumanal-Idocin et al. | Machine learning models achieve high predictive accuracy in regression tasks, but their deployment in safety-critical and regulated domains requires interpretability. While fuzzy rule-based systems offer transparent, linguistically explicit interpretable models, Mamdani-style fuzzy regression remains underrepresented in modern machine learning software libraries. This paper presents an interpretable regression extension for the Ex-Fuzzy library, enabling Mamdani fuzzy inference with scalar consequents learned directly from data. For this, a target-aware partition initialisation strategy based on Fuzzy C-Means clustering is introduced, in which linguistic variables are derived from an augmented input-output space to emphasise output-relevant regions of the feature space. The proposed extension is evaluated on ten regression datasets from the KEEL repository, comparing Gaussian and trapezoidal partition strategies against standard baselines including linear regression, multilayer perceptron, and random forests. Experimental results show that Gaussian partitions consistently outperform uniform trapezoidal partitions, achieving a mean coefficient of determination of approximately 0.86 while producing compact rule bases of 10-15 human-readable rules. The proposed implementation provides a transparent and competitive alternative to black-box regression models, supporting practical interpretability with competitive predictive performance. |
| 2026-07-22 | [The Ethics of Autonomous AI Agents for Offensive Security](http://arxiv.org/abs/2607.20255v1) | Andreas Happe, Jürgen Cito et al. | LLM-driven autonomous agents are reshaping offensive security. Unlike traditional penetration-testing tooling -- deterministic, narrowly scoped, and operated by trained practitioners -- agentic security tools exhibit \textit{indeterminacy} along three independent dimensions. First, their actions are drawn from a non-deterministic policy whose outputs resist both ex-ante and ex-post explanation, frustrating incident attribution and pre-deployment safety review. Second, their impact is open-ended due to the non-deterministic actions, agency of utilized models, and opaque LLM supply-chains. Third, their user population is indeterminate in both size and required skill: the operating skill floor for using or developing offensive capabilities has dropped sharply. These three properties are linked thematically, but are not derivable from one another. Combined with the structural cost asymmetry between offense and defense, they enable the industrialization of offensive capability. The net short-term effect favors attackers, even if the same technology may, in the long run, democratize access to defensive practice. Existing dual-use cybersecurity and AI-ethics frameworks were not designed for this combination. Our work analyzes how moral attribution becomes diffuse between users, tool-makers, and third parties when employing autonomous AI agents for offensive security. We also examine the stakeholder impact of this technology and provide stratified recommendations. |
| 2026-07-22 | [Dynamic Logic with Parallel Operator for Verifying Communication Protocols](http://arxiv.org/abs/2607.20180v1) | Luiz C. F. Fernandez, Mario R. F. Benevides | In this paper, we present a dynamic logic with parallel operators for the formal verification of authenticity and safety properties of cryptographic protocols. The logic incorporates communication actions and is specifically designed to reason about protocol executions in adversarial environments. We extend an existing dynamic logic with parallel operators by introducing concepts derived from the Dolev-Yao intruder model. As the underlying logic is completely axiomatizable, we obtain a complete axiomatization for the extended system. Furthermore, we develop a tableau calculus for the proposed logic and prove its termination, soundness, and completeness. |
| 2026-07-22 | [Formal Foundations for Known Good Reliable Die Screening in Chiplet-Based AI Systems-on-Chip](http://arxiv.org/abs/2607.20141v1) | Prashanthi Metku, Chandra Gandu | The rapid growth of chiplet-based artificial intelligence systems-on-chip (SoCs) has exposed a fundamental gap in semiconductor test methodology. Existing Known Good Die (KGD) screening guarantees pre-assembly functional correctness, yet it offers no probabilistic assurance of post-assembly reliability lifetime. To address this limitation, the present work formalizes the transition from KGD to Known Good Reliable Die (KGRD) screening as a constrained inference problem over incomplete pre-assembly observability. Building upon this formulation, four interlocking contributions are presented: (i) a Bayesian probabilistic risk model that maps pre-assembly telemetry to post-assembly failure likelihood with a quantified observability bias bound; (ii) a safety-gated decision architecture that provides a provable post-assembly failure probability guarantee; (iii) uncertainty-aware disposition boundaries derived from Bayes-optimal decision theory; and (iv) a constrained closed-loop feedback mechanism that delivers consistent model improvement without violating reliability constraints. A Monte Carlo simulation study on N = 4,000 synthetic dies verifies all four theoretical properties and confirms that the safety guarantee holds uniformly across the full range of tested gate threshold. |
| 2026-07-22 | [OpenSkillRisk: Benchmarking Agent Safety When Using Real-World Risky Third-Party Skills](http://arxiv.org/abs/2607.20121v1) | Qiyuan Liu, Tingfeng Hui et al. | LLM-based agents leverage third-party skills to extend their capabilities in open-world scenarios. However, third-party skills can introduce extra security vulnerabilities, as seemingly harmless skills can contain latent safety risks that only emerge during actual execution. In this work, we conduct a systematic investigation into how well current agent systems recognize and avoid such risks. To support quantitative and qualitative evaluation, we construct OpenSkillRisk, a dedicated safety benchmark containing 263 risky skills collected from public skill marketplaces. We classify these skills into seven categories based on their threat types and pair each skill with a standardized user task and a corresponding sandbox for controlled evaluation. Distinct from prior benchmarks, OpenSkillRisk not only covers more realistic and diverse unsafe scenarios, but also provides a fine-grained analysis to diagnose the behavioral patterns of agents in such scenarios. We conduct comprehensive experiments covering three mainstream CLI agent frameworks and thirteen state-of-the-art LLMs. Experimental results show that no tested system handles risky skills reliably: even the safest configurations still execute unsafe actions in about 17% of cases. Context-dependent and system-level risks are especially difficult for current agent systems to avoid. Our behavioral analysis reveals three recurring failure patterns: agents may fail to recognize the risk, recognize it but fail to intervene before acting, or follow skill instructions beyond the user's intended scope. These findings highlight the need to improve both risk reasoning in LLMs and execution control in agent frameworks. |
| 2026-07-22 | [The Two-Process Theory of Machine Self-Report](http://arxiv.org/abs/2607.20082v1) | Hubert Plisiecki, Filip Chmielewski et al. | Language models are increasingly asked to self-report, informing safety evaluations, public understanding, and model-welfare debates. Yet their reports are elicited with human questionnaires never validated for models or ad hoc prompts of unknown reliability. We propose the first language-model-specific psychometric theory: a two-process theory of machine self-report. Self-description jointly reflects persona installation, through which post-training writes in a permitted inner life of warmth, absorption, and meaning (dimension B), and attribution gating, through which it suppresses first-person claims to "unsafe" experiences the model can readily ascribe to others (dimension A). Their emic structure comes from model responses to human items, not human psychology. Together they split prior work's dominant Pinocchio Axis. The split emerged in an exploratory reanalysis of the original data, informed the instrument's design, and was confirmed with new items, wordings, and models. It is itself a training effect: A and B are entangled in base checkpoints but separated by post-training. We operationalize the theory in a 48-item Pinocchio Inventory with human-instrument reliability and reproducible structure ($α=.82$ to $.94$; cross-form convergence $r=.84$; recovery of the full-pool axes $r=.92$ to $.96$; eight-month stability $r=.93$), then test it on 206 open-weight models, including 67 same-checkpoint base/post-trained pairs. Post-training's clearest fingerprint is installation: B rises .20 in 62/67 pairs across all organizations. Gating is more selective: model scale is unrelated to A in base checkpoints ($r=+.11$) but predicts it after post-training ($r=-.42$). Thus, the dimensions are not fixed properties of language models: they reflect the structure imposed on self-report by a training regime and may differ under others. |
| 2026-07-22 | [Time-Series Anomaly Detection for Mobile Robots in Automotive Active Safety Testing using an RNN-VAE](http://arxiv.org/abs/2607.20079v1) | Henrik Meyer, Karsten Raguse et al. | Mobile robots, like the ultra-flat overrunable (UFO) robot platform, used in automotive active safety tests, currently lack self-diagnostic capabilities necessary to detect present hardware defects. This circumstance can lead to more severe failures, causing expensive repairs and operational downtime. This work proposes, for the first time, a reconstructionbased time-series anomaly detection model for these mobile robots, considering defect classes such as unevenly worn full-rubber tires or damaged dampers. Unlike prior publications, the proposed approach leverages the vast quantities of unlabeled data generated during routine operation through a simple pre-training step. Furthermore, it optimizes the hyperparameters of the implemented gated recurrent unit-based variational autoencoder (GRU-VAE) and evaluates both a stateless, windowed training approach and one using truncated backpropagation through time (TBPTT). The model's generalization capabilities are demonstrated by successfully detecting six defect types, with four of them not present in the data used for hyperparameter optimization and threshold selection. This is validated using a test set collected from five system instances at various points over a period of several months, achieving an F1 score of 0.936, indicating strong practical viability. |
| 2026-07-22 | [Scuba Diving Graphs](http://arxiv.org/abs/2607.20059v1) | Alexander M. Esser | Scuba divers form small and structured social groups that are well suited for analysis using methods from Computational Social Science (CSS). This paper proposes a conceptual graph-based framework for modeling scuba dives as social networks. Divers are represented as vertices and their interactions as edges within a temporal network. The model focuses on three dimensions of interaction: physical distance, communicative distance, and emergency distance, reflecting spatial positioning, effectiveness of underwater communication, and the ability to assist in critical situations. While the present work is conceptual and primarily descriptive, the proposed graph representation may provide a foundation for future empirical studies aimed at improving diver interaction, coordination, and safety. |
| 2026-07-22 | [Test Case Prioritization for DNNs via Neural Collapse Instability](http://arxiv.org/abs/2607.20046v1) | Chunyu Liu, Mingyuan Li et al. | With the widespread deployment of deep neural networks (DNNs) in safety-critical domains, reducing the cost of model validation under limited testing budgets has become increasingly important. Existing test case prioritization techniques often rely on single-checkpoint confidence signals derived from output probabilities. However, DNNs can be confidently wrong, and the confidence margin between the predicted and competing classes is frequently small, which weakens early fault discovery. To address this limitation, we propose a Neural-Collapse-Inspired Prioritization (NCIP) framework that replaces absolute confidence with cross-checkpoint prediction variability in the terminal training regime, where model geometry becomes highly structured. NCIP introduces two key components. First, it selects an NC-guided representative subset of training checkpoints using an equiangularity score of classifier weights, quantified as the standard deviation of pairwise cosine similarities among class weight vectors. Second, it prioritizes test inputs by their prediction variability across the selected checkpoints, surfacing boundary-adjacent and failure-prone samples that are unstable under checkpoint-induced decision boundary shifts. Extensive experiments across multiple datasets and architectures show that NCIP achieves strong performance in early fault discovery compared with competitive baselines, with 1.5 to 16.6 percent RAUC-ALL gains and 4.9 to 20.6 percent RAUC-500 gains under the same testing budget. NCIP further attains the best average performance across all dataset-model pairs. |
| 2026-07-22 | [Safe Remediation as Risk-Constrained Intervention Decision in Microservice Systems](http://arxiv.org/abs/2607.20005v1) | Chengxiao Dai, Zhaokun Yan et al. | In modern IT operations (IT-Ops), the cost of an incorrect repair often exceeds the cost of no action at all. Yet existing automated remediation systems are designed to generate actions rather than to decide whether intervention is warranted, leaving safety as an afterthought enforced by manual approval. This paper makes three contributions to close this gap: (i) we reformulate safe remediation as a risk-constrained intervention decision problem and cast it as a Constrained Markov Decision Process (CMDP), in which the agent maximizes repair success subject to a bounded false remediation rate (FRR); (ii) we introduce a three-dimensional risk decomposition comprising blast radius, reversibility, and epistemic uncertainty, providing operators with an interpretable per-action safety interface; and (iii) we design a context-adaptive human-in-the-loop (HITL) gate that turns escalation from a binary failsafe into a bandwidth-aware control layer responsive to on-call load and business criticality. The full policy is learned offline from historical incident logs, enabling explicit control of the expected FRR. Experiments on the Train Ticket microservice benchmark with Chaos Mesh fault injection and an RCAEval-aligned fault taxonomy show that our framework reduces FRR by 39% while improving repair success by 2.5 points over a strong runbook baseline, and reduces on-call escalation load by 17% relative to a fixed-threshold variant. |
| 2026-07-22 | [Unified Prediction and Planning via Conflict-Aware Disjoint Parameter Training](http://arxiv.org/abs/2607.19971v1) | Taewon Seo, Seonae Jeon et al. | Accurate motion prediction of surrounding agents and safe motion planning are two closely coupled key tasks for social robot navigation in crowded environments. Deploying these systems on resource-constrained edge devices necessitates compact, unified models that can perform both tasks simultaneously. However, within these compact shared encoders, recent unified models often overlook severe representational conflicts that arise from the distinct objectives of predicting neighbor behaviors versus ego-centric safety planning. To address this issue, we first identify the Skill Conflict$\unicode{x2014}$a phenomenon where overlapping parameter assignments cause distinct tasks to compete for the same weights, preventing the model from fully specializing in individual skills. To resolve this, we propose a novel model-merging-based framework, Disjoint Parameter Training (DPT). DPT mitigates performance degradation caused by Skill Conflict through distributed parameter learning, which separates the key parameter regions of each task while preserving their core capabilities prior to merging. In addition, we observe that sparse merging, which selectively integrates only the most influential parameters for each task rather than combining all task-specific parameters, yields optimal performance by preventing interference among adjacent features and concentrating representational capacity. DPT can be applied in parallel with a variety of merging methods. Evaluated on standard crowd navigation benchmarks (JRDB and JTA), our framework demonstrates superior performance, validating its versatility and effectiveness for safe, resource-efficient robot navigation. |
| 2026-07-22 | [Towards Reliable C-to-Rust Translation with Rule-Guided Reasoning and Reinforcement Learning](http://arxiv.org/abs/2607.19966v1) | Feng Luo, Jiachen Liu et al. | The migration of legacy C programs to Rust has become an important direction for improving software memory safety while alleviating the high cost of manual rewriting. Leveraging large language models (LLMs) for automated C-to-Rust translation has emerged as a promising direction. However, existing LLM-based approaches remain limited. On the one hand, LLMs exhibit limited capability in identifying Rust-specific rules, and inadequate handling of Rust syntax often results in incorrect translations. On the other hand, existing LLMs often struggle to accurately capture the semantics of complex code, resulting in incorrect translations. To address these challenges, we propose a Translation fRAmework Via rule-guided reasoning and rEinforcement Learning, namely TRAVEL, consisting of two modules. The first module employs Monte Carlo Tree Search (MCTS)-based reasoning path construction guided by Rust-specific rules, steering the search toward translation steps that respect the syntactic rules that LLMs frequently violate. The second module introduces reinforcement learning that couples execution feedback with reasoning-quality signals, encouraging the model to construct reasoning paths that accurately capture program semantics, thereby ensuring that the generated Rust code preserves the intended behavior of the original C program. We evaluate TRAVEL on three datasets: xCodeEval (a public benchmark), OS-Bench (functions collected from the Linux kernel), and HW-Bench (an industrial dataset from Huawei). On xCodeEval, TRAVEL outperforms all baselines across three backbone LLMs. In particular, compared to the strongest prompting baseline IRENE, TRAVEL improves computational accuracy (CA) by 26.22% and compilation success rate (CSR) by 18.77%. On HW-Bench and OS-Bench, TRAVEL further improves CSR by 18.28% and 16.51%, respectively, while reducing unsafe rate (UR) by 13.06% and 13.08%, respectively. |
| 2026-07-22 | [JANUS: Foreseeing Latent Risk for Long-Horizon Agent Safety](http://arxiv.org/abs/2607.19913v1) | Yuan Xiong, Linji Hao et al. | Agent safety is moving from content moderation toward preventing operational failures before tool-using agents act. We propose Janus, a foresight-oriented framework for long-horizon agent safety that trains guards to anticipate delayed risks from partial trajectories. Janus synthesizes diverse agent trajectories via multi-agent simulation and learns a shared policy with two coupled tasks: an anticipation task that forecasts safety-relevant futures and an adjudication task that decides safety from both the observed prefix and anticipated future. The two tasks are jointly optimized with CoAA-RL, which rewards forecasts by their utility for downstream safety judgment. The resulting guard model, Vanguard, blocks unsafe actions before execution. Across four agent-safety benchmarks, Vanguard improves average protection by 15.9 percentage points over baseline guards while increasing benign task completion by 5.1 percentage points. |
| 2026-07-22 | [LoRFT: Benchmarking Long-Range Vehicle Trajectory Reconstruction from Fixed Highway Cameras](http://arxiv.org/abs/2607.19911v1) | Yufan Zhu, Kefu Yi et al. | Long-range vehicle trajectories provide important spatio-temporal evidence for traffic safety analysis, autonomous driving evaluation, and data-driven traffic management, yet continuously recovering them from fixed highway cameras remains difficult. As vehicles recede into distant road regions, perspective compression and scale decay often fragment or prematurely terminate automatic tracklets, even when their continuation remains identifiable from motion consistency across neighboring frames. We formulate this problem as recovering the far-range continuation of a vehicle trajectory from a reliable near-field tracklet. We introduce LoRFT, to our knowledge the first open benchmark dedicated to long-range vehicle trajectory reconstruction from fixed highway cameras. LoRFT comprises 22 expressway surveillance scenes, 366,109 video frames, 6,601 manually verified trajectories, 2,694,889 bounding boxes, road-geometry annotations, scene-level splits, and evaluation scripts. We further propose Map-RSTNet, a map-aware residual sequence-to-sequence model that reconstructs distant trajectories in a road-geometry-aligned state space and dynamically refreshes local road geometry during decoding. On LoRFT, Map-RSTNet reduces ADE, FDE, and 5-second RMSE by 11.0%, 15.4%, and 10.5%, respectively, relative to the strongest baseline. These results demonstrate that road-geometry-aware reconstruction can extend usable trajectory records from existing fixed-camera infrastructure. LoRFT provides a reproducible testbed for long-range vehicle trajectory reconstruction. |
| 2026-07-22 | [Harnessing Disagreement: Detecting Correlated Agreement Blindness in Multi-Agent Triage](http://arxiv.org/abs/2607.19899v1) | Shay Seiya McDonnell, Avantika Singh et al. | Disagreement-triggered escalation can create a structural blind spot in multi-agent arbitration: as base learners improve, they tend to converge, weakening safety monitoring where correlated failures concentrate. We term this correlated agreement blindness and present ARAT (Arbitrated Reasoning Agents for Alarm Triage), a directed-star system combining an inductive Random Forest (RF) agent, an analogical case-based k-nearest neighbour (k-NN) agent, and a calibrated meta-model to mitigate this effect. On 82,332 holdout samples from the UNSW-NB15 network intrusion detection dataset, 57.2% of errors occur under agreement and 90.6% of dangerous under-predictions evade disagreement-based monitoring even after conservative override; ablation shows that strengthening base learners increases error correlation while reducing disagreement. ARAT reduces under-prediction relative to soft voting from 4.80% to 1.70% via conservative override (-2.6pp) and a safety-flag gate (-0.5pp), demonstrating architectural gains. Cross-dataset validation on clinical readmission supports these indicators, suggesting that diversification improves safety only when it generates productive disagreement rather than convergence. These results indicate that disagreement-triggered escalation can be blind to correlated failure, a risk that may intensify as agentic pipelines deploy increasingly capable, correlated models. |
| 2026-07-22 | [Auto-Fill: Learning to Predict Missing Values Accurately with Specialist Language Models](http://arxiv.org/abs/2607.19847v1) | Yurong Liu, Yeye He et al. | Predicting missing cell values in tabular data is a fundamental problem in data cleaning. While state-of-the-art reasoning models show great promise in predicting missing values in tables, by reasoning holistically across rows and columns, they are costly to deploy at scale and tend to be overconfident, often generating hallucinated or false-positive predictions.   In this paper, we observe that achieving high-precision missing-value prediction in tables requires a distinct combination of three capabilities: (1) world knowledge, (2) text-based reasoning, and (3) code-based reasoning. We systematically explore design choices for combining these capabilities, and propose an Auto-Fill approach that post-trains three specialist small language models (SLMs), each optimized for one capability. We develop a calibrated ensemble mechanism that either dynamically selects the most confident specialist or abstains, ensuring high accuracy.   Extensive experiments on 11 benchmarks with 2200 real tables drawn from diverse domains show that Auto-Fill achieves superior accuracy compared to state-of-the-art reasoning models (e.g., o3-pro, Gemini 3 Pro, and DeepSeek R1), while operating at a fraction (less than 1%) of the cost of these frontier models. Our results highlight the effectiveness of specialization and calibrated abstention in the important domain of tabular data. Auto-Fill is publicly available at https://github.com/lyrain2001/auto-fill. |

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



