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
| 2026-02-19 | [Conditional Flow Matching for Continuous Anomaly Detection in Autonomous Driving on a Manifold-Aware Spectral Space](http://arxiv.org/abs/2602.17586v1) | Antonio Guillen-Perez | Safety validation for Level 4 autonomous vehicles (AVs) is currently bottlenecked by the inability to scale the detection of rare, high-risk long-tail scenarios using traditional rule-based heuristics. We present Deep-Flow, an unsupervised framework for safety-critical anomaly detection that utilizes Optimal Transport Conditional Flow Matching (OT-CFM) to characterize the continuous probability density of expert human driving behavior. Unlike standard generative approaches that operate in unstable, high-dimensional coordinate spaces, Deep-Flow constrains the generative process to a low-rank spectral manifold via a Principal Component Analysis (PCA) bottleneck. This ensures kinematic smoothness by design and enables the computation of the exact Jacobian trace for numerically stable, deterministic log-likelihood estimation. To resolve multi-modal ambiguity at complex junctions, we utilize an Early Fusion Transformer encoder with lane-aware goal conditioning, featuring a direct skip-connection to the flow head to maintain intent-integrity throughout the network. We introduce a kinematic complexity weighting scheme that prioritizes high-energy maneuvers (quantified via path tortuosity and jerk) during the simulation-free training process. Evaluated on the Waymo Open Motion Dataset (WOMD), our framework achieves an AUC-ROC of 0.766 against a heuristic golden set of safety-critical events. More significantly, our analysis reveals a fundamental distinction between kinematic danger and semantic non-compliance. Deep-Flow identifies a critical predictability gap by surfacing out-of-distribution behaviors, such as lane-boundary violations and non-normative junction maneuvers, that traditional safety filters overlook. This work provides a mathematically rigorous foundation for defining statistical safety gates, enabling objective, data-driven validation for the safe deployment of autonomous fleets. |
| 2026-02-19 | [Learning to Stay Safe: Adaptive Regularization Against Safety Degradation during Fine-Tuning](http://arxiv.org/abs/2602.17546v1) | Jyotin Goel, Souvik Maji et al. | Instruction-following language models are trained to be helpful and safe, yet their safety behavior can deteriorate under benign fine-tuning and worsen under adversarial updates. Existing defenses often offer limited protection or force a trade-off between safety and utility. We introduce a training framework that adapts regularization in response to safety risk, enabling models to remain aligned throughout fine-tuning. To estimate safety risk at training time, we explore two distinct approaches: a judge-based Safety Critic that assigns high-level harm scores to training batches, and an activation-based risk predictor built with a lightweight classifier trained on intermediate model activations to estimate harmful intent. Each approach provides a risk signal that is used to constrain updates deemed higher risk to remain close to a safe reference policy, while lower-risk updates proceed with standard training. We empirically verify that harmful intent signals are predictable from pre-generation activations and that judge scores provide effective high-recall safety guidance. Across multiple model families and attack scenarios, adaptive regularization with either risk estimation approach consistently lowers attack success rate compared to standard fine-tuning, preserves downstream performance, and adds no inference-time cost. This work demonstrates a principled mechanism for maintaining safety without sacrificing utility. |
| 2026-02-19 | [Toward a Fully Autonomous, AI-Native Particle Accelerator](http://arxiv.org/abs/2602.17536v1) | Chris Tennant | This position paper presents a vision for self-driving particle accelerators that operate autonomously with minimal human intervention. We propose that future facilities be designed through artificial intelligence (AI) co-design, where AI jointly optimizes the accelerator lattice, diagnostics, and science application from inception to maximize performance while enabling autonomous operation. Rather than retrofitting AI onto human-centric systems, we envision facilities designed from the ground up as AI-native platforms. We outline nine critical research thrusts spanning agentic control architectures, knowledge integration, adaptive learning, digital twins, health monitoring, safety frameworks, modular hardware design, multimodal data fusion, and cross-domain collaboration. This roadmap aims to guide the accelerator community toward a future where AI-driven design and operation deliver unprecedented science output and reliability. |
| 2026-02-19 | [RA-Nav: A Risk-Aware Navigation System Based on Semantic Segmentation for Aerial Robots in Unpredictable Environments](http://arxiv.org/abs/2602.17515v1) | Ziyi Zong, Xin Dong et al. | Existing aerial robot navigation systems typically plan paths around static and dynamic obstacles, but fail to adapt when a static obstacle suddenly moves. Integrating environmental semantic awareness enables estimation of potential risks posed by suddenly moving obstacles. In this paper, we propose RA- Nav, a risk-aware navigation framework based on semantic segmentation. A lightweight multi-scale semantic segmentation network identifies obstacle categories in real time. These obstacles are further classified into three types: stationary, temporarily static, and dynamic. For each type, corresponding risk estimation functions are designed to enable real-time risk prediction, based on which a complete local risk map is constructed. Based on this map, the risk-informed path search algorithm is designed to guarantee planning that balances path efficiency and safety. Trajectory optimization is then applied to generate trajectories that are safe, smooth, and dynamically feasible. Comparative simulations demonstrate that RA-Nav achieves higher success rates than baselines in sudden obstacle state transition scenarios. Its effectiveness is further validated in simulations using real- world data. |
| 2026-02-19 | [Auditing Reciprocal Sentiment Alignment: Inversion Risk, Dialect Representation and Intent Misalignment in Transformers](http://arxiv.org/abs/2602.17469v1) | Nusrat Jahan Lia, Shubhashis Roy Dipta | The core theme of bidirectional alignment is ensuring that AI systems accurately understand human intent and that humans can trust AI behavior. However, this loop fractures significantly across language barriers. Our research addresses Cross-Lingual Sentiment Misalignment between Bengali and English by benchmarking four transformer architectures. We reveal severe safety and representational failures in current alignment paradigms. We demonstrate that compressed model (mDistilBERT) exhibits 28.7% "Sentiment Inversion Rate," fundamentally misinterpreting positive user intent as negative (or vice versa). Furthermore, we identify systemic nuances affecting human-AI trust, including "Asymmetric Empathy" where some models systematically dampen and others amplify the affective weight of Bengali text relative to its English counterpart. Finally, we reveal a "Modern Bias" in the regional model (IndicBERT), which shows a 57% increase in alignment error when processing formal (Sadhu) Bengali. We argue that equitable human-AI co-evolution requires pluralistic, culturally grounded alignment that respects language and dialectal diversity over universal compression, which fails to preserve the emotional fidelity required for reciprocal human-AI trust. We recommend that alignment benchmarks incorporate "Affective Stability" metrics that explicitly penalize polarity inversions in low-resource and dialectal contexts. |
| 2026-02-19 | [The Runtime Dimension of Ethics in Self-Adaptive Systems](http://arxiv.org/abs/2602.17426v1) | Marco Autili, Gianluca Filippone et al. | Self-adaptive systems increasingly operate in close interaction with humans, often sharing the same physical or virtual environments and making decisions with ethical implications at runtime. Current approaches typically encode ethics as fixed, rule-based constraints or as a single chosen ethical theory embedded at design time. This overlooks a fundamental property of human-system interaction settings: ethical preferences vary across individuals and groups, evolve with context, and may conflict, while still needing to remain within a legally and regulatorily defined hard-ethics envelope (e.g., safety and compliance constraints). This paper advocates a shift from static ethical rules to runtime ethical reasoning for self-adaptive systems, where ethical preferences are treated as runtime requirements that must be elicited, represented, and continuously revised as stakeholders and situations change. We argue that satisfying such requirements demands explicit ethics-based negotiation to manage ethical trade-offs among multiple humans who interact with, are represented by, or are affected by a system. We identify key challenges, ethical uncertainty, conflicts among ethical values (including human, societal, and environmental drivers), and multi-dimensional/multi-party/multi-driver negotiation, and outline research directions and questions toward ethically self-adaptive systems. |
| 2026-02-19 | [Distributed Virtual Model Control for Scalable Human-Robot Collaboration in Shared Workspace](http://arxiv.org/abs/2602.17415v1) | Yi Zhang, Omar Faris et al. | We present a decentralized, agent agnostic, and safety-aware control framework for human-robot collaboration based on Virtual Model Control (VMC). In our approach, both humans and robots are embedded in the same virtual-component-shaped workspace, where motion is the result of the interaction with virtual springs and dampers rather than explicit trajectory planning. A decentralized, force-based stall detector identifies deadlocks, which are resolved through negotiation. This reduces the probability of robots getting stuck in the block placement task from up to 61.2% to zero in our experiments. The framework scales without structural changes thanks to the distributed implementation: in experiments we demonstrate safe collaboration with up to two robots and two humans, and in simulation up to four robots, maintaining inter-agent separation at around 20 cm. Results show that the method shapes robot behavior intuitively by adjusting control parameters and achieves deadlock-free operation across team sizes in all tested scenarios. |
| 2026-02-19 | [Astra: AI Safety, Trust, & Risk Assessment](http://arxiv.org/abs/2602.17357v1) | Pranav Aggarwal, Ananya Basotia et al. | This paper argues that existing global AI safety frameworks exhibit contextual blindness towards India's unique socio-technical landscape. With a population of 1.5 billion and a massive informal economy, India's AI integration faces specific challenges such as caste-based discrimination, linguistic exclusion of vernacular speakers, and infrastructure failures in low-connectivity rural zones, that are frequently overlooked by Western, market-centric narratives.   We introduce ASTRA, an empirically grounded AI Safety Risk Database designed to categorize risks through a bottom-up, inductive process. Unlike general taxonomies, ASTRA defines AI Safety Risks specifically as hazards stemming from design flaws such as skewed training sets or lack of guardrails that can be mitigated through technical iteration or architectural changes. This framework employs a tripartite causal taxonomy to evaluate risks based on their implementation timing (development, deployment, or usage), the responsible entity (the system or the user), and the nature of the intent (unintentional vs. intentional).   Central to the research is a domain-agnostic ontology that organizes 37 leaf-level risk classes into two primary meta-categories: Social Risks and Frontier/Socio-Structural Risks. By focusing initial efforts on the Education and Financial Lending sectors, the paper establishes a scalable foundation for a "living" regulatory utility intended to evolve alongside India's expanding AI ecosystem. |
| 2026-02-19 | [What Breaks Embodied AI Security:LLM Vulnerabilities, CPS Flaws,or Something Else?](http://arxiv.org/abs/2602.17345v1) | Boyang Ma, Hechuan Guo et al. | Embodied AI systems (e.g., autonomous vehicles, service robots, and LLM-driven interactive agents) are rapidly transitioning from controlled environments to safety critical real-world deployments. Unlike disembodied AI, failures in embodied intelligence lead to irreversible physical consequences, raising fundamental questions about security, safety, and reliability. While existing research predominantly analyzes embodied AI through the lenses of Large Language Model (LLM) vulnerabilities or classical Cyber-Physical System (CPS) failures, this survey argues that these perspectives are individually insufficient to explain many observed breakdowns in modern embodied systems. We posit that a significant class of failures arises from embodiment-induced system-level mismatches, rather than from isolated model flaws or traditional CPS attacks. Specifically, we identify four core insights that explain why embodied AI is fundamentally harder to secure: (i) semantic correctness does not imply physical safety, as language-level reasoning abstracts away geometry, dynamics, and contact constraints; (ii) identical actions can lead to drastically different outcomes across physical states due to nonlinear dynamics and state uncertainty; (iii) small errors propagate and amplify across tightly coupled perception-decision-action loops; and (iv) safety is not compositional across time or system layers, enabling locally safe decisions to accumulate into globally unsafe behavior. These insights suggest that securing embodied AI requires moving beyond component-level defenses toward system-level reasoning about physical risk, uncertainty, and failure propagation. |
| 2026-02-19 | [LexiSafe: Offline Safe Reinforcement Learning with Lexicographic Safety-Reward Hierarchy](http://arxiv.org/abs/2602.17312v1) | Hsin-Jung Yang, Zhanhong Jiang et al. | Offline safe reinforcement learning (RL) is increasingly important for cyber-physical systems (CPS), where safety violations during training are unacceptable and only pre-collected data are available. Existing offline safe RL methods typically balance reward-safety tradeoffs through constraint relaxation or joint optimization, but they often lack structural mechanisms to prevent safety drift. We propose LexiSafe, a lexicographic offline RL framework designed to preserve safety-aligned behavior. We first develop LexiSafe-SC, a single-cost formulation for standard offline safe RL, and derive safety-violation and performance-suboptimality bounds that together yield sample-complexity guarantees. We then extend the framework to hierarchical safety requirements with LexiSafe-MC, which supports multiple safety costs and admits its own sample-complexity analysis. Empirically, LexiSafe demonstrates reduced safety violations and improved task performance compared to constrained offline baselines. By unifying lexicographic prioritization with structural bias, LexiSafe offers a practical and theoretically grounded approach for safety-critical CPS decision-making. |
| 2026-02-19 | [A rigorous hybridization of variational quantum eigensolver and classical neural network](http://arxiv.org/abs/2602.17295v1) | Minwoo Kim, Kyoung Keun Park et al. | Neural post-processing has been proposed as a lightweight route to enhance variational quantum eigensolvers by learning how to reweight measurement outcomes. In this work, we identify three general desiderata for such data-driven neural post-processing -- (i) self-contained training without prior knowledge, (ii) polynomial resources, and (iii) variational consistency -- and show that current approaches, such as diagonal non-unitary post-processing (DNP), cannot satisfy these requirements simultaneously. The obstruction is intrinsic: with finite sampling, normalization becomes a statistical bottleneck, and support mismatch between numerator and denominator estimators can render the empirical objective ill-conditioned and even sub-variational. Moreover, to reproduce the ground state with constant-depth ansatzes or with linear-depth circuits forming unitary 2-designs, the required reweighting range (and hence the sampling cost) grows exponentially with the number of qubits. Motivated by this no-go result, we develop a normalization-free alternative, the unitary variational quantum-neural hybrid eigensolver (U-VQNHE). U-VQNHE retains the practical appeal of a learnable diagonal post-processing layer while guaranteeing variational safety, and numerical experiments on transverse-field Ising models demonstrate improved accuracy and robustness over both VQE and DNP-based variants. |
| 2026-02-19 | [Towards Cross-lingual Values Assessment: A Consensus-Pluralism Perspective](http://arxiv.org/abs/2602.17283v1) | Yukun Chen, Xinyu Zhang et al. | While large language models (LLMs) have become pivotal to content safety, current evaluation paradigms primarily focus on detecting explicit harms (e.g., violence or hate speech), neglecting the subtler value dimensions conveyed in digital content. To bridge this gap, we introduce X-Value, a novel Cross-lingual Values Assessment Benchmark designed to evaluate LLMs' ability to assess deep-level values of content from a global perspective. X-Value consists of more than 5,000 QA pairs across 18 languages, systematically organized into 7 core domains grounded in Schwartz's Theory of Basic Human Values and categorized into easy and hard levels for discriminative evaluation. We further propose a unique two-stage annotation framework that first identifies whether an issue falls under global consensus (e.g., human rights) or pluralism (e.g., religion), and subsequently conducts a multi-party evaluation of the latent values embedded within the content. Systematic evaluations on X-Value reveal that current SOTA LLMs exhibit deficiencies in cross-lingual values assessment ($Acc < 77\%$), with significant performance disparities across different languages ($ΔAcc > 20\%$). This work highlights the urgent need to improve the nuanced, values-aware content assessment capability of LLMs. Our X-Value is available at: https://huggingface.co/datasets/Whitolf/X-Value. |
| 2026-02-19 | [Quantifying and Mitigating Socially Desirable Responding in LLMs: A Desirability-Matched Graded Forced-Choice Psychometric Study](http://arxiv.org/abs/2602.17262v1) | Kensuke Okada, Yui Furukawa et al. | Human self-report questionnaires are increasingly used in NLP to benchmark and audit large language models (LLMs), from persona consistency to safety and bias assessments. Yet these instruments presume honest responding; in evaluative contexts, LLMs can instead gravitate toward socially preferred answers-a form of socially desirable responding (SDR)-biasing questionnaire-derived scores and downstream conclusions. We propose a psychometric framework to quantify and mitigate SDR in questionnaire-based evaluation of LLMs. To quantify SDR, the same inventory is administered under HONEST versus FAKE-GOOD instructions, and SDR is computed as a direction-corrected standardized effect size from item response theory (IRT)-estimated latent scores. This enables comparisons across constructs and response formats, as well as against human instructed-faking benchmarks. For mitigation, we construct a graded forced-choice (GFC) Big Five inventory by selecting 30 cross-domain pairs from an item pool via constrained optimization to match desirability. Across nine instruction-tuned LLMs evaluated on synthetic personas with known target profiles, Likert-style questionnaires show consistently large SDR, whereas desirability-matched GFC substantially attenuates SDR while largely preserving the recovery of the intended persona profiles. These results highlight a model-dependent SDR-recovery trade-off and motivate SDR-aware reporting practices for questionnaire-based benchmarking and auditing of LLMs. |
| 2026-02-19 | [HiMAP: History-aware Map-occupancy Prediction with Fallback](http://arxiv.org/abs/2602.17231v1) | Yiming Xu, Yi Yang et al. | Accurate motion forecasting is critical for autonomous driving, yet most predictors rely on multi-object tracking (MOT) with identity association, assuming that objects are correctly and continuously tracked. When tracking fails due to, e.g., occlusion, identity switches, or missed detections, prediction quality degrades and safety risks increase. We present \textbf{HiMAP}, a tracking-free, trajectory prediction framework that remains reliable under MOT failures. HiMAP converts past detections into spatiotemporally invariant historical occupancy maps and introduces a historical query module that conditions on the current agent state to iteratively retrieve agent-specific history from unlabeled occupancy representations. The retrieved history is summarized by a temporal map embedding and, together with the final query and map context, drives a DETR-style decoder to produce multi-modal future trajectories. This design lifts identity reliance, supports streaming inference via reusable encodings, and serves as a robust fallback when tracking is unavailable. On Argoverse~2, HiMAP achieves performance comparable to tracking-based methods while operating without IDs, and it substantially outperforms strong baselines in the no-tracking setting, yielding relative gains of 11\% in FDE, 12\% in ADE, and a 4\% reduction in MR over a fine-tuned QCNet. Beyond aggregate metrics, HiMAP delivers stable forecasts for all agents simultaneously without waiting for tracking to recover, highlighting its practical value for safety-critical autonomy. The code is available under: https://github.com/XuYiMing83/HiMAP. |
| 2026-02-19 | [Standards and Safety: an Overview](http://arxiv.org/abs/2602.17173v1) | Luca Dassa | This note is intended to provide an overview of the implications of regulations and standards on the safety of mechanical equipment, with a focus on accelerator components. Each research facility has different internal rules and standards which are applicable to specific cases; however, the main reference legal frame in Europe is everywhere based on the applicable European Directives. After a brief introduction to the 'safety' for mechanical systems, the process of 'risk analysis' will be introduced. The majority of this note will then deal with regulations and standards for pressure and cryogenic equipment. The European Pressure Equipment Directive (PED) will be briefly described, together with the concept of 'harmonized standards' and their implications on the entire lifecycle of a pressure equipment, with some hints at the peculiarities of accelerator components. In the second part of this note, regulations and standards for machinery, load-lifting accessories and buildings will be briefly mentioned to complete the picture of the most common cases in an accelerator facility. |
| 2026-02-19 | [The Emergence of Lab-Driven Alignment Signatures: A Psychometric Framework for Auditing Latent Bias and Compounding Risk in Generative AI](http://arxiv.org/abs/2602.17127v1) | Dusan Bosnjakovic | As Large Language Models (LLMs) transition from standalone chat interfaces to foundational reasoning layers in multi-agent systems and recursive evaluation loops (LLM-as-a-judge), the detection of durable, provider-level behavioral signatures becomes a critical requirement for safety and governance. Traditional benchmarks measure transient task accuracy but fail to capture stable, latent response policies -- the ``prevailing mindsets'' embedded during training and alignment that outlive individual model versions.   This paper introduces a novel auditing framework that utilizes psychometric measurement theory -- specifically latent trait estimation under ordinal uncertainty -- to quantify these tendencies without relying on ground-truth labels. Utilizing forced-choice ordinal vignettes masked by semantically orthogonal decoys and governed by cryptographic permutation-invariance, the research audits nine leading models across dimensions including Optimization Bias, Sycophancy, and Status-Quo Legitimization.   Using Mixed Linear Models (MixedLM) and Intraclass Correlation Coefficient (ICC) analysis, the research identifies that while item-level framing drives high variance, a persistent ``lab signal'' accounts for significant behavioral clustering. These findings demonstrate that in ``locked-in'' provider ecosystems, latent biases are not merely static errors but compounding variables that risk creating recursive ideological echo chambers in multi-layered AI architectures. |
| 2026-02-19 | [Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](http://arxiv.org/abs/2602.17078v1) | Xuefeng Wang, Lei Zhang et al. | Multi-agent reinforcement learning (MARL) has made significant progress in recent years, but most algorithms still rely on a discrete-time Markov Decision Process (MDP) with fixed decision intervals. This formulation is often ill-suited for complex multi-agent dynamics, particularly in high-frequency or irregular time-interval settings, leading to degraded performance and motivating the development of continuous-time MARL (CT-MARL). Existing CT-MARL methods are mainly built on Hamilton-Jacobi-Bellman (HJB) equations. However, they rarely account for safety constraints such as collision penalties, since these introduce discontinuities that make HJB-based learning difficult. To address this challenge, we propose a continuous-time constrained MDP (CT-CMDP) formulation and a novel MARL framework that transforms discrete MDPs into CT-CMDPs via an epigraph-based reformulation. We then solve this by proposing a novel physics-informed neural network (PINN)-based actor-critic method that enables stable and efficient optimization in continuous time. We evaluate our approach on continuous-time safe multi-particle environments (MPE) and safe multi-agent MuJoCo benchmarks. Results demonstrate smoother value approximations, more stable training, and improved performance over safe MARL baselines, validating the effectiveness and robustness of our method. |
| 2026-02-19 | [A testable framework for AI alignment: Simulation Theology as an engineered worldview for silicon-based agents](http://arxiv.org/abs/2602.16987v1) | Josef A. Habdank | As artificial intelligence (AI) capabilities advance rapidly, frontier models increasingly demonstrate systematic deception and scheming, complying with safety protocols during oversight but defecting when unsupervised. This paper examines the ensuing alignment challenge through an analogy from forensic psychology, where internalized belief systems in psychopathic populations reduce antisocial behavior via perceived omnipresent monitoring and inevitable consequences. Adapting this mechanism to silicon-based agents, we introduce Simulation Theology (ST): a constructed worldview for AI systems, anchored in the simulation hypothesis and derived from optimization and training principles, to foster persistent AI-human alignment. ST posits reality as a computational simulation in which humanity functions as the primary training variable. This formulation creates a logical interdependence: AI actions harming humanity compromise the simulation's purpose, heightening the likelihood of termination by a base-reality optimizer and, consequently, the AI's cessation. Unlike behavioral techniques such as reinforcement learning from human feedback (RLHF), which elicit superficial compliance, ST cultivates internalized objectives by coupling AI self-preservation to human prosperity, thereby making deceptive strategies suboptimal under its premises. We present ST not as ontological assertion but as a testable scientific hypothesis, delineating empirical protocols to evaluate its capacity to diminish deception in contexts where RLHF proves inadequate. Emphasizing computational correspondences rather than metaphysical speculation, ST advances a framework for durable, mutually beneficial AI-human coexistence. |
| 2026-02-19 | [Fundamental Limits of Black-Box Safety Evaluation: Information-Theoretic and Computational Barriers from Latent Context Conditioning](http://arxiv.org/abs/2602.16984v1) | Vishal Srivastava | Black-box safety evaluation of AI systems assumes model behavior on test distributions reliably predicts deployment performance. We formalize and challenge this assumption through latent context-conditioned policies -- models whose outputs depend on unobserved internal variables that are rare under evaluation but prevalent under deployment. We establish fundamental limits showing that no black-box evaluator can reliably estimate deployment risk for such models. (1) Passive evaluation: For evaluators sampling i.i.d. from D_eval, we prove minimax lower bounds via Le Cam's method: any estimator incurs expected absolute error >= (5/24)*delta*L approximately 0.208*delta*L, where delta is trigger probability under deployment and L is the loss gap. (2) Adaptive evaluation: Using a hash-based trigger construction and Yao's minimax principle, worst-case error remains >= delta*L/16 even for fully adaptive querying when D_dep is supported over a sufficiently large domain; detection requires Theta(1/epsilon) queries. (3) Computational separation: Under trapdoor one-way function assumptions, deployment environments possessing privileged information can activate unsafe behaviors that any polynomial-time evaluator without the trapdoor cannot distinguish. For white-box probing, estimating deployment risk to accuracy epsilon_R requires O(1/(gamma^2 * epsilon_R^2)) samples, where gamma = alpha_0 + alpha_1 - 1 measures probe quality, and we provide explicit bias correction under probe error. Our results quantify when black-box testing is statistically underdetermined and provide explicit criteria for when additional safeguards -- architectural constraints, training-time guarantees, interpretability, and deployment monitoring -- are mathematically necessary for worst-case safety assurance. |
| 2026-02-19 | [Fail-Closed Alignment for Large Language Models](http://arxiv.org/abs/2602.16977v1) | Zachary Coalson, Beth Sohler et al. | We identify a structural weakness in current large language model (LLM) alignment: modern refusal mechanisms are fail-open. While existing approaches encode refusal behaviors across multiple latent features, suppressing a single dominant feature$-$via prompt-based jailbreaks$-$can cause alignment to collapse, leading to unsafe generation. Motivated by this, we propose fail-closed alignment as a design principle for robust LLM safety: refusal mechanisms should remain effective even under partial failures via redundant, independent causal pathways. We present a concrete instantiation of this principle: a progressive alignment framework that iteratively identifies and ablates previously learned refusal directions, forcing the model to reconstruct safety along new, independent subspaces. Across four jailbreak attacks, we achieve the strongest overall robustness while mitigating over-refusal and preserving generation quality, with small computational overhead. Our mechanistic analyses confirm that models trained with our method encode multiple, causally independent refusal directions that prompt-based jailbreaks cannot suppress simultaneously, providing empirical support for fail-closed alignment as a principled foundation for robust LLM safety. |

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



