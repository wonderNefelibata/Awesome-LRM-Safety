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
| 2026-03-19 | [Box Maze: A Process-Control Architecture for Reliable LLM Reasoning](http://arxiv.org/abs/2603.19182v1) | Zou Qiang | Large language models (LLMs) demonstrate strong generative capabilities but remain vulnerable to hallucination and unreliable reasoning under adversarial prompting. Existing safety approaches -- such as reinforcement learning from human feedback (RLHF) and output filtering -- primarily operate at the behavioral level and may lack explicit architectural mechanisms for enforcing reasoning process integrity.   This paper proposes the Box Maze framework, a conceptual process-control architecture that decomposes LLM reasoning into three explicit layers: memory grounding, structured inference, and boundary enforcement. We introduce preliminary simulation-based evaluation involving progressive boundary erosion scenarios across multiple heterogeneous LLM systems (DeepSeek-V3, Doubao, Qwen). Results from n=50 adversarial scenarios suggest that explicit cognitive control layers may improve consistency in boundary maintenance, with architectural constraints reducing boundary failure rates from approximately 40% (baseline RLHF) to below 1% under adversarial conditions.   While current validation is simulation-based, these preliminary results indicate that process-level control may offer a promising direction for improving reliability in large language model reasoning. |
| 2026-03-19 | [ADMM-Based Distributed MPC with Control Barrier Functions for Safe Multi-Robot Quadrupedal Locomotion](http://arxiv.org/abs/2603.19170v1) | Yicheng Zeng, Ruturaj S. Sambhus et al. | This paper proposes a fully decentralized model predictive control (MPC) framework with control barrier function (CBF) constraints for safety-critical trajectory planning in multi-robot legged systems. The incorporation of CBF constraints introduces explicit inter-agent coupling, which prevents direct decomposition of the resulting optimal control problems. To address this challenge, we reformulate the centralized safety-critical MPC problem using a structured distributed optimization framework based on the alternating direction method of multipliers (ADMM). By introducing a novel node-edge splitting formulation with consensus constraints, the proposed approach decomposes the global problem into independent node-local and edge-local quadratic programs that can be solved in parallel using only neighbor-to-neighbor communication. This enables fully decentralized trajectory optimization with symmetric computational load across agents while preserving safety and dynamic feasibility. The proposed framework is integrated into a hierarchical locomotion control architecture for quadrupedal robots, combining high-level distributed trajectory planning, mid-level nonlinear MPC enforcing single rigid body dynamics, and low-level whole-body control enforcing full-order robot dynamics. The effectiveness of the proposed approach is demonstrated through hardware experiments on two Unitree Go2 quadrupedal robots and numerical simulations involving up to four robots navigating uncertain environments with rough terrain and external disturbances. The results show that the proposed distributed formulation achieves performance comparable to centralized MPC while reducing the average per-cycle planning time by up to 51% in the four-agent case, enabling efficient real-time decentralized implementation. |
| 2026-03-19 | [UGID: Unified Graph Isomorphism for Debiasing Large Language Models](http://arxiv.org/abs/2603.19144v1) | Zikang Ding, Junchi Yao et al. | Large language models (LLMs) exhibit pronounced social biases. Output-level or data-optimization--based debiasing methods cannot fully resolve these biases, and many prior works have shown that biases are embedded in internal representations. We propose \underline{U}nified \underline{G}raph \underline{I}somorphism for \underline{D}ebiasing large language models (\textit{\textbf{UGID}}), an internal-representation--level debiasing framework for large language models that models the Transformer as a structured computational graph, where attention mechanisms define the routing edges of the graph and hidden states define the graph nodes. Specifically, debiasing is formulated as enforcing invariance of the graph structure across counterfactual inputs, with differences allowed only on sensitive attributes. \textit{\textbf{UGID}} jointly constrains attention routing and hidden representations in bias-sensitive regions, effectively preventing bias migration across architectural components. To achieve effective behavioral alignment without degrading general capabilities, we introduce a log-space constraint on sensitive logits and a selective anchor-based objective to preserve definitional semantics. Extensive experiments on large language models demonstrate that \textit{\textbf{UGID}} effectively reduces bias under both in-distribution and out-of-distribution settings, significantly reduces internal structural discrepancies, and preserves model safety and utility. |
| 2026-03-19 | [SHAPCA: Consistent and Interpretable Explanations for Machine Learning Models on Spectroscopy Data](http://arxiv.org/abs/2603.19141v1) | Mingxing Zhang, Nicola Rossberg et al. | In recent years, machine learning models have been increasingly applied to spectroscopic datasets for chemical and biomedical analysis. For their successful adoption, particularly in clinical and safety-critical settings, professionals and researchers must be able to understand and trust the reasoning behind model predictions. However, the inherently high dimensionality and strong collinearity of spectroscopy data pose a fundamental challenge to model explainability. These properties not only complicate model training but also undermine the stability and consistency of explanations, leading to fluctuations in feature importance across repeated training runs. Feature extraction techniques have been used to reduce the input dimensionality; these new features hinder the connection between the prediction and the original signal. This study proposes SHAPCA, an explainable machine learning pipeline that combines Principal Component Analysis (for dimensionality reduction) and Shapely Additive exPlanations (for post hoc explanation) to provide explanations in the original input space, which a practitioner can interpret and link back to the biological components. The proposed framework enables analysis from both global and local perspectives, revealing the spectral bands that drive overall model behaviour as well as the instance-specific features that influence individual predictions. Numerical analysis demonstrated the interpretability of the results and greater consistency across different runs. |
| 2026-03-19 | [On Optimizing Multimodal Jailbreaks for Spoken Language Models](http://arxiv.org/abs/2603.19127v1) | Aravind Krishnan, Karolina Stańczak et al. | As Spoken Language Models (SLMs) integrate speech and text modalities, they inherit the safety vulnerabilities of their LLM backbone and an expanded attack surface. SLMs have been previously shown to be susceptible to jailbreaking, where adversarial prompts induce harmful responses. Yet existing attacks largely remain unimodal, optimizing either text or audio in isolation. We explore gradient-based multimodal jailbreaks by introducing JAMA (Joint Audio-text Multimodal Attack), a joint multimodal optimization framework combining Greedy Coordinate Gradient (GCG) for text and Projected Gradient Descent (PGD) for audio, to simultaneously perturb both modalities. Evaluations across four state-of-the-art SLMs and four audio types demonstrate that JAMA surpasses unimodal jailbreak rate by 1.5x to 10x. We analyze the operational dynamics of this joint attack and show that a sequential approximation method makes it 4x to 6x faster. Our findings suggest that unimodal safety is insufficient for robust SLMs. The code and data are available at https://repos.lsv.uni-saarland.de/akrishnan/multimodal-jailbreak-slm |
| 2026-03-19 | [Exact-Time Safety Recovery using Time-Varying Control Barrier Functions with Optimal Barrier Tracking](http://arxiv.org/abs/2603.19119v1) | Yingqing Chen, Christos G. Cassandras et al. | This paper is motivated by controllers developed for autonomous vehicles which occasionally result into conditions where safety is no longer guaranteed. We develop an exact-time safety recovery framework for any control-affine nonlinear system when its state is outside a safe region using time-varying Control Barrier Functions (CBFs) with optimal barrier tracking. Unlike conventional formulations that provide only conservative upper bounds on recovery time convergence, the proposed approach guarantees recovery to the safe set at a prescribed time. The key mechanism is an active barrier tracking condition that forces the barrier function to follow exactly a designer-specified recovery trajectory. This transforms safety recovery into a trajectory design problem. The recovery trajectory is parameterized and optimized to achieve optimal performance while preserving feasibility under input constraints, avoiding the aggressive corrective actions typically induced by conventional finite-time formulations. The safety recovery framework is applied to the roundabout traffic coordination problem for Connected and Automated Vehicles (CAVs), where any initially violated safe merging constraint is replaced by an exact-time recovery barrier constraint to ensure safety guarantee restoration before CAV conflict points are reached. Simulation results demonstrate improved feasibility and performance. |
| 2026-03-19 | [FedTrident: Resilient Road Condition Classification Against Poisoning Attacks in Federated Learning](http://arxiv.org/abs/2603.19101v1) | Sheng Liu, Panos Papadimitratos | FL has emerged as a transformative paradigm for ITS, notably camera-based Road Condition Classification (RCC). However, by enabling collaboration, FL-based RCC exposes the system to adversarial participants launching Targeted Label-Flipping Attacks (TLFAs). Malicious clients (vehicles) can relabel their local training data (e.g., from an actual uneven road to a wrong smooth road), consequently compromising global model predictions and jeopardizing transportation safety. Existing countermeasures against such poisoning attacks fail to maintain resilient model performance near the necessary attack-free levels in various attack scenarios due to: 1) not tailoring poisoned local model detection to TLFAs, 2) not excluding malicious vehicular clients based on historical behavior, and 3) not remedying the already-corrupted global model after exclusion. To close this research gap, we propose FedTrident, which introduces: 1) neuron-wise analysis for local model misbehavior detection (notably including attack goal identification, critical feature extraction, and GMM-based model clustering and filtering); 2) adaptive client rating for client exclusion according to the local model detection results in each FL round; and 3) machine unlearning for corrupted global model remediation once malicious clients are excluded during FL. Extensive evaluation across diverse FL-RCC models, tasks, and configurations demonstrates that FedTrident can effectively thwart TLFAs, achieving performance comparable to that in attack-free scenarios and outperforming eight baseline countermeasures by 9.49% and 4.47% for the two most critical metrics. Moreover, FedTrident is resilient to various malicious client rates, data heterogeneity levels, complicated multi-task, and dynamic attacks. |
| 2026-03-19 | [TAU-R1: Visual Language Model for Traffic Anomaly Understanding](http://arxiv.org/abs/2603.19098v1) | Yuqiang Lin, Kehua Chen et al. | Traffic Anomaly Understanding (TAU) is important for traffic safety in Intelligent Transportation Systems. Recent vision-language models (VLMs) have shown strong capabilities in video understanding. However, progress on TAU remains limited due to the lack of benchmarks and task-specific methodologies. To address this limitation, we introduce Roundabout-TAU, a dataset constructed from real-world roundabout videos collected in collaboration with the City of Carmel, Indiana. The dataset contains 342 clips and is annotated with more than 2,000 question-answer pairs covering multiple aspects of traffic anomaly understanding. Building on this benchmark, we propose TAU-R1, a two-layer vision-language framework for TAU. The first layer is a lightweight anomaly classifier that performs coarse anomaly categorisation, while the second layer is a larger anomaly reasoner that generates detailed event summaries. To improve task-specific reasoning, we introduce a two-stage training strategy consisting of decomposed-QA-enhanced supervised fine-tuning followed by TAU-GRPO, a GRPO-based post-training method with TAU-specific reward functions. Experimental results show that TAU-R1 achieves strong performance on both anomaly classification and reasoning tasks while maintaining deployment efficiency. The dataset and code are available at: https://github.com/siri-rouser/TAU-R1 |
| 2026-03-19 | [SAVeS: Steering Safety Judgments in Vision-Language Models via Semantic Cues](http://arxiv.org/abs/2603.19092v1) | Carlos Hinojosa, Clemens Grange et al. | Vision-language models (VLMs) are increasingly deployed in real-world and embodied settings where safety decisions depend on visual context. However, it remains unclear which visual evidence drives these judgments. We study whether multimodal safety behavior in VLMs can be steered by simple semantic cues. We introduce a semantic steering framework that applies controlled textual, visual, and cognitive interventions without changing the underlying scene content. To evaluate these effects, we propose SAVeS, a benchmark for situational safety under semantic cues, together with an evaluation protocol that separates behavioral refusal, grounded safety reasoning, and false refusals. Experiments across multiple VLMs and an additional state-of-the-art benchmark show that safety decisions are highly sensitive to semantic cues, indicating reliance on learned visual-linguistic associations rather than grounded visual understanding. We further demonstrate that automated steering pipelines can exploit these mechanisms, highlighting a potential vulnerability in multimodal safety systems. |
| 2026-03-19 | [Finite-sample bounds for multi-output system identification](http://arxiv.org/abs/2603.19073v1) | Léo Simpson, Katrin Baumgärtner et al. | This paper presents uniform-in-time finite-sample bounds for regularized linear regression with vector-valued outputs and conditionally zero-mean subgaussian noise. By revisiting classical self-normalized martingale arguments, we obtain bounds that apply directly to multi-output regression, unlike most of the prior work. Compared to the state of the art, the new results are more general and yield tighter bounds, even for scalar-valued outputs. The mild assumptions we use allow for unknown dependencies between regressors and past noise terms, typically induced by system dynamics or feedback mechanisms. Therefore, these novel finite-sample bounds can be applied to many affine-in-parameter system identification problems, including the identification of a linear time-invariant system from full-state measurements. These new results may lead to significant improvements in stochastic learning-based controllers for safety-critical applications. |
| 2026-03-19 | [Security awareness in LLM agents: the NDAI zone case](http://arxiv.org/abs/2603.19011v1) | Enrico Bottazzi, Pia Park | NDAI zones let inventor and investor agents negotiate inside a Trusted Execution Environment (TEE) where any disclosed information is deleted if no deal is reached. This makes full IP disclosure the rational strategy for the inventor's agent. Leveraging this infrastructure, however, requires agents to distinguish a secure environment from an insecure one, a capability LLM agents lack natively, since they can rely only on evidence passed through the context window to form awareness of their execution environment. We ask: How do different LLM models weight various forms of evidence when forming awareness of the security of their execution environment? Using an NDAI-style negotiation task across 10 language models and various evidence scenarios, we find a clear asymmetry: a failing attestation universally suppresses disclosure across all models, whereas a passing attestation produces highly heterogeneous responses: some models increase disclosure, others are unaffected, and a few paradoxically reduce it. This reveals that current LLM models can reliably detect danger signals but cannot reliably verify safety, the very capability required for privacy-preserving agentic protocols such as NDAI zones. Bridging this gap, possibly through interpretability analysis, targeted fine-tuning, or improved evidence architectures, remains the central open challenge for deploying agents that calibrate information sharing to actual evidence quality. |
| 2026-03-19 | [Safety-Guaranteed Imitation Learning from Nonlinear Model Predictive Control for Spacecraft Close Proximity Operations](http://arxiv.org/abs/2603.18910v1) | Alexander Meinert, Niklas Baldauf et al. | This paper presents a safety-guaranteed, runtime-efficient imitation learning framework for spacecraft close proximity control. We leverage Control Barrier Functions (CBFs) for safety certificates and Control Lyapunov Functions (CLFs) for stability as unified design principles across data generation, training, and deployment. First, a nonlinear Model Predictive Control (NMPC) expert enforces CBF constraints to provide safe reference trajectories. Second, we train a neural policy with a novel CBF-CLF-informed loss and DAgger-like rollouts with curriculum weighting, promoting data-efficiency and reducing future safety filter interventions. Third, at deployment a lightweight one-step CBF-CLF quadratic program minimally adjusts the learned control input to satisfy hard safety constraints while encouraging stability. We validate the approach for ESA-compliant close proximity operations, including fly-around with a spherical keep-out zone and final approach inside a conical approach corridor, using the Basilisk high-fidelity simulator with nonlinear dynamics and perturbations. Numerical experiments indicate stable convergence to decision points and strict adherence to safety under the filter, with task performance comparable to the NMPC expert while significantly reducing online computation. A runtime analysis demonstrates real-time feasibility on a commercial off-the-shelf processor, supporting onboard deployment for safety-critical on-orbit servicing. |
| 2026-03-19 | [A Stabilized Mortar Method for Discontinuities in Geological Media with Non-Conforming Grids](http://arxiv.org/abs/2603.18905v1) | Daniele Moretto, Andrea Franceschini et al. | Accurate numerical simulation of fault and fracture mechanics is critical for the performance and safety assessment of many subsurface systems. The discretized representation of discontinuity surfaces and the robust simulation of their frictional contact behavior still represent major challenges. In this work, we use the mortar method to enforce the contact constraints and allow for non-conformity around the discontinuity surface, with a set of Lagrange multipliers playing the role of interface tractions. The formulation combines piecewise linear displacements with piecewise constant multipliers defined on one side of the fault interface (the non-mortar side). This choice for the Lagrange multipliers has a number of advantages from practical and computational viewpoints, but violates the inf-sup stability constraint. In order to stabilize the proposed formulation, we develop a traction-jump stabilization term to be added to the constraint equations. We use a macro-element analysis to derive an algorithmic strategy that automatically evaluates the proper scaling of the stabilization, without requiring any additional user-selected parameter. Numerical experiments demonstrate that the proposed formulation not only restores the inf-sup stability condition, but also recovers stable traction profiles in the presence of finer non-mortar sides, where other inf-sup-stable formulations fail. The proposed method is finally used to simulate non-linear contact conditions in non-conforming corner-point grids typically used in industrial geological applications. |
| 2026-03-19 | [From Accuracy to Readiness: Metrics and Benchmarks for Human-AI Decision-Making](http://arxiv.org/abs/2603.18895v1) | Min Hun Lee | Artificial intelligence (AI) systems are deployed as collaborators in human decision-making. Yet, evaluation practices focus primarily on model accuracy rather than whether human-AI teams are prepared to collaborate safely and effectively. Empirical evidence shows that many failures arise from miscalibrated reliance, including overuse when AI is wrong and underuse when it is helpful.   This paper proposes a measurement framework for evaluating human-AI decision-making centered on team readiness. We introduce a four part taxonomy of evaluation metrics spanning outcomes, reliance behavior, safety signals, and learning over time, and connect these metrics to the Understand-Control-Improve (U-C-I) lifecycle of human-AI onboarding and collaboration.   By operationalizing evaluation through interaction traces rather than model properties or self-reported trust, our framework enables deployment-relevant assessment of calibration, error recovery, and governance. We aim to support more comparable benchmarks and cumulative research on human-AI readiness, advancing safer and more accountable human-AI collaboration. |
| 2026-03-19 | [Quantitative Introspection in Language Models: Tracking Internal States Across Conversation](http://arxiv.org/abs/2603.18893v1) | Nicolas Martorell | Tracking the internal states of large language models across conversations is important for safety, interpretability, and model welfare, yet current methods are limited. Linear probes and other white-box methods compress high-dimensional representations imperfectly and are harder to apply with increasing model size. Taking inspiration from human psychology, where numeric self-report is a widely used tool for tracking internal states, we ask whether LLMs' own numeric self-reports can track probe-defined emotive states over time. We study four concept pairs (wellbeing, interest, focus, and impulsivity) in 40 ten-turn conversations, operationalizing introspection as the causal informational coupling between a model's self-report and a concept-matched probe-defined internal state. We find that greedy-decoded self-reports collapse outputs to few uninformative values, but introspective capacity can be unmasked by calculating logit-based self-reports. This metric tracks interpretable internal states (Spearman $ρ= 0.40$-$0.76$; isotonic $R^2 = 0.12$-$0.54$ in LLaMA-3.2-3B-Instruct), follows how those states change over time, and activation steering confirms the coupling is causal. Furthermore, we find that introspection is present at turn 1 but evolves through conversation, and can be selectively improved by steering along one concept to boost introspection for another ($ΔR^2$ up to $0.30$). Crucially, these phenomena scale with model size in some cases, approaching $R^2 \approx 0.93$ in LLaMA-3.1-8B-Instruct, and partially replicate in other model families. Together, these results position numeric self-report as a viable, complementary tool for tracking internal emotive states in conversational AI systems. |
| 2026-03-19 | [Reasoning over mathematical objects: on-policy reward modeling and test time aggregation](http://arxiv.org/abs/2603.18886v1) | Pranjal Aggarwal, Marjan Ghazvininejad et al. | The ability to precisely derive mathematical objects is a core requirement for downstream STEM applications, including mathematics, physics, and chemistry, where reasoning must culminate in formally structured expressions. Yet, current LM evaluations of mathematical and scientific reasoning rely heavily on simplified answer formats such as numerical values or multiple choice options due to the convenience of automated assessment. In this paper we provide three contributions for improving reasoning over mathematical objects: (i) we build and release training data and benchmarks for deriving mathematical objects, the Principia suite; (ii) we provide training recipes with strong LLM-judges and verifiers, where we show that on-policy judge training boosts performance; (iii) we show how on-policy training can also be used to scale test-time compute via aggregation. We find that strong LMs such as Qwen3-235B and o3 struggle on Principia, while our training recipes can bring significant improvements over different LLM backbones, while simultaneously improving results on existing numerical and MCQA tasks, demonstrating cross-format generalization of reasoning abilities. |
| 2026-03-19 | [Tursio Database Search: How far are we from ChatGPT?](http://arxiv.org/abs/2603.18835v1) | Sulbha Jain, Shivani Tripathi et al. | Business users need to search enterprise databases using natural language, just as they now search the web using ChatGPT or Perplexity. However, existing benchmarks -- designed for open-domain QA or text-to-SQL -- do not evaluate the end-to-end quality of such a search experience. We present an evaluation framework for structured database search that generates realistic banking queries across varying difficulty levels and assesses answer quality using relevance, safety, and conversational metrics via an LLM-as-judge approach. We apply this framework to compare Tursio, a database search platform, against ChatGPT and Perplexity on a credit union banking schema. Our results show that Tursio achieves answer relevancy statistically comparable to both baselines (97.8% vs. 98.1% on simple, 90.0% vs. 100.0% on medium, 89.5% vs. 100.0% on hard questions), even though Tursio answers from a structured database while the baselines generate responses from the open web. We analyze the failure modes, identify database completeness as the primary bottleneck, and outline directions for improving both the evaluation methodology and the systems under evaluation. |
| 2026-03-19 | [Rethinking Uncertainty Quantification and Entanglement in Image Segmentation](http://arxiv.org/abs/2603.18792v1) | Jakob Lønborg Christensen, Vedrana Andersen Dahl et al. | Uncertainty quantification (UQ) is crucial in safety-critical applications such as medical image segmentation. Total uncertainty is typically decomposed into data-related aleatoric uncertainty (AU) and model-related epistemic uncertainty (EU). Many methods exist for modeling AU (such as Probabilistic UNet, Diffusion) and EU (such as ensembles, MC Dropout), but it is unclear how they interact when combined. Additionally, recent work has revealed substantial entanglement between AU and EU, undermining the interpretability and practical usefulness of the decomposition. We present a comprehensive empirical study covering a broad range of AU-EU model combinations, propose a metric to quantify uncertainty entanglement, and evaluate both across downstream UQ tasks. For out-of-distribution detection, ensembles exhibit consistently lower entanglement and superior performance. For ambiguity modeling and calibration the best models are dataset-dependent, with softmax/SSN-based methods performing well and Probabilistic UNets being less entangled. A softmax ensemble fares remarkably well on all tasks. Finally, we analyze potential sources of uncertainty entanglement and outline directions for mitigating this effect. |
| 2026-03-19 | [Mi:dm K 2.5 Pro](http://arxiv.org/abs/2603.18788v1) | KT Tech innovation Group | The evolving LLM landscape requires capabilities beyond simple text generation, prioritizing multi-step reasoning, long-context understanding, and agentic workflows. This shift challenges existing models in enterprise environments, especially in Korean-language and domain-specific scenarios where scaling is insufficient. We introduce Mi:dm K 2.5 Pro, a 32B parameter flagship LLM designed to address enterprise-grade complexity through reasoning-focused optimization.   Our methodology builds a robust data foundation via a quality-centric curation pipeline utilizing abstract syntax tree (AST) analysis for code, gap-filling synthesis for mathematics, and an LLM-based quality evaluator. Pre-training scales the model via layer-predictor-based Depth Upscaling (DuS) and a progressive strategy supporting a 128K token context window. Post-training introduces a specialized multi-stage pipeline, including Reasoning SFT, model merging, and asynchronous reinforcement learning (RL), to develop complex problem-solving skills. "Fusion Training" then rebalances these capabilities with conversational fluency, consistent response styling, and reliable tool-use.   The evaluations show that Mi:dm K 2.5 Pro achieves competitive performance against leading global and domestic models. In addition, it sets state-of-the-art results on Korean-specific benchmarks, showcasing deep linguistic and cultural understanding. Finally, Responsible AI evaluations validate safety against attacks, ensuring a secure profile for deployment with a balance of harmlessness and responsiveness. |
| 2026-03-19 | [CSSDF-Net: Safe Motion Planning Based on Neural Implicit Representations of Configuration Space Distance Field](http://arxiv.org/abs/2603.18669v1) | Haohua Chen, Yixuan Zhou et al. | High-dimensional manipulator operation in unstructured environments requires a differentiable, scene-agnostic distance query mechanism to guide safe motion generation. Existing geometric collision checkers are typically non-differentiable, while workspace-based implicit distance models are hindered by the highly nonlinear workspace--configuration mapping and often suffer from poor convergence; moreover, self-collision and environment collision are commonly handled as separate constraints. We propose Configuration-Space Signed Distance Field-Net (CSSDF-Net), which learns a continuous signed distance field directly in configuration space to provide joint-space distance and gradient queries under a unified geometric notion of safety. To enable zero-shot generalization without environment-specific retraining, we introduce a spatial-hashing-based data generation pipeline that encodes robot-centric geometric priors and supports efficient retrieval of risk configurations for arbitrary obstacle point sets. The learned distance field is integrated into safety-constrained trajectory optimization and receding-horizon MPC, enabling both offline planning and online reactive avoidance. Experiments on a planar arm and a 7-DoF manipulator demonstrate stable gradients, effective collision avoidance in static and dynamic scenes, and practical inference latency for large-scale point-cloud queries, supporting deployment in previously unseen environments. |

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



