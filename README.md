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
| 2026-02-17 | [Decision Quality Evaluation Framework at Pinterest](http://arxiv.org/abs/2602.15809v1) | Yuqi Tian, Robert Paine et al. | Online platforms require robust systems to enforce content safety policies at scale. A critical component of these systems is the ability to evaluate the quality of moderation decisions made by both human agents and Large Language Models (LLMs). However, this evaluation is challenging due to the inherent trade-offs between cost, scale, and trustworthiness, along with the complexity of evolving policies. To address this, we present a comprehensive Decision Quality Evaluation Framework developed and deployed at Pinterest. The framework is centered on a high-trust Golden Set (GDS) curated by subject matter experts (SMEs), which serves as a ground truth benchmark. We introduce an automated intelligent sampling pipeline that uses propensity scores to efficiently expand dataset coverage. We demonstrate the framework's practical application in several key areas: benchmarking the cost-performance trade-offs of various LLM agents, establishing a rigorous methodology for data-driven prompt optimization, managing complex policy evolution, and ensuring the integrity of policy content prevalence metrics via continuous validation. The framework enables a shift from subjective assessments to a data-driven and quantitative practice for managing content safety systems. |
| 2026-02-17 | [The Geometry of Alignment Collapse: When Fine-Tuning Breaks Safety](http://arxiv.org/abs/2602.15799v1) | Max Springer, Chung Peng Lee et al. | Fine-tuning aligned language models on benign tasks unpredictably degrades safety guardrails, even when training data contains no harmful content and developers have no adversarial intent. We show that the prevailing explanation, that fine-tuning updates should be orthogonal to safety-critical directions in high-dimensional parameter space, offers false reassurance: we show this orthogonality is structurally unstable and collapses under the dynamics of gradient descent. We then resolve this through a novel geometric analysis, proving that alignment concentrates in low-dimensional subspaces with sharp curvature, creating a brittle structure that first-order methods cannot detect or defend. While initial fine-tuning updates may indeed avoid these subspaces, the curvature of the fine-tuning loss generates second-order acceleration that systematically steers trajectories into alignment-sensitive regions. We formalize this mechanism through the Alignment Instability Condition, three geometric properties that, when jointly satisfied, lead to safety degradation. Our main result establishes a quartic scaling law: alignment loss grows with the fourth power of training time, governed by the sharpness of alignment geometry and the strength of curvature coupling between the fine-tuning task and safety-critical parameters. These results expose a structural blind spot in the current safety paradigm. The dominant approaches to safe fine-tuning address only the initial snapshot of a fundamentally dynamic problem. Alignment fragility is not a bug to be patched; it is an intrinsic geometric property of gradient descent on curved manifolds. Our results motivate the development of curvature-aware methods, and we hope will further enable a shift in alignment safety analysis from reactive red-teaming to predictive diagnostics for open-weight model deployment. |
| 2026-02-17 | [Three-Dimensional Optical-Electrical Simulation of Cs2AgBiBr6 Double Perovskite Solar Cells](http://arxiv.org/abs/2602.15759v1) | Md Shanian Moed, Adnan Amin Siddiquee et al. | Despite significant advances in lead-free perovskite photovoltaics, achieving a balance among environmental safety and high optoelectronic performance remains challenging. The inorganic double perovskite Cs2AgBiBr6 has emerged as a promising candidate owing to its robust three-dimensional crystal structure and suitable visible-range bandgap. However, best power conversion efficiencies (PCEs) for Cs2AgBiBr6 solar cells reported so far - 6.37% experimentally and 27.78% in numerical studies - remain below the theoretical performance potential, largely due to suboptimal charge transport layers, and interface-related recombination losses. Here, we address this gap using a 3D finite-element method (FEM) implemented in COMSOL Multiphysics, which couples optical simulations with semiconductor drift-diffusion transport. To our knowledge, this work represents the first comprehensive 3D FEM-based study of a double halide perovskite solar cell. Screening of 25 electron transport layer (ETL)-hole transport layer (HTL) combinations identifies CeO2 and P3HT as the optimal ETL and HTL respectively. Device performance is further analyzed through systematic variation of layer thicknesses, doping concentrations and defect densities within the FTO/CeO2/Cs2AgBiBr6/P3HT/Au architecture. Under optimized parameters, the simulated device achieves a PCE of 31.76%, representing the theoretical upper bound predicted by the model. Overall, this work demonstrates 3D physics-based device engineering as a decisive pathway for overcoming efficiency bottlenecks in lead-free double perovskite photovoltaics. |
| 2026-02-17 | [Estimating Human Muscular Fatigue in Dynamic Collaborative Robotic Tasks with Learning-Based Models](http://arxiv.org/abs/2602.15684v1) | Feras Kiki, Pouya P. Niaz et al. | Assessing human muscle fatigue is critical for optimizing performance and safety in physical human-robot interaction(pHRI). This work presents a data-driven framework to estimate fatigue in dynamic, cyclic pHRI using arm-mounted surface electromyography(sEMG). Subject-specific machine-learning regression models(Random Forest, XGBoost, and Linear Regression predict the fraction of cycles to fatigue(FCF) from three frequency-domain and one time-domain EMG features, and are benchmarked against a convolutional neural network(CNN) that ingests spectrograms of filtered EMG. Framing fatigue estimation as regression (rather than classification) captures continuous progression toward fatigue, supporting earlier detection, timely intervention, and adaptive robot control. In experiments with ten participants, a collaborative robot under admittance control guided repetitive lateral (left-right) end-effector motions until muscular fatigue. Average FCF RMSE across participants was 20.8+/-4.3% for the CNN, 23.3+/-3.8% for Random Forest, 24.8+/-4.5% for XGBoost, and 26.9+/-6.1% for Linear Regression. To probe cross-task generalization, one participant additionally performed unseen vertical (up-down) and circular repetitions; models trained only on lateral data were tested directly and largely retained accuracy, indicating robustness to changes in movement direction, arm kinematics, and muscle recruitment, while Linear Regression deteriorated. Overall, the study shows that both feature-based ML and spectrogram-based DL can estimate remaining work capacity during repetitive pHRI, with the CNN delivering the lowest error and the tree-based models close behind. The reported transfer to new motion patterns suggests potential for practical fatigue monitoring without retraining for every task, improving operator protection and enabling fatigue-aware shared autonomy, for safer fatigue-adaptive pHRI control. |
| 2026-02-17 | [Complexity and Misspecification](http://arxiv.org/abs/2602.15674v1) | Drew Fudenberg, Florian Mudekereza | We propose a tractable unified framework to study the evolution and interaction of model-misspecification concerns and complexity aversion in repeated decision problems. This aims to capture environments where decision makers worry that their models are misspecified while also disliking overly complex models. We find that pathological cycles caused by endogenous concerns for misspecification can be eliminated by penalizing complex models and show that such preferences for simplicity tend to favor safety, which can enhance welfare in the long run. We use our framework to provide new microfoundations for pervasive empirical phenomena such as "scale heterogeneity" in discrete-choice analysis, "probability neglect" in behavioral economics, and "home bias" in international finance. |
| 2026-02-17 | [CARE Drive A Framework for Evaluating Reason-Responsiveness of Vision Language Models in Automated Driving](http://arxiv.org/abs/2602.15645v1) | Lucas Elbert Suryana, Farah Bierenga et al. | Foundation models, including vision language models, are increasingly used in automated driving to interpret scenes, recommend actions, and generate natural language explanations. However, existing evaluation methods primarily assess outcome based performance, such as safety and trajectory accuracy, without determining whether model decisions reflect human relevant considerations. As a result, it remains unclear whether explanations produced by such models correspond to genuine reason responsive decision making or merely post hoc rationalizations. This limitation is especially significant in safety critical domains because it can create false confidence. To address this gap, we propose CARE Drive, Context Aware Reasons Evaluation for Driving, a model agnostic framework for evaluating reason responsiveness in vision language models applied to automated driving. CARE Drive compares baseline and reason augmented model decisions under controlled contextual variation to assess whether human reasons causally influence decision behavior. The framework employs a two stage evaluation process. Prompt calibration ensures stable outputs. Systematic contextual perturbation then measures decision sensitivity to human reasons such as safety margins, social pressure, and efficiency constraints. We demonstrate CARE Drive in a cyclist overtaking scenario involving competing normative considerations. Results show that explicit human reasons significantly influence model decisions, improving alignment with expert recommended behavior. However, responsiveness varies across contextual factors, indicating uneven sensitivity to different types of reasons. These findings provide empirical evidence that reason responsiveness in foundation models can be systematically evaluated without modifying model parameters. |
| 2026-02-17 | [Latency-aware Human-in-the-Loop Reinforcement Learning for Semantic Communications](http://arxiv.org/abs/2602.15640v1) | Peizheng Li, Xinyi Lin et al. | Semantic communication promises task-aligned transmission but must reconcile semantic fidelity with stringent latency guarantees in immersive and safety-critical services. This paper introduces a time-constrained human-in-the-loop reinforcement learning (TC-HITL-RL) framework that embeds human feedback, semantic utility, and latency control within a semantic-aware Open radio access network (RAN) architecture. We formulate semantic adaptation driven by human feedback as a constrained Markov decision process (CMDP) whose state captures semantic quality, human preferences, queue slack, and channel dynamics, and solve it via a primal--dual proximal policy optimization algorithm with action shielding and latency-aware reward shaping. The resulting policy preserves PPO-level semantic rewards while tightening the variability of both air-interface and near-real-time RAN intelligent controller processing budgets. Simulations over point-to-multipoint links with heterogeneous deadlines show that TC-HITL-RL consistently meets per-user timing constraints, outperforms baseline schedulers in reward, and stabilizes resource consumption, providing a practical blueprint for latency-aware semantic adaptation. |
| 2026-02-17 | [The Stationarity Bias: Stratified Stress-Testing for Time-Series Imputation in Regulated Dynamical Systems](http://arxiv.org/abs/2602.15637v1) | Amirreza Dolatpour Fathkouhi, Alireza Namazi et al. | Time-series imputation benchmarks employ uniform random masking and shape-agnostic metrics (MSE, RMSE), implicitly weighting evaluation by regime prevalence. In systems with a dominant attractor -- homeostatic physiology, nominal industrial operation, stable network traffic -- this creates a systematic \emph{Stationarity Bias}: simple methods appear superior because the benchmark predominantly samples the easy, low-entropy regime where they trivially succeed. We formalize this bias and propose a \emph{Stratified Stress-Test} that partitions evaluation into Stationary and Transient regimes. Using Continuous Glucose Monitoring (CGM) as a testbed -- chosen for its rigorous ground-truth forcing functions (meals, insulin) that enable precise regime identification -- we establish three findings with broad implications:(i)~Stationary Efficiency: Linear interpolation achieves state-of-the-art reconstruction during stable intervals, confirming that complex architectures are computationally wasteful in low-entropy regimes.(ii)~Transient Fidelity: During critical transients (post-prandial peaks, hypoglycemic events), linear methods exhibit drastically degraded morphological fidelity (DTW), disproportionate to their RMSE -- a phenomenon we term the \emph{RMSE Mirage}, where low pointwise error masks the destruction of signal shape.(iii)~Regime-Conditional Model Selection: Deep learning models preserve both pointwise accuracy and morphological integrity during transients, making them essential for safety-critical downstream tasks. We further derive empirical missingness distributions from clinical trials and impose them on complete training data, preventing models from exploiting unrealistically clean observations and encouraging robustness under real-world missingness. This framework generalizes to any regulated system where routine stationarity dominates critical transients. |
| 2026-02-17 | [Multi-Objective Coverage via Constraint Active Search](http://arxiv.org/abs/2602.15595v1) | Zakaria Shams Siam, Xuefeng Liu et al. | In this paper, we formulate the new multi-objective coverage (MOC) problem where our goal is to identify a small set of representative samples whose predicted outcomes broadly cover the feasible multi-objective space. This problem is of great importance in many critical real-world applications, e.g., drug discovery and materials design, as this representative set can be evaluated much faster than the whole feasible set, thus significantly accelerating the scientific discovery process. Existing works cannot be directly applied as they either focus on sample space coverage or multi-objective optimization that targets the Pareto front. However, chemically diverse samples often yield identical objective profiles, and safety constraints are usually defined on the objectives. To solve this MOC problem, we propose a novel search algorithm, MOC-CAS, which employs an upper confidence bound-based acquisition function to select optimistic samples guided by Gaussian process posterior predictions. For enabling efficient optimization, we develop a smoothed relaxation of the hard feasibility test and derive an approximate optimizer. Compared to the competitive baselines, we show that our MOC-CAS empirically achieves superior performances across large-scale protein-target datasets for SARS-CoV-2 and cancer, each assessed on five objectives derived from SMILES-based features. |
| 2026-02-17 | [Req2Road: A GenAI Pipeline for SDV Test Artifact Generation and On-Vehicle Execution](http://arxiv.org/abs/2602.15591v1) | Denesa Zyberaj, Lukasz Mazur et al. | Testing functionality in Software-Defined Vehicles is challenging because requirements are written in natural language, specifications combine text, tables, and diagrams, while test assets are scattered across heterogeneous toolchains. Large Language Models and Vision-Language Models are used to extract signals and behavioral logic to automatically generate Gherkin scenarios, which are then converted into runnable test scripts. The Vehicle Signal Specification (VSS) integration standardizes signal references, supporting portability across subsystems and test benches. The pipeline uses retrieval-augmented generation to preselect candidate VSS signals before mapping. We evaluate the approach on the safety-relevant Child Presence Detection System, executing the generated tests in a virtual environment and on an actual vehicle. Our evaluation covers Gherkin validity, VSS mapping quality, and end-to-end executability. Results show that 32 of 36 requirements (89\%) can be transformed into executable scenarios in our setting, while human review and targeted substitutions remain necessary. This paper is a feasibility and architectural demonstration of an end-to-end requirements-to-test pipeline for SDV subsystems, evaluated on a CPDS case in simulation and Vehicle-in-the-Loop settings. |
| 2026-02-17 | [Constraining Streaming Flow Models for Adapting Learned Robot Trajectory Distributions](http://arxiv.org/abs/2602.15567v1) | Jieting Long, Dechuan Liu et al. | Robot motion distributions often exhibit multi-modality and require flexible generative models for accurate representation. Streaming Flow Policies (SFPs) have recently emerged as a powerful paradigm for generating robot trajectories by integrating learned velocity fields directly in action space, enabling smooth and reactive control. However, existing formulations lack mechanisms for adapting trajectories post-training to enforce safety and task-specific constraints. We propose Constraint-Aware Streaming Flow (CASF), a framework that augments streaming flow policies with constraint-dependent metrics that reshape the learned velocity field during execution. CASF models each constraint, defined in either the robot's workspace or configuration space, as a differentiable distance function that is converted into a local metric and pulled back into the robot's control space. Far from restricted regions, the resulting metric reduces to the identity; near constraint boundaries, it smoothly attenuates or redirects motion, effectively deforming the underlying flow to maintain safety. This allows trajectories to be adapted in real time, ensuring that robot actions respect joint limits, avoid collisions, and remain within feasible workspaces, while preserving the multi-modal and reactive properties of streaming flow policies. We demonstrate CASF in simulated and real-world manipulation tasks, showing that it produces constraint-satisfying trajectories that remain smooth, feasible, and dynamically consistent, outperforming standard post-hoc projection baselines. |
| 2026-02-17 | [Efficient Road Renovation Scheduling under Uncertainty using Lower Bound Pruning](http://arxiv.org/abs/2602.15554v1) | Robbert Bosch, Patricia Rogetzer et al. | Urban infrastructure degrades over time, necessitating periodic renovation to maintain functionality and safety. When renovation is delayed beyond the infrastructure's remaining lifespan, costly emergency interventions become necessary to prevent failure. Decision makers must therefore balance expected emergency intervention costs against traffic congestion impacts. We formalize this trade-off as a road network maintenance scheduling problem with uncertain deadlines, which presents optimization challenges including computationally expensive evaluation and an exponentially growing solution space. To address these challenges, this paper contributes a hybrid optimization approach combining machine learning with genetic algorithms for large-scale infrastructure renovation scheduling under uncertainty. We formulate the problem as a bi-level multi-objective optimization problem that explicitly accounts for uncertain infrastructure lifespans through probabilistic failure models. We develop a progressive lower bound evaluation method that integrates machine learning surrogate models with a multi-objective genetic algorithm to improve solution quality by enabling more iterations within fixed computational budgets. We demonstrate the method's effectiveness on substantially larger problem instances (76 projects) than previously addressed in the literature, achieving statistically significant improvements across multiple performance metrics by increasing computational efficiency up to 40 times compared to standard approaches. |
| 2026-02-17 | [Approximation Theory for Lipschitz Continuous Transformers](http://arxiv.org/abs/2602.15503v1) | Takashi Furuya, Davide Murari et al. | Stability and robustness are critical for deploying Transformers in safety-sensitive settings. A principled way to enforce such behavior is to constrain the model's Lipschitz constant. However, approximation-theoretic guarantees for architectures that explicitly preserve Lipschitz continuity have yet to be established. In this work, we bridge this gap by introducing a class of gradient-descent-type in-context Transformers that are Lipschitz-continuous by construction. We realize both MLP and attention blocks as explicit Euler steps of negative gradient flows, ensuring inherent stability without sacrificing expressivity. We prove a universal approximation theorem for this class within a Lipschitz-constrained function space. Crucially, our analysis adopts a measure-theoretic formalism, interpreting Transformers as operators on probability measures, to yield approximation guarantees independent of token count. These results provide a rigorous theoretical foundation for the design of robust, Lipschitz continuous Transformer architectures. |
| 2026-02-17 | [LLM-as-Judge on a Budget](http://arxiv.org/abs/2602.15481v1) | Aadirupa Saha, Aniket Wagde et al. | LLM-as-a-judge has emerged as a cornerstone technique for evaluating large language models by leveraging LLM reasoning to score prompt-response pairs. Since LLM judgments are stochastic, practitioners commonly query each pair multiple times to estimate mean scores accurately. This raises a critical challenge: given a fixed computational budget $B$, how to optimally allocate queries across $K$ prompt-response pairs to minimize estimation error? % We present a principled variance-adaptive approach leveraging multi-armed bandit theory and concentration inequalities. Our method dynamically allocates queries based on estimated score variances, concentrating resources where uncertainty is highest. Further, our algorithm is shown to achieve a worst-case score-estimation error of $\tilde{O}\left(\sqrt{\frac{\sum_{i=1}^K σ_i^2}{B}}\right)$, $σ_i^2$ being the unknown score variance for pair $i \in [K]$ with near-optimal budget allocation. % Experiments on \emph{Summarize-From-Feedback} and \emph{HelpSteer2} demonstrate that our method significantly outperforms uniform allocation, reducing worst-case estimation error while maintaining identical budgets. Our work establishes a theoretical foundation for efficient LLM evaluation with practical implications for AI safety, model alignment, and automated assessment at scale. |
| 2026-02-17 | [Benchmarking IoT Time-Series AD with Event-Level Augmentations](http://arxiv.org/abs/2602.15457v1) | Dmitry Zhevnenko, Ilya Makarov et al. | Anomaly detection (AD) for safety-critical IoT time series should be judged at the event level: reliability and earliness under realistic perturbations. Yet many studies still emphasize point-level results on curated base datasets, limiting value for model selection in practice. We introduce an evaluation protocol with unified event-level augmentations that simulate real-world issues: calibrated sensor dropout, linear and log drift, additive noise, and window shifts. We also perform sensor-level probing via mask-as-missing zeroing with per-channel influence estimation to support root-cause analysis. We evaluate 14 representative models on five public anomaly datasets (SWaT, WADI, SMD, SKAB, TEP) and two industrial datasets (steam turbine, nuclear turbogenerator) using unified splits and event aggregation. There is no universal winner: graph-structured models transfer best under dropout and long events (e.g., on SWaT under additive noise F1 drops 0.804->0.677 for a graph autoencoder, 0.759->0.680 for a graph-attention variant, and 0.762->0.756 for a hybrid graph attention model); density/flow models work well on clean stationary plants but can be fragile to monotone drift; spectral CNNs lead when periodicity is strong; reconstruction autoencoders become competitive after basic sensor vetting; predictive/hybrid dynamics help when faults break temporal dependencies but remain window-sensitive. The protocol also informs design choices: on SWaT under log drift, replacing normalizing flows with Gaussian density reduces high-stress F1 from ~0.75 to ~0.57, and fixing a learned DAG gives a small clean-set gain (~0.5-1.0 points) but increases drift sensitivity by ~8x. |
| 2026-02-17 | [Flux pumping and bifurcated relaxations of helical core in 3D magnetohydrodynamic modelling of ASDEX Upgrade plasmas](http://arxiv.org/abs/2602.15431v1) | H. Zhang, M. Hoelzl et al. | Flux pumping was achieved in recent hybrid scenario experiments in the ASDEX Upgrade (AUG) tokamak, which is characterized by a sawtooth-free helical quiescent state and the anomalous radial redistribution of toroidal current density and poloidal magnetic flux. In this article, the self-regulation mechanism of the AUG core plasma during flux pumping is investigated at realistic parameters using the JOREK code based on the two-temperature, nonlinear, full magnetohydrodynamic (MHD) model. A key milestone in AUG flux pumping modelling is achieved by quantitatively reproducing the clamped current density and safety factor profiles in the plasma core, demonstrating the effectiveness of the dynamo effect in sustaining the flux pumping state. The dynamo term, that is of particular interest, is primarily generated by the pressure-gradient driven m/n = 1/1 quasi-interchange-like MHD instability. The work systematically extrapolates the parameter regimes of flux pumping from the above AUG base case by scanning dissipation coefficients and plasma beta. The simulation results reveal bifurcated plasma behaviours at different Hartmann numbers, including distinct states such as flux pumping (helical core with a flat current density), sawteeth (periodic kink-cycling), single crash (without subsequent cycle), and quasi-stationary magnetic island (peaked current density). Transitions from marginal flux pumping state to sawteeth are observed in long-term simulations. The relationships between system dissipation, plasma beta, and different plasma states are carefully analyzed. For practical purposes, the potential operational window for flux pumping, as determined by plasma density and temperature, is estimated. The modelling efforts advance the understanding of flux pumping and facilitate the development of a fast surrogate model for efficient evaluation of flux pumping. |
| 2026-02-17 | [Improving LLM Reliability through Hybrid Abstention and Adaptive Detection](http://arxiv.org/abs/2602.15391v1) | Ankit Sharma, Nachiket Tapas et al. | Large Language Models (LLMs) deployed in production environments face a fundamental safety-utility trade-off either a strict filtering mechanisms prevent harmful outputs but often block benign queries or a relaxed controls risk unsafe content generation. Conventional guardrails based on static rules or fixed confidence thresholds are typically context-insensitive and computationally expensive, resulting in high latency and degraded user experience. To address these limitations, we introduce an adaptive abstention system that dynamically adjusts safety thresholds based on real-time contextual signals such as domain and user history. The proposed framework integrates a multi-dimensional detection architecture composed of five parallel detectors, combined through a hierarchical cascade mechanism to optimize both speed and precision. The cascade design reduces unnecessary computation by progressively filtering queries, achieving substantial latency improvements compared to non-cascaded models and external guardrail systems. Extensive evaluation on mixed and domain-specific workloads demonstrates significant reductions in false positives, particularly in sensitive domains such as medical advice and creative writing. The system maintains high safety precision and near-perfect recall under strict operating modes. Overall, our context-aware abstention framework effectively balances safety and utility while preserving performance, offering a scalable solution for reliable LLM deployment. |
| 2026-02-17 | [Fine-Tuning LLMs to Generate Economical and Reliable Actions for the Power Grid](http://arxiv.org/abs/2602.15350v1) | Mohamad Chehade, Hao Zhu | Public Safety Power Shutoffs (PSPS) force rapid topology changes that can render standard operating points infeasible, requiring operators to quickly identify corrective transmission switching actions that reduce load shedding while maintaining acceptable voltage behavior. We present a verifiable, multi-stage adaptation pipeline that fine-tunes an instruction-tuned large language model (LLM) to generate \emph{open-only} corrective switching plans from compact PSPS scenario summaries under an explicit switching budget. First, supervised fine-tuning distills a DC-OPF MILP oracle into a constrained action grammar that enables reliable parsing and feasibility checks. Second, direct preference optimization refines the policy using AC-evaluated preference pairs ranked by a voltage-penalty metric, injecting voltage-awareness beyond DC imitation. Finally, best-of-$N$ selection provides an inference-time addition by choosing the best feasible candidate under the target metric. On IEEE 118-bus PSPS scenarios, fine-tuning substantially improves DC objective values versus zero-shot generation, reduces AC power-flow failure from 50\% to single digits, and improves voltage-penalty outcomes on the common-success set. Code and data-generation scripts are released to support reproducibility. |
| 2026-02-17 | [Noncooperative Coordination for Decentralized Air Traffic Management](http://arxiv.org/abs/2602.15333v1) | Jaehan Im | Decentralized air traffic management requires coordination among self-interested stakeholders operating under shared safety and capacity constraints, where conventional centralized or implicitly cooperative models do not adequately capture this setting. We develop a unified perspective on noncooperative coordination, in which system-level outcomes emerge by designing incentives and assigning signals that reshape individual optimality rather than imposing cooperation or enforcement. We advance this framework along three directions: scalable equilibrium engineering via reduced-rank and uncertainty-aware correlated equilibria, decentralized mechanism design for equilibrium selection without enforcement, and structured noncooperative dynamics with convergence guarantees. Beyond these technical contributions, we discuss core design principles that govern incentive-compatible coordination in decentralized systems. Together, these results establish a foundation for scalable, robust coordination in safety-critical air traffic systems. |
| 2026-02-17 | [Complex-Valued Unitary Representations as Classification Heads for Improved Uncertainty Quantification in Deep Neural Networks](http://arxiv.org/abs/2602.15283v1) | Akbar Anbar Jafari, Cagri Ozcinar et al. | Modern deep neural networks achieve high predictive accuracy but remain poorly calibrated: their confidence scores do not reliably reflect the true probability of correctness. We propose a quantum-inspired classification head architecture that projects backbone features into a complex-valued Hilbert space and evolves them under a learned unitary transformation parameterised via the Cayley map. Through a controlled hybrid experimental design - training a single shared backbone and comparing lightweight interchangeable heads - we isolate the effect of complex-valued unitary representations on calibration. Our ablation study on CIFAR-10 reveals that the unitary magnitude head (complex features evolved under a Cayley unitary, read out via magnitude and softmax) achieves an Expected Calibration Error (ECE) of 0.0146, representing a 2.4x improvement over a standard softmax head (0.0355) and a 3.5x improvement over temperature scaling (0.0510). Surprisingly, replacing the softmax readout with a Born rule measurement layer - the quantum-mechanically motivated approach - degrades calibration to an ECE of 0.0819. On the CIFAR-10H human-uncertainty benchmark, the wave function head achieves the lowest KL-divergence (0.336) to human soft labels among all compared methods, indicating that complex-valued representations better capture the structure of human perceptual ambiguity. We provide theoretical analysis connecting norm-preserving unitary dynamics to calibration through feature-space geometry, report negative results on out-of-distribution detection and sentiment analysis to delineate the method's scope, and discuss practical implications for safety-critical applications. Code is publicly available. |

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



