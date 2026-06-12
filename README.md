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
| 2026-06-11 | [Beyond Runtime Enforcement: Shield Synthesis as Defensibility Analysis for Adversarial Networks](http://arxiv.org/abs/2606.13621v1) | Achraf Hsain, Sultan Almuhammadi | Shielded reinforcement learning is typically presented as a runtime safety mechanism that compiles temporal-logic specifications into automata restricting an agent's actions. We argue this is the wrong product. The same automata-theoretic machinery -- specification compilation, product game construction, attractor computation, and winning-region extraction -- is better read as a design-time analytical instrument whose outputs are structural insights about a system rather than runtime constraints on a deployed agent.   We instantiate this through a constrained two-player safety game for network defense. The two specifications are enforced asymmetrically: the defender specification defines the unsafe region of the game, whereas the attacker specification restricts the adversary's legal actions during attractor computation. Solving the game yields a defensibility verdict -- a formal certificate that a topology-specification pair is or is not defensible -- with the associated winning region and shield.   Beyond the binary verdict, we derive topology-level metrics from the attractor structure and combine them with post-convergence behavior from shield-constrained adversarial multi-agent reinforcement learning. Together these form a defensibility fingerprint capturing both a network's formal safety properties and its operational behavior under adaptive play.   A what-if analysis shows that formal defensibility and operational effectiveness capture distinct aspects of security: small architectural changes can produce large shifts in operational outcomes while leaving formal safety margins nearly unchanged. Shield synthesis is thus most valuable not as a deployment mechanism for safe agents, but as a framework for answering architectural questions about whether, where, and how a system can be defended. The defensibility verdict is the output, not the safe policy. |
| 2026-06-11 | [Beyond the IT Checklist: Engineering a Reasonable Standard of Care for Cyber Safety](http://arxiv.org/abs/2606.13612v1) | Matthew E. Jablonski, Linton Wells et al. | Current U.S. cyber policy, centered on security, often treats documentation of controls and incident reports as a proxy for safety in the built environment. This paper argues that such an approach is inadequate for cyber-physical systems, where digital failures can produce kinetic harm. We construct and code a corpus of critical infrastructure policy documents (N=292, 2000-2025) to examine how "reasonable care" is operationalized across the NIST SP 800-160 Vol.~2 resilience lifecycle. The resulting maps show that obligations are concentrated in the Anticipate phase and emphasize administrative compliance, while Withstand and Recover phases rely heavily on delegated references to IT-focused control catalogs that are poorly aligned with physics-based hazards. We identify three major disconnects: miscalibrated delegated standards, recovery defined as notification rather than engineered navigation, and uneven adaptation requirements across sectors. We then propose a modernized standard of care anchored in hazard-specific traceability, structured assurance cases, and cyber resiliency engineering. Finally, we recommend that federal policy pair these engineering obligations with targeted incentives so that resilient architectures for critical infrastructure become a viable business decision rather than an unfunded expectation. |
| 2026-06-11 | [Probabilistic, Resource-Aware, Asynchronous, Out-of-Order Choreographies](http://arxiv.org/abs/2606.13520v1) | Mako Bates, Steven Baldasty et al. | Futures-based implementations of out-of-order choreographies can substantially improve latency and throughput, but their actual behavior depends on resources such as communication delay, computation time, failures, and recovery. Existing formal models such as Ozone's O3 describe which executions are possible, but do not directly explain how likely those executions are or how long they take. In this work we present AsInst, a probabilistic, resource-aware language for modeling the semantics of asynchronous choreographies with out-of-order execution. AsInst programs are interpreted as temporal Bayesian networks that model both the values produced at runtime and the times at which they become available. We prove that this central semantics correctly captures a corresponding futures-style network semantics. We also show that AsInst can encode Ozone-style select-and-merge conditionals, and we use case studies to model communication-failure recovery and analyze runtime performance. |
| 2026-06-11 | [Exploring Systems-Thinking Approaches to Loss of Control Risk](http://arxiv.org/abs/2606.13474v1) | Aurelio Carlucci, Sean P. Fillingham et al. | Internal deployment of agentic AI systems for coding and research creates a sociotechnical control problem that extends beyond model behaviour. We treat internal-deployment Loss of Control as the inability to reliably constrain, audit, reverse, or halt AI-mediated changes to code, infrastructure, evaluation, or deployment processes in time to prevent serious organisational or societal harms. We ask whether established systems-safety methods can identify risks that model-level evaluations may miss. Using a generic frontier-lab coding-agent scenario reconstructed from public materials, we apply STECA, STPA, and FRAM. The analyses surface complementary findings: published frameworks can leave governance responsibilities and feedback loops externally unverifiable; delays in monitoring and intervention can make otherwise valid control actions ineffective; and routine operational variability can gradually erode the calibration and independence of safeguards. We argue that frontier-AI risk management should pair model-focused evaluations with systems-level hazard analysis and operational assurance that tracks whether controls remain effective over time. |
| 2026-06-11 | [Embodied Opinion Dynamics for Safety-Critical Motion Control in Dynamic Environments](http://arxiv.org/abs/2606.13465v1) | Zhiqi Tang, Yu Xing | This paper proposes a novel adaptive control framework that embeds nonlinear opinion dynamics within the dynamical sensorimotor layers of an automated vehicle governed by second-order nonholonomic bicycle kinematics. The framework enables an ego vehicle to perform adaptive decision-making and achieve safe motion control under interaction uncertainty with non-cooperative neighboring agents. We consider a representative case study in which an ego vehicle autonomously attempts to merge into a lane occupied by human-driven or automated vehicles whose intentions are unknown. Within the proposed framework, the ego vehicle adaptively selects and executes merging versus non-merging behaviors in response to changing environmental conditions. Formal safety guarantees, as well as equilibrium and stability analyses of the closed-loop system, are provided. Numerical simulations further demonstrate the effectiveness of the proposed approach. |
| 2026-06-11 | [PolyFlow: Safe and Efficient Polytope-Constrained Flow Matching with Constraint Embedding and Projection-free Update](http://arxiv.org/abs/2606.13400v1) | Jianming Ma, Qiyue Yang et al. | While flow-based generative models have demonstrated strong performance across a wide range of domains, deploying them in safety-critical physical systems remains challenging due to strict constraint requirements. Existing approaches typically enforce safety through post-hoc corrections, which incur substantial computational overhead and may distort the learned distribution. We propose PolyFlow, a polytope-constrained flow matching framework that embeds constraints directly into the model and flow dynamics. PolyFlow introduces a discrete-time flow formulation and a projection-free architecture, which eliminate the discretization error and guarantee strict satisfaction of arbitrary polyhedral constraints, without the need for expensive iterative solvers. Experimental results show that PolyFlow achieves zero constraint violation while maintaining high distributional fidelity across a range of planning and control tasks. Compared to state-of-the-art constrained generation baselines, PolyFlow significantly reduces inference latency and demonstrates a favorable trade-off between safety, efficiency, and generative quality. Code is available on https://github.com/MJianM/PolyFlow. |
| 2026-06-11 | [Enhanced Photocurrent Response in Epitaxial 0.5PZT-0.5PFN Multiferroic Thin Films](http://arxiv.org/abs/2606.13399v1) | Lucia Imhoff, Miguel A. Rengifo et al. | The exploration of novel multiferroic materials with strong coupling between ferroelectric polarization and photovoltaic effects is crucial for next-generation optoelectronic devices. In this study, we characterized highly oriented 0.5Pb(Zr0.52Ti0.48)O3-0.5Pb(Fe0.5Nb0.5)O3 multiferroic thin films grown by pulsed laser deposition on SrTiO3 (001) substrates with a SrRuO3 bottom electrode. The films exhibited excellent crystalline quality, with a single perovskite phase and (001) orientation. They displayed good ferroelectric properties (remanent polarization $\sim$17 $μ$C/cm$^2$, PUND; coercive field $\sim$150 kV/cm), alongside weak ferromagnetic behavior at room temperature (remanence 1.30 emu/cm$^3$; coercive field 90 Oe). Photovoltaic measurements demonstrated a robust, polarization-dependent photoresponse under 403 nm monochromatic laser illumination, achieving Jsc values between $\pm$20 $μ$A/cm$^2$. This compelling observation confirms the intrinsic coupling between ferroelectric polarization and photovoltaic effects, highlighting the considerable promise of these single-phase multiferroic thin films for advanced photovoltaic and optoelectronic memory applications. |
| 2026-06-11 | [Navigating the Safety-Fidelity Trade-off: Massive-Variate Time Series Forecasting for Power Systems via Probabilistic Scenarios](http://arxiv.org/abs/2606.13338v1) | Kaijie Xu, Anqi Wang et al. | Probabilistic forecasting models are increasingly deployed on multivariate systems with distinct channel physics and operational constraints, but existing benchmarks evaluate neither property at scale. Public canonical multivariate benchmarks cap out at 2,000 channels, while power-system benchmarks either lack temporal structure or probabilistic evaluation. We introduce PowerPhase, a probabilistic forecasting benchmark built on six transmission grids ranging from 2,000 to 36,964 jointly forecasted channels, more than an order of magnitude beyond popular canonical multivariate benchmarks. Each target trajectory is the output of an AC power-flow solve, and PowerPhase ships with constraint-aware metrics, including Safety_mBrier, NECV, and CVaR-alpha, that complement CRPS and Distortion. Across eight baselines and three seeds, distributional accuracy and constraint satisfaction rank models differently, a trade-off we term safety-fidelity. We further propose PowerForge, a scenario-based quantile forecaster with type-specific decoding heads and a causal bridge between variable groups, which achieves the best average rank on every grid. |
| 2026-06-11 | [Feasibility Assessment of Remote Driving via Latency Analysis of ITS-G5 and Cellular Networks in the MASA Living Lab](http://arxiv.org/abs/2606.13292v1) | Gaetano Orazio Cauchi, Antonio Solida et al. | Remote driving has gained increasing attention as a key enabler for connected and automated vehicles. Yet its practical deployment hinges on wireless networks' ability to guarantee low, predictable latency. In this paper, we present an extensive latency analysis of ITS-G5 and cellular (5G) technologies within the Modena Automotive Smart Area (MASA), a real-world, city-scale testbed equipped with a distributed intelligent transportation infrastructure. By conducting controlled experiments under varying network loads and traffic conditions, we measure network and end-to-end latency components relevant to remote driving, in which the uplink consists of a continuous video stream transmitted from the vehicle to the remote operator, and the downlink conveys control commands back to the car. Measurements conducted under diverse conditions reveal how latency and variability differ across the two technologies and how infrastructure coverage impacts video-stream transmission performance. Based on the observed latency distributions and reliability metrics, we assess the practical feasibility and safety margins of remote driving in mixed network environments. The results provide actionable insights for future teleoperation deployments and motivate hybrid communication strategies that combine the strengths of ITS-G5 and cellular networks. |
| 2026-06-11 | [Multi-Field Hybrid Retrieval-Augmented Generation for Maritime Accident Root Cause Analysis](http://arxiv.org/abs/2606.13249v1) | Seongjin Kim, Sungil Kim | Maritime accident adjudication reports contain critical tribunal findings for root cause analysis (RCA), yet retrieving relevant precedents and drafting consistent reports from decades of records remains labor-intensive. This paper proposes a multi-field hybrid retrieval-augmented generation (RAG) framework for automated maritime RCA, utilizing a comprehensive dataset of 13,329 Korea Maritime Safety Tribunal (KMST) reports (1971-2025). We transform raw adjudications into a structured knowledge base of "incident cards", indexing three distinct fields-Summary, Causes, and Disposition-alongside a hierarchical L1/L2 cause taxonomy. Our retrieval strategy employs a field-aware hybrid approach, fusing sparse and dense rankings via Reciprocal Rank Fusion (RRF). Given the lack of large-scale expert relevance labels, we evaluate retrieval performance using ceiling-normalized recall and nDCG based on a metadata-derived proxy relevance score. Experimental results demonstrate that our proposed retrieval significantly outperforms baseline methods, improving NormRecall@100 from 0.18 to 0.55. Furthermore, grounding the generator on the retrieved precedents enhances RCA generation quality over an LLM-only baseline, increasing the LLM-as-a-judge score from 3.34 to 3.72. These findings suggest that field-aware RAG can substantially streamline maritime safety investigation workflows by enabling faster precedent search and more consistent, evidence-based RCA drafting. |
| 2026-06-11 | [Embedding ISO 10218 Safety Compliance in Robots via Control Barrier Functions for Human-Robot Collaboration](http://arxiv.org/abs/2606.13203v1) | Federico Parma, Cesare Tonola et al. | Human-Robot Collaboration (HRC) requires strict adherence to safety standards, such as ISO 10218, to prevent harmful interactions. Standard Speed and Separation Monitoring (SSM) filters calculate safe robotic speeds based on conservative assumptions, such as constant human velocity, which prevents accurate predictions of minimum separation distances and causes unnecessary operational halts. This paper proposes a Control Barrier Function (CBF) that explicitly incorporates human acceleration data to analytically forward-predict the minimum human-robot separation distance during a worst-case robotic stopping trajectory. To guarantee safety at the control level, this predictive CBF is integrated as an inequality constraint within a Sequential Quadratic Programming (SQP) framework. Specifically, two methods are proposed: Method I, a CBF-constrained PD safety filter; and Method II, a task-scaling SQP controller that enforces a spatial tube constraint. Simulated and real-world experiments on a UR10e robot evaluate the two proposed methods against a standard industrial SSM module baseline. Results demonstrate that Method II dynamically modulates execution speed and confines spatial deviations. Compared to Method I, Method II achieves a 63\% reduction in mean trajectory error and avoids excessive evasive manoeuvres, ensuring high task throughput while complying with ISO 10218 SSM guidelines. |
| 2026-06-11 | [MPC for underactuated spacecraft control with a Lyapunov supervised physics-informed neural network correction layer](http://arxiv.org/abs/2606.13113v1) | Amirhossein Ayanmanesh Motlaghmofrad, Carlo Cena et al. | Underactuated spacecraft faces controllability limitations and heightened sensitivity to environmental disturbances, complicating attitude maneuvering and stabilization. Due to the lack of control authority along the underactuated axis, conventional controllers cannot directly stabilize all attitude components and therefore require reference planning strategies. Furthermore, MPC approaches remain sensitive to inertia uncertainty and unmodeled dynamic couplings, resulting in degraded tracking performance under mismatch. To address these issues, we consider a hierarchical architecture integrating three layers: (i) a nonlinear model predictive controller (NMPC) for constraint and underactuation-aware maneuver planning and nominal closed-loop stability under actuator limits; (ii) a physics-informed neural network (PINN) trained offline on simulation data to estimate residual disturbance torques, with loss terms that enforce consistency with rigid-body rotational dynamics; (iii) a Lyapunov-based supervisory safety mechanism that evaluates the learned correction online and bounds or suppresses its influence to preserve the stability properties of the baseline controller. The architecture is evaluated in a high-fidelity simulation environment modelling reaction wheel dynamics, actuator saturation, and environmental disturbances. Monte Carlo studies show statistically significant reductions in steady-state attitude error relative to standalone NMPC while maintaining robust behavior under uncertainty. The supervisory layer ensures graceful degradation to purely model-based control when the learning-based augmentation is unreliable. |
| 2026-06-11 | [Functional Cache Grafting: Robust and Rapid Code-Policy Synthesis for Embodied Agents](http://arxiv.org/abs/2606.13097v1) | Saehun Chun, Wonje Choi et al. | Code-writing large language models (CodeLLMs) generate executable code policies for embodied agents by translating natural language goals and environmental constraints into structured control programs. However, policy generation in open-domain embodied environments suffers from two fundamental limitations: (i) delayed decoding caused by repetitive prefill computation over long prompts, and (ii) limited robustness due to fully generative decoding, which often produces API mismatches, missing safety guards, and unstable control logic. To address these limitations, we present FCGraft, a Functional Cache Grafting framework. FCGraft maintains a library of function-level validated code skeletons and their associated prompt-level Transformer key-value (KV) caches, and synthesizes new policies by retrieving relevant functions and grafting their KV caches when a new task is provided. Given retrieved function caches, FCGraft performs cache grafting via stitching, which composes cached function segments into a composite policy, and patching, which locally adapts only the necessary code regions to satisfy task-specific parameters and constraints with minimal additional decoding. By eliminating redundant prefill computation, this approach reduces generation latency, while reusing validated control structures improves robustness over prompt-level caching methods RAGCache, achieving 18.31% higher task success rate and 2.3x faster policy synthesis. |
| 2026-06-11 | [Efficient Estimation of A-basis and B-Basis Value under Epistemic Uncertainty using Importance Sampling and Control Variates](http://arxiv.org/abs/2606.13094v1) | Elton Donfack-Siewe, Jérôme Morio et al. | In aerospace certification and other safety-critical domains, conservative quantile estimation such as A- and B-basis values is essential to guarantee reliability. While these metrics are traditionally derived from experimental campaigns, this work focuses on their estimation using a validated deterministic numerical model. The problem is formulated under mixed aleatory-epistemic uncertainty, accounting for limited material data, finite sampling effects, and surrogate modeling errors. We propose a methodology for estimating conservative design quantiles with statistical guarantees under mixed uncertainties. The proposed method leverages importance sampling and control variates to achieve accurate and efficient estimates within a fixed computational budget. One key point is the surrogate model's role solely as a variance reduction device, which guarantees unbiased and consistent quantile estimation. By explicitly integrating all sources of uncertainty, the proposed framework provides a numerical alternative to estimate A-basis and B-Basis. Furthermore, Sobol-based sensitivity indices are obtained at no additional cost, offering insight into the dominant epistemic sources. Numerical experiments on structural models demonstrate the method's reliability and computational efficiency. In particular, the application to large-scale industrial simulations confirms its suitability for aerospace certification workflows and highlights its relevance for real world engineering environments. |
| 2026-06-11 | [Effects of Social Interactions in Self-Organising Railway Traffic Management](http://arxiv.org/abs/2606.13068v1) | Fabio Oddi, Federico Naldini et al. | Recent research is exploring self-organised traffic management as a solution for scaling to complex real-world networks. In such a system, trains predict their neighbourhood, produce traffic plan hypotheses, and agree via consensus with neighbours on a future traffic plan to be implemented. This paper investigates a structural parameter within this pipeline: the predictive neighbourhood horizon. The horizon is used by trains to identify future potential conflicts with neighbours, and to establish the local interaction topology, that is, the subset of trains to negotiate with. As the primary design variable, the horizon directly determines the size and density of the social interaction graph, whereas its impact on the complexity of local sub-problems and the distributed consensus dynamics represents a trade-off to be explored. Through a closed-loop simulation framework the study evaluates how variations of the horizon impact the overall decentralised coordination process, from initial conflict detection to distributed schedule consensus. The analysis focuses on investigating the potential trade-off introduced by the horizon choice: balancing local tractability and computational responsiveness with the need for global schedule coherence and feasibility in safety-critical environments. Contrary to intuition, our empirical results indicate that the short time horizons suffice, while long values compromise local tractability and computational responsiveness with no gain in global schedule optimality. |
| 2026-06-11 | [SciR: A Controllable Benchmark for Scientific Reasoning in LLMs](http://arxiv.org/abs/2606.13020v1) | Pierre Beckmann, Marco Valentino et al. | Three paradigmatic forms of inference recur across scientific reasoning: deduction, induction, and causal abduction. Reliably evaluating LLMs on these in scientific settings is currently out of reach: scientific benchmarks built on human annotations are costly and lack mechanistic ground truth, while synthetic logical-reasoning benchmarks do not resemble real scientific documents. We introduce SciR, a benchmark that combines multi-paradigm reasoning with controllable scientific rendering, anchored on three paradigmatic scientific problems. Tasks are generated from formal objects (deduction tree, inductive rule hypothesis, causal graph) to guarantee verifiable answers, then rendered into multi-document scientific discourse via per-track domain-tuned genres. The construction lets us independently vary two difficulty axes: how hard it is to extract the key information needed for inference, and how hard the principled inference itself is. We test six models. Both axes hurt every model, and their effects compound. The rendering even hurts neurosymbolic pipelines, which hand inference to a verified solver. The two axes yield a per-model extraction-vs-inference profile: for instance, reasoning models like deepseek-r1 mostly surpass non-reasoning instruct models on the inference axis. To our knowledge, SciR is the first multi-paradigm scientific-reasoning benchmark with parametric control on both extraction and inference difficulty. |
| 2026-06-11 | [A Machine Learning Framework for Real-Time Personalized Ergonomic Pose Analysis](http://arxiv.org/abs/2606.12988v1) | Manex Atxa, Bruno Simoes et al. | This paper introduces a new methodology for real-time prediction of ergonomic and non-ergonomic human poses using volumetric video data in three dimensions. Although the methodology was designed for ergonomic assessments, it can be adapted to other applications requiring real-time analysis of human posture. One aspect that makes this system stand out is its ability to analyze 3D point clouds during the assessment, enabling computation from multiple angles. This overcomes a critical limitation of cameras which provide often a fixed viewpoint, thereby restricting the data available for a thorough postural evaluation, especially when occlusions occur. The system continuously and automatically performs pose inference using the chosen perspective on the real-time streaming data; however, only the poses manually selected and labeled by the user are used to train the personalized deep learning classifier. The methodology has been refined through a case study in which RGB-D cameras captured subjects performing load-lifting tasks, enabling real-time skeletal labeling. The model was trained on this data and, following the training phase, performs inference on new streaming data in real time. This research offers a scalable and pragmatic approach for real-time ergonomic evaluation by combining state-of-the-art 3D data technologies and traditional 2D pose estimation algorithms. It addresses the increasing need for safety and health monitoring in workplace environments, marking a notable contribution to the domain. |
| 2026-06-11 | [An Embodied Simulation Platform, Benchmark, and Data-Efficient Augmentation Framework for Wet-Lab Robotics](http://arxiv.org/abs/2606.12936v1) | Zhe Liu, Huanbo Jin et al. | Wet-lab robots can improve the reproducibility, throughput, and safety of biomedical experiments, but scaling their learning requires customizable simulators for safe and reproducible task generation, open editable laboratory assets, and efficient pipelines that turn limited demonstrations into usable training data. We present Pipette, an embodied simulation platform, benchmark, and data-efficient augmentation framework for wet-lab robot learning. Pipette releases over 43 open-source and re-editable wet-lab assets, together with an extensible asset-building pipeline. A key component of Pipette is its simulation-based data augmentation pipeline, replaying human demonstrations in simulation, applies lighting, camera, speed, and action perturbations, and filters generated episodes with automatic task success checks, rapidly expanding usable training data from limited manual demonstrations. We further introduce an 11-task wet-lab embodied benchmark covering sample handling, culture-ware manipulation, device operation, and precision placement. With only 30 demonstrations per task, ACT achieves 65.5% average success rate, while simulation augmentation improves SmolVLA from 44.1% to 74.7% and π0 from 40.4% to 46.5%, validating the effectiveness of Pipette for data-efficient VLA training and evaluation. Pipette also supports natural-language-driven scene construction and task registration, lowering the barrier for non-expert users to define new wet-lab robotic tasks. |
| 2026-06-11 | [MAStrike: Shapley-Guided Collusive Red-Teaming on Multi-Agent Systems](http://arxiv.org/abs/2606.12918v1) | Chejian Xu, Zhaorun Chen et al. | Hierarchical multi-agent systems (MAS) are rapidly being deployed in high-stakes workflows across domains such as finance and software engineering. In these systems, safety and security are inherently distributed across role-specialized agents, significantly expanding the attack surface, particularly under coordinated adversarial behaviors such as privilege escalation and cross-agent collusion. Existing red-teaming approaches for MAS remain limited: they rely on heuristic selection of target agents and perturb isolated message streams, leaving critical questions unanswered as which agents are most responsible for system safety, and how compromised agents can coordinate to bypass defenses. We propose MAStrike, a closed-loop framework for collusive red-teaming in hierarchical MAS. We propose the first agent-level Shapley value analysis for MAS, quantifying each agent's marginal contribution to system robustness under task-specific distributions. GGuided by this attribution, MAStrike identifies vulnerable agent coalitions and generates coordinated, role-aware adversarial manipulations. These attacks are iteratively refined through structured causal diagnosis, attributing failure cases to uncompromised agents that block adversarial attempts. We further build a comprehensive MAS red-teaming benchmark and controllable environments spanning diverse hierarchical topologies and domains, including finance, software engineering, and CRM. Extensive experiments across MAS built on multiple frontier models show that MAStrike substantially outperforms heuristic baselines. Our analysis further uncovers non-trivial Shapley value distributions and higher-order interaction structures among agents, revealing critical vulnerabilities and coordination patterns that are overlooked by prior single-agent or template-based methods. |
| 2026-06-11 | [SafeLLM: Extraction as a Hallucination-Resistant Alternative to Rewriting in Safety-Critical Settings](http://arxiv.org/abs/2606.12897v1) | Julia Ive, Felix Jozsa et al. | Large language models (LLMs) are increasingly used to access organisational documentation, including standard operating procedures (SOPs), HR policies and institutional guidelines. However, retrieval-augmented generation (RAG) systems that rely on free-form rewriting can introduce hallucinations and unstable trade-offs between completeness and conciseness, particularly in safety- and compliance-critical settings. Objectives: To evaluate extraction as a hallucination-resistant alternative to rewriting-based RAG and compare strategies that balance precision, recall and safety across document types and model scales. Methods: We compare multiple prompting strategies, including line-number-based source selection, extraction of relevant guideline sentences with explicit safety annotations, and a multi-stage pipeline that refines draft answers using supporting evidence from source guidelines. Experiments are conducted on documents of varying length and structure, including local NHS acute care and oncology guidelines and UK-wide NICE guidelines, using both frontier-scale and locally deployable models. Performance is assessed using automatic metrics and human expert evaluation of relevance and completeness. Results: Line-number selection achieves the strongest results, outperforming direct copying and safety-focused strategies across both large and small models while maintaining high term recall (up to 95%) and close alignment with source text. Safety-oriented approaches improve precision but introduce systematic omissions, while multi-stage filtering further amplifies this trade-off. Performance varies with document structure: line-based extraction excels in protocol-like content, whereas alternative strategies perform better on more verbose documents (up to 97% term recall). |

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



